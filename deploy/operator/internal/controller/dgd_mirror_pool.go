/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// poolRegistration is what etcd discovery holds for one mirror-rollout worker:
// the Dynamo namespace and worker ID of its instance, and the taints of its
// base model card.
type poolRegistration struct {
	namespace string
	workerID  string
	taints    []string
}

// poolRegistry reads and sets the pool roles of mirror-rollout workers.
type poolRegistry interface {
	// Registrations returns the registration of each of pods that published a
	// pool member record and a base model card.
	Registrations(ctx context.Context, pods map[types.NamespacedName]bool) (map[types.NamespacedName]poolRegistration, error)
	// SetRole replaces the caller-managed taints of a registered worker.
	SetRole(ctx context.Context, worker *mirrorWorker, taints []string) error
}

// poolStore is the subset of etcd a poolRegistry needs.
type poolStore interface {
	// List returns every key under prefix with its value.
	List(ctx context.Context, prefix string) (map[string][]byte, error)
	Put(ctx context.Context, key string, value []byte) error
}

type etcdPoolStore struct {
	client *clientv3.Client
}

func (s etcdPoolStore) List(ctx context.Context, prefix string) (map[string][]byte, error) {
	response, err := s.client.Get(ctx, prefix, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("list %s: %w", prefix, err)
	}
	entries := make(map[string][]byte, len(response.Kvs))
	for _, kv := range response.Kvs {
		entries[string(kv.Key)] = kv.Value
	}
	return entries, nil
}

func (s etcdPoolStore) Put(ctx context.Context, key string, value []byte) error {
	if _, err := s.client.Put(ctx, key, string(value)); err != nil {
		return fmt.Errorf("put %s: %w", key, err)
	}
	return nil
}

type storePoolRegistry struct {
	store poolStore
}

// poolMember is a worker's record under consts.PoolMembersPrefix.
type poolMember struct {
	PodNamespace string `json:"pod_namespace"`
	PodName      string `json:"pod_name"`
}

// modelCardRecord is the part of a registered model card a rollout reads.
type modelCardRecord struct {
	InstanceID  json.Number `json:"instance_id"`
	ModelSuffix string      `json:"model_suffix"`
	CardJSON    struct {
		RuntimeConfig struct {
			Taints []string `json:"taints"`
		} `json:"runtime_config"`
	} `json:"card_json"`
}

// splitPoolKey splits "<prefix><dynamo namespace>/<worker id in hex>" into the
// namespace and the decimal worker ID.
func splitPoolKey(key, prefix string) (string, string, bool) {
	namespace, hexID, found := strings.Cut(strings.TrimPrefix(key, prefix), "/")
	if !found || namespace == "" || strings.Contains(hexID, "/") {
		return "", "", false
	}
	id, err := strconv.ParseUint(hexID, 16, 64)
	if err != nil {
		return "", "", false
	}
	return namespace, strconv.FormatUint(id, 10), true
}

func (r storePoolRegistry) Registrations(ctx context.Context, pods map[types.NamespacedName]bool) (map[types.NamespacedName]poolRegistration, error) {
	records, err := r.store.List(ctx, consts.PoolMembersPrefix)
	if err != nil {
		return nil, err
	}
	// Read the cards of the namespaces these Pods' workers registered in.
	type workerKey struct{ namespace, workerID string }
	members := map[workerKey]types.NamespacedName{}
	incarnations := map[types.NamespacedName]int{}
	cards := map[string]map[string][]string{}
	for key, value := range records {
		namespace, workerID, ok := splitPoolKey(key, consts.PoolMembersPrefix)
		if !ok {
			continue
		}
		var member poolMember
		if err := json.Unmarshal(value, &member); err != nil {
			continue
		}
		pod := types.NamespacedName{Namespace: member.PodNamespace, Name: member.PodName}
		if !pods[pod] {
			continue
		}
		members[workerKey{namespace, workerID}] = pod
		incarnations[pod]++
		if _, read := cards[namespace]; read {
			continue
		}
		cards[namespace], err = r.baseCardTaints(ctx, namespace)
		if err != nil {
			return nil, err
		}
	}

	registrations := make(map[types.NamespacedName]poolRegistration, len(members))
	for worker, pod := range members {
		// A Pod whose restarted worker registered before the previous one's
		// lease expired has two records; it is unregistered until one remains.
		if incarnations[pod] > 1 {
			continue
		}
		taints, carded := cards[worker.namespace][worker.workerID]
		if !carded {
			continue
		}
		registrations[pod] = poolRegistration{namespace: worker.namespace, workerID: worker.workerID, taints: taints}
	}
	return registrations, nil
}

// baseCardTaints returns the taints of every base model card in namespace, by
// decimal worker ID.
func (r storePoolRegistry) baseCardTaints(ctx context.Context, namespace string) (map[string][]string, error) {
	entries, err := r.store.List(ctx, consts.ModelCardsPrefix+namespace+"/")
	if err != nil {
		return nil, err
	}
	taints := map[string][]string{}
	for _, value := range entries {
		var card modelCardRecord
		if err := json.Unmarshal(value, &card); err != nil || card.ModelSuffix != "" {
			continue
		}
		id, err := strconv.ParseUint(card.InstanceID.String(), 10, 64)
		if err != nil {
			continue
		}
		cardTaints := card.CardJSON.RuntimeConfig.Taints
		if cardTaints == nil {
			cardTaints = []string{}
		}
		taints[strconv.FormatUint(id, 10)] = cardTaints
	}
	return taints, nil
}

func (r storePoolRegistry) SetRole(ctx context.Context, worker *mirrorWorker, taints []string) error {
	id, err := strconv.ParseUint(worker.workerID, 10, 64)
	if err != nil {
		return fmt.Errorf("worker %s of pod %s: %w", worker.workerID, worker.pod.Name, err)
	}
	if taints == nil {
		taints = []string{}
	}
	value, err := json.Marshal(map[string][]string{"taints": taints})
	if err != nil {
		return err
	}
	key := fmt.Sprintf("%s%s/%x/role", consts.PoolRolesPrefix, worker.namespace, id)
	return r.store.Put(ctx, key, value)
}

// poolWatcher turns changes to pool member records and model cards into
// reconciles of the DGD that owns the worker, and deletes the role record of
// a worker whose member record ends.
type poolWatcher struct {
	client *clientv3.Client
	reader client.Reader
	events chan event.GenericEvent
}

func newPoolWatcher(etcd *clientv3.Client, reader client.Reader) *poolWatcher {
	return &poolWatcher{client: etcd, reader: reader, events: make(chan event.GenericEvent, 1024)}
}

// Start watches until ctx ends, re-establishing the watch after any failure.
func (w *poolWatcher) Start(ctx context.Context) error {
	logger := logf.FromContext(ctx).WithName("mirror-pool-watcher")
	for ctx.Err() == nil {
		if err := w.watch(ctx); err != nil && ctx.Err() == nil {
			logger.Error(err, "Pool watch failed; retrying")
			select {
			case <-ctx.Done():
			case <-time.After(5 * time.Second):
			}
		}
	}
	return nil
}

func (w *poolWatcher) watch(ctx context.Context) error {
	response, err := w.client.Get(ctx, consts.PoolMembersPrefix, clientv3.WithPrefix())
	if err != nil {
		return err
	}
	members := map[string]poolMember{}
	for _, kv := range response.Kvs {
		var member poolMember
		if json.Unmarshal(kv.Value, &member) == nil {
			members[strings.TrimPrefix(string(kv.Key), consts.PoolMembersPrefix)] = member
			w.enqueue(ctx, member)
		}
	}

	// Remove the roles of workers that left while nothing watched.
	roles, err := w.client.Get(ctx, consts.PoolRolesPrefix, clientv3.WithPrefix(), clientv3.WithKeysOnly())
	if err != nil {
		return err
	}
	for _, kv := range roles.Kvs {
		worker := strings.TrimSuffix(strings.TrimPrefix(string(kv.Key), consts.PoolRolesPrefix), "/role")
		if _, live := members[worker]; !live {
			if _, err := w.client.Delete(ctx, string(kv.Key)); err != nil {
				return err
			}
		}
	}

	watchCtx, cancel := context.WithCancel(clientv3.WithRequireLeader(ctx))
	defer cancel()
	from := clientv3.WithRev(response.Header.Revision + 1)
	memberEvents := w.client.Watch(watchCtx, consts.PoolMembersPrefix, clientv3.WithPrefix(), from, clientv3.WithPrevKV())
	cardEvents := w.client.Watch(watchCtx, consts.ModelCardsPrefix, clientv3.WithPrefix(), from)
	for {
		select {
		case <-ctx.Done():
			return nil
		case batch, ok := <-memberEvents:
			if !ok || batch.Err() != nil {
				return fmt.Errorf("member watch ended: %v", batch.Err())
			}
			for _, ev := range batch.Events {
				worker := strings.TrimPrefix(string(ev.Kv.Key), consts.PoolMembersPrefix)
				if ev.Type == clientv3.EventTypeDelete {
					member := members[worker]
					delete(members, worker)
					// The worker is gone; so is any role set for it.
					if _, err := w.client.Delete(ctx, consts.PoolRolesPrefix+worker+"/role"); err != nil {
						return err
					}
					w.enqueue(ctx, member)
					continue
				}
				var member poolMember
				if json.Unmarshal(ev.Kv.Value, &member) == nil {
					members[worker] = member
					w.enqueue(ctx, member)
				}
			}
		case batch, ok := <-cardEvents:
			if !ok || batch.Err() != nil {
				return fmt.Errorf("model card watch ended: %v", batch.Err())
			}
			for _, ev := range batch.Events {
				// <namespace>/<component>/<endpoint>/<worker id in hex>[/<suffix>]
				parts := strings.Split(strings.TrimPrefix(string(ev.Kv.Key), consts.ModelCardsPrefix), "/")
				if len(parts) < 4 {
					continue
				}
				if member, found := members[parts[0]+"/"+parts[3]]; found {
					w.enqueue(ctx, member)
				}
			}
		}
	}
}

// enqueue reconciles the DGD that owns the member's Pod.
func (w *poolWatcher) enqueue(ctx context.Context, member poolMember) {
	if member.PodName == "" {
		return
	}
	pod := &corev1.Pod{}
	if err := w.reader.Get(ctx, types.NamespacedName{Namespace: member.PodNamespace, Name: member.PodName}, pod); err != nil {
		return
	}
	requests := mapDGDWorkerPodToRequests(ctx, pod)
	if len(requests) == 0 {
		return
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	dgd.Namespace, dgd.Name = requests[0].Namespace, requests[0].Name
	select {
	case w.events <- event.GenericEvent{Object: dgd}:
	case <-ctx.Done():
	}
}
