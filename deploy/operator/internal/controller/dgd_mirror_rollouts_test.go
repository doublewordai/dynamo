/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// mirrorFleet is a test DSL for one mirror-rollout DGD with a worker component
// and the Pods and discovery records of its workers.
type mirrorFleet struct {
	t         *testing.T
	dgd       *nvidiacomv1beta1.DynamoGraphDeployment
	objects   []client.Object
	pool      *memoryPoolStore
	created   time.Time
	discovery configv1alpha1.DiscoveryBackend
}

func newMirrorFleet(t *testing.T) *mirrorFleet {
	return &mirrorFleet{
		t: t,
		dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "graph",
				Namespace:   "serving",
				UID:         "graph-uid",
				Annotations: map[string]string{consts.KubeAnnotationMirrorRollouts: "true"},
			},
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
					ComponentName: "worker",
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(2)),
				}},
			},
		},
		pool:      newMemoryPoolStore(),
		created:   time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC),
		discovery: configv1alpha1.DiscoveryBackendEtcd,
	}
}

// worker adds a Ready worker Pod of generation hash, registered in dynamo
// namespace ns with worker id id and taints.
func (f *mirrorFleet) worker(name, hash, ns string, id uint64, taints ...string) *mirrorFleet {
	f.created = f.created.Add(time.Minute)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         f.dgd.Namespace,
			UID:               types.UID(name + "-uid"),
			CreationTimestamp: metav1.NewTime(f.created),
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: f.dgd.Name,
				consts.KubeLabelDynamoComponent:           "worker",
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypeWorker,
				consts.KubeLabelDynamoSelector:            f.dgd.Name + "-worker",
				consts.KubeLabelDynamoWorkerHash:          hash,
			},
		},
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
	f.objects = append(f.objects, pod)
	f.pool.register(pod, ns, id, taints)
	return f
}

// legacyWorker adds a Ready worker Pod of generation hash that publishes no
// pool member record: one started before the opt-in, or by a runtime without
// pool roles.
func (f *mirrorFleet) legacyWorker(name, hash string) *mirrorFleet {
	f.worker(name, hash, "unused", 0)
	for key := range f.pool.entries {
		if strings.HasPrefix(key, consts.PoolMembersPrefix+"unused/") || strings.HasPrefix(key, consts.ModelCardsPrefix+"unused/") {
			delete(f.pool.entries, key)
		}
	}
	return f
}

// memoryPoolStore plays etcd discovery and the workers' side of the pool
// role channel: a role written for a worker replaces its card's pool taints,
// keeping the worker's own and topology taints.
type memoryPoolStore struct {
	entries map[string][]byte
	// roles holds the taints last set for each Pod.
	roles map[string][]string
}

func newMemoryPoolStore() *memoryPoolStore {
	return &memoryPoolStore{entries: map[string][]byte{}, roles: map[string][]string{}}
}

// register publishes the member record and base model card a worker Pod
// registers in dynamo namespace ns with worker id id and taints.
func (s *memoryPoolStore) register(pod *corev1.Pod, ns string, id uint64, taints []string) {
	member, _ := json.Marshal(poolMember{PodNamespace: pod.Namespace, PodName: pod.Name})
	s.entries[fmt.Sprintf("%s%s/%x", consts.PoolMembersPrefix, ns, id)] = member
	s.publishCard(ns, id, taints)
}

func (s *memoryPoolStore) publishCard(ns string, id uint64, taints []string) {
	card, _ := json.Marshal(map[string]any{
		"type":        "Model",
		"namespace":   ns,
		"component":   "backend",
		"endpoint":    "generate",
		"instance_id": id,
		"card_json":   map[string]any{"runtime_config": map[string]any{"taints": taints}},
	})
	s.entries[fmt.Sprintf("%s%s/backend/generate/%x", consts.ModelCardsPrefix, ns, id)] = card
}

func (s *memoryPoolStore) List(_ context.Context, prefix string) (map[string][]byte, error) {
	entries := map[string][]byte{}
	for key, value := range s.entries {
		if strings.HasPrefix(key, prefix) {
			entries[key] = value
		}
	}
	return entries, nil
}

func (s *memoryPoolStore) Put(_ context.Context, key string, value []byte) error {
	s.entries[key] = value
	worker, isRole := strings.CutPrefix(key, consts.PoolRolesPrefix)
	if !isRole {
		return nil
	}
	worker = strings.TrimSuffix(worker, "/role")

	// The worker applies the role to its card, as its runtime would.
	var role struct {
		Taints []string `json:"taints"`
	}
	if err := json.Unmarshal(value, &role); err != nil {
		return err
	}
	var member poolMember
	if err := json.Unmarshal(s.entries[consts.PoolMembersPrefix+worker], &member); err != nil {
		return err
	}
	ns, hexID, _ := strings.Cut(worker, "/")
	id, err := strconv.ParseUint(hexID, 16, 64)
	if err != nil {
		return err
	}
	s.roles[member.PodName] = role.Taints

	// A worker whose card is not registered applies the role once it registers.
	cardJSON, carded := s.entries[fmt.Sprintf("%s%s/backend/generate/%s", consts.ModelCardsPrefix, ns, hexID)]
	if !carded {
		return nil
	}
	var card modelCardRecord
	if err := json.Unmarshal(cardJSON, &card); err != nil {
		return err
	}
	taints := slices.Clone(role.Taints)
	for _, taint := range card.CardJSON.RuntimeConfig.Taints {
		if !strings.HasPrefix(taint, consts.MirrorTaintPrefix) {
			taints = append(taints, taint)
		}
	}
	s.publishCard(ns, id, taints)
	return nil
}

func (f *mirrorFleet) build() (client.Client, *memoryPoolStore, *mirrorRolloutReconciler) {
	scheme := runtime.NewScheme()
	require.NoError(f.t, nvidiacomv1alpha1.AddToScheme(scheme))
	require.NoError(f.t, nvidiacomv1beta1.AddToScheme(scheme))
	require.NoError(f.t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(append([]client.Object{f.dgd}, f.objects...)...).
		WithIndex(&corev1.Pod{}, dgdComponentPodIndex, dgdComponentPodIndexValues).
		WithStatusSubresource(&nvidiacomv1alpha1.DynamoMirrorPair{}).
		Build()
	rollout := newDGDWorkerRolloutReconciler(kubeClient, events.NewFakeRecorder(100))
	rollout.config = &configv1alpha1.OperatorConfiguration{Discovery: configv1alpha1.DiscoveryConfiguration{Backend: f.discovery}}
	rollout.pool = storePoolRegistry{store: f.pool}
	return kubeClient, f.pool, newMirrorRolloutReconciler(rollout)
}

func rollingContext(newHash string) dynamo.RollingUpdateContext {
	return dynamo.RollingUpdateContext{
		NewWorkerHash:                      newHash,
		OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
	}
}

func onlyPair(t *testing.T, kubeClient client.Client) *nvidiacomv1alpha1.DynamoMirrorPair {
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	require.NoError(t, kubeClient.List(context.Background(), pairs))
	require.Len(t, pairs.Items, 1)
	return &pairs.Items[0]
}

func setVerdict(t *testing.T, kubeClient client.Client, pair *nvidiacomv1alpha1.DynamoMirrorPair, verdict string) {
	meta.SetStatusCondition(&pair.Status.Conditions, metav1.Condition{
		Type: verdict, Status: metav1.ConditionTrue, Reason: "Judged", Message: "test judge",
	})
	require.NoError(t, kubeClient.Status().Update(context.Background(), pair))
}

func TestMirrorRolloutPromotesParkedWorkersOutsideARollout(t *testing.T) {
	ctx := context.Background()

	t.Log("A mirror-rollout deployment with one parked worker and no rollout in progress")
	fleet := newMirrorFleet(t).worker("w-a", "gen1", "ns-gen1", 11, consts.ParkedMirrorTaint, "zone-a", "dynamo.topology/rack=r1")
	_, taints, mirrors := fleet.build()

	t.Log("Reconcile promotes the parked worker, whose card keeps its own taints")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, dynamo.RollingUpdateContext{NewWorkerHash: "gen1"}))
	assert.Equal(t, map[string][]string{"w-a": {}}, taints.roles)
	registrations, err := storePoolRegistry{store: taints}.Registrations(ctx, map[types.NamespacedName]bool{{Namespace: "serving", Name: "w-a"}: true})
	require.NoError(t, err)
	assert.ElementsMatch(t, []string{"zone-a", "dynamo.topology/rack=r1"}, registrations[types.NamespacedName{Namespace: "serving", Name: "w-a"}].taints)
}

func TestMirrorRolloutPairsPromotesAndCountsServingWorkers(t *testing.T) {
	ctx := context.Background()

	t.Log("Two serving old workers and one parked new worker during a rollout")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("old-b", "gen1", "ns-gen1", 12).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")

	t.Log("The first reconcile pairs the new worker with the oldest serving old worker")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	pair := onlyPair(t, kubeClient)
	assert.Equal(t, "old-a", pair.Spec.Shadowed.PodName)
	assert.Equal(t, "11", pair.Spec.Shadowed.WorkerID)
	assert.Equal(t, "ns-gen1", pair.Spec.Shadowed.DynamoNamespace)
	assert.True(t, pair.Spec.FirstPair)
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhasePending, pair.Status.Phase)

	t.Log("The next reconcile assigns the mirror taint, then the one after starts mirroring")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	assert.Equal(t, []string{consts.MirrorTaintPrefix + "ns-gen1/11"}, taints.roles["new-a"])
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	pair = onlyPair(t, kubeClient)
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring, pair.Status.Phase)
	require.NotNil(t, pair.Status.MirroringSince)

	t.Log("The mirror counts as idle while it mirrors")
	idle, err := mirrors.rollout.idlePoolWorkers(ctx, fleet.dgd, "worker")
	require.NoError(t, err)
	assert.Equal(t, map[string]int32{"gen2": 1}, idle)

	t.Log("An approval marks the shadowed worker for removal and promotes the mirror")
	setVerdict(t, kubeClient, pair, nvidiacomv1alpha1.DynamoMirrorPairConditionApproved)
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	assert.Equal(t, []string{}, taints.roles["new-a"])
	shadowed := &corev1.Pod{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: "serving", Name: "old-a"}, shadowed))
	assert.Equal(t, consts.PromotedShadowDeletionCost, shadowed.Annotations[consts.KubeAnnotationPodDeletionCost])

	t.Log("Once the promotion shows in discovery the pair is promoted and the worker serves")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhasePromoted, onlyPair(t, kubeClient).Status.Phase)
	idle, err = mirrors.rollout.idlePoolWorkers(ctx, fleet.dgd, "worker")
	require.NoError(t, err)
	assert.Empty(t, idle)
}

func TestMirrorRolloutRejectionStopsTheGeneration(t *testing.T) {
	ctx := context.Background()

	t.Log("A mirroring pair and a second parked new worker")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("old-b", "gen1", "ns-gen1", 12).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	}
	pair := onlyPair(t, kubeClient)
	require.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring, pair.Status.Phase)

	t.Log("A rejection aborts the pair and parks its mirror again")
	setVerdict(t, kubeClient, pair, nvidiacomv1alpha1.DynamoMirrorPairConditionRejected)
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	pair = onlyPair(t, kubeClient)
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, pair.Status.Phase)
	assert.Equal(t, mirrorPairAbortRejected, pair.Status.Reason)
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])

	t.Log("Later reconciles neither pair nor promote a worker of the rejected generation")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	onlyPair(t, kubeClient)
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])
}

func TestMirrorRolloutParksAMirrorWhoseShadowedWorkerLeft(t *testing.T) {
	ctx := context.Background()

	t.Log("A mirroring pair")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	}

	t.Log("The shadowed worker leaves before a verdict")
	shadowed := &corev1.Pod{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: "serving", Name: "old-a"}, shadowed))
	require.NoError(t, kubeClient.Delete(ctx, shadowed))

	t.Log("The pair aborts and its mirror parks for a new pair")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	pair := onlyPair(t, kubeClient)
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, pair.Status.Phase)
	assert.Equal(t, mirrorPairAbortShadowedGone, pair.Status.Reason)
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])
}

func TestMirrorRolloutLeavesOtherDeploymentsAlone(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		discovery   configv1alpha1.DiscoveryBackend
	}{
		{name: "a deployment that did not opt in", discovery: configv1alpha1.DiscoveryBackendEtcd},
		{
			name:        "a deployment on Kubernetes discovery",
			annotations: map[string]string{consts.KubeAnnotationMirrorRollouts: "true"},
			discovery:   configv1alpha1.DiscoveryBackendKubernetes,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			t.Log("A parked worker in a deployment mirror rollouts do not cover")
			fleet := newMirrorFleet(t).worker("w-a", "gen1", "ns-gen1", 11, consts.ParkedMirrorTaint)
			fleet.dgd.Annotations = tt.annotations
			fleet.discovery = tt.discovery
			kubeClient, taints, mirrors := fleet.build()

			t.Log("Reconcile does nothing")
			require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
			assert.Empty(t, taints.roles)
			pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
			require.NoError(t, kubeClient.List(ctx, pairs))
			assert.Empty(t, pairs.Items)
		})
	}
}

func TestMirrorRolloutRejectionAbortsTheGenerationsOtherOpenPairs(t *testing.T) {
	ctx := context.Background()

	t.Log("Two mirroring pairs of one generation")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("old-b", "gen1", "ns-gen1", 12).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint).
		worker("new-b", "gen2", "ns-gen2", 22, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	}
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	require.NoError(t, kubeClient.List(ctx, pairs))
	require.Len(t, pairs.Items, 2)
	byMirror := map[string]*nvidiacomv1alpha1.DynamoMirrorPair{}
	for i := range pairs.Items {
		byMirror[pairs.Items[i].Spec.Mirror.PodName] = &pairs.Items[i]
	}

	t.Log("One pair is rejected while the other is approved")
	setVerdict(t, kubeClient, byMirror["new-a"], nvidiacomv1alpha1.DynamoMirrorPairConditionRejected)
	setVerdict(t, kubeClient, byMirror["new-b"], nvidiacomv1alpha1.DynamoMirrorPairConditionApproved)

	t.Log("Neither mirror is promoted and both park again")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	require.NoError(t, kubeClient.List(ctx, pairs))
	reasons := map[string]string{}
	for _, pair := range pairs.Items {
		assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, pair.Status.Phase)
		reasons[pair.Spec.Mirror.PodName] = pair.Status.Reason
	}
	assert.Equal(t, map[string]string{"new-a": mirrorPairAbortRejected, "new-b": mirrorPairAbortGenerationRejected}, reasons)
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-b"])
	shadowed := &corev1.Pod{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: "serving", Name: "old-b"}, shadowed))
	assert.Empty(t, shadowed.Annotations[consts.KubeAnnotationPodDeletionCost])
}

func TestMirrorRolloutHoldsOldWorkersUntilTheMirrorIsPromoted(t *testing.T) {
	tests := []struct {
		name        string
		newTaints   []string
		expectedOld int32
	}{
		{name: "a parked new worker retires no old worker", newTaints: []string{consts.ParkedMirrorTaint}, expectedOld: 2},
		{name: "a mirroring new worker retires no old worker", newTaints: []string{consts.MirrorTaintPrefix + "ns-old/11"}, expectedOld: 2},
		{name: "a promoted new worker retires one old worker", newTaints: nil, expectedOld: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("A two-replica mirror-rollout worker with one surge slot, one Kubernetes-ready new worker")
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(2)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentRollingUpdateMaxSurge:       "1",
						KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
					},
				},
			})
			dgd.Annotations = map[string]string{
				consts.AnnotationCurrentWorkerHashV2: testOldWorkerHash,
				consts.KubeAnnotationMirrorRollouts:  "true",
			}
			newHash := betaDGDWorkersSpecHash(t, dgd)
			dcd := func(hash string, spec, available int32) *nvidiacomv1beta1.DynamoComponentDeployment {
				return createTestDCD(t, dgd, &nvidiacomv1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-worker-" + hash[:8],
						Namespace: "default",
						Labels: map[string]string{
							consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
							consts.KubeLabelDynamoWorkerHash:          hash,
						},
					},
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ComponentType: consts.ComponentTypeWorker,
							ServiceName:   "worker",
							Replicas:      ptr.To(spec),
						},
					},
					Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
						Service: &nvidiacomv1alpha1.ServiceReplicaStatus{Replicas: spec, AvailableReplicas: ptr.To(available)},
					},
				})
			}
			newPod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "worker-new",
					Namespace: "default",
					UID:       "worker-new-uid",
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
						consts.KubeLabelDynamoComponent:           "worker",
						consts.KubeLabelDynamoWorkerHash:          newHash,
					},
				},
				Status: corev1.PodStatus{
					Phase:      corev1.PodRunning,
					Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
				},
			}
			pool := newMemoryPoolStore()
			pool.register(newPod, "ns-new", 21, tt.newTaints)
			r := createTestReconcilerWithStatus(dgd, withObjects(
				dcd(testOldWorkerHash, 2, 2),
				dcd(newHash, 1, 1),
				newPod,
			))
			r.config = &configv1alpha1.OperatorConfiguration{Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
			r.pool = storePoolRegistry{store: pool}

			t.Log("The old generation shrinks only once the new worker serves")
			rollingCtx, err := r.buildRollingUpdateContext(context.Background(), dgd)
			require.NoError(t, err)
			assert.Equal(t, tt.expectedOld, rollingCtx.OldWorkerReplicaTargetsByComponent["worker"])
			assert.Equal(t, int32(1), rollingCtx.NewWorkerReplicaTargetsByComponent["worker"])
		})
	}
}

func TestPoolRegistryJoinsMembersWithTheirBaseCards(t *testing.T) {
	tests := []struct {
		name    string
		entries map[string]string
		want    map[types.NamespacedName]poolRegistration
	}{
		{
			name: "a member with its base card",
			entries: map[string]string{
				consts.PoolMembersPrefix + "ns/b":                  `{"pod_namespace":"serving","pod_name":"w-a"}`,
				consts.ModelCardsPrefix + "ns/backend/generate/b":  `{"instance_id":11,"card_json":{"runtime_config":{"taints":["t"]}}}`,
				consts.ModelCardsPrefix + "ns/backend/generate/c":  `{"instance_id":12,"card_json":{"runtime_config":{"taints":["other"]}}}`,
				consts.ModelCardsPrefix + "ns2/backend/generate/b": `{"instance_id":11,"card_json":{"runtime_config":{"taints":["elsewhere"]}}}`,
			},
			want: map[types.NamespacedName]poolRegistration{
				{Namespace: "serving", Name: "w-a"}: {namespace: "ns", workerID: "11", taints: []string{"t"}},
			},
		},
		{
			name: "a member of a Pod not asked about",
			entries: map[string]string{
				consts.PoolMembersPrefix + "ns/b":                 `{"pod_namespace":"serving","pod_name":"w-other"}`,
				consts.ModelCardsPrefix + "ns/backend/generate/b": `{"instance_id":11,"card_json":{"runtime_config":{"taints":["t"]}}}`,
			},
			want: map[types.NamespacedName]poolRegistration{},
		},
		{
			name: "a member whose card has not registered",
			entries: map[string]string{
				consts.PoolMembersPrefix + "ns/b": `{"pod_namespace":"serving","pod_name":"w-a"}`,
			},
			want: map[types.NamespacedName]poolRegistration{},
		},
		{
			name: "an adapter card is not the base card",
			entries: map[string]string{
				consts.PoolMembersPrefix + "ns/b":                       `{"pod_namespace":"serving","pod_name":"w-a"}`,
				consts.ModelCardsPrefix + "ns/backend/generate/b/lora1": `{"instance_id":11,"model_suffix":"lora1","card_json":{"runtime_config":{"taints":["t"]}}}`,
			},
			want: map[types.NamespacedName]poolRegistration{},
		},
		{
			name: "a malformed member key",
			entries: map[string]string{
				consts.PoolMembersPrefix + "ns/not-hex":           `{"pod_namespace":"serving","pod_name":"w-a"}`,
				consts.ModelCardsPrefix + "ns/backend/generate/b": `{"instance_id":11,"card_json":{"runtime_config":{"taints":["t"]}}}`,
			},
			want: map[types.NamespacedName]poolRegistration{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Seed etcd with the records")
			store := newMemoryPoolStore()
			for key, value := range tt.entries {
				store.entries[key] = []byte(value)
			}

			t.Log("Registrations joins each member with its base card")
			pods := map[types.NamespacedName]bool{{Namespace: "serving", Name: "w-a"}: true}
			got, err := storePoolRegistry{store: store}.Registrations(context.Background(), pods)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMirrorRolloutRollsOrdinarilyFromWorkersThatPredateTheOptIn(t *testing.T) {
	ctx := context.Background()

	t.Log("Old workers without pool records and one parked new worker during a rollout")
	fleet := newMirrorFleet(t).
		legacyWorker("old-a", "gen1").
		legacyWorker("old-b", "gen1").
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()

	t.Log("The old workers count as serving")
	idle, err := mirrors.rollout.idlePoolWorkers(ctx, fleet.dgd, "worker")
	require.NoError(t, err)
	assert.Equal(t, map[string]int32{"gen2": 1}, idle)

	t.Log("Nothing can mirror them, so the new worker is promoted without a pair")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	assert.Equal(t, map[string][]string{"new-a": {}}, taints.roles)
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	require.NoError(t, kubeClient.List(ctx, pairs))
	assert.Empty(t, pairs.Items)
}

func TestMirrorRolloutWaitsForAnOldPoolWorkerToRegister(t *testing.T) {
	ctx := context.Background()

	t.Log("An old pool worker whose records have not appeared yet, and a parked new worker")
	fleet := newMirrorFleet(t).
		legacyWorker("old-a", "gen1").
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	fleet.objects[0].SetAnnotations(map[string]string{consts.KubeAnnotationMirrorRollouts: "true"})
	kubeClient, taints, mirrors := fleet.build()

	t.Log("Reconcile neither promotes the new worker nor pairs it")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	assert.Empty(t, taints.roles)
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	require.NoError(t, kubeClient.List(ctx, pairs))
	assert.Empty(t, pairs.Items)
}

func TestMirrorRolloutCountsAnUnregisteredPoolWorkerAsIdle(t *testing.T) {
	ctx := context.Background()

	t.Log("A Ready new pool worker whose records have not appeared yet")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		legacyWorker("new-a", "gen2")
	fleet.objects[1].SetAnnotations(map[string]string{consts.KubeAnnotationMirrorRollouts: "true"})
	_, _, mirrors := fleet.build()

	t.Log("It serves nothing yet, so it is idle")
	idle, err := mirrors.rollout.idlePoolWorkers(ctx, fleet.dgd, "worker")
	require.NoError(t, err)
	assert.Equal(t, map[string]int32{"gen2": 1}, idle)
}

func TestMirrorRolloutKeepsARejectedGenerationParked(t *testing.T) {
	ctx := context.Background()

	t.Log("Generation gen2 is rejected while its worker mirrors")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	}
	setVerdict(t, kubeClient, onlyPair(t, kubeClient), nvidiacomv1alpha1.DynamoMirrorPairConditionRejected)
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	require.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])

	t.Log("The deployment moves on to gen3 and settles with the gen2 worker still present")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, dynamo.RollingUpdateContext{NewWorkerHash: "gen3"}))

	t.Log("The rejected worker stays parked")
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])
}

func TestMirrorRolloutAbortsPairsOfASupersededGeneration(t *testing.T) {
	ctx := context.Background()

	t.Log("A gen2 worker mirrors an old worker")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	}
	require.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring, onlyPair(t, kubeClient).Status.Phase)

	t.Log("The deployment rolls to gen3 before a verdict")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen3")))

	t.Log("The pair aborts as superseded and its mirror parks")
	pair := onlyPair(t, kubeClient)
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, pair.Status.Phase)
	assert.Equal(t, mirrorPairAbortSuperseded, pair.Status.Reason)
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])
}

func TestMirrorRolloutAbortsAPairWhoseShadowedWorkerRestarted(t *testing.T) {
	ctx := context.Background()

	t.Log("A mirroring pair")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("old-b", "gen1", "ns-gen1", 12).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	}
	pair := onlyPair(t, kubeClient)
	require.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring, pair.Status.Phase)

	t.Log("The shadowed worker restarts in its Pod and registers a new worker ID")
	shadowed := &corev1.Pod{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: "serving", Name: pair.Spec.Shadowed.PodName}, shadowed))
	for key := range taints.entries {
		if strings.HasSuffix(key, fmt.Sprintf("/%x", uint64(11))) {
			delete(taints.entries, key)
		}
	}
	taints.register(shadowed, "ns-gen1", 13, nil)

	t.Log("The pair aborts as its shadowed worker gone, and the mirror parks")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	require.NoError(t, kubeClient.List(ctx, pairs))
	var aborted *nvidiacomv1alpha1.DynamoMirrorPair
	for i := range pairs.Items {
		if pairs.Items[i].Name == pair.Name {
			aborted = &pairs.Items[i]
		}
	}
	require.NotNil(t, aborted)
	assert.Equal(t, mirrorPairAbortShadowedGone, aborted.Status.Reason)
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])
}

func TestMirrorRolloutPromotesParkedWorkersWithNothingToMirror(t *testing.T) {
	ctx := context.Background()

	t.Log("A rollout whose old workers are all gone, and a parked new worker")
	fleet := newMirrorFleet(t).worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	_, taints, mirrors := fleet.build()

	t.Log("Nothing can be mirrored, so the new worker is promoted")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	assert.Equal(t, map[string][]string{"new-a": {}}, taints.roles)
}

func TestMirrorRolloutRetriesAPairUnderANewName(t *testing.T) {
	ctx := context.Background()

	t.Log("An earlier pair of the same two Pods aborted")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	fleet.objects = append(fleet.objects, &nvidiacomv1alpha1.DynamoMirrorPair{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "new-a.old-a",
			Namespace: "serving",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
				consts.KubeLabelDynamoComponent:           "worker",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoMirrorPairSpec{
			WorkerHash: "gen2",
			Mirror:     nvidiacomv1alpha1.DynamoMirrorPairWorker{PodName: "new-a"},
			Shadowed:   nvidiacomv1alpha1.DynamoMirrorPairWorker{PodName: "old-a"},
		},
		Status: nvidiacomv1alpha1.DynamoMirrorPairStatus{Phase: nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, Reason: mirrorPairAbortShadowedGone},
	})
	kubeClient, _, mirrors := fleet.build()

	t.Log("Reconcile pairs them again under the next attempt's name")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	retried := &nvidiacomv1alpha1.DynamoMirrorPair{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: "serving", Name: "new-a.old-a.1"}, retried))
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhasePending, retried.Status.Phase)
}

func TestMirrorRolloutKeepsAnApprovedServingMirrorWhenAnotherPairIsRejected(t *testing.T) {
	ctx := context.Background()

	t.Log("Two mirroring pairs of one generation")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("old-b", "gen1", "ns-gen1", 12).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint).
		worker("new-b", "gen2", "ns-gen2", 22, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	}
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	require.NoError(t, kubeClient.List(ctx, pairs))
	require.Len(t, pairs.Items, 2)
	byMirror := map[string]*nvidiacomv1alpha1.DynamoMirrorPair{}
	for i := range pairs.Items {
		byMirror[pairs.Items[i].Spec.Mirror.PodName] = &pairs.Items[i]
	}

	t.Log("new-a is approved and serving before new-b is rejected")
	setVerdict(t, kubeClient, byMirror["new-a"], nvidiacomv1alpha1.DynamoMirrorPairConditionApproved)
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	require.Equal(t, []string{}, taints.roles["new-a"])
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(byMirror["new-b"]), byMirror["new-b"]))
	setVerdict(t, kubeClient, byMirror["new-b"], nvidiacomv1alpha1.DynamoMirrorPairConditionRejected)

	t.Log("The rejection parks new-b but new-a stays promoted")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	assert.Equal(t, []string{}, taints.roles["new-a"])
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-b"])
}

func TestMirrorRolloutCountsMarkedWorkersServingOffEtcdDiscovery(t *testing.T) {
	ctx := context.Background()

	t.Log("A marked, Ready worker with no pool records in a deployment on Kubernetes discovery")
	fleet := newMirrorFleet(t).legacyWorker("w-a", "gen1")
	fleet.objects[0].SetAnnotations(map[string]string{consts.KubeAnnotationMirrorRollouts: "true"})
	fleet.discovery = configv1alpha1.DiscoveryBackendKubernetes
	_, _, mirrors := fleet.build()

	t.Log("It booted without a pool role, so it serves")
	idle, err := mirrors.rollout.idlePoolWorkers(ctx, fleet.dgd, "worker")
	require.NoError(t, err)
	assert.Empty(t, idle)
}

func TestMirrorRolloutBoundsLongPairNames(t *testing.T) {
	ctx := context.Background()

	t.Log("Two worker Pods with long names")
	long := strings.Repeat("w", 140)
	fleet := newMirrorFleet(t).
		worker(long+"-old", "gen1", "ns-gen1", 11).
		worker(long+"-new", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, _, mirrors := fleet.build()

	t.Log("Their pair gets a name within the object-name limit")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	pair := onlyPair(t, kubeClient)
	assert.LessOrEqual(t, len(pair.Name), 253)
	assert.True(t, strings.HasPrefix(pair.Name, long+"-new."))
}

func TestMirrorRolloutActsOnAnApprovalOnlyOnceTheMirrorMirrors(t *testing.T) {
	ctx := context.Background()

	t.Log("A pending pair approved before its mirror is assigned")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	pair := onlyPair(t, kubeClient)
	require.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhasePending, pair.Status.Phase)
	setVerdict(t, kubeClient, pair, nvidiacomv1alpha1.DynamoMirrorPairConditionApproved)

	t.Log("The mirror is assigned first, not promoted")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	assert.Equal(t, []string{consts.MirrorTaintPrefix + "ns-gen1/11"}, taints.roles["new-a"])

	t.Log("Once it mirrors, the approval promotes it")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	require.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring, onlyPair(t, kubeClient).Status.Phase)
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	assert.Equal(t, []string{}, taints.roles["new-a"])
}

func TestPoolRegistryLeavesAPodWithTwoIncarnationsUnregistered(t *testing.T) {
	t.Log("A restarted worker registered before its previous incarnation's lease expired")
	store := newMemoryPoolStore()
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "serving", Name: "w-a"}}
	store.register(pod, "ns", 11, nil)
	store.register(pod, "ns", 12, []string{consts.ParkedMirrorTaint})

	t.Log("The Pod is unregistered until one record remains")
	got, err := storePoolRegistry{store: store}.Registrations(context.Background(), map[types.NamespacedName]bool{{Namespace: "serving", Name: "w-a"}: true})
	require.NoError(t, err)
	assert.Empty(t, got)
}

func TestMirrorRolloutParksAMirrorThatLostItsRegistration(t *testing.T) {
	ctx := context.Background()

	t.Log("A mirroring pair")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	for range 3 {
		require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	}
	require.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring, onlyPair(t, kubeClient).Status.Phase)

	t.Log("The mirror's model card disappears while its Pod lives")
	delete(taints.entries, fmt.Sprintf("%sns-gen2/backend/generate/%x", consts.ModelCardsPrefix, uint64(21)))

	t.Log("The pair ends and the mirror's recorded identity gets the parked role for when it re-registers")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	pair := onlyPair(t, kubeClient)
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, pair.Status.Phase)
	assert.Equal(t, mirrorPairAbortMirrorGone, pair.Status.Reason)
	assert.Equal(t, []string{consts.ParkedMirrorTaint}, taints.roles["new-a"])
}

func TestMirrorRolloutWaitsForAPendingMirrorToRegister(t *testing.T) {
	ctx := context.Background()

	t.Log("A pending pair whose mirror's card is not observed yet")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	kubeClient, taints, mirrors := fleet.build()
	rolling := rollingContext("gen2")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	require.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhasePending, onlyPair(t, kubeClient).Status.Phase)
	delete(taints.entries, fmt.Sprintf("%sns-gen2/backend/generate/%x", consts.ModelCardsPrefix, uint64(21)))

	t.Log("The pair waits")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rolling))
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairPhasePending, onlyPair(t, kubeClient).Status.Phase)
}

func TestMirrorRolloutIgnoresPairsOfAPreviousDeployment(t *testing.T) {
	ctx := context.Background()

	t.Log("A rejected pair of this generation left by a deleted DGD of the same name")
	fleet := newMirrorFleet(t).
		worker("old-a", "gen1", "ns-gen1", 11).
		worker("new-a", "gen2", "ns-gen2", 21, consts.ParkedMirrorTaint)
	fleet.objects = append(fleet.objects, &nvidiacomv1alpha1.DynamoMirrorPair{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "new-a.old-a",
			Namespace: "serving",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
				consts.KubeLabelDynamoComponent:           "worker",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoMirrorPairSpec{
			WorkerHash: "gen2",
			Mirror:     nvidiacomv1alpha1.DynamoMirrorPairWorker{PodName: "new-a"},
			Shadowed:   nvidiacomv1alpha1.DynamoMirrorPairWorker{PodName: "old-a"},
		},
		Status: nvidiacomv1alpha1.DynamoMirrorPairStatus{Phase: nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, Reason: mirrorPairAbortRejected},
	})
	kubeClient, _, mirrors := fleet.build()

	t.Log("This deployment's generation still pairs")
	require.NoError(t, mirrors.Reconcile(ctx, fleet.dgd, rollingContext("gen2")))
	pair := &nvidiacomv1alpha1.DynamoMirrorPair{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: "serving", Name: "new-a.old-a.1"}, pair))
	assert.True(t, metav1.IsControlledBy(pair, fleet.dgd))
}

func TestMirrorRolloutKeepsTheDeploymentPendingWhileWorkersAreParked(t *testing.T) {
	ctx := context.Background()

	t.Log("A mirror-rollout deployment whose only worker is still parked")
	fleet := newMirrorFleet(t).worker("w-a", "gen1", "ns-gen1", 11, consts.ParkedMirrorTaint)
	_, _, mirrors := fleet.build()
	ready := ReconcileResult{
		State: nvidiacomv1beta1.DGDStateSuccessful,
		ComponentStatus: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
			"worker": {Replicas: 1, ReadyReplicas: ptr.To(int32(1)), AvailableReplicas: ptr.To(int32(1))},
		},
	}

	t.Log("The deployment is pending and the parked worker is not available")
	got, err := mirrors.applyPoolReadiness(ctx, fleet.dgd, "gen1", ready)
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.DGDStatePending, got.State)
	assert.Equal(t, int32(0), *got.ComponentStatus["worker"].AvailableReplicas)
}

func TestMirrorRolloutRestoresOldCapacityAfterARejection(t *testing.T) {
	t.Log("A four-replica mirror-rollout worker whose old generation already shrank to three")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(4)),
			Annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge:       "1",
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "1",
			},
		},
	})
	dgd.UID = "test-dgd-uid"
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHashV2: testOldWorkerHash,
		consts.KubeAnnotationMirrorRollouts:  "true",
	}
	newHash := betaDGDWorkersSpecHash(t, dgd)
	dcd := func(hash string, spec, available int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		return createTestDCD(t, dgd, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker-" + hash[:8],
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          hash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					ServiceName:   "worker",
					Replicas:      ptr.To(spec),
				},
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				Service: &nvidiacomv1alpha1.ServiceReplicaStatus{Replicas: spec, AvailableReplicas: ptr.To(available)},
			},
		})
	}
	rejected := &nvidiacomv1alpha1.DynamoMirrorPair{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-new.worker-old",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoComponent:           "worker",
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment",
				Name: "test-dgd", UID: dgd.UID, Controller: ptr.To(true),
			}},
		},
		Spec:   nvidiacomv1alpha1.DynamoMirrorPairSpec{WorkerHash: newHash},
		Status: nvidiacomv1alpha1.DynamoMirrorPairStatus{Phase: nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, Reason: mirrorPairAbortRejected},
	}
	newPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "worker-new",
			Namespace:   "default",
			Annotations: map[string]string{consts.KubeAnnotationMirrorRollouts: "true"},
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoComponent:           "worker",
				consts.KubeLabelDynamoWorkerHash:          newHash,
			},
		},
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
	pool := newMemoryPoolStore()
	pool.register(newPod, "ns-new", 21, []string{consts.ParkedMirrorTaint})
	r := createTestReconcilerWithStatus(dgd, withObjects(dcd(testOldWorkerHash, 3, 3), dcd(newHash, 1, 1), rejected, newPod))
	r.config = &configv1alpha1.OperatorConfiguration{Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
	r.pool = storePoolRegistry{store: pool}

	t.Log("The old generation returns to four and the rejected one keeps no parked worker")
	rollingCtx, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, int32(4), rollingCtx.OldWorkerReplicaTargetsByComponent["worker"])
	assert.Equal(t, int32(4), rollingCtx.OldWorkerReplicaTargetsByDCD["test-dgd-worker-"+testOldWorkerHash[:8]])
	assert.Equal(t, int32(0), rollingCtx.NewWorkerReplicaTargetsByComponent["worker"])
}
