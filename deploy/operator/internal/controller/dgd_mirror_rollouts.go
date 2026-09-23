/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"crypto/sha256"
	"fmt"
	"slices"
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomirrorpairs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomirrorpairs/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;patch

const (
	mirrorPairAbortRejected = "Rejected"
	// mirrorPairAbortGenerationRejected ends an open pair of a generation
	// another pair of which was rejected.
	mirrorPairAbortGenerationRejected = "GenerationRejected"
	mirrorPairAbortMirrorGone         = "MirrorGone"
	mirrorPairAbortShadowedGone       = "ShadowedGone"
	// mirrorPairAbortSuperseded ends an open pair whose generation is no
	// longer the one the deployment rolls to.
	mirrorPairAbortSuperseded = "Superseded"
)

// mirrorWorker is one worker Pod as a mirror rollout sees it: its Kubernetes
// state and the registration it published to etcd discovery.
type mirrorWorker struct {
	pod   *corev1.Pod
	ready bool
	// pool reports a worker the operator started with a pool role: its Pod
	// carries the mirror-rollout mark and its deployment uses etcd discovery.
	pool       bool
	registered bool
	namespace  string
	workerID   string
	taints     []string
}

func (w mirrorWorker) workerHash() string {
	return w.pod.Labels[consts.KubeLabelDynamoWorkerHash]
}

func (w mirrorWorker) hasTaint(taint string) bool {
	return slices.Contains(w.taints, taint)
}

func (w mirrorWorker) mirroring() bool {
	return slices.ContainsFunc(w.taints, func(taint string) bool {
		return strings.HasPrefix(taint, consts.MirrorTaintPrefix)
	})
}

func (w mirrorWorker) parked() bool {
	return w.registered && w.hasTaint(consts.ParkedMirrorTaint)
}

// poolWorker reports a worker the operator started with a pool role.
func (w mirrorWorker) poolWorker() bool {
	return w.pool
}

// serving reports a worker that clients can reach: Ready and not a mirror. A
// pool worker serves only once its registration shows no mirror taint; a
// worker started before the opt-in takes no pool role and serves when Ready.
func (w mirrorWorker) serving() bool {
	if !w.ready {
		return false
	}
	if w.registered {
		return !w.mirroring()
	}
	return !w.poolWorker()
}

// mirrorTaint is the taint a worker shadowing w publishes.
func (w mirrorWorker) mirrorTaint() string {
	return consts.MirrorTaintPrefix + w.namespace + "/" + w.workerID
}

// readMirrorWorkers returns every live Pod of a DGD component with its
// discovery registration, oldest first.
func (r *dgdWorkerRolloutReconciler) readMirrorWorkers(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) ([]mirrorWorker, error) {
	if r.pool == nil {
		return nil, fmt.Errorf("mirror rollouts need etcd discovery; the operator has no etcd address")
	}
	pods, err := r.listDGDComponentPods(ctx, dgd, componentName)
	if err != nil {
		return nil, err
	}
	sort.Slice(pods, func(i, j int) bool {
		if pods[i].CreationTimestamp.Equal(&pods[j].CreationTimestamp) {
			return pods[i].Name < pods[j].Name
		}
		return pods[i].CreationTimestamp.Before(&pods[j].CreationTimestamp)
	})
	podKeys := make(map[types.NamespacedName]bool, len(pods))
	for i := range pods {
		podKeys[client.ObjectKeyFromObject(&pods[i])] = true
	}
	registrations, err := r.pool.Registrations(ctx, podKeys)
	if err != nil {
		return nil, fmt.Errorf("read pool registrations: %w", err)
	}

	// Join each live Pod with the registration its worker published. A Pod
	// with none yet is unregistered; the pool watch reconciles again when its
	// records appear.
	etcdDiscovery := false
	for i := range dgd.Spec.Components {
		if component := &dgd.Spec.Components[i]; component.ComponentName == componentName && r.config != nil {
			etcdDiscovery = dynamo.ComponentDiscoveryBackend(dgd, component, r.config.Discovery.Backend) == configv1alpha1.DiscoveryBackendEtcd
		}
	}
	workers := make([]mirrorWorker, 0, len(pods))
	for i := range pods {
		pod := &pods[i]
		if pod.DeletionTimestamp != nil || isTerminalPhase(pod.Status.Phase) {
			continue
		}
		worker := mirrorWorker{
			pod:   pod,
			ready: podReady(pod),
			pool:  etcdDiscovery && pod.Annotations[consts.KubeAnnotationMirrorRollouts] == consts.KubeLabelValueTrue,
		}
		if registration, found := registrations[client.ObjectKeyFromObject(pod)]; found {
			worker.registered = true
			worker.namespace = registration.namespace
			worker.workerID = registration.workerID
			worker.taints = registration.taints
		}
		workers = append(workers, worker)
	}
	return workers, nil
}

func podReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

// idlePoolWorkers counts, by generation, the Ready workers of a component that
// serve nothing because they are parked or mirroring. A pool generation is
// available only as far as its workers serve, including an older generation
// left behind when the deployment opts out. It reads etcd only for a component
// that opted in or still runs pool workers.
func (r *dgdWorkerRolloutReconciler) idlePoolWorkers(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) (map[string]int32, error) {
	if r.pool == nil {
		return nil, nil
	}
	if !r.mirrorRolloutComponent(dgd, componentName) {
		pods, err := r.listDGDComponentPods(ctx, dgd, componentName)
		if err != nil {
			return nil, err
		}
		if !slices.ContainsFunc(pods, func(pod corev1.Pod) bool {
			return pod.Annotations[consts.KubeAnnotationMirrorRollouts] == consts.KubeLabelValueTrue
		}) {
			return nil, nil
		}
	}
	workers, err := r.readMirrorWorkers(ctx, dgd, componentName)
	if err != nil {
		return nil, err
	}
	idle := map[string]int32{}
	for _, worker := range workers {
		if worker.ready && !worker.serving() {
			idle[worker.workerHash()]++
		}
	}
	return idle, nil
}

// mirrorRolloutSpec reports whether component of dgd rolls by mirror pairs on
// this reconciler's pathway.
func (r *dgdWorkerRolloutReconciler) mirrorRolloutSpec(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) bool {
	return r.config != nil && dynamo.MirrorRolloutComponent(dgd, component, r.config.Discovery.Backend)
}

// mirrorRolloutComponent reports whether the named component of dgd rolls by mirror pairs.
func (r *dgdWorkerRolloutReconciler) mirrorRolloutComponent(dgd *nvidiacomv1beta1.DynamoGraphDeployment, componentName string) bool {
	for i := range dgd.Spec.Components {
		if dgd.Spec.Components[i].ComponentName == componentName {
			return r.mirrorRolloutSpec(dgd, &dgd.Spec.Components[i])
		}
	}
	return false
}

// mirrorRolloutReconciler runs the pairs of a mirror rollout. It assigns each
// parked new-generation worker the old worker it replaces, promotes a pair
// once a judge approves it, and promotes parked workers no rollout needs.
type mirrorRolloutReconciler struct {
	rollout *dgdWorkerRolloutReconciler
}

func newMirrorRolloutReconciler(rollout *dgdWorkerRolloutReconciler) *mirrorRolloutReconciler {
	return &mirrorRolloutReconciler{rollout: rollout}
}

// Reconcile drives every mirror-rollout component of dgd from the observed
// workers, their registrations and the component's pairs.
func (m *mirrorRolloutReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) error {
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !m.rollout.mirrorRolloutSpec(dgd, component) {
			continue
		}
		if err := m.reconcileComponent(ctx, dgd, component.ComponentName, rollingUpdateCtx); err != nil {
			return fmt.Errorf("mirror rollout of component %s: %w", component.ComponentName, err)
		}
	}
	return nil
}

func (m *mirrorRolloutReconciler) reconcileComponent(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) error {
	workers, err := m.rollout.readMirrorWorkers(ctx, dgd, componentName)
	if err != nil {
		return err
	}
	byPod := make(map[string]*mirrorWorker, len(workers))
	for i := range workers {
		byPod[workers[i].pod.Name] = &workers[i]
	}
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	listed := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	if err := m.rollout.List(ctx, listed, client.InNamespace(dgd.Namespace), client.MatchingLabels{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoComponent:           componentName,
	}); err != nil {
		return fmt.Errorf("list mirror pairs: %w", err)
	}

	// Only this DGD's pairs count; a DGD recreated under the same name starts clean.
	for i := range listed.Items {
		if metav1.IsControlledBy(&listed.Items[i], dgd) {
			pairs.Items = append(pairs.Items, listed.Items[i])
		}
	}
	newHash := rollingUpdateCtx.NewWorkerHash
	_, rolling := rollingUpdateCtx.OldWorkerReplicaTargetsByComponent[componentName]

	// A rejection of any pair stops the whole generation, including its other open pairs.
	rejectedHashes := map[string]bool{}
	newGenerationPaired := false
	for i := range pairs.Items {
		pair := &pairs.Items[i]
		if pair.Spec.WorkerHash == newHash {
			newGenerationPaired = true
		}
		verdict, decided := pair.Verdict()
		if pair.Status.Reason == mirrorPairAbortRejected || (decided && verdict == nvidiacomv1alpha1.DynamoMirrorPairConditionRejected) {
			rejectedHashes[pair.Spec.WorkerHash] = true
		}
	}
	rejected := rejectedHashes[newHash]

	// An open pair of a superseded generation ends, and its mirror parks.
	for i := range pairs.Items {
		pair := &pairs.Items[i]
		if pair.Spec.WorkerHash == newHash || pair.Status.Phase == nvidiacomv1alpha1.DynamoMirrorPairPhasePromoted || pair.Status.Phase == nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted {
			continue
		}
		if mirror := byPod[pair.Spec.Mirror.PodName]; mirror != nil && mirror.registered && mirror.mirroring() && !mirror.parked() {
			if err := m.rollout.pool.SetRole(ctx, mirror, []string{consts.ParkedMirrorTaint}); err != nil {
				return err
			}
		}
		if err := m.setPairPhase(ctx, pair, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, mirrorPairAbortSuperseded); err != nil {
			return err
		}
	}

	// Settle each open pair against its workers and verdict.
	pairedMirrors := map[string]bool{}
	pairedShadows := map[string]bool{}
	for i := range pairs.Items {
		pair := &pairs.Items[i]
		if pair.Spec.WorkerHash != newHash {
			continue
		}
		if pair.Status.Phase == nvidiacomv1alpha1.DynamoMirrorPairPhasePromoted || pair.Status.Phase == nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted {
			continue
		}
		open, err := m.settlePair(ctx, dgd, pair, byPod, rejected)
		if err != nil {
			return err
		}
		if open {
			pairedMirrors[pair.Spec.Mirror.PodName] = true
			pairedShadows[pair.Spec.Shadowed.PodName] = true
		}
	}

	// Outside a rollout, promote every parked worker of a generation no judge
	// rejected. During one, parked workers wait for pairs or for removal.
	if !rolling {
		for i := range workers {
			worker := &workers[i]
			if !worker.parked() || rejectedHashes[worker.workerHash()] {
				continue
			}
			if err := m.rollout.pool.SetRole(ctx, worker, nil); err != nil {
				return err
			}
		}
		return nil
	}
	// A rejected generation shrinks to its serving workers; its parked and
	// mirroring Pods go first.
	if rejected {
		for i := range workers {
			worker := &workers[i]
			if worker.workerHash() != newHash || worker.serving() ||
				worker.pod.Annotations[consts.KubeAnnotationPodDeletionCost] == consts.PromotedShadowDeletionCost {
				continue
			}
			patch := client.MergeFrom(worker.pod.DeepCopy())
			if worker.pod.Annotations == nil {
				worker.pod.Annotations = map[string]string{}
			}
			worker.pod.Annotations[consts.KubeAnnotationPodDeletionCost] = consts.PromotedShadowDeletionCost
			if err := m.rollout.Patch(ctx, worker.pod, patch); err != nil {
				return fmt.Errorf("mark rejected pod %s for removal: %w", worker.pod.Name, err)
			}
		}
		return nil
	}

	// Pair each unpaired parked new-generation worker with the oldest serving
	// old-generation worker that nothing shadows yet.
	candidates := []*mirrorWorker{}
	legacy := false
	oldServing, oldPending := false, false
	for i := range workers {
		worker := &workers[i]
		if worker.workerHash() == newHash {
			continue
		}
		if worker.pod.Annotations[consts.KubeAnnotationPodDeletionCost] == consts.PromotedShadowDeletionCost {
			continue
		}
		oldServing = oldServing || worker.serving()
		oldPending = oldPending || (worker.ready && worker.poolWorker() && !worker.registered)
		if !worker.serving() || pairedShadows[worker.pod.Name] {
			continue
		}
		// A serving worker that started before the opt-in takes no pool role,
		// so nothing can mirror it. A pool worker not yet registered waits.
		if !worker.registered {
			legacy = legacy || !worker.poolWorker()
			continue
		}
		candidates = append(candidates, worker)
	}

	// Take shadows in the order the rollout removes old replicas: oldest
	// generation first, then oldest Pod, so a promoted mirror's shadowed Pod is
	// in the generation that scales down.
	generationAge, err := m.oldGenerationCreation(ctx, dgd, newHash)
	if err != nil {
		return err
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := generationAge[candidates[i].workerHash()], generationAge[candidates[j].workerHash()]
		return a.Before(&b)
	})

	// Old workers that predate the opt-in cannot be mirrored, and with no old
	// worker serving or registering there is nothing to mirror: this
	// generation's parked workers serve without pairs.
	if len(candidates) == 0 && (legacy || (!oldServing && !oldPending)) {
		for i := range workers {
			worker := &workers[i]
			if worker.workerHash() != newHash || !worker.parked() || pairedMirrors[worker.pod.Name] {
				continue
			}
			if err := m.rollout.pool.SetRole(ctx, worker, nil); err != nil {
				return err
			}
		}
		return nil
	}
	for i := range workers {
		mirror := &workers[i]
		if len(candidates) == 0 {
			break
		}
		if mirror.workerHash() != newHash || !mirror.ready || !mirror.parked() || pairedMirrors[mirror.pod.Name] {
			continue
		}
		shadowed := candidates[0]
		candidates = candidates[1:]
		// Every listed pair of these Pods, a previous DGD's included, holds a name.
		attempt := 0
		for j := range listed.Items {
			if listed.Items[j].Spec.Mirror.PodName == mirror.pod.Name && listed.Items[j].Spec.Shadowed.PodName == shadowed.pod.Name {
				attempt++
			}
		}
		if err := m.createPair(ctx, dgd, componentName, newHash, !newGenerationPaired, attempt, mirror, shadowed); err != nil {
			return err
		}
		newGenerationPaired = true
	}
	return nil
}

// mirrorGenerationRejected reports whether a judge rejected a pair of this
// DGD's component in generation workerHash.
func (r *dgdWorkerRolloutReconciler) mirrorGenerationRejected(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	workerHash string,
) (bool, error) {
	if !r.mirrorRolloutComponent(dgd, componentName) {
		return false, nil
	}
	pairs := &nvidiacomv1alpha1.DynamoMirrorPairList{}
	if err := r.List(ctx, pairs, client.InNamespace(dgd.Namespace), client.MatchingLabels{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoComponent:           componentName,
	}); err != nil {
		return false, fmt.Errorf("list mirror pairs: %w", err)
	}
	for i := range pairs.Items {
		pair := &pairs.Items[i]
		if pair.Spec.WorkerHash != workerHash || !metav1.IsControlledBy(pair, dgd) {
			continue
		}
		verdict, decided := pair.Verdict()
		if pair.Status.Reason == mirrorPairAbortRejected || (decided && verdict == nvidiacomv1alpha1.DynamoMirrorPairConditionRejected) {
			return true, nil
		}
	}
	return false, nil
}

// applyPoolReadiness keeps a mirror-rollout component out of Ready while any
// Ready worker of generation newHash is parked or mirroring, and counts only
// serving workers of any generation as available.
func (m *mirrorRolloutReconciler) applyPoolReadiness(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newHash string,
	result ReconcileResult,
) (ReconcileResult, error) {
	var notServing []string
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !m.rollout.mirrorRolloutSpec(dgd, component) {
			continue
		}
		idle, err := m.rollout.idlePoolWorkers(ctx, dgd, component.ComponentName)
		if err != nil {
			return result, err
		}
		var count int32
		for _, n := range idle {
			count += n
		}
		if count == 0 {
			continue
		}
		if idle[newHash] > 0 {
			notServing = append(notServing, fmt.Sprintf("%s: %d workers parked or mirroring", component.ComponentName, idle[newHash]))
		}
		if status, found := result.ComponentStatus[component.ComponentName]; found {
			if status.ReadyReplicas != nil {
				status.ReadyReplicas = ptr.To(max(*status.ReadyReplicas-count, 0))
			}
			if status.AvailableReplicas != nil {
				status.AvailableReplicas = ptr.To(max(*status.AvailableReplicas-count, 0))
			}
			result.ComponentStatus[component.ComponentName] = status
		}
	}
	if len(notServing) == 0 {
		return result, nil
	}
	result.State = nvidiacomv1beta1.DGDStatePending
	result.Reason = "pool_workers_not_serving"
	result.Message = Message(strings.Join(notServing, "; "))
	return result, nil
}

// settlePair moves one open pair forward and reports whether it stays open.
// generationRejected aborts the pair whatever its own verdict.
func (m *mirrorRolloutReconciler) settlePair(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	pair *nvidiacomv1alpha1.DynamoMirrorPair,
	byPod map[string]*mirrorWorker,
	generationRejected bool,
) (bool, error) {
	mirror := byPod[pair.Spec.Mirror.PodName]
	shadowed := byPod[pair.Spec.Shadowed.PodName]
	assigned := consts.MirrorTaintPrefix + pair.Spec.Shadowed.DynamoNamespace + "/" + pair.Spec.Shadowed.WorkerID
	verdict, decided := pair.Verdict()

	// A mirror whose Pod lives but whose registration is not observed waits
	// for it; one that lost its registration after mirroring parks under the
	// identity the pair recorded, so it can pair again once it re-registers.
	if mirror != nil && !mirror.registered && !generationRejected {
		if pair.Status.Phase != nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring {
			return true, nil
		}
		recorded := &mirrorWorker{pod: mirror.pod, namespace: pair.Spec.Mirror.DynamoNamespace, workerID: pair.Spec.Mirror.WorkerID}
		if err := m.rollout.pool.SetRole(ctx, recorded, []string{consts.ParkedMirrorTaint}); err != nil {
			return true, err
		}
		return false, m.setPairPhase(ctx, pair, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, mirrorPairAbortMirrorGone)
	}

	// A worker that restarted under the same Pod registers a new worker ID:
	// it is not the worker the pair recorded.
	if mirror != nil && mirror.registered && mirror.workerID != pair.Spec.Mirror.WorkerID {
		mirror = nil
	}
	if shadowed != nil && shadowed.registered && shadowed.workerID != pair.Spec.Shadowed.WorkerID {
		shadowed = nil
	}

	// A rejected pair ends the generation's promotions; its mirror parks again,
	// as does the mirror of every other open pair of that generation. A mirror
	// approved and already serving before the rejection stays promoted.
	rejectedHere := decided && verdict == nvidiacomv1alpha1.DynamoMirrorPairConditionRejected
	approvedAndServing := decided && !rejectedHere && mirror != nil && mirror.registered && !mirror.mirroring()
	if (generationRejected && !approvedAndServing) || rejectedHere {
		if mirror != nil && mirror.registered {
			if err := m.rollout.pool.SetRole(ctx, mirror, []string{consts.ParkedMirrorTaint}); err != nil {
				return true, err
			}
		}
		if !rejectedHere {
			return false, m.setPairPhase(ctx, pair, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, mirrorPairAbortGenerationRejected)
		}
		if err := m.setPairPhase(ctx, pair, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, mirrorPairAbortRejected); err != nil {
			return true, err
		}
		m.rollout.recorder.Eventf(dgd, pair, corev1.EventTypeWarning, "MirrorPairRejected", "Rollout",
			"Mirror %s of %s was rejected; no further worker of generation %s will be promoted",
			pair.Spec.Mirror.PodName, pair.Spec.Shadowed.PodName, pair.Spec.WorkerHash)
		return false, nil
	}

	// A pair whose mirror left cannot be promoted.
	if mirror == nil || !mirror.registered {
		return false, m.setPairPhase(ctx, pair, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, mirrorPairAbortMirrorGone)
	}

	// An approved pair marks its shadowed worker for removal, then promotes the
	// mirror. An approval counts once the mirror is observed mirroring.
	if decided && pair.Status.Phase == nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring {
		if shadowed != nil && shadowed.pod.Annotations[consts.KubeAnnotationPodDeletionCost] != consts.PromotedShadowDeletionCost {
			patch := client.MergeFrom(shadowed.pod.DeepCopy())
			if shadowed.pod.Annotations == nil {
				shadowed.pod.Annotations = map[string]string{}
			}
			shadowed.pod.Annotations[consts.KubeAnnotationPodDeletionCost] = consts.PromotedShadowDeletionCost
			if err := m.rollout.Patch(ctx, shadowed.pod, patch); err != nil {
				return true, fmt.Errorf("mark shadowed pod %s for removal: %w", shadowed.pod.Name, err)
			}
		}
		if mirror.mirroring() {
			return true, m.rollout.pool.SetRole(ctx, mirror, nil)
		}
		if err := m.setPairPhase(ctx, pair, nvidiacomv1alpha1.DynamoMirrorPairPhasePromoted, ""); err != nil {
			return true, err
		}
		m.rollout.recorder.Eventf(dgd, pair, corev1.EventTypeNormal, "MirrorPairPromoted", "Rollout",
			"Promoted %s in place of %s", pair.Spec.Mirror.PodName, pair.Spec.Shadowed.PodName)
		return false, nil
	}

	// An undecided pair whose shadowed worker left parks its mirror for a new pair.
	if shadowed == nil || !shadowed.serving() {
		if err := m.rollout.pool.SetRole(ctx, mirror, []string{consts.ParkedMirrorTaint}); err != nil {
			return true, err
		}
		return false, m.setPairPhase(ctx, pair, nvidiacomv1alpha1.DynamoMirrorPairPhaseAborted, mirrorPairAbortShadowedGone)
	}

	// An assigned mirror starts the judges' clock; an unassigned one is assigned.
	if mirror.hasTaint(assigned) {
		if pair.Status.Phase == nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring {
			return true, nil
		}
		base := pair.DeepCopy()
		pair.Status.Phase = nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring
		now := metav1.Now()
		pair.Status.MirroringSince = &now
		return true, m.patchPairStatus(ctx, pair, base)
	}
	return true, m.rollout.pool.SetRole(ctx, mirror, []string{assigned})
}

// oldGenerationCreation returns the creation time of each old worker
// generation's DCD, by worker hash.
func (m *mirrorRolloutReconciler) oldGenerationCreation(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newHash string,
) (map[string]metav1.Time, error) {
	dcds, err := m.rollout.listOldWorkerDCDs(ctx, dgd, newHash)
	if err != nil {
		return nil, fmt.Errorf("list old worker DCDs: %w", err)
	}
	created := make(map[string]metav1.Time, len(dcds))
	for i := range dcds {
		created[dcds[i].Labels[consts.KubeLabelDynamoWorkerHash]] = dcds[i].CreationTimestamp
	}
	return created, nil
}

// createPair records a new pair; the next reconcile assigns its mirror.
func (m *mirrorRolloutReconciler) createPair(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	workerHash string,
	firstPair bool,
	attempt int,
	mirror *mirrorWorker,
	shadowed *mirrorWorker,
) error {
	// A retried pair of the same two Pods takes the next attempt's name. A
	// name over the object-name limit keeps the mirror's Pod name and a hash.
	name := mirror.pod.Name + "." + shadowed.pod.Name
	if attempt > 0 {
		name = fmt.Sprintf("%s.%d", name, attempt)
	}
	if len(name) > validation.DNS1123SubdomainMaxLength {
		sum := sha256.Sum256([]byte(name))
		suffix := fmt.Sprintf(".%x", sum[:8])
		prefix := mirror.pod.Name[:min(len(mirror.pod.Name), validation.DNS1123SubdomainMaxLength-len(suffix))]
		name = strings.TrimRight(prefix, ".-") + suffix
	}
	pair := &nvidiacomv1alpha1.DynamoMirrorPair{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: dgd.Namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           componentName,
				consts.KubeLabelDynamoWorkerHash:          workerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoMirrorPairSpec{
			GraphDeploymentName: dgd.Name,
			ComponentName:       componentName,
			WorkerHash:          workerHash,
			FirstPair:           firstPair,
			Mirror: nvidiacomv1alpha1.DynamoMirrorPairWorker{
				PodName:         mirror.pod.Name,
				DynamoNamespace: mirror.namespace,
				WorkerID:        mirror.workerID,
			},
			Shadowed: nvidiacomv1alpha1.DynamoMirrorPairWorker{
				PodName:         shadowed.pod.Name,
				DynamoNamespace: shadowed.namespace,
				WorkerID:        shadowed.workerID,
			},
		},
	}
	if err := controllerutil.SetControllerReference(dgd, pair, m.rollout.Scheme()); err != nil {
		return err
	}
	if err := m.rollout.Create(ctx, pair); err != nil {
		// The pair exists already; its watch event brings it into the next reconcile.
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("create mirror pair %s: %w", pair.Name, err)
	}
	m.rollout.recorder.Eventf(dgd, pair, corev1.EventTypeNormal, "MirrorPairCreated", "Rollout",
		"%s mirrors %s (%s)", mirror.pod.Name, shadowed.pod.Name, shadowed.mirrorTaint())

	// Record the phase so judges see a pending pair.
	base := pair.DeepCopy()
	pair.Status.Phase = nvidiacomv1alpha1.DynamoMirrorPairPhasePending
	return m.patchPairStatus(ctx, pair, base)
}

func (m *mirrorRolloutReconciler) setPairPhase(
	ctx context.Context,
	pair *nvidiacomv1alpha1.DynamoMirrorPair,
	phase nvidiacomv1alpha1.DynamoMirrorPairPhase,
	reason string,
) error {
	base := pair.DeepCopy()
	pair.Status.Phase = phase
	pair.Status.Reason = reason
	return m.patchPairStatus(ctx, pair, base)
}

// patchPairStatus writes the operator's status fields under an optimistic
// lock, so a judge's concurrent verdict is never overwritten.
func (m *mirrorRolloutReconciler) patchPairStatus(
	ctx context.Context,
	pair *nvidiacomv1alpha1.DynamoMirrorPair,
	base *nvidiacomv1alpha1.DynamoMirrorPair,
) error {
	patch := client.MergeFromWithOptions(base, client.MergeFromWithOptimisticLock{})
	if err := m.rollout.Status().Patch(ctx, pair, patch); err != nil {
		return fmt.Errorf("update mirror pair %s status: %w", pair.Name, err)
	}
	return nil
}
