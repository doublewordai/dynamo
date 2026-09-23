/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	corev1 "k8s.io/api/core/v1"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
)

// MirrorRolloutComponent reports whether component takes part in dgd's mirror
// rollouts: a single-node worker of a deployment that opted in and uses etcd
// discovery, through which the rollout reads and sets its workers' roles.
func MirrorRolloutComponent(
	dgd *v1beta1.DynamoGraphDeployment,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	defaultDiscoveryBackend configv1alpha1.DiscoveryBackend,
) bool {
	return mirrorRolloutWorker(dgd, component) &&
		ComponentDiscoveryBackend(dgd, component, defaultDiscoveryBackend) == configv1alpha1.DiscoveryBackendEtcd
}

// mirrorRolloutWorker reports a single-node worker, rolled by replacement, of
// a deployment that opted into mirror rollouts. A Recreate component drains
// its old workers first, so nothing would be left to mirror.
func mirrorRolloutWorker(
	dgd *v1beta1.DynamoGraphDeployment,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
) bool {
	return dgd.GetAnnotations()[commonconsts.KubeAnnotationMirrorRollouts] == commonconsts.KubeLabelValueTrue &&
		IsWorkerComponent(string(component.ComponentType)) &&
		!component.IsMultinode() &&
		GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component)[commonconsts.KubeAnnotationMirrorDeploymentStrategy] != "Recreate"
}

// ComponentDiscoveryBackend resolves the discovery backend a component's
// workers are rendered with: the component's own annotations over the graph's.
func ComponentDiscoveryBackend(
	dgd *v1beta1.DynamoGraphDeployment,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	defaultDiscoveryBackend configv1alpha1.DiscoveryBackend,
) configv1alpha1.DiscoveryBackend {
	return controller_common.GetDiscoveryBackend(defaultDiscoveryBackend, GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component))
}

// markMirrorRolloutWorker marks the pod template of a single-node worker in a
// deployment that opted into mirror rollouts, so its workers boot parked. The
// mark follows the deployment's opt-in alone: a pod template carries it only
// when the operator set it.
func markMirrorRolloutWorker(
	dgd *v1beta1.DynamoGraphDeployment,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
) {
	if !mirrorRolloutWorker(dgd, component) {
		if component.PodTemplate != nil {
			delete(component.PodTemplate.Annotations, commonconsts.KubeAnnotationMirrorRollouts)
		}
		return
	}
	ensurePodTemplate(component).Annotations[commonconsts.KubeAnnotationMirrorRollouts] = commonconsts.KubeLabelValueTrue
}

// parkedPoolRoleEnv makes a mirror-rollout worker boot parked: registered, but
// serving nothing until the operator assigns or promotes it through etcd.
// Workers on another discovery backend have no such channel, so they boot as
// ordinary workers.
func parkedPoolRoleEnv(context ComponentContext) []corev1.EnvVar {
	if !context.MirrorRollouts ||
		context.Discovery.Backend != configv1alpha1.DiscoveryBackendEtcd ||
		context.numberOfNodes > 1 {
		return nil
	}
	return []corev1.EnvVar{{Name: commonconsts.PoolRoleEnvVar, Value: commonconsts.ParkedPoolRole}}
}
