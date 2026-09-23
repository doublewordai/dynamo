/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package consts

const (
	// KubeAnnotationMirrorRollouts set to "true" on a DynamoGraphDeployment rolls
	// its single-node worker components by mirror pairs: every new worker
	// mirrors the old worker it replaces until a judge approves it.
	KubeAnnotationMirrorRollouts = "nvidia.com/mirror-rollouts"

	// KubeAnnotationMirrorDeploymentStrategy is the annotation selecting a
	// component's rollout strategy; mirror rollouts need one that keeps old
	// workers to mirror.
	KubeAnnotationMirrorDeploymentStrategy = "nvidia.com/deployment-strategy"

	// PoolRoleEnvVar declares a worker's pool role to the Dynamo runtime.
	PoolRoleEnvVar = "DYN_POOL_ROLE"
	// ParkedPoolRole makes a worker a mirror of a worker that never exists, so
	// it registers but neither serves nor receives copies until the operator
	// assigns or clears its mirror taint.
	ParkedPoolRole = "mirror:dynamo-parked/0"

	// MirrorTaintPrefix begins the model taint a mirror worker publishes; the
	// rest names the worker it shadows as <dynamo namespace>/<worker id>.
	MirrorTaintPrefix = "dynamo.pool/mirror-of="
	// ParkedMirrorTaint is the taint a worker started with ParkedPoolRole publishes.
	ParkedMirrorTaint = MirrorTaintPrefix + "dynamo-parked/0"
	// TopologyTaintPrefix begins taints the runtime derives itself; a worker
	// rejects a role that sets them.
	TopologyTaintPrefix = "dynamo.topology/"

	// PoolMembersPrefix holds, under <dynamo namespace>/<worker id in hex>, the
	// Pod of every worker that booted with a pool role. Each record is bound to
	// its worker's discovery lease.
	PoolMembersPrefix = "v1/pool_members/"
	// PoolRolesPrefix holds, under <dynamo namespace>/<worker id in hex>/role,
	// the caller-managed taints the operator sets on a pool worker. The worker
	// watches its record and applies each value to its model card.
	PoolRolesPrefix = "v1/pool_roles/"
	// ModelCardsPrefix holds the model cards workers register in discovery,
	// under <dynamo namespace>/<component>/<endpoint>/<worker id in hex>.
	ModelCardsPrefix = "v1/mdc/"

	// KubeAnnotationPodDeletionCost ranks Pods for removal when their
	// ReplicaSet scales down; the lowest cost goes first.
	KubeAnnotationPodDeletionCost = "controller.kubernetes.io/pod-deletion-cost"
	// PromotedShadowDeletionCost marks the old worker whose mirror was promoted.
	PromotedShadowDeletionCost = "-1000000"
)
