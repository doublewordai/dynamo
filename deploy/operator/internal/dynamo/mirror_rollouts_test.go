/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestMirrorRolloutWorkersAreMarked(t *testing.T) {
	tests := []struct {
		name       string
		optedIn    bool
		specMark   bool
		component  v1beta1.DynamoComponentDeploymentSharedSpec
		wantMarked bool
	}{
		{
			name:       "a single-node worker of an opted-in deployment",
			optedIn:    true,
			component:  v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: commonconsts.ComponentTypeWorker},
			wantMarked: true,
		},
		{
			name:      "a worker of a deployment that did not opt in",
			component: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: commonconsts.ComponentTypeWorker},
		},
		{
			name:      "a worker of a deployment whose spec annotations carry the mark without the deployment opting in",
			specMark:  true,
			component: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: commonconsts.ComponentTypeWorker},
		},
		{
			name: "a worker whose own pod template carries the mark without the deployment opting in",
			component: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				ComponentType: commonconsts.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{commonconsts.KubeAnnotationMirrorRollouts: "true"},
				}},
			},
		},
		{
			name:    "a multinode worker",
			optedIn: true,
			component: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				ComponentType: commonconsts.ComponentTypeWorker,
				Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
			},
		},
		{
			name:    "a worker rolled by Recreate",
			optedIn: true,
			component: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				ComponentType: commonconsts.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{commonconsts.KubeAnnotationMirrorDeploymentStrategy: "Recreate"},
				}},
			},
		},
		{
			name:      "a frontend",
			optedIn:   true,
			component: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: commonconsts.ComponentTypeFrontend},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Render the deployment's component")
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "serving"},
				Spec:       v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{tt.component}},
			}
			if tt.optedIn {
				dgd.Annotations = map[string]string{commonconsts.KubeAnnotationMirrorRollouts: "true"}
			}
			if tt.specMark {
				dgd.Spec.Annotations = map[string]string{commonconsts.KubeAnnotationMirrorRollouts: "true"}
			}
			dcds, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{NewWorkerHash: "gen1"})
			require.NoError(t, err)

			t.Log("Only a mirror-rollout worker's pod template carries the mark")
			annotations := GetPodTemplateAnnotations(&dcds[tt.component.ComponentName].Spec.DynamoComponentDeploymentSharedSpec)
			assert.Equal(t, tt.wantMarked, annotations[commonconsts.KubeAnnotationMirrorRollouts] == commonconsts.KubeLabelValueTrue)
		})
	}
}

func TestMirrorRolloutWorkersBootParkedUnderEtcdDiscovery(t *testing.T) {
	tests := []struct {
		name       string
		context    ComponentContext
		wantParked bool
	}{
		{
			name:       "a marked single-node worker on etcd discovery",
			context:    ComponentContext{MirrorRollouts: true, numberOfNodes: 1, Discovery: DiscoveryContext{Backend: configv1alpha1.DiscoveryBackendEtcd}},
			wantParked: true,
		},
		{
			name:    "a marked worker on Kubernetes discovery",
			context: ComponentContext{MirrorRollouts: true, numberOfNodes: 1, Discovery: DiscoveryContext{Backend: configv1alpha1.DiscoveryBackendKubernetes}},
		},
		{
			name:    "a marked multinode worker on etcd discovery",
			context: ComponentContext{MirrorRollouts: true, numberOfNodes: 2, Discovery: DiscoveryContext{Backend: configv1alpha1.DiscoveryBackendEtcd}},
		},
		{
			name:    "an unmarked worker",
			context: ComponentContext{numberOfNodes: 1, Discovery: DiscoveryContext{Backend: configv1alpha1.DiscoveryBackendEtcd}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the worker container")
			container, err := NewWorkerDefaults().GetBaseContainer(tt.context)
			require.NoError(t, err)

			t.Log("Only a parked worker declares the parked pool role")
			var roles []string
			for _, env := range container.Env {
				if env.Name == commonconsts.PoolRoleEnvVar {
					roles = append(roles, env.Value)
				}
			}
			if tt.wantParked {
				assert.Equal(t, []string{commonconsts.ParkedPoolRole}, roles)
			} else {
				assert.Empty(t, roles)
			}
		})
	}
}

func TestMirrorRolloutComponentFollowsTheComponentDiscoveryBackend(t *testing.T) {
	tests := []struct {
		name       string
		dgdDefault configv1alpha1.DiscoveryBackend
		component  map[string]string
		want       bool
	}{
		{name: "etcd by default", dgdDefault: configv1alpha1.DiscoveryBackendEtcd, want: true},
		{
			name:       "a worker that overrides etcd with Kubernetes discovery",
			dgdDefault: configv1alpha1.DiscoveryBackendEtcd,
			component:  map[string]string{commonconsts.KubeAnnotationDynamoDiscoveryBackend: string(configv1alpha1.DiscoveryBackendKubernetes)},
		},
		{
			name:       "a worker that overrides Kubernetes with etcd discovery",
			dgdDefault: configv1alpha1.DiscoveryBackendKubernetes,
			component:  map[string]string{commonconsts.KubeAnnotationDynamoDiscoveryBackend: string(configv1alpha1.DiscoveryBackendEtcd)},
			want:       true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("An opted-in deployment with one worker component")
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: commonconsts.ComponentTypeWorker}
			if tt.component != nil {
				component.PodTemplate = &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Annotations: tt.component}}
			}
			dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{
				Name: "graph", Annotations: map[string]string{commonconsts.KubeAnnotationMirrorRollouts: "true"},
			}}

			t.Log("The component takes part only on its own etcd discovery")
			assert.Equal(t, tt.want, MirrorRolloutComponent(dgd, &component, tt.dgdDefault))
		})
	}
}

func TestMirrorRolloutParkedRoleOverridesAPodTemplatePoolRole(t *testing.T) {
	tests := []struct {
		name     string
		marked   bool
		wantRole string
	}{
		{name: "a marked worker boots parked whatever its pod template sets", marked: true, wantRole: commonconsts.ParkedPoolRole},
		{name: "an unmarked worker keeps its pod template's role", wantRole: "user-role"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("A worker whose main container sets its own pool role")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				ComponentType: commonconsts.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: commonconsts.MainContainerName,
						Env:  []corev1.EnvVar{{Name: commonconsts.PoolRoleEnvVar, Value: "user-role"}},
					}}},
				},
			}
			if tt.marked {
				component.PodTemplate.Annotations[commonconsts.KubeAnnotationMirrorRollouts] = commonconsts.KubeLabelValueTrue
			}
			config := &configv1alpha1.OperatorConfiguration{Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}

			t.Log("Render the worker's pod")
			podSpec, err := GenerateBasePodSpec(component, BackendFrameworkVLLM, &mockSecretsRetriever{}, "graph", "serving",
				RoleMain, 1, config, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)

			t.Log("The main container declares exactly one pool role")
			var roles []string
			for _, env := range podSpec.Containers[0].Env {
				if env.Name == commonconsts.PoolRoleEnvVar {
					roles = append(roles, env.Value)
				}
			}
			assert.Equal(t, []string{tt.wantRole}, roles)
		})
	}
}

func TestMirrorRolloutMarkParksOnlyWorkersOfTheComponentPathway(t *testing.T) {
	etcdConfig := &configv1alpha1.OperatorConfiguration{Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
	markedWorker := func() *v1beta1.DynamoComponentDeploymentSharedSpec {
		return &v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: "worker",
			ComponentType: commonconsts.ComponentTypeWorker,
			PodTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{commonconsts.KubeAnnotationMirrorRollouts: commonconsts.KubeLabelValueTrue},
			}},
		}
	}
	poolRoles := func(podSpec *corev1.PodSpec) []string {
		var roles []string
		for _, env := range podSpec.Containers[0].Env {
			if env.Name == commonconsts.PoolRoleEnvVar {
				roles = append(roles, env.Value)
			}
		}
		return roles
	}
	dcd := func(owned bool) *v1beta1.DynamoComponentDeployment {
		dcd := &v1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "graph-worker", Namespace: "serving"},
			Spec: v1beta1.DynamoComponentDeploymentSpec{
				BackendFramework:                    string(BackendFrameworkVLLM),
				DynamoComponentDeploymentSharedSpec: *markedWorker(),
			},
		}
		if owned {
			dcd.OwnerReferences = []metav1.OwnerReference{{Kind: "DynamoGraphDeployment", Name: "graph"}}
		}
		return dcd
	}

	t.Log("A DCD a DGD owns boots its marked worker parked")
	podSpec, err := GenerateBasePodSpecForController(dcd(true), &mockSecretsRetriever{}, etcdConfig, RoleMain,
		commonconsts.MultinodeDeploymentTypeGrove, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
	require.NoError(t, err)
	assert.Equal(t, []string{commonconsts.ParkedPoolRole}, poolRoles(podSpec))

	t.Log("A standalone DCD with the mark does not")
	podSpec, err = GenerateBasePodSpecForController(dcd(false), &mockSecretsRetriever{}, etcdConfig, RoleMain,
		commonconsts.MultinodeDeploymentTypeGrove, staticContainerGPUCount(0), GenerateBasePodSpecForControllerOptions{})
	require.NoError(t, err)
	assert.Empty(t, poolRoles(podSpec))

	t.Log("Nor does a worker rendered for Grove, even in a DGD that opted in")
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name: "graph", Namespace: "serving",
		Annotations: map[string]string{commonconsts.KubeAnnotationMirrorRollouts: commonconsts.KubeLabelValueTrue},
	}}
	podSpec, err = GeneratePodSpecForComponent(markedWorker(), BackendFrameworkVLLM, &mockSecretsRetriever{}, dgd, RoleMain, 1,
		etcdConfig, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
	require.NoError(t, err)
	assert.Empty(t, poolRoles(podSpec))
}
