/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DynamoMirrorPairPhase is where a mirror pair is in its lifecycle.
type DynamoMirrorPairPhase string

const (
	// DynamoMirrorPairPhasePending: the operator created the pair and is
	// assigning the new worker its shadowed worker.
	DynamoMirrorPairPhasePending DynamoMirrorPairPhase = "Pending"
	// DynamoMirrorPairPhaseMirroring: the new worker publishes the mirror taint
	// and receives a copy of every request placed on the shadowed worker. A
	// judge may now set a verdict.
	DynamoMirrorPairPhaseMirroring DynamoMirrorPairPhase = "Mirroring"
	// DynamoMirrorPairPhasePromoted: the pair was approved and the new worker
	// serves; the shadowed worker is the next one its generation retires.
	DynamoMirrorPairPhasePromoted DynamoMirrorPairPhase = "Promoted"
	// DynamoMirrorPairPhaseAborted: the pair ended without promotion, because it
	// was rejected or one of its workers left.
	DynamoMirrorPairPhaseAborted DynamoMirrorPairPhase = "Aborted"
)

const (
	// DynamoMirrorPairConditionApproved is set to True by a judge to promote the
	// new worker in place of the shadowed one.
	DynamoMirrorPairConditionApproved = "Approved"
	// DynamoMirrorPairConditionRejected is set to True by a judge to abort the
	// pair. The operator then promotes no further worker of that generation.
	DynamoMirrorPairConditionRejected = "Rejected"
)

// DynamoMirrorPairSpec names a new-generation worker mirroring the old-generation
// worker it replaces during a mirror rollout. The operator writes it; judges
// read it to decide the verdict.
type DynamoMirrorPairSpec struct {
	// GraphDeploymentName is the DynamoGraphDeployment being rolled out.
	GraphDeploymentName string `json:"graphDeploymentName"`
	// ComponentName is the worker component of that deployment.
	ComponentName string `json:"componentName"`
	// WorkerHash identifies the new worker generation.
	WorkerHash string `json:"workerHash"`
	// FirstPair is true for the first pair of this generation in this
	// component, so a judge can hold it longer than the rest.
	// +optional
	FirstPair bool `json:"firstPair,omitempty"`
	// Mirror is the new-generation worker.
	Mirror DynamoMirrorPairWorker `json:"mirror"`
	// Shadowed is the old-generation worker the mirror receives copies of.
	Shadowed DynamoMirrorPairWorker `json:"shadowed"`
}

// DynamoMirrorPairWorker identifies one worker of a pair, as a Pod and as the
// Dynamo worker that Pod registers.
type DynamoMirrorPairWorker struct {
	// PodName is the worker's Pod.
	PodName string `json:"podName"`
	// DynamoNamespace is the Dynamo namespace the worker registers in.
	DynamoNamespace string `json:"dynamoNamespace"`
	// WorkerID is the worker's Dynamo instance id, in decimal.
	WorkerID string `json:"workerID"`
}

// DynamoMirrorPairStatus is the pair's progress and verdict.
type DynamoMirrorPairStatus struct {
	// Phase is set by the operator.
	// +optional
	Phase DynamoMirrorPairPhase `json:"phase,omitempty"`
	// MirroringSince is when the mirror started receiving copies.
	// +optional
	MirroringSince *metav1.Time `json:"mirroringSince,omitempty"`
	// Reason explains an Aborted phase.
	// +optional
	Reason string `json:"reason,omitempty"`
	// Conditions carry the verdict. Judges set Approved or Rejected to True;
	// the first to become True decides the pair.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="Deployment",type="string",JSONPath=".spec.graphDeploymentName"
// +kubebuilder:printcolumn:name="Component",type="string",JSONPath=".spec.componentName"
// +kubebuilder:printcolumn:name="Mirror",type="string",JSONPath=".spec.mirror.podName"
// +kubebuilder:printcolumn:name="Shadowed",type="string",JSONPath=".spec.shadowed.podName"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName=dmp

// DynamoMirrorPair is one step of a mirror rollout: a new-generation worker
// that receives copies of an old-generation worker's requests until a judge
// approves it to replace that worker, or rejects it.
type DynamoMirrorPair struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoMirrorPairSpec   `json:"spec,omitempty"`
	Status DynamoMirrorPairStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoMirrorPairList contains a list of DynamoMirrorPair.
type DynamoMirrorPairList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoMirrorPair `json:"items"`
}

// Verdict reports the condition type a judge set to True first, if any.
// Transition times have second precision; a tie counts as a rejection.
func (p *DynamoMirrorPair) Verdict() (verdict string, decided bool) {
	var first *metav1.Condition
	for i := range p.Status.Conditions {
		condition := &p.Status.Conditions[i]
		if condition.Status != metav1.ConditionTrue {
			continue
		}
		if condition.Type != DynamoMirrorPairConditionApproved && condition.Type != DynamoMirrorPairConditionRejected {
			continue
		}
		if first == nil || condition.LastTransitionTime.Before(&first.LastTransitionTime) ||
			(condition.LastTransitionTime.Equal(&first.LastTransitionTime) && condition.Type == DynamoMirrorPairConditionRejected) {
			first = condition
		}
	}
	if first == nil {
		return "", false
	}
	return first.Type, true
}
