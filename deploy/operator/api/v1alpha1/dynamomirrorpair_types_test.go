/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoMirrorPairVerdict(t *testing.T) {
	early := metav1.NewTime(time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC))
	late := metav1.NewTime(early.Add(time.Second))
	condition := func(kind string, status metav1.ConditionStatus, at metav1.Time) metav1.Condition {
		return metav1.Condition{Type: kind, Status: status, LastTransitionTime: at}
	}
	tests := []struct {
		name        string
		conditions  []metav1.Condition
		wantVerdict string
		wantDecided bool
	}{
		{name: "no verdict"},
		{
			name:        "the earlier verdict wins",
			conditions:  []metav1.Condition{condition(DynamoMirrorPairConditionRejected, metav1.ConditionTrue, late), condition(DynamoMirrorPairConditionApproved, metav1.ConditionTrue, early)},
			wantVerdict: DynamoMirrorPairConditionApproved,
			wantDecided: true,
		},
		{
			name:        "a false condition is no verdict",
			conditions:  []metav1.Condition{condition(DynamoMirrorPairConditionRejected, metav1.ConditionFalse, early), condition(DynamoMirrorPairConditionApproved, metav1.ConditionTrue, late)},
			wantVerdict: DynamoMirrorPairConditionApproved,
			wantDecided: true,
		},
		{
			name:        "a tie is a rejection",
			conditions:  []metav1.Condition{condition(DynamoMirrorPairConditionApproved, metav1.ConditionTrue, early), condition(DynamoMirrorPairConditionRejected, metav1.ConditionTrue, early)},
			wantVerdict: DynamoMirrorPairConditionRejected,
			wantDecided: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Read the verdict of a pair with the conditions")
			pair := &DynamoMirrorPair{Status: DynamoMirrorPairStatus{Conditions: tt.conditions}}
			verdict, decided := pair.Verdict()
			assert.Equal(t, tt.wantVerdict, verdict)
			assert.Equal(t, tt.wantDecided, decided)
		})
	}
}
