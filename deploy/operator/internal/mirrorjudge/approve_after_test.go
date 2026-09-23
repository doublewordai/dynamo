/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mirrorjudge

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

func TestApproveAfterApprovesOnceTheWindowHasPassed(t *testing.T) {
	ctx := context.Background()
	start := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)

	t.Log("A pair that started mirroring at start")
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))
	pair := &nvidiacomv1alpha1.DynamoMirrorPair{
		ObjectMeta: metav1.ObjectMeta{Name: "new-a.old-a", Namespace: "serving"},
		Status: nvidiacomv1alpha1.DynamoMirrorPairStatus{
			Phase:          nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring,
			MirroringSince: &metav1.Time{Time: start},
		},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(pair).
		WithStatusSubresource(&nvidiacomv1alpha1.DynamoMirrorPair{}).
		Build()
	now := start.Add(4 * time.Second)
	judge := &ApproveAfter{Client: kubeClient, After: 10 * time.Second, Now: func() time.Time { return now }}
	request := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "serving", Name: "new-a.old-a"}}

	t.Log("Before the window the judge waits for the remaining time")
	result, err := judge.Reconcile(ctx, request)
	require.NoError(t, err)
	assert.Equal(t, 6*time.Second, result.RequeueAfter)
	require.NoError(t, kubeClient.Get(ctx, request.NamespacedName, pair))
	_, decided := pair.Verdict()
	assert.False(t, decided)

	t.Log("After the window the judge approves")
	now = start.Add(10 * time.Second)
	_, err = judge.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, request.NamespacedName, pair))
	verdict, decided := pair.Verdict()
	assert.True(t, decided)
	assert.Equal(t, nvidiacomv1alpha1.DynamoMirrorPairConditionApproved, verdict)
}
