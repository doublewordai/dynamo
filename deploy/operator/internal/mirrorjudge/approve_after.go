/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package mirrorjudge holds the reference judge for mirror rollouts: it approves
// every pair once its mirror has received copies for a fixed time.
package mirrorjudge

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

// ApproveAfter approves each mirroring pair After its mirror started receiving copies.
type ApproveAfter struct {
	Client client.Client
	After  time.Duration
	// Now is the judge's clock.
	Now func() time.Time
}

// Reconcile approves one pair when its time is up and requeues it until then.
func (j *ApproveAfter) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	pair := &nvidiacomv1alpha1.DynamoMirrorPair{}
	if err := j.Client.Get(ctx, request.NamespacedName, pair); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if pair.Status.Phase != nvidiacomv1alpha1.DynamoMirrorPairPhaseMirroring || pair.Status.MirroringSince == nil {
		return ctrl.Result{}, nil
	}
	if _, decided := pair.Verdict(); decided {
		return ctrl.Result{}, nil
	}

	// Wait out the mirroring window before approving.
	remaining := pair.Status.MirroringSince.Add(j.After).Sub(j.Now())
	if remaining > 0 {
		return ctrl.Result{RequeueAfter: remaining}, nil
	}
	meta.SetStatusCondition(&pair.Status.Conditions, metav1.Condition{
		Type:    nvidiacomv1alpha1.DynamoMirrorPairConditionApproved,
		Status:  metav1.ConditionTrue,
		Reason:  "Elapsed",
		Message: fmt.Sprintf("approved by approve-after judge after mirroring for %s", j.After),
	})
	return ctrl.Result{}, j.Client.Status().Update(ctx, pair)
}

// SetupWithManager registers the judge on mirror pairs.
func (j *ApproveAfter) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoMirrorPair{}).
		Named("mirror-judge-approve-after").
		Complete(j)
}
