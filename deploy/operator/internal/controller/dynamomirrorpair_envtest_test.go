//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

var _ = Describe("DynamoMirrorPair", func() {
	It("keeps its spec immutable while judges set its status", func() {
		ctx := context.Background()

		By("creating a pair")
		pair := &nvidiacomv1alpha1.DynamoMirrorPair{
			ObjectMeta: metav1.ObjectMeta{Name: "new-a.old-a", Namespace: envtestNamespace},
			Spec: nvidiacomv1alpha1.DynamoMirrorPairSpec{
				GraphDeploymentName: "graph",
				ComponentName:       "worker",
				WorkerHash:          "gen2",
				Mirror:              nvidiacomv1alpha1.DynamoMirrorPairWorker{PodName: "new-a", DynamoNamespace: "ns-gen2", WorkerID: "21"},
				Shadowed:            nvidiacomv1alpha1.DynamoMirrorPairWorker{PodName: "old-a", DynamoNamespace: "ns-gen1", WorkerID: "11"},
			},
		}
		Expect(k8sClient.Create(ctx, pair)).To(Succeed())

		By("retargeting its shadowed worker")
		retargeted := pair.DeepCopy()
		retargeted.Spec.Shadowed.PodName = "old-b"
		err := k8sClient.Update(ctx, retargeted)
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an invalid error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("spec is immutable"))

		By("approving it through its status")
		meta.SetStatusCondition(&pair.Status.Conditions, metav1.Condition{
			Type: nvidiacomv1alpha1.DynamoMirrorPairConditionApproved, Status: metav1.ConditionTrue, Reason: "Judged",
		})
		Expect(k8sClient.Status().Update(ctx, pair)).To(Succeed())
	})
})
