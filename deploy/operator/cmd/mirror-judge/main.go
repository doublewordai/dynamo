/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Command mirror-judge is the reference judge for mirror rollouts. It approves
// every DynamoMirrorPair once the mirror has received copies for
// --approve-after. It exists for tests and local rehearsals: it is not built
// into the operator image or deployed by the Helm chart. A production
// deployment runs its own gate controller that sets each pair's Approved or
// Rejected condition from the pair's metrics. To run this one against the
// current kubeconfig context:
//
//	go run ./cmd/mirror-judge --approve-after=10m
//
// It needs get, list and watch on dynamomirrorpairs and update on
// dynamomirrorpairs/status.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/mirrorjudge"
)

func main() {
	after := flag.Duration("approve-after", 10*time.Second,
		"How long a mirror receives copies before its pair is approved; must not be negative")
	flag.Parse()
	ctrl.SetLogger(zap.New())
	log := ctrl.Log.WithName("mirror-judge")
	if *after < 0 {
		log.Error(fmt.Errorf("--approve-after is %s", *after), "a negative window would approve every pair at once")
		os.Exit(2)
	}

	// Build a manager that knows the pair type.
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		log.Error(err, "register scheme")
		os.Exit(1)
	}
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{Scheme: scheme})
	if err != nil {
		log.Error(err, "create manager")
		os.Exit(1)
	}

	// Run the approve-after judge until the process is signalled.
	judge := &mirrorjudge.ApproveAfter{Client: mgr.GetClient(), After: *after, Now: time.Now}
	if err := judge.SetupWithManager(mgr); err != nil {
		log.Error(err, "set up judge")
		os.Exit(1)
	}
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		log.Error(err, "run manager")
		os.Exit(1)
	}
}
