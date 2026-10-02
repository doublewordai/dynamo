/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/yaml"
)

// The crd-apply init container sends each CRD as a server-side-apply request.
// kube-apiserver rejects request bodies larger than 3 MiB ("Request entity too
// large: limit is 3145728"), and etcd will not store an object larger than its
// max-request-bytes (1.5 MiB by default). Keep both limits here with headroom
// so a regenerated CRD cannot silently grow past them again.
const (
	maxRequestBytes = 2 * 1024 * 1024 // 2 MiB: headroom below the 3 MiB request limit
	maxStoredBytes  = 1 * 1024 * 1024 // 1 MiB: headroom below the 1.5 MiB etcd default
)

// TestCRDsFitSizeLimits guards against the regression where regenerating the
// CRDs for k8s.io 0.37 pushed DynamoGraphDeployment past the API server's
// request limit, making the operator's crd-apply init container fail on every
// start. It measures exactly what crd-apply sends: unmarshal the generated
// YAML with sigs.k8s.io/yaml and marshal it back, which matches the init
// container's request body (and is larger than the file on disk).
func TestCRDsFitSizeLimits(t *testing.T) {
	crdsDir := filepath.Join("..", "..", "config", "crd", "bases")
	entries, err := os.ReadDir(crdsDir)
	if err != nil {
		t.Fatalf("unable to read CRD directory %s: %v", crdsDir, err)
	}

	var checked int
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}

		filePath := filepath.Join(crdsDir, entry.Name())
		data, err := os.ReadFile(filePath)
		if err != nil {
			t.Fatalf("unable to read %s: %v", filePath, err)
		}

		crd := &apiextensionsv1.CustomResourceDefinition{}
		if err := yaml.Unmarshal(data, crd); err != nil {
			t.Fatalf("unable to unmarshal %s: %v", filePath, err)
		}

		requestBody, err := yaml.Marshal(crd)
		if err != nil {
			t.Fatalf("unable to marshal %s: %v", filePath, err)
		}
		storedBody, err := json.Marshal(crd)
		if err != nil {
			t.Fatalf("unable to marshal %s as JSON: %v", filePath, err)
		}

		t.Logf("%s: request=%d bytes (limit %d), stored=%d bytes (limit %d)",
			entry.Name(), len(requestBody), maxRequestBytes, len(storedBody), maxStoredBytes)

		if len(requestBody) > maxRequestBytes {
			t.Errorf("CRD %s marshals to %d bytes, over the %d-byte request guard; "+
				"crd-apply would be rejected by the API server's 3 MiB limit",
				entry.Name(), len(requestBody), maxRequestBytes)
		}
		if len(storedBody) > maxStoredBytes {
			t.Errorf("CRD %s serializes to %d bytes, over the %d-byte storage guard; etcd may reject it",
				entry.Name(), len(storedBody), maxStoredBytes)
		}
		checked++
	}

	if checked == 0 {
		t.Fatalf("no CRD YAML files found in %s", crdsDir)
	}
}
