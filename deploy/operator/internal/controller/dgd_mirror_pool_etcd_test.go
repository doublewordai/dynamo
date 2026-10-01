/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// startTestEtcd runs the etcd binary envtest installs, or skips the test when
// KUBEBUILDER_ASSETS does not provide one.
func startTestEtcd(t *testing.T) *clientv3.Client {
	t.Helper()
	binary := filepath.Join(os.Getenv("KUBEBUILDER_ASSETS"), "etcd")
	if _, err := os.Stat(binary); os.Getenv("KUBEBUILDER_ASSETS") == "" || err != nil {
		t.Skip("KUBEBUILDER_ASSETS does not provide an etcd binary")
	}
	freePort := func() int {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		defer func() { _ = listener.Close() }()
		return listener.Addr().(*net.TCPAddr).Port
	}
	clientURL := fmt.Sprintf("http://127.0.0.1:%d", freePort())
	peerURL := fmt.Sprintf("http://127.0.0.1:%d", freePort())
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, binary,
		"--data-dir", t.TempDir(),
		"--listen-client-urls", clientURL, "--advertise-client-urls", clientURL,
		"--listen-peer-urls", peerURL, "--initial-advertise-peer-urls", peerURL,
		"--initial-cluster", "default="+peerURL)
	require.NoError(t, cmd.Start())
	t.Cleanup(func() {
		cancel()
		_ = cmd.Wait()
	})

	etcd, err := clientv3.New(clientv3.Config{Endpoints: []string{clientURL}, DialTimeout: 5 * time.Second})
	require.NoError(t, err)
	t.Cleanup(func() { _ = etcd.Close() })
	require.Eventually(t, func() bool {
		attempt, done := context.WithTimeout(context.Background(), time.Second)
		defer done()
		_, err := etcd.Get(attempt, "health")
		return err == nil
	}, 30*time.Second, 100*time.Millisecond)
	return etcd
}

func TestPoolWatcherRemovesOnlyRolesOfWorkersStillAbsent(t *testing.T) {
	ctx := context.Background()
	etcd := startTestEtcd(t)
	put := func(key, value string) {
		_, err := etcd.Put(ctx, key, value)
		require.NoError(t, err)
	}
	member := `{"pod_namespace":"serving","pod_name":"%s"}`

	t.Log("A live worker with a role, and the role of a worker that left")
	put(consts.PoolMembersPrefix+"ns/a", fmt.Sprintf(member, "worker-a"))
	put(consts.PoolRolesPrefix+"ns/a/role", `{"taints":[]}`)
	put(consts.PoolRolesPrefix+"ns/b/role", `{"taints":[]}`)

	t.Log("The watcher snapshots the members")
	snapshot, err := etcd.Get(ctx, consts.PoolMembersPrefix, clientv3.WithPrefix())
	require.NoError(t, err)
	members := map[string]poolMember{}
	for _, kv := range snapshot.Kvs {
		members[string(kv.Key)[len(consts.PoolMembersPrefix):]] = poolMember{}
	}

	t.Log("A worker registers and is given a role after the snapshot")
	put(consts.PoolMembersPrefix+"ns/c", fmt.Sprintf(member, "worker-c"))
	put(consts.PoolRolesPrefix+"ns/c/role", `{"taints":[]}`)

	t.Log("Cleanup removes only the role of the worker that left")
	watcher := &poolWatcher{client: etcd}
	require.NoError(t, watcher.removeOrphanRoles(ctx, members, snapshot.Header.Revision))
	roles, err := etcd.Get(ctx, consts.PoolRolesPrefix, clientv3.WithPrefix(), clientv3.WithKeysOnly())
	require.NoError(t, err)
	keys := make([]string, 0, len(roles.Kvs))
	for _, kv := range roles.Kvs {
		keys = append(keys, string(kv.Key))
	}
	assert.ElementsMatch(t, []string{consts.PoolRolesPrefix + "ns/a/role", consts.PoolRolesPrefix + "ns/c/role"}, keys)
}
