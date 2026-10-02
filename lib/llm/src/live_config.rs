// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-model routing settings re-read from a file while the frontend runs.
//!
//! When `DYN_LIVE_CONFIG_PATH` names a file, the frontend reads it every
//! `DYN_LIVE_CONFIG_POLL_SECS` seconds (default 5). A Kubernetes ConfigMap
//! mounted as a volume (not through `subPath`) updates in place, so editing the
//! ConfigMap changes routing without restarting frontends or workers.
//!
//! ```yaml
//! models:
//!   my-served-model:
//!     router:
//!       overlap_score_credit: 1.5
//!       prefill_load_scale: 1.0
//!       router_temperature: 0.0
//! ```
//!
//! Models are keyed by the model name the frontend receives in the request.
//! The `router` fields are the per-request [`RouterConfigOverride`] fields and
//! are validated the same way. A field set on the request itself takes
//! precedence over the file. Other keys under a model are ignored, so one file
//! can also carry settings that other processes read.
//!
//! A file that fails to parse or validate is logged and ignored, and the last
//! valid contents stay in force. A missing file means no overrides.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use arc_swap::ArcSwap;
use dynamo_kv_router::RouterConfigOverride;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

pub const LIVE_CONFIG_PATH_ENV: &str = "DYN_LIVE_CONFIG_PATH";
pub const LIVE_CONFIG_POLL_SECS_ENV: &str = "DYN_LIVE_CONFIG_POLL_SECS";
const DEFAULT_POLL: Duration = Duration::from_secs(5);

static STORE: LazyLock<LiveConfigStore> = LazyLock::new(LiveConfigStore::default);

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveConfigFile {
    #[serde(default)]
    models: HashMap<String, ModelSection>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelSection {
    #[serde(default)]
    router: Option<RouterSection>,
}

/// The [`RouterConfigOverride`] fields, rejecting unknown keys so a misspelt
/// setting fails validation instead of silently doing nothing.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterSection {
    overlap_score_credit: Option<f64>,
    prefill_load_scale: Option<f64>,
    router_temperature: Option<f64>,
    assume_kv_reuse: Option<bool>,
    track_prefill_tokens: Option<bool>,
    shared_cache_multiplier: Option<f64>,
}

impl TryFrom<RouterSection> for RouterConfigOverride {
    type Error = String;

    fn try_from(section: RouterSection) -> Result<Self, Self::Error> {
        let RouterSection {
            overlap_score_credit,
            prefill_load_scale,
            router_temperature,
            assume_kv_reuse,
            track_prefill_tokens,
            shared_cache_multiplier,
        } = section;
        let config = RouterConfigOverride {
            overlap_score_credit,
            prefill_load_scale,
            router_temperature,
            assume_kv_reuse,
            track_prefill_tokens,
            shared_cache_multiplier,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Validated file contents.
#[derive(Debug, Default)]
pub struct LiveConfig {
    router: HashMap<String, RouterConfigOverride>,
}

impl LiveConfig {
    pub fn parse(contents: &str) -> anyhow::Result<Self> {
        let file: LiveConfigFile = if contents.trim().is_empty() {
            LiveConfigFile::default()
        } else {
            serde_yaml::from_str(contents)?
        };
        let mut router = HashMap::new();
        for (model, section) in file.models {
            if let Some(section) = section.router {
                let config = RouterConfigOverride::try_from(section)
                    .map_err(|error| anyhow::anyhow!("models.{model}.router: {error}"))?;
                router.insert(model, config);
            }
        }
        Ok(Self { router })
    }

    pub fn router_override(&self, model: &str) -> Option<&RouterConfigOverride> {
        self.router.get(model)
    }
}

/// Holds the config currently in force. Requests read it without locking.
#[derive(Default)]
pub struct LiveConfigStore {
    current: ArcSwap<LiveConfig>,
}

impl LiveConfigStore {
    pub fn load(&self) -> Arc<LiveConfig> {
        self.current.load_full()
    }

    pub fn store(&self, config: LiveConfig) {
        self.current.store(Arc::new(config));
    }

    /// The override to route `model` with: fields set on the request win, and
    /// the file fills the rest.
    pub fn effective_router_override<'a>(
        &self,
        model: &str,
        request: Option<&'a RouterConfigOverride>,
    ) -> Option<Cow<'a, RouterConfigOverride>> {
        let config = self.current.load();
        let Some(file) = config.router_override(model) else {
            return request.map(Cow::Borrowed);
        };
        Some(Cow::Owned(match request {
            Some(request) => fill_unset(request, file),
            None => file.clone(),
        }))
    }
}

fn fill_unset(request: &RouterConfigOverride, file: &RouterConfigOverride) -> RouterConfigOverride {
    let RouterConfigOverride {
        overlap_score_credit,
        prefill_load_scale,
        router_temperature,
        assume_kv_reuse,
        track_prefill_tokens,
        shared_cache_multiplier,
    } = request.clone();
    RouterConfigOverride {
        overlap_score_credit: overlap_score_credit.or(file.overlap_score_credit),
        prefill_load_scale: prefill_load_scale.or(file.prefill_load_scale),
        router_temperature: router_temperature.or(file.router_temperature),
        assume_kv_reuse: assume_kv_reuse.or(file.assume_kv_reuse),
        track_prefill_tokens: track_prefill_tokens.or(file.track_prefill_tokens),
        shared_cache_multiplier: shared_cache_multiplier.or(file.shared_cache_multiplier),
    }
}

/// The process-wide store the frontend routes with.
pub fn store() -> &'static LiveConfigStore {
    &STORE
}

/// Starts polling `DYN_LIVE_CONFIG_PATH` if it is set.
pub fn spawn_from_env(cancel: CancellationToken) {
    let Some(path) = std::env::var_os(LIVE_CONFIG_PATH_ENV).map(PathBuf::from) else {
        return;
    };
    let poll = match std::env::var(LIVE_CONFIG_POLL_SECS_ENV) {
        Ok(value) => match value.parse::<f64>() {
            Ok(secs) if secs.is_finite() && secs > 0.0 => Duration::from_secs_f64(secs),
            _ => {
                tracing::warn!(
                    value,
                    "{LIVE_CONFIG_POLL_SECS_ENV} is not a positive number of seconds; using {}s",
                    DEFAULT_POLL.as_secs()
                );
                DEFAULT_POLL
            }
        },
        Err(_) => DEFAULT_POLL,
    };
    tokio::spawn(watch(path, poll, store(), cancel));
}

async fn watch(
    path: PathBuf,
    poll: Duration,
    store: &'static LiveConfigStore,
    cancel: CancellationToken,
) {
    tracing::info!(path = %path.display(), poll_secs = poll.as_secs_f64(), "Watching live config");
    let mut last = None;
    let mut ticker = tokio::time::interval(poll);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticker.tick() => {}
        }
        reload(&path, &mut last, store).await;
    }
}

/// Re-reads `path` and swaps in its contents if they changed and are valid.
/// `last` holds the contents seen on the previous read, valid or not, so a bad
/// file is reported once rather than on every poll.
async fn reload(path: &Path, last: &mut Option<String>, store: &LiveConfigStore) {
    let contents = match tokio::fs::read_to_string(path).await {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            tracing::error!(path = %path.display(), %error, "Cannot read live config; keeping the current one");
            return;
        }
    };
    if last.as_deref() == Some(contents.as_str()) {
        return;
    }
    match LiveConfig::parse(&contents) {
        Ok(config) => {
            tracing::info!(path = %path.display(), router = ?config.router, "Applied live config");
            store.store(config);
        }
        Err(error) => {
            tracing::error!(path = %path.display(), %error, "Invalid live config; keeping the current one");
        }
    }
    *last = Some(contents);
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUBIC: &str = r#"
models:
  cubic:
    router:
      overlap_score_credit: 1.5
      router_temperature: 0.0
    admission:
      queue_margin: 20
  other:
    engine:
      max_running_requests: 8
"#;

    fn cubic_credit(store: &LiveConfigStore) -> Option<f64> {
        store
            .load()
            .router_override("cubic")
            .and_then(|config| config.overlap_score_credit)
    }

    #[test]
    fn parses_router_sections_and_ignores_other_model_keys() {
        let config = LiveConfig::parse(CUBIC).unwrap();
        let cubic = config.router_override("cubic").unwrap();
        assert_eq!(cubic.overlap_score_credit, Some(1.5));
        assert_eq!(cubic.router_temperature, Some(0.0));
        assert_eq!(cubic.prefill_load_scale, None);
        assert!(config.router_override("other").is_none());
    }

    #[test]
    fn empty_file_has_no_overrides() {
        assert!(LiveConfig::parse("").unwrap().router.is_empty());
        assert!(LiveConfig::parse("  \n").unwrap().router.is_empty());
    }

    #[test]
    fn rejects_misspelt_and_invalid_settings() {
        for contents in [
            "models: {m: {router: {overlap_score_credits: 1.5}}}",
            "models: {m: {router: {overlap_score_credit: -1.0}}}",
            "models: {m: {router: {shared_cache_multiplier: 2.0}}}",
            "modles: {m: {router: {overlap_score_credit: 1.5}}}",
            "models: [1, 2]",
        ] {
            assert!(LiveConfig::parse(contents).is_err(), "accepted {contents}");
        }
    }

    #[test]
    fn request_fields_win_and_the_file_fills_the_rest() {
        let store = LiveConfigStore::default();
        store.store(LiveConfig::parse(CUBIC).unwrap());

        let from_file = store.effective_router_override("cubic", None).unwrap();
        assert_eq!(from_file.overlap_score_credit, Some(1.5));

        let request = RouterConfigOverride {
            overlap_score_credit: Some(0.0),
            ..Default::default()
        };
        let merged = store
            .effective_router_override("cubic", Some(&request))
            .unwrap();
        assert_eq!(merged.overlap_score_credit, Some(0.0));
        assert_eq!(merged.router_temperature, Some(0.0));

        assert!(store.effective_router_override("unknown", None).is_none());
        let passthrough = store
            .effective_router_override("unknown", Some(&request))
            .unwrap();
        assert!(matches!(passthrough, Cow::Borrowed(_)));
    }

    #[tokio::test]
    async fn reload_applies_valid_changes_and_keeps_the_last_good_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.yaml");
        let store = LiveConfigStore::default();
        let mut last = None;

        reload(&path, &mut last, &store).await;
        assert!(store.load().router.is_empty());

        std::fs::write(&path, CUBIC).unwrap();
        reload(&path, &mut last, &store).await;
        assert_eq!(cubic_credit(&store), Some(1.5));

        std::fs::write(
            &path,
            "models: {cubic: {router: {overlap_score_credit: -1}}}",
        )
        .unwrap();
        reload(&path, &mut last, &store).await;
        assert_eq!(cubic_credit(&store), Some(1.5));

        std::fs::write(
            &path,
            "models: {cubic: {router: {overlap_score_credit: 1.25}}}",
        )
        .unwrap();
        reload(&path, &mut last, &store).await;
        assert_eq!(cubic_credit(&store), Some(1.25));

        std::fs::remove_file(&path).unwrap();
        reload(&path, &mut last, &store).await;
        assert!(store.load().router.is_empty());
    }

    #[tokio::test]
    async fn reload_follows_a_configmap_style_symlink_swap() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("..2026_10_02_a");
        let second = dir.path().join("..2026_10_02_b");
        std::fs::create_dir(&first).unwrap();
        std::fs::create_dir(&second).unwrap();
        std::fs::write(first.join("live.yaml"), CUBIC).unwrap();
        std::fs::write(
            second.join("live.yaml"),
            "models: {cubic: {router: {overlap_score_credit: 3.0}}}",
        )
        .unwrap();
        let data = dir.path().join("..data");
        std::os::unix::fs::symlink(&first, &data).unwrap();
        std::os::unix::fs::symlink(data.join("live.yaml"), dir.path().join("live.yaml")).unwrap();

        let path = dir.path().join("live.yaml");
        let store = LiveConfigStore::default();
        let mut last = None;
        reload(&path, &mut last, &store).await;
        assert_eq!(cubic_credit(&store), Some(1.5));

        // The kubelet swaps `..data` with a rename, as here.
        let staged = dir.path().join("..data_tmp");
        std::os::unix::fs::symlink(&second, &staged).unwrap();
        std::fs::rename(&staged, &data).unwrap();
        reload(&path, &mut last, &store).await;
        assert_eq!(cubic_credit(&store), Some(3.0));
    }
}
