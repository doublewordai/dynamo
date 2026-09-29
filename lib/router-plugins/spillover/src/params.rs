// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy parameters, read from the router-policy YAML `parameters` mapping.
//!
//! ```yaml
//! parameters:
//!   models:
//!     "zai-org/GLM-5.3@interactive":
//!       occupancy_threshold: 0.9
//!       hosted_capacity_blocks: 30000
//!       failover_penalty_blocks: 200
//!       pending_weight_blocks: 4
//!       tiers:
//!         - {name: openrouter, dp_ranks: [1000, 1999], penalty_blocks: 200, weight_blocks: 8}
//!         - {name: provider-b, dp_ranks: [2000, 2999], penalty_blocks: 200, weight_blocks: 40}
//! ```

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct SpilloverParameters {
    /// Settings keyed by Dynamo model name (the routing partition's `model_name`).
    pub models: BTreeMap<String, ModelParameters>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelParameters {
    /// Fraction of `hosted_capacity_blocks` at which a hosted worker counts as full. In (0, 1].
    pub occupancy_threshold: f64,
    /// KV capacity of one hosted worker rank, in blocks. Greater than 0.
    pub hosted_capacity_blocks: f64,
    /// Cost added to a hosted worker at or over the occupancy threshold. At least 0.
    pub failover_penalty_blocks: f64,
    /// Cost per active request on any worker. At least 0.
    pub pending_weight_blocks: f64,
    /// Proxy tiers. Workers whose DP rank falls in no tier are hosted workers.
    #[serde(default)]
    pub tiers: Vec<TierParameters>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TierParameters {
    /// Name used in logs and metrics.
    pub name: String,
    /// Inclusive DP rank range that identifies this tier's proxy workers.
    pub dp_ranks: [u32; 2],
    /// Fixed "always full" cost for every worker in this tier. At least 0.
    pub penalty_blocks: f64,
    /// Preference among tiers: smaller is preferred. At least 0.
    pub weight_blocks: f64,
}

impl SpilloverParameters {
    /// Reject unusable settings at startup. See the field docs for each bound.
    /// Also reject overlapping or inverted tier rank ranges within a model, and ranges that
    /// include rank 0 (hosted workers use low ranks).
    pub fn validate(&self) -> Result<(), String> {
        for (model, params) in &self.models {
            params.validate(model)?;
        }
        Ok(())
    }

    pub fn for_model(&self, model_name: &str) -> Option<&ModelParameters> {
        self.models.get(model_name)
    }
}

impl ModelParameters {
    /// The tier whose inclusive `dp_ranks` range contains `dp_rank`, if any.
    pub fn tier_for_rank(&self, dp_rank: u32) -> Option<&TierParameters> {
        self.tiers
            .iter()
            .find(|tier| (tier.dp_ranks[0]..=tier.dp_ranks[1]).contains(&dp_rank))
    }

    /// Reject unusable values for one model. `model` only names the offending model in errors.
    fn validate(&self, model: &str) -> Result<(), String> {
        if !self.occupancy_threshold.is_finite()
            || self.occupancy_threshold <= 0.0
            || self.occupancy_threshold > 1.0
        {
            return Err(format!(
                "model {model:?}: occupancy_threshold must be in (0, 1]"
            ));
        }
        if !self.hosted_capacity_blocks.is_finite() || self.hosted_capacity_blocks <= 0.0 {
            return Err(format!(
                "model {model:?}: hosted_capacity_blocks must be a positive finite number"
            ));
        }
        for (field, value) in [
            ("failover_penalty_blocks", self.failover_penalty_blocks),
            ("pending_weight_blocks", self.pending_weight_blocks),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "model {model:?}: {field} must be a finite number >= 0"
                ));
            }
        }
        for (index, tier) in self.tiers.iter().enumerate() {
            tier.validate(model, index)?;
        }
        for (index, tier) in self.tiers.iter().enumerate() {
            if self.tiers[..index]
                .iter()
                .any(|other| other.name == tier.name)
            {
                return Err(format!(
                    "model {model:?}: tier name {:?} is duplicated",
                    tier.name
                ));
            }
        }
        for (index, tier) in self.tiers.iter().enumerate() {
            for other in &self.tiers[index + 1..] {
                if tier.dp_ranks[0] <= other.dp_ranks[1] && other.dp_ranks[0] <= tier.dp_ranks[1] {
                    return Err(format!(
                        "model {model:?}: tier {:?} dp_ranks overlap tier {:?}",
                        tier.name, other.name
                    ));
                }
            }
        }
        Ok(())
    }
}

impl TierParameters {
    /// Reject unusable values for one tier. `index` names the offending tier in errors.
    fn validate(&self, model: &str, index: usize) -> Result<(), String> {
        if self.name.is_empty() {
            return Err(format!(
                "model {model:?}: tier[{index}].name must not be empty"
            ));
        }
        for (field, value) in [
            ("penalty_blocks", self.penalty_blocks),
            ("weight_blocks", self.weight_blocks),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "model {model:?}: tier {:?} {field} must be a finite number >= 0",
                    self.name
                ));
            }
        }
        let [start, end] = self.dp_ranks;
        if start == 0 {
            return Err(format!(
                "model {model:?}: tier {:?} dp_ranks start must be > 0",
                self.name
            ));
        }
        if start > end {
            return Err(format!(
                "model {model:?}: tier {:?} dp_ranks start must be <= end",
                self.name
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type ModelMutation = fn(&mut ModelParameters);
    type TierMutation = fn(&mut TierParameters);

    fn valid_model() -> ModelParameters {
        ModelParameters {
            occupancy_threshold: 0.9,
            hosted_capacity_blocks: 1000.0,
            failover_penalty_blocks: 200.0,
            pending_weight_blocks: 10.0,
            tiers: vec![
                TierParameters {
                    name: "x".into(),
                    dp_ranks: [1000, 1999],
                    penalty_blocks: 200.0,
                    weight_blocks: 0.0,
                },
                TierParameters {
                    name: "y".into(),
                    dp_ranks: [2000, 2999],
                    penalty_blocks: 200.0,
                    weight_blocks: 50.0,
                },
            ],
        }
    }

    fn validate(model: ModelParameters) -> Result<(), String> {
        let mut params = SpilloverParameters::default();
        params.models.insert("test-model".into(), model);
        params.validate()
    }

    #[test]
    fn parses_and_accepts_module_doc_example() {
        // The module doc wraps this mapping in the router-policy `parameters` key.
        let yaml = r#"
parameters:
  models:
    "zai-org/GLM-5.3@interactive":
      occupancy_threshold: 0.9
      hosted_capacity_blocks: 30000
      failover_penalty_blocks: 200
      pending_weight_blocks: 4
      tiers:
        - {name: openrouter, dp_ranks: [1000, 1999], penalty_blocks: 200, weight_blocks: 8}
        - {name: provider-b, dp_ranks: [2000, 2999], penalty_blocks: 200, weight_blocks: 40}
"#;
        let doc: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let params: SpilloverParameters =
            serde_yaml::from_value(doc["parameters"].clone()).unwrap();
        assert!(params.validate().is_ok());
        let model = params.for_model("zai-org/GLM-5.3@interactive").unwrap();
        assert_eq!(model.hosted_capacity_blocks, 30000.0);
        assert_eq!(model.tiers.len(), 2);
        assert_eq!(model.tiers[1].name, "provider-b");
    }

    #[test]
    fn accepts_boundary_occupancy_threshold() {
        let mut model = valid_model();
        model.occupancy_threshold = 1.0;
        assert!(validate(model).is_ok());
    }

    #[test]
    fn rejects_bad_scalars_naming_model_and_field() {
        let cases: [(ModelMutation, &str, &str); 6] = [
            (
                |m| m.occupancy_threshold = 0.0,
                "occupancy_threshold",
                "(0, 1]",
            ),
            (
                |m| m.occupancy_threshold = 1.5,
                "occupancy_threshold",
                "(0, 1]",
            ),
            (
                |m| m.occupancy_threshold = f64::NAN,
                "occupancy_threshold",
                "(0, 1]",
            ),
            (
                |m| m.hosted_capacity_blocks = 0.0,
                "hosted_capacity_blocks",
                "positive",
            ),
            (
                |m| m.failover_penalty_blocks = -1.0,
                "failover_penalty_blocks",
                ">= 0",
            ),
            (
                |m| m.pending_weight_blocks = f64::INFINITY,
                "pending_weight_blocks",
                ">= 0",
            ),
        ];
        for (mutate, field, expected) in cases {
            let mut model = valid_model();
            mutate(&mut model);
            let error = validate(model).unwrap_err();
            assert!(error.contains("test-model"), "{error}");
            assert!(error.contains(field), "{error}");
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn rejects_bad_tier_values() {
        let cases: [(TierMutation, &str); 4] = [
            (|t| t.name.clear(), "name"),
            (|t| t.penalty_blocks = -1.0, "penalty_blocks"),
            (|t| t.weight_blocks = f64::NAN, "weight_blocks"),
            (|t| t.dp_ranks = [2000, 1000], "dp_ranks"),
        ];
        for (mutate, field) in cases {
            let mut model = valid_model();
            mutate(&mut model.tiers[0]);
            let error = validate(model).unwrap_err();
            assert!(error.contains("test-model"), "{error}");
            assert!(error.contains(field), "{error}");
        }
    }

    #[test]
    fn rejects_tier_rank_zero() {
        let mut model = valid_model();
        model.tiers[0].dp_ranks = [0, 999];
        let error = validate(model).unwrap_err();
        assert!(error.contains("test-model") && error.contains("dp_ranks"));
    }

    #[test]
    fn rejects_duplicate_tier_names() {
        let mut model = valid_model();
        model.tiers[1].name = "x".into();
        let error = validate(model).unwrap_err();
        assert!(error.contains("test-model") && error.contains("duplicate"));
    }

    #[test]
    fn rejects_overlapping_tier_ranges() {
        let mut model = valid_model();
        model.tiers[1].dp_ranks = [1500, 2500];
        let error = validate(model).unwrap_err();
        assert!(error.contains("test-model") && error.contains("overlap"));
    }

    #[test]
    fn tier_for_rank_edges() {
        let model = valid_model();
        assert!(model.tier_for_rank(0).is_none());
        assert!(model.tier_for_rank(999).is_none());
        assert_eq!(model.tier_for_rank(1000).unwrap().name, "x");
        assert_eq!(model.tier_for_rank(1500).unwrap().name, "x");
        assert_eq!(model.tier_for_rank(1999).unwrap().name, "x");
        assert_eq!(model.tier_for_rank(2000).unwrap().name, "y");
        assert_eq!(model.tier_for_rank(2999).unwrap().name, "y");
        assert!(model.tier_for_rank(3000).is_none());
        // Hosted ranks sit below every proxy range.
        assert!(model.tier_for_rank(7).is_none());
    }
}
