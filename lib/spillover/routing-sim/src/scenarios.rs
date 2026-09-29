//! The built-in scenarios, embedded so the library and binary do not depend on the working
//! directory. `tests/scenarios.rs` also loads them from `scenarios/` to keep the files honest.

use crate::config::Scenario;

macro_rules! builtin_scenarios {
    ($($name:literal),* $(,)?) => {
        /// Names of the shipped scenarios.
        pub const NAMES: &[&str] = &[$($name),*];

        /// The YAML text of a shipped scenario, or `None` for an unknown name.
        pub fn text(name: &str) -> Option<&'static str> {
            match name {
                $($name => Some(include_str!(concat!("../scenarios/", $name, ".yaml"))),)*
                _ => None,
            }
        }
    };
}

builtin_scenarios![
    "low_load",
    "overload_ramp",
    "stickiness",
    "proxy_rate_limited",
    "no_parameters",
    "hosted_outage",
];

/// Parse a shipped scenario by name.
pub fn load_builtin(name: &str) -> anyhow::Result<Scenario> {
    let text = text(name).ok_or_else(|| anyhow::anyhow!("unknown scenario: {name}"))?;
    Scenario::parse(text)
}
