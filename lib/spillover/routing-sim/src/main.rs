// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `routing-sim <scenario.yaml> [--json out.json] [--markdown out.md] [--heuristic]`
//! `routing-sim sweep <scenario.yaml> --param name=v1,v2 [--param ...] [--jobs N]
//!                    [--json out.json] [--markdown out.md]`
//!
//! Runs one scenario and prints a markdown report, or sweeps a grid of policy settings.
//! `--heuristic` uses the development stand-in selector; the default is the real spillover policy.

use std::path::PathBuf;

use dw_routing_sim::sweep::{ParamSpec, SweepResult};
use dw_routing_sim::{HeuristicSelector, PolicySelector, Scenario, Selector, check_assertions};
use dynamo_kv_router::KvRouterConfig;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("sweep") {
        return run_sweep(&args[1..]);
    }
    run_scenario(&args)
}

fn run_scenario(args: &[String]) -> anyhow::Result<()> {
    let mut scenario_path: Option<PathBuf> = None;
    let mut json: Option<PathBuf> = None;
    let mut markdown: Option<PathBuf> = None;
    let mut heuristic = false;

    let mut args = args.iter().cloned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = Some(next_path(&mut args, "--json")?),
            "--markdown" => markdown = Some(next_path(&mut args, "--markdown")?),
            "--heuristic" => heuristic = true,
            "-h" | "--help" => {
                println!(
                    "usage: routing-sim <scenario.yaml> [--json out.json] [--markdown out.md] [--heuristic]"
                );
                return Ok(());
            }
            other => {
                if scenario_path.is_some() {
                    anyhow::bail!("unexpected argument: {other}");
                }
                scenario_path = Some(PathBuf::from(other));
            }
        }
    }

    let scenario_path = scenario_path.ok_or_else(|| anyhow::anyhow!("missing scenario path"))?;
    let scenario = Scenario::load(&scenario_path)?;
    let params = scenario.policy.parameters();

    let mut selector: Box<dyn Selector> = if heuristic {
        Box::new(HeuristicSelector::new(
            params
                .for_model(&scenario.policy.model)
                .cloned()
                .expect("scenario policy has a model"),
        ))
    } else {
        Box::new(PolicySelector::new(
            &KvRouterConfig::default(),
            &scenario.policy.model,
            &params,
            scenario.seed,
        ))
    };

    let report = dw_routing_sim::run_scenario_with_default_reference(&scenario, selector.as_mut());
    let markdown_text = report.markdown();
    print!("{markdown_text}");
    if let Some(path) = &markdown {
        std::fs::write(path, &markdown_text)?;
    }
    if let Some(path) = &json {
        std::fs::write(path, report.json())?;
    }

    let failures = check_assertions(&scenario, &report);
    if !failures.is_empty() {
        eprintln!("\nassertion failures:");
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        std::process::exit(1);
    }
    Ok(())
}

fn run_sweep(args: &[String]) -> anyhow::Result<()> {
    let mut scenario_path: Option<PathBuf> = None;
    let mut params: Vec<ParamSpec> = Vec::new();
    let mut json: Option<PathBuf> = None;
    let mut markdown: Option<PathBuf> = None;
    let mut jobs: Option<usize> = None;

    let mut args = args.iter().cloned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--param" => {
                let text = next_value(&mut args, "--param")?;
                params.push(ParamSpec::parse(&text)?);
            }
            "--json" => json = Some(next_path(&mut args, "--json")?),
            "--markdown" => markdown = Some(next_path(&mut args, "--markdown")?),
            "--jobs" => {
                let text = next_value(&mut args, "--jobs")?;
                jobs = Some(text.parse().map_err(|_| {
                    anyhow::anyhow!("--jobs expects a positive integer (got {text:?})")
                })?);
            }
            "-h" | "--help" => {
                println!(
                    "usage: routing-sim sweep <scenario.yaml> --param name=v1,v2 [--param ...] \
                     [--jobs N] [--json out.json] [--markdown out.md]"
                );
                return Ok(());
            }
            other => {
                if scenario_path.is_some() {
                    anyhow::bail!("unexpected argument: {other}");
                }
                scenario_path = Some(PathBuf::from(other));
            }
        }
    }

    if params.is_empty() {
        anyhow::bail!("sweep needs at least one --param name=v1,v2");
    }
    let scenario_path = scenario_path.ok_or_else(|| anyhow::anyhow!("missing scenario path"))?;
    let scenario = Scenario::load(&scenario_path)?;
    let jobs = jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
    });

    let result = dw_routing_sim::sweep::run(&scenario, &params, jobs)?;
    write_sweep(&result, markdown.as_deref(), json.as_deref())
}

fn write_sweep(
    result: &SweepResult,
    markdown: Option<&std::path::Path>,
    json: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let markdown_text = result.markdown();
    print!("{markdown_text}");
    if let Some(path) = markdown {
        std::fs::write(path, &markdown_text)?;
    }
    if let Some(path) = json {
        std::fs::write(path, result.json())?;
    }
    Ok(())
}

fn next_path(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a path"))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
}
