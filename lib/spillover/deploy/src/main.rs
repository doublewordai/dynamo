//! `spillover-deploy generate --input lib/spillover/deploy/config/deployments.yaml --out <dir>`
//! `spillover-deploy check --input lib/spillover/deploy/config/deployments.yaml`

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "spillover-deploy",
    about = "Generate Dynamo router-policy and proxy-worker configs from one deployment file"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write router-policy.yaml and one proxy config per (deployment, tier, replica).
    Generate {
        /// Deployment description to read.
        #[arg(long)]
        input: PathBuf,
        /// Directory to write the generated files into.
        #[arg(long)]
        out: PathBuf,
    },
    /// Validate the deployment file and the output it would produce, writing nothing.
    Check {
        /// Deployment description to read.
        #[arg(long)]
        input: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Generate { input, out } => dw_spillover_deploy::generate(&input, &out),
        Command::Check { input } => dw_spillover_deploy::check(&input),
    }
}
