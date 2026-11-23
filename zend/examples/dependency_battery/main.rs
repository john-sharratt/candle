//! `dependency_battery` — the codebase dependency battery of
//! `docs/unbounded_agents.md` §9.12, against a running zend.
//!
//! - `run`: ask every question of `battery.yaml` under each of the seven arms,
//!   three runs each, sixteen at a time and interleaved (`schedule.rs`); each
//!   answer a fresh conversation over the daemon's HTTP API with tools off and
//!   the arm's retrieval dials, scored (`score.rs`), read back for what its
//!   projection points placed (`placement.rs`) and deleted, one JSON line
//!   each; the card's load sampled beside it (`telemetry.rs`), and the
//!   member's K/V census (`GET /v1/kv/census`) taken as it starts and ends.
//! - `analyse`: every figure's data and table of §9.12, §6.5 and §9.6.9 from
//!   those lines (`analyse.rs`, `report.rs`).
//! - `direct`: the direct questions, generated from the source by rule
//!   (`direct.rs`), as `battery.yaml` lists them.
//!
//! ```sh
//! cargo run -p zend --release --example dependency_battery -- run \
//!   --zend http://192.168.1.2:8081 --node 223ebb209828d84c --repo D:\prog\candle \
//!   --out docs/results/battery_128/answers.jsonl
//! cargo run -p zend --release --example dependency_battery -- analyse \
//!   --answers docs/results/battery_128/answers.jsonl --out docs/results/battery_128
//! ```

mod analyse;
mod arms;
mod battery;
mod client;
mod direct;
mod load;
mod placement;
mod report;
mod run;
mod schedule;
mod score;
mod stats;
mod telemetry;

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use serde_json::Value;
use tracing_subscriber::EnvFilter;

use battery::{checked, Battery};
use score::words;

#[derive(Parser)]
#[command(about = "The §9.12 dependency battery against a running zend")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Ask the battery.
    Run {
        /// The member's HTTP base, e.g. `http://192.168.1.2:8081`, reached
        /// from the member's own machine.
        #[arg(long)]
        zend: String,
        /// The member that answers every question: its zend node id, 16 hex
        /// digits, as `cluster_status` lists it.
        #[arg(long)]
        node: String,
        /// The repository the questions are about, for its files and its
        /// grounding vocabulary.
        #[arg(long)]
        repo: PathBuf,
        /// The battery; defaults to the one beside this example.
        #[arg(long)]
        battery: Option<PathBuf>,
        /// Where the answers' lines go; a run is taken up again from it.
        #[arg(long)]
        out: PathBuf,
        /// Where the telemetry samples go; beside `out` by default.
        #[arg(long)]
        telemetry: Option<PathBuf>,
        /// The thread the battery's conversations are named under.
        #[arg(long, default_value = "battery-128")]
        thread: String,
        #[arg(long, default_value_t = 3)]
        runs: u32,
        #[arg(long, default_value_t = 16)]
        concurrency: usize,
        /// The queue's shuffle seed.
        #[arg(long, default_value_t = analyse::SEED)]
        seed: u64,
        /// Ask only these question ids.
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
    },
    /// Write every figure's data and table from a run's answers.
    Analyse {
        #[arg(long)]
        answers: PathBuf,
        /// The telemetry samples; beside `answers` by default.
        #[arg(long)]
        telemetry: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the direct questions generated from `repo`.
    Direct {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value_t = 43)]
        count: usize,
        #[arg(long, default_value_t = analyse::SEED)]
        seed: u64,
    },
}

/// Every identifier-like word in the repository's source and documents.
fn vocabulary(repo: &Path) -> anyhow::Result<HashSet<String>> {
    let mut vocab = HashSet::new();
    for entry in ignore::WalkBuilder::new(repo).build() {
        let entry = entry?;
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_file()
            && matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("rs" | "cu" | "cuh" | "md" | "yaml" | "toml")
            )
        {
            vocab.extend(words(name).map(str::to_string));
            if let Ok(text) = std::fs::read_to_string(p) {
                vocab.extend(words(&text).map(str::to_string));
            }
        }
    }
    Ok(vocab)
}

fn default_battery() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/dependency_battery/battery.yaml")
}

fn beside(path: &Path, name: &str) -> PathBuf {
    path.parent().unwrap_or_else(|| Path::new(".")).join(name)
}

fn samples(path: &Path) -> anyhow::Result<Vec<Value>> {
    let Ok(f) = File::open(path) else {
        return Ok(Vec::new());
    };
    let (samples, cut) = telemetry::read(BufReader::new(f))
        .with_context(|| format!("reading {}", path.display()))?;
    if !cut.is_empty() {
        tracing::warn!(lines = ?cut, "telemetry lines cut short by a stopped launch, left out");
    }
    Ok(samples)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().cmd {
        Cmd::Run {
            zend,
            node,
            repo,
            battery,
            out,
            telemetry,
            thread,
            runs,
            concurrency,
            seed,
            only,
        } => {
            let battery = checked(
                Battery::load(&battery.unwrap_or_else(default_battery))?,
                &repo,
            )?;
            let questions = battery
                .questions
                .into_iter()
                .filter(|q| only.is_empty() || only.contains(&q.id))
                .collect();
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let telemetry = telemetry.unwrap_or_else(|| beside(&out, "telemetry.jsonl"));
            run::run(run::Plan {
                base: &zend,
                node: &node,
                thread: &thread,
                questions,
                vocabulary: vocabulary(&repo)?,
                runs,
                seed,
                concurrency,
                out: &out,
                telemetry: &telemetry,
                census_dir: out.parent().unwrap_or_else(|| Path::new(".")),
            })
            .await
        }
        Cmd::Analyse {
            answers,
            telemetry,
            out,
        } => {
            let lines = run::lines(&answers)?;
            let telemetry = telemetry.unwrap_or_else(|| beside(&answers, "telemetry.jsonl"));
            let census = beside(&answers, "census_start.json");
            let bf16_per_token = match fs::read_to_string(&census) {
                Ok(text) => load::bf16_per_token(
                    &serde_json::from_str(&text)
                        .with_context(|| format!("parsing {}", census.display()))?,
                ),
                Err(_) => None,
            };
            report::write(&lines, &samples(&telemetry)?, bf16_per_token, &out)?;
            println!("wrote {}", out.join("summary.md").display());
            Ok(())
        }
        Cmd::Direct { repo, count, seed } => {
            let files = direct::sources(&repo)?;
            let all: Vec<String> = files.iter().map(|(rel, _)| rel.clone()).collect();
            let cands = direct::candidates(&files);
            eprintln!("{} candidates", cands.len());
            print!("{}", direct::yaml(&direct::pick(cands, count, seed, &all)));
            Ok(())
        }
    }
}
