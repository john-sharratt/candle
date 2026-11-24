//! `dependency_battery` — the codebase dependency battery of
//! `docs/unbounded_agents.md` §9.12, against a running zend.
//!
//! - `run`: ask every question of `battery.yaml` under each arm (`arms.rs`) —
//!   or those `--arms` names (`selection.rs`) — up to its highest run, what
//!   the results file does not already record, sixteen at a time and
//!   interleaved (`schedule.rs`); each answer a fresh conversation over the
//!   daemon's HTTP API with the arm's tools mode, retrieval dials and prompt
//!   (`prompt.rs`; R1's files retrieved by BM25, `rag/`), scored
//!   (`score.rs`), read back for what its projection points placed
//!   (`placement.rs`) and deleted, one JSON line each; the card's load
//!   sampled beside it (`telemetry.rs`), and the member's K/V census
//!   (`GET /v1/kv/census`) taken as it starts and ends.
//! - `analyse`: every figure's data and table of §9.12, §6.5 and §9.6.9 from
//!   those lines (`analyse.rs`, `report.rs`).
//! - `figures`: the paper's ten data figures as inline-SVG snippets, drawn
//!   from those CSVs (`figures/`).
//! - `direct`: the direct questions, generated from the source by rule
//!   (`direct.rs`), as `battery.yaml` lists them.
//!
//! ```sh
//! cargo run -p zend --release --example dependency_battery -- run \
//!   --zend http://192.168.1.2:8081 --node 223ebb209828d84c --repo D:\prog\candle \
//!   --out docs/results/battery_128/answers.jsonl
//! # the two baselines' three runs, and three more of C1 and B1, interleaved:
//! cargo run -p zend --release --example dependency_battery -- run \
//!   --zend http://192.168.1.2:8081 --node 223ebb209828d84c --repo D:\prog\candle \
//!   --out docs/results/battery_128/answers.jsonl --arms r1,t1,c1:6,b1:6 --runs 3
//! cargo run -p zend --release --example dependency_battery -- analyse \
//!   --answers docs/results/battery_128/answers.jsonl --kv-layers 12 \
//!   --out docs/results/battery_128
//! ```

mod analyse;
mod arms;
mod battery;
mod client;
mod direct;
mod figures;
mod grounding;
mod hits;
mod load;
mod placement;
mod prompt;
mod rag;
mod report;
mod report_baselines;
mod run;
mod schedule;
mod score;
mod seen;
mod selection;
mod stats;
mod telemetry;
mod view;

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use serde_json::Value;
use tracing_subscriber::EnvFilter;

use arms::Arm;
use battery::{checked, Battery, Question};
use rag::context::Packed;
use rag::Retriever;
use score::words;
use selection::arm_runs;

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
        /// The highest run each arm is asked to; the runs `out` already
        /// records are not asked again.
        #[arg(long, default_value_t = 3)]
        runs: u32,
        /// Ask only these arms, by id; `id:n` asks that arm to run `n`
        /// instead of `--runs`. Every arm when none is named.
        #[arg(long, value_delimiter = ',')]
        arms: Vec<String>,
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
        /// The K/V layers of the model that answered: the K/V census counts a
        /// token once per layer, the format census once. Flash-Next holds K/V
        /// on 12 of its 48 layers.
        #[arg(long)]
        kv_layers: u32,
        #[arg(long)]
        out: PathBuf,
    },
    /// Draw the paper's ten data figures as inline-SVG snippets from the
    /// CSVs `analyse` wrote into `dir`, into `dir/figures`.
    Figures {
        #[arg(long)]
        dir: PathBuf,
        /// The files the provenance gallery ranked the needle questions
        /// among: the random ranking's denominator.
        #[arg(long)]
        gallery_files: u32,
        /// The BF16 bytes one resident token's K/V takes over every K/V
        /// layer, as `summary.md`'s load table gives it.
        #[arg(long)]
        bf16_per_token: f64,
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

/// R1's files block for each of `questions`, from `repo` indexed without
/// `exclude` — the battery, whose `expect` lists are its answers, and the
/// folder its results go to.
fn retrieval(
    repo: &Path,
    exclude: &[PathBuf],
    questions: &[Question],
) -> anyhow::Result<HashMap<String, Packed>> {
    let retriever = Retriever::build(repo, exclude)?;
    println!(
        "R1 retrieves from {} files of {}",
        retriever.files(),
        repo.display()
    );
    Ok(questions
        .iter()
        .map(|q| (q.id.clone(), retriever.retrieve(&q.question)))
        .collect())
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
            arms,
            concurrency,
            seed,
            only,
        } => {
            let arms = arm_runs(&arms, runs)?;
            let battery_path = battery.unwrap_or_else(default_battery);
            let battery = checked(Battery::load(&battery_path)?, &repo)?;
            let questions: Vec<Question> = battery
                .questions
                .into_iter()
                .filter(|q| only.is_empty() || only.contains(&q.id))
                .collect();
            let results_dir = out.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
            fs::create_dir_all(&results_dir)?;
            let retrieval = if arms.iter().any(|(a, _)| *a == Arm::R1) {
                retrieval(&repo, &[battery_path, results_dir], &questions)?
            } else {
                HashMap::new()
            };
            let telemetry = telemetry.unwrap_or_else(|| beside(&out, "telemetry.jsonl"));
            run::run(run::Plan {
                base: &zend,
                node: &node,
                thread: &thread,
                questions,
                vocabulary: vocabulary(&repo)?,
                arms,
                retrieval,
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
            kv_layers,
            out,
        } => {
            let lines = run::lines(&answers)?;
            let telemetry = telemetry.unwrap_or_else(|| beside(&answers, "telemetry.jsonl"));
            let census = beside(&answers, "census_start.json");
            let bf16_per_token = match fs::read_to_string(&census) {
                Ok(text) => load::bf16_per_token(
                    &serde_json::from_str(&text)
                        .with_context(|| format!("parsing {}", census.display()))?,
                    kv_layers,
                ),
                Err(_) => None,
            };
            report::write(&lines, &samples(&telemetry)?, bf16_per_token, &out)?;
            println!("wrote {}", out.join("summary.md").display());
            Ok(())
        }
        Cmd::Figures {
            dir,
            gallery_files,
            bf16_per_token,
        } => {
            for path in figures::write(&dir, gallery_files, bf16_per_token)? {
                println!("wrote {}", path.display());
            }
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
