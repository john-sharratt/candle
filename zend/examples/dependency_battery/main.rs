//! `dependency_battery` — the codebase dependency battery of
//! `docs/unbounded_agents.md` §9.12, run against a workspace's substrate.
//!
//! Each question in `battery.yaml` is put to the loaded model as a fresh
//! conversation with tools off (`ZendSession::ask`), and each answer's final
//! projection is recorded by layer and collection. **Measured 2026-11-10, that
//! projection holds no turn and no collection member**: a conversation forked
//! from the base conversation with the chat's projection does not draw on the
//! code-reading layer as a production chat does, so on this path the arms
//! measure the model without its memory
//! (`docs/results/dependency_battery_2026-11-10.md`). The battery's production
//! scores come from the chat itself. Each answer is scored for recall against the
//! question's `expect` list and for grounding against the repository's own
//! vocabulary (`score.rs`), and written as one JSON line.
//!
//! The arms are one run each:
//!
//! - `full` — the chat as it runs: reprojection during the decode;
//! - `one-shot` — the working set its opening projection chose, never
//!   re-selected during the decode;
//! - `no-memory` — the code-reading layer out of service: the model and its
//!   system prompt alone.
//!
//! The model takes the whole card and the substrate is opened read-only, so
//! stop the daemon first:
//! ```sh
//! cargo run -p zend --release --example dependency_battery -- \
//!   --workspace D:\prog --repo D:\prog\candle --agent-checkouts D:\agent --arm full \
//!   --out docs/results/dependency_battery_full.jsonl
//! ```

mod score;

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent::checkouts::Checkouts;
use agent::config::{AgentConfig, CheckoutConfig, SweepConfig};
use agent::jobs::Jobs;
use anyhow::{bail, Context};
use candle_conversation::persistence::SUBSTRATE_DIR;
use candle_conversation::projection::SystemItem;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;
use zend::api::chat::{apply_tools_dial, dial_selection};
use zend::config::DaemonConfig;
use zend::session::{AskSpec, ZendSession};
use zend::substrate_git::open_workspace;
use zend::types::ToolMode;
use zend_vfs::sandbox::programs::policy;
use zend_vfs::{Priority, Workspace};

use score::{answer, grounding, recall, words};

#[derive(Parser)]
#[command(about = "Run the §9.12 dependency battery against a workspace's substrate, read-only")]
struct Args {
    /// The workspace whose `substrate/` holds the code-reading layer.
    #[arg(long)]
    workspace: PathBuf,
    /// The repository the questions are about, read for the grounding
    /// vocabulary.
    #[arg(long)]
    repo: PathBuf,
    /// The folder holding a checkout of each workspace repository by name —
    /// the daemon's agent's (`motherd/agent.yaml`).
    #[arg(long)]
    agent_checkouts: PathBuf,
    #[arg(long, value_enum)]
    arm: Arm,
    /// The battery; defaults to the one beside this example.
    #[arg(long)]
    battery: Option<PathBuf>,
    /// Where the JSON lines go.
    #[arg(long)]
    out: PathBuf,
    /// The decode limit per question, in tokens.
    #[arg(long, default_value_t = 3_072)]
    max_tokens: usize,
    /// The thinking-effort dial, 0 (off) to 4.
    #[arg(long, default_value_t = 2)]
    effort: u8,
    /// Ask only these question ids.
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
}

#[derive(Clone, Copy, ValueEnum, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Arm {
    Full,
    OneShot,
    NoMemory,
}

#[derive(Deserialize)]
struct Battery {
    questions: Vec<Question>,
}

#[derive(Deserialize, Clone)]
struct Question {
    id: String,
    category: String,
    question: String,
    expect: Vec<String>,
}

#[derive(Serialize)]
struct Line<'a> {
    arm: Arm,
    id: &'a str,
    category: &'a str,
    recall: f64,
    missed: Vec<String>,
    grounding: Option<f64>,
    ungrounded: Vec<String>,
    tokens: usize,
    projections: usize,
    /// The final projection's turns by layer: what the answer was decoded
    /// against.
    projected: BTreeMap<String, usize>,
    /// The final projection's system-prompt collections: members selected,
    /// members in all, and the selected members' tokens.
    collections: BTreeMap<String, (usize, usize, u32)>,
    /// Whether the answer issued a tool call — with tools off, a sign the
    /// model looked for the code rather than recalling it.
    tool_call: bool,
    decode_secs: f64,
    answer: &'a str,
}

/// The agent the session's repositories are read through, as the daemon's
/// agent file lays it out — a checkout of each manifest repository under
/// `checkouts`, by name — but never pushed and with no journal sync, its jobs
/// folder beside the results. Every question runs with tools off, so nothing
/// is run on it; the engine needs one to load.
fn agent_for(workspace: &Path, checkouts: &Path, out: &Path) -> anyhow::Result<Arc<Jobs>> {
    let jobs_dir = out
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("battery_agent_jobs");
    let cfg = AgentConfig {
        jobs_dir,
        slots: Some(1),
        stage_dir: None,
        checkouts: Workspace::manifest(workspace)?
            .into_iter()
            .map(|spec| CheckoutConfig {
                dir: checkouts.join(&spec.name),
                name: spec.name,
                push: false,
                journal: false,
            })
            .collect(),
        sweep: SweepConfig::default(),
    };
    let checkouts = Checkouts::open(&cfg, &policy()).context("opening the agent's checkout")?;
    Ok(Jobs::new(checkouts, 1, Priority::Normal))
}

/// Every identifier-like word in the repository's source and documents.
fn vocabulary(repo: &Path) -> anyhow::Result<HashSet<String>> {
    let mut vocab = HashSet::new();
    let mut stack = vec![repo.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let p = e?.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if p.is_dir() {
                if !matches!(name, "target" | ".git" | "node_modules") {
                    stack.push(p);
                }
            } else if matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("rs" | "cu" | "cuh" | "md" | "yaml" | "toml")
            ) {
                vocab.extend(words(name).map(str::to_string));
                if let Ok(text) = std::fs::read_to_string(&p) {
                    vocab.extend(words(&text).map(str::to_string));
                }
            }
        }
    }
    Ok(vocab)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    if !args.workspace.join(SUBSTRATE_DIR).is_dir() {
        bail!("{} holds no substrate directory", args.workspace.display());
    }
    let battery_path = args.battery.clone().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/dependency_battery/battery.yaml")
    });
    let battery: Battery = serde_yaml::from_str(
        &std::fs::read_to_string(&battery_path)
            .with_context(|| format!("reading {}", battery_path.display()))?,
    )
    .context("parsing the battery")?;
    let questions: Vec<Question> = battery
        .questions
        .into_iter()
        .filter(|q| args.only.is_empty() || args.only.contains(&q.id))
        .collect();
    let vocab = vocabulary(&args.repo)?;
    println!(
        "{} questions · arm {:?} · vocabulary {} words",
        questions.len(),
        args.arm,
        vocab.len()
    );

    let disabled: HashSet<String> = match args.arm {
        Arm::NoMemory => ["code_reading".to_string()].into_iter().collect(),
        Arm::Full | Arm::OneShot => HashSet::new(),
    };
    // The workspace as the daemon opens it: every manifest repository read
    // from the substrate, through an agent holding a checkout of each.
    let config = DaemonConfig {
        disabled_layers: disabled,
        read_only_substrate: true,
        agent: Some(agent_for(
            &args.workspace,
            &args.agent_checkouts,
            &args.out,
        )?),
        ..DaemonConfig::new(open_workspace(&args.workspace)?)
    };
    let session = Arc::new(ZendSession::new(config));
    session.start_loading();
    session.wait_ready().await;

    let mut selection = dial_selection(Some(args.effort), None, Some(args.effort != 0));
    apply_tools_dial(&mut selection, ToolMode::None);
    let mut out =
        File::create(&args.out).with_context(|| format!("creating {}", args.out.display()))?;
    let mut by_category: BTreeMap<String, (f64, f64, usize, usize)> = BTreeMap::new();
    for q in &questions {
        let spec = AskSpec {
            id: format!("{:?}/{}", args.arm, q.id),
            question: q.question.clone(),
            max_tokens: args.max_tokens,
            selection: selection.clone(),
            reproject: !matches!(args.arm, Arm::OneShot),
        };
        let runner = Arc::clone(&session);
        let outcome = tokio::task::spawn_blocking(move || runner.ask(&spec))
            .await
            .context("the question's task")??;
        let said = answer(&outcome.text);
        let (r, missed) = recall(said, &q.expect);
        let g = grounding(said, &vocab);
        let line = Line {
            arm: args.arm,
            id: &q.id,
            category: &q.category,
            recall: r,
            missed,
            grounding: g.as_ref().map(|(g, _)| *g),
            ungrounded: g.map(|(_, u)| u).unwrap_or_default(),
            tokens: outcome.tokens,
            projections: outcome.projections,
            projected: outcome
                .last_projection
                .iter()
                .flat_map(|p| p.selection.turns.iter())
                .fold(BTreeMap::new(), |mut m, t| {
                    *m.entry(t.layer.clone()).or_insert(0) += 1;
                    m
                }),
            collections: outcome
                .last_projection
                .iter()
                .flat_map(|p| p.selection.system.iter())
                .filter_map(|item| match item {
                    SystemItem::Collection { name, sections, .. } => Some((
                        name.clone(),
                        (
                            sections.iter().filter(|s| s.selected).count(),
                            sections.len(),
                            sections
                                .iter()
                                .filter(|s| s.selected)
                                .map(|s| s.tokens)
                                .sum(),
                        ),
                    )),
                    _ => None,
                })
                .collect(),
            tool_call: said.contains("<tool_call>") || said.contains("<tool_use>"),
            decode_secs: outcome.decode_secs,
            answer: said,
        };
        writeln!(out, "{}", serde_json::to_string(&line)?)?;
        out.flush()?;
        println!(
            "{:<4} {:<13} recall {:>5.1}%  grounding {:>6}  {} tokens  {} projections  \
             projected {:?}  collections {:?}  tool call {}  missed {:?}",
            q.id,
            q.category,
            r * 100.0,
            line.grounding
                .map_or("—".to_string(), |g| format!("{:.1}%", g * 100.0)),
            line.tokens,
            line.projections,
            line.projected,
            line.collections,
            line.tool_call,
            line.missed
        );
        let e = by_category.entry(q.category.clone()).or_default();
        e.0 += r;
        if let Some(g) = line.grounding {
            e.1 += g;
            e.3 += 1;
        }
        e.2 += 1;
    }

    println!("\n| Arm | Category | Questions | Mean recall | Mean grounding |");
    println!("|---|---|---:|---:|---:|");
    for (cat, (r, g, n, ng)) in &by_category {
        println!(
            "| {:?} | {cat} | {n} | {:.1}% | {} |",
            args.arm,
            r / *n as f64 * 100.0,
            if *ng == 0 {
                "—".to_string()
            } else {
                format!("{:.1}%", g / *ng as f64 * 100.0)
            }
        );
    }
    Ok(())
}
