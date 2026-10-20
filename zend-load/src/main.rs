//! `zend-load`: the load ladder for a zend cluster, run through its gateway.
//!
//! Each rung runs its clients at once, each holding one conversation through
//! several turns, and prints per rung how requests ended, which member served
//! them, the spread of the time to first text and of the longest silence —
//! the stall a person feels — and the aggregate rate of text. Every
//! conversation the run made is deleted at the end.
//!
//! ```text
//! zend-load --levels light,medium,heavy --tasks decode,read
//! zend-load --gateway http://192.168.0.6:8081 --patterns heavy-read,blend,shift
//! ```
//!
//! `--patterns` runs named, fixed mixes ([`pattern`]) in place of the ladder,
//! each part's row printed and then the pattern's as a whole. `--gateway` may
//! name one member's own listener as well as the gateway, so the same pattern
//! measures a member alone and the cluster. `ingest-alone` follows an ingest
//! the operator started to its end ([`ingest`]); `ingest` does the same with
//! 12 decode clients starting 60 s after its backlog appears.

mod cleanup;
mod client;
mod ingest;
mod ladder;
mod pattern;
mod report;
mod run;
mod sample;
mod sse;
mod tasks;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use clap::Parser;

use client::Target;
use ingest::{watch, Watched};
use ladder::rung;
use pattern::{pattern, run_pattern};
use report::Row;
use tasks::{run_mark, Task};
use tokio::time::Instant;

#[derive(Parser)]
#[command(about = "Load ladder for a zend cluster, through its gateway")]
struct Args {
    /// The gateway's base URL.
    #[arg(long, default_value = "http://127.0.0.1")]
    gateway: String,
    /// The site the gateway routes zend by.
    #[arg(long, default_value = "code.tokera.com")]
    host: String,
    /// Rungs to run, in order: light, medium, heavy.
    #[arg(long, value_delimiter = ',', default_value = "light,medium,heavy")]
    levels: Vec<String>,
    /// Tasks per rung: decode (passthrough), read (zen-code on the candle
    /// code), write (zen-code editing battle-cities in its overlay).
    #[arg(long, value_delimiter = ',', default_value = "decode,read,write")]
    tasks: Vec<String>,
    /// Named patterns to run in place of the ladder: heavy-decode,
    /// heavy-read, blend, shift, story-x16.
    #[arg(long, value_delimiter = ',')]
    patterns: Vec<String>,
    /// The longest one turn may take before it counts as cut.
    #[arg(long, default_value_t = 900)]
    turn_limit_s: u64,
    /// Leave the run's conversations in place.
    #[arg(long)]
    keep: bool,
    /// Print every request's measurements, not only each rung's summary.
    #[arg(long)]
    samples: bool,
    /// Write every answered turn's reply to this folder, one file per turn
    /// named `<scenario>-c<client>-t<turn>.txt`: what a wrong rewrite said.
    #[arg(long)]
    replies: Option<PathBuf>,
}

/// Write `ran`'s replies under `dir`, named for `scenario`; a reply that cannot
/// be written is reported and the run goes on.
fn keep_replies(dir: &Path, scenario: &str, ran: &run::Ran) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("  replies: {} not created: {e}", dir.display());
        return;
    }
    for (client, turn, text) in &ran.replies {
        let path = dir.join(format!("{scenario}-c{client}-t{turn}.txt"));
        if let Err(e) = std::fs::write(&path, text) {
            eprintln!("  replies: {} not written: {e}", path.display());
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut tasks = Vec::new();
    for name in &args.tasks {
        match Task::parse(name) {
            Some(t) => tasks.push(t),
            None => bail!("no task `{name}`: decode, read, write or story"),
        }
    }
    let mut scenarios = Vec::new();
    for level in &args.levels {
        for task in &tasks {
            match rung(level, *task) {
                Some(s) => scenarios.push(s),
                None => bail!("no level `{level}`: light, medium or heavy"),
            }
        }
    }
    let base = args.gateway.trim_end_matches('/').to_string();
    let target = Arc::new(Target {
        http: reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()?,
        chat_url: format!("{base}/v1/chat/completions"),
        host: args.host.clone(),
    });
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let limit = Duration::from_secs(args.turn_limit_s);
    let mut rows = Vec::new();
    if !args.patterns.is_empty() {
        let mut patterns = Vec::new();
        for name in &args.patterns {
            if matches!(name.as_str(), "ingest" | "ingest-alone") {
                continue;
            }
            match pattern(name) {
                Some(p) => patterns.push(p),
                None => bail!(
                    "no pattern `{name}`: heavy-decode, heavy-read, blend, shift, story-x16, \
                     ingest or ingest-alone"
                ),
            }
        }
        for name in args
            .patterns
            .iter()
            .filter(|n| matches!(n.as_str(), "ingest" | "ingest-alone"))
        {
            let beside = (name == "ingest").then(|| ladder::Scenario {
                name: "ingest-decode".into(),
                task: Task::Decode,
                clients: INGEST_DECODE_CLIENTS,
                turns: 2,
                max_tokens: Task::Decode.max_tokens(),
            });
            let ran = run_ingest(
                Arc::clone(&target),
                &base,
                beside,
                &format!("zl{stamp}"),
                limit,
            )
            .await;
            println!("{}", describe_ingest(name, &ran.watched));
            if let Some((scenario, run_name, decode)) = &ran.decode {
                if args.samples {
                    for s in &decode.samples {
                        println!("  {s}");
                    }
                }
                let row = Row::of(&scenario.name, &decode.samples, decode.wall);
                println!("{row}");
                rows.push(row);
                if !args.keep {
                    let mark = format!("{} ", run_mark(run_name));
                    let (gone, kept) =
                        cleanup::delete_run(&target, &base, &decode.conversations, &mark).await;
                    println!("  cleanup: {gone} conversations deleted, {kept} refused");
                }
            }
        }
        for p in &patterns {
            println!("running pattern {} ...", p.name);
            let started = Instant::now();
            let parts = run_pattern(Arc::clone(&target), p, &format!("zl{stamp}"), limit).await;
            let wall = started.elapsed();
            let mut all = Vec::new();
            for (scenario, run_name, ran) in &parts {
                if args.samples {
                    for s in &ran.samples {
                        println!("  {s}");
                    }
                }
                if let Some(dir) = &args.replies {
                    keep_replies(dir, &scenario.name, ran);
                }
                let row = Row::of(&scenario.name, &ran.samples, ran.wall);
                println!("{row}");
                rows.push(row);
                all.extend(ran.samples.iter().cloned());
                if !args.keep {
                    let mark = format!("{} ", run_mark(run_name));
                    let (gone, kept) =
                        cleanup::delete_run(&target, &base, &ran.conversations, &mark).await;
                    println!("  cleanup: {gone} conversations deleted, {kept} refused");
                }
            }
            if parts.len() > 1 {
                let row = Row::of(&p.name, &all, wall);
                println!("{row}");
                rows.push(row);
            }
        }
        println!("\nsummary");
        for row in &rows {
            println!("{row}");
        }
        return Ok(());
    }
    for scenario in &scenarios {
        let run_name = format!("zl{stamp}-{}", scenario.name);
        println!(
            "running {}: {} clients x {} turns ...",
            scenario.name, scenario.clients, scenario.turns
        );
        let ran = run::run(Arc::clone(&target), scenario, &run_name, limit).await;
        if args.samples {
            for s in &ran.samples {
                println!("  {s}");
            }
        }
        if let Some(dir) = &args.replies {
            keep_replies(dir, &scenario.name, &ran);
        }
        let row = Row::of(&scenario.name, &ran.samples, ran.wall);
        println!("{row}");
        rows.push(row);
        if !args.keep {
            let mark = format!("{} ", run_mark(&run_name));
            let (gone, kept) = cleanup::delete_run(&target, &base, &ran.conversations, &mark).await;
            println!("  cleanup: {gone} conversations deleted, {kept} refused");
        }
    }
    println!("\nsummary");
    for row in &rows {
        println!("{row}");
    }
    Ok(())
}

/// Decode clients started beside an ingest.
const INGEST_DECODE_CLIENTS: usize = 12;

/// How long after the backlog appears they start.
const INGEST_DECODE_AFTER: Duration = Duration::from_secs(60);

/// How long a watch waits for the ingest's backlog to appear.
const INGEST_APPEAR_WAIT: Duration = Duration::from_secs(30 * 60);

/// The longest a watched ingest may run.
const INGEST_LIMIT: Duration = Duration::from_secs(4 * 60 * 60);

struct IngestRan {
    watched: Watched,
    decode: Option<(ladder::Scenario, String, run::Ran)>,
}

/// Watch an ingest at `base`, and with `beside`, run those decode clients
/// starting [`INGEST_DECODE_AFTER`] after its backlog appears.
async fn run_ingest(
    target: Arc<Target>,
    base: &str,
    beside: Option<ladder::Scenario>,
    stamp: &str,
    limit: Duration,
) -> IngestRan {
    let (appeared, wait) = tokio::sync::oneshot::channel::<()>();
    let decode = beside.map(|scenario| {
        let target = Arc::clone(&target);
        let run_name = format!("{stamp}-{}", scenario.name);
        tokio::spawn(async move {
            if wait.await.is_err() {
                return None;
            }
            tokio::time::sleep(INGEST_DECODE_AFTER).await;
            let ran = run::run(target, &scenario, &run_name, limit).await;
            Some((scenario, run_name, ran))
        })
    });
    let mut appeared = Some(appeared);
    let watched = watch(&target, base, INGEST_APPEAR_WAIT, INGEST_LIMIT, || {
        if let Some(tx) = appeared.take() {
            let _ = tx.send(());
        }
    })
    .await;
    drop(appeared);
    let decode = match decode {
        Some(h) => h.await.ok().flatten(),
        None => None,
    };
    IngestRan { watched, decode }
}

fn describe_ingest(name: &str, w: &Watched) -> String {
    match (w.wall, w.units_per_minute()) {
        (Some(wall), Some(rate)) => format!(
            "{name:<14} ingest of {} units in {:.0}s ({rate:.1} units/min), backlog appeared after {:.0}s",
            w.units,
            wall.as_secs_f64(),
            w.appeared_after.map_or(0.0, |d| d.as_secs_f64())
        ),
        _ if w.appeared_after.is_none() => format!(
            "{name:<14} no ingest backlog appeared within {} min",
            INGEST_APPEAR_WAIT.as_secs() / 60
        ),
        _ => format!(
            "{name:<14} ingest of {}+ units still running after {} h",
            w.units,
            INGEST_LIMIT.as_secs() / 3600
        ),
    }
}
