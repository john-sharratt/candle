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
//! measures a member alone and the cluster.

mod cleanup;
mod client;
mod ladder;
mod pattern;
mod report;
mod run;
mod sample;
mod sse;
mod tasks;

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use clap::Parser;

use client::Target;
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
    /// heavy-read, blend, shift.
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut tasks = Vec::new();
    for name in &args.tasks {
        match Task::parse(name) {
            Some(t) => tasks.push(t),
            None => bail!("no task `{name}`: decode, read or write"),
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
            match pattern(name) {
                Some(p) => patterns.push(p),
                None => bail!("no pattern `{name}`: heavy-decode, heavy-read, blend or shift"),
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
