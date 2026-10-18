//! `coop-lab`: fit each member's forward cost from its wave logs, and compare
//! placement policies on the cluster simulator.
//!
//! ```text
//! coop-lab fit --member BlackWorld=motherd.log.2,motherd.log --member WhiteWorld=ww.json
//! coop-lab sim
//! ```
//!
//! A log file is a plain log, or the JSON a member's `/v1/logs` answers (its
//! `lines`, or each member's under `members[].body.lines`).

use std::fs;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::Value;

use coop::curve::{curve, Point};
use coop::experts::ExpertGeometry;
use coop::fit::{fit, Fitted};
use coop::sample::{Channel, Sample};
use coop::sim::engine::{simulate, Outage};
use coop::sim::link::Link;
use coop::sim::model::MemberModel;
use coop::sim::policy::{Concentrate, EarliestFinish, Policy, Today};
use coop::sim::report::Report;
use coop::sim::workload::{generate, Conversation, Kind};
use coop::wave_log::parse;

#[derive(Parser)]
#[command(about = "Cooperative admission: member cost fits and cluster placement experiments")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fit each member's forward cost from its scheduler wave log.
    Fit {
        /// `NAME=FILE[,FILE…]`, once per member.
        #[arg(long, required = true)]
        member: Vec<String>,
    },
    /// Run the placement policies over the heavy rungs' workloads.
    Sim {
        /// Conversations per scenario.
        #[arg(long, default_value_t = 24)]
        conversations: usize,
        /// Seconds over which they arrive.
        #[arg(long, default_value_t = 10.0)]
        spread: f64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Fit { member } => fit_members(&member),
        Command::Sim {
            conversations,
            spread,
            seed,
        } => {
            sim(conversations, spread, seed);
            Ok(())
        }
    }
}

fn policies() -> Vec<Box<dyn Policy>> {
    vec![
        Box::new(Today),
        Box::new(Concentrate { floor_hit: 0.9 }),
        Box::new(EarliestFinish),
    ]
}

fn sim(n: usize, spread: f64, seed: u64) {
    let models = [MemberModel::black_world(), MemberModel::white_world()];
    for m in &models {
        if !m.measured {
            println!(
                "note: {} is assumed from its gate figures, not measured",
                m.name
            );
        }
    }
    let half = n / 2;
    let mut mixed = generate(Kind::Chat, half, spread, seed);
    for mut c in generate(Kind::Read, n - half, spread, seed + 1) {
        c.id += half;
        mixed.push(c);
    }
    let scenarios: Vec<(&str, Vec<Conversation>)> = vec![
        ("chat", generate(Kind::Chat, n, spread, seed)),
        ("read", generate(Kind::Read, n, spread, seed)),
        ("chat + read", mixed),
    ];
    for (name, work) in &scenarios {
        for outage in [
            None,
            Some(Outage {
                member: 1,
                at: 30.0,
            }),
        ] {
            let title = match outage {
                None => format!("{name}, {n} conversations"),
                Some(o) => format!(
                    "{name}, {n} conversations, {} drops at {} s",
                    models[o.member].name, o.at
                ),
            };
            println!("\n{title}");
            print_header();
            for mut p in policies() {
                print_row(&simulate(
                    &models,
                    work,
                    p.as_mut(),
                    Link::gigabit(),
                    outage,
                ));
            }
        }
    }
}

fn print_header() {
    println!(
        "  {:<30} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>9} {:>6} {:>4} {:>5}",
        "policy",
        "tok/s",
        "first50",
        "first95",
        "gap95",
        "gapmax",
        "slow95",
        "jain",
        "turns B/W",
        "linkGB",
        "rst",
        "moves"
    );
}

fn print_row(r: &Report) {
    let split = r
        .per_member
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("/");
    println!(
        "  {:<30} {:>7.0} {:>7.1}s {:>7.1}s {:>7.2}s {:>7.2}s {:>8.1} {:>6.3} {:>9} {:>6.1} {:>4} {:>5}",
        r.policy,
        r.tokens_per_s,
        r.first_token_p50,
        r.first_token_p95,
        r.longest_gap_p95,
        r.longest_gap_max,
        r.slowdown_p95,
        r.fairness,
        split,
        r.link_gb,
        r.restarted,
        r.moves
    );
}

fn fit_members(members: &[String]) -> Result<()> {
    let geometry = ExpertGeometry::FLASH_NEXT;
    for spec in members {
        let Some((name, files)) = spec.split_once('=') else {
            bail!("--member takes NAME=FILE[,FILE…], not {spec:?}");
        };
        let mut samples = Vec::new();
        for f in files.split(',') {
            samples.extend(samples_of(f)?);
        }
        println!("{name}: {} samples", samples.len());
        let alone: Vec<Sample> = samples.iter().filter(|s| s.alone).cloned().collect();
        for (label, set) in [("all windows", &samples), ("channel alone", &alone)] {
            for channel in [Channel::Decode, Channel::Prefill] {
                match fit(set, channel, &geometry) {
                    Some(f) => println!("  {channel:?} ({label}): {}", describe(&f)),
                    None => println!("  {channel:?} ({label}): too few samples"),
                }
            }
        }
        println!("  decode, alone:");
        print_curve(&curve(&samples, Channel::Decode, |s| s.alone));
        println!("  decode, beside prefill:");
        print_curve(&curve(&samples, Channel::Decode, |s| !s.alone));
        println!("  prefill, all:");
        print_curve(&curve(&samples, Channel::Prefill, |_| true));
    }
    Ok(())
}

fn print_curve(points: &[Point]) {
    println!(
        "    {:>6} {:>7} {:>8} {:>7} {:>8} {:>10} {:>8} {:>6} {:>9}",
        "rows≥", "windows", "fwds", "rows", "tokens", "kv", "ms", "hit", "tok/s"
    );
    for p in points {
        println!(
            "    {:>6} {:>7} {:>8} {:>7.1} {:>8.0} {:>10.0} {:>8.1} {:>6.3} {:>9.0}",
            p.from,
            p.windows,
            p.fwds,
            p.rows,
            p.tokens,
            p.kv,
            p.ms,
            p.hit,
            p.tokens_per_s()
        );
    }
}

/// Every sample in the file at `path`.
fn samples_of(path: &str) -> Result<Vec<Sample>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let lines: Vec<String> = if text.trim_start().starts_with('{') {
        json_lines(&serde_json::from_str(&text).with_context(|| format!("parsing {path}"))?)
    } else {
        text.lines().map(String::from).collect()
    };
    Ok(lines.iter().flat_map(|l| parse(l)).collect())
}

/// The log lines of a `/v1/logs` answer.
fn json_lines(v: &Value) -> Vec<String> {
    let strings = |l: &Value| -> Vec<String> {
        l.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    match v["members"].as_array() {
        Some(ms) => ms
            .iter()
            .flat_map(|m| strings(&m["body"]["lines"]))
            .collect(),
        None => strings(&v["lines"]),
    }
}

fn describe(f: &Fitted) -> String {
    let c = &f.cost;
    format!(
        "base {:.2} ms + {:.4} ms/token + {:.3} ms/MKV + {:.2} ms/GB resident + {:.2} ms/GB missed \
         (R² {:.3}, rms {:.1} ms, {} windows, {} fwds)",
        c.base_ms,
        c.token_ms,
        c.mkv_ms,
        c.resident_gb_ms,
        c.missed_gb_ms,
        f.r2,
        f.rms_ms,
        f.samples,
        f.fwds
    )
}
