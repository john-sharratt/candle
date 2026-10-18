//! `coop-lab`: fit each member's forward cost from its wave logs.
//!
//! ```text
//! coop-lab --member BlackWorld=motherd.log.2,motherd.log --member WhiteWorld=ww.json
//! ```
//!
//! A file is a plain log, or the JSON a member's `/v1/logs` answers (its
//! `lines`, or each member's under `members[].body.lines`).

use std::fs;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::Value;

use coop::curve::{curve, Point};
use coop::experts::ExpertGeometry;
use coop::fit::{fit, Fitted};
use coop::sample::{Channel, Sample};
use coop::wave_log::parse;

#[derive(Parser)]
#[command(about = "Fit each zend member's forward cost from its scheduler wave log")]
struct Args {
    /// `NAME=FILE[,FILE…]`, once per member.
    #[arg(long, required = true)]
    member: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let geometry = ExpertGeometry::FLASH_NEXT;
    for spec in &args.member {
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
        for (label, keep) in [
            (
                "decode, alone",
                Box::new(|s: &Sample| s.alone) as Box<dyn Fn(&Sample) -> bool>,
            ),
            ("decode, beside prefill", Box::new(|s: &Sample| !s.alone)),
        ] {
            println!("  {label}:");
            print_curve(&curve(&samples, Channel::Decode, keep));
        }
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
