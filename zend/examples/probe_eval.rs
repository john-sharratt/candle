//! Retrieval eval for the `repo_map` **probe layer** — does a question about a
//! folder actually retrieve that folder?
//!
//! Drives a running daemon over `POST /v1/substrate/project`, which captures the
//! query's live decode-Q, scores it against the substrate, and returns the
//! selected projection tiles while writing nothing. No extra VRAM, and no risk
//! of the measurement contaminating the corpus it measures.
//!
//! Reads `.zend/probe_holdout.json`, written by the ingest. Two populations live
//! in it and they answer different questions:
//!
//! * **held-out** (`resident: false`) — admissible candidates that lost on slots
//!   alone and were never ingested. The corpus has never seen them, so this is
//!   the quality measure.
//! * **resident** (`resident: true`) — the probes that WERE ingested. Their own
//!   signatures are in the corpus, so they self-match; a pass proves only
//!   self-consistency. Scored anyway as a necessary condition — a probe that
//!   cannot retrieve its own folder from that position is broken beyond
//!   argument, and one that retrieves somebody else's is actively harmful.
//!
//! ```text
//! cargo run -p zend --example probe_eval --release -- \
//!     --url http://127.0.0.1:8080 --limit 400 --wait-ready
//! ```

use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use zend::repo_scan::probe::harness::{Scoreboard, Trial};
use zend::repo_scan::probe::Register;
use zend::repo_scan::probe_pass::{read_holdout, HoldoutQuery};

const REPO_LAYER: &str = "repo_map";

#[derive(Deserialize)]
struct Tile {
    #[serde(default)]
    layer: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    score: f32,
    #[serde(default)]
    selected: bool,
}

#[derive(Deserialize)]
struct ProjView {
    #[serde(default)]
    tiles: Vec<Tile>,
}

#[derive(Deserialize)]
struct StatusResp {
    #[serde(default)]
    state: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut url = "http://127.0.0.1:8080".to_string();
    let mut workspace = std::path::PathBuf::from(".");
    let mut limit: Option<usize> = None;
    let mut wait_ready = false;
    let mut verbose = false;
    let mut include_resident = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--url" => url = args.next().context("--url needs a value")?,
            "--workspace" => {
                workspace = std::path::PathBuf::from(args.next().context("--workspace")?)
            }
            "--limit" => limit = Some(args.next().context("--limit needs N")?.parse()?),
            "--wait-ready" => wait_ready = true,
            "--verbose" => verbose = true,
            "--include-resident" => include_resident = true,
            other => anyhow::bail!("unknown arg {other:?}"),
        }
    }
    let url = url.trim_end_matches('/').to_string();

    let all = read_holdout(&workspace).context("reading the probe holdout file")?;
    let held: Vec<HoldoutQuery> = all
        .iter()
        .filter(|q| include_resident || !q.resident)
        .cloned()
        .collect();
    anyhow::ensure!(!held.is_empty(), "holdout file has no queries to score");

    // Even-stride sample so a cap still spans every directory rather than
    // truncating to whichever sorted first — the file is ordered by directory,
    // so a head-truncation would score `candle-core/` and nothing else.
    let cases = even_sample(&held, limit.unwrap_or(held.len()));
    eprintln!(
        "{} of {} queries sampled ({} held-out, {} resident, {} directories)",
        cases.len(),
        held.len(),
        all.iter().filter(|q| !q.resident).count(),
        all.iter().filter(|q| q.resident).count(),
        {
            let mut dirs: Vec<&str> = all.iter().map(|q| q.dir.as_str()).collect();
            dirs.sort_unstable();
            dirs.dedup();
            dirs.len()
        },
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()?;

    if wait_ready {
        eprintln!("waiting for the daemon at {url} to finish loading…");
        loop {
            if let Ok(r) = client.get(format!("{url}/v1/status")).send().await {
                if let Ok(s) = r.json::<StatusResp>().await {
                    if s.state != "loading" {
                        eprintln!("daemon state = {:?} — proceeding", s.state);
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    }

    let mut trials: Vec<Trial> = Vec::new();
    let mut unavailable = 0usize;
    for (i, case) in cases.iter().enumerate() {
        let resp = client
            .post(format!("{url}/v1/substrate/project"))
            .json(&serde_json::json!({ "text": case.query }))
            .send()
            .await
            .with_context(|| format!("POST project for {:?}", case.query))?;
        if !resp.status().is_success() {
            unavailable += 1;
            continue;
        }
        let view: ProjView = resp.json().await.context("parsing the projection view")?;
        // Selected repo_map tiles, best score first. `selected` is what the
        // projection actually injected, which is the decision being measured —
        // a tile that scored well but lost its budget slot did not retrieve.
        let mut ranked: Vec<(f32, String)> = view
            .tiles
            .iter()
            .filter(|t| t.layer == REPO_LAYER && t.selected)
            .map(|t| (t.score, t.label.clone()))
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let trial = Trial {
            query: case.query.clone(),
            expected_dir: case.dir.clone(),
            register: Register::from_id(&case.register).unwrap_or(Register::Systemic),
            resident: case.resident,
            ranked: ranked.into_iter().map(|(_, label)| label).collect(),
        };
        if verbose {
            eprintln!(
                "  [{:>4}/{}] {} {:<10} {} -> {:?}",
                i + 1,
                cases.len(),
                if trial.hit_at_1() { "HIT " } else { "miss" },
                case.register,
                case.dir,
                trial.ranked.first(),
            );
        } else if (i + 1) % 50 == 0 {
            eprintln!("  {} / {}", i + 1, cases.len());
        }
        trials.push(trial);
    }

    if unavailable > 0 {
        eprintln!("{unavailable} queries could not be scored (daemon not serving projections)");
    }

    let board = Scoreboard::tally(&trials);
    println!();
    println!("── PROBE RETRIEVAL ──────────────────────────────────────────────");
    println!("held-out trials  {}", board.held_out.trials);
    println!(
        "  hit@1          {:>6.1}%   ({}/{})",
        board.held_out.hit_at_1_pct(),
        board.held_out.hit_at_1,
        board.held_out.trials,
    );
    println!(
        "  hit@3          {:>6.1}%   ({}/{})",
        board.held_out.hit_at_3_pct(),
        board.held_out.hit_at_3,
        board.held_out.trials,
    );
    println!("  MRR            {:>6.3}", board.held_out.mrr());
    if board.resident.trials > 0 {
        println!();
        println!(
            "resident (self-match, necessary condition only)  hit@1 {:.1}%  hit@3 {:.1}%  n={}",
            board.resident.hit_at_1_pct(),
            board.resident.hit_at_3_pct(),
            board.resident.trials,
        );
    }
    println!();
    println!("per register (held-out):");
    for (name, score) in &board.by_register {
        println!(
            "  {name:<12} hit@1 {:>6.1}%  hit@3 {:>6.1}%  MRR {:>5.3}  n={}",
            score.hit_at_1_pct(),
            score.hit_at_3_pct(),
            score.mrr(),
            score.trials,
        );
    }
    let usurpers = board.top_usurpers(8);
    if !usurpers.is_empty() {
        println!();
        println!("folders most often taking first place from the right answer:");
        for (dir, n) in usurpers {
            println!("  {n:>4}x  {dir}");
        }
    }
    println!();
    println!("worst misses:");
    for miss in board.misses.iter().take(12) {
        println!(
            "  [{}] {} \n        expected {}  got {:?}",
            miss.register.id(),
            miss.query,
            miss.expected_dir,
            miss.ranked.first(),
        );
    }
    Ok(())
}

/// Even-stride sample of `all`, preserving order.
fn even_sample(all: &[HoldoutQuery], n: usize) -> Vec<HoldoutQuery> {
    if n >= all.len() {
        return all.to_vec();
    }
    (0..n).map(|i| all[i * all.len() / n].clone()).collect()
}
