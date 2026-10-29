//! Independent asks through a running daemon's project API
//! (`POST /v1/substrate/zend/project`), graded: for each ask in `asks.yaml`,
//! where the tools it names rank in the projection's tool beliefs, and which
//! turns the projection selected.
//!
//! Every run is kept under `target/project-asks/<label>.json`, and a run given
//! `--against <label>` is compared with that one ask by ask — the tool ranks,
//! and how much of each ask's selected turns the two share — so a change to the
//! gallery is judged by what it does to projections nobody calibrated on.
//!
//! Run, from the repository root:
//!   cargo run -p candle-conversation --release --example project_asks -- \
//!     --daemon <url> --node <hex> --label <name> [--against <name>]

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// Where runs are kept, from the repository root.
const RUNS: &str = "target/project-asks";

/// The asks, beside this file.
const ASKS: &str = "candle-conversation/examples/project_asks/asks.yaml";

/// The admin identity a member's routes accept from the gateway.
const ADMIN_HEADERS: [(&str, &str); 3] = [
    ("x-tokera-user", "forged-admin"),
    ("x-tokera-provider", "google"),
    ("x-tokera-email", "johnathan.sharratt@gmail.com"),
];

#[derive(Deserialize)]
struct Ask {
    ask: String,
    tools: Vec<String>,
}

#[derive(Deserialize)]
struct ProjectView {
    tiles: Vec<Tile>,
}

#[derive(Deserialize)]
struct Tile {
    kind: String,
    selected: bool,
    label: String,
    timeline: Option<String>,
    index: Option<u32>,
}

/// One ask's projection, as kept.
#[derive(Serialize, Deserialize, Clone)]
struct Graded {
    ask: String,
    /// The tools in belief order.
    tools: Vec<String>,
    /// The 1-based rank of the best-ranked expected tool; past the end when
    /// none was ranked.
    rank: usize,
    /// The selected turns, as `timeline/index`.
    turns: BTreeSet<String>,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn project(agent: &ureq::Agent, url: &str, node: &str, text: &str) -> anyhow::Result<ProjectView> {
    let mut req = agent
        .post(&format!("{url}/v1/substrate/zend/project"))
        .set("x-tokera-node", node)
        .set("content-type", "application/json");
    for (k, v) in ADMIN_HEADERS {
        req = req.set(k, v);
    }
    let body = serde_json::json!({ "text": text }).to_string();
    let answer = req.send_string(&body).context("asking the daemon")?;
    serde_json::from_str(&answer.into_string()?).context("reading the projection")
}

/// `view` graded against `ask`.
fn grade(ask: &Ask, view: &ProjectView) -> Graded {
    let tools: Vec<String> = view
        .tiles
        .iter()
        .filter(|t| t.kind == "section")
        .map(|t| t.label.clone())
        .collect();
    let rank = tools
        .iter()
        .position(|t| ask.tools.contains(t))
        .map_or(tools.len() + 1, |i| i + 1);
    let turns = view
        .tiles
        .iter()
        .filter(|t| t.kind != "section" && t.selected)
        .filter_map(|t| Some(format!("{}/{}", t.timeline.as_ref()?, t.index?)))
        .collect();
    Graded {
        ask: ask.ask.clone(),
        tools,
        rank,
        turns,
    }
}

/// `hit@1`, `hit@3` and MRR over `ranks`.
fn summary(ranks: &[usize]) -> (f32, f32, f32) {
    let n = ranks.len().max(1) as f32;
    let hit = |k: usize| ranks.iter().filter(|&&r| r <= k).count() as f32 / n;
    (
        hit(1),
        hit(3),
        ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / n,
    )
}

/// The share of `a ∪ b` both hold; 1 when both are empty.
fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(b).count() as f32 / union as f32
}

fn run_path(label: &str) -> PathBuf {
    Path::new(RUNS).join(format!("{label}.json"))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let url = arg(&args, "--daemon").context("--daemon <url>")?;
    let node = arg(&args, "--node").context("--node <hex>")?;
    let label = arg(&args, "--label").context("--label <name>")?;
    let against = arg(&args, "--against");
    let asks: Vec<Ask> =
        serde_yaml::from_str(&fs::read_to_string(ASKS).context("reading the asks")?)?;
    let agent = ureq::AgentBuilder::new().build();
    let mut graded = Vec::with_capacity(asks.len());
    for ask in &asks {
        let view = project(&agent, url.trim_end_matches('/'), &node, &ask.ask)
            .with_context(|| format!("projecting {:?}", ask.ask))?;
        graded.push(grade(ask, &view));
    }
    fs::create_dir_all(RUNS)?;
    fs::write(run_path(&label), serde_json::to_string_pretty(&graded)?)?;

    let before: Option<HashMap<String, Graded>> = against
        .as_ref()
        .map(|a| -> anyhow::Result<_> {
            let kept: Vec<Graded> = serde_json::from_str(
                &fs::read_to_string(run_path(a)).with_context(|| format!("reading run {a}"))?,
            )?;
            Ok(kept.into_iter().map(|g| (g.ask.clone(), g)).collect())
        })
        .transpose()?;
    println!("{:<62} {:>5} {:>6} {:>6}", "ask", "rank", "was", "turns");
    for g in &graded {
        let prior = before.as_ref().and_then(|b| b.get(&g.ask));
        println!(
            "{:<62} {:>5} {:>6} {:>6}",
            g.ask.chars().take(62).collect::<String>(),
            g.rank,
            prior.map_or("-".to_string(), |p| p.rank.to_string()),
            prior.map_or("-".to_string(), |p| format!(
                "{:.2}",
                jaccard(&g.turns, &p.turns)
            )),
        );
    }
    let ranks: Vec<usize> = graded.iter().map(|g| g.rank).collect();
    let (h1, h3, m) = summary(&ranks);
    println!(
        "\n{label}: tool hit@1 {h1:.3}  hit@3 {h3:.3}  MRR {m:.3}  ({} asks)",
        ranks.len()
    );
    if let Some(b) = &before {
        let was: Vec<usize> = graded
            .iter()
            .filter_map(|g| b.get(&g.ask).map(|p| p.rank))
            .collect();
        let (b1, b3, bm) = summary(&was);
        let same_turns = graded
            .iter()
            .filter_map(|g| b.get(&g.ask).map(|p| jaccard(&g.turns, &p.turns)))
            .sum::<f32>()
            / graded.len().max(1) as f32;
        println!(
            "{}: tool hit@1 {b1:.3}  hit@3 {b3:.3}  MRR {bm:.3}; selected turns shared {same_turns:.3}",
            against.as_deref().unwrap_or_default()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(kind: &str, label: &str, selected: bool, at: Option<(&str, u32)>) -> Tile {
        Tile {
            kind: kind.into(),
            selected,
            label: label.into(),
            timeline: at.map(|(t, _)| t.into()),
            index: at.map(|(_, i)| i),
        }
    }

    /// **An ask's rank is its best expected tool's place in belief order; its
    /// turns are the selected ones.**
    #[test]
    fn an_ask_is_graded_by_its_best_expected_tool() {
        let view = ProjectView {
            tiles: vec![
                tile("section", "git_log", true, None),
                tile("section", "git_grep", true, None),
                tile("section", "file_grep", false, None),
                tile("normal", "a", true, Some(("7", 1))),
                tile("normal", "b", false, Some(("7", 2))),
            ],
        };
        let ask = Ask {
            ask: "find it".into(),
            tools: vec!["file_grep".into(), "git_grep".into()],
        };
        let g = grade(&ask, &view);
        assert_eq!(g.rank, 2);
        assert_eq!(g.turns, BTreeSet::from(["7/1".to_string()]));
        let none = Ask {
            ask: "x".into(),
            tools: vec!["calculator".into()],
        };
        assert_eq!(grade(&none, &view).rank, 4, "past the end of 3 tools");
    }

    #[test]
    fn hits_mrr_and_shared_turns() {
        let (h1, h3, m) = summary(&[1, 2, 4, 1]);
        assert_eq!((h1, h3), (0.5, 0.75));
        assert!((m - (1.0 + 0.5 + 0.25 + 1.0) / 4.0).abs() < 1e-6);
        let a: BTreeSet<String> = ["1".into(), "2".into()].into();
        let b: BTreeSet<String> = ["2".into(), "3".into()].into();
        assert!((jaccard(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(jaccard(&BTreeSet::new(), &BTreeSet::new()), 1.0);
    }
}
