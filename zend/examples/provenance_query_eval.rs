//! Model-gated retrieval eval for the `code_read` layer — HTTP driver.
//!
//! Drives a **running** zend daemon over its HTTP API (`POST
//! /v1/substrate/project`), so it reuses the daemon's already-loaded model and
//! adds **no extra VRAM** — the safe way to measure retrieval quality while a
//! corpus is resident. `substrate_project` captures each query's live decode-Q,
//! scores it against the substrate, and returns the selected projection tiles,
//! writing NOTHING to the substrate.
//!
//! For each authored query in the fixture it checks whether the `code_reading` /
//! `scopes` selection lands on the query's expected file, and reports hit@budget
//! / hit@1 / MRR / per-style / distractor-confusion — to stdout and, optionally,
//! as a self-contained HTML file.
//!
//! Unlike `substrate_inspect belief-eval` (self-match — a scope retrieving its
//! own file), this measures whether a real natural-language / symbol / task
//! query hits the right file.
//!
//! ```text
//! cargo run -p zend --example provenance_query_eval --release --features cuda -- \
//!     --url http://127.0.0.1:80 --queries zend/examples/provenance_queries.json \
//!     --limit 40 --html target/provenance_eval.html --wait-ready
//! ```
//! The daemon serves the projection API only once fully loaded; `--wait-ready`
//! polls `/v1/status` until it leaves the `loading` state before running.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct QuerySet {
    queries: Vec<QueryCase>,
}

#[derive(Deserialize, Clone)]
struct QueryCase {
    id: String,
    query: String,
    expected_path: String,
    style: String,
    #[serde(default)]
    distractor_group: Option<String>,
}

/// One tile of the daemon's `ProjectView` response (subset we need).
#[derive(Deserialize)]
struct Tile {
    #[serde(default)]
    layer: String,
    #[serde(default)]
    group: String,
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
    #[serde(default)]
    detail: String,
    #[serde(default)]
    loading: Option<LoadingResp>,
}

#[derive(Deserialize)]
struct LoadingResp {
    #[serde(default)]
    current: String,
    #[serde(default)]
    progress: f32,
}

const CODE_LAYER: &str = "code_reading";
const CODE_GROUP: &str = "scopes";

struct SelectedScope {
    path: String,
    score: f32,
}

struct Outcome {
    case: QueryCase,
    selected: Vec<SelectedScope>,
    /// 1-based rank of the expected file among the selected scopes; `None` = not selected.
    hit_rank: Option<usize>,
    hit_score: Option<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── args ──────────────────────────────────────────────────────────────
    let mut url = "http://127.0.0.1:80".to_string();
    let mut queries_path = PathBuf::from("zend/examples/provenance_queries.json");
    let mut html_path: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut verbose = false;
    let mut wait_ready = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--url" => url = args.next().context("--url needs a value")?,
            "--queries" => {
                queries_path = PathBuf::from(args.next().context("--queries needs a path")?)
            }
            "--html" => html_path = Some(PathBuf::from(args.next().context("--html needs a path")?)),
            "--limit" => limit = Some(args.next().context("--limit needs N")?.parse()?),
            "--verbose" => verbose = true,
            "--wait-ready" => wait_ready = true,
            other => anyhow::bail!("unknown arg {other:?}"),
        }
    }
    let url = url.trim_end_matches('/').to_string();

    // ── fixture ───────────────────────────────────────────────────────────
    let qs: QuerySet = serde_json::from_str(
        &std::fs::read_to_string(&queries_path)
            .with_context(|| format!("reading {}", queries_path.display()))?,
    )
    .context("parsing query fixture")?;
    let all = qs.queries;
    // Even-stride sample so a cap still spans every subsystem/family in the
    // (subsystem-grouped) fixture instead of truncating to the first N.
    let cases = even_sample(&all, limit.unwrap_or(all.len()));
    let mut group_of: HashMap<String, String> = HashMap::new();
    for c in &all {
        if let Some(g) = &c.distractor_group {
            group_of.insert(c.expected_path.clone(), g.clone());
        }
    }
    eprintln!(
        "{} of {} queries sampled from {}",
        cases.len(),
        all.len(),
        queries_path.display()
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    // ── wait for the daemon to leave `loading` ────────────────────────────
    if wait_ready {
        eprintln!("waiting for daemon at {url} to finish loading…");
        loop {
            match client.get(format!("{url}/v1/status")).send().await {
                Ok(r) => {
                    if let Ok(s) = r.json::<StatusResp>().await {
                        if s.state != "loading" {
                            eprintln!("daemon state = {:?} — proceeding", s.state);
                            break;
                        }
                        let (cur, prog) = s
                            .loading
                            .map(|l| (l.current, l.progress))
                            .unwrap_or_default();
                        eprintln!(
                            "  loading: {cur} {:.0}% ({})",
                            prog * 100.0,
                            if s.detail.is_empty() { "" } else { &s.detail }
                        );
                    }
                }
                Err(e) => eprintln!("  status poll failed ({e}); retrying"),
            }
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    }

    // ── run queries ───────────────────────────────────────────────────────
    let mut outcomes: Vec<Outcome> = Vec::with_capacity(cases.len());
    let mut unavailable = 0usize;
    for (i, case) in cases.iter().enumerate() {
        let resp = client
            .post(format!("{url}/v1/substrate/project"))
            .json(&serde_json::json!({ "text": case.query }))
            .send()
            .await
            .with_context(|| format!("POST project for {}", case.id))?;
        // The daemon answers 503 while it is still loading (the projection API
        // only serves once fully loaded). Record unavailable queries and carry on
        // so the run still produces a report rather than aborting on the first one.
        if !resp.status().is_success() {
            unavailable += 1;
            if verbose {
                eprintln!(
                    "  [{:>3}/{}] --   {:<18} HTTP {} (daemon not serving projections yet)",
                    i + 1,
                    cases.len(),
                    case.id,
                    resp.status().as_u16()
                );
            }
            outcomes.push(Outcome { case: case.clone(), selected: Vec::new(), hit_rank: None, hit_score: None });
            continue;
        }
        let view: ProjView = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                unavailable += 1;
                eprintln!("  {} : could not decode ProjectView ({e})", case.id);
                outcomes.push(Outcome { case: case.clone(), selected: Vec::new(), hit_rank: None, hit_score: None });
                continue;
            }
        };

        let mut selected: Vec<SelectedScope> = view
            .tiles
            .iter()
            .filter(|t| t.layer == CODE_LAYER && t.group == CODE_GROUP && t.selected)
            .map(|t| SelectedScope {
                path: t.label.clone(),
                score: t.score,
            })
            .collect();
        selected.sort_by(|a, b| b.score.total_cmp(&a.score));

        let hit_rank = selected
            .iter()
            .position(|s| path_matches(&s.path, &case.expected_path))
            .map(|p| p + 1);
        let hit_score = hit_rank.map(|r| selected[r - 1].score);

        if verbose {
            let mark = match hit_rank {
                Some(1) => "OK1",
                Some(_) => "OK ",
                None => "MISS",
            };
            let picks: Vec<String> = selected
                .iter()
                .map(|s| format!("{}[{:.0}]", short(&s.path), s.score))
                .collect();
            eprintln!(
                "  [{:>3}/{}] {mark:<4} {:<12} {:<18} -> {}",
                i + 1,
                cases.len(),
                case.style,
                case.id,
                picks.join(" ")
            );
        }

        outcomes.push(Outcome {
            case: case.clone(),
            selected,
            hit_rank,
            hit_score,
        });
    }

    print_report(&outcomes, &group_of);
    if unavailable > 0 {
        eprintln!(
            "\n⚠ {unavailable}/{} queries returned HTTP 503 — the daemon is still loading and \
             is not serving projections yet. This run only verified the harness pipeline; \
             re-run once the daemon is ready for real hit-rates.",
            outcomes.len()
        );
    }
    if let Some(p) = &html_path {
        std::fs::write(p, render_html(&outcomes, &group_of, &url, unavailable))
            .with_context(|| format!("writing {}", p.display()))?;
        eprintln!("\nHTML report → {}", p.display());
    }
    Ok(())
}

/// Evenly-spaced sample of `k` items from `all` (all of them if `k >= len`).
fn even_sample(all: &[QueryCase], k: usize) -> Vec<QueryCase> {
    if k >= all.len() || all.is_empty() {
        return all.to_vec();
    }
    (0..k)
        .map(|i| all[i * all.len() / k].clone())
        .collect()
}

/// File-level match. The tile `label` for a code turn resolves to its file path;
/// tolerate a suffix mismatch so a labeling quirk doesn't read as a miss.
fn path_matches(label: &str, expected: &str) -> bool {
    let l = label.trim().replace('\\', "/");
    let e = expected.trim().replace('\\', "/");
    !l.is_empty() && (l == e || l.ends_with(&e) || e.ends_with(&l))
}

fn short(p: &str) -> String {
    let t = p.trim_end_matches('/').replace('\\', "/");
    let parts: Vec<&str> = t.rsplit('/').take(2).collect();
    parts.iter().rev().cloned().collect::<Vec<_>>().join("/")
}

// ── metrics ───────────────────────────────────────────────────────────────

struct Metrics {
    n: usize,
    hit_budget: usize,
    hit1: usize,
    mrr: f64,
    mean_set: f64,
    mean_hit_score: f32,
    empty: usize,
}

fn compute(outcomes: &[Outcome]) -> Metrics {
    let n = outcomes.len().max(1);
    let hit_scores: Vec<f32> = outcomes.iter().filter_map(|o| o.hit_score).collect();
    Metrics {
        n: outcomes.len(),
        hit_budget: outcomes.iter().filter(|o| o.hit_rank.is_some()).count(),
        hit1: outcomes.iter().filter(|o| o.hit_rank == Some(1)).count(),
        mrr: outcomes
            .iter()
            .map(|o| o.hit_rank.map(|r| 1.0 / r as f64).unwrap_or(0.0))
            .sum::<f64>()
            / n as f64,
        mean_set: outcomes.iter().map(|o| o.selected.len()).sum::<usize>() as f64 / n as f64,
        mean_hit_score: if hit_scores.is_empty() {
            0.0
        } else {
            hit_scores.iter().sum::<f32>() / hit_scores.len() as f32
        },
        empty: outcomes.iter().filter(|o| o.selected.is_empty()).count(),
    }
}

fn styles_of(outcomes: &[Outcome]) -> Vec<String> {
    let mut s: Vec<String> = outcomes.iter().map(|o| o.case.style.clone()).collect();
    s.sort();
    s.dedup();
    s
}

/// Per-family miss counts: (misses, misses where a same-family sibling was selected).
fn distractor_confusion(
    outcomes: &[Outcome],
    group_of: &HashMap<String, String>,
) -> Vec<(String, usize, usize)> {
    let mut fam: HashMap<String, (usize, usize)> = HashMap::new();
    for o in outcomes.iter().filter(|o| o.hit_rank.is_none()) {
        let Some(g) = &o.case.distractor_group else {
            continue;
        };
        let e = fam.entry(g.clone()).or_insert((0, 0));
        e.0 += 1;
        let stolen = o.selected.iter().any(|s| {
            let key = s.path.trim().replace('\\', "/");
            group_of.get(&key).map(|x| x == g).unwrap_or(false)
        });
        if stolen {
            e.1 += 1;
        }
    }
    let mut rows: Vec<(String, usize, usize)> =
        fam.into_iter().map(|(g, (m, s))| (g, m, s)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    rows
}

fn print_report(outcomes: &[Outcome], group_of: &HashMap<String, String>) {
    let m = compute(outcomes);
    if m.n == 0 {
        println!("no queries.");
        return;
    }
    let pct = |k: usize| 100.0 * k as f64 / m.n as f64;
    println!("\n═══ code_read provenance query eval ═══\n");
    println!("Overall (n={}):", m.n);
    println!("  hit@budget : {:>5.1}%   ({}/{})", pct(m.hit_budget), m.hit_budget, m.n);
    println!("  hit@1      : {:>5.1}%   ({}/{})", pct(m.hit1), m.hit1, m.n);
    println!("  MRR        : {:>6.3}", m.mrr);
    println!("  mean scopes/query : {:>4.2}", m.mean_set);
    println!("  mean expected score (on hit) : {:>6.1}  (0–1000 band)", m.mean_hit_score);
    if m.empty > 0 {
        println!("  ⚠ {}/{} queries selected ZERO scopes", m.empty, m.n);
    }
    println!("\nBy style:");
    for st in styles_of(outcomes) {
        let sub: Vec<&Outcome> = outcomes.iter().filter(|o| o.case.style == st).collect();
        let sn = sub.len().max(1);
        let sb = sub.iter().filter(|o| o.hit_rank.is_some()).count();
        let s1 = sub.iter().filter(|o| o.hit_rank == Some(1)).count();
        println!(
            "  {st:<12} (n={:>3}): hit@budget {:>5.1}%  | hit@1 {:>5.1}%",
            sub.len(),
            100.0 * sb as f64 / sn as f64,
            100.0 * s1 as f64 / sn as f64
        );
    }
    let conf = distractor_confusion(outcomes, group_of);
    println!("\nDistractor confusion (missed family queries with a sibling selected instead):");
    if conf.is_empty() {
        println!("  (none)");
    }
    for (g, miss, stolen) in &conf {
        println!("  {g:<24} {miss} miss, {stolen} sibling-stolen");
    }
    println!("\nMisses:");
    let misses: Vec<&Outcome> = outcomes.iter().filter(|o| o.hit_rank.is_none()).collect();
    if misses.is_empty() {
        println!("  (none)");
    }
    for o in &misses {
        let picks: Vec<String> =
            o.selected.iter().map(|s| format!("{}[{:.0}]", short(&s.path), s.score)).collect();
        println!(
            "  {:<12} {:<18} expected {:<32} | {}",
            o.case.style,
            o.case.id,
            short(&o.case.expected_path),
            if picks.is_empty() { "∅".into() } else { picks.join(" ") }
        );
    }
}

// ── HTML report ─────────────────────────────────────────────────────────────

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn render_html(
    outcomes: &[Outcome],
    group_of: &HashMap<String, String>,
    url: &str,
    unavailable: usize,
) -> String {
    let m = compute(outcomes);
    let pct = |k: usize| if m.n == 0 { 0.0 } else { 100.0 * k as f64 / m.n as f64 };
    let mut h = String::new();
    h.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    h.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
    h.push_str("<title>code_read provenance query eval</title><style>");
    h.push_str(CSS);
    h.push_str("</style></head><body><main>");
    h.push_str("<h1>code_read provenance query eval</h1>");
    h.push_str(&format!(
        "<p class=\"sub\">Authored query → does the projection select the expected file? \
         Driven against <code>{}</code>. hit@budget = expected file is in the selected scopes; \
         hit@1 = it is the top-scored scope.</p>",
        esc(url)
    ));

    // summary cards
    h.push_str("<section class=\"cards\">");
    let card = |label: &str, val: String, sub: &str| {
        format!(
            "<div class=\"card\"><div class=\"val\">{val}</div><div class=\"lbl\">{label}</div><div class=\"cs\">{sub}</div></div>"
        )
    };
    h.push_str(&card("hit@budget", format!("{:.0}%", pct(m.hit_budget)), &format!("{}/{}", m.hit_budget, m.n)));
    h.push_str(&card("hit@1", format!("{:.0}%", pct(m.hit1)), &format!("{}/{}", m.hit1, m.n)));
    h.push_str(&card("MRR", format!("{:.3}", m.mrr), "over selected rank"));
    h.push_str(&card("mean scopes", format!("{:.2}", m.mean_set), "selected / query"));
    h.push_str(&card("mean hit score", format!("{:.0}", m.mean_hit_score), "0–1000 band"));
    h.push_str("</section>");
    if unavailable > 0 {
        h.push_str(&format!(
            "<p class=\"warn\">⚠ {unavailable}/{} queries returned HTTP 503 — the daemon was still \
             loading and not serving projections. This report only verified the pipeline; the \
             numbers above are not real hit-rates. Re-run once the daemon is ready.</p>",
            m.n
        ));
    } else if m.empty > 0 {
        h.push_str(&format!(
            "<p class=\"warn\">⚠ {}/{} queries selected zero scopes.</p>",
            m.empty, m.n
        ));
    }

    // by style
    h.push_str("<h2>By query style</h2><table><thead><tr><th>style</th><th>n</th><th>hit@budget</th><th>hit@1</th></tr></thead><tbody>");
    for st in styles_of(outcomes) {
        let sub: Vec<&Outcome> = outcomes.iter().filter(|o| o.case.style == st).collect();
        let sn = sub.len().max(1);
        let sb = sub.iter().filter(|o| o.hit_rank.is_some()).count();
        let s1 = sub.iter().filter(|o| o.hit_rank == Some(1)).count();
        h.push_str(&format!(
            "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{:.0}%</td><td class=\"num\">{:.0}%</td></tr>",
            esc(&st), sub.len(), 100.0 * sb as f64 / sn as f64, 100.0 * s1 as f64 / sn as f64
        ));
    }
    h.push_str("</tbody></table>");

    // distractor confusion
    let conf = distractor_confusion(outcomes, group_of);
    if !conf.is_empty() {
        h.push_str("<h2>Distractor confusion</h2><table><thead><tr><th>near-dup family</th><th>misses</th><th>sibling selected instead</th></tr></thead><tbody>");
        for (g, miss, stolen) in &conf {
            h.push_str(&format!(
                "<tr><td>{}</td><td class=\"num\">{miss}</td><td class=\"num\">{stolen}</td></tr>",
                esc(g)
            ));
        }
        h.push_str("</tbody></table>");
    }

    // per-query detail
    h.push_str("<h2>Every query</h2><table class=\"detail\"><thead><tr><th>#</th><th>style</th><th>query</th><th>expected file</th><th>result</th><th>selected scopes (score)</th></tr></thead><tbody>");
    for (i, o) in outcomes.iter().enumerate() {
        let (cls, res) = match o.hit_rank {
            Some(1) => ("ok", "✓ #1".to_string()),
            Some(r) => ("ok", format!("✓ #{r}")),
            None => ("miss", "✗ miss".to_string()),
        };
        let picks: String = o
            .selected
            .iter()
            .map(|s| {
                let hit = path_matches(&s.path, &o.case.expected_path);
                format!(
                    "<span class=\"pick{}\">{} <b>{:.0}</b></span>",
                    if hit { " phit" } else { "" },
                    esc(&short(&s.path)),
                    s.score
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        h.push_str(&format!(
            "<tr class=\"{cls}\"><td class=\"num\">{}</td><td>{}</td><td class=\"q\">{}</td><td class=\"path\">{}</td><td class=\"res\">{res}</td><td>{}</td></tr>",
            i + 1,
            esc(&o.case.style),
            esc(&o.case.query),
            esc(&short(&o.case.expected_path)),
            if picks.is_empty() { "<span class=\"pick\">∅</span>".into() } else { picks }
        ));
    }
    h.push_str("</tbody></table></main></body></html>");
    h
}

const CSS: &str = r#"
:root{--bg:#f7f7f5;--fg:#1a1a1a;--mut:#6b6b6b;--card:#fff;--line:#e3e3df;--ok:#1f7a3d;--okbg:#eaf6ee;--miss:#a11;--missbg:#fbecec;--acc:#2b5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e8e6;--mut:#9a9a97;--card:#1f2126;--line:#2e3138;--ok:#5cd68a;--okbg:#17301f;--miss:#f08a8a;--missbg:#301717;--acc:#79a6f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mut);max-width:70ch;margin:0 0 24px}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:.9em}
.cards{display:flex;gap:14px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;min-width:130px;flex:1}
.card .val{font-size:28px;font-weight:650;letter-spacing:-.02em}.card .lbl{color:var(--fg);font-weight:600;margin-top:2px}.card .cs{color:var(--mut);font-size:12px}
.warn{background:var(--missbg);color:var(--miss);padding:10px 14px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.num{text-align:right;font-variant-numeric:tabular-nums}
tr.ok .res{color:var(--ok);font-weight:600;white-space:nowrap}tr.miss .res{color:var(--miss);font-weight:600;white-space:nowrap}
tr.miss{background:var(--missbg)}
td.q{max-width:340px;color:var(--fg)}td.path{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:var(--mut);white-space:nowrap}
.pick{display:inline-block;background:var(--line);border-radius:5px;padding:1px 6px;margin:1px 2px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;white-space:nowrap}
.pick.phit{background:var(--okbg);color:var(--ok);font-weight:600}
"#;
