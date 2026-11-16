//! The scoring lab: the dependency battery's questions ranked against the
//! `code_reading` files under a formula read from a file, and re-ranked every
//! time that file is saved — so the formula is iterated without restarting
//! the daemon or rebuilding this harness.
//!
//! Each question is asked once of a running daemon: `/project` for its probe
//! (the question's wide-Q, as a chat's opening captures it) and production's
//! own pick ([`crate::lab_probes`]), and `/scan` for the raw per-token,
//! per-fold-group votes of the content and the calibration competitions on
//! the daemon's GPU gallery ([`crate::remote_gallery`]). Both are cached on
//! disk. Everything after that — gating, weighing, correcting, ranking — is
//! the formula's ([`crate::lab_formula`]), on the host, in milliseconds.
//!
//! Judged against each question's `files` in the battery: where the files
//! that answer it rank, whether one is in the top `budget_files`, and what
//! production placed for the same probe.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Context;
use serde::Deserialize;

use crate::corpus::File;
use crate::lab_formula::Formula;
use crate::lab_probes::{probed, Probed};
use crate::ranking::order;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::sweep::TokenVotes;

/// How often the formula file is looked at in `--watch`.
const POLL: Duration = Duration::from_millis(400);

/// A battery question, as the lab reads it.
#[derive(Deserialize)]
struct BatteryQuestion {
    id: String,
    category: String,
    question: String,
    #[serde(default)]
    files: Vec<String>,
}

#[derive(Deserialize)]
struct Battery {
    questions: Vec<BatteryQuestion>,
}

/// One question, probed and scanned: what every formula scores.
pub struct Asked {
    pub id: String,
    pub category: String,
    /// The files that answer it, by index; empty when no file does.
    pub answers: Vec<usize>,
    /// Answer paths no loaded file holds.
    pub missing: Vec<String>,
    pub content: TokenVotes,
    pub calibration: TokenVotes,
    /// Production's pick, by file index, best first.
    pub placed: Vec<usize>,
}

/// The files of `files` whose path is `answer` within its repository: the
/// path itself, or it after a `<repo>/` prefix.
pub fn answer_files(files: &[File], answer: &str) -> Vec<usize> {
    files
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            f.path.as_deref().is_some_and(|p| {
                p == answer
                    || p.strip_suffix(answer)
                        .is_some_and(|head| head.ends_with('/'))
            })
        })
        .map(|(i, _)| i)
        .collect()
}

pub fn run(
    files: &[File],
    battery: &Path,
    formula_path: &Path,
    watch: bool,
    daemon: &Daemon,
) -> anyhow::Result<()> {
    let battery: Battery = serde_yaml::from_str(
        &fs::read_to_string(battery).with_context(|| format!("reading {}", battery.display()))?,
    )
    .context("parsing the battery")?;
    let gpu = RemoteGallery::new(files, daemon);
    let by_timeline: HashMap<u64, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.timeline, i))
        .collect();
    let t = Instant::now();
    let mut asked: Vec<Asked> = Vec::with_capacity(battery.questions.len());
    for q in &battery.questions {
        let Probed { probe, placed } =
            probed(daemon, &q.question).with_context(|| format!("probing {}", q.id))?;
        let content = gpu
            .votes(
                &[probe.as_slice()],
                &|_, _| false,
                Questions::Without,
                &|_, _| false,
            )?
            .remove(0);
        let calibration = gpu
            .votes(
                &[probe.as_slice()],
                &|_, _| true,
                Questions::With,
                &|_, _| false,
            )?
            .remove(0);
        let mut answers = Vec::new();
        let mut missing = Vec::new();
        for a in &q.files {
            match answer_files(files, a).as_slice() {
                [] => missing.push(a.clone()),
                found => answers.extend_from_slice(found),
            }
        }
        asked.push(Asked {
            id: q.id.clone(),
            category: q.category.clone(),
            answers,
            missing,
            content,
            calibration,
            placed: placed
                .iter()
                .filter_map(|tl| by_timeline.get(tl).copied())
                .collect(),
        });
    }
    println!(
        "lab: {} questions probed and scanned in {:.1}s — formula {}",
        asked.len(),
        t.elapsed().as_secs_f64(),
        formula_path.display()
    );
    for a in asked.iter().filter(|a| !a.missing.is_empty()) {
        println!("  {}: no loaded file is {:?}", a.id, a.missing);
    }

    let paths: Vec<&str> = files
        .iter()
        .map(|f| f.path.as_deref().unwrap_or(""))
        .collect();
    let tokens: Vec<usize> = files
        .iter()
        .map(|f| f.turns.iter().map(|t| t.sigs.len()).sum())
        .collect();
    let mut seen: Option<SystemTime> = None;
    loop {
        let stamp = fs::metadata(formula_path).and_then(|m| m.modified()).ok();
        if seen.is_none() || stamp != seen {
            seen = stamp.or(Some(SystemTime::UNIX_EPOCH));
            let text = fs::read_to_string(formula_path).unwrap_or_default();
            match Formula::parse(&text) {
                Ok(formula) => report(&formula, &asked, &paths, &tokens),
                Err(e) => println!("\nformula refused: {e:#}"),
            }
        }
        if !watch {
            return Ok(());
        }
        sleep(POLL);
    }
}

/// Every question scored under `formula`, ranked and judged.
fn report(formula: &Formula, asked: &[Asked], paths: &[&str], tokens: &[usize]) {
    let t = Instant::now();
    let raw: Vec<Vec<f32>> = asked
        .iter()
        .map(|a| formula.raw(&a.content, &a.calibration, paths, tokens))
        .collect();
    let scored = formula.corrected(&raw);
    let budget = formula.spec.budget_files;
    let short = |p: &str| p.rsplit('/').next().unwrap_or(p).to_string();

    println!("\n── formula scored in {} ms ──", t.elapsed().as_millis());
    println!(
        "{:<4} {:<13} {:>6} {:>7} {:>6}   top files (answers marked *)",
        "id", "category", "best", "in top", "prod"
    );
    let mut judged = Vec::new();
    for (a, s) in asked.iter().zip(&scored) {
        if a.answers.is_empty() {
            continue;
        }
        let ranked = order(s);
        let rank_of = |f: usize| {
            ranked
                .iter()
                .position(|&x| x == f)
                .map_or(usize::MAX, |r| r + 1)
        };
        let best = a
            .answers
            .iter()
            .map(|&f| rank_of(f))
            .min()
            .unwrap_or(usize::MAX);
        let covered = a.answers.iter().filter(|&&f| rank_of(f) <= budget).count();
        let prod = a.answers.iter().filter(|f| a.placed.contains(f)).count();
        let top: Vec<String> = ranked
            .iter()
            .take(5)
            .map(|&f| {
                let mark = if a.answers.contains(&f) { "*" } else { "" };
                format!("{mark}{}", short(paths[f]))
            })
            .collect();
        println!(
            "{:<4} {:<13} {:>6} {:>4}/{:<2} {:>3}/{:<2}  {}",
            a.id,
            a.category,
            best,
            covered,
            a.answers.len(),
            prod,
            a.answers.len(),
            top.join(", ")
        );
        let docs = ranked
            .iter()
            .take(budget)
            .filter(|&&f| is_document(paths[f]))
            .count();
        judged.push((best, covered, a.answers.len(), prod, docs));
    }
    let n = judged.len().max(1) as f32;
    let hit = |k: usize| judged.iter().filter(|j| j.0 <= k).count() as f32 / n;
    let mrr = judged.iter().map(|j| 1.0 / j.0 as f32).sum::<f32>() / n;
    let coverage = judged.iter().map(|j| j.1 as f32 / j.2 as f32).sum::<f32>() / n;
    let prod_hit = judged.iter().filter(|j| j.3 > 0).count() as f32 / n;
    let docs = judged.iter().map(|j| j.4 as f32).sum::<f32>() / (n * budget as f32);
    println!(
        "\n{} questions: hit@1 {:.2}  hit@5 {:.2}  hit@{budget} {:.2}  MRR {:.3}  answer files in top \
         {budget} {:.2}  documents in top {budget} {:.2}  | production placed an answer {:.2}",
        judged.len(),
        hit(1),
        hit(5),
        hit(budget),
        mrr,
        coverage,
        docs,
        prod_hit
    );
}

/// Whether `path` is prose rather than code — what the type shares call docs.
fn is_document(path: &str) -> bool {
    [".md", ".markdown", ".mdx", ".txt", ".rst", ".adoc"]
        .iter()
        .any(|ext| path.ends_with(ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str) -> File {
        File {
            timeline: 0,
            path: Some(path.to_string()),
            turns: Vec::new(),
            questions: Vec::new(),
        }
    }

    /// **An answer path finds its file under its repository's prefix**, and
    /// only a whole path: `mod.rs` alone names no `scheduler/mod.rs`.
    #[test]
    fn an_answer_names_a_file_by_its_path_within_the_repository() {
        let files = [
            file("candle/candle-conversation/src/scheduler/mod.rs"),
            file("candle/candle-conversation/src/projection/mod.rs"),
            file("candle-conversation/src/scheduler/mod.rs"),
        ];
        assert_eq!(
            answer_files(&files, "candle-conversation/src/scheduler/mod.rs"),
            vec![0, 2]
        );
        assert_eq!(answer_files(&files, "scheduler/mod.rs"), vec![0, 2]);
        assert!(answer_files(&files, "duler/mod.rs").is_empty());
        assert!(answer_files(&files, "candle/x.rs").is_empty());
    }

    #[test]
    fn prose_is_a_document_and_code_is_not() {
        assert!(is_document("candle/README.md"));
        assert!(is_document("candle/docs/notes.txt"));
        assert!(!is_document("candle/zend/src/lib.rs"));
        assert!(!is_document("candle/zend/src/prompts/projection.yaml"));
    }
}
