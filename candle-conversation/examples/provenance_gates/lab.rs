//! The scoring lab: the dependency battery's questions ranked against the
//! `code_reading` files under a formula read from a file, and re-ranked every
//! time that file is saved — so the formula is iterated without restarting
//! the daemon or rebuilding this harness.
//!
//! Each question is asked once of a running daemon: `/project` for its probe
//! (the question's wide-Q, as a chat's opening captures it) and production's
//! own pick ([`crate::lab_probes`]), and `/scan` for the raw per-token,
//! per-fold-group votes of the content and the calibration competitions on
//! the daemon's GPU gallery ([`crate::remote_gallery`]). A fixed sample of
//! other files' calibration questions is scanned the same way, as the
//! background a file is read against when it scores on a question that is
//! not about it. All of it is cached on disk. Everything after that — fusing,
//! gating, correcting, ranking — is the formula's ([`crate::lab_formula`]),
//! on the host, in milliseconds.
//!
//! The formula file holds one formula, or `variants:` — named formulas each
//! scored to one summary line, for a sweep.
//!
//! Judged against each question's `files` in the battery: where the files
//! that answer it rank, whether one is in the top `budget_files`, and what
//! production placed for the same probe.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Context;
use candle_conversation::persistence::content_hash::ContentHasher;
use candle_conversation::projection::TimelineId;
use candle_conversation::provenance::window_score::{group_ranked, FlatSigs, Ranked};
use candle_conversation::provenance::{encode_wide_sigs, WideQSig, FOLD_GROUPS};
use candle_conversation::substrate::Substrate;
use serde::Deserialize;

use crate::corpus::{File, Gallery};
use crate::lab_formula::{Formula, FormulaSpec, Sample, TokenVote, Votes};
use crate::lab_probes::{probed, Probed};
use crate::ranking::order;
use crate::ranking::token_files;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};

/// How often the formula file is looked at in `--watch`.
const POLL: Duration = Duration::from_millis(400);

/// Background probes: one calibration question from every this-many-th
/// calibrated file.
const BACKGROUND_STRIDE: usize = 60;

/// Files a CPU ranking keeps per token and group: enough for `topk` up to
/// one less, each vote measured over the next.
const RANK_KEPT: usize = 9;

/// Where the CPU rankings are kept, from the repository root: each file with
/// the gallery token that agreed, which token runs read.
const RANK_CACHE_DIR: &str = "target/provenance-topk-cache-at";

/// The longest one question's CPU ranking may take before the lab stops
/// ranking for the rest of the run: an experiment, not a wait.
const RANK_BUDGET: Duration = Duration::from_secs(60);

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

/// A formula file: one formula, or named variants.
#[derive(Deserialize)]
#[serde(untagged)]
enum FormulaFile {
    Variants {
        variants: BTreeMap<String, FormulaSpec>,
    },
    One(FormulaSpec),
}

/// One probe's votes in both competitions, as the daemon scanned them, and
/// its content competition ranked on the CPU when the lab was asked to.
struct Scanned {
    content: Vec<TokenVote>,
    calibration: Vec<TokenVote>,
    ranked: Option<Vec<[Ranked; FOLD_GROUPS]>>,
}

impl Scanned {
    fn votes(&self) -> Votes<'_> {
        Votes {
            content: &self.content,
            calibration: &self.calibration,
            ranked: self.ranked.as_deref(),
        }
    }
}

/// One question, probed and scanned: what every formula scores.
struct Asked {
    id: String,
    category: String,
    /// The files that answer it, by index; empty when no file does.
    answers: Vec<usize>,
    /// Answer paths no loaded file holds.
    missing: Vec<String>,
    scanned: Scanned,
    /// Production's pick, by file index, best first.
    placed: Vec<usize>,
}

/// Everything a formula is scored against.
struct Bench<'a> {
    asked: Vec<Asked>,
    /// Other files' calibration questions, each with the file it is about.
    background: Vec<(usize, Scanned)>,
    paths: Vec<&'a str>,
    tokens: Vec<usize>,
    chain: HashSet<usize>,
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

/// The files of `files` that are links of the priming chain: every ancestor
/// any file inherits. Every conversation opens on them already.
pub fn chain_files(sub: &Substrate, files: &[File]) -> HashSet<usize> {
    let mut ancestors: HashSet<u64> = HashSet::new();
    for f in files {
        if let Some(tl) = TimelineId::from_raw(f.timeline) {
            ancestors.extend(
                sub.inherited_chain(tl)
                    .into_iter()
                    .filter(|a| *a != tl)
                    .map(|a| a.raw()),
            );
        }
    }
    files
        .iter()
        .enumerate()
        .filter(|(_, f)| ancestors.contains(&f.timeline))
        .map(|(i, _)| i)
        .collect()
}

/// `probe`'s votes in both competitions on the daemon's gallery, its own
/// calibration question `own` — `(file, question)` — left out.
fn scan(
    gpu: &RemoteGallery,
    probe: &[WideQSig],
    own: Option<(usize, usize)>,
) -> anyhow::Result<Scanned> {
    let content = gpu
        .votes(&[probe], &|_, _| false, Questions::Without, &|_, _| false)?
        .remove(0);
    let calibration = gpu
        .votes(&[probe], &|_, _| true, Questions::With, &|f, q| {
            Some((f, q)) == own
        })?
        .remove(0);
    Ok(Scanned {
        content,
        calibration,
        ranked: None,
    })
}

/// A ranked token as the cache keeps it: per group, `(mean, std, top)`.
type KeptRanking = Vec<Vec<(f32, f32, Vec<(usize, u32, u32)>)>>;

/// `probe` ranked on the CPU against the content gallery `flat` — each
/// token's best [`RANK_KEPT`] files per group — from the cache when it holds
/// it.
fn ranked(
    probe: &[WideQSig],
    flat: &FlatSigs,
    token_file: &[usize],
    n_files: usize,
) -> anyhow::Result<Vec<[Ranked; FOLD_GROUPS]>> {
    let mut h = ContentHasher::new();
    h.update(&encode_wide_sigs(probe));
    h.update(&(flat.len() as u64).to_le_bytes());
    let key = h.finish();
    let kept = PathBuf::from(RANK_CACHE_DIR).join(format!("{:016x}{:016x}.json", key.hi, key.lo));
    if let Ok(text) = fs::read_to_string(&kept) {
        let rows: KeptRanking = serde_json::from_str(&text).context("reading a kept ranking")?;
        return Ok(rows
            .into_iter()
            .map(|groups| {
                let mut it = groups
                    .into_iter()
                    .map(|(mean, std, top)| Ranked { mean, std, top });
                std::array::from_fn(|_| {
                    it.next().unwrap_or(Ranked {
                        mean: 0.0,
                        std: 1.0,
                        top: Vec::new(),
                    })
                })
            })
            .collect());
    }
    let out = group_ranked(probe, flat, token_file, n_files, RANK_KEPT);
    let rows: KeptRanking = out
        .iter()
        .map(|groups| {
            groups
                .iter()
                .map(|r| (r.mean, r.std, r.top.clone()))
                .collect()
        })
        .collect();
    fs::create_dir_all(RANK_CACHE_DIR).context("making the ranking cache")?;
    fs::write(&kept, serde_json::to_string(&rows)?).context("keeping the ranking")?;
    Ok(out)
}

/// What the lab is run on.
pub struct LabRun<'a> {
    pub files: &'a [File],
    pub gallery: &'a Gallery,
    pub chain: HashSet<usize>,
    pub battery: &'a Path,
    pub formula: &'a Path,
    /// Re-score every time the formula file is saved.
    pub watch: bool,
    /// Rank each question's content competition on the CPU, for `topk`.
    pub topk: bool,
    pub daemon: &'a Daemon,
}

pub fn run(run: LabRun) -> anyhow::Result<()> {
    let LabRun {
        files,
        gallery,
        chain,
        battery,
        formula: formula_path,
        watch,
        topk,
        daemon,
    } = run;
    let token_file = token_files(gallery, None);
    let mut rank_budget_spent = false;
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
        let mut scanned = scan(&gpu, &probe, None)?;
        if topk && !rank_budget_spent {
            let t_rank = Instant::now();
            scanned.ranked = Some(ranked(&probe, &gallery.flat, &token_file, files.len())?);
            let took = t_rank.elapsed();
            if took > RANK_BUDGET {
                rank_budget_spent = true;
                println!(
                    "  {}: the CPU ranking took {:.0}s, past the {}s budget — no more are ranked",
                    q.id,
                    took.as_secs_f64(),
                    RANK_BUDGET.as_secs()
                );
            }
        }
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
            scanned,
            placed: placed
                .iter()
                .filter_map(|tl| by_timeline.get(tl).copied())
                .collect(),
        });
    }
    let calibrated: Vec<usize> = (0..files.len())
        .filter(|&f| !files[f].questions.is_empty())
        .collect();
    let mut background = Vec::new();
    for &f in calibrated.iter().step_by(BACKGROUND_STRIDE) {
        let q = &files[f].questions[0];
        background.push((f, scan(&gpu, q.body_sigs(), Some((f, 0)))?));
    }
    println!(
        "lab: {} questions and {} background probes scanned in {:.1}s, {} chain files — formula {}",
        asked.len(),
        background.len(),
        t.elapsed().as_secs_f64(),
        chain.len(),
        formula_path.display()
    );
    for a in asked.iter().filter(|a| !a.missing.is_empty()) {
        println!("  {}: no loaded file is {:?}", a.id, a.missing);
    }
    let bench = Bench {
        asked,
        background,
        paths: files
            .iter()
            .map(|f| f.path.as_deref().unwrap_or(""))
            .collect(),
        tokens: files
            .iter()
            .map(|f| f.turns.iter().map(|t| t.sigs.len()).sum())
            .collect(),
        chain,
    };

    let mut seen: Option<SystemTime> = None;
    loop {
        let stamp = fs::metadata(formula_path).and_then(|m| m.modified()).ok();
        if seen.is_none() || stamp != seen {
            seen = stamp.or(Some(SystemTime::UNIX_EPOCH));
            let text = fs::read_to_string(formula_path).unwrap_or_default();
            if let Err(e) = score_file(&bench, &text) {
                println!("\nformula refused: {e:#}");
            }
        }
        if !watch {
            return Ok(());
        }
        sleep(POLL);
    }
}

/// Score the formula file `text` holds: one formula in detail, or each named
/// variant to a summary line.
fn score_file(bench: &Bench, text: &str) -> anyhow::Result<()> {
    let file: FormulaFile = if text.trim().is_empty() {
        FormulaFile::One(FormulaSpec::default())
    } else {
        serde_yaml::from_str(text)?
    };
    match file {
        FormulaFile::One(spec) => {
            let formula = Formula::new(spec)?;
            let t = Instant::now();
            let judged = judge(bench, &formula, true);
            println!("\n── formula scored in {} ms ──", t.elapsed().as_millis());
            println!("{}", summary(&judged, formula.spec.budget_files));
        }
        FormulaFile::Variants { variants } => {
            println!("\n── {} variants ──", variants.len());
            for (name, spec) in variants {
                let formula = Formula::new(spec).with_context(|| format!("variant {name}"))?;
                let judged = judge(bench, &formula, false);
                println!("{name:<28} {}", summary(&judged, formula.spec.budget_files));
            }
        }
    }
    Ok(())
}

/// One question's judgement: the best answer file's rank, the answer files in
/// the budget, the answer files, production's answer files placed, and the
/// documents in the budget.
struct Judged {
    best: usize,
    covered: usize,
    answers: usize,
    prod: usize,
    docs: usize,
}

/// Every question scored under `formula` and judged; with `detail`, a line
/// each.
fn judge(bench: &Bench, formula: &Formula, detail: bool) -> Vec<Judged> {
    let raw: Vec<Vec<f32>> = bench
        .asked
        .iter()
        .map(|a| formula.raw(&a.scanned.votes(), &bench.paths, &bench.tokens))
        .collect();
    let background: Vec<Sample> = bench
        .background
        .iter()
        .map(|(f, s)| Sample {
            about: Some(*f),
            scores: formula.raw(&s.votes(), &bench.paths, &bench.tokens),
        })
        .collect();
    let scored = formula.kind_normalized(
        &formula.corrected(&raw, &background),
        &background,
        &bench.paths,
    );
    let budget = formula.spec.budget_files;
    let short = |p: &str| p.rsplit('/').next().unwrap_or(p).to_string();
    if detail {
        println!(
            "{:<4} {:<13} {:>6} {:>7} {:>6} {:>6} {:>9}  top files (answers marked *)",
            "id", "category", "best", "in top", "prod", "scored", "answer"
        );
    }
    let mut judged = Vec::new();
    for (a, s) in bench.asked.iter().zip(&scored) {
        if a.answers.is_empty() {
            continue;
        }
        let ranked: Vec<usize> = order(s)
            .into_iter()
            .filter(|&f| formula.ranks(f, bench.paths[f], &bench.chain))
            .collect();
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
        let docs = ranked
            .iter()
            .take(budget)
            .filter(|&&f| is_document(bench.paths[f]))
            .count();
        if detail {
            let top: Vec<String> = ranked
                .iter()
                .take(5)
                .map(|&f| {
                    let mark = if a.answers.contains(&f) { "*" } else { "" };
                    format!("{mark}{}", short(bench.paths[f]))
                })
                .collect();
            let scoring = ranked.iter().filter(|&&f| s[f] > 0.0).count();
            let best_score = a.answers.iter().map(|&f| s[f]).fold(0.0f32, f32::max);
            println!(
                "{:<4} {:<13} {:>6} {:>4}/{:<2} {:>3}/{:<2} {:>6} {:>9.1}  {}",
                a.id,
                a.category,
                best,
                covered,
                a.answers.len(),
                prod,
                a.answers.len(),
                scoring,
                best_score,
                top.join(", ")
            );
        }
        judged.push(Judged {
            best,
            covered,
            answers: a.answers.len(),
            prod,
            docs,
        });
    }
    judged
}

/// The summary line for `judged` at `budget`.
fn summary(judged: &[Judged], budget: usize) -> String {
    let n = judged.len().max(1) as f32;
    let hit = |k: usize| judged.iter().filter(|j| j.best <= k).count() as f32 / n;
    let mrr = judged.iter().map(|j| 1.0 / j.best as f32).sum::<f32>() / n;
    let coverage = judged
        .iter()
        .map(|j| j.covered as f32 / j.answers as f32)
        .sum::<f32>()
        / n;
    let prod = judged.iter().filter(|j| j.prod > 0).count() as f32 / n;
    let docs = judged.iter().map(|j| j.docs as f32).sum::<f32>() / (n * budget as f32);
    format!(
        "n {}  hit@1 {:.2}  hit@5 {:.2}  hit@{budget} {:.2}  MRR {:.3}  cover {:.2}  docs {:.2}  | prod {:.2}",
        judged.len(),
        hit(1),
        hit(5),
        hit(budget),
        mrr,
        coverage,
        docs,
        prod
    )
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

    /// **A formula file is one formula or named variants.**
    #[test]
    fn a_formula_file_is_one_formula_or_named_variants() {
        let one: FormulaFile = serde_yaml::from_str("keep: 0.5").unwrap();
        assert!(matches!(one, FormulaFile::One(s) if s.keep == 0.5));
        let many: FormulaFile =
            serde_yaml::from_str("variants:\n  a: { keep: 0.5 }\n  b: { hub: divide }\n").unwrap();
        match many {
            FormulaFile::Variants { variants } => {
                assert_eq!(variants.keys().cloned().collect::<Vec<_>>(), ["a", "b"]);
            }
            FormulaFile::One(_) => panic!("variants read as one formula"),
        }
    }
}
