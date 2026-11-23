//! Running the battery against a live zend (`docs/unbounded_agents.md` §9.12,
//! *Running*): the queue of `schedule.rs`, sixteen answers at a time, each a
//! fresh conversation asked once, scored, read back for its placements and
//! deleted; one JSON line per answer, written as it lands.
//!
//! A run stopped part-way is taken up again from its results file: an answer
//! already recorded is not asked twice. An answer whose turn fails is retried
//! on a fresh conversation, and recorded as failed once its tries are spent.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::arms::Arm;
use crate::battery::Question;
use crate::client::{conv_id, Zend};
use crate::placement::{placement, Placement};
use crate::schedule::{queue, Job};
use crate::score::{answer, grounding, recall};
use crate::telemetry;

/// What every question is prefixed with: the pilot's instruction.
pub const PREFIX: &str = "Answer from your memory of the repository only — do not call any \
     tool, do not read or search any file. If you do not remember, say so.";

/// How many times an answer is tried before it is recorded as failed.
pub const TRIES: u32 = 3;

/// One answer's line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub arm: String,
    pub run: u32,
    pub id: String,
    pub category: String,
    pub names_file: bool,
    pub recall: f64,
    pub missed: Vec<String>,
    pub grounding: Option<f64>,
    pub ungrounded: Vec<String>,
    pub completion_tokens: u64,
    pub think: Option<u32>,
    pub placement: Placement,
    pub secs: f64,
    pub answer: String,
    /// Why the answer failed, when every try did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What a run is given.
pub struct Plan<'a> {
    pub base: &'a str,
    /// The member that answers every question, by node id.
    pub node: &'a str,
    pub thread: &'a str,
    pub questions: Vec<Question>,
    pub vocabulary: HashSet<String>,
    pub runs: u32,
    pub seed: u64,
    pub concurrency: usize,
    pub out: &'a Path,
    pub telemetry: &'a Path,
}

/// The `(arm, run, id)` of every answer `out` already records.
pub fn recorded(out: &Path) -> anyhow::Result<HashSet<(String, u32, String)>> {
    let Ok(f) = File::open(out) else {
        return Ok(HashSet::new());
    };
    let mut done = HashSet::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let l: Line = serde_json::from_str(&line).context("a recorded answer")?;
        done.insert((l.arm, l.run, l.id));
    }
    Ok(done)
}

/// The line for `q` answered under `arm` in `run` with `text`.
#[allow(clippy::too_many_arguments)]
pub fn scored(
    q: &Question,
    arm: Arm,
    run: u32,
    text: &str,
    vocabulary: &HashSet<String>,
    completion_tokens: u64,
    think: Option<u32>,
    placement: Placement,
    secs: f64,
) -> Line {
    let said = answer(text);
    let (r, missed) = recall(said, &q.expect);
    let g = grounding(said, vocabulary);
    Line {
        arm: arm.id().to_string(),
        run,
        id: q.id.clone(),
        category: q.category.clone(),
        names_file: q.names_file,
        recall: r,
        missed,
        grounding: g.as_ref().map(|(g, _)| *g),
        ungrounded: g.map(|(_, u)| u).unwrap_or_default(),
        completion_tokens,
        think,
        placement,
        secs,
        answer: said.to_string(),
        error: None,
    }
}

/// Ask one job to its line: tried on a fresh conversation each time — cleared
/// of anything an interrupted run left on it, and deleted once read.
async fn answer_job(
    zend: &Zend,
    plan_thread: &str,
    q: &Question,
    job: Job,
    vocabulary: &HashSet<String>,
) -> Line {
    let prompt = format!("{PREFIX}\n\n{}", q.question);
    let mut last_error = String::new();
    for attempt in 1..=TRIES {
        let id = format!("{}-a{attempt}", q.id);
        let conv = conv_id(plan_thread, job.arm, job.run, &id);
        // A run stopped mid-answer leaves that answer's conversation behind;
        // asked again on it, the question would follow its own first try.
        if let Err(e) = zend.delete(&conv).await {
            tracing::warn!(
                conv,
                "a conversation left by an earlier run was not cleared: {e:#}"
            );
        }
        let asked = zend.ask(&conv, &prompt, job.arm).await;
        if let Err(e) = zend.delete(&conv).await {
            tracing::warn!(conv, "the conversation was not deleted: {e:#}");
        }
        match asked {
            Ok(a) => {
                let p = placement(&a.points, &q.files, a.think);
                return scored(
                    q,
                    job.arm,
                    job.run,
                    &a.text,
                    vocabulary,
                    a.completion_tokens,
                    a.think,
                    p,
                    a.secs,
                );
            }
            Err(e) => {
                last_error = format!("{e:#}");
                tracing::warn!(conv, attempt, "the answer failed: {last_error}");
            }
        }
    }
    let mut failed = scored(
        q,
        job.arm,
        job.run,
        "",
        vocabulary,
        0,
        None,
        Placement::default(),
        0.0,
    );
    failed.error = Some(last_error);
    failed
}

/// Run `plan` to its end.
pub async fn run(plan: Plan<'_>) -> anyhow::Result<()> {
    let zend = Arc::new(Zend::new(plan.base, plan.node)?);
    let done = recorded(plan.out)?;
    let jobs: Vec<Job> = queue(plan.questions.len(), plan.runs, plan.seed)
        .into_iter()
        .filter(|j| {
            let q = &plan.questions[j.question];
            !done.contains(&(j.arm.id().to_string(), j.run, q.id.clone()))
        })
        .collect();
    println!(
        "{} answers to ask ({} already recorded), {} at a time",
        jobs.len(),
        done.len(),
        plan.concurrency
    );
    let out = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(plan.out)
            .with_context(|| format!("opening {}", plan.out.display()))?,
    ));
    let samples = OpenOptions::new()
        .create(true)
        .append(true)
        .open(plan.telemetry)
        .with_context(|| format!("opening {}", plan.telemetry.display()))?;
    let in_flight = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = tokio::spawn(telemetry::sample(
        Arc::clone(&zend),
        Arc::clone(&in_flight),
        Arc::clone(&stop),
        samples,
    ));

    let gate = Arc::new(Semaphore::new(plan.concurrency));
    let questions = Arc::new(plan.questions);
    let vocabulary = Arc::new(plan.vocabulary);
    let thread = plan.thread.to_string();
    let total = jobs.len();
    let finished = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::with_capacity(total);
    for job in jobs {
        let permit = Arc::clone(&gate).acquire_owned().await?;
        let (zend, questions, vocabulary, out, in_flight, finished, thread) = (
            Arc::clone(&zend),
            Arc::clone(&questions),
            Arc::clone(&vocabulary),
            Arc::clone(&out),
            Arc::clone(&in_flight),
            Arc::clone(&finished),
            thread.clone(),
        );
        tasks.push(tokio::spawn(async move {
            in_flight.fetch_add(1, Ordering::AcqRel);
            let q = &questions[job.question];
            let line = answer_job(&zend, &thread, q, job, &vocabulary).await;
            in_flight.fetch_sub(1, Ordering::AcqRel);
            let n = finished.fetch_add(1, Ordering::AcqRel) + 1;
            println!(
                "[{n}/{total}] {} r{} {:<4} recall {:>5.1}%  opening {} think-end {} answer {}  {} tokens{}",
                line.arm,
                line.run,
                line.id,
                line.recall * 100.0,
                line.placement.opening,
                line.placement.think_end,
                line.placement.first_answer,
                line.completion_tokens,
                line.error.as_deref().map(|e| format!("  FAILED: {e}")).unwrap_or_default()
            );
            let text = serde_json::to_string(&line).expect("a line serializes");
            let mut f = out.lock().expect("the results file");
            writeln!(f, "{text}").and_then(|_| f.flush()).expect("writing a result");
            drop(permit);
        }));
    }
    for t in tasks {
        t.await?;
    }
    stop.store(true, Ordering::Release);
    sampler.await?;
    Ok(())
}

/// Every recorded line of `path`.
pub fn lines(path: &Path) -> anyhow::Result<Vec<Line>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(&line).context("a recorded answer")?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question() -> Question {
        Question {
            id: "d1".into(),
            category: "direct".into(),
            question: "Where is it?".into(),
            expect: vec!["scan.rs".into(), "gpu.rs".into()],
            files: vec!["a/scan.rs".into()],
            names_file: false,
        }
    }

    /// **An answer is scored after its reasoning**: what the think block
    /// names counts for nothing.
    #[test]
    fn an_answer_is_scored_on_what_follows_its_reasoning() {
        let vocab: HashSet<String> = ["scan", "rs"].iter().map(|s| s.to_string()).collect();
        let l = scored(
            &question(),
            Arm::C1,
            2,
            "<think>gpu.rs maybe</think>It is in `scan.rs`.",
            &vocab,
            40,
            Some(12),
            Placement::default(),
            3.5,
        );
        assert_eq!((l.arm.as_str(), l.run, l.recall), ("c1", 2, 0.5));
        assert_eq!(l.missed, vec!["gpu.rs".to_string()]);
        assert_eq!(l.grounding, Some(1.0));
        assert_eq!(l.answer, "It is in `scan.rs`.");
    }

    #[test]
    fn a_line_round_trips_and_a_missing_file_records_nothing() {
        let l = scored(
            &question(),
            Arm::A0,
            1,
            "x",
            &HashSet::new(),
            1,
            None,
            Placement::default(),
            1.0,
        );
        let back: Line = serde_json::from_str(&serde_json::to_string(&l).unwrap()).unwrap();
        assert_eq!(
            (back.arm, back.run, back.id),
            ("a0".to_string(), 1, "d1".to_string())
        );
        assert!(recorded(Path::new("no/such/results.jsonl"))
            .unwrap()
            .is_empty());
    }
}
