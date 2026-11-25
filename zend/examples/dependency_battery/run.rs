//! Running the battery against a live zend (`docs/unbounded_agents.md` §9.12,
//! *Running*): the queue of `schedule.rs`, sixteen answers at a time, each a
//! fresh conversation asked once, scored, read back for its placements and
//! deleted; one JSON line per answer, written as it lands.
//!
//! A run stopped part-way is taken up again from its results file: an answer
//! already recorded is not asked twice. An answer whose turn fails is retried
//! on a fresh conversation, and recorded as failed once its tries are spent.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use tokio::sync::{OnceCell, Semaphore};

use crate::arms::Arm;
use crate::battery::Question;
use crate::client::{conv_id, passthrough_conv_id, Zend};
use crate::placement::{placement, Placement};
use crate::probe::{budget_chars, messages, paths as probe_paths};
use crate::prompt::prompt;
use crate::rag::context::{Packed, RetrievedFile};
use crate::rag::{mark, Retriever};
use crate::schedule::{queue, Job, Recorded};
use crate::score::{answer, grounding, recall};
use crate::selected::selected;
use crate::telemetry;
use crate::view::{of_packed, of_point, with_named, View};

/// How many times an answer is tried before it is recorded as failed.
pub const TRIES: u32 = 3;

/// One question's selection, shared by the probe arms: C1's selected paths,
/// or why the selection failed, set by the first arm to need it.
type SharedSelection = OnceCell<Result<Vec<String>, String>>;

/// The chunk heads the K/V census samples.
pub const CENSUS_SAMPLE: usize = 4_096;

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
    /// R1's files, as its prompt carried them, best first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieved: Vec<RetrievedFile>,
    /// On R1's lines: whether an answer file was among `retrieved`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_hit: Option<bool>,
    /// The tool calls the daemon ran during the turn, as its history gives
    /// them; absent from the lines recorded before they were read back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<u32>,
    /// Which expected items were in what the model had in view as it began
    /// to answer; absent from the lines recorded before it was read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<View>,
}

/// Where P1's selected files are read from: the repository's text files, and
/// the repository's name as the workspace prefixes its readings' paths.
pub struct Selection {
    pub files: Retriever,
    pub repo: String,
}

/// What a run is given.
pub struct Plan<'a> {
    pub base: &'a str,
    /// The member that answers every question, by node id.
    pub node: &'a str,
    pub thread: &'a str,
    pub questions: Vec<Question>,
    pub vocabulary: HashSet<String>,
    /// The arms asked, each up to its highest run.
    pub arms: Vec<(Arm, u32)>,
    /// R1's files block for each question, by id; empty when R1 is not asked.
    pub retrieval: HashMap<String, Packed>,
    /// What P1 reads its selected files from; `None` when P1 is not asked.
    pub selection: Option<Arc<Selection>>,
    pub seed: u64,
    pub concurrency: usize,
    pub out: &'a Path,
    pub telemetry: &'a Path,
    /// Where the K/V census taken as the run starts and as it ends goes:
    /// `census_start.json` and `census_end.json` in this folder.
    pub census_dir: &'a Path,
}

/// The `(arm, run, id)` of every answer `out` already records with an
/// answer. A line whose every try failed records no answer, so its question
/// is asked again; the analysis reads only answered lines.
pub fn recorded(out: &Path) -> anyhow::Result<Recorded> {
    let Ok(f) = File::open(out) else {
        return Ok(Recorded::new());
    };
    let mut done = Recorded::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let l: Line = serde_json::from_str(&line).context("a recorded answer")?;
        if l.error.is_none() {
            done.insert((l.arm, l.run, l.id));
        }
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
        retrieved: Vec::new(),
        retrieval_hit: None,
        tool_calls: None,
        view: None,
    }
}

/// `line` given R1's files for `q`, each answer file among them marked, and
/// what of `q`'s expected items they held.
pub fn with_retrieval(mut line: Line, q: &Question, packed: &Packed) -> Line {
    let mut files = packed.files.clone();
    line.retrieval_hit = Some(mark(&mut files, &q.files));
    line.retrieved = files;
    if line.error.is_none() {
        line.view = Some(with_named(
            of_packed(packed, &q.files, &q.expect),
            &line.answer,
            &q.expect,
        ));
    }
    line
}

/// The tag one launch names its conversations with: its start, in hex
/// seconds. A deleted conversation's id is tombstoned for good, so no launch
/// may ask on an id another used — a run taken up again after an interruption
/// asks every answer it still owes on ids of its own.
pub fn launch_tag(unix_secs: u64) -> String {
    format!("{unix_secs:x}")
}

/// The id one try of one question takes under launch `tag`.
pub fn attempt_id(qid: &str, tag: &str, attempt: u32) -> String {
    format!("{qid}-{tag}-a{attempt}")
}

/// Ask one job to its line: tried on a fresh conversation each time, deleted
/// once read.
async fn answer_job(
    zend: &Zend,
    plan_thread: &str,
    tag: &str,
    q: &Question,
    job: Job,
    vocabulary: &HashSet<String>,
    retrieved: Option<&Packed>,
) -> Line {
    let line = asked_line(zend, plan_thread, tag, q, job, vocabulary, retrieved).await;
    match retrieved {
        Some(packed) => with_retrieval(line, q, packed),
        None => line,
    }
}

/// P1's files for `q`: C1 asks it on a conversation of its own, and the
/// readings the point governing its answer's first token placed are packed as
/// R1's files are (`selected.rs`). Tried as [`asked_line`] tries an answer.
async fn select_files(
    zend: &Zend,
    plan_thread: &str,
    tag: &str,
    q: &Question,
    job: Job,
    sel: &Selection,
) -> Result<Packed, String> {
    selected_paths(zend, plan_thread, tag, q, job, sel)
        .await
        .map(|paths| sel.files.pack_paths(&paths))
}

/// The repository-relative paths C1's selection placed for `q`, strongest
/// first, as [`select_files`] packs them.
async fn selected_paths(
    zend: &Zend,
    plan_thread: &str,
    tag: &str,
    q: &Question,
    job: Job,
    sel: &Selection,
) -> Result<Vec<String>, String> {
    let ask = prompt(Arm::C1, &q.question, "");
    let mut last_error = String::new();
    let mut attempt = 1;
    let mut asked_n = 0;
    while attempt <= TRIES {
        asked_n += 1;
        let conv = conv_id(
            plan_thread,
            job.arm,
            job.run,
            &attempt_id(&format!("{}-sel", q.id), tag, asked_n),
        );
        let asked = zend.ask(&conv, &ask, Arm::C1).await;
        if let Err(e) = zend.delete(&conv).await {
            tracing::warn!(conv, "the conversation was not deleted: {e:#}");
        }
        match asked {
            Ok(a) => return Ok(selected(&a.points, a.think, &sel.repo)),
            Err(e) => {
                last_error = format!("{e:#}");
                tracing::warn!(conv, attempt, "the selection failed: {last_error}");
                if zend.was_down().await {
                    tracing::warn!(conv, "the member was down; asked again, no try spent");
                } else {
                    attempt += 1;
                }
            }
        }
    }
    Err(last_error)
}

/// The line for `job` when every try failed with `error`.
fn failed_line(q: &Question, job: Job, vocabulary: &HashSet<String>, error: String) -> Line {
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
    failed.error = Some(error);
    failed
}

/// A probe arm's line for `q` (`probe.rs`): `paths`, the shared selection,
/// packed under the arm's budget and placed where the arm places them, asked
/// on the passthrough route and its conversation deleted once answered.
async fn probe_line(
    zend: &Zend,
    q: &Question,
    job: Job,
    vocabulary: &HashSet<String>,
    sel: &Selection,
    paths: &[String],
) -> Line {
    let packed = sel
        .files
        .pack_paths_under(&probe_paths(job.arm, paths), budget_chars(job.arm));
    let (system, user) = messages(job.arm, &q.question, &packed);
    let mut last_error = String::new();
    let mut attempt = 1;
    while attempt <= TRIES {
        let asked = zend.ask_passthrough(&system, &user).await;
        let conv = passthrough_conv_id(&system, &user);
        if let Err(e) = zend.delete(&conv).await {
            tracing::warn!(conv, "the conversation was not deleted: {e:#}");
        }
        match asked {
            Ok(a) => {
                let line = scored(
                    q,
                    job.arm,
                    job.run,
                    &a.text,
                    vocabulary,
                    a.completion_tokens,
                    a.think,
                    Placement::default(),
                    a.secs,
                );
                return with_retrieval(line, q, &packed);
            }
            Err(e) => {
                last_error = format!("{e:#}");
                tracing::warn!(id = q.id, attempt, "the probe failed: {last_error}");
                if zend.was_down().await {
                    tracing::warn!(id = q.id, "the member was down; asked again, no try spent");
                } else {
                    attempt += 1;
                }
            }
        }
    }
    failed_line(q, job, vocabulary, last_error)
}

/// One job's line, as [`answer_job`] records it before its retrieval.
async fn asked_line(
    zend: &Zend,
    plan_thread: &str,
    tag: &str,
    q: &Question,
    job: Job,
    vocabulary: &HashSet<String>,
    retrieved: Option<&Packed>,
) -> Line {
    let prompt = prompt(
        job.arm,
        &q.question,
        retrieved.map_or("", |p| p.text.as_str()),
    );
    let mut last_error = String::new();
    // `asked_n` names each conversation, so no two asks share one; `attempt`
    // counts only the tries that were the answer's own to fail.
    let mut attempt = 1;
    let mut asked_n = 0;
    while attempt <= TRIES {
        asked_n += 1;
        let conv = conv_id(
            plan_thread,
            job.arm,
            job.run,
            &attempt_id(&q.id, tag, asked_n),
        );
        let asked = zend.ask(&conv, &prompt, job.arm).await;
        if let Err(e) = zend.delete(&conv).await {
            tracing::warn!(conv, "the conversation was not deleted: {e:#}");
        }
        match asked {
            Ok(a) => {
                let p = placement(&a.points, &q.files, a.think);
                let mut line = scored(
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
                line.tool_calls = Some(a.tool_calls);
                line.view = a.first_answer.as_ref().map(|detail| {
                    with_named(
                        of_point(detail, &q.files, &q.expect),
                        &line.answer,
                        &q.expect,
                    )
                });
                return line;
            }
            Err(e) => {
                last_error = format!("{e:#}");
                tracing::warn!(conv, attempt, "the answer failed: {last_error}");
                if zend.was_down().await {
                    tracing::warn!(conv, "the member was down; asked again, no try spent");
                } else {
                    attempt += 1;
                }
            }
        }
    }
    failed_line(q, job, vocabulary, last_error)
}

/// Run `plan` to its end.
pub async fn run(plan: Plan<'_>) -> anyhow::Result<()> {
    let zend = Arc::new(Zend::new(plan.base, plan.node)?);
    let done = recorded(plan.out)?;
    let ids: Vec<String> = plan.questions.iter().map(|q| q.id.clone()).collect();
    let jobs: Vec<Job> = queue(&ids, &plan.arms, &done, plan.seed);
    if let Some(j) = jobs
        .iter()
        .find(|j| j.arm == Arm::R1 && !plan.retrieval.contains_key(&ids[j.question]))
    {
        bail!(
            "R1 is asked {} with no files retrieved for it",
            ids[j.question]
        );
    }
    if plan.selection.is_none() && jobs.iter().any(|j| j.arm == Arm::P1 || j.arm.probe()) {
        bail!("P1 or a probe arm is asked with no repository to read its selected files from");
    }
    // The probe arms share one selection per question: the first to need it
    // asks C1, the others wait for that ask.
    let shared_selection: Arc<HashMap<String, Arc<SharedSelection>>> = Arc::new(
        plan.questions
            .iter()
            .map(|q| (q.id.clone(), Arc::new(OnceCell::new())))
            .collect(),
    );
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
    // A measurement of retrieval is of a settled memory: every file read,
    // calibrated and normalized before the first question.
    zend.until_settled().await?;
    census(&zend, &plan.census_dir.join("census_start.json")).await?;
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
    let retrieval = Arc::new(plan.retrieval);
    let selection = plan.selection;
    let thread = plan.thread.to_string();
    let tag = Arc::new(launch_tag(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    ));
    println!("this launch's conversations are tagged {tag}");
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
        let tag = Arc::clone(&tag);
        let retrieval = Arc::clone(&retrieval);
        let selection = selection.clone();
        let shared_selection = Arc::clone(&shared_selection);
        tasks.push(tokio::spawn(async move {
            in_flight.fetch_add(1, Ordering::AcqRel);
            let q = &questions[job.question];
            let line = match (job.arm, selection.as_deref()) {
                (arm, Some(sel)) if arm.probe() => {
                    let paths = shared_selection[&q.id]
                        .get_or_init(|| selected_paths(&zend, &thread, &tag, q, job, sel))
                        .await;
                    match paths {
                        Ok(paths) => probe_line(&zend, q, job, &vocabulary, sel, paths).await,
                        Err(e) => failed_line(q, job, &vocabulary, e.clone()),
                    }
                }
                (Arm::P1, Some(sel)) => match select_files(&zend, &thread, &tag, q, job, sel).await
                {
                    Ok(packed) => {
                        answer_job(&zend, &thread, &tag, q, job, &vocabulary, Some(&packed)).await
                    }
                    Err(e) => failed_line(q, job, &vocabulary, e),
                },
                _ => {
                    let files = (job.arm == Arm::R1).then(|| retrieval.get(&q.id)).flatten();
                    answer_job(&zend, &thread, &tag, q, job, &vocabulary, files).await
                }
            };
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
    census(&zend, &plan.census_dir.join("census_end.json")).await
}

/// The answering member's K/V census, written to `path`.
async fn census(zend: &Zend, path: &Path) -> anyhow::Result<()> {
    let c = zend
        .sample(&format!("/v1/kv/census?sample={CENSUS_SAMPLE}"))
        .await
        .context("the K/V census")?;
    std::fs::write(path, serde_json::to_string_pretty(&c)?)
        .with_context(|| format!("writing {}", path.display()))?;
    println!("K/V census written to {}", path.display());
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
    use std::env::temp_dir;
    use std::fs;
    use std::process;

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

    /// **No two launches share a conversation id**: a deleted id is
    /// tombstoned for good, so each launch's tries carry its own tag.
    #[test]
    fn each_launch_asks_on_ids_of_its_own() {
        assert_eq!(launch_tag(0x6923_ab10), "6923ab10");
        assert_eq!(attempt_id("t07", "6923ab10", 2), "t07-6923ab10-a2");
        assert_ne!(
            attempt_id("t07", &launch_tag(100), 1),
            attempt_id("t07", &launch_tag(101), 1)
        );
    }

    /// **R1's line names the files its prompt carried**, the answer file
    /// marked and the hit recorded; a line recorded before these fields reads
    /// back without them.
    #[test]
    fn r1s_line_records_what_it_retrieved() {
        let packed = Packed {
            text: String::new(),
            files: vec![
                RetrievedFile {
                    path: "b/other.rs".into(),
                    rank: 1,
                    score: 3.0,
                    chars: 10,
                    cut: false,
                    answer_file: false,
                },
                RetrievedFile {
                    path: "a/scan.rs".into(),
                    rank: 2,
                    score: 2.0,
                    chars: 4,
                    cut: true,
                    answer_file: false,
                },
            ],
        };
        let base = scored(
            &question(),
            Arm::R1,
            1,
            "scan.rs",
            &HashSet::new(),
            3,
            None,
            Placement::default(),
            1.0,
        );
        let l = with_retrieval(base, &question(), &packed);
        assert_eq!(l.retrieval_hit, Some(true));
        assert_eq!(
            l.retrieved
                .iter()
                .map(|f| (f.rank, f.answer_file))
                .collect::<Vec<_>>(),
            vec![(1, false), (2, true)]
        );
        let text = serde_json::to_string(&l).unwrap();
        assert!(text.contains("\"retrieval_hit\":true"));
        assert!(
            !text.contains("tool_calls"),
            "an unread count is not written"
        );

        let old = r#"{"arm":"a1","run":1,"id":"d05","category":"direct","names_file":false,
            "recall":0.0,"missed":[],"grounding":1.0,"ungrounded":[],"completion_tokens":263,
            "think":153,"placement":{"points":7,"in_thinking":3,"opening":false,
            "think_end":false,"first_answer":false,"ever":false,"series":[[0,false]]},
            "secs":39.5,"answer":"x"}"#;
        let back: Line = serde_json::from_str(old).unwrap();
        assert_eq!(
            (back.retrieved.len(), back.retrieval_hit, back.tool_calls),
            (0, None, None)
        );
        assert_eq!(back.view, None);
        assert_eq!(
            l.view.as_ref().map(|v| v.in_files.clone()),
            Some(vec![false, false]),
            "R1's view is read from its packed block, empty here"
        );
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

    /// A line whose every try failed is not a recorded answer: its question
    /// is asked again, and the answered line beside it is.
    #[test]
    fn a_failed_line_is_asked_again() {
        let answered = scored(
            &question(),
            Arm::C1,
            4,
            "x",
            &HashSet::new(),
            1,
            None,
            Placement::default(),
            1.0,
        );
        let mut failed = scored(
            &question(),
            Arm::B1,
            4,
            "",
            &HashSet::new(),
            0,
            None,
            Placement::default(),
            0.0,
        );
        failed.error = Some("connection refused".to_string());
        let path = temp_dir().join(format!("battery-recorded-{}.jsonl", process::id()));
        fs::write(
            &path,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&answered).unwrap(),
                serde_json::to_string(&failed).unwrap()
            ),
        )
        .unwrap();
        let done = recorded(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(done.contains(&("c1".to_string(), 4, "d1".to_string())));
        assert!(!done.contains(&("b1".to_string(), 4, "d1".to_string())));
        assert_eq!(done.len(), 1);
    }
}
