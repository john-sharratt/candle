//! Does a calibration question's thinking earn its place?
//!
//! A calibration turn is a question and the thinking it starts. A live probe
//! is a question at a turn's opening and the model's thinking after it, so
//! each is asked: a held-out question's words, and its thinking. Each is
//! scored as production scores it — the files' content, plus half the
//! calibration competition — with the competition's windows whole, cut to
//! their questions, or cut to their thinking, so each half's part shows.
//!
//! Only files calibrated with thinking take part in the competition (mid-pass
//! the rest still hold questions alone); every file's content does.

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::provenance::WideQSig;

use crate::anomaly;
use crate::attribution::{attributed, background_mass, Chunks, BATCH, KEEP};
use crate::corpus::{File, Question};
use crate::judge::{header, Judge};
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::sweep::summed;

/// Questions per file the competition holds, beyond question 0.
const GALLERY: usize = 8;

/// The calibration competition's weight, production's.
const WEIGHT: f32 = 0.5;

/// What of each competition window stays.
#[derive(Clone, Copy)]
enum Keep {
    Whole,
    Question,
    Thinking,
}

/// `a + w·b` per file, per probe.
pub fn combined(a: &[Vec<f32>], b: &[Vec<f32>], w: f32) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p + w * q).collect())
        .collect()
}

/// Whether `q` was calibrated with its thinking.
pub fn thought(q: &Question) -> bool {
    !q.answer.is_empty()
}

/// What of `q`'s signatures `keep` names.
fn part(q: &Question, keep: Keep) -> &[WideQSig] {
    match keep {
        Keep::Whole => &q.sigs,
        Keep::Question => q.body_sigs(),
        Keep::Thinking => &q.sigs[q.answer.clone()],
    }
}

/// The ranges cut from `q`'s turn so only what `keep` names stays.
fn cuts(q: &Question, keep: Keep) -> Vec<(u64, usize, usize)> {
    let sid = turn_stream_id(q.timeline, q.turn).0;
    let n = q.sigs.len();
    let kept = match keep {
        Keep::Whole => return Vec::new(),
        Keep::Question => q.body.clone(),
        Keep::Thinking => q.answer.clone(),
    };
    [(sid, 0, kept.start), (sid, kept.end, n)]
        .into_iter()
        .filter(|&(_, s, e)| s < e)
        .collect()
}

pub fn run(files: &[File], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let n = files.len();
    // Files whose first `GALLERY + 1` questions all carry thinking.
    let ready: Vec<bool> = files
        .iter()
        .map(|f| f.questions.len() > GALLERY && f.questions[..=GALLERY].iter().all(thought))
        .collect();
    let asked: Vec<usize> = (0..n).filter(|&fi| ready[fi]).collect();
    println!(
        "thinking: {} files calibrated with thinking (of {} calibrated)",
        asked.len(),
        files.iter().filter(|f| !f.questions.is_empty()).count()
    );
    anyhow::ensure!(!asked.is_empty(), "no file is calibrated with thinking yet");
    let masked = |fi: usize, qi: usize| !ready[fi] || qi == 0 || qi > GALLERY;

    for (name, probe) in [("question", Keep::Question), ("thinking", Keep::Thinking)] {
        let held: Vec<(usize, &[WideQSig])> = asked
            .iter()
            .map(|&fi| (fi, part(&files[fi].questions[0], probe)))
            .collect();
        let content = Judge {
            gpu: &gpu,
            held_out: &held,
            hubs: &[],
            n,
            questions: Questions::Without,
            masked_question: &|_, _| false,
        };
        let c = content.scores(&[])?;
        header(&format!("{name} probes ({})", held.len()));
        content.print("content alone", &c);
        for (label, keep) in [
            ("question + thinking", Keep::Whole),
            ("question only", Keep::Question),
            ("thinking only", Keep::Thinking),
        ] {
            let excluded: Vec<(u64, usize, usize)> = asked
                .iter()
                .flat_map(|&fi| files[fi].questions[1..=GALLERY].iter())
                .flat_map(|q| cuts(q, keep))
                .collect();
            let q = competition(&gpu, &held, n, &masked, &excluded)?;
            content.print(&format!("  competition alone, {label}"), &q);
            content.print(
                &format!("  content + {WEIGHT} x {label}"),
                &combined(&c, &q, WEIGHT),
            );
        }
    }

    // A live probe holds both halves: asked as one turn, and as its two halves
    // asked apart with their scores summed.
    let probes = |keep| -> Vec<(usize, &[WideQSig])> {
        asked
            .iter()
            .map(|&fi| (fi, part(&files[fi].questions[0], keep)))
            .collect()
    };
    let (whole, question, thinking) = (
        probes(Keep::Whole),
        probes(Keep::Question),
        probes(Keep::Thinking),
    );
    let judge = |held| Judge {
        gpu: &gpu,
        held_out: held,
        hubs: &[],
        n,
        questions: Questions::Without,
        masked_question: &|_, _| false,
    };
    let (jw, jq, jt) = (judge(&whole), judge(&question), judge(&thinking));
    let (cw, cq, ct) = (jw.scores(&[])?, jq.scores(&[])?, jt.scores(&[])?);
    let (qw, qq, qt) = (
        competition(&gpu, &whole, n, &masked, &[])?,
        competition(&gpu, &question, n, &masked, &[])?,
        competition(&gpu, &thinking, n, &masked, &[])?,
    );
    let (c_sum, q_sum) = (combined(&cq, &ct, 1.0), combined(&qq, &qt, 1.0));
    header(&format!(
        "live probes, question and thinking ({})",
        whole.len()
    ));
    jw.print("one turn, content alone", &cw);
    jw.print("halves summed, content alone", &c_sum);
    for &w in &FUSION_WEIGHTS {
        jw.print(&format!("  one turn, w {w}"), &combined(&cw, &qw, w));
        jw.print(
            &format!("  halves summed, w {w}"),
            &combined(&c_sum, &q_sum, w),
        );
    }

    // The competition's own generic chunks — thinking openers every question
    // shares — cut as content's are: their mass is what they win on other
    // files' questions past the gallery's, asked of the competition alone.
    let chunks = Chunks::of(files, false, &|fi, qi| !masked(fi, qi));
    let background: Vec<(usize, &[WideQSig])> = asked
        .iter()
        .flat_map(|&fi| {
            files[fi].questions[GALLERY + 1..]
                .iter()
                .filter(|q| thought(q))
                .map(move |q| (fi, q.sigs.as_slice()))
        })
        .collect();
    let bg = attributed(&gpu, &chunks, &background, false, Questions::With, &masked)?;
    let mass = background_mass(&chunks, &background, &bg);
    println!(
        "\ncompetition chunks {}, background {} questions",
        mass.len(),
        background.len()
    );
    // What the most generic competition chunks are: their question and where
    // in its turn they sit.
    let mut by_mass: Vec<usize> = (0..mass.len()).collect();
    by_mass.sort_by(|&a, &b| mass[b].total_cmp(&mass[a]));
    let total: f32 = mass.iter().sum();
    for &c in by_mass.iter().take(SHOWN) {
        let (sid, start, end) = chunks.range_of[c];
        let fi = chunks.file_of[c];
        let Some((qi, q)) = files[fi]
            .questions
            .iter()
            .enumerate()
            .find(|(_, q)| turn_stream_id(q.timeline, q.turn).0 == sid)
        else {
            continue;
        };
        let part = if end <= q.body.end && start >= q.body.start {
            "question"
        } else if start >= q.answer.start && end <= q.answer.end {
            "thinking"
        } else {
            "spans/framing"
        };
        println!(
            "{:>10.0} ({:>4.1}%)  {} q{qi} [{start}..{end}] of {} (question {:?}, thinking {:?}) {part}",
            mass[c],
            100.0 * mass[c] / total.max(1e-6),
            files[fi].path.as_deref().unwrap_or("<no path>"),
            q.sigs.len(),
            q.body,
            q.answer
        );
    }
    for (name, probe) in [("question", Keep::Question), ("thinking", Keep::Thinking)] {
        let held: Vec<(usize, &[WideQSig])> = asked
            .iter()
            .map(|&fi| (fi, part(&files[fi].questions[0], probe)))
            .collect();
        let content = Judge {
            gpu: &gpu,
            held_out: &held,
            hubs: &[],
            n,
            questions: Questions::Without,
            masked_question: &|_, _| false,
        };
        let c = content.scores(&[])?;
        header(&format!("{name} probes: weight and competition cut"));
        for &k in &CUT_KS {
            let cut: Vec<(u64, usize, usize)> = match k {
                Some(k) => anomaly::excluded(&mass, &chunks.file_of, k, CUT_KEEP)
                    .into_iter()
                    .map(|c| chunks.range_of[c])
                    .collect(),
                None => Vec::new(),
            };
            let q = competition(&gpu, &held, n, &masked, &cut)?;
            let label = k.map_or("no cut".to_string(), |k| format!("k {k} ({})", cut.len()));
            for &w in &WEIGHTS {
                content.print(&format!("  {label}, w {w}"), &combined(&c, &q, w));
            }
        }
    }
    Ok(())
}

/// The most generic competition chunks shown.
const SHOWN: usize = 25;

/// Thresholds of the competition cut tried, `None` for none.
const CUT_KS: [Option<f32>; 4] = [None, Some(5.0), Some(4.0), Some(3.0)];

/// The share of each file's competition chunks a cut leaves it.
const CUT_KEEP: f32 = 0.5;

/// Weights of the calibration competition tried.
const WEIGHTS: [f32; 3] = [0.25, 0.5, 1.0];

/// Weights of the calibration competition tried on live probes.
const FUSION_WEIGHTS: [f32; 5] = [0.1, 0.25, 0.35, 0.5, 0.75];

/// `held` asked of the calibration competition alone — every file turn masked
/// — with `masked` questions left out and `excluded` cut.
pub fn competition(
    gpu: &RemoteGallery,
    held: &[(usize, &[WideQSig])],
    n: usize,
    masked: &dyn Fn(usize, usize) -> bool,
    excluded: &[(u64, usize, usize)],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(held.len());
    for batch in held.chunks(BATCH) {
        let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
        let votes = gpu.votes_without(&sigs, &|_, _| true, Questions::With, masked, excluded)?;
        out.extend(votes.iter().map(|v| summed(v, &[1.0, 1.0, 1.0], KEEP, n)));
    }
    Ok(out)
}
