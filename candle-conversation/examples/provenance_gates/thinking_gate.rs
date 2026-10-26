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

use crate::attribution::{BATCH, KEEP};
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
fn combined(a: &[Vec<f32>], b: &[Vec<f32>], w: f32) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p + w * q).collect())
        .collect()
}

/// Whether `q` was calibrated with its thinking.
fn thought(q: &Question) -> bool {
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
    Ok(())
}

/// `held` asked of the calibration competition alone — every file turn masked
/// — with `masked` questions left out and `excluded` cut.
fn competition(
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
