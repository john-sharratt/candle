//! The rules gate 4 and the questions gate rank files by, each scored on the
//! daemon's gallery: today's scorer with and without the calibration questions in
//! the gallery; the over-peak hub correction, which gate 4 withdrew; and
//! surprise against each file's background ([`crate::background`]), alone and
//! blended with the score's size.

use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::hubness::{corrected, peak, HubCorrection};
use candle_conversation::provenance::WideQSig;

use crate::background::{learn, question_probes, scaled_scores, to_ref, Background};
use crate::corpus::File;
use crate::remote_gallery::{Questions, RemoteGallery};
use crate::sweep::{background_sources, summed};

/// The probe length a peak is scaled to: the live probe cap.
const T_REF: usize = 256;

/// Background probes a peak is learned from: production's warm.
const BACKGROUND_PROBES: usize = 64;

/// How a rule turns a probe's summed votes into file scores.
#[derive(Clone, Copy)]
pub enum Fold {
    /// The votes as they are, or over a hub correction's peak.
    Hub(HubCorrection),
    /// Surprise against each file's background, plus `blend × ln(1 + x)`.
    Surprise { blend: f32 },
}

/// One way of ranking files.
pub struct Rule {
    pub name: &'static str,
    pub weights: [f32; 3],
    pub keep: f32,
    pub fold: Fold,
    pub questions: Questions,
}

pub const RULES: [Rule; 6] = [
    Rule {
        name: "today",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: Fold::Hub(HubCorrection::None),
        questions: Questions::Without,
    },
    Rule {
        name: "today + questions",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: Fold::Hub(HubCorrection::None),
        questions: Questions::With,
    },
    Rule {
        name: "over-peak + questions",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: Fold::Hub(HubCorrection::OverPeak { floor: 100.0 }),
        questions: Questions::With,
    },
    Rule {
        name: "surprise + questions",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: Fold::Surprise { blend: 0.0 },
        questions: Questions::With,
    },
    Rule {
        name: "surprise+0.5 + questions",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: Fold::Surprise { blend: 0.5 },
        questions: Questions::With,
    },
    Rule {
        name: "surprise+1.0 + questions",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: Fold::Surprise { blend: 1.0 },
        questions: Questions::With,
    },
];

/// Every rule's file scores for a probe, its peaks and backgrounds learned
/// once.
pub struct Scorer<'g, 'a> {
    gpu: &'g RemoteGallery<'a>,
    n_files: usize,
    /// Each rule's per-file background peak; empty for a rule that needs none.
    peaks: Vec<Vec<f32>>,
    /// Each file's background against the gallery with the questions in it,
    /// when a surprise rule needs one.
    backgrounds: Vec<Background>,
}

impl<'g, 'a> Scorer<'g, 'a> {
    pub fn new(
        gpu: &'g RemoteGallery<'a>,
        files: &[File],
        rule: &BoundaryRule,
    ) -> anyhow::Result<Self> {
        let sources = background_sources(files, rule, BACKGROUND_PROBES);
        let mut peaks = Vec::with_capacity(RULES.len());
        for r in &RULES {
            if !matches!(r.fold, Fold::Hub(HubCorrection::OverPeak { .. })) {
                peaks.push(Vec::new());
                continue;
            }
            let mut with_len = Vec::with_capacity(sources.len());
            for (fi, sigs) in &sources {
                let votes = gpu.votes(
                    &[sigs.as_slice()],
                    &|f, _| f == *fi,
                    r.questions,
                    &|f, _| f == *fi,
                )?;
                let s = summed(&votes[0], &r.weights, r.keep, files.len());
                with_len.push((s, sigs.len()));
            }
            peaks.push(peak(&with_len, files.len(), T_REF));
        }
        let backgrounds = if RULES
            .iter()
            .any(|r| matches!(r.fold, Fold::Surprise { .. }))
        {
            let probes = question_probes(files);
            let scaled = scaled_scores(gpu, &probes, Questions::With, files.len())?;
            learn(&probes, &scaled, files.len())
        } else {
            Vec::new()
        };
        Ok(Scorer {
            gpu,
            n_files: files.len(),
            peaks,
            backgrounds,
        })
    }

    /// Every rule's score per file for `probe`, against the gallery without
    /// the turns `masked` names (as `(file, turn)`) and the questions
    /// `masked_question` names (as `(file, question)`).
    pub fn score(
        &self,
        probe: &[WideQSig],
        masked: &dyn Fn(usize, usize) -> bool,
        masked_question: &dyn Fn(usize, usize) -> bool,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let without = self
            .gpu
            .votes(&[probe], masked, Questions::Without, masked_question)?;
        let with = self
            .gpu
            .votes(&[probe], masked, Questions::With, masked_question)?;
        Ok(RULES
            .iter()
            .zip(&self.peaks)
            .map(|(r, peaks)| {
                let votes = match r.questions {
                    Questions::Without => &without[0],
                    Questions::With => &with[0],
                };
                let s = summed(votes, &r.weights, r.keep, self.n_files);
                match r.fold {
                    Fold::Hub(HubCorrection::None) => s,
                    Fold::Hub(h) => corrected(&s, peaks, h, probe.len(), T_REF),
                    Fold::Surprise { blend } => s
                        .iter()
                        .zip(&self.backgrounds)
                        .map(|(&x, b)| b.blended(to_ref(x, probe.len()), blend))
                        .collect(),
                }
            })
            .collect())
    }
}
