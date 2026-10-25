//! The rules gate 4 and the questions gate rank files by, each scored on the
//! GPU gallery: today's scorer, and the live one — over-peak hub correction —
//! each with and without the calibration questions in the gallery.

use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::hubness::{corrected, peak, HubCorrection};
use candle_conversation::provenance::WideQSig;

use crate::corpus::File;
use crate::gpu_gallery::{GpuGallery, Questions};
use crate::sweep::{background_sources, summed};

/// The probe length a peak is scaled to: the live probe cap.
const T_REF: usize = 256;

/// Background probes a rule's peaks are learned from: production's warm.
const BACKGROUND_PROBES: usize = 64;

/// One way of ranking files.
pub struct Rule {
    pub name: &'static str,
    pub weights: [f32; 3],
    pub keep: f32,
    pub fold: HubCorrection,
    pub questions: Questions,
}

pub const RULES: [Rule; 4] = [
    Rule {
        name: "today",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: HubCorrection::None,
        questions: Questions::Without,
    },
    Rule {
        name: "today + questions",
        weights: [1.0, 1.0, 1.0],
        keep: 0.25,
        fold: HubCorrection::None,
        questions: Questions::With,
    },
    Rule {
        name: "live",
        weights: [0.0, 0.0, 1.0],
        keep: 0.25,
        fold: HubCorrection::OverPeak { floor: 100.0 },
        questions: Questions::Without,
    },
    Rule {
        name: "live + questions",
        weights: [0.0, 0.0, 1.0],
        keep: 0.25,
        fold: HubCorrection::OverPeak { floor: 100.0 },
        questions: Questions::With,
    },
];

/// Every rule's file scores for a probe, its background peaks learned once.
pub struct Scorer<'g, 'a> {
    gpu: &'g GpuGallery<'a>,
    n_files: usize,
    /// Each rule's per-file background peak; empty for a rule that needs none.
    peaks: Vec<Vec<f32>>,
}

impl<'g, 'a> Scorer<'g, 'a> {
    pub fn new(
        gpu: &'g GpuGallery<'a>,
        files: &[File],
        rule: &BoundaryRule,
    ) -> anyhow::Result<Self> {
        let sources = background_sources(files, rule, BACKGROUND_PROBES);
        let mut peaks = Vec::with_capacity(RULES.len());
        for r in &RULES {
            if !matches!(r.fold, HubCorrection::OverPeak { .. }) {
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
        Ok(Scorer {
            gpu,
            n_files: files.len(),
            peaks,
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
                    HubCorrection::None => s,
                    h => corrected(&s, peaks, h, probe.len(), T_REF),
                }
            })
            .collect())
    }
}
