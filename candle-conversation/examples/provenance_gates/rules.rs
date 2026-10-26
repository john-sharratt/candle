//! The rules gate 4 and the questions gate rank files by, each scored on the
//! daemon's gallery as production's belief scan scores a `cases: file` group:
//! the files' content in one competition, the calibration questions and their
//! thinking in a competition of their own, the file's score its content's
//! plus `calibration × ` its calibration's. A rule names each competition's
//! fold-group weighting and the calibration's weight.

use candle_conversation::provenance::WideQSig;

use crate::remote_gallery::{Questions, RemoteGallery};
use crate::sweep::summed;

/// The needle gate's keep fraction, production's.
const KEEP: f32 = 0.25;

/// One way of ranking files.
pub struct Rule {
    pub name: &'static str,
    /// The content competition's fold-group weights.
    pub content: [f32; 3],
    /// The calibration competition's fold-group weights.
    pub calibration_groups: [f32; 3],
    /// The calibration competition's weight against the content's; 0 leaves
    /// it out.
    pub calibration: f32,
}

pub const RULES: [Rule; 5] = [
    Rule {
        name: "content [1,1,1]",
        content: [1.0, 1.0, 1.0],
        calibration_groups: [1.0, 1.0, 1.0],
        calibration: 0.0,
    },
    Rule {
        name: "[1,1,1] + 0.5 x [1,1,1] (code_reading)",
        content: [1.0, 1.0, 1.0],
        calibration_groups: [1.0, 1.0, 1.0],
        calibration: 0.5,
    },
    Rule {
        name: "[0,1,1] + 0.25 x [0,1,1]",
        content: [0.0, 1.0, 1.0],
        calibration_groups: [0.0, 1.0, 1.0],
        calibration: 0.25,
    },
    Rule {
        name: "[0,1,1] + 0.25 x [0,0,1]",
        content: [0.0, 1.0, 1.0],
        calibration_groups: [0.0, 0.0, 1.0],
        calibration: 0.25,
    },
    Rule {
        name: "[0,1,1] + 0.5 x [0,0,1]",
        content: [0.0, 1.0, 1.0],
        calibration_groups: [0.0, 0.0, 1.0],
        calibration: 0.5,
    },
];

/// Every rule's file scores for a probe.
pub struct Scorer<'g, 'a> {
    gpu: &'g RemoteGallery<'a>,
    n_files: usize,
}

impl<'g, 'a> Scorer<'g, 'a> {
    pub fn new(gpu: &'g RemoteGallery<'a>, n_files: usize) -> Self {
        Scorer { gpu, n_files }
    }

    /// Every rule's score per file for `probe`: its content against the
    /// gallery without the turns `masked` names (as `(file, turn)`), its
    /// calibration against every calibration question but those
    /// `masked_question` names (as `(file, question)`).
    pub fn score(
        &self,
        probe: &[WideQSig],
        masked: &dyn Fn(usize, usize) -> bool,
        masked_question: &dyn Fn(usize, usize) -> bool,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let content = self
            .gpu
            .votes(&[probe], masked, Questions::Without, &|_, _| false)?;
        let calibration =
            self.gpu
                .votes(&[probe], &|_, _| true, Questions::With, masked_question)?;
        Ok(RULES
            .iter()
            .map(|r| {
                let c = summed(&content[0], &r.content, KEEP, self.n_files);
                let q = summed(&calibration[0], &r.calibration_groups, KEEP, self.n_files);
                c.iter()
                    .zip(&q)
                    .map(|(a, b)| a + r.calibration * b)
                    .collect()
            })
            .collect())
    }
}
