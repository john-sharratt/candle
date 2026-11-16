//! The scoring lab's formula: how a probe's votes become a score per file.
//!
//! Read from a YAML file the lab watches, so changing a weight, a gate or a
//! correction re-scores every question at once, with no daemon restarted and
//! no harness rebuilt. Every field has production's value as its default, so
//! an empty file scores as today's belief scan does.

use regex::Regex;
use serde::Deserialize;

use candle_conversation::provenance::window_score::{gate, Vote};
use candle_conversation::provenance::FOLD_GROUPS;

/// One probe token's vote in each fold group, as the daemon's scan returns it.
pub type TokenVote = [Option<Vote>; FOLD_GROUPS];

/// A formula as written. See the module documentation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FormulaSpec {
    /// How many files the lab treats as placed — recall is "an answer file in
    /// the top this many".
    pub budget_files: usize,
    /// The content competition's fold-group weights.
    pub groups: [f32; 3],
    /// The needle gate: the top fraction of probe tokens by summed vote kept.
    pub keep: f32,
    /// Each kept vote raised to this power before it is summed: below 1
    /// flattens a few huge votes, above 1 sharpens them.
    pub vote_power: f32,
    /// A file's score divided by its signed tokens to this power: 0 leaves
    /// size alone, 0.5 is a square-root length correction.
    pub length_power: f32,
    /// The calibration competition's weight against the content's; 0 leaves
    /// it out.
    pub calibration: f32,
    /// The calibration competition's fold-group weights.
    pub calibration_groups: [f32; 3],
    /// Correct each file by how much it scores on questions that are not
    /// about it: its mean score over the other questions of the run.
    pub hub: Hub,
    /// Per-path multipliers, first match wins; a path matching none is ×1.
    pub types: Vec<TypeWeight>,
}

impl Default for FormulaSpec {
    fn default() -> Self {
        Self {
            budget_files: 10,
            groups: [1.0, 1.0, 1.0],
            keep: 0.25,
            vote_power: 1.0,
            length_power: 0.0,
            calibration: 0.5,
            calibration_groups: [1.0, 1.0, 1.0],
            hub: Hub::None,
            types: Vec::new(),
        }
    }
}

/// How a file's score on other questions corrects its score on this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hub {
    /// No correction.
    None,
    /// Minus its mean over the other questions.
    Subtract,
    /// Divided by its mean over the other questions (plus one).
    Divide,
}

/// A multiplier for the files whose path a pattern matches.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeWeight {
    pub files: String,
    pub weight: f32,
}

/// A formula, its patterns compiled.
pub struct Formula {
    pub spec: FormulaSpec,
    types: Vec<(Regex, f32)>,
}

impl Formula {
    /// Compile `spec`, refusing a pattern that does not compile and a gate
    /// outside `(0, 1]`.
    pub fn new(spec: FormulaSpec) -> anyhow::Result<Self> {
        anyhow::ensure!(
            spec.keep > 0.0 && spec.keep <= 1.0,
            "keep {} must be in (0, 1]",
            spec.keep
        );
        anyhow::ensure!(spec.budget_files > 0, "budget_files must be at least 1");
        let types = spec
            .types
            .iter()
            .map(|t| Ok((Regex::new(&t.files)?, t.weight)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { spec, types })
    }

    /// Parse and compile the formula in `text`.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let spec: FormulaSpec = if text.trim().is_empty() {
            FormulaSpec::default()
        } else {
            serde_yaml::from_str(text)?
        };
        Self::new(spec)
    }

    /// Each file's raw score for one probe — before the hub correction, which
    /// needs every question's: its content votes and its calibration votes
    /// gated, raised, summed, length-corrected and typed. `paths[f]` and
    /// `tokens[f]` are file `f`'s path and signed tokens.
    pub fn raw(
        &self,
        content: &[TokenVote],
        calibration: &[TokenVote],
        paths: &[&str],
        tokens: &[usize],
    ) -> Vec<f32> {
        let s = &self.spec;
        let n = paths.len();
        let mut out = self.summed(content, &s.groups, n);
        if s.calibration != 0.0 {
            for (o, c) in out
                .iter_mut()
                .zip(self.summed(calibration, &s.calibration_groups, n))
            {
                *o += s.calibration * c;
            }
        }
        for (f, o) in out.iter_mut().enumerate() {
            if s.length_power != 0.0 {
                *o /= (tokens[f].max(1) as f32).powf(s.length_power);
            }
            *o *= self.type_weight(paths[f]);
        }
        out
    }

    /// `raw[q]` for every question `q`, corrected by each file's mean over the
    /// other questions as [`FormulaSpec::hub`] says.
    pub fn corrected(&self, raw: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if self.spec.hub == Hub::None || raw.len() < 2 {
            return raw.to_vec();
        }
        let n = raw[0].len();
        let total: Vec<f32> = (0..n).map(|f| raw.iter().map(|r| r[f]).sum()).collect();
        let others = (raw.len() - 1) as f32;
        raw.iter()
            .map(|r| {
                (0..n)
                    .map(|f| {
                        let mean = (total[f] - r[f]) / others;
                        match self.spec.hub {
                            Hub::Subtract => r[f] - mean,
                            Hub::Divide => r[f] / (1.0 + mean),
                            Hub::None => r[f],
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// The multiplier for `path`: the first type it matches, else 1.
    pub fn type_weight(&self, path: &str) -> f32 {
        self.types
            .iter()
            .find(|(r, _)| r.is_match(path))
            .map_or(1.0, |(_, w)| *w)
    }

    /// Votes weighed by `groups`, gated at the formula's keep, each raised to
    /// its power, summed per file.
    fn summed(&self, votes: &[TokenVote], groups: &[f32; 3], n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for v in gate(votes, groups, self.spec.keep) {
            if v.file < n {
                out[v.file] += v.weight.signum() * v.weight.abs().powf(self.spec.vote_power);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vote(file: usize, weight: f32) -> Option<Vote> {
        Some(Vote {
            token: usize::MAX,
            file,
            weight,
        })
    }

    /// Four probe tokens voting in group 0 only: file 0 by 8, 1 by 4, 2 by 2
    /// and 1 by 1.
    fn votes() -> Vec<TokenVote> {
        vec![
            [vote(0, 8.0), None, None],
            [vote(1, 4.0), None, None],
            [vote(2, 2.0), None, None],
            [vote(1, 1.0), None, None],
        ]
    }

    const PATHS: [&str; 3] = ["candle/README.md", "candle/a.rs", "candle/b.rs"];

    /// **An empty formula is production's**: every group weighed 1, the top
    /// quarter of tokens kept — here the one 8-vote — and the calibration at
    /// half weight.
    #[test]
    fn an_empty_formula_scores_as_production() {
        let f = Formula::parse("").unwrap();
        assert_eq!(f.spec.keep, 0.25);
        assert_eq!(f.spec.calibration, 0.5);
        let calibration = vec![[vote(2, 6.0), None, None]];
        let raw = f.raw(&votes(), &calibration, &PATHS, &[10, 10, 10]);
        assert_eq!(raw, vec![8.0, 0.0, 3.0]);
    }

    /// **Every knob does what it says**: the whole gate keeps all four votes,
    /// a square root flattens them, a length power divides by size, and a
    /// type weight multiplies the files it matches.
    #[test]
    fn each_knob_changes_the_score_as_written() {
        let f = Formula::parse("keep: 1.0\ncalibration: 0\n").unwrap();
        assert_eq!(
            f.raw(&votes(), &[], &PATHS, &[1, 1, 1]),
            vec![8.0, 5.0, 2.0]
        );

        let f = Formula::parse("keep: 1.0\ncalibration: 0\nvote_power: 0.5\n").unwrap();
        let r = f.raw(&votes(), &[], &PATHS, &[1, 1, 1]);
        assert!((r[0] - 8f32.sqrt()).abs() < 1e-6 && (r[1] - 3.0).abs() < 1e-6);

        let f = Formula::parse("keep: 1.0\ncalibration: 0\nlength_power: 1.0\n").unwrap();
        assert_eq!(
            f.raw(&votes(), &[], &PATHS, &[4, 5, 1]),
            vec![2.0, 1.0, 2.0]
        );

        let f = Formula::parse(
            "keep: 1.0\ncalibration: 0\ntypes:\n  - { files: '\\.md$', weight: 0.25 }\n",
        )
        .unwrap();
        assert_eq!(
            f.raw(&votes(), &[], &PATHS, &[1, 1, 1]),
            vec![2.0, 5.0, 2.0]
        );
    }

    /// **A hub correction reads a file against the other questions**: a file
    /// that scores the same on every question loses its lead.
    #[test]
    fn a_hub_correction_reads_a_file_against_the_other_questions() {
        let raw = vec![vec![10.0, 3.0], vec![10.0, 0.0], vec![10.0, 0.0]];
        let f = Formula::parse("hub: subtract").unwrap();
        assert_eq!(f.corrected(&raw)[0], vec![0.0, 3.0]);
        let f = Formula::parse("hub: divide").unwrap();
        assert_eq!(f.corrected(&raw)[0], vec![10.0 / 11.0, 3.0]);
        assert_eq!(Formula::parse("").unwrap().corrected(&raw), raw);
    }

    /// **A formula the arithmetic cannot use is refused**, saying why.
    #[test]
    fn a_bad_formula_is_refused() {
        let err = |t: &str| Formula::parse(t).err().unwrap().to_string();
        assert_eq!(err("keep: 0"), "keep 0 must be in (0, 1]");
        assert_eq!(err("budget_files: 0"), "budget_files must be at least 1");
        assert!(err("types:\n  - { files: '(', weight: 1 }").contains("regex"));
        assert!(err("kepe: 0.5").contains("unknown field"));
    }
}
