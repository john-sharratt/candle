//! The scoring lab's formula: how a probe's votes become a score per file.
//!
//! Read from a YAML file the lab watches, so changing a weight, a gate or a
//! correction re-scores every question at once, with no daemon restarted and
//! no harness rebuilt. Every field has production's value as its default, so
//! an empty file scores as today's belief scan does.

use std::collections::HashSet;

use regex::Regex;
use serde::Deserialize;

use candle_conversation::provenance::window_score::{gate, Ranked, Vote};
use candle_conversation::provenance::FOLD_GROUPS;

/// One probe token's vote in each fold group, as the daemon's scan returns it.
pub type TokenVote = [Option<Vote>; FOLD_GROUPS];

/// The tail a full hit sits at: 1000 is beyond 99 in 100 of a background, as
/// production's normalization reads it (`docs/provenance_score_normalization.md`
/// §3.2).
const FULL_HIT_TAIL: f32 = 0.01;

/// Scoring samples a file needs for a background of its own; fewer, and it is
/// read against every file's pooled samples — production's `min_background`.
const MIN_BACKGROUND: usize = 8;

/// A formula as written. See the module documentation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FormulaSpec {
    /// How many files the lab treats as placed — recall is "an answer file in
    /// the top this many".
    pub budget_files: usize,
    /// How the fold groups' votes combine.
    pub fusion: Fusion,
    /// Files each probe token votes for in each group of the content
    /// competition: its best `topk`, file `r` by `z_r × (a_r − a_{topk+1})`
    /// — 1 is production's single `z × margin` vote. Above 1 the content is
    /// scored from the lab's CPU ranking ([`Votes::ranked`]), additively.
    pub topk: usize,
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
    /// Correct each file by what it scores on questions that are not about
    /// it.
    pub hub: Hub,
    /// Where the questions that are not about a file come from.
    pub hub_from: HubFrom,
    /// Leave the priming chain's files out of the ranking: every conversation
    /// opens on them already.
    pub exclude_chain: bool,
    /// Paths left out of the ranking, by pattern.
    pub exclude: Vec<String>,
    /// Per-path multipliers, first match wins; a path matching none is ×1.
    pub types: Vec<TypeWeight>,
}

impl Default for FormulaSpec {
    fn default() -> Self {
        Self {
            budget_files: 10,
            fusion: Fusion::Additive,
            topk: 1,
            groups: [1.0, 1.0, 1.0],
            keep: 0.25,
            vote_power: 1.0,
            length_power: 0.0,
            calibration: 0.5,
            calibration_groups: [1.0, 1.0, 1.0],
            hub: Hub::None,
            hub_from: HubFrom::Background,
            exclude_chain: false,
            exclude: Vec::new(),
            types: Vec::new(),
        }
    }
}

/// How the fold groups' votes combine into a file's score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fusion {
    /// Every group's weighed votes gated together and summed — production's.
    Additive,
    /// Each group gated on its own and the tallies summed, a file scoring
    /// only when the content group (0) gave it anything — the content axes'
    /// law (`docs/tool_selection_provenance_results.md` §25.5): an identity-
    /// group spike with no content agreement counts for nothing.
    ContentGated,
}

/// How a file's scores on other questions correct its score on this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hub {
    /// No correction.
    None,
    /// Minus its mean background score.
    Subtract,
    /// Divided by its mean background score (plus one).
    Divide,
    /// Read as tail surprise against its background, as production
    /// normalizes: `1000 × −ln P(background ≥ x) / ln(1 / 0.01)`.
    Surprise,
}

/// Which probes are a file's background.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HubFrom {
    /// The other battery questions.
    Others,
    /// The background probes the lab scanned: other files' calibration
    /// questions.
    Background,
    /// Both.
    Both,
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
    exclude: Vec<Regex>,
}

/// One background probe's raw scores, and the file it is about.
pub struct Sample {
    pub about: Option<usize>,
    pub scores: Vec<f32>,
}

/// One probe's votes in both competitions.
pub struct Votes<'a> {
    pub content: &'a [TokenVote],
    pub calibration: &'a [TokenVote],
    /// The content competition ranked on the CPU — each probe token's best
    /// files per group — when the lab computed it; what `topk` above 1 reads.
    pub ranked: Option<&'a [[Ranked; FOLD_GROUPS]]>,
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
        let exclude = spec
            .exclude
            .iter()
            .map(|p| Ok(Regex::new(p)?))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            spec,
            types,
            exclude,
        })
    }

    /// Parse and compile the formula in `text` — the tests' way in; the lab
    /// reads a formula file through `lab::score_file`, which also takes
    /// variants.
    #[cfg(test)]
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let spec: FormulaSpec = if text.trim().is_empty() {
            FormulaSpec::default()
        } else {
            serde_yaml::from_str(text)?
        };
        Self::new(spec)
    }

    /// Each file's raw score for one probe — before the hub correction, which
    /// needs the background's: its content votes and its calibration votes
    /// fused, raised, summed, length-corrected and typed. `paths[f]` and
    /// `tokens[f]` are file `f`'s path and signed tokens.
    pub fn raw(&self, votes: &Votes, paths: &[&str], tokens: &[usize]) -> Vec<f32> {
        let s = &self.spec;
        let n = paths.len();
        let mut out = match (s.topk, votes.ranked) {
            (k, Some(ranked)) if k > 1 => self.top_k_summed(ranked, &s.groups, k, n),
            _ => self.fused(votes.content, &s.groups, n),
        };
        if s.calibration != 0.0 {
            let cal = self.fused(votes.calibration, &s.calibration_groups, n);
            for (o, c) in out.iter_mut().zip(cal) {
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

    /// Each question's `raw` scores corrected against each file's background
    /// as [`FormulaSpec::hub`] and [`FormulaSpec::hub_from`] say:
    /// `background` are the raw scores of the background probes, each with
    /// the file it is about — a probe is a hit for that file, never part of
    /// its background — and a question's own scores are never part of its.
    pub fn corrected(&self, raw: &[Vec<f32>], background: &[Sample]) -> Vec<Vec<f32>> {
        if self.spec.hub == Hub::None || raw.is_empty() {
            return raw.to_vec();
        }
        let n = raw[0].len();
        (0..raw.len())
            .map(|q| {
                let mut samples: Vec<(Option<usize>, &Vec<f32>)> = Vec::new();
                if self.spec.hub_from != HubFrom::Background {
                    samples.extend(others(raw, q).map(|r| (None, r)));
                }
                if self.spec.hub_from != HubFrom::Others {
                    samples.extend(background.iter().map(|s| (s.about, &s.scores)));
                }
                if samples.is_empty() {
                    return raw[q].clone();
                }
                let pooled: Vec<f32> = samples
                    .iter()
                    .flat_map(|(_, s)| s.iter().copied())
                    .filter(|&x| x > 0.0)
                    .collect();
                (0..n)
                    .map(|f| {
                        let x = raw[q][f];
                        let mine: Vec<f32> = samples
                            .iter()
                            .filter(|(about, _)| *about != Some(f))
                            .map(|(_, s)| s[f])
                            .collect();
                        match self.spec.hub {
                            Hub::Subtract => x - mean(&mine),
                            Hub::Divide => x / (1.0 + mean(&mine)),
                            Hub::Surprise => {
                                let scored: Vec<f32> =
                                    mine.iter().copied().filter(|&v| v > 0.0).collect();
                                let against = if scored.len() >= MIN_BACKGROUND {
                                    scored.as_slice()
                                } else {
                                    pooled.as_slice()
                                };
                                surprise(x, against)
                            }
                            Hub::None => x,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Whether file `f` at `path` takes part in the ranking: not a chain link
    /// when chain links are left out, and matching no exclusion.
    pub fn ranks(&self, f: usize, path: &str, chain: &HashSet<usize>) -> bool {
        !(self.spec.exclude_chain && chain.contains(&f))
            && !self.exclude.iter().any(|r| r.is_match(path))
    }

    /// The multiplier for `path`: the first type it matches, else 1.
    pub fn type_weight(&self, path: &str) -> f32 {
        self.types
            .iter()
            .find(|(r, _)| r.is_match(path))
            .map_or(1.0, |(_, w)| *w)
    }

    /// `votes` fused into a score per file of `n` under the formula's law.
    fn fused(&self, votes: &[TokenVote], groups: &[f32; 3], n: usize) -> Vec<f32> {
        match self.spec.fusion {
            Fusion::Additive => self.summed(votes, groups, n),
            Fusion::ContentGated => {
                let mut out = vec![0.0f32; n];
                let mut gate_tally = vec![0.0f32; n];
                for g in 0..FOLD_GROUPS {
                    let mut one = [0.0f32; 3];
                    one[g] = groups[g];
                    if one[g] == 0.0 {
                        continue;
                    }
                    let tally = self.summed(votes, &one, n);
                    if g == 0 {
                        gate_tally.clone_from(&tally);
                    }
                    for (o, t) in out.iter_mut().zip(tally) {
                        *o += t;
                    }
                }
                for (o, g) in out.iter_mut().zip(gate_tally) {
                    if g <= 0.0 {
                        *o = 0.0;
                    }
                }
                out
            }
        }
    }

    /// Each probe token's votes for its best `k` files per group, weighed by
    /// `groups`: file `r` by `z_r × (a_r − a_{k+1})`, `z_r` its agreement's
    /// standing over the token's distribution. The tokens whose summed votes
    /// are in the top `keep` fraction keep them, each raised to its power and
    /// summed per file — production's gate over a token's every vote.
    fn top_k_summed(
        &self,
        ranked: &[[Ranked; FOLD_GROUPS]],
        groups: &[f32; 3],
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let per_token: Vec<Vec<(usize, f32)>> = ranked
            .iter()
            .map(|groups_of| {
                let mut out = Vec::new();
                for (g, r) in groups_of.iter().enumerate() {
                    let w = groups[g];
                    if w == 0.0 || r.top.is_empty() {
                        continue;
                    }
                    let floor = r.top.get(k).map_or(0, |t| t.1);
                    for &(file, a) in r.top.iter().take(k) {
                        let z = ((a as f32 - r.mean) / r.std.max(1e-3)).max(0.0);
                        let v = w * z * a.saturating_sub(floor) as f32;
                        if v > 0.0 {
                            out.push((file, v));
                        }
                    }
                }
                out
            })
            .collect();
        let mags: Vec<f32> = per_token
            .iter()
            .map(|v| v.iter().map(|x| x.1).sum())
            .collect();
        let mut out = vec![0.0f32; n];
        if mags.is_empty() {
            return out;
        }
        let n_keep = ((self.spec.keep * mags.len() as f32).ceil() as usize).clamp(1, mags.len());
        let mut sorted = mags.clone();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let thresh = sorted[n_keep - 1];
        for (votes, m) in per_token.iter().zip(&mags) {
            if *m < thresh {
                continue;
            }
            for &(file, v) in votes {
                if file < n {
                    out[file] += v.powf(self.spec.vote_power);
                }
            }
        }
        out
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

/// Every question of `raw` but `q`.
fn others(raw: &[Vec<f32>], q: usize) -> impl Iterator<Item = &Vec<f32>> {
    raw.iter()
        .enumerate()
        .filter(move |(i, _)| *i != q)
        .map(|(_, r)| r)
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

/// `x` read as tail surprise against `background`'s scoring samples:
/// `1000 × −ln P(background ≥ x) / ln(1 / FULL_HIT_TAIL)`, add-one smoothed,
/// rising one nat per nat of `ln(1 + x)` past the whole background so two
/// scores beyond it still rank by size. A score of no evidence reads zero.
pub fn surprise(x: f32, background: &[f32]) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let n = background.len() as f32;
    let above = background.iter().filter(|&&b| b >= x).count() as f32;
    let tail = (above + 1.0) / (n + 1.0);
    let past = background.iter().copied().fold(0.0f32, f32::max).min(x);
    let beyond = (1.0 + x).ln() - (1.0 + past).ln();
    1000.0 * (-tail.ln() + beyond) / (1.0 / FULL_HIT_TAIL).ln()
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

    fn raw(f: &Formula, content: &[TokenVote], tokens: &[usize]) -> Vec<f32> {
        f.raw(
            &Votes {
                content,
                calibration: &[],
                ranked: None,
            },
            &PATHS,
            tokens,
        )
    }

    /// One token ranked in group 0 only: files 1, 0, 2 at agreements 90, 80,
    /// 70, over a distribution of mean 50 and spread 10.
    fn ranked_one() -> Vec<[Ranked; FOLD_GROUPS]> {
        let empty = Ranked {
            mean: 0.0,
            std: 1.0,
            top: Vec::new(),
        };
        vec![[
            Ranked {
                mean: 50.0,
                std: 10.0,
                top: vec![(1, 90), (0, 80), (2, 70)],
            },
            empty.clone(),
            empty,
        ]]
    }

    /// **Top-k voting gives each of a token's best files a vote** over the
    /// `(k+1)`-th: at k = 2, file 1 by z 4 × (90 − 70) and file 0 by
    /// z 3 × (80 − 70); file 2, the floor, by nothing. At k = 1 the CPU
    /// ranking is not read — the daemon's single vote stands.
    #[test]
    fn top_k_gives_each_best_file_a_vote_over_the_next() {
        let ranked = ranked_one();
        let f = Formula::parse("topk: 2\ncalibration: 0").unwrap();
        let got = f.raw(
            &Votes {
                content: &[],
                calibration: &[],
                ranked: Some(&ranked),
            },
            &PATHS,
            &[1, 1, 1],
        );
        assert_eq!(got, vec![30.0, 80.0, 0.0]);
        let one = Formula::parse("calibration: 0").unwrap();
        let got = one.raw(
            &Votes {
                content: &votes(),
                calibration: &[],
                ranked: Some(&ranked),
            },
            &PATHS,
            &[1, 1, 1],
        );
        assert_eq!(got, vec![8.0, 0.0, 0.0], "k = 1 reads the daemon's votes");
    }

    /// **An empty formula is production's**: every group weighed 1, the top
    /// quarter of tokens kept — here the one 8-vote — and the calibration at
    /// half weight.
    #[test]
    fn an_empty_formula_scores_as_production() {
        let f = Formula::parse("").unwrap();
        assert_eq!((f.spec.keep, f.spec.calibration), (0.25, 0.5));
        assert_eq!(f.spec.fusion, Fusion::Additive);
        let calibration = vec![[vote(2, 6.0), None, None]];
        let got = f.raw(
            &Votes {
                content: &votes(),
                calibration: &calibration,
                ranked: None,
            },
            &PATHS,
            &[10, 10, 10],
        );
        assert_eq!(got, vec![8.0, 0.0, 3.0]);
    }

    /// **Every knob does what it says**: the whole gate keeps all four votes,
    /// a square root flattens them, a length power divides by size, and a
    /// type weight multiplies the files it matches.
    #[test]
    fn each_knob_changes_the_score_as_written() {
        let f = Formula::parse("keep: 1.0\ncalibration: 0\n").unwrap();
        assert_eq!(raw(&f, &votes(), &[1, 1, 1]), vec![8.0, 5.0, 2.0]);

        let f = Formula::parse("keep: 1.0\ncalibration: 0\nvote_power: 0.5\n").unwrap();
        let r = raw(&f, &votes(), &[1, 1, 1]);
        assert!((r[0] - 8f32.sqrt()).abs() < 1e-6 && (r[1] - 3.0).abs() < 1e-6);

        let f = Formula::parse("keep: 1.0\ncalibration: 0\nlength_power: 1.0\n").unwrap();
        assert_eq!(raw(&f, &votes(), &[4, 5, 1]), vec![2.0, 1.0, 2.0]);

        let f = Formula::parse(
            "keep: 1.0\ncalibration: 0\ntypes:\n  - { files: '\\.md$', weight: 0.25 }\n",
        )
        .unwrap();
        assert_eq!(raw(&f, &votes(), &[1, 1, 1]), vec![2.0, 5.0, 2.0]);
    }

    /// **The content gate mutes an identity spike**: file 1 is voted for only
    /// in the identity groups, file 0 in all three; gated, file 1 scores
    /// nothing and file 0 keeps all three groups' votes.
    #[test]
    fn the_content_gate_mutes_an_identity_spike() {
        let spiky = vec![
            [vote(0, 1.0), vote(0, 1.0), vote(0, 1.0)],
            [None, vote(1, 9.0), vote(1, 9.0)],
        ];
        let additive = Formula::parse("keep: 1.0\ncalibration: 0").unwrap();
        assert_eq!(raw(&additive, &spiky, &[1, 1, 1]), vec![3.0, 18.0, 0.0]);
        let gated = Formula::parse("keep: 1.0\ncalibration: 0\nfusion: content_gated").unwrap();
        assert_eq!(raw(&gated, &spiky, &[1, 1, 1]), vec![3.0, 0.0, 0.0]);
    }

    /// **A hub correction reads a file against questions not about it**: a
    /// file that scores the same on every question loses its lead, from the
    /// other questions or from the background probes.
    #[test]
    fn a_hub_correction_reads_a_file_against_the_other_questions() {
        let raw = vec![vec![10.0, 3.0], vec![10.0, 0.0], vec![10.0, 0.0]];
        let f = Formula::parse("hub: subtract\nhub_from: others").unwrap();
        assert_eq!(f.corrected(&raw, &[])[0], vec![0.0, 3.0]);
        let f = Formula::parse("hub: divide\nhub_from: others").unwrap();
        assert_eq!(f.corrected(&raw, &[])[0], vec![10.0 / 11.0, 3.0]);
        // From the background probes: one about file 1 is that file's hit, not
        // its background, so file 1 is read against the other alone.
        let background = vec![
            Sample {
                about: None,
                scores: vec![10.0, 0.0],
            },
            Sample {
                about: Some(1),
                scores: vec![10.0, 8.0],
            },
        ];
        let f = Formula::parse("hub: subtract").unwrap();
        assert_eq!(f.corrected(&raw[..1], &background)[0], vec![0.0, 3.0]);
        assert_eq!(Formula::parse("").unwrap().corrected(&raw, &[]), raw);
    }

    /// **Tail surprise**: a score past every background sample of one file
    /// reads higher than the same score inside another's, zero reads zero,
    /// and a score beyond the whole background keeps rising with its size.
    #[test]
    fn tail_surprise_reads_a_score_against_its_background() {
        let quiet = [1.0; 9];
        let loud = [50.0; 9];
        assert!(surprise(20.0, &quiet) > surprise(20.0, &loud));
        assert_eq!(surprise(0.0, &quiet), 0.0);
        assert!(surprise(200.0, &quiet) > surprise(100.0, &quiet));
        // Inside the background: the tail alone. 9 samples ≥ 20 of 9 reads
        // −ln(10/10) = 0.
        assert_eq!(surprise(20.0, &loud), 0.0);
    }

    /// **A chain link or an excluded path takes no part in the ranking.**
    #[test]
    fn chain_links_and_excluded_paths_are_left_out() {
        let chain: HashSet<usize> = [0].into();
        let f = Formula::parse("exclude_chain: true\nexclude: ['^journals/']").unwrap();
        assert!(!f.ranks(0, "candle/README.md", &chain));
        assert!(!f.ranks(1, "journals/zen/x.yaml", &chain));
        assert!(f.ranks(2, "candle/a.rs", &chain));
        assert!(Formula::parse("")
            .unwrap()
            .ranks(0, "candle/README.md", &chain));
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
