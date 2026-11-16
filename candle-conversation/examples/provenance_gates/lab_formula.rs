//! The scoring lab's formula: how a probe's votes become a score per file.
//!
//! Read from a YAML file the lab watches, so changing a weight, a gate or a
//! correction re-scores every question at once, with no daemon restarted and
//! no harness rebuilt. Every field has production's value as its default, so
//! an empty file scores as today's belief scan does.

use std::collections::{HashMap, HashSet};
use std::iter::once;

use regex::Regex;
use serde::Deserialize;

use candle_conversation::provenance::window_score::{gate, Ranked, Vote};
use candle_conversation::provenance::FOLD_GROUPS;

use crate::lab_runs::{run_lengths, RunOrder};

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
    /// Which of the probe's tokens are scored: the question, the decoded
    /// answer, or both.
    pub part: Part,
    /// How the fold groups' votes combine.
    pub fusion: Fusion,
    /// Files each probe token votes for in each group of the content
    /// competition: its best `topk`, file `r` by `z_r × (a_r − a_{topk+1})`
    /// — 1 is production's single `z × margin` vote. Above 1 the content is
    /// scored from the lab's CPU ranking ([`Votes::ranked`]), additively.
    pub topk: usize,
    /// Under `topk` above 1: how a token's vote is split among its best
    /// files.
    pub vote_law: VoteLaw,
    /// Under `topk` above 1: how much each probe token counts.
    pub token_weight: TokenWeight,
    /// Under `topk` above 1: how a file's kept votes combine.
    pub aggregate: Aggregate,
    /// The content competition's fold-group weights.
    pub groups: [f32; 3],
    /// Under `topk` above 1, per fold group: a vote in a token run of length
    /// `L` ([`crate::lab_runs`]) is multiplied by `1 + run_bonus × (L − 1)`.
    pub run_bonus: [f32; 3],
    /// Under `topk` above 1, per fold group: a vote in a run shorter than
    /// this is dropped — 1 keeps every vote.
    pub run_min: [usize; 3],
    /// How far apart, in gallery tokens, two neighbouring probe tokens'
    /// matches in one file may lie and still link.
    pub run_slack: u32,
    /// Which way a run's matches may step through the file.
    pub run_order: RunOrder,
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
    /// How the content and calibration scores combine.
    pub blend: Blend,
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
    /// Compare each file first against files of its own kind — its
    /// extension — so a kind that scores on everything starts no higher than
    /// one that scores on little, with no weight set by hand.
    pub type_norm: TypeNorm,
}

/// How a token's vote is split among its best files, `a_r` the agreement of
/// the file at rank `r`, over the token's distribution of mean `μ` and
/// spread `σ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteLaw {
    /// `z_r × (a_r − a_{k+1})` — production's `z × margin` at k = 1.
    ZMargin,
    /// `z_r` alone.
    ZOnly,
    /// `1 / (r + 1)`: by rank, whatever the agreements.
    ReciprocalRank,
    /// `exp((a_r − a_0) / σ)`: a near-tie shares the vote, a clear winner
    /// keeps it.
    Softmax,
    /// One a file.
    Count,
}

/// How much a probe token counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenWeight {
    /// Every token alike.
    None,
    /// By how far its best match stands above its distribution,
    /// `(a_0 − μ) / σ` — a token every file matches alike counts little.
    TopZ,
    /// By how far its best match stands above its k-th, `(a_0 − a_k) / σ`
    /// — a token that cannot tell its best files apart counts little.
    Spread,
}

/// How a file's kept votes combine into its score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregate {
    /// The sum of its votes, each raised to `vote_power`.
    Sum,
    /// The sum of `ln(1 + v)`: concave, so many votes outweigh one loud one.
    LogSum,
    /// How many distinct probe tokens voted for it.
    Tokens,
    /// That count times its largest vote.
    TokensTimesBest,
}

/// How a file is read against files of its own kind (its extension).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeNorm {
    /// Not at all.
    None,
    /// Divided by the mean of its kind's scoring files on this question.
    QuestionMean,
    /// As a z-score among its kind's scoring files on this question.
    QuestionZ,
    /// Divided by its kind's mean score on the background probes — how loud
    /// that kind is on questions that are not about it.
    BackgroundMean,
}

impl Default for FormulaSpec {
    fn default() -> Self {
        Self {
            budget_files: 10,
            part: Part::Whole,
            fusion: Fusion::Additive,
            topk: 1,
            vote_law: VoteLaw::ZMargin,
            token_weight: TokenWeight::None,
            aggregate: Aggregate::Sum,
            groups: [1.0, 1.0, 1.0],
            run_bonus: [0.0; 3],
            run_min: [1; 3],
            run_slack: 1,
            run_order: RunOrder::Forward,
            keep: 0.25,
            vote_power: 1.0,
            length_power: 0.0,
            calibration: 0.5,
            calibration_groups: [1.0, 1.0, 1.0],
            blend: Blend::Sum,
            hub: Hub::None,
            hub_from: HubFrom::Background,
            exclude_chain: false,
            exclude: Vec::new(),
            types: Vec::new(),
            type_norm: TypeNorm::None,
        }
    }
}

/// How the content and calibration competitions' scores combine, the
/// calibration weighed by [`FormulaSpec::calibration`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Blend {
    /// Raw scores summed.
    Sum,
    /// Each competition as a z-score among the files it scores, then summed:
    /// neither wins by being on a louder scale.
    ZSum,
    /// Reciprocal-rank fusion, `1 / (60 + rank)` in each competition that
    /// scores the file: only order counts.
    Rrf,
    /// The larger of the two z-scores: a file either competition is sure of
    /// ranks high.
    ZMax,
}

/// `content` and `calibration` scores combined as `blend` says, the
/// calibration weighed by `weight`. A file neither competition scores stays
/// at 0.
fn blended(content: &[f32], calibration: &[f32], weight: f32, blend: Blend) -> Vec<f32> {
    match blend {
        Blend::Sum => content
            .iter()
            .zip(calibration)
            .map(|(c, k)| c + weight * k)
            .collect(),
        Blend::ZSum | Blend::ZMax => {
            let (zc, zk) = (scoring_z(content), scoring_z(calibration));
            zc.iter()
                .zip(&zk)
                .map(|(&c, &k)| {
                    let k = k.map(|k| weight * k);
                    let v = match (c, k, blend) {
                        (None, None, _) => return 0.0,
                        (Some(c), Some(k), Blend::ZMax) => c.max(k),
                        (c, k, _) => c.unwrap_or(0.0) + k.unwrap_or(0.0),
                    };
                    (v + Z_FLOOR).max(f32::MIN_POSITIVE)
                })
                .collect()
        }
        Blend::Rrf => {
            let (rc, rk) = (ranks(content), ranks(calibration));
            rc.iter()
                .zip(&rk)
                .map(|(c, k)| {
                    c.map_or(0.0, |r| 1.0 / (RRF_K + r as f32))
                        + weight * k.map_or(0.0, |r| 1.0 / (RRF_K + r as f32))
                })
                .collect()
        }
    }
}

/// Reciprocal-rank fusion's damping: the conventional 60.
const RRF_K: f32 = 60.0;

/// Added to a blended z-score so a scored file stays above the unscored
/// files' 0: a z-score below the mean is still a vote.
const Z_FLOOR: f32 = 10.0;

/// Each positive score's z-score among the positive scores; `None` for a
/// file the competition gave nothing.
fn scoring_z(scores: &[f32]) -> Vec<Option<f32>> {
    let pos: Vec<f32> = scores.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        return vec![None; scores.len()];
    }
    let mean = pos.iter().sum::<f32>() / pos.len() as f32;
    let var = pos.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / pos.len() as f32;
    let std = var.sqrt().max(f32::EPSILON);
    scores
        .iter()
        .map(|&v| (v > 0.0).then(|| (v - mean) / std))
        .collect()
}

/// Each positive score's rank among the positive scores, best 0, ties by
/// file; `None` for a file the competition gave nothing.
fn ranks(scores: &[f32]) -> Vec<Option<usize>> {
    let mut order: Vec<usize> = (0..scores.len()).filter(|&f| scores[f] > 0.0).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut out = vec![None; scores.len()];
    for (r, f) in order.into_iter().enumerate() {
        out[f] = Some(r);
    }
    out
}

/// A path's kind: its extension, or the empty kind for a path with none.
pub fn kind_of(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.rsplit_once('.').map_or("", |(_, ext)| ext)
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
    /// Probe tokens that are the prefilled question, the rest being the
    /// decoded answer; `None` for a probe with no answer (a background
    /// probe), which every [`Part`] reads whole.
    pub question: Option<usize>,
}

/// Which of a probe's tokens a formula scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// The prefilled question: what a turn's opening projection reads.
    Question,
    /// The decoded answer alone.
    Thinking,
    /// Both: what a turn reprojects on once it has thought.
    Whole,
}

impl Part {
    /// The token range of a probe of `len` tokens, `question` of them the
    /// question's, this part covers.
    fn range(self, question: Option<usize>, len: usize) -> std::ops::Range<usize> {
        match (self, question) {
            (_, None) | (Part::Whole, _) => 0..len,
            (Part::Question, Some(q)) => 0..q.min(len),
            (Part::Thinking, Some(q)) => q.min(len)..len,
        }
    }
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
        let part = |len: usize| s.part.range(votes.question, len);
        let content = &votes.content[part(votes.content.len())];
        let calibration = &votes.calibration[part(votes.calibration.len())];
        let ranked = votes.ranked.map(|r| &r[part(r.len())]);
        let mut out = match (s.topk, ranked) {
            (k, Some(ranked)) if k > 1 => self.top_k_summed(ranked, &s.groups, k, n),
            _ => self.fused(content, &s.groups, n),
        };
        if s.calibration != 0.0 {
            // The calibration keeps its single vote: its token matches one
            // file's question cleanly, and spreading that vote over the next
            // files dilutes it (MRR 0.53 → 0.40 on the battery).
            let cal = self.fused(calibration, &s.calibration_groups, n);
            out = blended(&out, &cal, s.calibration, s.blend);
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

    /// `scored` read against each file's kind as [`FormulaSpec::type_norm`]
    /// says: per question, over the kind's scoring files there; or over the
    /// kind's scores on the `background` probes. A kind with nothing to be
    /// read against leaves its files as they are; a score of no evidence
    /// stays zero.
    pub fn kind_normalized(
        &self,
        scored: &[Vec<f32>],
        background: &[Sample],
        paths: &[&str],
    ) -> Vec<Vec<f32>> {
        let kinds: Vec<&str> = paths.iter().map(|p| kind_of(p)).collect();
        let stats = |rows: &mut dyn Iterator<Item = &Vec<f32>>| {
            let mut by_kind: HashMap<&str, (f32, f32, usize)> = HashMap::new();
            for row in rows {
                for (f, &x) in row.iter().enumerate() {
                    if x > 0.0 {
                        let e = by_kind.entry(kinds[f]).or_default();
                        e.0 += x;
                        e.1 += x * x;
                        e.2 += 1;
                    }
                }
            }
            by_kind
        };
        let background_stats = match self.spec.type_norm {
            TypeNorm::BackgroundMean => stats(&mut background.iter().map(|s| &s.scores)),
            _ => HashMap::new(),
        };
        scored
            .iter()
            .map(|row| {
                let per_question = match self.spec.type_norm {
                    TypeNorm::QuestionMean | TypeNorm::QuestionZ => stats(&mut once(row)),
                    _ => HashMap::new(),
                };
                row.iter()
                    .enumerate()
                    .map(|(f, &x)| {
                        if x <= 0.0 {
                            // Below every scoring file, whatever its z.
                            return if self.spec.type_norm == TypeNorm::QuestionZ {
                                f32::MIN
                            } else {
                                x
                            };
                        }
                        let k = kinds[f];
                        match self.spec.type_norm {
                            TypeNorm::None => x,
                            TypeNorm::QuestionMean => match per_question.get(k) {
                                Some(&(sum, _, n)) => x / (sum / n as f32),
                                None => x,
                            },
                            // A kind's lone scoring file is its kind's
                            // average: z 0.
                            TypeNorm::QuestionZ => match per_question.get(k) {
                                Some(&(sum, sq, n)) if n > 1 => {
                                    let mean = sum / n as f32;
                                    let var = (sq / n as f32 - mean * mean).max(0.0);
                                    (x - mean) / var.sqrt().max(f32::EPSILON)
                                }
                                _ => 0.0,
                            },
                            TypeNorm::BackgroundMean => match background_stats.get(k) {
                                Some(&(sum, _, n)) => x / (sum / n as f32),
                                None => x,
                            },
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
    /// `groups`, under the formula's [`VoteLaw`] and [`TokenWeight`]. The
    /// tokens whose summed votes are in the top `keep` fraction keep them —
    /// production's gate over a token's every vote — and each file's kept
    /// votes are combined by the formula's [`Aggregate`].
    fn top_k_summed(
        &self,
        ranked: &[[Ranked; FOLD_GROUPS]],
        groups: &[f32; 3],
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let s = &self.spec;
        let runs: [Option<Vec<Vec<usize>>>; FOLD_GROUPS] = std::array::from_fn(|g| {
            (s.run_bonus[g] != 0.0 || s.run_min[g] > 1)
                .then(|| run_lengths(ranked, g, k, s.run_slack, s.run_order))
        });
        let per_token: Vec<Vec<(usize, f32)>> = ranked
            .iter()
            .enumerate()
            .map(|(ti, groups_of)| {
                let mut out = Vec::new();
                for (g, r) in groups_of.iter().enumerate() {
                    let w = groups[g];
                    if w == 0.0 || r.top.is_empty() {
                        continue;
                    }
                    let token = self.token_weight(r, k);
                    for (rank, v) in self.law_votes(r, k) {
                        let run = runs[g].as_ref().map_or(1.0, |lens| {
                            let len = lens[ti][rank];
                            if len < s.run_min[g] {
                                0.0
                            } else {
                                1.0 + s.run_bonus[g] * (len - 1) as f32
                            }
                        });
                        let v = w * token * v * run;
                        if v > 0.0 {
                            out.push((r.top[rank].0, v));
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
        // Per file: the sum of its raised votes, the sum of `ln(1 + v)`, its
        // largest vote, and how many distinct kept tokens voted for it.
        let mut sum = vec![0.0f32; n];
        let mut logs = vec![0.0f32; n];
        let mut best = vec![0.0f32; n];
        let mut tokens = vec![0u32; n];
        for (votes, m) in per_token.iter().zip(&mags) {
            if *m < thresh {
                continue;
            }
            let mut seen: Vec<usize> = Vec::new();
            for &(file, v) in votes {
                if file >= n {
                    continue;
                }
                sum[file] += v.powf(self.spec.vote_power);
                logs[file] += v.ln_1p();
                best[file] = best[file].max(v);
                if !seen.contains(&file) {
                    seen.push(file);
                    tokens[file] += 1;
                }
            }
        }
        for f in 0..n {
            out[f] = match self.spec.aggregate {
                Aggregate::Sum => sum[f],
                Aggregate::LogSum => logs[f],
                // Ties between equal counts broken by the votes' sum, which
                // no count can reach: it is below one.
                Aggregate::Tokens => {
                    tokens[f] as f32 + sum[f] / (1.0 + sum.iter().copied().fold(0.0, f32::max))
                }
                Aggregate::TokensTimesBest => tokens[f] as f32 * best[f],
            };
        }
        out
    }

    /// Each of `r`'s best `k` files' vote under the formula's [`VoteLaw`], by
    /// its rank in `r.top`.
    fn law_votes(&self, r: &Ranked, k: usize) -> Vec<(usize, f32)> {
        let floor = r.top.get(k).map_or(0, |t| t.1);
        let std = r.std.max(1e-3);
        let top = r.top[0].1 as f32;
        r.top
            .iter()
            .take(k)
            .enumerate()
            .map(|(rank, &(_, a, _))| {
                let z = ((a as f32 - r.mean) / std).max(0.0);
                let v = match self.spec.vote_law {
                    VoteLaw::ZMargin => z * a.saturating_sub(floor) as f32,
                    VoteLaw::ZOnly => z,
                    VoteLaw::ReciprocalRank => 1.0 / (rank + 1) as f32,
                    VoteLaw::Softmax => ((a as f32 - top) / std).exp(),
                    VoteLaw::Count => 1.0,
                };
                (rank, v)
            })
            .collect()
    }

    /// How much a token's ranking `r` counts, under the formula's
    /// [`TokenWeight`].
    fn token_weight(&self, r: &Ranked, k: usize) -> f32 {
        let std = r.std.max(1e-3);
        let top = r.top[0].1 as f32;
        match self.spec.token_weight {
            TokenWeight::None => 1.0,
            TokenWeight::TopZ => ((top - r.mean) / std).max(0.0),
            TokenWeight::Spread => {
                let floor = r.top.get(k).map_or(0, |t| t.1) as f32;
                ((top - floor) / std).max(0.0)
            }
        }
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

    /// **The two competitions blend by sum, by z-score, by rank or by the
    /// surer of the two**, and a file neither scores stays at 0.
    #[test]
    fn content_and_calibration_blend_by_scale_free_laws() {
        let content = [4.0, 2.0, 0.0, 0.0];
        let calibration = [0.0, 1.0, 3.0, 0.0];
        assert_eq!(
            blended(&content, &calibration, 0.5, Blend::Sum),
            vec![4.0, 2.5, 1.5, 0.0]
        );
        // Content z: ±1 over {4, 2}; calibration z: ∓1 over {1, 3}.
        assert_eq!(
            blended(&content, &calibration, 1.0, Blend::ZSum),
            vec![11.0, 8.0, 11.0, 0.0]
        );
        assert_eq!(
            blended(&content, &calibration, 1.0, Blend::ZMax),
            vec![11.0, 9.0, 11.0, 0.0]
        );
        let rrf = blended(&content, &calibration, 1.0, Blend::Rrf);
        assert_eq!(
            rrf,
            vec![1.0 / 60.0, 1.0 / 61.0 + 1.0 / 61.0, 1.0 / 60.0, 0.0]
        );
    }

    fn raw(f: &Formula, content: &[TokenVote], tokens: &[usize]) -> Vec<f32> {
        f.raw(
            &Votes {
                content,
                calibration: &[],
                ranked: None,
                question: None,
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
                top: vec![(1, 90, 0), (0, 80, 0), (2, 70, 0)],
            },
            empty.clone(),
            empty,
        ]]
    }

    /// **A part is the question's tokens, the answer's, or both**, and a
    /// probe with no answer is read whole by every part.
    #[test]
    fn a_part_cuts_the_probe_at_its_question() {
        assert_eq!(Part::Question.range(Some(3), 10), 0..3);
        assert_eq!(Part::Thinking.range(Some(3), 10), 3..10);
        assert_eq!(Part::Whole.range(Some(3), 10), 0..10);
        assert_eq!(Part::Thinking.range(None, 10), 0..10);
        assert_eq!(Part::Thinking.range(Some(12), 10), 10..10);
    }

    /// **A vote in a token run is multiplied by its bonus, and a run too
    /// short is dropped.** Two tokens each rank file 1 (gallery tokens 10,
    /// 11: one run of two) and file 0 (tokens 30, 70: two runs of one), at
    /// z 4 × (90 − 70) = 80 and z 3 × (80 − 70) = 30, over the floor file 2.
    #[test]
    fn token_runs_boost_or_gate_a_vote() {
        let token = |a1: u32, a0: u32| Ranked {
            mean: 50.0,
            std: 10.0,
            top: vec![(1, 90, a1), (0, 80, a0), (2, 70, 0)],
        };
        let empty = Ranked {
            mean: 0.0,
            std: 1.0,
            top: Vec::new(),
        };
        let ranked = vec![
            [token(10, 30), empty.clone(), empty.clone()],
            [token(11, 70), empty.clone(), empty],
        ];
        let score = |spec: &str| {
            Formula::parse(spec).unwrap().raw(
                &Votes {
                    content: &[],
                    calibration: &[],
                    ranked: Some(&ranked),
                    question: None,
                },
                &PATHS,
                &[1, 1, 1],
            )
        };
        let base = "topk: 2\ncalibration: 0\nkeep: 1.0\n";
        assert_eq!(score(base), vec![60.0, 160.0, 0.0]);
        assert_eq!(
            score(&format!("{base}run_bonus: [1.0, 0, 0]")),
            vec![60.0, 320.0, 0.0]
        );
        assert_eq!(
            score(&format!("{base}run_min: [2, 1, 1]")),
            vec![0.0, 160.0, 0.0]
        );
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
                question: None,
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
                question: None,
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
                question: None,
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

    /// **Each vote law, token weight and aggregate does what it says**, on
    /// one token ranking files 1, 0, 2 at 90, 80, 70 over mean 50, spread 10.
    #[test]
    fn each_vote_law_token_weight_and_aggregate_does_what_it_says() {
        let ranked = ranked_one();
        let score = |yaml: &str| {
            Formula::parse(&format!("topk: 2\ncalibration: 0\n{yaml}"))
                .unwrap()
                .raw(
                    &Votes {
                        content: &[],
                        calibration: &[],
                        ranked: Some(&ranked),
                        question: None,
                    },
                    &PATHS,
                    &[1, 1, 1],
                )
        };
        assert_eq!(score("vote_law: reciprocal_rank"), vec![0.5, 1.0, 0.0]);
        assert_eq!(score("vote_law: count"), vec![1.0, 1.0, 0.0]);
        assert_eq!(score("vote_law: z_only"), vec![3.0, 4.0, 0.0]);
        let soft = score("vote_law: softmax");
        assert!((soft[0] - (-1.0f32).exp()).abs() < 1e-6 && soft[1] == 1.0);
        // A token standing 4 spreads above its mean counts 4; its best two
        // files are 2 spreads apart from the third.
        assert_eq!(
            score("vote_law: count\ntoken_weight: top_z"),
            vec![4.0, 4.0, 0.0]
        );
        assert_eq!(
            score("vote_law: count\ntoken_weight: spread"),
            vec![2.0, 2.0, 0.0]
        );
        // One token: each file it voted for counts one token, ties broken by
        // the votes' sum below one.
        let tokens = score("aggregate: tokens");
        assert!(tokens[1] > tokens[0] && tokens[0] > 1.0 && tokens[1] < 2.0);
        assert_eq!(score("aggregate: tokens_times_best"), vec![30.0, 80.0, 0.0]);
        let logs = score("aggregate: log_sum");
        assert!((logs[1] - 81f32.ln()).abs() < 1e-5);
    }

    /// **A kind is a path's extension**; a path with none is the empty kind.
    #[test]
    fn a_kind_is_the_extension() {
        assert_eq!(kind_of("candle/docs/x.md"), "md");
        assert_eq!(kind_of("candle/zend/src/lib.rs"), "rs");
        assert_eq!(kind_of("candle/Makefile"), "");
        assert_eq!(kind_of("candle/.github/x"), "");
    }

    /// **Read against its kind, a loud kind starts no higher**: documents
    /// scoring 10 and 20 and code 2 and 4 on one question — divided by their
    /// kind's mean, each pair reads 2/3 and 4/3; as z-scores −1 and 1; a file
    /// of no evidence stays 0, or falls below all under z.
    #[test]
    fn a_kind_is_read_against_itself() {
        let paths = ["a.md", "b.md", "c.rs", "d.rs", "e.rs"];
        let scored = vec![vec![10.0, 20.0, 2.0, 4.0, 0.0]];
        let mean = Formula::parse("type_norm: question_mean").unwrap();
        let got = mean.kind_normalized(&scored, &[], &paths);
        let want = [2.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 0.0];
        for (g, w) in got[0].iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
        let z = Formula::parse("type_norm: question_z").unwrap();
        assert_eq!(
            z.kind_normalized(&scored, &[], &paths)[0],
            vec![-1.0, 1.0, -1.0, 1.0, f32::MIN]
        );
        // Against the background: documents average 30 there, code 3.
        let background = vec![Sample {
            about: None,
            scores: vec![30.0, 30.0, 3.0, 3.0, 3.0],
        }];
        let bg = Formula::parse("type_norm: background_mean").unwrap();
        let got = bg.kind_normalized(&scored, &background, &paths);
        assert_eq!(got[0][1], 20.0 / 30.0);
        assert_eq!(got[0][3], 4.0 / 3.0);
        assert_eq!(
            Formula::parse("")
                .unwrap()
                .kind_normalized(&scored, &[], &paths),
            scored
        );
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
