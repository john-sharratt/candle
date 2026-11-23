//! The battery's analysis (`docs/unbounded_agents.md` §9.12, *Scoring and
//! statistics*, and §6.5): every figure's data and every table's numbers,
//! computed from the answers' lines alone.
//!
//! A question's score in an arm is its recall averaged over the runs that
//! answered it. Intervals are 95% percentile-bootstrap intervals over
//! questions — for a difference, over the per-question paired differences —
//! from [`RESAMPLES`] resamples of a fixed seed, so the figures reproduce
//! exactly. Contrasts are tested by Wilcoxon signed-rank on the paired
//! differences and Holm-corrected across the five.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::arms::{Arm, ARMS, CONTRASTS};
use crate::placement::placed_at;
use crate::run::Line;
use crate::stats::{bootstrap_mean_interval, holm, mean, wilcoxon};

pub const RESAMPLES: usize = 10_000;
pub const SEED: u64 = 0x2026_1123;

/// Each arm's per-question scores: the mean recall of its answered runs.
pub type Scores = BTreeMap<Arm, BTreeMap<String, f64>>;

pub fn scores(lines: &[Line]) -> Scores {
    let mut sums: BTreeMap<Arm, BTreeMap<String, (f64, usize)>> = BTreeMap::new();
    for l in lines.iter().filter(|l| l.error.is_none()) {
        let Some(arm) = Arm::of(&l.arm) else { continue };
        let e = sums
            .entry(arm)
            .or_default()
            .entry(l.id.clone())
            .or_default();
        e.0 += l.recall;
        e.1 += 1;
    }
    sums.into_iter()
        .map(|(arm, qs)| {
            (
                arm,
                qs.into_iter()
                    .map(|(id, (s, n))| (id, s / n as f64))
                    .collect(),
            )
        })
        .collect()
}

/// Figure 9.12-1's row for one arm.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmRow {
    pub arm: Arm,
    pub questions: usize,
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    /// Each run's mean recall over its answered questions.
    pub run_means: Vec<f64>,
    /// Questions at full marks in the per-question score.
    pub full_marks: usize,
}

pub fn arm_rows(lines: &[Line], scores: &Scores) -> Vec<ArmRow> {
    ARMS.iter()
        .filter_map(|&arm| {
            let qs: Vec<f64> = scores.get(&arm)?.values().copied().collect();
            let (lo, hi) = bootstrap_mean_interval(&qs, RESAMPLES, SEED);
            let mut runs: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
            for l in lines
                .iter()
                .filter(|l| l.error.is_none() && l.arm == arm.id())
            {
                runs.entry(l.run).or_default().push(l.recall);
            }
            Some(ArmRow {
                arm,
                questions: qs.len(),
                mean: mean(&qs),
                lo,
                hi,
                run_means: runs.values().map(|r| mean(r)).collect(),
                full_marks: qs.iter().filter(|s| **s >= 1.0).count(),
            })
        })
        .collect()
}

/// The paired per-question differences `a − b` over the questions both
/// arms scored, by id.
pub fn paired(scores: &Scores, a: Arm, b: Arm) -> Vec<(String, f64)> {
    let (Some(sa), Some(sb)) = (scores.get(&a), scores.get(&b)) else {
        return Vec::new();
    };
    sa.iter()
        .filter_map(|(id, x)| sb.get(id).map(|y| (id.clone(), x - y)))
        .collect()
}

/// Figure 9.12-2's row for one contrast.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastRow {
    pub name: &'static str,
    pub arm: Arm,
    pub against: Arm,
    pub questions: usize,
    pub effect: f64,
    pub lo: f64,
    pub hi: f64,
    pub p: f64,
    pub p_holm: f64,
}

pub fn contrasts(scores: &Scores) -> Vec<ContrastRow> {
    let mut rows: Vec<ContrastRow> = CONTRASTS
        .iter()
        .map(|&(name, arm, against)| {
            let d: Vec<f64> = paired(scores, arm, against)
                .into_iter()
                .map(|(_, d)| d)
                .collect();
            let (lo, hi) = if d.is_empty() {
                (f64::NAN, f64::NAN)
            } else {
                bootstrap_mean_interval(&d, RESAMPLES, SEED)
            };
            ContrastRow {
                name,
                arm,
                against,
                questions: d.len(),
                effect: mean(&d),
                lo,
                hi,
                p: wilcoxon(&d).p,
                p_holm: 0.0,
            }
        })
        .collect();
    let adjusted = holm(&rows.iter().map(|r| r.p).collect::<Vec<_>>());
    for (r, p) in rows.iter_mut().zip(adjusted) {
        r.p_holm = p;
    }
    rows
}

/// Figure 9.12-4: the gain `a − b` with its interval, for the questions that
/// name their answer's file and for those that do not.
pub fn gain_by_naming(
    scores: &Scores,
    a: Arm,
    b: Arm,
    names_file: &BTreeMap<String, bool>,
) -> [(bool, usize, f64, f64, f64); 2] {
    let d = paired(scores, a, b);
    [true, false].map(|named| {
        let xs: Vec<f64> = d
            .iter()
            .filter(|(id, _)| names_file.get(id) == Some(&named))
            .map(|(_, x)| *x)
            .collect();
        let (lo, hi) = if xs.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            bootstrap_mean_interval(&xs, RESAMPLES, SEED)
        };
        (named, xs.len(), mean(&xs), lo, hi)
    })
}

/// Figure 9.12-5: for `arm`, at each token position of `grid`, the share of
/// questions whose answer file is placed there — each question's share the
/// mean over its answered runs — with its interval.
pub fn retrieval_curve(lines: &[Line], arm: Arm, grid: &[u32]) -> Vec<(u32, f64, f64, f64)> {
    let mut by_q: BTreeMap<&str, Vec<&Line>> = BTreeMap::new();
    for l in lines
        .iter()
        .filter(|l| l.error.is_none() && l.arm == arm.id())
    {
        by_q.entry(l.id.as_str()).or_default().push(l);
    }
    grid.iter()
        .map(|&t| {
            let shares: Vec<f64> = by_q
                .values()
                .map(|ls| {
                    mean(
                        &ls.iter()
                            .map(|l| placed_at(&l.placement.series, t) as u8 as f64)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let (lo, hi) = if shares.is_empty() {
                (f64::NAN, f64::NAN)
            } else {
                bootstrap_mean_interval(&shares, RESAMPLES, SEED)
            };
            (t, mean(&shares), lo, hi)
        })
        .collect()
}

/// Figure 9.12-6: mean recall of the projecting arms' answers decoded with an
/// answer file placed for their first answer token, and without, each with
/// its interval over answers.
pub fn recall_by_placement(lines: &[Line]) -> [(bool, usize, f64, f64, f64); 2] {
    [true, false].map(|placed| {
        let xs: Vec<f64> = lines
            .iter()
            .filter(|l| l.error.is_none())
            .filter(|l| Arm::of(&l.arm).is_some_and(Arm::projects))
            .filter(|l| l.placement.first_answer == placed)
            .map(|l| l.recall)
            .collect();
        let (lo, hi) = if xs.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            bootstrap_mean_interval(&xs, RESAMPLES, SEED)
        };
        (placed, xs.len(), mean(&xs), lo, hi)
    })
}

/// §6.5's row for one arm: the share of answers with an answer file placed at
/// the opening, at the reasoning's end and for the answer's first token; the
/// mean reprojections inside the think block; and, of the answers whose
/// opening missed the file, the share placed by the first answer token.
#[derive(Debug, Clone, PartialEq)]
pub struct ReachRow {
    pub arm: Arm,
    pub answers: usize,
    pub opening: f64,
    pub think_end: f64,
    pub first_answer: f64,
    pub in_thinking: f64,
    pub brought_in: f64,
}

pub fn reach(lines: &[Line], arm: Arm) -> ReachRow {
    let ls: Vec<&Line> = lines
        .iter()
        .filter(|l| l.error.is_none() && l.arm == arm.id())
        .collect();
    let share =
        |f: &dyn Fn(&Line) -> bool| mean(&ls.iter().map(|l| f(l) as u8 as f64).collect::<Vec<_>>());
    let missed: Vec<&&Line> = ls.iter().filter(|l| !l.placement.opening).collect();
    ReachRow {
        arm,
        answers: ls.len(),
        opening: share(&|l| l.placement.opening),
        think_end: share(&|l| l.placement.think_end),
        first_answer: share(&|l| l.placement.first_answer),
        in_thinking: mean(
            &ls.iter()
                .map(|l| l.placement.in_thinking as f64)
                .collect::<Vec<_>>(),
        ),
        brought_in: mean(
            &missed
                .iter()
                .map(|l| l.placement.first_answer as u8 as f64)
                .collect::<Vec<_>>(),
        ),
    }
}

/// Every numeric leaf of `v`, by its JSON pointer below `prefix`.
pub fn numeric_leaves(v: &Value, prefix: &str, out: &mut BTreeMap<String, f64>) {
    match v {
        Value::Number(n) => {
            if let Some(x) = n.as_f64() {
                out.insert(prefix.to_string(), x);
            }
        }
        Value::Object(m) => {
            for (k, x) in m {
                numeric_leaves(x, &format!("{prefix}/{k}"), out);
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                numeric_leaves(x, &format!("{prefix}/{i}"), out);
            }
        }
        _ => {}
    }
}

/// The telemetry samples as one table: seconds from the first sample, the
/// answers in flight, and every numeric field either read carried, a column
/// each — Figures 9.6-1 to 9.6-3 are drawn from its columns.
pub fn telemetry_table(samples: &[Value]) -> (Vec<String>, Vec<Vec<String>>) {
    let t0 = samples
        .iter()
        .filter_map(|s| s.get("t_ms")?.as_f64())
        .fold(f64::INFINITY, f64::min);
    let rows: Vec<BTreeMap<String, f64>> = samples
        .iter()
        .map(|s| {
            let mut m = BTreeMap::new();
            for key in ["telemetry", "memory"] {
                if let Some(v) = s.get(key) {
                    numeric_leaves(v, key, &mut m);
                }
            }
            m
        })
        .collect();
    let mut columns: Vec<String> = rows.iter().flat_map(|r| r.keys().cloned()).collect();
    columns.sort();
    columns.dedup();
    let mut header = vec!["t_s".to_string(), "in_flight".to_string()];
    header.extend(columns.iter().cloned());
    let body = samples
        .iter()
        .zip(&rows)
        .map(|(s, r)| {
            let mut row = vec![
                format!(
                    "{:.1}",
                    (s.get("t_ms").and_then(Value::as_f64).unwrap_or(t0) - t0) / 1000.0
                ),
                s.get("in_flight")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    .to_string(),
            ];
            row.extend(
                columns
                    .iter()
                    .map(|c| r.get(c).map(|x| x.to_string()).unwrap_or_default()),
            );
            row
        })
        .collect();
    (header, body)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::placement::Placement;

    fn line(arm: Arm, run: u32, id: &str, recall: f64) -> Line {
        Line {
            arm: arm.id().into(),
            run,
            id: id.into(),
            category: "direct".into(),
            names_file: false,
            recall,
            missed: Vec::new(),
            grounding: None,
            ungrounded: Vec::new(),
            completion_tokens: 0,
            think: None,
            placement: Placement::default(),
            secs: 0.0,
            answer: String::new(),
            error: None,
        }
    }

    /// **A question's score is its runs' mean**, and a failed answer counts
    /// for none of them.
    #[test]
    fn a_score_is_the_mean_of_the_answered_runs() {
        let mut failed = line(Arm::C1, 3, "q1", 0.0);
        failed.error = Some("boom".into());
        let lines = vec![
            line(Arm::C1, 1, "q1", 1.0),
            line(Arm::C1, 2, "q1", 0.5),
            failed,
            line(Arm::B1, 1, "q1", 0.25),
        ];
        let s = scores(&lines);
        assert_eq!(s[&Arm::C1]["q1"], 0.75);
        assert_eq!(s[&Arm::B1]["q1"], 0.25);
        assert_eq!(paired(&s, Arm::C1, Arm::B1), vec![("q1".to_string(), 0.5)]);
        let rows = arm_rows(&lines, &s);
        let c1 = rows.iter().find(|r| r.arm == Arm::C1).unwrap();
        assert_eq!(
            (c1.questions, c1.mean, c1.run_means.clone()),
            (1, 0.75, vec![1.0, 0.5])
        );
    }

    /// **Every contrast is a paired difference**, Holm-adjusted across the
    /// five; with C1 one point above B1 on every question the effect is
    /// exactly one and its interval collapses onto it.
    #[test]
    fn a_contrast_is_the_mean_paired_difference() {
        let mut lines = Vec::new();
        for i in 0..20 {
            let id = format!("q{i}");
            lines.push(line(Arm::C1, 1, &id, 1.0));
            lines.push(line(Arm::B1, 1, &id, 0.0));
        }
        let rows = contrasts(&scores(&lines));
        let r = rows
            .iter()
            .find(|r| r.name == "Reasoning-time reprojection")
            .unwrap();
        assert_eq!((r.questions, r.effect, r.lo, r.hi), (20, 1.0, 1.0, 1.0));
        assert!(r.p < 1e-3 && r.p_holm >= r.p);
        let none = rows.iter().find(|r| r.name == "The memory").unwrap();
        assert_eq!(none.questions, 0);
        assert!(none.effect.is_nan());
    }

    #[test]
    fn the_gain_is_split_by_whether_the_question_names_its_file() {
        let mut lines = vec![
            line(Arm::C1, 1, "named", 1.0),
            line(Arm::B1, 1, "named", 1.0),
            line(Arm::C1, 1, "unnamed", 1.0),
            line(Arm::B1, 1, "unnamed", 0.0),
        ];
        lines[0].names_file = true;
        let names: BTreeMap<String, bool> =
            [("named".to_string(), true), ("unnamed".to_string(), false)].into();
        let [named, unnamed] = gain_by_naming(&scores(&lines), Arm::C1, Arm::B1, &names);
        assert_eq!((named.0, named.1, named.2), (true, 1, 0.0));
        assert_eq!((unnamed.0, unnamed.1, unnamed.2), (false, 1, 1.0));
    }

    /// **The curve reads each answer's governing point**: placed from token
    /// 256 in one run of two, so half the question's share from there on.
    #[test]
    fn the_retrieval_curve_follows_each_answers_points() {
        let mut a = line(Arm::C1, 1, "q", 0.0);
        a.placement.series = vec![(0, false), (256, true)];
        let mut b = line(Arm::C1, 2, "q", 0.0);
        b.placement.series = vec![(0, false)];
        let curve = retrieval_curve(&[a, b], Arm::C1, &[0, 128, 256, 512]);
        let shares: Vec<f64> = curve.iter().map(|(_, s, _, _)| *s).collect();
        assert_eq!(shares, vec![0.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn reach_counts_what_the_reasoning_brought_in() {
        let mut found = line(Arm::C1, 1, "q1", 1.0);
        found.placement = Placement {
            opening: false,
            think_end: true,
            first_answer: true,
            in_thinking: 4,
            ..Placement::default()
        };
        let mut missed = line(Arm::C1, 1, "q2", 0.0);
        missed.placement.in_thinking = 2;
        let mut opened = line(Arm::C1, 1, "q3", 1.0);
        opened.placement = Placement {
            opening: true,
            first_answer: true,
            ..Placement::default()
        };
        let r = reach(&[found, missed, opened], Arm::C1);
        assert_eq!(r.answers, 3);
        assert_eq!(r.opening, 1.0 / 3.0);
        assert_eq!(r.first_answer, 2.0 / 3.0);
        assert_eq!(r.in_thinking, 2.0);
        assert_eq!(r.brought_in, 0.5);
        let [placed, unplaced] = recall_by_placement(&[line(Arm::A0, 1, "q", 1.0)]);
        assert_eq!(
            (placed.1, unplaced.1),
            (0, 0),
            "an arm that projects nothing is no evidence"
        );
    }

    #[test]
    fn telemetry_becomes_one_column_per_numeric_field() {
        let samples = vec![
            json!({"t_ms": 1000, "in_flight": 3, "telemetry": {"mix": {"q4": 0.5}}, "memory": {"vram": [7]}}),
            json!({"t_ms": 6000, "in_flight": 16, "telemetry": {"mix": {"q4": 0.6, "q8": 0.1}}, "memory": {"error": "x"}}),
        ];
        let (header, rows) = telemetry_table(&samples);
        assert_eq!(
            header,
            vec![
                "t_s",
                "in_flight",
                "memory/vram/0",
                "telemetry/mix/q4",
                "telemetry/mix/q8"
            ]
        );
        assert_eq!(rows[0], vec!["0.0", "3", "7", "0.5", ""]);
        assert_eq!(rows[1], vec!["5.0", "16", "", "0.6", "0.1"]);
    }
}
