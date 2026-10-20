//! A scenario's samples, summed up: how many requests ended which way, where
//! they were served, the spread of the waits a person feels, and the
//! aggregate rate of text across the cluster.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use crate::sample::{Outcome, Sample};

/// One scenario's summary.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub scenario: String,
    pub requests: usize,
    pub answered: usize,
    pub failed: usize,
    pub refused: usize,
    pub cut: usize,
    /// The first failure's message, for the report.
    pub first_failure: Option<String>,
    /// Requests per member, by the name the gateway's `x-served-by` gave.
    pub by_member: BTreeMap<String, usize>,
    pub first_text: Spread,
    pub silence: Spread,
    /// Text chunks per second of wall time, across every request.
    pub text_rate: f64,
    pub wall: Duration,
    /// Of the answered turns with a correct output to check, how many gave it,
    /// and how many were checked.
    pub correct: (usize, usize),
}

/// The 50th and 95th percentiles and the maximum, nearest rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Spread {
    pub p50: Duration,
    pub p95: Duration,
    pub max: Duration,
}

impl Spread {
    pub fn of(mut values: Vec<Duration>) -> Spread {
        if values.is_empty() {
            return Spread::default();
        }
        values.sort();
        let rank = |p: f64| {
            let i = ((p * values.len() as f64).ceil() as usize).max(1) - 1;
            values[i.min(values.len() - 1)]
        };
        Spread {
            p50: rank(0.50),
            p95: rank(0.95),
            max: *values.last().expect("not empty"),
        }
    }
}

impl Row {
    pub fn of(scenario: &str, samples: &[Sample], wall: Duration) -> Row {
        let count = |f: fn(&Outcome) -> bool| samples.iter().filter(|s| f(&s.outcome)).count();
        let mut by_member = BTreeMap::new();
        for s in samples {
            *by_member.entry(s.member.clone()).or_insert(0) += 1;
        }
        let chunks: usize = samples.iter().map(|s| s.text_chunks).sum();
        Row {
            scenario: scenario.to_string(),
            requests: samples.len(),
            answered: count(|o| *o == Outcome::Answered),
            failed: count(|o| matches!(o, Outcome::Failed(_))),
            refused: count(|o| matches!(o, Outcome::Refused(_))),
            cut: count(|o| matches!(o, Outcome::Cut(_))),
            first_failure: samples.iter().find_map(|s| match &s.outcome {
                Outcome::Failed(m) | Outcome::Refused(m) => Some(m.clone()),
                Outcome::Cut(m) => Some(format!("cut: {m}")),
                _ => None,
            }),
            by_member,
            first_text: Spread::of(samples.iter().filter_map(|s| s.first_text).collect()),
            silence: Spread::of(samples.iter().map(|s| s.longest_silence).collect()),
            text_rate: chunks as f64 / wall.as_secs_f64().max(1e-9),
            wall,
            correct: (
                samples.iter().filter(|s| s.rewrite == Some(true)).count(),
                samples.iter().filter(|s| s.rewrite.is_some()).count(),
            ),
        }
    }
}

fn secs(d: Duration) -> String {
    format!("{:.1}", d.as_secs_f64())
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let members: Vec<String> = self
            .by_member
            .iter()
            .map(|(m, n)| format!("{m} {n}"))
            .collect();
        write!(
            f,
            "{:<14} req {:>3}  ok {:>3}  fail {:>2}  refused {:>2}  cut {:>2} | first text p50 {}s p95 {}s max {}s | silence p50 {}s p95 {}s max {}s | {:.1} chunks/s over {}s | {}",
            self.scenario,
            self.requests,
            self.answered,
            self.failed,
            self.refused,
            self.cut,
            secs(self.first_text.p50),
            secs(self.first_text.p95),
            secs(self.first_text.max),
            secs(self.silence.p50),
            secs(self.silence.p95),
            secs(self.silence.max),
            self.text_rate,
            secs(self.wall),
            members.join(", "),
        )?;
        if self.correct.1 > 0 {
            write!(f, " | rewrite ok {}/{}", self.correct.0, self.correct.1)?;
        }
        if let Some(m) = &self.first_failure {
            write!(f, "\n               first failure: {m}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    fn sample(member: &str, outcome: Outcome, first: Option<u64>, silence: u64) -> Sample {
        Sample {
            scenario: "medium".into(),
            client: 0,
            turn: 0,
            member: member.into(),
            outcome,
            first_text: first.map(s),
            total: s(10),
            longest_silence: s(silence),
            text_chunks: 50,
            tools: Vec::new(),
            rewrite: None,
        }
    }

    /// Nearest-rank percentiles: of 1..=20 s, p50 is 10 s and p95 is 19 s.
    #[test]
    fn spreads_are_nearest_rank() {
        let spread = Spread::of((1..=20).rev().map(s).collect());
        assert_eq!(
            spread,
            Spread {
                p50: s(10),
                p95: s(19),
                max: s(20),
            }
        );
        assert_eq!(Spread::of(vec![]), Spread::default());
        assert_eq!(Spread::of(vec![s(3)]).p95, s(3));
    }

    /// Outcomes, members, spreads and the text rate are summed per scenario.
    #[test]
    fn a_row_sums_its_samples() {
        let samples = vec![
            sample("BlackWorld", Outcome::Answered, Some(2), 3),
            sample("BlackWorld", Outcome::Answered, Some(4), 5),
            sample("WhiteWorld", Outcome::Failed("stalled".into()), None, 1),
            sample("WhiteWorld", Outcome::Cut("closed".into()), Some(6), 9),
        ];
        let row = Row::of("medium", &samples, s(20));
        assert_eq!(
            (row.requests, row.answered, row.failed, row.refused, row.cut),
            (4, 2, 1, 0, 1)
        );
        assert_eq!(row.first_failure.as_deref(), Some("stalled"));
        assert_eq!(row.by_member["BlackWorld"], 2);
        assert_eq!(row.by_member["WhiteWorld"], 2);
        assert_eq!(row.first_text.p50, s(4));
        assert_eq!(row.silence.max, s(9));
        assert_eq!(row.text_rate, 10.0);
        assert_eq!(row.correct, (0, 0));
        assert!(!row.to_string().contains("rewrite"));
    }

    /// Turns with a correct output to check are counted, and the row says how
    /// many gave it.
    #[test]
    fn a_row_counts_its_correct_rewrites() {
        let mut samples = vec![
            sample("BlackWorld", Outcome::Answered, Some(2), 3),
            sample("BlackWorld", Outcome::Answered, Some(2), 3),
            sample("WhiteWorld", Outcome::Answered, Some(2), 3),
        ];
        samples[0].rewrite = Some(true);
        samples[1].rewrite = Some(false);
        samples[2].rewrite = Some(true);
        let row = Row::of("story-x16", &samples, s(20));
        assert_eq!(row.correct, (2, 3));
        assert!(row.to_string().contains("| rewrite ok 2/3"));
    }
}
