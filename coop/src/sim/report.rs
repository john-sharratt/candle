//! What a simulated run comes to, per turn and over the run: throughput, the
//! wait for a reply's first token, the longest silence inside one, and how
//! evenly the slowdown fell across conversations.

/// One turn as it ran.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRecord {
    pub conversation: usize,
    pub turn: usize,
    /// The member whose lease it ran under.
    pub member: usize,
    /// When it was due: arrival, or the end of the person's pause.
    pub due: f64,
    pub first_token: f64,
    pub end: f64,
    /// The longest gap between two of its tokens.
    pub longest_gap: f64,
    pub tokens: usize,
    /// Its time alone on an idle decode member.
    pub ideal: f64,
    /// It was cut by a member dropping and run again from its start.
    pub restarted: bool,
}

/// The run's figures.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub policy: &'static str,
    pub turns: usize,
    /// Generated tokens over the run's span.
    pub tokens_per_s: f64,
    pub first_token_p50: f64,
    pub first_token_p95: f64,
    pub longest_gap_p95: f64,
    pub longest_gap_max: f64,
    /// Turn time over its ideal, p95 over turns.
    pub slowdown_p95: f64,
    /// Jain's index of the per-conversation mean slowdown: 1 is perfectly
    /// even, `1/n` one conversation taking everything.
    pub fairness: f64,
    /// Turns run under each member's lease.
    pub per_member: Vec<usize>,
    pub link_gb: f64,
    pub restarted: usize,
    /// Turns whose lease went to a different member than the conversation's
    /// previous turn ran on.
    pub moves: usize,
}

/// The `q` quantile of `xs`, nearest rank.
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let i = ((q * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1;
    v[i]
}

/// Jain's fairness index of `xs`.
pub fn jain(xs: &[f64]) -> f64 {
    let sum: f64 = xs.iter().sum();
    let sq: f64 = xs.iter().map(|x| x * x).sum();
    if sq == 0.0 {
        return 1.0;
    }
    sum * sum / (xs.len() as f64 * sq)
}

impl Report {
    pub fn of(
        policy: &'static str,
        records: &[TurnRecord],
        members: usize,
        start: f64,
        link_bytes: f64,
    ) -> Report {
        let end = records.iter().map(|r| r.end).fold(start, f64::max);
        let tokens: usize = records.iter().map(|r| r.tokens).sum();
        let first: Vec<f64> = records.iter().map(|r| r.first_token - r.due).collect();
        let gaps: Vec<f64> = records.iter().map(|r| r.longest_gap).collect();
        let slow: Vec<f64> = records
            .iter()
            .map(|r| (r.end - r.due) / r.ideal.max(1e-9))
            .collect();
        let conversations = records
            .iter()
            .map(|r| r.conversation + 1)
            .max()
            .unwrap_or(0);
        let mut by_conv = vec![(0.0f64, 0usize); conversations];
        for (r, s) in records.iter().zip(&slow) {
            by_conv[r.conversation].0 += s;
            by_conv[r.conversation].1 += 1;
        }
        let means: Vec<f64> = by_conv
            .iter()
            .filter(|(_, n)| *n > 0)
            .map(|(s, n)| s / *n as f64)
            .collect();
        let mut per_member = vec![0; members];
        for r in records {
            per_member[r.member] += 1;
        }
        let mut ordered: Vec<&TurnRecord> = records.iter().collect();
        ordered.sort_by_key(|r| (r.conversation, r.turn));
        let moves = ordered
            .windows(2)
            .filter(|w| w[0].conversation == w[1].conversation && w[0].member != w[1].member)
            .count();
        Report {
            policy,
            turns: records.len(),
            tokens_per_s: tokens as f64 / (end - start).max(1e-9),
            first_token_p50: quantile(&first, 0.5),
            first_token_p95: quantile(&first, 0.95),
            longest_gap_p95: quantile(&gaps, 0.95),
            longest_gap_max: gaps.iter().copied().fold(0.0, f64::max),
            slowdown_p95: quantile(&slow, 0.95),
            fairness: jain(&means),
            per_member,
            link_gb: link_bytes / 1e9,
            restarted: records.iter().filter(|r| r.restarted).count(),
            moves,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_are_nearest_rank() {
        let xs = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(quantile(&xs, 0.5), 3.0);
        assert_eq!(quantile(&xs, 0.95), 5.0);
        assert_eq!(quantile(&xs, 0.0), 1.0);
        assert!(quantile(&[], 0.5).is_nan());
    }

    #[test]
    fn jain_is_one_when_even_and_one_over_n_when_one_takes_all() {
        assert_eq!(jain(&[2.0, 2.0, 2.0, 2.0]), 1.0);
        assert_eq!(jain(&[1.0, 0.0, 0.0, 0.0]), 0.25);
    }

    fn turn(conversation: usize, member: usize, due: f64, end: f64) -> TurnRecord {
        TurnRecord {
            conversation,
            turn: 0,
            member,
            due,
            first_token: due + 1.0,
            end,
            longest_gap: 0.5,
            tokens: 100,
            ideal: 2.0,
            restarted: false,
        }
    }

    /// Throughput is over the run's span; slowdown is per turn against its
    /// ideal; turns count under their member.
    #[test]
    fn a_report_reads_its_turns() {
        let r = Report::of(
            "p",
            &[turn(0, 0, 0.0, 4.0), turn(1, 1, 0.0, 8.0)],
            2,
            0.0,
            2e9,
        );
        assert_eq!(r.tokens_per_s, 25.0);
        assert_eq!(r.first_token_p50, 1.0);
        assert_eq!(r.slowdown_p95, 4.0);
        assert!((r.fairness - 0.9).abs() < 1e-12);
        assert_eq!(r.per_member, vec![1, 1]);
        assert_eq!(r.link_gb, 2.0);
        assert_eq!(r.moves, 0, "two conversations, one turn each");
    }

    /// A conversation whose second turn ran elsewhere moved once.
    #[test]
    fn a_turn_elsewhere_is_a_move() {
        let mut second = turn(0, 1, 5.0, 9.0);
        second.turn = 1;
        let r = Report::of("p", &[turn(0, 0, 0.0, 4.0), second], 2, 0.0, 0.0);
        assert_eq!(r.moves, 1);
    }
}
