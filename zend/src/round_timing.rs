//! Where a round's time went: queued for the GPU, prefilling, decoding
//! (`docs/zen_coding_challenge_fixes.md` Cause 1).
//!
//! A coding task's wall time was 89–94% the model's, and its call listing
//! could not say which part: 128 rounds of 73 s each, against per-round
//! prefill of seconds and decode rates that came to far less. Each round now
//! records its own split under its turn, in the conversation's metadata
//! ([`key`]), and `timeline_read`'s `calls` view shows it beside the round's
//! calls with the conversation's totals, so a slow task reads as queueing, as
//! prefill or as decode before anything is changed to speed it up.

use std::collections::BTreeMap;
use std::fmt;

/// The prefix of a round's metadata key, [`key`].
pub const PREFIX: &str = "round.";

/// The metadata key a round of turn `turn` records its timing under.
pub fn key(turn: u32) -> String {
    format!("{PREFIX}{turn}")
}

/// One round's time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoundTiming {
    /// From the submit to the turn's end, less its prefill and decode: the
    /// time it waited — for the GPU, behind other sessions, for its
    /// reprojection.
    pub queued_ms: u64,
    /// The tokens its own prefill forwarded.
    pub prefill_tokens: u64,
    pub prefill_ms: u64,
    /// The tokens it generated — its reasoning and its calls together.
    pub generated: u64,
    pub decode_ms: u64,
}

impl RoundTiming {
    /// A round that took `wall_ms` from its submit to its end, of which the
    /// engine spent `prefill_ms` prefilling `prefill_tokens` and `decode_ms`
    /// decoding `generated`.
    pub fn of(
        wall_ms: u64,
        prefill_tokens: u64,
        prefill_ms: u64,
        generated: u64,
        decode_ms: u64,
    ) -> RoundTiming {
        RoundTiming {
            queued_ms: wall_ms.saturating_sub(prefill_ms + decode_ms),
            prefill_tokens,
            prefill_ms,
            generated,
            decode_ms,
        }
    }

    /// As its metadata value: `queued=… prefill=…/…ms decode=…/…ms`.
    pub fn value(&self) -> String {
        format!(
            "queued={} prefill={}/{}ms decode={}/{}ms",
            self.queued_ms, self.prefill_tokens, self.prefill_ms, self.generated, self.decode_ms
        )
    }

    /// A value [`Self::value`] wrote; `None` for one it did not.
    pub fn parse(value: &str) -> Option<RoundTiming> {
        let mut words = value.split_whitespace();
        let queued_ms = words.next()?.strip_prefix("queued=")?.parse().ok()?;
        let pair = |word: Option<&str>, name: &str| -> Option<(u64, u64)> {
            let (n, ms) = word?
                .strip_prefix(name)?
                .strip_suffix("ms")?
                .split_once('/')?;
            Some((n.parse().ok()?, ms.parse().ok()?))
        };
        let (prefill_tokens, prefill_ms) = pair(words.next(), "prefill=")?;
        let (generated, decode_ms) = pair(words.next(), "decode=")?;
        if words.next().is_some() {
            return None;
        }
        Some(RoundTiming {
            queued_ms,
            prefill_tokens,
            prefill_ms,
            generated,
            decode_ms,
        })
    }

    fn add(&mut self, other: &RoundTiming) {
        self.queued_ms += other.queued_ms;
        self.prefill_tokens += other.prefill_tokens;
        self.prefill_ms += other.prefill_ms;
        self.generated += other.generated;
        self.decode_ms += other.decode_ms;
    }
}

/// Milliseconds as seconds, to a tenth.
fn secs(ms: u64) -> String {
    format!("{}.{} s", ms / 1000, (ms % 1000) / 100)
}

impl fmt::Display for RoundTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "queued {}, prefill {} tok in {}, decode {} tok in {}",
            secs(self.queued_ms),
            self.prefill_tokens,
            secs(self.prefill_ms),
            self.generated,
            secs(self.decode_ms)
        )
    }
}

/// Every round's timing a conversation's metadata holds, by turn.
pub fn rounds(custom: &BTreeMap<String, String>) -> BTreeMap<u32, RoundTiming> {
    custom
        .iter()
        .filter_map(|(k, v)| {
            let turn = k.strip_prefix(PREFIX)?.parse().ok()?;
            Some((turn, RoundTiming::parse(v)?))
        })
        .collect()
}

/// The totals of `rounds`, with how many there are.
pub fn total<'a>(rounds: impl IntoIterator<Item = &'a RoundTiming>) -> (usize, RoundTiming) {
    rounds
        .into_iter()
        .fold((0, RoundTiming::default()), |(n, mut sum), r| {
            sum.add(r);
            (n + 1, sum)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The wait is what the engine's own times do not account for**, and a
    /// round whose times exceed its wall clock waited none.
    #[test]
    fn the_wait_is_the_wall_clock_the_engine_did_not_spend() {
        let r = RoundTiming::of(10_000, 512, 1_400, 210, 6_100);
        assert_eq!(r.queued_ms, 2_500);
        assert_eq!(RoundTiming::of(100, 1, 80, 1, 40).queued_ms, 0);
    }

    #[test]
    fn a_round_reads_back_as_it_was_written() {
        let r = RoundTiming::of(10_000, 512, 1_400, 210, 6_100);
        assert_eq!(
            r.value(),
            "queued=2500 prefill=512/1400ms decode=210/6100ms"
        );
        assert_eq!(RoundTiming::parse(&r.value()), Some(r));
        assert_eq!(RoundTiming::parse("queued=x"), None);
        assert_eq!(
            RoundTiming::parse(&format!("{} garbage", r.value())),
            None,
            "a value with more than its three fields is not one this wrote"
        );
        assert_eq!(
            r.to_string(),
            "queued 2.5 s, prefill 512 tok in 1.4 s, decode 210 tok in 6.1 s"
        );
    }

    #[test]
    fn the_rounds_of_a_conversation_are_read_by_turn_and_totalled() {
        let a = RoundTiming::of(10_000, 512, 1_400, 210, 6_100);
        let b = RoundTiming::of(3_000, 100, 300, 50, 1_000);
        let custom = BTreeMap::from([
            (key(4), a.value()),
            (key(12), b.value()),
            ("dial_think".to_string(), "on".to_string()),
            (key(13), "unreadable".to_string()),
        ]);
        let got = rounds(&custom);
        assert_eq!(got, BTreeMap::from([(4, a), (12, b)]));
        let (n, sum) = total(got.values());
        assert_eq!(n, 2);
        assert_eq!(
            sum,
            RoundTiming {
                queued_ms: 4_200,
                prefill_tokens: 612,
                prefill_ms: 1_700,
                generated: 260,
                decode_ms: 7_100
            }
        );
    }
}
