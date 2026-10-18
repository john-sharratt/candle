//! One request, measured: when its stream's events arrived relative to the
//! request, and how it ended. What a person waiting on the reply feels is the
//! time to the first text and the longest stretch in which nothing at all
//! arrived — a stall — which [`Sample::of`] reads off the arrival times.

use std::fmt;
use std::time::Duration;

use crate::sse::Item;

/// How a request ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The stream ended with `[DONE]` and no failure.
    Answered,
    /// The turn reported a failure in-band.
    Failed(String),
    /// A status other than 200, or a transport error.
    Refused(String),
    /// The stream ended without `[DONE]`: how it ended — the run's deadline,
    /// the server closing it, or a transport error.
    Cut(String),
}

/// One request's measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub scenario: String,
    /// The client that sent it, and which of its turns it was (from 0); the
    /// run sets them.
    pub client: usize,
    pub turn: usize,
    pub member: String,
    pub outcome: Outcome,
    /// From sending the request to the first text.
    pub first_text: Option<Duration>,
    /// From sending the request to the end of the stream.
    pub total: Duration,
    /// The longest time in which no event arrived, the wait for the first
    /// included.
    pub longest_silence: Duration,
    /// Chunks that carried text: about one per generated token.
    pub text_chunks: usize,
    /// The tools the turn ran, in the order it started them.
    pub tools: Vec<String>,
}

impl Sample {
    /// Read `events`, each with its arrival time since the request was sent,
    /// the time the stream ended, and how it ended, which is the cut's reason
    /// when no `[DONE]` came.
    pub fn of(
        scenario: &str,
        member: &str,
        events: &[(Duration, Item)],
        ended: Duration,
        end: &str,
    ) -> Sample {
        let mut first_text = None;
        let mut text_chunks = 0;
        let mut outcome = Outcome::Cut(end.to_string());
        let mut last = Duration::ZERO;
        let mut longest = Duration::ZERO;
        let mut tools = Vec::new();
        for (at, item) in events {
            longest = longest.max(at.saturating_sub(last));
            last = *at;
            if item.has_text() {
                text_chunks += 1;
                first_text.get_or_insert(*at);
            }
            match item {
                Item::Tools(names) => tools.extend(names.iter().cloned()),
                Item::Failed(m) => outcome = Outcome::Failed(m.clone()),
                Item::Done if matches!(outcome, Outcome::Cut(_)) => outcome = Outcome::Answered,
                _ => {}
            }
        }
        Sample {
            scenario: scenario.to_string(),
            client: 0,
            turn: 0,
            member: member.to_string(),
            outcome,
            first_text,
            total: ended,
            longest_silence: longest,
            text_chunks,
            tools,
        }
    }

    /// A request that never streamed.
    pub fn refused(scenario: &str, member: &str, why: String, ended: Duration) -> Sample {
        Sample {
            scenario: scenario.to_string(),
            client: 0,
            turn: 0,
            member: member.to_string(),
            outcome: Outcome::Refused(why),
            first_text: None,
            total: ended,
            longest_silence: ended,
            text_chunks: 0,
            tools: Vec::new(),
        }
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let first = self
            .first_text
            .map_or("-".to_string(), |d| format!("{:.1}s", d.as_secs_f64()));
        write!(
            f,
            "client {} turn {} on {}: {:?}, first text {first}, longest silence {:.1}s, total {:.1}s, {} text chunks, tools [{}]",
            self.client,
            self.turn,
            self.member,
            self.outcome,
            self.longest_silence.as_secs_f64(),
            self.total.as_secs_f64(),
            self.text_chunks,
            self.tools.join(" "),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// The first text, the longest silence — including the wait before any
    /// event — and the text chunks are read off the arrival times.
    fn text(content: &str, reasoning: usize) -> Item {
        Item::Text {
            content: content.into(),
            reasoning,
        }
    }

    #[test]
    fn a_stream_is_measured_from_its_arrivals() {
        let events = vec![
            (ms(400), Item::Progress("prefill".into())),
            (ms(1_000), Item::Tools(vec!["file_read".into()])),
            (ms(2_400), Item::Progress("prefill".into())),
            (ms(2_500), text("", 4)),
            (ms(2_550), text("", 0)),
            (ms(2_600), text("Once", 0)),
            (ms(2_610), Item::Done),
        ];
        let s = Sample::of("light", "BlackWorld", &events, ms(2_620), "closed");
        assert_eq!(s.outcome, Outcome::Answered);
        assert_eq!(s.first_text, Some(ms(2_500)));
        assert_eq!(s.longest_silence, ms(1_400));
        assert_eq!(s.text_chunks, 2);
        assert_eq!(s.tools, vec!["file_read".to_string()]);
        assert_eq!(s.total, ms(2_620));
    }

    /// An in-band failure is a failure even though `[DONE]` follows; a
    /// stream with no `[DONE]` was cut.
    #[test]
    fn a_failure_and_a_cut_are_told_apart() {
        let failed = Sample::of(
            "heavy",
            "WhiteWorld",
            &[
                (ms(10), Item::Failed("stalled".into())),
                (ms(11), Item::Done),
            ],
            ms(12),
            "closed",
        );
        assert_eq!(failed.outcome, Outcome::Failed("stalled".into()));
        let cut = Sample::of(
            "heavy",
            "WhiteWorld",
            &[(ms(10), text("a", 0))],
            ms(50),
            "closed",
        );
        assert_eq!(cut.outcome, Outcome::Cut("closed".into()));
    }
}
