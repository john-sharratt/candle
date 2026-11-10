//! What a reload found missing from the remote turns it holds as pending.
//!
//! A remote turn is adopted only once its `Tokens` and every `Chunk` of its
//! block range are here ([`super::turn_is_complete`]); until then it is a gap.
//! A count of gaps says how many are waiting, not on what, and the three
//! answers mean different faults:
//!
//! - **no tokens** — the turn's own records never arrived;
//! - **no chunks** — its K/V was never written where replication carries it,
//!   or every chunk was dropped on arrival;
//! - **some chunks** — part of its K/V arrived, or the layer count a turn is
//!   judged against differs from the one it was written with.
//!
//! So the reload reports each, with a few turns to look up by hand.

/// How many sample turns a census keeps.
pub const SAMPLES: usize = 4;

/// One pending turn as the census saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
    pub timeline: u64,
    pub turn: u32,
    pub tokens: bool,
    pub chunks_held: usize,
    pub chunks_wanted: usize,
}

/// The pending turns a reload found, by what each lacks.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PendingCensus {
    /// Turns whose `Tokens` record is missing.
    pub no_tokens: usize,
    /// Turns holding tokens but no chunk at all.
    pub no_chunks: usize,
    /// Turns holding tokens and some chunks, short of the count wanted.
    pub some_chunks: usize,
    /// Turns whose records are all here but could not be read.
    pub unreadable: usize,
    /// The first few turns noted, as found.
    pub samples: Vec<Sample>,
}

impl PendingCensus {
    /// Count a pending turn holding `held` of `wanted` chunks, with or
    /// without its tokens.
    pub fn note(&mut self, timeline: u64, turn: u32, tokens: bool, held: usize, wanted: usize) {
        if !tokens {
            self.no_tokens += 1;
        } else if held == 0 {
            self.no_chunks += 1;
        } else {
            self.some_chunks += 1;
        }
        self.sample(Sample {
            timeline,
            turn,
            tokens,
            chunks_held: held,
            chunks_wanted: wanted,
        });
    }

    /// Count a pending turn whose records are here but unreadable.
    pub fn note_unreadable(&mut self, timeline: u64, turn: u32) {
        self.unreadable += 1;
        self.sample(Sample {
            timeline,
            turn,
            tokens: true,
            chunks_held: 0,
            chunks_wanted: 0,
        });
    }

    fn sample(&mut self, s: Sample) {
        if self.samples.len() < SAMPLES {
            self.samples.push(s);
        }
    }

    /// Every pending turn counted.
    pub fn total(&self) -> usize {
        self.no_tokens + self.no_chunks + self.some_chunks + self.unreadable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_shortfall_is_counted_apart() {
        let mut c = PendingCensus::default();
        c.note(1, 0, false, 0, 26);
        c.note(1, 1, true, 0, 26);
        c.note(2, 0, true, 13, 26);
        c.note_unreadable(3, 0);
        assert_eq!(
            (c.no_tokens, c.no_chunks, c.some_chunks, c.unreadable),
            (1, 1, 1, 1)
        );
        assert_eq!(c.total(), 4);
    }

    #[test]
    fn the_first_few_are_kept_as_samples() {
        let mut c = PendingCensus::default();
        for t in 0..10 {
            c.note(7, t, true, 0, 13);
        }
        assert_eq!(c.samples.len(), SAMPLES);
        assert_eq!(
            c.samples[0],
            Sample {
                timeline: 7,
                turn: 0,
                tokens: true,
                chunks_held: 0,
                chunks_wanted: 13,
            }
        );
        assert_eq!(c.no_chunks, 10);
    }
}
