//! What the simulated cluster is asked to do: conversations arriving, each a
//! run of turns, each turn a run of segments — prefill something, then decode
//! — which is how a zend turn with tool rounds runs: decode a call, prefill
//! its result, decode on.
//!
//! Generated from a seed by a fixed linear congruential generator, so a
//! scenario is the same every run and the policies are compared on the same
//! arrivals.

/// One prefill-then-decode stretch of a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub prefill: usize,
    pub decode: usize,
}

/// One turn: the pause before it (the person reading and typing), and its
/// segments.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub think_s: f64,
    pub segments: Vec<Segment>,
}

/// One conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Conversation {
    pub id: usize,
    pub arrive_s: f64,
    /// Attended context before its first turn: the projection's system
    /// prompt and selected history.
    pub context: usize,
    pub turns: Vec<Turn>,
}

/// A deterministic stream of uniform numbers.
#[derive(Debug, Clone)]
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Lcg {
        let mixed = seed.wrapping_mul(6364136223846793005);
        Lcg(mixed.wrapping_add(1442695040888963407))
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[lo, hi]`.
    pub fn between(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    pub fn int(&mut self, lo: usize, hi: usize) -> usize {
        self.between(lo as f64, hi as f64 + 0.999).floor() as usize
    }
}

/// The kinds of conversation the zend-load rungs drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Passthrough chat: a prompt, a reply, short follow-ups.
    Chat,
    /// zen-code reading the candle tree: tool rounds with large results.
    Read,
}

/// `n` conversations of `kind`, arriving over `spread_s`.
pub fn generate(kind: Kind, n: usize, spread_s: f64, seed: u64) -> Vec<Conversation> {
    let mut r = Lcg::new(seed);
    (0..n)
        .map(|id| {
            let arrive_s = r.between(0.0, spread_s);
            let (context, turns) = match kind {
                Kind::Chat => (
                    r.int(8_000, 24_000),
                    (0..3)
                        .map(|t| Turn {
                            think_s: if t == 0 { 0.0 } else { r.between(5.0, 30.0) },
                            segments: vec![Segment {
                                prefill: if t == 0 {
                                    r.int(800, 2_500)
                                } else {
                                    r.int(30, 300)
                                },
                                decode: r.int(150, 400),
                            }],
                        })
                        .collect(),
                ),
                Kind::Read => (
                    r.int(60_000, 240_000),
                    (0..2)
                        .map(|t| Turn {
                            think_s: if t == 0 { 0.0 } else { r.between(10.0, 60.0) },
                            segments: (0..r.int(3, 6))
                                .map(|s| Segment {
                                    prefill: if s == 0 {
                                        r.int(1_000, 4_000)
                                    } else {
                                        r.int(1_500, 8_000)
                                    },
                                    decode: r.int(60, 250),
                                })
                                .collect(),
                        })
                        .collect(),
                ),
            };
            Conversation {
                id,
                arrive_s,
                context,
                turns,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seed gives the same scenario every time, and another seed another.
    #[test]
    fn a_seed_is_a_scenario() {
        assert_eq!(
            generate(Kind::Read, 8, 10.0, 7),
            generate(Kind::Read, 8, 10.0, 7)
        );
        assert_ne!(
            generate(Kind::Read, 8, 10.0, 7),
            generate(Kind::Read, 8, 10.0, 8)
        );
    }

    /// Chats are three single-segment turns, a long first prompt then short
    /// follow-ups; reads are tool rounds; every arrival is inside the spread.
    #[test]
    fn kinds_have_their_shapes() {
        for c in generate(Kind::Chat, 50, 10.0, 1) {
            assert!((0.0..=10.0).contains(&c.arrive_s));
            assert_eq!(c.turns.len(), 3);
            assert!(c.turns.iter().all(|t| t.segments.len() == 1));
            assert!(c.turns[0].segments[0].prefill >= 800);
            assert!(c.turns[1].segments[0].prefill <= 300);
            assert_eq!(c.turns[0].think_s, 0.0);
        }
        for c in generate(Kind::Read, 50, 10.0, 1) {
            assert!(c.turns.iter().all(|t| (3..=6).contains(&t.segments.len())));
        }
    }

    /// Uniforms stay in range.
    #[test]
    fn uniforms_stay_in_range() {
        let mut r = Lcg::new(3);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
            assert!((5..=9).contains(&r.int(5, 9)));
        }
    }
}
