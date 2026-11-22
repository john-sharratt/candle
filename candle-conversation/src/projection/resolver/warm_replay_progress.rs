//! How far the dialogue normalization replay has got — the `/v1/status` bar a
//! daemon shows while its score backgrounds re-learn after a load.
//!
//! The replay's turns are counted in when they are handed out
//! ([`Conversation::take_warm_replay`]) and out as each is scored or dropped,
//! so a re-arm while an earlier replay still drains adds to the count rather
//! than resetting it. The count settles to empty the moment the scored catch
//! up, so the next replay starts from `0 / 0`.

use std::sync::Mutex;

/// `(scored, total)` of the replay now draining, under one lock so a reader
/// never sees one without the other.
#[derive(Default)]
pub struct WarmReplayProgress {
    count: Mutex<(usize, usize)>,
}

impl WarmReplayProgress {
    /// `n` more turns handed out to be scored.
    pub fn add(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut c = self.count.lock().unwrap_or_else(|e| e.into_inner());
        c.1 += n;
    }

    /// One handed-out turn scored.
    pub fn scored_one(&self) {
        let mut c = self.count.lock().unwrap_or_else(|e| e.into_inner());
        c.0 += 1;
        settle(&mut c);
    }

    /// `n` handed-out turns that will never be scored — their substrate
    /// detached before the scheduler reached them.
    pub fn dropped(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut c = self.count.lock().unwrap_or_else(|e| e.into_inner());
        c.1 = c.1.saturating_sub(n);
        settle(&mut c);
    }

    /// `(scored, total)` while a replay drains; `None` once nothing is left.
    pub fn snapshot(&self) -> Option<(usize, usize)> {
        let c = self.count.lock().unwrap_or_else(|e| e.into_inner());
        (c.1 > 0).then_some(*c)
    }
}

fn settle(c: &mut (usize, usize)) {
    if c.0 >= c.1 {
        *c = (0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::WarmReplayProgress;

    #[test]
    fn counts_up_then_settles_empty() {
        let p = WarmReplayProgress::default();
        assert_eq!(p.snapshot(), None);
        p.add(3);
        assert_eq!(p.snapshot(), Some((0, 3)));
        p.scored_one();
        p.scored_one();
        assert_eq!(p.snapshot(), Some((2, 3)));
        p.scored_one();
        assert_eq!(p.snapshot(), None, "scored caught up: empty");
        p.add(2);
        assert_eq!(p.snapshot(), Some((0, 2)), "the next replay starts at zero");
    }

    #[test]
    fn a_rearm_mid_drain_adds_to_the_count() {
        let p = WarmReplayProgress::default();
        p.add(2);
        p.scored_one();
        p.add(4);
        assert_eq!(p.snapshot(), Some((1, 6)));
    }

    #[test]
    fn dropped_turns_let_the_count_settle() {
        let p = WarmReplayProgress::default();
        p.add(5);
        p.scored_one();
        p.dropped(4);
        assert_eq!(p.snapshot(), None);
        p.dropped(3);
        assert_eq!(p.snapshot(), None, "saturates, never underflows");
        p.add(0);
        assert_eq!(p.snapshot(), None, "a zero hand-out is a no-op");
    }
}
