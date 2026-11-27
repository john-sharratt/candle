//! Holes in a slot's position space: positions no K/V sits at
//! (`docs/placed_section_layout.md` §3.1, §4.1).
//!
//! A reserved range the projection leaves partly empty, and the gap between two
//! ranges, are positions a later chunk is read beyond. A hole is a lead on the
//! chunk after it, so every chunk's RoPE base, the position frontier and the
//! token count stay three different numbers: the token count is the tokens the
//! slot holds, the frontier is where its next token is read, and the frontier
//! exceeds the count by the slot's holes.

use super::*;

impl ChunkedKvBacking {
    /// Leave `n` positions empty at `batch_idx`'s tail: the content the slot
    /// gains next is read `n` positions further on.
    ///
    /// A tail holding tokens is closed first with a fresh writer chunk, which
    /// takes the hole as its lead, so no token is ever written after the hole
    /// into the chunk before it. An empty writer tail takes the hole as it is.
    pub fn push_position_gap(&self, batch_idx: usize, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let n = u32::try_from(n)
            .map_err(|_| candle::Error::Msg(format!("push_position_gap: {n} positions")))?;
        let tail_holds_tokens = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let Some(slot) = state.sequences.get(batch_idx).and_then(|s| s.as_ref()) else {
                candle::bail!("push_position_gap: slot {batch_idx} not allocated");
            };
            slot.last_chunk().is_some_and(|c| c.usage > 0)
        };
        if tail_holds_tokens {
            self.push_empty_writer_chunk(batch_idx)?;
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) else {
            candle::bail!("push_position_gap: slot {batch_idx} not allocated");
        };
        slot.push_lead(n);
        Ok(())
    }

    /// Drop `batch_idx`'s trailing chunks that hold no token, keeping every
    /// hole they carried pending at the new tail. Returns the chunks dropped;
    /// `Ok(0)` for an unallocated slot.
    pub fn trim_empty_tail(&self, batch_idx: usize) -> Result<usize> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        Ok(state
            .sequences
            .get_mut(batch_idx)
            .and_then(|s| s.as_mut())
            .map_or(0, |s| s.trim_empty_tail()))
    }

    /// The positions `batch_idx` leaves empty — every hole before its tail.
    /// `Ok(0)` for an unallocated slot.
    pub fn lead_total(&self, batch_idx: usize) -> Result<usize> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        Ok(state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .map_or(0, |s| s.lead_total()))
    }

    /// The position `batch_idx`'s next token is read at: its evicted front,
    /// every chunk's lead and tokens, and the hole pending at its tail. `Ok(0)`
    /// for an unallocated slot.
    pub fn position_frontier(&self, batch_idx: usize) -> Result<usize> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        Ok(state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .map_or(0, |s| s.position_frontier()))
    }
}
