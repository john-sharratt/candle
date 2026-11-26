//! One slot's prompt K/V or index given to another
//! ([`SchedulerRequest::Transplant`]): what tells a fault in the K/V a slot
//! attends over from a fault in the index that chooses what it attends to.
//!
//! A slot described at its turn's end ([`SchedulerRequest::DescribeAtSeal`])
//! is captured with it — its chunks on every layer, their handles held so the
//! seal emptying the slot frees nothing, and its index pages. A second slot
//! holding the same prompt in the same chunk windows can then take the
//! captured chunks in place of its own, the captured pages in place of its
//! index, or both, before its own turn decodes over them.

use candle_nn::kv_cache::SealedSequence;

use super::*;

/// A described slot's chunks on every layer and its index pages, each page
/// with the tokens it covers.
pub(super) struct SlotCapture {
    pub(super) layers: Vec<SealedSequence>,
    pub(super) pages: Vec<(usize, Vec<u8>)>,
}

/// How many of the leading items, of the token widths `widths`, make exactly
/// `tokens`: the chunks — or pages — a prompt of `tokens` occupies. Refused
/// when no prefix sums to it.
pub(super) fn leading_covering(
    widths: &[usize],
    tokens: usize,
) -> Result<usize, ConversationError> {
    let mut held = 0;
    for (i, &w) in widths.iter().enumerate() {
        if held == tokens {
            return Ok(i);
        }
        held += w;
        if held > tokens {
            break;
        }
    }
    if held == tokens {
        return Ok(widths.len());
    }
    Err(ConversationError::Channel(format!(
        "transplant: no leading run of {widths_len} widths makes {tokens} tokens",
        widths_len = widths.len()
    )))
}

impl Scheduler {
    /// Give `slot`, holding the first `prompt_tokens` of the captured slot's
    /// prompt in the same windows, the capture's K/V for them (`kv`), its
    /// index pages for them (`index`), or both. The capture is taken either
    /// way.
    pub(super) fn transplant(
        &mut self,
        slot: SequenceId,
        capture: u64,
        prompt_tokens: usize,
        kv: bool,
        index: bool,
    ) -> Result<(), ConversationError> {
        let cap = self
            .captures
            .remove(&capture)
            .ok_or_else(|| ConversationError::Channel(format!("no capture {capture}")))?;
        if kv {
            let backings = self.session.backings();
            for (backing, sealed) in backings.iter().zip(&cap.layers) {
                let widths: Vec<usize> = sealed
                    .chunks
                    .iter()
                    .map(|c| c.token_count as usize)
                    .collect();
                let n = leading_covering(&widths, prompt_tokens)?;
                let mut prompt = sealed.clone();
                prompt.chunks.truncate(n);
                prompt.token_count = prompt_tokens;
                backing
                    .truncate_sequence_to_blocks(slot.0, 0)
                    .map_err(ConversationError::Model)?;
                backing
                    .inject_sealed_at_tail(slot.0, &prompt)
                    .map_err(ConversationError::Model)?;
            }
        }
        if index {
            let widths: Vec<usize> = cap.pages.iter().map(|(t, _)| *t).collect();
            let n = leading_covering(&widths, prompt_tokens)?;
            self.model
                .reset_positional_state(slot.0)
                .map_err(ConversationError::Model)?;
            for (_, page) in &cap.pages[..n] {
                self.model
                    .push_positional_state(slot.0, page, &PieceKey::of(page))
                    .map_err(ConversationError::Model)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A prompt takes the leading items that sum to it exactly**, an empty
    /// item at its end left with what follows.
    #[test]
    fn a_prompt_takes_the_leading_items_that_sum_to_it() {
        assert_eq!(leading_covering(&[32, 8, 0, 5, 20], 40).unwrap(), 2);
        assert_eq!(leading_covering(&[32, 8, 5], 45).unwrap(), 3);
        assert_eq!(leading_covering(&[3, 273], 0).unwrap(), 0);
    }

    /// **A prompt that ends inside an item is refused.**
    #[test]
    fn a_prompt_ending_inside_an_item_is_refused() {
        assert!(leading_covering(&[32, 8], 36).is_err());
        assert!(leading_covering(&[32], 40).is_err());
    }
}
