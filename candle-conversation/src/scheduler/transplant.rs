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

/// The first `n` chunks of `captured`, each one `own` names taken from `ours`
/// instead. `window` is a chunk's `(offset, tokens)`: a chunk kept from `ours`
/// must lie in the captured one's window, or the slot does not hold the same
/// prompt in the same chunks and the mix is refused.
pub(super) fn mixed_chunks<T: Clone>(
    captured: &[T],
    ours: &[T],
    n: usize,
    own: &[usize],
    window: impl Fn(&T) -> (u16, u16),
) -> Result<Vec<T>, ConversationError> {
    let mut out: Vec<T> = captured[..n].to_vec();
    for &i in own {
        let (Some(c), Some(o)) = (captured.get(i).filter(|_| i < n), ours.get(i)) else {
            return Err(ConversationError::Channel(format!(
                "transplant: chunk {i} kept as the slot's own is past the prompt's {n} \
                 chunks or the slot's {} chunks",
                ours.len()
            )));
        };
        if window(c) != window(o) {
            return Err(ConversationError::Channel(format!(
                "transplant: chunk {i} is {:?} on the slot but {:?} in the capture",
                window(o),
                window(c)
            )));
        }
        out[i] = o.clone();
    }
    Ok(out)
}

impl Scheduler {
    /// Give `slot`, holding the first `prompt_tokens` of the captured slot's
    /// prompt in the same windows, the capture's K/V for them (`kv`), its
    /// index pages for them (`index`), or both. With `kv`, the chunks
    /// `own_chunks` names keep the slot's own K/V. The capture stays until it is
    /// released ([`SchedulerRequest::ReleaseCapture`]), so one described slot
    /// can be given to as many slots as ask for it.
    pub(super) fn transplant(
        &mut self,
        slot: SequenceId,
        capture: u64,
        prompt_tokens: usize,
        kv: bool,
        index: bool,
        own_chunks: &[usize],
    ) -> Result<(), ConversationError> {
        let cap = self
            .captures
            .get(&capture)
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
                if own_chunks.is_empty() {
                    prompt.chunks.truncate(n);
                } else {
                    // The slot's own chunks, held before it is cut back: what
                    // the kept anchors are read from.
                    let ours = backing
                        .record_turn(slot.0)
                        .map_err(ConversationError::Model)?;
                    prompt.chunks =
                        mixed_chunks(&sealed.chunks, &ours.chunks, n, own_chunks, |c| {
                            (c.offset, c.token_count)
                        })?;
                }
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

    /// **The named chunks are the slot's own, every other the capture's**, cut
    /// to the prompt.
    #[test]
    fn the_named_chunks_are_the_slots_own() {
        let captured = [("c0", 0, 32), ("c1", 0, 32), ("c2", 0, 5), ("c3", 0, 9)];
        let ours = [("o0", 0, 32), ("o1", 0, 32), ("o2", 0, 5), ("o3", 0, 9)];
        let window = |c: &(&str, u16, u16)| (c.1, c.2);
        let mixed = mixed_chunks(&captured, &ours, 3, &[0, 2], window).unwrap();
        let names: Vec<&str> = mixed.iter().map(|c| c.0).collect();
        assert_eq!(names, vec!["o0", "c1", "o2"]);
        let none = mixed_chunks(&captured, &ours, 3, &[], window).unwrap();
        let names: Vec<&str> = none.iter().map(|c| c.0).collect();
        assert_eq!(names, vec!["c0", "c1", "c2"]);
    }

    /// **A kept chunk past the prompt, or in another window, is refused.**
    #[test]
    fn a_kept_chunk_out_of_place_is_refused() {
        let captured = [("c0", 0, 32), ("c1", 0, 7)];
        let ours = [("o0", 0, 32), ("o1", 0, 8)];
        let window = |c: &(&str, u16, u16)| (c.1, c.2);
        assert!(mixed_chunks(&captured, &ours, 1, &[1], window).is_err());
        assert!(mixed_chunks(&captured, &ours, 2, &[1], window).is_err());
        assert!(mixed_chunks(&captured, &ours[..1], 2, &[1], window).is_err());
    }

    /// **A prompt that ends inside an item is refused.**
    #[test]
    fn a_prompt_ending_inside_an_item_is_refused() {
        assert!(leading_covering(&[32, 8], 36).is_err());
        assert!(leading_covering(&[32], 40).is_err());
    }
}
