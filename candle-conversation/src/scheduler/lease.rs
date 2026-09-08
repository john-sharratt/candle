//! The decode lease's boundary: what a spent turn's K/V carries to the warm
//! tier, and the arithmetic that keeps the resumed slot consistent.

use candle_nn::kv_cache::SealedSequence;

/// Where a layer's chunk list is cut so it covers exactly `tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Cut {
    /// Chunks kept, counted from the front. Everything from here on is dropped.
    pub keep: usize,
    /// The kept tail chunk's token count after the cut — `None` when the cut
    /// falls exactly on a chunk boundary and no chunk is trimmed.
    pub last: Option<u16>,
    /// Tokens the layer held past `tokens`: trimmed rows plus dropped chunks.
    pub surplus: usize,
}

/// Cut a chunk list whose windows hold `usages` tokens each so it covers
/// exactly `tokens`, or all of it when it holds fewer.
///
/// Pure over the usages, so the boundary cases are tested without a backing:
/// a cut on a chunk boundary keeps the chunk before it whole and drops
/// everything after, including empty writer chunks; a cut inside a chunk trims
/// that chunk; a list already short of `tokens` is untouched.
pub(super) fn cut_at(usages: &[u16], tokens: usize) -> Cut {
    let total: usize = usages.iter().map(|&u| u as usize).sum();
    if total <= tokens {
        return Cut {
            keep: usages.len(),
            last: None,
            surplus: 0,
        };
    }
    let mut cum = 0usize;
    for (i, &usage) in usages.iter().enumerate() {
        let usage = usage as usize;
        if cum + usage >= tokens {
            let want = tokens - cum;
            return if want == 0 {
                Cut {
                    keep: i,
                    last: None,
                    surplus: total - tokens,
                }
            } else if want == usage {
                // The cut lands on this chunk's end: kept whole, nothing trimmed.
                Cut {
                    keep: i + 1,
                    last: None,
                    surplus: total - tokens,
                }
            } else {
                Cut {
                    keep: i + 1,
                    last: Some(want as u16),
                    surplus: total - tokens,
                }
            };
        }
        cum += usage;
    }
    Cut {
        keep: usages.len(),
        last: None,
        surplus: 0,
    }
}

/// Trim every layer of a parked turn's snapshot to the `tokens` its sequence
/// has committed, answering the largest surplus any layer carried.
///
/// **A resume re-derives the slot's offset from the chunks it injects, so the
/// snapshot has to cover exactly the committed tokens.** A slot's chunks can
/// hold rows past its recorded offset — a speculative block's rejected tail
/// that a rollback clamped at a sealed boundary, an empty writer chunk pushed
/// for the next step — and the prefill-slot header build refuses a slot whose
/// chunks cover more than its offset counts, because the surplus displaces
/// every position written after it. Run 12 measured exactly that on the first
/// decode after each resume — `slices cover 1455 tokens but the slot's recorded
/// offset is 1453 (delta +2)` — and the refusal failed every sequence in the
/// wave. Cutting here makes the invariant hold by construction, whatever the
/// slot looked like at the moment its lease ran out.
pub(super) fn trim_sealed_to_tokens(layers: &mut [SealedSequence], tokens: usize) -> usize {
    let mut surplus = 0usize;
    for seq in layers.iter_mut() {
        let usages: Vec<u16> = seq.chunks.iter().map(|c| c.token_count).collect();
        let cut = cut_at(&usages, tokens);
        if cut.surplus == 0 {
            continue;
        }
        surplus = surplus.max(cut.surplus);
        seq.chunks.truncate(cut.keep);
        if let (Some(last), Some(chunk)) = (cut.last, seq.chunks.last_mut()) {
            chunk.token_count = last;
        }
        seq.token_count = tokens;
    }
    surplus
}

#[cfg(test)]
mod tests {
    use super::{cut_at, Cut};

    /// The measured case: a full block table plus the two rows the offset does
    /// not count, in the partial tail, and an empty writer chunk behind it.
    #[test]
    fn a_surplus_in_the_tail_is_trimmed_and_the_empty_writer_chunk_dropped() {
        let usages = [32u16, 32, 32, 32, 12, 0];
        assert_eq!(
            cut_at(&usages, 32 * 4 + 10),
            Cut {
                keep: 5,
                last: Some(10),
                surplus: 2,
            }
        );
    }

    /// A cut on a chunk boundary keeps the chunk before it whole and drops
    /// everything after — no chunk is trimmed to zero and kept.
    #[test]
    fn a_cut_on_a_boundary_drops_what_follows_without_trimming() {
        let usages = [32u16, 32, 5];
        assert_eq!(
            cut_at(&usages, 64),
            Cut {
                keep: 2,
                last: None,
                surplus: 5,
            }
        );
    }

    /// A list that already covers exactly the committed tokens, or fewer, is
    /// untouched: the resume then injects it whole.
    #[test]
    fn a_list_at_or_under_the_offset_is_left_alone() {
        let exact = [32u16, 32, 7];
        assert_eq!(
            cut_at(&exact, 71),
            Cut {
                keep: 3,
                last: None,
                surplus: 0,
            }
        );
        assert_eq!(
            cut_at(&exact, 100),
            Cut {
                keep: 3,
                last: None,
                surplus: 0,
            },
            "a short layer is not the trim's to fix"
        );
        assert_eq!(
            cut_at(&[], 10),
            Cut {
                keep: 0,
                last: None,
                surplus: 0,
            }
        );
    }

    /// Whole trailing chunks past the cut count toward the surplus along with
    /// the trimmed rows.
    #[test]
    fn dropped_chunks_count_toward_the_surplus() {
        let usages = [32u16, 32, 32, 32];
        assert_eq!(
            cut_at(&usages, 40),
            Cut {
                keep: 2,
                last: Some(8),
                surplus: 88,
            }
        );
    }
}
