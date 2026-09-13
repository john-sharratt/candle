//! Does a live decode row's offset agree with the layer it is serialised for?
//!
//! A decode header describes the writer chunk with a length derived from the
//! row's **offset** (`rebuild_decode_gpu_chunks`: `write_len = seq_offset − Σ
//! usage before the writer`), not from what the chunk holds. For a live row the
//! two are the same number: the offset is the position the new token is written
//! at, and everything before it is committed history. When they differ the
//! header misdescribes the layer to the kernel in one of two ways, both silent:
//!
//! * **Offset ahead** — the header counts a position the layer never wrote. The
//!   attention reads it (allocation poison under `tensor-assert`, the slot's
//!   previous tenant otherwise), and once the claimed length reaches
//!   `CHUNK_SIZE` the fused scatter's `within < CHUNK_SIZE` guard drops the new
//!   token's K/V write outright — while the per-step usage advance still counts
//!   it, which is how a single hole becomes permanent history.
//! * **Offset behind** — the header hides committed tokens, and the new token is
//!   written over one of them.
//!
//! The prefill header build refuses this already (`build_slot_headers` checks
//! its slices against the recorded offset in both directions). The decode and
//! draft builds did not check it at all. This is that check, as an instrument:
//! it names the layer, the sequence and the numbers the first time a live row
//! disagrees — the moment the skew is created, not the wave that later reads it.
//!
//! Live rows only. A snapshot row — a verify block re-reading positions it
//! already holds, a per-token prefill snapshot — is serialised at an offset
//! below the layer's length on purpose.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::types::SequenceState;

/// Disagreements logged in full before the instrument drops to powers of two.
const REPORTED_IN_FULL: usize = 64;

/// Every disagreement seen, for the occurrence count each report carries.
static REPORTS: AtomicUsize = AtomicUsize::new(0);

/// A live row whose header length and writer chunk disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WriterDisagreement {
    /// The writer chunk (`decode_write_chunk_idx`).
    pub writer: usize,
    /// Tokens the writer chunk holds.
    pub usage: usize,
    /// Tokens the header will claim it holds.
    pub write_len: usize,
    /// Tokens the layer holds in all.
    pub committed: usize,
}

/// How `seq`, serialised as a live decode row at `seq_offset`, would misdescribe
/// its writer chunk — `None` when the header would be truthful.
pub(super) fn writer_disagreement(
    seq: &SequenceState,
    seq_offset: usize,
) -> Option<WriterDisagreement> {
    let chunks = seq.chunks_slice();
    if chunks.is_empty() {
        return None;
    }
    // The same writer and the same arithmetic `rebuild_decode_gpu_chunks` uses,
    // so a disagreement here is exactly a lie in the header it builds.
    let writer = seq.decode_write_chunk_idx();
    let before: usize = chunks[..writer].iter().map(|c| c.usage as usize).sum();
    let usage = chunks[writer].usage as usize;
    let write_len = seq_offset.saturating_sub(before);
    (write_len != usage).then(|| WriterDisagreement {
        writer,
        usage,
        write_len,
        committed: chunks.iter().map(|c| c.usage as usize).sum(),
    })
}

/// Whether the `n`th disagreement is logged: all of the first
/// [`REPORTED_IN_FULL`], then each power of two, so a skew that persists for
/// thousands of steps reads as one finding with a count rather than a flood.
fn logs(n: usize) -> bool {
    n <= REPORTED_IN_FULL || n.is_power_of_two()
}

/// Check `seq` as a live row of `layer` at `seq_offset`, and report it if its
/// header would misdescribe the writer chunk.
pub(super) fn audit(layer: usize, seq_idx: usize, seq_offset: usize, seq: &SequenceState) {
    let Some(d) = writer_disagreement(seq, seq_offset) else {
        return;
    };
    let n = REPORTS.fetch_add(1, Ordering::Relaxed) + 1;
    if !logs(n) {
        return;
    }
    let ahead = d.write_len > d.usage;
    tracing::error!(
        target: "candle_nn::kv_cache::writer_len",
        layer,
        seq = seq_idx,
        seq_offset,
        writer = d.writer,
        usage = d.usage,
        write_len = d.write_len,
        committed = d.committed,
        occurrence = n,
        "decode row offset {} of its layer: the header claims {} tokens in writer chunk {} \
         which holds {} (layer holds {} in all, row offset {seq_offset}) — {}",
        if ahead { "AHEAD" } else { "BEHIND" },
        d.write_len,
        d.writer,
        d.usage,
        d.committed,
        if ahead {
            "the attention reads a position this layer never wrote, and at a full claimed \
             length the scatter drops the new token's K/V while the usage advance counts it"
        } else {
            "committed tokens drop out of the header and the new token is written over one"
        }
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{logs, writer_disagreement, WriterDisagreement, REPORTED_IN_FULL};
    use crate::kv_cache::chunked::types::SequenceState;
    use crate::kv_cache::chunked::{ChunkGid, ChunkWindow, HeadGids};

    fn window(usage: u32) -> ChunkWindow {
        ChunkWindow {
            gids: HeadGids::uniform(ChunkGid::detached(0), 1),
            usage,
            offset: 0,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(Vec::new()),
            v_fmt: Arc::new(Vec::new()),
            meta: None,
        }
    }

    fn layer(usages: &[u32], writer_start: usize) -> SequenceState {
        let mut s = SequenceState::new(None);
        for &u in usages {
            s.push_chunk(window(u));
        }
        s.set_writer_start_idx(writer_start);
        s
    }

    #[test]
    fn a_live_row_at_its_layers_length_agrees() {
        assert_eq!(writer_disagreement(&layer(&[32, 31], 0), 63), None);
    }

    /// The fault's own numbers: a writer holding 31 of 32, described at an
    /// offset one past what the layer holds, is claimed full — the length at
    /// which the fused scatter drops the write.
    #[test]
    fn one_ahead_claims_a_position_the_layer_never_wrote() {
        assert_eq!(
            writer_disagreement(&layer(&[32, 31], 0), 64),
            Some(WriterDisagreement {
                writer: 1,
                usage: 31,
                write_len: 32,
                committed: 63
            })
        );
    }

    #[test]
    fn one_behind_hides_a_committed_token() {
        assert_eq!(
            writer_disagreement(&layer(&[32, 31], 0), 62),
            Some(WriterDisagreement {
                writer: 1,
                usage: 31,
                write_len: 30,
                committed: 63
            })
        );
    }

    /// The captured layout: a sealed partial chunk, a sealed empty pad, then
    /// the writer. The writer is found past the boundary, and its length is
    /// measured from everything before it — the empty pad included.
    #[test]
    fn the_writer_is_found_past_a_sealed_partial_and_its_pad() {
        let l = layer(&[10, 0, 31], 2);
        assert_eq!(writer_disagreement(&l, 41), None);
        assert_eq!(
            writer_disagreement(&l, 42),
            Some(WriterDisagreement {
                writer: 2,
                usage: 31,
                write_len: 32,
                committed: 41
            })
        );
    }

    /// A trailing empty chunk the decode ensure pushed is not the writer while
    /// the chunk before it still has room.
    #[test]
    fn a_trailing_empty_chunk_is_not_the_writer() {
        assert_eq!(writer_disagreement(&layer(&[32, 31, 0], 0), 63), None);
    }

    #[test]
    fn an_empty_layer_has_nothing_to_misdescribe() {
        assert_eq!(writer_disagreement(&layer(&[], 0), 5), None);
    }

    #[test]
    fn reports_are_logged_in_full_then_at_powers_of_two() {
        assert!((1..=REPORTED_IN_FULL).all(logs));
        assert!(!logs(REPORTED_IN_FULL + 1));
        assert!(logs(128));
        assert!(!logs(129));
        assert!(logs(1 << 20));
    }
}
