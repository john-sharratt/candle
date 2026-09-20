//! The layers of one sequence must describe the same token windows.
//!
//! A slot's thirteen KV layers each keep their own chunk list. Those lists are
//! allowed to differ in one harmless way — a layer may carry a trailing *empty*
//! writer chunk the others do not, left by a windowed creep prefill or by
//! `reconcile_block_counts` padding a lagging layer up to the longest. It holds
//! no token, so no position moves and every token-count check passes.
//!
//! It stops being harmless the moment something is **appended**: the layers
//! carrying the empty chunk take the new content one block further along than
//! the layers without it, the empty chunk becomes *interior*, and the layers now
//! disagree about which chunk holds which tokens. Nothing downstream can repair
//! that — `heal_tail_divergence` refuses an interior difference as mid-history
//! corruption, and the slot is wedged for the rest of its life.
//!
//! These tests pin the two places that have to make the difference go away:
//! trimming the live layers before an append, and dropping it from a snapshot
//! before it can be persisted and re-injected somewhere else.

// `batched_inference` is compiled only with CUDA, though nothing here needs a
// device: the whole fixture is chunk bookkeeping on `Device::Cpu`.
#![cfg(feature = "cuda")]

use candle::{Device, Result};
use candle_transformers::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, KvLayers,
};

/// Enough layers to hold a skew on some and not others.
const LAYERS: usize = 4;
/// Layers `[0, SKEWED)` get the odd empty writer chunk.
const SKEWED: usize = 2;
/// `candle-nn`'s chunk width, shared with the CUDA side.
const CHUNK: usize = 32;

fn session() -> Result<BatchedInferenceSession> {
    BatchedInferenceSession::new(
        KvLayers::stream_only(LAYERS),
        4,
        32,
        &Device::Cpu,
        BatchedConfig::default(),
    )
}

/// Put `tokens` worth of real content on every layer and advance the slot.
///
/// `ensure_for_offset` only allocates the chunks — a chunk's `usage` is
/// committed by the write, which is what `set_len` stands in for here. Both
/// steps matter: a chunk with capacity and no usage is exactly the empty writer
/// chunk these tests are about, so a fixture that skipped the commit would
/// build a slot made entirely of them.
fn fill(s: &mut BatchedInferenceSession, seq: usize, at: usize, tokens: usize) -> Result<()> {
    for b in s.backings() {
        b.ensure_for_offset(seq, at, tokens)?;
        b.set_len(seq, at + tokens);
    }
    s.advance_sequence(seq, tokens)
}

fn block_counts(s: &BatchedInferenceSession, seq: usize) -> Vec<Option<usize>> {
    s.backings()
        .iter()
        .map(|b| b.sequence_block_count(seq))
        .collect()
}

/// 40 tokens: one full chunk and a partial, so the record has real structure to
/// land wrongly if the append starts from different blocks.
fn donor_record(
    s: &mut BatchedInferenceSession,
) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
    let donor = s.create_sequence()?;
    fill(s, donor, 0, CHUNK + 8)?;
    s.snapshot_sequence_per_layer(donor)
}

/// The production failure, in miniature: a slot whose first two layers carry an
/// empty writer chunk takes an injected record at a different block index on
/// those layers than on the rest, and every later read sees two layer groups
/// disagreeing about chunk `n`.
#[test]
fn injecting_onto_a_partly_padded_slot_keeps_every_layer_aligned() -> Result<()> {
    let mut s = session()?;
    let sealed = donor_record(&mut s)?;

    let target = s.create_sequence()?;
    fill(&mut s, target, 0, CHUNK)?;
    // The skew, exactly as `reconcile_block_counts` and a paused creep leave it:
    // real on some layers, absent on others, and empty either way.
    for b in &s.backings()[..SKEWED] {
        b.push_empty_writer_chunk(target)?;
    }
    assert_eq!(
        block_counts(&s, target),
        vec![Some(2), Some(2), Some(1), Some(1)],
        "the fixture must start skewed, or the test proves nothing"
    );

    s.inject_sealed_at_tail(target, &sealed)?;

    let counts = block_counts(&s, target);
    assert!(
        counts.iter().all(|c| *c == counts[0]),
        "the injected record landed at different block indices per layer: {counts:?}"
    );
    // The real assertion: the layers describe the same windows, chunk for chunk.
    // This is the check that fires in production as "chunked decode layout
    // diverged across layers".
    s.snapshot_sequence_per_layer(target)?;
    Ok(())
}

/// Trimming is lossless: it takes the empty chunks and nothing else, and the
/// slot's token count is untouched.
#[test]
fn trimming_removes_trailing_empty_chunks_and_no_content() -> Result<()> {
    let mut s = session()?;
    let seq = s.create_sequence()?;
    fill(&mut s, seq, 0, CHUNK + 8)?;
    for b in s.backings() {
        b.push_empty_writer_chunk(seq)?;
        b.push_empty_writer_chunk(seq)?;
    }
    assert_eq!(block_counts(&s, seq), vec![Some(4); LAYERS]);

    s.trim_empty_tail_chunks(seq)?;

    assert_eq!(
        block_counts(&s, seq),
        vec![Some(2); LAYERS],
        "both empty chunks go; the full chunk and the partial stay"
    );
    assert_eq!(
        s.sequence_offset(seq),
        Some(CHUNK + 8),
        "an empty chunk holds no token, so the offset must not move"
    );
    Ok(())
}

/// A snapshot is what gets persisted and re-injected into other conversations,
/// so the empty chunk must not survive the capture — otherwise the skew is
/// copied into every slot that later borrows the record.
#[test]
fn a_snapshot_carries_no_trailing_empty_chunk() -> Result<()> {
    let mut s = session()?;
    let seq = s.create_sequence()?;
    fill(&mut s, seq, 0, CHUNK + 8)?;
    for b in s.backings() {
        b.push_empty_writer_chunk(seq)?;
    }

    let sealed = s.snapshot_sequence_per_layer(seq)?;

    for (li, layer) in sealed.iter().enumerate() {
        assert_eq!(
            layer.chunks.len(),
            2,
            "layer {li} captured its empty writer chunk: {:?}",
            layer
                .chunks
                .iter()
                .map(|c| c.token_count)
                .collect::<Vec<_>>()
        );
        assert_eq!(layer.token_count, CHUNK + 8);
    }
    Ok(())
}
