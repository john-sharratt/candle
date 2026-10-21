//! `BatchedInferenceSession::splice_sealed` against a CPU session: a sealed
//! piece put in the middle of a slot, on every layer, with the sequence's
//! offset moving by the change in tokens.

use candle::{DType, Device};
use candle_nn::kv_cache::SealedSequence;

use super::batched_inference::{BatchedConfig, BatchedInferenceSession, KvLayers};

const LAYERS: usize = 3;

fn session() -> BatchedInferenceSession {
    BatchedInferenceSession::new(
        KvLayers::stream_only(LAYERS),
        1,
        16,
        &Device::Cpu,
        BatchedConfig::default()
            .with_initial_seq_len(32)
            .with_dtype(DType::F32),
    )
    .expect("CPU session")
}

/// A sealed piece of `tokens` tokens, recorded off a slot of its own on every
/// layer.
fn piece(s: &mut BatchedInferenceSession, tokens: usize) -> Vec<SealedSequence> {
    let src = s.create_sequence().expect("slot");
    s.ensure_capacity(&[src], tokens).expect("capacity");
    for b in s.backings() {
        b.set_len(src, tokens);
    }
    s.advance_sequence(src, tokens).expect("advance");
    s.snapshot_sequence_per_layer(src).expect("snapshot")
}

/// Every layer's chunk token counts for `seq` — the same on every layer, or
/// the splice desynced them.
fn layout(s: &BatchedInferenceSession, seq: usize) -> Vec<usize> {
    let per_layer = s.snapshot_sequence_per_layer(seq).expect("snapshot");
    let first: Vec<usize> = per_layer[0]
        .chunks
        .iter()
        .map(|c| c.token_count as usize)
        .collect();
    for (li, l) in per_layer.iter().enumerate() {
        let here: Vec<usize> = l.chunks.iter().map(|c| c.token_count as usize).collect();
        assert_eq!(here, first, "layer {li} parted from layer 0");
    }
    first
}

/// `[a | b | c]` with `b` replaced by `d` on every layer: the slot reads
/// `[a | d | c]`, the offset moves by `d − b`, and the range names where `d`
/// now sits.
#[test]
fn a_piece_spliced_into_the_middle_lands_on_every_layer() {
    let mut s = session();
    let (a, b, c, d) = (
        piece(&mut s, 40),
        piece(&mut s, 7),
        piece(&mut s, 3),
        piece(&mut s, 45),
    );
    let dst = s.create_sequence().expect("dst");
    for p in [&a, &b, &c] {
        s.inject_sealed_at_tail(dst, p).expect("inject");
    }
    assert_eq!(layout(&s, dst), [32, 8, 7, 3]);
    assert_eq!(s.sequence_offset(dst), Some(50));

    let placed = s.splice_sealed(dst, 2, 3, &d).expect("splice");
    assert_eq!(placed.range, (2, 4), "d's two chunks sit where b was");
    assert_eq!(placed.tokens, 45);
    assert!(!placed.was_cut());
    assert_eq!(layout(&s, dst), [32, 8, 32, 13, 3]);
    assert_eq!(s.sequence_offset(dst), Some(50 - 7 + 45));
}

/// Evicting a range moves the offset down by its tokens; inserting at the
/// front moves every block after it.
#[test]
fn a_splice_evicts_and_inserts_anywhere() {
    let mut s = session();
    let (a, b, c) = (piece(&mut s, 5), piece(&mut s, 7), piece(&mut s, 3));
    let dst = s.create_sequence().expect("dst");
    for p in [&a, &b, &c] {
        s.inject_sealed_at_tail(dst, p).expect("inject");
    }
    let none: Vec<SealedSequence> = b
        .iter()
        .map(|l| SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: l.chunk_size,
            location: l.location,
        })
        .collect();
    s.splice_sealed(dst, 1, 2, &none).expect("evict");
    assert_eq!(layout(&s, dst), [5, 3]);
    assert_eq!(s.sequence_offset(dst), Some(8));

    let placed = s.splice_sealed(dst, 0, 0, &b).expect("insert");
    assert_eq!(placed.range, (0, 1));
    assert_eq!(layout(&s, dst), [7, 5, 3]);
    assert_eq!(s.sequence_offset(dst), Some(15));
}

/// A splice the backing refuses — a range reaching into the writer's chunks —
/// leaves the slot and its offset as they stood.
#[test]
fn a_refused_splice_leaves_every_layer_as_it_stood() {
    let mut s = session();
    let a = piece(&mut s, 5);
    let dst = s.create_sequence().expect("dst");
    s.inject_sealed_at_tail(dst, &a).expect("inject");
    s.ensure_capacity(&[dst], 40).expect("writer chunk");
    let before = layout(&s, dst);
    let blocks = s.sequence_block_count(dst).expect("blocks");
    assert!(blocks > before.len(), "a writer chunk past the sealed one");
    assert!(
        s.splice_sealed(dst, 0, blocks, &a).is_err(),
        "the writer's chunks are not sealed history"
    );
    assert_eq!(layout(&s, dst), before);
    assert_eq!(s.sequence_block_count(dst), Some(blocks));
    assert_eq!(s.sequence_offset(dst), Some(5));
}
