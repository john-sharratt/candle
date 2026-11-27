//! A sequence's position frontier apart from its token count
//! (`batched_inference::positions`), against a CPU session: a hole pushed
//! between two sealed pieces moves the position and not the offset, on every
//! layer alike.

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

/// Every layer's holes for `seq`: the same on every layer.
fn holes(s: &BatchedInferenceSession, seq: usize) -> usize {
    let per_layer: Vec<usize> = s
        .backings()
        .iter()
        .map(|b| b.lead_total(seq).expect("holes"))
        .collect();
    for (li, &h) in per_layer.iter().enumerate() {
        assert_eq!(h, per_layer[0], "layer {li} parted from layer 0");
    }
    per_layer[0]
}

/// **A hole between two pieces moves the position, not the offset.**
#[test]
fn a_hole_between_two_pieces_moves_the_position_not_the_offset() {
    let mut s = session();
    let (a, b) = (piece(&mut s, 40), piece(&mut s, 7));
    let dst = s.create_sequence().expect("dst");
    s.inject_sealed_at_tail(dst, &a).expect("inject a");
    assert_eq!(s.sequence_offset(dst), Some(40));
    assert_eq!(s.sequence_position(dst), Some(40));

    s.push_sequence_position_gap(dst, 100).expect("gap");
    assert_eq!(s.sequence_offset(dst), Some(40));
    assert_eq!(s.sequence_position(dst), Some(140));
    assert_eq!(holes(&s, dst), 100);

    s.inject_sealed_at_tail(dst, &b).expect("inject b");
    assert_eq!(s.sequence_offset(dst), Some(47));
    assert_eq!(s.sequence_position(dst), Some(147));
    assert_eq!(holes(&s, dst), 100);
    for backing in s.backings() {
        assert_eq!(backing.position_frontier(dst).expect("frontier"), 147);
    }
}

/// **No hole, no difference.**
#[test]
fn without_a_hole_the_position_is_the_offset() {
    let mut s = session();
    let a = piece(&mut s, 40);
    let dst = s.create_sequence().expect("dst");
    s.inject_sealed_at_tail(dst, &a).expect("inject");
    assert_eq!(s.sequence_position(dst), s.sequence_offset(dst));
    assert_eq!(holes(&s, dst), 0);
}

/// **A gap on a sequence that does not exist is refused.**
#[test]
fn a_gap_on_a_missing_sequence_is_refused() {
    let s = session();
    assert!(s.push_sequence_position_gap(5, 10).is_err());
}
