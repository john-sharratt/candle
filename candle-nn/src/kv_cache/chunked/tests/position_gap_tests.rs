//! Holes in a slot's position space (`sequence_ops::position_gap`): each chunk's
//! RoPE base, the position frontier and the token count, against exact values.

use candle::{DType, Device};

use crate::kv_cache::chunked::ChunkedKvBacking;

fn backing() -> ChunkedKvBacking {
    ChunkedKvBacking::new(2, 1, 32, DType::BF16, &Device::Cpu, 256).unwrap()
}

/// Every chunk's RoPE base, as the decode serialiser derives it.
fn rope_bases(backing: &ChunkedKvBacking, slot: usize) -> Vec<i32> {
    let state = backing.state.read().expect("lock");
    let seq = state.sequences[slot].as_ref().expect("slot");
    (0..seq.block_count()).map(|b| seq.rope_pos(b)).collect()
}

/// Every chunk's `(lead, usage)`.
fn shape(backing: &ChunkedKvBacking, slot: usize) -> Vec<(u32, u32)> {
    let state = backing.state.read().expect("lock");
    let seq = state.sequences[slot].as_ref().expect("slot");
    seq.chunks_slice()
        .iter()
        .map(|c| (c.lead, c.usage))
        .collect()
}

/// **A hole on an empty slot is pending until a chunk takes it.**
#[test]
fn a_hole_on_an_empty_slot_waits_for_the_first_chunk() {
    let b = backing();
    b.alloc_sequence().unwrap();
    b.push_position_gap(0, 10).unwrap();
    assert_eq!(b.lead_total(0).unwrap(), 10);
    assert_eq!(b.position_frontier(0).unwrap(), 10);
    b.reserve_glue_gap_chunk(0, 5).unwrap();
    // The 5-token chunk takes the hole: it is read at 10..15; the empty
    // writer after it starts at 15.
    assert_eq!(shape(&b, 0), vec![(10, 5), (0, 0)]);
    assert_eq!(rope_bases(&b, 0), vec![10, 15]);
    assert_eq!(b.position_frontier(0).unwrap(), 15);
    assert_eq!(b.lead_total(0).unwrap(), 10);
}

/// **A hole after tokens closes the tail and opens a writer behind the hole.**
#[test]
fn a_hole_after_tokens_opens_a_writer_behind_it() {
    let b = backing();
    b.alloc_sequence().unwrap();
    b.reserve_glue_gap_chunk(0, 3).unwrap(); // [0, 3), then an empty writer
    b.reserve_glue_gap_chunk(0, 4).unwrap(); // [3, 7), then an empty writer
    b.push_position_gap(0, 100).unwrap();
    // The empty writer tail takes the hole as it stands.
    assert_eq!(shape(&b, 0), vec![(0, 3), (0, 0), (0, 4), (100, 0)]);
    assert_eq!(rope_bases(&b, 0), vec![0, 3, 3, 107]);
    assert_eq!(b.position_frontier(0).unwrap(), 107);
    b.reserve_glue_gap_chunk(0, 2).unwrap(); // [107, 109)
    assert_eq!(
        shape(&b, 0),
        vec![(0, 3), (0, 0), (0, 4), (100, 0), (0, 2), (0, 0)]
    );
    assert_eq!(rope_bases(&b, 0), vec![0, 3, 3, 107, 107, 109]);
    assert_eq!(b.position_frontier(0).unwrap(), 109);
    assert_eq!(b.lead_total(0).unwrap(), 100);
}

/// **Holes add up, and a truncate drops the hole pending after the old tail.**
#[test]
fn holes_add_up_and_a_truncate_drops_the_pending_one() {
    let b = backing();
    b.alloc_sequence().unwrap();
    b.push_position_gap(0, 7).unwrap();
    b.push_position_gap(0, 9).unwrap();
    assert_eq!(b.position_frontier(0).unwrap(), 16);
    {
        let mut state = b.state.write().expect("lock");
        state.sequences[0]
            .as_mut()
            .expect("slot")
            .truncate_chunks(0);
    }
    // Nothing to truncate: the pending hole stands.
    assert_eq!(b.position_frontier(0).unwrap(), 16);
    b.reserve_glue_gap_chunk(0, 1).unwrap();
    assert_eq!(rope_bases(&b, 0), vec![16, 17]);
    {
        let mut state = b.state.write().expect("lock");
        let seq = state.sequences[0].as_mut().expect("slot");
        seq.push_lead(4); // the empty writer takes it: [17 + 4)
        seq.truncate_chunks(1);
    }
    assert_eq!(shape(&b, 0), vec![(16, 1)]);
    assert_eq!(b.position_frontier(0).unwrap(), 17);
}

/// **Trimming an empty tail keeps its holes pending at the new tail.**
#[test]
fn trimming_an_empty_tail_keeps_its_holes() {
    let b = backing();
    b.alloc_sequence().unwrap();
    b.reserve_glue_gap_chunk(0, 3).unwrap(); // [0, 3), then an empty writer
    b.push_position_gap(0, 20).unwrap(); // the empty writer takes it
    b.push_empty_writer_chunk(0).unwrap();
    b.push_position_gap(0, 5).unwrap(); // the second empty writer takes it
    assert_eq!(shape(&b, 0), vec![(0, 3), (20, 0), (5, 0)]);
    assert_eq!(b.position_frontier(0).unwrap(), 28);

    assert_eq!(b.trim_empty_tail(0).unwrap(), 2);
    assert_eq!(shape(&b, 0), vec![(0, 3)]);
    assert_eq!(b.position_frontier(0).unwrap(), 28);
    assert_eq!(b.lead_total(0).unwrap(), 25);

    b.reserve_glue_gap_chunk(0, 4).unwrap();
    assert_eq!(shape(&b, 0), vec![(0, 3), (25, 4), (0, 0)]);
    assert_eq!(rope_bases(&b, 0), vec![0, 28, 32]);
}

/// **No hole, no change: positions are the running token count.**
#[test]
fn without_a_hole_positions_are_the_token_count() {
    let b = backing();
    b.alloc_sequence().unwrap();
    b.reserve_glue_gap_chunk(0, 3).unwrap();
    b.reserve_glue_gap_chunk(0, 7).unwrap();
    assert_eq!(rope_bases(&b, 0), vec![0, 3, 3, 10]);
    assert_eq!(b.position_frontier(0).unwrap(), 10);
    assert_eq!(b.lead_total(0).unwrap(), 0);
}
