//! Does dropping a conversation give its memory back?
//!
//! `free_sequence` releases a slot's GIDs by RAII, and the chunks return to the
//! pool. That is the *chunk* level, and it is not the level anyone cares about:
//! a region comes back only when an arena is **wholly** empty, because that is
//! what `has_reclaimable` tests —
//!
//! ```ignore
//! fn has_reclaimable(&self) -> bool {
//!     tables.values().any(|t| t.live_count() == 0)
//! }
//! ```
//!
//! One surviving chunk therefore pins a whole arena. So "the drop path works"
//! and "the memory came back" are separate claims, and only the first has ever
//! been tested. These assert the second.
//!
//! # Why this is worth its own file
//!
//! Measured on the 27B gate: `live` regions never fall. A single-context config
//! holds 21 regions when it runs first and 84 when it follows a wider one, and
//! after a 20-context config it sits at 404 — where the next config, running a
//! quarter of the load, still reports 405. The weight zone is squeezed to half
//! its size for the rest of the run and every subsequent row is slower.
//!
//! Two candidate causes, and they call for completely different fixes: the drop
//! path is broken, or the drop path works and the arenas are **fragmented**.
//! These tests separate them, and the answer is neither — freeing every sequence
//! returns every arena, and freeing half returns half (measured: 6 arenas
//! occupied, 3 held). So the drop path works and the pool packs well enough that
//! whole-arena reclaim is sufficient. What the gate is missing is the *drop*: it
//! never calls `free_sequence` between configs, and `prune()` clears cached
//! embedding variants rather than KV.
//!
//! # These are CPU arenas
//!
//! Which is what makes them fast and unconditional, and it is also their limit.
//! GPU arenas are pooled globally across every layer sharing a head config, so a
//! GPU pool sees an allocation interleaving these cannot produce. A pass here
//! says the *logic* is sound; it does not say the device pool fragments no worse.

use crate::kv_cache::chunked::backing::ChunkedKvBacking;
use candle::{DType, Device, Tensor};

/// Heads and head-dim per chunk.
///
/// **Chosen so the pool spans several arenas, not one.** An arena holds a
/// region's worth of chunks, so a small head config fits thousands and the whole
/// fixture lands in a single arena — where every question these tests ask has a
/// trivial answer. Measured with a 4×32 config: six sequences of 512 tokens
/// occupied exactly **one** arena, and the fragmentation assertion below passed
/// while observing nothing at all. At 16×128 a chunk is ~256 KiB, so a few
/// thousand tokens spread across a handful of arenas, which is the regime the
/// whole-arena reclaim rule actually has to cope with.
const HEADS: usize = 16;
const HEAD_DIM: usize = 128;

/// A backing with room for several sequences.
fn backing() -> ChunkedKvBacking {
    ChunkedKvBacking::new(8, HEADS, HEAD_DIM, DType::BF16, &Device::Cpu, 8192).unwrap()
}

/// Fill `batch_idx` with `tokens` tokens of KV.
fn seed(b: &ChunkedKvBacking, batch_idx: usize, tokens: usize) {
    let k = Tensor::ones((1, HEADS, tokens, HEAD_DIM), DType::BF16, &Device::Cpu).unwrap();
    let v = Tensor::ones((1, HEADS, tokens, HEAD_DIM), DType::BF16, &Device::Cpu).unwrap();
    b.write_contiguous(batch_idx, 0, &k, &v).unwrap();
    b.set_len(batch_idx, tokens);
}

/// A fixture that occupies one arena cannot answer any question these tests ask,
/// so every one of them checks it first.
fn assert_spans_several_arenas(occupied: usize, what: &str) {
    assert!(
        occupied >= 3,
        "{what}: the fixture occupies {occupied} arena(s) — too few for the \
         whole-arena reclaim rule to be under test at all"
    );
}

/// **The whole point: a conversation that ends gives its arenas back.**
///
/// Every sequence is freed, so nothing is live, so *every* arena is wholly empty
/// and must be reclaimable. If this fails, the drop path itself is broken and no
/// amount of compaction or demotion would help.
#[test]
fn freeing_every_sequence_empties_every_arena() {
    let b = backing();
    let base = b.arena_count().unwrap();

    let mut slots = Vec::new();
    for _ in 0..6 {
        slots.push(b.alloc_sequence().unwrap());
    }
    for &s in &slots {
        seed(&b, s, 2048);
    }
    let peak = b.arena_count().unwrap();
    assert_spans_several_arenas(peak - base, "freeing_every_sequence");

    for &s in &slots {
        b.free_sequence(s).unwrap();
    }
    b.release_empty_arenas().unwrap();

    let after = b.arena_count().unwrap();
    assert_eq!(
        after,
        base,
        "every sequence was freed and {} of {peak} arenas are still held — \
         dropping a conversation is not returning its memory",
        after - base
    );
}

/// **A row that finishes should leave the pool near-empty**, which is the
/// property a gate can assert between configs.
///
/// Stated as occupancy rather than as an exact count so it says something about
/// the *shape* of the answer: after the workload that filled the pool has gone,
/// what is left must be a rounding error, not a fraction.
#[test]
fn a_finished_workload_leaves_no_meaningful_occupancy() {
    let b = backing();
    let base = b.arena_count().unwrap();

    for round in 0..3 {
        let mut slots = Vec::new();
        for _ in 0..4 {
            slots.push(b.alloc_sequence().unwrap());
        }
        for &s in &slots {
            seed(&b, s, 2048);
        }
        assert_spans_several_arenas(b.arena_count().unwrap() - base, "a_finished_workload round");
        for &s in &slots {
            b.free_sequence(s).unwrap();
        }
        b.release_empty_arenas().unwrap();

        let after = b.arena_count().unwrap();
        assert_eq!(
            after,
            base,
            "round {round}: {} arenas outlived the workload that created them — \
             a later round starts with less memory than the one before it, which \
             is the ratchet",
            after - base
        );
    }
}

/// **Fragmentation, isolated.** Free *half* the sequences and ask what is left.
///
/// This is the case the whole-empty reclaim rule cannot serve: the survivors are
/// spread across the arenas the departed shared, so almost nothing is wholly
/// empty and almost nothing comes back — even though the live chunks would fit
/// in a fraction of the arenas still held.
///
/// The assertion is deliberately generous: what is held afterwards must be
/// within a factor of two of what the live chunks actually need. A tighter bound
/// would be a claim about the allocator's packing; this is a claim that the pool
/// is not holding an order of magnitude more than it uses.
#[test]
fn freeing_half_the_sequences_returns_a_proportionate_share() {
    let b = backing();
    let base = b.arena_count().unwrap();

    let mut slots = Vec::new();
    for _ in 0..6 {
        slots.push(b.alloc_sequence().unwrap());
    }
    // One write per sequence. A loop of repeated `seed` calls looks like an
    // interleave and is not: `write_contiguous` starts at offset 0 every time,
    // so the later rounds overwrite the earlier ones and the sequence ends at
    // the last round's length — which is how the first version of this fixture
    // came to occupy two arenas while appearing to write four times as much.
    for &s in &slots {
        seed(&b, s, 2048);
    }
    let full = b.arena_count().unwrap();
    assert_spans_several_arenas(full - base, "freeing_half");

    // Half the conversations end.
    for &s in slots.iter().step_by(2) {
        b.free_sequence(s).unwrap();
    }
    b.release_empty_arenas().unwrap();
    let held = b.arena_count().unwrap() - base;
    let occupied = full - base;

    // Half the conversations ended, so about half the arenas should come back.
    // `+ 1` for the rounding when an odd survivor shares a partly-filled arena.
    assert!(
        held <= occupied / 2 + 1,
        "half the sequences ended and {held} of {occupied} arenas are still held — \
         the survivors' chunks are spread across arenas the departed shared, so \
         whole-arena reclaim frees almost nothing. This is the fragmentation a \
         compaction pass would recover; `GidPool::reclaimable_arenas` already \
         computes the amount and has no consumer."
    );
}
