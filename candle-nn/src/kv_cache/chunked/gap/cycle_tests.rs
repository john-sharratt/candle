//! The gap cycle on the device (`docs/kv_quantize_in_the_gap.md` §3): gap
//! after gap, catch-up then quantize, every complete block of every live
//! sequence comes out as today's seal makes it — however small the gap's
//! table, with nothing taken while a gap is outstanding, and with a block whose
//! bands were relocated under its gap quantized afresh rather than installed.

use std::sync::Arc;
use std::time::Duration;

use candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async;
use candle::DType;

use super::super::compress::band_ptr;
use super::super::compression_policy::CompressionPolicy;
use super::super::head_gids::HeadGids;
use super::cycle::{GapCycle, GapSeq};
use super::demand::GapKind;
use super::install_tests::{assert_installed, live, live_bands, Live};
use super::launch_tests::{write_turn, LAYERS};
use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::{
    compact_backings, CompactionRefused, CompactionReport, KvFormat, QuantFormat,
};

/// `slot` with every layer under `policy`.
fn under(slot: usize, policy: &CompressionPolicy) -> Vec<GapSeq> {
    vec![GapSeq {
        seq: slot,
        layer_policy: vec![*policy; LAYERS],
    }]
}

/// Run gaps — catch-up, then quantize — until a quantize takes nothing and
/// none is outstanding, synchronising between them. The gaps it took.
fn run_until_quiet(cycle: &mut GapCycle, l: &Live, policy: &CompressionPolicy) -> usize {
    let seqs = under(l.slot, policy);
    for gaps in 0..16 {
        cycle.catch_up(&l.layers).unwrap();
        let q = cycle.quantize(&l.layers, &seqs, GapKind::Prefill).unwrap();
        if q.jobs == 0 && !cycle.is_outstanding() {
            return gaps;
        }
        cycle.wait().unwrap();
    }
    panic!("the cycle did not settle in 16 gaps");
}

#[test]
fn a_cycle_quantizes_every_complete_block_as_today() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let mut cycle = GapCycle::new(&l.layers, 1024).unwrap();
    let gaps = run_until_quiet(&mut cycle, &l, &policy);
    assert_eq!(gaps, 1, "one gap holds the whole table");
    for blk in 0..3 {
        assert_installed(&l, blk);
    }
    assert_eq!(live_bands(&l, 0..3), l.want);
}

/// Two sequences in one gap, each under its own policy: the adaptive one comes
/// out as today's seal makes it, the forced one in its forced K format, and a
/// sequence the gap is not handed stays native.
#[test]
fn each_sequence_is_quantized_under_its_own_policy() {
    let adaptive = CompressionPolicy::new(5);
    let Some(l) = live(&adaptive) else {
        return;
    };
    let forced = CompressionPolicy::new(5).with_override_k_quant(Some(QuantFormat::Q8_0));
    let second_slot = |base: u32| -> usize {
        let slots: Vec<usize> = l
            .layers
            .iter()
            .enumerate()
            .map(|(i, b)| write_turn(b, &l.device, (i as u32 + 1) * base))
            .collect();
        assert!(slots.iter().all(|&s| s == slots[0]));
        slots[0]
    };
    let other = second_slot(1000);
    let native = second_slot(3000);
    let mut seqs = under(l.slot, &adaptive);
    seqs.extend(under(other, &forced));
    let mut cycle = GapCycle::new(&l.layers, 1024).unwrap();
    for _ in 0..16 {
        cycle.catch_up(&l.layers).unwrap();
        let q = cycle.quantize(&l.layers, &seqs, GapKind::Prefill).unwrap();
        if q.jobs == 0 && !cycle.is_outstanding() {
            break;
        }
        cycle.wait().unwrap();
    }
    assert_eq!(live_bands(&l, 0..3), l.want);
    let q8 = KvFormat::Quantized(QuantFormat::Q8_0);
    for b in &l.layers {
        let state = b.state.read().unwrap();
        let seq = state.sequences[other].as_ref().unwrap();
        for blk in 0..3 {
            let c = &seq.chunks_slice()[blk];
            assert!(!c.takes_writes(), "layer {} block {blk}", b.layer_idx);
            assert!(
                c.k_fmt.iter().all(|&t| KvFormat::from_tag(t) == Some(q8)),
                "layer {} block {blk}: K forced to Q8_0",
                b.layer_idx
            );
        }
        let untouched = state.sequences[native].as_ref().unwrap();
        for blk in 0..3 {
            assert!(
                untouched.chunks_slice()[blk].takes_writes(),
                "layer {} block {blk}: the sequence left native stays native",
                b.layer_idx
            );
        }
    }
}

/// A table smaller than the blocks complete takes what fits and moves the
/// cursor past only that: the rest is the next gap's, and every block ends up
/// quantized.
#[test]
fn a_table_cut_to_the_gaps_room_takes_the_rest_next_gap() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    // Room for one block on every layer.
    let mut cycle = GapCycle::new(&l.layers, LAYERS).unwrap();
    let gaps = run_until_quiet(&mut cycle, &l, &policy);
    assert_eq!(gaps, 3, "one block a gap");
    assert_eq!(live_bands(&l, 0..3), l.want);
}

/// While a gap is outstanding, a quantize takes nothing: no cursor moves.
#[test]
fn nothing_is_taken_while_a_gap_is_outstanding() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let seqs = under(l.slot, &policy);
    let mut cycle = GapCycle::new(&l.layers, LAYERS).unwrap();
    let first = cycle.quantize(&l.layers, &seqs, GapKind::Prefill).unwrap();
    assert_eq!(first.jobs, LAYERS);
    let again = cycle.quantize(&l.layers, &seqs, GapKind::Prefill).unwrap();
    assert!(again.outstanding && again.jobs == 0);
    let cursor = l.layers[0].gap_cursor_for_seq(l.slot).unwrap();
    assert_eq!(cursor, 1, "only the first gap's block was taken");
}

/// A block whose native bands are relocated while its gap runs — what a
/// compaction does — is not installed from the bytes the gap read: it is
/// re-queued with the bands it holds now and quantized by the next gap.
#[test]
fn a_block_relocated_under_its_gap_is_quantized_afresh() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let seqs = under(l.slot, &policy);
    let mut cycle = GapCycle::new(&l.layers, 1024).unwrap();
    cycle.quantize(&l.layers, &seqs, GapKind::Prefill).unwrap();
    relocate(&l, 0, 1);
    cycle.wait().unwrap();
    let done = cycle.catch_up(&l.layers).unwrap();
    assert!(done.caught_up);
    assert_eq!(done.installed, 3 * LAYERS - 1);
    assert_eq!(
        done.deferred, 1,
        "the relocated block waits for the next gap"
    );
    let b = &l.layers[0];
    {
        let state = b.state.read().unwrap();
        let seq = state.sequences[l.slot].as_ref().unwrap();
        assert!(
            seq.chunks_slice()[1].takes_writes(),
            "not installed from old bytes"
        );
    }
    let next = cycle.quantize(&l.layers, &seqs, GapKind::Prefill).unwrap();
    assert_eq!((next.jobs, next.retried), (1, 1));
    cycle.wait().unwrap();
    let last = cycle.catch_up(&l.layers).unwrap();
    assert_eq!(last.installed, 1);
    for blk in 0..3 {
        assert_installed(&l, blk);
    }
    assert_eq!(
        live_bands(&l, 0..3),
        l.want,
        "the afresh quantize is today's too"
    );
}

/// Relocate `layer`'s block `block` of the live sequence: fresh native slots,
/// its bytes copied over on the device's stream, as a compaction copies —
/// waited for.
fn relocate(l: &Live, layer: usize, block: usize) {
    let b = &l.layers[layer];
    let f16 = b
        .inner
        .arena_key_for(KvFormat::Float(DType::F16), ArenaLocation::Gpu)
        .unwrap();
    let band_bytes = b.elems_per_chunk() * DType::F16.size_in_bytes();
    let stream = l.cuda.cuda_stream();
    {
        let mut state = b.state.write().unwrap();
        let seq = state.sequences[l.slot].as_mut().unwrap();
        let old = seq.chunks_slice()[block].clone();
        let fresh: Vec<_> = old
            .gids
            .as_slice()
            .iter()
            .map(|g| {
                let n = b.alloc_chunk_for_key(f16).unwrap();
                let (src, dst) = b
                    .inner
                    .storage
                    .read(|s| {
                        (
                            band_ptr(s, g, "src").unwrap(),
                            band_ptr(s, &n, "dst").unwrap(),
                        )
                    })
                    .unwrap();
                // SAFETY: two band slots of one native class, both live.
                unsafe { memcpy_dtod_async(dst, src, band_bytes, stream.cu_stream()).unwrap() };
                n
            })
            .collect();
        seq.set_block_gids(
            block,
            HeadGids::from_vec(fresh),
            old.k_pal.clone(),
            old.v_pal.clone(),
            old.k_scale.clone(),
            old.v_scale.clone(),
            old.k_fmt.clone(),
            old.v_fmt.clone(),
        );
    }
    l.device.synchronize().unwrap();
}

/// A gap whose block was relocated under it, caught up: layer 0's block 1
/// stands deferred, every other block installed.
fn one_deferred(l: &Live, policy: &CompressionPolicy) -> GapCycle {
    let mut cycle = GapCycle::new(&l.layers, 1024).unwrap();
    cycle
        .quantize(&l.layers, &under(l.slot, policy), GapKind::Prefill)
        .unwrap();
    relocate(l, 0, 1);
    cycle.wait().unwrap();
    let done = cycle.catch_up(&l.layers).unwrap();
    assert_eq!((done.installed, done.deferred), (3 * LAYERS - 1, 1));
    cycle
}

/// A pass over `l`'s pools, its holders these backings alone, and what it
/// answered.
fn pass(l: &Live, cycle: &GapCycle) -> Result<CompactionReport, CompactionRefused> {
    compact_backings(
        &l.layers,
        Some(cycle),
        Duration::from_millis(80),
        &mut |_| Ok(()),
    )
}

/// A pass that ran relocated nothing it could not name; a refusal is not the
/// gap's.
fn assert_witnessed(outcome: Result<CompactionReport, CompactionRefused>) {
    match outcome {
        Ok(report) => assert_eq!(report.unwitnessed, 0, "{report:?}"),
        Err(e) => assert_ne!(e, CompactionRefused::GapOutstanding),
    }
}

/// A pass refuses while a gap is outstanding — its lease and its jobs hold
/// slots no block table names — and runs once the gap is caught up, with
/// every slot it relocates named by a holder it swept.
#[test]
fn a_pass_refuses_while_a_gap_is_outstanding() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let mut cycle = GapCycle::new(&l.layers, 1024).unwrap();
    let q = cycle
        .quantize(&l.layers, &under(l.slot, &policy), GapKind::Prefill)
        .unwrap();
    assert_eq!(q.jobs, 3 * LAYERS);
    assert!(cycle.is_outstanding());
    assert!(matches!(
        pass(&l, &cycle),
        Err(CompactionRefused::GapOutstanding)
    ));
    cycle.wait().unwrap();
    let done = cycle.catch_up(&l.layers).unwrap();
    assert_eq!(done.installed, 3 * LAYERS);
    assert!(!cycle.is_outstanding());
    assert_witnessed(pass(&l, &cycle));
}

/// A deferred block is kept by where it stands and holds none of its bands:
/// between gaps its block table is the one holder of them, which is the
/// holder a compaction sweeps.
#[test]
fn a_deferred_block_holds_no_band_between_gaps() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let cycle = one_deferred(&l, &policy);
    assert_eq!(cycle.deferred_at(), vec![(0, l.slot, 1)]);
    {
        let state = l.layers[0].state.read().unwrap();
        let gids = &state.sequences[l.slot].as_ref().unwrap().chunks_slice()[1].gids;
        assert_eq!(
            Arc::strong_count(&gids.0),
            1,
            "the block table alone holds the deferred block's bands"
        );
    }
    assert_witnessed(pass(&l, &cycle));
}

/// A deferred block whose sequence is freed before its next gap goes with
/// the sequence: nothing in the cycle keeps its slots, a pass finds nothing
/// of it to relocate, and the next gap drops it rather than retrying it.
#[test]
fn a_deferred_block_of_a_freed_sequence_lets_its_slots_go() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let mut cycle = one_deferred(&l, &policy);
    let band = {
        let state = l.layers[0].state.read().unwrap();
        state.sequences[l.slot].as_ref().unwrap().chunks_slice()[1]
            .gids
            .as_slice()[0]
            .clone()
    };
    assert_eq!(band.strong_count(), 2, "the block table and this clone");
    for b in &l.layers {
        b.free_sequence(l.slot).unwrap();
    }
    assert_eq!(
        band.strong_count(),
        1,
        "this clone alone: the cycle holds none"
    );
    drop(band);
    assert_witnessed(pass(&l, &cycle));
    let q = cycle.quantize(&l.layers, &[], GapKind::Prefill).unwrap();
    assert_eq!((q.jobs, q.retried), (0, 0), "the stale retry is dropped");
    assert!(cycle.deferred_at().is_empty());
}
