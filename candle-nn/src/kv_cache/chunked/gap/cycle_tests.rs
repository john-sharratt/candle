//! The gap cycle on the device (`docs/kv_quantize_in_the_gap.md` §3): gap
//! after gap, catch-up then quantize, every complete block of every live
//! sequence comes out as today's seal makes it — however small the gap's
//! table, with nothing taken while a gap is outstanding, and with a block whose
//! bands were relocated under its gap quantized afresh rather than installed.

use candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async;
use candle::DType;

use super::super::compress::band_ptr;
use super::super::compression_policy::CompressionPolicy;
use super::super::head_gids::HeadGids;
use super::cycle::GapCycle;
use super::demand::GapKind;
use super::install_tests::{assert_installed, live, live_bands, Live};
use super::launch_tests::LAYERS;
use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::KvFormat;

/// Run gaps — catch-up, then quantize — until a quantize takes nothing and
/// none is outstanding, synchronising between them. The gaps it took.
fn run_until_quiet(cycle: &mut GapCycle, l: &Live, policy: &CompressionPolicy) -> usize {
    let policies = vec![*policy; LAYERS];
    for gaps in 0..16 {
        cycle.catch_up(&l.layers).unwrap();
        let q = cycle
            .quantize(&l.layers, &[l.slot], &policies, GapKind::Prefill)
            .unwrap();
        if q.jobs == 0 && !cycle.is_outstanding() {
            return gaps;
        }
        l.device.synchronize().unwrap();
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
    let policies = vec![policy; LAYERS];
    let mut cycle = GapCycle::new(&l.layers, LAYERS).unwrap();
    let first = cycle
        .quantize(&l.layers, &[l.slot], &policies, GapKind::Prefill)
        .unwrap();
    assert_eq!(first.jobs, LAYERS);
    let again = cycle
        .quantize(&l.layers, &[l.slot], &policies, GapKind::Prefill)
        .unwrap();
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
    let policies = vec![policy; LAYERS];
    let mut cycle = GapCycle::new(&l.layers, 1024).unwrap();
    cycle
        .quantize(&l.layers, &[l.slot], &policies, GapKind::Prefill)
        .unwrap();
    // Relocate layer 0's block 1: fresh native slots, its bytes copied over on
    // the device's stream, as a compaction copies.
    let b = &l.layers[0];
    let f16 = b
        .inner
        .arena_key_for(KvFormat::Float(DType::F16), ArenaLocation::Gpu)
        .unwrap();
    let band_bytes = b.elems_per_chunk() * DType::F16.size_in_bytes();
    let stream = l.cuda.cuda_stream();
    {
        let mut state = b.state.write().unwrap();
        let seq = state.sequences[l.slot].as_mut().unwrap();
        let old = seq.chunks_slice()[1].clone();
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
            1,
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
    let done = cycle.catch_up(&l.layers).unwrap();
    assert!(done.caught_up);
    assert_eq!(done.installed, 3 * LAYERS - 1);
    assert_eq!(
        done.deferred, 1,
        "the relocated block waits for the next gap"
    );
    {
        let state = b.state.read().unwrap();
        let seq = state.sequences[l.slot].as_ref().unwrap();
        assert!(
            seq.chunks_slice()[1].takes_writes(),
            "not installed from old bytes"
        );
    }
    let next = cycle
        .quantize(&l.layers, &[l.slot], &policies, GapKind::Prefill)
        .unwrap();
    assert_eq!((next.jobs, next.retried), (1, 1));
    l.device.synchronize().unwrap();
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
