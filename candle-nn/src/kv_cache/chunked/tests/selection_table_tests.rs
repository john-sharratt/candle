//! Tests for the selection table's `(chunk, head)` row layout.
//!
//! The table is keyed by `chunk_idx * n_kv_head + head_idx`, and each of a
//! row's four palette sub-entries describes **one band** — populated from that
//! band's own gid (base pointer, slot stride) and the chunk's own format tag.
//!
//! Before the re-index the row was keyed by arena and all four sub-entries were
//! written identically, so per-band format variety could only come from
//! `arena_idx` differing per band. Size classes put bands of different formats
//! in one region, which collapses `arena_idx` — so the sub-entry dimension has
//! to carry the variety instead. These tests pin that contract from the host
//! side; the model gate exercises the kernel half.
//!
//! See `docs/archived/arena_unification.md` §2 and §5 step 1.

use std::sync::Arc;

use candle::{DType, Device, Tensor};

use crate::kv_cache::arena_table::{ArenaFormatTag, PaletteSubEntry, PerHeadEntry, N_PALETTE};
use crate::kv_cache::chunked::head_gids::ChunkBands;
use crate::kv_cache::chunked::ChunkedKvBacking;

/// A CPU backing with one written sequence, so at least one arena exists and
/// the live chunk's gids resolve to a real slot stride.
fn backing_with_one_chunk() -> ChunkedKvBacking {
    let backing = ChunkedKvBacking::new(1, 2, 32, DType::BF16, &Device::Cpu, 64).unwrap();
    backing.alloc_sequence().unwrap();
    let k = Tensor::ones((1, 2, 32, 32), DType::BF16, &Device::Cpu).unwrap();
    let v = Tensor::ones((1, 2, 32, 32), DType::BF16, &Device::Cpu).unwrap();
    backing.write_contiguous(0, 0, &k, &v).unwrap();
    backing
}

/// Decode the `(k_tag, v_tag)` pair out of a sub-entry's packed metadata
/// column, mirroring `PaletteSubEntry::to_cols`.
fn tags_at(row: &[i64], palette: usize) -> (u8, u8) {
    let meta = row[palette * PaletteSubEntry::COLS + 6];
    (((meta >> 16) & 0xFF) as u8, ((meta >> 8) & 0xFF) as u8)
}

/// **The load-bearing assertion for the re-index.** Give one chunk four
/// different K formats and four different V formats across its palette bands,
/// and the row must report all eight distinctly.
///
/// A table built from arena state cannot pass this: the bands share an arena,
/// so it would report one format four times.
#[test]
fn every_palette_sub_entry_carries_its_own_band_format() {
    let backing = backing_with_one_chunk();
    let sealed = backing
        .live_chunks_as_sealed(0)
        .expect("sequence should have a live chunk");
    let chunk = sealed.first().expect("one chunk");
    let n_kv_head = backing.n_kv_head();

    // Deliberately non-adjacent tags, so an off-by-one in the sub-entry
    // indexing shows up as an obvious mismatch rather than a plausible one.
    let k_tags = [
        ArenaFormatTag::Q8_0,
        ArenaFormatTag::Q4_KS,
        ArenaFormatTag::R16,
        ArenaFormatTag::Q2_0,
    ];
    let v_tags = [
        ArenaFormatTag::F16,
        ArenaFormatTag::Q5_1,
        ArenaFormatTag::Q3_0,
        ArenaFormatTag::Q0,
    ];
    let bands = ChunkBands {
        gids: chunk.gids.clone(),
        k_fmt: Arc::new(
            (0..n_kv_head)
                .flat_map(|_| k_tags.iter().map(|t| t.as_u8()))
                .collect(),
        ),
        v_fmt: Arc::new(
            (0..n_kv_head)
                .flat_map(|_| v_tags.iter().map(|t| t.as_u8()))
                .collect(),
        ),
    };

    let table = backing
        .inner
        .per_head_table_host(std::slice::from_ref(&bands))
        .unwrap();
    assert_eq!(
        table.len(),
        n_kv_head * PerHeadEntry::COLS,
        "one row per (chunk, head)"
    );

    for h in 0..n_kv_head {
        let row = &table[h * PerHeadEntry::COLS..(h + 1) * PerHeadEntry::COLS];
        for p in 0..N_PALETTE {
            assert_eq!(
                tags_at(row, p),
                (k_tags[p].as_u8(), v_tags[p].as_u8()),
                "head {h} palette {p} must carry its own band's formats"
            );
        }
    }
}

/// Rows are laid out `chunk_idx * n_kv_head + head_idx` — the index the kernel
/// computes. Two chunks with different tags must land in different rows.
#[test]
fn rows_are_keyed_by_chunk_then_head() {
    let backing = backing_with_one_chunk();
    let sealed = backing.live_chunks_as_sealed(0).unwrap();
    let chunk = sealed.first().unwrap();
    let n_kv_head = backing.n_kv_head();
    let per_chunk = n_kv_head * N_PALETTE;

    let mk = |tag: ArenaFormatTag| ChunkBands {
        gids: chunk.gids.clone(),
        k_fmt: Arc::new(vec![tag.as_u8(); per_chunk]),
        v_fmt: Arc::new(vec![tag.as_u8(); per_chunk]),
    };
    let chunks = [mk(ArenaFormatTag::Q8_0), mk(ArenaFormatTag::Q4_0)];

    let table = backing.inner.per_head_table_host(&chunks).unwrap();
    assert_eq!(table.len(), 2 * n_kv_head * PerHeadEntry::COLS);

    for (ci, tag) in [ArenaFormatTag::Q8_0, ArenaFormatTag::Q4_0]
        .into_iter()
        .enumerate()
    {
        for h in 0..n_kv_head {
            let off = (ci * n_kv_head + h) * PerHeadEntry::COLS;
            let row = &table[off..off + PerHeadEntry::COLS];
            for p in 0..N_PALETTE {
                assert_eq!(
                    tags_at(row, p),
                    (tag.as_u8(), tag.as_u8()),
                    "row for chunk {ci} head {h} palette {p}"
                );
            }
        }
    }
}

/// Every sub-entry resolves its band's address independently. On a CPU backing
/// the base pointer is 0, but the slot stride is real: for this fixture's
/// BF16 arena it is `CHUNK_SIZE(32) × sub_head_dim(32/4 = 8) × 2 bytes`.
#[test]
fn sub_entries_carry_a_real_slot_stride() {
    let backing = backing_with_one_chunk();
    let sealed = backing.live_chunks_as_sealed(0).unwrap();
    let chunk = sealed.first().unwrap();
    let n_kv_head = backing.n_kv_head();
    let bands = ChunkBands::from_sealed(chunk);

    let table = backing
        .inner
        .per_head_table_host(std::slice::from_ref(&bands))
        .unwrap();
    for h in 0..n_kv_head {
        let row = &table[h * PerHeadEntry::COLS..(h + 1) * PerHeadEntry::COLS];
        for p in 0..N_PALETTE {
            let base = p * PaletteSubEntry::COLS;
            // The stride the kernel steps by is the arena's **class** stride,
            // not the band's payload (`docs/archived/arena_unification.md` invariant 8).
            // This fixture's BF16 band is 32 tokens x 8 dims x 2 B = 512 B,
            // which is its own rung, so here the two happen to coincide —
            // `every_format_above_the_catch_all_lands_exactly` is why. The row
            // must carry the class stride regardless, which is what asserting
            // against the looked-up class (not the literal) checks.
            let cls = crate::kv_cache::chunked::size_class::class_for_format(
                crate::kv_cache::KvFormat::Float(candle::DType::BF16),
                32 * 8,
            )
            .unwrap();
            assert_eq!(cls.bytes(), 512);
            let stride = cls.bytes() as i64;
            assert_eq!(row[base + 4], stride, "head {h} palette {p} K stride");
            assert_eq!(row[base + 5], stride, "head {h} palette {p} V stride");
            assert_eq!(
                row[base + 2],
                0,
                "byte offsets are zero: a slot is one band"
            );
            assert_eq!(
                row[base + 3],
                0,
                "byte offsets are zero: a slot is one band"
            );
        }
    }
}

/// An empty job list produces an empty table, not a row of garbage.
#[test]
fn empty_job_list_yields_an_empty_table() {
    let backing = backing_with_one_chunk();
    assert!(backing.inner.per_head_table_host(&[]).unwrap().is_empty());
}
