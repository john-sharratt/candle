//! The size-class ladder: fixed-stride slot sizes shared by every KV format.
//!
//! A chunk slot's stride is chosen from a small fixed ladder rather than from
//! the format that happens to occupy it. A chunk of format `F` takes one slot
//! of the smallest class whose byte size is at least `F`'s payload, and the
//! trailing pad is never read as data.
//!
//! This is the single mechanism behind cross-format fungibility: two chunks of
//! different formats that land in the same class share one region and one free
//! list, so reclaiming a slot from one format makes it immediately available to
//! the other. Under per-format arenas that was impossible — a free Q4_0 slot
//! was invisible to a Q8_0 allocation however much of it there was.
//!
//! # Coverage is over the format space, not the policy subset
//!
//! The ladder covers all 22 [`QuantFormat`] variants and all four float
//! dtypes, deliberately — not just the formats the current
//! [`CompressionPolicy`](super::CompressionPolicy) selects. The threshold
//! tables are provisional and re-derived per model, the `override_k_quant` /
//! `override_v_quant` knobs can force any format at any time, and a cold-loaded
//! chunk carries whatever tag was persisted, possibly by an earlier policy. A
//! class costs one row in a `const` table; a missing class costs 30–40 % pad on
//! every chunk of that format, forever.
//!
//! # Two lengths, never conflated
//!
//! [`payload_bytes`] is what the format actually occupies — every copy length,
//! every persisted image, every checksum. [`SizeClass::bytes`] is the address
//! stride — `base + idx * stride`, and the extent that must be zeroed on
//! recycle because the next tenant may be any format. Conflating them either
//! moves pad over PCIe and grows on-disk images, or addresses the wrong slot.
//! See `docs/archived/arena_unification.md` invariant 8.

use candle::DType;

use crate::kv_cache::arena_table::ArenaFormatTag;
use crate::kv_cache::{KvFormat, QuantFormat};

use super::types::{CHUNK_SIZE, TARGET_ARENA_BYTES};

/// Slot strides, ascending: **coarse where stranding dominates, exact where
/// bandwidth does.**
///
/// Above the bottom rung there is one rung per distinct payload, so nothing
/// rounds up. The stride is what the attention and selection kernels step by,
/// so a padded slot is re-read on every pass over the cache for as long as the
/// chunk lives — there is nothing to buy by rounding a known size to a prettier
/// one. The bottom rung is the deliberate exception: eight formats from 32 B to
/// 288 B share it, because each would otherwise stamp a whole 16 MiB region for
/// a trickle of chunks (§3.10 problem 3). A12 governs where that line sits.
///
/// Derived from the C-level formats first, which left the two fixed ones
/// rounding up — `Q8_0` 1088→1152 (5.6 %) and `Q4_0` 576→640 (10.0 %). Those
/// were the gate's only two configs below the pre-unification baseline, five
/// runs of five, at the width where KV bandwidth dominates. Adding their rungs
/// moved both above it. See `docs/archived/arena_unification_results.md`, step 6.
///
/// # Head dims other than 128
///
/// The rungs above were sized for the production palette-4 geometry —
/// `head_dim 128 / N_PALETTE 4 = 32`, so `CHUNK_SIZE(32) * 32 = 1024` elements
/// a slot. A slot's payload scales with `head_dim`, so the supported dims
/// {64, 96, 128, 256, 512} need coverage up to `head_dim 512`, where `R16` and
/// `F32` occupy 16384 B. Only the top rungs are added: the intermediate
/// payloads at those widths round up into existing rungs, which wastes
/// bandwidth on a non-production geometry rather than adding classes that
/// would strand regions on the production one.
/// `every_kv_format_maps_to_a_class` pins coverage at *every* supported dim —
/// it pinned only 128 before, which is how a chunk of `R16` at `head_dim 256`
/// came to have nowhere to live.
///
/// `head_dim 512` is DeepSeek's latent width. The single-latent backing runs
/// its 16×32-dim bands (1024-elem slots) in steady state, but it is
/// *constructed* at the GQA width — `warm_protected_arenas` mints the writer
/// arenas at `512 / N_PALETTE = 128` dims before `set_single_latent` drops
/// them and re-mints at band width — so the construction-time R16 payload
/// (16384 B) must have a rung to pass through.
pub const LADDER: [usize; 15] = [
    320,   // catch-all: Q0, Q0_V, Q0_X, Q0_M2, Q0_M4, Q1_S, Q1_A, Q2_S, Q2_0
    384,   // Q2_1
    448,   // Q3_0
    512,   // Q3_1
    576,   // Q4_0
    640,   // Q4_KS
    704,   // Q5_0
    768,   // Q5_1
    1024,  // F8E4M3
    1088,  // Q8_0
    1152,  // Q8_KS, Q8_1
    2048,  // F16, BF16
    4096,  // R16, F32 — and F16/BF16 at head_dim 256
    8192,  // R16, F32 at head_dim 256
    16384, // R16, F32 at head_dim 512 (DeepSeek latent constructed at GQA width)
];

/// Raw-gid stride: `raw = region_idx * GID_STRIDE + chunk_idx`.
///
/// A fixed power of two so gid decode is shift/mask on both the Rust and CUDA
/// sides rather than div/mod. It must exceed the largest
/// [`SizeClass::chunks_per_region`] (52,428, from the 320 B class), which
/// [`gid_stride_exceeds_max_chunks_per_region`](tests) asserts.
pub const GID_STRIDE: usize = 1 << 16;

/// One rung of [`LADDER`], as an index into it.
///
/// Ordered by size, so `SizeClass::promote` is `+1` and the ordering is also
/// the promotion order used when a class is starved (see
/// `docs/archived/arena_unification.md` §3.4, scarcity-only promotion).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SizeClass(u8);

impl SizeClass {
    /// Number of rungs in the ladder.
    pub const COUNT: usize = LADDER.len();

    /// The rung at `index`, or `None` if past the top of the ladder.
    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        (index < Self::COUNT).then_some(Self(index as u8))
    }

    /// The rung at `index`, in a `const` context. Panics at compile time if
    /// `index` is past the top of the ladder, so a named constant class can
    /// never silently drift off the end when the ladder is edited.
    pub const fn at(index: usize) -> Self {
        assert!(index < Self::COUNT, "size class index past the ladder");
        Self(index as u8)
    }

    /// This rung's position in [`LADDER`].
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// The slot stride in bytes — address arithmetic and zero-on-recycle.
    #[inline]
    pub fn bytes(self) -> usize {
        LADDER[self.0 as usize]
    }

    /// Chunk slots in one [`TARGET_ARENA_BYTES`] region at this stride.
    #[inline]
    pub fn chunks_per_region(self) -> usize {
        TARGET_ARENA_BYTES / self.bytes()
    }

    /// The next rung up, or `None` at the top. Used by scarcity-only class
    /// promotion: a chunk may occupy a larger slot (more pad, reads unaffected
    /// because they use the format's own bytes) when its own class has no free
    /// slot and no free region can be claimed.
    #[inline]
    pub fn promote(self) -> Option<Self> {
        Self::from_index(self.index() + 1)
    }

    /// Every rung, ascending.
    pub fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).map(|i| Self(i as u8))
    }
}

/// Elements one chunk slot holds for a given palette sub-band width.
///
/// A slot is one `(head, palette-band, side)` of `CHUNK_SIZE` tokens, so
/// `CHUNK_SIZE * sub_head_dim` elements. Passing `sub_head_dim` explicitly is
/// deliberate: the legacy `arena_bytes_per_chunk` assumed
/// `sub_head_dim == CHUNK_SIZE`, which holds only for `head_dim / N_PALETTE ==
/// 32` and silently mis-sized every arena for any other geometry
/// (`docs/archived/arena_unification.md` A9).
#[inline]
pub fn elems_per_chunk(sub_head_dim: usize) -> usize {
    CHUNK_SIZE * sub_head_dim
}

/// Bytes a chunk of `format` actually occupies — **not** its slot stride.
///
/// This is the length for every copy, every persisted image, and every
/// checksum. Quantized formats are counted in whole blocks, because a block's
/// size is not generally divisible by its element count (`Q4_0` is 18 bytes for
/// 32 elements), so per-element arithmetic cannot round-trip it.
pub fn payload_bytes(format: KvFormat, elems_per_chunk: usize) -> usize {
    match format {
        KvFormat::Float(dtype) => elems_per_chunk * dtype.size_in_bytes(),
        KvFormat::Quantized(qf) => {
            let block = qf.block_size();
            debug_assert_eq!(
                elems_per_chunk % block,
                0,
                "chunk elems {elems_per_chunk} not divisible by {qf:?} block size {block}"
            );
            (elems_per_chunk / block) * qf.bytes_per_block()
        }
    }
}

/// Bytes a chunk whose recorded tag is `tag` occupies inside its slot.
///
/// The tag is the *only* record of a band's format once arenas are untyped
/// byte slabs, so this is the entry point every copy length, persisted image
/// and checksum goes through on the read side —
/// [`payload_bytes`] is its twin on the write side, where a `KvFormat` is
/// still in hand.
///
/// `None` for a tag that names no `KvFormat`: `Invalid`, the GGML K-quants,
/// and the other formats the KV cache never allocates. A caller holding such
/// a tag does not know how many bytes to move and must fail rather than guess.
#[inline]
pub fn payload_bytes_for_tag(tag: ArenaFormatTag, elems_per_chunk: usize) -> Option<usize> {
    tag.to_kv_format()
        .map(|f| payload_bytes(f, elems_per_chunk))
}

/// The smallest class that fits `payload`, or `None` if it exceeds the ladder.
#[inline]
pub fn class_for_payload(payload: usize) -> Option<SizeClass> {
    LADDER
        .iter()
        .position(|&b| b >= payload)
        .and_then(SizeClass::from_index)
}

/// The class a chunk of `format` is allocated from.
///
/// `None` means the ladder does not cover this format at this geometry — a
/// configuration error, not a runtime condition, since
/// [`every_kv_format_maps_to_a_class`](tests) pins coverage for the production
/// geometry.
#[inline]
pub fn class_for_format(format: KvFormat, elems_per_chunk: usize) -> Option<SizeClass> {
    class_for_payload(payload_bytes(format, elems_per_chunk))
}

/// Every float dtype the KV cache stores. `KvFormat::Float` accepts any
/// [`DType`], but only these four reach an arena (see
/// `ArenaFormatTag::from_kv_format`, which maps the rest to `Invalid`).
pub const KV_FLOAT_DTYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F8E4M3];

/// Every `KvFormat` an arena can hold: the four float dtypes plus all
/// [`QuantFormat`] variants. Used by coverage tests and by diagnostics that
/// walk the whole format space.
pub fn all_kv_formats() -> impl Iterator<Item = KvFormat> {
    use strum::IntoEnumIterator;
    KV_FLOAT_DTYPES
        .into_iter()
        .map(KvFormat::Float)
        .chain(QuantFormat::iter().map(KvFormat::Quantized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::arena_table::N_PALETTE;

    /// The production palette-4 geometry: `head_dim 128 / N_PALETTE 4 = 32`,
    /// so a slot is `CHUNK_SIZE(32) * 32 = 1024` elements.
    const PROD_ELEMS: usize = 1024;

    #[test]
    fn ladder_is_strictly_increasing() {
        for w in LADDER.windows(2) {
            assert!(
                w[0] < w[1],
                "ladder must be strictly increasing: {LADDER:?}"
            );
        }
    }

    /// Every head dim the paged kernels accept. A slot's payload scales with
    /// `head_dim / N_PALETTE`, so the ladder has to cover all of them, not just
    /// the one the model in front of us happens to use.
    ///
    /// Kept in step with the `64 | 96 | 128 | 256` guards in
    /// `paged_decode_attn` and `paged_prefill_attn_varlen_chunks`; 512 is the
    /// DeepSeek single-latent width, whose backing passes through the GQA
    /// geometry at construction (see the ladder's head-dim note).
    const SUPPORTED_HEAD_DIMS: [usize; 5] = [64, 96, 128, 256, 512];

    /// Coverage: every format an arena can hold maps to a class, at every head
    /// dim the kernels accept. A `None` here means a chunk of that format has
    /// nowhere to live.
    ///
    /// This asserted `PROD_ELEMS` alone until a decode test at `head_dim 256`
    /// died on `no size class covers Quantized(R16) at 2048 elems/chunk`: the
    /// ladder stopped at 4096 B, and `R16`/`F32` need 8192 B at that width.
    /// Pinning only the geometry we happen to ship is what let a supported one
    /// go uncovered, so the loop runs over all of them.
    #[test]
    fn every_kv_format_maps_to_a_class() {
        for head_dim in SUPPORTED_HEAD_DIMS {
            let elems = elems_per_chunk(head_dim / N_PALETTE);
            for fmt in all_kv_formats() {
                let payload = payload_bytes(fmt, elems);
                assert!(
                    class_for_format(fmt, elems).is_some(),
                    "{fmt:?} at head_dim {head_dim} ({elems} elems/chunk) has payload \
                     {payload} B, past the top of the ladder {LADDER:?}"
                );
            }
        }
    }

    /// **Above the catch-all rung, every format lands exactly.**
    ///
    /// A slot's stride is what the attention and selection kernels read per
    /// band, so padding is not merely wasted capacity — it is wasted
    /// *bandwidth*, on every pass over the cache, for as long as the chunk
    /// lives. Rounding a known size up to a prettier one buys nothing.
    ///
    /// The bottom rung is the deliberate exception. Below it sit eight formats
    /// from 32 B (`Q0`) to 288 B (`Q2_S`), each of which would otherwise stamp
    /// a whole 16 MiB region for a trickle of chunks — §3.10's problem 3,
    /// rare-class region stranding, which is the reason classes are coarser
    /// than formats at all. So the rule is: coarse where stranding dominates,
    /// exact where bandwidth does. A12 governs the boundary — split the low end
    /// only once sub-320 formats exceed ~2 % of live slots.
    ///
    /// Measured, and the reason this test exists: the ladder was first derived
    /// from the adaptive C-level formats, so the fixed ones missed — `Q8_0` at
    /// 1088 B rounded to 1152 (5.6 %), `Q4_0` at 576 B to 640 (10.0 %). The
    /// gate's two 20-context configs are exactly `Q8_0×20` and `Q4_0×20`, and
    /// they were the only two of sixteen below the pre-unification baseline,
    /// five runs out of five — at the width where KV bandwidth dominates.
    #[test]
    fn every_format_above_the_catch_all_lands_exactly() {
        let catch_all = LADDER[0];
        let mut padded: Vec<String> = Vec::new();
        for fmt in all_kv_formats() {
            let payload = payload_bytes(fmt, PROD_ELEMS);
            if payload <= catch_all {
                continue;
            }
            let Some(class) = class_for_format(fmt, PROD_ELEMS) else {
                continue; // coverage is `every_kv_format_maps_to_a_class`'s job
            };
            if class.bytes() != payload {
                padded.push(format!(
                    "{fmt:?}: payload {payload} B -> class {} B ({} B, {:.1} % wasted on every read)",
                    class.bytes(),
                    class.bytes() - payload,
                    (class.bytes() - payload) as f64 / class.bytes() as f64 * 100.0,
                ));
            }
        }
        assert!(
            padded.is_empty(),
            "the ladder {LADDER:?} needs a rung for each of these:\n  {}",
            padded.join("\n  ")
        );
    }

    /// The catch-all rung really is a catch-all: everything under it shares one
    /// class, so a rare format never strands a region of its own.
    #[test]
    fn small_formats_share_the_bottom_rung() {
        let catch_all = LADDER[0];
        let small: Vec<KvFormat> = all_kv_formats()
            .filter(|&f| payload_bytes(f, PROD_ELEMS) <= catch_all)
            .collect();
        assert!(
            small.len() > 1,
            "the bottom rung should absorb several formats"
        );
        for fmt in small {
            let class = class_for_format(fmt, PROD_ELEMS).expect("covered");
            assert_eq!(
                class.bytes(),
                catch_all,
                "{fmt:?} should share the catch-all rung"
            );
        }
    }

    /// The class must be the *smallest* that fits — a format landing in a
    /// larger class than necessary is silent waste on every chunk of it.
    #[test]
    fn class_is_smallest_that_fits() {
        for fmt in all_kv_formats() {
            let payload = payload_bytes(fmt, PROD_ELEMS);
            let class = class_for_format(fmt, PROD_ELEMS).expect("covered");
            assert!(
                class.bytes() >= payload,
                "{fmt:?}: class {} B does not fit payload {payload} B",
                class.bytes()
            );
            if let Some(i) = class.index().checked_sub(1) {
                assert!(
                    LADDER[i] < payload,
                    "{fmt:?}: payload {payload} B also fits class {} B — not the smallest",
                    LADDER[i]
                );
            }
        }
    }

    /// **The u16 recycle-link invariant.** `ArenaRefcounts.counts` is
    /// `Vec<AtomicU16>`, and a free slot's word holds the recycle-stack link —
    /// which may be the empty-stack sentinel `chunks_per_region` itself. So the
    /// bound is `<= 65_535`, not `< 65_536`: `Q0_M4` at exactly 65,536 chunks
    /// made the sentinel wrap to 0 and alias slot 0 (`docs/archived/arena_unification.md`
    /// §2.1). The 320 B minimum caps every class at 52,428.
    #[test]
    fn every_class_fits_u16_recycle_links() {
        for class in SizeClass::all() {
            let chunks = class.chunks_per_region();
            assert!(
                chunks <= u16::MAX as usize,
                "class {} B yields {chunks} chunks/region, past the u16 link bound {}",
                class.bytes(),
                u16::MAX
            );
        }
    }

    /// **The class follows the model's palette width, not `CHUNK_SIZE`.**
    ///
    /// A slot holds `CHUNK_SIZE × (head_dim / N_PALETTE)` elements. The retired
    /// `arena_bytes_per_chunk` hard-coded `CHUNK_SIZE × CHUNK_SIZE`, which is
    /// the same number only at `head_dim == 128` — it returned 8,192 chunks for
    /// an F16 arena at *every* geometry and so silently mis-sized every arena on
    /// any other model (`docs/archived/arena_unification.md` §9). This pins that the
    /// ladder reads the geometry: same format, different `head_dim`, different
    /// class.
    #[test]
    fn the_class_follows_the_palette_width_not_chunk_size() {
        let f16 = KvFormat::Float(DType::F16);
        let class_at = |head_dim: usize| {
            let elems = elems_per_chunk((head_dim / N_PALETTE).max(1));
            class_for_format(f16, elems).expect("F16 is covered at every tested geometry")
        };

        // Production Qwen3 geometry: head_dim 128 ⇒ sub-band 32 ⇒ 1024 elems ⇒
        // 2048 B. This is the one geometry at which the retired constant agreed.
        let prod = class_at(128);
        assert_eq!(prod.bytes(), 2048);
        assert_eq!(prod.chunks_per_region(), 8192);

        // Half the head width halves the payload and drops a rung — where the
        // retired constant would still have said 2048 B / 8192 chunks.
        let narrow = class_at(64);
        assert_eq!(narrow.bytes(), 1024);
        assert_eq!(narrow.chunks_per_region(), 16_384);
        assert_ne!(
            narrow.chunks_per_region(),
            prod.chunks_per_region(),
            "chunks/region must track head_dim, or arena sizing is geometry-blind again"
        );
    }

    #[test]
    fn gid_stride_exceeds_max_chunks_per_region() {
        let max = SizeClass::all()
            .map(|c| c.chunks_per_region())
            .max()
            .expect("ladder is non-empty");
        assert_eq!(max, 52_428, "smallest class sets the max chunks/region");
        assert!(
            GID_STRIDE > max,
            "GID_STRIDE {GID_STRIDE} must exceed max chunks/region {max}"
        );
    }

    #[test]
    fn gid_stride_is_a_power_of_two() {
        assert!(
            GID_STRIDE.is_power_of_two(),
            "GID_STRIDE {GID_STRIDE} must be a power of two so decode is shift/mask"
        );
    }

    /// **Raw byte goldens**, per repo policy for anything serialization- or
    /// quantization-shaped: the exact payload of one 1024-element chunk in
    /// every format. These are `32 blocks * bytes_per_block`, and each
    /// `bytes_per_block` is locked to its CUDA counterpart by a
    /// `static_assert` in `candle-kernels/src/blocks.cuh`. A change here means
    /// a block struct changed and the CUDA side must be checked.
    #[test]
    fn payload_bytes_match_block_goldens() {
        use QuantFormat as Q;
        let cases: [(KvFormat, usize); 26] = [
            // Floats: elems * width.
            (KvFormat::Float(DType::F32), 4096),
            (KvFormat::Float(DType::F16), 2048),
            (KvFormat::Float(DType::BF16), 2048),
            (KvFormat::Float(DType::F8E4M3), 1024),
            // Quants: 32 blocks of 32 elements.
            (KvFormat::Quantized(Q::Q0), 32),
            (KvFormat::Quantized(Q::Q0_V), 64),
            (KvFormat::Quantized(Q::Q0_X), 64),
            (KvFormat::Quantized(Q::Q0_M2), 96),
            (KvFormat::Quantized(Q::Q1_S), 160),
            (KvFormat::Quantized(Q::Q1_A), 192),
            (KvFormat::Quantized(Q::Q0_M4), 256),
            (KvFormat::Quantized(Q::Q2_S), 288),
            (KvFormat::Quantized(Q::Q2_0), 320),
            (KvFormat::Quantized(Q::Q2_A), 320),
            (KvFormat::Quantized(Q::Q2_1), 384),
            (KvFormat::Quantized(Q::Q3_0), 448),
            (KvFormat::Quantized(Q::Q3_1), 512),
            (KvFormat::Quantized(Q::Q4_0), 576),
            (KvFormat::Quantized(Q::Q4_1), 640),
            (KvFormat::Quantized(Q::Q4_KS), 640),
            (KvFormat::Quantized(Q::Q5_0), 704),
            (KvFormat::Quantized(Q::Q5_1), 768),
            (KvFormat::Quantized(Q::Q8_0), 1088),
            (KvFormat::Quantized(Q::Q8_1), 1152),
            (KvFormat::Quantized(Q::Q8_KS), 1152),
            (KvFormat::Quantized(Q::R16), 4096),
        ];
        for (fmt, want) in cases {
            assert_eq!(
                payload_bytes(fmt, PROD_ELEMS),
                want,
                "{fmt:?} payload for a 1024-element chunk"
            );
        }
        // The table above must be exhaustive over the format space, or a new
        // format could slip in without a golden.
        assert_eq!(
            cases.len(),
            all_kv_formats().count(),
            "golden table must cover every KvFormat"
        );
    }

    /// **The tag round-trip.** `ArenaFormatTag::to_kv_format` is the inverse of
    /// `from_kv_format`, and it is now load-bearing: with arenas untyped, the
    /// tag is the only path from a persisted byte back to a byte length. A
    /// format that fails to round-trip would have its bands silently truncated
    /// or over-read on every copy.
    #[test]
    fn every_kv_format_round_trips_through_its_tag() {
        for fmt in all_kv_formats() {
            let tag = ArenaFormatTag::from_kv_format(fmt);
            assert_ne!(tag, ArenaFormatTag::Invalid, "{fmt:?} has no tag");
            assert_eq!(
                tag.to_kv_format(),
                Some(fmt),
                "{fmt:?} -> {tag:?} did not round-trip"
            );
        }
    }

    /// A tag that names no `KvFormat` yields no length. The alternative — a
    /// zero, or a default width — turns a corrupt tag byte into a silent
    /// short copy, which is exactly the class of bug the tags exist to make
    /// impossible.
    #[test]
    fn unmapped_tags_have_no_payload() {
        for tag in [
            ArenaFormatTag::Invalid,
            ArenaFormatTag::Q4_K,
            ArenaFormatTag::Q6_K,
            ArenaFormatTag::P2,
            ArenaFormatTag::QAWQ,
            ArenaFormatTag::F8E5M2,
        ] {
            assert_eq!(
                payload_bytes_for_tag(tag, PROD_ELEMS),
                None,
                "{tag:?} must not report a payload length"
            );
        }
    }

    /// `payload_bytes_for_tag` and `payload_bytes` are the same quantity read
    /// from the two directions — the tag side and the format side. They are
    /// used by opposite halves of every copy, so a divergence is a
    /// write-length/read-length mismatch.
    #[test]
    fn tag_payload_matches_format_payload() {
        for fmt in all_kv_formats() {
            let tag = ArenaFormatTag::from_kv_format(fmt);
            assert_eq!(
                payload_bytes_for_tag(tag, PROD_ELEMS),
                Some(payload_bytes(fmt, PROD_ELEMS)),
                "{fmt:?} disagrees between tag and format"
            );
        }
    }

    /// The known rounding losses, pinned so a ladder edit has to confront them
    /// rather than silently changing them.
    ///
    /// Every loss above the catch-all is now zero — the ladder gained rungs for
    /// the formats that used to round up, because the padding is re-read on
    /// every attention pass and nothing was buying it. What remains is the
    /// small end, which is coarse on purpose: it is the trade audit A12 gates
    /// on, and `Q0` at 90 % is its honest worst case.
    #[test]
    fn known_rounding_waste_is_unchanged() {
        let waste = |fmt: KvFormat| -> f64 {
            let p = payload_bytes(fmt, PROD_ELEMS);
            let c = class_for_format(fmt, PROD_ELEMS).unwrap().bytes();
            (c - p) as f64 / c as f64 * 100.0
        };
        // Q0 32 -> 320 = 90 %: the honest cost of one fungible small class.
        assert!((waste(KvFormat::Quantized(QuantFormat::Q0)) - 90.0).abs() < 0.01);
        // Everything the engine actually stores in bulk is exact, including the
        // two that were not: F8E4M3 (was 1024 -> 1152, 11.1 %) and Q8_0/Q4_0.
        for exact in [
            KvFormat::Float(DType::F8E4M3),
            KvFormat::Float(DType::F16),
            KvFormat::Quantized(QuantFormat::Q4_0),
            KvFormat::Quantized(QuantFormat::Q8_0),
            KvFormat::Quantized(QuantFormat::Q4_KS),
            KvFormat::Quantized(QuantFormat::Q8_KS),
        ] {
            assert_eq!(waste(exact), 0.0, "{exact:?} should land on a rung");
        }
    }

    /// Formats that share a class share a region and a free list. This is the
    /// property the whole initiative rests on and it is currently false — under
    /// per-format arenas each of these lives in its own pool.
    #[test]
    fn formats_sharing_a_class_are_fungible() {
        let class_of = |f: KvFormat| class_for_format(f, PROD_ELEMS).expect("covered").index();
        // The entire sub-320 tail collapses into one class.
        let small = [
            QuantFormat::Q0,
            QuantFormat::Q0_V,
            QuantFormat::Q0_X,
            QuantFormat::Q0_M2,
            QuantFormat::Q1_S,
            QuantFormat::Q1_A,
            QuantFormat::Q0_M4,
            QuantFormat::Q2_S,
            QuantFormat::Q2_0,
            QuantFormat::Q2_A,
        ];
        let c0 = class_of(KvFormat::Quantized(small[0]));
        for q in small {
            assert_eq!(
                class_of(KvFormat::Quantized(q)),
                c0,
                "{q:?} must share the smallest class"
            );
        }
        // Same-size formats still share: Q4_1 and Q4_KS are both 640 B, Q8_1
        // and Q8_KS both 1152.
        assert_eq!(
            class_of(KvFormat::Quantized(QuantFormat::Q4_1)),
            class_of(KvFormat::Quantized(QuantFormat::Q4_KS))
        );
        assert_eq!(
            class_of(KvFormat::Quantized(QuantFormat::Q8_1)),
            class_of(KvFormat::Quantized(QuantFormat::Q8_KS))
        );
        // Q8_0 (1088 B) no longer shares with Q8_KS (1152 B), and that is the
        // intended trade: they *can* co-occur — C4/C5 offer both as K
        // candidates — so separating them can cost one extra region in a
        // session that uses both. At ~15 k slots per region that is a rounding
        // error, against 5.6 % of every Q8_0 band read for as long as the chunk
        // lives. Fungibility is worth paying for at the small end, where a
        // format's whole demand may not fill a region; not here.
        assert_ne!(
            class_of(KvFormat::Quantized(QuantFormat::Q8_0)),
            class_of(KvFormat::Quantized(QuantFormat::Q8_KS))
        );
        // R16 (active K) and F32 share the top class.
        assert_eq!(
            class_of(KvFormat::Quantized(QuantFormat::R16)),
            class_of(KvFormat::Float(DType::F32))
        );
    }

    #[test]
    fn promote_walks_up_and_stops_at_the_top() {
        let mut c = SizeClass::from_index(0).unwrap();
        for i in 1..SizeClass::COUNT {
            c = c.promote().expect("not yet at the top");
            assert_eq!(c.index(), i);
        }
        assert!(c.promote().is_none(), "top class cannot promote");
    }

    /// A slot must never straddle a region boundary: `chunks_per_region *
    /// bytes <= TARGET_ARENA_BYTES`. Integer division makes this hold by
    /// construction; the test pins it against a future ladder edit that
    /// introduces a stride not dividing the region size.
    #[test]
    fn region_capacity_never_overruns() {
        for class in SizeClass::all() {
            let used = class.chunks_per_region() * class.bytes();
            assert!(
                used <= TARGET_ARENA_BYTES,
                "class {} B: {} chunks use {used} B of a {TARGET_ARENA_BYTES} B region",
                class.bytes(),
                class.chunks_per_region()
            );
        }
    }

    /// `elems_per_chunk` must come from the geometry, not from the assumption
    /// `sub_head_dim == CHUNK_SIZE` (audit A9). A model with `head_dim 256`
    /// gives `sub_head_dim 64` and doubles every payload.
    #[test]
    fn payload_scales_with_sub_head_dim() {
        assert_eq!(elems_per_chunk(32), 1024);
        assert_eq!(elems_per_chunk(64), 2048);
        let f16 = KvFormat::Float(DType::F16);
        assert_eq!(payload_bytes(f16, elems_per_chunk(32)), 2048);
        assert_eq!(payload_bytes(f16, elems_per_chunk(64)), 4096);
    }
}
