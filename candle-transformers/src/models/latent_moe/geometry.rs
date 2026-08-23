//! The compile-time shape of one KV latent.
//!
//! Every layer of the paged-latent stack — the CUDA kernels, the `KvHead` arena
//! record, the bit-exact mirror oracles — is written against a fixed
//! `(head_dim, rope_dim, n_bands)` triple. [`LatentGeometry`] is the single
//! declaration of that triple and of the divisibility rules that make the
//! kernels' tiling legal; everything else derives from it.
//!
//! # Why this is compile-time and not a runtime field
//!
//! `head_dim` is an array length, not a parameter. The kernels take it as a
//! template argument (`latent_decode_kernel<T, HEAD_DIM, ROPE_DIM, NPAL>`) so the
//! band loops unroll and the shared-memory tiles size statically, and the Rust
//! mirror oracles that gate those kernels bit-exactly carry latents as
//! `[f32; HEAD_DIM]`. Demoting the triple to runtime values would replace both
//! with dynamic shapes — dynamic shared memory and per-row `Vec` allocation in
//! the oracle — to serve a geometry no model currently has.
//!
//! So geometry is *selected* at compile time and *parameterized* everywhere
//! below. Supporting a second geometry is additive, and the additions are
//! enumerated in [`SUPPORTED`].

/// The shape of a single KV latent: its width, the size of its RoPE tail, and
/// how many quantization bands it is cut into.
///
/// Construct through [`LatentGeometry::new`], which enforces every rule the
/// kernels rely on at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentGeometry {
    /// Latent width. K and V are the same rows, so this is the width of both.
    pub head_dim: usize,
    /// Width of the rotated tail. Dims `[head_dim - rope_dim, head_dim)` carry
    /// RoPE; the leading `nope_dim` do not.
    pub rope_dim: usize,
    /// Number of equal quantization bands the latent is cut into. Each band
    /// gets its own format and scale in the `KvHead` record.
    pub n_bands: usize,
}

impl LatentGeometry {
    /// The number of dims one MMA k-step consumes. A band must be a whole
    /// number of these so each band maps onto `m16n8k32` tiles.
    pub const MMA_K: usize = 32;

    /// Build a geometry, checking every rule the kernels depend on.
    ///
    /// Panics at compile time (in a `const` context) when a rule is broken, so
    /// an illegal geometry cannot reach a kernel launch:
    ///
    /// * `n_bands` divides `head_dim` — bands are equal width.
    /// * the band width is a multiple of [`MMA_K`](Self::MMA_K) — each band is a
    ///   whole number of MMA k-steps.
    /// * `rope_dim` is non-empty and narrower than the latent, and falls on a
    ///   band boundary — no band straddles the rope/nope split, so the
    ///   asymmetric seal can keep the rope bands at full width while
    ///   compressing the nope bands.
    /// * `rope_dim` is even — RoPE rotates interleaved pairs.
    /// * `n_bands` is a power of two no wider than a warp — the prefill kernel
    ///   reduces across bands with `__shfl_down_sync` at width `n_bands`.
    pub const fn new(head_dim: usize, rope_dim: usize, n_bands: usize) -> Self {
        assert!(
            n_bands > 0 && head_dim.is_multiple_of(n_bands),
            "bands must be equal width"
        );
        let sub = head_dim / n_bands;
        assert!(
            sub.is_multiple_of(Self::MMA_K),
            "a band must be a whole number of MMA k-steps"
        );
        assert!(
            rope_dim > 0 && rope_dim < head_dim,
            "the rope tail must be non-empty and narrower than the latent"
        );
        assert!(
            (head_dim - rope_dim).is_multiple_of(sub),
            "the rope/nope split must land on a band boundary"
        );
        assert!(
            rope_dim.is_multiple_of(2),
            "interleaved RoPE needs an even rope dim"
        );
        assert!(
            n_bands.is_power_of_two() && n_bands <= 32,
            "band reductions use a warp shuffle at width n_bands"
        );
        Self {
            head_dim,
            rope_dim,
            n_bands,
        }
    }

    /// Width of the unrotated head of the latent.
    pub const fn nope_dim(&self) -> usize {
        self.head_dim - self.rope_dim
    }

    /// Dims per band.
    pub const fn sub_dim(&self) -> usize {
        self.head_dim / self.n_bands
    }

    /// Bands backing the unrotated region. Bands `[0, nope_bands)` are the nope
    /// region, `[nope_bands, n_bands)` the rope tail.
    pub const fn nope_bands(&self) -> usize {
        self.nope_dim() / self.sub_dim()
    }

    /// Byte size of one `KvHead` record (`slot_types.cuh` layout): the 4-bit
    /// per-dim palette map (`head_dim / 2` bytes — one map, since K ≡ V) plus a
    /// 26-byte `{k_ptr, v_ptr, k_fmt, v_fmt, k_scale, v_scale}` block per band.
    pub const fn kvhead_bytes(&self) -> usize {
        self.head_dim / 2 + self.n_bands * 26
    }

    /// f32 element count of one factored RoPE cos/sin table, given the table's
    /// hi/lo block heights. Each entry stores a `(sin, cos)` pair per frequency,
    /// and there are `rope_dim / 2` frequencies.
    pub const fn rope_table_len(&self, hi_dim: usize, lo_dim: usize) -> usize {
        (hi_dim + lo_dim) * (self.rope_dim / 2) * 2
    }
}

/// `head_dim = 512`, `rope_dim = 64`, `n_bands = 16` — 32-dim bands, with the
/// 64 rope dims isolated in the last two (dims `[448, 480)` and `[480, 512)`).
///
/// This is the geometry DeepSeek-V4-Flash uses and the only one the CUDA
/// kernels are currently instantiated for.
pub const D512_R64_B16: LatentGeometry = LatentGeometry::new(512, 64, 16);

/// Every geometry the kernels are built for.
///
/// A geometry in this list is launchable; a model declaring one that is not is
/// rejected by [`paged::assert_geometry`](super::paged::assert_geometry) at load.
/// Adding one means, in order:
///
/// 1. a `LatentGeometry` const here, added to this list;
/// 2. a second set of `extern "C"` entry points in
///    `candle-kernels/src/paged-latent/paged_latent_api_bf16.cu`, forwarding to
///    the same `launch_latent_{decode,prefill}` templates with the new triple.
///    The C ABI has no templates, so simultaneous geometries need separate
///    symbols; the `LM_HEAD_DIM`/`LM_ROPE_DIM`/`LM_NPAL` macros there name the
///    one this translation unit compiles, and `run_latent_geometry` reports it;
/// 3. an `n_palette()` arm in `candle-nn`'s `ChunkedKvBacking` so the arena
///    allocates the right number of bands per head.
///
/// Nothing else is geometry-dependent: the wave engine, gallery, and compressor
/// all read the triple from the active [`Arch`](super::arch::Arch).
pub const SUPPORTED: &[LatentGeometry] = &[D512_R64_B16];

/// Whether the kernels can launch `g`.
pub fn is_supported(g: LatentGeometry) -> bool {
    let mut i = 0;
    while i < SUPPORTED.len() {
        if SUPPORTED[i].head_dim == g.head_dim
            && SUPPORTED[i].rope_dim == g.rope_dim
            && SUPPORTED[i].n_bands == g.n_bands
        {
            return true;
        }
        i += 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{is_supported, LatentGeometry, D512_R64_B16, SUPPORTED};

    #[test]
    fn v4_geometry_derives_the_documented_layout() {
        let g = D512_R64_B16;
        assert_eq!(g.nope_dim(), 448);
        assert_eq!(g.sub_dim(), 32);
        assert_eq!(g.nope_bands(), 14);
        // 512/2 palette-map bytes + 16 bands × 26 bytes.
        assert_eq!(g.kvhead_bytes(), 256 + 416);
        // The rope tail occupies exactly the last two bands.
        assert_eq!(g.n_bands - g.nope_bands(), 2);
        assert_eq!(g.nope_bands() * g.sub_dim(), g.nope_dim());
    }

    #[test]
    fn rope_table_len_matches_the_kernel_table() {
        // ROPE_HI_DIM = 2048, ROPE_LO_DIM = 1024, 32 frequencies, (sin, cos).
        assert_eq!(
            D512_R64_B16.rope_table_len(2048, 1024),
            (2048 + 1024) * 32 * 2
        );
    }

    #[test]
    fn every_supported_geometry_is_legal() {
        for g in SUPPORTED {
            // Re-running the constructor re-checks every rule.
            let round = LatentGeometry::new(g.head_dim, g.rope_dim, g.n_bands);
            assert_eq!(round, *g);
            assert!(is_supported(round));
        }
    }

    #[test]
    fn an_unbuilt_geometry_is_not_supported() {
        // Legal shape, no kernel instantiation — dispatch must reject it.
        assert!(!is_supported(LatentGeometry::new(1024, 128, 32)));
    }
}
