//! Ties together the three places the latent geometry is declared, and checks
//! that a geometry no kernel is built for is refused.
//!
//! The host constants, the kernel macros and the candle-nn layout cannot import
//! one another, so a drift between them is invisible to every numerical test
//! and shows up only as wrong attention. This test compares them directly. A
//! model that declares a latent no kernel implements is refused at load as
//! well, which a comparison of host against kernel alone would let through.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    // Host mirrors of the latent kernels. Every one of these is a transcription:
    // the kernel's own loop bounds, its float literals to full width, and its
    // `fminf`/`fmaxf` NaN behaviour. Restyling them into iterator form or
    // trimming digits would break the line-by-line diff against the `.cu` that
    // makes a mismatch findable.
    #![allow(
        clippy::assign_op_pattern,
        clippy::excessive_precision,
        clippy::needless_range_loop,
        clippy::neg_cmp_op_on_partial_ord,
        clippy::type_complexity
    )]

    use super::*;

    pub(in super::super) const H: usize = 64;

    /// The geometry is declared in three places that cannot import each other —
    /// [`GEOMETRY`] here, the `LM_*` macros in `paged_latent_api_bf16.cu`, and
    /// `LATENT_N_BANDS` in candle-nn (which lays out the arena records but does
    /// not depend on candle-kernels). candle-transformers is the one crate that
    /// sees all three, so this is where they get tied together.
    ///
    /// A drift between them is invisible to every numerical test: the host would
    /// write `KvHead` bands at one set of offsets and the kernel would read
    /// another, yielding wrong attention rather than a fault.
    #[test]
    #[cfg(feature = "cuda")]
    fn host_geometry_matches_the_compiled_kernels() {
        use super::super::super::arch::Arch;
        use candle_nn::kv_cache::{LATENT_NOPE_BANDS, LATENT_N_BANDS};

        assert_eq!(
            candle_kernels::paged_latent::latent_geometry(),
            (HEAD_DIM, ROPE_DIM, N_BANDS),
            "compiled kernel geometry differs from paged.rs GEOMETRY"
        );
        assert_eq!(
            LATENT_N_BANDS, N_BANDS,
            "candle-nn arena band count differs"
        );
        assert_eq!(
            LATENT_NOPE_BANDS, NOPE_BANDS,
            "candle-nn nope/rope band split differs"
        );
        // And the full load-time check passes for a real model, which additionally
        // ties in the arch's declared geometry and the checkpoint's own dims.
        assert_geometry(&crate::models::deepseek4::DEEPSEEK_V4.defaults())
            .expect("geometry check passes when model, host, and kernels agree");
    }

    /// A model whose declared latent no kernel implements must be REJECTED, not
    /// launched. This is the case a successor architecture introduces — the one
    /// a host-vs-kernel constant comparison alone would let through — so it is
    /// checked with a config that claims a geometry nothing is built for.
    #[test]
    #[cfg(feature = "cuda")]
    fn a_model_wanting_an_unbuilt_geometry_is_refused() {
        use super::super::super::arch::Arch;
        let mut cfg = crate::models::deepseek4::DEEPSEEK_V4.defaults();
        // A legal latent (1024 = 32 bands of 32 dims, rope on a band boundary)
        // that no kernel is instantiated for.
        cfg.head_dim = 1024;
        cfg.rope_head_dim = 128;
        let err = assert_geometry(&cfg).expect_err("an unbuilt geometry must not load");
        let msg = err.to_string();
        // It must fail on the ARCH/CONFIG disagreement first — the arch still
        // declares 512 — which is itself the check the review found missing.
        assert!(
            msg.contains("declares latent geometry"),
            "expected the arch/config disagreement to be named, got: {msg}"
        );
    }
}
