//! FFI bindings for the dense int8 diffusion-transformer attention kernels.
//!
//! See `dit_attn_int8.cu` for the quantization grid and why it differs from the
//! paged kernel's. Every pointer is device memory on `stream`.

use core::ffi::c_void;

extern "C" {
    /// `softmax(q·kᵀ)·v` over a flat, unmasked sequence, int8 tensor-core.
    ///
    /// `q8`/`k8` are `[batch, heads, seq, head_dim]` int8 with `qs`/`ks` a f32
    /// scale per row; `v8` is `[batch, heads, head_dim, v_stride]` —
    /// **transposed** — with `vs` a f32 scale and `vmean` a f32 mean per dim.
    /// `out` is `[batch, heads, seq, head_dim]` bf16.
    ///
    /// `v_stride` is `seq` rounded up to a multiple of 16 and its pad must be
    /// zeroed: the kernel reads V a whole 16-byte vector at a time, so an
    /// unpadded row would be misaligned for any `seq` that is not itself a
    /// multiple of 16. Take it from [`v_stride`] rather than computing it again.
    ///
    /// `q` must arrive pre-scaled by `1/√head_dim`. `head_dim` other than 128 is
    /// a no-op rather than a wrong answer — the tiling is chosen against it.
    #[allow(clippy::too_many_arguments)]
    pub fn run_dit_attn_int8_bf16(
        q8: *const c_void,
        qs: *const c_void,
        k8: *const c_void,
        ks: *const c_void,
        v8: *const c_void,
        vs: *const c_void,
        vmean: *const c_void,
        out: *mut c_void,
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        v_stride: i32,
        stream: *mut c_void,
    );

    /// Symmetric int8 per row of a `[rows, cols]` bf16 tensor. `dst` has row
    /// stride `dst_stride` (≥ `cols`, tail zeroed) and `scale` is `[rows]` f32.
    ///
    /// **All three operands go through this one kernel**, including V: a
    /// transposed V's per-dim scale is a per-row scale, and V has to be
    /// transposed anyway for the PV MMA. `center` subtracts each row's mean and
    /// reports it in `mean` — which V wants and Q/K do not, because this
    /// architecture norms Q and K per head and leaves V alone.
    #[allow(clippy::too_many_arguments)]
    pub fn run_dit_quant_rows_bf16(
        src: *const c_void,
        dst: *mut c_void,
        scale: *mut c_void,
        mean: *mut c_void,
        rows: i32,
        cols: i32,
        dst_stride: i32,
        center: i32,
        stream: *mut c_void,
    );
}

/// V's padded row length for a sequence of `seq`.
///
/// **One definition, called by both sides.** The kernel reads V in 16-byte
/// vectors, so its rows have to be a multiple of 16 whatever the sequence is;
/// the allocation and the launch argument must agree exactly, and two copies of
/// `div_ceil(seq, 16) * 16` in different files is how they stop agreeing.
pub const fn v_stride(seq: usize) -> usize {
    seq.div_ceil(16) * 16
}
