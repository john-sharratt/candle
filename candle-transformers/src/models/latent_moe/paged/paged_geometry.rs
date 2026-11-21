//! Declares the latent geometry this build's kernels are compiled for, and
//! refuses to launch when the model, the host and the kernels disagree about it.
//!
//! The host lays out records and the kernels read them at offsets that both
//! derive from the same dimensions, so a disagreement produces wrong attention
//! rather than a fault. The declarations are compared at load, and a mismatch
//! is an error rather than an assertion that a build could compile out. The
//! file also builds the factored RoPE table the kernel reads, which covers
//! every position below two million.

use super::*;

/// The geometry this build's kernels are compiled for.
///
/// Everything below is derived from it, so the shape is declared once. Changing
/// it means adding the matching kernel instantiation first — see
/// [`SUPPORTED`](super::geometry::SUPPORTED) for the full checklist.
pub const GEOMETRY: LatentGeometry = D512_R64_B16;

pub const HEAD_DIM: usize = GEOMETRY.head_dim;

pub const ROPE_DIM: usize = GEOMETRY.rope_dim;

pub const NOPE_DIM: usize = GEOMETRY.nope_dim();

/// Single-latent band count. Must equal the `NPAL` template argument the
/// kernels are instantiated with in `paged_latent_api_bf16.cu` and
/// `LATENT_N_BANDS` in candle-nn.
pub const N_BANDS: usize = GEOMETRY.n_bands;

pub const SUB_DIM: usize = GEOMETRY.sub_dim();

pub const CHUNK: usize = 32;

/// KvHead record byte size (`slot_types.cuh` layout).
pub const KVHEAD_BYTES: usize = GEOMETRY.kvhead_bytes();

/// Refuse to launch unless the model, this build's host constants, and the
/// compiled kernels all describe the SAME latent.
///
/// Four declarations have to agree and none can see the others:
///
/// 1. `cfg.head_dim` / `cfg.rope_head_dim` — what the checkpoint actually holds;
/// 2. [`Arch::geometry`] — the shape the model says its kernels need;
/// 3. [`GEOMETRY`] here — the shape this build's host code lays records out for;
/// 4. `LM_HEAD_DIM` and friends in `paged_latent_api_bf16.cu` — the shape the
///    kernels were instantiated at, read back through `run_latent_geometry`.
///
/// A divergence between any pair is invisible at runtime: the host would write
/// `KvHead` bands at one set of offsets and the kernel would read another,
/// producing **wrong attention rather than a fault**. That is why this is a load
/// error and not an assertion someone might compile out.
///
/// (2) is the check that matters once a second model exists — a successor
/// declaring a 768-dim latent against kernels built for 512 is precisely the
/// mistake the `Arch` split makes possible, and comparing only (3) against (4)
/// would let it through.
#[cfg(feature = "cuda")]
pub fn assert_geometry(cfg: &Config) -> Result<()> {
    let g = cfg.arch.geometry();
    let arch_id = cfg.arch.id();

    // (1) vs (2) — the arch's declared latent must describe its own config.
    if g.head_dim != cfg.head_dim || g.rope_dim != cfg.rope_head_dim {
        candle::bail!(
            "{arch_id} declares latent geometry (head_dim {}, rope_dim {}) but its config says \
             (head_dim {}, rope_dim {}). Arch::geometry and the config describe the same \
             tensors, so one of them is wrong",
            g.head_dim,
            g.rope_dim,
            cfg.head_dim,
            cfg.rope_head_dim
        );
    }

    // (2) vs the built set — is any kernel compiled for this shape at all?
    if !super::super::geometry::is_supported(g) {
        candle::bail!(
            "{arch_id} needs latent geometry (head_dim {}, rope_dim {}, n_bands {}), which no \
             kernel is built for. Supported: {:?}. See latent_moe::geometry::SUPPORTED for what \
             adding one requires",
            g.head_dim,
            g.rope_dim,
            g.n_bands,
            super::super::geometry::SUPPORTED
        );
    }

    // (2)/(3) vs (4) — and this build's kernels must be the ones that got built.
    let compiled = candle_kernels::paged_latent::latent_geometry();
    let host = (HEAD_DIM, ROPE_DIM, N_BANDS);
    let wanted = (g.head_dim, g.rope_dim, g.n_bands);
    if compiled != wanted || host != wanted {
        candle::bail!(
            "latent geometry disagreement — {arch_id} wants {wanted:?}, host constants are \
             {host:?}, compiled kernels are {compiled:?} (head_dim, rope_dim, n_bands). \
             Launching would read KvHead bands at mismatched offsets. Align paged.rs GEOMETRY \
             with LM_HEAD_DIM/LM_ROPE_DIM/LM_NPAL in paged_latent_api_bf16.cu"
        );
    }
    Ok(())
}

/// Factored RoPE cos/sin table geometry (latent_common.cuh layout): position
/// `pos = hi·2¹⁰ + lo` splits the trig via the angle-addition identity, so two
/// small blocks — (sin, cos) of `(hi·2¹⁰)·f` and of `lo·f` — cover every
/// position below `ROPE_HI_DIM·2¹⁰` = 2M in ~768 KB per frequency set
/// (L2-resident). The attention kernel is a bounded WINDOW into the substrate
/// (the engine's context hard-cap is 1M), so every position it sees is < 2M and
/// lands in the table — the growing corpus is retrieved INTO that window, never
/// attended at raw substrate positions.
pub const ROPE_LO_BITS: usize = 10;

pub const ROPE_LO_DIM: usize = 1 << ROPE_LO_BITS;

// 1024
pub const ROPE_HI_DIM: usize = 2048;

// 2¹¹ → span 2²¹ ≈ 2M positions
/// f32 element count of one table: hi block + lo block, (sin, cos) per entry.
pub const ROPE_TAB_LEN: usize = GEOMETRY.rope_table_len(ROPE_HI_DIM, ROPE_LO_DIM);

/// Build the factored cos/sin table for one frequency set on-device (each
/// entry from the exact bit-mirrorable `rope_angle`/`ds_sincos` pair). Runs
/// once per set at model load; stream-ordered before any attention launch.
#[cfg(feature = "cuda")]
pub fn build_rope_table(rope_freqs: &Tensor) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let n_freqs = rope_freqs.dims1()?;
    if n_freqs != ROPE_DIM / 2 {
        candle::bail!("build_rope_table: {n_freqs} freqs != {}", ROPE_DIM / 2);
    }
    let dev = match rope_freqs.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("build_rope_table requires a CUDA tensor"),
    };
    let stream = dev.cuda_stream();
    let tab = Tensor::zeros(ROPE_TAB_LEN, DType::F32, rope_freqs.device())?;
    macro_rules! cuda_ptr {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            let ptr = match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<$ty>()?;
                    let (p, _guard) = slice.device_ptr(&stream);
                    p + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                }
                _ => candle::bail!("expected CUDA storage"),
            };
            ptr
        }};
    }
    let freq_p = cuda_ptr!(rope_freqs, f32);
    let tab_p = cuda_ptr!(&tab, f32);
    unsafe {
        candle_kernels::paged_latent::run_latent_rope_table_build(
            freq_p as *const f32,
            tab_p as *mut f32,
            n_freqs as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(tab)
}

/// Number of NOPE bands in the two-region cache (`[0, NOPE_DIM)` at `SUB_DIM`).
pub const NOPE_BANDS: usize = GEOMETRY.nope_bands();
