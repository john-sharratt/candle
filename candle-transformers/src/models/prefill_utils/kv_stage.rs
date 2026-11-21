//! Stages each long sequence's keys and values ahead of the prefill attention
//! kernel, so its blocks do not each decode the same positions.
//!
//! A sequence with enough query rows has every key position decoded once into a
//! stage buffer, while a short launch reads each column from its source. The
//! host read-back form exists so tests can check the staged rows against what the
//! kernel reads. The plan and the cut of a launch fix the buffer's size and the
//! windows the launch is divided into.

use super::*;

/// The pre-staged K/V of one prefill launch, read back to the host: per KV
/// head, one row per staged key position — int8 post-RoPE K codes, their FP16
/// per-32-dim-window scales (as bits), FP16 V (as bits). A staged sequence's
/// rows start at the staged kv lengths of the sequences before it.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct KvStageHost {
    /// Rows per KV head.
    pub positions: usize,
    /// Dims per row.
    pub head_dim: usize,
    pub(super) k_codes: Vec<i8>,
    pub(super) k_scales: Vec<u16>,
    pub(super) v: Vec<u16>,
}

#[cfg(feature = "cuda")]
impl KvStageHost {
    /// Row `row`'s K codes for KV head `head`, natural dim order.
    pub fn k_codes(&self, head: usize, row: usize) -> &[i8] {
        let at = (head * self.positions + row) * self.head_dim;
        &self.k_codes[at..at + self.head_dim]
    }

    /// Row `row`'s K window scales for KV head `head`, FP16 bits.
    pub fn k_scales(&self, head: usize, row: usize) -> &[u16] {
        let n_win = self.head_dim / 32;
        let at = (head * self.positions + row) * n_win;
        &self.k_scales[at..at + n_win]
    }

    /// Row `row`'s V for KV head `head`, FP16 bits, natural dim order.
    pub fn v(&self, head: usize, row: usize) -> &[u16] {
        let at = (head * self.positions + row) * self.head_dim;
        &self.v[at..at + self.head_dim]
    }
}

/// The pre-staging pass of a prefill launch alone — what
/// [`paged_prefill_batched`] stages ahead of its attention kernel for
/// `caches` about to prefill `q_lens` tokens (packed `k`/`v`, flat
/// `[Σ q_lens, n_kv_head, head_dim]` in the arena's compute dtype) at
/// `offsets` — read back to the host. Writes nothing to the caches; the
/// writer region must already be claimed, as for the launch itself.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_kv_stage(
    caches: &[&mut KvCache],
    offsets: &[usize],
    k: &Tensor,
    v: &Tensor,
    q_lens: &[usize],
    n_kv_head: usize,
    head_dim: usize,
    rope: &RopeRungs,
    rope_interleaved: bool,
    generation: &Generation,
) -> Result<KvStageHost> {
    if !int8_prefill_head_dim(head_dim) {
        candle::bail!("prefill kv stage: head_dim {head_dim} has no int8 prefill kernel");
    }
    let device = k.device().clone();
    let Device::Cuda(dev) = &device else {
        candle::bail!("prefill kv stage: K must be a CUDA tensor");
    };
    let plan = KvStagePlan::new(q_lens, offsets);
    if plan.positions == 0 {
        candle::bail!(
            "prefill kv stage: no sequence of {q_lens:?} has {PREFILL_KV_STAGE_MIN_Q_LEN} rows"
        );
    }
    let kv_dtype_code: i32 = match k.dtype() {
        DType::F16 => 1,
        DType::BF16 => 2,
        dt => candle::bail!("prefill kv stage: unsupported K/V dtype {dt:?}"),
    };
    let total_q: usize = q_lens.iter().sum();
    for (t, what) in [(k, "k"), (v, "v")] {
        if t.dims() != [total_q, n_kv_head, head_dim]
            || !t.is_contiguous()
            || t.dtype() != k.dtype()
        {
            candle::bail!("prefill kv stage: {what} is {:?}, not packed", t.shape());
        }
    }
    let b_sz = q_lens.len();
    let mut cu = Vec::with_capacity(b_sz + 1);
    cu.push(0u32);
    for &l in q_lens {
        cu.push(cu[cu.len() - 1] + l as u32);
    }
    let cu = Tensor::from_vec(cu, b_sz + 1, &device)?;
    let q_lens_dev = Tensor::from_vec(
        q_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
        b_sz,
        &device,
    )?;
    let kv_lens_dev = Tensor::from_vec(
        offsets
            .iter()
            .zip(q_lens)
            .map(|(&o, &l)| (o + l) as u32)
            .collect::<Vec<_>>(),
        b_sz,
        &device,
    )?;
    let headers = build_slot_headers(
        caches,
        q_lens,
        generation,
        &RefCell::new(None),
        offsets,
        rope,
        &device,
        None,
    )?;

    let bytes = plan.bytes(n_kv_head, head_dim);
    let stage = KernelOutput::<u8>::new(dev, bytes, None)?;
    let stream = dev.cuda_stream();
    let ptr_of = |t: &Tensor| -> Result<u64> {
        let (s, l) = t.storage_and_layout();
        let Storage::Cuda(c) = &*s else {
            candle::bail!("prefill kv stage: expected a CUDA tensor")
        };
        let base = match t.dtype() {
            DType::U32 => {
                c.as_cuda_slice::<u32>()?
                    .slice(l.start_offset()..)
                    .device_ptr(&stream)
                    .0
            }
            DType::F16 => {
                c.as_cuda_slice::<f16>()?
                    .slice(l.start_offset()..)
                    .device_ptr(&stream)
                    .0
            }
            DType::BF16 => {
                c.as_cuda_slice::<bf16>()?
                    .slice(l.start_offset()..)
                    .device_ptr(&stream)
                    .0
            }
            dt => candle::bail!("prefill kv stage: unexpected dtype {dt:?}"),
        };
        Ok(base)
    };
    let (k_ptr, v_ptr) = (ptr_of(k)?, ptr_of(v)?);
    let (cu_ptr, ql_ptr, kv_ptr) = (ptr_of(&cu)?, ptr_of(&q_lens_dev)?, ptr_of(&kv_lens_dev)?);
    {
        let (stage_ptr, _guard) = stage.device_ptr(&stream);
        candle::set_kernel_breadcrumb("run_paged_prefill_kv_stage", file!(), line!());
        // SAFETY: every pointer is a live device buffer of the extent the
        // kernel addresses — the packed K/V and the length tables built above,
        // the uploaded headers held by `headers`, and `bytes` of stage.
        unsafe {
            run_paged_prefill_kv_stage(
                k_ptr as *const c_void,
                v_ptr as *const c_void,
                headers.headers_ptr as *const u8,
                cu_ptr as *const u32,
                ql_ptr as *const u32,
                kv_ptr as *const u32,
                b_sz as i32,
                n_kv_head as i32,
                head_dim as i32,
                kv_dtype_code,
                rope.ffi()?,
                rope_interleaved as i32,
                stream.cu_stream() as *mut c_void,
                stage_ptr as *mut u8,
                bytes as i64,
                plan.positions as i64,
                PREFILL_KV_STAGE_MIN_Q_LEN as i32,
                plan.max_kv_len as i32,
            );
        }
    }
    let raw = stage
        .into_tensor(dev.clone(), bytes)
        .to_owned_tensor()?
        .to_vec1::<u8>()?;
    drop(headers);

    // The planes as `prefill_kv_stage_bytes` lays them out.
    let rows = n_kv_head * plan.positions;
    let align = |b: usize| b.div_ceil(BUMP_ALIGNMENT) * BUMP_ALIGNMENT;
    let s_off = align(rows * head_dim);
    let v_off = s_off + align(rows * (head_dim / 32) * 2);
    let half = |b: &[u8]| -> Vec<u16> {
        b.chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect()
    };
    Ok(KvStageHost {
        positions: plan.positions,
        head_dim,
        k_codes: raw[..rows * head_dim].iter().map(|&b| b as i8).collect(),
        k_scales: half(&raw[s_off..s_off + rows * (head_dim / 32) * 2]),
        v: half(&raw[v_off..v_off + rows * head_dim * 2]),
    })
}

/// What an uncut pre-staging pass stages (`candle-kernels/src/paged-prefill/kv_stage.cuh`):
/// every key position of each sequence with at least
/// [`PREFILL_KV_STAGE_MIN_Q_LEN`] rows — the bulk chunks, whose blocks would
/// otherwise each decode the positions they select — and the deepest of them.
/// A launch of short sequences (a verify window, a decode-shaped row) stages
/// nothing and reads every column from its source.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct KvStagePlan {
    /// The stage's rows per KV head: [`prefill_kv_stage_positions`].
    pub(super) positions: usize,
    /// The deepest staged sequence's kv length — the pre-staging grid's extent.
    pub(super) max_kv_len: usize,
}

/// How one prefill attention launch is cut and what it carves
/// ([`PrefillKvStageLayout`]): the buffer's layout, each row group's key
/// windows, and the deepest staged sequence.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PrefillLaunchCut {
    pub(super) layout: PrefillKvStageLayout,
    /// Key windows per row group — the launcher's loop.
    pub(super) group_chunks: Vec<u32>,
    /// The deepest staged sequence's kv length — the pre-staging grid's extent.
    pub(super) max_kv_len: usize,
}

#[cfg(feature = "cuda")]
impl PrefillLaunchCut {
    pub(super) fn new(
        q_lens: &[usize],
        offsets: &[usize],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        bounds: PrefillLaunchBounds,
    ) -> Self {
        let layout =
            PrefillKvStageLayout::new(q_lens, offsets, n_head, n_kv_head, head_dim, bounds);
        Self {
            layout,
            group_chunks: layout.group_chunks(q_lens, offsets),
            max_kv_len: KvStagePlan::new(q_lens, offsets).max_kv_len,
        }
    }
}

#[cfg(feature = "cuda")]
impl KvStagePlan {
    pub(super) fn new(q_lens: &[usize], offsets: &[usize]) -> Self {
        Self {
            positions: prefill_kv_stage_positions(q_lens, offsets),
            max_kv_len: q_lens
                .iter()
                .zip(offsets)
                .filter(|(&q, _)| q >= PREFILL_KV_STAGE_MIN_Q_LEN)
                .map(|(&q, &off)| off + q)
                .max()
                .unwrap_or(0),
        }
    }

    /// The stage buffer's length for `n_kv_head` heads of `head_dim`.
    pub(super) fn bytes(&self, n_kv_head: usize, head_dim: usize) -> usize {
        prefill_kv_stage_bytes(self.positions, n_kv_head, head_dim)
    }
}
