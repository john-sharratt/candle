//! The PLE block on the device: one GEMM and three fused launches in place of
//! [`super::ple::ple_apply_spans`]'s eager chain, every transient on the wave's
//! arena, and the residual updated in place.
//!
//! The eager chain stays as the definition — it is what the CPU oracle runs and
//! what the parity tests assert this against (see `simple/ple_fused.cu` for the
//! algebra). What changes is where the bytes go. The eager chain carves about
//! 930 KB a row, all of it on the CUDA pool because nothing in it has an arena
//! to inherit; this carves the gathered table rows, the key|value projection,
//! the conv's normed input and a span table — about 105 KB a row, on the phase
//! the caller opened — and writes nothing else.
//!
//! # The conv history is double-buffered
//!
//! Each sequence's `[hist, hc·d]` history is read through the span table where
//! it lies, and its successor is written into the sequence's spare buffer,
//! which then becomes its history ([`PleState::spare_hist`]). The history a
//! wave entered with is never written, so a failed wave's restore and a
//! verify's rewind both find it as it was. The spare is made at admission
//! ([`PleState::ensure_spare`]), so the block itself allocates no history.

use std::ffi::c_void;
use std::mem;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::wave_provenance::WaveTicket;
use candle::{DType, Device, LiveTensor, Result, Storage, Tensor};
use candle_kernels::simple::ple_fused::{run_ple_conv, run_ple_gate, PLE_SPAN_WORDS};

use super::config::PleConfig;
use super::device_operand::{with_operand, Operand};
use super::model::PleSource;
use super::ple::{ple_row_ids, PleSpan, PleWeights};
use crate::models::wave_buffers::{wave_empty_ticketed, wave_from_vec_ticketed};

/// The PLE weights in the shapes the fused launches read, built once at load.
#[derive(Debug, Clone)]
pub struct PleFusedWeights {
    /// `[hc·d + d, hidden]`: the key projection's rows, then the value's — one
    /// GEMM for both, its output the `kv` the gate kernel reads.
    pub key_value: Tensor,
    /// `[hc·d]` each, as in [`PleWeights`].
    pub norm_key: Tensor,
    pub norm_query: Tensor,
    pub norm_conv: Tensor,
    /// `[kern, hc·d]`: the checkpoint's `[hc·d, kern]` conv weight transposed,
    /// so a tap's weights for consecutive channels are consecutive in memory.
    pub conv_t: Tensor,
}

impl PleFusedWeights {
    /// Stack and transpose `w` for the fused launches — load-time copies, made
    /// once.
    pub fn from_weights(w: &PleWeights) -> Result<Self> {
        Ok(Self {
            key_value: Tensor::cat(&[&w.key, &w.value], 0)?,
            norm_key: w.norm_key.clone(),
            norm_query: w.norm_query.clone(),
            norm_conv: w.norm_conv.clone(),
            conv_t: w.conv.t()?.contiguous()?,
        })
    }
}

/// The PLE block over every sequence of a wave, on the device.
///
/// `res` is the wave's own `[ΣT, hc, n_embd]` residual, which `spans` tile in
/// order; it is updated in place, gated value then conv, as
/// [`super::ple::ple_apply_spans`] adds them. Every transient is carved from
/// `root`'s arena. Returns, per span that asked to capture them, a view of the
/// rows it appended to the conv history — on that arena, so a caller keeping
/// them past the phase copies them first.
pub fn ple_apply_spans_fused(
    res: &mut Tensor,
    spans: &mut [PleSpan<'_>],
    table: &dyn PleSource,
    w: &PleFusedWeights,
    cfg: &PleConfig,
    eps: f64,
    root: Option<WaveTicket>,
) -> Result<Vec<Option<Tensor>>> {
    let (total, hc, n_embd) = res.dims3()?;
    let hcd = hc * n_embd;
    let hist = cfg.conv_history();
    let dev = res.device().clone();
    let mut next = 0usize;
    for s in spans.iter() {
        if s.start != next || s.tokens.len() != s.len {
            candle::bail!(
                "ple spans: a span at row {} of {} rows ({} tokens) does not continue row {next}",
                s.start,
                s.len,
                s.tokens.len()
            );
        }
        next += s.len;
    }
    if next != total {
        candle::bail!("ple spans: spans cover {next} of the residual's {total} rows");
    }

    // The hash, per sequence on the host; one gather for every span's rows.
    let mut flat: Vec<u32> = Vec::new();
    for s in spans.iter_mut() {
        for token_rows in ple_row_ids(cfg, s.tokens, &mut s.state.prev) {
            flat.extend(token_rows);
        }
    }
    let hidden = w.key_value.dim(1)?;
    let emb = table.rows(&flat, root)?.reshape((total, hidden))?;
    // Key and value together; the projection inherits the rows' arena.
    let kv = emb.matmul(&w.key_value.t()?)?;
    let normalized = wave_empty_ticketed((total, hcd), DType::F32, &dev, root)?;
    launch_gate(&kv, res, w, &normalized, eps)?;

    // Each span's next history goes into its spare buffer — made at
    // admission, never here, and alternated with the history from then on.
    let mut writes: Vec<Tensor> = Vec::with_capacity(spans.len());
    for s in spans.iter_mut() {
        let spare = s.state.spare_hist.take().ok_or_else(|| {
            candle::Error::Msg(
                "ple: a sequence entered the forward with no spare history — admission \
                 makes it (`PleState::ensure_spare`), so the forward never allocates one"
                    .into(),
            )
        })?;
        if spare.dims() != [hist, hcd] {
            candle::bail!(
                "ple: spare history {:?} against a [{hist}, {hcd}] history",
                spare.dims()
            );
        }
        writes.push(spare);
    }
    let entries: Vec<SpanEntry<'_>> = spans
        .iter()
        .zip(&writes)
        .map(|(s, new_hist)| SpanEntry {
            start: s.start,
            len: s.len,
            hist: &s.state.conv_hist,
            new_hist,
        })
        .collect();
    let table = SpanTable::build(&entries, &dev, root)?;
    drop(entries);
    launch_conv(&normalized, &w.conv_t, &table, res, cfg)?;

    let mut captures = Vec::with_capacity(spans.len());
    for (s, written) in spans.iter_mut().zip(writes) {
        let read = mem::replace(&mut s.state.conv_hist, written);
        s.state.spare_hist = Some(read);
        captures.push(if s.capture {
            Some(normalized.narrow(0, s.start, s.len)?)
        } else {
            None
        });
    }
    Ok(captures)
}

/// `ple_gate`: the keyed injection added into `res` in place, and the conv's
/// normed input written to `normalized`.
///
///   kv `[n, hc·d + d]`, res `[n, hc, d]`, normalized `[n, hc·d]`
pub fn launch_gate(
    kv: &LiveTensor<'_>,
    res: &Tensor,
    w: &PleFusedWeights,
    normalized: &LiveTensor<'_>,
    eps: f64,
) -> Result<()> {
    let (n, hc, d) = res.dims3()?;
    let hcd = hc * d;
    let kv_stride = hcd + d;
    if kv.dims() != [n, kv_stride] || normalized.dims() != [n, hcd] {
        candle::bail!(
            "ple gate: kv {:?} and normalized {:?} for a [{n}, {hc}, {d}] residual",
            kv.dims(),
            normalized.dims()
        );
    }
    with_operand(kv, "ple gate: key|value", |kvo, stream| {
        with_operand(res, "ple gate: residual", |ro, _| {
            with_operand(&w.norm_key, "ple gate: key gain", |gko, _| {
                with_operand(&w.norm_query, "ple gate: query gain", |gqo, _| {
                    with_operand(&w.norm_conv, "ple gate: conv gain", |gco, _| {
                        with_operand(normalized, "ple gate: normalized", |no, _| {
                            let vec_ok = all_aligned(&[&kvo, &ro, &gko, &gqo, &gco, &no])
                                && kv_stride % 4 == 0;
                            candle::set_kernel_breadcrumb("run_ple_gate", file!(), line!());
                            unsafe {
                                run_ple_gate(
                                    kvo.ptr as *const f32,
                                    ro.ptr as *mut f32,
                                    gko.ptr as *const f32,
                                    gqo.ptr as *const f32,
                                    gco.ptr as *const f32,
                                    no.ptr as *mut f32,
                                    n as i32,
                                    hc as i32,
                                    d as i32,
                                    kv_stride as i32,
                                    eps as f32,
                                    (1.0 / (d as f64).sqrt()) as f32,
                                    i32::from(vec_ok),
                                    stream.cu_stream() as *mut c_void,
                                );
                            }
                        })
                    })
                })
            })
        })
    })??????;
    Ok(())
}

/// `ple_conv` and `ple_history`: the conv's SiLU added into `res` in place,
/// and every span's next history written into its `new_hist`.
pub fn launch_conv(
    normalized: &LiveTensor<'_>,
    conv_t: &Tensor,
    table: &SpanTable,
    res: &Tensor,
    cfg: &PleConfig,
) -> Result<()> {
    let (n, hc, d) = res.dims3()?;
    let hcd = hc * d;
    with_operand(normalized, "ple conv: normalized", |no, stream| {
        with_operand(conv_t, "ple conv: weight", |wo, _| {
            with_operand(res, "ple conv: residual", |ro, _| {
                let vec_ok = table.vec_ok && hcd % 4 == 0 && all_aligned(&[&no, &wo, &ro]);
                candle::set_kernel_breadcrumb("run_ple_conv", file!(), line!());
                unsafe {
                    run_ple_conv(
                        no.ptr as *const f32,
                        wo.ptr as *const f32,
                        table.ptr as *const c_void,
                        ro.ptr as *mut f32,
                        n as i32,
                        table.spans as i32,
                        hcd as i32,
                        cfg.conv_kernel as i32,
                        cfg.ngram_size as i32,
                        cfg.conv_history() as i32,
                        i32::from(vec_ok),
                        stream.cu_stream() as *mut c_void,
                    );
                }
            })
        })
    })???;
    Ok(())
}

/// One span as [`SpanTable::build`] lays it out.
pub struct SpanEntry<'a> {
    pub start: usize,
    pub len: usize,
    /// The history this wave reads.
    pub hist: &'a Tensor,
    /// Where its next history is written — never `hist` itself.
    pub new_hist: &'a Tensor,
}

/// The conv's per-span descriptors, uploaded: the invariant-2b table that lets
/// the kernel read each history where it lies.
pub struct SpanTable {
    /// Held so the device pointer below stays valid until the launch.
    _buf: Tensor,
    ptr: u64,
    spans: usize,
    /// Every history pointer in the table is 16-byte aligned.
    vec_ok: bool,
}

impl SpanTable {
    pub fn build(
        entries: &[SpanEntry<'_>],
        dev: &Device,
        root: Option<WaveTicket>,
    ) -> Result<Self> {
        let mut desc: Vec<i64> = Vec::with_capacity(PLE_SPAN_WORDS * entries.len());
        let mut vec_ok = true;
        for e in entries {
            if e.hist.same_storage(e.new_hist) {
                candle::bail!(
                    "ple conv: span at row {} writes its next history over the one it reads",
                    e.start
                );
            }
            let (hist_ptr, a) = device_ptr(e.hist, "ple conv: history")?;
            let (new_ptr, b) = device_ptr(e.new_hist, "ple conv: next history")?;
            vec_ok &= a && b;
            desc.extend([
                e.start as i64,
                e.len as i64,
                hist_ptr as i64,
                new_ptr as i64,
            ]);
        }
        let buf = wave_from_vec_ticketed(desc, (PLE_SPAN_WORDS * entries.len(),), dev, root)?;
        let ptr = table_ptr(&buf)?;
        Ok(Self {
            _buf: buf,
            ptr,
            spans: entries.len(),
            vec_ok,
        })
    }
}

fn all_aligned(ops: &[&Operand]) -> bool {
    ops.iter().all(|o| o.vec_ok)
}

/// A dense F32 buffer's device pointer and whether it is 16-byte aligned, for
/// a descriptor that outlives the closure [`with_operand`] scopes.
///
/// Sound to hold past the call because the tensors it is asked about — a
/// sequence's history buffers — are held by the caller until the launch that
/// reads the descriptor is enqueued.
fn device_ptr(t: &Tensor, what: &str) -> Result<(u64, bool)> {
    with_operand(t, what, |o, _| (o.ptr, o.vec_ok))
}

/// The span table's device pointer. Held past the call for the same reason as
/// [`device_ptr`]: [`SpanTable`] keeps the buffer until the launch is enqueued.
fn table_ptr(t: &LiveTensor<'_>) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cs) = &*storage else {
        candle::bail!("ple conv: span table is not on the device");
    };
    let stream = cs.device().cuda_stream();
    let slice = cs.as_cuda_slice::<i64>()?.slice(layout.start_offset()..);
    let (ptr, _guard) = slice.device_ptr(&stream);
    Ok(ptr)
}

#[cfg(test)]
mod tests {
    use candle::Device;

    use super::super::ple_bench::{gate_once, gate_waves};

    /// The bench's admissible gap: reassociation of the norms' reductions.
    const GATE: f32 = 2e-5;

    /// Fused against the eager chain on the device — the residual, every
    /// sequence's next history and the captured rows — across decode rows, a
    /// segment shorter than the history and one longer, and an odd width that
    /// takes the scalar path. Needs a card; there is nothing to check without
    /// one.
    #[test]
    fn the_fused_block_is_the_eager_chain() {
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        for (d, heads, lens) in [
            (128usize, 16usize, &[1usize, 1, 1][..]),
            (128, 16, &[3, 1, 20, 6][..]),
            (66, 2, &[5, 12][..]),
        ] {
            let gap = gate_once(&dev, 4, d, heads, lens, 0x51).unwrap();
            assert!(
                gap <= GATE,
                "d={d} spans {lens:?}: fused vs eager gap {gap}"
            );
        }
    }

    /// The history a wave reads is never the buffer it writes: over three
    /// consecutive waves each one reads what the last wrote and parks the
    /// buffer it read as the spare, and every wave still matches the eager
    /// chain.
    #[test]
    fn the_history_alternates_between_two_buffers() {
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let gap = gate_waves(&dev, 4, 128, 16, &[4, 1, 11], 3, 0x52).unwrap();
        assert!(gap <= GATE, "gap {gap}");
    }
}
