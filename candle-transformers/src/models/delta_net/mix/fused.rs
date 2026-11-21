//! The fused CUDA mixer, which runs the conv, the recurrence and the output
//! epilogue as a few launches over the whole wave, and the tally that records why
//! a wave fell back to the tensor-op path.
//!
//! The fused path runs only when the dtypes, geometry and device all allow it.
//! When one condition fails, the wave runs the same arithmetic as tensor ops, and
//! the tally names the condition that rejected it, since a silent fallback shows
//! up as slowness rather than as an error. Decode spans run as one conv launch
//! and one step launch however many sessions the wave carries, reading each
//! session's state through a pointer table. Capture and replay hooks copy and
//! rewrite a long span's scan inputs for speculative decoding.

use super::*;

/// Whether a fused-path guard admitted the wave, recording **which condition
/// rejected it** when it did not.
///
/// The guards are long conjunctions, and a miss on any one silently drops the
/// whole wave to the tensor-op fallback — the same arithmetic at a fraction of
/// the speed. That is a cliff you cannot see in a timing: the span table shows
/// `dn:mix` inflated ~89× per call and says nothing about why, and
/// `docs/qwen36_performance_plan.md` T1 is explicit that timings are the wrong
/// instrument here (the profile feature's own sync distorts width rows ~2×).
///
/// So this counts paths taken, not time spent, and names the first failing
/// condition. "The fused path ran 40 times and fell back 200" is a different
/// problem from "it never ran", and "it fell back on `qkv:f32`" is a different
/// problem again from "it fell back on `head_dim`".
///
/// # Cost, on a per-layer per-wave path
///
/// The conditions arrive as a slice of **thunks**, not of `bool`s, so the walk
/// short-circuits exactly as the `&&` chain it replaced did — the `state:cuda`
/// predicate is `O(n_seqs)` and must not run once the cheap dtype checks have
/// already rejected. The tally is a fixed array of atomics indexed by position,
/// so recording is one relaxed `fetch_add`: no hashing, no map, no lock. An
/// earlier version took a global `Mutex<HashMap>` and hashed a string pair on
/// every call while claiming to be free; it was neither.
#[cfg(feature = "cuda")]
pub(super) const MAX_DISPATCH_CONDS: usize = 12;

/// `[path][0]` is the fused count; `[path][i + 1]` counts rejections by
/// condition `i`. One path, so one row.
#[cfg(feature = "cuda")]
pub(super) static DISPATCH_TALLY: [[std::sync::atomic::AtomicU64; MAX_DISPATCH_CONDS + 1]; 1] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; MAX_DISPATCH_CONDS + 1] }; 1];

/// The paths, in `DISPATCH_TALLY` row order.
#[cfg(feature = "cuda")]
pub(super) const DISPATCH_PATHS: [&str; 1] = ["dn:mix"];

#[cfg(feature = "cuda")]
pub(super) fn dispatch_verdict(row: usize, conds: &[(&'static str, &dyn Fn() -> bool)]) -> bool {
    debug_assert!(conds.len() <= MAX_DISPATCH_CONDS);
    let reject = conds.iter().position(|(_, ok)| !ok());
    let slot = reject.map_or(0, |i| i + 1);
    DISPATCH_TALLY[row][slot].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    reject.is_none()
}

/// The tally [`dispatch_verdict`] has accumulated: `(path, outcome, count)`,
/// where the outcome is `FUSED` or the first condition that rejected. Sorted
/// most-frequent first. Zero rows are omitted.
#[cfg(feature = "cuda")]
pub fn dispatch_tally() -> Vec<(&'static str, &'static str, u64)> {
    let names: [&[&str]; 1] = [&[
        "device",
        "head_dim",
        "qkv:f32",
        "alpha:f32",
        "beta:f32",
        "z:f32",
        "dt_bias:f32",
        "a:f32",
        "norm:f32",
        "state:cuda",
        "short:conv",
    ]];
    let mut v = Vec::new();
    for (row, path) in DISPATCH_PATHS.iter().enumerate() {
        for (slot, cell) in DISPATCH_TALLY[row].iter().enumerate() {
            let n = cell.load(std::sync::atomic::Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            let outcome = if slot == 0 {
                "FUSED"
            } else {
                names[row].get(slot - 1).copied().unwrap_or("?")
            };
            v.push((*path, outcome, n));
        }
    }
    v.sort_by_key(|e| Reverse(e.2));
    v
}

/// The fused mixer up to its epilogue — the conv, the scan or decode step —
/// returning the whole-wave output `o` the norm-gate epilogue reads, or `None`
/// when this layer's dtypes, geometry or device rule the fused kernels out.
#[cfg(feature = "cuda")]
pub(super) fn fused_mix<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    ticket: Option<WaveTicket>,
) -> Result<Option<LiveTensor<'w>>> {
    let (t, _) = p.qkv.dims2()?;
    let d = dims.head_dim;
    let (qkv, w) = (&p.qkv, c);
    let key_dim = dims.key_dim();

    // The fused path: everything from the conv to the output projection in
    // one launch over the short spans, three over the long ones, two over the
    // decode spans, plus one epilogue launch over the whole wave. The conv
    // kernels' SiLU + Q|K-norm epilogue makes their output THE operand buffer —
    // q, k and v are strided views of it, the GQA broadcast is an index
    // (`kh = h % h_k`), the read scale is applied on load, the gates are
    // computed in-kernel, and each span writes its own rows of one shared
    // output. So the split/norm launches and their full-width copy, the
    // repeat/scale materialisations, the per-span operand copies, the output
    // concatenation, and the epilogue's three full-width intermediates do not
    // exist here. The tensor-op path (`mix_spans_by_ops`) is the same
    // arithmetic (the kernels are parity-locked to it) and serves CPU,
    // reference dtypes, and any other head geometry.
    //
    // A short multi-row span has one kernel, the short-span scan, compiled for
    // one conv width; a wave carrying such a span at another width runs by
    // ops whole.
    let fused_ok = dispatch_verdict(
        0,
        &[
            ("device", &|| qkv.device().is_cuda()),
            ("head_dim", &|| {
                d == super::super::cuda::DELTA_NET_PREFILL_DIM
            }),
            ("qkv:f32", &|| qkv.dtype() == DType::F32),
            ("alpha:f32", &|| p.alpha_lin.dtype() == DType::F32),
            ("beta:f32", &|| p.beta_lin.dtype() == DType::F32),
            ("z:f32", &|| p.z.dtype() == DType::F32),
            ("dt_bias:f32", &|| c.dt_bias.dtype() == DType::F32),
            ("a:f32", &|| c.a.dtype() == DType::F32),
            ("norm:f32", &|| c.norm.dtype() == DType::F32),
            ("state:cuda", &|| {
                seqs.iter().all(|s| s.state.s.device().is_cuda())
            }),
            ("short:conv", &|| {
                dims.conv_kernel == DELTA_NET_SHORT_SPAN_CONV
                    || !seqs.iter().any(|s| s.len > 1 && s.len <= TIMELINE_ROWS)
            }),
        ],
    );
    if !fused_ok {
        return Ok(None);
    }
    // The decode spans run as ONE conv launch and ONE step launch however
    // many sessions the wave carries, through the pointer table. The hot path
    // receives the table from the wave driver (built once per forward); the
    // reference path and unit tests, which call in without one, build a
    // per-layer table here — same kernels, same arithmetic, the upload merely
    // happens closer to the launch.
    let local = match table {
        Some(_) => None,
        None if seqs.iter().any(|s| s.len == 1) => {
            Some(super::super::cuda::build_layer_table(seqs, ticket)?)
        }
        None => None,
    };
    let table = table.or(local.as_ref());
    // The multi-row spans get the same treatment, built here because unlike the
    // decode table their extents are a property of THIS wave's packing rather
    // than of the sequences, so there is nothing for the driver to hoist out of
    // the layer sweep. Split by length: a short span — a verify block, a short
    // prompt — runs the short-span scan into its timeline, one launch doing the
    // conv, the intra-chunk solve and the state pass's reads on chip; a long
    // span runs the conv + scan pair into the prefill scratch. A wave carrying
    // both launches both.
    let (short, long) = super::super::cuda::build_span_tables(seqs, qkv)?;

    let conved = conv_fused_spans(
        qkv,
        w.conv,
        2 * key_dim,
        rms_eps as f32,
        table,
        long.as_ref(),
    )?;
    let o = conved.empty_beside((t, dims.value_dim()), DType::F32)?;
    let fused = super::super::cuda::DeltaNetFused {
        conved: &conved,
        alpha: &p.alpha_lin,
        blin: &p.beta_lin,
        dt_bias: c.dt_bias,
        a: c.a,
        o: &o,
        // See the read-scale comment on the fallback below: this is
        // load-bearing through the epilogue's epsilon floor.
        q_scale: (1.0 / (d as f64).sqrt()) as f32,
    };
    if let Some(tbl) = table {
        super::super::cuda::delta_net_decode_batch(&fused, tbl)?;
    }
    if let Some(sp) = short.as_ref() {
        super::super::cuda::delta_net_short_span_scan(&fused, qkv, w.conv, sp, rms_eps as f32)?;
    }
    if let Some(sp) = long.as_ref() {
        super::super::cuda::delta_net_prefill_scan(&fused, sp)?;
    }
    capture_spans(seqs, &conved, p)?;
    replay_spans(seqs, &fused, dims)?;
    Ok(Some(o))
}

/// Copy each captured span's scan inputs — its rows of the conv output and of
/// the two gate projections — into the rows its hook carries.
#[cfg(feature = "cuda")]
pub(super) fn capture_spans(
    seqs: &[DeltaNetSeq<'_>],
    conved: &LiveTensor<'_>,
    p: &DeltaNetProjections<'_>,
) -> Result<()> {
    for s in seqs {
        let SpanHook::Capture(rows) = &s.hook else {
            continue;
        };
        rows.qkv
            .slice_set(&conved.narrow(0, s.start, s.len)?, 0, 0)?;
        rows.alpha
            .slice_set(&p.alpha_lin.narrow(0, s.start, s.len)?, 0, 0)?;
        rows.beta
            .slice_set(&p.beta_lin.narrow(0, s.start, s.len)?, 0, 0)?;
    }
    Ok(())
}

/// For each replayed span, run the record's rows through the span's entering
/// state with the wave's own scan and write the result to the span's advanced
/// state — over what the live scan wrote there, on the same stream, so the
/// span's outputs are its own and the state it carries on is the record's.
#[cfg(feature = "cuda")]
pub(super) fn replay_spans(
    seqs: &mut [DeltaNetSeq<'_>],
    fused: &super::super::cuda::DeltaNetFused<'_, '_>,
    dims: &DeltaNetDims,
) -> Result<()> {
    for s in seqs.iter_mut() {
        let SpanHook::Replay(rows) = &s.hook else {
            continue;
        };
        let rows = rows.clone();
        let n = rows.tokens()?;
        let o = rows.qkv.empty_beside((n, dims.value_dim()), DType::F32)?;
        let replay = super::super::cuda::DeltaNetFused {
            conved: &rows.qkv,
            alpha: &rows.alpha,
            blin: &rows.beta,
            dt_bias: fused.dt_bias,
            a: fused.a,
            o: &o,
            q_scale: fused.q_scale,
        };
        let one = [DeltaNetSeq {
            start: 0,
            len: n,
            state: &mut *s.state,
            write: s.write.clone(),
            hook: SpanHook::None,
        }];
        let (_, Some(table)) = super::super::cuda::build_span_tables(&one, &rows.qkv)? else {
            candle::bail!("delta_net: a replay of {n} rows is not a long span");
        };
        super::super::cuda::delta_net_prefill_scan(&replay, &table)?;
    }
    Ok(())
}

/// A hook is a long span's: a short span writes a timeline, which neither a
/// capture nor a replay has a state to stand beside.
pub(super) fn check_span_hooks(seqs: &[DeltaNetSeq<'_>]) -> Result<()> {
    match seqs
        .iter()
        .find(|s| !s.hook.is_none() && s.len <= TIMELINE_ROWS)
    {
        Some(s) => candle::bail!(
            "delta_net_mix_spans: a span of {} rows carries a capture or replay, which a span of \
             at most {TIMELINE_ROWS} rows cannot",
            s.len
        ),
        None => Ok(()),
    }
}
