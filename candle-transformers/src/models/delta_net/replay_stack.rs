//! The speculative rewind's DeltaNet replay over a **stack** of recurrent
//! layers, in one launch per kernel.
//!
//! A rewind re-runs each verifying sequence's accepted prefix through every
//! recurrent layer, from the state the block was entered with. No layer's
//! replay reads another's output — each starts from its own entering state and
//! its own stashed operands — so the layers are as independent as the spans
//! are, and they share the launch the same way: the prefill kernels'
//! `blockIdx.z` covers `layer · spans + span`. Replayed a layer at a time the
//! same work was 36 launch triples on Flash-Next, each a few microseconds of
//! kernel behind a launch gap; stacked it is one triple.
//!
//! The kernels are the very ones the verify wave ran (`delta_net_conv_prefill`,
//! `delta_net_prefill_scan`), so the replay retraces the wave's arithmetic by
//! construction — bit-identity to the wave is the contract a rewind rests on.
//! What differs per layer reaches them two ways: the layer's own allocations
//! (stashed projections, constants) through a device table, and everything the
//! launch carves (conv output, scan transients, discarded output, state
//! pointers) as one layer-stacked buffer each.

use candle::{DType, Device, LiveTensor, Result};
use candle_kernels::delta_net::{
    run_delta_net_conv_prefill_f32, run_delta_net_prefill_intra_f32,
    run_delta_net_prefill_state_f32, DELTA_NET_LAYER_OPS, DELTA_NET_MAX_LAYER_SPANS,
    DELTA_NET_PREFILL_CHUNK, DELTA_NET_PREFILL_DIM,
};

use super::cuda::{f32_ptr, typed_ptr};
use super::mix::{DeltaNetConstants, DeltaNetOut, DeltaNetProjections, DeltaNetState};
use super::types::DeltaNetDims;
use crate::models::operand_guard::expect_dense_dtype;

/// One rewinding sequence's rows of the stash buffers: the same rows in every
/// layer, because every layer's stash is laid out by the same cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplaySpan {
    /// First stash row.
    pub start: usize,
    /// Rows replayed — the accepted prefix.
    pub len: usize,
}

/// One span's four carried buffers in one layer, as device addresses: the conv
/// tail and state the block entered with, and the halves the replay writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayStates {
    pub tail_in: u64,
    pub tail_out: u64,
    pub state_in: u64,
    pub state_out: u64,
}

impl ReplayStates {
    /// The addresses of `entering` and `out`, checked against `dims`.
    ///
    /// The conv tails must be separate buffers: the conv's blocks that read the
    /// entering tail run alongside the one that writes the advance. The states
    /// may share one (the state pass holds its tile in shared memory before it
    /// stores).
    pub fn of(entering: &DeltaNetState, out: &DeltaNetOut, dims: &DeltaNetDims) -> Result<Self> {
        let d = dims.head_dim;
        let state = (dims.n_v_heads, d, d);
        let tail = (dims.conv_dim(), dims.conv_kernel - 1);
        if entering.s.dims3()? != state || out.s.dims3()? != state {
            candle::bail!(
                "delta_net replay stack: states {:?} / {:?} against the mixer's {state:?}",
                entering.s.dims(),
                out.s.dims()
            );
        }
        if entering.conv_tail.dims2()? != tail || out.conv_tail.dims2()? != tail {
            candle::bail!(
                "delta_net replay stack: conv tails {:?} / {:?} against the mixer's {tail:?}",
                entering.conv_tail.dims(),
                out.conv_tail.dims()
            );
        }
        if entering.conv_tail.same_storage(&out.conv_tail) {
            candle::bail!(
                "delta_net replay stack: a span's entering and advanced conv tails must be \
                 separate buffers"
            );
        }
        Ok(Self {
            tail_in: f32_ptr(&entering.conv_tail, "conv tail in")?,
            tail_out: f32_ptr(&out.conv_tail, "conv tail out")?,
            state_in: f32_ptr(&entering.s, "state in")?,
            state_out: f32_ptr(&out.s, "state out")?,
        })
    }
}

/// One recurrent layer of a stack: its stashed projections and constants, and
/// each span's carried buffers in span order.
pub struct StackedLayer<'a, 'w> {
    pub p: &'a DeltaNetProjections<'w>,
    pub c: &'a DeltaNetConstants<'a>,
    pub states: Vec<ReplayStates>,
}

/// Advance every span's carried state in every layer of `layers` over its rows
/// of that layer's stash — states only; the scan's output is written and
/// discarded.
///
/// `conved` is the conv output, `[layers · rows, conv_dim]` F32 where `rows` is
/// the stash's row count, and the replay's **provenance root**: the stash names
/// no arena, so the tables and every transient are placed beside `conved` on
/// whatever span the caller carved it from. It arrives uninitialised — the conv
/// writes every row the scan reads (invariant 6).
///
/// Spans must be disjoint and ascending, and need not tile the stash: an accept
/// that kept fewer rows than its block leaves rows no kernel touches. Every span
/// runs through the prefill kernels whatever its length, one row included: the
/// wave ran the stashed rows as prefill rows, and retracing a one-row accept
/// through the decode kernels instead would be a different reduction order.
pub fn delta_net_replay_stack(
    layers: &[StackedLayer<'_, '_>],
    spans: &[ReplaySpan],
    dims: &DeltaNetDims,
    eps: f32,
    conved: &LiveTensor<'_>,
) -> Result<()> {
    let (g, n) = (layers.len(), spans.len());
    if g == 0 || n == 0 {
        candle::bail!("delta_net replay stack: {g} layers over {n} spans");
    }
    if g * n > DELTA_NET_MAX_LAYER_SPANS {
        candle::bail!(
            "delta_net replay stack: {g} layers × {n} spans pass the launch's \
             {DELTA_NET_MAX_LAYER_SPANS}"
        );
    }
    let d = dims.head_dim;
    if d != DELTA_NET_PREFILL_DIM {
        candle::bail!(
            "delta_net replay stack: the prefill kernels are compiled for d == \
             {DELTA_NET_PREFILL_DIM}, got {d}"
        );
    }
    let (h_v, h_k) = (dims.n_v_heads, dims.n_k_heads);
    if h_k == 0 || !h_v.is_multiple_of(h_k) {
        candle::bail!("delta_net replay stack: h_v {h_v} must be a multiple of h_k {h_k}");
    }
    let (conv_dim, value_dim) = (dims.conv_dim(), dims.value_dim());
    let qk_channels = 2 * dims.key_dim();
    if !qk_channels.is_multiple_of(256) {
        candle::bail!(
            "delta_net replay stack: qk_channels {qk_channels} must be a multiple of 256"
        );
    }
    let rows = layers[0].p.qkv.dim(0)?;
    let mut cursor = 0usize;
    for (i, s) in spans.iter().enumerate() {
        if s.start < cursor || s.len == 0 {
            candle::bail!(
                "delta_net replay stack: span {i} at {}+{} overlaps the one before it \
                 (ends at {cursor}) — spans must be disjoint and ascending",
                s.start,
                s.len
            );
        }
        cursor = s.start + s.len;
    }
    if cursor > rows {
        candle::bail!("delta_net replay stack: spans reach row {cursor} of a {rows}-row stash");
    }
    expect_dense_dtype(conved, DType::F32, "replay stack conved")?;
    if conved.dims() != [g * rows, conv_dim] {
        candle::bail!(
            "delta_net replay stack: conved is {:?}, the stack convolves [{}, {conv_dim}]",
            conved.dims(),
            g * rows
        );
    }

    // ONE upload: every layer's operand pointers, then every layer's four rows
    // of span pointers — the order `DnLayerOps` and `dn_span` read.
    let mut table: Vec<i64> = Vec::with_capacity(g * (DELTA_NET_LAYER_OPS + 4 * n));
    for (l, layer) in layers.iter().enumerate() {
        let p = layer.p;
        if p.qkv.dims2()? != (rows, conv_dim) {
            candle::bail!(
                "delta_net replay stack: layer {l}'s qkv is {:?}, the stack's is [{rows}, \
                 {conv_dim}]",
                p.qkv.dims()
            );
        }
        if p.alpha_lin.dims2()? != (rows, h_v) || p.beta_lin.dims2()? != (rows, h_v) {
            candle::bail!(
                "delta_net replay stack: layer {l}'s gate projections are not [{rows}, {h_v}]"
            );
        }
        if layer.c.conv.dims2()? != (conv_dim, dims.conv_kernel) {
            candle::bail!(
                "delta_net replay stack: layer {l}'s conv kernel is {:?}, not [{conv_dim}, {}]",
                layer.c.conv.dims(),
                dims.conv_kernel
            );
        }
        if layer.c.dt_bias.dims1()? != h_v || layer.c.a.dims1()? != h_v {
            candle::bail!("delta_net replay stack: layer {l}'s dt_bias/a are not [{h_v}]");
        }
        if layer.states.len() != n {
            candle::bail!(
                "delta_net replay stack: layer {l} carries {} span states for {n} spans",
                layer.states.len()
            );
        }
        for ptr in [
            f32_ptr(&p.qkv, "qkv")?,
            f32_ptr(layer.c.conv, "conv kernel")?,
            f32_ptr(&p.alpha_lin, "alpha")?,
            f32_ptr(&p.beta_lin, "beta_lin")?,
            f32_ptr(layer.c.dt_bias, "dt_bias")?,
            f32_ptr(layer.c.a, "a")?,
        ] {
            table.push(ptr as i64);
        }
    }
    for layer in layers {
        table.extend(layer.states.iter().map(|s| s.tail_in as i64));
        table.extend(layer.states.iter().map(|s| s.tail_out as i64));
        table.extend(layer.states.iter().map(|s| s.state_in as i64));
        table.extend(layer.states.iter().map(|s| s.state_out as i64));
    }
    let mut extents: Vec<u32> = Vec::with_capacity(2 * n);
    extents.extend(spans.iter().map(|s| s.start as u32));
    extents.extend(spans.iter().map(|s| s.len as u32));
    let max_len = spans.iter().map(|s| s.len).max().unwrap_or(0);

    let table_len = table.len();
    let table = conved.from_vec_beside(table, table_len)?;
    let extents = conved.from_vec_beside(extents, (2, n))?;
    let o = conved.empty_beside((g * rows, value_dim), DType::F32)?;
    let u = conved.empty_beside((g, h_v, rows, d), DType::F32)?;
    let w = conved.empty_beside((g, h_v, rows, d), DType::F32)?;
    let kq = conved.empty_beside((g, h_v, rows, DELTA_NET_PREFILL_CHUNK), DType::F32)?;
    let g_cs = conved.empty_beside((g, h_v, rows), DType::F32)?;

    let dev = match conved.device() {
        Device::Cuda(dv) => dv.clone(),
        _ => candle::bail!("delta_net replay stack: conved must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    let raw = stream.cu_stream() as *mut core::ffi::c_void;
    let ops_p = typed_ptr(&table, DType::I64, "layer table")?;
    let ptrs_p = ops_p + (g * DELTA_NET_LAYER_OPS * std::mem::size_of::<i64>()) as u64;
    let spans_p = typed_ptr(&extents, DType::U32, "span extents")?;
    let conved_p = f32_ptr(conved, "conved")?;
    let v_p = conved_p + ((conv_dim - value_dim) * std::mem::size_of::<f32>()) as u64;
    let (o_p, u_p, w_p) = (f32_ptr(&o, "o")?, f32_ptr(&u, "u")?, f32_ptr(&w, "w")?);
    let (kq_p, gcs_p) = (f32_ptr(&kq, "kq")?, f32_ptr(&g_cs, "g_cs")?);
    let q_scale = (1.0 / (d as f64).sqrt()) as f32;
    // The per-layer operands the table supplies are ignored by the kernels in
    // a stacked launch; the first layer's are passed so no argument is null.
    let first = &layers[0];
    let (x0, k0) = (
        f32_ptr(&first.p.qkv, "qkv")?,
        f32_ptr(first.c.conv, "conv")?,
    );
    let (a0, b0) = (
        f32_ptr(&first.p.alpha_lin, "alpha")?,
        f32_ptr(&first.p.beta_lin, "beta_lin")?,
    );
    let (dt0, an0) = (
        f32_ptr(first.c.dt_bias, "dt_bias")?,
        f32_ptr(first.c.a, "a")?,
    );
    unsafe {
        run_delta_net_conv_prefill_f32(
            x0 as *const f32,
            k0 as *const f32,
            conved_p as *mut f32,
            ptrs_p as *const i64,
            spans_p as *const u32,
            n as i32,
            ops_p as *const i64,
            g as i32,
            rows as i32,
            max_len as i32,
            conv_dim as i32,
            dims.conv_kernel as i32,
            qk_channels as i32,
            eps,
            raw,
        );
        run_delta_net_prefill_intra_f32(
            conved_p as *const f32,
            v_p as *const f32,
            a0 as *const f32,
            b0 as *const f32,
            dt0 as *const f32,
            an0 as *const f32,
            u_p as *mut f32,
            w_p as *mut f32,
            kq_p as *mut f32,
            gcs_p as *mut f32,
            spans_p as *const u32,
            n as i32,
            ops_p as *const i64,
            g as i32,
            max_len as i32,
            rows as i32,
            h_v as i32,
            h_k as i32,
            conv_dim as i32,
            q_scale,
            raw,
        );
        run_delta_net_prefill_state_f32(
            conved_p as *const f32,
            u_p as *const f32,
            w_p as *const f32,
            kq_p as *const f32,
            gcs_p as *const f32,
            o_p as *mut f32,
            ptrs_p as *const i64,
            spans_p as *const u32,
            n as i32,
            ops_p as *const i64,
            g as i32,
            rows as i32,
            h_v as i32,
            h_k as i32,
            conv_dim as i32,
            q_scale,
            raw,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::delta_net::cuda::{
        delta_net_conv_prefill, delta_net_prefill_scan, DeltaNetFused,
    };
    use crate::models::delta_net::DeltaNetSpanTable;
    use candle::Tensor;
    use candle_nn::kv_cache::{DELTA_NET_REPLAY_LAYER_OPS, DELTA_NET_REPLAY_MAX_LAYER_SPANS};

    fn lcg(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// One layer's stashed operands and constants, and per span the entering
    /// buffers plus two sets of output buffers — one for the wave's launch, one
    /// for the stack's.
    struct Layer {
        p: DeltaNetProjections<'static>,
        conv: Tensor,
        dt_bias: Tensor,
        a: Tensor,
        norm: Tensor,
        entering: Vec<DeltaNetState>,
        wave_out: Vec<DeltaNetOut>,
        stack_out: Vec<DeltaNetOut>,
    }

    /// **The stack is the wave's arithmetic, to the bit.** Three layers of a
    /// mixer with GQA (2 K heads under 4 V heads) and two rewinding spans — a
    /// three-row accept and a one-row accept, with an unreplayed row between
    /// them — replayed once as a stack and once a layer at a time through the
    /// very wrappers the verify wave calls. Every advanced state and conv tail
    /// must match byte for byte: a rewind the sequence can tell apart from the
    /// wave is a rewind that changed the conversation.
    #[test]
    fn a_stacked_replay_is_each_layers_wave_launch_bit_for_bit() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let dims = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 4,
        };
        let (rows, n_layers) = (6usize, 3usize);
        let spans = [
            ReplaySpan { start: 0, len: 3 },
            ReplaySpan { start: 4, len: 1 },
        ];
        let (conv_dim, h_v, d) = (dims.conv_dim(), dims.n_v_heads, dims.head_dim);
        let tail_cols = dims.conv_kernel - 1;
        let eps = 1e-6f32;
        let zeros = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &gpu).unwrap();
        let out = || -> DeltaNetOut {
            DeltaNetOut {
                s: zeros(&[h_v, d, d]),
                conv_tail: zeros(&[conv_dim, tail_cols]),
            }
        };
        let layers: Vec<Layer> = (0..n_layers as u64)
            .map(|l| {
                let seed = 100 * (l + 1);
                Layer {
                    p: DeltaNetProjections {
                        qkv: lcg(&[rows, conv_dim], seed, 2.0, &gpu),
                        z: lcg(&[rows, dims.value_dim()], seed + 1, 1.0, &gpu),
                        beta_lin: lcg(&[rows, h_v], seed + 2, 2.0, &gpu),
                        alpha_lin: lcg(&[rows, h_v], seed + 3, 2.0, &gpu),
                    },
                    conv: lcg(&[conv_dim, dims.conv_kernel], seed + 4, 1.0, &gpu),
                    dt_bias: lcg(&[h_v], seed + 5, 1.0, &gpu),
                    a: (lcg(&[h_v], seed + 6, 1.0, &gpu).abs().unwrap() + 0.1)
                        .unwrap()
                        .neg()
                        .unwrap(),
                    norm: zeros(&[d]),
                    entering: (0..spans.len() as u64)
                        .map(|s| DeltaNetState {
                            s: lcg(&[h_v, d, d], seed + 10 + s, 0.1, &gpu),
                            conv_tail: lcg(&[conv_dim, tail_cols], seed + 20 + s, 1.0, &gpu),
                        })
                        .collect(),
                    wave_out: spans.iter().map(|_| out()).collect(),
                    stack_out: spans.iter().map(|_| out()).collect(),
                }
            })
            .collect();
        let consts: Vec<DeltaNetConstants<'_>> = layers
            .iter()
            .map(|l| DeltaNetConstants {
                dt_bias: &l.dt_bias,
                a: &l.a,
                conv: &l.conv,
                norm: &l.norm,
            })
            .collect();

        // The wave's launches, a layer at a time.
        for (l, layer) in layers.iter().enumerate() {
            let mut ptrs: Vec<i64> = Vec::new();
            for col in 0..4 {
                for (s, entering) in layer.entering.iter().enumerate() {
                    let o = &layer.wave_out[s];
                    let t = [&entering.conv_tail, &o.conv_tail, &entering.s, &o.s][col];
                    ptrs.push(f32_ptr(t, "table").unwrap() as i64);
                }
            }
            let mut ext: Vec<u32> = spans.iter().map(|s| s.start as u32).collect();
            ext.extend(spans.iter().map(|s| s.len as u32));
            let table = DeltaNetSpanTable {
                ptrs: Tensor::from_vec(ptrs, (4, spans.len()), &gpu).unwrap(),
                spans: Tensor::from_vec(ext, (2, spans.len()), &gpu).unwrap(),
                n: spans.len(),
                max_len: 3,
            };
            let conved = zeros(&[rows, conv_dim]);
            delta_net_conv_prefill(
                &layer.p.qkv,
                &layer.conv,
                &table,
                2 * dims.key_dim(),
                eps,
                &conved,
            )
            .unwrap();
            let o = zeros(&[rows, dims.value_dim()]);
            let fused = DeltaNetFused {
                conved: &conved,
                alpha: &layer.p.alpha_lin,
                blin: &layer.p.beta_lin,
                dt_bias: consts[l].dt_bias,
                a: consts[l].a,
                o: &o,
                q_scale: (1.0 / (d as f64).sqrt()) as f32,
            };
            delta_net_prefill_scan(&fused, &table).unwrap();
        }

        // The stack, every layer in one launch triple.
        let stacked: Vec<StackedLayer<'_, '_>> = layers
            .iter()
            .zip(&consts)
            .map(|(layer, c)| StackedLayer {
                p: &layer.p,
                c,
                states: layer
                    .entering
                    .iter()
                    .zip(&layer.stack_out)
                    .map(|(e, o)| ReplayStates::of(e, o, &dims).unwrap())
                    .collect(),
            })
            .collect();
        let conved = zeros(&[n_layers * rows, conv_dim]);
        delta_net_replay_stack(&stacked, &spans, &dims, eps, &conved).unwrap();

        for (l, layer) in layers.iter().enumerate() {
            for s in 0..spans.len() {
                let (w, k) = (&layer.wave_out[s], &layer.stack_out[s]);
                let state = bits(&w.s);
                assert!(
                    state.iter().any(|&b| b != 0),
                    "layer {l} span {s}: the wave left the state unwritten"
                );
                assert_eq!(state, bits(&k.s), "layer {l} span {s}: state");
                assert_eq!(
                    bits(&w.conv_tail),
                    bits(&k.conv_tail),
                    "layer {l} span {s}: conv tail"
                );
            }
        }
    }

    /// The plan prices the stack with its own copies of the kernels' table
    /// width and grid extent, out of reach of a build without CUDA; they must
    /// be the kernels' numbers.
    #[test]
    fn the_plan_prices_the_kernels_table_and_grid() {
        assert_eq!(DELTA_NET_REPLAY_LAYER_OPS, DELTA_NET_LAYER_OPS);
        assert_eq!(DELTA_NET_REPLAY_MAX_LAYER_SPANS, DELTA_NET_MAX_LAYER_SPANS);
    }

    /// A stack refuses spans that overlap, and a conv buffer not sized for
    /// every layer it holds.
    #[test]
    fn a_stack_refuses_overlapping_spans_and_a_short_conv_buffer() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let dims = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 1,
            conv_kernel: 4,
        };
        let conv_dim = dims.conv_dim();
        let z = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &gpu).unwrap();
        let p = DeltaNetProjections {
            qkv: z(&[4, conv_dim]),
            z: z(&[4, 128]),
            beta_lin: z(&[4, 1]),
            alpha_lin: z(&[4, 1]),
        };
        let (dt, a, conv, norm) = (z(&[1]), z(&[1]), z(&[conv_dim, 4]), z(&[128]));
        let c = DeltaNetConstants {
            dt_bias: &dt,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let entering = DeltaNetState {
            s: z(&[1, 128, 128]),
            conv_tail: z(&[conv_dim, 3]),
        };
        let out = DeltaNetOut {
            s: z(&[1, 128, 128]),
            conv_tail: z(&[conv_dim, 3]),
        };
        let st = ReplayStates::of(&entering, &out, &dims).unwrap();
        let layer = |states: Vec<ReplayStates>| StackedLayer {
            p: &p,
            c: &c,
            states,
        };
        let overlapping = [
            ReplaySpan { start: 0, len: 2 },
            ReplaySpan { start: 1, len: 1 },
        ];
        let err = delta_net_replay_stack(
            &[layer(vec![st, st])],
            &overlapping,
            &dims,
            1e-6,
            &z(&[4, conv_dim]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("disjoint and ascending"), "{err}");
        let one = [ReplaySpan { start: 0, len: 2 }];
        let err = delta_net_replay_stack(
            &[layer(vec![st]), layer(vec![st])],
            &one,
            &dims,
            1e-6,
            &z(&[4, conv_dim]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("the stack convolves [8,"), "{err}");
    }
}
