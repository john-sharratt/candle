//! Replay the captured MoE gate GEMM offline and ask whether it is deterministic.
//!
//! This test consumes the bundle written by
//! `models::moe_gemm_capture::capture_if_armed` when the assert instrumentation
//! catches `moe.gate_out.L*` going non-finite in production. It exists to split
//! one fork that no amount of in-process logging can:
//!
//! * **The replay reproduces the bad output.** Then those bytes really do
//!   multiply out to that value, the kernel is behaving as written, and the
//!   fault is upstream in whatever produced the operand or the tile tables. The
//!   search becomes a bisect over the inputs, which is tractable offline.
//!
//! * **The replay produces something different** — or differs between its own
//!   runs. Then the bytes the kernel read in production were not the bytes that
//!   were supposed to be there: a copy that had not landed, a buffer recycled
//!   under an in-flight read, an arena lease outliving its generation. That is a
//!   lifetime/ordering fault and a completely different search.
//!
//! Run after a capture:
//! ```text
//! cargo test -p candle-transformers --features cuda,tensor-assert --test moe_gemm_replay -- --nocapture
//! ```
//!
//! With no dump present the test reports that and passes — it is a diagnostic
//! keyed to a specific capture, not a gate on the build.

#![cfg(all(feature = "cuda", feature = "tensor-assert"))]

use candle::quantized::cuda::{grouped_qmatmul_dev_q8a128, Q8a128Operand};
use candle::quantized::GgmlDType;
use candle::tensor_assert::Replay;
use candle::{DType, Device, Result};
use candle_transformers::models::nan_capture::DUMP_DIR;
use cudarc::driver::{CudaSlice, DevicePtr};

/// How many times to re-run the identical call. Two would show a difference;
/// more raises the chance of catching an intermittent one, and the whole replay
/// is milliseconds.
const RUNS: usize = 16;

fn parse_dtype(s: &str) -> Result<GgmlDType> {
    Ok(match s {
        "Q4_KO" => GgmlDType::Q4_KO,
        "Q5_KO" => GgmlDType::Q5_KO,
        "Q6_KO" => GgmlDType::Q6_KO,
        "Q8_KO" => GgmlDType::Q8_KO,
        "Q2_KO" => GgmlDType::Q2_KO,
        "MXFP4_KO" => GgmlDType::MXFP4_KO,
        other => candle::bail!("replay: unhandled weight dtype {other:?}"),
    })
}

fn parse_out_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        other => candle::bail!("replay: unhandled output dtype {other:?}"),
    })
}

/// Summarise a raw output buffer without interpreting its dtype beyond width.
fn stats(bytes: &[u8], dt: DType) -> (usize, usize, f32, f32) {
    let vals: Vec<f32> = match dt {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        _ => Vec::new(),
    };
    let nan = vals.iter().filter(|v| v.is_nan()).count();
    let inf = vals.iter().filter(|v| v.is_infinite()).count();
    let lo = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let hi = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    (nan, inf, lo, hi)
}

/// Which output rows are non-finite, and whether they are the rows the tile
/// tables actually asked the kernel to compute.
///
/// The grouped GEMM writes only the rows its tiles cover. The gather that fills
/// its operand allocates uninitialised (hot-path invariant 6: a buffer a kernel
/// fully overwrites must not be memset first), so the rows past the last valid
/// assignment hold whatever the arena held before — and a GEMM over those rows
/// produces garbage by construction. That garbage is *supposed* to be
/// unreachable: the scatter reads only valid rows.
///
/// So there are two completely different stories behind a NaN here, and they
/// are told apart by exactly one question — are the bad rows covered by a tile?
///
/// * **Uncovered.** The NaN is padding, harmless, and an assert on this tensor
///   is measuring noise rather than a fault. The real signal is downstream of
///   the scatter.
/// * **Covered.** A row the kernel was told to compute came out non-finite, and
///   the operand or the weights behind that row are the fault.
fn analyse_rows(
    r: &Replay,
    out: &[u8],
    dt: DType,
    ncols: usize,
    launch_tiles: usize,
) -> Result<()> {
    let esz = dt.size_in_bytes();
    let nrows = out.len() / (ncols * esz);
    let row_bad = |row: usize| -> bool {
        let (lo, hi) = (row * ncols * esz, (row + 1) * ncols * esz);
        let (nan, inf, _, _) = stats(&out[lo..hi], dt);
        nan > 0 || inf > 0
    };

    let starts = r.typed::<i32>("tile_b_start")?;
    let cnts = r.typed::<i32>("tile_b_cnt")?;
    let mut covered = vec![false; nrows];
    let mut covered_rows = 0usize;
    for t in 0..launch_tiles.min(starts.len()).min(cnts.len()) {
        let (s, c) = (starts[t].max(0) as usize, cnts[t].max(0) as usize);
        for row in s..(s + c).min(nrows) {
            if !covered[row] {
                covered[row] = true;
                covered_rows += 1;
            }
        }
    }

    let mut bad_covered = 0usize;
    let mut bad_uncovered = 0usize;
    let mut first_bad_covered: Option<usize> = None;
    for row in 0..nrows {
        if !row_bad(row) {
            continue;
        }
        if covered[row] {
            bad_covered += 1;
            first_bad_covered.get_or_insert(row);
        } else {
            bad_uncovered += 1;
        }
    }

    println!(
        "rows: {nrows} total, {covered_rows} covered by a tile, {} not covered",
        nrows - covered_rows
    );
    println!("non-finite rows: {bad_covered} covered, {bad_uncovered} uncovered");

    if bad_covered == 0 {
        println!(
            "VERDICT: every non-finite row is UNCOVERED — this is the uninitialised tail of the \
             gather operand, which no tile computes and the scatter never reads. It is not the \
             fault, and an assert on this tensor is measuring padding. Move the assert past the \
             scatter and re-capture."
        );
    } else {
        println!(
            "VERDICT: {bad_covered} row(s) the kernel was TOLD to compute came out non-finite \
             (first: row {}). The operand rows and expert weights behind those rows are the \
             fault — bisect from there.",
            first_bad_covered.unwrap_or(0)
        );
    }
    Ok(())
}

/// Decode the int8 operand and ask whether its bad rows are the output's bad rows.
///
/// The quantized *values* in a q8a128 operand are integers and cannot be
/// non-finite; only its per-group scales can. So a NaN row here is a NaN scale,
/// and the row sets settle where to look next:
///
/// * **They match.** The GEMM is faithfully multiplying a NaN operand — the
///   fault is in whatever produced the gather's source, upstream of the MoE.
/// * **The operand is clean.** Finite inputs and finite weights produced NaN
///   output, which leaves the tile tables addressing rows or experts that were
///   never written.
fn analyse_operand(
    r: &Replay,
    cu: &candle::cuda_backend::CudaDevice,
    out: &[u8],
    out_dt: DType,
    out_cols: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    use candle::tensor_assert::scratch::{dequantize_flat_into, QTYPE_Q8A128V};

    let bytes = r.bytes("stacked")?;
    let src = cu.memcpy_stod(&bytes).map_err(candle::Error::wrap)?;
    let n = rows * cols;
    let dst = unsafe { cu.alloc::<f32>(n) }.map_err(candle::Error::wrap)?;
    let stream = cu.cuda_stream();
    let (sp, _sg) = src.device_ptr(&stream);
    let (dp, _dg) = dst.device_ptr(&stream);
    dequantize_flat_into(sp, dp, n, QTYPE_Q8A128V);
    stream.synchronize().map_err(candle::Error::wrap)?;
    let vals = cu.memcpy_dtov(&dst).map_err(candle::Error::wrap)?;

    let esz = out_dt.size_in_bytes();
    let mut op_bad = 0usize;
    let mut both = 0usize;
    let mut op_only = 0usize;
    let mut out_only = 0usize;
    let mut first_op_bad: Option<usize> = None;
    for row in 0..rows {
        let ob = vals[row * cols..(row + 1) * cols]
            .iter()
            .any(|v| !v.is_finite());
        let (lo, hi) = (row * out_cols * esz, (row + 1) * out_cols * esz);
        let (nan, inf, _, _) = stats(&out[lo..hi], out_dt);
        let rb = nan > 0 || inf > 0;
        if ob {
            op_bad += 1;
            first_op_bad.get_or_insert(row);
        }
        match (ob, rb) {
            (true, true) => both += 1,
            (true, false) => op_only += 1,
            (false, true) => out_only += 1,
            (false, false) => {}
        }
    }
    let finite_lo = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let finite_hi = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    println!(
        "operand: {op_bad} of {rows} rows non-finite, finite=[{finite_lo:e}, {finite_hi:e}] \
         (first bad row {:?})",
        first_op_bad
    );
    println!("row agreement: both={both} operand-only={op_only} output-only={out_only}");

    if op_bad > 0 && out_only == 0 {
        println!(
            "VERDICT: the GEMM is faithfully multiplying a NON-FINITE OPERAND. Every bad output \
             row has a bad operand row behind it, so the fault is upstream of this matmul — in \
             the gather's source, i.e. the layer's normed activation and the residual that fed \
             it. The MoE is a victim here, not the cause."
        );
    } else if op_bad == 0 {
        println!(
            "VERDICT: the operand is entirely FINITE and the weights are proven finite, yet the \
             output is not. That leaves the tile tables — rows or experts addressed that were \
             never written. Inspect tile_expert/tile_b_start against the operand's extent."
        );
    } else {
        println!(
            "VERDICT: MIXED — {out_only} output row(s) went bad with a finite operand row and a \
             finite weight. Those rows are the ones to bisect; the rest follow their operand."
        );
    }
    Ok(())
}

/// Audit the tile tables against the extents the kernel will index with them.
///
/// The grouped GEMM does, per tile `t`,
/// `weight_ptrs[expert_base + tile_expert[t]]` and reads `tile_b_cnt[t]` rows
/// starting at `tile_b_start[t]`. Nothing in the kernel range-checks either.
/// So when the operand is finite and the weights are finite and the output is
/// not, the remaining way to produce a wrong number is to address the wrong
/// thing — an expert id outside the table, or a row span past the operand.
///
/// Both are visible here as pure arithmetic on the dumped tables, with no
/// device involved.
/// Decode every expert slot in the dump and report which ones are corrupt.
///
/// One corrupt slot and 255 clean ones is a targeted overwrite of that slot's
/// VRAM; a contiguous run of corrupt slots is something that wrote across a
/// span; all of them corrupt would mean the pointer table, not the memory.
/// The shape of the damage is what names the writer.
fn scan_all_experts(r: &Replay, cu: &candle::cuda_backend::CudaDevice) -> Result<()> {
    let w_dtype = parse_dtype(&r.get::<String>("weight_dtype")?)?;
    let w_nrows: usize = r.get("weight_nrows")?;
    let w_cols: usize = r.get("cols")?;
    let num_experts: usize = r.get("num_experts")?;
    let shape = candle::Shape::from((w_nrows, w_cols));
    let n_w = w_nrows * w_cols;

    let dst = unsafe { cu.alloc::<f32>(n_w) }.map_err(candle::Error::wrap)?;
    let stream = cu.cuda_stream();
    let (dp, _dg) = dst.device_ptr(&stream);

    let mut corrupt: Vec<usize> = Vec::new();
    let mut worst = f32::NEG_INFINITY;
    for e in 0..num_experts {
        let Ok(bytes) = r.bytes(&format!("weight_{e:05}")) else {
            continue;
        };
        let src = cu.memcpy_stod(&bytes).map_err(candle::Error::wrap)?;
        let (sp, _sg) = src.device_ptr(&stream);
        candle::tensor_assert::scratch::dequantize_into(sp, dp, &shape, w_dtype)?;
        stream.synchronize().map_err(candle::Error::wrap)?;
        let v = cu.memcpy_dtov(&dst).map_err(candle::Error::wrap)?;
        let bad = v.iter().any(|x| !x.is_finite());
        let mag = v
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(0f32, |a, b| a.max(b.abs()));
        // A KO weight of this checkpoint lives in [-0.78, 0.78]; anything an
        // order of magnitude past that is damage even without a NaN in it.
        if bad || mag > 8.0 {
            corrupt.push(e);
            worst = worst.max(mag);
        }
    }
    println!(
        "expert slot scan: {} of {num_experts} corrupt (worst finite magnitude {worst:e})",
        corrupt.len()
    );
    if corrupt.is_empty() {
        return Ok(());
    }
    println!("  corrupt slots: {corrupt:?}");
    let contiguous = corrupt.windows(2).all(|w| w[1] == w[0] + 1);
    if corrupt.len() == 1 {
        println!(
            "  SHAPE: a SINGLE slot. Something wrote over exactly one expert's VRAM — an \
             aliased allocation handed out on top of a resident weight, not a broad overrun."
        );
    } else if contiguous {
        println!(
            "  SHAPE: a CONTIGUOUS RUN of {} slots ({}..={}). That is one write of a known \
             length landing at the wrong base — look for a buffer whose size matches.",
            corrupt.len(),
            corrupt[0],
            corrupt[corrupt.len() - 1]
        );
    } else {
        println!(
            "  SHAPE: {} SCATTERED slots. Not one overrun — either several independent \
             aliases or a strided writer.",
            corrupt.len()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn analyse_tiles(
    r: &Replay,
    cu: &candle::cuda_backend::CudaDevice,
    out: &[u8],
    dt: DType,
    ncols: usize,
    rows: usize,
    num_experts: usize,
    launch_tiles: usize,
) -> Result<()> {
    let experts = r.typed::<i32>("tile_expert")?;
    let starts = r.typed::<i32>("tile_b_start")?;
    let cnts = r.typed::<i32>("tile_b_cnt")?;
    let n = launch_tiles
        .min(experts.len())
        .min(starts.len())
        .min(cnts.len());

    let rng = |v: &[i32]| -> (i32, i32) {
        v[..n]
            .iter()
            .fold((i32::MAX, i32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)))
    };
    let (e_lo, e_hi) = rng(&experts);
    let (s_lo, s_hi) = rng(&starts);
    let (c_lo, c_hi) = rng(&cnts);
    println!(
        "tiles({n}): expert=[{e_lo}, {e_hi}] vs experts={num_experts} | \
         start=[{s_lo}, {s_hi}] cnt=[{c_lo}, {c_hi}] vs rows={rows}"
    );

    let bad_expert: Vec<usize> = (0..n)
        .filter(|&t| experts[t] < 0 || experts[t] as usize >= num_experts)
        .collect();
    let bad_span: Vec<usize> = (0..n)
        .filter(|&t| {
            let (s, c) = (starts[t], cnts[t]);
            s < 0 || c < 0 || (s as i64 + c as i64) > rows as i64
        })
        .collect();

    if !bad_expert.is_empty() {
        println!(
            "OUT-OF-RANGE EXPERT ID in {} tile(s); first tile {} → expert {}",
            bad_expert.len(),
            bad_expert[0],
            experts[bad_expert[0]]
        );
    }
    if !bad_span.is_empty() {
        println!(
            "OUT-OF-RANGE ROW SPAN in {} tile(s); first tile {} → start {} cnt {}",
            bad_span.len(),
            bad_span[0],
            starts[bad_span[0]],
            cnts[bad_span[0]]
        );
    }

    // Which tiles actually cover the bad output rows, and what they carry.
    let esz = dt.size_in_bytes();
    let n_out_rows = out.len() / (ncols * esz);
    let mut shown = 0;
    for row in 0..n_out_rows {
        let (lo, hi) = (row * ncols * esz, (row + 1) * ncols * esz);
        let (nan, inf, _, _) = stats(&out[lo..hi], dt);
        if nan == 0 && inf == 0 {
            continue;
        }
        for t in 0..n {
            let (s, c) = (starts[t].max(0) as usize, cnts[t].max(0) as usize);
            if row >= s && row < s + c {
                println!(
                    "  bad row {row} ← tile {t}: expert={} start={} cnt={}",
                    experts[t], starts[t], cnts[t]
                );
                shown += 1;
                break;
            }
        }
        if shown >= 8 {
            println!("  … (further bad rows elided)");
            break;
        }
    }

    if bad_expert.is_empty() && bad_span.is_empty() {
        println!(
            "tile tables are IN RANGE — so the addresses are right and the question moves to \
             what is AT them."
        );
        // Which experts the bad rows route to, deduplicated.
        let mut guilty: Vec<i32> = Vec::new();
        for row in 0..n_out_rows {
            let (lo, hi) = (row * ncols * esz, (row + 1) * ncols * esz);
            let (nan, inf, _, _) = stats(&out[lo..hi], dt);
            if nan == 0 && inf == 0 {
                continue;
            }
            for t in 0..n {
                let (s, c) = (starts[t].max(0) as usize, cnts[t].max(0) as usize);
                if row >= s && row < s + c {
                    if !guilty.contains(&experts[t]) {
                        guilty.push(experts[t]);
                    }
                    break;
                }
            }
        }
        println!("experts behind the bad rows: {guilty:?}");
        scan_all_experts(r, cu)?;
        // Decode those experts' ACTUAL bytes, as captured. The load-time audit
        // read every weight in the grid and found them finite; this reads the
        // same weight again from the moment it produced a NaN. If it decodes
        // non-finite now, the bytes in VRAM changed after load — the expert
        // cache streams slots pinned→VRAM, and a slot overwritten while a
        // kernel reads it looks exactly like this.
        let w_dtype = parse_dtype(&r.get::<String>("weight_dtype")?)?;
        let w_nrows: usize = r.get("weight_nrows")?;
        let w_cols: usize = r.get("cols")?;
        let shape = candle::Shape::from((w_nrows, w_cols));
        let n_w = w_nrows * w_cols;
        for e in &guilty {
            let name = format!("weight_{:05}", *e as usize);
            let Ok(bytes) = r.bytes(&name) else {
                println!("  {name}: not in the dump");
                continue;
            };
            let src = cu.memcpy_stod(&bytes).map_err(candle::Error::wrap)?;
            let dst = unsafe { cu.alloc::<f32>(n_w) }.map_err(candle::Error::wrap)?;
            let stream = cu.cuda_stream();
            let (sp, _sg) = src.device_ptr(&stream);
            let (dp, _dg) = dst.device_ptr(&stream);
            candle::tensor_assert::scratch::dequantize_into(sp, dp, &shape, w_dtype)?;
            stream.synchronize().map_err(candle::Error::wrap)?;
            let v = cu.memcpy_dtov(&dst).map_err(candle::Error::wrap)?;
            let nan = v.iter().filter(|x| x.is_nan()).count();
            let inf = v.iter().filter(|x| x.is_infinite()).count();
            let lo = v
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f32::INFINITY, f32::min);
            let hi = v
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            let zeros = v.iter().filter(|x| **x == 0.0).count();
            println!(
                "  expert {e}: nan={nan} inf={inf} zeros={zeros}/{n_w} finite=[{lo:e}, {hi:e}]"
            );
            if nan > 0 || inf > 0 {
                println!(
                    "  VERDICT: EXPERT {e}'s WEIGHT IS NON-FINITE IN VRAM at the moment it was \
                     used — yet every weight in the grid decoded finite at load. The bytes \
                     changed after load, which makes this the expert cache's streaming path, \
                     not the MoE arithmetic."
                );
            } else if zeros == n_w {
                println!(
                    "  VERDICT: EXPERT {e}'s SLOT IS ALL ZEROS — the weight was never written, \
                     or was written and then cleared. A zero KO scale dequantizes to zero here \
                     but the GEMM's own loader can divide by it."
                );
            }
        }
        return Ok(());
    } else {
        println!(
            "VERDICT: the tile tables ADDRESS OUT OF RANGE. The kernel dereferences \
             weight_ptrs[expert_base + tile_expert[t]] with no bounds check, so this reads \
             memory that is not an expert weight — which is exactly how a finite operand and \
             finite weights produce a non-finite result. The fault is in moe_bucketize, which \
             writes these tables."
        );
    }
    Ok(())
}

#[test]
fn the_captured_moe_gemm_replays_identically() -> Result<()> {
    let Ok(r) = Replay::open(DUMP_DIR) else {
        println!(
            "no capture at {DUMP_DIR} — run the daemon under `--features tensor-assert` until \
             it panics with a capture, then re-run this test"
        );
        return Ok(());
    };

    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cu) = &dev else {
        candle::bail!("replay needs a CUDA device")
    };

    // A general checkpoint fired before the gate GEMM did, which is the chain
    // working: the first panic is the first corruption, so an earlier site
    // winning means the GEMM was downstream of it. There is no matmul to replay
    // in that case — report what the checkpoint saw and stop.
    if r.has("checkpoint") {
        let name: String = r.get("checkpoint")?;
        let out = r.bytes("out")?;
        let dt = parse_out_dtype(&r.get::<String>("out_dtype")?)?;
        let (n, i, lo, hi) = stats(&out, dt);
        println!("FIRST CORRUPTION: checkpoint {name}");
        println!(
            "  shape={} dtype={dt:?}",
            r.get::<String>("out_shape").unwrap_or_default()
        );
        println!("  re-read from dump: nan={n} inf={i} finite=[{lo:e}, {hi:e}]");
        println!(
            "  capture saw:       nan={} inf={} (min={} max={})",
            r.get::<String>("saw.nan")?,
            r.get::<String>("saw.inf")?,
            r.get::<String>("saw.min")?,
            r.get::<String>("saw.max")?
        );
        let saw_nan: usize = r.get("saw.nan")?;
        let saw_inf: usize = r.get("saw.inf")?;
        if (saw_nan, saw_inf) != (n, i) {
            panic!(
                "THE BUFFER CHANGED UNDER THE CAPTURE: assert counted nan={saw_nan} \
                 inf={saw_inf}, re-reading the dumped bytes gives nan={n} inf={i}. Same \
                 allocation, moments apart, with a fence between — something is still \
                 writing to it."
            );
        }
        println!(
            "  every checkpoint upstream of this one passed on the same pass, so this is \
             where the value first went non-finite"
        );
        return Ok(());
    }

    let layer: usize = r.get("layer")?;
    let rows: usize = r.get("rows")?;
    let cols: usize = r.get("cols")?;
    let expert_base: usize = r.get("expert_base")?;
    let num_experts: usize = r.get("num_experts")?;
    let weight_nrows: usize = r.get("weight_nrows")?;
    let launch_tiles: usize = r.get("launch_tiles")?;
    let w_dtype = parse_dtype(&r.get::<String>("weight_dtype")?)?;
    let out_dtype = parse_out_dtype(&r.get::<String>("out_dtype")?)?;

    println!(
        "replaying layer {layer}: rows={rows} cols={cols} experts={num_experts} \
         base={expert_base} nrows={weight_nrows} tiles={launch_tiles} \
         w={w_dtype:?} out={out_dtype:?}"
    );

    // ── Rebuild the exact call ────────────────────────────────────────────
    // The weight table is rebuilt at the SAME indices production used, rather
    // than densified with `expert_base` folded away: the kernel indexes it at
    // `expert_base + tile_expert[t]`, and a remapped table would be a different
    // call than the one being reproduced.
    let stacked_bytes = r.bytes("stacked")?;
    let stacked_dev = cu
        .memcpy_stod(&stacked_bytes)
        .map_err(candle::Error::wrap)?;
    let stacked = Q8a128Operand::new(stacked_dev, rows, cols);

    let mut weights: Vec<CudaSlice<u8>> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let b = r.bytes(&format!("weight_{e:05}"))?;
        weights.push(cu.memcpy_stod(&b).map_err(candle::Error::wrap)?);
    }
    let stream = cu.cuda_stream();
    let mut table = vec![0u64; expert_base + num_experts];
    for (e, w) in weights.iter().enumerate() {
        let (p, _g) = w.device_ptr(&stream);
        table[expert_base + e] = p;
    }
    let table_dev = cu.memcpy_stod(&table).map_err(candle::Error::wrap)?;

    let tile_expert = cu
        .memcpy_stod(&r.typed::<i32>("tile_expert")?)
        .map_err(candle::Error::wrap)?;
    let tile_b_start = cu
        .memcpy_stod(&r.typed::<i32>("tile_b_start")?)
        .map_err(candle::Error::wrap)?;
    let tile_b_cnt = cu
        .memcpy_stod(&r.typed::<i32>("tile_b_cnt")?)
        .map_err(candle::Error::wrap)?;

    // ── Run it RUNS times ────────────────────────────────────────────────
    let mut outs: Vec<Vec<u8>> = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let out = grouped_qmatmul_dev_q8a128(
            &stacked,
            &table_dev,
            expert_base,
            num_experts,
            w_dtype,
            weight_nrows,
            &tile_expert,
            &tile_b_start,
            &tile_b_cnt,
            launch_tiles,
            cu,
        )?;
        let (storage, layout) = out.storage_and_layout();
        let candle::Storage::Cuda(s) = &*storage else {
            candle::bail!("replay: output is not CUDA")
        };
        let esz = out.dtype().size_in_bytes();
        let base = s.slice.device_ptr(&stream) + (layout.start_offset() * esz) as u64;
        let n = out.elem_count() * esz;
        // SAFETY: `base`/`n` are the output tensor's own extent, and `out` is
        // alive for this block. The view is leaked rather than dropped because
        // it borrows memory it does not own.
        let view: CudaSlice<u8> = unsafe { stream.upgrade_device_ptr::<u8>(base, n) };
        let host = cu.memcpy_dtov(&view).map_err(candle::Error::wrap)?;
        std::mem::forget(view);
        outs.push(host);
    }

    // ── Compare ──────────────────────────────────────────────────────────
    let production = r.bytes("out")?;
    let (n0, i0, lo0, hi0) = stats(&production, out_dtype);
    println!("production out: nan={n0} inf={i0} finite=[{lo0:e}, {hi0:e}]");

    // What the assert kernel SAW in production, carried in the manifest. This
    // is a second, independent reading of the same buffer taken at capture
    // time: if it disagrees with re-reading the dumped bytes here, the bytes
    // changed between the check and the copy — which is itself the answer, and
    // it would be invisible without recording both.
    let saw_nan: usize = r.get("saw.nan")?;
    let saw_inf: usize = r.get("saw.inf")?;
    println!(
        "capture saw:    nan={saw_nan} inf={saw_inf} (min={} max={})",
        r.get::<String>("saw.min")?,
        r.get::<String>("saw.max")?
    );
    if (saw_nan, saw_inf) != (n0, i0) {
        panic!(
            "THE BUFFER CHANGED UNDER THE CAPTURE: the assert kernel counted nan={saw_nan} \
             inf={saw_inf}, but re-reading the very bytes that were then copied to disk gives \
             nan={n0} inf={i0}. Both readings are of the same allocation, moments apart, with \
             a fence between them — so something is still writing to it. That is a lifetime or \
             ordering fault in the producer, and it is upstream of anything the replay below \
             could show."
        );
    }

    let self_consistent = outs.iter().all(|o| *o == outs[0]);
    let (n1, i1, lo1, hi1) = stats(&outs[0], out_dtype);
    println!("replay out:     nan={n1} inf={i1} finite=[{lo1:e}, {hi1:e}]");
    println!("replay self-consistent across {RUNS} runs: {self_consistent}");

    if !self_consistent {
        let differing = outs.iter().filter(|o| **o != outs[0]).count();
        panic!(
            "THE KERNEL IS NOT DETERMINISTIC: {differing} of {RUNS} runs over byte-identical \
             inputs disagreed. The fault is inside the kernel or its launch, not in the \
             operands — look for a race on the accumulator or an unsynchronised read."
        );
    }

    if outs[0].len() != production.len() {
        panic!(
            "replay produced {} bytes against production's {} — the rebuilt call does not \
             match the captured one, so nothing below it can be trusted",
            outs[0].len(),
            production.len()
        );
    }

    if outs[0] == production {
        println!(
            "VERDICT: deterministic AND bit-identical to production. The captured bytes really \
             do produce this output, so the kernel is behaving as written and the fault is \
             upstream — bisect the operand and the tile tables."
        );
        analyse_rows(&r, &production, out_dtype, weight_nrows, launch_tiles)?;
        analyse_operand(&r, cu, &production, out_dtype, weight_nrows, rows, cols)?;
        analyse_tiles(
            &r,
            cu,
            &production,
            out_dtype,
            weight_nrows,
            rows,
            num_experts,
            launch_tiles,
        )?;
    } else {
        let diff = outs[0]
            .iter()
            .zip(&production)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "VERDICT: deterministic but DIFFERENT from production ({diff} of {} bytes). The \
             kernel over these bytes does not reproduce what production got, so production was \
             not reading these bytes — a copy that had not landed, a recycled buffer, or a \
             lease outliving its generation.",
            production.len()
        );
    }
    Ok(())
}
