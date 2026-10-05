//! Individual-kernel microbench + `ncu` target for the fused PLE launches
//! (`ple_gate`, `ple_conv` with its `ple_history`), with its own correctness
//! gate — §0.4 rule 4: a kernel that has not been measured is not finished.
//!
//! The arrangement is `hyper::bench`'s, for the reasons that module gives:
//!
//! - **Gate before timing, fused against eager, on the same device.** The eager
//!   chain ([`super::ple::ple_apply_spans`]) is the definition; on the device it
//!   runs the same `fast_exp` sigmoid the kernels call, so the only admissible
//!   difference is reassociation — the gate's dot product and its norms reduce
//!   in a different order, and `rg` is taken from `gate²·Σv²` rather than from
//!   the rounded gated values. [`GATE`] sits there and nowhere near a changed
//!   formula. The gate covers the shapes the conv's span table has to get right:
//!   decode rows, a prefill segment shorter than the history, one longer, and an
//!   odd width that forces the scalar path.
//! - **Size past the L2.** Both kernels are memory-shaped, so a working set
//!   inside the card's 96 MiB L2 reports the cache's bandwidth. The timed
//!   geometry is printed against it.
//!
//! Each kernel's GB/s is its own minimum traffic over its own time — for a
//! kernel this shape, the whole verdict: at the card's achievable bandwidth
//! there is nothing left to win.

use candle::wave_provenance::WaveTicket;
use candle::{DType, Device, Result, Tensor};

use super::config::PleConfig;
use super::model::PleSource;
use super::ple::{ple_apply_spans, PleSpan, PleState, PleWeights};
use super::ple_fused::{
    launch_conv, launch_gate, ple_apply_spans_fused, PleFusedWeights, SpanEntry, SpanTable,
};

/// The card's L2. A timed working set below this measures cache, not memory.
const L2_BYTES: usize = 96 * 1024 * 1024;

/// Admissible fused-vs-eager gap, relative to the reference's magnitude:
/// reassociation of 2,560-wide reductions and nothing more.
const GATE: f32 = 2e-5;

/// One benchmark point.
#[derive(Clone, Copy, Debug)]
pub struct PleBenchCfg {
    /// Rows in the wave — the axis that decides the working set.
    pub tokens: usize,
    /// Sequences the rows are split across.
    pub spans: usize,
    pub hc: usize,
    pub n_embd: usize,
    pub warmup: usize,
    pub iters: usize,
    pub seed: u64,
}

impl PleBenchCfg {
    /// Qwen3.8-Flash-Next's PLE geometry: 4 streams over `n_embd` 2560, the
    /// conv 4 taps dilated by the 3-gram order.
    pub fn qwen4exp(tokens: usize, spans: usize) -> Self {
        Self {
            tokens,
            spans,
            hc: 4,
            n_embd: 2560,
            warmup: 10,
            iters: 50,
            seed: 0x0005_EED0_F91E,
        }
    }

    fn hcd(&self) -> usize {
        self.hc * self.n_embd
    }

    /// Bytes `ple_gate` must move: the key|value row in, the residual in and
    /// out, the normed conv input out.
    pub fn gate_bytes(&self) -> usize {
        self.tokens * (4 * self.hcd() + self.n_embd) * std::mem::size_of::<f32>()
    }

    /// Bytes `ple_conv` must move: the normed input in once, the residual in
    /// and out.
    pub fn conv_bytes(&self) -> usize {
        3 * self.tokens * self.hcd() * std::mem::size_of::<f32>()
    }
}

/// The hash geometry at `n_embd` over `heads` heads (16 in production), each
/// over a small table so the gathers stay cheap. The embedding a row gathers
/// is `heads × head_dim` wide, which must be `n_embd`.
fn ple_cfg(n_embd: usize, heads: usize) -> PleConfig {
    PleConfig {
        layer: 1,
        ngram_size: 3,
        heads_per_ngram: heads / 2,
        conv_kernel: 4,
        eos_token_id: 7,
        multipliers: vec![
            0x9E37_79B9_7F4A_7C15,
            0xC2B2_AE3D_27D4_EB4F,
            0x1656_67B1_9E37_79F9,
        ],
        head_offsets: (0..heads as u64).map(|h| h * 64).collect(),
        head_vocab_sizes: vec![64; heads],
        head_dim: n_embd / heads,
    }
}

/// A table held on the device, gathered with an ordinary `index_select`.
struct DeviceTable(Tensor);

impl PleSource for DeviceTable {
    fn rows(&self, ids: &[u32], _root: Option<WaveTicket>) -> Result<Tensor> {
        let idx = Tensor::from_vec(ids.to_vec(), ids.len(), self.0.device())?;
        self.0.index_select(&idx, 0)
    }
}

fn lcg(shape: &[usize], seed: u64, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let vals: Vec<f32> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect();
    Tensor::from_vec(vals, shape, dev)
}

/// Largest elementwise gap, relative to the reference's own magnitude.
fn rel_gap(got: &Tensor, want: &Tensor) -> Result<f32> {
    let g = got.flatten_all()?.to_vec1::<f32>()?;
    let w = want.flatten_all()?.to_vec1::<f32>()?;
    if g.len() != w.len() {
        candle::bail!("ple bench: {} elements against {}", g.len(), w.len());
    }
    let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    Ok(g.iter()
        .zip(&w)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale)
}

fn weights(hc: usize, d: usize, kern: usize, seed: u64, dev: &Device) -> Result<PleWeights> {
    let hcd = hc * d;
    Ok(PleWeights {
        key: lcg(&[hcd, d], seed ^ 1, dev)?.affine(0.05, 0.)?,
        value: lcg(&[d, d], seed ^ 2, dev)?.affine(0.05, 0.)?,
        norm_key: lcg(&[hcd], seed ^ 3, dev)?.affine(0.2, 1.0)?,
        norm_query: lcg(&[hcd], seed ^ 4, dev)?.affine(0.2, 1.0)?,
        norm_conv: lcg(&[hcd], seed ^ 5, dev)?.affine(0.2, 1.0)?,
        conv: lcg(&[hcd, kern], seed ^ 6, dev)?.affine(0.5, 0.)?,
    })
}

/// One sequence: its tokens this wave and the state it enters with — a
/// nonzero history, so a tap that reads it is checked against real values.
fn seq_state(cfg: &PleConfig, hcd: usize, seed: u64, dev: &Device) -> Result<PleState> {
    Ok(PleState {
        conv_hist: lcg(&[cfg.conv_history(), hcd], seed, dev)?,
        spare_hist: None,
        prev: vec![(seed % 50) as u32 + 11, (seed % 37) as u32 + 13],
    })
}

/// Every sequence of a wave as a span over `states`, all capturing.
fn spans_of<'a>(states: &'a mut [PleState], tokens: &'a [Vec<u32>]) -> Vec<PleSpan<'a>> {
    let mut start = 0;
    states
        .iter_mut()
        .zip(tokens)
        .map(|(state, toks)| {
            let s = PleSpan {
                start,
                len: toks.len(),
                tokens: toks,
                state,
                capture: true,
            };
            start += toks.len();
            s
        })
        .collect()
}

/// Fused against eager for one wave of `lens` rows — see [`gate_waves`].
pub fn gate_once(
    dev: &Device,
    hc: usize,
    d: usize,
    heads: usize,
    lens: &[usize],
    seed: u64,
) -> Result<f32> {
    gate_waves(dev, hc, d, heads, lens, 1, seed)
}

/// Fused against eager for `waves` consecutive waves of `lens` rows over the
/// same sequences, every output of every wave compared: the residual, each
/// sequence's next history, and the captured rows.
///
/// From the second wave on it also checks the double buffer: a wave reads the
/// history the wave before it wrote, and leaves the buffer it read as the
/// sequence's spare — never the one it wrote.
pub fn gate_waves(
    dev: &Device,
    hc: usize,
    d: usize,
    heads: usize,
    lens: &[usize],
    waves: usize,
    seed: u64,
) -> Result<f32> {
    let cfg = ple_cfg(d, heads);
    if cfg.n_heads() * cfg.head_dim != d {
        candle::bail!("ple bench: {heads} heads cannot gather a {d}-wide embedding");
    }
    let hcd = hc * d;
    let w = weights(hc, d, cfg.conv_kernel, seed, dev)?;
    let fw = PleFusedWeights::from_weights(&w)?;
    let rows = cfg.head_offsets.last().copied().unwrap_or(0) as usize + 64;
    let table = DeviceTable(lcg(&[rows, cfg.head_dim], seed ^ 9, dev)?);
    let total: usize = lens.iter().sum();
    let tokens: Vec<Vec<u32>> = lens
        .iter()
        .enumerate()
        .map(|(i, &l)| (0..l).map(|t| ((t * 7 + i * 13) % 41) as u32 + 1).collect())
        .collect();
    let mut eager_states: Vec<PleState> = (0..lens.len())
        .map(|i| seq_state(&cfg, hcd, seed ^ (0x100 + i as u64), dev))
        .collect::<Result<_>>()?;
    let mut fused_states: Vec<PleState> = eager_states.iter().map(PleState::snapshot).collect();
    // What admission does before every forward.
    for s in fused_states.iter_mut() {
        s.ensure_spare()?;
    }

    let eps = 1e-6;
    let mut worst = 0f32;
    for wave in 0..waves {
        let res = lcg(&[total, hc, d], seed ^ (0x200 + wave as u64), dev)?;
        let entering: Vec<Tensor> = fused_states.iter().map(|s| s.conv_hist.clone()).collect();
        let (want, want_caps) = {
            let mut spans = spans_of(&mut eager_states, &tokens);
            ple_apply_spans(&res, &mut spans, &table, &w, &cfg, eps, None)?
        };
        let mut got = res.copy()?;
        let got_caps = {
            let mut spans = spans_of(&mut fused_states, &tokens);
            ple_apply_spans_fused(&mut got, &mut spans, &table, &fw, &cfg, eps, None)?
        };

        worst = worst.max(rel_gap(&got, &want)?);
        for ((f, e), read) in fused_states.iter().zip(&eager_states).zip(&entering) {
            worst = worst.max(rel_gap(&f.conv_hist, &e.conv_hist)?);
            if f.prev != e.prev {
                candle::bail!("ple bench: hash windows diverged");
            }
            let spare = f
                .spare_hist
                .as_ref()
                .ok_or_else(|| candle::Error::msg("ple bench: no spare after a wave"))?;
            if !spare.same_storage(read) || f.conv_hist.same_storage(read) {
                candle::bail!(
                    "ple bench: wave {wave} did not alternate — the history it read must become \
                     the spare and the one it wrote the history"
                );
            }
        }
        for (f, e) in got_caps.iter().zip(&want_caps) {
            match (f, e) {
                (Some(f), Some(e)) => worst = worst.max(rel_gap(f, e)?),
                _ => candle::bail!("ple bench: a span captured on one side only"),
            }
        }
    }
    Ok(worst)
}

/// Time `f` after `warmup` untimed calls, returning µs per call.
fn time_call(dev: &Device, cfg: &PleBenchCfg, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..cfg.warmup {
        f()?;
    }
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..cfg.iters {
        f()?;
    }
    dev.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / cfg.iters as f64)
}

/// Gate the kernels against the eager chain, then time each one.
pub fn run_ple_kernels(dev: &Device, cfg: PleBenchCfg) -> Result<()> {
    // Decode rows; a segment shorter than the 9-row history and one longer; an
    // odd width (two heads of 33) that cannot vectorise.
    let shapes: [(usize, usize, &[usize]); 3] = [
        (cfg.n_embd, 16, &[1, 1, 1, 1]),
        (cfg.n_embd, 16, &[3, 1, 20, 6]),
        (66, 2, &[5, 12]),
    ];
    let mut worst = 0f32;
    for (d, heads, lens) in shapes {
        let gap = gate_once(dev, cfg.hc, d, heads, lens, cfg.seed)?;
        if gap > GATE {
            candle::bail!(
                "ple bench: fused disagrees with eager by {gap} at d={d} spans {lens:?} — past \
                 reassociation, so the kernels compute something else and timing them \
                 measures nothing"
            );
        }
        worst = worst.max(gap);
    }
    println!("correctness gate: fused vs eager on device, worst rel gap {worst:.2e}");

    let (t, hc, d) = (cfg.tokens, cfg.hc, cfg.n_embd);
    let hcd = cfg.hcd();
    let pc = ple_cfg(d, 16);
    let w = PleFusedWeights::from_weights(&weights(hc, d, pc.conv_kernel, cfg.seed, dev)?)?;
    let kv = lcg(&[t, hcd + d], cfg.seed ^ 0x31, dev)?;
    let res = lcg(&[t, hc, d], cfg.seed ^ 0x32, dev)?;
    let normalized = lcg(&[t, hcd], cfg.seed ^ 0x33, dev)?;
    let per = t / cfg.spans.max(1);
    let hists: Vec<Tensor> = (0..cfg.spans)
        .map(|i| lcg(&[pc.conv_history(), hcd], cfg.seed ^ (0x40 + i as u64), dev))
        .collect::<Result<_>>()?;
    let news: Vec<Tensor> = (0..cfg.spans)
        .map(|_| Tensor::empty((pc.conv_history(), hcd), DType::F32, dev))
        .collect::<Result<_>>()?;
    let entries: Vec<SpanEntry<'_>> = (0..cfg.spans)
        .map(|i| SpanEntry {
            start: i * per,
            len: if i + 1 == cfg.spans { t - i * per } else { per },
            hist: &hists[i],
            new_hist: &news[i],
        })
        .collect();
    let table = SpanTable::build(&entries, dev, None)?;

    let ws = cfg.gate_bytes().max(cfg.conv_bytes());
    println!(
        "geometry {hc} x {d} = {hcd} wide | {t} rows over {} spans | largest pass {:.1} MiB ({})",
        cfg.spans,
        ws as f64 / (1024.0 * 1024.0),
        if ws > L2_BYTES {
            "PAST L2"
        } else {
            "INSIDE L2 — measuring cache"
        },
    );

    // The residual accumulates across iterations; the values stay finite over
    // any iteration count a benchmark runs.
    let gate_us = time_call(dev, &cfg, || launch_gate(&kv, &res, &w, &normalized, 1e-6))?;
    let conv_us = time_call(dev, &cfg, || {
        launch_conv(&normalized, &w.conv_t, &table, &res, &pc)
    })?;
    for (name, us, bytes) in [
        ("ple_gate", gate_us, cfg.gate_bytes()),
        ("ple_conv", conv_us, cfg.conv_bytes()),
    ] {
        println!(
            "{name:<9} {us:>9.1} us/call   {:>7.1} GB/s   ({:.1} MiB moved)",
            bytes as f64 / (us * 1e-6) / 1e9,
            bytes as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
}
