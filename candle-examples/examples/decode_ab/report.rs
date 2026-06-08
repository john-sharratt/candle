//! Report rendering for the A/B harness (terminal + markdown).

use crate::metrics::Metrics;

/// Outcome of one (scenario, format) parity cell.
pub enum CompareOutcome {
    /// Both kernels ran; parity computed.
    Ran { metrics: Metrics, passed: bool },
    /// The fixture could not be built for this pairing (format/shape combo).
    Skipped(String),
}

/// Outcome of one (scenario, format) golden cell — both kernels vs FP32 truth.
pub enum GoldenOutcome {
    Ran {
        v2: Metrics,
        fused: Metrics,
        ab_mae: f32,
        passed: bool,
    },
    Skipped(String),
}

pub struct GoldenRow {
    pub scenario: String,
    pub format: String,
    pub outcome: GoldenOutcome,
}

/// Render the golden (ground-truth) table. The pass gate is on **cosine**: it
/// is robust to quantization precision (even L7 Q2/Q1 stays ≥ ~0.96) but craters
/// on a structural bug (wrong K-read / palette routing → ≲ 0.6), so it flags
/// correctness regressions without false-failing on low-bit formats.
pub fn render_golden(rows: &[GoldenRow], cosine_tol: f32) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# decode golden check — both kernels vs FP32 ground truth (identity RoPE)\n\n\
         Pass gate: V2 **and** fused cosine-vs-golden ≥ {cosine_tol:.4} (structural \
         correctness; quant precision shows up as MAE, not cosine). The 1-bit Q1_S \
         format uses a relaxed floor (0.80) — its sign-only V legitimately tops out \
         near 0.87, where a structural bug would still crater below ~0.6.\n\n",
    ));
    s.push_str(
        "| scenario | format | status | v2 cos | v2 MAE | fused cos | fused MAE | A/B MAE |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    let (mut np, mut nf, mut ns) = (0usize, 0usize, 0usize);
    for r in rows {
        match &r.outcome {
            GoldenOutcome::Ran {
                v2,
                fused,
                ab_mae,
                passed,
            } => {
                if *passed {
                    np += 1;
                } else {
                    nf += 1;
                }
                s.push_str(&format!(
                    "| {} | {} | {} | {:.5} | {:.3e} | {:.5} | {:.3e} | {:.3e} |\n",
                    r.scenario,
                    r.format,
                    if *passed { "✅ pass" } else { "❌ FAIL" },
                    v2.cosine,
                    v2.mae,
                    fused.cosine,
                    fused.mae,
                    ab_mae,
                ));
            }
            GoldenOutcome::Skipped(why) => {
                ns += 1;
                s.push_str(&format!(
                    "| {} | {} | ⊘ skip | — | — | — | — | {why} |\n",
                    r.scenario, r.format
                ));
            }
        }
    }
    s.push_str(&format!(
        "\n**{np} pass, {nf} fail, {ns} skipped** of {} cells.\n",
        rows.len()
    ));
    s
}

pub struct CompareRow {
    pub scenario: String,
    pub format: String,
    pub outcome: CompareOutcome,
}

pub struct BenchRow {
    pub scenario: String,
    pub format: String,
    /// Active decode slots = tokens produced per call (for tokens/s).
    pub num_slots: usize,
    /// Median per-call **GPU kernel** time in microseconds (CUDA-event timed).
    pub v2_us: f64,
    pub fused_us: f64,
}

impl BenchRow {
    /// Speedup of fused over v2 (>1 = fused faster).
    pub fn speedup(&self) -> f64 {
        if self.fused_us > 0.0 {
            self.v2_us / self.fused_us
        } else {
            f64::NAN
        }
    }
    /// Decode tokens/s for a per-call latency in µs.
    fn toks_per_s(&self, us: f64) -> f64 {
        if us > 0.0 {
            self.num_slots as f64 * 1.0e6 / us
        } else {
            f64::NAN
        }
    }
}

/// Render the compare table as GitHub-flavored markdown.
pub fn render_compare(rows: &[CompareRow], tol: (f32, f32, f32)) -> String {
    let (mae_tol, max_tol, cos_tol) = tol;
    let mut s = String::new();
    s.push_str(&format!(
        "# decode A/B — parity (V2 reference vs fused-attn-v1)\n\n\
         Pass criterion: MAE ≤ {mae_tol:.1e}, max-abs ≤ {max_tol:.1e}, cosine ≥ {cos_tol:.5}\n\n",
    ));
    s.push_str("| scenario | format | status | MAE | max-abs | cosine | worst head |\n");
    s.push_str("|---|---|---|---|---|---|---|\n");
    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_skip = 0usize;
    for r in rows {
        match &r.outcome {
            CompareOutcome::Ran { metrics, passed } => {
                if *passed {
                    n_pass += 1;
                } else {
                    n_fail += 1;
                }
                let status = if *passed { "✅ pass" } else { "❌ FAIL" };
                s.push_str(&format!(
                    "| {} | {} | {} | {:.3e} | {:.3e} | {:.5} | h{} ({:.3e}) |\n",
                    r.scenario,
                    r.format,
                    status,
                    metrics.mae,
                    metrics.max_abs,
                    metrics.cosine,
                    metrics.worst_head,
                    metrics.worst_head_mae,
                ));
            }
            CompareOutcome::Skipped(why) => {
                n_skip += 1;
                s.push_str(&format!(
                    "| {} | {} | ⊘ skip | — | — | — | {why} |\n",
                    r.scenario, r.format,
                ));
            }
        }
    }
    s.push_str(&format!(
        "\n**{n_pass} pass, {n_fail} fail, {n_skip} skipped** of {} cells.\n",
        rows.len()
    ));
    s
}

/// Render the bench table as markdown. Times are pure GPU kernel time (CUDA
/// events); tokens/s = num_slots / per-call-time. Target for Track A: ≥ 2×.
pub fn render_bench(rows: &[BenchRow]) -> String {
    let mut s = String::new();
    s.push_str("# decode bench — V2 vs fused-attn-v1 (CUDA-event GPU kernel time)\n\n");
    s.push_str("| scenario | format | slots | V2 µs | fused µs | V2 tok/s | fused tok/s | speedup |\n");
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    for r in rows {
        let sp = r.speedup();
        let marker = if sp >= 2.0 {
            "🟢"
        } else if sp >= 1.0 {
            "🟡"
        } else {
            "🔴"
        };
        s.push_str(&format!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.0} | {:.0} | {marker} {:.2}× |\n",
            r.scenario,
            r.format,
            r.num_slots,
            r.v2_us,
            r.fused_us,
            r.toks_per_s(r.v2_us),
            r.toks_per_s(r.fused_us),
            sp,
        ));
    }
    if !rows.is_empty() {
        let gm: f64 = rows.iter().map(|r| r.speedup().ln()).sum::<f64>() / rows.len() as f64;
        s.push_str(&format!(
            "\n**Geomean speedup: {:.2}×** (🟢 ≥2× target, 🟡 1–2×, 🔴 slower)\n",
            gm.exp()
        ));
    }
    s
}
