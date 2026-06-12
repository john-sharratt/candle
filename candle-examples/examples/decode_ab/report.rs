//! Report rendering for the int8 paged-decode harness (terminal + markdown).

use crate::metrics::Metrics;

/// Outcome of one (scenario, format) golden cell — the int8 kernel vs FP32 truth.
pub enum GoldenOutcome {
    Ran { metrics: Metrics, passed: bool },
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
        "# decode golden check — int8 kernel vs FP32 ground truth (identity RoPE)\n\n\
         Pass gate: int8 cosine-vs-golden ≥ {cosine_tol:.4} (structural correctness; \
         quant precision shows up as MAE, not cosine). The 1-bit Q1_S format uses a \
         relaxed floor (0.80) — its sign-only V legitimately tops out near 0.87, where \
         a structural bug would still crater below ~0.6.\n\n",
    ));
    s.push_str("| scenario | format | status | int8 cos | int8 MAE | max-abs | worst head |\n");
    s.push_str("|---|---|---|---|---|---|---|\n");
    let (mut np, mut nf, mut ns) = (0usize, 0usize, 0usize);
    for r in rows {
        match &r.outcome {
            GoldenOutcome::Ran { metrics, passed } => {
                if *passed {
                    np += 1;
                } else {
                    nf += 1;
                }
                s.push_str(&format!(
                    "| {} | {} | {} | {:.5} | {:.3e} | {:.3e} | h{} ({:.3e}) |\n",
                    r.scenario,
                    r.format,
                    if *passed { "✅ pass" } else { "❌ FAIL" },
                    metrics.cosine,
                    metrics.mae,
                    metrics.max_abs,
                    metrics.worst_head,
                    metrics.worst_head_mae,
                ));
            }
            GoldenOutcome::Skipped(why) => {
                ns += 1;
                s.push_str(&format!(
                    "| {} | {} | ⊘ skip | — | — | — | {why} |\n",
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

pub struct BenchRow {
    pub scenario: String,
    pub format: String,
    /// Active decode slots = tokens produced per call (for tokens/s).
    pub num_slots: usize,
    /// Median per-call **GPU kernel** time in microseconds (CUDA-event timed).
    pub int8_us: f64,
}

impl BenchRow {
    /// Decode tokens/s for the per-call latency.
    fn toks_per_s(&self) -> f64 {
        if self.int8_us > 0.0 {
            self.num_slots as f64 * 1.0e6 / self.int8_us
        } else {
            f64::NAN
        }
    }
}

/// Render the bench table as markdown. Times are pure GPU kernel time (CUDA
/// events); tokens/s = num_slots / per-call-time.
pub fn render_bench(rows: &[BenchRow]) -> String {
    let mut s = String::new();
    s.push_str("# decode bench — int8 kernel (CUDA-event GPU kernel time)\n\n");
    s.push_str("| scenario | format | slots | int8 µs | int8 tok/s |\n");
    s.push_str("|---|---|---|---|---|\n");
    for r in rows {
        s.push_str(&format!(
            "| {} | {} | {} | {:.1} | {:.0} |\n",
            r.scenario,
            r.format,
            r.num_slots,
            r.int8_us,
            r.toks_per_s(),
        ));
    }
    s
}
