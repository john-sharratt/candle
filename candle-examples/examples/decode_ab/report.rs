//! Report rendering for the int8 paged-decode harness (terminal + markdown).

use crate::metrics::Metrics;

/// Outcome of one (scenario, format) golden cell: the int8 kernel against FP32
/// attention over the stored context (`stored`, the gate) and over the
/// unquantized truth (`truth`, the format's precision).
pub enum GoldenOutcome {
    Ran {
        stored: Metrics,
        truth: Metrics,
        passed: bool,
    },
    Skipped(String),
}

pub struct GoldenRow {
    pub scenario: String,
    pub format: String,
    pub outcome: GoldenOutcome,
}

/// Render the golden table. The pass gate is the cosine against FP32 attention
/// over the context AS STORED: the storage format's loss is in the reference,
/// so what remains is the kernel's own arithmetic, and a structural bug — a
/// wrong block, rank, scale, side or codebook — shows at any precision. The
/// cosine and MAE against the unquantized truth are the format's precision,
/// reported and not gated (the model gates hold precision to the models'
/// calibrated thresholds).
pub fn render_golden(stage: &str, rows: &[GoldenRow], stored_cosine_tol: f32) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# {stage} golden check — int8 kernel vs FP32 attention (identity RoPE)\n\n\
         Pass gate: cosine vs FP32 attention over the stored context ≥ {stored_cosine_tol} \
         (structural correctness at any storage precision). `truth cos` / `truth MAE` are \
         against the unquantized context — the storage format's precision, not gated.\n\n",
    ));
    s.push_str(
        "| scenario | format | status | stored cos | stored MAE | stored max | truth cos | truth MAE | worst head (truth) |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    let (mut np, mut nf, mut ns) = (0usize, 0usize, 0usize);
    for r in rows {
        match &r.outcome {
            GoldenOutcome::Ran {
                stored,
                truth,
                passed,
            } => {
                if *passed {
                    np += 1;
                } else {
                    nf += 1;
                }
                s.push_str(&format!(
                    "| {} | {} | {} | {:.6} | {:.3e} | {:.3e} | {:.5} | {:.3e} | h{} ({:.3e}) |\n",
                    r.scenario,
                    r.format,
                    if *passed { "✅ pass" } else { "❌ FAIL" },
                    stored.cosine,
                    stored.mae,
                    stored.max_abs,
                    truth.cosine,
                    truth.mae,
                    truth.worst_head,
                    truth.worst_head_mae,
                ));
            }
            GoldenOutcome::Skipped(why) => {
                ns += 1;
                s.push_str(&format!(
                    "| {} | {} | ⊘ skip | — | — | — | — | — | {why} |\n",
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

/// One (scenario, format) cell of the stage profile: every production path
/// that touches the stored arena, each as device time.
pub struct ProfileRow {
    pub scenario: String,
    pub format: String,
    /// Sealing the prefilled context into the arena format (format selection
    /// and palette conversion), summed over slots, in µs; zero for a directly
    /// typed arena.
    pub seal_us: f64,
    /// Median decode step, in µs.
    pub decode_us: f64,
    /// Tokens per slot in each timed prefill step.
    pub prefill_tokens: usize,
    /// Median batched prefill step over the stored context, in µs.
    pub prefill_us: f64,
}

/// Render the stage profile as markdown.
pub fn render_profile(rows: &[ProfileRow]) -> String {
    let mut s = String::new();
    s.push_str("# decode_ab profile — device time per stage (CUDA events)\n\n");
    s.push_str(
        "seal = format selection + palette conversion of the prefilled context \
         (all slots); decode = one step, median; prefill = one batched step of \
         `tok` new tokens per slot over the stored context, median.\n\n",
    );
    s.push_str("| scenario | format | seal µs | decode µs | prefill tok | prefill µs |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    for r in rows {
        s.push_str(&format!(
            "| {} | {} | {:.1} | {:.1} | {} | {:.1} |\n",
            r.scenario, r.format, r.seal_us, r.decode_us, r.prefill_tokens, r.prefill_us,
        ));
    }
    s
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
