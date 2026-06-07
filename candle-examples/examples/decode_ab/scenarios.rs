//! Decode scenarios — the shape/layout axis of the A/B matrix.
//!
//! Each [`Scenario`] fully determines the synthetic decode problem: head
//! counts (MHA / GQA / MQA), head dim, how many context tokens were prefilled
//! (which sets the chunk/slice layout the decode kernel must scan), the batch
//! size (active decode slots), the RoPE layout, and the compute dtype.

use candle::DType;

/// A single decode shape/layout configuration.
#[derive(Clone, Debug)]
pub struct Scenario {
    pub name: &'static str,
    pub n_q_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Context tokens prefilled before the decode step. Drives chunk count and
    /// the partial-tail layout (`ctx_len % 32`).
    pub ctx_len: usize,
    /// Active decode slots (batch).
    pub num_slots: usize,
    /// RoPE layout: true = Llama (interleaved pairs), false = Qwen/GPT-2 halves.
    pub rope_interleaved: bool,
    /// Compute dtype of Q / k_new / v_new / output. The kernel typed-dispatch
    /// keys on this (F16 vs BF16/F8E4M3).
    pub compute: DType,
}

impl Scenario {
    /// Tokens visible to attention after the decode token is scattered.
    pub fn kv_len(&self) -> usize {
        self.ctx_len + 1
    }

    /// Head dims the kernels support (both V2 and fused Track-A): 64/96/128/256.
    pub fn head_dim_supported(&self) -> bool {
        matches!(self.head_dim, 64 | 96 | 128 | 256)
    }
}

/// The default scenario matrix: every scenario except those currently known to
/// crash a kernel and poison the CUDA context (see [`all_scenarios`]). This is
/// what runs when no `--scenarios` filter is given, so the default sweep
/// completes cleanly end to end.
pub fn default_scenarios() -> Vec<Scenario> {
    all_scenarios()
        .into_iter()
        .filter(|s| !CONTEXT_FATAL_KNOWN.contains(&s.name))
        .collect()
}

/// Scenarios currently known to illegal-address a kernel (poisoning the CUDA
/// context for the rest of the process) and therefore excluded from the
/// default sweep. `mha_hd256_ctx128`: V2 paged-decode illegal-addresses at
/// head_dim=256 even with a plain F16 arena — the kernel compiles an hd256
/// launcher but it faults (suspected shared-memory overrun). Still runnable via
/// `--scenarios mha_hd256_ctx128` for investigation.
pub const CONTEXT_FATAL_KNOWN: &[&str] = &["mha_hd256_ctx128"];

/// The full scenario universe, including ones flagged `context_fatal_known`.
/// `--scenarios <name>` resolves against this list so a known-bad scenario can
/// still be run in isolation for investigation.
pub fn all_scenarios() -> Vec<Scenario> {
    let f16 = DType::F16;
    vec![
        // ── Attention-sink / minimal context ───────────────────────────
        Scenario {
            name: "sink_only_ctx4",
            n_q_head: 8,
            n_kv_head: 8,
            head_dim: 128,
            ctx_len: 4, // only the 4 protected sink tokens
            num_slots: 1,
            rope_interleaved: true,
            compute: f16,
        },
        // ── MHA, single partial chunk ───────────────────────────────────
        Scenario {
            name: "mha_ctx31_partial",
            n_q_head: 16,
            n_kv_head: 16,
            head_dim: 64,
            ctx_len: 31,
            num_slots: 1,
            rope_interleaved: true,
            compute: f16,
        },
        // ── GQA 4:1, multi-chunk with partial tail ──────────────────────
        Scenario {
            name: "gqa4_ctx200",
            n_q_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            ctx_len: 200, // 6 full chunks + 8
            num_slots: 1,
            rope_interleaved: true,
            compute: f16,
        },
        // ── MQA, deep context ───────────────────────────────────────────
        Scenario {
            name: "mqa_ctx1024",
            n_q_head: 16,
            n_kv_head: 1,
            head_dim: 128,
            ctx_len: 1024,
            num_slots: 1,
            rope_interleaved: true,
            compute: f16,
        },
        // ── GQA 3:1 (Llama-3.2-3B-ish), batched slots ──────────────────
        Scenario {
            name: "gqa3_ctx512_b8",
            n_q_head: 24,
            n_kv_head: 8,
            head_dim: 128,
            ctx_len: 512,
            num_slots: 8,
            rope_interleaved: true,
            compute: f16,
        },
        // ── Non-interleaved RoPE (Qwen-style) ───────────────────────────
        Scenario {
            name: "gqa4_ctx256_ropehalf",
            n_q_head: 16,
            n_kv_head: 4,
            head_dim: 128,
            ctx_len: 256,
            num_slots: 1,
            rope_interleaved: false,
            compute: f16,
        },
        // ── head_dim = 256 ─────────────────────────────────────────────
        Scenario {
            name: "mha_hd256_ctx128",
            n_q_head: 8,
            n_kv_head: 8,
            head_dim: 256,
            ctx_len: 128,
            num_slots: 1,
            rope_interleaved: true,
            compute: f16,
        },
        // ── BF16 compute path ──────────────────────────────────────────
        Scenario {
            name: "gqa4_ctx256_bf16",
            n_q_head: 16,
            n_kv_head: 4,
            head_dim: 128,
            ctx_len: 256,
            num_slots: 1,
            rope_interleaved: true,
            compute: DType::BF16,
        },
        // ── Deep context, large batch (throughput-leaning) ─────────────
        Scenario {
            name: "gqa4_ctx2048_b16",
            n_q_head: 16,
            n_kv_head: 4,
            head_dim: 128,
            ctx_len: 2048,
            num_slots: 16,
            rope_interleaved: true,
            compute: f16,
        },
    ]
}

/// Perf-focused scenarios for `bench`: **batch = 8** (so 8 query rows can fill
/// the INT8 MMA's M=16 dimension), GQA 3:1, hd128 (Llama-3.2-3B-ish), swept over
/// context depth. This is the regime Track A's INT8 path is meant to win in —
/// `bench` uses these by default when no `--scenarios` is given.
pub fn perf_scenarios() -> Vec<Scenario> {
    let mk = |name: &'static str, ctx: usize| Scenario {
        name,
        n_q_head: 24,
        n_kv_head: 8,
        head_dim: 128,
        ctx_len: ctx,
        num_slots: 8,
        rope_interleaved: true,
        compute: DType::F16,
    };
    vec![
        mk("perf_b8_ctx128", 128),
        mk("perf_b8_ctx512", 512),
        mk("perf_b8_ctx1024", 1024),
        mk("perf_b8_ctx2048", 2048),
    ]
}

/// Filter scenarios by a comma-separated list of names (`--scenarios a,b`).
pub fn select_scenarios(filter: &str) -> Result<Vec<Scenario>, String> {
    let mut universe = all_scenarios();
    universe.extend(perf_scenarios());
    let mut out = Vec::new();
    let mut unknown = Vec::new();
    for want in filter.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        match universe.iter().find(|s| s.name == want) {
            Some(s) => out.push(s.clone()),
            None => unknown.push(want.to_string()),
        }
    }
    if !unknown.is_empty() {
        return Err(format!("unknown scenario name(s): {}", unknown.join(", ")));
    }
    Ok(out)
}
