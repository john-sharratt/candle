//! Arena storage formats exercised by the A/B harness.
//!
//! An [`ArenaFmt`] names the on-arena storage of the sealed K/V chunks. The
//! "compute" dtype (the dtype of the freshly-projected Q / k_new / v_new and
//! of the kernel output) is carried separately on the scenario, since the
//! decode kernel reads the sealed arena via per-palette format tags while
//! computing in F16 or BF16.

use candle::DType;
use candle_nn::kv_cache::{KvFormat, QuantFormat};

/// One arena storage format the decode kernels are asked to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaFmt {
    /// Lossless / near-lossless float arena (F16, BF16, F8E4M3).
    Float(DType),
    /// Backing arena requested as `Quantized(qf)` — but note the sealed chunks
    /// actually land in the R16 capture format until a quantize pass runs, so
    /// this exercises the R16 read path, not real block quant.
    Quant(QuantFormat),
    /// REAL palette4 quantization (hd128 only): prefill → seal → run
    /// `quantize_sealed_in_place` so the decoded chunks are genuinely quantized.
    /// `override_fmt = Some(qf)` forces a uniform format + **unity** palette;
    /// `None` uses the adaptive selection, which produces **non-unity** palette
    /// maps (the configuration that exposed the production K-decode bug).
    RealQuant { level: u8, override_fmt: Option<QuantFormat> },
}

impl ArenaFmt {
    /// Short stable label used in report tables and `--formats` filters.
    pub fn label(&self) -> String {
        match self {
            ArenaFmt::Float(DType::F16) => "f16".to_string(),
            ArenaFmt::Float(DType::BF16) => "bf16".to_string(),
            ArenaFmt::Float(DType::F8E4M3) => "f8e4m3".to_string(),
            ArenaFmt::Float(_) => "float?".to_string(),
            ArenaFmt::Quant(q) => quant_label(*q).to_string(),
            ArenaFmt::RealQuant { level, override_fmt: Some(qf) } => {
                format!("rq-uni-{}-L{level}", quant_label(*qf))
            }
            ArenaFmt::RealQuant { level, override_fmt: None } => format!("rq-adaptive-L{level}"),
        }
    }

    /// The `KvFormat` to construct the backing arena with. RealQuant builds a
    /// float source arena and quantizes after sealing.
    pub fn kv_format(&self) -> KvFormat {
        match self {
            ArenaFmt::Float(dt) => KvFormat::Float(*dt),
            ArenaFmt::Quant(q) => KvFormat::Quantized(*q),
            ArenaFmt::RealQuant { .. } => KvFormat::Float(DType::F16),
        }
    }

    /// Structural-correctness cosine floor for the golden gate. Most formats use
    /// the caller's default. The extreme **1-bit symmetric** format Q1_S encodes
    /// V as sign × one per-block magnitude, which legitimately reconstructs at
    /// ~0.87 cosine vs FP32 (V2 hits the same floor) — well above the ~0.6 a
    /// structural read/decode bug would crater to. So it gets a relaxed floor
    /// that still fails on a real bug. (Q1_A keeps two scales and clears 0.93.)
    pub fn golden_cosine_floor(&self, default: f32) -> f32 {
        let qf = match self {
            ArenaFmt::Quant(q) => Some(*q),
            ArenaFmt::RealQuant { override_fmt: Some(q), .. } => Some(*q),
            _ => None,
        };
        match qf {
            Some(QuantFormat::Q1_S) => default.min(0.80),
            _ => default,
        }
    }
}

fn quant_label(q: QuantFormat) -> &'static str {
    use QuantFormat::*;
    match q {
        Q4_0 => "q4_0",
        Q4_1 => "q4_1",
        Q5_0 => "q5_0",
        Q5_1 => "q5_1",
        Q8_0 => "q8_0",
        Q8_1 => "q8_1",
        Q4_KS => "q4_ks",
        Q8_KS => "q8_ks",
        Q2_0 => "q2_0",
        Q3_0 => "q3_0",
        R16 => "r16",
        Q0 => "q0",
        Q1_S => "q1_s",
        Q2_S => "q2_s",
        Q2_A => "q2_a",
        Q2_1 => "q2_1",
        Q3_1 => "q3_1",
        Q0_V => "q0_v",
        Q1_A => "q1_a",
        Q0_X => "q0_x",
        Q0_M2 => "q0_m2",
        Q0_M4 => "q0_m4",
    }
}

/// The default arena-format set: float baselines plus the well-established
/// block-quant formats spanning the compression tiers. This is what runs when
/// no `--formats` filter is given.
pub fn default_formats() -> Vec<ArenaFmt> {
    use QuantFormat::*;
    vec![
        ArenaFmt::Float(DType::F16),
        ArenaFmt::Float(DType::BF16),
        ArenaFmt::Float(DType::F8E4M3),
        ArenaFmt::Quant(Q8_0),
        ArenaFmt::Quant(Q8_KS),
        ArenaFmt::Quant(Q5_0),
        ArenaFmt::Quant(Q4_0),
        ArenaFmt::Quant(Q4_1),
        ArenaFmt::Quant(Q4_KS),
        ArenaFmt::Quant(Q3_0),
        ArenaFmt::Quant(Q2_0),
        ArenaFmt::Quant(Q2_S),
    ]
}

/// The quant-coverage format axis for the `suite` command's **codec sweep** (run
/// against the cheap shallow/mid [`suite_scenarios`](crate::scenarios::suite_scenarios)):
/// the F16 baseline plus every compression codec the decode kernel reads through
/// its real palette quantization. The native-INT8 arenas (Q8_0/Q4_0/Q2_0 +
/// adaptive) run at both ends of the compression ladder — L0 (near-lossless
/// reference) and L7 (aggressive) — to bracket the quant-error range; the
/// remaining read-through families run at L0. A subset of [`all_formats`], so
/// every label here resolves via `select_formats`.
pub fn quant_formats() -> Vec<ArenaFmt> {
    use QuantFormat::*;
    let mut v = vec![ArenaFmt::Float(DType::F16)];
    for level in [0u8, 7] {
        for of in [Q8_0, Q4_0, Q2_0] {
            v.push(ArenaFmt::RealQuant { level, override_fmt: Some(of) });
        }
        v.push(ArenaFmt::RealQuant { level, override_fmt: None });
    }
    for qf in [Q5_0, Q3_0, Q4_KS, Q8_KS, Q2_S, Q1_S, Q1_A, Q0, Q0_M2, Q0_M4, Q0_X] {
        v.push(ArenaFmt::RealQuant { level: 0, override_fmt: Some(qf) });
    }
    v
}

/// Production-format axis for the `suite` command's **depth/scale sweep** (run
/// against the expensive deep [`suite_deep_scenarios`](crate::scenarios::suite_deep_scenarios)):
/// the F16 correctness baseline plus the native-INT8 arenas Q8_0 and Q4_0 at both
/// ends of the compression ladder (L0 reference, L7 aggressive). The deep/large
/// fixtures are slow to build and the codec under read doesn't change the
/// deep-scan or split-KV path, so the aggressive codecs stay out of the deep
/// sweep — they're already covered by [`quant_formats`] at the cheap shapes.
pub fn deep_formats() -> Vec<ArenaFmt> {
    use QuantFormat::*;
    vec![
        ArenaFmt::Float(DType::F16),
        ArenaFmt::RealQuant { level: 0, override_fmt: Some(Q8_0) },
        ArenaFmt::RealQuant { level: 7, override_fmt: Some(Q8_0) },
        ArenaFmt::RealQuant { level: 0, override_fmt: Some(Q4_0) },
        ArenaFmt::RealQuant { level: 7, override_fmt: Some(Q4_0) },
    ]
}

/// Every arena format the kernels claim to support — the exhaustive sweep used
/// under `--all-formats`. The exotic capture/experimental formats are included;
/// any that cannot be used as a uniform sealing format are reported as skipped
/// per (scenario, format) rather than aborting the run.
pub fn all_formats() -> Vec<ArenaFmt> {
    use QuantFormat::*;
    let mut v = vec![
        ArenaFmt::Float(DType::F16),
        ArenaFmt::Float(DType::BF16),
        ArenaFmt::Float(DType::F8E4M3),
    ];
    for q in [
        Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q4_KS, Q8_KS, Q2_0, Q3_0, R16, Q0, Q1_S, Q2_S, Q2_A,
        Q2_1, Q3_1, Q0_V, Q1_A, Q0_X, Q0_M2, Q0_M4,
    ] {
        v.push(ArenaFmt::Quant(q));
    }
    // Real palette4 quantization (hd128 only). Unity (override → uniform int8
    // arena, the skip-dequant target) vs adaptive (non-unity palette). Q8_0/
    // Q4_0/Q2_0 cover 8/4/2-bit native-INT8 arenas for the memory-bound bench.
    for level in [0u8, 1, 3, 5, 7] {
        v.push(ArenaFmt::RealQuant { level, override_fmt: Some(QuantFormat::Q8_0) });
        v.push(ArenaFmt::RealQuant { level, override_fmt: Some(QuantFormat::Q4_0) });
        v.push(ArenaFmt::RealQuant { level, override_fmt: Some(QuantFormat::Q2_0) });
        v.push(ArenaFmt::RealQuant { level, override_fmt: None });
    }
    // Remaining read-through passthrough families (level 0; the override forces
    // the format). Exercises every BlockInt8 typed worker, not just Q8_0/Q4_0/Q2_0.
    for qf in [
        QuantFormat::Q5_0,
        QuantFormat::Q3_0,
        QuantFormat::Q4_KS,
        QuantFormat::Q8_KS,
        QuantFormat::Q2_S,
        QuantFormat::Q1_S,
        QuantFormat::Q1_A,
        QuantFormat::Q0,
        QuantFormat::Q0_M2,
        QuantFormat::Q0_M4,
        QuantFormat::Q0_X,
    ] {
        v.push(ArenaFmt::RealQuant { level: 0, override_fmt: Some(qf) });
    }
    v
}

/// Resolve a `--formats a,b,c` filter (by label) against the full universe.
/// Unknown labels are returned in the `Err` for the caller to surface.
pub fn select_formats(filter: &str) -> Result<Vec<ArenaFmt>, String> {
    let universe = all_formats();
    let mut out = Vec::new();
    let mut unknown = Vec::new();
    for want in filter.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        match universe.iter().find(|f| f.label() == want) {
            Some(f) => out.push(*f),
            None => unknown.push(want.to_string()),
        }
    }
    if !unknown.is_empty() {
        return Err(format!("unknown format label(s): {}", unknown.join(", ")));
    }
    Ok(out)
}
