//! Names the inference modes a session can run in, each mapping to a compute
//! dtype and a storage format for K and V.
//!
//! Float modes use one dtype for compute and storage. Quantized modes compute in
//! F32 over quantized KV. Adaptive levels 0 to 10 carry a candidate list per
//! side. The selection kernel tries the most compressed candidate first and keeps
//! the first whose error stays within threshold, and a block that passes none
//! stays F16. The paged kernels store F16 or BF16 only, so an F32 mode stores
//! BF16 to match the compute collapse.

use super::*;

/// Inference mode specifying both compute dtype and KV cache storage format.
///
/// This enum provides a high-level way to configure inference:
/// - Float modes (F32, F16, BF16): Use the specified dtype for both compute and KV storage
/// - Quantized modes: Use F32 compute with quantized KV storage for memory savings
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    /// Full precision F32 compute and KV storage
    F32,
    /// Half precision F16 compute and KV storage
    F16,
    /// BFloat16 compute and KV storage
    BF16,
    /// F32 compute with Q8_0 quantized KV cache (8-bit int)
    Q8_0,
    /// F32 compute with Q8_1 quantized KV cache (8-bit int)
    Q8_1,
    /// F32 compute with Q8_KS quantized KV cache (8-bit int with sub-block scales)
    Q8_KS,
    /// F32 compute with Q4_0 quantized KV cache (4-bit int)
    Q4_0,
    /// F32 compute with Q4_1 quantized KV cache (4-bit int)
    Q4_1,
    /// F32 compute with Q4_KS quantized KV cache (4-bit with sub-block scales)
    Q4_KS,
    /// F32 compute with Q3_0 quantized KV cache (3-bit int)
    Q3_0,
    /// F32 compute with Q2_0 quantized KV cache (2-bit int)
    Q2_0,
    /// F16 compute with R16 KV cache (raw F16 with Q-capture space)
    R16,
    /// Composite types
    Q8_Q4,
    Q8_Q8KS,
    Q8_Q4KS,
    Q8_Q2_A,
    Q8_Q2_S,
    Q8_Q1_S,
    /// Adaptive compression level 0: near-lossless. K={Q8_KS,Q8_0,Q2_0}, V={Q8_0,Q4_0,Q2_0}.
    /// Most K blocks stay F16 fallback; V predominantly Q8_0.
    C0,
    /// Adaptive compression level 1. K={Q8_KS,Q8_0,Q4_1,Q2_0}, V={Q8_0,Q4_0,Q2_0}.
    /// K: Q8_0≈81%, small Q8_KS fraction; V: Q8_0≈93%.
    C1,
    /// Adaptive compression level 2. K={Q8_KS,Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_1,Q4_0,Q2_0}.
    /// K: Q8_0≈93%, Q4 formats emerging; V: Q4_1/Q4_0 mix.
    C2,
    /// Adaptive compression level 3. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_1,Q4_0,Q2_0}.
    /// K: Q8_0≈76%, Q4 mix≈21%; V: Q4_0 dominant.
    C3,
    /// Adaptive compression level 4. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_0,Q3_0,Q2_0}.
    /// K: Q4_0≈25%, Q8_0≈24%; V: Q4_0≈81%, Q3_0 entering.
    C4,
    /// Adaptive compression level 5. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_0,Q3_0,Q2_0}.
    /// K: Q4_0≈28%, Q3_0≈18%; V: Q3_0≈70%.
    C5,
    /// Adaptive compression level 6. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q3_0≈43%, Q4_0≈20%; V: Q3_0≈70%, Q2_0≈25%.
    C6,
    /// Adaptive compression level 7. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q3_0≈49%, Q2_0≈25%; V: Q2_0≈59%, Q3_0≈37%.
    C7,
    /// Adaptive compression level 8. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q2_0≈48%, Q3_0≈42%; V: Q2_0≈94%.
    C8,
    /// Adaptive compression level 9: maximum compression. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q2_0≈64%, Q3_0≈30%; V: Q2_0≈95%.
    C9,
    /// Adaptive compression level 10: K same as C9, V pushed further.
    /// K: same as C9; V: Q2_0 dominant.
    C10,
}

impl InferenceMode {
    /// Get the compute dtype for activations.
    pub fn compute_dtype(&self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            // All quantized modes use F32 for compute to maintain precision
            Self::Q8_0
            | Self::Q4_0
            | Self::Q8_1
            | Self::Q4_1
            | Self::Q4_KS
            | Self::Q8_KS
            | Self::Q2_0
            | Self::Q3_0
            | Self::Q8_Q8KS
            | Self::Q8_Q4KS
            | Self::Q8_Q4
            | Self::Q8_Q2_A
            | Self::Q8_Q2_S
            | Self::Q8_Q1_S
            | Self::C0
            | Self::C1
            | Self::C2
            | Self::C3
            | Self::C4
            | Self::C5
            | Self::C6
            | Self::C7
            | Self::C8
            | Self::C9
            | Self::C10 => DType::F32,
            // R16 uses F16 compute (it stores F16 natively)
            Self::R16 => DType::F16,
        }
    }

    /// Get the KV cache storage format for K.
    ///
    /// Returns the first *quantized* candidate if one exists, otherwise the
    /// first entry.  For adaptive modes with float fallback (C0-C2) this
    /// returns the highest-fidelity quant (e.g. Q8_0) so that `storage_policy`
    /// produces `GpuQuant` and reconcile can route chunks through the adaptive
    /// selection kernel.  The float entry is a per-block fallback within the
    /// kernel, not the storage target.
    pub fn k_format(&self) -> KvFormat {
        let cands = self.k_candidates();
        cands
            .iter()
            .find(|f| f.is_quantized())
            .copied()
            .unwrap_or(cands[0])
    }

    /// Get the KV cache storage format for V.
    ///
    /// Same logic as [`k_format`]: returns first quantized candidate when
    /// available, falling back to the first entry for pure-float modes.
    pub fn v_format(&self) -> KvFormat {
        let cands = self.v_candidates();
        cands
            .iter()
            .find(|f| f.is_quantized())
            .copied()
            .unwrap_or(cands[0])
    }

    /// K cache candidate formats for per-block adaptive selection.
    ///
    /// K cache candidate formats for per-block adaptive selection.
    ///
    /// Ordered highest-to-lowest fidelity.  The selection kernel evaluates
    /// candidates in ascending BPE order (best compression first) and selects
    /// the most aggressive format whose magnitude-weighted error stays within
    /// threshold.  If no quantized candidate passes, the block stays F16.
    ///
    /// A single-element list means uniform application (no selection kernel).
    /// Multiple elements trigger the adaptive selection kernel.
    pub fn k_candidates(&self) -> Vec<KvFormat> {
        use KvFormat::{Float, Quantized as Q};
        use QuantFormat::*;

        if let Some(level) = self.compression_level() {
            return CompressionPolicy::production_candidates(level).0;
        }

        match self {
            // -- Composite models - K and V formats differ but are statically defined, no selection kernel --
            Self::Q8_Q4
            | Self::Q8_Q8KS
            | Self::Q8_Q4KS
            | Self::Q8_Q2_S
            | Self::Q8_Q2_A
            | Self::Q8_Q1_S => vec![Q(Q8_1)],
            // ── Uniform modes — single candidate ──
            Self::Q8_0 => vec![Q(Q8_0)],
            Self::Q8_1 => vec![Q(Q8_1)],
            Self::Q8_KS => vec![Q(Q8_KS)],
            Self::Q4_0 => vec![Q(Q4_0)],
            Self::Q4_1 => vec![Q(Q4_1)],
            Self::Q4_KS => vec![Q(Q4_KS)],
            Self::Q3_0 => vec![Q(Q3_0)],
            Self::Q2_0 => vec![Q(Q2_0)],
            Self::R16 => vec![Q(QuantFormat::R16)],
            // ── Float modes ──
            // Paged attention kernels store/read F16/BF16 only, so F32 KV is unrepresentable. The
            // F32 reference mode stores BF16 — matching the F32→BF16 compute collapse already in the
            // prefill/decode paths. Model compute follows the arena dtype (caches.dtype()).
            Self::F32 => vec![Float(DType::BF16)],
            Self::F16 => vec![Float(DType::F16)],
            Self::BF16 => vec![Float(DType::BF16)],
            _ => vec![Float(DType::F16)],
        }
    }

    /// V cache candidate formats for per-block adaptive selection.
    ///
    /// Mirror of [`k_candidates`] for the V cache.  Uses cosine distance
    /// as the error metric.  Always 4 candidates per adaptive level.
    pub fn v_candidates(&self) -> Vec<KvFormat> {
        use KvFormat::{Float, Quantized as Q};
        use QuantFormat::*;

        if let Some(level) = self.compression_level() {
            return CompressionPolicy::production_candidates(level).1;
        }

        match self {
            // -- Composite models - K and V formats differ but are statically defined, no selection kernel --
            Self::Q8_Q8KS => vec![Q(Q8_KS)],
            Self::Q8_Q4KS => vec![Q(Q4_KS)],
            Self::Q8_Q2_S => vec![Q(Q2_S)],
            Self::Q8_Q2_A => vec![Q(Q2_A)],
            Self::Q8_Q1_S => vec![Q(Q1_S)],
            Self::Q8_Q4 => vec![Q(Q4_1)],
            // ── Uniform modes — single candidate ──
            Self::Q8_0 => vec![Q(Q8_0)],
            Self::Q8_1 => vec![Q(Q8_1)],
            Self::Q8_KS => vec![Q(Q8_KS)],
            Self::Q4_0 => vec![Q(Q4_0)],
            Self::Q4_1 => vec![Q(Q4_1)],
            Self::Q4_KS => vec![Q(Q4_KS)],
            Self::Q3_0 => vec![Q(Q3_0)],
            Self::Q2_0 => vec![Q(Q2_0)],
            // R16 is K-only (K@F16 + Q-capture space). V uses plain F16.
            Self::R16 => vec![Float(DType::F16)],
            // ── Float modes ──
            // F32 KV is unrepresentable in the paged kernels (F16/BF16 only); store BF16 to match
            // the F32→BF16 compute collapse.
            Self::F32 => vec![Float(DType::BF16)],
            Self::F16 => vec![Float(DType::F16)],
            Self::BF16 => vec![Float(DType::BF16)],
            _ => vec![Float(DType::F16)],
        }
    }

    /// Get the compression level (0-10) for adaptive modes, or `None` for
    /// uniform/non-compression modes. Presence of a level signals that the
    /// adaptive selection kernel should run.
    pub fn compression_level(&self) -> Option<u8> {
        match self {
            Self::C0 => Some(0),
            Self::C1 => Some(1),
            Self::C2 => Some(2),
            Self::C3 => Some(3),
            Self::C4 => Some(4),
            Self::C5 => Some(5),
            Self::C6 => Some(6),
            Self::C7 => Some(7),
            Self::C8 => Some(8),
            Self::C9 => Some(9),
            Self::C10 => Some(10),
            _ => None,
        }
    }

    /// Check if this mode uses quantized KV cache.
    ///
    /// True when at least one candidate is a quantized format.
    pub fn is_quantized(&self) -> bool {
        self.k_candidates().iter().any(|f| f.is_quantized())
            || self.v_candidates().iter().any(|f| f.is_quantized())
    }
}
