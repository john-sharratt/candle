use super::ops as cpu;
use super::params::SELECT_BLOCK;
use crate::kv_cache::{KvFormat, QuantFormat};

#[cfg(feature = "cuda")]
use candle::quantized::{cuda::ggml_to_select_qtype, GgmlDType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleSide {
    Key,
    Value,
}

impl SampleSide {
    pub fn label(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Value => "value",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum SampleFormat {
    F16,
    BF16,
    Q8KS,
    Q8_1,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4KS,
    Q4_1,
    Q4_0,
    Q3_1,
    Q3_0,
    Q2_1,
    Q2A,
    Q2S,
    Q2_0,
    Q1S,
    Q0,
    Q0_V,
    Q1_A,
    Q0_X,
    Q0_M2,
    Q0_M4,
}

impl SampleFormat {
    pub const Q8_KS: Self = Self::Q8KS;
    pub const Q4_KS: Self = Self::Q4KS;
    pub const Q2_A: Self = Self::Q2A;
    pub const Q2_S: Self = Self::Q2S;
    pub const Q1_S: Self = Self::Q1S;
    pub const Q0_V: Self = Self::Q0_V;
    pub const Q1_A: Self = Self::Q1_A;
    pub const Q0_X: Self = Self::Q0_X;
    pub const Q0_M2: Self = Self::Q0_M2;
    pub const Q0_M4: Self = Self::Q0_M4;

    /// All variants in declaration order. Keep in sync with the enum — adding a
    /// variant without updating ALL causes `sample_format_coverage` to fail.
    pub const ALL: &'static [Self] = &[
        Self::F16, Self::BF16,
        Self::Q8KS, Self::Q8_1, Self::Q8_0,
        Self::Q5_1, Self::Q5_0,
        Self::Q4KS, Self::Q4_1, Self::Q4_0,
        Self::Q3_1, Self::Q3_0,
        Self::Q2_1, Self::Q2A, Self::Q2S, Self::Q2_0,
        Self::Q1S,
        Self::Q0, Self::Q0_V, Self::Q1_A, Self::Q0_X, Self::Q0_M2, Self::Q0_M4,
    ];

    pub fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16)
    }

    pub fn from_kv_format(format: KvFormat) -> Option<Self> {
        match format {
            KvFormat::Float(candle::DType::F16) => Some(Self::F16),
            KvFormat::Float(candle::DType::BF16) => Some(Self::BF16),
            KvFormat::Quantized(QuantFormat::Q8_KS) => Some(Self::Q8KS),
            KvFormat::Quantized(QuantFormat::Q8_1) => Some(Self::Q8_1),
            KvFormat::Quantized(QuantFormat::Q8_0) => Some(Self::Q8_0),
            KvFormat::Quantized(QuantFormat::Q5_1) => Some(Self::Q5_1),
            KvFormat::Quantized(QuantFormat::Q5_0) => Some(Self::Q5_0),
            KvFormat::Quantized(QuantFormat::Q4_KS) => Some(Self::Q4KS),
            KvFormat::Quantized(QuantFormat::Q4_1) => Some(Self::Q4_1),
            KvFormat::Quantized(QuantFormat::Q4_0) => Some(Self::Q4_0),
            KvFormat::Quantized(QuantFormat::Q3_1) => Some(Self::Q3_1),
            KvFormat::Quantized(QuantFormat::Q3_0) => Some(Self::Q3_0),
            KvFormat::Quantized(QuantFormat::Q2_1) => Some(Self::Q2_1),
            KvFormat::Quantized(QuantFormat::Q2_A) => Some(Self::Q2A),
            KvFormat::Quantized(QuantFormat::Q2_S) => Some(Self::Q2S),
            KvFormat::Quantized(QuantFormat::Q2_0) => Some(Self::Q2_0),
            KvFormat::Quantized(QuantFormat::Q1_S) => Some(Self::Q1S),
            KvFormat::Quantized(QuantFormat::Q0) => Some(Self::Q0),
            KvFormat::Quantized(QuantFormat::Q0_V) => Some(Self::Q0_V),
            KvFormat::Quantized(QuantFormat::Q1_A) => Some(Self::Q1_A),
            KvFormat::Quantized(QuantFormat::Q0_X) => Some(Self::Q0_X),
            KvFormat::Quantized(QuantFormat::Q0_M2) => Some(Self::Q0_M2),
            KvFormat::Quantized(QuantFormat::Q0_M4) => Some(Self::Q0_M4),
            _ => None,
        }
    }

    pub fn to_quant_format(self) -> Option<QuantFormat> {
        match self {
            Self::Q8KS => Some(QuantFormat::Q8_KS),
            Self::Q8_1 => Some(QuantFormat::Q8_1),
            Self::Q8_0 => Some(QuantFormat::Q8_0),
            Self::Q5_1 => Some(QuantFormat::Q5_1),
            Self::Q5_0 => Some(QuantFormat::Q5_0),
            Self::Q4KS => Some(QuantFormat::Q4_KS),
            Self::Q4_1 => Some(QuantFormat::Q4_1),
            Self::Q4_0 => Some(QuantFormat::Q4_0),
            Self::Q3_1 => Some(QuantFormat::Q3_1),
            Self::Q3_0 => Some(QuantFormat::Q3_0),
            Self::Q2_1 => Some(QuantFormat::Q2_1),
            Self::Q2A => Some(QuantFormat::Q2_A),
            Self::Q2S => Some(QuantFormat::Q2_S),
            Self::Q2_0 => Some(QuantFormat::Q2_0),
            Self::Q1S => Some(QuantFormat::Q1_S),
            Self::Q0 => Some(QuantFormat::Q0),
            Self::Q0_V => Some(QuantFormat::Q0_V),
            Self::Q1_A => Some(QuantFormat::Q1_A),
            Self::Q0_X => Some(QuantFormat::Q0_X),
            Self::Q0_M2 => Some(QuantFormat::Q0_M2),
            Self::Q0_M4 => Some(QuantFormat::Q0_M4),
            Self::F16 | Self::BF16 => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn to_ggml_dtype(self) -> GgmlDType {
        match self {
            Self::F16 => GgmlDType::F16,
            Self::BF16 => GgmlDType::BF16,
            Self::Q8KS => GgmlDType::Q8_KS,
            Self::Q8_1 => GgmlDType::Q8_1,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q5_1 => GgmlDType::Q5_1,
            Self::Q5_0 => GgmlDType::Q5_0,
            Self::Q4KS => GgmlDType::Q4_KS,
            Self::Q4_1 => GgmlDType::Q4_1,
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q3_1 => GgmlDType::Q3_1,
            Self::Q3_0 => GgmlDType::Q3_0,
            Self::Q2_1 => GgmlDType::Q2_1,
            Self::Q2A => GgmlDType::Q2_A,
            Self::Q2S => GgmlDType::Q2_S,
            Self::Q2_0 => GgmlDType::Q2_0,
            Self::Q1S => GgmlDType::Q1_S,
            Self::Q0 => GgmlDType::Q0,
            Self::Q0_V => GgmlDType::Q0_V,
            Self::Q1_A => GgmlDType::Q1_A,
            Self::Q0_X => GgmlDType::Q0_X,
            Self::Q0_M2 => GgmlDType::Q0_M2,
            Self::Q0_M4 => GgmlDType::Q0_M4,
        }
    }

    pub fn try_from_cuda_tag(code: i32) -> candle::Result<Self> {
        // Codes match SELECT_FMT_* defines in select_kv_format.cuh, which are
        // aligned to GgmlDType discriminants after the format-code migration.
        match code {
            1 => Ok(Self::F16),
            2 => Ok(Self::BF16),
            7 => Ok(Self::Q8_0),
            8 => Ok(Self::Q8_1),
            10 => Ok(Self::Q8KS),
            12 => Ok(Self::Q5_0),
            13 => Ok(Self::Q5_1),
            15 => Ok(Self::Q4_0),
            16 => Ok(Self::Q4_1),
            18 => Ok(Self::Q4KS),
            19 => Ok(Self::Q3_0),
            20 => Ok(Self::Q3_1),
            22 => Ok(Self::Q2_0),
            23 => Ok(Self::Q2_1),
            25 => Ok(Self::Q2S),
            26 => Ok(Self::Q2A),
            27 => Ok(Self::Q1S),
            28 => Ok(Self::Q0_V),
            29 => Ok(Self::Q1_A),
            30 => Ok(Self::Q0_X),
            31 => Ok(Self::Q0_M2),
            32 => Ok(Self::Q0_M4),
            33 => Ok(Self::Q0),
            // 99 and other unrecognized codes → error; callers may special-case floats via SELECT_FMT_F16/BF16
            _ => candle::bail!("unknown sampled-selection CUDA tag {code}"),
        }
    }

    pub fn from_cuda_tag(code: i32) -> Self {
        Self::try_from_cuda_tag(code)
            .unwrap_or_else(|_| panic!("unknown sampled-selection CUDA tag {code}"))
    }

    #[cfg(feature = "cuda")]
    pub fn to_cuda_tag(self) -> i32 {
        ggml_to_select_qtype(self.to_ggml_dtype())
            .unwrap_or_else(|_| panic!("missing sampled-selection CUDA tag for {self}"))
    }

    pub fn float_range(self) -> f32 {
        match self {
            // FP8 formats — outer scale is required to map values into the
            // format's fixed representable range.
            Self::Q2A => 1792.0,
            Self::Q2S => 672.0,
            Self::Q1S
            | Self::Q0
            | Self::Q0_V
            | Self::Q1_A
            | Self::Q0_X
            | Self::Q0_M2
            | Self::Q0_M4 => 448.0,
            // Block-scale formats — outer scale is actively harmful because it
            // overrides the block's adaptive internal scale d. Passthrough.
            Self::Q8KS
            | Self::Q8_1
            | Self::Q8_0
            | Self::Q5_1
            | Self::Q5_0
            | Self::Q4KS
            | Self::Q4_1
            | Self::Q4_0
            | Self::Q3_1
            | Self::Q3_0
            | Self::Q2_1
            | Self::Q2_0
            | Self::F16
            | Self::BF16 => 0.0,
        }
    }

    pub fn bits_per_elem(self) -> f32 {
        match self {
            Self::F16 => 16.0,
            Self::BF16 => 16.0,
            Self::Q8KS => 9.0,
            Self::Q8_1 => 9.0,
            Self::Q8_0 => 8.5,
            Self::Q5_1 => 6.0,
            Self::Q5_0 => 5.5,
            Self::Q4KS => 5.0,
            Self::Q4_1 => 5.0,
            Self::Q4_0 => 4.5,
            Self::Q3_1 => 4.0,
            Self::Q3_0 => 3.5,
            Self::Q2_1 => 3.0,
            Self::Q2A => 2.5,
            Self::Q2S => 2.25,
            Self::Q2_0 => 2.5,
            Self::Q1S => 1.25,
            Self::Q0 => 0.25,
            Self::Q0_V | Self::Q0_X => 0.5,
            Self::Q0_M2 => 0.75,
            Self::Q1_A => 1.5,
            Self::Q0_M4 => 2.0,
        }
    }

    /// Sorted ascending by bits-per-element (higher = less compressed = "worst case").
    /// Ties broken by format family. Must match format_table_index_cuda in the CUDA kernel.
    /// Total 23 entries (indices 0-22).
    pub fn table_index(self) -> usize {
        match self {
            Self::Q0     => 0,   // 0.25 bpe
            Self::Q0_X   => 1,   // 0.5  bpe
            Self::Q0_V   => 2,   // 0.5  bpe
            Self::Q0_M2  => 3,   // 0.75 bpe
            Self::Q1S    => 4,   // 1.25 bpe
            Self::Q1_A   => 5,   // 1.5  bpe (6 bytes / 32 elem)
            Self::Q0_M4  => 6,   // 2.0  bpe (8 bytes / 32 elem)
            Self::Q2S    => 7,   // 2.25 bpe
            Self::Q2_0   => 8,   // 2.5  bpe
            Self::Q2A => 9,   // 2.5  bpe
            Self::Q2_1   => 10,  // 3.0  bpe
            Self::Q3_0   => 11,  // 3.5  bpe
            Self::Q3_1   => 12,  // 4.0  bpe
            Self::Q4_0   => 13,  // 4.5  bpe
            Self::Q4_1   => 14,  // 5.0  bpe
            Self::Q4KS   => 15,  // 5.0  bpe
            Self::Q5_0   => 16,  // 5.5  bpe
            Self::Q5_1   => 17,  // 6.0  bpe
            Self::Q8_0   => 18,  // 8.5  bpe
            Self::Q8_1   => 19,  // 9.0  bpe
            Self::Q8KS   => 20,  // 9.0  bpe
            Self::F16    => 21,  // 16.0 bpe
            Self::BF16   => 22,  // 16.0 bpe
        }
    }

    pub fn from_table_index(idx: usize) -> Self {
        match idx {
            0  => Self::Q0,
            1  => Self::Q0_X,
            2  => Self::Q0_V,
            3  => Self::Q0_M2,
            4  => Self::Q1S,
            5  => Self::Q1_A,
            6  => Self::Q0_M4,
            7  => Self::Q2S,
            8  => Self::Q2_0,
            9  => Self::Q2A,
            10 => Self::Q2_1,
            11 => Self::Q3_0,
            12 => Self::Q3_1,
            13 => Self::Q4_0,
            14 => Self::Q4_1,
            15 => Self::Q4KS,
            16 => Self::Q5_0,
            17 => Self::Q5_1,
            18 => Self::Q8_0,
            19 => Self::Q8_1,
            20 => Self::Q8KS,
            21 => Self::F16,
            22 => Self::BF16,
            _  => Self::F16,
        }
    }

    pub fn grid_label(self) -> &'static str {
        match self {
            Self::Q0 => "Q0",
            Self::Q0_V => "Q0V",
            Self::Q1_A => "Q1A",
            Self::Q0_X => "Q0X",
            Self::Q0_M2 => "QM2",
            Self::Q0_M4 => "QM4",
            Self::Q1S => "Q1S",
            Self::Q2S => "Q2S",
            Self::Q2_0 => "Q20",
            Self::Q2A => "Q2A",
            Self::Q2_1 => "Q21",
            Self::Q3_0 => "Q30",
            Self::Q3_1 => "Q31",
            Self::Q4_0 => "Q40",
            Self::Q4_1 => "Q41",
            Self::Q4KS => "Q4K",
            Self::Q8_0 => "Q80",
            Self::Q8_1 => "Q81",
            Self::Q8KS => "Q8K",
            Self::Q5_0 => "Q50",
            Self::Q5_1 => "Q51",
            Self::BF16 => "BFL",
            Self::F16 => "F16",
        }
    }

    pub fn apply_quant(self, block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
        match self {
            Self::F16 => *block,
            Self::BF16 => cpu::round_trip_bf16(block),
            Self::Q8KS => cpu::round_trip_q8_ks(block),
            Self::Q8_1 => cpu::round_trip_q8_1(block),
            Self::Q8_0 => cpu::round_trip_q8_0(block),
            Self::Q5_1 => cpu::round_trip_q5_1(block),
            Self::Q5_0 => cpu::round_trip_q5_0(block),
            Self::Q4KS => cpu::round_trip_q4_ks(block),
            Self::Q4_1 => cpu::round_trip_q4_1(block),
            Self::Q4_0 => cpu::round_trip_q4_0(block),
            Self::Q3_1 => cpu::round_trip_q3_1(block),
            Self::Q3_0 => cpu::round_trip_q3_0(block),
            Self::Q2_1 => cpu::round_trip_q2_1(block),
            Self::Q2A => cpu::round_trip_q2_a(block),
            Self::Q2S => cpu::round_trip_q2_s(block),
            Self::Q2_0 => cpu::round_trip_q2_0(block),
            Self::Q1S => cpu::round_trip_q1_s(block),
            Self::Q0 => cpu::round_trip_q0(block),
            Self::Q0_V => cpu::round_trip_q0_v(block),
            Self::Q1_A => cpu::round_trip_q1_a(block),
            Self::Q0_X => cpu::round_trip_q0_x(block),
            Self::Q0_M2 => cpu::round_trip_q0_m2(block),
            Self::Q0_M4 => cpu::round_trip_q0_m4(block),
        }
    }
}

impl std::fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::Q8KS => write!(f, "Q8_KS"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q4KS => write!(f, "Q4_KS"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q3_1 => write!(f, "Q3_1"),
            Self::Q3_0 => write!(f, "Q3_0"),
            Self::Q2_1 => write!(f, "Q2_1"),
            Self::Q2A => write!(f, "Q2_A"),
            Self::Q2S => write!(f, "Q2_S"),
            Self::Q2_0 => write!(f, "Q2_0"),
            Self::Q1S => write!(f, "Q1_S"),
            Self::Q0 => write!(f, "Q0"),
            Self::Q0_V => write!(f, "Q0_V"),
            Self::Q1_A => write!(f, "Q1_A"),
            Self::Q0_X => write!(f, "Q0_X"),
            Self::Q0_M2 => write!(f, "Q0_M2"),
            Self::Q0_M4 => write!(f, "Q0_M4"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SampleFormat;

    // Every SELECT_FMT_* tag that the CUDA kernel can emit as a KV format,
    // paired with the expected SampleFormat.  Must stay in sync with
    // select_kv_format.cuh.  If a new format is added there, add it here too —
    // the count assertion below will catch the mismatch at test time.
    const KNOWN_KV_TAGS: &[(i32, SampleFormat)] = &[
        (1,  SampleFormat::F16),
        (2,  SampleFormat::BF16),
        (7,  SampleFormat::Q8_0),
        (8,  SampleFormat::Q8_1),
        (10, SampleFormat::Q8KS),
        (12, SampleFormat::Q5_0),
        (13, SampleFormat::Q5_1),
        (15, SampleFormat::Q4_0),
        (16, SampleFormat::Q4_1),
        (18, SampleFormat::Q4KS),
        (19, SampleFormat::Q3_0),
        (20, SampleFormat::Q3_1),
        (22, SampleFormat::Q2_0),
        (23, SampleFormat::Q2_1),
        (25, SampleFormat::Q2S),
        (26, SampleFormat::Q2A),
        (27, SampleFormat::Q1S),
        (28, SampleFormat::Q0_V),
        (29, SampleFormat::Q1_A),
        (30, SampleFormat::Q0_X),
        (31, SampleFormat::Q0_M2),
        (32, SampleFormat::Q0_M4),
        (33, SampleFormat::Q0),
    ];

    #[test]
    fn sample_format_coverage() {
        // Every tag in KNOWN_KV_TAGS must decode to the expected format.
        for &(tag, expected) in KNOWN_KV_TAGS {
            let got = SampleFormat::try_from_cuda_tag(tag)
                .unwrap_or_else(|_| panic!("KV tag {tag} rejected — add it to try_from_cuda_tag"));
            assert_eq!(got, expected, "tag {tag} decoded to wrong format");
        }

        // KNOWN_KV_TAGS and SampleFormat::ALL must stay the same size.
        // If you add a variant to one, add it to the other.
        assert_eq!(
            KNOWN_KV_TAGS.len(),
            SampleFormat::ALL.len(),
            "KNOWN_KV_TAGS ({}) and SampleFormat::ALL ({}) are out of sync",
            KNOWN_KV_TAGS.len(),
            SampleFormat::ALL.len(),
        );

        // Non-KV SELECT_FMT codes must be rejected (F32, R16, K-quants, FP8, etc.)
        for &tag in &[0i32, 3, 4, 5, 6, 9, 11, 14, 17, 21, 24, 34, 35, 99] {
            assert!(
                SampleFormat::try_from_cuda_tag(tag).is_err(),
                "non-KV tag {tag} should be rejected by try_from_cuda_tag"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sample_format_cuda_tag_roundtrip() {
        // For every format in SampleFormat::ALL, to_cuda_tag() → try_from_cuda_tag()
        // must round-trip back to the same format.
        for &fmt in SampleFormat::ALL {
            let tag = fmt.to_cuda_tag();
            let roundtrip = SampleFormat::try_from_cuda_tag(tag)
                .unwrap_or_else(|_| panic!("try_from_cuda_tag failed for {fmt} (tag={tag})"));
            assert_eq!(roundtrip, fmt, "CUDA tag roundtrip mismatch for {fmt}");
        }
    }
}

#[derive(Debug, Clone)]
pub struct ErrorSurface {
    pub n_batch: usize,
    pub n_head: usize,
    pub n_dim: usize,
    pub n_quant: usize,
    pub chunk_size: usize,
    pub side: SampleSide,
    pub data: Vec<f32>,
    /// Per-block Q·K attention relevance weights, populated by the GPU sampling
    /// kernel for downstream analysis.  `None` on the CPU path.
    pub q_relevance: Option<Vec<f32>>,
}

impl ErrorSurface {
    pub fn index_of(
        &self,
        batch_item: usize,
        head_dim: usize,
        quant_index: usize,
        head: usize,
    ) -> usize {
        (((batch_item * self.n_dim) + head_dim) * self.n_quant + quant_index) * self.n_head + head
    }

    pub fn get(&self, batch_item: usize, head_dim: usize, quant_index: usize, head: usize) -> f32 {
        self.data[self.index_of(batch_item, head_dim, quant_index, head)]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompressionSummary {
    pub ideal_bpe: f64,
    pub head_bpe: f64,
    pub palette4_bpe: f64,
    pub ideal_cr: f64,
    pub head_cr: f64,
    pub palette4_cr: f64,
}
