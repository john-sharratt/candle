//! The enumeration of the quantized and float formats the workspace stores
//! weights and KV in, with the GGUF file codes that map each one to disk.
//!
//! Each variant has a block geometry, and its bits per weight decide how many
//! bytes a weight puts on the bus, which under layer streaming decides decode
//! speed. The GGUF on-disk numbering differs from the in-workspace discriminants,
//! so the two file-code functions are the one translation boundary and are each
//! other's inverse. The KO twins are lane-major GPU layouts that the int8
//! tensor-core matmul reads. They are never written to disk, and their codes
//! exist only so that they round-trip.

use super::*;

#[repr(u32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    /// R16: Raw F16 + reserved Q-capture space (128 bytes per 32 elements)
    R16 = 3,
    /// P2: 2-bit palette index (1 byte per 4 head_dim positions, pure arena routing)
    P2 = 4,
    /// AWQ 4-bit with group size 128
    QAWQ = 5,
    QAWQ_G64 = 6,
    /// Quant blocks
    Q8_0 = 7,
    Q8_1 = 8,
    Q8_K = 9,
    Q8_KS = 10,

    Q6_K = 11,

    Q5_0 = 12,
    Q5_1 = 13,
    Q5_K = 14,

    Q4_0 = 15,
    Q4_1 = 16,
    Q4_K = 17,
    Q4_KS = 18,

    Q3_0 = 19,
    Q3_1 = 20,
    Q3_K = 21,

    Q2_0 = 22,
    Q2_1 = 23,
    Q2_K = 24,
    Q2_S = 25,
    Q2_A = 26,

    Q1_S = 27,

    Q0_V = 28,
    Q1_A = 29,
    Q0_X = 30,
    Q0_M2 = 31,
    Q0_M4 = 32,
    Q0 = 33,

    F8E4M3 = 34,
    F8E5M2 = 35,

    U8 = 36,
    I8 = 37,
    U16 = 38,
    I16 = 39,
    U32 = 40,
    I32 = 41,
    U64 = 42,
    I64 = 43,
    F64 = 44,

    /// Lane-major per-128 affine weight format for the q8a128 int8 tensor-core matmul.
    /// NOT a byte-permute of a stored K-quant: it carries its own per-128 `(scale, min)`
    /// (one pair per 128 K per row) and is re-quantized straight from F32 (see
    /// `ko_quant::quantize_ko`), so it is lossy vs the source — not bit-identical. Q4_KO is
    /// 4-bit; the de-interleaved lane-major layout lets each int8-matmul lane pull its 4
    /// sub-uint32s in one wide LDS. Value 45 mirrors `QTYPE_Q4_KO` / `QType::Q4_KO`; the
    /// Q5/Q6/Q8_KO twins follow at 46-48.
    Q4_KO = 45,
    /// Lane-major per-128 affine KO twins (5/6/8-bit) — the same re-quantized-from-F32 format
    /// as Q4_KO, NOT a permutation of Q5_K/Q6_K/Q8_K. GPU-only weight formats; mirror
    /// `QTYPE_Q*_KO` / `QType::Q*_KO`.
    Q5_KO = 46,
    Q6_KO = 47,
    Q8_KO = 48,

    /// OCP MXFP4 (4-bit micro-scaling FP4): 32 elems/block, one E8M0 scale byte + 16
    /// nibbles indexing the E2M1 table. GGUF file code 39. The native trained format
    /// for the DeepSeek-V4 routed experts. See [`k_quants::BlockMXFP4`].
    MXFP4 = 49,

    /// Lane-major per-sub MXFP4 for the q8a128 int8 tensor-core matmul — the KO twin the
    /// MXFP4 routed experts repack to (an exact byte permutation of the native blocks; no
    /// requant). The E2M1 nibbles are kept **4-bit**; the kernel runs one int32 MMA per
    /// 32-K sub and folds each with its own E8M0 power-of-two scale in FP — the per-32
    /// scales apply exactly. Stays 4-bit → fits in RAM where Q6_KO/Q8_KO don't. GPU-only;
    /// value 50 mirrors `QTYPE_MXFP4_KO` / `QType::MXFP4_KO`. See
    /// `ko_quant::mxfp4_native_to_ko_gpu_chunk` + `loader/mxfp4.cuh`.
    MXFP4_KO = 50,

    /// Lane-major per-128 affine KO twin at **2-bit** — the smallest KO weight. Same
    /// re-quantized-from-F32 per-128 `(scale, min)` affine format as `Q4_KO`, but each value is
    /// a 2-bit crumb (0..3): the 128-K tile's quants pack into 32 B (`block_c_q2_KO_k128`,
    /// `int qs[8]`) vs Q4_KO's 64 B, so a chunk is 288 B / 1024 elems (~2.25 bpw). Its 2-bit
    /// crumb layout mirrors the high-2-bit crumb region Q6_KO already carries (`cr0`/`cr1` at
    /// `lane*8 + sub*2`), used here as the whole value. GPU-only (built by requantizing from F32
    /// on-device, like the other KO twins); read by the `q2_ko_int8_f32_grouped` int8 kernel.
    /// Value 51 mirrors `QTYPE_Q2_KO` / `QType::Q2_KO`. See `ko_quant::quantize_q2_ko`.
    Q2_KO = 51,

    /// Lane-major per-128 affine KO twin at **3-bit** — `Q3_K`'s same-width twin, and the
    /// reason it exists. Without it `Q3_K` had no KO form and `to_ko` rounded it *up* to
    /// `Q4_KO`; under layer streaming that upcast is paid as PCIe bytes on every forward, and
    /// on the 27B at Q3_K_M it inflated the layer pack by 12.8%.
    ///
    /// Carries no `ql` plane: the value is the 2-bit **crumb** (bits 0-1, `Q2_KO`'s whole-value
    /// region) plus the 1-bit **hi** plane (bit 2, `Q5_KO`'s region) — so a chunk is
    /// `256 + 128 + 32` = 416 B / 1024 elems (**3.25 bpw**) and both planes are layouts the
    /// kernels already read. GPU-only, built by requantizing from F32 on-device like the other
    /// KO twins; read by the `q3_ko_int8_f32_grouped` int8 kernel. Value 52 mirrors
    /// `QTYPE_Q3_KO` / `QType::Q3_KO`. See [`ko_quant::ko_params`].
    Q3_KO = 52,
}

impl GgmlDType {
    /// Whether a CUDA tensor of this type dequantizes straight to BF16 on the device
    /// ([`QTensor::dequantize_bf16`]) with no F32 intermediate: the block formats the
    /// dequantize kernel reads, and MXFP4. Float and integer storage, P2, and the KO twins
    /// (which refuse any float dequantize) go through F32 instead.
    pub fn dequantizes_to_bf16(self) -> bool {
        use GgmlDType::*;
        matches!(
            self,
            Q4_0 | Q4_1
                | Q5_0
                | Q5_1
                | Q8_0
                | Q8_1
                | Q2_K
                | Q3_K
                | Q4_K
                | Q5_K
                | Q6_K
                | Q8_K
                | QAWQ
                | QAWQ_G64
                | Q4_KS
                | Q8_KS
                | Q2_0
                | Q3_0
                | R16
                | Q0
                | Q1_S
                | Q2_S
                | Q2_A
                | Q2_1
                | Q3_1
                | Q0_V
                | Q1_A
                | Q0_X
                | Q0_M2
                | Q0_M4
                | MXFP4
        )
    }

    /// Storage cost in bits per weight, scales and all.
    ///
    /// `type_size × 8 / block_size` — the number that decides how many bytes a weight puts on
    /// the bus, which under layer streaming is what decides decode. Named here because three
    /// call sites compare widths (the `to_ko` table's test, the narrowing policy, and the
    /// repack's refusal to widen) and a formula written out three times is one that drifts.
    ///
    /// It counts the *storage*, not the value grid: `Q4_1` is 5.0 bpw and `Q4_KO` 4.25, yet
    /// both hold 4-bit values — the difference is a scale-and-min per 32 against one per 128.
    /// So it bounds bandwidth exactly and precision only loosely.
    pub fn bits_per_weight(self) -> f64 {
        self.type_size() as f64 * 8.0 / self.block_size() as f64
    }

    /// True for the lane-major per-128 KO weight formats — every affine width
    /// (`Q2_KO`/`Q3_KO`/`Q4_KO`/`Q5_KO`/`Q6_KO`/`Q8_KO`) plus `MXFP4_KO`.
    /// These are the only weight layouts the q8a128 int8 tensor-core matmul can read, so the
    /// `DynamicTensor::Int8` matmul path guards on this.
    pub fn is_ko(self) -> bool {
        matches!(
            self,
            Self::Q2_KO
                | Self::Q3_KO
                | Self::Q4_KO
                | Self::Q5_KO
                | Self::Q6_KO
                | Self::Q8_KO
                | Self::MXFP4_KO
        )
    }

    /// The KO weight twin used for int8 optimization, selected by [`Int8Mode`].
    ///
    /// [`Int8Mode::Performance`] picks the **same-width** twin (2-bit→`Q2_KO`, 3-bit→`Q3_KO`,
    /// 4-bit→`Q4_KO`, 5-bit→`Q5_KO`, 6-bit→`Q6_KO`, 8-bit→`Q8_KO`). It takes the
    /// per-32→per-128 granularity hit on the weight but is the fastest and smallest.
    ///
    /// **Same-width is now a rule with no exceptions, and it did not used to be.** `Q2_K` and
    /// `Q3_K` both mapped to `Q4_KO` because those twins did not exist — a 71% and a 31%
    /// inflation respectively, invisible while weights were resident and paid VRAM once. Under
    /// layer streaming a weight's width is PCIe bytes on *every forward*, and on the 27B at
    /// Q3_K_M the `Q3_K → Q4_KO` upcast alone inflated the layer pack by 12.8%
    /// (`docs/archived/qwen38_layer_streaming.md` §2.3). Rounding *down* was never the alternative:
    /// `Q2_KO`'s four levels floor at `rel_l2 ≈ 0.325`, a quality loss rather than a repack.
    ///
    /// [`Int8Mode::Precision`] steps the source one notch up the ladder
    /// (`Q2_KO` < `Q3_KO` < `Q4_KO` < `Q5_KO` < `Q6_KO` < `Q8_KO`): 2-bit→`Q3_KO`,
    /// 3-bit→`Q4_KO`, 4-bit→`Q5_KO`, 5-bit→`Q6_KO`. The extra bit absorbs the granularity loss
    /// so the re-quant of the already-quantized weight is near-lossless. At the top of the
    /// ladder there is no finer twin, so 6-bit→`Q6_KO` and 8-bit→`Q8_KO` are same-width in both
    /// modes.
    ///
    /// Errors for [`Int8Mode::Off`] (no KO twin) and for dtypes with no KO form.
    pub fn to_ko(self, mode: Int8Mode) -> Result<Self> {
        // Already a KO format (e.g. a pre-repacked / prepared weight): the twin is itself, so
        // the optimize/geometry step is a no-op regardless of mode — nothing to repack.
        if self.is_ko() {
            return Ok(self);
        }
        match mode {
            Int8Mode::Off => crate::bail!("to_ko: Int8Mode::Off has no KO weight twin"),
            Int8Mode::Performance => Ok(match self {
                // **The Q2_K row is the one to measure before trusting.** Q2_K
                // carries a `(scale, min)` per *16* elements inside its
                // 256-superblock; Q2_KO is a single per-128 affine over four
                // levels, which the note above records as flooring at
                // `rel_l2 ≈ 0.325`. Nothing currently gates the difference:
                // `ko_quant`'s 0.400 ceiling scores Q2_KO **from f32** and never
                // sees the Q2_K→Q2_KO requant, so this row would pass whatever it
                // cost. Left as it is — the same-width rule is deliberate and
                // pinned by `the_int8_twin_table_is_same_width_then_one_step_up`
                // — but the gap between the ceiling and this path is real, and
                // the fix is a measurement of the requant rather than a guess at
                // the mapping.
                Self::Q2_K => Self::Q2_KO,
                // Q3_K → Q3_KO is the clearer of the two: eight levels against
                // Q3_K's own 3-bit grid, and shrinking it is the stated PCIe
                // motive.
                Self::Q3_K => Self::Q3_KO,
                Self::Q4_0 | Self::Q4_1 | Self::Q4_K => Self::Q4_KO,
                Self::Q5_0 | Self::Q5_1 | Self::Q5_K => Self::Q5_KO,
                Self::Q6_K => Self::Q6_KO,
                Self::Q8_0 | Self::Q8_1 | Self::Q8_K => Self::Q8_KO,
                // **A float source gets the widest twin there is**, and both modes agree
                // because there is no wider KO grid than Q8_KO to step up to — the `Q8_*`
                // row above lands here for the same reason.
                //
                // `repack_ko_into` builds it directly: `dequantize_f32_into` widens a float
                // band with a cast where it would dequantize a quantized one, so the route
                // is the same banded one. That is why this row exists at all — without it
                // `to_ko` failed on a float projection and the caller fell back to
                // dequantizing the whole tensor and re-quantizing it to `Q8_0`, which is
                // four whole-tensor buffers where the banded route needs 48 MiB.
                Self::F32 | Self::F16 | Self::BF16 => Self::Q8_KO,
                // MXFP4 keeps its native 4-bit E2M1 nibbles (exact byte permutation, no
                // requant to a wider grid; the kernel folds each per-32 E8M0 sub exactly),
                // so it fits in RAM where Q6_KO/Q8_KO don't, with no weight-side loss.
                Self::MXFP4 => Self::MXFP4_KO,
                other => crate::bail!("no KO weight form for {other:?}"),
            }),
            Int8Mode::Precision => Ok(match self {
                Self::Q2_K => Self::Q3_KO,
                Self::Q3_K => Self::Q4_KO,
                Self::Q4_0 | Self::Q4_1 | Self::Q4_K => Self::Q5_KO,
                Self::Q5_0 | Self::Q5_1 | Self::Q5_K => Self::Q6_KO,
                Self::Q6_K => Self::Q6_KO,
                Self::Q8_0 | Self::Q8_1 | Self::Q8_K => Self::Q8_KO,
                // **A float source gets the widest twin there is**, and both modes agree
                // because there is no wider KO grid than Q8_KO to step up to — the `Q8_*`
                // row above lands here for the same reason.
                //
                // `repack_ko_into` builds it directly: `dequantize_f32_into` widens a float
                // band with a cast where it would dequantize a quantized one, so the route
                // is the same banded one. That is why this row exists at all — without it
                // `to_ko` failed on a float projection and the caller fell back to
                // dequantizing the whole tensor and re-quantizing it to `Q8_0`, which is
                // four whole-tensor buffers where the banded route needs 48 MiB.
                Self::F32 | Self::F16 | Self::BF16 => Self::Q8_KO,
                // Same 4-bit per-sub twin as Performance — the per-sub fold is already
                // weight-exact, so there is no wider grid to step up to.
                Self::MXFP4 => Self::MXFP4_KO,
                other => crate::bail!("no KO weight form for {other:?}"),
            }),
        }
    }

    /// Map an in-workspace integer (`GgmlDType as u32` discriminant) back to
    /// the corresponding `GgmlDType` variant.  This is the identity mapping —
    /// `from_u32(Self::Q4_K as u32)` returns `Self::Q4_K`.
    ///
    /// ⚠ This is NOT the GGUF/GGML on-disk file-format code.  Use
    /// `from_gguf_file_code` when reading raw integers from GGUF/GGML files.
    ///
    /// Every other integer used for a quant format inside the workspace —
    /// the C++ `QType` in `block_compact.cuh`, `ArenaFormat::*` in
    /// `arena_table.cuh`, `SELECT_FMT_*` in `select_kv_format.cuh`, the
    /// KV-side Rust `QType` in `candle_kernels/src/simple/quantized.rs`,
    /// the matmul-side Rust `QType` in `candle_kernels/src/quantized/api.rs`,
    /// and every `qtype: i32` FFI argument — all use `GgmlDType as u32`.
    pub fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::BF16,
            3 => Self::R16,
            4 => Self::P2,
            5 => Self::QAWQ,
            6 => Self::QAWQ_G64,
            7 => Self::Q8_0,
            8 => Self::Q8_1,
            9 => Self::Q8_K,
            10 => Self::Q8_KS,
            11 => Self::Q6_K,
            12 => Self::Q5_0,
            13 => Self::Q5_1,
            14 => Self::Q5_K,
            15 => Self::Q4_0,
            16 => Self::Q4_1,
            17 => Self::Q4_K,
            18 => Self::Q4_KS,
            19 => Self::Q3_0,
            20 => Self::Q3_1,
            21 => Self::Q3_K,
            22 => Self::Q2_0,
            23 => Self::Q2_1,
            24 => Self::Q2_K,
            25 => Self::Q2_S,
            26 => Self::Q2_A,
            27 => Self::Q1_S,
            28 => Self::Q0_V,
            29 => Self::Q1_A,
            30 => Self::Q0_X,
            31 => Self::Q0_M2,
            32 => Self::Q0_M4,
            33 => Self::Q0,
            34 => Self::F8E4M3,
            35 => Self::F8E5M2,
            36 => Self::U8,
            37 => Self::I8,
            38 => Self::U16,
            39 => Self::I16,
            40 => Self::U32,
            41 => Self::I32,
            42 => Self::U64,
            43 => Self::I64,
            44 => Self::F64,
            45 => Self::Q4_KO,
            46 => Self::Q5_KO,
            47 => Self::Q6_KO,
            48 => Self::Q8_KO,
            49 => Self::MXFP4,
            50 => Self::MXFP4_KO,
            51 => Self::Q2_KO,
            52 => Self::Q3_KO,
            _ => crate::bail!("unknown dtype discriminant {u}"),
        };
        Ok(dtype)
    }

    /// Return the in-workspace integer representation (`GgmlDType as u32`).
    ///
    /// ⚠ This is NOT the GGUF/GGML on-disk file-format code.  Use
    /// `to_gguf_file_code` when writing integers to GGUF/GGML files.
    pub fn to_u32(self) -> u32 {
        self as u32
    }

    /// Decode a GGUF / GGML **on-disk file-format** dtype code into our
    /// in-memory `GgmlDType`.
    ///
    /// ⚠ THIS IS THE SINGLE TRANSLATION BOUNDARY between the on-disk GGUF
    /// integer and everything inside the workspace.  The on-disk numbering
    /// (upstream llama.cpp) differs from the workspace discriminants:
    ///   * On-disk: Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, BF16=30, …
    ///   * In-workspace: Q4_K=17, Q5_K=14, Q6_K=11, Q8_K=9, BF16=2, …
    ///
    /// `from_gguf_file_code` and `to_gguf_file_code` are each other's inverse.
    pub(crate) fn from_gguf_file_code(u: u32) -> Result<Self> {
        let dtype = match u {
            // Standard GGML file codes (match llama.cpp ggml.h).
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            // Standard ggml integer / f64 tensor codes (llama.cpp writes these for
            // non-quant tensors such as the hash-routing `tid2eid` table). candle's own
            // writer uses the 231+ codes, but external files use these — read both.
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            // MXFP4 (OCP micro-scaling FP4) — DeepSeek-V4 routed experts.
            39 => Self::MXFP4,
            // AWQ types use IDs 100+ to avoid conflicts with GGML types
            100 => Self::QAWQ,
            101 => Self::QAWQ_G64,
            // Candle-specific extensions use IDs 200+ to stay clear of future GGML / AWQ additions.
            200 => Self::Q4_KS,
            201 => Self::Q8_KS,
            202 => Self::Q2_0,
            203 => Self::Q3_0,
            204 => Self::Q2_S,
            205 => Self::Q2_A,
            206 => Self::Q1_S,
            207 => Self::R16,
            208 => Self::Q0,
            209 => Self::Q3_1,
            210 => Self::Q2_1,
            211 => Self::P2,
            212 => Self::Q0_V,
            213 => Self::Q1_A,
            214 => Self::Q0_X,
            215 => Self::Q0_M2,
            216 => Self::Q0_M4,
            217 => Self::F8E4M3,
            218 => Self::F8E5M2,
            // KO byte-permuted twins — never actually on disk, but kept round-trippable
            // with `to_gguf_file_code` for consistency.
            219 => Self::Q4_KO,
            220 => Self::Q5_KO,
            221 => Self::Q6_KO,
            222 => Self::Q8_KO,
            223 => Self::MXFP4_KO,
            224 => Self::Q2_KO,
            225 => Self::Q3_KO,
            230 => Self::F64,
            231 => Self::U8,
            232 => Self::I8,
            233 => Self::U16,
            234 => Self::I16,
            235 => Self::U32,
            236 => Self::I32,
            237 => Self::U64,
            238 => Self::I64,
            _ => crate::bail!("unknown gguf file dtype code {u}"),
        };
        Ok(dtype)
    }

    /// Encode a `GgmlDType` back into its GGUF / GGML on-disk file-format code.
    ///
    /// ⚠ MUST be the exact inverse of `from_gguf_file_code` above — the output
    /// is the integer written to disk in GGUF tensor metadata, NOT the
    /// `#[repr(u32)]` discriminant of this enum. `pub` for the same streaming
    /// GGUF composers as [`gguf_file::Value::write`].
    pub fn to_gguf_file_code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::BF16 => 30,
            Self::QAWQ => 100,
            Self::QAWQ_G64 => 101,
            Self::Q4_KS => 200,
            Self::Q8_KS => 201,
            Self::Q2_0 => 202,
            Self::Q3_0 => 203,
            Self::Q2_S => 204,
            Self::Q2_A => 205,
            Self::Q1_S => 206,
            Self::R16 => 207,
            Self::Q0 => 208,
            Self::Q3_1 => 209,
            Self::Q2_1 => 210,
            Self::P2 => 211,
            Self::Q0_V => 212,
            Self::Q1_A => 213,
            Self::Q0_X => 214,
            Self::Q0_M2 => 215,
            Self::Q0_M4 => 216,
            Self::F8E4M3 => 217,
            Self::F8E5M2 => 218,
            Self::MXFP4 => 39,
            Self::F64 => 230,
            Self::U8 => 231,
            Self::I8 => 232,
            Self::U16 => 233,
            Self::I16 => 234,
            Self::U32 => 235,
            Self::I32 => 236,
            Self::U64 => 237,
            Self::I64 => 238,
            // KO twins are GPU-only and never actually written to disk; codes kept for
            // exhaustiveness / round-trip with `from_gguf_file_code`.
            Self::Q4_KO => 219,
            Self::Q5_KO => 220,
            Self::Q6_KO => 221,
            Self::Q8_KO => 222,
            Self::MXFP4_KO => 223,
            Self::Q2_KO => 224,
            Self::Q3_KO => 225,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F64 => Box::new(vec![f64::zeros(); elem_count]),
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::F8E4M3 => Box::new(vec![BlockQ8_0::zeros(); elem_count]),
            Self::F8E5M2 => Box::new(vec![BlockQ8_0::zeros(); elem_count]),
            Self::U8 => Box::new(vec![u8::zeros(); elem_count]),
            Self::I8 => Box::new(vec![i8::zeros(); elem_count]),
            Self::U16 => Box::new(vec![u16::zeros(); elem_count]),
            Self::I16 => Box::new(vec![i16::zeros(); elem_count]),
            Self::U32 => Box::new(vec![u32::zeros(); elem_count]),
            Self::I32 => Box::new(vec![i32::zeros(); elem_count]),
            Self::U64 => Box::new(vec![u64::zeros(); elem_count]),
            Self::I64 => Box::new(vec![i64::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2_K => Box::new(vec![BlockQ2_K::zeros(); elem_count / BlockQ2_K::BLCK_SIZE]),
            Self::Q3_K => Box::new(vec![BlockQ3_K::zeros(); elem_count / BlockQ3_K::BLCK_SIZE]),
            Self::Q4_K => Box::new(vec![BlockQ4_K::zeros(); elem_count / BlockQ4_K::BLCK_SIZE]),
            Self::Q5_K => Box::new(vec![BlockQ5_K::zeros(); elem_count / BlockQ5_K::BLCK_SIZE]),
            Self::Q6_K => Box::new(vec![BlockQ6_K::zeros(); elem_count / BlockQ6_K::BLCK_SIZE]),
            Self::Q8_K => Box::new(vec![BlockQ8_K::zeros(); elem_count / BlockQ8_K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
            Self::QAWQ => Box::new(vec![BlockQAWQ::zeros(); elem_count / BlockQAWQ::BLCK_SIZE]),
            Self::QAWQ_G64 => Box::new(vec![
                BlockQAWQ_G64::zeros();
                elem_count / BlockQAWQ_G64::BLCK_SIZE
            ]),
            Self::Q4_KS => Box::new(vec![
                BlockQ4_KS::zeros();
                elem_count / BlockQ4_KS::BLCK_SIZE
            ]),
            Self::Q8_KS => Box::new(vec![
                BlockQ8_KS::zeros();
                elem_count / BlockQ8_KS::BLCK_SIZE
            ]),
            Self::Q2_0 => Box::new(vec![BlockQ2_0::zeros(); elem_count / BlockQ2_0::BLCK_SIZE]),
            Self::Q3_0 => Box::new(vec![BlockQ3_0::zeros(); elem_count / BlockQ3_0::BLCK_SIZE]),
            Self::R16 => Box::new(vec![BlockR16::zeros(); elem_count / BlockR16::BLCK_SIZE]),
            Self::Q0 => Box::new(vec![BlockQ0::zeros(); elem_count / BlockQ0::BLCK_SIZE]),
            Self::Q1_S => Box::new(vec![BlockQ1S::zeros(); elem_count / BlockQ1S::BLCK_SIZE]),
            Self::Q2_S => Box::new(vec![BlockQ2S::zeros(); elem_count / BlockQ2S::BLCK_SIZE]),
            Self::Q2_A => Box::new(vec![BlockQ2A::zeros(); elem_count / BlockQ2A::BLCK_SIZE]),
            Self::Q2_1 => Box::new(vec![BlockQ2_1::zeros(); elem_count / BlockQ2_1::BLCK_SIZE]),
            Self::Q3_1 => Box::new(vec![BlockQ3_1::zeros(); elem_count / BlockQ3_1::BLCK_SIZE]),
            Self::P2 => Box::new(vec![BlockP2::zeros(); elem_count / BlockP2::BLCK_SIZE]),
            Self::Q0_V => Box::new(vec![BlockQ0V::zeros(); elem_count / BlockQ0V::BLCK_SIZE]),
            Self::Q1_A => Box::new(vec![BlockQ1A::zeros(); elem_count / BlockQ1A::BLCK_SIZE]),
            Self::Q0_X => Box::new(vec![BlockQ0X::zeros(); elem_count / BlockQ0X::BLCK_SIZE]),
            Self::Q0_M2 => Box::new(vec![BlockQ0M2::zeros(); elem_count / BlockQ0M2::BLCK_SIZE]),
            Self::Q0_M4 => Box::new(vec![BlockQ0M4::zeros(); elem_count / BlockQ0M4::BLCK_SIZE]),
            Self::Q4_KO => Box::new(vec![
                BlockQ4_KO::zeros();
                elem_count / BlockQ4_KO::BLCK_SIZE
            ]),
            Self::Q5_KO => Box::new(vec![
                BlockQ5_KO::zeros();
                elem_count / BlockQ5_KO::BLCK_SIZE
            ]),
            Self::Q6_KO => Box::new(vec![
                BlockQ6_KO::zeros();
                elem_count / BlockQ6_KO::BLCK_SIZE
            ]),
            Self::Q8_KO => Box::new(vec![
                BlockQ8_KO::zeros();
                elem_count / BlockQ8_KO::BLCK_SIZE
            ]),
            Self::MXFP4 => Box::new(vec![
                BlockMXFP4::zeros();
                elem_count / BlockMXFP4::BLCK_SIZE
            ]),
            // MXFP4_KO is a GPU-only lane-major chunk (built by repacking MXFP4 on-device,
            // never as CPU blocks) — there is no host block struct to zero-fill.
            Self::MXFP4_KO => {
                panic!("MXFP4_KO has no CPU block form; build it via repack from MXFP4 on CUDA")
            }
            // Q2_KO is a GPU-only lane-major chunk (built by requantizing F32 on-device, or by
            // `ko_quant::quantize_q2_ko` for the host prepare path) — no host block struct.
            Self::Q2_KO => {
                panic!("Q2_KO has no CPU block form; build it via quantize_q2_ko / repack on CUDA")
            }
            // Q3_KO likewise: a crumb+hi lane-major chunk with no `ql` plane, so there is no
            // host block struct — `ko_quant::quantize_ko` emits it, or the on-device repack.
            Self::Q3_KO => {
                panic!("Q3_KO has no CPU block form; build it via quantize_ko / repack on CUDA")
            }
        }
    }
    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 => 1,
            Self::F8E5M2 => 1,
            Self::U8 => 1,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 2,
            Self::U32 => 4,
            Self::I32 => 4,
            Self::U64 => 8,
            Self::I64 => 8,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2_K => std::mem::size_of::<BlockQ2_K>(),
            Self::Q3_K => std::mem::size_of::<BlockQ3_K>(),
            Self::Q4_K => std::mem::size_of::<BlockQ4_K>(),
            Self::Q5_K => std::mem::size_of::<BlockQ5_K>(),
            Self::Q6_K => std::mem::size_of::<BlockQ6_K>(),
            Self::Q8_K => std::mem::size_of::<BlockQ8_K>(),
            Self::QAWQ => std::mem::size_of::<BlockQAWQ>(),
            Self::QAWQ_G64 => std::mem::size_of::<BlockQAWQ_G64>(),
            Self::Q4_KS => std::mem::size_of::<BlockQ4_KS>(),
            Self::Q8_KS => std::mem::size_of::<BlockQ8_KS>(),
            Self::Q2_0 => std::mem::size_of::<BlockQ2_0>(),
            Self::Q3_0 => std::mem::size_of::<BlockQ3_0>(),
            Self::R16 => std::mem::size_of::<BlockR16>(),
            Self::Q0 => std::mem::size_of::<BlockQ0>(),
            Self::Q1_S => std::mem::size_of::<BlockQ1S>(),
            Self::Q2_S => std::mem::size_of::<BlockQ2S>(),
            Self::Q2_A => std::mem::size_of::<BlockQ2A>(),
            Self::Q2_1 => std::mem::size_of::<BlockQ2_1>(),
            Self::Q3_1 => std::mem::size_of::<BlockQ3_1>(),
            Self::P2 => std::mem::size_of::<BlockP2>(),
            Self::Q0_V => std::mem::size_of::<BlockQ0V>(),
            Self::Q1_A => std::mem::size_of::<BlockQ1A>(),
            Self::Q0_X => std::mem::size_of::<BlockQ0X>(),
            Self::Q0_M2 => std::mem::size_of::<BlockQ0M2>(),
            Self::Q0_M4 => std::mem::size_of::<BlockQ0M4>(),
            // KO compact blocks: same byte size as their K counterpart, just reordered.
            // KO twins are GPU-only lane-major chunks sized by `ko_quant::ko_chunk_bytes`
            // (128 elems/block, ko_chunk_bytes/8 bytes/block) — NOT the CPU `BlockQ*_KO`
            // struct layout. `type_size` is what the GGUF header uses to lay out and slice
            // tensor data (offset accounting on write, length on read), so it MUST equal the
            // bytes `quantize_ko` actually emits or offsets drift (past-EOF reads). Mirrors
            // MXFP4_KO returning `72` (= 576/8) directly rather than a Block struct size.
            Self::Q4_KO => 68,  // ko_chunk_bytes 544 / 8
            Self::Q5_KO => 84,  // ko_chunk_bytes 672 / 8
            Self::Q6_KO => 100, // ko_chunk_bytes 800 / 8
            Self::Q8_KO => 132, // ko_chunk_bytes 1056 / 8
            Self::MXFP4 => std::mem::size_of::<BlockMXFP4>(),
            // 576-byte GPU chunk per 1024 elements (512 nibbles + 32 E8M0 + 32 dm) → 72 B
            // per 128, keeping bytes == n_blocks × type_size for the [N,K] storage.
            Self::MXFP4_KO => 72,
            // 288-byte GPU chunk per 1024 elements (256 B of 2-bit crumbs + 32 dm) → 36 B per
            // 128. `ko_chunk_bytes(Q2_KO) = 288`.
            Self::Q2_KO => 36,
            Self::Q3_KO => 52, // ko_chunk_bytes 416 / 8
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F64 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::F8E4M3 => 1,
            Self::F8E5M2 => 1,
            Self::U8 => 1,
            Self::I8 => 1,
            Self::U16 => 1,
            Self::I16 => 1,
            Self::U32 => 1,
            Self::I32 => 1,
            Self::U64 => 1,
            Self::I64 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => {
                k_quants::QK_K
            }
            Self::QAWQ | Self::QAWQ_G64 => k_quants::QK_AWQ,
            Self::Q4_KS => k_quants::QK_Q4_KS,
            Self::Q8_KS => k_quants::QK_Q8_KS,
            Self::Q2_0 => k_quants::QK2_0,
            Self::Q3_0 => k_quants::QK3_0,
            Self::R16 => k_quants::QK_R16,
            Self::Q0 => k_quants::QK_Q0,
            Self::Q1_S => k_quants::QK1_S,
            Self::Q2_S => k_quants::QK2_S,
            Self::Q2_A => k_quants::QK2_A,
            Self::Q2_1 => k_quants::QK2_1,
            Self::Q3_1 => k_quants::QK3_1,
            Self::P2 => k_quants::QK_P2,
            Self::Q0_V => k_quants::QK_Q0_V,
            Self::Q1_A => k_quants::QK1_A,
            Self::Q0_X => k_quants::QK_Q0_X,
            Self::Q0_M2 => k_quants::QK_Q0_M2,
            Self::Q0_M4 => k_quants::QK_Q0_M4,
            // KO compact blocks hold 128 elements (K/128 granularity).
            Self::Q4_KO => BlockQ4_KO::BLCK_SIZE,
            Self::Q5_KO => BlockQ5_KO::BLCK_SIZE,
            Self::Q6_KO => BlockQ6_KO::BLCK_SIZE,
            Self::Q8_KO => BlockQ8_KO::BLCK_SIZE,
            Self::MXFP4 => k_quants::QK_MXFP4,
            // K/128 granularity (the collapse is per 128-K tile), like the other KO twins.
            Self::MXFP4_KO => 128,
            // K/128 granularity like the other KO twins (per-128 affine).
            Self::Q2_KO => 128,
            Self::Q3_KO => 128,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    fn size(&self) -> usize;
    fn clone_box(&self) -> Box<dyn QuantizedType>;
}

impl<T: k_quants::GgmlType + Send + Sync + Clone + 'static> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }

    fn clone_box(&self) -> Box<dyn QuantizedType> {
        Box::new(self.clone())
    }
}
