//! Which KV threshold row an engine runs under, by its experts' format.
//!
//! The rows are calibrated per expert format (`docs/qwen38_flash_next.md`,
//! Phase 6): narrower experts leave the model less margin for K/V error, so the
//! C10 edge a row is placed against moves with them.

use candle::quantized::GgmlDType;
use candle_nn::kv_cache::{KvErrorThresholdFactors, QWEN4EXP_KV_FACTORS, QWEN4EXP_Q2KO_KV_FACTORS};

/// The threshold row for experts stored as `format`.
///
/// `Q2_KO` has its own measured row. `Q3_KO` — the 32 GiB rung, calibrated on no
/// card of that size — takes the tighter `Q2_KO` row: narrower experts have
/// less margin, so the narrow row costs `Q3_KO` some ratio and cannot cost it
/// validity. Every wider format runs the row derived on `Q4_KO`.
pub fn kv_factors_for(format: GgmlDType) -> KvErrorThresholdFactors {
    match format {
        GgmlDType::Q2_KO | GgmlDType::Q3_KO => QWEN4EXP_Q2KO_KV_FACTORS,
        _ => QWEN4EXP_KV_FACTORS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(f: KvErrorThresholdFactors) -> [f32; 4] {
        [f.k_hi, f.k_low, f.v_hi, f.v_low]
    }

    #[test]
    fn each_format_takes_its_row() {
        assert_eq!(
            row(kv_factors_for(GgmlDType::Q2_KO)),
            row(QWEN4EXP_Q2KO_KV_FACTORS)
        );
        assert_eq!(
            row(kv_factors_for(GgmlDType::Q3_KO)),
            row(QWEN4EXP_Q2KO_KV_FACTORS)
        );
        for f in [GgmlDType::Q4_KO, GgmlDType::Q8_0] {
            assert_eq!(row(kv_factors_for(f)), row(QWEN4EXP_KV_FACTORS), "{f:?}");
        }
    }

    /// The narrow rung's row is never looser than the wide one on any axis —
    /// narrower experts have less margin, not more.
    #[test]
    fn the_narrow_row_is_no_looser() {
        let (narrow, wide) = (row(QWEN4EXP_Q2KO_KV_FACTORS), row(QWEN4EXP_KV_FACTORS));
        for (n, w) in narrow.iter().zip(wide) {
            assert!(*n <= w, "{narrow:?} is looser than {wide:?}");
        }
    }
}
