//! What the claim kernel needs to know about a format, by select code
//! (`docs/kv_quantize_in_the_gap.md` §3.2).
//!
//! The selection writes each band's format as a select code. The claim turns a
//! code into the size class whose leased slot the band takes, and into the
//! format byte the destination record carries. Both are
//! tables the host builds once per geometry and uploads with the gap, so the
//! device never needs the format ladder.

use candle::quantized::cuda::{ggml_dtype_to_arena_fmt_code, ggml_to_select_qtype};
use candle::Result;

use candle::cuda_backend::kernels::simple::kv_gap_claim::N_CODES;

use super::super::size_class::{all_kv_formats, class_for_format, SizeClass};
use crate::kv_cache::KvFormat;

/// The class a code with no format at this geometry maps to.
pub(crate) const NO_CLASS: i8 = -1;

/// Per select code: the size class its bands claim from, and the record format
/// byte its destination record carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodeTables {
    pub class: [i8; N_CODES],
    pub fmt: [u8; N_CODES],
}

impl CodeTables {
    /// The tables for chunks of `elems_per_chunk` elements per band: every
    /// quantized format the ladder covers, at its select code.
    pub(crate) fn new(elems_per_chunk: usize) -> Result<Self> {
        if SizeClass::COUNT > i8::MAX as usize {
            candle::bail!("gap codes: {} size classes do not fit i8", SizeClass::COUNT);
        }
        let mut class = [NO_CLASS; N_CODES];
        let mut fmt = [0u8; N_CODES];
        for format in all_kv_formats() {
            let KvFormat::Quantized(q) = format else {
                continue;
            };
            let ggml = q.to_ggml_dtype();
            // Formats the selection cannot name have no code and no row.
            let Ok(code) = ggml_to_select_qtype(ggml) else {
                continue;
            };
            let Some(c) = class_for_format(format, elems_per_chunk) else {
                continue;
            };
            let code = usize::try_from(code)
                .ok()
                .filter(|&c| c < N_CODES)
                .ok_or_else(|| {
                    candle::Error::Msg(format!("gap codes: {q:?} has select code {code}"))
                })?;
            let c = c.index() as i8;
            if class[code] != NO_CLASS && class[code] != c {
                candle::bail!("gap codes: select code {code} maps to two classes");
            }
            class[code] = c;
            fmt[code] = ggml_dtype_to_arena_fmt_code(ggml)?;
        }
        Ok(Self { class, fmt })
    }

    /// Refuse codes the tables do not cover. Run over a policy's candidates
    /// and overrides before a launch, so the claim never meets a code it
    /// cannot place.
    pub(crate) fn check_covers(&self, codes: &[i32]) -> Result<()> {
        for &code in codes {
            let covered = usize::try_from(code)
                .ok()
                .and_then(|c| self.class.get(c))
                .is_some_and(|&c| c != NO_CLASS);
            if !covered {
                candle::bail!("gap codes: select code {code} has no size class here");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle::quantized::GgmlDType;

    use super::*;
    use crate::kv_cache::QuantFormat;

    /// Production geometry: head_dim 128, four palettes, 32 tokens a chunk.
    const EPC: usize = 32 * 32;

    fn code(d: GgmlDType) -> usize {
        ggml_to_select_qtype(d).unwrap() as usize
    }

    #[test]
    fn a_code_carries_its_formats_class_and_record_byte() {
        let t = CodeTables::new(EPC).unwrap();
        for (q, byte) in [
            (QuantFormat::Q8_0, 7u8),
            (QuantFormat::Q4_0, 15),
            (QuantFormat::R16, 3),
        ] {
            let c = code(q.to_ggml_dtype());
            let class = class_for_format(KvFormat::Quantized(q), EPC).unwrap();
            assert_eq!(t.class[c], class.index() as i8, "{q:?}");
            assert_eq!(t.fmt[c], byte, "{q:?}");
        }
    }

    #[test]
    fn the_float_codes_have_no_class() {
        let t = CodeTables::new(EPC).unwrap();
        assert_eq!(t.class[0], NO_CLASS);
        assert!(t.check_covers(&[0]).is_err());
    }

    #[test]
    fn coverage_is_checked_per_code() {
        let t = CodeTables::new(EPC).unwrap();
        let q8 = code(GgmlDType::Q8_0) as i32;
        assert!(t.check_covers(&[q8]).is_ok());
        assert!(t.check_covers(&[q8, -1]).is_err());
        assert!(t.check_covers(&[q8, N_CODES as i32]).is_err());
    }
}
