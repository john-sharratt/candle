//! A pack whose matmul weights are stored as their KO twins, so a loader
//! copies every tensor to the card as it is.
//!
//! A load that builds its twins — upload the source, repack on the device,
//! drop the source — pays a scratch buffer and a kernel per weight on every
//! load, for bytes that come out the same every time. A twinned pack does that
//! once, at build, on the host (`prepare::write_ko_twin`, the converter the
//! offline prepare step uses): each tensor [`ko_target`] gives a twin at the
//! pack's mode is written as that twin, and the GGUF reader uploads KO bytes
//! with no reinterpretation (`ggml_file::qtensor_from_ggml`).
//!
//! The bytes are this build's converter's, so the pack records each
//! `(source → twin)` pair's fingerprint (`repack_fingerprint`) and a pack whose
//! pairs no longer reproduce is stale — the same rule the repacked sections
//! keep.

use super::compose::Composition;
use super::keys::TWINS;
use super::prints::{from_metadata, to_metadata};
use crate::models::repack_fingerprint::{check_prints, pair_fingerprint_with, PairPrint};
use candle::quantized::gguf_file::Value;
use candle::quantized::prepare::{ko_target, write_ko_twin};
use candle::quantized::{GgmlDType, Int8Mode};
use candle::{Result, Shape};
use std::collections::HashMap;

/// Store every tensor of `composition` that has a KO twin at `mode` as that
/// twin — all but those named in `verbatim`, tensors the model reads densely
/// rather than through a matmul — and record the fingerprint of each pair used.
pub fn twin_tensors(composition: &mut Composition, mode: Int8Mode, verbatim: &[&str]) {
    let mut pairs: Vec<(GgmlDType, GgmlDType)> = Vec::new();
    for t in &mut composition.tensors {
        if verbatim.contains(&t.name.as_str()) {
            continue;
        }
        if let Some(ko) = ko_target(&t.name, &Shape::from(t.dims.clone()), t.from, mode) {
            t.dtype = ko;
            if !pairs.contains(&(t.from, ko)) {
                pairs.push((t.from, ko));
            }
        }
    }
    pairs.sort_by_key(|&(s, d)| (s.to_u32(), d.to_u32()));
    let prints: Vec<PairPrint> = pairs
        .into_iter()
        .map(|(src_dtype, dtype)| PairPrint {
            src_dtype,
            dtype,
            fp: twin_fingerprint(src_dtype, dtype),
        })
        .collect();
    for (k, v) in to_metadata(TWINS, &prints) {
        composition.set_metadata(&k, v);
    }
}

/// The fingerprint of `src → ko` as this build's host converter produces it.
fn twin_fingerprint(src: GgmlDType, ko: GgmlDType) -> u64 {
    pair_fingerprint_with(src, ko, |bytes, rows, cols| {
        let mut out = Vec::new();
        write_ko_twin(bytes, &[rows, cols], src, ko, &mut out)?;
        Ok(out)
    })
}

/// Refuse a twinned pack whose twins this build's converter would not write:
/// one that records no pairs, or a pair whose fingerprint moved.
pub fn check_twins(metadata: &HashMap<String, Value>) -> Result<()> {
    check_prints(&from_metadata(TWINS, metadata)?, twin_fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::model_pack::compose::{Origin, PackedTensor};

    fn tensor(name: &str, from: GgmlDType, dims: &[usize]) -> PackedTensor {
        PackedTensor {
            name: name.into(),
            origin: Origin::Source(0),
            dtype: from,
            from,
            dims: dims.to_vec(),
            offset: 0,
        }
    }

    fn composition() -> Composition {
        Composition {
            metadata: Vec::new(),
            tensors: vec![
                tensor("v.blk.0.attn_qkv.weight", GgmlDType::F16, &[3456, 1152]),
                tensor("v.blk.0.ffn_up.weight", GgmlDType::F16, &[4304, 1152]),
                tensor("v.blk.0.ln1.weight", GgmlDType::F32, &[1152]),
                tensor("v.position_embd.weight", GgmlDType::F16, &[2304, 1152]),
            ],
        }
    }

    fn dtypes(c: &Composition) -> Vec<(GgmlDType, GgmlDType)> {
        c.tensors.iter().map(|t| (t.from, t.dtype)).collect()
    }

    /// **A tileable projection is stored as its twin**; one whose rows do not
    /// tile (4 304 % 32), a vector, and a table named verbatim keep their
    /// file's type — and the one pair used is recorded with its fingerprint.
    #[test]
    fn tileable_projections_are_twinned_and_the_pair_recorded() {
        let mut c = composition();
        twin_tensors(&mut c, Int8Mode::Precision, &["v.position_embd.weight"]);
        use GgmlDType::{F16, F32, Q8_KO};
        assert_eq!(
            dtypes(&c),
            [(F16, Q8_KO), (F16, F16), (F32, F32), (F16, F16)]
        );
        assert_eq!(c.tensors[0].byte_len(), 3456 / 8 * (1152 / 128) * 1056);
        assert_eq!(c.tensors[0].source_len(), 3456 * 1152 * 2);

        let m: HashMap<String, Value> = c.metadata.into_iter().collect();
        assert_eq!(m["zen.pack.twins.count"].to_u32().unwrap(), 1);
        assert_eq!(m["zen.pack.twins.0.src"].to_u32().unwrap(), F16.to_u32());
        assert_eq!(m["zen.pack.twins.0.dst"].to_u32().unwrap(), Q8_KO.to_u32());
        assert!(check_twins(&m).is_ok());
    }

    /// A mode with no int8 matmul twins nothing, and records that it did not.
    #[test]
    fn an_off_pack_holds_its_files_types() {
        let mut c = composition();
        twin_tensors(&mut c, Int8Mode::Off, &[]);
        assert!(c.tensors.iter().all(|t| t.dtype == t.from));
        let m: HashMap<String, Value> = c.metadata.into_iter().collect();
        assert_eq!(m["zen.pack.twins.count"].to_u32().unwrap(), 0);
        assert!(check_twins(&m).is_ok());
    }

    /// A recorded pair whose fingerprint this build does not reproduce is
    /// refused by name, and a pack recording no twin table is refused.
    #[test]
    fn a_moved_fingerprint_or_a_missing_table_is_refused() {
        let mut c = composition();
        twin_tensors(&mut c, Int8Mode::Precision, &[]);
        let mut m: HashMap<String, Value> = c.metadata.into_iter().collect();
        let fp = m["zen.pack.twins.0.fp"].to_u64().unwrap();
        m.insert("zen.pack.twins.0.fp".into(), Value::U64(fp ^ 1));
        let e = check_twins(&m).unwrap_err().to_string();
        assert!(e.contains("F16"), "{e}");
        assert!(check_twins(&HashMap::new()).is_err());
    }
}
