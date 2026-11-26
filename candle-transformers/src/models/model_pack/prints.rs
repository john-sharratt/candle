//! A table of repack fingerprints in a pack's metadata.
//!
//! A pack that stores weights some converter produced records, for each
//! `(source → stored)` pair it used, the fingerprint of that converter
//! (`repack_fingerprint`), so a pack whose pairs no longer reproduce is stale.
//! Each table lives under a base key of its own — one per converter, since two
//! converters' fingerprints of one pair are different numbers:
//! `{base}.count`, then `{base}.{i}.{src,dst,fp}` for each pair, the two dtypes
//! as `GgmlDType::to_u32`.

use crate::models::repack_fingerprint::PairPrint;
use candle::quantized::gguf_file::Value;
use candle::quantized::GgmlDType;
use candle::Result;
use std::collections::HashMap;

/// `prints` as metadata entries under `base`.
pub(crate) fn to_metadata(base: &str, prints: &[PairPrint]) -> Vec<(String, Value)> {
    let mut out = vec![(format!("{base}.count"), Value::U32(prints.len() as u32))];
    for (i, p) in prints.iter().enumerate() {
        out.push((format!("{base}.{i}.src"), Value::U32(p.src_dtype.to_u32())));
        out.push((format!("{base}.{i}.dst"), Value::U32(p.dtype.to_u32())));
        out.push((format!("{base}.{i}.fp"), Value::U64(p.fp)));
    }
    out
}

/// The table under `base` in `m`; an error when it is absent or partial.
pub(crate) fn from_metadata(base: &str, m: &HashMap<String, Value>) -> Result<Vec<PairPrint>> {
    let get = |key: String| {
        m.get(&key)
            .ok_or_else(|| candle::Error::Msg(format!("model pack: no {key}")))
    };
    let count = get(format!("{base}.count"))?.to_u32()? as usize;
    (0..count)
        .map(|i| {
            Ok(PairPrint {
                src_dtype: GgmlDType::from_u32(get(format!("{base}.{i}.src"))?.to_u32()?)?,
                dtype: GgmlDType::from_u32(get(format!("{base}.{i}.dst"))?.to_u32()?)?,
                fp: get(format!("{base}.{i}.fp"))?.to_u64()?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table writes as its count and three keys a pair, and reads back.
    #[test]
    fn a_table_round_trips_under_its_base() {
        let prints = [
            PairPrint {
                src_dtype: GgmlDType::Q8_0,
                dtype: GgmlDType::Q8_KO,
                fp: 0x0123_4567_89ab_cdef,
            },
            PairPrint {
                src_dtype: GgmlDType::Q8_0,
                dtype: GgmlDType::Q4_KO,
                fp: 7,
            },
        ];
        let entries = to_metadata("zen.x", &prints);
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            [
                "zen.x.count",
                "zen.x.0.src",
                "zen.x.0.dst",
                "zen.x.0.fp",
                "zen.x.1.src",
                "zen.x.1.dst",
                "zen.x.1.fp"
            ]
        );
        let m: HashMap<String, Value> = entries.into_iter().collect();
        assert_eq!(from_metadata("zen.x", &m).unwrap(), prints);
        assert!(from_metadata("zen.y", &m).is_err());
    }
}
