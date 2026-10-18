//! What the dense block holds, by kind of weight, read off the pack's tensor table.
//!
//! The dense block is a bump allocator with no ledger: `claim_dense` hands out
//! ground and keeps only the total. Every tensor the load reads to the device is
//! there — all but the n-gram table, which a host row cache serves, and the routed
//! experts, which the expert cache streams — so the file says what fills it. The
//! figures are the file's bytes; the block also holds each weight's matmul row
//! padding and the KO twins' tiling, which is what the measured block exceeds this
//! by.

use std::collections::BTreeMap;

use candle::quantized::gguf_file::TensorInfo;

/// The n-gram (PLE) table, which never reaches the device.
const PLE_TABLE: &str = "per_layer_token_embd.weight";

/// One kind of weight's share of the dense block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DenseShare {
    pub tensors: usize,
    pub bytes: usize,
}

/// The kind of weight `name` is, or `None` for one that is not device-resident.
pub fn dense_kind(name: &str) -> Option<&'static str> {
    if name == PLE_TABLE || name.ends_with("_exps.weight") {
        return None;
    }
    let kind = if name == "token_embd.weight" {
        "token embedding"
    } else if name.starts_with("output") {
        "LM head"
    } else if name.contains("shexp") {
        "shared expert"
    } else if name.contains("ffn_gate_inp") || name.contains("exp_probs") {
        "router"
    } else if name.contains("nextn") || name.contains("mtp") {
        "MTP head"
    } else if name.contains("hc_") {
        "HC"
    } else if name.contains("indexer") {
        "QSA indexer"
    } else if name.contains("ssm") || name.contains("attn_qkv") || name.contains("attn_gate") {
        // A DeltaNet layer's input projections are named `attn_qkv` and
        // `attn_gate`; a full-attention layer's are `attn_q`, `attn_k`, `attn_v`.
        "DeltaNet mixer"
    } else if name.contains("attn") {
        "full attention"
    } else if name.contains("norm") {
        "norms"
    } else if name.contains("per_layer") || name.contains("ple") {
        "PLE projections"
    } else {
        "other"
    };
    Some(kind)
}

/// Every device-resident tensor's bytes in the file, by kind.
pub fn dense_inventory<'a>(
    tensors: impl IntoIterator<Item = (&'a String, &'a TensorInfo)>,
) -> BTreeMap<&'static str, DenseShare> {
    let mut out: BTreeMap<&'static str, DenseShare> = BTreeMap::new();
    for (name, info) in tensors {
        let Some(kind) = dense_kind(name) else {
            continue;
        };
        let bytes =
            info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let share = out.entry(kind).or_default();
        share.tensors += 1;
        share.bytes += bytes;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::GgmlDType;
    use std::cmp::Reverse;
    use std::collections::HashMap;

    fn info(dtype: GgmlDType, dims: &[usize]) -> TensorInfo {
        TensorInfo {
            ggml_dtype: dtype,
            shape: dims.into(),
            offset: 0,
        }
    }

    /// The table is split by kind, and the two tensors the device never holds —
    /// the n-gram table and the routed experts — are left out.
    #[test]
    fn the_inventory_sums_by_kind_and_skips_what_the_device_never_holds() {
        let t: HashMap<String, TensorInfo> = [
            ("token_embd.weight", info(GgmlDType::Q8_0, &[8, 64])),
            ("blk.0.attn_q.weight", info(GgmlDType::Q8_0, &[4, 64])),
            ("blk.1.attn_q.weight", info(GgmlDType::Q8_0, &[4, 64])),
            (
                "blk.0.ffn_down_exps.weight",
                info(GgmlDType::Q8_0, &[64, 64]),
            ),
            (PLE_TABLE, info(GgmlDType::Q8_0, &[1000, 64])),
        ]
        .into_iter()
        .map(|(n, i)| (n.to_string(), i))
        .collect();
        let inv = dense_inventory(&t);
        // Q8_0: 32 elements in 34 bytes.
        assert_eq!(
            inv["token embedding"],
            DenseShare {
                tensors: 1,
                bytes: 16 * 34
            }
        );
        assert_eq!(
            inv["full attention"],
            DenseShare {
                tensors: 2,
                bytes: 2 * 8 * 34
            }
        );
        assert_eq!(
            inv.len(),
            2,
            "experts and the n-gram table are not resident"
        );
    }

    /// The Flash-Next pack's dense inventory, by kind, with every tensor the
    /// table could not name listed so the kinds can be extended.
    #[test]
    #[ignore = "probe: cargo test -p candle-transformers --lib qwen4exp::dense_inventory -- --ignored --nocapture"]
    fn flash_next_dense_inventory() -> candle::Result<()> {
        use crate::models::model_pack::cache_root;
        use candle::quantized::gguf_file::Content;

        let dir = cache_root().join("unsloth--Qwen3.8-Flash-Next-GGUF");
        let pack = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(".pack.gguf"))
            })
            .ok_or_else(|| candle::Error::Msg(format!("no pack in {dir:?}")))?;
        let mut file = std::fs::File::open(&pack)?;
        let content = Content::read(&mut file)?;
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        println!("{}", pack.display());
        let inv = dense_inventory(&content.tensor_infos);
        let total: usize = inv.values().map(|s| s.bytes).sum();
        for (kind, s) in &inv {
            println!(
                "  {kind:<18} {:>5} tensors {:>9.1} MiB",
                s.tensors,
                mib(s.bytes)
            );
        }
        println!(
            "  {:<18} {:>5}         {:>9.1} MiB",
            "total",
            "",
            mib(total)
        );
        // By tensor, with the layer index taken out: what each kind is made of.
        let mut by_suffix: BTreeMap<(String, String), DenseShare> = BTreeMap::new();
        for (name, info) in &content.tensor_infos {
            if dense_kind(name).is_none() {
                continue;
            }
            let suffix = match name.strip_prefix("blk.") {
                Some(rest) => rest.split_once('.').map_or(rest, |(_, s)| s).to_string(),
                None => name.clone(),
            };
            let bytes = info.shape.elem_count() / info.ggml_dtype.block_size()
                * info.ggml_dtype.type_size();
            let share = by_suffix
                .entry((suffix, format!("{:?}", info.ggml_dtype)))
                .or_default();
            share.tensors += 1;
            share.bytes += bytes;
        }
        let mut rows: Vec<_> = by_suffix.into_iter().collect();
        rows.sort_by_key(|r| Reverse(r.1.bytes));
        for ((suffix, dtype), s) in rows {
            println!(
                "    {suffix:<36} {dtype:<6} ×{:<3} {:>8.1} MiB",
                s.tensors,
                mib(s.bytes)
            );
        }
        Ok(())
    }
}
