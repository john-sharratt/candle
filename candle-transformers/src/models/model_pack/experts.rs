//! A checkpoint's routed experts, located for the build.
//!
//! Every routed family here stores an MoE block's experts as three merged
//! `[n_expert, out, in]` tensors — gate, up, down — so an expert's bytes are a
//! fixed stride into each. The families differ only in the tensors' names,
//! which [`ExpertNames`] supplies: the qwen lineage's convention, or a
//! sparse-latent model's own [`Arch`].

use crate::models::expert_lre::MmapExpertRef;
use crate::models::latent_moe::arch::Ffn;
use crate::models::latent_moe::{block_count, Arch, Weight};
use candle::quantized::gguf_file::{Content, TensorInfo};
use candle::Result;

/// The three merged expert tensors of one MoE block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertNames {
    /// The block's index in the checkpoint (`blk.{n}`).
    pub block: u32,
    pub gate: String,
    pub up: String,
    pub down: String,
}

impl ExpertNames {
    /// The qwen lineage's names: `blk.{n}.ffn_{gate,up,down}_exps.weight`.
    pub fn qwen(block: u32) -> Self {
        let p = format!("blk.{block}");
        Self {
            block,
            gate: format!("{p}.ffn_gate_exps.weight"),
            up: format!("{p}.ffn_up_exps.weight"),
            down: format!("{p}.ffn_down_exps.weight"),
        }
    }

    pub fn all(&self) -> [&str; 3] {
        [&self.gate, &self.up, &self.down]
    }
}

/// Every block of `content` carrying the qwen lineage's merged experts, in
/// block order — the order the loaders number MoE layers in.
pub fn qwen_expert_blocks(content: &Content) -> Vec<ExpertNames> {
    let mut blocks: Vec<u32> = content
        .tensor_infos
        .keys()
        .filter_map(|k| {
            k.strip_prefix("blk.")?
                .strip_suffix(".ffn_gate_exps.weight")?
                .parse()
                .ok()
        })
        .collect();
    blocks.sort_unstable();
    blocks.into_iter().map(ExpertNames::qwen).collect()
}

/// Every block of `content` carrying `arch`'s routed experts, in block order,
/// every name taken from the arch.
///
/// The sparse-latent family names every tensor through its [`Arch`] and never by
/// another model's convention, so its build does the same: the blocks are the
/// model's own count ([`block_count`], as its loader reads it), and a block is
/// routed when the arch's gate tensor for it is in the file.
pub fn latent_expert_blocks(content: &Content, arch: &dyn Arch) -> Vec<ExpertNames> {
    (0..block_count(&content.metadata, arch))
        .filter_map(|layer| {
            let name = |f: Ffn| arch.weight(layer, Weight::RoutedExperts(f));
            let gate = name(Ffn::Gate);
            content
                .tensor_infos
                .contains_key(&gate)
                .then(|| ExpertNames {
                    block: layer as u32,
                    gate,
                    up: name(Ffn::Up),
                    down: name(Ffn::Down),
                })
        })
        .collect()
}

/// Per-expert references into `content`'s file for each of `layers`, and the
/// expert count they agree on.
///
/// Refuses a block missing one of its three tensors, a tensor that is not the
/// merged 3-D form, and blocks that disagree on how many experts they hold — a
/// stride walked over the wrong count lands in the wrong expert.
pub fn expert_refs(
    content: &Content,
    layers: &[ExpertNames],
) -> Result<(Vec<Vec<MmapExpertRef>>, usize)> {
    let mut all = Vec::with_capacity(layers.len());
    let mut n_expert = None;
    for names in layers {
        let info = |name: &str| -> Result<&TensorInfo> {
            content.tensor_infos.get(name).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "model pack: block {} has no {name} — a partial expert set cannot be packed",
                    names.block
                ))
            })
        };
        let (g, u, d) = (info(&names.gate)?, info(&names.up)?, info(&names.down)?);
        for (name, t) in names.all().into_iter().zip([g, u, d]) {
            let dims = t.shape.dims();
            if dims.len() != 3 {
                candle::bail!(
                    "model pack: {name} is {}-D — experts are packed from the merged \
                     [n_expert, out, in] form only",
                    dims.len()
                );
            }
            match n_expert {
                None => n_expert = Some(dims[0]),
                Some(n) if n != dims[0] => candle::bail!(
                    "model pack: {name} holds {} experts where earlier blocks hold {n}",
                    dims[0]
                ),
                Some(_) => {}
            }
        }
        let stride = |t: &TensorInfo| -> usize {
            let elems: usize = t.shape.dims()[1..].iter().product();
            elems / t.ggml_dtype.block_size() * t.ggml_dtype.type_size()
        };
        let base = |t: &TensorInfo| (content.tensor_data_offset + t.offset) as usize;
        let (gs, us, ds) = (stride(g), stride(u), stride(d));
        let n = n_expert.unwrap_or(0);
        all.push(
            (0..n)
                .map(|e| MmapExpertRef {
                    gate_offset: base(g) + e * gs,
                    gate_len: gs,
                    up_offset: base(u) + e * us,
                    up_len: us,
                    down_offset: base(d) + e * ds,
                    down_len: ds,
                    gate_shape: g.shape.dims()[1..].to_vec(),
                    up_shape: u.shape.dims()[1..].to_vec(),
                    down_shape: d.shape.dims()[1..].to_vec(),
                    gate_dtype: g.ggml_dtype,
                    up_dtype: u.ggml_dtype,
                    down_dtype: d.ggml_dtype,
                })
                .collect(),
        );
    }
    Ok((all, n_expert.unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::VersionedMagic;
    use candle::quantized::GgmlDType;
    use candle::Shape;
    use std::collections::HashMap;

    fn content(tensors: &[(&str, Vec<usize>)]) -> Content {
        let mut infos = HashMap::new();
        let mut at = 0u64;
        for (name, dims) in tensors {
            infos.insert(
                (*name).to_string(),
                TensorInfo {
                    ggml_dtype: GgmlDType::Q8_0,
                    shape: Shape::from(dims.clone()),
                    offset: at,
                },
            );
            at += 1 << 20;
        }
        Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos: infos,
            tensor_data_offset: 4096,
        }
    }

    fn block(b: u32, n: usize) -> Vec<(String, Vec<usize>)> {
        let names = ExpertNames::qwen(b);
        vec![
            (names.gate.clone(), vec![n, 64, 128]),
            (names.up.clone(), vec![n, 64, 128]),
            (names.down.clone(), vec![n, 128, 64]),
        ]
    }

    fn owned(v: &[(String, Vec<usize>)]) -> Vec<(&str, Vec<usize>)> {
        v.iter().map(|(n, d)| (n.as_str(), d.clone())).collect()
    }

    /// Blocks are found by their gate tensor and returned in block order, not
    /// the hash map's — `blk.10` after `blk.9`.
    #[test]
    fn blocks_come_back_in_numeric_order() {
        let mut t = block(10, 4);
        t.extend(block(9, 4));
        t.extend(block(2, 4));
        let c = content(&owned(&t));
        let blocks: Vec<u32> = qwen_expert_blocks(&c).iter().map(|n| n.block).collect();
        assert_eq!(blocks, [2, 9, 10]);
    }

    /// Expert `e`'s projection is `e` strides into its merged tensor, a stride
    /// being one `[out, in]` slab of the tensor's dtype.
    #[test]
    fn an_experts_bytes_are_a_stride_into_its_tensor() {
        let t = block(0, 4);
        let c = content(&owned(&t));
        let (refs, n) = expert_refs(&c, &qwen_expert_blocks(&c)).unwrap();
        assert_eq!(n, 4);
        let slab = 64 * 128 / 32 * 34; // Q8_0: 34 bytes per 32
        let gate_at = (c.tensor_data_offset + c.tensor_infos[&t[0].0].offset) as usize;
        assert_eq!(refs[0][3].gate_offset, gate_at + 3 * slab);
        assert_eq!(refs[0][3].gate_len, slab);
        assert_eq!(refs[0][3].down_shape, [128, 64]);
    }

    #[test]
    fn blocks_disagreeing_on_expert_count_are_refused() {
        let mut t = block(0, 4);
        t.extend(block(1, 8));
        let c = content(&owned(&t));
        let e = expert_refs(&c, &qwen_expert_blocks(&c))
            .unwrap_err()
            .to_string();
        assert!(e.contains("holds 8 experts"), "{e}");
    }

    /// A sparse-latent model's experts are found under its own arch's names —
    /// the test arch's are deliberately unlike the qwen lineage's, so a build
    /// that fell back to `blk.{n}.ffn_*_exps` would find nothing here.
    #[test]
    fn latent_experts_are_named_through_the_arch() {
        use crate::models::latent_moe::arch::test_arch::TEST_ARCH;
        let n_layers = TEST_ARCH.defaults().n_layers;
        assert!(n_layers >= 3, "the test arch has {n_layers} layers");
        let routed = [1usize, n_layers - 1];
        let mut t: Vec<(String, Vec<usize>)> = Vec::new();
        for &layer in &routed {
            for (f, dims) in [
                (Ffn::Gate, vec![4, 64, 128]),
                (Ffn::Up, vec![4, 64, 128]),
                (Ffn::Down, vec![4, 128, 64]),
            ] {
                t.push((TEST_ARCH.weight(layer, Weight::RoutedExperts(f)), dims));
            }
        }
        let c = content(&owned(&t));
        assert!(qwen_expert_blocks(&c).is_empty());
        let blocks = latent_expert_blocks(&c, &TEST_ARCH);
        let got: Vec<u32> = blocks.iter().map(|b| b.block).collect();
        assert_eq!(got, [1, n_layers as u32 - 1]);
        assert_eq!(
            blocks[0].gate,
            TEST_ARCH.weight(1, Weight::RoutedExperts(Ffn::Gate))
        );
        let (_, n) = expert_refs(&c, &blocks).unwrap();
        assert_eq!(n, 4);
    }

    #[test]
    fn a_partial_block_is_refused() {
        let mut t = block(0, 4);
        t.pop();
        let c = content(&owned(&t));
        let e = expert_refs(&c, &qwen_expert_blocks(&c))
            .unwrap_err()
            .to_string();
        assert!(e.contains("partial expert set"), "{e}");
    }
}
