//! Qwen3.8-Flash-Next — the routed (MoE) Qwen3.8's model file, named beside
//! its `quantized_qwen35_moe` / `quantized_qwen36_moe` siblings.
//!
//! HF `model_type: qwen4_exp`, GGUF arch `qwen4exp`: a 48-layer 3:1 hybrid
//! (36 Gated DeltaNet / 12 QSA attention) over a 4-stream gated residual,
//! 512-expert top-10 MoE on every layer, an n-gram PLE injection at layer 1,
//! and a host-resident 320M-row embedding table behind a bounded RAM row
//! cache. The family machinery lives in [`super::qwen4exp`]; this file pins
//! the checkpoints (the Q8_0 split and its W4A16-imported Q4_KO-expert
//! sibling) and holds the batched-forward gates. Design doc:
//! `docs/qwen38_flash_next.md` (§12 is the frozen schema this file loads
//! against).
//!
//! **Not** the model of `quantized_qwen38.rs` — that is the dense 27B of the
//! `qwen35` lineage. Both carry "Qwen3.8" branding; the `_moe` suffix is the
//! same split the 3.5/3.6 siblings use.

use std::path::PathBuf;

use candle::quantized::GgmlDType;
use candle::{Device, Result};

use super::quant_ladder;
use super::qwen4exp::prepare::{ExpertSource, Recipe, SourceFile, SourceRole};
use super::qwen4exp::{load_oracle_model, Qwen4ExpModel};

/// The tokenizer, pinned to the canonical base repo.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.8-Flash-Next";
pub const TOKENIZER_REV: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";

/// The bring-up checkpoint: unsloth's **plain Q8_0** split, revision-pinned.
///
/// Q8_0 rather than a Q4-class quant, and that is a measurement, not a
/// preference: both community sub-8-bit conversions (unsloth `UD-*`,
/// AtomicChat `AD-*`) carry IQ-family tensors this codebase cannot read —
/// verified from the shard headers, dtype codes 20/21, exactly the trap
/// `quantized_qwen38.rs` documents as "no rung is a `UD-` file". The §0.2
/// per-machine expert formats are produced by requantizing locally from this
/// file. Six shards; shard 1 is metadata-only, shard 3 is the isolated PLE
/// table.
///
/// The revision is the one that also carries the MTP draft head (`MTP/`). The
/// six `Q8_0` shards are byte-identical to the earlier `c8b5954a` pin — the same
/// LFS object ids — so moving the pin changed no weight.
pub const QWEN4EXP_REPO: &str = "unsloth/Qwen3.8-Flash-Next-GGUF";
pub const QWEN4EXP_REV: &str = "38bb39ee97821de2c9009abb7e93950eec396e66";
pub const QWEN4EXP_Q8_0_SHARDS: usize = 6;

/// The six `Q8_0` shards at [`QWEN4EXP_REV`]: `(path, bytes, LFS SHA-256)`.
pub const QWEN4EXP_Q8_0_FILES: [(&str, u64, &str); QWEN4EXP_Q8_0_SHARDS] = [
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
        10_946_624,
        "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf",
        682_434_912,
        "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf",
        54_400_261_312,
        "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf",
        49_446_841_216,
        "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf",
        49_668_930_400,
        "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf",
        34_015_618_784,
        "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5",
    ),
];

/// The MTP draft head at [`QWEN4EXP_REV`], at `Q8_0`: `(path, bytes, LFS SHA-256)`.
/// Its dense weights stay `Q8_0` in the engine artifact; its experts take the
/// trunk's width (`quant_ladder::drafter_format`).
pub const QWEN4EXP_MTP_FILE: (&str, u64, &str) = (
    "MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
    4_137_429_120,
    "cd87e5d1a4dadaeed63e35929f3b2f28d13e081b4cd32e00f2835095ec09351e",
);

/// Shard `i` (1-based) of the pinned Q8_0 split.
pub fn q8_0_shard_name(i: usize) -> String {
    format!("Q8_0/Qwen3.8-Flash-Next-Q8_0-{i:05}-of-{QWEN4EXP_Q8_0_SHARDS:05}.gguf")
}

/// The W4A16 source for the Q4_KO expert import (`qwen4exp::convert`):
/// compressed-tensors int4, group_size 128, symmetric — the same per-128
/// affine quantization Q4_KO stores, so the import is bit-exact and carries
/// the release's AWQ calibration into our resident format. Only shards 2–5
/// hold the quantized trunk; shard 1 is the BF16 PLE table + embeddings and
/// is never fetched.
pub const QWEN4EXP_W4A16_REPO: &str = "wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16";
pub const QWEN4EXP_W4A16_REV: &str = "0939125b929543a783ce700c90e36dd1a575c00c";
pub const QWEN4EXP_W4A16_EXPERT_SHARDS: [usize; 4] = [2, 3, 4, 5];

/// Safetensors shard `i` of the W4A16 release.
pub fn w4a16_shard_name(i: usize) -> String {
    format!("model-{i:05}-of-00005.safetensors")
}

/// The W4A16 expert shards ([`QWEN4EXP_W4A16_EXPERT_SHARDS`]) at
/// [`QWEN4EXP_W4A16_REV`]: `(path, bytes, LFS SHA-256)`.
pub const QWEN4EXP_W4A16_FILES: [(&str, u64, &str); 4] = [
    (
        "model-00002-of-00005.safetensors",
        20_007_503_112,
        "dcb830243a6f1f56f8849cdc829724383c3fa104ba793e7f879a19927d4e8a32",
    ),
    (
        "model-00003-of-00005.safetensors",
        19_960_933_248,
        "00b849022ecb3fae6c8d6cdde768da899abb47c25bfc44d87e8c931f11b124cb",
    ),
    (
        "model-00004-of-00005.safetensors",
        20_008_579_224,
        "af83800aeef9ee47e49fc0a9769f67617108934626e79a2e30051ef48f338a7d",
    ),
    (
        "model-00005-of-00005.safetensors",
        13_134_032_728,
        "b4e2d8c38b466774703b831ccbfc89f1c78e726d2296bb7eaaf2d3b31933fadd",
    ),
];

/// The version of the engine build's output bytes. Part of the recipe, so a
/// change to what the build writes for an unchanged set of sources names a new
/// artifact and every machine rebuilds.
pub const ENGINE_CONVERTER_VERSION: u32 = 1;

/// The engine artifact's recipe for a card of `vram_gib`.
///
/// The expert width is [`quant_ladder::expert_format`]'s: `Q4_KO` is the
/// bit-exact W4A16 import, a narrower KO rung is requantized from the `Q8_0`
/// split, and above every rung the split's own `Q8_0` stands. The trunk, the
/// n-gram table and the draft head's dense weights are `Q8_0` on every rung.
pub fn engine_recipe(vram_gib: u64) -> Recipe {
    engine_recipe_at(quant_ladder::expert_format(vram_gib))
}

/// The engine artifact's recipe with its routed experts at `experts` — the
/// rung named directly rather than through a card's VRAM, for a caller that
/// already knows which artifact it wants (a preset naming its expert width).
/// `None` leaves the experts at the split's own `Q8_0`.
pub fn engine_recipe_at(experts: Option<GgmlDType>) -> Recipe {
    let pinned =
        |role, repo, revision, (path, bytes, sha256): (&'static str, u64, &'static str)| {
            SourceFile {
                role,
                repo,
                revision,
                path,
                bytes,
                sha256,
            }
        };
    let mut sources: Vec<SourceFile> = QWEN4EXP_Q8_0_FILES
        .into_iter()
        .map(|f| pinned(SourceRole::Trunk, QWEN4EXP_REPO, QWEN4EXP_REV, f))
        .collect();
    sources.push(pinned(
        SourceRole::DraftHead,
        QWEN4EXP_REPO,
        QWEN4EXP_REV,
        QWEN4EXP_MTP_FILE,
    ));
    let (experts, expert_source) = match experts {
        Some(GgmlDType::Q4_KO) => {
            sources.extend(QWEN4EXP_W4A16_FILES.into_iter().map(|f| {
                pinned(
                    SourceRole::ExpertImport,
                    QWEN4EXP_W4A16_REPO,
                    QWEN4EXP_W4A16_REV,
                    f,
                )
            }));
            (GgmlDType::Q4_KO, ExpertSource::AwqImport)
        }
        Some(ko) => (ko, ExpertSource::Requantized),
        None => (GgmlDType::Q8_0, ExpertSource::Verbatim),
    };
    Recipe {
        sources,
        trunk: GgmlDType::Q8_0,
        head_dense: GgmlDType::Q8_0,
        experts,
        expert_source,
        // The drafter's experts take the trunk's width on every rung
        // (`quant_ladder::drafter_format`): slots are sized to the widest layer.
        head_experts: experts,
        converter_version: ENGINE_CONVERTER_VERSION,
    }
}

/// Where engine artifacts live: zend's model cache, under this repo's folder —
/// the directory `zend`'s prepared-artifact resolution reads.
pub fn engine_artifact_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("zend")
        .join("models")
        .join(QWEN4EXP_REPO.replace('/', "--"))
}

/// Load the reference (oracle) model from the pinned split's first shard
/// path. Refuses a non-`qwen4exp` file and validates the frozen geometry —
/// a silent architecture change fails here, not in a kernel.
pub fn oracle_from_gguf_path(
    first_shard: &std::path::Path,
    device: &Device,
) -> Result<Qwen4ExpModel> {
    let model = load_oracle_model(first_shard, device)?;
    let cfg = &model.cfg;
    if cfg.num_layers != 48
        || cfg.attn_head_dim != 256
        || cfg.hidden_size != 2560
        || cfg.moe.n_experts != 512
        || cfg.hc.count != 4
    {
        candle::bail!(
            "quantized_qwen38_moe: checkpoint geometry diverged from the frozen schema \
             (layers {}, head_dim {}, hidden {}, experts {}, hc {})",
            cfg.num_layers,
            cfg.attn_head_dim,
            cfg.hidden_size,
            cfg.moe.n_experts,
            cfg.hc.count
        );
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::batch_test::utils::{account_model_load, TestConfig, TestMode};
    use candle::quantized::gguf_file::Content;
    use candle::Tensor;
    use hf_hub::RepoType;
    use std::collections::HashSet;
    use std::fs::File;

    fn pinned_shards() -> Result<Vec<PathBuf>> {
        (1..=QWEN4EXP_Q8_0_SHARDS)
            .map(|i| {
                hf_get(
                    QWEN4EXP_REPO,
                    RepoType::Model,
                    QWEN4EXP_REV,
                    &q8_0_shard_name(i),
                )
            })
            .collect()
    }

    /// This card's engine artifact: resolved from zend's model cache, or built
    /// there from the pinned sources ([`prepare_engine_artifact`]).
    fn engine_gguf() -> Result<PathBuf> {
        use crate::models::batch_test::test_helpers::HfSourceStore;
        use crate::models::qwen4exp::prepare::prepare_engine;
        let device = Device::new_cuda(0)?;
        let recipe = engine_recipe(quant_ladder::device_vram_gib(&device)?);
        prepare_engine(&recipe, &engine_artifact_dir(), &HfSourceStore, &device)
    }

    /// The pin table and the shard-name function describe the same six files.
    #[test]
    fn the_q8_0_pins_name_the_split() {
        for (i, (path, _, sha)) in QWEN4EXP_Q8_0_FILES.iter().enumerate() {
            assert_eq!(*path, q8_0_shard_name(i + 1));
            assert_eq!(sha.len(), 64);
        }
        for (&i, (path, _, _)) in QWEN4EXP_W4A16_EXPERT_SHARDS
            .iter()
            .zip(QWEN4EXP_W4A16_FILES.iter())
        {
            assert_eq!(*path, w4a16_shard_name(i));
        }
    }

    /// Each card in the fleet gets the rung `quant_ladder` names, the sources
    /// that rung needs, and a distinct artifact.
    #[test]
    fn each_rung_has_its_own_recipe() {
        let laptop = engine_recipe(16);
        assert_eq!(laptop.experts, GgmlDType::Q2_KO);
        assert_eq!(laptop.head_experts, GgmlDType::Q2_KO);
        assert_eq!(laptop.expert_source, ExpertSource::Requantized);
        assert_eq!(laptop.trunk, GgmlDType::Q8_0);
        assert_eq!(laptop.head_dense, GgmlDType::Q8_0);
        assert_eq!(laptop.sources_of(SourceRole::Trunk).len(), 6);
        assert_eq!(laptop.sources_of(SourceRole::DraftHead).len(), 1);
        assert!(laptop.sources_of(SourceRole::ExpertImport).is_empty());

        let workstation = engine_recipe(32);
        assert_eq!(workstation.experts, GgmlDType::Q3_KO);
        assert_eq!(workstation.expert_source, ExpertSource::Requantized);

        let blackwell = engine_recipe(72);
        assert_eq!(blackwell.experts, GgmlDType::Q4_KO);
        assert_eq!(blackwell.expert_source, ExpertSource::AwqImport);
        assert_eq!(blackwell.sources_of(SourceRole::ExpertImport).len(), 4);

        let above = engine_recipe(96);
        assert_eq!(above.experts, GgmlDType::Q8_0);
        assert_eq!(above.expert_source, ExpertSource::Verbatim);

        let names: HashSet<String> = [&laptop, &workstation, &blackwell, &above]
            .iter()
            .map(|r| r.artifact_name())
            .collect();
        assert_eq!(names.len(), 4, "two rungs share an artifact name");
        assert!(laptop
            .artifact_name()
            .starts_with("Qwen3.8-Flash-Next-Q2_KOEXP-"));
    }

    /// Does this checkpoint carry a NextN / MTP draft head?
    ///
    /// Speculative decode needs a drafter, and the cheapest one is the head the
    /// checkpoint ships — a single block that reuses the trunk's `lm_head`
    /// (`qwen35::mtp`). Whether Qwen3.8-Flash-Next has one is a property of the
    /// released weights, not of this engine, so it is answered by reading them
    /// rather than by reasoning about the config: this prints every metadata key
    /// and tensor name that mentions the head, so a "no" is a *checked* no.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     the_checkpoint_draft_head_inventory -- --ignored --nocapture
    #[test]
    #[ignore = "reads the merged engine GGUF's header"]
    fn the_checkpoint_draft_head_inventory() -> Result<()> {
        use crate::models::latent_moe::GgufModel;
        let merged = engine_gguf()?;
        let gguf = GgufModel::open(&[merged])?;
        let mut meta: Vec<String> = gguf
            .metadata
            .iter()
            .filter(|(k, _)| {
                let k = k.to_lowercase();
                k.contains("nextn") || k.contains("mtp") || k.contains("block_count")
            })
            .map(|(k, v)| format!("{k} = {v:?}"))
            .collect();
        meta.sort();
        println!("metadata:");
        for m in &meta {
            println!("  {m}");
        }
        let names = gguf.tensor_names();
        let mut heads: Vec<&String> = names
            .iter()
            .filter(|n| {
                let l = n.to_lowercase();
                // The head's own tensors AND its whole block: the block is what
                // says how the head wires into a hyper-connection stack, and
                // the norms' widths are what say which collapse each one is.
                l.contains("nextn") || l.contains("mtp") || n.starts_with("blk.48.")
            })
            .collect();
        heads.sort();
        println!("draft-head tensors: {}", heads.len());
        // With shapes: the head's own geometry is what decides how it wires to
        // a stack whose residual is `hc` streams wide. `hnorm` spanning
        // `hc_dim` rather than `n_embd` is what says it norms the WIDE residual
        // and the head's mixer narrows it afterwards, and `nextn.hc_head_norm`'s
        // width says the same about the output side.
        for h in heads.iter().take(24) {
            match gguf.info(h) {
                Some(i) => println!("  {h}  {:?} {:?}", i.shape.dims(), i.ggml_dtype),
                None => println!("  {h}  (no info)"),
            }
        }
        // The trunk's own block count, to see whether extra blocks are present
        // beyond the 48 the engine loads.
        let mut max_blk = 0usize;
        for n in names.iter() {
            if let Some(rest) = n.strip_prefix("blk.") {
                if let Some((idx, _)) = rest.split_once('.') {
                    if let Ok(i) = idx.parse::<usize>() {
                        max_blk = max_blk.max(i + 1);
                    }
                }
            }
        }
        println!("highest blk.N present: {max_blk} (engine loads 48)");

        // The merged file is a CONVERTED artifact, so a head could have been
        // dropped on the way. Ask the pinned upstream Q8_0 split too — that is
        // the released checkpoint, and its answer is the one that decides
        // whether a shipped draft head exists at all.
        let src = pinned_shards()?;
        let src_gguf = GgufModel::open(&src)?;
        let src_names = src_gguf.tensor_names();
        let mut src_heads: Vec<&String> = src_names
            .iter()
            .filter(|n| {
                let n = n.to_lowercase();
                n.contains("nextn") || n.contains("mtp")
            })
            .collect();
        src_heads.sort();
        let mut src_blk = 0usize;
        for n in src_names.iter() {
            if let Some(rest) = n.strip_prefix("blk.") {
                if let Some((idx, _)) = rest.split_once('.') {
                    if let Ok(i) = idx.parse::<usize>() {
                        src_blk = src_blk.max(i + 1);
                    }
                }
            }
        }
        println!(
            "upstream Q8_0: {} tensors, draft-head tensors {}, highest blk.N {src_blk}",
            src_names.len(),
            src_heads.len()
        );
        for h in src_heads.iter().take(24) {
            println!("  {h}");
        }
        Ok(())
    }

    /// **The prepared artifact's head block and top-level tensors**, from its
    /// header alone — no source is read, so this runs on a machine whose
    /// sources were released after the build.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     the_engine_artifact_head_inventory -- --ignored --nocapture
    #[test]
    #[ignore = "reads the prepared engine artifact's header"]
    fn the_engine_artifact_head_inventory() -> Result<()> {
        use crate::models::qwen4exp::prepare::prepared;
        let device = Device::new_cuda(0)?;
        let recipe = engine_recipe(quant_ladder::device_vram_gib(&device)?);
        let path = prepared(&recipe, &engine_artifact_dir())?.ok_or_else(|| {
            candle::Error::Msg(format!("{} is not prepared", recipe.artifact_name()))
        })?;
        let content = Content::read(&mut File::open(&path)?)?;
        let mut names: Vec<&String> = content
            .tensor_infos
            .keys()
            .filter(|n| n.starts_with("blk.48.") || !n.starts_with("blk."))
            .collect();
        names.sort();
        for n in names {
            let i = &content.tensor_infos[n];
            println!("  {n:<44} {:?} {:?}", i.shape.dims(), i.ggml_dtype);
        }
        Ok(())
    }

    /// The MTP draft-head sidecar, read through **our own loader**.
    ///
    /// The release's GGUF lineage carries no head
    /// ([`the_checkpoint_draft_head_inventory`]), but the published weights do,
    /// and a community conversion ships them as the sidecar
    /// `Qwen35LoadOptions::mtp_path` already knows how to take. Whether *this
    /// engine* can read it is a different question from whether the file is
    /// well-formed, so this opens it with `GgufModel` and reports the geometry
    /// the loader will need — a compatibility claim made by the loader itself.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     the_mtp_sidecar_is_loadable -- --ignored --nocapture
    #[test]
    #[ignore = "reads the MTP sidecar GGUF's header"]
    fn the_mtp_sidecar_is_loadable() -> Result<()> {
        use crate::models::latent_moe::GgufModel;
        let path = hf_get(
            QWEN4EXP_REPO,
            RepoType::Model,
            QWEN4EXP_REV,
            QWEN4EXP_MTP_FILE.0,
        )?;
        let mut gguf = GgufModel::open(&[path])?;
        let arch = gguf.metadata.get("general.architecture");
        println!("architecture: {arch:?}");
        for k in [
            "qwen4exp.block_count",
            "qwen4exp.nextn_predict_layers",
            "qwen4exp.embedding_length",
            "qwen4exp.expert_count",
            "qwen4exp.expert_used_count",
            "qwen4exp.attention.head_count",
            "qwen4exp.attention.head_count_kv",
            "qwen4exp.attention.key_length",
            "qwen4exp.hyper_connection.count",
            "qwen4exp.hyper_connection.low_rank",
        ] {
            println!("  {k} = {:?}", gguf.metadata.get(k));
        }
        let mut names = gguf.tensor_names();
        names.sort();
        println!("tensors: {}", names.len());
        for n in &names {
            let dims = gguf.info(n).map(|i| i.shape.dims().to_vec());
            let dt = gguf.info(n).map(|i| i.ggml_dtype);
            println!("  {n:<44} {dims:?} {dt:?}");
        }

        // **Are the head's `token_embd` / `output` the trunk's?**
        //
        // The self-contained head carries both so it can run standalone, and
        // `mtp_use_dedicated_embeddings: false` says they ARE the trunk's —
        // which is what makes dropping them at merge safe. Bytes decide it.
        let device = Device::new_cuda(0)?;
        let mut trunk = GgufModel::open(&pinned_shards()?)?;
        for name in ["token_embd.weight", "output.weight"] {
            let a = gguf.qtensor(name, &device)?.dequantize(&device)?;
            let b = trunk.qtensor(name, &device)?.dequantize(&device)?;
            if a.dims() != b.dims() {
                println!(
                    "  {name}: shapes differ {:?} vs {:?} — HEAD'S OWN",
                    a.dims(),
                    b.dims()
                );
                continue;
            }
            let d = (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            println!(
                "  {name:<26} max|Δ| vs trunk = {d:.3e}  =>  {}",
                if d == 0.0 {
                    "DUPLICATE, safe to drop"
                } else {
                    "HEAD'S OWN, must keep"
                }
            );
        }
        Ok(())
    }

    /// **Build this card's engine artifact**, or confirm the one on disk.
    ///
    /// The recipe is [`engine_recipe`] for the card's VRAM: the `Q8_0` trunk
    /// and n-gram table verbatim, the MTP head folded in as the block past the
    /// trunk (`docs/qwen38_flash_next.md` §14 — a NextN head "is a layer of the
    /// model, not a sidecar", and its 512 experts join the same grid), and the
    /// routed experts at the rung's width. A present artifact whose stamp
    /// matches is used as it stands; otherwise the pinned sources are fetched,
    /// verified, built from and then deleted.
    #[test]
    #[ignore = "downloads the pinned sources (~192 GB on a requant rung) and writes the \
                engine artifact (~95 GB at Q2_KO) into zend's model cache, then deletes the \
                sources. Run with: cargo test --release --features cuda -p candle-transformers \
                --lib quantized_qwen38_moe::tests::prepare_engine_artifact \
                -- --ignored --nocapture --test-threads=1"]
    fn prepare_engine_artifact() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let recipe = engine_recipe(quant_ladder::device_vram_gib(&device)?);
        println!("recipe {}:\n{}", recipe.digest(), recipe.canonical());
        let t = std::time::Instant::now();
        let path = engine_gguf()?;
        println!(
            "✓ engine artifact ready ({:.0}s): {} ({} GiB)",
            t.elapsed().as_secs_f32(),
            path.display(),
            std::fs::metadata(&path)?.len() >> 30
        );
        Ok(())
    }

    /// GPU engine smoke: load the merged Q4KOEXP artifact onto the card, run
    /// the probe prompt through `forward_wave` (prefill + greedy decode), and
    /// require the oracle's own continuation. The first run of the wave path
    /// end to end: embed → GR → GDN spans → paged attention → 512-expert MoE
    /// → PLE → head, all through `drive_wave`.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF (builds the expert pack on first \
                run) and needs a GPU"]
    fn test_engine_wave_paris_smoke() -> Result<()> {
        use crate::models::batched_inference::BatchedConfig;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;
        use candle::IndexOp;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let t0 = std::time::Instant::now();
        let gpu = Qwen4ExpGpu::load(&merged, &device, Int8Mode::auto(&device))?;
        println!("✓ engine loaded in {:.0}s", t0.elapsed().as_secs_f32());
        let model = Qwen4ExpBatched::new(gpu)?;

        let tok = tokenizer()?;
        let prompt = "The capital of France is";
        let ids: Vec<u32> = tok
            .encode(prompt, true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", ids.len());

        use crate::models::batched_inference::ManagedBatchedModel;
        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = ManagedBatchedModel::num_layers(&model);

        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let t1 = std::time::Instant::now();
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        println!(
            "✓ prefill {} tokens in {:.1}s",
            ids.len(),
            t1.elapsed().as_secs_f32()
        );
        session.advance_sequence(seq, ids.len())?;
        let logits = step.logits_owned()?;
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        let mut gen = vec![next];
        let t2 = std::time::Instant::now();
        for _ in 0..11 {
            let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            next = step.logits_owned()?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            gen.push(next);
        }
        let dt = t2.elapsed().as_secs_f32();
        let text = tok
            .decode(&gen, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        println!("generated ids: {gen:?}");
        println!("continuation: {text:?}");
        println!("decode: {:.2} tok/s", 11.0 / dt);
        // The exact greedy continuation, not a substring probe. Decode here is
        // deterministic — same checkpoint, same prompt, argmax at every step —
        // so the token ids are a fixed expected value and anything else is a
        // regression worth failing on.
        //
        // A `contains("Paris")` check passed while this model was answering
        // `" Paris*\",\n    \"What is the capital of France?\","`: the first
        // token survives almost any corruption downstream of it, because it is
        // produced by prefill before the damage lands. Giving the draft head a
        // KV layer put an unwritten layer into the trunk's stream aggregates,
        // which clamped the sequence offset back to 0 on every wave entry
        // (`KvLayers`), and the substring check reported ok throughout.
        const EXPECTED: [u32; 12] = [
            11751, 13, 561, 6511, 314, 9564, 369, 19241, 13, 561, 6511, 314,
        ];
        assert_eq!(
            gen.as_slice(),
            EXPECTED.as_slice(),
            "greedy continuation drifted: {text:?}"
        );
        Ok(())
    }

    /// **The draft head proposes the tokens the trunk actually produces.**
    ///
    /// The bring-up gate for the head, and deliberately upstream of the whole
    /// verify path: speculation is lossless, so a head wired subtly wrong costs
    /// no correctness anywhere — it just proposes tokens nothing accepts, and
    /// the only symptom is a speedup that never arrives. That is invisible in
    /// an end-to-end measurement, which reports "speculation does not pay here"
    /// in exactly the same shape whether the ladder is too deep or the head is
    /// reading its input through the wrong norm.
    ///
    /// So this asks the narrow question directly. After the same prefill
    /// [`test_engine_wave_paris_smoke`] uses, the trunk's continuation is known
    /// exactly — `[13, 561, 6511, 314]` after the committed `11751` — so the
    /// head's proposals can be compared against ground truth rather than
    /// against a threshold.
    ///
    /// The floor is the FIRST proposal, not a count. Proposal `j > 0` is
    /// conditioned on the head's own earlier guesses, so a head that is right
    /// once and then drifts is still working; a head whose very first token is
    /// wrong is reading a different model than the trunk.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_draft_head_proposes_the_trunks_tokens \
                -- --ignored --nocapture --test-threads=1"]
    fn test_draft_head_proposes_the_trunks_tokens() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;
        use candle::IndexOp;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&merged, &device, Int8Mode::auto(&device))?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;
        let ids: Vec<u32> = tok
            .encode("The capital of France is", true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = ManagedBatchedModel::num_layers(&model);
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let committed = step.logits_owned()?[0]
            .i(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        // The head must be loaded at all: a budget of 0 here would make every
        // assertion below vacuous.
        assert!(
            model.draft_budget(1) > 0,
            "this artifact reports a zero draft budget at width 1 — it carries no NextN head"
        );
        let proposals = model.speculative_draft(&mut session, &[seq], &[committed], 4)?;
        let got = &proposals[0];
        // What the trunk really says next, from the smoke's pinned continuation.
        const TRUTH: [u32; 4] = [13, 561, 6511, 314];
        let matched = got
            .iter()
            .zip(TRUTH.iter())
            .take_while(|(a, b)| a == b)
            .count();
        println!("committed: {committed}");
        println!("drafted:   {got:?}");
        println!("truth:     {TRUTH:?}");
        println!("accepted prefix: {matched} of {}", got.len());

        assert!(
            !got.is_empty(),
            "the head proposed nothing — it had no seed, which after a prefill means \
             `head_wave_pass` never ran"
        );
        assert_eq!(
            got[0], TRUTH[0],
            "the head's FIRST proposal disagrees with the trunk: drafted {got:?} against \
             {TRUTH:?}. Every later position is conditioned on the head's own guesses, so \
             this one is the wiring check — a mismatch here means the head's input \
             assembly or output collapse is reading the model differently than the trunk."
        );

        // The head's KV must be exactly where it was: a walk leaves no trace,
        // and a head layer left longer than the trunk's is a skew the next wave
        // would "heal" by truncating a token the caller was already given.
        let lens: Vec<usize> = session
            .sequence_caches(seq)
            .expect("live slot")
            .caches
            .iter()
            .map(|c| {
                let mut n = 0usize;
                c.k_cache().chunked_visit_live_chunks(|it| {
                    for ch in it {
                        n += ch.token_count as usize;
                    }
                });
                n
            })
            .collect();
        println!("layer token counts after drafting: {lens:?}");
        assert!(
            lens.windows(2).all(|w| w[0] == w[1]),
            "the draft walk left the layers uneven: {lens:?}"
        );
        Ok(())
    }

    /// **The model stops on its own.** A chat turn that asks for a one-word
    /// answer, decoded until the model emits an end token or a cap is reached.
    ///
    /// [`test_engine_wave_paris_smoke`] cannot answer this and never could: it
    /// prompts with the bare completion prefix `"The capital of France is"` and
    /// runs a fixed 11 iterations. Nothing there asks the model to finish, so
    /// continuing into "The capital of Germany is Berlin" is the *correct*
    /// response to that input — the loop simply stops counting. Termination is
    /// a different property and needs a prompt that has an end.
    ///
    /// Three things have to line up for this to pass, and each fails
    /// differently:
    ///
    /// 1. **The chat template is right.** Wrong turn markers and the model
    ///    continues the transcript — writing the user's next turn instead of
    ///    ending its own.
    /// 2. **Thinking is closed.** This checkpoint carries `<think>` / `</think>`
    ///    (248068/248069). Qwen3's non-thinking form pre-fills the assistant
    ///    turn with an already-closed block; without it the model reasons for
    ///    hundreds of tokens and a cap this size looks exactly like a failure
    ///    to stop.
    /// 3. **The end token is reachable at all.** `<|im_end|>` is 248046 —
    ///    *not* the 248044 the config names as `eos_token_id`, which is
    ///    `<|endoftext|>` and ends a document rather than a turn. A stop list
    ///    built from `eos_token_id` alone never fires on a chat turn, which is
    ///    why the conversation layer resolves both by name.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_engine_stops_on_end_of_turn \
                -- --ignored --nocapture --test-threads=1"]
    fn test_engine_stops_on_end_of_turn() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;
        use candle::IndexOp;

        /// `<|im_end|>` — the end of an assistant TURN.
        const IM_END: u32 = 248_046;
        /// `<|endoftext|>` — the end of a DOCUMENT, and what the config calls
        /// `eos_token_id`. Accepted as a stop here because either one means the
        /// model chose to finish.
        const ENDOFTEXT: u32 = 248_044;
        /// Generous enough that a model which genuinely wants to keep talking
        /// is not mistaken for one that stopped.
        const MAX_NEW: usize = 48;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&merged, &device, Int8Mode::auto(&device))?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;

        // Qwen3's chat form with thinking disabled: the assistant turn opens
        // with an already-closed think block, so the model answers directly.
        let prompt = "<|im_start|>user\nWhat is the capital of France? \
                      Answer with a single word.<|im_end|>\n\
                      <|im_start|>assistant\n<think>\n\n</think>\n\n";
        // `false`: the template above supplies its own special tokens, so the
        // post-processor must not add another set around them.
        let ids: Vec<u32> = tok
            .encode(prompt, false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", ids.len());

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = ManagedBatchedModel::num_layers(&model);

        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let mut next = step.logits_owned()?[0]
            .i(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        let mut gen: Vec<u32> = Vec::new();
        let mut stopped_at: Option<u32> = None;
        for _ in 0..MAX_NEW {
            if next == IM_END || next == ENDOFTEXT {
                stopped_at = Some(next);
                break;
            }
            gen.push(next);
            let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            next = step.logits_owned()?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
        }
        let answer = tok
            .decode(&gen, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        println!("answer ids: {gen:?}");
        println!("answer: {answer:?}");
        println!(
            "stopped on: {}",
            match stopped_at {
                Some(IM_END) => "<|im_end|>".to_string(),
                Some(ENDOFTEXT) => "<|endoftext|>".to_string(),
                Some(other) => format!("token {other}"),
                None => format!("NOTHING — still generating after {MAX_NEW} tokens"),
            }
        );

        let stop = stopped_at.ok_or_else(|| {
            candle::Error::Msg(format!(
                "the model never ended its turn: {MAX_NEW} tokens generated with no \
                 <|im_end|> ({IM_END}) or <|endoftext|> ({ENDOFTEXT}) — got {answer:?}"
            ))
        })?;
        assert!(
            stop == IM_END || stop == ENDOFTEXT,
            "stopped on an unexpected token {stop}"
        );
        assert!(
            answer.contains("Paris"),
            "the model ended its turn but answered {answer:?} rather than Paris — \
             termination is right, the answer is not"
        );
        // A one-word instruction, honoured. Loose enough not to police
        // punctuation or a leading space, tight enough that a model which
        // ended its turn only after a paragraph still fails.
        assert!(
            gen.len() <= 4,
            "asked for a single word and got {} tokens: {answer:?}",
            gen.len()
        );
        Ok(())
    }

    /// **QSA at depth on the engine.** A prompt past the 2051-position
    /// selection width, so the 12 full-attention layers genuinely select
    /// instead of degenerating to dense — the regime the engine could not
    /// reach before the indexer was built.
    ///
    /// What this asserts, and why each part is here:
    ///
    /// 1. **The indexer engaged.** `qsa_rows_selected` counts rows a
    ///    selection was built for; zero would mean the whole run took the
    ///    identity path and the rest of the test proved nothing.
    /// 2. **Narrowing is free.** The same engine runs the same prompt with the
    ///    budget widened past the depth, which makes `selection_engages` false
    ///    and the read dense. The selected run must produce the **same tokens**
    ///    — not merely an answer that still mentions the needle. This is the
    ///    claim QSA makes, and it is the one a subtly wrong selection (an
    ///    off-by-one in the block arithmetic, a mis-indexed row) fails.
    /// 3. **The read reached the needle at all.** It sits in the FIRST
    ///    sentence, thousands of tokens behind the query, and its distinctive
    ///    prefix appears in both answers. Without this, two identically broken
    ///    reads would agree and pass (2).
    /// 4. **It is deterministic**, run to run, which a race in the streaming
    ///    top-k or in the per-sequence cache would break.
    /// 5. **Decode carries it.** The continuation is generated one token at a
    ///    time, so the decode path's selection (one row per slot, built
    ///    against a cache the prefill filled) is exercised too.
    ///
    /// **The dense control is load-bearing, and its absence made this test
    /// wrong.** Asserting the needle alone conflates two failures: a selection
    /// that dropped the block, and a prompt the checkpoint does not answer. It
    /// was the second — the prompt was a bare continuation ("Answer: the label
    /// was"), which an instruct model answers by switching into assistant mode
    /// rather than finishing the sentence, and the dense read failed it exactly
    /// as the selected one did.
    ///
    /// The selection ITSELF is pinned against the CPU oracle's
    /// `qsa_selection_mask` in `qwen4exp::indexer` and the attention's
    /// honouring of it in `tests/qsa_kernel_tests.rs`, both without a model in
    /// the way — this is the end-to-end statement those two make possible.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and prefills >2100 tokens. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_engine_qsa_at_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn test_engine_qsa_at_depth() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;
        use candle::IndexOp;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let t0 = std::time::Instant::now();
        let mut gpu = Qwen4ExpGpu::load(&merged, &device, Int8Mode::auto(&device))?;
        let released_top_k = gpu.cfg.indexer.top_k;
        let width = released_top_k + 4 - 1;
        println!(
            "✓ engine loaded in {:.0}s (selection width {width})",
            t0.elapsed().as_secs_f32()
        );
        // **The DENSE control, on the same model and the same prompt.** A needle
        // that the selected read loses says nothing on its own: the continuation
        // could be lost to the selection, or the checkpoint could simply not
        // answer this prompt. Widening the budget past the prompt makes
        // `selection_engages` false — the same code path, reading every cell —
        // so the two runs differ in exactly one thing.
        let dense_top_k = 1 << 20;
        gpu.cfg.indexer.top_k = dense_top_k;
        let mut model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;

        // The needle first, then filler until the query is far past the
        // budget: everything between is uniform, so a dense read and a
        // selected read see very different things.
        let mut text = String::from(
            "The expedition's logbook opens with a single line: the sample recovered \
             from the northern ridge was labelled AMBERGRIS-7.\n\n",
        );
        let filler =
            "The team recorded the weather each morning and stowed the equipment each night. ";
        while tok
            .encode(text.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .len()
            < 2400
        {
            text.push_str(filler);
        }
        // **In the shape the checkpoint was trained on**, with thinking
        // suppressed — the same framing `test_engine_stops_on_end_of_turn`
        // uses. Asked as a bare continuation ("Answer: the label was"), this
        // is an instruct model handed something that is not a turn: it
        // recovers the first token of the needle and then switches into
        // assistant mode, emitting `\n\n<think>` instead of finishing the
        // word. Measured — the DENSE read did it too, so the shape was
        // failing the test, not the selection. The empty `<think></think>`
        // keeps the answer inside the seven decode steps this measures.
        let text = format!(
            "<|im_start|>user\n{text}\n\nWhat label was written on the sample from the \
             northern ridge? Answer with the label alone.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let ids: Vec<u32> = tok
            .encode(text.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        assert!(
            ids.len() > width,
            "prompt of {} tokens is inside the {width}-cell budget — selection would be \
             the identity and this test would assert nothing",
            ids.len()
        );
        println!("prompt: {} tokens", ids.len());

        let n_layers = ManagedBatchedModel::num_layers(&model);
        let run = |model: &Qwen4ExpBatched| -> Result<(String, u64)> {
            let before = model.qsa_rows_selected();
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let seq = session.create_sequence()?;
            let prompt = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let t = std::time::Instant::now();
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, ids.len())?;
            println!("  prefill {:.1}s", t.elapsed().as_secs_f32());
            let logits = step.logits_owned()?;
            let m = logits[0].abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(m.is_finite(), "non-finite logits under QSA selection");
            // **The turn's stop tokens, and the decode honours them.**
            // `<|im_end|>` ends an assistant turn and `<|endoftext|>` ends a
            // document; past either, the model is continuing something it
            // already finished and what it emits is not a claim about
            // anything. Decoding through it made this test compare that
            // garbage: two runs answered `AMBER` identically and then filled
            // the remaining steps differently — one with `<|endoftext|>` then
            // `<|im_start|>` repeats, the other with `<|im_end|>` then an
            // invented user turn — and the determinism assertion read that as
            // a race in the selection.
            const IM_END: u32 = 248_046;
            const ENDOFTEXT: u32 = 248_044;
            let stop = |t: u32| t == IM_END || t == ENDOFTEXT;

            let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            let mut gen = vec![next];
            // Decode is timed on its own: with the read capped at 2051 cells
            // this rate is the STEADY STATE for any depth, which is what makes
            // it the denominator §6.2's kernel question needs.
            let t_dec = std::time::Instant::now();
            const DECODE_STEPS: usize = 7;
            for _ in 0..DECODE_STEPS {
                if stop(next) {
                    break;
                }
                let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
                let step = model.forward_wave(
                    &mut session,
                    &[seq],
                    std::slice::from_ref(&t),
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    n_layers,
                    None,
                )?;
                session.advance_sequence(seq, 1)?;
                next = step.logits_owned()?[0]
                    .i(0)?
                    .argmax(0)?
                    .to_scalar::<u32>()?;
                gen.push(next);
            }
            let dec = t_dec.elapsed().as_secs_f64();
            println!(
                "  decode {DECODE_STEPS} steps in {:.2}s = {:.1} tok/s ({:.1} ms/step)",
                dec,
                DECODE_STEPS as f64 / dec,
                1000.0 * dec / DECODE_STEPS as f64
            );
            model.release_sequence(seq)?;
            // **The answer, without the marker that ends it.** `<|im_end|>` and
            // `<|endoftext|>` both mean the model chose to finish, and which of
            // the two wins is a coin-flip between near-tied logits that a
            // selection reading 2,051 of 2,444 cells is entitled to move.
            // Comparing it would be asserting bit-identical logits, which
            // narrowing does not promise and does not need to: what has to
            // survive the narrowing is the answer.
            let body: Vec<u32> = gen.iter().copied().take_while(|&t| !stop(t)).collect();
            let out = tok
                .decode(&body, false)
                .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
            Ok((out, model.qsa_rows_selected() - before))
        };

        // ── The dense control first ──────────────────────────────────────────
        // Same engine, same prompt, budget above the depth so the selection is
        // the identity. What this run recovers is the ceiling; what the selected
        // run recovers is measured against it, and only the DIFFERENCE is QSA's.
        let (dense, dense_rows) = run(&model)?;
        println!("dense control: {dense:?}  (QSA rows selected: {dense_rows})");
        assert_eq!(
            dense_rows, 0,
            "the dense control narrowed {dense_rows} row(s) — a budget of {dense_top_k} was \
             supposed to put every cell inside it, so this is not a control at all"
        );

        model.set_selection_budget(released_top_k)?;
        let (first, rows) = run(&model)?;
        println!("continuation: {first:?}  (QSA rows selected: {rows})");
        // §6.2: what the decode kernel's share actually is at a QSA-capped
        // read, on the geometry the stale kernel comment calls exotic (hpg 12
        // at head_dim 256, which runs single-stage). Measure before touching
        // the kernel — with `--features profile`, otherwise this is empty.
        {
            use crate::models::profile::{gpu_drain_blocking, pipeline_snapshot_and_reset};
            gpu_drain_blocking();
            let mut snap = pipeline_snapshot_and_reset();
            if !snap.entries.is_empty() {
                snap.entries
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let total: f64 = snap.entries.iter().map(|e| e.1).sum();
                println!("\n=== QSA-depth profile (total {total:.1} ms) ===");
                for (name, ms, calls) in snap.entries.iter().take(18) {
                    println!(
                        "  {name:<28} {ms:>9.1} ms  {:>5.1}%  ×{calls}",
                        100.0 * ms / total
                    );
                }
            }
        }
        assert!(
            rows > 0,
            "no row was ever narrowed — the indexer never engaged, so nothing here \
             tested the selection"
        );
        // **The needle's distinctive prefix, not the whole label.** Greedy
        // decode answers this prompt with the single token `AMBER` and then
        // ends the turn — the DENSE read does exactly the same, so the rest of
        // `AMBERGRIS-7` is this checkpoint's terseness and not something a read
        // can recover. `AMBER` still only comes from the needle: nothing in the
        // filler resembles it, so this remains a retrieval claim rather than a
        // formatting one, which is what the "plausible alternative" the doc
        // above worries about would fail.
        const NEEDLE: &str = "AMBER";
        assert!(
            dense.to_uppercase().contains(NEEDLE),
            "the DENSE read lost the needle: {dense:?} — nothing below is about the \
             selection, because reading every cell did not answer this prompt"
        );
        assert!(
            first.to_uppercase().contains(NEEDLE),
            "the needle did not survive the SELECTED read: {first:?} — the dense control \
             recovered it ({dense:?}), so the selection dropped the block that held it"
        );
        // **The claim QSA actually makes, and the strongest one available
        // here.** Reading 2,051 of 2,444 cells produced the same answer as
        // reading all of them — not merely an answer that still mentions the
        // needle, but the same tokens. A selection that dropped the needle's
        // block, or kept it and reordered the rest, fails here while a
        // `contains` check would pass.
        assert_eq!(
            first, dense,
            "the selected read diverged from the dense one: {first:?} against {dense:?} — \
             {rows} row(s) were narrowed, and narrowing is supposed to leave the answer alone"
        );
        let (second, _) = run(&model)?;
        assert_eq!(first, second, "selection at depth is nondeterministic");
        println!("✓ QSA engaged, dense-identical, needle recovered, deterministic");
        Ok(())
    }

    /// Layer-range NaN bisect over the GPU wave (§7.3 method): run the prompt
    /// through `[0, k)` for growing `k` and report the residual's max-abs —
    /// the first non-finite `k` names the layer, and the schedule names the
    /// subsystem (3:1 GDN/attention, PLE at layer 1, MoE everywhere).
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF; diagnostic"]
    fn test_engine_wave_nan_bisect() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&merged, &device, Int8Mode::auto(&device))?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;
        let ids: Vec<u32> = tok
            .encode("The capital of France is", true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let n_layers = ManagedBatchedModel::num_layers(&model);
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;

        for k in [0usize, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 40, 47, 48] {
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let seq = session.create_sequence()?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt_t),
                &[],
                &[],
                0,
                k.min(n_layers),
                None,
            )?;
            if k < n_layers {
                let r = step
                    .into_residual()
                    .ok_or_else(|| candle::Error::Msg("expected a residual".into()))?;
                let m = r
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_scalar::<f32>()?;
                println!("layers [0,{k:>2}): residual max|x| = {m:e}");
            } else {
                let lg = step.logits_owned()?;
                let m = lg[0]
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_scalar::<f32>()?;
                println!("layers [0,{k:>2}) + head: logits max|x| = {m:e}");
            }
        }
        Ok(())
    }

    fn tokenizer() -> Result<tokenizers::Tokenizer> {
        let p = hf_get(
            TOKENIZER_REPO,
            RepoType::Model,
            TOKENIZER_REV,
            "tokenizer.json",
        )?;
        tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| candle::Error::Msg(format!("load tokenizer: {e}")))
    }

    /// **The gate's config ladder** — the BF16 widths and the C rungs, in the
    /// order they are read.
    ///
    /// Shared by the plain gate and the speculative one so the two tables are
    /// the same measurement with one variable changed. A speculative run over a
    /// different ladder would answer a different question, and the difference
    /// would be invisible in the output: both print a table of rungs.
    ///
    /// The DeepSeek gate's shape — cold ×1, the batched widths, then a warm ×1
    /// to read steady state — followed by:
    ///
    /// **The C-ladder.** Compressed KV. `QWEN4EXP_KV_FACTORS` is the row these
    /// calibrate, and the two ends do different jobs: **C5 is the operating
    /// point** (what zend runs) and **C10 is the probe** — tuned to pass just
    /// under its breaking edge so the row can be placed with knowledge of how
    /// much edge is left. A red C10 means the thresholds drifted past it, and
    /// the fix is the factor row rather than a widened tolerance.
    ///
    /// Two things are unusual here and are why this row could not be inherited
    /// from the 3.6 sibling. Only **12 of 48** layers hold K/V, so a level's
    /// error touches a quarter of the stack; and QSA caps the read at 2051
    /// cells, so above that depth a block's error reaches the output only if
    /// the indexer selects it. These rungs run at gate depth (~713 tokens),
    /// where selection is the identity — the row they produce is a bound for
    /// deeper contexts, not a measurement of them.
    ///
    /// C5 rather than C4 for the middle rung: it is the level zend actually
    /// runs (`session.rs` sets `compression_level(5)`), so the gate covers the
    /// operating point directly instead of leaving it to be inferred between
    /// two rungs either side of it. C10 at two widths rather than one: the pair
    /// is what shows the top rung holding as the cohort grows, which is exactly
    /// where a row tuned at a single width quietly stops covering the next (the
    /// 3.5 sibling needed a retune for precisely that). A closing C10 ×1 sets
    /// the top rung's single-sequence rates beside BF16 ×1's, so the cost of
    /// maximum compression reads straight off the table.
    ///
    /// **The ×16 rung is gated on VRAM.** What bounds width here is per-session
    /// state, not the checkpoint: 36 GDN layers carry 256 MiB of recurrent state
    /// a sequence (both halves of the store), so sixteen want 4 GiB before any
    /// K/V or the 3.3 GiB tier a sixteen-wide decode stands. On the 16 GB card the
    /// dense trunk plus an expert zone on its floor leave 221 regions against the
    /// 275 that needs — measured, and refused by the weight side on its floor.
    /// Twenty-four GiB is the smallest card in the fleet with that room; below it
    /// the rung is skipped and says so, because a narrower run is still a green
    /// run and silence would let reduced coverage read as a pass.
    fn gate_ladder(vram_gib: u64) -> Vec<TestConfig> {
        use crate::models::batched_inference::InferenceMode;

        let wide = vram_gib >= 24;
        if !wide {
            println!(
                "  - BF16 ×16 skipped: a {vram_gib} GiB card, under the 24 GiB gate — sixteen \
                 sequences' recurrent state does not fit beside the trunk"
            );
        }
        let widths: &[usize] = if wide {
            &[1, 4, 8, 16, 1]
        } else {
            &[1, 4, 8, 1]
        };
        let mut configs: Vec<TestConfig> = widths
            .iter()
            .map(|&n| TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect();
        configs.extend([0usize, 5, 8].map(|level| TestConfig {
            mode: match level {
                0 => InferenceMode::C0,
                5 => InferenceMode::C5,
                _ => InferenceMode::C8,
            },
            use_batched: true,
            num_contexts: 2,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        }));
        configs.extend([2usize, 8, 1].map(|n| TestConfig {
            mode: InferenceMode::C10,
            use_batched: true,
            num_contexts: n,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        }));
        configs
    }

    /// **The speculative ladder** — every compression rung, and the widths.
    ///
    /// Wider than [`gate_ladder`] on purpose. That one is a *calibration* gate:
    /// it samples C0/C5/C8/C10 because those four are what pin
    /// `QWEN4EXP_KV_FACTORS`, and running the levels between them would cost
    /// time to re-measure a row already bounded by its ends. This one asks a
    /// different question — what speculation is worth at each operating point —
    /// and the answer is not interpolable between rungs, because the accepted
    /// prefix depends on how much the compressed K/V has moved the target's own
    /// argmaxes. A level where acceptance collapses is a level where the
    /// drafter and the target have stopped agreeing, and only the rung itself
    /// shows that.
    ///
    /// The 32-context row is the width limit, and it is here rather than in the
    /// plain gate because it is where the rewind stash is priced: the stash is
    /// `width × (budget + 1)` rows across all 36 DeltaNet layers, so 32 is the
    /// width at which `affordable_draft_budget` is expected to start clamping
    /// the ladder's depth. A row that runs and reports a *shallower* effective
    /// budget there is the clamp working; a row that OOMs is the clamp priced
    /// wrong.
    fn speculative_ladder() -> Vec<TestConfig> {
        use crate::models::batched_inference::InferenceMode;

        let mut configs: Vec<TestConfig> = [1usize, 4, 8, 16, 32]
            .into_iter()
            .map(|n| TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect();
        // Every rung, C0 through C10, at the width the C-ladder is read at.
        configs.extend(
            [
                InferenceMode::C0,
                InferenceMode::C1,
                InferenceMode::C2,
                InferenceMode::C3,
                InferenceMode::C4,
                InferenceMode::C5,
                InferenceMode::C6,
                InferenceMode::C7,
                InferenceMode::C8,
                InferenceMode::C9,
                InferenceMode::C10,
            ]
            .map(|mode| TestConfig {
                mode,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }),
        );
        // The operating point and the probe at width too, since a rung that
        // holds at 2 can still lose the cohort at 8 — which is exactly the
        // failure the 3.5 sibling's retune was for.
        configs.extend(
            [InferenceMode::C5, InferenceMode::C10].map(|mode| TestConfig {
                mode,
                use_batched: true,
                num_contexts: 8,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }),
        );
        // A trailing single context, the way the plain gate ends: the leading
        // ×1 is cold — expert cache empty, arenas unbuilt — so it prices a
        // first request rather than the steady state. This one runs against
        // everything the ladder above warmed, which is the number a
        // single-session client actually sees.
        configs.push(TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        });
        configs
    }

    /// How deep this checkpoint's head is swept. Its own ceiling, not the
    /// hybrid's: a one-block NextN head is trained to predict a single position
    /// and is used recursively past that, so each extra row is conditioned on
    /// the head's own previous guess and pays less than the one before it. Four
    /// is far enough to see where that stops paying.
    const QWEN38_MAX_DRAFT: usize = 4;

    fn tokenizer_json() -> Result<String> {
        let p = hf_get(
            TOKENIZER_REPO,
            RepoType::Model,
            TOKENIZER_REV,
            "tokenizer.json",
        )?;
        std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
    }

    /// **The gate.** The same iterating `test_parallel_batched_forwarding`
    /// every production model here has (`docs/qwen38_flash_next.md` §11): the
    /// BF16 config ladder at the widths the DeepSeek gate runs, per-session
    /// StoryRewrite validation at the 100% threshold, the performance table,
    /// and the expert-pipeline stats. The C-ladder rungs wait on this model's
    /// own `PRODUCTION_*` threshold derivation (§8 item 6) — KV compression
    /// rows are per-model and per-machine and nothing carries over.
    /// **Speculative decode, swept.** The batched gate's shape and its
    /// StoryRewrite validation, run once per draft budget so the table shows
    /// what each extra read-ahead row buys.
    ///
    /// Measured here rather than on a short probe because speculation is a
    /// property of the *decode loop*: the win is tokens per driver step, and a
    /// handful of tokens is a handful of steps, where the prefill and the first
    /// block dominate whatever the drafter does.
    ///
    /// **Budget 0 is the baseline in the same harness**, not a number carried
    /// over from another run — the comparison has to hold the KV rungs, the
    /// expert cache's warmth and the session count fixed, and the only way to
    /// be sure of that is to measure both here.
    ///
    /// Speculation is lossless: every budget must produce the same validity, so
    /// a rung that goes red at depth is a rewind bug, never a quality tradeoff.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_speculative_ladder \
                -- --ignored --nocapture --test-threads=1"]
    fn test_speculative_ladder() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::ManagedBatchedModel;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        println!("\n=== Qwen3.8-Flash-Next: speculative ladder (production budget) ===\n");
        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);

        // **No `with_speculative`.** The default is `DraftBudget::Adaptive`,
        // which asks the model at each config's width — production behaviour,
        // and the whole point of this table. Pinning a budget would measure a
        // configuration nothing runs, and would hide the one thing the wide
        // rows exist to show: `affordable_draft_budget` clamping the depth a
        // 32-wide cohort can stash for.
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(7200);

        params.run(speculative_ladder(), || {
            let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
            let m = Qwen4ExpBatched::new(gpu)?;
            // A ladder that silently fell back to plain decode would still pass
            // — speculation is lossless, so the only symptom is the speedup
            // going away. Assert the drafter is really there.
            assert!(
                m.draft_budget(1) > 0,
                "this artifact reports a zero draft budget at width 1 — it carries no \
                 NextN head, so every row here would report plain decode"
            );
            println!("✓ Model loaded\n");
            Ok(m)
        })
    }

    /// **Which K format breaks it — one format per run, through the gate that
    /// reproduces.**
    ///
    /// C6 adds exactly four K candidates C5 never had — `Q1_S`, `Q2_A`, `Q2_S`
    /// and `Q8_1` — and acceptance falls 97.6% → 2.9% across that boundary.
    /// What says this is a *discrete* defect rather than precision loss is that
    /// C6 through C10 are all equally bad (1.5–4.9%) while their compression
    /// ratios keep climbing, 5.61× to 7.53×. Graduated loss would make C10 far
    /// worse than C6; a single bad format that any of those levels can select
    /// makes them all the same.
    ///
    /// So each suspect is pinned as the ONLY K format, via
    /// `BatchedConfig::override_k_quant`, and run through the gate's own
    /// StoryRewrite — the harness that actually reproduces the 97.6% figure.
    /// A control from C5's set (`Q4_0`) runs first: it must hold, or the
    /// override itself is what breaks things and no other row means anything.
    ///
    /// Only the sealed prefill is compressed — the live decode tail stays F16 —
    /// so a bad format here corrupts the *history* the head attends over while
    /// the tokens around it stay clean. That is consistent with proposals that
    /// read fluently and are never the trunk's.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_which_k_format_breaks_drafting \
                -- --ignored --nocapture --test-threads=1"]
    fn test_which_k_format_breaks_drafting() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;
        use candle_nn::kv_cache::QuantFormat;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let model = account_model_load(&device, || {
            let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
            Qwen4ExpBatched::new(gpu)
        })?;

        // `Q4_0` is the control — it is in C5's set, which drafts at 97.6%.
        // The rest are exactly what C6 adds.
        let suspects = [
            ("Q4_0 (control, in C5)", QuantFormat::Q4_0),
            ("Q1_S", QuantFormat::Q1_S),
            ("Q2_A", QuantFormat::Q2_A),
            ("Q2_S", QuantFormat::Q2_S),
            ("Q8_1", QuantFormat::Q8_1),
        ];

        for (label, fmt) in suspects {
            println!("\n=== K pinned to {label} ===\n");
            let params = TestParams::new(256, &tok, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_int8mode(int8mode)
                .with_override_k_quant(Some(fmt))
                .with_timeout_secs(3600);
            // One config: the level that works, so the ONLY variable is which
            // format its keys actually land in.
            let configs = vec![TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }];
            let ok = params.run_loaded(configs, &model).is_ok();
            println!("  {label}: validity {}", if ok { "PASS" } else { "FAIL" });
        }
        Ok(())
    }

    /// **Why acceptance collapses between C5 and C6 — the drafter, side by
    /// side, at both levels.**
    ///
    /// Drafted-token acceptance is 97.6% at C5 and 2.9% at C6, then flat at
    /// 1.5–4.9% through C10 no matter how much lossier the level gets. That
    /// shape is not compression noise decorrelating a prediction — noise gives
    /// a gradient, and a level far lossier than C6 would be far worse than C6.
    /// Flat-and-near-zero across five rungs is something **breaking** at the
    /// C5→C6 boundary and staying broken.
    ///
    /// The trunk is not what breaks: C6, C7 and C8 all validate 100%, so the
    /// target produces the same text it always did. Only the head's proposals
    /// collapse.
    ///
    /// So this runs the same sequence at both levels, deep enough that the KV
    /// has actually been compressed, and reports:
    ///
    /// * **which proposal position first misses** — position 0 missing means
    ///   the head is reading bad state; only later positions missing means the
    ///   walk's own recurrence;
    /// * **the rope depth each level hands the drafter**, and the head layer's
    ///   block count against the trunk's. `draft_walk`'s module docs call out
    ///   that a rope table which does not cover the drafted positions is
    ///   "silent wrong RoPE on every drafted position … it could only ever
    ///   surface as acceptance quietly collapsing", which is this symptom
    ///   exactly.
    ///
    /// A short prompt cannot see any of it: chunks are 32 tokens, so nothing is
    /// compressed until they fill, and C5 and C6 are then the same run.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_drafter_c5_vs_c6 \
                -- --ignored --nocapture --test-threads=1"]
    fn test_drafter_c5_vs_c6() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, InferenceMode, ManagedBatchedModel};
        use crate::models::draft_walk::draft_rope_depth;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;
        use candle::IndexOp;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&merged, &device, Int8Mode::auto(&device))?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;
        let head_kv = model
            .engine()
            .cfg
            .mtp_kv_layer()
            .ok_or_else(|| candle::Error::msg("no draft head"))?;

        // Long enough that chunks fill and the policy actually compresses.
        let para = "The archivist walked the long gallery each morning, counting the \
                    lamps that had failed overnight and noting them in a ledger nobody \
                    else read. ";
        let prompt = para.repeat(24);
        let ids: Vec<u32> = tok
            .encode(prompt.as_str(), true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", ids.len());

        const DRAFT: usize = 4;
        // Sampled at several depths, not one. A single sample cannot tell "the
        // drafter is broken here" from "the drafter degrades with depth", and
        // those want different explanations. BF16 is in the list as the arm
        // with no compression at all — if it tracks the compressed rungs, the
        // variable is depth; if it holds while they fall, it is compression.
        const CHECKPOINTS: [usize; 4] = [8, 32, 64, 96];

        for mode in [
            InferenceMode::BF16,
            InferenceMode::C0,
            InferenceMode::C5,
            InferenceMode::C6,
        ] {
            println!("\n=== {mode:?} ===");
            // **The formats, not only the level.** Setting `compression_level`
            // alone leaves the KV in the default float format, so nothing is
            // ever compressed and every rung behaves identically — a null
            // result that looks exactly like "the rungs are the same". The
            // harness builds a rung from all three fields; so must this.
            let config = BatchedConfig {
                k_format: mode.k_format(),
                v_format: mode.v_format(),
                compression_level: mode.compression_level(),
                ..Default::default()
            };
            let mut session = model.create_batched_session(config)?;
            let seq = session.create_sequence()?;
            let n_layers = ManagedBatchedModel::num_layers(&model);

            let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt_t),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, ids.len())?;
            // **Seal the prefilled history, as the gate does.** Compression
            // happens at seal, not at write, so a run that only decodes leaves
            // every chunk open and float — and then C0, C5 and C6 produce
            // byte-identical output, which reads as "the rungs do not differ"
            // rather than "nothing was compressed". Mirrors
            // `run_batched_config`, which seals after prefill for the same
            // reason its %Quantized column is meaningful.
            session.quantize_and_seal_sequences(&[seq], true)?;
            let mut next = step.logits_owned()?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;

            // Decode past the point where chunks have filled and been sealed,
            // so the KV the head reads is genuinely compressed.
            let plain = |session: &mut _, t: u32| -> Result<u32> {
                let x = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
                let s = model.forward_wave(
                    session,
                    &[seq],
                    std::slice::from_ref(&x),
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    n_layers,
                    None,
                )?;
                session.advance_sequence(seq, 1)?;
                s.logits_owned()?[0].i(0)?.argmax(0)?.to_scalar::<u32>()
            };
            let mut done = 0usize;
            for &mark in &CHECKPOINTS {
                while done < mark {
                    next = plain(&mut session, next)?;
                    done += 1;
                }
                let depth = draft_rope_depth(&session, &[seq], head_kv)?;
                let caches = session.sequence_caches(seq).expect("live slot");
                let trunk_blocks = caches.caches[0].k_cache().chunked_max_blocks();
                let head_blocks = caches.caches[head_kv].k_cache().chunked_max_blocks();
                let offset = session.sequence_offset(seq).unwrap_or(0);

                let proposals = model.speculative_draft(&mut session, &[seq], &[next], DRAFT)?;
                let drafted = proposals[0].clone();

                // Ground truth: what the trunk actually produces next. These
                // decodes advance the sequence, so the next checkpoint simply
                // continues from here.
                let mut truth: Vec<u32> = Vec::with_capacity(DRAFT);
                let mut cur = next;
                for _ in 0..DRAFT {
                    cur = plain(&mut session, cur)?;
                    truth.push(cur);
                }
                done += DRAFT;
                next = cur;
                let matched = drafted
                    .iter()
                    .zip(truth.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                println!(
                    "  +{mark:<3} offset {offset:<4} rope {depth}blk trunk {trunk_blocks} \
                     head {head_blocks} | accepted {matched}/{DRAFT}  drafted {drafted:?} \
                     truth {truth:?}"
                );
            }
            model.release_sequence(seq)?;
        }
        Ok(())
    }

    /// **C9/C10 drift from the reference without speculation, and this is the
    /// record of that.**
    ///
    /// Measured, at 256 generated tokens, width 2, every draft budget:
    ///
    /// | budget | C8 | C9 | C10 |
    /// |---|---|---|---|
    /// | 0 (plain decode) | exact | diverges @505/506 | diverges @505/440 |
    /// | 1 | exact | @505/506 | @505/443 |
    /// | 2 | exact | @505/506 | @505/440 |
    /// | 4 | exact | @505/506 | @505/440 |
    ///
    /// Same characters at every depth, and present with **no speculation at
    /// all**. So the top rungs are not a speculation defect: they lose the
    /// reference around 110–130 generated tokens on their own.
    ///
    /// **The standing gate cannot see this**, and that is the point worth
    /// keeping. `test_parallel_batched_forwarding` generates **64** tokens —
    /// roughly 250 characters — and the divergence starts past 440. Its C10 row
    /// passing is a true statement about the first 64 tokens and says nothing
    /// about the rung.
    ///
    /// C9 is also exactly where the **Q0 family** first becomes selectable for
    /// keys (`Q0`, a 1-byte zero block; `Q0_V`/`Q0_X`/`Q0_M2`/`Q0_M4`, whole-
    /// block summaries at 2–8 bytes per 32 elements); C8 and below have none.
    /// That remains a correlation, but now with a *plain-decode* defect — which
    /// makes it a `QWEN4EXP_KV_FACTORS` threshold question rather than anything
    /// to do with the draft head.
    ///
    /// This test exists so nobody re-attributes it to the drafter. It cost
    /// three wrong controls to establish: a substring filter that ran twelve
    /// models' gates, a run that aborted before it validated anything, and a
    /// gate whose generation was too short to reach the divergence at all.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_top_rung_divergence_vs_budget \
                -- --ignored --nocapture --test-threads=1"]
    fn test_top_rung_divergence_vs_budget() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let model = account_model_load(&device, || {
            let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
            Qwen4ExpBatched::new(gpu)
        })?;

        // The two rungs that drift, at the width they drift at, plus C8 as the
        // control that must stay exact at every budget.
        let configs: Vec<TestConfig> = [InferenceMode::C8, InferenceMode::C9, InferenceMode::C10]
            .into_iter()
            .map(|mode| TestConfig {
                mode,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect();

        let sound: Vec<TestConfig> = configs.iter().take(1).cloned().collect();
        let top: Vec<TestConfig> = configs.iter().skip(1).cloned().collect();

        let run = |set: &[TestConfig], budget: usize, label: &str| -> Result<bool> {
            println!("\n=== {label}, draft budget {budget} ===\n");
            let params = TestParams::new(256, &tok, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_int8mode(int8mode)
                .with_speculative(budget)
                .with_timeout_secs(3600);
            Ok(params.run_loaded(set.to_vec(), &model).is_ok())
        };

        // **C8 is the assertion.** A rung that plain decode holds must stay
        // exact at every depth — that is what lossless means, and it is the
        // only part of this test that can fail for speculation's sake.
        for budget in [0usize, 1, 2, 4] {
            assert!(
                run(&sound, budget, "C8 (sound rung)")?,
                "C8 diverged at draft budget {budget} — a rung plain decode holds exactly \
                 must survive speculation, so this is a rewind defect"
            );
        }

        // **C9/C10 are the record.** They drift from the reference around
        // 110-130 generated tokens under PLAIN DECODE, which is why the
        // standing gate — 64 tokens — has never seen it. Asserted the way it
        // measures rather than the way it ought to be, so that a future run
        // where speculation *does* make them worse is visible as a change.
        let mut top_ok: Vec<usize> = Vec::new();
        for budget in [0usize, 1, 2, 4] {
            if run(&top, budget, "C9/C10 (top rungs)")? {
                top_ok.push(budget);
            }
        }
        println!("\nbudgets at which C9/C10 held: {top_ok:?}  (expected: none)");
        assert!(
            top_ok.is_empty(),
            "C9/C10 held at budgets {top_ok:?} — they diverge under plain decode today, so \
             holding is a change worth looking at, not a pass"
        );
        Ok(())
    }

    /// **The draft-budget sweep.** Holds the width fixed and varies the depth,
    /// which is how the ladder's brackets were derived — distinct from
    /// [`test_speculative_ladder`], which holds the ladder and varies the rung.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_speculative_decode \
                -- --ignored --nocapture --test-threads=1"]
    fn test_speculative_decode() -> Result<()> {
        use crate::models::quantized_qwen35::tests::speculative_gate;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let device = Device::new_cuda(0)?;
        speculative_gate(
            "Qwen3.8-Flash-Next",
            int8mode,
            &[1, 4],
            &tok,
            QWEN38_MAX_DRAFT,
            &device,
            || {
                let device = Device::new_cuda(0)?;
                let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// **Depth**: the batched forward at 32K and 128K of KV.
    ///
    /// The ladder above reports this model at ~700 tokens, which is where its
    /// per-token cost is cleanest but says nothing about the property the
    /// engine is built for. This runs the same forward with a prompt deep
    /// enough to make the KV cache the dominant resident object, at three
    /// points on the compression ladder, so prefill rate, decode rate and
    /// compression can be read against depth rather than against width.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and runs a 128K-token prompt. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_long_context_scaling \
                -- --ignored --nocapture --test-threads=1"]
    fn test_long_context_scaling() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        long_context_gate(
            "Qwen3.8-Flash-Next (qwen4exp, 250B-A13B)",
            int8mode,
            &tok,
            Dialect::qwen35(),
            // `qwen4exp.context_length` in the merged engine GGUF.
            262_144,
            &[
                (
                    32_768,
                    &[InferenceMode::BF16, InferenceMode::C5, InferenceMode::C10][..],
                ),
                (131_072, &[InferenceMode::BF16, InferenceMode::C10][..]),
            ],
            1,
            64,
            DepthTask::Coherence,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// **Where does decode time go as the cache grows?**
    ///
    /// `test_long_context_scaling` measures two depths, which is enough to say
    /// that decode falls faster than it should (10.2 → 1.2 t/s across 4×) and
    /// not enough to say *which* part of the step is responsible. This walks a
    /// depth ladder at one compression rung so every span's cost can be read
    /// against depth as a curve: a term that is constant in depth stays flat, a
    /// term linear in depth doubles with each rung, and the one that does is
    /// the wall.
    ///
    /// BF16 only and a short generate, because the question is the *shape* of
    /// the decode step, not its absolute rate — and the prefill needed to reach
    /// 128K already dominates the run's wall clock.
    ///
    /// Run under `--features cuda,profile`, which is what fills the pipeline
    /// tables; without it the harness still reports throughput and the spans
    /// are inert.
    #[test]
    #[ignore = "profiling run: loads the ~124 GB merged engine GGUF and prefills to 128K. \
                Run with: cargo test --release --features cuda,profile \
                -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_decode_vs_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_decode_vs_depth() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let bf16 = &[InferenceMode::BF16][..];
        long_context_gate(
            "Qwen3.8-Flash-Next decode profile",
            int8mode,
            &tok,
            Dialect::qwen35(),
            262_144,
            &[
                (8_192, bf16),
                (16_384, bf16),
                (32_768, bf16),
                (65_536, bf16),
                (131_072, bf16),
            ],
            1,
            32,
            DepthTask::Coherence,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// The width ladder's task, run at the depth ladder's depths.
    ///
    /// [`profile_decode_vs_depth`] and [`test_parallel_batched_forwarding`]
    /// differ in THREE ways at once — depth, task, and whether QSA engages —
    /// so their decode rates cannot be read against each other, and the ~80 t/s
    /// of the width ladder against the ~29 t/s of the depth sweep is mostly the
    /// draft head accepting 4.85 tokens a step instead of 2.58 on a rewrite it
    /// can largely copy.
    ///
    /// This row holds the task fixed and varies only depth: the padding goes in
    /// FRONT of the same story the ladder rewrites, so acceptance stays in the
    /// ladder's regime while the prefix grows to 128K. It is the honest answer
    /// to "what does conversational-quality decode cost at depth", which
    /// neither of the other two tables gives on its own.
    #[test]
    #[ignore = "profiling run: loads the ~124 GB merged engine GGUF and prefills to 128K. \
                Run with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_story_rewrite_vs_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_story_rewrite_vs_depth() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let bf16 = &[InferenceMode::BF16][..];
        long_context_gate(
            "Qwen3.8-Flash-Next story rewrite",
            int8mode,
            &tok,
            Dialect::qwen35(),
            262_144,
            &[
                (8_192, bf16),
                (16_384, bf16),
                (32_768, bf16),
                (65_536, bf16),
                (131_072, bf16),
            ],
            1,
            // The ladder's generation length, so the speculative loop runs the
            // same number of steps per row as the table this is comparable to.
            64,
            DepthTask::Rewrite,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// The token-rate probe: the gate's harness over the three rungs that
    /// decide whether a change helped — one sequence, the widest BF16 fleet
    /// this card runs, and the compressed pair — and its performance table,
    /// in a few minutes rather than the full ladder's fifteen.
    ///
    /// **Rates only.** The transient tier's exactness is the gate's verdict
    /// ([`test_parallel_batched_forwarding`]), judged over every rung; asserted
    /// here it would fail on a slack no rate depends on and take the table with
    /// it, which is the one thing this run exists to print.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_token_rate \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_token_rate() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let params = TestParams::new(64, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_exact_tier(false)
            .with_timeout_secs(1800);
        let rung = |mode, n| TestConfig {
            mode,
            use_batched: true,
            num_contexts: n,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        };
        let configs = vec![
            rung(InferenceMode::BF16, 1),
            rung(InferenceMode::BF16, 8),
            rung(InferenceMode::C5, 2),
        ];
        params.run(configs, || {
            Qwen4ExpBatched::new(Qwen4ExpGpu::load(&merged, &device, int8mode)?)
        })
    }

    /// The cost of compression alone: BF16 ×1 and C10 ×1 **interleaved**, twice
    /// over, so neither mode always runs later in the process than the other.
    ///
    /// The gate runs its compressed rungs after its BF16 ones, ten minutes into
    /// a hot card, and there every kernel slows — the expert GEMMs, which never
    /// read K/V, by as much as the attention that does. Alternating the two
    /// modes puts each at both an early and a late position, so a gap that
    /// survives the interleave is compression's and one that follows position
    /// is the card's.
    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::compression_rate_interleaved \
                -- --ignored --nocapture --test-threads=1"]
    fn compression_rate_interleaved() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let params = TestParams::new(64, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_exact_tier(false)
            .with_timeout_secs(1800);
        let rung = |mode| TestConfig {
            mode,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        };
        let configs = vec![
            rung(InferenceMode::BF16),
            rung(InferenceMode::C10),
            rung(InferenceMode::BF16),
            rung(InferenceMode::C10),
        ];
        params.run(configs, || {
            Qwen4ExpBatched::new(Qwen4ExpGpu::load(&merged, &device, int8mode)?)
        })
    }

    #[test]
    #[ignore = "loads the ~124 GB merged engine GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_parallel_batched_forwarding \
                -- --ignored --nocapture --test-threads=1"]
    fn test_parallel_batched_forwarding() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        println!("\n=== Qwen3.8-Flash-Next (qwen4exp) batched forwarding ===\n");
        let merged = engine_gguf()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);

        let params = TestParams::new(64, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600);

        let configs = gate_ladder(quant_ladder::device_vram_gib(&device)?);

        let load = || {
            let gpu = Qwen4ExpGpu::load(&merged, &device, int8mode)?;
            // The checkpoint must still be the geometry this engine was
            // validated for — a silent architecture change fails here, not in
            // a kernel.
            assert_eq!(gpu.cfg.num_layers, 48);
            assert_eq!(gpu.cfg.attn_head_dim, 256);
            assert_eq!(gpu.cfg.moe.n_experts, 512);
            assert_eq!(gpu.cfg.moe.n_experts_used, 10);
            let m = Qwen4ExpBatched::new(gpu)?;
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

    /// The Phase-1 oracle gate: `forward_batched` end to end on the real
    /// checkpoint. Five checks —
    ///
    /// 1. the split loader + config against the frozen schema;
    /// 2. finite logits over real prompts, **batched** (two sessions in one
    ///    call — the oracle is batched from the start, §0.6);
    /// 3. a session's logits do not depend on its batch-mates (×1 ≡ ×N);
    /// 4. segmented forward ≡ one-shot — the three carried state classes
    ///    (GDN, PLE conv+hash window, QSA index cache) survive a boundary;
    /// 5. greedy continuation is deterministic, non-degenerate, and
    ///    semantically right ("The capital of France is" → Paris) — the
    ///    check that catches self-consistent-but-wrong geometry.
    #[test]
    #[ignore = "reads the pinned 188 GB Q8_0 split from the HF cache and runs the CPU \
                reference forward (on-demand expert reads; needs ~24 GB RAM). Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_forward_batched_oracle \
                -- --ignored --nocapture --test-threads=1"]
    fn test_forward_batched_oracle() -> Result<()> {
        println!("\n=== Qwen3.8-Flash-Next forward_batched oracle ===\n");
        let shards = pinned_shards()?;
        let device = Device::Cpu;
        let t_load = std::time::Instant::now();
        let model = oracle_from_gguf_path(&shards[0], &device)?;
        println!(
            "✓ loaded: {} layers ({} attention), hidden {}, {} experts top-{}, \
             hc {}×, PLE layer {} ({:.1}s)",
            model.cfg.num_layers,
            model.cfg.n_attention_layers(),
            model.cfg.hidden_size,
            model.cfg.moe.n_experts,
            model.cfg.moe.n_experts_used,
            model.cfg.hc.count,
            model.cfg.ple.layer,
            t_load.elapsed().as_secs_f32(),
        );

        let tok = tokenizer()?;
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(tok
                .encode(s, false)
                .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
                .get_ids()
                .to_vec())
        };
        let prompt_a = "The capital of France is";
        let prompt_b = "Water is made of hydrogen and";
        let a = encode(prompt_a)?;
        let b = encode(prompt_b)?;
        println!("prompts: {:?} tokens / {:?} tokens", a.len(), b.len());

        // (2) Batched prefill: two sessions, one call.
        let mut pair = [model.new_session()?, model.new_session()?];
        let t_prefill = std::time::Instant::now();
        let both = model.forward_batched(&[&a, &b], &mut pair)?;
        println!(
            "✓ batched prefill ({:.1}s)",
            t_prefill.elapsed().as_secs_f32()
        );
        let last_a = both[0].narrow(0, a.len() - 1, 1)?;
        let last_b = both[1].narrow(0, b.len() - 1, 1)?;
        for (name, l) in [("A", &last_a), ("B", &last_b)] {
            let m = l.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(m.is_finite(), "session {name}: non-finite logits");
        }

        // (3) ×1 ≡ ×N: session A alone must match session A in the pair.
        let mut solo = model.new_session()?;
        let alone = model
            .forward_batched(&[&a], std::slice::from_mut(&mut solo))?
            .remove(0);
        let d = alone
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "batching changed session A's logits by {d}");
        println!("✓ ×1 ≡ ×2 (max |Δ| = {d:.2e})");

        // (4) Segmented ≡ one-shot, across all three carried state classes.
        let mut seg = model.new_session()?;
        let cut = a.len() / 2;
        let s1 = model
            .forward_batched(&[&a[..cut]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let s2 = model
            .forward_batched(&[&a[cut..]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let stitched = Tensor::cat(&[s1, s2], 0)?;
        let d = stitched
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "segmented forward diverged from one-shot by {d}");
        println!("✓ segmented ≡ one-shot (max |Δ| = {d:.2e})");

        // (5) Greedy continuation, BOTH sessions stepped through
        // `forward_batched` together — decode is batched too.
        let argmax = |l: &Tensor| -> Result<u32> { l.get(0)?.argmax(0)?.to_scalar::<u32>() };
        let mut next = [argmax(&last_a)?, argmax(&last_b)?];
        let mut out = [vec![next[0]], vec![next[1]]];
        let t_dec = std::time::Instant::now();
        for _ in 0..8 {
            let seg_a = [next[0]];
            let seg_b = [next[1]];
            let ls = model.forward_batched(&[&seg_a, &seg_b], &mut pair)?;
            next = [argmax(&ls[0])?, argmax(&ls[1])?];
            out[0].push(next[0]);
            out[1].push(next[1]);
        }
        println!(
            "✓ greedy ×2 decode, 8 steps ({:.1}s)",
            t_dec.elapsed().as_secs_f32()
        );
        for (prompt, ids) in [(prompt_a, &out[0]), (prompt_b, &out[1])] {
            let text = tok
                .decode(ids, false)
                .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
            println!("{prompt}|{text}");
        }
        let text_a = tok
            .decode(&out[0], false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        assert!(
            text_a.contains("Paris"),
            "the model did not complete {prompt_a:?} with Paris — got {text_a:?}"
        );
        let distinct: HashSet<u32> = out[0].iter().copied().collect();
        assert!(distinct.len() >= 4, "degenerate continuation: {:?}", out[0]);
        println!("\n=== oracle gate green ===");
        Ok(())
    }

    /// The Q4_KO-expert recheck: convert the W4A16 release's experts into the
    /// pinned split (first run only — the sibling split is cached beside the
    /// Q8_0 shards), then run the oracle gate on the converted artifact and
    /// compare its last-position logits against the Q8_0 reference.
    ///
    /// What each half proves:
    /// - the conversion itself asserts **bit-exactness** per expert
    ///   (`dequant_ko` ≡ the AWQ release's own dequantization), so a green
    ///   conversion means our Q4_KO bytes ARE the calibrated AWQ weights;
    /// - the gate then proves the model still *works* at 4-bit experts —
    ///   Paris/oxygen, ×1 ≡ ×2, segmented ≡ one-shot — and the cross-model
    ///   comparison quantifies the Q8→AWQ-Q4 expert swap (informational: two
    ///   different quantizations of the same weights, no bit-equality to
    ///   expect; argmax agreement on the probe prompts is the meaningful bar).
    #[test]
    #[ignore = "downloads 73 GB of W4A16 expert shards on first run, writes the ~72 GB \
                Q4KOEXP sibling split beside the Q8_0 shards, and runs the CPU oracle \
                twice. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_forward_batched_oracle_q4ko_experts \
                -- --ignored --nocapture --test-threads=1"]
    fn test_forward_batched_oracle_q4ko_experts() -> Result<()> {
        use crate::models::qwen4exp::convert::convert_w4a16_experts;

        println!("\n=== Qwen3.8-Flash-Next Q4_KO-expert oracle recheck ===\n");
        let q8_shards = pinned_shards()?;
        let awq_shards: Vec<PathBuf> = QWEN4EXP_W4A16_EXPERT_SHARDS
            .iter()
            .map(|&i| {
                hf_get(
                    QWEN4EXP_W4A16_REPO,
                    RepoType::Model,
                    QWEN4EXP_W4A16_REV,
                    &w4a16_shard_name(i),
                )
            })
            .collect::<Result<_>>()?;

        // The sibling split lives beside the Q8_0 shards; conversion is
        // resumable and skipped entirely once all six outputs exist.
        let out_dir = q8_shards[0]
            .parent()
            .ok_or_else(|| candle::Error::Msg("q8 shard has no parent dir".into()))?
            .to_path_buf();
        let t_conv = std::time::Instant::now();
        // The conversion runs its fused repack kernel on the GPU; the oracle
        // itself stays on the CPU below.
        let cuda = Device::new_cuda(0)?;
        let converted = convert_w4a16_experts(&q8_shards, &awq_shards, &out_dir, &cuda)?;
        println!(
            "✓ Q4KOEXP split ready ({:.0}s): {:?}",
            t_conv.elapsed().as_secs_f32(),
            converted[0].file_name().unwrap_or_default()
        );

        let device = Device::Cpu;
        let model = oracle_from_gguf_path(&converted[0], &device)?;
        println!("✓ converted split loads through the standard path");

        let tok = tokenizer()?;
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(tok
                .encode(s, false)
                .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
                .get_ids()
                .to_vec())
        };
        let prompt_a = "The capital of France is";
        let prompt_b = "Water is made of hydrogen and";
        let a = encode(prompt_a)?;
        let b = encode(prompt_b)?;

        // The oracle gate's checks on the converted model.
        let mut pair = [model.new_session()?, model.new_session()?];
        let both = model.forward_batched(&[&a, &b], &mut pair)?;
        let last_a = both[0].narrow(0, a.len() - 1, 1)?;

        let mut solo = model.new_session()?;
        let alone = model
            .forward_batched(&[&a], std::slice::from_mut(&mut solo))?
            .remove(0);
        let d = alone
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "batching changed session A's logits by {d}");

        let mut seg = model.new_session()?;
        let cut = a.len() / 2;
        let s1 = model
            .forward_batched(&[&a[..cut]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let s2 = model
            .forward_batched(&[&a[cut..]], std::slice::from_mut(&mut seg))?
            .remove(0);
        let stitched = Tensor::cat(&[s1, s2], 0)?;
        let d = stitched
            .sub(&both[0])?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(d < 2e-3, "segmented forward diverged from one-shot by {d}");
        println!("✓ ×1 ≡ ×2 and segmented ≡ one-shot on Q4_KO experts");

        // Greedy continuation through batched decode — the semantic bar.
        let argmax = |l: &Tensor| -> Result<u32> { l.get(0)?.argmax(0)?.to_scalar::<u32>() };
        let mut next = [
            argmax(&last_a)?,
            argmax(&both[1].narrow(0, b.len() - 1, 1)?)?,
        ];
        let mut out = [vec![next[0]], vec![next[1]]];
        for _ in 0..8 {
            let (seg_a, seg_b) = ([next[0]], [next[1]]);
            let ls = model.forward_batched(&[&seg_a, &seg_b], &mut pair)?;
            next = [argmax(&ls[0])?, argmax(&ls[1])?];
            out[0].push(next[0]);
            out[1].push(next[1]);
        }
        for (prompt, ids) in [(prompt_a, &out[0]), (prompt_b, &out[1])] {
            let text = tok
                .decode(ids, false)
                .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
            println!("{prompt}|{text}");
        }
        if let Some(s) = model.ple_table.cache_stats() {
            println!(
                "PLE row cache: {} hits / {} misses ({:.1}% hit rate), {} evictions",
                s.hits,
                s.misses,
                s.hit_rate() * 100.0,
                s.evictions
            );
        }
        let text_a = tok
            .decode(&out[0], false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        assert!(
            text_a.contains("Paris"),
            "Q4_KO experts lost the completion: {text_a:?}"
        );

        // Cross-model comparison against the Q8_0 reference, same prompt.
        drop(model);
        let reference = oracle_from_gguf_path(&q8_shards[0], &device)?;
        let mut ref_sess = reference.new_session()?;
        let ref_logits = reference
            .forward_batched(&[&a], std::slice::from_mut(&mut ref_sess))?
            .remove(0);
        let ref_last = ref_logits.narrow(0, a.len() - 1, 1)?;
        let q4_last = alone.narrow(0, a.len() - 1, 1)?;
        let ref_arg = argmax(&ref_last)?;
        let q4_arg = argmax(&q4_last)?;
        let diff = q4_last
            .sub(&ref_last)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        println!(
            "Q8_0 vs Q4_KO experts, last position: argmax {} vs {} ({}), max |Δlogit| = {diff:.3}",
            ref_arg,
            q4_arg,
            if ref_arg == q4_arg {
                "agree"
            } else {
                "DISAGREE"
            },
        );
        assert_eq!(
            ref_arg, q4_arg,
            "the probe prompt's next token moved under the expert swap — \
             quantify before trusting the rung"
        );
        println!("\n=== Q4_KO-expert recheck green ===");
        Ok(())
    }

    /// QSA at depth: a prompt past the 2051-position selection width, so the
    /// 12 attention layers genuinely select instead of degenerating to dense.
    /// Determinism + finiteness here; token-level parity in the sparse regime
    /// is Phase 1's llama.cpp comparison.
    #[test]
    #[ignore = "reads the pinned 188 GB Q8_0 split and runs a >2051-token CPU prefill \
                (tens of minutes). Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_forward_batched_qsa_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn test_forward_batched_qsa_depth() -> Result<()> {
        let shards = pinned_shards()?;
        let device = Device::Cpu;
        let model = oracle_from_gguf_path(&shards[0], &device)?;
        let tok = tokenizer()?;

        let base = "The expedition kept meticulous notes on every specimen it catalogued. ";
        let mut long = String::new();
        while tok
            .encode(long.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .len()
            < 2100
        {
            long.push_str(base);
        }
        long.push_str("The very first sentence of these notes said the expedition kept");
        let ids = tok
            .encode(long.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let width = model.cfg.indexer.top_k + 4 - 1;
        assert!(ids.len() > width, "prompt too short to engage selection");
        println!("prompt: {} tokens (selection width {width})", ids.len());

        let run = || -> Result<u32> {
            let mut s = model.new_session()?;
            let l = model
                .forward_batched(&[&ids], std::slice::from_mut(&mut s))?
                .remove(0);
            let last = l.narrow(0, ids.len() - 1, 1)?;
            let m = last.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(m.is_finite(), "non-finite logits under QSA selection");
            last.get(0)?.argmax(0)?.to_scalar::<u32>()
        };
        let t = std::time::Instant::now();
        let first = run()?;
        println!(
            "✓ deep prefill ({:.0}s); next token id {first} = {:?}",
            t.elapsed().as_secs_f32(),
            tok.decode(&[first], false).ok()
        );
        let second = run()?;
        assert_eq!(first, second, "selection at depth is nondeterministic");
        Ok(())
    }
}
