//! Resolve a prepared engine artifact, or build it.
//!
//! [`prepared`] answers from disk alone: the artifact named by the recipe,
//! present, carrying the recipe's stamp, with its directory consistent with the
//! recipe (see [`verify`]). When that holds no source file is fetched.
//!
//! [`prepare_engine`] builds it otherwise:
//!
//! 1. fetch every pinned source and check its length and SHA-256;
//! 2. convert the draft head's experts to the recipe's width;
//! 3. for an AWQ-import recipe, import the W4A16 experts into a sibling split;
//! 4. merge the trunk split (and the head) into one GGUF, requantizing the
//!    trunk's experts as they stream when the recipe says so, and stamping the
//!    recipe into the metadata;
//! 5. read the result's header back and verify it;
//! 6. delete the sources and the intermediates — the artifact carries
//!    everything the engine reads, the n-gram table included.
//!
//! Every step writes to a temporary name and renames on success, so an
//! interrupted build is resumed rather than mistaken for a finished one. Step 6
//! also runs whenever a finished artifact is found, so a build interrupted
//! between the merge and the cleanup does not leave its sources behind.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle::quantized::gguf_file::{Content, Value};
use candle::quantized::GgmlDType;
use candle::{Device, Result};

use super::recipe::{ExpertSource, Recipe, SourceFile, SourceRole, TAG_HEX};
use super::requant::requant_experts;
use super::store::{fetch_verified, release, SourceStore};
use crate::models::qwen4exp::convert::{
    convert_mtp_sidecar, convert_w4a16_experts, is_expert_tensor, merge_gguf_files, TensorRewrite,
    Verbatim,
};

const BLOCK_COUNT_KEY: &str = "qwen4exp.block_count";
const NEXTN_KEY: &str = "qwen4exp.nextn_predict_layers";

/// The three routed-expert projections every block carries.
const EXPERT_PROJECTIONS: u32 = 3;

/// The recipe's artifact in `dir`, when it is present and verifies.
///
/// A file under the recipe's name that does not verify is an error rather than
/// a miss: the name carries the recipe's digest, so a mismatch there is damage,
/// not an old build, and rebuilding over it would hide that.
pub fn prepared(recipe: &Recipe, dir: &Path) -> Result<Option<PathBuf>> {
    let path = dir.join(recipe.artifact_name());
    if !path.exists() {
        return Ok(None);
    }
    let content = Content::read(&mut File::open(&path)?)?;
    verify(recipe, &path, &content)?;
    Ok(Some(path))
}

/// The recipe's artifact in `dir`, built from `store` when it is not there.
pub fn prepare_engine(
    recipe: &Recipe,
    dir: &Path,
    store: &dyn SourceStore,
    device: &Device,
) -> Result<PathBuf> {
    if let Some(path) = prepared(recipe, dir)? {
        clean_up(recipe, dir, store)?;
        return Ok(path);
    }
    std::fs::create_dir_all(dir)?;
    let tag = &recipe.digest()[..TAG_HEX];
    let t = Instant::now();

    let all: Vec<&SourceFile> = recipe.sources.iter().collect();
    let paths = fetch_verified(store, &all)?;
    let path_of = |role: SourceRole| -> Vec<PathBuf> {
        all.iter()
            .zip(&paths)
            .filter(|(s, _)| s.role == role)
            .map(|(_, p)| p.clone())
            .collect()
    };
    let mut trunk = path_of(SourceRole::Trunk);
    trunk.sort();
    let heads = path_of(SourceRole::DraftHead);
    let imports = path_of(SourceRole::ExpertImport);
    tracing::info!(
        sources = all.len(),
        secs = t.elapsed().as_secs_f32(),
        "engine sources fetched and verified"
    );

    let mut splits = match recipe.expert_source {
        ExpertSource::AwqImport => {
            convert_w4a16_experts(&trunk, &imports, &awq_dir(dir, tag), device)?
        }
        ExpertSource::Verbatim | ExpertSource::Requantized => trunk.clone(),
    };

    let mut overrides: Vec<(String, Value)> = recipe.stamp().to_vec();
    if let [head] = &heads[..] {
        // The trunk's shards declare its blocks and no head; the merged file
        // has one more, and `nextn_predict_layers` is what makes the config
        // parser split trunk from head (`num_layers = block_count − nextn`).
        // The head file must name its block exactly there, or its tensors land
        // under a prefix the loader never reads.
        let trunk_blocks = u32_meta(
            &Content::read(&mut File::open(&trunk[0])?)?,
            BLOCK_COUNT_KEY,
        )?;
        let head_content = Content::read(&mut File::open(head)?)?;
        let head_at = u32_meta(&head_content, BLOCK_COUNT_KEY)?
            .checked_sub(u32_meta(&head_content, NEXTN_KEY)?)
            .ok_or_else(|| candle::Error::Msg(format!("{head:?}: nextn exceeds block_count")))?;
        if head_at != trunk_blocks {
            candle::bail!(
                "draft head {head:?} sits at blk.{head_at}, the trunk has {trunk_blocks} blocks"
            );
        }
        let converted = mtp_path(dir, tag);
        convert_mtp_sidecar(head, &converted, recipe.head_experts, device)?;
        splits.push(converted);
        overrides.push((BLOCK_COUNT_KEY.to_string(), Value::U32(trunk_blocks + 1)));
        overrides.push((NEXTN_KEY.to_string(), Value::U32(1)));
    }

    let dst = dir.join(recipe.artifact_name());
    let requant = ExpertRequant {
        target: recipe.experts,
        device,
    };
    let rewrite: &dyn TensorRewrite = match recipe.expert_source {
        ExpertSource::Requantized => &requant,
        ExpertSource::Verbatim | ExpertSource::AwqImport => &Verbatim,
    };
    merge_gguf_files(&splits, &dst, &overrides, rewrite)?;

    let content = Content::read(&mut File::open(&dst)?)?;
    if let Err(e) = verify(recipe, &dst, &content) {
        std::fs::remove_file(&dst)?;
        return Err(e);
    }
    let freed = clean_up(recipe, dir, store)?;
    tracing::info!(
        artifact = %dst.display(),
        gib = std::fs::metadata(&dst)?.len() as f64 / (1u64 << 30) as f64,
        sources_freed_gib = freed as f64 / (1u64 << 30) as f64,
        secs = t.elapsed().as_secs_f32(),
        "engine artifact built"
    );
    Ok(dst)
}

/// Where the AWQ import writes its sibling split for the recipe tagged `tag`.
fn awq_dir(dir: &Path, tag: &str) -> PathBuf {
    dir.join(format!("awq-{tag}"))
}

/// Where the converted draft head is written for the recipe tagged `tag`.
fn mtp_path(dir: &Path, tag: &str) -> PathBuf {
    dir.join(format!("mtp-{tag}.gguf"))
}

/// Delete the recipe's sources and intermediates. Idempotent — what is already
/// gone is skipped — so it runs on every resolve, not only after a build.
/// Returns the source bytes freed.
fn clean_up(recipe: &Recipe, dir: &Path, store: &dyn SourceStore) -> Result<u64> {
    let all: Vec<&SourceFile> = recipe.sources.iter().collect();
    let freed = release(store, &all)?;
    let tag = &recipe.digest()[..TAG_HEX];
    let awq = awq_dir(dir, tag);
    if awq.is_dir() {
        std::fs::remove_dir_all(&awq)?;
    }
    let mtp = mtp_path(dir, tag);
    for p in [mtp.clone(), mtp.with_extension("gguf.tmp")] {
        if p.exists() {
            std::fs::remove_file(&p)?;
        }
    }
    Ok(freed)
}

/// Requantizes the trunk's routed experts to `target` as the merge streams them.
///
/// Only a non-KO source is requantized: an expert already in a KO format was
/// placed there by an earlier step at the width the recipe asked of it (the
/// draft head's, converted on its own), and quantizing it again would change it.
struct ExpertRequant<'a> {
    target: GgmlDType,
    device: &'a Device,
}

impl TensorRewrite for ExpertRequant<'_> {
    fn target(&self, name: &str, _: &[usize], dtype: GgmlDType) -> Option<GgmlDType> {
        (is_expert_tensor(name) && !dtype.is_ko() && dtype != self.target).then_some(self.target)
    }

    fn emit(
        &self,
        _: &str,
        dims: &[usize],
        dtype: GgmlDType,
        src: &[u8],
        target: GgmlDType,
    ) -> Result<Vec<u8>> {
        requant_experts(src, dims, dtype, target, self.device)
    }
}

fn u32_meta(content: &Content, key: &str) -> Result<u32> {
    match content.metadata.get(key) {
        Some(Value::U32(v)) => Ok(*v),
        other => candle::bail!("engine artifact: {key} is {other:?}, expected U32"),
    }
}

/// The checks a finished artifact must pass, from its header:
///
/// - the recipe's stamp;
/// - every block's three routed-expert projections present, at the width the
///   recipe names (the draft head's block at the head's);
/// - when the file declares a draft head, the head's own input projection at
///   the head's block;
/// - a file long enough to hold every tensor its directory lists.
fn verify(recipe: &Recipe, path: &Path, content: &Content) -> Result<()> {
    if !recipe.stamped_in(content) {
        candle::bail!(
            "engine artifact {path:?} does not carry recipe {} — it was built from something \
             else under this recipe's name",
            recipe.digest()
        );
    }
    let blocks = u32_meta(content, BLOCK_COUNT_KEY)?;
    let nextn = match content.metadata.get(NEXTN_KEY) {
        Some(Value::U32(v)) => *v,
        _ => 0,
    };
    let head = blocks.checked_sub(nextn).ok_or_else(|| {
        candle::Error::Msg(format!(
            "engine artifact {path:?}: nextn {nextn} > blocks {blocks}"
        ))
    })?;
    let head_prefix = format!("blk.{head}.");
    if nextn > 0
        && !content
            .tensor_infos
            .contains_key(&format!("{head_prefix}nextn.eh_proj.weight"))
    {
        candle::bail!("engine artifact {path:?} declares a draft head but has no blk.{head}.nextn");
    }
    let mut end = 0u64;
    let mut experts = 0u32;
    for (name, info) in &content.tensor_infos {
        let bytes = (info.shape.elem_count() / info.ggml_dtype.block_size()
            * info.ggml_dtype.type_size()) as u64;
        end = end.max(info.offset + bytes);
        if !is_expert_tensor(name) {
            continue;
        }
        experts += 1;
        let want = if nextn > 0 && name.starts_with(&head_prefix) {
            recipe.head_experts
        } else {
            recipe.experts
        };
        if info.ggml_dtype != want {
            candle::bail!(
                "engine artifact {path:?}: {name} is {:?}, the recipe names {want:?}",
                info.ggml_dtype
            );
        }
    }
    if experts != EXPERT_PROJECTIONS * blocks {
        candle::bail!(
            "engine artifact {path:?}: {experts} expert tensors for {blocks} blocks — every \
             block carries {EXPERT_PROJECTIONS}"
        );
    }
    let len = std::fs::metadata(path)?.len();
    if len < content.tensor_data_offset + end {
        candle::bail!(
            "engine artifact {path:?} is {len} bytes, its directory needs {}",
            content.tensor_data_offset + end
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::recipe::hex;
    use super::super::store::sha256_file;
    use super::*;
    use candle::quantized::gguf_file;
    use candle::quantized::QTensor;
    use candle::Tensor;
    use half::f16;
    use std::collections::HashSet;

    /// A store over one directory: every source is fetched from, and cached
    /// only in, `dir/<path>`.
    struct DirStore(PathBuf);

    impl SourceStore for DirStore {
        fn fetch(&self, file: &SourceFile) -> Result<PathBuf> {
            Ok(self.0.join(file.path))
        }
        fn cached_copies(&self, file: &SourceFile) -> Vec<PathBuf> {
            vec![self.0.join(file.path)]
        }
    }

    fn f32_tensor(dims: &[usize], seed: f32) -> QTensor {
        let n: usize = dims.iter().product();
        let v: Vec<f32> = (0..n).map(|i| seed + i as f32 * 0.5).collect();
        QTensor::quantize(
            &Tensor::from_vec(v, dims, &Device::Cpu).unwrap(),
            GgmlDType::F32,
        )
        .unwrap()
    }

    /// Write a GGUF of `blocks` blocks (plus an `extra` tensor), each with three
    /// `[2, 32, 32]` F32 expert tensors; a nonzero `nextn` puts a NextN input
    /// projection on the block `blocks − nextn` and declares it.
    fn write_gguf(path: &Path, blocks: u32, nextn: u32, extra: &str) {
        let mut owned: Vec<(String, QTensor)> = Vec::new();
        for b in 0..blocks {
            for p in ["gate", "up", "down"] {
                owned.push((
                    format!("blk.{b}.ffn_{p}_exps.weight"),
                    f32_tensor(&[2, 32, 32], b as f32),
                ));
            }
        }
        if nextn > 0 {
            owned.push((
                format!("blk.{}.nextn.eh_proj.weight", blocks - nextn),
                f32_tensor(&[32, 64], 9.0),
            ));
        }
        owned.push((extra.to_string(), f32_tensor(&[32], 1.0)));
        let arch = Value::String("qwen4exp".into());
        let bc = Value::U32(blocks);
        let nx = Value::U32(nextn);
        let meta: Vec<(&str, &Value)> = vec![
            ("general.architecture", &arch),
            (BLOCK_COUNT_KEY, &bc),
            (NEXTN_KEY, &nx),
        ];
        let tensors: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut f = File::create(path).unwrap();
        gguf_file::write(&mut f, &meta, &tensors).unwrap();
    }

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    /// A Verbatim recipe over files already in `dir`.
    fn recipe_over(dir: &Path, trunk: &[&str], head: &str) -> Recipe {
        let file = |role, name: &str| {
            let p = dir.join(name);
            SourceFile {
                role,
                repo: "org/model",
                revision: "abc",
                path: leak(name.to_string()),
                bytes: std::fs::metadata(&p).unwrap().len(),
                sha256: leak(sha256_file(&p).unwrap()),
            }
        };
        let mut sources: Vec<SourceFile> =
            trunk.iter().map(|n| file(SourceRole::Trunk, n)).collect();
        sources.push(file(SourceRole::DraftHead, head));
        Recipe {
            sources,
            trunk: GgmlDType::Q8_0,
            head_dense: GgmlDType::Q8_0,
            experts: GgmlDType::F32,
            expert_source: ExpertSource::Verbatim,
            head_experts: GgmlDType::F32,
            converter_version: 1,
        }
    }

    /// End to end on the Verbatim rung: two trunk shards and a head become one
    /// stamped artifact holding every block's experts and the head's block, the
    /// sources are deleted, and a second resolve answers from disk.
    #[test]
    fn a_verbatim_recipe_builds_verifies_and_releases_its_sources() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&src).unwrap();
        // Trunk: two shards. The first carries blocks 0 and 1 and the block
        // count, as a split's first shard does; the second carries only a
        // non-expert tensor, as the n-gram table's shard does.
        write_gguf(&src.join("a.gguf"), 2, 0, "token_embd.weight");
        {
            let t = f32_tensor(&[32], 2.0);
            let bc = Value::U32(2);
            let meta: Vec<(&str, &Value)> = vec![(BLOCK_COUNT_KEY, &bc)];
            let mut f = File::create(src.join("b.gguf")).unwrap();
            gguf_file::write(&mut f, &meta, &[("output.weight", &t)]).unwrap();
        }
        // The head: one block past the trunk's two.
        {
            let mut owned: Vec<(String, QTensor)> = Vec::new();
            for p in ["gate", "up", "down"] {
                owned.push((
                    format!("blk.2.ffn_{p}_exps.weight"),
                    f32_tensor(&[2, 32, 32], 7.0),
                ));
            }
            owned.push((
                "blk.2.nextn.eh_proj.weight".into(),
                f32_tensor(&[32, 64], 9.0),
            ));
            owned.push(("token_embd.weight".into(), f32_tensor(&[32], 3.0)));
            let bc = Value::U32(3);
            let nx = Value::U32(1);
            let meta: Vec<(&str, &Value)> = vec![(BLOCK_COUNT_KEY, &bc), (NEXTN_KEY, &nx)];
            let tensors: Vec<(&str, &QTensor)> =
                owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
            let mut f = File::create(src.join("head.gguf")).unwrap();
            gguf_file::write(&mut f, &meta, &tensors).unwrap();
        }
        let recipe = recipe_over(&src, &["a.gguf", "b.gguf"], "head.gguf");
        let store = DirStore(src.clone());

        let built = prepare_engine(&recipe, &out, &store, &Device::Cpu).unwrap();
        assert_eq!(built, out.join(recipe.artifact_name()));
        let content = Content::read(&mut File::open(&built).unwrap()).unwrap();
        assert!(recipe.stamped_in(&content));
        assert!(matches!(
            content.metadata.get(BLOCK_COUNT_KEY),
            Some(Value::U32(3))
        ));
        assert!(matches!(
            content.metadata.get(NEXTN_KEY),
            Some(Value::U32(1))
        ));
        let names: HashSet<&str> = content.tensor_infos.keys().map(|s| s.as_str()).collect();
        for b in 0..3 {
            assert!(names.contains(format!("blk.{b}.ffn_up_exps.weight").as_str()));
        }
        assert!(names.contains("blk.2.nextn.eh_proj.weight"));
        // The head's copy of the embedding is the trunk's and was dropped.
        assert_eq!(content.tensor_infos["token_embd.weight"].shape.dims(), [32]);
        for f in ["a.gguf", "b.gguf", "head.gguf"] {
            assert!(!src.join(f).exists(), "{f} was not released");
        }
        let leftovers: Vec<_> = std::fs::read_dir(&out).unwrap().collect();
        assert_eq!(leftovers.len(), 1, "only the artifact remains in {out:?}");

        // Resolved from disk: the sources are gone, so a fetch would fail.
        assert_eq!(prepared(&recipe, &out).unwrap(), Some(built.clone()));
        assert_eq!(
            prepare_engine(&recipe, &out, &store, &Device::Cpu).unwrap(),
            built
        );
    }

    /// A build interrupted after its merge left the sources on disk; the next
    /// resolve finds the artifact and releases them.
    #[test]
    fn a_resolve_releases_sources_a_finished_build_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&src).unwrap();
        write_gguf(&src.join("a.gguf"), 1, 0, "token_embd.weight");
        {
            let mut owned: Vec<(String, QTensor)> = Vec::new();
            for p in ["gate", "up", "down"] {
                owned.push((
                    format!("blk.1.ffn_{p}_exps.weight"),
                    f32_tensor(&[2, 32, 32], 7.0),
                ));
            }
            owned.push((
                "blk.1.nextn.eh_proj.weight".into(),
                f32_tensor(&[32, 64], 9.0),
            ));
            let bc = Value::U32(2);
            let nx = Value::U32(1);
            let meta: Vec<(&str, &Value)> = vec![(BLOCK_COUNT_KEY, &bc), (NEXTN_KEY, &nx)];
            let tensors: Vec<(&str, &QTensor)> =
                owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
            let mut f = File::create(src.join("head.gguf")).unwrap();
            gguf_file::write(&mut f, &meta, &tensors).unwrap();
        }
        let recipe = recipe_over(&src, &["a.gguf"], "head.gguf");
        // Keep copies so the "interrupted" state can be put back.
        let keep = dir.path().join("keep");
        std::fs::create_dir_all(&keep).unwrap();
        for f in ["a.gguf", "head.gguf"] {
            std::fs::copy(src.join(f), keep.join(f)).unwrap();
        }
        let store = DirStore(src.clone());
        prepare_engine(&recipe, &out, &store, &Device::Cpu).unwrap();
        for f in ["a.gguf", "head.gguf"] {
            std::fs::copy(keep.join(f), src.join(f)).unwrap();
        }
        prepare_engine(&recipe, &out, &store, &Device::Cpu).unwrap();
        for f in ["a.gguf", "head.gguf"] {
            assert!(!src.join(f).exists(), "{f} survived a resolve");
        }
    }

    /// A head file whose block is not the one past the trunk is refused before
    /// anything is written.
    #[test]
    fn a_head_at_the_wrong_block_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&src).unwrap();
        write_gguf(&src.join("a.gguf"), 2, 0, "token_embd.weight");
        // Declares its head at blk.4 against a two-block trunk.
        write_gguf(&src.join("head.gguf"), 5, 1, "output.weight");
        let recipe = recipe_over(&src, &["a.gguf"], "head.gguf");
        let err = prepare_engine(&recipe, &out, &DirStore(src.clone()), &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("sits at blk.4, the trunk has 2 blocks"),
            "{err}"
        );
        assert!(!out.join(recipe.artifact_name()).exists());
    }

    /// `verify` against hand-made headers: each check names what it refuses.
    #[test]
    fn verify_refuses_each_inconsistency() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_gguf(&src.join("a.gguf"), 1, 0, "x.weight");
        write_gguf(&src.join("h.gguf"), 2, 1, "y.weight");
        let recipe = recipe_over(&src, &["a.gguf"], "h.gguf");
        let stamped = |blocks: u32, nextn: u32, with_head: bool, dtype: GgmlDType| {
            let p = dir
                .path()
                .join(format!("v{blocks}{nextn}{with_head}{dtype:?}.gguf"));
            let mut owned: Vec<(String, QTensor)> = Vec::new();
            for b in 0..blocks {
                for pj in ["gate", "up", "down"] {
                    let t = f32_tensor(&[2, 32, 32], 0.0);
                    let t = if dtype == GgmlDType::F16 {
                        QTensor::quantize(&t.dequantize(&Device::Cpu).unwrap(), GgmlDType::F16)
                            .unwrap()
                    } else {
                        t
                    };
                    owned.push((format!("blk.{b}.ffn_{pj}_exps.weight"), t));
                }
            }
            if with_head {
                owned.push((
                    format!("blk.{}.nextn.eh_proj.weight", blocks - nextn),
                    f32_tensor(&[32, 64], 0.0),
                ));
            }
            let [(k1, v1), (k2, v2)] = recipe.stamp();
            let bc = Value::U32(blocks);
            let nx = Value::U32(nextn);
            let meta: Vec<(&str, &Value)> = vec![
                (k1.as_str(), &v1),
                (k2.as_str(), &v2),
                (BLOCK_COUNT_KEY, &bc),
                (NEXTN_KEY, &nx),
            ];
            let tensors: Vec<(&str, &QTensor)> =
                owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
            let mut f = File::create(&p).unwrap();
            gguf_file::write(&mut f, &meta, &tensors).unwrap();
            let c = Content::read(&mut File::open(&p).unwrap()).unwrap();
            verify(&recipe, &p, &c).map_err(|e| e.to_string())
        };
        assert!(stamped(2, 1, true, GgmlDType::F32).is_ok());
        assert!(stamped(2, 0, false, GgmlDType::F32).is_ok());
        assert!(stamped(2, 1, false, GgmlDType::F32)
            .unwrap_err()
            .contains("has no blk.1.nextn"));
        assert!(stamped(2, 1, true, GgmlDType::F16)
            .unwrap_err()
            .contains("the recipe names F32"));
        assert!(stamped(1, 2, false, GgmlDType::F32)
            .unwrap_err()
            .contains("nextn 2 > blocks 1"));
    }

    /// The merge must write exactly what its rewrite emits, at the dtype and
    /// length the directory promised — a rewrite that emits F16 for F32
    /// experts yields F16 tensors of half the bytes, and verbatim tensors stay.
    #[test]
    fn a_merge_rewrite_changes_only_what_it_targets() {
        struct ToF16;
        impl TensorRewrite for ToF16 {
            fn target(&self, name: &str, _: &[usize], dtype: GgmlDType) -> Option<GgmlDType> {
                (is_expert_tensor(name) && dtype == GgmlDType::F32).then_some(GgmlDType::F16)
            }
            fn emit(
                &self,
                _: &str,
                _: &[usize],
                _: GgmlDType,
                src: &[u8],
                _: GgmlDType,
            ) -> Result<Vec<u8>> {
                Ok(src
                    .chunks_exact(4)
                    .flat_map(|b| {
                        f16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes()
                    })
                    .collect())
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.gguf");
        write_gguf(&a, 1, 0, "norm.weight");
        let dst = dir.path().join("m.gguf");
        merge_gguf_files(&[a], &dst, &[], &ToF16).unwrap();
        let c = Content::read(&mut File::open(&dst).unwrap()).unwrap();
        assert_eq!(
            c.tensor_infos["blk.0.ffn_gate_exps.weight"].ggml_dtype,
            GgmlDType::F16
        );
        assert_eq!(c.tensor_infos["norm.weight"].ggml_dtype, GgmlDType::F32);
        let mut f = File::open(&dst).unwrap();
        let t = c
            .tensor(&mut f, "blk.0.ffn_gate_exps.weight", &Device::Cpu)
            .unwrap();
        let v: Vec<f32> = t
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(&v[..3], &[0.0, 0.5, 1.0]);
        let n = c.tensor(&mut f, "norm.weight", &Device::Cpu).unwrap();
        let v: Vec<f32> = n.dequantize(&Device::Cpu).unwrap().to_vec1().unwrap();
        assert_eq!(&v[..2], &[1.0, 1.5]);
    }

    /// The requantizer leaves an expert already in a KO format alone — the
    /// draft head's, converted on its own — and non-experts always.
    #[test]
    fn expert_requant_targets_only_non_ko_experts() {
        let r = ExpertRequant {
            target: GgmlDType::Q2_KO,
            device: &Device::Cpu,
        };
        let e = "blk.3.ffn_up_exps.weight";
        assert_eq!(r.target(e, &[], GgmlDType::Q8_0), Some(GgmlDType::Q2_KO));
        assert_eq!(r.target(e, &[], GgmlDType::Q3_KO), None);
        assert_eq!(r.target(e, &[], GgmlDType::Q2_KO), None);
        assert_eq!(r.target("blk.3.attn_q.weight", &[], GgmlDType::Q8_0), None);
    }

    #[test]
    fn the_intermediate_paths_carry_the_tag() {
        let d = Path::new("root");
        assert_eq!(awq_dir(d, "0123abcd4567"), d.join("awq-0123abcd4567"));
        assert_eq!(mtp_path(d, "0123abcd4567"), d.join("mtp-0123abcd4567.gguf"));
        // `hex` is the same encoding the tag is cut from.
        assert_eq!(hex(&[0x01, 0x23]), "0123");
    }
}
