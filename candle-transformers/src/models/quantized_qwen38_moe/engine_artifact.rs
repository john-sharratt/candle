//! The engine artifact each card builds, and the recipe that names it.
//!
//! A card's expert width comes from the quantisation ladder: Q4_KO where the
//! bit-exact import applies, a narrower rung requantised from the Q8_0 split, and
//! the split's own Q8_0 above every rung. The trunk, the n-gram table and the draft
//! head's dense weights are Q8_0 on every rung. The recipe's digest names the
//! artifact, so a pack built from another recipe is another pack, while the model
//! digest is common to every rung, so a substrate written under one opens under
//! another. Preparing an artifact fetches about 190 GB of pinned sources, so a
//! probe resolves an artifact that is already built and never prepares one.

use super::*;

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

/// The model every rung's engine artifact holds, whatever its expert width:
/// the release's trunk and draft-head sources ([`Recipe::model_digest`]). What
/// names Flash-Next to a substrate, so one written under one rung opens under
/// another.
pub fn engine_model_digest() -> [u8; 32] {
    engine_recipe_at(None).model_digest()
}

/// Where engine artifacts are prepared, and their packs live: the model cache,
/// under this repo's folder.
pub fn engine_artifact_dir() -> PathBuf {
    cache_root().join(QWEN4EXP_REPO.replace('/', "--"))
}

/// The model-pack request for the engine whose routed experts are at `experts`
/// (`None` for the split's own `Q8_0`) and whose sections are packed for
/// `mode` (`None` for the mode this card picks).
///
/// The checkpoint is the recipe's artifact, and its "revision" is the recipe's
/// digest — so a pack built from another recipe is another pack.
pub fn engine_request(experts: Option<GgmlDType>, mode: Option<Int8Mode>) -> PackRequest {
    let recipe = engine_recipe_at(experts);
    PackRequest::of(
        PackFamily::Routed,
        (QWEN4EXP_REPO, &recipe.digest(), &recipe.artifact_name()),
        (TOKENIZER_REPO, TOKENIZER_REV),
        mode,
    )
}

/// A recipe store that fetches nothing: an artifact missing from the model
/// cache is reported as the build step it is, never downloaded.
pub(super) struct PreparedOnly;

impl SourceStore for PreparedOnly {
    fn fetch(&self, file: &SourceFile) -> Result<PathBuf> {
        candle::bail!(
            "the Flash-Next engine is not prepared on this machine and {} is not here to \
             prepare it from — the Flash-Next gate builds it \
             (quantized_qwen38_moe::tests::test_parallel_batched_forwarding)",
            file.path
        )
    }

    fn cached_copies(&self, _: &SourceFile) -> Vec<PathBuf> {
        Vec::new()
    }
}

/// This card's engine pack — resolved, or built from an artifact already
/// prepared here, but **never prepared**: that fetches ~190 GB of pinned
/// sources, which a probe must not start as a side effect of asking for a path.
/// The gate (`tests::test_parallel_batched_forwarding`) is what prepares it.
#[cfg(feature = "cuda")]
pub fn prepared_engine_pack(
    device: &Device,
    mode: Option<Int8Mode>,
    tokenizer: &dyn Fn(&str, &str) -> Result<String>,
) -> Result<PathBuf> {
    let experts = quant_ladder::expert_format(quant_ladder::device_vram_gib(device)?);
    let fetch = EngineFetch {
        recipe: engine_recipe_at(experts),
        dir: engine_artifact_dir(),
        store: &PreparedOnly,
        tokenizer,
        device,
    };
    model_pack(
        &engine_request(experts, mode),
        &cache_root(),
        device,
        &fetch,
    )
}

/// This card's engine pack at `mode`, from the model cache — prepared from the
/// pinned sources when neither the pack nor the artifact is there.
///
/// **An artifact already prepared is only resolved.** `prepare_engine`
/// releases the recipe's sources whenever its store can find them, which is
/// right for a build and wrong here: the oracle gates read the same ~188 GB
/// Q8_0 split, so a gate that only needed the artifact would delete the split
/// and the next oracle gate would download it again. A store that fetches
/// nothing also releases nothing.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn engine_pack(device: &Device, mode: Option<Int8Mode>) -> Result<PathBuf> {
    use crate::models::batch_test::test_helpers::{hf_get, HfSourceStore};
    use crate::models::qwen4exp::prepare::prepared;
    use hf_hub::RepoType;

    let experts = quant_ladder::expert_format(quant_ladder::device_vram_gib(device)?);
    let recipe = engine_recipe_at(experts);
    let dir = engine_artifact_dir();
    let store: &dyn SourceStore = if prepared(&recipe, &dir)?.is_some() {
        &PreparedOnly
    } else {
        &HfSourceStore
    };
    let tokenizer = |repo: &str, rev: &str| -> Result<String> {
        let p = hf_get(repo, RepoType::Model, rev, "tokenizer.json")?;
        std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
    };
    let fetch = EngineFetch {
        recipe,
        dir,
        store,
        tokenizer: &tokenizer,
        device,
    };
    model_pack(
        &engine_request(experts, mode),
        &cache_root(),
        device,
        &fetch,
    )
}
