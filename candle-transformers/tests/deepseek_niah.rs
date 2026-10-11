//! Step-6 needle-in-a-haystack recall for DeepSeek-V4-Flash.
//!
//! Requires `--features cuda,ruler-bench`. It drives the RULER generator from
//! `models::batch_test`, which the library gates on
//! `any(test, feature = "ruler-bench")` — and the library's `test` cfg is NOT
//! active when building an integration test, so only the feature can reach it.
//! Without the gate below this file fails to COMPILE under a plain
//! `--features cuda`, taking the whole test target with it.
//!
//! Plants one "magic number" needle at a random depth inside a long filler
//! haystack and checks the model recovers it — the retrieval gate for the
//! provenance corpus (FloatGallery + BDP recall + Indexer top-k). The prompt
//! runs at a context length that far exceeds the fixed `window_size`, so the
//! needle can ONLY be found through the compressed corpus, not the sliding
//! window: this is the §L "addressable context grows, attended set stays
//! bounded" property under test end-to-end.
//!
//! Ignored (loads DeepSeek-V4-Flash on CUDA + per-token prefill is minutes at
//! a few-thousand-token context).

#![cfg(all(feature = "cuda", feature = "ruler-bench"))]

use std::path::{Path, PathBuf};

use candle::quantized::Int8Mode;
use candle::{Device, Result};
use candle_transformers::models::batch_test::ruler_gen::{
    generate_ruler_samples, run_ruler_eval, score_ruler_sample, RulerSample, RulerTask,
};
use candle_transformers::models::deepseek4::DEEPSEEK_V4;
use candle_transformers::models::latent_moe::{BatchedEngine, Engine};
use candle_transformers::models::model_pack::{
    cache_root, local_rev, model_pack, HubFetch, LocalFetch, ModelPack, PackFamily, PackRequest,
};

const KO_DIR: &str = r"D:\models\deepseek-v4-flash-mxfp4";
const KO_FILE: &str = "DeepSeek-V4-Flash-0731-MXFP4_KO.gguf";
/// The pack's repo label: the source is local, and the provenance says so.
const KO_LABEL: &str = "local/deepseek-v4-flash-mxfp4";
const TOKENIZER_REPO: &str = "deepseek-ai/DeepSeek-V4-Flash-0731";

/// The merged checkpoint's model pack, or `None` when this machine has neither
/// the pack nor the file it is built from. The source is local; the hub is
/// asked for the tokenizer alone.
fn ko_pack(device: &Device) -> Result<Option<PathBuf>> {
    let rev = local_rev(&Path::new(KO_DIR).join(KO_FILE));
    let request = PackRequest::of(
        PackFamily::Latent(&DEEPSEEK_V4),
        (KO_LABEL, &rev, KO_FILE),
        (TOKENIZER_REPO, "main"),
        Some(Int8Mode::Performance),
    );
    let fetch = LocalFetch {
        repo: KO_LABEL.into(),
        dir: PathBuf::from(KO_DIR),
        other: &HubFetch,
    };
    match model_pack(&request, &cache_root(), device, &fetch) {
        Ok(path) => Ok(Some(path)),
        Err(_) if !Path::new(KO_DIR).join(KO_FILE).exists() => Ok(None),
        Err(e) => Err(e),
    }
}

/// Re-frame a Qwen-ChatML RULER sample into the DeepSeek dialect. The haystack
/// body + question is model-agnostic; only the role markers differ, so we lift
/// the middle (between the user opener and the assistant header) and wrap it in
/// DeepSeek's `<｜begin▁of▁sentence｜>…<｜User｜>…<｜Assistant｜>` frame.
fn reframe_deepseek(qwen_input: &str) -> String {
    // Qwen user body starts after `/no_think\n`; the assistant header begins at
    // the first `<|im_end|><|im_start|>assistant`.
    let after_user = qwen_input
        .split_once("/no_think\n")
        .map(|(_, rest)| rest)
        .unwrap_or(qwen_input);
    let body = after_user
        .split_once("<|im_end|><|im_start|>assistant")
        .map(|(mid, _)| mid)
        .unwrap_or(after_user);
    format!(
        "<｜begin▁of▁sentence｜>You are a helpful assistant.\
         <｜User｜>{body}<｜Assistant｜>"
    )
}

#[test]
#[ignore = "loads DeepSeek-V4-Flash on CUDA; per-token prefill is minutes at multi-k context"]
fn deepseek_niah_single_recall() -> candle::Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("[skip] no CUDA device");
            return Ok(());
        }
    };
    let Some(pack) = ko_pack(&device)? else {
        eprintln!("[skip] KO gguf absent: {KO_DIR}\\{KO_FILE}");
        return Ok(());
    };
    // The tokenizer the pack was built with, from inside it.
    let opened = ModelPack::open(&pack)?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(opened.tokenizer_json()?.as_bytes())
        .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
    let eos = tokenizer
        .token_to_id("<｜end▁of▁sentence｜>")
        .expect("deepseek eos id");

    // Context length WELL beyond window_size (128) so the needle lives only in
    // the compressed corpus, not the sliding window. Two samples at random
    // depths — recall must hold regardless of where the needle lands.
    let ctx_len = 2048usize;
    let raw = generate_ruler_samples(&tokenizer, RulerTask::NiahSingle1, ctx_len, 2, 0x0D1A);
    let samples: Vec<RulerSample> = raw
        .into_iter()
        .map(|s| RulerSample {
            input: reframe_deepseek(&s.input),
            outputs: s.outputs,
        })
        .collect();

    let engine = Engine::load(&pack, &DEEPSEEK_V4, &device)?;
    let model = BatchedEngine::new(engine)?;

    let t0 = std::time::Instant::now();
    let preds = run_ruler_eval(
        &model,
        &tokenizer,
        &samples,
        None,
        16,
        &[eos],
        &device,
        None,
    )?;
    eprintln!(
        "[niah] {} samples in {:.1}s",
        samples.len(),
        t0.elapsed().as_secs_f32()
    );

    let mut hits = 0usize;
    for (i, (pred, sample)) in preds.iter().zip(&samples).enumerate() {
        let ok = score_ruler_sample(RulerTask::NiahSingle1, pred, &sample.outputs);
        eprintln!(
            "[niah] sample {i}: needle={:?} pred={:?} -> {}",
            sample.outputs, pred, ok
        );
        hits += ok as usize;
    }
    assert!(
        hits == samples.len(),
        "needle recall failed: {hits}/{} samples found the magic number at {ctx_len}-token depth",
        samples.len()
    );
    Ok(())
}
