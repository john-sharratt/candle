/// Shared helpers for RULER and other benchmark tests.
///
/// These are only compiled under `#[cfg(test)]` — they exist solely to reduce
/// boilerplate in per-model test modules.

/// Download `{hf_repo}/tokenizer.json` via the HF Hub API and return a
/// ready-to-use `Tokenizer`.
///
/// # Arguments
/// * `hf_repo` – HF model id, e.g. `"Qwen/Qwen3-8B"`.
pub fn load_hf_tokenizer(hf_repo: &str) -> candle::Result<tokenizers::Tokenizer> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| candle::Error::Msg(format!("HF API init: {e}")))?;
    let repo = api.model(hf_repo.to_string());
    let path = repo
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(format!("tokenizer download ({hf_repo}): {e}")))?;
    let json = std::fs::read_to_string(&path)
        .map_err(|e| candle::Error::Msg(format!("tokenizer read: {e}")))?;
    tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .map_err(|e| candle::Error::Msg(format!("tokenizer parse: {e}")))
}

/// Download a GGUF file from the HF Hub and return its local `PathBuf`.
///
/// # Arguments
/// * `hf_repo`   – HF model id, e.g. `"unsloth/Qwen3-8B-GGUF"`.
/// * `filename`  – file inside the repo, e.g. `"Qwen3-8B-Q4_K_M.gguf"`.
/// * `revision`  – git ref, usually `"main"`.
pub fn download_hf_gguf(
    hf_repo: &str,
    filename: &str,
    revision: &str,
) -> candle::Result<std::path::PathBuf> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| candle::Error::Msg(format!("HF API init: {e}")))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        hf_repo.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));
    let path = repo
        .get(filename)
        .map_err(|e| candle::Error::Msg(format!("model download ({hf_repo}/{filename}): {e}")))?;
    Ok(path)
}

/// Read a GGUF file from disk and return its parsed
/// [`candle::quantized::gguf_file::Content`] together with the open file
/// handle (needed for tensor data reads).
pub fn open_gguf(
    path: &std::path::Path,
) -> candle::Result<(
    candle::quantized::gguf_file::Content,
    std::fs::File,
)> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| candle::Error::Msg(format!("open {:?}: {e}", path)))?;
    let content = candle::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| candle::Error::Msg(format!("read gguf {:?}: {e}", path)))?;
    Ok((content, file))
}
