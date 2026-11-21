//! Helpers for the Flash-Next tests.
//!
//! These open this card's engine pack, the tokenizer, and the pinned shard list
//! that the gates load against.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::batch_test::test_helpers::hf_get;

    use hf_hub::RepoType;

    pub(in super::super) fn pinned_shards() -> Result<Vec<PathBuf>> {
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

    /// This card's engine pack at the mode this card picks.
    pub(in super::super) fn engine() -> Result<PathBuf> {
        let device = Device::new_cuda(0)?;
        engine_pack(&device, Some(Int8Mode::auto(&device)))
    }

    pub(in super::super) fn tokenizer() -> Result<tokenizers::Tokenizer> {
        let p = hf_get(
            TOKENIZER_REPO,
            RepoType::Model,
            TOKENIZER_REV,
            "tokenizer.json",
        )?;
        tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| candle::Error::Msg(format!("load tokenizer: {e}")))
    }

    pub(in super::super) fn tokenizer_json() -> Result<String> {
        let p = hf_get(
            TOKENIZER_REPO,
            RepoType::Model,
            TOKENIZER_REV,
            "tokenizer.json",
        )?;
        std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
    }
}
