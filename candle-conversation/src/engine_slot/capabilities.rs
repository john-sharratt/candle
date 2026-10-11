//! What a tenant needs of the model it runs on.
//!
//! One model serves every tenant in a process (`docs/shared_model_host.md`
//! §5), so each tenant declares what it needs *of* a model and is refused at
//! start — by name, with the missing capability — when the model lacks it. A
//! change of model then fails at startup rather than in a stencil compile
//! midway through a load, or quietly, by running with no tools.

use crate::models::DialectType;

/// What a tenant needs of a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Strings that must each encode to exactly one token — the structural
    /// markers a tenant's stencils and parsers key on.
    pub tokens: &'static [&'static str],
    /// The chat dialect its prompts and stencils are written for.
    pub dialect: DialectType,
}

impl Capabilities {
    /// Check `tokenizer` and the model's `dialect` against these needs.
    /// `Err` names every capability the model lacks.
    pub fn check(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        dialect: DialectType,
    ) -> Result<(), String> {
        let mut missing = Vec::new();
        for token in self.tokens {
            let n = tokenizer
                .encode(*token, false)
                .map(|e| e.get_ids().len())
                .unwrap_or(0);
            if n != 1 {
                missing.push(format!("`{token}` as one token (it encodes to {n})"));
            }
        }
        if dialect != self.dialect {
            missing.push(format!(
                "the {:?} dialect (the model speaks {dialect:?})",
                self.dialect
            ));
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::{AddedToken, Tokenizer};

    /// A vocabulary in which `<think>` is one token and `<tool_call>` is not.
    fn tokenizer() -> Tokenizer {
        let vocab = [("[UNK]".to_string(), 0u32), ("x".to_string(), 1)]
            .into_iter()
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut t = Tokenizer::new(model);
        t.with_pre_tokenizer(Some(Whitespace {}));
        t.add_special_tokens(&[AddedToken::from("<think>", true)]);
        t
    }

    #[test]
    fn a_model_with_every_capability_passes() {
        let needs = Capabilities {
            tokens: &["<think>"],
            dialect: DialectType::Qwen35,
        };
        assert_eq!(needs.check(&tokenizer(), DialectType::Qwen35), Ok(()));
    }

    #[test]
    fn every_missing_capability_is_named() {
        let needs = Capabilities {
            tokens: &["<think>", "<tool_call>"],
            dialect: DialectType::Qwen35,
        };
        let err = needs.check(&tokenizer(), DialectType::ChatML).unwrap_err();
        assert!(err.contains("`<tool_call>` as one token"), "{err}");
        assert!(err.contains("the Qwen35 dialect"), "{err}");
        assert!(!err.contains("`<think>`"), "{err}");
    }
}
