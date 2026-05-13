use crate::token_buffer::TokenBuffer;

/// Streaming token decoder.
///
/// Wraps a tokenizer to produce text incrementally as tokens arrive,
/// handling partial UTF-8 sequences correctly. Adapted from the
/// `TokenOutputStream` in `candle-examples`.
///
/// # Design
///
/// BPE tokenizers with byte-fallback (e.g. Qwen) decode individual byte
/// tokens as U+FFFD but merge consecutive byte tokens into proper characters.
/// A sliding-window approach that decodes a subset of tokens would see
/// different text than the full decode, breaking byte-offset-based diffing.
///
/// Instead, this decoder always decodes ALL accumulated tokens on every call
/// and tracks how much of the full decoded string has been emitted via a
/// single `emitted_len` counter. The full decode is always consistent,
/// so the emitted prefix is stable and the diff is correct.
///
/// Performance: tokenizer decode is ~microseconds for a few thousand tokens,
/// negligible versus GPU inference time (~10ms+ per token).
pub(crate) struct TokenStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: TokenBuffer,
    /// Byte length of the full decoded text that has already been emitted.
    emitted_len: usize,
    /// When `true`, special tokens are stripped from decoded text.
    /// When `false` (show-hidden mode), they are included verbatim.
    skip_special_tokens: bool,
}

impl TokenStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: TokenBuffer::new(),
            emitted_len: 0,
            skip_special_tokens: true,
        }
    }

    /// Create a token stream that includes special tokens in decoded output.
    pub fn new_show_special(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: TokenBuffer::new(),
            emitted_len: 0,
            skip_special_tokens: false,
        }
    }

    fn decode(&self, tokens: &[u32]) -> candle::Result<String> {
        self.tokenizer
            .decode(tokens, self.skip_special_tokens)
            .map_err(|e| candle::Error::Msg(format!("tokenizer decode error: {e}")))
    }

    /// Feed a new token. Returns the new text fragment if a complete
    /// character boundary was reached, `None` otherwise.
    pub fn next_token(&mut self, token: u32) -> candle::Result<Option<String>> {
        self.tokens.push(token);
        let text = self.decode(&self.tokens)?;

        if text.len() > self.emitted_len && text.is_char_boundary(self.emitted_len) {
            let new_part = &text[self.emitted_len..];

            // Decide whether the new fragment is safe to emit:
            //
            // skip_special_tokens mode: require last char to be alphanumeric.
            //   This gates emission until byte-fallback sequences complete
            //   (U+FFFD is not alphanumeric) and punctuation accumulates
            //   until the next word character. Flushed by decode_rest().
            //
            // show-hidden mode: emit unless the fragment contains U+FFFD,
            //   which signals an incomplete byte-fallback sequence. This is
            //   more permissive (allows punctuation, angle brackets for
            //   special tokens like <|im_end|>) while still buffering
            //   partial emoji / multi-byte characters.
            let safe = if self.skip_special_tokens {
                new_part
                    .chars()
                    .last()
                    .map_or(false, |c| c.is_alphanumeric())
            } else {
                !new_part.contains('\u{FFFD}')
            };

            if safe {
                let result = new_part.to_string();
                self.emitted_len = text.len();
                return Ok(Some(result));
            }
        }
        Ok(None)
    }

    /// Flush any remaining partial text that hasn't been emitted yet.
    pub fn decode_rest(&self) -> candle::Result<Option<String>> {
        let text = self.decode(&self.tokens)?;
        if text.len() > self.emitted_len && text.is_char_boundary(self.emitted_len) {
            Ok(Some(text[self.emitted_len..].to_string()))
        } else if text.len() > self.emitted_len {
            // Boundary mismatch at end of turn — emit full remaining text.
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }

    /// Decode all tokens into a single string.
    pub fn decode_all(&self) -> candle::Result<String> {
        self.decode(&self.tokens)
    }
}
