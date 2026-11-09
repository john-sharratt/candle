//! A round's text, decoded token by token as it streams — each step costing
//! what the step adds, not a decode of the whole round.
//!
//! Built on tokenizers' [`DecodeStream`] (special tokens skipped, as the full
//! decode skips them): a step releases the text its token completes, holding a
//! token back while it ends mid-character. **The text only ever grows.** Should
//! a step fail — its window no longer decodes to a prefix of what it released
//! (`InvalidPrefix`), or the decode itself errs — the stream falls back to the
//! full decode of the round's ids until it can resume: a full decode that
//! extends the text appends the extension, one that does not appends nothing
//! that step; and once a full decode extends the text and does not end in
//! U+FFFD, a fresh `DecodeStream` takes over from the next id. That is sound
//! for a byte-level decoder, which needs no context across a whole character.

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{DecodeStream, Tokenizer};

/// The replacement character a decode ends in while a character is cut.
const CUT: char = '\u{FFFD}';

type Stream<'t> = DecodeStream<
    't,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// One round's text, as its tokens stream.
pub struct TextStream<'t> {
    tokenizer: &'t Tokenizer,
    /// `None` while falling back to full decodes.
    stream: Option<Stream<'t>>,
    ids: Vec<u32>,
    text: String,
}

impl<'t> TextStream<'t> {
    pub fn new(tokenizer: &'t Tokenizer) -> Self {
        TextStream {
            tokenizer,
            stream: Some(tokenizer.decode_stream(true)),
            ids: Vec::new(),
            text: String::new(),
        }
    }

    /// Take `id` in, appending what it releases. Answers the byte offset in
    /// [`Self::text`] where the new bytes start — its length before the step.
    pub fn push(&mut self, id: u32) -> usize {
        let from = self.text.len();
        self.ids.push(id);
        match self.stream.as_mut().map(|s| s.step(id)) {
            Some(Ok(Some(released))) => self.text.push_str(&released),
            Some(Ok(None)) => {}
            Some(Err(_)) | None => {
                self.stream = None;
                self.fall_back();
            }
        }
        from
    }

    /// Append what the full decode adds to the text, and resume streaming once
    /// it has extended the text by whole characters.
    fn fall_back(&mut self) {
        let Ok(full) = self.tokenizer.decode(&self.ids, true) else {
            return;
        };
        if full.len() > self.text.len() && full.starts_with(&self.text) {
            self.text.push_str(&full[self.text.len()..]);
            if !full.ends_with(CUT) {
                self.stream = Some(self.tokenizer.decode_stream(true));
            }
        }
    }

    /// The text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// How many tokens have been taken in.
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use std::array::from_fn;

    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::replace::{Replace, ReplacePattern};

    use super::*;

    /// A tokenizer over `vocab` with no merges, decoding through `decoder`.
    fn tokenizer<const N: usize>(
        vocab: [&str; N],
        byte_fallback: bool,
        decoder: DecoderWrapper,
    ) -> Tokenizer {
        let vocab: [(String, u32); N] = from_fn(|i| (vocab[i].to_string(), i as u32));
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(byte_fallback)
            .build()
            .unwrap();
        let mut t = Tokenizer::new(bpe);
        t.with_decoder(Some(decoder));
        t
    }

    /// **A character cut across two byte tokens is released whole**: `é` as
    /// `<0xC3>` then `<0xA9>` adds nothing, then exactly its two bytes.
    #[test]
    fn a_character_split_across_byte_tokens_is_released_whole() {
        let t = tokenizer(
            ["<0xC3>", "<0xA9>"],
            true,
            DecoderWrapper::ByteFallback(ByteFallback::new()),
        );
        let mut s = TextStream::new(&t);
        assert_eq!(s.push(0), 0);
        assert_eq!(s.text(), "");
        assert_eq!(s.push(1), 0);
        assert_eq!(s.text().as_bytes(), [0xC3, 0xA9]);
        assert_eq!(s.tokens(), 2);
    }

    /// **A step whose window no longer extends the text falls back, and the
    /// text never shrinks.** The decoder rewrites a trailing `ab` (fused, then
    /// replaced) so `a`, `b` decodes as `XYZ` — `InvalidPrefix` — and the text
    /// keeps `a`; `c` undoes the rewrite, the full decode `abc` extends it and
    /// streaming resumes, so the text ends equal to the full decode.
    #[test]
    fn a_step_that_rewrites_the_past_falls_back_and_resumes() {
        let rewrite = Replace::new(ReplacePattern::Regex("ab$".into()), "XYZ").unwrap();
        let t = tokenizer(
            ["a", "b", "c"],
            false,
            DecoderWrapper::Sequence(Sequence::new(vec![
                DecoderWrapper::Fuse(Fuse::new()),
                DecoderWrapper::Replace(rewrite),
            ])),
        );
        assert_eq!(t.decode(&[0, 1], true).unwrap(), "XYZ");
        let mut s = TextStream::new(&t);
        s.push(0);
        assert_eq!(s.text(), "a");
        assert_eq!(s.push(1), 1);
        assert_eq!(s.text(), "a", "nothing taken back, nothing added");
        assert_eq!(s.push(2), 1);
        assert_eq!(s.text(), "abc");
        assert_eq!(s.push(0), 3);
        assert_eq!(s.text(), t.decode(&[0, 1, 2, 0], true).unwrap());
        assert_eq!(s.text(), "abca");
    }
}
