//! [`candle_conversation::text_stream`] against every real chat vocabulary in
//! this machine's hub cache: the text streamed token by token is the full
//! decode, byte for byte, over chat tags, multi-byte text and an emoji cut
//! across byte tokens. Skipped, loudly, when no Qwen tokenizer is here; the
//! prescribed model's own gate is `host`'s `text_stream_model_vocab`.

#[path = "common/real_vocab.rs"]
mod real_vocab;

use candle_conversation::text_stream::TextStream;
use tokenizers::Tokenizer;

use real_vocab::for_each_tokenizer;

/// Characters rare enough that a vocabulary spells them in byte tokens; the
/// first one this vocabulary splits goes into the fixture.
const RARE: [&str; 5] = ["𓂀", "𐍈", "ꙮ", "🫠", "🦀"];

/// A reply holding everything a stream must carry whole, `cut` among it: a
/// character this vocabulary splits across byte tokens.
fn fixture(cut: &str) -> String {
    format!(
        "<think>\nThe file is réseau.rs — read it {cut} first.\n</think>\n\n\
         <tool_call>\n{{\"name\": \"file_read\", \"arguments\": {{\"path\": \"naïve/日本.rs\"}}}}\n\
         </tool_call>\nDone: {cut}{cut}, ü, 中文."
    )
}

/// The fixture's ids, holding a character this vocabulary splits across byte
/// tokens — the case a stream holds back.
fn ids_with_a_split_character(name: &str, tok: &Tokenizer) -> Vec<u32> {
    let cut = RARE
        .into_iter()
        .find(|c| {
            let ids = tok.encode(*c, false).unwrap().get_ids().to_vec();
            ids.len() > 1 && tok.decode(&ids[..1], true).unwrap().ends_with('\u{FFFD}')
        })
        .unwrap_or_else(|| panic!("{name}: none of {RARE:?} is split across byte tokens"));
    tok.encode(fixture(cut).as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec()
}

#[test]
fn the_streamed_text_is_the_full_decode() {
    for_each_tokenizer(|name, tok| {
        let ids = ids_with_a_split_character(name, tok);
        let full = tok.decode(&ids, true).unwrap();
        let mut stream = TextStream::new(tok);
        for &id in &ids {
            let from = stream.push(id);
            assert!(from <= stream.text().len(), "{name}");
            assert!(full.starts_with(stream.text()), "{name}: not a prefix");
        }
        assert_eq!(stream.text().as_bytes(), full.as_bytes(), "{name}");
        assert_eq!(stream.tokens(), ids.len(), "{name}");
    });
}
