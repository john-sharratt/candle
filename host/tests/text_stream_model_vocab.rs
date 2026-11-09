//! The streamed text is the full decode for the prescribed model's own
//! tokenizer (`docs/performance_patch.md` Z1, §7) — a landing requirement, so
//! it fails, never skips, when the tokenizer cannot be had. Ignored by default:
//! run it with `cargo test --release -p host --test text_stream_model_vocab --
//! --ignored --nocapture`.

use candle_conversation::text_stream::TextStream;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use host::MODEL;
use tokenizers::Tokenizer;

/// Characters rare enough that a vocabulary spells them in byte tokens; the
/// first one the model's vocabulary splits goes into the fixture.
const RARE: [&str; 5] = ["𓂀", "𐍈", "ꙮ", "🫠", "🦀"];

/// A reply holding everything a stream must carry whole: the model's reasoning
/// and call tags, multi-byte text, and `cut`, a character split across byte
/// tokens.
fn fixture(cut: &str) -> String {
    format!(
        "<think>\nThe file is réseau.rs — read it {cut} first.\n</think>\n\n\
         <tool_call>\n{{\"name\": \"file_read\", \"arguments\": {{\"path\": \"naïve/日本.rs\"}}}}\n\
         </tool_call>\nDone: {cut}{cut}, ü, 中文."
    )
}

#[test]
#[ignore = "fetches the prescribed model's tokenizer; run before landing Z1"]
fn the_streamed_text_is_the_full_decode_for_the_prescribed_model() {
    let spec = MODEL.spec();
    let repo = Repo::with_revision(
        spec.tokenizer_repo.clone(),
        RepoType::Model,
        spec.tokenizer_rev.clone(),
    );
    let path = Api::new()
        .expect("the hub API")
        .repo(repo)
        .get("tokenizer.json")
        .unwrap_or_else(|e| {
            panic!(
                "{}: tokenizer.json could not be had: {e}",
                spec.tokenizer_repo
            )
        });
    let tok = Tokenizer::from_file(&path).expect("tokenizer.json parses");

    let cut = RARE
        .into_iter()
        .find(|c| {
            let ids = tok.encode(*c, false).unwrap().get_ids().to_vec();
            ids.len() > 1 && tok.decode(&ids[..1], true).unwrap().ends_with('\u{FFFD}')
        })
        .unwrap_or_else(|| panic!("none of {RARE:?} is split across byte tokens"));

    let ids = tok
        .encode(fixture(cut).as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let full = tok.decode(&ids, true).unwrap();
    let mut stream = TextStream::new(&tok);
    for &id in &ids {
        stream.push(id);
        assert!(
            full.starts_with(stream.text()),
            "not a prefix of the full decode"
        );
    }
    assert_eq!(stream.text().as_bytes(), full.as_bytes());
    println!(
        "{}: {} tokens streamed, {} bytes, `{cut}` split, equal to the full decode",
        spec.tokenizer_repo,
        ids.len(),
        full.len()
    );
}
