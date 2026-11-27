//! The knowledge base's pieces on every cached Qwen vocabulary: sealed one by
//! one, they must hold exactly the tokens the assembled text holds, or a seam
//! shows the model a sequence it never saw
//! (`docs/placed_section_layout.md` §3.5c).

#[path = "common/real_vocab.rs"]
mod real_vocab;

use candle_conversation::files::render::{listing_section, section_text};
use candle_conversation::files::{assemble_pieces, Chunk, LineSpan, RenderedFile};
use tokenizers::Tokenizer;

use real_vocab::for_each_tokenizer;

fn ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false)
        .unwrap_or_else(|e| panic!("encoding {text:?}: {e}"))
        .get_ids()
        .to_vec()
}

fn chunk(first: u32, last: u32, text: String) -> Chunk {
    Chunk {
        lines: LineSpan { first, last },
        text,
    }
}

/// `files`' pieces tokenized one by one, joined, against the assembled text
/// tokenized whole.
fn assert_pieces_hold(name: &str, tok: &Tokenizer, files: &[RenderedFile]) {
    let pieces = assemble_pieces(files);
    let apart: Vec<u32> = pieces.iter().flat_map(|p| ids(tok, p)).collect();
    let whole = ids(tok, &pieces.concat());
    assert_eq!(
        apart, whole,
        "{name}: pieces {pieces:?} tokenize differently from their text"
    );
}

/// **The assembled pieces tokenize as the knowledge text**: a file of two
/// sections in `file_read`'s format, the first ending on the cut marker, a
/// section placed alone out of its file's order, and a folder, on every cached
/// Qwen vocabulary.
#[test]
fn assembled_pieces_tokenize_as_the_knowledge_text() {
    for_each_tokenizer(|name, tok| {
        let path = "candle/zend/src/ingest_lease.rs";
        let first = section_text(
            path,
            "```rust file=candle/zend/src/ingest_lease.rs part=a1B2 lines=1-3\n \
             +1 //! Leases.\n +2 \n +3 use std::time::Duration;\n```\n\
             end of candle/zend/src/ingest_lease.rs lines 1-3 of 6 — next: lines=4-6\n",
            false,
        );
        let second = section_text(
            path,
            "```rust file=candle/zend/src/ingest_lease.rs part=c3D4 lines=4-6 scope=\"fn claim_worker\"\n \
             +1 pub fn claim_worker() -> Share {\n +2     ask().unwrap_or(Share::Theirs)\n +3 }\n```\n\
             end of candle/zend/src/ingest_lease.rs lines 4-6 of 6 — the end of the file\n",
            true,
        );
        let rust = RenderedFile {
            path: path.into(),
            chunks: vec![chunk(1, 3, first.clone()), chunk(4, 6, second.clone())],
        };
        let out_of_order = RenderedFile {
            path: path.into(),
            chunks: vec![chunk(4, 6, second), chunk(1, 3, first)],
        };
        let folder = RenderedFile {
            path: "candle/zend/src/".into(),
            chunks: vec![chunk(
                1,
                3,
                listing_section("candle/zend/src/", "api/\ncalibrate/\nlib.rs 2459\n"),
            )],
        };
        assert_pieces_hold(name, tok, std::slice::from_ref(&rust));
        assert_pieces_hold(name, tok, std::slice::from_ref(&out_of_order));
        assert_pieces_hold(name, tok, &[rust, folder]);
    });
}
