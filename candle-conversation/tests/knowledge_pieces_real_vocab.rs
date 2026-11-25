//! The knowledge base's pieces on every cached Qwen vocabulary: sealed one by
//! one, they must hold exactly the tokens the assembled text holds, or a seam
//! shows the model a sequence it never saw
//! (`docs/provenance_system_collection.md` §3, §12 D2–D3).

#[path = "common/real_vocab.rs"]
mod real_vocab;

use candle_conversation::files::{assemble_pieces, cut_lines, Chunk, LineSpan, RenderedFile};
use tokenizers::Tokenizer;

use real_vocab::for_each_tokenizer;

fn ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false)
        .unwrap_or_else(|e| panic!("encoding {text:?}: {e}"))
        .get_ids()
        .to_vec()
}

fn chunk(first: u32, last: u32, text: &str) -> Chunk {
    Chunk {
        lines: LineSpan { first, last },
        text: text.to_string(),
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

/// **The assembled pieces tokenize as the knowledge text**: a file placed
/// whole, a file with a gap between two runs, and two files, on every cached
/// Qwen vocabulary.
#[test]
fn assembled_pieces_tokenize_as_the_knowledge_text() {
    for_each_tokenizer(|name, tok| {
        let rust = RenderedFile {
            path: "candle/zend/src/ingest_lease.rs".into(),
            chunks: vec![
                chunk(1, 3, "//! Leases.\n\nuse std::time::Duration;\n"),
                chunk(
                    4,
                    6,
                    "pub fn claim_worker() -> Share {\n    ask().unwrap_or(Share::Theirs)\n}\n",
                ),
            ],
        };
        let gapped = RenderedFile {
            path: "candle/docs/design.md".into(),
            chunks: vec![
                chunk(1, 2, "# Design\n\nThe close row.\n"),
                chunk(40, 41, "| a | b |\n|---|---|\n"),
            ],
        };
        let folder = RenderedFile {
            path: "candle/zend/src/".into(),
            chunks: vec![chunk(1, 3, "api/\ncalibrate/\nlib.rs 2459\n")],
        };
        assert_pieces_hold(name, tok, std::slice::from_ref(&rust));
        assert_pieces_hold(name, tok, std::slice::from_ref(&gapped));
        assert_pieces_hold(name, tok, &[rust, gapped, folder]);
    });
}

/// **Chunks the cut makes hold their seams on the real vocabulary**: a file
/// with blank lines at every cut is cut where the seams tokenize alike, and
/// its chunks, assembled, tokenize as the file does.
#[test]
fn a_cut_file_assembles_as_its_text_tokenizes() {
    for_each_tokenizer(|name, tok| {
        let text = "fn a() {}\n\nfn b() {}\n\n\nfn c() {}\n    \n}\n";
        let chunks = cut_lines(text, 1, |s| ids(tok, s));
        assert_eq!(chunks.concat_text(), text, "{name}: the cut lost text");
        assert_pieces_hold(
            name,
            tok,
            &[RenderedFile {
                path: "candle/a.rs".into(),
                chunks,
            }],
        );
    });
}

trait ConcatText {
    fn concat_text(&self) -> String;
}

impl ConcatText for Vec<Chunk> {
    fn concat_text(&self) -> String {
        self.iter().map(|c| c.text.as_str()).collect()
    }
}
