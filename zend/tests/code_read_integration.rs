//! Tier-2 integration test for the `code_reading` layer's
//! tool-call conversation shape.
//!
//! Each carved scope emits four turns:
//!
//!   1. user (prefill)    — "Read X lines N-M and summarize..."
//!   2. assistant (prefill) — `<tool_call>{...}</tool_call>`
//!   3. user (prefill)    — `<tool_response>...</tool_response>`
//!   4. assistant (DECODED) — model-generated one-sentence summary
//!
//! The [`RecordingTurnSink`] captures each call's `was_decoded`
//! flag; tests assert the alternating pattern + the content of the
//! prefilled turns.  The decoded turn returns a deterministic stub
//! so tests don't need a model loaded.

use std::fs;
use std::path::Path;

use zend::code_read::ingest_code_reading_into_sink;
use zend::loading::LoadProgress;
use zend::repo_scan::walk_workspace;
use zend::turn_sink::RecordingTurnSink;

fn fixture(name: &str) -> tempfile::TempDir {
    let _ = name;
    tempfile::tempdir().expect("tempdir")
}

fn write(root: &Path, rel: &str, body: &[u8]) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

/// Number of turns emitted per carved scope under the new tool-call
/// shape — one prefill pair plus one decode pair.
const TURNS_PER_SCOPE: usize = 2;

#[test]
fn code_read_emits_two_turn_pairs_per_scope() {
    let dir = fixture("phase2");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "src/lib.rs",
        b"pub fn alpha() {}\npub fn beta() {}\n",
    );
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    let (n_scopes, _state) = ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();
    assert!(n_scopes >= 2, "expected ≥2 scopes (alpha + beta), got {n_scopes}");
    assert_eq!(
        sink.turns.len(),
        n_scopes * TURNS_PER_SCOPE,
        "each scope emits {TURNS_PER_SCOPE} sink calls (one prefill pair + one decode pair)",
    );
}

#[test]
fn code_read_alternates_prefill_then_decode_per_scope() {
    let dir = fixture("alternation");
    let root = dir.path().to_path_buf();
    write(&root, "src/lib.rs", b"pub fn alpha() {}\npub fn beta() {}\n");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    for chunk in sink.turns.chunks(2) {
        assert_eq!(chunk.len(), 2);
        assert!(!chunk[0].2, "first call per scope is a prefill");
        assert!(chunk[1].2, "second call per scope is a decode");
    }
}

#[test]
fn code_read_first_user_prompt_asks_for_one_sentence_summary() {
    let dir = fixture("user_prompt");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "src/auth/handler.rs",
        b"pub fn validate_token() { let _ = 1; }\n",
    );
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    let user1 = &sink.turns[0].0;
    assert!(
        user1.starts_with("Read `src/auth/handler.rs` lines "),
        "user prompt should start with the read instruction, got: {user1:?}"
    );
    assert!(user1.contains("summarize it in a single sentence"));
}

#[test]
fn code_read_first_assistant_is_hermes_tool_call() {
    let dir = fixture("tool_call");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "src/lib.rs",
        b"pub fn alpha() { let _ = 1; }\n",
    );
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    let assistant1 = &sink.turns[0].1;
    assert!(assistant1.starts_with("<tool_call>"));
    assert!(assistant1.ends_with("</tool_call>"));
    assert!(assistant1.contains("\"name\":\"read_file\""));
    assert!(assistant1.contains("\"path\":\"src/lib.rs\""));
    assert!(assistant1.contains("\"start_line\":"));
    assert!(assistant1.contains("\"end_line\":"));
}

#[test]
fn code_read_second_user_is_tool_response_with_fenced_code() {
    let dir = fixture("tool_response");
    let root = dir.path().to_path_buf();
    let src = b"pub fn one() {}\npub fn two() {\n    let x = 42;\n}\n";
    write(&root, "src/lib.rs", src);
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    let tr = sink
        .turns
        .iter()
        .find(|(u, _, _)| u.contains("<tool_response>") && u.contains("pub fn two"))
        .expect("tool_response carrying fn two");
    assert!(tr.0.starts_with("<tool_response>\n"));
    assert!(tr.0.ends_with("</tool_response>"));
    assert!(tr.0.contains("```rust"));
    assert!(tr.0.contains("pub fn two() {"));
    assert!(tr.0.contains("let x = 42;"));
    // The decode-paired assistant text is the stubbed summary.
    assert_eq!(tr.1, "[fake summary]");
    assert!(tr.2, "the tool_response pair is the decoded turn");
}

#[test]
fn code_read_prefill_never_emits_tool_call_tags_in_user_prompt() {
    // The first-turn user prompt (the natural-language ask) must
    // NEVER carry `<tool_call>` tags — those are reserved for the
    // assistant tool-call echo.  Were they to leak into the user
    // turn, the dialogue layer's tool-call extractor could not
    // parse a real tool call separated from this prefill marker.
    let dir = fixture("no_user_tool_call");
    let root = dir.path().to_path_buf();
    write(&root, "src/lib.rs", b"pub fn alpha() {}\n");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    for (user, _, _decoded) in &sink.turns {
        let is_tool_response = user.contains("<tool_response>");
        if !is_tool_response {
            assert!(
                !user.contains("<tool_call>"),
                "natural-language user prompt must not contain <tool_call>: {user:?}"
            );
        }
    }
}

#[test]
fn code_read_skips_files_outside_watch_patterns() {
    let dir = fixture("filter");
    let root = dir.path().to_path_buf();
    write(&root, "src/lib.rs", b"pub fn alpha() {}\n");
    write(&root, "data/blob.bin", b"\x00\x01\x02");
    write(&root, "image.svg", b"<svg/>");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    for (u, _, _) in &sink.turns {
        assert!(!u.contains("blob.bin"));
        assert!(!u.contains("image.svg"));
    }
}

#[test]
fn code_read_falls_back_to_fixed_window_on_unknown_language() {
    let dir = fixture("fallback");
    let root = dir.path().to_path_buf();
    let lines: String = (0..250).map(|i| format!("line {i}\n")).collect();
    write(&root, "notes.txt", lines.as_bytes());
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    let scope_prompts: Vec<&str> = sink
        .turns
        .iter()
        .filter(|(u, _, _decoded)| u.contains("notes.txt") && u.starts_with("Read"))
        .map(|(u, _, _)| u.as_str())
        .collect();
    assert!(
        scope_prompts.len() >= 3,
        "250-line file should split into ≥ 3 windows, got {}",
        scope_prompts.len()
    );
}

#[test]
fn code_read_progress_reports_real_fraction() {
    let dir = fixture("progress");
    let root = dir.path().to_path_buf();
    write(&root, "src/a.rs", b"pub fn one() {}\n");
    write(&root, "src/b.rs", b"pub fn two() {}\n");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    let (n_scopes, _state) = ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();
    assert_eq!(n_scopes * TURNS_PER_SCOPE, sink.turns.len());

    let snap = progress.snapshot().expect("still loading");
    assert!(snap.progress >= 0.999, "should be at 100% after ingest");
}
