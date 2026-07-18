//! Tier-2 integration test for the `code_reading` layer's per-file
//! tool-call conversation shape.
//!
//! Each file becomes ONE conversation: one prefill turn per carved part, whose
//! stored ChatML has three role segments —
//!       user      : "Source excerpt — `X` lines N-M:"
//!       assistant : `<tool_call>{...}</tool_call>`
//!       user      : `<tool_response>...</tool_response>`
//! The `(user, assistant)` pair the sink records fuses the last two into its
//! `assistant` string with the dialect role boundary spliced between them, so
//! the tool response reconstructs as its own user turn (Hermes/Qwen convention).
//!
//! There is no inline whole-file summary decode: the file summary is the
//! async summary tree's root, built by the summariser over these recorded
//! scope turns. The [`RecordingTurnSink`] captures every `(user, assistant)`
//! prefill pair so tests can verify the conversation shape without a model.

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

#[test]
fn code_read_emits_one_prefill_per_part() {
    let dir = fixture("per_file_shape");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "src/lib.rs",
        b"pub fn alpha() {}\npub fn beta() {}\n",
    );
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    let (n_scopes, _state) =
        ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();
    assert!(
        n_scopes >= 2,
        "expected ≥2 scopes (alpha + beta), got {n_scopes}"
    );

    // Single file → exactly one prefill turn per carved part, no summary decode
    // (the whole-file summary is the async summariser's rollup, not inline).
    assert_eq!(
        sink.turns.len(),
        n_scopes,
        "one prefill turn per carved part"
    );
    // Gather-scope tags: every scope turn carries `["code", <path>]` so it lands
    // in the code-tagged provenance gallery.
    for (_, _, tags) in &sink.turns {
        assert_eq!(tags, &["code", "src/lib.rs"]);
    }
}

#[test]
fn code_read_every_turn_is_a_part_prefill() {
    let dir = fixture("all_prefills");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "src/lib.rs",
        b"pub fn alpha() {}\npub fn beta() {}\n",
    );
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    // Every recorded turn is a prefilled part read — the file summary is not
    // decoded inline anymore.
    for (user, _, _) in &sink.turns {
        assert!(
            user.starts_with("Source excerpt — `src/lib.rs` lines "),
            "part user prompt reads a line range, got: {user:?}"
        );
        assert!(
            !user.to_lowercase().contains("summarize"),
            "part prompts must not ask for a summary: {user:?}"
        );
    }
}

#[test]
fn code_read_part_assistant_is_hermes_tool_call_then_response() {
    let dir = fixture("tool_call");
    let root = dir.path().to_path_buf();
    write(&root, "src/lib.rs", b"pub fn alpha() { let _ = 1; }\n");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    // The assistant slot fuses the four-segment exchange: <tool_call>, then the
    // call→response boundary, the <tool_response>, the response→close boundary,
    // and finally the assistant read-confirmation. So the response lands in its
    // own user turn and the exchange closes on an assistant turn.
    let assistant1 = &sink.turns[0].1;
    assert!(assistant1.starts_with("<tool_call>"));
    assert!(assistant1.contains("</tool_call>"));
    assert!(assistant1.contains("\"name\":\"read_file\""));
    assert!(assistant1.contains("\"path\":\"src/lib.rs\""));
    assert!(assistant1.contains("\"start_line\":"));
    assert!(assistant1.contains("\"end_line\":"));
    assert!(assistant1.contains("<tool_response>"));
    assert!(assistant1.contains("</tool_response>"));
    // Closes on the assistant read-confirmation, NOT on the user tool response.
    assert!(!assistant1.ends_with("</tool_response>"));
    assert!(assistant1.trim_end().ends_with('.'));
    assert!(assistant1.contains("Read `src/lib.rs` lines "));

    // Boundary 1 sits BETWEEN the call and the response: the assistant turn ends
    // (`<|im_end|>`) and a user turn opens (`<|im_start|>user`). Without it the
    // model reads the response as the assistant answering its own call.
    let call_end = assistant1.find("</tool_call>").unwrap();
    let resp_start = assistant1.find("<tool_response>").unwrap();
    let between = &assistant1[call_end + "</tool_call>".len()..resp_start];
    assert_eq!(
        between, "<|im_end|>\n<|im_start|>user\n",
        "expected an assistant→user role boundary between call and response, got {between:?}"
    );

    // Boundary 2 sits BETWEEN the response and the closing confirmation: the user
    // turn ends and an assistant turn opens, so the exchange doesn't hang on a
    // user turn.
    let resp_end = assistant1.find("</tool_response>").unwrap();
    let ack_start = assistant1.find("Read `src/lib.rs`").unwrap();
    let between2 = &assistant1[resp_end + "</tool_response>".len()..ack_start];
    assert_eq!(
        between2, "<|im_end|>\n<|im_start|>assistant\n",
        "expected a user→assistant role boundary between response and confirmation, got {between2:?}"
    );
}

#[test]
fn code_read_part_assistant_is_tool_response_with_fenced_code() {
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
        .find(|(_, a, _)| a.contains("<tool_response>") && a.contains("pub fn two"))
        .expect("part turn carrying fn two in its tool_response");
    // The content lives in the assistant slot of a prefill turn (the tool
    // response segment, which is now followed by the closing confirmation).
    assert!(tr.1.contains("<tool_response>\n"));
    assert!(tr.1.contains("</tool_response>"));
    assert!(tr.1.contains("```rust"));
    assert!(tr.1.contains("pub fn two() {"));
    assert!(tr.1.contains("let x = 42;"));
    // The part-read user prompt is the labelled excerpt header.
    assert!(tr.0.starts_with("Source excerpt — `src/lib.rs` lines "));
}

#[test]
fn code_read_user_prompts_never_carry_tool_markup() {
    // No user prompt may contain `<tool_call>` / `<tool_response>` markup.
    // Those tags are reserved for the prefilled assistant slot; were they to
    // leak into a user turn, the dialogue layer's tool-call extractor could
    // not tell a real tool call apart from this prefill marker.
    let dir = fixture("no_user_tool_markup");
    let root = dir.path().to_path_buf();
    write(&root, "src/lib.rs", b"pub fn alpha() {}\n");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    for (user, _, _) in &sink.turns {
        assert!(
            !user.contains("<tool_call>"),
            "user prompt must not contain <tool_call>: {user:?}"
        );
        assert!(
            !user.contains("<tool_response>"),
            "user prompt must not contain <tool_response>: {user:?}"
        );
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

    for (u, a, _) in &sink.turns {
        assert!(!u.contains("blob.bin") && !a.contains("blob.bin"));
        assert!(!u.contains("image.svg") && !a.contains("image.svg"));
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

    // Every turn is a part read ("Source excerpt"); a 250-line file splits
    // into several fixed windows.
    let part_reads: Vec<&str> = sink
        .turns
        .iter()
        .filter(|(u, _, _)| u.contains("notes.txt") && u.starts_with("Source excerpt"))
        .map(|(u, _, _)| u.as_str())
        .collect();
    assert!(
        part_reads.len() >= 3,
        "250-line file should split into ≥ 3 windows, got {}",
        part_reads.len()
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
    let (n_scopes, _state) =
        ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    // Two files → every turn is a part prefill; no inline summary decodes.
    assert_eq!(sink.turns.len(), n_scopes, "one prefill per carved part");

    let snap = progress.snapshot().expect("still loading");
    assert!(snap.progress >= 0.999, "should be at 100% after ingest");
}
