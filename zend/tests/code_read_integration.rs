//! Tier-2 integration test for the `code_reading` layer's per-file
//! tool-round-trip conversation shape.
//!
//! Each carved scope is ingested as a **tool round-trip of two coupled turns**
//! (see `Sequence::ingest_scope_roundtrip`):
//!   Turn A (call):     user("Summarize `X` (lines N-M) …") → assistant(<tool_call>)
//!   Turn B (response):  user(<tool_response>…source…)      → assistant(DECODED summary)
//!
//! Recording it as two turns — rather than one baked four-segment blob — is what
//! keeps the inter-turn role seams (`user_start` / `assistant_end` between and
//! around the turns) as REGENERATED live glue: a baked seam goes stale when the
//! scope is re-injected at a different position mid-dialogue. So the sink records
//! **two** turns per scope, and the assistant/user strings carry **no** baked
//! role markers.
//!
//! The [`RecordingTurnSink`] has no model, so the response turn's summary is
//! recorded empty here (the real engine decodes it under `/no_think`); these
//! tests assert the call turn's shape, the response turn's tool_response, and the
//! absence of baked boundary markers. The decode is exercised in the scheduler.

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

/// The recorded turns split into (call, response) pairs — even indices are call
/// turns, odd indices are their response turns.
fn is_call_turn(i: usize) -> bool {
    i % 2 == 0
}

#[test]
fn code_read_emits_two_coupled_turns_per_scope() {
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

    // Each scope is a tool round-trip = TWO turns (call + response).
    assert_eq!(
        sink.turns.len(),
        2 * n_scopes,
        "two coupled turns (call + response) per carved scope"
    );
    // Gather-scope tags: every turn carries `["code", <path>]` so it lands in the
    // code-tagged provenance gallery (and out of the untagged dialogue partition).
    for (_, _, tags) in &sink.turns {
        assert_eq!(tags, &["code", "src/lib.rs"]);
    }
}

#[test]
fn code_read_call_turn_is_summarize_request_then_tool_call() {
    let dir = fixture("call_shape");
    let root = dir.path().to_path_buf();
    write(&root, "src/lib.rs", b"pub fn alpha() { let _ = 1; }\n");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    // The FIRST turn (the call): user is the two-sentence summarise request; the
    // assistant is JUST the `<tool_call>` — no baked boundary markers, no
    // tool_response (that's the next, coupled turn). The seams are regenerated.
    let (call_user, call_assistant, _) = &sink.turns[0];
    assert!(call_user.starts_with("Summarize `src/lib.rs` (lines "));
    assert!(call_user.contains("in no more than two sentences"));
    assert!(call_assistant.starts_with("<tool_call>"));
    assert!(call_assistant.trim_end().ends_with("</tool_call>"));
    assert!(call_assistant.contains("\"name\":\"file_read\""));
    assert!(call_assistant.contains("\"path\":\"src/lib.rs\""));
    assert!(call_assistant.contains("\"start_line\":"));
    assert!(call_assistant.contains("\"end_line\":"));
    // NO tool_response and NO baked role markers in the call turn.
    assert!(!call_assistant.contains("<tool_response>"));
    assert!(!call_assistant.contains("<|im_end|>"));
    assert!(!call_assistant.contains("<|im_start|>"));
}

#[test]
fn code_read_response_turn_is_tool_response_with_fenced_code() {
    let dir = fixture("response_shape");
    let root = dir.path().to_path_buf();
    let src = b"pub fn one() {}\npub fn two() {\n    let x = 42;\n}\n";
    write(&root, "src/lib.rs", src);
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    // The tool_response comes back in a USER turn (the Hermes/Qwen convention) —
    // the second, coupled half of the round-trip. Its assistant half is the
    // DECODED summary, empty in this model-less sink.
    let (resp_user, resp_assistant, _) = sink
        .turns
        .iter()
        .enumerate()
        .find(|(i, (u, _, _))| !is_call_turn(*i) && u.contains("pub fn two"))
        .map(|(_, t)| t)
        .expect("a response turn carrying fn two in its tool_response");
    assert!(resp_user.contains("<tool_response>\n"));
    assert!(resp_user.contains("</tool_response>"));
    assert!(resp_user.contains("```rust"));
    assert!(resp_user.contains("pub fn two() {"));
    assert!(resp_user.contains("let x = 42;"));
    // The summary is decoded by the real engine; the model-less sink records it empty.
    assert_eq!(resp_assistant, "");
    // No baked role markers in the response user turn either.
    assert!(!resp_user.contains("<|im_end|>"));
    assert!(!resp_user.contains("<|im_start|>"));
}

#[test]
fn code_read_call_user_never_carries_tool_markup() {
    // The CALL user prompt is a plain summarise request — it must never contain
    // `<tool_call>` / `<tool_response>` markup (that would confuse the dialogue
    // layer's tool-call extractor). The tool_response legitimately lands in the
    // RESPONSE user turn (a tool result is a user turn), so that one is exempt.
    let dir = fixture("no_call_markup");
    let root = dir.path().to_path_buf();
    write(&root, "src/lib.rs", b"pub fn alpha() {}\n");
    let map = walk_workspace(&root);

    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    ingest_code_reading_into_sink(&mut sink, &root, &map, &progress).unwrap();

    for (i, (user, _, _)) in sink.turns.iter().enumerate() {
        assert!(
            !user.contains("<tool_call>"),
            "no user turn may contain <tool_call>: {user:?}"
        );
        if is_call_turn(i) {
            assert!(
                !user.contains("<tool_response>"),
                "the call user prompt must not contain <tool_response>: {user:?}"
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

    // Each scope's CALL turn is "Summarize `notes.txt` (lines …)"; a 250-line file
    // splits into several fixed windows.
    let call_reads: Vec<&str> = sink
        .turns
        .iter()
        .enumerate()
        .filter(|(i, (u, _, _))| {
            is_call_turn(*i) && u.contains("notes.txt") && u.starts_with("Summarize ")
        })
        .map(|(_, (u, _, _))| u.as_str())
        .collect();
    assert!(
        call_reads.len() >= 3,
        "250-line file should split into ≥ 3 windows, got {}",
        call_reads.len()
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

    // Two files → every scope is a two-turn round-trip.
    assert_eq!(
        sink.turns.len(),
        2 * n_scopes,
        "two coupled turns per carved scope"
    );

    let snap = progress.snapshot().expect("still loading");
    assert!(snap.progress >= 0.999, "should be at 100% after ingest");
}
