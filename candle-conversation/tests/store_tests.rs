//! Unit tests for candle-conversation store module (no GPU needed).
//!
//! Tests are organized into sections:
//!
//!   § 1  Basic roundtrip
//!   § 2  Text edge cases
//!   § 3  On-disk YAML kind / role
//!   § 4  Tombstones
//!   § 5  KV-cache view field
//!   § 6  Attentional signature (TurnSignature)
//!   § 7  turns_written counter
//!   § 8  SubstrateStore::create / open / path
//!   § 9  Multi-session behaviour
//!   § 10 Partial-write recovery
//!   § 11 Stress / ordering

use std::io::Write;

use candle_conversation::store::{
    Base64Bytes, SubstrateStore, SystemPromptSegment, TurnRecord, TurnSignature,
};
use candle_conversation::tree::TurnType;
use candle_conversation::turn::Role;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a minimal but fully-populated [`TurnSignature`] suitable for tests.
fn test_signature() -> TurnSignature {
    let band = Base64Bytes::from_i8_slice(&[1i8; 128]);
    TurnSignature {
        k_lower: band.clone(),
        k_mid: band.clone(),
        k_upper: band.clone(),
        q_lower: band.clone(),
        q_mid: band.clone(),
        q_upper: band.clone(),
        scale_kl: 0.0234,
        scale_km: 0.0188,
        scale_ku: 0.0301,
        scale_ql: 0.0174,
        scale_qm: 0.0213,
        scale_qu: 0.0398,
    }
}

// ── § 1 Basic roundtrip ───────────────────────────────────────────────────────

#[test]
fn store_roundtrip_single_turn() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.store");

    let record = TurnRecord::new(0, Role::User, "Hello, world!", vec![]);

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store.append_turn(record).unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 0);
    assert_eq!(turns[0].text, "Hello, world!");
    assert_eq!(turns[0].role, Role::User);
}

#[test]
fn store_roundtrip_multiple_turns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.store");

    let records = vec![
        TurnRecord::new(0, Role::User, "You are helpful.", vec![]),
        TurnRecord::new(1, Role::User, "What is 2+2?", vec![0]),
        TurnRecord::new(2, Role::Assistant, "4", vec![0, 1]),
    ];

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        for r in &records {
            store.append_turn(r.clone()).unwrap();
        }
    }

    let read_back = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(read_back.len(), 3);
    for (expected, actual) in records.iter().zip(read_back.iter()) {
        assert_eq!(expected.turn_id, actual.turn_id);
        assert_eq!(expected.text, actual.text);
    }
}

#[test]
fn store_empty_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_text.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(42, Role::Assistant, "", vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].text, "");
}

#[test]
fn store_unicode_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unicode.store");

    let text = "こんにちは世界 🌍 Ñoño";
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(7, Role::User, text, vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].text, text);
}

#[test]
fn store_mixed_roles_roundtrip() {
    // Both User and Assistant roles must be preserved through a full roundtrip.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roles.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "question", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "answer", vec![0]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(2, Role::User, "follow-up", vec![0, 1]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].role, Role::User);
    assert_eq!(turns[1].role, Role::Assistant);
    assert_eq!(turns[2].role, Role::User);
}

// ── § 2 Text edge cases ───────────────────────────────────────────────────────

#[test]
fn store_text_multiline_preserved() {
    // Internal newlines must survive the YAML block-scalar roundtrip.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multiline.store");

    let text = "line one\nline two\nline three";
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, text, vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].text, text);
}

#[test]
fn store_text_trailing_newline_normalized() {
    // The store appends a trailing '\n' before YAML serialization (to force
    // block scalar style) then strips it on read.  Text that already ends with
    // '\n' therefore has that newline removed; text without one is unchanged.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trail_nl.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        // No trailing newline — roundtrips as-is.
        store
            .append_turn(TurnRecord::new(0, Role::User, "hello", vec![]))
            .unwrap();
        // Trailing newline — stripped on read.
        store
            .append_turn(TurnRecord::new(1, Role::User, "world\n", vec![]))
            .unwrap();
        // Internal newlines preserved; only trailing newline is stripped.
        store
            .append_turn(TurnRecord::new(2, Role::User, "a\nb\nc\n", vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].text, "hello");
    assert_eq!(turns[1].text, "world");
    assert_eq!(turns[2].text, "a\nb\nc");
}

#[test]
fn store_text_yaml_special_characters() {
    // Characters with structural meaning in YAML must not corrupt the log.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("special.store");

    let text = "mapping: {key: value}\nlist: [a, b, c]\n# looks like a comment\n&anchor *alias";
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, text, vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    // Trailing '\n' is stripped by normalization; all other content preserved.
    assert_eq!(turns[0].text, text.trim_end_matches('\n'));
}

#[test]
fn store_text_block_scalar_style_in_yaml() {
    // Text fields must be serialized with YAML literal block scalar (`|`) so
    // the log is readable in any text editor.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("block.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "some text", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "reply", vec![0]))
            .unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        raw.contains("text: |"),
        "expected block scalar `|` in:\n{raw}"
    );
}

#[test]
fn store_large_text_roundtrip() {
    // A turn with ~10 KB of text must roundtrip without truncation.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.store");

    let paragraph = "The quick brown fox jumps over the lazy dog. ";
    let text: String = paragraph.repeat(230); // ≈ 10 KB

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::Assistant, &text, vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].text, text);
}

// ── § 3 On-disk YAML kind / role ──────────────────────────────────────────────

#[test]
fn store_user_turn_kind_in_yaml() {
    // User turns must be serialized as `kind: user_turn`.  There must be no
    // separate `role:` field — the role is encoded in the variant name.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user_kind.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "hello", vec![]))
            .unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        raw.contains("kind: user_turn"),
        "missing `kind: user_turn` in:\n{raw}"
    );
    assert!(
        !raw.contains("role:"),
        "unexpected `role:` field in:\n{raw}"
    );
}

#[test]
fn store_assistant_turn_kind_in_yaml() {
    // Assistant turns must use `kind: assistant_turn`.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("asst_kind.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::Assistant, "hi", vec![]))
            .unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        raw.contains("kind: assistant_turn"),
        "missing `kind: assistant_turn` in:\n{raw}"
    );
    assert!(
        !raw.contains("role:"),
        "unexpected `role:` field in:\n{raw}"
    );
}

#[test]
fn store_yaml_documents_start_with_separator() {
    // Every YAML document in the stream must start with `---` so the file is
    // valid for both multi-doc parsers and human readers.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sep.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "a", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "b", vec![0]))
            .unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    let sep_count = raw.matches("\n---\n").count() + raw.starts_with("---\n") as usize;
    assert!(sep_count >= 2, "expected ≥ 2 `---` separators in:\n{raw}");
}

// ── § 4 Tombstones ────────────────────────────────────────────────────────────

#[test]
fn store_tombstone_filters_turn() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tombstone.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "to be deleted", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "kept", vec![0]))
            .unwrap();
        store.append_tombstone(0).unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 1);
    assert_eq!(turns[0].text, "kept");
}

#[test]
fn store_tombstone_multiple() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_tomb.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        for i in 0u64..5 {
            store
                .append_turn(TurnRecord::new(i, Role::User, format!("turn {i}"), vec![]))
                .unwrap();
        }
        store.append_tombstone(1).unwrap();
        store.append_tombstone(3).unwrap();
    }

    let ids: Vec<u64> = SubstrateStore::read_all(&path)
        .unwrap()
        .iter()
        .map(|t| t.turn_id)
        .collect();
    assert_eq!(ids, vec![0, 2, 4]);
}

#[test]
fn store_tombstone_all_turns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("all_tomb.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "gone", vec![]))
            .unwrap();
        store.append_tombstone(0).unwrap();
    }

    assert!(SubstrateStore::read_all(&path).unwrap().is_empty());
}

#[test]
fn store_tombstone_unknown_id_is_ignored() {
    // A tombstone for a turn_id that was never written produces no error and
    // does not suppress any real turn.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unknown_tomb.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store.append_tombstone(99).unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::User, "real", vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 1);
}

#[test]
fn store_tombstone_persists_across_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cross_session_tomb.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "deleted", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "kept", vec![0]))
            .unwrap();
        store.append_tombstone(0).unwrap();
    }
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(2, Role::User, "new turn", vec![1]))
            .unwrap();
    }

    let ids: Vec<u64> = SubstrateStore::read_all(&path)
        .unwrap()
        .iter()
        .map(|t| t.turn_id)
        .collect();
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn store_tombstone_does_not_increment_turns_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("count_tomb.store");

    let mut store = SubstrateStore::open(&path).unwrap();
    store
        .append_turn(TurnRecord::new(0, Role::User, "hello", vec![]))
        .unwrap();
    assert_eq!(store.turns_written(), 1);
    store.append_tombstone(0).unwrap();
    assert_eq!(store.turns_written(), 1); // unchanged
}

#[test]
fn store_tombstone_duplicate_is_idempotent() {
    // Tombstoning the same turn_id twice must not error and must suppress the
    // turn exactly once.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup_tomb.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "text", vec![]))
            .unwrap();
        store.append_tombstone(0).unwrap();
        store.append_tombstone(0).unwrap();
    }

    assert!(SubstrateStore::read_all(&path).unwrap().is_empty());
}

#[test]
fn store_tombstone_written_before_its_turn_still_suppresses() {
    // read_all computes the tombstone set from all documents regardless of
    // position.  A tombstone that appears before the turn it targets must still
    // cause that turn to be filtered out.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tombstone_pre.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store.append_tombstone(5).unwrap();
        store
            .append_turn(TurnRecord::new(5, Role::User, "should be gone", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(6, Role::User, "should remain", vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 6);
}

#[test]
fn store_tombstone_interleaved_with_new_appends() {
    // Tombstoning in the middle of an append sequence must not corrupt the log.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("interleaved.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "q1", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "a1", vec![0]))
            .unwrap();
        store.append_tombstone(0).unwrap();
        store
            .append_turn(TurnRecord::new(2, Role::User, "q2", vec![1]))
            .unwrap();
        store.append_tombstone(1).unwrap();
        store
            .append_turn(TurnRecord::new(3, Role::Assistant, "a2", vec![2]))
            .unwrap();
    }

    let ids: Vec<u64> = SubstrateStore::read_all(&path)
        .unwrap()
        .iter()
        .map(|t| t.turn_id)
        .collect();
    assert_eq!(ids, vec![2, 3]);
}

// ── § 5 KV-cache view field ───────────────────────────────────────────────────

#[test]
fn store_view_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("view.store");

    let view = vec![0u64, 1, 5, 23, 99];
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(
                100,
                Role::Assistant,
                "response",
                view.clone(),
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].view, view);
}

#[test]
fn store_view_empty_persists() {
    // An empty view (no history for the first turn) must roundtrip as `[]`.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_view.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "hi", vec![]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert!(turns[0].view.is_empty());
}

#[test]
fn store_view_large_ids() {
    // view entries are u64 — values near the u64 maximum must survive.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("view_large.store");

    let view = vec![0u64, u32::MAX as u64, u64::MAX / 2];
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "reply", view.clone()))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].view, view);
}

// ── § 6 Attentional signature ─────────────────────────────────────────────────

#[test]
fn store_signature_roundtrip() {
    // All six band vectors and six scale values must survive a full roundtrip.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sig.store");

    let sig = test_signature();
    let mut record = TurnRecord::new(0, Role::Assistant, "text", vec![]);
    record.signature = Some(sig.clone());

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store.append_turn(record).unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    let s = turns[0]
        .signature
        .as_ref()
        .expect("signature should be present");
    assert_eq!(s.k_lower.as_bytes(), sig.k_lower.as_bytes());
    assert_eq!(s.k_mid.as_bytes(), sig.k_mid.as_bytes());
    assert_eq!(s.k_upper.as_bytes(), sig.k_upper.as_bytes());
    assert_eq!(s.q_lower.as_bytes(), sig.q_lower.as_bytes());
    assert_eq!(s.q_mid.as_bytes(), sig.q_mid.as_bytes());
    assert_eq!(s.q_upper.as_bytes(), sig.q_upper.as_bytes());
    assert!((s.scale_kl - sig.scale_kl).abs() < 1e-6);
    assert!((s.scale_km - sig.scale_km).abs() < 1e-6);
    assert!((s.scale_ku - sig.scale_ku).abs() < 1e-6);
    assert!((s.scale_ql - sig.scale_ql).abs() < 1e-6);
    assert!((s.scale_qm - sig.scale_qm).abs() < 1e-6);
    assert!((s.scale_qu - sig.scale_qu).abs() < 1e-6);
}

#[test]
fn store_signature_absent_from_yaml_when_none() {
    // When no signature is attached the `signature:` key must not appear in
    // the YAML (`skip_serializing_if = "Option::is_none"`).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_sig.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::Assistant, "no sig", vec![]))
            .unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains("signature:"),
        "unexpected `signature:` in:\n{raw}"
    );
}

#[test]
fn store_user_turn_has_no_signature_in_yaml() {
    // User turns use the `user_turn` schema, which has no signature field.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user_no_sig.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "question", vec![]))
            .unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains("signature:"),
        "unexpected `signature:` in user turn:\n{raw}"
    );
}

#[test]
fn store_signature_keys_present_in_yaml_when_set() {
    // When a signature is present all twelve keys must be visible in raw YAML.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sig_keys.store");

    let mut record = TurnRecord::new(0, Role::Assistant, "text", vec![]);
    record.signature = Some(test_signature());

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store.append_turn(record).unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    for key in &[
        "k_lower", "k_mid", "k_upper", "q_lower", "q_mid", "q_upper", "scale_kl", "scale_km",
        "scale_ku", "scale_ql", "scale_qm", "scale_qu",
    ] {
        assert!(
            raw.contains(key),
            "missing `{key}` in signature YAML:\n{raw}"
        );
    }
}

#[test]
fn store_signature_survives_tombstone_and_reopen() {
    // A signed assistant turn that is NOT tombstoned must keep its signature
    // across file close and reopen.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sig_tomb.store");

    let sig = test_signature();
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "q", vec![]))
            .unwrap();

        let mut record = TurnRecord::new(1, Role::Assistant, "a", vec![0]);
        record.signature = Some(sig.clone());
        store.append_turn(record).unwrap();

        // Tombstone the user turn, not the assistant turn.
        store.append_tombstone(0).unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 1);
    let s = turns[0]
        .signature
        .as_ref()
        .expect("signature must survive tombstone of another turn");
    assert_eq!(s.k_lower.as_bytes(), sig.k_lower.as_bytes());
}

// ── § 7 turns_written counter ─────────────────────────────────────────────────

#[test]
fn store_turns_written_increments_per_append() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("counter.store");
    let mut store = SubstrateStore::open(&path).unwrap();

    assert_eq!(store.turns_written(), 0);
    store
        .append_turn(TurnRecord::new(0, Role::User, "a", vec![]))
        .unwrap();
    assert_eq!(store.turns_written(), 1);
    store
        .append_turn(TurnRecord::new(1, Role::Assistant, "b", vec![0]))
        .unwrap();
    assert_eq!(store.turns_written(), 2);
    store
        .append_turn(TurnRecord::new(2, Role::User, "c", vec![]))
        .unwrap();
    assert_eq!(store.turns_written(), 3);
}

#[test]
fn store_turns_written_resets_on_new_session() {
    // The counter reflects only the current open session; re-opening a file
    // that already has turns starts the counter at 0.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reset_counter.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "a", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::User, "b", vec![]))
            .unwrap();
        assert_eq!(store.turns_written(), 2);
    }

    let fresh = SubstrateStore::open(&path).unwrap();
    assert_eq!(fresh.turns_written(), 0);
}

// ── § 8 SubstrateStore::create / open / path ───────────────────────────────

#[test]
fn store_create_with_header_then_read_turns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("with_header.store");

    let system_prompt = vec![
        SystemPromptSegment::Static {
            text: "You are a helpful assistant.".to_string(),
        },
        SystemPromptSegment::Section {
            name: "mood".to_string(),
        },
        SystemPromptSegment::Section {
            name: "conversation_history".to_string(),
        },
    ];

    {
        let mut store = SubstrateStore::create(&path, system_prompt).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "Hello", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "Hi there!", vec![0]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].turn_id, 0);
    assert_eq!(turns[1].turn_id, 1);
    assert_eq!(turns[1].view, vec![0u64]);

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("kind: conversation"));
    assert!(raw.contains("You are a helpful assistant."));
    assert!(raw.contains("kind: section"));
    assert!(raw.contains("name: mood"));
}

#[test]
fn store_create_fails_if_file_exists() {
    // `create` must error when the target path already exists; callers must
    // use `open` to append to an existing conversation.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exists.store");

    SubstrateStore::create(&path, vec![]).unwrap();
    assert!(
        SubstrateStore::create(&path, vec![]).is_err(),
        "expected an error creating a store at an existing path"
    );
}

#[test]
fn store_read_all_header_only_returns_empty() {
    // A file with only the conversation header and no turn documents must
    // return an empty vec.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("header_only.store");

    SubstrateStore::create(
        &path,
        vec![SystemPromptSegment::Static {
            text: "system".to_string(),
        }],
    )
    .unwrap();

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert!(turns.is_empty());
}

#[test]
fn store_open_creates_file_if_absent() {
    // Opening a path that does not exist must create the file silently.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("new.store");

    assert!(!path.exists());
    let mut store = SubstrateStore::open(&path).unwrap();
    assert!(path.exists());

    store
        .append_turn(TurnRecord::new(0, Role::User, "hello", vec![]))
        .unwrap();
    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
}

#[test]
fn store_path_accessor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("path_check.store");
    let store = SubstrateStore::open(&path).unwrap();
    assert_eq!(store.path(), path.as_path());
}

// ── § 9 Multi-session behaviour ───────────────────────────────────────────────

#[test]
fn store_append_across_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "first", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "second", vec![0]))
            .unwrap();
    }
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(2, Role::User, "third", vec![0, 1]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[2].text, "third");
}

#[test]
fn store_multiple_sessions_interleaved_tombstones() {
    // Session 1 writes, session 2 tombstones and writes more, session 3 tombstones again.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_session_tomb.store");

    // Session 1: turns 0, 1, 2
    {
        let mut s = SubstrateStore::open(&path).unwrap();
        for i in 0u64..3 {
            s.append_turn(TurnRecord::new(i, Role::User, format!("t{i}"), vec![]))
                .unwrap();
        }
    }
    // Session 2: tombstone 0, add turns 3 and 4
    {
        let mut s = SubstrateStore::open(&path).unwrap();
        s.append_tombstone(0).unwrap();
        s.append_turn(TurnRecord::new(3, Role::User, "t3", vec![]))
            .unwrap();
        s.append_turn(TurnRecord::new(4, Role::User, "t4", vec![]))
            .unwrap();
    }
    // Session 3: tombstone 2 and 4
    {
        let mut s = SubstrateStore::open(&path).unwrap();
        s.append_tombstone(2).unwrap();
        s.append_tombstone(4).unwrap();
    }

    let ids: Vec<u64> = SubstrateStore::read_all(&path)
        .unwrap()
        .iter()
        .map(|t| t.turn_id)
        .collect();
    assert_eq!(ids, vec![1, 3]);
}

// ── § 10 Partial-write recovery ───────────────────────────────────────────────

#[test]
fn store_read_all_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.store");
    std::fs::write(&path, "").unwrap();

    assert!(SubstrateStore::read_all(&path).unwrap().is_empty());
}

#[test]
fn store_read_all_whitespace_only_file() {
    // A file containing only whitespace is treated as empty.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ws.store");
    std::fs::write(&path, "\n\n\n").unwrap();

    assert!(SubstrateStore::read_all(&path).unwrap().is_empty());
}

#[test]
fn store_recovers_from_truncated_last_document() {
    // Simulate a crash mid-write: two valid turns followed by a partial
    // document missing the required `view` field.  read_all must return the
    // two valid turns and silently discard the incomplete tail.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "safe turn", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(
                1,
                Role::Assistant,
                "safe response",
                vec![0],
            ))
            .unwrap();
    }

    // Append an incomplete document (missing the required `view:` field).
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        write!(
            f,
            "---\nkind: user_turn\nturn_id: 2\ntext: |\n  write was cut short\n"
        )
        .unwrap();
        f.flush().unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(
        turns.len(),
        2,
        "partial document should be silently dropped"
    );
    assert_eq!(turns[0].turn_id, 0);
    assert_eq!(turns[1].turn_id, 1);
}

#[test]
fn store_recovers_from_malformed_last_document() {
    // Like the truncation test but the corrupt tail contains syntactically
    // broken YAML rather than a structurally incomplete document.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "good", vec![]))
            .unwrap();
    }

    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        write!(f, "---\n{{broken: [yaml: without closing bracket\n").unwrap();
        f.flush().unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 0);
}

#[test]
fn store_recovers_from_unknown_kind_in_last_document() {
    // A document with an unrecognised `kind:` (e.g. a future format version)
    // at the tail must be silently dropped rather than crashing read_all.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unknown_kind.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "known", vec![]))
            .unwrap();
    }

    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        write!(
            f,
            "---\nkind: future_format_v99\nturn_id: 1\nsome_new_field: x\n"
        )
        .unwrap();
        f.flush().unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
}

// ── § 11 On-disk format & stress ─────────────────────────────────────────────

#[test]
fn store_yaml_is_human_readable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readable.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(42, Role::User, "check format", vec![]))
            .unwrap();
        store.append_tombstone(42).unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("kind: user_turn"));
    assert!(raw.contains("turn_id: 42"));
    assert!(raw.contains("check format"));
    assert!(raw.contains("kind: tombstone"));
    assert!(!raw.contains("token_ids:"));
}

#[test]
fn store_many_turns_preserve_order() {
    // 100 turns written across two sessions must come back in insertion order
    // with all text and turn_id values intact.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        for i in 0u64..50 {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            store
                .append_turn(TurnRecord::new(i, role, format!("turn {i}"), vec![]))
                .unwrap();
        }
    }
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        for i in 50u64..100 {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            store
                .append_turn(TurnRecord::new(i, role, format!("turn {i}"), vec![]))
                .unwrap();
        }
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 100);
    for (i, t) in turns.iter().enumerate() {
        assert_eq!(t.turn_id, i as u64, "turn_id mismatch at position {i}");
        assert_eq!(t.text, format!("turn {i}"), "text mismatch at position {i}");
    }
}

// ── § 12 Summary turns ────────────────────────────────────────────────────────

#[test]
fn store_summary_turn_sleep_roundtrip() {
    // A Sleep summary turn roundtrips with the correct turn_type and covers.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sleep.store");

    let covers = vec![1u64, 2, 3];
    let view = vec![1u64, 2, 3];
    let text = "The day felt heavy with unresolved tension.";

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                10,
                TurnType::Sleep,
                text,
                covers.clone(),
                view.clone(),
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 10);
    assert_eq!(turns[0].turn_type, TurnType::Sleep);
    assert_eq!(turns[0].text, text);
    assert_eq!(turns[0].covers, covers);
    assert_eq!(turns[0].view, view);
    assert_eq!(turns[0].role, Role::Assistant);
}

#[test]
fn store_summary_turn_thought_roundtrip() {
    // Thought (daydream) turns roundtrip correctly.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("thought.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                5,
                TurnType::Thought,
                "A flicker of recognition.",
                vec![2, 3],
                vec![2, 3],
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].turn_type, TurnType::Thought);
    assert_eq!(turns[0].text, "A flicker of recognition.");
}

#[test]
fn store_summary_turn_reason_roundtrip() {
    // Reason (planning) turns roundtrip correctly.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reason.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                7,
                TurnType::Reason,
                "The plan: stay patient.",
                vec![4, 5, 6],
                vec![4, 5, 6],
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].turn_type, TurnType::Reason);
    assert_eq!(turns[0].covers, vec![4, 5, 6]);
}

#[test]
fn store_summary_turn_kind_in_yaml() {
    // summary_turn entries must appear as `kind: summary_turn` and include
    // a `turn_type:` field and a `covers:` list in the raw YAML.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("summary_yaml.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                20,
                TurnType::Sleep,
                "dream text",
                vec![10, 11],
                vec![10, 11],
            ))
            .unwrap();
    }

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        raw.contains("kind: summary_turn"),
        "missing `kind: summary_turn` in:\n{raw}"
    );
    assert!(
        raw.contains("turn_type: sleep"),
        "missing `turn_type: sleep` in:\n{raw}"
    );
    assert!(raw.contains("covers:"), "missing `covers:` in:\n{raw}");
    assert!(
        raw.contains("text: |"),
        "expected block scalar `|` in:\n{raw}"
    );
    // Role must not leak as a separate field.
    assert!(
        !raw.contains("role:"),
        "unexpected `role:` field in:\n{raw}"
    );
}

#[test]
fn store_summary_covers_large_set() {
    // covers can reference many sub-turns, including Reality and other summaries.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_covers.store");

    let covers: Vec<u64> = (0u64..20).collect();
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                99,
                TurnType::Sleep,
                "summary of many",
                covers.clone(),
                covers.clone(),
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].covers, covers);
}

#[test]
fn store_summary_covers_other_summaries() {
    // A summary can cover other summary turns — supports recursive hierarchies.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recursive_summary.store");

    // Level-1: summarizes Reality turns 1 and 2.
    // Level-2: summarizes level-1 summary (turn 10) and Reality turn 3.
    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::User, "q1", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(2, Role::Assistant, "a1", vec![1]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(3, Role::User, "q2", vec![1, 2]))
            .unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                10,
                TurnType::Sleep,
                "level-1 summary",
                vec![1, 2],
                vec![1, 2],
            ))
            .unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                11,
                TurnType::Reason,
                "level-2 meta-summary",
                vec![10, 3],
                vec![10, 3],
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 5);

    let lvl1 = turns.iter().find(|t| t.turn_id == 10).unwrap();
    assert_eq!(lvl1.turn_type, TurnType::Sleep);
    assert_eq!(lvl1.covers, vec![1, 2]);

    let lvl2 = turns.iter().find(|t| t.turn_id == 11).unwrap();
    assert_eq!(lvl2.turn_type, TurnType::Reason);
    assert_eq!(lvl2.covers, vec![10, 3]);
}

#[test]
fn store_summary_turn_tombstone_suppresses_it() {
    // Tombstoning a summary turn removes it, just like Reality turns.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("summary_tomb.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::User, "q", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                2,
                TurnType::Sleep,
                "to be removed",
                vec![1],
                vec![1],
            ))
            .unwrap();
        store.append_tombstone(2).unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, 1);
}

#[test]
fn store_summary_turn_with_signature() {
    // Signatures roundtrip correctly on summary turns.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("summary_sig.store");

    let sig = test_signature();
    let mut record =
        TurnRecord::new_summary(5, TurnType::Thought, "a thought", vec![3, 4], vec![3, 4]);
    record.signature = Some(sig.clone());

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store.append_turn(record).unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    let s = turns[0]
        .signature
        .as_ref()
        .expect("signature must be present");
    assert_eq!(s.k_upper.as_bytes(), sig.k_upper.as_bytes());
    assert!((s.scale_qu - sig.scale_qu).abs() < 1e-6);
}

#[test]
fn store_summary_turn_empty_covers() {
    // A summary_turn with an empty covers list is valid (e.g. a Reason plan
    // not attributable to specific prior turns).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_covers.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                1,
                TurnType::Reason,
                "the plan going forward",
                vec![],
                vec![],
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns[0].covers, Vec::<u64>::new());
    assert_eq!(turns[0].turn_type, TurnType::Reason);
}

#[test]
fn store_summary_mixed_stream_ordering() {
    // A mixed stream of Reality and summary turns preserves insertion order
    // on read, and all fields are distinct between variants.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::User, "hello", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(2, Role::Assistant, "hi", vec![1]))
            .unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                3,
                TurnType::Sleep,
                "dream about the greeting",
                vec![1, 2],
                vec![1, 2],
            ))
            .unwrap();
        store
            .append_turn(TurnRecord::new(
                4,
                Role::User,
                "how are you?",
                vec![1, 2, 3],
            ))
            .unwrap();
        store
            .append_turn(TurnRecord::new_summary(
                5,
                TurnType::Thought,
                "a flicker",
                vec![4],
                vec![4],
            ))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert_eq!(turns.len(), 5);
    assert_eq!(turns[0].turn_id, 1);
    assert_eq!(turns[0].turn_type, TurnType::Reality);
    assert_eq!(turns[1].turn_id, 2);
    assert_eq!(turns[1].turn_type, TurnType::Reality);
    assert_eq!(turns[2].turn_id, 3);
    assert_eq!(turns[2].turn_type, TurnType::Sleep);
    assert_eq!(turns[3].turn_id, 4);
    assert_eq!(turns[3].turn_type, TurnType::Reality);
    assert_eq!(turns[4].turn_id, 5);
    assert_eq!(turns[4].turn_type, TurnType::Thought);
}

#[test]
fn store_reality_turns_have_empty_covers() {
    // Reality turns always produce empty covers — the field is only meaningful
    // for summary turns.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reality_covers.store");

    {
        let mut store = SubstrateStore::open(&path).unwrap();
        store
            .append_turn(TurnRecord::new(0, Role::User, "q", vec![]))
            .unwrap();
        store
            .append_turn(TurnRecord::new(1, Role::Assistant, "a", vec![0]))
            .unwrap();
    }

    let turns = SubstrateStore::read_all(&path).unwrap();
    assert!(
        turns[0].covers.is_empty(),
        "user turn should have empty covers"
    );
    assert!(
        turns[1].covers.is_empty(),
        "assistant turn should have empty covers"
    );
}
