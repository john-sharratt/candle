//! Integration tests for the summarization pipeline.
//!
//! Verifies that feeding N turns into a [`Sequence`] with a low
//! `summarize_every` value triggers [`SummarizationTask`] synchronously
//! (inside [`Sequence::send`]), and that the resulting
//! [`ConversationSegment`] node added to the tree contains well-formed,
//! non-empty summary text.
//!
//! Uses `Qwen2-0.5B-Instruct Q4_0` (~0.4 GB) — the smallest available preset
//! — for speed.  Set `summarize_on_day_boundary: false` so only the
//! count-based trigger is active and results are deterministic.
//!
//! Run with:
//! ```bash
//! cargo test -p candle-conversation --features hub \
//!     --test summarization_tests -- --nocapture
//! ```

use candle_conversation::{
    models::{Model, ModelBuilder},
    ConversationEngine, ConversationNode, ConversationTreeConfig, SamplingConfig, SequenceConfig,
};
use std::time::{Duration, Instant};

// ────────────────────────────────────────────────────────────────────────────
// Test harness — each test loads its own engine instance
// ────────────────────────────────────────────────────────────────────────────

const SUMM_MODEL: Model = Model::Qwen2_0_5B;

fn summ_builder() -> ModelBuilder {
    SUMM_MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(64)
        .max_concurrent(4)
}

fn engine() -> ConversationEngine {
    let device =
        candle::Device::cuda_if_available(0).expect("CUDA device required for integration tests");
    eprintln!("\n=== Loading {} ===", SUMM_MODEL);
    let start = Instant::now();
    let e = summ_builder()
        .engine(&device)
        .expect("failed to load model");
    eprintln!("   Loaded in {:.2}s\n", start.elapsed().as_secs_f64());
    e
}

/// Build a `SequenceConfig` using `summarize_every = n` and no day-boundary trigger.
fn config_with_every(summarize_every: u32) -> SequenceConfig {
    let mut cfg = summ_builder().conversation_config();
    cfg.tree = ConversationTreeConfig {
        summarize_every,
        summarize_on_day_boundary: false,
        ..ConversationTreeConfig::default()
    };
    cfg
}

/// Build a `SequenceConfig` with both turn-level and segment-level
/// summarization thresholds set, and no day-boundary trigger.
///
/// Uses `summarization_max_tokens: 32` so each summary call is fast (avoids
/// the 0.5B model spending 30+ seconds generating 256 copied tokens).
fn config_with_every_and_segment_every(
    summarize_every: u32,
    segment_summarize_every: u32,
) -> SequenceConfig {
    let mut cfg = summ_builder().conversation_config();
    cfg.tree = ConversationTreeConfig {
        summarize_every,
        segment_summarize_every,
        summarize_on_day_boundary: false,
        summarization_max_tokens: 32,
        ..ConversationTreeConfig::default()
    };
    cfg
}

fn system_prompt() -> String {
    summ_builder().format_system_prompt()
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Count segment nodes in the current tree.
fn segment_count(conv: &candle_conversation::Sequence) -> usize {
    conv.tree()
        .nodes()
        .filter(|n| matches!(n, ConversationNode::Segment(_)))
        .count()
}

/// Return the last segment node in the tree, or `None`.
fn last_segment(
    conv: &candle_conversation::Sequence,
) -> Option<candle_conversation::ConversationSegment> {
    conv.tree()
        .nodes()
        .filter_map(|n| n.as_segment())
        .last()
        .cloned()
}

/// Poll for a segment node to appear in the tree with an explicit wall-clock
/// timeout.
///
/// Because [`Sequence::send`] already blocks until the summarization task
/// completes (via `run_task_blocking_inner`), the segment is normally present
/// the instant `send` returns.  This helper guards against regressions where
/// the synchronous drain is accidentally removed, preventing a test hang.
fn wait_for_any_segment(conv: &candle_conversation::Sequence, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if segment_count(conv) > 0 {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

/// Feeding exactly `summarize_every` turns should append exactly one segment
/// node to the tree by the time the triggering `send()` returns.
#[test]
#[ignore]
fn test_summarization_triggers_at_threshold() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    // Exactly summarize_every = 3 turns.
    for i in 1..=3u32 {
        let resp = conv
            .send_turn(&format!("Message number {i}."))
            .expect("send failed");
        eprintln!("Turn {i}: {}", resp.text);
    }

    let found = wait_for_any_segment(&conv, Duration::from_secs(30));
    assert!(
        found,
        "expected a segment node after {} turns with summarize_every=3, but none appeared \
         within the timeout",
        3
    );
    eprintln!("✓ segment appeared after 3 turns");
    conv.close().ok();
}

/// The summary text inside the segment must be non-empty.
///
/// An empty summary would indicate that the summarization inference returned
/// nothing, or the `TokenizedText` was not populated.
#[test]
#[ignore]
fn test_summarization_segment_text_is_nonempty() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=3u32 {
        conv.send_turn(&format!("Turn {i} here."))
            .expect("send failed");
    }

    let seg = last_segment(&conv).expect("no segment node found");
    let text = seg.inner().summary_text.text().to_string();
    eprintln!("Summary text ({} chars): {:?}", text.len(), text);

    assert!(!text.is_empty(), "segment summary_text should not be empty");
    conv.close().ok();
}

/// The summary must be at least 10 characters long, ensuring the model
/// actually generated meaningful content rather than returning a trivial or
/// degenerate response.
#[test]
#[ignore]
fn test_summarization_segment_has_minimum_length() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=3u32 {
        conv.send_turn(&format!("Statement {i}: the sky is blue."))
            .expect("send failed");
    }

    let seg = last_segment(&conv).expect("no segment node found");
    let text = seg.inner().summary_text.text();
    eprintln!(
        "Summary length: {} chars — {:?}",
        text.len(),
        &text[..text.len().min(120)]
    );

    assert!(
        text.len() >= 10,
        "summary text is too short ({} chars) — expected at least 10",
        text.len()
    );
    conv.close().ok();
}

/// After 3 turns with `summarize_every=3` the segment's `SegmentId` must
/// span turns 1 through 3 inclusive.
#[test]
#[ignore]
fn test_summarization_segment_covers_correct_turn_range() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=3u32 {
        conv.send_turn(&format!("Turn {i}.")).expect("send failed");
    }

    let seg = last_segment(&conv).expect("no segment node found");
    let sid = seg.inner().segment_id;

    eprintln!(
        "Segment: start_turn.seq={}, end_turn.seq={}",
        sid.start_turn.seq, sid.end_turn.seq
    );

    assert_eq!(
        sid.start_turn.seq, 1,
        "segment start_turn.seq should be 1 (first turn), got {}",
        sid.start_turn.seq
    );
    assert_eq!(
        sid.end_turn.seq, 3,
        "segment end_turn.seq should be 3 (last turn), got {}",
        sid.end_turn.seq
    );
    conv.close().ok();
}

/// After summarization the segment is the *parent* of the turns it
/// summarises.  The top-level `nodes` vec should contain exactly 1 node
/// (the segment), and that segment's `children` should hold the 3 turns.
#[test]
#[ignore]
fn test_summarization_turn_nodes_preserved_after_segment() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=3u32 {
        conv.send_turn(&format!("Turn {i}.")).expect("send failed");
    }

    let nodes: Vec<_> = conv.tree().nodes().collect();
    eprintln!("Top-level node count: {}", nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        match n {
            ConversationNode::Turn(t) => {
                eprintln!(
                    "  [{}] Turn(seq={}, user={:?})",
                    i,
                    t.inner().turn_id.seq,
                    &t.inner().user.text()[..t.inner().user.text().len().min(40)]
                );
            }
            ConversationNode::Segment(s) => {
                eprintln!(
                    "  [{}] Segment({}..={}, {} children)",
                    i,
                    s.inner().segment_id.start_turn.seq,
                    s.inner().segment_id.end_turn.seq,
                    s.inner().children.len()
                );
                for (j, child) in s.inner().children.iter().enumerate() {
                    if let ConversationNode::Turn(ct) = child {
                        eprintln!("       child[{}] Turn(seq={})", j, ct.inner().turn_id.seq);
                    }
                }
            }
        }
    }

    // Top-level list: exactly 1 node (the segment).
    assert_eq!(
        nodes.len(),
        1,
        "expected 1 top-level node (segment), got {}",
        nodes.len()
    );

    let seg = nodes[0].as_segment().expect("top-level node should be a Segment (test_summarization_turn_nodes_preserved_after_segment)");
    let child_turn_count = seg
        .inner()
        .children
        .iter()
        .filter(|n| n.as_turn().is_some())
        .count();
    assert_eq!(
        child_turn_count, 3,
        "segment should have 3 child turns, got {}",
        child_turn_count
    );

    conv.close().ok();
}

/// The segment is the structural parent of the turns it summarises: the
/// top-level `nodes` vec should hold exactly the segment, and its `children`
/// should contain the turns in ascending seq order.
#[test]
#[ignore]
fn test_summarization_segment_is_parent_of_turns() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=3u32 {
        conv.send_turn(&format!("Message number {i}."))
            .expect("send failed");
    }

    let nodes: Vec<_> = conv.tree().nodes().collect();

    // One top-level node: the segment.
    assert_eq!(
        nodes.len(),
        1,
        "expected 1 top-level node after summarisation, got {}",
        nodes.len()
    );

    let seg = nodes[0]
        .as_segment()
        .expect("top-level node should be a Segment");
    let children = &seg.inner().children;

    assert_eq!(
        children.len(),
        3,
        "segment should have exactly 3 child nodes, got {}",
        children.len()
    );

    // Every child is a Turn with the expected seq (1, 2, 3 in order).
    for (expected_seq, child) in (1u32..=3).zip(children.iter()) {
        let turn = child
            .as_turn()
            .expect("each child of the segment should be a Turn");
        assert_eq!(
            turn.inner().turn_id.seq,
            expected_seq,
            "child turn seq: expected {expected_seq}, got {}",
            turn.inner().turn_id.seq
        );
    }

    // The segment's own SegmentId bounds match its children.
    assert_eq!(seg.inner().segment_id.start_turn.seq, 1);
    assert_eq!(seg.inner().segment_id.end_turn.seq, 3);

    conv.close().ok();
}

/// Feeding only `summarize_every - 1` turns must NOT produce a segment node.
/// The trigger fires at turn N, not before it.
#[test]
#[ignore]
fn test_summarization_does_not_trigger_early() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    // Only 2 turns — below threshold of 3.
    for i in 1..=2u32 {
        conv.send_turn(&format!("Early turn {i}."))
            .expect("send failed");
    }

    let segs = segment_count(&conv);
    eprintln!("Segment count after {} / {} turns: {}", 2, 3, segs);
    assert_eq!(
        segs, 0,
        "no segment should appear before reaching summarize_every=3 turns"
    );

    conv.close().ok();
}

/// With `summarize_every=3` feeding 6 turns should produce exactly two
/// separate segment nodes: one covering turns 1–3 and another covering 4–6.
#[test]
#[ignore]
fn test_summarization_second_round_triggers() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=6u32 {
        let resp = conv
            .send_turn(&format!("Round turn {i}."))
            .expect("send failed");
        eprintln!(
            "Turn {i}: {} ({} tokens)",
            resp.text, resp.stats.tokens_generated
        );
    }

    // Poll briefly — both summarizations should already be complete.
    let found = wait_for_any_segment(&conv, Duration::from_secs(60));
    assert!(found, "expected at least one segment after 6 turns");

    let seg_count = segment_count(&conv);
    eprintln!("Total segment count after 6 turns: {}", seg_count);
    assert_eq!(
        seg_count, 2,
        "expected exactly 2 segments for 6 turns with summarize_every=3, got {}",
        seg_count
    );

    let nodes: Vec<_> = conv.tree().nodes().collect();
    let segs: Vec<_> = nodes.iter().filter_map(|n| n.as_segment()).collect();
    // First segment: 1..=3, second: 4..=6.
    assert_eq!(segs[0].inner().segment_id.start_turn.seq, 1);
    assert_eq!(segs[0].inner().segment_id.end_turn.seq, 3);
    assert_eq!(segs[1].inner().segment_id.start_turn.seq, 4);
    assert_eq!(segs[1].inner().segment_id.end_turn.seq, 6);

    conv.close().ok();
}

/// The segment's `ordering_seq()` must equal `end_turn.seq` — confirming
/// chronological ordering of mixed node types.
#[test]
#[ignore]
fn test_summarization_segment_ordering_seq() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=3u32 {
        conv.send_turn(&format!("Order test turn {i}."))
            .expect("send failed");
    }

    let seg = last_segment(&conv).expect("no segment node found");
    let expected_seq = seg.inner().segment_id.end_turn.seq;
    let ordering = seg.node_id().ordering_seq();

    eprintln!(
        "Segment ordering_seq={}, end_turn.seq={}",
        ordering, expected_seq
    );
    assert_eq!(
        ordering, expected_seq,
        "segment ordering_seq() should equal end_turn.seq"
    );
    conv.close().ok();
}

/// Feed a conversation with specific facts, trigger summarization, and verify
/// the summary is at least somewhat coherent: it contains either a word from
/// the conversation topics or has sufficient length (> 20 chars).
///
/// With a 0.5 B model the summary quality is modest, so the bar is low but
/// above a degenerate empty/single-token response.
#[test]
#[ignore]
fn test_summarization_topic_summary_is_coherent() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    let r1 = conv
        .send_turn("My name is Alice and I enjoy hiking.")
        .expect("turn 1 failed");
    eprintln!("Turn 1 response: {}", r1.text);

    let r2 = conv
        .send_turn("I live in a small town near the mountains.")
        .expect("turn 2 failed");
    eprintln!("Turn 2 response: {}", r2.text);

    let r3 = conv
        .send_turn("My favourite hobby is photography.")
        .expect("turn 3 failed");
    eprintln!("Turn 3 response: {}", r3.text);

    let seg = last_segment(&conv).expect("no segment node found after topic conversation");
    let text = seg.inner().summary_text.text().to_string();
    eprintln!("Topic summary ({} chars): {:?}", text.len(), text);

    // Must be non-trivially long (> 20 chars).
    assert!(
        text.len() > 20,
        "summary should be > 20 chars for a coherent distillation, got {} chars: {:?}",
        text.len(),
        text
    );

    // As a soft check: at least one of the topic words should appear. With
    // Qwen2-0.5B this is aspirational but not guaranteed under argmax, so
    // we only log a warning rather than fail.
    let lower = text.to_lowercase();
    let topic_words = ["alice", "hik", "mountain", "photo", "town", "hobby"];
    let hit = topic_words.iter().any(|w| lower.contains(w));
    if !hit {
        eprintln!(
            "NOTE: summary does not contain any of {:?} — model may have paraphrased",
            topic_words
        );
    }

    conv.close().ok();
}

/// Verify that after summarization fires the conversation can still accept
/// further turns without error.  The KV-cache state must remain consistent
/// across the summarization task completion.
#[test]
#[ignore]
fn test_summarization_conversation_continues_after_segment() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(3))
        .expect("new_conversation failed");

    for i in 1..=3u32 {
        conv.send_turn(&format!("Pre-summary turn {i}."))
            .expect("pre-summary send failed");
    }

    // Segment should now exist.
    assert_eq!(segment_count(&conv), 1, "expected 1 segment after 3 turns");

    // Continue with two more turns — must not panic or error.
    let r4 = conv
        .send_turn("Can you recap what we discussed?")
        .expect("post-summary turn 4 failed");
    let r5 = conv
        .send_turn("Thanks, that is helpful.")
        .expect("post-summary turn 5 failed");

    eprintln!("Post-summary turn 4: {}", r4.text);
    eprintln!("Post-summary turn 5: {}", r5.text);

    assert!(!r4.text.is_empty(), "turn 4 should produce output");
    assert!(!r5.text.is_empty(), "turn 5 should produce output");

    // Top-level nodes: 1 segment (containing turns 1-3 as children) + Turn 4 + Turn 5 = 3.
    assert_eq!(
        conv.tree().nodes().count(),
        3,
        "expected 1 segment + 2 unsummarised turns = 3 top-level nodes, got {}",
        conv.tree().nodes().count()
    );

    conv.close().ok();
}

/// Using `summarize_every=1` (every single turn) should produce a segment
/// after the very first `send()`.
#[test]
#[ignore]
fn test_summarization_every_one_trigger() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(1))
        .expect("new_conversation failed");

    let resp = conv.send_turn("Hello world.").expect("send failed");
    eprintln!("Response: {}", resp.text);

    let found = wait_for_any_segment(&conv, Duration::from_secs(30));
    assert!(
        found,
        "with summarize_every=1 a segment should appear after the very first turn"
    );

    let seg = last_segment(&conv).unwrap();
    eprintln!("Segment text: {:?}", seg.inner().summary_text.text());
    assert!(!seg.inner().summary_text.text().is_empty());

    conv.close().ok();
}

/// Verify `summarize_every=0` (disabled) never fires, even after many turns.
#[test]
#[ignore]
fn test_summarization_disabled_when_every_zero() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every(0))
        .expect("new_conversation failed");

    for i in 1..=5u32 {
        conv.send_turn(&format!("No-summary turn {i}."))
            .expect("send failed");
    }

    let segs = segment_count(&conv);
    assert_eq!(
        segs, 0,
        "summarize_every=0 should disable summarization entirely, got {} segments",
        segs
    );

    conv.close().ok();
}
/// Recursive (segment-of-segments) summarization: with `summarize_every=3`
/// and `segment_summarize_every=2`, feeding 6 turns should produce:
///
/// - Level-1 `Seg(1..=3)` after turn 3.
/// - Level-1 `Seg(4..=6)` after turn 6.
/// - Immediately, the 2 level-1 segments trigger a level-2 `Seg(1..=6)` that
///   absorbs both as its children.
///
/// Final top-level structure:
/// ```
/// nodes[0] = Seg(1..=6)                       ← level-2 parent
///   .children[0] = Seg(1..=3)                 ← level-1 child
///     .children[0..2] = Turn1..Turn3
///   .children[1] = Seg(4..=6)                 ← level-1 child
///     .children[0..2] = Turn4..Turn6
/// ```
#[test]
#[ignore]
fn test_recursive_summarization_two_levels() {
    let eng = engine();
    let mut conv = eng
        .new_conversation(&system_prompt(), config_with_every_and_segment_every(3, 2))
        .expect("new_conversation failed");

    // Use insert_turn (prefill-only, no decode) so the 6 setup turns are fast.
    // Summarization tasks still do real inference when they fire.
    for i in 1..=6u32 {
        let asst = format!("Acknowledged turn {i}.");
        conv.insert_turn(&format!("Recursive test turn {i}."), &asst)
            .expect("insert_turn failed");
        eprintln!("Inserted turn {i}");
    }

    let nodes: Vec<_> = conv.tree().nodes().collect();
    eprintln!("Top-level node count after 6 turns: {}", nodes.len());

    // Exactly one top-level node: the level-2 segment.
    assert_eq!(
        nodes.len(),
        1,
        "expected 1 top-level node (level-2 segment), got {}",
        nodes.len()
    );

    let level2 = nodes[0]
        .as_segment()
        .expect("top-level node should be a Segment");

    eprintln!(
        "Level-2: Seg({}..={}) summary={:?}",
        level2.inner().segment_id.start_turn.seq,
        level2.inner().segment_id.end_turn.seq,
        &level2.inner().summary_text.text()[..level2.inner().summary_text.text().len().min(80)]
    );

    // Level-2 segment spans turns 1..=6.
    assert_eq!(level2.inner().segment_id.start_turn.seq, 1);
    assert_eq!(level2.inner().segment_id.end_turn.seq, 6);

    // Level-2 should have exactly 2 Segment children (the level-1 segments).
    let l2_children = &level2.inner().children;
    assert_eq!(
        l2_children.len(),
        2,
        "level-2 segment should have 2 children, got {}",
        l2_children.len()
    );

    for (ci, child) in l2_children.iter().enumerate() {
        let l1 = child
            .as_segment()
            .expect("level-2 child should itself be a Segment");
        let expected_start = (ci as u32) * 3 + 1;
        let expected_end = expected_start + 2;
        eprintln!(
            "  Level-1 child[{}]: Seg({}..={}) with {} turn children",
            ci,
            l1.inner().segment_id.start_turn.seq,
            l1.inner().segment_id.end_turn.seq,
            l1.inner().children.len()
        );
        assert_eq!(l1.inner().segment_id.start_turn.seq, expected_start);
        assert_eq!(l1.inner().segment_id.end_turn.seq, expected_end);
        assert_eq!(
            l1.inner().children.len(),
            3,
            "level-1 child[{}] should have 3 turn children",
            ci
        );
        for (ti, tc) in l1.inner().children.iter().enumerate() {
            let turn = tc
                .as_turn()
                .expect("level-1 child's children should be Turns");
            assert_eq!(turn.inner().turn_id.seq, expected_start + ti as u32);
        }
    }

    assert!(
        !level2.inner().summary_text.text().is_empty(),
        "level-2 summary text should not be empty"
    );

    conv.close().ok();
}
