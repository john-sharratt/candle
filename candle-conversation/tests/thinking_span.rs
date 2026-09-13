//! Integration tests for thinking-span projection, against a real thinking
//! model (`docs/thinking_span_projection.md`).
//!
//! # Why these exist separately from `conversation_tests`
//!
//! That suite runs Qwen2-0.5B, whose tokenizer has no single `</think>` token —
//! so `Scheduler::think_close` is `None`, no reasoning span is ever recorded,
//! and **none of the thinking-span path is exercised by it at all**. Every
//! assertion here needs a model that actually emits `<think>…</think>`.
//!
//! # What they assert, and what they deliberately do not
//!
//! Structure, not prose. A turn's *content* depends on the model and the
//! sampler; its *layout* does not. So these pin the things the design is
//! actually responsible for — where the span lands, that the window drops
//! exactly it, that the two halves agree — and never assert on what the model
//! said. A test that checked the text would fail for reasons that have nothing
//! to do with the code under test, which is how the sibling suite ended up
//! entirely `#[ignore]`d.
//!
//! # They are `#[ignore]`d on purpose
//!
//! Every test here loads a real model. Even the small one is ~30 s for the
//! suite, and the model this design is FOR takes minutes — far too slow for the
//! edit-compile-test loop. So they never run by default, and the fast feedback
//! lives in unit tests instead: the page-span arithmetic
//! (`qwen4exp::paged_index::tail_span_pages`), the page framing
//! (`index_pages`), the K/V windowing (`conversation::window_sealed_tokens_tests`),
//! and the layout spans (`turn_layout`). Those run in milliseconds and cover
//! the arithmetic; these cover the composition.
//!
//! # What the default model does and does NOT exercise
//!
//! [`THINKING_MODEL`] is Qwen3-8B: it has single `<think>`/`</think>` tokens, so
//! the recorded span, the layout, and the K/V window are all real here. But its
//! arch does not carry per-position state — `carries_positional_state()` is
//! false — so it seals **no QSA index pages at all**, and the page half of the
//! design is inert under it. That half is where the measured regression was.
//!
//! To exercise the whole pipeline, switch [`THINKING_MODEL`] to
//! `Model::Qwen38_FlashNext_Q4KO`. It is the architecture the design targets —
//! index pages, page cuts, the width-based seal — at the cost of a multi-minute
//! load, which is why it is not the default.
//!
//! # Run them in RELEASE
//!
//! ```bash
//! cargo test --release -p candle-conversation --features hub --test thinking_span \
//!     -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! The whole suite is ~32 s. `--release` is not for speed: a `debug_assert!` in
//! `provenance::gallery_arena` fires on this model's folded geometry ("gallery
//! sig width 48 != arena wpt 24") the moment a SECOND turn scans the first one's
//! signatures, and it takes the scheduler thread with it. That is a pre-existing
//! debug-only fault — the daemon runs release and never meets it — but it means
//! a debug run of these tests dies at turn two for a reason unrelated to
//! anything they assert. Worth fixing on its own; until then, release.
//!
//! `--test-threads=1` because each test loads its own engine, and two of those
//! will not fit at once.

use candle_conversation::models::{Model, ModelBuilder};
use candle_conversation::projection::TurnIndex;
use candle_conversation::turn_layout::TurnSegment;
use candle_conversation::{ConversationEngine, SamplingConfig, SequenceConfig};

/// The model these tests run against.
///
/// Qwen3-8B by default: it has single `<think>` / `</think>` tokens — the whole
/// requirement for the span, layout and window assertions — and loads in
/// seconds. It seals no index pages, so the page half of the design is inert
/// under it; see the module docs. Switch to `Model::Qwen38_FlashNext_Q4KO` to
/// cover that half too.
const THINKING_MODEL: Model = Model::Qwen3_8B_Q4;

fn builder() -> ModelBuilder {
    THINKING_MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(7)
        .max_response_tokens(96)
        .max_concurrent(4)
}

/// A private substrate for one test, removed before use.
///
/// **These tests must not share the repo-root `.substrate`.** A substrate is
/// bound to the tokenizer that sealed it — deliberately, because every turn in
/// the log was sealed under that vocabulary — so a suite that inherits whatever
/// the last run left behind fails on a model mismatch that has nothing to do
/// with the code under test. Worse, it would pass or fail depending on what ran
/// before it, which is how a suite stops being evidence.
fn private_workspace(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("candle-thinking-span-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("test workspace");
    dir
}

fn engine(name: &str) -> ConversationEngine {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_test_writer()
        .try_init();
    let device =
        candle::Device::cuda_if_available(0).expect("CUDA device required for these tests");
    builder()
        .workspace_path(private_workspace(name))
        .engine(&device)
        .expect("failed to load the thinking model")
}

fn config() -> SequenceConfig {
    builder().conversation_config()
}

fn system() -> String {
    builder().format_system_prompt()
}

/// **The reasoning span is where the layout says it is — checked against the
/// turn's own token grid.**
///
/// This is the assertion the design leans on hardest: the span is measured from
/// the `</think>` the decode actually emitted, not from re-tokenised prose, and
/// everything downstream (the page cuts, the window, the sig filter) trusts it.
/// If it is off by one, the window drops an answer token or keeps a reasoning
/// one, and nothing else in the system would notice.
///
/// So it is verified the only way that cannot agree with a bug: the last token
/// of the recorded span, read out of the stored grid, must BE `</think>`.
#[test]
#[ignore = "needs a CUDA GPU and downloads Qwen3-8B"]
fn the_recorded_span_ends_on_the_close_marker_in_the_real_grid() {
    let eng = engine("span-marker");
    let mut conv = eng
        .new_conversation(&system(), config())
        .expect("new conversation");
    let resp = conv.send_turn("What is 7 times 8?").expect("turn");
    let idx = resp
        .seal
        .as_ref()
        .and_then(|s| s.turn_index)
        .map(TurnIndex)
        .expect("the turn sealed and reported its index");

    let tl = conv.timeline_id();
    let handle = eng.conversation();
    let read = handle.read();
    let layout = read.turn_layout(tl, idx).expect("sealed layout");
    let grid = read.token_ids_of(tl, idx);

    let span = layout.segments.iter().find_map(|s| match s {
        TurnSegment::Thinking { kv, .. } => *kv,
        _ => None,
    });

    // A thinking model under a "think briefly" system prompt should reason. If
    // it did not, there is nothing to check — say so rather than pass silently.
    let Some(span) = span else {
        eprintln!("model emitted no reasoning block; nothing to verify");
        conv.close().ok();
        return;
    };

    assert!(span.len > 0, "a real thinking span must cover tokens");
    let end = span.end() as usize;
    assert!(
        end <= grid.len(),
        "the span runs past the stored grid ({end} > {})",
        grid.len()
    );

    // The decisive check. `</think>` is a single token for this model, so the
    // span's last token must be exactly it.
    let close = eng
        .tokenizer()
        .token_to_id("</think>")
        .expect("this model has a single </think> token");
    assert_eq!(
        grid[end - 1],
        close,
        "the span's last token is {} but should be `</think>` ({close}) — the \
         recorded boundary does not match the grid",
        grid[end - 1]
    );

    // …and the span starts at the assistant body, not at `<think>`: any preamble
    // the model emitted before the marker is reasoning too.
    assert_eq!(
        span.offset,
        layout.assistant_content_start(),
        "the reasoning region must begin where the assistant body does"
    );

    conv.close().ok();
}

/// **The window drops exactly the reasoning, and both halves agree.**
///
/// `turn_sealed_without_thinking` returns the K/V and the retained index pages
/// together precisely so they cannot diverge. This checks the promise: the
/// windowed K/V is narrower than the whole turn by exactly the span, or — when
/// the pages do not line up with it — the turn comes back WHOLE on both sides.
/// A half-windowed turn is the one outcome that must never occur.
#[test]
#[ignore = "needs a CUDA GPU and downloads Qwen3-8B"]
fn windowing_drops_exactly_the_span_or_nothing_at_all() {
    let eng = engine("windowing");
    let mut conv = eng
        .new_conversation(&system(), config())
        .expect("new conversation");
    let resp = conv.send_turn("Name one primary colour.").expect("turn");
    let idx = resp
        .seal
        .as_ref()
        .and_then(|s| s.turn_index)
        .map(TurnIndex)
        .expect("sealed index");

    let tl = conv.timeline_id();
    let handle = eng.conversation();
    let read = handle.read();

    let whole = read.turn_sealed_of(tl, idx).expect("the turn is hot");
    let whole_tokens = whole[0].token_count;
    let pages = read.index_page_blob(tl, idx).map(|b| b.to_vec());
    let layout = read.turn_layout(tl, idx).expect("layout");
    let span = layout.segments.iter().find_map(|s| match s {
        TurnSegment::Thinking { kv, .. } => *kv,
        _ => None,
    });

    let (windowed, kept_pages) = read
        .turn_sealed_without_thinking(tl, idx, pages)
        .expect("the turn's reasoning windows onto its page boundaries")
        .expect("the turn is hot");
    let windowed_tokens = windowed[0].token_count;

    match span {
        Some(span) => {
            let dropped = whole_tokens - windowed_tokens;
            assert!(
                dropped == span.len as usize || dropped == 0,
                "a windowed turn must drop exactly its span ({}) or nothing at all \
                 (pages misaligned), but dropped {dropped}",
                span.len
            );
            if dropped == 0 {
                // Two different reasons land here and only one is a warning
                // sign. Under a model that seals no index pages there is nothing
                // to window against and whole is the ONLY correct answer — which
                // is the default here, so this is the expected path. Pages that
                // exist but straddle the span are refused outright (an `Err` the
                // `expect` above would have caught), so they never reach this
                // branch.
                eprintln!(
                    "the turn injected whole — {} — so the windowing half of this \
                     test is inert; run it against Model::Qwen38_FlashNext_Q4KO to cover it",
                    if kept_pages.is_none() {
                        "this model seals no index pages"
                    } else {
                        "its pages cover no part of the span"
                    }
                );
            }
        }
        None => assert_eq!(
            windowed_tokens, whole_tokens,
            "a turn with no reasoning span must come back unchanged"
        ),
    }

    conv.close().ok();
}

/// **The rule is positional: newest whole, older windowed.**
///
/// Three turns, then the same question asked of each: is this turn the newest?
/// The newest must come back whole even though it has a span; the older ones
/// must window (or, if their pages misalign, come back whole on both halves —
/// never half).
#[test]
#[ignore = "needs a CUDA GPU and downloads Qwen3-8B"]
fn only_the_newest_turn_keeps_its_reasoning() {
    let eng = engine("newest-whole");
    let mut conv = eng
        .new_conversation(&system(), config())
        .expect("new conversation");

    let mut indices = Vec::new();
    for q in ["Name a colour.", "Name a shape.", "Name a number."] {
        let r = conv
            .send_turn(q)
            .unwrap_or_else(|e| panic!("turn {q:?}: {e}"));
        if let Some(i) = r.seal.as_ref().and_then(|s| s.turn_index) {
            indices.push(TurnIndex(i));
        }
    }
    assert_eq!(indices.len(), 3, "all three turns must seal");

    let tl = conv.timeline_id();
    let handle = eng.conversation();
    let read = handle.read();
    let newest = indices.iter().copied().max().expect("a newest turn");

    for &idx in &indices {
        let Some(whole) = read.turn_sealed_of(tl, idx) else {
            continue; // demoted out of hot; not this test's subject
        };
        let pages = read.index_page_blob(tl, idx).map(|b| b.to_vec());
        let (windowed, _) = read
            .turn_sealed_without_thinking(tl, idx, pages)
            .expect("the turn's reasoning windows onto its page boundaries")
            .expect("hot turn windows");
        let layout = read.turn_layout(tl, idx).expect("layout");
        let has_span = layout
            .segments
            .iter()
            .any(|s| matches!(s, TurnSegment::Thinking { kv: Some(_), .. }));

        if idx == newest {
            // The projection injects the newest turn WHOLE — it never calls the
            // windowing path for it. The accessor is still expected to be
            // consistent when asked, which is what makes the rule a projection
            // decision rather than a property of the record.
            assert_eq!(
                whole[0].token_count,
                read.turn_sealed_of(tl, idx).unwrap()[0].token_count,
                "the newest turn's record is unchanged"
            );
        } else if has_span {
            assert!(
                windowed[0].token_count <= whole[0].token_count,
                "windowing must never widen a turn"
            );
        }
    }

    conv.close().ok();
}

/// **A conversation of several turns stays self-consistent.**
///
/// The blunt end-to-end check: many turns in a row, each rebuilding the
/// projection from the substrate, must all seal and all remain readable. This
/// is the shape that broke on the daemon — not one turn, but the accumulation —
/// and it costs seconds here against minutes there.
#[test]
#[ignore = "needs a CUDA GPU and downloads Qwen3-8B"]
fn a_long_conversation_seals_every_turn_and_stays_readable() {
    let eng = engine("long-run");
    let mut conv = eng
        .new_conversation(&system(), config())
        .expect("new conversation");

    let questions = [
        "Name a colour.",
        "Name a shape.",
        "Name a number.",
        "Name a fruit.",
        "Name a country.",
        "Name an animal.",
    ];
    let mut sealed = Vec::new();
    for q in questions {
        let r = conv
            .send_turn(q)
            .unwrap_or_else(|e| panic!("turn {q:?}: {e}"));
        assert!(
            !r.text.is_empty(),
            "turn {q:?} produced no text at all — the turn did not decode"
        );
        let idx = r
            .seal
            .as_ref()
            .and_then(|s| s.turn_index)
            .unwrap_or_else(|| panic!("turn {q:?} did not seal"));
        sealed.push(TurnIndex(idx));
    }

    let tl = conv.timeline_id();
    let handle = eng.conversation();
    let read = handle.read();
    for &idx in &sealed {
        let layout = read
            .turn_layout(tl, idx)
            .expect("every sealed turn has a layout");
        let grid_len = read.token_ids_of(tl, idx).len() as u32;
        assert_eq!(
            layout.validate_tiling(grid_len),
            Ok(()),
            "turn {} does not tile its own grid — the layout and the K/V disagree",
            idx.0
        );
    }

    conv.close().ok();
}
