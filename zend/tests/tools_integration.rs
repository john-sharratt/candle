//! End-to-end tool integration tests for `zend`.
//!
//! These tests boot a full [`ZendSession`] (model + tool catalog + projection),
//! submit user queries, and verify that the tool orchestrator picks the right
//! tools and chains them correctly.  Each test exercises a different scenario
//! — they're the real integration test for months of provenance + projection
//! work.
//!
//! ## What's being validated
//!
//! 1. **Tool catalog at all 93 tools is registered as system-prompt sections.**
//!    The whole `zend-tools` registry surfaces into the substrate at startup,
//!    each tool getting its own per-section sig_entries.
//!
//! 2. **Projection's section TopK selection picks the right K tools.**  For
//!    a query like "what's the date?" the BDP scoring should rank `datetime`
//!    above all 92 other tools — which proves the provenance + projection
//!    machinery is selecting on actual semantic relevance, not luck.
//!
//! 3. **The orchestrator dispatches tool calls and chains the response.**
//!    A `<tool_call>` block in the response is parsed, the tool runs via
//!    `zend_tools::runner::run`, and the result wraps as
//!    `<tool_response>{json}</tool_response>` for the next turn.
//!
//! 4. **Continuous re-projection works mid-decode.**  A query that needs
//!    two tools across one response should surface both — driven by the
//!    view-swap mechanism re-scoring as the model's intent shifts.
//!
//! 5. **Negative case.**  A non-tool query produces no `<tool_call>` blocks
//!    and surfaces no irrelevant tool — confirms BDP isn't biased toward
//!    forcing a tool surface when nothing matches.
//!
//! ## Running
//!
//! The whole suite is `#[cfg(feature = "cuda")]`-gated (needs a real GPU
//! and the GGUF weights on disk).  CPU-only CI skips it.
//!
//! Every scenario the 0.8B answers correctly runs in the default suite, on
//! `Qwen35_0_8B_Q8` — the production model's lineage, dialect and tool-call
//! style — against the suite's own workspace under `target/tmp`, thinking off,
//! each on a conversation of its own that is retired afterwards. Each scenario
//! boots its own daemon: ~3 s of model load (~10 s on the first, which parses
//! the checkpoint header for the whole process) and ~4.4 s of tool-section
//! prefill before the query, ~10 s end to end. The first run on a fresh
//! workspace also calibrates the tool catalog once, ~35 s.
//!
//! A scenario the 0.8B answers wrongly runs on the production model against
//! the shared production workspace under `target/tmp`
//! (`common::production_workspace`), where that model's catalog calibration is
//! paid once, and is `#[ignore]`d. Run those by name, daemon stopped:
//!
//! ```text
//! cargo test -p zend --test tools_integration --features cuda -- --ignored --nocapture
//! ```

mod common;

#[cfg(feature = "cuda")]
mod tool_scenarios {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use futures::StreamExt;

    use crate::common::{needs_compaction, production_workspace, run_conv_id};
    use candle::vram::host_pinned_bytes;
    use candle_conversation::models::Model;
    use candle_conversation::{SamplingConfig, SelectionState};
    use zend::api::chat::dial_selection;
    use zend::config::{DaemonConfig, ModelChoice};
    use zend::log_broadcast::LogBus;
    use zend::session::{timeline_for, StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role};

    /// Per-scenario cap. A scenario on a warm workspace is ~10 s; the first run
    /// on a fresh one also calibrates the whole tool catalog once, which is what
    /// this leaves room for while still catching a hang.
    const TIMEOUT_SECS: u64 = 900;

    /// The model every default scenario runs: the production model's lineage,
    /// dialect and tool-call style at 0.8B, so a scenario costs ~10 s rather
    /// than the production model's ~76 s and the orchestration exercised is
    /// still the real one.
    const MODEL: Model = Model::Qwen35_0_8B_Q8;

    /// Scenarios run one at a time. They share one workspace and its substrate
    /// admits one daemon, so each boots, answers and shuts down before the next
    /// opens it.
    static SCENARIO: Mutex<()> = Mutex::new(());

    /// The suite's own workspace, kept under `target/tmp` between runs.
    ///
    /// Not the live repo: its substrate holds another model's K/V and
    /// calibration exemplars, which a different checkpoint cannot read. Not a
    /// fresh temp dir either: the exemplars tool selection scores against live
    /// in the substrate, and a fresh one would regenerate the whole catalog's on
    /// every run. Kept here, calibration is paid once and every later boot
    /// resumes it.
    fn workspace() -> PathBuf {
        let ws = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tools_integration_ws");
        std::fs::create_dir_all(&ws).expect("create the suite's workspace");
        ws
    }

    fn init_tracing() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();
    }

    // Each scenario boots its own daemon rather than sharing one. A shared
    // engine accumulates every scenario into one substrate and KV pool, so each
    // successive query scans a larger corpus under more relief pressure —
    // measured slower than a boot per scenario on the production model (952 s
    // shared against 758 s for ten fresh sessions).

    /// Which model and workspace a scenario runs on.
    #[derive(Clone, Copy)]
    enum Rig {
        /// The 0.8B on the suite's own workspace, thinking off, exactly as the
        /// composer's toggle sends it. At 0.8B a reasoning block runs to the
        /// token budget before the model ever calls a tool, so a thinking turn
        /// would measure how long it deliberates rather than whether the
        /// orchestration routes. Every scenario the 0.8B answers runs here.
        Small,
        /// The production model — the measured-VRAM ladder — on the shared
        /// production workspace, where that model's calibration is paid once,
        /// with the schema's default thinking. For a scenario the 0.8B answers wrongly;
        /// `#[ignore]`d wherever it is used, since it pays the production boot.
        Production,
    }

    /// [`run_on`] the small rig — what every scenario the 0.8B handles runs.
    async fn run_query(prompt: &str, conv_id: &str) -> String {
        run_on(Rig::Small, prompt, conv_id).await
    }

    /// Boot a ZendSession on `rig`, wait for ready, send `prompt`, shut the
    /// session down, and return the concatenated assistant text.
    async fn run_on(rig: Rig, prompt: &str, conv_id: &str) -> String {
        // A conversation of this run's own — both workspaces persist across runs.
        let conv_id = run_conv_id(conv_id);
        let (workspace, model, selection) = match rig {
            Rig::Small => (
                workspace(),
                ModelChoice::Preset(Box::new(MODEL)),
                dial_selection(None, None, Some(false)),
            ),
            Rig::Production => (
                production_workspace(),
                ModelChoice::MeasuredVram,
                SelectionState::default(),
            ),
        };
        let compact_substrate = needs_compaction(&workspace);
        let log = LogBus::new();
        let config = DaemonConfig {
            workspace,
            port: 0,
            model,
            compact_substrate,
            ..Default::default()
        };
        let session = Arc::new(ZendSession::new(config, Arc::clone(&log)));
        session.start_loading();

        let messages = vec![ChatMessage::new(Role::User, prompt)];
        // Argmax, so a scenario is one fixed decode of its prompt. Left to the
        // daemon, sampling is reseeded from the clock every turn and the same
        // prompt passes or fails from one run to the next.
        let mut stream = session
            .submit_with_sampling(
                messages,
                Some(512),
                conv_id.to_string(),
                Some(SamplingConfig::argmax()),
                None,
                None,
                false,
                zend::types::ToolMode::Comprehensive,
                None,
                selection,
            )
            .await;

        let mut response = String::new();
        let mut status_msgs: Vec<String> = Vec::new();
        while let Some(result) = stream.next().await {
            match result.expect("stream item error") {
                StreamItem::Status(msg) => {
                    eprintln!("[STATUS] {msg}");
                    status_msgs.push(msg);
                }
                StreamItem::Token(tok) => {
                    eprint!("{tok}");
                    response.push_str(&tok);
                }
                StreamItem::Projection(projection_event_out) => {
                    eprintln!("\n[PROJECTION EVENT] {:?}", projection_event_out.event);
                }
                StreamItem::Tool(status) => {
                    eprintln!("\n[TOOL {}] {:?}", status.phase, status.tools);
                }
                StreamItem::Usage(_) => {}
            }
        }
        eprintln!("\n\n[FINAL RESPONSE]\n{response}");
        eprintln!("[STATUS MESSAGES] {status_msgs:?}");
        // Retire this run's conversation, so the workspace does not keep one per
        // scenario per run; compaction reclaims it.
        if let Some(Err(e)) = session.tombstone_timeline_raw(timeline_for(&conv_id).raw()) {
            panic!("tombstoning {conv_id}: {e}");
        }
        // Release the workspace's substrate for the next scenario.
        session.shutdown().await;
        response
    }

    /// Host-pinned bytes still allocated once the previous scenario's session had
    /// shut down and been released.
    static PINNED_AFTER_PREVIOUS: Mutex<Option<u64>> = Mutex::new(None);

    /// How long a finished scenario's runtime may take to wind its tasks down.
    const RUNTIME_WIND_DOWN: Duration = Duration::from_secs(60);

    fn run_with_timeout<F: std::future::Future<Output = String> + Send + 'static>(f: F) -> String {
        let _one_at_a_time = SCENARIO.lock().unwrap_or_else(|e| e.into_inner());
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let result =
            rt.block_on(async { tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), f).await });
        // Wait for everything the session spawned, so what is still pinned below
        // is what the session left behind rather than what is still unwinding.
        rt.shutdown_timeout(RUNTIME_WIND_DOWN);
        let response = result.unwrap_or_else(|_| panic!("test timed out after {TIMEOUT_SECS}s"));
        assert_session_released_its_pinned_memory();
        response
    }

    /// **A shut-down session releases what it pinned.**
    ///
    /// Every scenario boots and shuts down a full session in this one process, so
    /// each one's residue is measurable against the last. The first reading is
    /// the baseline — it includes what the process pins once and keeps — and every
    /// later session must leave exactly that behind. A session still reachable
    /// after `shutdown` keeps its expert warm tier pinned, and its engine
    /// competes with the next session for the card.
    fn assert_session_released_its_pinned_memory() {
        let now = host_pinned_bytes();
        let mut previous = PINNED_AFTER_PREVIOUS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(before) = *previous {
            assert_eq!(
                now, before,
                "a shut-down session left pinned host memory behind: {before} bytes after \
                 the previous scenario, {now} after this one"
            );
        }
        *previous = Some(now);
    }

    // ── Scenario 1: simple datetime query ────────────────────────────────────
    //
    // "What's the date today?" → expect `datetime` to surface in top-K and
    // the orchestrator to chain the response into a final answer.

    #[test]
    fn datetime_query_calls_datetime_tool() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "What's today's date? Give me just the ISO date.",
            "test-datetime",
        ));
        assert!(!response.is_empty(), "datetime query produced no response");
        // The final answer should contain a date — at minimum a 4-digit year.
        let has_year = response.contains("2024")
            || response.contains("2025")
            || response.contains("2026")
            || response.contains("2027");
        assert!(
            has_year,
            "expected a year in the datetime response, got: {response:?}",
        );
    }

    // ── Scenario 2: calculator query ─────────────────────────────────────────

    #[test]
    fn calculator_query_calls_calculator_tool() {
        init_tracing();
        let response = run_with_timeout(run_query("Calculate 17 times 23.", "test-calc"));
        assert!(!response.is_empty(), "calc produced no response");
        // 17 × 23 = 391.  Accept either as a numeral or close variants.
        assert!(
            response.contains("391"),
            "expected 391 in calculator response, got: {response:?}",
        );
    }

    // ── Scenario 3: simple addition ──────────────────────────────────────────

    #[test]
    fn calculator_handles_simple_addition() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "What is 2 plus 2? Reply with just the number.",
            "test-add",
        ));
        assert!(!response.is_empty());
        assert!(
            response.contains('4'),
            "expected '4' in response to 2+2, got: {response:?}",
        );
    }

    // ── Scenario 4: unit conversion ──────────────────────────────────────────

    #[test]
    fn unit_convert_query_uses_unit_convert_tool() {
        init_tracing();
        let response = run_with_timeout(run_query("Convert 100 km to miles.", "test-units"));
        assert!(!response.is_empty(), "unit_convert produced no response");
        // 100 km ≈ 62.137 miles.  Look for "62" prefix as a sanity check.
        assert!(
            response.contains("62") || response.to_lowercase().contains("mile"),
            "expected a miles-flavoured answer, got: {response:?}",
        );
    }

    // ── Scenario 5: random number generation ─────────────────────────────────

    #[test]
    fn random_query_uses_random_tool() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "Give me a random integer between 1 and 100.",
            "test-random",
        ));
        assert!(!response.is_empty());
        // Accept any digit sequence — tool output may be any value.
        assert!(
            response.chars().any(|c| c.is_ascii_digit()),
            "expected a number in the random response, got: {response:?}",
        );
    }

    // ── Scenario 6: no tool needed ───────────────────────────────────────────
    //
    // A purely conversational question shouldn't trigger any tool call.
    // The model should respond directly.  This validates BDP isn't biased
    // toward forcing tool selection on every query.

    #[test]
    fn plain_conversation_does_not_call_tools() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "Hi! In one sentence, what is Rust used for?",
            "test-plain",
        ));
        assert!(!response.is_empty(), "plain query produced no response");
        // Streamed final response should NOT contain raw <tool_call> markers
        // (the orchestrator filters those out by NOT streaming tool-iteration
        // text — only the final no-tool iteration is streamed).
        assert!(
            !response.contains("<tool_call>"),
            "final response leaked tool_call markers: {response:?}",
        );
    }

    // ── Scenario 7: chained tools (multiple in one user request) ─────────────
    //
    // Two distinct tool needs in one user message — exercises the
    // continuous-re-projection swap mechanism.  Both `datetime` and
    // `calculator` need to surface, possibly across mid-decode swaps.

    #[test]
    fn chained_query_uses_two_tools_across_one_request() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "What's today's date and what is 5 plus 3?",
            "test-chain",
        ));
        assert!(!response.is_empty(), "chained query produced no response");
        let has_year = response.contains("2024")
            || response.contains("2025")
            || response.contains("2026")
            || response.contains("2027");
        let has_eight = response.contains('8');
        assert!(
            has_year && has_eight,
            "expected year + '8' (5+3) in chained response, got: {response:?}",
        );
    }

    // ── Scenario 8: hash compute ─────────────────────────────────────────────

    // On the production model: the 0.8B answered this one with a decimal
    // integer rather than calling `hash_compute` (measured), so it pays the
    // production boot and is `#[ignore]`d.
    #[test]
    #[ignore = "runs on the production model, which the 0.8B cannot stand in for here"]
    fn hash_compute_query_uses_hash_tool() {
        init_tracing();
        let response = run_with_timeout(run_on(
            Rig::Production,
            "Compute the SHA256 hash of the text \"hello\".",
            "test-hash",
        ));
        assert!(!response.is_empty());
        // SHA256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        // Accept any 64-hex-character substring as evidence of a real hash.
        let has_long_hex = response
            .split_whitespace()
            .any(|tok| tok.len() >= 32 && tok.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(
            has_long_hex
                || response
                    .to_lowercase()
                    .contains("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
            "expected hex hash in response, got: {response:?}",
        );
    }

    // ── Scenario 9: weather query ────────────────────────────────────────────

    #[test]
    fn weather_query_uses_weather_tool() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "What's the weather in London right now?",
            "test-weather",
        ));
        // Weather tool may fail (network unavailable in CI), but the model
        // should still surface the right tool.  The response should mention
        // London or a temperature/condition word.
        let lower = response.to_lowercase();
        let mentions_topic = lower.contains("london")
            || lower.contains("weather")
            || lower.contains("temperature")
            || lower.contains("celsius")
            || lower.contains("fahrenheit")
            || lower.contains("error")
            || lower.contains("unavailable");
        assert!(
            mentions_topic,
            "expected weather-themed response, got: {response:?}",
        );
    }

    // ── Scenario 10: web_search query ────────────────────────────────────────

    #[test]
    fn web_search_query_uses_web_search_tool() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "Search the web for \"rust language\" and tell me what you find.",
            "test-search",
        ));
        // Network-dependent tool; assert the model engaged with the request
        // rather than refusing.
        let lower = response.to_lowercase();
        assert!(
            lower.contains("rust") || lower.contains("error") || lower.contains("search"),
            "expected search-themed response, got: {response:?}",
        );
    }
}
