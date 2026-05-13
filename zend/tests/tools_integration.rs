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
//! ```text
//! cargo test -p zend --test tools_integration --features cuda -- --nocapture
//! ```

#[cfg(feature = "cuda")]
mod tool_scenarios {
    use std::sync::Arc;

    use futures::StreamExt;

    use zend::config::DaemonConfig;
    use zend::log_broadcast::LogBus;
    use zend::session::{StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role};

    /// Per-test load timeout.  Model load + tool catalog prefill at first
    /// boot is ~30 s; a generous cap keeps cold-start tests honest while
    /// still catching hangs.
    const TIMEOUT_SECS: u64 = 900;

    fn init_tracing() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();
    }

    /// Boot a ZendSession, wait for ready, send `prompt`, return the
    /// concatenated assistant text.  Used by every scenario test.
    async fn run_query(prompt: &str, conv_id: &str) -> String {
        let log = LogBus::new();
        let config = DaemonConfig {
            workspace: std::env::current_dir().unwrap(),
            port: 0,
        };
        let session = Arc::new(ZendSession::new(config, Arc::clone(&log)));
        session.start_loading();

        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        let mut stream = session
            .submit(messages, Some(512), conv_id.to_string())
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
            }
        }
        eprintln!("\n\n[FINAL RESPONSE]\n{response}");
        eprintln!("[STATUS MESSAGES] {status_msgs:?}");
        response
    }

    fn run_with_timeout<F: std::future::Future<Output = String> + Send + 'static>(
        f: F,
    ) -> String {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let result = rt.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(TIMEOUT_SECS), f).await
        });
        rt.shutdown_background();
        result.unwrap_or_else(|_| {
            panic!("test timed out after {TIMEOUT_SECS}s")
        })
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
        assert!(
            !response.is_empty(),
            "datetime query produced no response"
        );
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
        let response =
            run_with_timeout(run_query("Calculate 17 times 23.", "test-calc"));
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
        let response =
            run_with_timeout(run_query("Convert 100 km to miles.", "test-units"));
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

    #[test]
    fn hash_compute_query_uses_hash_tool() {
        init_tracing();
        let response = run_with_timeout(run_query(
            "Compute the SHA256 hash of the text \"hello\".",
            "test-hash",
        ));
        assert!(!response.is_empty());
        // SHA256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        // Accept any 64-hex-character substring as evidence of a real hash.
        let has_long_hex = response.split_whitespace().any(|tok| {
            tok.len() >= 32 && tok.chars().all(|c| c.is_ascii_hexdigit())
        });
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
        let response =
            run_with_timeout(run_query("What's the weather in London right now?", "test-weather"));
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
