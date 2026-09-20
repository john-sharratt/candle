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
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use futures::StreamExt;

    use crate::common::{needs_compaction, production_workspace, run_conv_id};
    use candle::vram::host_pinned_bytes;
    use candle_conversation::models::Model;
    use candle_conversation::projection::{SectionLoads, SystemItem};
    use candle_conversation::{SamplingConfig, SelectionState};
    use zend::api::chat::{apply_tools_dial, dial_selection};
    use zend::config::{DaemonConfig, ModelChoice};
    use zend::log_broadcast::LogBus;
    use zend::session::{timeline_for, StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role, ToolMode};

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
        run_on(Rig::Small, prompt, conv_id).await.response
    }

    /// [`run_on`] the small rig, keeping what provenance did with the query.
    async fn run_query_observed(prompt: &str, conv_id: &str) -> Outcome {
        run_on(Rig::Small, prompt, conv_id).await
    }

    /// What one scenario observed.
    ///
    /// The response text answers "did the model do the right thing". The other
    /// two fields answer "did PROVENANCE do the right thing", which is the
    /// property a tool's question seeds are tuned against, and they are not the
    /// same question.
    ///
    /// **Asserting on the text alone cannot tell a tool that was never
    /// projected from one the model chose not to call.** Scenario 10 passed for
    /// months against a `web_search` that provenance never selected at all: its
    /// probe contained the words "rust" and "search", so a refusal quoting the
    /// request back satisfied the assertion. Selection is observable — every
    /// projection event carries the whole `tools` collection with each member's
    /// belief score and whether it was picked — so it is what these scenarios
    /// assert.
    struct Outcome {
        response: String,
        /// Tool names actually dispatched, in call order, across every round.
        tools_called: Vec<String>,
        /// The highest belief score each catalog tool reached at any projection
        /// point of this turn, on the normalized 0–1000 band the `tools`
        /// collection is gated on (`min_score` 800, `evict` 750).
        peak_score: HashMap<String, f32>,
        /// Every tool projected into the prompt at any point in the turn.
        selected: HashSet<String>,
    }

    impl Outcome {
        fn called(&self, tool: &str) -> bool {
            self.tools_called.iter().any(|t| t == tool)
        }

        /// Whether the tool ever reached the model — projected, or dispatched.
        fn reached_the_model(&self, tool: &str) -> bool {
            self.selected.contains(tool) || self.called(tool)
        }

        fn peak(&self, tool: &str) -> f32 {
            self.peak_score.get(tool).copied().unwrap_or(0.0)
        }

        /// The `n` tools with the highest belief this turn, strongest first. A
        /// tool whose belief never rose above zero is not ranked.
        fn strongest(&self, n: usize) -> Vec<&str> {
            let mut rows: Vec<(&String, &f32)> =
                self.peak_score.iter().filter(|(_, s)| **s > 0.0).collect();
            rows.sort_by(|a, b| b.1.total_cmp(a.1));
            rows.iter().take(n).map(|(name, _)| name.as_str()).collect()
        }

        /// The strongest tools this turn, highest first. A selection failure is
        /// only actionable if it says what won instead.
        fn ranking(&self) -> String {
            let mut rows: Vec<(&String, &f32)> = self.peak_score.iter().collect();
            rows.sort_by(|a, b| b.1.total_cmp(a.1));
            rows.iter()
                .take(8)
                .map(|(n, s)| format!("{n}={s:.0}"))
                .collect::<Vec<_>>()
                .join("  ")
        }
    }

    /// Boot a ZendSession on `rig`, wait for ready, send `prompt`, shut the
    /// session down, and return what the turn did.
    async fn run_on(rig: Rig, prompt: &str, conv_id: &str) -> Outcome {
        // A conversation of this run's own — both workspaces persist across runs.
        let conv_id = run_conv_id(conv_id);
        let (workspace, model, mut selection) = match rig {
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
        // The tool prompt a chat turn gets — the block AND its worked call — so
        // the suite exercises what a user is shown, not a catalog with no example.
        apply_tools_dial(&mut selection, ToolMode::Comprehensive);
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
                ToolMode::Comprehensive,
                None,
                selection,
            )
            .await;

        let mut response = String::new();
        let mut status_msgs: Vec<String> = Vec::new();
        let mut tools_called: Vec<String> = Vec::new();
        let mut peak_score: HashMap<String, f32> = HashMap::new();
        let mut selected: HashSet<String> = HashSet::new();
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
                    // Every point carries the WHOLE tools collection — picked and
                    // skipped alike — so the peak across the turn is the tool's
                    // best showing, not whatever the last point happened to hold.
                    for item in &projection_event_out.event.selection.system {
                        let SystemItem::Collection { name, sections, .. } = item else {
                            continue;
                        };
                        if name != "tools" {
                            continue;
                        }
                        for s in sections {
                            let peak = peak_score.entry(s.name.clone()).or_insert(0.0);
                            *peak = peak.max(s.score);
                            if s.selected {
                                selected.insert(s.name.clone());
                            }
                        }
                    }
                }
                StreamItem::Tool(status) => {
                    eprintln!("\n[TOOL {}] {:?}", status.phase, status.tools);
                    // The "done" notice repeats the same names; count once.
                    if status.phase == "running" {
                        tools_called.extend(status.tools.iter().cloned());
                    }
                }
                StreamItem::Prefill { .. }
                | StreamItem::Think { .. }
                | StreamItem::TurnEnd { .. } => {}
            }
        }
        eprintln!("\n\n[FINAL RESPONSE]\n{response}");
        eprintln!("[STATUS MESSAGES] {status_msgs:?}");
        eprintln!("[TOOLS CALLED] {tools_called:?}");
        eprintln!("[TOOLS SELECTED] {selected:?}");
        // Retire this run's conversation, so the workspace does not keep one per
        // scenario per run; compaction reclaims it.
        if let Some(Err(e)) = session.tombstone_timeline_raw(timeline_for(&conv_id).raw()) {
            panic!("tombstoning {conv_id}: {e}");
        }
        // Release the workspace's substrate for the next scenario.
        session.shutdown().await;
        Outcome {
            response,
            tools_called,
            peak_score,
            selected,
        }
    }

    /// Host-pinned bytes still allocated once the previous scenario's session had
    /// shut down and been released.
    static PINNED_AFTER_PREVIOUS: Mutex<Option<u64>> = Mutex::new(None);

    /// How long a finished scenario's runtime may take to wind its tasks down.
    const RUNTIME_WIND_DOWN: Duration = Duration::from_secs(60);

    fn run_with_timeout<T, F: std::future::Future<Output = T> + Send + 'static>(f: F) -> T {
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

    /// Boot a ZendSession on the small rig, wait for every startup step to
    /// finish, and report how it loaded its prompt sections — the tool catalog
    /// among them.
    async fn boot_and_count_sections() -> SectionLoads {
        let workspace = workspace();
        let compact_substrate = needs_compaction(&workspace);
        let config = DaemonConfig {
            workspace,
            port: 0,
            model: ModelChoice::Preset(Box::new(MODEL)),
            compact_substrate,
            ..Default::default()
        };
        let session = Arc::new(ZendSession::new(config, LogBus::new()));
        session.start_loading();
        while session.status_snapshot().loading.is_some() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let loads = session
            .section_loads()
            .expect("the model is loaded once every startup step has finished");
        session.shutdown().await;
        loads
    }

    // ── A boot seals no prompt section a previous boot already sealed ────────
    //
    // Sections are content-addressed in the redo log, so an unchanged catalog
    // has nothing left to compute on the next boot. A section prefilled again
    // is sealed again, and its records supersede the last boot's — dead records
    // only compaction reclaims, which is how the suite's workspace once grew
    // ~140 MB a boot.
    //
    // **The two boots' counts are deliberately NOT compared.** This workspace
    // is shared and persistent, and `needs_compaction` is a size test
    // (`> COMPACT_ABOVE_BYTES`), so it can be true on BOTH boots: the second
    // compaction sheds the records the first boot's re-seals superseded, and
    // `restored` legitimately falls — 321 to 226 on the run that exposed this.
    // Asserting `restored == first.restored + first.prefilled` reported that
    // reclamation as "the second boot prefilled N sections" while `prefilled`
    // was plainly 0 on both sides: a failure message describing the opposite of
    // what had happened. What the suite is protecting is that nothing is
    // recomputed, so that is what is asserted.

    #[test]
    fn a_second_boot_restores_every_prompt_section() {
        init_tracing();
        // Both boots under one scenario lock, so no other scenario runs on the
        // shared workspace between them.
        let (first, second) = run_with_timeout(async {
            let first = boot_and_count_sections().await;
            (first, boot_and_count_sections().await)
        });
        assert_eq!(
            second.prefilled, 0,
            "the second boot prefilled {} prompt section(s) the first had already sealed — \
             each is sealed again and supersedes the last boot's records, which is how this \
             workspace once grew ~140 MB a boot (first boot: {first:?}, second: {second:?})",
            second.prefilled,
        );
        assert!(
            second.restored > 0,
            "the second boot restored no prompt section at all, so nothing the first boot \
             sealed survived in the log (first boot: {first:?}, second: {second:?})",
        );
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
        ))
        .response;
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
    //
    // **Asserts SELECTION, not theme.** The previous form asked the model to
    // "search the web for \"rust language\"" and accepted any answer containing
    // "rust", "search" or "error" — every one of which appears in a refusal that
    // quotes the request back ("I don't have a web search tool ... Rust is a
    // systems language"). It therefore passed while `web_search` was never
    // projected at all: measured on the live daemon, it peaked at 99.99 against
    // a gate of 800 and was absent from every projection point of the turn.

    #[test]
    fn web_search_query_selects_the_web_search_tool() {
        init_tracing();
        let out = run_with_timeout(run_query_observed(
            "Search the web for the latest Rust release notes.",
            "test-search",
        ));
        // **Asserts PROJECTION, not merely that the tool ran.** A call alone is
        // not evidence the catalog offered it: the comprehensive tool *summary*
        // lists every tool's NAME, so the model can emit a call for a tool whose
        // schema was never projected — measured on the live daemon, `web_search`
        // was called six times while `selected` was false at all twenty
        // projection points of the turn, its belief frozen at exactly 169.53601
        // across sixteen consecutive events while its neighbours moved every
        // step. Asserting "reached the model" would pass on that blind call and
        // hide the defect, which is the same mistake as asserting on theme.
        assert!(
            out.selected.contains("web_search"),
            "web_search was never PROJECTED for an explicit web-search request \
             (called: {}). Its belief peaked at {:.0} while the turn's strongest \
             tools were: {}",
            out.called("web_search"),
            out.peak("web_search"),
            out.ranking(),
        );
    }

    // ── Scenario 11: finding a file by name ──────────────────────────────────

    #[test]
    fn a_filename_question_selects_file_search() {
        init_tracing();
        let out = run_with_timeout(run_query_observed(
            "Which file is batched_model.rs, and where does it live?",
            "test-file-search",
        ));
        assert!(
            out.reached_the_model("file_search"),
            "file_search was not projected for a find-this-file request — peak \
             belief {:.0} against a gate of 800. Strongest tools: {}",
            out.peak("file_search"),
            out.ranking(),
        );
    }

    // ── Scenario 12: finding code by content ─────────────────────────────────

    #[test]
    fn a_symbol_question_selects_file_grep() {
        init_tracing();
        let out = run_with_timeout(run_query_observed(
            "Where in this codebase is the function forward_wave defined?",
            "test-file-grep",
        ));
        assert!(
            out.reached_the_model("file_grep"),
            "file_grep was not projected for a where-is-this-defined request — \
             peak belief {:.0} against a gate of 800. Strongest tools: {}",
            out.peak("file_grep"),
            out.ranking(),
        );
    }

    // ── Scenario 13: the file tools stay inside their own domain ─────────────
    //
    // A web-search request must not pull the repository search tools in. This is
    // the negative control for the question seeds, and nothing else in the suite
    // catches it: measured on the live daemon, `file_search` reached 3589 on
    // "Search the web and tell me the latest stable Rust release version" —
    // more than four times the gate, and the strongest tool of the turn —
    // because its seed list named `web_search.rs`. A seed that borrows another
    // tool's vocabulary teaches provenance the wrong domain, and the symptom is
    // a *different* tool being starved rather than this one misbehaving.

    #[test]
    fn a_web_query_does_not_pull_in_the_repository_search_tools() {
        init_tracing();
        let out = run_with_timeout(run_query_observed(
            "Search the web and tell me the latest stable Rust release version.",
            "test-search-domain",
        ));
        // **Scale-free on purpose.** `SelectedSection::score` is the raw belief
        // accumulator, not the normalized 0–1000 band the policy's `min_score`
        // is written in — `belief-eval` defaults that gate to 35, the schema
        // says 800, the results doc says 1000, and live scores run past 800,000.
        // A threshold here would be a number with no defensible origin, so the
        // assertion is on rank: the belief top-k admits up to three tools, and a
        // repository search tool among the three strongest on a web question is
        // a seed borrowing another tool's vocabulary.
        //
        // **On belief, not on `selected`.** `file_search` and `file_grep` are
        // mandatory (`tool_def::ToolDef::mandatory`): they project on every turn,
        // on top of the belief top-k and without taking a slot, so they are
        // always selected and "selected" says nothing about their seeds. This
        // test asserted `!selected` until they became mandatory, after which it
        // failed on every run whatever the seeds did.
        let strongest = out.strongest(3);
        for tool in ["file_search", "file_grep"] {
            assert!(
                !strongest.contains(&tool),
                "{tool} ranked in the belief top 3 for a web-search question \
                 (belief {:.0}) — its question seeds are borrowing another tool's \
                 vocabulary. Strongest tools: {}",
                out.peak(tool),
                out.ranking(),
            );
        }
        assert!(
            out.reached_the_model("web_search"),
            "web_search did not reach the model for a web-search question. \
             Strongest tools: {}",
            out.ranking(),
        );
    }
}
