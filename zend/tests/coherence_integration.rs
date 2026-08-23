//! Coherence integration test — runs the same multi-section
//! "tour of the codebase" prompt the user invoked by hand, and asserts
//! the model produced a coherent on-topic response.
//!
//! The test is intentionally **deterministic**: we drive the sampler
//! with `SamplingConfig::argmax()` (temperature = 0, greedy decoding).
//! That removes the high-variance behaviour of the production default
//! (temp=0.8, top_k=40, top_p=0.95) which can occasionally pick `<|im_end|>`
//! as the first sampled token on a perfectly fine K/V state — a
//! single bad draw is not the signal we want when guarding against
//! kernel-level regressions.
//!
//! What the test guards against
//! ────────────────────────────
//! 1. **K/V coherence regression on multi-section projection.**  The
//!    failure mode this test was originally written for is "model
//!    emits EOS as the first sampled token because the substrate-as-
//!    parent section injection feeds it K/V whose chunk layout the
//!    kernels mishandle."  Under greedy decoding that's a stable
//!    signal: a healthy model picks a real content token; a broken
//!    K/V state lets EOS dominate the logits.
//! 2. **Tokenizer/streaming wiring.**  The test asserts at least one
//!    streamed token chunk reaches the caller.
//! 3. **On-topic check (loose).**  The first ~50 tokens of a greedy
//!    answer should mention at least one codebase-flavoured word
//!    (crate, module, candle, rust, file, code).  A model that drifts
//!    off-topic at the first token usually means the projection or
//!    prefill is feeding wrong context.
//!
//! Run with:
//!
//! ```text
//! cargo test -p zend --test coherence_integration --features cuda -- --nocapture
//! ```

#[cfg(feature = "cuda")]
mod coherence {
    use std::sync::Arc;

    use candle_conversation::SamplingConfig;
    use futures::StreamExt;

    use zend::config::DaemonConfig;
    use zend::log_broadcast::LogBus;
    use zend::session::{StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role};

    /// Generous timeout — model load (Qwen3-30B-A3B) + tool catalog
    /// prefill is ~30-60s cold; the generation itself caps at
    /// `max_tokens × ~70ms` (~9 s for 128 tokens at the configured TPS).
    const TIMEOUT_SECS: u64 = 900;

    /// Cap the response length.  We don't need a full essay to confirm
    /// the kernel is healthy — a few dozen tokens of coherent content
    /// is enough.  Lower than the production cap keeps the test fast.
    const MAX_RESPONSE_TOKENS: usize = 128;

    /// Minimum tokens we expect from a healthy model.  The early-EOS
    /// failure mode emits 0–2 tokens.  Greedy decoding on a healthy
    /// model emits much more than that for a codebase-tour prompt.
    /// Set well above the failure-mode regime but well below typical
    /// healthy output so transient sampler quirks don't false-fail us.
    const MIN_EXPECTED_TOKENS: usize = 20;

    fn init_tracing() {
        // EnvFilter picks up RUST_LOG (or falls back to a wide-but-
        // targeted default that includes the per-token sampling traces
        // and the turn-complete KV-cache token dump).  These two
        // targets are the canonical "what did the model see / say?"
        // diagnostics — keeping them on by default means a failing run
        // leaves a complete forensic trail in the test output.
        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "info,\
                 candle_conversation::scheduler::sampling=trace,\
                 candle_conversation::scheduler::context_dump=info,\
                 candle_conversation::scheduler::section_dump=info,\
                 candle_conversation::scheduler::projection_dump=info,\
                 candle_conversation::scheduler::view_create=info,\
                 candle_conversation::scheduler::reproject=trace,\
                 candle_conversation::scheduler::reproject_ranges=info,\
                 candle_transformers::models::slot_state=trace",
            )
        });
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    }

    /// Boot a ZendSession, submit `prompt` with the supplied sampling
    /// config, return `(response_text, token_count_proxy)`.
    /// `token_count_proxy` is the number of `StreamItem::Token` events
    /// — each emitted chunk of streamed text counts as one event, so
    /// it's a loose upper bound on whitespace-stripped word count and
    /// a tight lower bound on "did we generate anything substantial?"
    async fn run_query(
        prompt: &str,
        conv_id: &str,
        max_tokens: Option<usize>,
        sampling: Option<SamplingConfig>,
    ) -> (String, usize) {
        let log = LogBus::new();
        let config = DaemonConfig {
            workspace: std::env::current_dir().unwrap(),
            port: 0,
            ..Default::default()
        };
        let session = Arc::new(ZendSession::new(config, Arc::clone(&log)));
        session.start_loading();

        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        let mut stream = session
            .submit_with_sampling(
                messages,
                max_tokens,
                conv_id.to_string(),
                sampling,
                None,
                None,
                false,
                zend::types::ToolMode::Comprehensive,
                None,
                candle_conversation::SelectionState::default(),
            )
            .await;

        let mut response = String::new();
        let mut token_events: usize = 0;
        let mut status_msgs: Vec<String> = Vec::new();
        while let Some(result) = stream.next().await {
            match result.expect("stream item error") {
                StreamItem::Status(msg) => {
                    eprintln!("[STATUS] {msg}");
                    status_msgs.push(msg);
                }
                StreamItem::Token(tok) => {
                    token_events += 1;
                    eprint!("{tok}");
                    response.push_str(&tok);
                }
                StreamItem::Projection(projection_event_out) => {
                    eprintln!("\n[PROJECTION EVENT] {:?}", projection_event_out.event);
                }
                StreamItem::Tool(status) => {
                    eprintln!("\n[TOOL {}] {:?}", status.phase, status.tools);
                }
            }
        }
        eprintln!("\n\n[FINAL RESPONSE — {token_events} token events]\n{response}");
        eprintln!("[STATUS MESSAGES] {status_msgs:?}");
        (response, token_events)
    }

    fn run_with_timeout<F: std::future::Future<Output = (String, usize)> + Send + 'static>(
        f: F,
    ) -> (String, usize) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let result = rt.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(TIMEOUT_SECS), f).await
        });
        rt.shutdown_background();
        result.unwrap_or_else(|_| panic!("test timed out after {TIMEOUT_SECS}s"))
    }

    /// Greedy decode on the multi-section "tour of the codebase"
    /// prompt.  A healthy K/V projection produces a long, on-topic
    /// answer; the early-EOS failure mode emits 0–2 tokens.
    ///
    /// Loads the real model and decodes a full tour, so it runs for minutes and
    /// is `#[ignore]`d out of the default sweep. Run it deliberately:
    /// `cargo test -p zend --features cuda --release --test coherence_integration -- --ignored`
    #[test]
    #[ignore]
    fn codebase_tour_produces_substantial_answer() {
        init_tracing();
        // Greedy / argmax — deterministic so a failure is reproducible
        // and a pass is a real "kernel + projection are healthy"
        // signal, not luck of the sampler.
        let sampling = SamplingConfig::argmax();
        let (response, token_events) = run_with_timeout(run_query(
            "Give me a tour of the codebase — main crates, key files, \
             and how everything connects.",
            "test-tour",
            Some(MAX_RESPONSE_TOKENS),
            Some(sampling),
        ));
        assert!(
            !response.trim().is_empty(),
            "tour query produced no response — the model emitted EOS \
             before generating any content tokens, or the stream was \
             closed without any token events.  This is the canonical \
             K/V-projection regression symptom.",
        );
        assert!(
            token_events >= MIN_EXPECTED_TOKENS,
            "expected ≥ {MIN_EXPECTED_TOKENS} token events for a \
             'tour of the codebase' reply under greedy decoding, got \
             only {token_events}.  Greedy decoding is deterministic — \
             this is a real regression, not sampler noise.\n\
             Response text: {response:?}",
        );
        // On-topic sanity: greedy generation that's gone off the rails
        // typically drifts immediately.  A healthy answer to the tour
        // prompt should mention at least one of these words within
        // the first MAX_RESPONSE_TOKENS tokens.  Loose check — the
        // exact wording varies across model snapshots.
        let lower = response.to_lowercase();
        let mentions_codebase_topic = lower.contains("crate")
            || lower.contains("module")
            || lower.contains("candle")
            || lower.contains("rust")
            || lower.contains("file")
            || lower.contains("code");
        assert!(
            mentions_codebase_topic,
            "expected at least one codebase-related word \
             (crate/module/candle/rust/file/code) in the response, \
             got: {response:?}",
        );
    }
}
