//! CONTROL test for reprojection cost.
//!
//! Boots the real engine against an isolated copy of the live substrate, primes
//! a deep on-topic conversation ONCE, then submits a control prompt whose
//! projection selects many turns — the baseline we measure before/after
//! optimizing the glue prefill.
//!
//! Fast iteration by design:
//!   - Substrate copied to `target/reproject_control_ws/` once and reused.
//!   - The primed conversation persists in that copy, so subsequent runs SKIP
//!     priming and just resume it (`submit` reads history from the timeline).
//!   - `REPRIME=1` deletes the workspace for a clean fresh baseline.
//!
//! The always-on instrumentation prints, for the control projection:
//!   reproject (zero-copy rebuild) … apply_ms=… turns=… sections=… segments=…
//!   apply_segments breakdown … n_prefill=… prefill_tokens=…
//!
//! Run:
//!   cargo test -p zend --test reproject_control --features cuda -- --ignored --nocapture
//! grep stderr between "=== CONTROL PROJECTION START/END ===".

#[cfg(feature = "cuda")]
mod control {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use futures::StreamExt;

    use zend::config::DaemonConfig;
    use zend::log_broadcast::LogBus;
    use zend::session::{StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role};

    const TIMEOUT_SECS: u64 = 1200;
    const CONV_ID: &str = "reproject-control";

    /// On-topic questions that build a deep, selectable conversation so the
    /// control projection retrieves many turns.
    const PRIMING: &[&str] = &[
        "Walk me through the chunked KV cache backing in candle-nn — how are chunks laid out?",
        "How does the CompressionPolicy choose K and V formats per 32-token block?",
        "Explain the gid pool allocator: how chunks are claimed and freed.",
        "How does the reproject zero-copy rebuild reuse sealed turns without recompute?",
        "Describe the provenance BDP scan and how Q vectors are captured during decode.",
        "How does elevate_to_hot move turns across the GPU/RAM/NVMe tiers?",
    ];

    const CONTROL_PROMPT: &str =
        "Summarize the full KV cache tiering and reprojection architecture in detail.";

    fn init_tracing() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();
    }

    fn candle_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf()
    }

    /// Isolated workspace (one-time substrate copy). `(workspace, needs_priming)`.
    /// `REPRIME=1` wipes it for a clean fresh baseline.
    fn workspace() -> (PathBuf, bool) {
        let root = candle_root();
        let dst_root = root.join("target").join("reproject_control_ws");
        if std::env::var("REPRIME").is_ok() {
            let _ = std::fs::remove_dir_all(&dst_root);
        }
        let dst_sub = dst_root.join(".substrate");
        let dst_log = dst_sub.join("substrate.log");
        let sentinel = dst_root.join(".control_primed");

        if dst_log.exists() {
            let primed = sentinel.exists();
            eprintln!(
                "reusing substrate copy at {} (primed={primed})",
                dst_root.display()
            );
            return (dst_root, !primed);
        }
        std::fs::create_dir_all(&dst_sub).unwrap();
        let src_sub = root.join(".substrate");
        let src_log = src_sub.join("substrate.log");
        eprintln!(
            "one-time copy substrate {} ({} MB) -> {}",
            src_log.display(),
            std::fs::metadata(&src_log)
                .map(|m| m.len() / 1_000_000)
                .unwrap_or(0),
            dst_log.display()
        );
        std::fs::copy(&src_log, &dst_log).expect("copy substrate.log");
        let tok = src_sub.join("tokenizer.json");
        if tok.exists() {
            std::fs::copy(&tok, dst_sub.join("tokenizer.json")).ok();
        }
        (dst_root, true)
    }

    #[test]
    #[ignore]
    fn reproject_control() {
        init_tracing();
        let (ws, needs_priming) = workspace();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build runtime");
        let result = rt.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(TIMEOUT_SECS),
                run(ws.clone(), needs_priming),
            )
            .await
        });
        rt.shutdown_background();
        result.unwrap_or_else(|_| panic!("timed out after {TIMEOUT_SECS}s"));

        if needs_priming {
            std::fs::write(ws.join(".control_primed"), b"1").ok();
        }
    }

    async fn submit_drain(session: &Arc<ZendSession>, prompt: &str, max_tokens: usize) -> usize {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        let mut stream = session
            .submit(messages, Some(max_tokens), CONV_ID.to_string(), None, None)
            .await;
        let mut n = 0usize;
        let mut text = String::new();
        while let Some(result) = stream.next().await {
            match result.expect("stream item error") {
                StreamItem::Status(msg) => eprintln!("[STATUS] {msg}"),
                StreamItem::Token(t) => {
                    text.push_str(&t);
                    n += 1;
                }
            }
        }
        // Print the decoded text so coherence can be eyeballed — a wrong
        // gap-fill `col_actual_pos` still emits tokens, just garbage ones.
        eprintln!("[DECODED {n}] {text}");
        n
    }

    async fn run(workspace: PathBuf, needs_priming: bool) {
        let log = LogBus::new();
        let config = DaemonConfig {
            workspace,
            port: 0,
            skip_code_read: true,
            skip_repo_scan: true,
            ..Default::default()
        };
        let session = Arc::new(ZendSession::new(config, Arc::clone(&log)));
        session.start_loading();

        if needs_priming {
            eprintln!("priming '{CONV_ID}' with {} turns…", PRIMING.len());
            for (i, p) in PRIMING.iter().enumerate() {
                let n = submit_drain(&session, p, 60).await;
                eprintln!("[PRIME {i}] decoded {n} tokens");
            }
        }

        // Decode past the every_n_tokens=64 reproject cadence so a mid-decode
        // reproject fires — that churning reproject IS the slow control case
        // (the submit projection is cache-stable and not what we're measuring).
        eprintln!("=== CONTROL PROJECTION START ===");
        // Decode well past several every_n_tokens=64 boundaries so multiple
        // mid-decode reprojects fire: with budget-aware eviction the first loads
        // the working set cold, but subsequent reprojects should find it still
        // hot (cold_to_hot → ~0).
        let n = submit_drain(&session, CONTROL_PROMPT, 256).await;
        eprintln!("=== CONTROL PROJECTION END (decoded {n} tokens) ===");
        assert!(n > 0, "model produced no tokens");
    }
}
