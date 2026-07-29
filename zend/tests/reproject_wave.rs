//! 2-conversation WAVE test for the cross-conversation gap-fill.
//!
//! Two conversations decode CONCURRENTLY past the reproject cadence, so
//! `drain_pending_reprojections` collects both into one batch and fires a single
//! multi-slot gap-fill forward (`gap-fill wave: n_slots=2`). Validates that the
//! flat multi-slot `col_actual_pos` packing is correct — both conversations must
//! produce coherent output, not just one.
//!
//! Fast iteration: substrate copied once to `target/reproject_wave_ws/`, the two
//! primed conversations persist there and are reused. `REPRIME=1` wipes it.
//!
//! Run:
//!   cargo test -p zend --test reproject_wave --features cuda -- --ignored --nocapture
//! Look for "gap-fill wave: ... n_slots=2" and the two "[DECODED ...]" lines
//! between "=== WAVE PROJECTION START/END ===".

#[cfg(feature = "cuda")]
mod wave {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use futures::StreamExt;

    use zend::config::DaemonConfig;
    use zend::log_broadcast::LogBus;
    use zend::session::{StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role};

    const TIMEOUT_SECS: u64 = 1200;
    const CONV_A: &str = "reproject-wave-a";
    const CONV_B: &str = "reproject-wave-b";

    /// A few on-topic turns so each conversation has selectable history that
    /// triggers boundary-glue churn on the control reproject.
    const PRIMING: &[&str] = &[
        "Walk me through the chunked KV cache backing in candle-nn — how are chunks laid out?",
        "How does the CompressionPolicy choose K and V formats per 32-token block?",
        "Explain the gid pool allocator: how chunks are claimed and freed.",
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
    fn workspace() -> (PathBuf, bool) {
        let root = candle_root();
        let dst_root = root.join("target").join("reproject_wave_ws");
        if std::env::var("REPRIME").is_ok() {
            let _ = std::fs::remove_dir_all(&dst_root);
        }
        let dst_sub = dst_root.join(".substrate");
        let dst_log = dst_sub.join("substrate.log");
        let sentinel = dst_root.join(".wave_primed");

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
        (dst_root, true)
    }

    #[test]
    #[ignore]
    fn reproject_wave() {
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
            std::fs::write(ws.join(".wave_primed"), b"1").ok();
        }
    }

    async fn submit_drain(
        session: &Arc<ZendSession>,
        conv_id: &str,
        prompt: &str,
        max_tokens: usize,
    ) -> (usize, String) {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        let mut stream = session
            .submit(
                messages,
                Some(max_tokens),
                conv_id.to_string(),
                None,
                None,
                false,
                zend::types::ToolMode::Comprehensive,
                None,
                candle_conversation::SelectionState::default(),
            )
            .await;
        let mut n = 0usize;
        let mut text = String::new();
        while let Some(result) = stream.next().await {
            match result.expect("stream item error") {
                StreamItem::Status(_) => {}
                StreamItem::Token(t) => {
                    text.push_str(&t);
                    n += 1;
                }
                StreamItem::Projection(_) => {}
                StreamItem::Tool(_) => {}
            }
        }
        (n, text)
    }

    async fn run(workspace: PathBuf, needs_priming: bool) {
        let log = LogBus::new();
        let config = DaemonConfig {
            workspace,
            port: 0,
            disabled_layers: ["repo_map", "code_reading"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        };
        let session = Arc::new(ZendSession::new(config, Arc::clone(&log)));
        session.start_loading();

        if needs_priming {
            for conv in [CONV_A, CONV_B] {
                eprintln!("priming '{conv}' with {} turns…", PRIMING.len());
                for (i, p) in PRIMING.iter().enumerate() {
                    let (n, _) = submit_drain(&session, conv, p, 60).await;
                    eprintln!("[PRIME {conv} {i}] decoded {n} tokens");
                }
            }
        }

        // Concurrent control submits: both conversations decode in the same
        // scheduler wave, so they cross the every_n_tokens reproject cadence
        // together and `drain_pending_reprojections` batches their boundary glue
        // into one multi-slot gap-fill forward.
        eprintln!("=== WAVE PROJECTION START ===");
        let (a, b) = tokio::join!(
            submit_drain(&session, CONV_A, CONTROL_PROMPT, 80),
            submit_drain(&session, CONV_B, CONTROL_PROMPT, 80),
        );
        let (na, text_a) = a;
        let (nb, text_b) = b;
        eprintln!("=== WAVE PROJECTION END (a={na} b={nb}) ===");
        eprintln!("[DECODED A {na}] {text_a}");
        eprintln!("[DECODED B {nb}] {text_b}");

        assert!(na > 0, "conversation A produced no tokens");
        assert!(nb > 0, "conversation B produced no tokens");
    }
}
