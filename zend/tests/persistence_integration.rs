/// End-to-end persistence test — the P8 daemon-integration exit gate.
///
/// Boots a `ZendSession` against a throwaway workspace, runs a multi-turn
/// conversation, performs the graceful-shutdown checkpoint, then **reopens
/// the substrate from disk** and asserts the turns were recorded durably —
/// the continuity a daemon restart relies on.
///
/// Run with:
///   cargo test -p zend --features cuda --test persistence_integration -- --nocapture
///
/// Gated on `cuda`: it needs a real GPU and the GGUF model on disk, so it
/// is skipped in CPU-only CI. The substrate format itself is covered by the
/// 73 CPU tests in `candle_conversation::persistence`.
#[cfg(feature = "cuda")]
mod persistence {
    use std::sync::Arc;

    use futures::StreamExt;

    use candle_conversation::persistence::{SubstratePersistence, SUBSTRATE_DIR};
    use candle_conversation::substrate::Substrate;
    use zend::config::DaemonConfig;
    use zend::log_broadcast::LogBus;
    use zend::session::{StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role};

    const TIMEOUT_SECS: u64 = 600;

    fn init_tracing() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();
    }

    /// Drain a submitted turn to completion, returning the joined response.
    async fn run_turn(session: &Arc<ZendSession>, conv_id: &str, prompt: &str) -> String {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        let mut stream = session
            .submit(
                messages,
                Some(64),
                conv_id.to_string(),
                None,
                None,
                false,
                zend::types::ToolMode::Comprehensive,
                candle_conversation::SelectionState::default(),
            )
            .await;
        let mut response = String::new();
        while let Some(result) = stream.next().await {
            if let StreamItem::Token(tok) = result.expect("stream item error") {
                response.push_str(&tok);
            }
        }
        response
    }

    #[test]
    fn turns_persist_and_recover_across_a_simulated_restart() {
        init_tracing();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");

        // A throwaway workspace — the substrate log lands in its `.substrate/`.
        let workspace = std::env::temp_dir().join(format!(
            "zend_p8_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let ws = workspace.clone();
        let result = rt.block_on(async move {
            tokio::time::timeout(
                std::time::Duration::from_secs(TIMEOUT_SECS),
                run_session(ws),
            )
            .await
        });
        rt.shutdown_background();
        result.unwrap_or_else(|_| panic!("test timed out after {TIMEOUT_SECS} s"));

        // ── Simulated restart: reopen the substrate straight from disk ──────
        let log_path = workspace.join(SUBSTRATE_DIR).join("seg-0000000001.log");
        assert!(
            log_path.exists(),
            "the daemon must leave a substrate log at {}",
            log_path.display(),
        );
        // Per-stream runtime state moved from the manifest into the
        // walker-populated `Substrate` (see `open_in_with_substrate`),
        // so reopen through that path and count the streams that came
        // back with a declaration — one per recorded turn.
        let mut substrate = Substrate::new();
        let _recovered = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
            .expect("substrate must reopen cleanly");
        let turn_streams = substrate
            .all_stream_ids()
            .filter(|id| {
                substrate
                    .stream_of(*id)
                    .map(|s| s.decl.is_some())
                    .unwrap_or(false)
            })
            .count();
        assert!(
            turn_streams >= 2,
            "expected the two recorded turns to survive the restart, found {turn_streams}",
        );

        std::fs::remove_dir_all(&workspace).ok();
    }

    async fn run_session(workspace: std::path::PathBuf) {
        let log = LogBus::new();
        let config = DaemonConfig {
            workspace,
            port: 0,
            ..Default::default()
        };
        let session = Arc::new(ZendSession::new(config, Arc::clone(&log)));
        session.start_loading();

        // Two turns on one conversation — the multi-turn continuity case.
        let r1 = run_turn(
            &session,
            "p8-conv",
            "What is 2 + 2? Reply with just the number.",
        )
        .await;
        eprintln!("[TURN 1] {r1}");
        assert!(!r1.is_empty(), "turn 1 produced no tokens");

        let r2 = run_turn(
            &session,
            "p8-conv",
            "Now add 3 to that. Reply with just the number.",
        )
        .await;
        eprintln!("[TURN 2] {r2}");
        assert!(!r2.is_empty(), "turn 2 produced no tokens");

        // Graceful shutdown — flush + checkpoint the redo log.
        session.shutdown().await;
    }
}
