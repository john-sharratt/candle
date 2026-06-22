/// End-to-end session test — no HTTP layer, no CLI.
///
/// Boots a ZendSession directly, submits a turn, and verifies that:
///   1. Status messages arrive while the model is loading.
///   2. The model eventually produces tokens.
///   3. The response is coherent (contains the expected answer).
///
/// Run with:
///   cargo test -p zend --test integration -- --nocapture
///
/// The test is gated on the `cuda` feature because it requires a real GPU
/// and the GGUF model file on disk.  It is skipped in CPU-only CI.
#[cfg(feature = "cuda")]
mod conversation {
    use std::sync::Arc;

    use futures::StreamExt;

    use zend::config::DaemonConfig;
    use zend::log_broadcast::LogBus;
    use zend::session::{StreamItem, ZendSession};
    use zend::types::{ChatMessage, Role};

    const TIMEOUT_SECS: u64 = 600;

    /// Submits immediately after `start_loading()` so the stream must carry
    /// status events while the inference engine initialises.
    ///
    /// Uses a manually-built runtime so we can call `shutdown_background()`
    /// on timeout — `#[tokio::test]` would hang waiting for the model-load
    /// `spawn_blocking` thread to finish even after the test panics.
    fn init_tracing() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init(); // try_init so repeated calls in parallel tests don't panic
    }

    #[test]
    fn streams_status_messages_then_answer() {
        init_tracing();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");

        // timeout() must be constructed inside block_on — its Sleep future
        // calls Handle::current() eagerly and panics if there is no runtime yet.
        let result = rt.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(TIMEOUT_SECS),
                run_conversation(),
            )
            .await
        });

        // Forcefully drop the runtime without waiting for spawn_blocking threads
        // (the model-load thread may still be running if we timed out).
        rt.shutdown_background();

        result.unwrap_or_else(|_| {
            panic!("test timed out after {TIMEOUT_SECS} s — model did not load or respond in time")
        });
    }

    async fn run_conversation() {
        let log = LogBus::new();
        let config = DaemonConfig {
            workspace: std::env::current_dir().unwrap(),
            port: 0, // not starting HTTP
            ..Default::default()
        };

        let session = Arc::new(ZendSession::new(config, Arc::clone(&log)));

        // Trigger model load — this spawns the background initialisation task.
        session.start_loading();

        // Submit immediately; the stream must carry status events until ready.
        let messages = vec![ChatMessage {
            role: Role::User,
            content: "What is 2 + 2?  Reply with just the number.".to_string(),
        }];
        let mut stream = session
            .submit(messages, Some(64), "test-conv".to_string())
            .await;

        let mut status_msgs: Vec<String> = Vec::new();
        let mut response = String::new();

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
                StreamItem::Projection(_) => {}
            }
        }

        eprintln!("\n\n[FULL RESPONSE]\n{response}");
        eprintln!("[STATUS MESSAGES] {status_msgs:?}");

        assert!(
            !status_msgs.is_empty(),
            "expected at least one StreamItem::Status during model load, got none"
        );
        assert!(!response.is_empty(), "model produced no tokens");
        assert!(
            response.contains('4'),
            "expected '4' in the response to '2+2', got: {response:?}"
        );
    }
}
