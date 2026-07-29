//! End-to-end control for the multi-turn "answers the previous question"
//! duplication bug.
//!
//! Symptom (production, `/no_think`): after a *recall* turn ("what question did
//! i ask at the start?" → "You asked, 'Give me a tour…'"), a following math
//! turn ("what is 2 + 2?") reproduces the recall answer **verbatim** instead of
//! answering.
//!
//! Root cause (fixed): the paged-prefill kernel's hoisted palette-routing table
//! was built once per tile from the tile's FIRST chunk slice. Under cum-token
//! addressing a tile straddles two slices whenever the prefix contains a
//! partial chunk (every sealed turn tail), and the straddle positions were
//! routed through the wrong slice's palette map — garbling every quantized
//! turn after the first partial chunk in the projection. The kernel-level
//! regression test for the exact mechanism is
//! `candle-transformers::models::prefill_utils::tests::
//! correctness_prefill_straddle_shuffled_pal_map`; this test is the full-stack
//! control that drives a real conversation through projection, adaptive
//! quantization, and reprojection.
//!
//! The control is `#[ignore]`d (loads the 30B model, ~4 min, GPU-exclusive) —
//! run it explicitly (needs CUDA + the GGUF model; drives a fresh conversation):
//!
//! ```text
//! RUST_LOG=info cargo test -p zend --test duplication_replay \
//!   --features cuda replay::repro_dup -- --exact --ignored --nocapture
//! ```

#[cfg(feature = "cuda")]
mod replay {
    use std::path::PathBuf;
    use std::sync::Arc;

    use candle_conversation::{OptionalState, SamplingConfig, SelectionState, NO_THINK_SELECTOR};
    use futures::StreamExt;

    use zend::config::DaemonConfig;
    use zend::log_broadcast::LogBus;
    use zend::session::{StreamItem, ZendSession};
    use zend::types::ToolMode;
    use zend::types::{ChatMessage, Role};

    const MAX_RESPONSE_TOKENS: usize = 128;

    fn init_tracing() {
        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    }

    /// Production dials for the failing scenario: thinking OFF (`/no_think`),
    /// standard length, tools off — mirrors `chat::dial_selection(effort=0)`.
    fn no_think_standard_selection() -> SelectionState {
        let mut sel = SelectionState::new();
        sel.select("thinking_effort", "off");
        sel.select("response_length", "standard");
        sel.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
        sel.set_optional("reasoning_stance", OptionalState::Absent);
        sel.set_optional("tools_enabled", OptionalState::Absent);
        sel
    }

    fn build_session() -> Arc<ZendSession> {
        // CARGO_MANIFEST_DIR is `<root>/zend`, so its parent is the workspace
        // (which holds `.substrate/substrate.log`).
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("zend has a parent dir")
            .to_path_buf();

        let log = LogBus::new();
        let config = DaemonConfig {
            workspace,
            port: 0,
            // Bring the daemon up without the workspace ingest sweep — this test
            // exercises decode replay, not retrieval.
            disabled_layers: ["repo_map", "code_reading"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        };
        // NOTE: caller must invoke `session.start_loading()` from WITHIN the Tokio
        // runtime (inside `block_on`) — the workspace watcher spawns onto the
        // ambient runtime and panics if none is current.
        Arc::new(ZendSession::new(config, Arc::clone(&log)))
    }

    /// Submit one user turn on `conv_id` (argmax sampling) and return the
    /// decoded answer, newline-flattened.
    async fn ask(session: &Arc<ZendSession>, conv_id: &str, prompt: &str) -> String {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        let mut stream = session
            .submit_with_sampling(
                messages,
                Some(MAX_RESPONSE_TOKENS),
                conv_id.to_string(),
                Some(SamplingConfig::argmax()),
                None,
                None,
                false,
                ToolMode::None,
                None,
                no_think_standard_selection(),
            )
            .await;
        let fut = async {
            let mut resp = String::new();
            while let Some(r) = stream.next().await {
                if let StreamItem::Token(t) = r.expect("stream item") {
                    eprint!("{t}");
                    resp.push_str(&t);
                }
            }
            resp
        };
        tokio::time::timeout(std::time::Duration::from_secs(600), fut)
            .await
            .expect("turn timed out")
            .trim()
            .replace('\n', " ")
    }

    /// Drive the escalating math/recall ladder: a tour turn, then six rounds of
    /// (math question, recall question). Returns one `(math, math_answer,
    /// recall_answer)` triple per round. The recall re-selects the whole history
    /// each round, so every round exercises reprojection over the accumulated
    /// quantized turns — the pattern the duplication reproduced on.
    fn drive_escalating_ladder(
        rt: &tokio::runtime::Runtime,
        session: &Arc<ZendSession>,
        conv: &str,
        disable_summariser: bool,
    ) -> Vec<(String, String, String)> {
        let math = ["1 + 1", "2 + 2", "4 + 4", "6 + 6", "8 + 8", "10 + 10"];
        const RECALL_Q: &str = "what question did i ask at the start of the conversation?";
        rt.block_on(async {
            let tour = ask(
                session,
                conv,
                "Give me a tour of the codebase — main crates, key files, and how everything connects.",
            )
            .await;
            eprintln!("\n===== start (tour) =====\n{tour}\n");
            if disable_summariser {
                // Timeline now exists (first turn submitted); switch its feed off.
                let ok = session.set_conversation_summarize(conv, false);
                eprintln!("[summariser DISABLED for {conv}: applied={}]", ok.is_some());
            }

            let mut out = Vec::new();
            for m in math {
                let ma = ask(session, conv, &format!("what is {m}?")).await;
                eprintln!("\n===== {m} =====\n{ma}\n");
                // Let the async summariser / tier-migration process the new turns
                // before the recall pulls the history back in (mirrors production).
                tokio::time::sleep(std::time::Duration::from_secs(6)).await;
                let ra = ask(session, conv, RECALL_Q).await;
                eprintln!("\n===== recall after {m} =====\n{ra}\n");
                out.push((m.to_string(), ma, ra));
                tokio::time::sleep(std::time::Duration::from_secs(6)).await;
            }
            out
        })
    }

    /// Print the ladder table and return the indices of duplicated math rounds.
    fn print_ladder(label: &str, rounds: &[(String, String, String)]) -> Vec<usize> {
        let math_dup = |a: &str| -> bool {
            let l = a.to_lowercase();
            l.contains("you asked") || l.contains("tour of the codebase")
        };
        eprintln!("\n=========== ESCALATING LADDER [{label}] ===========");
        let mut bad_rounds: Vec<usize> = Vec::new();
        for (i, (m, ma, ra)) in rounds.iter().enumerate() {
            let dup = math_dup(ma);
            if dup {
                bad_rounds.push(i);
            }
            let ma_short: String = ma.chars().take(48).collect();
            let ra_short: String = ra.chars().take(48).collect();
            eprintln!(
                "  round {i} {m:7}  math[{}] {ma_short:?}\n             recall {ra_short:?}",
                if dup { "DUP " } else { "ok  " }
            );
        }
        eprintln!(
            "  => {} / {} math answers duplicated",
            bad_rounds.len(),
            rounds.len()
        );
        eprintln!("=================================================\n");
        bad_rounds
    }

    /// The duplication control: drive the ladder on a fresh conversation with
    /// everything at production settings (adaptive quantization, summariser on,
    /// persistence on) and assert no math turn reproduces the recall answer.
    /// Before the prefill palette-routing fix this failed at 3–6 of 6 rounds.
    #[test]
    #[ignore = "loads the 30B model and drives 13 live turns (~4 min, GPU-exclusive); run explicitly"]
    fn repro_dup() {
        init_tracing();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let session = build_session();
        rt.block_on(async { session.start_loading() });

        // Unique conv id per run so repeated runs don't pile onto one timeline.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let conv = format!("repro-dup-{nonce}");
        let rounds = drive_escalating_ladder(&rt, &session, &conv, false);
        rt.shutdown_background();

        let bad = print_ladder("repro-dup", &rounds);
        eprintln!(
            "\n>>> DUPLICATION: {} / {} math turns duplicated <<<\n",
            bad.len(),
            rounds.len()
        );
        assert!(
            bad.is_empty(),
            "DUP reproduced — {} math turn(s) duplicated (rounds {bad:?}). \
             The prefill straddle palette-routing bug (or a new corruption with \
             the same symptom) is back; see the ladder table above.",
            bad.len()
        );
    }
}
