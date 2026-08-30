//! Rung 4 (step 5): DeepSeek-V4-Flash inside the CONVERSATION ENGINE over a
//! temp-rooted substrate — the paged-kernel wave model behind the full
//! scheduler/persistence stack.
//!
//! Gates:
//!  (a) the conversation engine answers "Paris" (the same semantic golden the
//!      per-token engine and the wave model passed, now end-to-end);
//!  (b) persist → reboot: a fresh engine over the SAME workspace replays the
//!      substrate (the prior conversation's label survives the restart) and
//!      still answers correctly on a new conversation.
//!
//! Live continuation INTO a resumed timeline (reattaching a `Sequence` to a
//! persisted conversation) rides the step-7 reassembly work — the substrate
//! data-side of resume is what this test pins down.
//!
//! The workspace is ALWAYS an explicit `TempDir`: `workspace_path` unset
//! silently roots `.substrate/` in the current directory.

use std::path::PathBuf;
use std::time::Instant;

use candle_conversation::models::{Model, ModelArch, ModelSpec};
use candle_conversation::SamplingConfig;
use candle_transformers::models::dialect::{Dialect, DialectType};

/// Route `tracing` warnings/errors to the test output — a swallowed seal or
/// persistence failure must be visible, not silent. `RUST_LOG` overrides the
/// default `warn` filter for targeted diagnostics.
fn init_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Coarse dial: any RUST_LOG value raises the level to DEBUG (the
        // workspace tracing-subscriber has no env-filter feature).
        let level = if std::env::var_os("RUST_LOG").is_some() {
            tracing::Level::DEBUG
        } else {
            tracing::Level::WARN
        };
        let _ = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_test_writer()
            .try_init();
    });
}

fn ko_gguf() -> PathBuf {
    PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4").join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf")
}

/// The tokenizer from the local HF hub cache (populated by the engine-level
/// goldens); resolved via the hub API so a cold cache downloads it.
fn tokenizer_json() -> Option<PathBuf> {
    let api = hf_hub::api::sync::Api::new().ok()?;
    api.model("deepseek-ai/DeepSeek-V4-Flash-0731".to_string())
        .get("tokenizer.json")
        .ok()
}

fn deepseek_spec(model_path: &std::path::Path) -> ModelSpec {
    let model_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    ModelSpec {
        arch: ModelArch::DeepSeekV4,
        chat_format: DialectType::DeepSeek,
        dialect: Dialect::deepseek(),
        model_repo: String::new(), // local file only — never downloaded
        model_filename: model_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        model_bytes,
        tokenizer_repo: "deepseek-ai/DeepSeek-V4-Flash-0731".to_string(),
        tokenizer_rev: String::new(),
        default_system_prompt: "You are a concise, factual assistant.".to_string(),
        max_seq_len: 4096,
        default_sampling: SamplingConfig::argmax(),
        supports_thinking: true,
        non_thinking_sampling: None,
    }
}

/// The two parts run as SEPARATE cargo invocations (separate processes) over
/// this fixed workspace: a reboot IS a process boundary — one process cannot
/// pin two ~6.6 GB expert-staging arenas — and the process split mirrors the
/// real daemon-restart semantics exactly.
fn rung4_workspace() -> PathBuf {
    std::env::temp_dir().join("deepseek_rung4_workspace")
}

fn skip_gates() -> Option<(candle::Device, PathBuf, PathBuf)> {
    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("[skip] no CUDA device");
            return None;
        }
    };
    let gguf = ko_gguf();
    if !gguf.exists() {
        eprintln!("[skip] KO gguf absent: {}", gguf.display());
        return None;
    }
    let tok = match tokenizer_json() {
        Some(t) => t,
        None => {
            eprintln!("[skip] tokenizer.json unavailable");
            return None;
        }
    };
    Some((device, gguf, tok))
}

#[test]
#[ignore = "loads DeepSeek-V4-Flash on CUDA (minutes); run part1 then part2 as separate invocations"]
fn deepseek_rung4_part1_paris() -> candle_conversation::Result<()> {
    init_tracing();
    let Some((device, gguf, tok)) = skip_gates() else {
        return Ok(());
    };
    let ws = rung4_workspace();
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).expect("workspace dir");

    let mut builder = Model::custom(deepseek_spec(&gguf))
        .model_path(&gguf)
        .tokenizer_path(&tok)
        .sampling(SamplingConfig::argmax())
        .max_response_tokens(24)
        .workspace_path(&ws);
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();
    let t0 = Instant::now();
    let engine = builder.engine(&device)?;
    eprintln!("[rung4] engine up in {:.1}s", t0.elapsed().as_secs_f32());

    let mut conv = engine.new_conversation(&system_prompt, config)?;
    let timeline = conv.timeline_id();
    let resp = conv.send_turn("What is the capital of France? Reply with only the city name.")?;
    eprintln!(
        "[rung4] response: {:?} ({} tok, {:.2} tok/s)",
        resp.text, resp.stats.tokens_generated, resp.stats.tokens_per_second
    );
    // The turn MUST have sealed into the substrate — a `None` here means the
    // seal path silently skipped and part 2's reboot would find nothing.
    eprintln!("[rung4] seal present: {}", resp.seal.is_some());
    assert!(
        resp.seal.is_some(),
        "dialogue turn did not seal into the substrate"
    );
    // STRICT: must match the per-token CPU/raw-template golden exactly — a
    // verbose or run-on answer means EOS/template drift and must stay red.
    assert_eq!(
        resp.text.trim(),
        "Paris",
        "conversation engine must answer exactly \"Paris\" like the reference path"
    );
    engine.set_conversation_metadata(timeline, "rung4", "paris")?;
    // Engine drop drains persistence; part 2 (a new process) verifies it.
    Ok(())
}

#[test]
#[ignore = "run AFTER part1, as a separate invocation (the reboot)"]
fn deepseek_rung4_part2_reboot() -> candle_conversation::Result<()> {
    init_tracing();
    let Some((device, gguf, tok)) = skip_gates() else {
        return Ok(());
    };
    let ws = rung4_workspace();
    if !ws.join(".substrate").exists() {
        eprintln!("[skip] part1's workspace absent — run part1 first");
        return Ok(());
    }

    let mut builder = Model::custom(deepseek_spec(&gguf))
        .model_path(&gguf)
        .tokenizer_path(&tok)
        .sampling(SamplingConfig::argmax())
        .max_response_tokens(24)
        .workspace_path(&ws);
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();
    let t0 = Instant::now();
    let engine = builder.engine(&device)?;
    eprintln!("[rung4] reboot in {:.1}s", t0.elapsed().as_secs_f32());

    // The startup substrate replay runs on the scheduler thread — wait for it
    // to finish before querying (readers key off `finished`, not done==total).
    let status = engine.substrate_reload_status();
    let deadline = Instant::now() + std::time::Duration::from_secs(120);
    loop {
        let (done, total, finished) = status.snapshot();
        if finished {
            eprintln!("[rung4] substrate replayed: {done}/{total} turns");
            break;
        }
        if Instant::now() > deadline {
            panic!("substrate reload did not finish within 120s ({done}/{total})");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // (b1) the persisted conversation survived the reboot: the metadata
    // written in part 1 is findable in the replayed substrate.
    let found = engine.find_conversations_by_metadata("rung4", "paris");
    assert!(
        !found.is_empty(),
        "persisted conversation metadata lost across reboot"
    );

    // (b2) the rebooted engine still answers correctly.
    let mut conv = engine.new_conversation(&system_prompt, config)?;
    let resp = conv.send_turn("What is the capital of Japan? Reply with only the city name.")?;
    eprintln!("[rung4] post-reboot response: {:?}", resp.text);
    assert_eq!(
        resp.text.trim(),
        "Tokyo",
        "post-reboot engine must answer exactly \"Tokyo\" like the reference path"
    );

    let _ = std::fs::remove_dir_all(&ws);
    Ok(())
}

/// Step-7 rung-4: two turns in ONE live session (no reboot). Turn 2 reprojects
/// turn 1 — reassembling its sealed KV behind a re-rendered inter-turn boundary
/// — so the France→Japan exchange straddles a live seam. Both answers must stay
/// crisp: turn 1 "Paris", then across the seam turn 2 "Tokyo". This is the
/// end-to-end seam gate (the compression-seam fold matters when a group
/// straddles the boundary; at this short depth the window still covers it, so a
/// pass here shows the seam machinery is coherent, and a regression would show
/// the boundary handling corrupting the second turn).
#[test]
#[ignore = "loads DeepSeek-V4-Flash on CUDA (minutes); single-process 2-turn seam"]
fn deepseek_rung4_part3_live_seam() -> candle_conversation::Result<()> {
    init_tracing();
    let Some((device, gguf, tok)) = skip_gates() else {
        return Ok(());
    };
    let ws = std::env::temp_dir().join("deepseek_rung4_seam_workspace");
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).expect("workspace dir");

    let mut builder = Model::custom(deepseek_spec(&gguf))
        .model_path(&gguf)
        .tokenizer_path(&tok)
        .sampling(SamplingConfig::argmax())
        .max_response_tokens(24)
        .workspace_path(&ws);
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();
    let engine = builder.engine(&device)?;

    let mut conv = engine.new_conversation(&system_prompt, config)?;
    let r1 = conv.send_turn("What is the capital of France? Reply with only the city name.")?;
    eprintln!("[seam] turn 1: {:?}", r1.text);
    assert_eq!(r1.text.trim(), "Paris", "turn 1 must answer \"Paris\"");

    // Turn 2 rides the live seam over turn 1.
    let r2 = conv.send_turn("What is the capital of Japan? Reply with only the city name.")?;
    eprintln!("[seam] turn 2 (across seam): {:?}", r2.text);
    assert_eq!(
        r2.text.trim(),
        "Tokyo",
        "turn 2 across the live seam must answer \"Tokyo\""
    );

    let _ = std::fs::remove_dir_all(&ws);
    Ok(())
}
