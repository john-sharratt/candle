//! Tier-3 end-to-end smoke for the `repo_map` + `code_reading`
//! ingestion pipeline.
//!
//! `#[ignore]`-d by default because it loads Qwen3-30B-A3B and
//! prefills a small synthetic workspace through both ingestion
//! passes, then issues a developer query that should retrieve
//! content from the foundational layers.  Run manually with:
//!
//! ```text
//! cargo test -p zend --test zen_code_phase12_smoke -- --ignored --nocapture
//! ```
//!
//! What this test guards:
//!
//! 1. `ingest_repo_map` and `ingest_code_reading` complete without
//!    error against a real engine.
//! 2. The two foundational layers' turns are reachable from the
//!    `dialogue` layer's BDP retrieval — a query that names a
//!    unique identifier surfaces the file that defines it.
//!
//! `phase12_recovers_from_substrate_restart` extends this to the
//! cross-restart path.

use std::fs;
use std::path::Path;
use std::sync::Mutex;

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::projection;
use candle_conversation::{ConversationEngine, SamplingConfig, Sequence, TurnEvent};

use zend::code_read::CodeReadState;
use zend::loading::LoadProgress;

const PROJECTION_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// Unique identifier the test workspace plants in exactly one file.
/// Picked to be highly unlikely to appear in the model's prior — if
/// the model knows about it, that knowledge can only have come from
/// the prefilled code-reading turns.
const PLANTED_FN_NAME: &str = "xyzzy_unique_identifier_42";
const PLANTED_FILE: &str = "src/widget/probe.rs";

fn cuda_device() -> Option<Device> {
    match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => Some(d),
        _ => None,
    }
}

fn init_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_test_writer()
            .try_init();
    });
}

fn build_fixture_workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "Cargo.toml",
        b"[package]\nname = \"demo-app\"\nversion = \"0.1.0\"\n",
    );
    write(
        &root,
        "src/lib.rs",
        b"pub mod widget;\n\npub fn hello() -> &'static str { \"hi\" }\n",
    );
    write(
        &root,
        "src/widget/mod.rs",
        b"pub mod probe;\n\npub struct Widget { pub id: u32 }\n",
    );
    write(
        &root,
        PLANTED_FILE,
        format!(
            "use crate::widget::Widget;\n\n\
             /// The single planted unique identifier.\n\
             pub fn {PLANTED_FN_NAME}(w: &Widget) -> u32 {{\n\
                 w.id + 1\n\
             }}\n"
        )
        .as_bytes(),
    );
    write(
        &root,
        "src/util.rs",
        b"pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    );
    write(&root, "README.md", b"# demo-app\n\nA tiny fixture repo.\n");
    dir
}

fn write(root: &Path, rel: &str, body: &[u8]) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

struct LoadedDaemon {
    engine: ConversationEngine,
    dialogue: Sequence,
    /// Held alive — dropping this tears down the prefilled K/V on the
    /// repo_map layer, which is what we're testing.
    #[allow(dead_code)]
    repo_map: Sequence,
    /// The code_reading pass holds no live Sequences: each per-file
    /// conversation's slot is freed after ingest while the substrate
    /// retains its sealed K/V, so retrieval reads it back from there.
    /// We keep the hash record purely so the field documents the pass.
    #[allow(dead_code)]
    code_read_state: CodeReadState,
}

fn load_daemon(workspace: &Path) -> LoadedDaemon {
    init_tracing();
    let device = cuda_device().expect("CUDA required for Tier-3 phase12 smoke");
    eprintln!(
        "=== Loading Qwen3-30B-A3B against {} ===",
        workspace.display()
    );
    let start = std::time::Instant::now();

    let dialect = Model::Qwen3_30B_A3B_Q4.spec().dialect.clone();
    let workspace_str = workspace.display().to_string();
    let mut proj_builder = projection::Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_YAML,
        &[("workspace", workspace_str.as_str())],
        Some(&dialect),
    )
    .expect("parse projection.yaml");
    let dialogue_layer = proj_builder.id_for_layer("dialogue").unwrap();
    let primary_group = proj_builder.id_for_group("primary_conversation").unwrap();
    let _ = zend::tools::install_tool_catalog(&mut proj_builder, dialogue_layer)
        .expect("install tool catalog");

    let mut builder = Model::Qwen3_30B_A3B_Q4
        .builder()
        .workspace_path(workspace)
        .sampling(SamplingConfig::argmax())
        .seed(0)
        .max_response_tokens(80)
        .thinking(false);
    let conv_config = builder.conversation_config();
    let engine = builder.engine(&device).expect("engine load");
    eprintln!("engine loaded in {:.1}s", start.elapsed().as_secs_f64());

    let tokenizer = engine.tokenizer().clone();
    proj_builder
        .tokenize_templates::<anyhow::Error, _>(|s| {
            let encoded = tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
            Ok(encoded.get_ids().to_vec())
        })
        .expect("tokenize templates");

    let proj_builder_repo_map = proj_builder.clone();
    let proj_builder_code_read = proj_builder.clone();

    let formatted_prompt = builder.format_system_prompt();
    let dialogue = engine
        .new_conversation_with_projection(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config.clone(),
        )
        .expect("new dialogue conv");
    eprintln!(
        "dialogue base built ({:.1}s)",
        start.elapsed().as_secs_f64()
    );

    let progress = LoadProgress::new();
    let (repo_map, walked, _cluster_state) = zend::repo_scan::ingest_repo_map(
        &engine,
        proj_builder_repo_map,
        workspace,
        conv_config.clone(),
        &progress,
    )
    .expect("repo map ingest");
    eprintln!(
        "repo_map ingestion done ({:.1}s) — {} files walked",
        start.elapsed().as_secs_f64(),
        walked.files.len()
    );
    // The per-file code_reading pool locks the engine for its brief
    // create/tombstone ops, so it takes a `&Mutex<ConversationEngine>`.
    // Mirror the daemon: wrap for the pass, then unwrap to hold on.
    let engine = Mutex::new(engine);
    let code_read_state = zend::code_read::ingest_code_reading(
        &engine,
        proj_builder_code_read,
        workspace,
        &walked,
        conv_config,
        &progress,
    )
    .expect("code reading ingest");
    let engine = engine.into_inner().expect("engine mutex not poisoned");
    eprintln!(
        "code_reading ingestion done ({:.1}s)",
        start.elapsed().as_secs_f64()
    );

    LoadedDaemon {
        engine,
        dialogue,
        repo_map,
        code_read_state,
    }
}

fn ask(seq: &mut Sequence, prompt: &str) -> String {
    use candle_conversation::TurnOptions;
    let handle = seq
        .submit_turn_with_options(
            prompt,
            TurnOptions {
                max_tokens: Some(80),
                sampling: Some(SamplingConfig::argmax()),
                ..Default::default()
            },
        )
        .expect("submit_turn");
    let mut response = String::new();
    let mut done = None;
    for event in handle.stream() {
        match event {
            TurnEvent::Done(resp) => {
                response = resp.text.clone();
                done = Some(resp);
                break;
            }
            TurnEvent::Error(e) => panic!("scheduler error: {e}"),
            _ => {}
        }
    }
    let resp = done.expect("turn produced Done");
    seq.finish_turn(handle, &resp).expect("finish_turn");
    response
}

#[test]
#[ignore = "Tier 3: loads Qwen3-30B-A3B + scans workspace + recall (~3 min)"]
fn phase12_recalls_a_known_function_in_the_test_fixture() {
    let dir = build_fixture_workspace();
    let mut daemon = load_daemon(dir.path());

    let prompt = format!(
        "Which file in this codebase defines the function `{PLANTED_FN_NAME}`? \
         Reply with just the file path."
    );
    let answer = ask(&mut daemon.dialogue, &prompt);
    eprintln!("=== answer ===\n{answer}\n=== end answer ===");

    // The response should mention the file path that contains the
    // planted function.  We accept either the exact path or just the
    // basename — the model may shorten it.
    assert!(
        answer.contains(PLANTED_FILE) || answer.contains("probe.rs"),
        "expected response to mention `{PLANTED_FILE}` or `probe.rs`, got: {answer:?}"
    );
    let _ = daemon.engine;
}

#[test]
#[ignore = "Tier 3: loads Qwen3-30B-A3B twice (~6 min) — repo_map + code_reading survive a restart"]
fn phase12_recovers_from_substrate_restart() {
    let dir = build_fixture_workspace();
    let workspace = dir.path().to_path_buf();

    // First load — populates the substrate redo log.
    {
        let mut daemon = load_daemon(&workspace);
        let _ = ask(&mut daemon.dialogue, "Say hi briefly.");
        // Drop the daemon — drives a shutdown checkpoint via Drop on
        // the engine, then the redo log is durable.
    }

    // Second load — same workspace.  The walker re-ingests; the
    // substrate restores prior turns.  Recall must still work.
    let mut daemon = load_daemon(&workspace);
    let prompt = format!(
        "Which file in this codebase defines `{PLANTED_FN_NAME}`? \
         File path only."
    );
    let answer = ask(&mut daemon.dialogue, &prompt);
    eprintln!("=== answer after restart ===\n{answer}\n=== end answer ===");
    assert!(
        answer.contains(PLANTED_FILE) || answer.contains("probe.rs"),
        "post-restart recall failed; got: {answer:?}"
    );
}
