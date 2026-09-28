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
//! 1. `BranchIngest::pass` — the sole ingestion entry point, also run by the
//!    daemon's background ingest worker — completes without error against a
//!    real engine over the fixture's branch, starting from an empty substrate
//!    (a fresh install).
//! 2. The two foundational layers' turns are reachable from the `dialogue`
//!    layer's BDP retrieval, scoped to the dialogue's own base — a query that
//!    names a unique identifier surfaces the file that defines it.
//!
//! `phase12_recovers_from_substrate_restart` extends this to the
//! cross-restart path.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::Instant;

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::projection;
use candle_conversation::stencil::ThinkMode;
use candle_conversation::{ConversationEngine, SamplingConfig, Sequence, TurnEvent, TurnOptions};
use tempfile::TempDir;
use tracing::Level;

use zend::branch_ingest::filter::IngestScope;
use zend::branch_ingest::{BranchIngest, LayerPass};
use zend::ingest::IngestMode;
use zend::loading::LoadProgress;
use zend::refresh_ctx::RefreshContext;
use zend::retrieval_scope::RetrievalScope;
use zend::tools::{install_tool_catalog, tool_catalog, ToolHost};
use zend::types::ToolMode;
use zend::workspace::single_repo;
use zend_tools::state::Secrets;

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
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_max_level(Level::WARN)
            .with_test_writer()
            .try_init();
    });
}

/// The fixture workspace's one repository, where every file is planted.
const FIXTURE_REPO: &str = "demo-app";

fn build_fixture_workspace() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join(FIXTURE_REPO);
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
    // The ingest reads branches, never the folder: commit it on `main`.
    for args in [
        &["init", "-q"][..],
        &["symbolic-ref", "HEAD", "refs/heads/main"],
        &["add", "-A"],
        &["commit", "-q", "-m", "fixture"],
    ] {
        let out = Command::new("git")
            .arg("-C")
            .arg(&root)
            .args([
                "-c",
                "core.hooksPath=",
                "-c",
                "user.name=T",
                "-c",
                "user.email=t@x",
            ])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    dir
}

fn write(root: &Path, rel: &str, body: &[u8]) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

/// Neither ingest layer holds a live Sequence: each per-directory / per-file
/// conversation's slot is freed after ingest while the substrate retains its
/// sealed K/V, so retrieval reads it back from there.
struct LoadedDaemon {
    engine: ConversationEngine,
    dialogue: Sequence,
}

fn load_daemon(workspace: &Path) -> LoadedDaemon {
    init_tracing();
    let device = cuda_device().expect("CUDA required for Tier-3 phase12 smoke");
    eprintln!(
        "=== Loading Qwen3-30B-A3B against {} ===",
        workspace.display()
    );
    let start = Instant::now();

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
    let _ = install_tool_catalog(&mut proj_builder).expect("install tool catalog");

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
    // The same three pieces `InferenceState::load` compiles once and threads
    // through `RefreshContext` — see `session.rs`. `code_reading`'s hidden
    // per-file conversations use these to frame identically to `dialogue`.
    let tool_stencil = engine
        .compile_tool_stencil(tool_catalog())
        .expect("tool stencil compile");
    let think_steering = engine
        .compile_think_steering()
        .expect("think steering compile");
    // `Quick`, not `Off` — see `RefreshContext::think_triggers`'s doc.
    let think_triggers = match &think_steering {
        Some(ts) => ts.registry_for(&tool_stencil, ThinkMode::Quick),
        None => Arc::clone(&tool_stencil),
    };
    let served = single_repo(workspace, FIXTURE_REPO).expect("workspace");
    let tool_host = ToolHost::new(&served, Arc::new(Secrets::empty())).expect("tool host");
    let tool_ctx = tool_host.context_for(ToolMode::Restricted, &tool_host.conversation_files());
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

    // `base_conv`'s counterpart for ingestion — see `InferenceState::ingest_bases`
    // in `session.rs`. The pass forks each unit's conversation off these
    // instead of each independently re-running the schema's "eager section
    // ingestion", exactly mirroring `dialogue` above.
    let repo_map_layer = proj_builder_repo_map.id_for_layer("repo_map").unwrap();
    let repo_map_group = proj_builder_repo_map.id_for_group("structure").unwrap();
    let repo_map_base = Mutex::new(
        engine
            .new_conversation_with_projection(
                &formatted_prompt,
                proj_builder_repo_map.clone(),
                repo_map_layer,
                repo_map_group,
                conv_config.clone(),
            )
            .expect("new repo_map ingest base"),
    );
    let code_read_layer = proj_builder_code_read.id_for_layer("code_reading").unwrap();
    let code_read_group = proj_builder_code_read.id_for_group("scopes").unwrap();
    let code_read_base = Mutex::new(
        engine
            .new_conversation_with_projection(
                &formatted_prompt,
                proj_builder_code_read.clone(),
                code_read_layer,
                code_read_group,
                conv_config.clone(),
            )
            .expect("new code_reading ingest base"),
    );

    let progress = Arc::new(LoadProgress::new());
    // The pass locks the engine for its brief create/tombstone ops, so it
    // takes a `&Mutex<ConversationEngine>` via `RefreshContext`. Mirror the
    // daemon: both layers' groups scoped, one pass of the branch ingest over
    // both layers — exactly as the worker's first pass runs it on a fresh
    // install — then the dialogue's scope set from its own files before it
    // is asked anything.
    engine.mark_group_scoped(repo_map_group);
    engine.mark_group_scoped(code_read_group);
    let engine = Mutex::new(engine);
    let scope = IngestScope::new("", None);
    let retrieval =
        RetrievalScope::new(Some(code_read_group), Some((repo_map_group, scope.clone())));
    let ctx = RefreshContext {
        engine: &engine,
        proj_builder: proj_builder_code_read,
        config: conv_config,
        formatted_prompt: &formatted_prompt,
        think_triggers,
        tool_ctx,
        retrieval: &retrieval,
    };
    let layers = [
        LayerPass {
            name: "repo_map",
            mode: IngestMode::Folders,
            scope: scope.clone(),
            base: &repo_map_base,
        },
        LayerPass {
            name: "code_reading",
            mode: IngestMode::Files,
            scope: scope.clone(),
            base: &code_read_base,
        },
    ];
    let changed = BranchIngest::default()
        .pass(&ctx, &served, &layers, &progress)
        .expect("branch ingest");
    eprintln!(
        "branch ingestion done ({:.1}s) — {}",
        start.elapsed().as_secs_f64(),
        if changed {
            "units ingested"
        } else {
            "every unit already committed"
        },
    );
    retrieval.apply(
        &engine,
        dialogue.timeline_id(),
        &tool_host.conversation_files(),
    );
    let engine = engine.into_inner().expect("engine mutex not poisoned");

    LoadedDaemon { engine, dialogue }
}

fn ask(seq: &mut Sequence, prompt: &str) -> String {
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

    // Second load — same workspace. Every unit is already committed, so the
    // pass ingests nothing; the substrate restores prior turns. Recall must
    // still work.
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
