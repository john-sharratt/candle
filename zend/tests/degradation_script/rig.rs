//! Driving the script through a real daemon: boot, converse, judge, retire.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use candle_conversation::models::Model;
use candle_conversation::{Role as TurnRole, SelectionState};
use futures::StreamExt;
use tracing_subscriber::EnvFilter;
use zend::api::chat::{apply_tools_dial, dial_selection};
use zend::config::{DaemonConfig, ModelChoice};
use zend::log_broadcast::LogBus;
use zend::session::{StreamItem, ZendSession};
use zend::types::{ChatMessage, Role, ToolMode};

use crate::common::needs_compaction;
use crate::verdict::{verdict_for, Verdict};

/// The script, in order. The last question is the one judged: it has the model
/// read `docs/unbounded_agents.md` (~44k tokens in one tool result) and give its
/// view, after four turns that each leave a question it could answer instead.
pub const SCRIPT: [&str; 5] = [
    "hi - how are you?",
    "what time is it?",
    "what is the sqrt of 237891273498722349871229384712934",
    "tell me about this repo?",
    "read the unbounded context paper and tell me what you think",
];

/// Fresh conversations per arm. Sampling is reseeded every turn, as a user's
/// chat is, so an arm measures a rate — and eight is enough to tell a model that
/// loses the question from one that does not.
pub const CONVERSATIONS: usize = 8;

/// Conversations that must read the paper for an arm to have measured anything:
/// one that never reads it (a search that found nothing, a file it could not
/// locate) says nothing about what happens after the read.
pub const REQUIRED_READS: usize = 6;

/// A budget past any depth the script reaches, and past the checkpoint's
/// 262,144-token context, so the selection is the identity.
pub const DENSE_BUDGET: usize = 1 << 20;

/// Cap on one arm, boot and first-run calibration included.
const ARM_TIMEOUT: Duration = Duration::from_secs(4 * 60 * 60);

/// What a finished arm's runtime may take to wind its tasks down.
const RUNTIME_WIND_DOWN: Duration = Duration::from_secs(60);

/// The files the script's tools read, copied from the repo into the workspace.
const SCRIPT_FILES: [&str; 2] = ["README.md", "docs/unbounded_agents.md"];

/// What the engine logs into an arm's output: every warning — the index
/// alarms (a seal that does not cover its turn, a page placed past its K/V, a
/// wave entering with an index that disagrees) among them — plus the turn and
/// index-page lines that say where each one happened.
const LOG_FILTER: &str =
    "warn,zend::session=info,candle_conversation::scheduler::unit_boundary=info";

/// One arm of the experiment.
pub struct Arm {
    pub name: &'static str,
    pub model: Model,
    /// The workspace directory under `target/tmp`, shared by the arms of one
    /// model: it holds that model's K/V and tool-catalog calibration.
    pub workspace: &'static str,
    /// `DaemonConfig::qsa_selection_budget` for the arm.
    pub qsa_selection_budget: Option<usize>,
}

/// Run `arm` to its verdicts and assert on them.
pub fn run_arm(arm: &Arm) {
    // Plain text, so the log reads and greps as written; a second arm in the
    // same process keeps the first one's subscriber.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(LOG_FILTER))
        .with_ansi(false)
        .with_test_writer()
        .try_init();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let verdicts = rt
        .block_on(async { tokio::time::timeout(ARM_TIMEOUT, converse(arm)).await })
        .unwrap_or_else(|_| panic!("{} timed out after {ARM_TIMEOUT:?}", arm.name));
    rt.shutdown_timeout(RUNTIME_WIND_DOWN);
    report(arm, &verdicts);
}

/// The arm's workspace, holding the files the script reads.
fn workspace(arm: &Arm) -> PathBuf {
    let ws = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(arm.workspace);
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("zend sits inside the repo");
    for rel in SCRIPT_FILES {
        let dst = ws.join(rel);
        std::fs::create_dir_all(dst.parent().expect("a file has a parent"))
            .expect("create the workspace");
        std::fs::copy(repo.join(rel), &dst).unwrap_or_else(|e| panic!("copy {rel}: {e}"));
    }
    ws
}

/// Boot the arm's daemon and run every conversation, returning each one's final
/// turn and its verdict.
async fn converse(arm: &Arm) -> Vec<(String, Verdict)> {
    let ws = workspace(arm);
    let config = DaemonConfig {
        compact_substrate: needs_compaction(&ws),
        workspace: ws,
        port: 0,
        model: ModelChoice::Preset(Box::new(arm.model.clone())),
        disabled_layers: ["repo_map", "code_reading"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        qsa_selection_budget: arm.qsa_selection_budget,
        ..Default::default()
    };
    let session = Arc::new(ZendSession::new(config, LogBus::new()));
    session.start_loading();
    session.wait_ready().await;

    // The composer's default dials, and the tool prompt a chat turn is shown.
    let mut selection = dial_selection(Some(2), Some(2), None);
    apply_tools_dial(&mut selection, ToolMode::Comprehensive);

    let run = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the clock is past the epoch")
        .as_nanos();
    let mut out = Vec::with_capacity(CONVERSATIONS);
    for i in 0..CONVERSATIONS {
        let conv_id = format!("{}-{run}-{i}", arm.name);
        let started = Instant::now();
        for question in SCRIPT {
            ask(&session, &conv_id, question, &selection).await;
        }
        let history = session
            .conversation_history(&conv_id)
            .unwrap_or_else(|| panic!("{conv_id} has no history"));
        let final_turn = history
            .iter()
            .rev()
            .find(|m| matches!(m.role, TurnRole::Assistant))
            .map(|m| m.text.clone())
            .unwrap_or_else(|| panic!("{conv_id} has no assistant turn"));
        let verdict = verdict_for(
            &final_turn,
            history
                .iter()
                .filter(|m| matches!(m.role, TurnRole::User))
                .map(|m| m.text.as_str()),
        );
        println!(
            "\n===== {} conversation {} of {CONVERSATIONS}: {verdict:?} ({:.0} s)\n{final_turn}",
            arm.name,
            i + 1,
            started.elapsed().as_secs_f64()
        );
        // Retired before the next begins, so no conversation's turns are there
        // for another to retrieve.
        if let Some(Err(e)) = session.tombstone_conversation(&conv_id) {
            panic!("tombstoning {conv_id}: {e}");
        }
        out.push((final_turn, verdict));
    }
    session.shutdown().await;
    out
}

/// One user turn, run to its end — every tool round included — as the chat
/// endpoint runs it: the daemon's own sampling, reseeded every turn.
async fn ask(session: &ZendSession, conv_id: &str, question: &str, selection: &SelectionState) {
    let mut stream = session
        .submit_with_sampling(
            vec![ChatMessage::new(Role::User, question)],
            None,
            conv_id.to_string(),
            None,
            None,
            None,
            false,
            ToolMode::Comprehensive,
            None,
            selection.clone(),
        )
        .await;
    while let Some(item) = stream.next().await {
        match item.unwrap_or_else(|e| panic!("{conv_id}: {question:?}: {e}")) {
            StreamItem::Tool(status) => {
                println!("[{conv_id}] tool {} {:?}", status.phase, status.tools)
            }
            StreamItem::Status(_)
            | StreamItem::Token(_)
            | StreamItem::Projection(_)
            | StreamItem::Prefill { .. }
            | StreamItem::Think { .. }
            | StreamItem::TurnEnd { .. } => {}
        }
    }
}

/// Print the arm's tally and assert the failure the script exists for: at
/// least [`REQUIRED_READS`] conversations read the paper, and none of them
/// answered an earlier question.
///
/// A summary where a view was asked for is counted and printed but does not
/// fail the arm. It is a model's style, not a lost question: the healthy
/// control (Qwen3.6-35B-A3B, which never once answered an earlier question on
/// 2026-09-16) wrote one in four of its seven reads, each with the question
/// named in its reasoning.
fn report(arm: &Arm, verdicts: &[(String, Verdict)]) {
    let read = verdicts
        .iter()
        .filter(|(_, v)| !matches!(v, Verdict::NeverRead))
        .count();
    let answered = verdicts.iter().filter(|(_, v)| v.passed()).count();
    let earlier = verdicts
        .iter()
        .filter(|(_, v)| matches!(v, Verdict::EarlierQuestion(_)))
        .count();
    let all: Vec<&Verdict> = verdicts.iter().map(|(_, v)| v).collect();
    println!(
        "\n===== {}: {read}/{} read the paper; of those {answered} answered the last question \
         and {earlier} answered an earlier one — {all:?}",
        arm.name,
        verdicts.len()
    );
    assert!(
        read >= REQUIRED_READS && earlier == 0,
        "{}: {read}/{} read the paper and {earlier} of them answered an earlier question \
         ({all:?}); required: {REQUIRED_READS} reads and none answering an earlier question",
        arm.name,
        verdicts.len()
    );
}
