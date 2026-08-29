//! Tier 3 — the hybrid lineage's recurrent state, end to end through zend.
//!
//! The last gate, deliberately. Every defect these tests can catch has a tier-1
//! or tier-2 test that fails first and localises better; what only this tier can
//! say is that a real daemon, against a real redo log, ingests and converses and
//! survives a restart. It is where "it works" is confirmed, not where anything
//! is debugged.
//!
//! ```text
//! cargo test -p zend --test hybrid_recurrent_state -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## Why the state is asserted, never just the text
//!
//! This is the whole reason the file exists. A conversation whose recurrent
//! state has silently zeroed **reads perfectly**. It is fluent, on-topic,
//! grammatical, and has simply forgotten — which no output-shaped assertion can
//! distinguish from a model that answered well. Every test below therefore
//! asserts on the state itself (via the snapshot the seal writes) and treats the
//! text as corroboration.
//!
//! That is the same failure signature as the three defects that cost this
//! lineage its original bring-up (`docs/qwen35_qwen38_models.md` §7.8): all
//! three left the model fluent, shape-correct and error-free.

use std::path::PathBuf;

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::persistence::record::SnapshotPayload;
use candle_conversation::projection;
use candle_conversation::projection::TimelineId;
use candle_conversation::{ConversationEngine, SamplingConfig, Sequence};

const PROJECTION_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// The hybrid under test. The point release shares Qwen3.5-35B's geometry —
/// 40 layers at 3:1, so 30 recurrent and 10 attention.
const HYBRID: Model = Model::Qwen36_35B_A3B_Q4;

/// Recurrent layers on this stack — what a complete snapshot must carry.
const EXPECTED_RECURRENT_LAYERS: usize = 30;

// ── helpers ──────────────────────────────────────────────────────────────────

/// The timeline's persisted recurrent snapshot, read back through the same path
/// a resume uses.
///
/// Polls, because the seal enqueues onto the off-thread writer and the record
/// lands asynchronously. A test that read once and found nothing would be
/// reporting its own timing.
fn snapshot_for(
    engine: &ConversationEngine,
    timeline: TimelineId,
    what: &str,
) -> Option<SnapshotPayload> {
    let conv = engine.conversation();
    for _ in 0..100 {
        match conv.read_recurrent_snapshot(timeline) {
            Ok(Some(p)) => return Some(p),
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
            Err(e) => panic!("{what}: snapshot indexed but unreadable: {e}"),
        }
    }
    None
}

/// True when a snapshot carries a state that is not all zeros.
///
/// The load-bearing assertion of this file. A zeroed state is exactly what a
/// conversation that has forgotten everything looks like, and it is invisible
/// from the outside.
fn state_is_non_zero(p: &SnapshotPayload) -> bool {
    p.layers
        .iter()
        .any(|l| l.state.iter().any(|&b| b != 0) || l.conv_tail.iter().any(|&b| b != 0))
}

fn build_engine(workspace: PathBuf, device: &Device) -> (ConversationEngine, Sequence) {
    let dialect = HYBRID.spec().dialect.clone();
    let workspace_str = workspace.display().to_string();
    let mut proj_builder = projection::Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_YAML,
        &[("workspace", workspace_str.as_str())],
        Some(&dialect),
    )
    .expect("parse projection.yaml");
    let dialogue_layer = proj_builder
        .id_for_layer("dialogue")
        .expect("dialogue layer");
    let primary_group = proj_builder
        .id_for_group("primary_conversation")
        .expect("primary group");

    let mut builder = HYBRID
        .builder()
        .workspace_path(workspace)
        .sampling(SamplingConfig::argmax())
        .seed(0)
        .max_response_tokens(40)
        // Suppression is best-effort here and the replies below show it: with an
        // explicit `.sampling(..)` the builder skips the model's non-thinking
        // sampling params, and its other lever puts `/no_think` in the *system
        // prompt*, which Qwen3 does not honour (the switch is read from the user
        // turn — see `conversation.rs`'s `turn_head_tokens`). So a real `<think>`
        // block in the output of these tests is the model's choice, not a defect.
        // Nothing here asserts on the text.
        .thinking(false);
    let conv_config = builder.conversation_config();
    let engine = builder.engine(device).expect("engine load");

    let tokenizer = engine.tokenizer().clone();
    proj_builder
        .tokenize_templates::<anyhow::Error, _>(|s| {
            let encoded = tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
            Ok(encoded.get_ids().to_vec())
        })
        .expect("tokenize templates");

    let formatted_prompt = builder.format_system_prompt();
    let conv = engine
        .new_conversation_with_projection(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config,
        )
        .expect("new conv");
    (engine, conv)
}

/// One complete turn: submit, wait, and seal. `send_turn` is the shape the
/// daemon uses, so the seal (and therefore the snapshot write) runs exactly as
/// it does in production.
fn say(conv: &mut Sequence, text: &str) -> String {
    let response = conv.send_turn(text).expect("send_turn");
    response.text
}

// ── P9.1 — ingest ────────────────────────────────────────────────────────────

/// **A workspace ingests and seals under the hybrid, and the snapshots land.**
///
/// Asserts three things the writer has to get right at once: that a snapshot is
/// written per sealed turn, that it carries every recurrent layer (a short
/// snapshot restores a partial state, which `import` would refuse — better to
/// catch it here than as a refusal on resume), and that the state is not zeros.
#[test]
#[ignore = "Tier 3: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB), seals several \
            turns and reads the redo log (~6 min). Run with: cargo test -p zend \
            --test hybrid_recurrent_state -- --ignored --nocapture --test-threads=1"]
fn hybrid_ingest_writes_a_complete_recurrent_snapshot_per_turn() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace = tmp.path().to_path_buf();
    let device = Device::new_cuda(0).expect("cuda");
    let (engine, mut conv) = build_engine(workspace, &device);
    let timeline = conv.timeline_id();

    for i in 0..3 {
        let reply = say(
            &mut conv,
            &format!("Turn {i}: name a colour and remember it."),
        );
        assert!(!reply.trim().is_empty(), "turn {i} produced no reply");
    }

    let last = snapshot_for(&engine, timeline, "ingest").expect(
        "no recurrent snapshot reached the redo log — this conversation cannot \
         be resumed, and nothing about the run would have said so",
    );
    assert_eq!(
        last.layers.len(),
        EXPECTED_RECURRENT_LAYERS,
        "snapshot carries {} layers, expected {EXPECTED_RECURRENT_LAYERS} — a \
         partial snapshot is refused by `import` on resume, so this would present \
         as an unresumable conversation rather than as a bad write",
        last.layers.len(),
    );
    assert!(
        state_is_non_zero(&last),
        "the sealed snapshot is all zeros: the recurrent layers carried nothing \
         through the conversation. Every reply above still read fluently."
    );
}

// ── P9.2 — continuity ────────────────────────────────────────────────────────

/// **Turns accumulate, and the state accumulates with them.**
///
/// The state at turn N must differ from the state at turn 1. Equality there is
/// the switch-gating defect — a view that starts from zero every turn — and it
/// is completely invisible in the text: the model still answers each turn from
/// its attention layers, fluently, having forgotten what the recurrent layers
/// held.
#[test]
#[ignore = "Tier 3: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and runs a \
            multi-turn conversation asserting the state advances (~8 min). Run \
            with: cargo test -p zend --test hybrid_recurrent_state -- --ignored \
            --nocapture --test-threads=1"]
fn hybrid_recurrent_state_advances_across_turns_and_is_non_zero_at_depth() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace = tmp.path().to_path_buf();
    let device = Device::new_cuda(0).expect("cuda");
    let (engine, mut conv) = build_engine(workspace, &device);
    let timeline = conv.timeline_id();

    say(
        &mut conv,
        "My favourite colour is vermilion. Acknowledge briefly.",
    );
    let after_first =
        snapshot_for(&engine, timeline, "first turn").expect("a snapshot after the first turn");

    for i in 0..8 {
        say(
            &mut conv,
            &format!("Filler turn {i}. Reply with one short sentence."),
        );
    }
    let after_many =
        snapshot_for(&engine, timeline, "later turns").expect("a snapshot after the later turns");
    assert!(
        after_many.turn_index > after_first.turn_index,
        "the snapshot did not advance past turn {} — the poll read the same \
         record twice and the comparison below would be vacuous",
        after_first.turn_index,
    );

    assert!(
        state_is_non_zero(&after_many),
        "the recurrent state is all zeros at depth — the model has been running \
         on its ten attention layers, fluently"
    );
    let same = after_first
        .layers
        .iter()
        .zip(after_many.layers.iter())
        .all(|(a, b)| a.state == b.state && a.conv_tail == b.conv_tail);
    assert!(
        !same,
        "the recurrent state at turn 9 is byte-identical to turn 1: nothing is \
         carrying it across turn boundaries. This is the defect that gates the \
         switch, and every reply in between read perfectly."
    );

    // Corroboration only — the assertions above are the test.
    let recall = say(&mut conv, "What is my favourite colour? One word.");
    eprintln!("[continuity] recall reply: {recall}");
}

// ── T6.3 — prompt branch checkpoints ─────────────────────────────────────────

/// **A new conversation starts with its system prompt in the recurrent layers,
/// not just in K/V.**
///
/// A conversation does not prefill its system prompt — it Arc-injects sealed
/// section K/V, and the wave never sees those tokens. On a recurrent stack that
/// leaves the state at zero while the attention layers hold the entire prompt:
/// the model's instructions are present to ten layers and absent from thirty.
/// P6 removes that by computing the state once per prompt branch at build time
/// and installing it before the first turn.
///
/// **Asserted on the path, not the text**, for the same reason as everything
/// else in this file: a conversation whose checkpoint failed to install has the
/// whole prompt in K/V and answers perfectly well. It just does not remember
/// the prompt in three quarters of its stack, and no output distinguishes that.
///
/// Two engines over one workspace, which separates the two halves:
/// - the **first** computes exactly one checkpoint — the branch it runs on, not
///   the tree's cross-product — and installs it;
/// - the **second** computes **zero**, because that branch is already on disk
///   under its content prefix, and still installs one.
///
/// A second engine that recomputed would mean the key does not round-trip; one
/// that installed nothing would mean the read path is broken. Both are silent.
///
/// **Exactly one, not "at least one", is the point.** The live tree has 200
/// branches (`no_think × persona × reasoning_stance × thinking_effort ×
/// response_length`), and an earlier version of this pass computed all of them
/// on every conversation open — 200 full-prompt prefills to use one. The upper
/// bound is the assertion that keeps it honest.
#[test]
#[ignore = "Tier 3: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) TWICE and runs \
            the branch-checkpoint prefill pass (~10 min). Run with: cargo test -p \
            zend --test hybrid_recurrent_state -- --ignored --nocapture \
            --test-threads=1"]
fn a_new_conversation_installs_its_prompt_branch_checkpoint() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace = tmp.path().to_path_buf();
    let device = Device::new_cuda(0).expect("cuda");

    let first_reply = {
        let (engine, mut conv) = build_engine(workspace.clone(), &device);
        let (computed, installed) = candle_conversation::branch_checkpoint_counts();
        assert_eq!(
            computed, 1,
            "a fresh workspace computed {computed} branch checkpoints; it runs on \
             one branch and should compute one. Zero means the pass did not run \
             and the recurrent layers know nothing of the system prompt; more \
             means it is walking the tree's cross-product and paying a \
             full-prompt prefill per unused branch before the first turn."
        );
        assert_eq!(
            installed, 1,
            "the checkpoint was computed but not installed — the work was done \
             and thrown away, and the conversation starts empty anyway"
        );
        let reply = say(&mut conv, "In one word, what are you?");
        drop(engine);
        reply
    };

    // The restart: same workspace, so every branch is already on disk.
    let (before_computed, before_installed) = candle_conversation::branch_checkpoint_counts();
    let (engine, mut conv) = build_engine(workspace, &device);
    let (computed, installed) = candle_conversation::branch_checkpoint_counts();
    assert_eq!(
        computed,
        before_computed,
        "the second engine recomputed {} checkpoint(s) that were already on disk — \
         the content-prefix key does not round-trip, so every start pays the full \
         pass again",
        computed - before_computed,
    );
    assert!(
        installed > before_installed,
        "the second engine installed no checkpoint despite every branch being on \
         disk — the read path is broken and the conversation starts empty"
    );

    let second_reply = say(&mut conv, "In one word, what are you?");
    eprintln!("[branch] first: {first_reply} | second: {second_reply}");
    drop(engine);
}

// ── T8.1 / P10.5 / P10.6 — what the state costs ──────────────────────────────

/// **The seal's state export, the snapshot write rate, and per-turn fork
/// traffic — measured, and each with the threshold that would make it a
/// problem.**
///
/// Three open questions in one run, because they share a workload and none of
/// them should be settled by reasoning about the numbers instead of taking
/// them:
///
/// - **T8.1 (gates P8)** — is the device→host state export visible in seal
///   latency? P8 is *"only if seal latency measures badly"*, and pinning the
///   export buffer to overlap the copy means async stream plumbing on the seal
///   path, whose failure mode is a race between the staging copy and the seal's
///   own writes: intermittently wrong resumed state, not reproducible on CPU.
///   Not a cost to pay against an unmeasured saving.
/// - **P10.5** — the F32 snapshot write rate, which decides whether snapshots
///   want bf16.
/// - **P10.6** — per-turn fork traffic. Views carry state now, so every
///   dialogue turn pays a device-to-device copy of the whole state, and the
///   ping-pong store removed the per-wave copy this used to be amortised
///   against.
///
/// Fails only on a threshold that would actually change a decision. Everything
/// else is printed: this is an instrument, and an instrument that fails on
/// noise gets muted.
#[test]
#[ignore = "Tier 3: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and seals six \
            turns to measure state cost (~6 min). Run with: cargo test -p zend \
            --test hybrid_recurrent_state -- --ignored --nocapture --test-threads=1"]
fn recurrent_state_cost_is_measured_not_assumed() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let device = Device::new_cuda(0).expect("cuda");
    // The engine must outlive the turns below; naming it `_engine` would drop it
    // at the end of the statement and take the scheduler with it.
    let (_engine, mut conv) = build_engine(tmp.path().to_path_buf(), &device);

    let (base_seals, base_us, base_bytes, base_forks) = candle_conversation::recurrent_state_cost();
    let turns = 6;
    let t0 = std::time::Instant::now();
    for i in 0..turns {
        say(
            &mut conv,
            &format!("Turn {i}: reply with one short sentence."),
        );
    }
    let wall = t0.elapsed();
    let (seals, us, bytes, forks) = candle_conversation::recurrent_state_cost();
    let (seals, us, bytes, forks) = (
        seals - base_seals,
        us - base_us,
        bytes - base_bytes,
        forks - base_forks,
    );
    assert!(seals > 0, "no turn sealed a snapshot — nothing to measure");

    let per_seal_ms = us as f64 / seals as f64 / 1000.0;
    let per_snapshot_mib = bytes as f64 / seals as f64 / (1024.0 * 1024.0);
    let wall_ms = wall.as_secs_f64() * 1000.0;
    let export_share = us as f64 / 1000.0 / wall_ms * 100.0;
    // One state is one state however it is moved, so a fork copies what a
    // snapshot encodes.
    let fork_mib = forks as f64 * per_snapshot_mib;

    eprintln!("[cost] {turns} turns in {wall_ms:.0} ms");
    eprintln!(
        "[cost] T8.1  seal export: {seals} exports, {per_seal_ms:.1} ms each \
         ({export_share:.2}% of wall)"
    );
    eprintln!(
        "[cost] P10.5 snapshot: {per_snapshot_mib:.1} MiB each, \
         {:.1} MiB/turn written",
        bytes as f64 / turns as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "[cost] P10.6 fork: {forks} state forks, ~{fork_mib:.0} MiB device-to-device \
         total (~{:.1} MiB/turn)",
        fork_mib / turns as f64
    );

    // The only threshold that changes a decision: if the synchronous export is
    // a tenth of turn latency, P8's async staging is worth its risk. Below
    // that it is not, and P8 stays closed.
    assert!(
        export_share < 10.0,
        "the seal's state export is {export_share:.1}% of turn wall time \
         ({per_seal_ms:.1} ms per seal) — P8 is no longer optional: pin the \
         export buffer and overlap the D2H"
    );
}

// ── P9.3 — restart ───────────────────────────────────────────────────────────

/// **Stop, restart, resume the timeline, continue coherently.**
///
/// The only test that exercises the resume path against a real redo log: the
/// first engine is dropped entirely, so nothing of its state survives but the
/// bytes on disk.
///
/// Asserts the resumed conversation's next snapshot is non-zero AND differs from
/// the pre-restart one. A resume that silently fell back to zeros would produce
/// a *fresh-looking* state — non-zero after a turn, but built from nothing —
/// so the differs-from check is what separates "resumed" from "restarted".
#[test]
#[ignore = "Tier 3: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) TWICE (fresh \
            engine after a simulated daemon restart), ~12 min. Run with: cargo \
            test -p zend --test hybrid_recurrent_state -- --ignored --nocapture \
            --test-threads=1"]
fn hybrid_survives_a_restart_and_resumes_its_recurrent_state() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace = tmp.path().to_path_buf();
    let device = Device::new_cuda(0).expect("cuda");

    let (timeline, pre_restart) = {
        let (engine, mut conv) = build_engine(workspace.clone(), &device);
        let timeline = conv.timeline_id();
        say(
            &mut conv,
            "Remember this: the passphrase is 'harbour lantern'. Acknowledge.",
        );
        say(&mut conv, "Also remember: my project is called Meridian.");
        let snap =
            snapshot_for(&engine, timeline, "pre-restart").expect("a snapshot before the restart");
        (timeline, snap)
        // Engine and conversation both drop here — the daemon is "stopped", and
        // the redo log is all that is left of the state.
    };
    assert!(
        state_is_non_zero(&pre_restart),
        "nothing worth resuming was written"
    );

    // Fresh engine over the same workspace — the restart.
    let (engine, base) = build_engine(workspace, &device);
    let mut resumed = base
        .fork_resuming(timeline)
        .expect("fork_resuming the timeline");
    let resumed_timeline = resumed.timeline_id();

    let reply = say(
        &mut resumed,
        "What is the passphrase? Answer with just the words.",
    );
    eprintln!("[restart] recall reply: {reply}");

    let post_restart = snapshot_for(&engine, resumed_timeline, "post-restart")
        .expect("a snapshot after the resumed turn");
    assert!(
        state_is_non_zero(&post_restart),
        "the resumed conversation's state is zeros"
    );
    let identical = pre_restart
        .layers
        .iter()
        .zip(post_restart.layers.iter())
        .all(|(a, b)| a.state == b.state);
    assert!(
        !identical,
        "the resumed turn did not advance the state at all"
    );
    assert!(
        post_restart.turn_index > pre_restart.turn_index,
        "the resumed conversation restarted its turn numbering ({} then {}) — it \
         began a new conversation rather than continuing the recovered one",
        pre_restart.turn_index,
        post_restart.turn_index,
    );

    drop(engine);
}
