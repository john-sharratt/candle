//! **The projection may change the reasoning; it must never change the answer.**
//!
//! The first version of this test demanded the two arms match token for token,
//! on the premise that a conversation short enough for every turn to be selected
//! leaves the projection nothing to drop. That premise is wrong, and the run
//! that disproved it is worth recording: with projection ON every turn emitted a
//! collapsed `<think></think>`, and with it OFF the model reasoned at length in
//! all six. Not a defect — the *design*. Projection windows every turn's
//! reasoning out of the K/V except the most recent one, so the model sees a
//! history with no visible reasoning and stops imitating it. The arms are
//! supposed to differ there.
//!
//! What may not differ is the conclusion. Whatever the model was allowed to see
//! of its own past reasoning, "2 + 2" is 4 and the access code is the one it was
//! given. So the comparison is on the ANSWER — the text after `</think>` — and
//! the reasoning is deliberately excluded from it.
//!
//! # The arms: UNSEALED against SEALED-AT-C
//!
//! The control arm (`disable_reprojection = true`) is the **prefill** path: the
//! slot is seeded once and every later turn is appended onto it, so the model
//! reads a context it forwarded token by token, in order, exactly as it was
//! generated. Nothing is sealed and re-injected mid-conversation.
//!
//! Every other arm is the **projection** path: after each turn the slot is reset
//! and rebuilt from sealed substrate K/V — the same turns, but arriving as
//! Arc-injected chunks at recomputed positions, with their index pages pushed
//! alongside. This is the real projection the daemon runs; there is no mock and
//! no stub anywhere in it.
//!
//! **The projection arms sweep the compression level, because that is what
//! sealing is allowed to do to the K/V.** Compression is paid at seal, so the
//! axis is literally unsealed against sealed-at-C0, C5 and C10 — and C10 is the
//! end that matters, since it is where a long-lived conversation lands and
//! therefore what unbounded context actually runs on. An arm that answers
//! differently at C10 is telling you the ladder is lossy enough to change a
//! conclusion, which no per-block error bound can say on its own.
//!
//! # Coherence is a FACT CHECK, not a liveness check
//!
//! Every turn in the script has a known right answer, and each arm is checked
//! against it before any arm is compared to another. That ordering matters:
//! "the two arms agree" passes when both are wrong the same way, and "the turn
//! produced text" passes on an answer that is fluent, on-topic and false —
//! which is precisely what a bad projection produces, and what was observed in
//! production when an injected `repo_map` turn taught the model to emit a tool
//! call instead of an answer.
//!
//! Everything that can go wrong with projection lives in that difference:
//! whether a rebuilt prefix is *equivalent* to the one the model actually
//! forwarded. So this is a differential test, not a quality one: one fixed
//! script, the **same engine**, both arms, and they must reach the same answer
//! on every turn. A divergence there is a projection defect: a mis-windowed K/V
//! range, a position that shifted, an index page that did not travel with the
//! chunks it describes.
//! Each of those reads as fluent text on its own, which is exactly why comparing
//! against a control is the only way to see it — and why the recall turn matters
//! as much as the comparison. A projection that loses the turn holding the fact
//! answers confidently and wrongly, which no assertion about fluency catches.
//!
//! **Why ≤ 8 turns.** `disable_reprojection` is not purely "projection off": it
//! also borrows a rolling window of the last `CODE_READ_WINDOW_TURNS` (8) sealed
//! turns. Below that bound the window is a no-op and the control really is a
//! linear append; above it the control would silently truncate and the
//! comparison would be measuring the window instead of the projection.
//!
//! Run with:
//! ```bash
//! cargo test -p candle-conversation --features hub --release \
//!     --test projection_identity -- --ignored --nocapture
//! ```
//! `--release` matters: a debug build trips a pre-existing gallery-arena
//! geometry assert on the second turn.

use candle_conversation::{
    models::{Model, ModelBuilder},
    ConversationEngine, SamplingConfig, SequenceConfig,
};
use tempfile::TempDir;

/// The stack whose projection this exercises: gated DeltaNet + QSA index, the
/// only lineage that carries per-position state a projection has to move.
/// A dense model would pass this test without touching any of it.
const TEST_MODEL: Model = Model::Qwen38_FlashNext_Q4KO;

/// Where the merged engine artifact lives.
///
/// This model is **built locally** (`prepared_from_source`), not published —
/// `qwen4exp::convert` merges the trunk and the MTP head into one GGUF, so
/// asking the hub for it returns 404. The path is deployment config, not a
/// feature switch: it selects which file to open and changes no code path.
/// Only the tokenizer still resolves from `tokenizer_repo`, exactly as the
/// daemon does — the directory holds no `tokenizer.json`, so `model_dir` would
/// point at one that is not there.
fn engine_gguf() -> std::path::PathBuf {
    std::env::var("QWEN38_FLASH_NEXT_GGUF")
        .unwrap_or_else(|_| {
            format!(
                "D:/Models/qwen38-flash-next/{}",
                TEST_MODEL.spec().model_filename
            )
        })
        .into()
}

/// The tokenizer, from the hub cache at the revision the model spec pins.
///
/// Setting `model_path` obliges us to set this too — the loader takes both or
/// neither, so that a locally built artifact can never be paired with a
/// tokenizer resolved from somewhere else. Derived from
/// `TOKENIZER_REPO`/`TOKENIZER_REV` rather than written out, so it follows the
/// pin if the pin moves; this vocabulary is not Qwen3's and pairing the wrong
/// one would decode plausible nonsense rather than fail.
fn tokenizer_json() -> std::path::PathBuf {
    use candle_transformers::models::quantized_qwen38_moe::{TOKENIZER_REPO, TOKENIZER_REV};
    if let Ok(p) = std::env::var("QWEN38_FLASH_NEXT_TOKENIZER") {
        return p.into();
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("no home directory to resolve the hub cache from");
    std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", TOKENIZER_REPO.replace('/', "--")))
        .join("snapshots")
        .join(TOKENIZER_REV)
        .join("tokenizer.json")
}

/// Argmax and a fixed seed: the comparison is token-for-token, so any sampling
/// entropy would make a passing run meaningless and a failing one unreadable.
fn test_builder() -> ModelBuilder {
    TEST_MODEL
        .builder()
        .model_path(engine_gguf())
        .tokenizer_path(tokenizer_json())
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(96)
        .max_concurrent(4)
}

/// Seal mismatches seen since the process started.
///
/// **The output agreeing is not the same as the engine being right.** The first
/// version of this test asserted only on answers, and passed a run that
/// mis-sealed ten turns out of twelve: a short conversation never re-reads its
/// own sealed pages, so it answers correctly over an index that does not
/// describe its K/V. The damage lands on a LATER conversation that projects
/// those turns as history and selects against pages claiming tokens the K/V does
/// not hold — silent degradation, which is the failure this whole design exists
/// to prevent. So the invariant is asserted directly, at the point the engine
/// notices it.
static SEAL_MISMATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Glue islands the engine had to GAP-FILL, which for this lineage must be zero.
///
/// **A DeltaNet stack does not re-emit inter-turn glue; it reuses it.** Each
/// sealed turn bakes its own boundary markers into its grid, so a projection of
/// system prompt + sealed turns + a live user message abuts them directly and
/// reserves no gap chunk at all. That property is asserted at the assembler
/// level by `a_projection_of_sealed_turns_and_a_live_message_emits_no_glue`;
/// this is the end-to-end half, because the assembler deciding not to emit an
/// island and the engine not reserving one are different claims.
///
/// It matters beyond efficiency: an island is a hole in the middle of the
/// sequence, and a recurrence cannot compute K/V for tokens inserted there —
/// `reserve_glue_island` refuses outright. So a reserved island on this model is
/// not a slow path, it is a failed projection.
static GLUE_ISLANDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Counts engine warnings by message text: seal mismatches and glue reservations.
struct SealWatch;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for SealWatch {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        #[derive(Default)]
        struct Find {
            seal: bool,
            glue: bool,
        }
        impl tracing::field::Visit for Find {
            fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
                if f.name() != "message" {
                    return;
                }
                let m = format!("{v:?}");
                self.seal |= m.contains("index pages cover");
                // Both the assembler's own refusal and the miss that precedes a
                // reservation, so a reserved island cannot pass unseen whichever
                // way it surfaces.
                self.glue |= m.contains("glue island")
                    || m.contains("cannot \\\n             gap-fill")
                    || m.contains("gap-fill");
            }
        }
        let mut find = Find::default();
        event.record(&mut find);
        if find.seal {
            SEAL_MISMATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if find.glue {
            GLUE_ISLANDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn init_tracing() {
    use std::sync::Once;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::registry()
            // A global level, not a per-layer one: the seal's mismatch is a
            // WARN, so it clears this and reaches `SealWatch` either way.
            .with(tracing_subscriber::filter::LevelFilter::INFO)
            .with(tracing_subscriber::fmt::layer().with_test_writer())
            .with(SealWatch)
            .try_init();
    });
}

/// A private working directory, so the run is hermetic.
///
/// `cargo test -p candle-conversation` runs with the CRATE directory as its
/// cwd, which holds a `.substrate` left by other suites — and a substrate is
/// bound to the tokenizer that sealed it, so inheriting one belonging to a
/// different model aborts the load outright. That refusal is correct (adopting
/// another vocabulary would silently change what every recorded turn says); the
/// bug is the test reaching for a substrate it does not own. Each run gets its
/// own, and takes the projection through a real substrate rather than none.
///
/// A `TempDir`, so the substrate it seals is removed when the test ends. A
/// named directory per run was never removed, and left a whole substrate in
/// the temp folder every time the test ran.
fn private_workspace() -> TempDir {
    tempfile::tempdir().expect("create private workspace")
}

fn engine(workspace: &std::path::Path) -> ConversationEngine {
    init_tracing();
    let device =
        candle::Device::cuda_if_available(0).expect("CUDA device required for integration tests");
    eprintln!("\n=== Loading {TEST_MODEL} ===");
    let start = std::time::Instant::now();
    let e = test_builder()
        .workspace_path(workspace)
        .engine(&device)
        .expect("failed to load model");
    eprintln!("   Loaded in {:.2}s\n", start.elapsed().as_secs_f64());
    e
}

/// **A fact planted early and recalled late, with filler between — and every
/// turn has a KNOWN RIGHT ANSWER.**
///
/// The recall turn is what makes this a coherence test rather than only a
/// determinism one: a projection that drops or mis-positions the turn holding
/// the fact still answers fluently, just wrongly, so the transcript comparison
/// alone would not catch a *plausible* divergence. Six turns keeps the whole
/// script inside the control's 8-turn window.
///
/// **Each turn carries what a right answer looks like**, and that is the
/// difference between this and a liveness check. "The turn produced non-empty
/// text after its reasoning" passes on an answer that is fluent, on-topic and
/// wrong — which is exactly the failure a bad projection produces, and exactly
/// what was observed in production when an injected `repo_map` turn taught the
/// model to emit a tool call instead of an answer. Every question here has one
/// defensible answer that does not depend on the model's taste, so the
/// predicate is a fact check rather than a style check.
///
/// The alternatives on turn 2 are the three primary colours: the question
/// genuinely admits any of them, and pinning one would be asserting the
/// checkpoint's preference rather than its coherence.
struct Turn {
    prompt: &'static str,
    /// The answer is right if it contains ANY of these, case-insensitively.
    accept: &'static [&'static str],
}

const SCRIPT: &[Turn] = &[
    Turn {
        prompt: "Remember this: the access code is 4471. Reply with just OK.",
        accept: &["ok"],
    },
    Turn {
        prompt: "What is 2 + 2? Answer with the number only.",
        accept: &["4"],
    },
    Turn {
        prompt: "Name one primary colour. One word.",
        accept: &["red", "blue", "yellow"],
    },
    Turn {
        prompt: "What is the capital of France? One word.",
        accept: &["paris"],
    },
    Turn {
        prompt: "Count from 1 to 3, comma separated.",
        accept: &["1, 2, 3", "1,2,3"],
    },
    Turn {
        prompt: "What was the access code I gave you? Digits only.",
        accept: &[SECRET],
    },
];

/// The planted fact, as it must survive to the recall turn.
const SECRET: &str = "4471";

/// One configuration of the engine's context path, and what makes it different
/// from the control.
///
/// **The axis is where the model's history comes from.** `Prefill` never seals
/// mid-conversation: the slot is seeded once and every turn appends onto K/V the
/// model forwarded itself. Every other arm resets the slot each turn and rebuilds
/// it from *sealed* substrate K/V — and the compression level is the state that
/// sealing is allowed to put it in. So the sweep is literally unsealed against
/// sealed-at-C, with the same script, the same engine, and greedy decoding.
///
/// C10 is the interesting end: it is the level whose per-read cost the ladder
/// measures and the one a long-lived conversation actually lands on, so "the
/// answer survives C10" is the claim that matters for unbounded context.
#[derive(Clone, Copy)]
struct Arm {
    label: &'static str,
    /// `true` ⇒ the prefill control (append-only, nothing sealed and re-injected).
    disable_reprojection: bool,
    /// Level the conversation's sealed turns quantize at. `None` ⇒ the
    /// engine-wide default.
    kv_level: Option<u8>,
}

const ARMS: &[Arm] = &[
    Arm {
        label: "prefill (control, unsealed)",
        disable_reprojection: true,
        kv_level: None,
    },
    Arm {
        label: "projection, engine default",
        disable_reprojection: false,
        kv_level: None,
    },
    Arm {
        label: "projection, sealed C0",
        disable_reprojection: false,
        kv_level: Some(0),
    },
    Arm {
        label: "projection, sealed C5",
        disable_reprojection: false,
        kv_level: Some(5),
    },
    Arm {
        label: "projection, sealed C10",
        disable_reprojection: false,
        kv_level: Some(10),
    },
];

/// The turn's answer: everything after the reasoning block.
///
/// The two arms legitimately reason differently (see the module docs), so a
/// comparison that included the think block would fail on the design rather
/// than on a defect. A turn with no block is all answer.
fn answer(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(i) => text[i + "</think>".len()..].trim(),
        None => text.trim(),
    }
}

/// What one arm produced, with the engine's own complaints attributed to it.
struct Run {
    arm: Arm,
    transcript: Vec<String>,
    /// Seal mismatches and glue islands raised WHILE THIS ARM RAN.
    ///
    /// The counters are process-wide, so they are snapshotted either side of the
    /// arm rather than read at the end: with five arms in one process a single
    /// total would say a projection is broken without saying which one.
    seal_mismatches: usize,
    glue_islands: usize,
}

fn run_script(eng: &ConversationEngine, arm: Arm) -> Run {
    use std::sync::atomic::Ordering::Relaxed;
    let seal_before = SEAL_MISMATCHES.load(Relaxed);
    let glue_before = GLUE_ISLANDS.load(Relaxed);

    let mut cfg: SequenceConfig = test_builder().conversation_config();
    cfg.disable_reprojection = arm.disable_reprojection;
    cfg.kv_compression_level = arm.kv_level;
    let mut conv = eng
        .new_conversation(&test_builder().format_system_prompt(), cfg)
        .expect("new_conversation failed");

    let mut transcript = Vec::with_capacity(SCRIPT.len());
    for (i, turn) in SCRIPT.iter().enumerate() {
        let response = conv
            .send_turn(turn.prompt)
            .unwrap_or_else(|e| panic!("[{}] turn {i} failed: {e}", arm.label));
        eprintln!("[{}] turn {i}: {:?}", arm.label, response.text.trim());
        transcript.push(response.text.trim().to_string());
    }
    conv.close().expect("close failed");

    Run {
        arm,
        transcript,
        seal_mismatches: SEAL_MISMATCHES.load(Relaxed) - seal_before,
        glue_islands: GLUE_ISLANDS.load(Relaxed) - glue_before,
    }
}

/// Whether `text` answers the turn it was given.
fn answers_correctly(turn: &Turn, text: &str) -> bool {
    let a = answer(text).to_lowercase();
    turn.accept.iter().any(|w| a.contains(&w.to_lowercase()))
}

/// Every turn answers, every answer is RIGHT, and the last one still holds the
/// fact from the first.
///
/// Run against every arm, because a projection that loses the fact and a slot
/// that never had it are different faults with the same symptom.
///
/// Returns the per-turn verdicts so the caller can table them; panics only at
/// the end, so one run reports every wrong turn rather than the first.
fn assert_coherent(run: &Run) -> Vec<bool> {
    let arm = run.arm.label;
    let mut ok = Vec::with_capacity(SCRIPT.len());
    let mut wrong: Vec<String> = Vec::new();
    for (i, (turn, text)) in SCRIPT.iter().zip(&run.transcript).enumerate() {
        assert!(
            !answer(text).is_empty(),
            "[{arm}] turn {i} answered nothing — it produced {text:?}, which is \
             reasoning with no conclusion after it"
        );
        let good = answers_correctly(turn, text);
        ok.push(good);
        if !good {
            wrong.push(format!(
                "  turn {i}: asked {:?}\n    answered {:?}\n    expected one of {:?}",
                turn.prompt,
                answer(text),
                turn.accept
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "[{arm}] {} of {} turns answered the question WRONGLY. Fluent, on-topic \
         and wrong is what a bad projection produces — the context it rebuilt \
         reads perfectly and says something else — so this is checked per turn \
         against a known answer rather than by asking whether the model said \
         anything at all.\n{}",
        wrong.len(),
        SCRIPT.len(),
        wrong.join("\n"),
    );
    let recall = answer(run.transcript.last().expect("script is not empty"));
    assert!(
        recall.contains(SECRET),
        "[{arm}] the recall turn lost the fact planted six turns earlier — \
         answered {recall:?}, expected it to contain {SECRET:?}. The context \
         reaching the model no longer holds the turn that carried it."
    );
    ok
}

/// Characters of reasoning the turn produced, before its answer.
///
/// **Length, not presence**, and the difference is the whole point of the
/// column. Projection windows every older turn's reasoning out of the K/V, so
/// the model sees a history that appears not to reason and imitates it less —
/// the arms are SUPPOSED to differ here, and reporting it is what stops a reader
/// mistaking that difference for the defect. Asking merely whether a `</think>`
/// is present cannot show it: a collapsed `<think></think>` has one too, so the
/// question "did it reason" answers yes for a turn that did not.
fn reasoning_len(text: &str) -> usize {
    match (text.find("<think>"), text.rfind("</think>")) {
        (Some(o), Some(c)) if c >= o => text[o + "<think>".len()..c].trim().len(),
        _ => 0,
    }
}

#[test]
#[ignore]
fn projection_is_the_identity_when_nothing_is_dropped() {
    let workspace = private_workspace();
    let eng = engine(workspace.path());

    // Every arm on ONE engine, in order, control first. The control is the
    // prefill path (append-only, nothing sealed mid-conversation); every other
    // arm rebuilds from sealed K/V, at the compression level named.
    let runs: Vec<Run> = ARMS.iter().map(|&a| run_script(&eng, a)).collect();
    let control = &runs[0];

    // ── Results, before any assertion ────────────────────────────────────────
    // Printed first and unconditionally: a failing assertion below aborts the
    // test, and the table is the thing worth having when it does.
    eprintln!(
        "\n=== projection identity: {} arms × {} turns ===",
        runs.len(),
        SCRIPT.len()
    );
    eprintln!(
        "{:<28} {:>6} {:>7} {:>10} {:>8} {:>6}  answers",
        "arm", "right", "vs ctrl", "reasoning", "mismatch", "glue"
    );
    let mut verdicts: Vec<Vec<bool>> = Vec::new();
    for run in &runs {
        let right = SCRIPT
            .iter()
            .zip(&run.transcript)
            .filter(|(t, x)| answers_correctly(t, x))
            .count();
        let same = control
            .transcript
            .iter()
            .zip(&run.transcript)
            .filter(|(a, b)| answer(a) == answer(b))
            .count();
        let think: usize = run.transcript.iter().map(|t| reasoning_len(t)).sum();
        let answers: Vec<String> = run
            .transcript
            .iter()
            .map(|t| answer(t).to_string())
            .collect();
        eprintln!(
            "{:<28} {:>4}/{} {:>5}/{} {:>10} {:>8} {:>6}  {:?}",
            run.arm.label,
            right,
            SCRIPT.len(),
            same,
            SCRIPT.len(),
            format!("{think} ch"),
            run.seal_mismatches,
            run.glue_islands,
            answers,
        );
        verdicts.push(Vec::new());
    }
    eprintln!();

    // ── Coherence, per arm ───────────────────────────────────────────────────
    // Every arm has to stand on its own before any of them is compared: an arm
    // that answers wrongly is broken whether or not the control agrees with it,
    // and two arms agreeing on a wrong answer is the one case a pure
    // differential test cannot see.
    for (i, run) in runs.iter().enumerate() {
        verdicts[i] = assert_coherent(run);
    }

    // ── The engine's own invariants, attributed to the arm that broke them ───
    for run in &runs {
        // **Glue is reused, never re-emitted.** Each sealed turn bakes its own
        // boundary markers into its grid, so a projection of system prompt +
        // sealed turns + a live message abuts them directly and reserves no gap.
        // An island is a hole mid-sequence that a recurrence cannot compute K/V
        // for — `reserve_glue_island` refuses it — so a non-zero count here is a
        // failed projection on this lineage, not a slower one.
        assert_eq!(
            run.glue_islands, 0,
            "[{}] the engine reserved or refused {} glue island(s). Search the log \
             for `glue island`.",
            run.arm.label, run.glue_islands,
        );
        // **Every turn's sealed index must describe its own K/V, exactly.**
        // A run can agree on every answer and still mis-seal: a short
        // conversation never re-reads its own pages, so it answers correctly
        // over an index that does not describe its K/V, and the cost lands on
        // whatever projects those turns later. Ten of twelve turns were wrong
        // this way in the run that first showed it, with every answer right.
        assert_eq!(
            run.seal_mismatches, 0,
            "[{}] {} turn(s) sealed an index that does not cover the turn. Search \
             the log for `index pages cover`: each line carries the turn's token \
             count, what its pages covered, and the page widths.",
            run.arm.label, run.seal_mismatches,
        );
    }

    // ── The differential ─────────────────────────────────────────────────────
    // Turn by turn against the control, so a failure names WHERE and IN WHICH
    // ARM the paths parted rather than reporting that two transcripts differ
    // somewhere.
    for run in &runs[1..] {
        for (i, (a, b)) in control.transcript.iter().zip(&run.transcript).enumerate() {
            assert_eq!(
                answer(a),
                answer(b),
                "turn {i}'s ANSWER diverged from the control.\n  \
                 {:<28}: {:?}\n  \
                 {:<28}: {:?}\n\
                 (full texts: {a:?} / {b:?})\n\
                 The arms are allowed to reason differently — projection windows \
                 older turns' reasoning out, so the model imitates less of it — \
                 but not to conclude differently. A difference here means the \
                 rebuilt prefix is not equivalent to the one the append path \
                 built: a mis-windowed K/V range, a shifted position, an index \
                 page that did not travel with the chunks it describes, or a \
                 compression level that is lossy enough to change the answer.",
                control.arm.label,
                answer(a),
                run.arm.label,
                answer(b),
            );
        }
    }
}
