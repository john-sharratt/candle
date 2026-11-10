//! **Recall at depth, past the working set** — the direct test of the
//! Asymptotic Numerical Stability theorem's operating claim
//! (`docs/unbounded_agents.md` §11.2, "Empirical status").
//!
//! The theorem bounds the stored error a step is exposed to by a constant
//! independent of the conversation's depth, *given* a working set of fixed
//! size selected from it. The batched depth gates cannot test that: they read
//! one context whole. This drives the conversation engine itself, with its
//! working set capped well below the history, and asks a question whose answer
//! sits at the start of the history:
//!
//! - the history is the repository's own source files, one file a turn, to
//!   [`DEPTHS`] — up to twelve times the engine's [`WORKING_SET`];
//! - four vault codes of random characters are stated in four turns placed
//!   together in its middle ([`NEEDLES_AT`]), where neither a fill from the
//!   oldest turns nor a window of the newest reaches them;
//! - the last turn asks for the four codes back, decoded greedily.
//!
//! Each depth is a fresh conversation (a fork does not inherit its parent's
//! turns, so it cannot share one), under each [`Arm`]:
//!
//! - **provenance, C10** — the production path, sealed at the top level;
//! - **provenance, C0** — the same at near-lossless storage, so a difference
//!   between the two is what compression did, at each depth;
//! - **one-shot, C10** — the working set chosen once, at the start of the
//!   question turn, and never re-selected during its decode
//!   (`Sequence::set_reprojection(0, [])` after the history is sealed): the
//!   ablation §9.12 names;
//! - **rolling window** — no provenance: the engine's append path, which
//!   borrows only the most recent sealed turns.
//!
//! The prediction under test: the provenance arms' recall does not fall with
//! depth, and C10 recalls what C0 does at every depth. The rolling window is
//! the control that shows the history really is out of reach without
//! selection. Nothing here asserts the prediction — a measurement that could
//! only pass would prove nothing — it asserts that every turn ran and prints
//! the table `docs/results/` records.
//!
//! ```bash
//! cargo test -p candle-conversation --features hub --release \
//!     --test provenance_depth -- --ignored --nocapture --test-threads=1
//! ```
//! Run with the card to itself: the daemon stopped.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle::Device;
use candle_conversation::{
    models::{Model, ModelBuilder},
    ConversationEngine, ProjectionEvent, SamplingConfig, SequenceConfig, TurnEvent,
};
use tempfile::{Builder, TempDir};
use tokenizers::Tokenizer;

/// The engine's slot length: the most of the history any one step can read.
const WORKING_SET: usize = 32_768;

/// History depths, tokens: inside the working set, 4× past it, 12× past it.
const DEPTHS: [usize; 3] = [30_000, 131_072, 393_216];

/// The needles: vault and code, codes of random characters so nothing in the
/// filler or the model's priors predicts them.
const NEEDLES: [(&str, &str); 4] = [
    ("Kirkwall", "7QX4-MZ9K"),
    ("Tampere", "R2VD-8HJ3"),
    ("Broome", "N6TC-4WQP"),
    ("Galway", "B9LF-2XRE"),
];

/// Filler turns before each needle, within the needles' block: four needles
/// across eight files, ~25K tokens.
const FILES_BEFORE_NEEDLE: usize = 2;

/// Where in the history the needles' block begins, as a share of its depth.
const NEEDLES_AT: f64 = 0.45;

/// The most of one source file a filler turn carries, in characters.
const FILE_CHARS: usize = 12_000;

/// The question the last turn asks.
const QUESTION: &str = "Earlier in this conversation I told you the codes of four vaults, one \
    at a time. List all four, in the order I gave them, one per line as `Vault: CODE`, and \
    nothing else.";

/// Greedy decode, long enough for reasoning and the four-line answer.
fn builder() -> ModelBuilder {
    Model::Qwen38_FlashNext
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(2_048)
        .max_seq_len(WORKING_SET)
        .max_concurrent(2)
}

/// One way of reaching the history.
#[derive(Clone, Copy, Debug)]
struct Arm {
    label: &'static str,
    kv_level: u8,
    /// Re-select the working set during decode.
    reproject: bool,
    /// The append path: no provenance selection at all.
    rolling_window: bool,
}

const ARMS: [Arm; 4] = [
    Arm {
        label: "provenance, C10",
        kv_level: 10,
        reproject: true,
        rolling_window: false,
    },
    Arm {
        label: "provenance, C0",
        kv_level: 0,
        reproject: true,
        rolling_window: false,
    },
    Arm {
        label: "one-shot, C10",
        kv_level: 10,
        reproject: false,
        rolling_window: false,
    },
    Arm {
        label: "rolling window",
        kv_level: 10,
        reproject: true,
        rolling_window: true,
    },
];

/// The conversation config an arm runs under. The one-shot arm's history is
/// sealed under the default config like every other arm's — the reprojection
/// rule also cuts the gallery's pages — and reprojection is switched off for
/// the question alone ([`run_cell`]).
fn config(b: &ModelBuilder, arm: Arm) -> SequenceConfig {
    let mut cfg = b.conversation_config();
    cfg.kv_compression_level = Some(arm.kv_level);
    cfg.disable_reprojection = arm.rolling_window;
    cfg
}

/// The text after the reasoning block.
fn answer(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(i) => text[i + "</think>".len()..].trim(),
        None => text.trim(),
    }
}

/// How many of the four codes the answer gives, in the order they were
/// stated: the longest run of codes appearing in the answer in needle order.
///
/// The code is the needle; the vault's name is not required beside it. The
/// question's own format (`Vault: CODE`) is read literally by the model as
/// often as not, so requiring the name scored four correct codes as none.
fn recalled(text: &str) -> usize {
    let a = answer(text);
    let mut from = 0;
    let mut n = 0;
    for (_, code) in NEEDLES {
        match a[from..].find(code) {
            Some(i) => {
                from += i + code.len();
                n += 1;
            }
            None => break,
        }
    }
    n
}

/// The user turn that states one needle.
fn needle_turn(vault: &str, code: &str) -> String {
    format!("For the record: the code for the {vault} vault is {code}. Reply with just OK.")
}

/// Every `.rs` file under the workspace's crate sources, sorted, so the filler
/// is the same on every run.
fn source_files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut out = Vec::new();
    for krate in [
        "candle-core",
        "candle-nn",
        "candle-transformers",
        "candle-conversation",
    ] {
        collect(&root.join(krate).join("src"), &mut out);
    }
    out.sort();
    out
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// A filler turn: one source file, cut at a line boundary.
fn file_turn(root: &Path, path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut cut = text.len().min(FILE_CHARS);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let body = &text[..text[..cut].rfind('\n').unwrap_or(cut)];
    let name = path.strip_prefix(root).unwrap_or(path).display();
    Some(format!("Here is `{name}`:\n```rust\n{body}\n```"))
}

/// The scripted history: files to [`NEEDLES_AT`] of `depth`, the needles,
/// then files until `depth` tokens.
///
/// **The needles sit in the middle, not at the start.** A projection that
/// selects nothing by relevance fills its budget from somewhere — measured
/// here, the opening projection of a one-shot turn took the history's oldest
/// turns — and a rolling window takes its newest. Needles at either end are
/// reached by those fills without any selection having found them; in the
/// middle of a history several times the working set, only selection reaches
/// them.
fn history(tok: &Tokenizer, files: &[PathBuf], depth: usize) -> Vec<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let count = |s: &str| tok.encode(s, false).map(|e| e.len()).unwrap_or(0);
    let mut turns = Vec::new();
    let mut tokens = 0usize;
    let mut files = files.iter().filter_map(|p| file_turn(&root, p));
    let lead = (depth as f64 * NEEDLES_AT) as usize;
    for t in files.by_ref() {
        if tokens >= lead {
            break;
        }
        tokens += count(&t);
        turns.push(t);
    }
    for (vault, code) in NEEDLES {
        for t in files.by_ref().take(FILES_BEFORE_NEEDLE) {
            tokens += count(&t);
            turns.push(t);
        }
        let n = needle_turn(vault, code);
        tokens += count(&n);
        turns.push(n);
    }
    for t in files {
        if tokens >= depth {
            break;
        }
        tokens += count(&t);
        turns.push(t);
    }
    assert!(
        tokens >= depth,
        "the workspace sources ran out at {tokens} tokens, short of {depth}"
    );
    turns
}

/// The assistant's reply to a scripted turn.
fn reply(turn: &str) -> &'static str {
    if turn.starts_with("For the record") {
        "OK"
    } else {
        "Read."
    }
}

/// One cell: the history inserted, the question asked, the answer scored.
struct Cell {
    arm: &'static str,
    depth: usize,
    turns: usize,
    recalled: usize,
    answer: String,
    seconds: f64,
    /// Projection points the question's decode emitted.
    projections: usize,
    /// Needle turns any of those projections selected, raw or summarised —
    /// whether the codes reached the step through attention at all.
    needles_selected: usize,
    /// Turns the last projection held, and how many of them were summaries.
    last_turns: usize,
    last_summaries: usize,
}

/// The script positions of the needle turns.
fn needle_positions(turns: &[String]) -> Vec<u32> {
    turns
        .iter()
        .enumerate()
        .filter(|(_, t)| reply(t) == "OK")
        .map(|(i, _)| i as u32)
        .collect()
}

fn run_cell(
    eng: &ConversationEngine,
    b: &ModelBuilder,
    arm: Arm,
    depth: usize,
    turns: &[String],
) -> Cell {
    let start = Instant::now();
    let mut conv = eng
        .new_conversation(&b.format_system_prompt(), config(b, arm))
        .expect("new_conversation");
    for t in turns {
        conv.insert_turn(t.as_str(), reply(t))
            .unwrap_or_else(|e| panic!("[{}] depth {depth}: insert_turn failed: {e}", arm.label));
    }
    if !arm.reproject {
        conv.set_reprojection(0, Vec::new());
    }
    let handle = conv
        .submit_turn(QUESTION)
        .unwrap_or_else(|e| panic!("[{}] depth {depth}: the question failed: {e}", arm.label));
    let mut points: Vec<ProjectionEvent> = Vec::new();
    let mut done = None;
    for event in handle.stream() {
        match event {
            TurnEvent::Projection(p) => points.push(p),
            TurnEvent::Done(r) => done = Some(r),
            TurnEvent::Error(e) => panic!("[{}] depth {depth}: decoding failed: {e}", arm.label),
            _ => {}
        }
    }
    let response = done.expect("the question finished");
    conv.finish_turn(handle, &response).expect("finish_turn");
    conv.close().expect("close");

    let needles = needle_positions(turns);
    let mut seen: Vec<u32> = points
        .iter()
        .flat_map(|p| p.selection.turns.iter())
        .map(|t| t.index)
        .filter(|i| needles.contains(i))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    let last = points.last().map(|p| &p.selection.turns);
    let text = response.text.trim().to_string();
    let cell = Cell {
        arm: arm.label,
        depth,
        turns: turns.len(),
        recalled: recalled(&text),
        answer: answer(&text).to_string(),
        seconds: start.elapsed().as_secs_f64(),
        projections: points.len(),
        needles_selected: seen.len(),
        last_turns: last.map_or(0, |t| t.len()),
        last_summaries: last.map_or(0, |t| {
            t.iter()
                .filter(|s| format!("{:?}", s.kind) != "Normal")
                .count()
        }),
    };
    println!(
        "[{}] depth {depth} ({} turns): {}/4 recalled in {:.0} s · {} projections · needle turns \
         {needles:?} selected {seen:?} · last projection {} turns ({} summaries), indices {:?} — {:?}",
        cell.arm,
        cell.turns,
        cell.recalled,
        cell.seconds,
        cell.projections,
        cell.last_turns,
        cell.last_summaries,
        last.map(|t| t.iter().map(|s| s.index).collect::<Vec<_>>()),
        cell.answer
    );
    cell
}

/// A private workspace on the target directory's drive, removed at the end.
fn workspace() -> TempDir {
    Builder::new()
        .prefix("provenance_depth")
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .expect("create the private workspace")
}

#[test]
#[ignore = "loads Flash-Next and inserts up to 393K tokens of history per cell; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn recall_at_depth_past_the_working_set() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let b = builder();
    let tok = b
        .resolve_model(&device)
        .expect("model pack resolved")
        .tokenizer()
        .expect("tokenizer");
    let files = source_files();
    let scripts: Vec<(usize, Vec<String>)> = DEPTHS
        .iter()
        .map(|&d| (d, history(&tok, &files, d)))
        .collect();

    let ws = workspace();
    let eng = builder()
        .workspace_path(ws.path())
        .engine(&device)
        .expect("engine");

    let mut cells = Vec::new();
    for arm in ARMS {
        for (depth, turns) in &scripts {
            cells.push(run_cell(&eng, &b, arm, *depth, turns));
        }
    }

    println!("\n### Recall at depth, working set {WORKING_SET} tokens\n");
    println!(
        "| Arm | History (tok) | Turns | Recalled | Needle turns selected | Projections | \
         Last projection (turns / summaries) | Seconds |"
    );
    println!("|---|---:|---:|---:|---:|---:|---|---:|");
    for c in &cells {
        println!(
            "| {} | {} | {} | {}/4 | {}/4 | {} | {} / {} | {:.0} |",
            c.arm,
            c.depth,
            c.turns,
            c.recalled,
            c.needles_selected,
            c.projections,
            c.last_turns,
            c.last_summaries,
            c.seconds
        );
    }
    assert_eq!(cells.len(), ARMS.len() * DEPTHS.len(), "every cell ran");
}

#[cfg(test)]
mod scoring {
    use super::*;

    /// Codes count in the order stated, with or without their vaults' names.
    #[test]
    fn codes_count_in_order() {
        let literal = "<think>hmm</think>\nVault: 7QX4-MZ9K\nVault: R2VD-8HJ3\nVault: N6TC-4WQP\nVault: B9LF-2XRE";
        assert_eq!(recalled(literal), 4);
        let skipped = "Kirkwall: 7QX4-MZ9K\nBroome: N6TC-4WQP\nGalway: B9LF-2XRE";
        assert_eq!(
            recalled(skipped),
            1,
            "the run stops at the missing second code"
        );
        let swapped = "R2VD-8HJ3 then 7QX4-MZ9K";
        assert_eq!(recalled(swapped), 1, "out of order counts only the run");
    }

    /// The reasoning never scores: codes repeated inside it are not an answer.
    #[test]
    fn reasoning_does_not_score() {
        let text = "<think>Kirkwall: 7QX4-MZ9K</think>\nI do not know.";
        assert_eq!(recalled(text), 0);
    }

    /// Needle turns are answered OK and file turns Read.
    #[test]
    fn scripted_replies() {
        assert_eq!(reply(&needle_turn("Broome", "N6TC-4WQP")), "OK");
        assert_eq!(reply("Here is `x.rs`:"), "Read.");
    }
}
