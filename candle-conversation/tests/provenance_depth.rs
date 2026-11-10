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
use std::sync::Arc;
use std::time::Instant;

use candle::Device;
use candle_conversation::decode_diag::{self, Arm as DiagCapture, Control};
use candle_conversation::projection::{Builder as Projection, TimelineId};
use candle_conversation::{
    models::{Model, ModelBuilder},
    ConversationEngine, ProjectionEvent, SamplingConfig, Sequence, SequenceConfig, TurnEvent,
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
fn history(tok: &Tokenizer, files: &[PathBuf], depth: usize, needles_at: f64) -> Vec<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let count = |s: &str| tok.encode(s, false).map(|e| e.len()).unwrap_or(0);
    let mut turns = Vec::new();
    let mut tokens = 0usize;
    let mut files = files.iter().filter_map(|p| file_turn(&root, p));
    let lead = (depth as f64 * needles_at) as usize;
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
        .map(|&d| (d, history(&tok, &files, d, NEEDLES_AT)))
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

/// One diagnosis arm: how the question's decode attends, and where the
/// needles sit in the history.
struct DiagArm {
    label: &'static str,
    control: Control,
    needles_at: f64,
}

const DIAG_DEPTH: usize = 131_072;

const DIAG_ARMS: [DiagArm; 6] = [
    DiagArm {
        label: "sparse (production)",
        control: Control::Sparse,
        needles_at: NEEDLES_AT,
    },
    DiagArm {
        label: "dense (every cell attended)",
        control: Control::Dense,
        needles_at: NEEDLES_AT,
    },
    DiagArm {
        label: "fresh (every piece re-placed)",
        control: Control::Fresh,
        needles_at: NEEDLES_AT,
    },
    DiagArm {
        label: "needles last, sparse",
        control: Control::Sparse,
        needles_at: 1.0,
    },
    DiagArm {
        label: "needles first, sparse",
        control: Control::Sparse,
        needles_at: 0.0,
    },
    DiagArm {
        label: "needles first, dense",
        control: Control::Dense,
        needles_at: 0.0,
    },
];

/// What one diagnosed question read, reduced to the needle pieces.
struct Diagnosis {
    label: &'static str,
    recalled: usize,
    answer: String,
    /// Needle turns in the last projection.
    projected: usize,
    /// The script positions of every turn the last projection held, in the
    /// order it placed them.
    selected: Vec<u32>,
    /// The needle pieces' slot positions, and the slot's end.
    needle_spans: Vec<(usize, usize)>,
    slot_end: usize,
    /// Attention-layer queries recorded, and how many read a needle piece.
    queries: usize,
    reading: usize,
    /// Cells taken from needle pieces, over every recorded query.
    needle_cells: usize,
    /// Cells taken over every recorded query.
    all_cells: usize,
    /// The best rank any needle block reached in any query.
    best_rank: Option<usize>,
}

fn diagnose(
    eng: &ConversationEngine,
    b: &ModelBuilder,
    arm: &DiagArm,
    turns: &[String],
) -> Diagnosis {
    let mut cfg = b.conversation_config();
    cfg.kv_compression_level = Some(10);
    let mut conv = eng
        .new_conversation(&b.format_system_prompt(), cfg)
        .expect("new_conversation");
    for t in turns {
        conv.insert_turn(t.as_str(), reply(t))
            .unwrap_or_else(|e| panic!("[{}] insert_turn failed: {e}", arm.label));
    }
    let timeline = conv.timeline_id().raw();
    assert!(
        decode_diag::arm(DiagCapture {
            control: arm.control,
            timeline,
            from: 0,
            live_at: usize::MAX,
        }),
        "another diagnostic decode holds the capture"
    );
    let handle = conv.submit_turn(QUESTION).expect("submit the question");
    let mut points: Vec<ProjectionEvent> = Vec::new();
    let mut done = None;
    for event in handle.stream() {
        match event {
            TurnEvent::Projection(p) => points.push(p),
            TurnEvent::Done(r) => done = Some(r),
            TurnEvent::Error(e) => panic!("[{}] decoding failed: {e}", arm.label),
            _ => {}
        }
    }
    let capture = decode_diag::disarm();
    let response = done.expect("the question finished");
    conv.finish_turn(handle, &response).expect("finish_turn");
    conv.close().expect("close");

    let needles = needle_positions(turns);
    let labels: Vec<String> = needles
        .iter()
        .map(|i| format!("turn:{timeline}:{i}"))
        .collect();
    let projected = points.last().map_or(0, |p| {
        p.selection
            .turns
            .iter()
            .filter(|t| needles.contains(&t.index))
            .count()
    });
    let needle_spans = capture
        .pieces
        .iter()
        .filter(|p| labels.contains(&p.label))
        .map(|p| (p.start, p.end))
        .collect();
    let slot_end = capture.pieces.iter().map(|p| p.end).max().unwrap_or(0);
    let mut d = Diagnosis {
        label: arm.label,
        recalled: recalled(&response.text),
        answer: answer(&response.text).to_string(),
        projected,
        selected: points.last().map_or_else(Vec::new, |p| {
            p.selection
                .turns
                .iter()
                .map(|t| t.index)
                .filter(|&i| (i as usize) < turns.len())
                .collect()
        }),
        needle_spans,
        slot_end,
        queries: capture.layers.len(),
        reading: 0,
        needle_cells: 0,
        all_cells: 0,
        best_rank: None,
    };
    for r in &capture.layers {
        d.all_cells += r.attended_cells;
        let hit: Vec<_> = r
            .pieces
            .iter()
            .filter(|p| labels.contains(&p.label))
            .collect();
        let cells: usize = hit.iter().map(|p| p.attended_cells).sum();
        if cells > 0 {
            d.reading += 1;
        }
        d.needle_cells += cells;
        for p in hit {
            d.best_rank = Some(d.best_rank.map_or(p.best_rank, |b| b.min(p.best_rank)));
        }
    }
    println!(
        "[{}] {}/4 recalled · needle turns in the last projection {}/4 · needle pieces at {:?} \
         of a {}-position slot · {} of {} layer queries read a needle piece · {} of {} cells \
         · best needle rank {:?} — {:?}",
        d.label,
        d.recalled,
        d.projected,
        d.needle_spans,
        d.slot_end,
        d.reading,
        d.queries,
        d.needle_cells,
        d.all_cells,
        d.best_rank,
        d.answer
    );
    println!(
        "[{}] the whole reply, reasoning included:\n{}\n[{}] end of reply",
        d.label,
        response.text.trim(),
        d.label
    );
    d
}

/// **Why recall fails past the working set**, at the failing cell (131K of
/// history, C10, the engine's 32K slot): the question's decode under the
/// capped diagnostic capture (`decode_diag`), which records for every
/// attention layer's query which pieces of the slot QSA's index chose.
///
/// - **sparse** is the production decode;
/// - **dense** attends every cell, so QSA's selection cannot be what drops the
///   needles;
/// - **fresh** places every piece anew at each reprojection, so nothing is
///   spliced;
/// - **needles last** states the same needles just before the question, so
///   they are near the decode cursor in the slot and in the recurrence alike.
#[test]
#[ignore = "loads Flash-Next and inserts 131K tokens of history per arm; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn why_recall_fails_past_the_working_set() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let b = builder();
    let tok = b
        .resolve_model(&device)
        .expect("model pack resolved")
        .tokenizer()
        .expect("tokenizer");
    let files = source_files();
    let ws = workspace();
    let eng = builder()
        .workspace_path(ws.path())
        .engine(&device)
        .expect("engine");
    let rows: Vec<Diagnosis> = DIAG_ARMS
        .iter()
        .map(|arm| {
            diagnose(
                &eng,
                &b,
                arm,
                &history(&tok, &files, DIAG_DEPTH, arm.needles_at),
            )
        })
        .collect();

    println!("\n### Why recall fails past the working set — {DIAG_DEPTH} tokens, C10\n");
    println!(
        "| Arm | Recalled | Needles projected | Needle pieces (slot positions) | Slot end | \
         Queries reading a needle | Needle cells / all | Best needle rank |"
    );
    println!("|---|---:|---:|---|---:|---:|---:|---:|");
    for d in &rows {
        println!(
            "| {} | {}/4 | {}/4 | {:?} | {} | {}/{} | {}/{} | {:?} |",
            d.label,
            d.recalled,
            d.projected,
            d.needle_spans,
            d.slot_end,
            d.reading,
            d.queries,
            d.needle_cells,
            d.all_cells,
            d.best_rank
        );
    }
    assert_eq!(rows.len(), DIAG_ARMS.len(), "every arm ran");
}

/// **The fully-forwarded control** for the failing cell: the turns the
/// production projection selected, in the order it placed them, inserted
/// alone into a fresh conversation, which then holds them whole inside the
/// working set — every turn's K/V computed in this slot's own context rather
/// than borrowed from the one it was first sealed in.
///
/// If the forwarded selection recalls what the projected one does not, the
/// cause is the borrowed K/V: a sealed turn carries the hidden states of the
/// context it was computed in. If it fails as well, the selection's
/// arrangement itself is what the model cannot read.
#[test]
#[ignore = "loads Flash-Next and inserts 131K tokens of history; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn recall_with_the_selection_forwarded() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let b = builder();
    let tok = b
        .resolve_model(&device)
        .expect("model pack resolved")
        .tokenizer()
        .expect("tokenizer");
    let files = source_files();
    let ws = workspace();
    let eng = builder()
        .workspace_path(ws.path())
        .engine(&device)
        .expect("engine");
    let script = history(&tok, &files, DIAG_DEPTH, NEEDLES_AT);
    let projected = diagnose(&eng, &b, &DIAG_ARMS[0], &script);
    let forwarded_turns: Vec<String> = projected
        .selected
        .iter()
        .map(|&i| script[i as usize].clone())
        .collect();
    let arm = DiagArm {
        label: "the same selection, forwarded",
        control: Control::Sparse,
        needles_at: NEEDLES_AT,
    };
    let forwarded = diagnose(&eng, &b, &arm, &forwarded_turns);
    println!(
        "\n### The fully-forwarded control — {DIAG_DEPTH} tokens, C10\n\n\
         | Arm | Turns | Recalled | Needles projected |\n|---|---:|---:|---:|\n\
         | projected from the history | {} | {}/4 | {}/4 |\n\
         | the same selection, forwarded | {} | {}/4 | {}/4 |\n\
         selection {:?}",
        script.len(),
        projected.recalled,
        projected.projected,
        forwarded_turns.len(),
        forwarded.recalled,
        forwarded.projected,
        projected.selected
    );
    assert!(
        !projected.selected.is_empty(),
        "the projected run recorded no selection to forward"
    );
}

/// Where in the 131K history the needles are sealed, for
/// [`recall_against_sealing_depth`].
const SEALING_DEPTHS: [f64; 6] = [0.0, 0.1, 0.2, 0.3, 0.45, 0.7];

/// **Recall against the depth the needles were sealed at**, everything else
/// held: 131K of history, C10, the production decode, the question last.
///
/// A projected question decodes through the recurrent state left by the whole
/// history whichever arm it is, so the question side does not vary here; what
/// varies is the recurrent state each needle turn was sealed through. A sealed
/// turn's attention K/V is computed from a residual stream that passed the
/// DeltaNet layers at its sealing, with the recurrence as it stood then, and
/// each insert's attention context is capped at the working set. If recall
/// falls with the depth of the stream behind the needles, the sealed K/V is
/// conditioned by the recurrence it was sealed through.
#[test]
#[ignore = "loads Flash-Next and inserts 131K tokens of history per depth; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn recall_against_sealing_depth() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let b = builder();
    let tok = b
        .resolve_model(&device)
        .expect("model pack resolved")
        .tokenizer()
        .expect("tokenizer");
    let files = source_files();
    let ws = workspace();
    let eng = builder()
        .workspace_path(ws.path())
        .engine(&device)
        .expect("engine");
    let count = |s: &str| tok.encode(s, false).map(|e| e.len()).unwrap_or(0);
    let mut rows = Vec::new();
    for at in SEALING_DEPTHS {
        let script = history(&tok, &files, DIAG_DEPTH, at);
        let first = needle_positions(&script)[0] as usize;
        let behind: usize = script[..first].iter().map(|t| count(t)).sum();
        let arm = DiagArm {
            label: "sealing depth",
            control: Control::Sparse,
            needles_at: at,
        };
        let d = diagnose(&eng, &b, &arm, &script);
        println!(
            "[sealed at {at:.2}] {behind} tokens of stream behind the first needle · {}/4 · \
             {}/4 projected",
            d.recalled, d.projected
        );
        rows.push((at, behind, d));
    }
    println!("\n### Recall against sealing depth — {DIAG_DEPTH} tokens, C10\n");
    println!(
        "| Needles at | Stream behind the first needle (tok) | Recalled | Needles projected | \
         Queries reading a needle | Answer |"
    );
    println!("|---:|---:|---:|---:|---:|---|");
    for (at, behind, d) in &rows {
        println!(
            "| {at:.2} | {behind} | {}/4 | {}/4 | {}/{} | {} |",
            d.recalled,
            d.projected,
            d.reading,
            d.queries,
            d.answer.replace('\n', " / ")
        );
    }
    assert_eq!(rows.len(), SEALING_DEPTHS.len(), "every depth ran");
}

/// A wider working set for [`recall_with_a_wider_working_set`].
const WIDE_WORKING_SET: usize = 65_536;

/// **The sealing-depth cliff, with the working set moved.** At a 32K slot,
/// recall holds for needles sealed with 18K of stream behind them and is gone
/// at 31K and beyond — the boundary at which a conversation's own turns start
/// to be prefilled against a projected slot rather than against its contiguous
/// history. With the slot at 64K, needles sealed 44K deep were prefilled
/// against the contiguous history again: if recall returns there, and fails
/// for needles sealed past 64K, the cause is sealing under a projection, not
/// the depth of the stream.
#[test]
#[ignore = "loads Flash-Next at a 64K slot and inserts 131K tokens of history per cell; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn recall_with_a_wider_working_set() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let b = builder().max_seq_len(WIDE_WORKING_SET);
    let tok = b
        .resolve_model(&device)
        .expect("model pack resolved")
        .tokenizer()
        .expect("tokenizer");
    let files = source_files();
    let ws = workspace();
    let eng = builder()
        .max_seq_len(WIDE_WORKING_SET)
        .workspace_path(ws.path())
        .engine(&device)
        .expect("engine");
    let count = |s: &str| tok.encode(s, false).map(|e| e.len()).unwrap_or(0);
    let mut rows = Vec::new();
    for at in [0.3, 0.6] {
        let script = history(&tok, &files, DIAG_DEPTH, at);
        let first = needle_positions(&script)[0] as usize;
        let behind: usize = script[..first].iter().map(|t| count(t)).sum();
        let arm = DiagArm {
            label: "64K working set",
            control: Control::Sparse,
            needles_at: at,
        };
        let d = diagnose(&eng, &b, &arm, &script);
        println!(
            "[64K slot, sealed at {at:.2}] {behind} tokens of stream behind the first needle · \
             {}/4 · {}/4 projected",
            d.recalled, d.projected
        );
        rows.push((at, behind, d.recalled, d.projected));
    }
    assert_eq!(rows.len(), 2, "both cells ran");
}

/// A dialogue window wide enough that a 131K conversation is never projected.
const WHOLE_WINDOW: usize = 262_144;

/// The plain-prompt projection [`ConversationEngine::new_conversation`] builds,
/// with its dialogue window set to `window`.
fn plain_projection(
    eng: &ConversationEngine,
    cfg: &SequenceConfig,
    prompt: &str,
    window: usize,
) -> Projection {
    let inner = prompt
        .strip_prefix(cfg.dialect.system_start)
        .unwrap_or(prompt);
    let inner = inner.strip_suffix(cfg.dialect.system_end).unwrap_or(inner);
    let frame = eng.plain_prompt_section(inner).expect("the frame section");
    let mut builder = Projection::for_plain_prompt(inner, frame);
    assert!(
        builder.set_layer_window("dialogue", window),
        "a plain prompt has a dialogue layer"
    );
    builder
}

/// **Sealing under projection, against reading under projection.** At 131K
/// of history with the needles sealed 65K deep, the dialogue window is set to
/// 262K (the conversation is never projected: each turn is prefilled against
/// its contiguous history, and read whole) or to the production 32K, once for
/// the inserts and again for the question.
#[test]
#[ignore = "loads Flash-Next and inserts 131K tokens of history per arm; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn sealing_or_reading_under_projection() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let b = builder();
    let tok = b
        .resolve_model(&device)
        .expect("model pack resolved")
        .tokenizer()
        .expect("tokenizer");
    let files = source_files();
    let ws = workspace();
    let eng = builder()
        .workspace_path(ws.path())
        .engine(&device)
        .expect("engine");
    let script = history(&tok, &files, DIAG_DEPTH, NEEDLES_AT);
    let needles = needle_positions(&script);
    let mut rows = Vec::new();
    for (seal, read) in [
        (WHOLE_WINDOW, WHOLE_WINDOW),
        (WHOLE_WINDOW, WORKING_SET),
        (WORKING_SET, WHOLE_WINDOW),
        (WORKING_SET, WORKING_SET),
    ] {
        let mut cfg = b.conversation_config();
        cfg.kv_compression_level = Some(10);
        let prompt = b.format_system_prompt();
        let sealing = plain_projection(&eng, &cfg, &prompt, seal);
        let layer = sealing.id_for_layer("dialogue").expect("dialogue layer");
        let group = sealing.schema().layers[0].groups[0].id;
        let mut conv = eng
            .new_conversation_with_projection(&prompt, sealing, layer, group, cfg.clone())
            .expect("new conversation");
        for t in &script {
            conv.insert_turn(t.as_str(), reply(t))
                .unwrap_or_else(|e| panic!("[seal {seal}] insert_turn failed: {e}"));
        }
        conv.set_projection(Arc::new(plain_projection(&eng, &cfg, &prompt, read)));
        let handle = conv.submit_turn(QUESTION).expect("submit the question");
        let mut points: Vec<ProjectionEvent> = Vec::new();
        let mut done = None;
        for event in handle.stream() {
            match event {
                TurnEvent::Projection(p) => points.push(p),
                TurnEvent::Done(r) => done = Some(r),
                TurnEvent::Error(e) => panic!("[seal {seal}, read {read}] decoding failed: {e}"),
                _ => {}
            }
        }
        let response = done.expect("the question finished");
        conv.finish_turn(handle, &response).expect("finish_turn");
        conv.close().expect("close");
        let last = points.last().map(|p| &p.selection.turns);
        let held = last.map_or(0, |t| t.len());
        let projected = last.map_or(0, |t| {
            t.iter().filter(|s| needles.contains(&s.index)).count()
        });
        let got = recalled(&response.text);
        println!(
            "[sealed under a {seal} window, read under {read}] {got}/4 · needles projected \
             {projected}/4 of {held} turns held — {:?}",
            answer(&response.text)
        );
        rows.push((seal, read, got, projected, held));
    }
    println!(
        "\n### Sealing or reading under projection — {DIAG_DEPTH} tokens, needles 65K deep, C10\n"
    );
    println!(
        "| Sealed under window | Read under window | Recalled | Needles projected | Turns held |"
    );
    println!("|---:|---:|---:|---:|---:|");
    for (seal, read, got, projected, held) in &rows {
        println!("| {seal} | {read} | {got}/4 | {projected}/4 | {held} |");
    }
    assert_eq!(rows.len(), 4, "every arm ran");
}

/// The workspace's repositories, in `D:\prog\workspace.yaml`'s order.
const CHAIN_REPOS: [&str; 4] = ["candle", "battle-cities", "journals", "yenbot"];

/// The anchor documents the priming chain reads from each repository's root,
/// in chain order (`zend/src/branch_ingest/prime.rs`, `ANCHOR_FILES`).
const CHAIN_ANCHORS: [&str; 4] = ["README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"];

/// The priming chain's reading, one turn a link: for each repository its root
/// listing, then each anchor document it has at its root, whole — the prefix
/// every `code_reading` file is sealed behind in production.
fn chain_turns() -> Vec<String> {
    let prog = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let mut turns = Vec::new();
    for repo in CHAIN_REPOS {
        let root = prog.join(repo);
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut names: Vec<(String, bool)> = entries
            .flatten()
            .map(|e| {
                (
                    e.file_name().to_string_lossy().into_owned(),
                    e.path().is_dir(),
                )
            })
            .filter(|(n, _)| !n.starts_with('.') && n != "target" && n != "substrate")
            .collect();
        names.sort();
        let listing: Vec<String> = names
            .iter()
            .map(|(n, dir)| if *dir { format!("{n}/") } else { n.clone() })
            .collect();
        turns.push(format!(
            "Here is the root listing of `{repo}`:\n```\n{}\n```",
            listing.join("\n")
        ));
        for anchor in CHAIN_ANCHORS {
            let Some((name, _)) = names
                .iter()
                .find(|(n, dir)| !dir && n.eq_ignore_ascii_case(anchor))
            else {
                continue;
            };
            if let Ok(text) = std::fs::read_to_string(root.join(name)) {
                turns.push(format!(
                    "Here is `{repo}/{name}`:\n```markdown\n{text}\n```"
                ));
            }
        }
    }
    turns
}

/// How the needles' conversation and the conversation that asks for them
/// stand to the priming chain.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Prefix {
    /// Sealed behind the system prompt alone; read by a reader that continues
    /// the needles' conversation.
    None,
    /// Sealed behind the chain; the needles' conversation is cut from it once
    /// sealed, so the reader reads the needles without the chain.
    SealedBehindReadWithout,
    /// Sealed behind the chain; the reader continues the needles'
    /// conversation, so it reads the chain and then the needles, as sealed,
    /// with the recurrence seeded from the lineage.
    SealedBehindReadWith,
    /// Sealed behind the chain and asked in the conversation that sealed them.
    AskedWhereSealed,
}

const PREFIX_ARMS: [(&str, Prefix); 4] = [
    ("no prefix", Prefix::None),
    (
        "sealed behind the chain, read without it",
        Prefix::SealedBehindReadWithout,
    ),
    (
        "sealed behind the chain, read with it",
        Prefix::SealedBehindReadWith,
    ),
    (
        "sealed behind the chain, asked where sealed",
        Prefix::AskedWhereSealed,
    ),
];

/// The question, asked under a diagnostic capture of `conv`'s decode: the
/// answer, and how many distinct pieces of `chain` and of `needles` the
/// slot held while it was decoded.
fn ask_counting_pieces(
    conv: &mut Sequence,
    chain: TimelineId,
    needles: TimelineId,
) -> (String, usize, usize) {
    assert!(
        decode_diag::arm(DiagCapture {
            control: Control::Sparse,
            timeline: conv.timeline_id().raw(),
            from: 0,
            live_at: usize::MAX,
        }),
        "another diagnostic decode holds the capture"
    );
    let handle = conv.submit_turn(QUESTION).expect("submit the question");
    let mut done = None;
    for event in handle.stream() {
        match event {
            TurnEvent::Done(r) => done = Some(r),
            TurnEvent::Error(e) => panic!("decoding failed: {e}"),
            _ => {}
        }
    }
    let capture = decode_diag::disarm();
    let response = done.expect("the question finished");
    conv.finish_turn(handle, &response).expect("finish_turn");
    let count = |tl: TimelineId| {
        let prefix = format!("turn:{}:", tl.raw());
        let mut labels: Vec<&str> = capture
            .pieces
            .iter()
            .map(|p| p.label.as_str())
            .filter(|l| l.starts_with(&prefix))
            .collect();
        labels.sort_unstable();
        labels.dedup();
        labels.len()
    };
    (response.text, count(chain), count(needles))
}

/// **Recall with and without the priming chain's prefix.** In production every
/// `code_reading` file's conversation is parented on the priming chain — the
/// workspace's root listings and anchor documents — before its reading starts,
/// with its recurrence seeded from the chain's end, so each of its turns is
/// sealed behind the chain (`zend/src/code_read/mod.rs`). A dialogue later
/// reads those turns' K/V.
///
/// The four needle turns stand in for one file's reading: sealed in their own
/// conversation, behind the chain or behind nothing, and asked for by a
/// reader that continues that conversation, with or without the chain in its
/// lineage — and, as the control that the chain really was in the sealing
/// slot, asked in the sealing conversation itself. The table counts the
/// chain's and the needles' pieces in the slot each question decoded through.
#[test]
#[ignore = "loads Flash-Next and seals the priming chain's documents; \
            run with --ignored --nocapture --test-threads=1, daemon stopped"]
fn recall_with_and_without_the_priming_prefix() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let b = builder();
    let tok = b
        .resolve_model(&device)
        .expect("model pack resolved")
        .tokenizer()
        .expect("tokenizer");
    let ws = workspace();
    let eng = builder()
        .workspace_path(ws.path())
        .engine(&device)
        .expect("engine");
    let mut cfg = b.conversation_config();
    cfg.kv_compression_level = Some(10);
    let prompt = b.format_system_prompt();

    let chain_script = chain_turns();
    let chain_tokens: usize = chain_script
        .iter()
        .map(|t| tok.encode(t.as_str(), false).map(|e| e.len()).unwrap_or(0))
        .sum();
    let mut chain_conv = eng
        .new_conversation(&prompt, cfg.clone())
        .expect("the chain's conversation");
    for t in &chain_script {
        chain_conv
            .insert_turn(t.as_str(), "Read.")
            .unwrap_or_else(|e| panic!("sealing a chain link failed: {e}"));
    }
    let chain = chain_conv.timeline_id();
    chain_conv.close().expect("close the chain");
    println!(
        "priming chain: {} links, {chain_tokens} tokens, timeline {}",
        chain_script.len(),
        chain.raw()
    );

    let mut rows = Vec::new();
    for (label, prefix) in PREFIX_ARMS {
        let mut sealing = eng
            .new_conversation(&prompt, cfg.clone())
            .expect("the needles' conversation");
        let needles = sealing.timeline_id();
        if prefix != Prefix::None {
            eng.set_forked_from(needles, chain)
                .expect("parent the needles on the chain");
            assert!(
                sealing
                    .seed_recurrent_from_lineage()
                    .expect("seed the recurrence"),
                "[{label}] the chain left no recurrent snapshot to seed from"
            );
        }
        for (vault, code) in NEEDLES {
            sealing
                .insert_turn(needle_turn(vault, code).as_str(), "OK")
                .unwrap_or_else(|e| panic!("[{label}] sealing a needle failed: {e}"));
        }
        let (text, chain_pieces, needle_pieces) = if prefix == Prefix::AskedWhereSealed {
            let asked = ask_counting_pieces(&mut sealing, chain, needles);
            sealing.close().expect("close");
            asked
        } else {
            sealing.close().expect("close");
            if prefix == Prefix::SealedBehindReadWithout {
                eng.clear_forked_from(needles)
                    .expect("cut the needles from the chain");
            }
            let mut reader = eng
                .new_conversation(&prompt, cfg.clone())
                .expect("the reader");
            eng.set_forked_from(reader.timeline_id(), needles)
                .expect("parent the reader on the needles");
            if prefix == Prefix::SealedBehindReadWith {
                reader
                    .seed_recurrent_from_lineage()
                    .expect("seed the reader's recurrence");
            }
            let asked = ask_counting_pieces(&mut reader, chain, needles);
            reader.close().expect("close");
            asked
        };
        let got = recalled(&text);
        println!(
            "[{label}] {got}/4 · chain pieces in the slot {chain_pieces} · needle pieces \
             {needle_pieces}/4 — {:?}",
            answer(&text)
        );
        println!(
            "[{label}] the whole reply:\n{}\n[{label}] end of reply",
            text.trim()
        );
        rows.push((
            label,
            got,
            chain_pieces,
            needle_pieces,
            answer(&text).to_string(),
        ));
    }

    println!(
        "\n### Recall with and without the priming prefix — chain {} links, {chain_tokens} \
         tokens, C10\n",
        chain_script.len()
    );
    println!("| Arm | Recalled | Chain pieces in the slot | Needle pieces | Answer |");
    println!("|---|---:|---:|---:|---|");
    for (label, got, chain_pieces, needle_pieces, a) in &rows {
        println!(
            "| {label} | {got}/4 | {chain_pieces} | {needle_pieces}/4 | {} |",
            a.replace('\n', " / ")
        );
    }
    assert_eq!(rows.len(), PREFIX_ARMS.len(), "every arm ran");
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
