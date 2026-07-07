//! Export the calibration trajectories the "Calibrating sections" load phase
//! decoded into the substrate — one Markdown file per authored `(tool, example)`
//! under `zend-tools/examples/` — as the source material for the
//! *prefill-instead-of-decode* calibration path.
//!
//! Each file holds the model's full turn for that example *except the system
//! block*: the `Tokens` record is already just the user prompt + the assistant
//! trajectory (the `<think>` and `<tool_call>`), because the system prompt is the
//! separate projection and never enters a turn's tokens.
//!
//! # Projection markers
//!
//! Every trajectory is annotated with [`MARKER`] tokens at the exact points a
//! projection fired during the live decode, so the prefill path can `split` on
//! the marker, prefill each segment, and generate one projection per marker —
//! reproducing the decode's projection sequence deterministically instead of
//! re-sampling it.
//!
//! The marker positions are **read from the substrate**, not guessed: each
//! turn's persisted [`ProjectionEvent`]s carry `[start_token, end_token)` spans
//! in generated-token space (`projection/event.rs`). The span *boundaries* are
//! the projection points — and because the first span starts at generated-token
//! `0`, a marker lands right after the user turn (the initial tool-scope
//! projection), then one per mid-decode reprojection, then one at the final seal.
//! Generated-token `0` is anchored at the first `<think>` after the
//! `<|im_start|>assistant` header (the user prompt + header before it is
//! prefilled scaffold, never decoded).
//!
//! Matching is by **user prompt**: for every authored example we find the
//! latest turn whose decoded text opens with that prompt, so the most recent
//! calibration run wins and truncated trajectories still export. Run from the
//! workspace root after a calibration run:
//!
//! ```text
//! cargo run -p zend --example export_calibration --release
//! ```

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_conversation::persistence::log_file::{read_record_at, LogFile, SUPERBLOCK_SIZE};
use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::walker;
use candle_conversation::projection::decode_events;
use candle_conversation::substrate::Substrate;
use candle_conversation::ProjectionEvent;
use tokenizers::Tokenizer;
use zend_tools::registry;

/// Sentinel string inserted at every projection point. The prefill calibration
/// path `split`s the trajectory on this (it is never tokenized — it is stripped
/// before each segment is encoded), so any content-unique string works; it
/// mirrors the `<|im_*|>` chat-marker style for readability and never appears in
/// real trajectory text.
const MARKER: &str = "<|projection|>";

/// One decoded turn: its text, token IDs, persisted projection events, and the
/// log offset of its `Tokens` record (higher offset = written later, so the
/// latest run wins on a tie).
struct DecodedTurn {
    text: String,
    ids: Vec<u32>,
    events: Vec<ProjectionEvent>,
    offset: u64,
}

/// First index at which `needle` occurs contiguously in `hay`, if any.
fn find_subseq(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| hay[i..i + needle.len()] == *needle)
}

/// Verify the prefill string-split is token-exact: splitting the marked text on
/// [`MARKER`] and re-encoding each piece (the prefill path's exact operation)
/// must reproduce the original token IDs, or the prefilled KV — and thus the
/// wide-Q signature — diverges from the decode baseline. Returns `Ok(None)` on an
/// exact match, else the first divergent token index and the two lengths.
fn roundtrip_divergence(
    marked: &str,
    ids: &[u32],
    tok: &Tokenizer,
) -> Result<Option<(usize, usize, usize)>> {
    let mut retok: Vec<u32> = Vec::new();
    for piece in marked.split(MARKER) {
        if piece.is_empty() {
            continue;
        }
        let enc = tok
            .encode(piece, false)
            .map_err(|e| anyhow::anyhow!("re-encode piece: {e}"))?;
        retok.extend_from_slice(enc.get_ids());
    }
    if retok == ids {
        return Ok(None);
    }
    let div = retok
        .iter()
        .zip(ids)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| retok.len().min(ids.len()));
    Ok(Some((div, retok.len(), ids.len())))
}

/// Reconstruct the trajectory text with a [`MARKER`] at every projection point,
/// using the substrate's persisted [`ProjectionEvent`] spans.
///
/// `think_ids` is the tokenization of `<think>`, the generated-token-0 anchor;
/// `assistant_ids` is `<|im_start|>assistant`, and the `<think>` search begins
/// after it so a `<think>` inside the user prompt is never matched by mistake.
/// The span boundaries (`{0, start_token, end_token, …, N}`) become marker
/// positions: `0` right after the user turn, each reprojection where it fired,
/// and `N` at the seal. Returns `(marked_text, marker_count)`.
fn inject_projection_markers(
    ids: &[u32],
    tok: &Tokenizer,
    think_ids: &[u32],
    assistant_ids: &[u32],
    events: &[ProjectionEvent],
) -> Result<(String, usize)> {
    let decode = |slice: &[u32]| -> Result<String> {
        tok.decode(slice, false)
            .map_err(|e| anyhow::anyhow!("detokenize segment: {e}"))
    };
    // Total generated tokens = the last span's end. With no spans there is
    // nothing to place; emit verbatim with a single trailing seal marker.
    let n = events
        .iter()
        .map(|e| e.end_token as usize)
        .max()
        .unwrap_or(0);
    if n == 0 {
        return Ok((format!("{}{MARKER}", decode(ids)?), 1));
    }

    // Generated-token 0 sits at the first `<think>` *after the assistant header*;
    // everything before is prefilled scaffold. Anchoring the search past the
    // header keeps a `<think>` appearing in the user prompt from stealing the
    // anchor. Fall back to aligning by the generated count when the anchor is
    // absent (no calibration case lacks a think, but stay defensive).
    let search_from = find_subseq(ids, assistant_ids)
        .map(|p| p + assistant_ids.len())
        .unwrap_or(0);
    let gen_start = find_subseq(&ids[search_from..], think_ids)
        .map(|r| search_from + r)
        .unwrap_or_else(|| ids.len().saturating_sub(n));
    let gen = &ids[gen_start..];
    let cap = n.min(gen.len());

    // Span boundaries in generated-token space = the projection points. Always
    // include 0 (initial, after the user turn) and `cap` (seal).
    let mut bset: BTreeSet<usize> = BTreeSet::new();
    bset.insert(0);
    bset.insert(cap);
    for e in events {
        bset.insert((e.start_token as usize).min(cap));
        bset.insert((e.end_token as usize).min(cap));
    }
    let boundaries: Vec<usize> = bset.into_iter().collect();

    let mut out = decode(&ids[..gen_start])?; // prompt + assistant header
    let mut prev = 0usize;
    let mut markers = 0usize;
    for &b in &boundaries {
        if b > prev {
            out.push_str(&decode(&gen[prev..b])?);
        }
        out.push_str(MARKER);
        markers += 1;
        prev = b;
    }
    // Any tail past the last generated token (e.g. a post-decode role-end) keeps
    // its place after the seal marker.
    if prev < gen.len() {
        out.push_str(&decode(&gen[prev..])?);
    }
    Ok((out, markers))
}

fn main() -> Result<()> {
    let log_path = PathBuf::from(".substrate/substrate.log");
    let out_root = PathBuf::from("zend-tools/examples");
    let mut log = LogFile::open(&log_path)
        .with_context(|| format!("opening substrate log {}", log_path.display()))?;

    // Replay the redo log into an in-RAM substrate so every stream's `Tokens`
    // record and projection-event blob is known.
    let mut substrate = Substrate::new();
    let (entries, _) = walker::collect(&mut log, SUPERBLOCK_SIZE)?;
    for e in &entries {
        substrate.apply_walker_entry(e);
    }

    // The tokenizer lives in a sidecar next to the log (the record itself is
    // hash-only).
    let sidecar = log_path
        .parent()
        .map(|p| p.join("tokenizer.json"))
        .unwrap_or_else(|| PathBuf::from("tokenizer.json"));
    let tok_bytes = std::fs::read(&sidecar)
        .with_context(|| format!("reading tokenizer sidecar {}", sidecar.display()))?;
    let tok = Tokenizer::from_bytes(&tok_bytes)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", sidecar.display()))?;

    // The `<think>` generation anchor, encoded once, plus the assistant header
    // that precedes the generated span — the `<think>` search starts after the
    // header so a `<think>` inside the user prompt can never be mistaken for it.
    let think_ids: Vec<u32> = tok
        .encode("<think>", false)
        .map_err(|e| anyhow::anyhow!("encoding <think>: {e}"))?
        .get_ids()
        .to_vec();
    let assistant_ids: Vec<u32> = tok
        .encode("<|im_start|>assistant", false)
        .map_err(|e| anyhow::anyhow!("encoding assistant header: {e}"))?
        .get_ids()
        .to_vec();

    // Decode every turn stream once, carrying its persisted projection events.
    // Special tokens are kept so the chat markers (`<|im_end|>`,
    // `<|im_start|>assistant`, `<think>`) survive into the exported trajectory.
    let mut turns: Vec<DecodedTurn> = Vec::new();
    for (_, stream) in substrate.all_streams() {
        if !matches!(stream.decl, Some(StreamDecl::Turn(_))) {
            continue;
        }
        let Some(loc) = stream.tokens else { continue };
        let rec = read_record_at(&mut log, loc.offset, loc.record_size)?;
        let ids = decode_token_ids(&rec.payload)?;
        let text = tok
            .decode(&ids, false)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
        let events = stream
            .projection_events
            .as_deref()
            .map(decode_events)
            .unwrap_or_default();
        turns.push(DecodedTurn {
            text,
            ids,
            events,
            offset: loc.offset,
        });
    }
    println!("decoded {} turn streams from the substrate", turns.len());

    let mut written = 0usize;
    let mut total_markers = 0usize;
    let mut roundtrip_exact = 0usize;
    let mut roundtrip_bad: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let mut no_events: Vec<String> = Vec::new();
    for tool in registry::all_tools() {
        for (i, example) in tool.examples.iter().enumerate() {
            if example.is_empty() {
                continue;
            }
            // The `Tokens` text opens with the user prompt (then `<|im_end|>`),
            // so a prefix match identifies this example's turn; pick the latest.
            let best = turns
                .iter()
                .filter(|t| t.text.starts_with(example))
                .max_by_key(|t| t.offset);
            let Some(turn) = best else {
                missing.push(format!("{} #{}", tool.name, i + 1));
                continue;
            };
            if turn.events.is_empty() {
                no_events.push(format!("{} #{}", tool.name, i + 1));
            }
            let (marked, markers) = inject_projection_markers(
                &turn.ids,
                &tok,
                &think_ids,
                &assistant_ids,
                &turn.events,
            )?;
            total_markers += markers;
            match roundtrip_divergence(&marked, &turn.ids, &tok)? {
                None => roundtrip_exact += 1,
                Some((div, got, want)) => {
                    // Distinguish a split-boundary artifact (fixable by snapping the
                    // marker) from a lossy full-text round-trip (would force storing IDs).
                    let full_ids = tok
                        .encode(turn.text.as_str(), false)
                        .map_err(|e| anyhow::anyhow!("full re-encode: {e}"))?;
                    let full_ok = full_ids.get_ids() == turn.ids.as_slice();
                    roundtrip_bad.push(format!(
                        "{} #{} (diverges at tok {div}, {got} vs {want}, full-text-roundtrips={full_ok})",
                        tool.name,
                        i + 1
                    ))
                }
            }
            std::fs::create_dir_all(&out_root)?;
            let path = out_root.join(format!("{}_{:02}.md", tool.name, i + 1));
            std::fs::write(&path, &marked)
                .with_context(|| format!("writing {}", path.display()))?;
            written += 1;
        }
    }

    println!(
        "wrote {written} example trajectories ({total_markers} projection markers, {:.1} avg) under {}",
        if written > 0 {
            total_markers as f64 / written as f64
        } else {
            0.0
        },
        out_root.display()
    );
    println!(
        "token round-trip (split on marker → re-encode → == original ids): {roundtrip_exact}/{written} exact"
    );
    if !roundtrip_bad.is_empty() {
        println!(
            "  {} NOT token-exact (prefill would diverge): {}",
            roundtrip_bad.len(),
            roundtrip_bad
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if !no_events.is_empty() {
        println!(
            "{} example(s) had no persisted projection events (seal-only marker): {}",
            no_events.len(),
            no_events.join(", ")
        );
    }
    if !missing.is_empty() {
        println!(
            "no substrate trajectory matched {} example(s): {}",
            missing.len(),
            missing.join(", ")
        );
    }
    Ok(())
}
