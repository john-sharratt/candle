//! Detailed anatomy of one calibration case: how many projection events and
//! signature windows it carries, and where each sits in the token sequence.
//!
//! ```text
//! cargo run -p zend --example dump_case_detail --release -- <workspace> <substring>
//! ```

use std::path::PathBuf;

use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{StreamDecl, StreamId};
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::{decode_events, SystemItem};
use candle_conversation::provenance::decode_wide_sigs;
use candle_conversation::substrate::Substrate;
use tokenizers::Tokenizer;

fn piece(tok: &Tokenizer, id: u32) -> String {
    tok.id_to_token(id)
        .unwrap_or_default()
        .replace('\u{0120}', "▁")
        .replace('\u{010A}', "⏎")
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let want = std::env::args().nth(2).unwrap_or_default();

    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate at {}: {e}", workspace.display()))?;
    let tok = Tokenizer::from_file(workspace.join(".substrate").join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // ── Aggregate: per-turn counts of projection events + wide-Q + signature tokens ─
    let mut turns = 0usize;
    let mut with_proj = 0usize;
    let mut proj_event_counts: Vec<usize> = Vec::new();
    let mut wideq_tok_counts: Vec<usize> = Vec::new();
    let (mut n_dead_tok, mut n_tot_tok) = (0usize, 0usize);
    for (_sid, e) in substrate.all_streams() {
        if !matches!(e.decl, Some(StreamDecl::Turn(_))) {
            continue;
        }
        turns += 1;
        if let Some(b) = &e.projection_events {
            let n = decode_events(b).len();
            if n > 0 {
                with_proj += 1;
                proj_event_counts.push(n);
            }
        }
        if let Some(hist) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) {
            // Complete per-token wide sign(Q) history (the widened Signatures).
            wideq_tok_counts.push(hist.len());
            let nheads = hist.first().map(|t| t.n_heads as u32).unwrap_or(0);
            let full = nheads * 128;
            n_dead_tok += hist.iter().filter(|t| t.popcount() == full).count();
            n_tot_tok += hist.len();
        }
    }
    let avg = |v: &[usize]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<usize>() as f64 / v.len() as f64
        }
    };
    let maxof = |v: &[usize]| v.iter().copied().max().unwrap_or(0);
    println!("=== AGGREGATE over {turns} turn streams ===");
    println!(
        "  turns with a projection event: {with_proj}   projection events per such turn: avg {:.2}, max {}",
        avg(&proj_event_counts),
        maxof(&proj_event_counts)
    );
    println!(
        "  wide-Q sign(Q) history tokens per turn: avg {:.1}, max {}  (complete per-token record)",
        avg(&wideq_tok_counts),
        maxof(&wideq_tok_counts),
    );
    println!(
        "  wide-Q DEAD tokens (all-ones/zero band): {n_dead_tok} of {n_tot_tok}  ({:.1}%)",
        100.0 * n_dead_tok as f64 / n_tot_tok.max(1) as f64
    );

    // ── Pick one case ───────────────────────────────────────────────────────────
    let mut chosen: Option<(StreamId, Vec<u32>, u32)> = None;
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(d)) = &e.decl else {
            continue;
        };
        let Some(ids) = persistence
            .read_tokens(&substrate, sid)
            .ok()
            .flatten()
            .and_then(|b| decode_token_ids(&b).ok())
        else {
            continue;
        };
        let text = tok.decode(&ids, false).unwrap_or_default();
        if want.is_empty() || text.contains(&want) {
            chosen = Some((sid, ids, d.assistant_content_start()));
            break;
        }
    }
    let Some((sid, ids, asst_start)) = chosen else {
        anyhow::bail!("no turn matched {want:?}");
    };
    let n_tok = ids.len();
    let asst = asst_start as usize;
    let pieces: Vec<String> = ids.iter().map(|&id| piece(&tok, id)).collect();

    // Locate think / tool_call markers by token position.
    let find = |needle: &str| pieces.iter().position(|p| p.contains(needle));
    let think_open = find("<think>");
    let think_close = find("</think>");
    let call_open = find("<tool_call>");
    let call_close = find("</tool_call>");

    // Projection events + wide-Q window for this case.
    let entry = substrate.stream_of(sid).unwrap();
    let events = entry
        .projection_events
        .as_ref()
        .map(|b| decode_events(b))
        .unwrap_or_default();
    let history = entry
        .wide_q_sigs
        .as_ref()
        .and_then(|b| decode_wide_sigs(b))
        .unwrap_or_default();

    println!("\n=== CASE stream {}  ({n_tok} tokens) ===", sid.0);
    println!("  prefill (user)  : positions 0..{asst}   ({asst} tok)");
    println!(
        "  decode (asst)   : positions {asst}..{n_tok}   ({} tok)",
        n_tok - asst
    );
    if let (Some(a), Some(b)) = (think_open, think_close) {
        println!(
            "  <think>…</think>: positions {a}..{}   ({} tok)",
            b + 1,
            b + 1 - a
        );
    }
    if let (Some(a), Some(b)) = (call_open, call_close) {
        println!(
            "  <tool_call>…    : positions {a}..{}   ({} tok)",
            b + 1,
            b + 1 - a
        );
    }
    println!(
        "\n  PROJECTION EVENTS: {}   (point model: each applies at one generated-token position)",
        events.len()
    );
    for (i, ev) in events.iter().enumerate() {
        // generated-token index → absolute position = asst_start + gen_index.
        let ps = asst + ev.start_token as usize;
        let sel: Vec<&str> = ev
            .selection
            .system
            .iter()
            .filter_map(|it| match it {
                SystemItem::Collection { sections, .. } => sections
                    .iter()
                    .find(|s| s.selected)
                    .map(|s| s.name.as_str()),
                _ => None,
            })
            .collect();
        println!(
            "    [{i}] gen {}  → position {ps}   selects: {}",
            ev.start_token,
            sel.join(", ")
        );
    }
    let nheads = history.first().map(|t| t.n_heads).unwrap_or(0);
    let full = nheads as u32 * 128;
    let dead = history.iter().filter(|t| t.popcount() == full).count();
    let med_pc = {
        let mut pc: Vec<u32> = history.iter().map(|t| t.popcount()).collect();
        pc.sort_unstable();
        pc.get(pc.len() / 2).copied().unwrap_or(0)
    };
    println!("\n  SIGNATURE RECORDS on this turn:");
    println!(
        "    · wide-Q `wide_q_sigs` : {} per-token entries, all layers × {nheads} heads; median popcount {med_pc} of {full}; {dead} dead",
        history.len(),
    );

    // ── Visual token map: a ruler + region bars ─────────────────────────────────
    println!("\n  TOKEN-SEQUENCE MAP (each column ≈ one token):");
    let bar = |lo: usize, hi: usize, ch: char| -> String {
        (0..n_tok)
            .map(|p| if p >= lo && p < hi { ch } else { ' ' })
            .collect()
    };
    // Downscale to ~120 cols so it fits a terminal.
    let width = 120usize.min(n_tok);
    let scale = |s: &str| -> String {
        (0..width)
            .map(|c| {
                let lo = c * n_tok / width;
                let hi = ((c + 1) * n_tok / width).max(lo + 1);
                // pick the densest non-space char in this column's span
                s[lo..hi.min(n_tok)]
                    .chars()
                    .find(|&ch| ch != ' ')
                    .unwrap_or(' ')
            })
            .collect()
    };
    let ruler: String = (0..width)
        .map(|c| {
            let p = c * n_tok / width;
            if p % 32 < (n_tok / width).max(1) {
                '|'
            } else {
                '·'
            }
        })
        .collect();
    println!(
        "    pos 0{}{n_tok}",
        " ".repeat(width.saturating_sub(format!("{n_tok}").len() + 1))
    );
    println!("        {ruler}");
    println!("    usr {}", scale(&bar(0, asst, 'U')));
    println!("    dec {}", scale(&bar(asst, n_tok, 'D')));
    if let (Some(a), Some(b)) = (think_open, think_close) {
        println!("    thk {}", scale(&bar(a, b + 1, 'T')));
    }
    if let (Some(a), Some(b)) = (call_open, call_close) {
        println!("    cal {}", scale(&bar(a, b + 1, 'C')));
    }
    for (i, ev) in events.iter().enumerate() {
        let ps = (asst + ev.start_token as usize).min(n_tok.saturating_sub(1));
        println!("    p{i:<2} {}", scale(&bar(ps, ps + 1, 'P')));
    }
    // The wide-Q history is one per-token signature covering the whole turn (its
    // length maps to the turn's tokens, prefill + decode).
    println!("    wQ  {}", scale(&bar(0, history.len().min(n_tok), 'W')));
    println!("\n    legend: U=user prefill  D=asst decode  T=<think>  C=<tool_call>  P=projection-event point");
    println!("            W=wide-Q per-token sign(Q) history");
    Ok(())
}
