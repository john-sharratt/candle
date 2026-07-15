//! Locate *where* the dead wide-Q tokens sit in a turn and diagnose the cause.
//!
//! A [`WideQSig`] token is "dead" when its popcount is `n_heads*128` (all sign
//! bits set ⇒ every Q component read as `>= 0`) or `0` — neither happens for real
//! zero-mean Q, so a dead token means the R16 `q[]` field was zero / uninitialized
//! / compressed-away at gather time. This tool classifies every dead token by
//! region so we can see the cause:
//!   · sink     — index 0..4          (attention-sink tokens, special scale)
//!   · body     — index 4..n_tok      (inside the turn's own token span)
//!   · trailing — index >= n_tok      (over-capture past the Tokens record)
//! and per-layer, so we can tell "all layers dead" (uninitialized read) from
//! "some layers dead" (those layers' R16 gone).
//!
//! ```text
//! cargo run -p zend --example dump_dead_sigs --release -- <workspace> [substring]
//! ```

use std::path::PathBuf;

use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{StreamDecl, StreamId};
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::{decode_wide_sigs, WideQSig};
use candle_conversation::substrate::Substrate;
use tokenizers::Tokenizer;

const HEAD_BITS: u32 = 128;
const N_KV_HEAD: usize = 4;

fn piece(tok: &Tokenizer, id: u32) -> String {
    tok.id_to_token(id)
        .unwrap_or_default()
        .replace('\u{0120}', "▁")
        .replace('\u{010A}', "⏎")
}

/// `(is_dead, all_ones)` — all_ones distinguishes the all-`>=0` read (zeroed q[])
/// from the all-`<0` read (popcount 0).
fn deadness(s: &WideQSig) -> (bool, bool) {
    let full = s.n_heads as u32 * HEAD_BITS;
    let pc = s.popcount();
    (pc == full || pc == 0, pc == full)
}

/// Per-layer popcount for one token: `n_kv_head` heads per layer, `head_dim` bits
/// each. Returns (layer_index, popcount, full) so a caller can see which layers
/// are degenerate.
fn per_layer_popcounts(s: &WideQSig) -> Vec<(usize, u32, u32)> {
    let wph = s.words_per_head();
    let n_layers = s.n_heads as usize / N_KV_HEAD;
    let full = (N_KV_HEAD as u32) * HEAD_BITS;
    (0..n_layers)
        .map(|l| {
            let lo = l * N_KV_HEAD * wph;
            let hi = (l + 1) * N_KV_HEAD * wph;
            let pc: u32 = s.words[lo..hi].iter().map(|w| w.count_ones()).sum();
            (l, pc, full)
        })
        .collect()
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

    // ── Aggregate: classify every dead token across all turns by region ─────────
    let (mut d_sink, mut d_body, mut d_trail, mut d_total) = (0usize, 0usize, 0usize, 0usize);
    let (mut n_tokens_all, mut n_sig_all) = (0usize, 0usize);
    let mut trailing_len_hist: Vec<usize> = Vec::new(); // history.len() - n_tok per turn
    for (sid, e) in substrate.all_streams() {
        if !matches!(e.decl, Some(StreamDecl::Turn(_))) {
            continue;
        }
        let Some(hist) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        let n_tok = persistence
            .read_tokens(&substrate, sid)
            .ok()
            .flatten()
            .and_then(|b| decode_token_ids(&b).ok())
            .map(|v| v.len())
            .unwrap_or(0);
        n_tokens_all += n_tok;
        n_sig_all += hist.len();
        if hist.len() > n_tok {
            trailing_len_hist.push(hist.len() - n_tok);
        }
        for (i, s) in hist.iter().enumerate() {
            if deadness(s).0 {
                d_total += 1;
                if i < 4 {
                    d_sink += 1;
                } else if i < n_tok {
                    d_body += 1;
                } else {
                    d_trail += 1;
                }
            }
        }
    }
    println!("=== AGGREGATE dead wide-Q classification (all turns) ===");
    println!(
        "  wide-Q tokens: {n_sig_all}   Tokens-record tokens: {n_tokens_all}   over-capture: {}",
        n_sig_all as isize - n_tokens_all as isize
    );
    println!(
        "  DEAD total {d_total}  =  sink(0..4) {d_sink}  +  body(4..n_tok) {d_body}  +  trailing(>=n_tok) {d_trail}"
    );
    if d_total > 0 {
        println!(
            "    sink {:.1}%   body {:.1}%   trailing {:.1}%",
            100.0 * d_sink as f64 / d_total as f64,
            100.0 * d_body as f64 / d_total as f64,
            100.0 * d_trail as f64 / d_total as f64,
        );
    }
    if !trailing_len_hist.is_empty() {
        let sum: usize = trailing_len_hist.iter().sum();
        let mx = trailing_len_hist.iter().copied().max().unwrap_or(0);
        println!(
            "  per-turn over-capture (wideQ.len - n_tok): avg {:.1}, max {}, turns {}",
            sum as f64 / trailing_len_hist.len() as f64,
            mx,
            trailing_len_hist.len()
        );
    }

    // ── Pick one case with a healthy number of dead tokens ──────────────────────
    let mut chosen: Option<(StreamId, Vec<u32>, usize, Vec<WideQSig>, usize)> = None;
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(d)) = &e.decl else {
            continue;
        };
        let Some(hist) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
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
        if !want.is_empty() && !text.contains(&want) {
            continue;
        }
        let dead = hist.iter().filter(|s| deadness(s).0).count();
        if dead > 0 {
            chosen = Some((
                sid,
                ids.clone(),
                d.assistant_content_start() as usize,
                hist,
                dead,
            ));
            if want.is_empty() {
                // Prefer a small, clear case: stop at the first with 3..40 dead.
                if (3..40).contains(&dead) {
                    break;
                }
            } else {
                break;
            }
        }
    }
    let Some((sid, ids, asst, hist, dead)) = chosen else {
        println!("\n(no turn with dead wide-Q tokens matched)");
        return Ok(());
    };
    let n_tok = ids.len();
    let full = hist
        .first()
        .map(|s| s.n_heads as u32 * HEAD_BITS)
        .unwrap_or(0);

    println!(
        "\n=== CASE stream {}  ({n_tok} Tokens, {} wide-Q, {dead} dead) ===",
        sid.0,
        hist.len()
    );
    println!(
        "  prefill 0..{asst}   decode {asst}..{n_tok}   over-capture {}..{}",
        n_tok,
        hist.len()
    );
    println!("  full popcount per token = {full}  (all-ones ⇒ q[] read all-zero)\n");

    // Per-index dead map with token strings and all-ones vs all-zero.
    for (i, s) in hist.iter().enumerate() {
        let (is_dead, all_ones) = deadness(s);
        if !is_dead {
            continue;
        }
        let region = if i < 4 {
            "SINK"
        } else if i < asst {
            "prefill"
        } else if i < n_tok {
            "decode"
        } else {
            "TRAILING"
        };
        let piece_s = if i < n_tok {
            format!("{:?}", piece(&tok, ids[i]))
        } else {
            "<past Tokens record>".to_string()
        };
        let kind = if all_ones {
            "all-ones (q=0)"
        } else {
            "all-zero (q<0)"
        };
        // Per-layer: how many of the 48 layers are degenerate for this token?
        let pl = per_layer_popcounts(s);
        let dead_layers = pl
            .iter()
            .filter(|(_, pc, lf)| *pc == *lf || *pc == 0)
            .count();
        println!(
            "  idx {i:>4} [{region:<8}] {kind}  dead-layers {dead_layers}/{}  tok {piece_s}",
            pl.len()
        );
    }

    Ok(())
}
