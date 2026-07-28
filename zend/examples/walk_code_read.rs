//! Walk turn streams in id order (ingest order) and score each for corruption:
//! the count of **token id 0** (renders as `!`, the garbage/NaN-KV decode token)
//! and CJK code points in the stored, decoded summary text. Prints the first
//! turns and flags the clean→garbage transition, so a rebuild can be checked for
//! where corruption enters.
//!
//! ```text
//! cargo run -p zend --example walk_code_read --release -- <workspace> [limit]
//! ```
//! Read-only: opens the append-only log and replays it; safe against a live
//! daemon (reads a consistent committed prefix).

use std::path::PathBuf;

use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{StreamDecl, StreamId};
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;
use tokenizers::Tokenizer;

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF)
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate at {}: {e}", workspace.display()))?;
    let tok = Tokenizer::from_file(workspace.join(".substrate").join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    println!("token id 0 => {:?}", tok.id_to_token(0));
    println!();

    let mut turns: Vec<u64> = substrate
        .all_streams()
        .filter(|(_, e)| matches!(e.decl, Some(StreamDecl::Turn(_))))
        .map(|(sid, _)| sid.0)
        .collect();
    turns.sort_unstable();
    println!(
        "{} turn streams total; walking first {}\n",
        turns.len(),
        limit
    );

    // How many readable turns to paste IN FULL for eyeball review, regardless
    // of clean/corrupt — set via env `SAMPLE_FULL` (default 8). Corrupt turns
    // are always printed (up to 25) on top of this.
    let sample_full: usize = std::env::var("SAMPLE_FULL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let (mut scanned, mut clean, mut bad, mut printed, mut sampled) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut tot_id0, mut tot_cjk, mut tot_tok) = (0usize, 0usize, 0usize);
    for sid in turns.iter().take(limit) {
        let stream_id = StreamId(*sid);
        let Some(tb) = persistence
            .read_tokens(&substrate, stream_id)
            .ok()
            .flatten()
        else {
            continue;
        };
        let Ok(ids) = decode_token_ids(&tb) else {
            continue;
        };
        scanned += 1;
        let zeros = ids.iter().filter(|&&i| i == 0).count();
        let text = tok.decode(&ids, false).unwrap_or_default();
        let cjk = text.chars().filter(|&c| is_cjk(c)).count();
        tot_id0 += zeros;
        tot_cjk += cjk;
        tot_tok += ids.len();
        // Paste the first `sample_full` readable turns verbatim (newlines kept)
        // so a human can read the actual stored summaries.
        if sampled < sample_full {
            sampled += 1;
            println!(
                "==== SAMPLE stream {} | tok={} | id0={} | cjk={} ====\n{}\n---- end sample ----\n",
                sid,
                ids.len(),
                zeros,
                cjk,
                text
            );
        }
        if zeros == 0 && cjk == 0 {
            clean += 1;
            continue;
        }
        bad += 1;
        if printed < 25 {
            printed += 1;
            // FULL text so the Chinese / '!' is visible in context — is it a
            // stray token in English, or a whole garbage/Chinese summary?
            println!(
                "==== CORRUPT stream {} | tok={} | id0={} | cjk={} ====\n{}\n",
                sid,
                ids.len(),
                zeros,
                cjk,
                text.replace('\n', " ⏎ ")
            );
        }
    }

    println!(
        "\n==== SCAN: {scanned} turns read | {clean} clean | {bad} corrupt ({:.1}%) | total tok={tot_tok} id0={tot_id0} ({:.3}%) cjk={tot_cjk} ({:.3}%) ====",
        if scanned > 0 { bad as f64 / scanned as f64 * 100.0 } else { 0.0 },
        if tot_tok > 0 { tot_id0 as f64 / tot_tok as f64 * 100.0 } else { 0.0 },
        if tot_tok > 0 { tot_cjk as f64 / tot_tok as f64 * 100.0 } else { 0.0 },
    );
    Ok(())
}
