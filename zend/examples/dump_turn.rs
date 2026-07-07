//! Dump one calibration turn (conversation) from the substrate: its full decoded
//! text, and the K/V chunk-format census (R16/F16 vs anything quantized).
//!
//! ```text
//! cargo run -p zend --example dump_turn --release -- <workspace> [match-substring]
//! ```
//! Without a match substring it picks the lowest-id turn stream.

use std::collections::BTreeMap;
use std::path::PathBuf;

use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;
use candle_nn::kv_cache::KvFormat;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let want = std::env::args().nth(2);

    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate at {}: {e}", workspace.display()))?;
    let tok = Tokenizer::from_file(workspace.join(".substrate").join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // Collect turn streams (id, token-count) in id order.
    let mut turns: Vec<u64> = substrate
        .all_streams()
        .filter(|(_, e)| matches!(e.decl, Some(StreamDecl::Turn(_))))
        .map(|(sid, _)| sid.0)
        .collect();
    turns.sort_unstable();

    // Pick: first turn whose decoded text contains `want`, else the lowest id.
    let mut chosen: Option<(u64, Vec<u32>, String)> = None;
    for sid in &turns {
        let stream_id = candle_conversation::persistence::streams::StreamId(*sid);
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
        let text = tok.decode(&ids, false).unwrap_or_default();
        match &want {
            Some(sub) if !text.contains(sub.as_str()) => continue,
            _ => {
                chosen = Some((*sid, ids, text));
                break;
            }
        }
    }
    let Some((sid_raw, ids, text)) = chosen else {
        anyhow::bail!("no turn stream matched {:?}", want);
    };
    let stream_id = candle_conversation::persistence::streams::StreamId(sid_raw);

    println!("=== turn stream {} — {} tokens ===\n", sid_raw, ids.len());
    println!("---- DECODED TEXT ----\n{text}\n---- END TEXT ----\n");

    // Per-token pieces (byte-BPE markers shown as ▁ / ⏎) so multi-token names are visible.
    let pieces: Vec<String> = ids
        .iter()
        .map(|&id| {
            tok.id_to_token(id)
                .unwrap_or_default()
                .replace('\u{0120}', "▁")
                .replace('\u{010A}', "⏎")
        })
        .collect();
    println!("---- {} TOKEN PIECES ----", pieces.len());
    println!("{}", pieces.join("│"));
    println!("---- END PIECES ----\n");

    // Chunk-format census over the turn's chunks. k_formats/v_formats hold one tag
    // per (head, palette) sub-band (n_kv_head × N_PALETTE = 16) per chunk.
    let chunk_indices: Vec<u64> = substrate
        .stream_of(stream_id)
        .map(|s| s.chunks.keys().copied().collect())
        .unwrap_or_default();
    let mut kfmt: BTreeMap<String, usize> = BTreeMap::new();
    let mut vfmt: BTreeMap<String, usize> = BTreeMap::new();
    let (mut k_total, mut v_total, mut read_ok, mut read_err) = (0usize, 0usize, 0usize, 0usize);
    for &ci in &chunk_indices {
        let payload = match persistence.read_chunk(&substrate, stream_id, ci) {
            Ok(p) => {
                read_ok += 1;
                p
            }
            Err(_) => {
                read_err += 1;
                continue;
            }
        };
        for &tag in &payload.k_formats {
            let name = KvFormat::from_tag(tag)
                .map(|f| format!("{f:?}"))
                .unwrap_or_else(|| format!("<unknown tag {tag}>"));
            *kfmt.entry(name).or_default() += 1;
            k_total += 1;
        }
        for &tag in &payload.v_formats {
            let name = KvFormat::from_tag(tag)
                .map(|f| format!("{f:?}"))
                .unwrap_or_else(|| format!("<unknown tag {tag}>"));
            *vfmt.entry(name).or_default() += 1;
            v_total += 1;
        }
    }

    println!(
        "---- CHUNK FORMAT CENSUS ({} chunks read, {} errors) ----",
        read_ok, read_err
    );
    let pct = |n: usize, tot: usize| {
        if tot > 0 {
            n as f64 / tot as f64 * 100.0
        } else {
            0.0
        }
    };
    println!("K sub-bands: {k_total}");
    for (name, n) in &kfmt {
        println!("  {:<18} {:>7}  ({:>5.1}%)", name, n, pct(*n, k_total));
    }
    println!("V sub-bands: {v_total}");
    for (name, n) in &vfmt {
        println!("  {:<18} {:>7}  ({:>5.1}%)", name, n, pct(*n, v_total));
    }
    Ok(())
}
