//! `substrate_census` — the K/V format census of a segmented substrate store
//! (`docs/unbounded_agents.md` §9.6.9): over every live chunk, how many
//! sub-bands each format holds, their bytes, and the bytes the same values
//! take at BF16 (`census.rs`), as one JSON object.
//!
//! It walks the store read-only. Take it with the store quiescent —
//! maintenance that retires a segment during the walk ends it with a read
//! error — by holding the daemon down for it:
//! ```sh
//! cargo run -p candle-conversation --release --example substrate_census -- \
//!   --store D:\prog\substrate > census.json
//! ```

mod census;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_conversation::persistence::log_file::{read_record_at, LogFile, SUPERBLOCK_SIZE};
use candle_conversation::persistence::record::{ChunkPayload, RecordType};
use candle_conversation::persistence::sealed_reader::SealedReader;
use candle_conversation::persistence::segment::SegmentId;
use candle_conversation::persistence::walker;
use candle_conversation::substrate::Substrate;
use clap::Parser;

use census::{Census, ChunkShape};

#[derive(Parser)]
#[command(about = "Read-only K/V format census of a substrate store")]
struct Cli {
    /// The segmented store: the `substrate` directory of `seg-*` files.
    #[arg(long)]
    store: PathBuf,
}

/// The store's segments, by id: `seg-<id>.log` sealed, `seg-<id>.active`
/// the one being written.
fn segments(store: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(store).with_context(|| format!("reading {}", store.display()))? {
        let path = e?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(rest) = name.strip_prefix("seg-") else {
            continue;
        };
        let Some(id) = rest
            .strip_suffix(".log")
            .or_else(|| rest.strip_suffix(".active"))
        else {
            continue;
        };
        if let Ok(id) = id.parse::<u64>() {
            out.push((id, path));
        }
    }
    out.sort();
    if out.is_empty() {
        bail!("{} holds no seg-* files", store.display());
    }
    Ok(out)
}

/// Whether the metadata walk reads a record's payload: not for anything the
/// substrate keeps by location, which the census reads itself.
fn walk_reads_payload(rt: RecordType) -> bool {
    !matches!(
        rt,
        RecordType::Chunk
            | RecordType::Tokens
            | RecordType::Snapshot
            | RecordType::HeaderIndex
            | RecordType::TurnIndexPage
            | RecordType::WideQSig
            | RecordType::ProjectionEvents
    )
}

/// The metadata substrate of every segment, walked in id order so
/// last-writer-wins resolves as the daemon's reload does.
fn substrate_of(store: &Path, segs: &[(u64, PathBuf)]) -> Result<Substrate> {
    let mut substrate = Substrate::new();
    substrate
        .payloads()
        .set_reader(Arc::new(SealedReader::offline(store.to_path_buf())));
    for (id, path) in segs {
        let mut log = LogFile::open_read_only(path)?;
        let (entries, _) = walker::collect_filtered(
            &mut log,
            SegmentId(*id),
            SUPERBLOCK_SIZE,
            walk_reads_payload,
        )?;
        for e in &entries {
            substrate.apply_walker_entry(e);
        }
        eprintln!("walked segment {id}: {} records", entries.len());
    }
    Ok(substrate)
}

/// The census over every live chunk: each stream's chunk locations, read back
/// from the segment that holds them.
fn formats(substrate: &Substrate, segs: &[(u64, PathBuf)]) -> Result<Census> {
    let mut logs: HashMap<u64, LogFile> = HashMap::new();
    for (id, path) in segs {
        logs.insert(*id, LogFile::open_read_only(path)?);
    }
    let mut census = Census::default();
    for (_, stream) in substrate.all_streams() {
        for loc in stream.chunks.values() {
            let log = logs
                .get_mut(&loc.segment.0)
                .with_context(|| format!("no segment {} for a live chunk", loc.segment.0))?;
            let rec = read_record_at(log, loc.offset, loc.record_size)?;
            let (meta, kv) = ChunkPayload::decode_with_kv_range(&rec.payload)?;
            let kv_len = kv.len() as u64;
            census.add(&ChunkShape {
                tokens: loc.token_count,
                k_formats: &meta.k_formats,
                v_formats: &meta.v_formats,
                kv_len,
                side_len: rec.payload.len() as u64 - kv_len,
            });
        }
    }
    Ok(census)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let segs = segments(&cli.store)?;
    let substrate = substrate_of(&cli.store, &segs)?;
    let census = formats(&substrate, &segs)?;
    eprintln!(
        "{} chunks, {} tokens: {:.2} GiB of K/V standing for {:.2} GiB at BF16 ({:.2}x); \
         {} unresolved",
        census.chunks,
        census.tokens,
        (census.kv_bytes + census.side_bytes) as f64 / (1u64 << 30) as f64,
        census.bf16_bytes as f64 / (1u64 << 30) as f64,
        census.ratio(),
        census.unresolved
    );
    println!("{}", serde_json::to_string_pretty(&census)?);
    Ok(())
}
