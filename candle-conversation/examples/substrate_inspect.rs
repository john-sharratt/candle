//! `substrate_inspect` — a read-only inspector for a substrate redo log.
//!
//! The substrate persistence layer (`docs/kv_tier_migration.md`) stores a
//! conversation workspace as an append-only log of content-addressed
//! records under `<workspace>/.substrate/substrate.log`. This tool opens
//! such a log read-only and renders it from several angles.
//!
//! ```text
//! cargo run -p candle-conversation --example substrate_inspect -- \
//!     .substrate/substrate.log summary
//!
//! Commands:
//!   summary               file + superblock overview, record histogram, live/dead
//!   headers               every record in append order (offset/type/ids/sizes)
//!   streams               per-stream manifest (turns + prompt sections)
//!   chunks  <stream-id>   KV chunk records for a stream (format, sizes, bytes)
//!   tokens  <stream-id>   decode a stream's Tokens record to token ids
//!   meta                  the live ModelSpec / Template payloads
//!   checkpoint            latest checkpoint + what it recovers to
//! ```
//!
//! `<stream-id>` accepts decimal or `0x`-prefixed hex (as printed by
//! `streams`).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use candle_conversation::persistence::checkpoint;
use candle_conversation::persistence::compaction;
use candle_conversation::persistence::log_file::{read_record_at, LogFile, SUPERBLOCK_SIZE};
use candle_conversation::persistence::manifest::Manifest;
use candle_conversation::persistence::record::{ChunkPayload, Record, RecordType};
use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{StreamDecl, StreamId};
use candle_conversation::persistence::walker;
use candle_nn::kv_cache::KvFormat;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(
    name = "substrate_inspect",
    about = "Read-only inspector for a substrate redo log"
)]
struct Cli {
    /// Path to the substrate log. Defaults to `.substrate/substrate.log`
    /// under the current directory (the workspace's redo log).
    #[arg(short, long, global = true)]
    log: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

/// The default log location — `<cwd>/.substrate/substrate.log`, where the
/// daemon writes the workspace redo log.
fn default_log_path() -> PathBuf {
    use candle_conversation::persistence::{ACTIVE_LOG_NAME, SUBSTRATE_DIR};
    PathBuf::from(SUBSTRATE_DIR).join(ACTIVE_LOG_NAME)
}

#[derive(Subcommand)]
enum Cmd {
    /// File + superblock overview, record histogram, live/dead ratio.
    Summary,
    /// Every record in append order: offset, type, ids, sizes.
    Headers,
    /// Per-stream manifest view (turns and prompt sections).
    Streams,
    /// KV chunk records for one stream.
    Chunks {
        /// Stream id, decimal or `0x`-hex.
        stream_id: String,
        /// Bytes of each chunk's KV blob to hex-preview (0 = none).
        #[arg(long, default_value_t = 16)]
        preview: usize,
    },
    /// Show one stream's `Tokens` record — decoded text by default, using the
    /// tokenizer embedded in the log's `Tokenizer` record.
    Tokens {
        /// Stream id, decimal or `0x`-hex.
        stream_id: String,
        /// Print raw numeric token ids instead of decoded text.
        #[arg(long)]
        ids: bool,
    },
    /// The live `ModelSpec` / `Template` payloads.
    Meta,
    /// Latest checkpoint and the manifest it recovers.
    Checkpoint,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let log_path = cli.log.clone().unwrap_or_else(default_log_path);
    let mut log = LogFile::open(&log_path).with_context(|| {
        format!(
            "opening log {} (pass --log to point elsewhere)",
            log_path.display()
        )
    })?;

    match cli.cmd {
        Cmd::Summary => summary(&log_path, &mut log)?,
        Cmd::Headers => headers(&mut log)?,
        Cmd::Streams => streams(&mut log)?,
        Cmd::Chunks { stream_id, preview } => chunks(&mut log, parse_stream_id(&stream_id)?, preview)?,
        Cmd::Tokens { stream_id, ids } => {
            tokens(&mut log, &log_path, parse_stream_id(&stream_id)?, ids)?
        }
        Cmd::Meta => meta(&mut log, &log_path)?,
        Cmd::Checkpoint => checkpoint_view(&mut log)?,
    }
    Ok(())
}

// ── Views ───────────────────────────────────────────────────────────────────

fn summary(path: &std::path::Path, log: &mut LogFile) -> Result<()> {
    let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let sb = log.superblock();
    let (entries, _) = walker::collect(log, SUPERBLOCK_SIZE)?;

    // Record-type histogram.
    let mut counts = [0usize; 9];
    let mut payload_bytes = 0u64;
    for e in &entries {
        counts[type_index(e.record.header.record_type)] += 1;
        payload_bytes += e.record.header.payload_len;
    }

    let manifest = build_manifest(log)?;
    let (turns, sections) = stream_kind_counts(&manifest);
    let live_chunks: usize = manifest.streams.values().map(|s| s.chunks.len()).sum();
    let dead = compaction::dead_record_ratio(log, &manifest)?;

    println!("file              {}", path.display());
    println!("size              {file_len} bytes ({} KiB)", file_len / 1024);
    println!("format version    {}", sb.format_version);
    println!(
        "latest checkpoint {}",
        if sb.latest_checkpoint_offset == 0 {
            "none".to_string()
        } else {
            format!("offset {}", sb.latest_checkpoint_offset)
        }
    );
    println!();
    println!("records           {} ({} payload bytes)", entries.len(), payload_bytes);
    for rt in ALL_TYPES {
        let n = counts[type_index(rt)];
        if n > 0 {
            println!("  {:<12} {n}", format!("{rt:?}"));
        }
    }
    println!();
    println!("streams           {} ({turns} turn, {sections} prompt-section)", manifest.streams.len());
    println!("live chunks       {live_chunks}");
    println!("model spec        {}", present(manifest.model_spec.is_some()));
    println!("template          {}", present(manifest.template.is_some()));
    println!(
        "tokenizer         {}",
        match manifest.tokenizer {
            Some(_) => "hash-only (32 bytes — sidecar holds the JSON)".to_string(),
            None => "no".to_string(),
        }
    );
    let sidecar_path = path
        .parent()
        .map(|p| p.join("tokenizer.json"))
        .unwrap_or_else(|| PathBuf::from("tokenizer.json"));
    let sidecar_status = match std::fs::metadata(&sidecar_path) {
        Ok(m) => format!(
            "{} ({:.1} MB)",
            sidecar_path.display(),
            m.len() as f64 / 1_048_576.0
        ),
        Err(_) => format!("{} (missing)", sidecar_path.display()),
    };
    println!("tokenizer sidecar {sidecar_status}");
    println!("dead-record ratio {:.1}% {}", dead * 100.0, compaction_hint(dead));
    Ok(())
}

fn headers(log: &mut LogFile) -> Result<()> {
    let (entries, outcome) = walker::collect(log, SUPERBLOCK_SIZE)?;
    println!(
        "{:>5}  {:>10}  {:<11}  {:>18}  {:>7}  {:>4}  {:>7}  {:>9}",
        "#", "offset", "type", "stream_id", "chunk", "fmt", "tokens", "payload"
    );
    for (i, e) in entries.iter().enumerate() {
        let h = &e.record.header;
        println!(
            "{i:>5}  {:>10}  {:<11}  {:>18}  {:>7}  {:>4}  {:>7}  {:>9}",
            e.offset,
            format!("{:?}", h.record_type),
            stream_hex(h.stream_id),
            h.chunk_index,
            h.format,
            h.token_count,
            h.payload_len,
        );
    }
    println!("\n{} records walked", outcome.records);
    Ok(())
}

fn streams(log: &mut LogFile) -> Result<()> {
    let manifest = build_manifest(log)?;
    if manifest.streams.is_empty() {
        println!("(no streams)");
        return Ok(());
    }
    // `manifest.streams` is keyed by the stream-id hash, which is unrelated to
    // time. Order by append order instead — the offset of each stream's first
    // record in the log — so the listing reads oldest-first, newest-last.
    let first_seen = first_seen_offsets(log)?;
    let mut ids: Vec<&StreamId> = manifest.streams.keys().collect();
    ids.sort_by_key(|id| first_seen.get(*id).copied().unwrap_or(u64::MAX));

    for (n, id) in ids.into_iter().enumerate() {
        let entry = &manifest.streams[id];
        let tok = if entry.tokens.is_some() { "y" } else { "n" };
        let sig = if entry.signatures.is_some() { "y" } else { "n" };
        let detail = match &entry.decl {
            Some(StreamDecl::Turn(t)) => format!(
                "exchange  idx={} chunks={} tok={} sig={}  conv={}",
                t.turn_index,
                entry.chunks.len(),
                tok,
                sig,
                t.timeline_id,
            ),
            Some(StreamDecl::PromptSection(s)) => format!(
                "section \"{}\"  chunks={} tok={} sig={}",
                s.debug_name,
                entry.chunks.len(),
                tok,
                sig,
            ),
            None => format!("(no decl)  chunks={}", entry.chunks.len()),
        };
        println!("[{n}] {}  {detail}", stream_hex(id.0));
    }
    Ok(())
}

/// The log offset where each stream's first record appears. Since the walker
/// returns records in ascending offset (append) order, the first occurrence
/// of a `stream_id` is its creation point — the chronological key.
fn first_seen_offsets(log: &mut LogFile) -> Result<std::collections::HashMap<StreamId, u64>> {
    let (entries, _) = walker::collect(log, SUPERBLOCK_SIZE)?;
    let mut first = std::collections::HashMap::new();
    for e in &entries {
        let sid = e.record.header.stream_id;
        if sid != 0 {
            first.entry(StreamId(sid)).or_insert(e.offset);
        }
    }
    Ok(first)
}

fn chunks(log: &mut LogFile, stream_id: StreamId, preview: usize) -> Result<()> {
    let manifest = build_manifest(log)?;
    let entry = manifest
        .streams
        .get(&stream_id)
        .with_context(|| format!("no stream {} in the log", stream_hex(stream_id.0)))?;
    if entry.chunks.is_empty() {
        println!("stream {} has no chunk records", stream_hex(stream_id.0));
        return Ok(());
    }
    let locs: Vec<u64> = entry.chunks.values().map(|l| l.offset).collect();
    let n_chunks = locs.len();

    // Accumulate the per-sub-band format distribution across every chunk, plus
    // a few coarse totals — the compressed view of how the turn is quantized.
    let mut k_dist: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut v_dist: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut total_kv = 0usize;
    let mut sub_bands = 0usize;
    let (mut full, mut partial) = (0usize, 0usize);
    let mut first_kv: Option<Vec<u8>> = None;
    for (i, &offset) in locs.iter().enumerate() {
        let rec = read_record_at(log, offset)?;
        let p = ChunkPayload::decode(&rec.payload)?;
        if i == 0 {
            sub_bands = p.k_formats.len();
        }
        for &t in &p.k_formats {
            *k_dist.entry(t).or_default() += 1;
        }
        for &t in &p.v_formats {
            *v_dist.entry(t).or_default() += 1;
        }
        total_kv += p.kv_bytes.len();
        if rec.header.token_count >= candle_nn::CHUNK_SIZE as u64 {
            full += 1;
        } else {
            partial += 1;
        }
        if preview > 0 && i == 0 {
            first_kv = Some(p.kv_bytes);
        }
    }

    println!(
        "stream {}  ({n_chunks} chunks, {sub_bands} sub-bands/chunk, {:.1} MB KV)",
        stream_hex(stream_id.0),
        total_kv as f64 / 1_048_576.0,
    );
    println!("  full chunks: {full}   partial-tail chunks: {partial}\n");
    print_fmt_distribution("K format distribution", &k_dist);
    print_fmt_distribution("V format distribution", &v_dist);
    if let Some(kv) = first_kv {
        println!("\nchunk[0] kv: {}", hex_preview(&kv, preview));
    }
    Ok(())
}

/// Print a format-tag → count distribution, most common first, with percent.
fn print_fmt_distribution(label: &str, dist: &std::collections::HashMap<u8, usize>) {
    let total: usize = dist.values().sum();
    println!("{label} ({total} sub-bands):");
    let mut rows: Vec<(u8, usize)> = dist.iter().map(|(&t, &c)| (t, c)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (tag, count) in rows {
        let name = KvFormat::from_tag(tag)
            .map(|f| format!("{f:?}"))
            .unwrap_or_else(|| format!("tag{tag}?"));
        let pct = if total > 0 {
            count as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        println!("  {name:<22} {count:>9}  ({pct:>5.1}%)");
    }
}

fn tokens(
    log: &mut LogFile,
    log_path: &std::path::Path,
    stream_id: StreamId,
    as_ids: bool,
) -> Result<()> {
    let manifest = build_manifest(log)?;
    let loc = manifest
        .streams
        .get(&stream_id)
        .and_then(|s| s.tokens)
        .with_context(|| format!("stream {} has no Tokens record", stream_hex(stream_id.0)))?;
    let rec = read_record_at(log, loc.offset)?;
    let ids = decode_token_ids(&rec.payload)?;
    println!("stream {}  ({} tokens)", stream_hex(stream_id.0), ids.len());

    if as_ids {
        println!("{ids:?}");
        return Ok(());
    }

    // Default: decode to text using the tokenizer sidecar next to the log.
    // Special tokens are kept so chat-format markers (`<|im_start|>` …) stay
    // visible — useful for inspecting a turn.
    let _ = &manifest; // manifest unused now that the tokenizer lives in a sidecar
    match load_log_tokenizer(log_path)? {
        Some(tok) => {
            let text = tok
                .decode(&ids, false)
                .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
            println!("{text}");
        }
        None => {
            eprintln!(
                "note: no `tokenizer.json` sidecar next to the log — showing raw ids. \
                 Use --ids to silence this."
            );
            println!("{ids:?}");
        }
    }
    Ok(())
}

/// Load the tokenizer from the sidecar file next to the active log
/// (`<log_dir>/tokenizer.json`). Returns `Ok(None)` when the sidecar is
/// missing — the substrate is then opaque to text decoding and the
/// caller should fall back to `--ids`.
fn load_log_tokenizer(log_path: &std::path::Path) -> Result<Option<Tokenizer>> {
    let sidecar = log_path
        .parent()
        .map(|p| p.join("tokenizer.json"))
        .unwrap_or_else(|| PathBuf::from("tokenizer.json"));
    let bytes = match std::fs::read(&sidecar) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).context(format!("read tokenizer sidecar {}", sidecar.display())),
    };
    let tok = Tokenizer::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("load tokenizer from {}: {e}", sidecar.display()))?;
    Ok(Some(tok))
}

fn meta(log: &mut LogFile, log_path: &std::path::Path) -> Result<()> {
    let manifest = build_manifest(log)?;
    print_singleton(log, "model spec", manifest.model_spec.map(|l| l.offset))?;
    print_singleton(log, "template", manifest.template.map(|l| l.offset))?;
    // Tokenizer bytes live in a sidecar; the record itself is just the
    // 32-byte SHA-256 digest.
    match manifest.tokenizer {
        None => println!("tokenizer   (none)"),
        Some(l) => {
            let sidecar = log_path
                .parent()
                .map(|p| p.join("tokenizer.json"))
                .unwrap_or_else(|| PathBuf::from("tokenizer.json"));
            let sidecar_size = std::fs::metadata(&sidecar)
                .ok()
                .map(|m| m.len())
                .unwrap_or(0);
            println!(
                "tokenizer   record {} bytes (hash-only) ;  sidecar {} ({} bytes)",
                l.payload_len,
                sidecar.display(),
                sidecar_size,
            );
        }
    }
    Ok(())
}

fn checkpoint_view(log: &mut LogFile) -> Result<()> {
    let hint = log.superblock().latest_checkpoint_offset;
    if hint == 0 {
        println!("no checkpoint has been written to this log");
        return Ok(());
    }
    println!("latest checkpoint at offset {hint}");
    let recovered = checkpoint::recover(log, hint)?;
    let (turns, sections) = stream_kind_counts(&recovered.manifest);
    println!(
        "recovers to: {} streams ({turns} turn, {sections} section), torn-tail={}",
        recovered.manifest.streams.len(),
        recovered.torn,
    );
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const ALL_TYPES: [RecordType; 9] = [
    RecordType::ModelSpec,
    RecordType::Template,
    RecordType::StreamDecl,
    RecordType::Chunk,
    RecordType::Tokens,
    RecordType::Signatures,
    RecordType::Commit,
    RecordType::Checkpoint,
    RecordType::Tokenizer,
];

fn type_index(rt: RecordType) -> usize {
    rt as usize - 1
}

fn build_manifest(log: &mut LogFile) -> Result<Manifest> {
    Ok(Manifest::build_from_walk(log, SUPERBLOCK_SIZE)?.0)
}

fn stream_kind_counts(manifest: &Manifest) -> (usize, usize) {
    let mut turns = 0;
    let mut sections = 0;
    for s in manifest.streams.values() {
        match &s.decl {
            Some(StreamDecl::Turn(_)) => turns += 1,
            Some(StreamDecl::PromptSection(_)) => sections += 1,
            None => {}
        }
    }
    (turns, sections)
}

fn print_singleton(log: &mut LogFile, label: &str, offset: Option<u64>) -> Result<()> {
    match offset {
        None => println!("{label:<11} (none)"),
        Some(off) => {
            let rec = read_record_at(log, off)?;
            println!("{label:<11} {} bytes", rec.payload.len());
            println!("            {}", payload_render(&rec));
        }
    }
    Ok(())
}

/// Render a payload as UTF-8 when it is printable, otherwise as a hex preview.
fn payload_render(rec: &Record) -> String {
    match std::str::from_utf8(&rec.payload) {
        Ok(s) if s.chars().all(|c| !c.is_control() || c == '\n' || c == '\t') => s.to_string(),
        _ => hex_preview(&rec.payload, 48),
    }
}

fn hex_preview(bytes: &[u8], max: usize) -> String {
    let shown: Vec<String> = bytes.iter().take(max).map(|b| format!("{b:02x}")).collect();
    let mut s = shown.join(" ");
    if bytes.len() > max {
        s.push_str(&format!(" … (+{} more)", bytes.len() - max));
    }
    s
}

fn stream_hex(id: u64) -> String {
    if id == 0 {
        "0".to_string()
    } else {
        format!("0x{id:016x}")
    }
}

fn parse_stream_id(s: &str) -> Result<StreamId> {
    let raw = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)
    } else {
        s.parse::<u64>()
    }
    .with_context(|| format!("invalid stream id '{s}' (use decimal or 0x-hex)"))?;
    Ok(StreamId(raw))
}

fn present(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn compaction_hint(ratio: f32) -> &'static str {
    if ratio >= 0.5 {
        "(compaction would reclaim significant space)"
    } else {
        ""
    }
}
