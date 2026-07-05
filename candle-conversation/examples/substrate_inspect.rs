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
//!   sections              prompt sections grouped by name, with per-branch
//!                         content address + KV fingerprint (verifies a
//!                         section-tree's branch variants were sealed distinctly)
//!   chunks  <stream-id>   KV chunk records for a stream (format, sizes, bytes)
//!   tokens  <stream-id>   decode a stream's Tokens record to token ids
//!   meta                  the live ModelSpec / Template payloads
//!   tool-summary          the cached tool-catalog summary (hash + text)
//!   checkpoint            latest checkpoint + what it recovers to
//!   projections [stream]  per-decode projection composition (the GUI panel data)
//!   tree                  per-timeline summary forest from TreeMetadata records
//! ```
//!
//! `<stream-id>` accepts decimal or `0x`-prefixed hex (as printed by
//! `streams`).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use std::collections::BTreeMap;

use candle_conversation::persistence::checkpoint;
use candle_conversation::persistence::compaction;
use candle_conversation::persistence::content_hash::ContentHash;
use candle_conversation::persistence::log_file::{read_record_at, LogFile, SUPERBLOCK_SIZE};
use candle_conversation::persistence::manifest::Manifest;
use candle_conversation::persistence::record::{
    ChunkPayload, Record, RecordType, ToolSummaryPayload,
};
use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{ContentAddress, StreamDecl, StreamId, TurnDecl};
use candle_conversation::persistence::walker;
use candle_conversation::projection::{
    decode_events, ProjectionEvent, SystemItem, TimelineId, TurnIndex,
};
use candle_conversation::substrate::{StreamRuntime, Substrate};
use candle_conversation::summary_tree::TurnKind;
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
    /// Prompt sections grouped by name, each variant with its content address
    /// (prefix + section hash) and a KV fingerprint. A section-tree's branch
    /// variants share a name and `section_hash` but differ in `prefix_hash` and
    /// KV — this view proves that directly from the persisted bytes.
    Sections,
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
    /// The cached tool-catalog summary (the `ToolSummary` singleton): its
    /// catalog hash and the full generated text.
    ToolSummary,
    /// Latest checkpoint and the manifest it recovers.
    Checkpoint,
    /// Per-decode projection composition — the same data the GUI's projection
    /// panel draws: for each decoded turn, the throughput plus what provenance
    /// materialized (token buckets) and which system-prompt sections it selected
    /// vs skipped and which conversation turns it pulled in.
    Projections {
        /// One turn stream (decimal or `0x`-hex). Omit to dump every decoded turn.
        stream_id: Option<String>,
    },
    /// Per-timeline summary forest reconstructed from `TreeMetadata` records:
    /// kind / level / children for every summary node, with peaks flagged.
    Tree {
        /// Restrict to one timeline (raw u64 id, as printed in the header).
        #[arg(long)]
        timeline: Option<u64>,
        /// Decode each summary node's text (and, for a SoT leaf, the source
        /// turn it compresses) so faithfulness can be eyeballed.
        #[arg(long)]
        text: bool,
    },
    /// Per-turn structural audit — the "is the user half actually in the K/V?"
    /// view.  For every turn it cross-references the persisted token_ids length
    /// against the summed chunk `token_count` (the real sealed-KV token count),
    /// the block span, and the content bounds (`user_content_start/end`,
    /// `assistant_content_start`).  Flags any turn whose sealed K/V is shorter
    /// than its token_ids — i.e. the user-message tokens never made it into the
    /// arena, so reprojection re-injects an assistant-only turn.
    TurnAudit {
        /// Restrict to one timeline (raw u64 id, as printed by `streams`).
        #[arg(long)]
        timeline: Option<u64>,
        /// Also decode the user/assistant body slices via the content bounds.
        #[arg(long)]
        text: bool,
    },
    /// Combined linear dump of a whole conversation in one pass: every turn in
    /// append (chronological) order with its forest kind (normal / SoT / SoS),
    /// `no_think`, token + per-layer KV counts, decoded user/assistant text, and
    /// the projection events that ran on it. The union of `streams` + `tree` +
    /// `turn-audit` + `tokens` + `projections`, so a conversation reads
    /// top-to-bottom without stitching several slow commands together.
    Dump {
        /// Restrict to one timeline (raw u64 id, as printed by `streams`). Omit
        /// to dump every conversation.
        #[arg(long)]
        timeline: Option<u64>,
        /// Untruncated text + every projection's full section/turn selection.
        #[arg(long)]
        full: bool,
    },
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
        Cmd::Sections => sections(&mut log, &log_path)?,
        Cmd::Chunks { stream_id, preview } => {
            chunks(&mut log, parse_stream_id(&stream_id)?, preview)?
        }
        Cmd::Tokens { stream_id, ids } => {
            tokens(&mut log, &log_path, parse_stream_id(&stream_id)?, ids)?
        }
        Cmd::Projections { stream_id } => {
            let only = stream_id.as_deref().map(parse_stream_id).transpose()?;
            projections(&mut log, only)?
        }
        Cmd::Meta => meta(&mut log, &log_path)?,
        Cmd::ToolSummary => tool_summary(&mut log)?,
        Cmd::Checkpoint => checkpoint_view(&mut log)?,
        Cmd::Tree { timeline, text } => tree(&mut log, &log_path, timeline, text)?,
        Cmd::TurnAudit { timeline, text } => turn_audit(&mut log, &log_path, timeline, text)?,
        Cmd::Dump { timeline, full } => dump(&mut log, timeline, full)?,
    }
    Ok(())
}

/// Render the per-decode projection composition for one or all decoded turns —
/// the same materialized-context account the GUI's projection panel shows. For
/// each turn we print every `ProjectionEvent` (one per reprojection span): the
/// decode throughput, the token buckets, the system-prompt sections provenance
/// selected vs skipped, and the conversation turns it pulled into the window.
fn projections(log: &mut LogFile, only: Option<StreamId>) -> Result<()> {
    let substrate = build_substrate(log)?;
    let first_seen = first_seen_offsets(log)?;
    let mut turns: Vec<(StreamId, u64, u32)> = substrate
        .all_streams()
        .filter_map(|(id, e)| match &e.decl {
            Some(StreamDecl::Turn(t)) => Some((id, t.timeline_id, t.turn_index)),
            _ => None,
        })
        .filter(|(id, _, _)| only.map_or(true, |o| *id == o))
        .collect();
    turns.sort_by_key(|(id, _, _)| first_seen.get(id).copied().unwrap_or(u64::MAX));

    if turns.is_empty() {
        println!(
            "(no turn streams{})",
            if only.is_some() {
                " matching that id"
            } else {
                ""
            }
        );
        return Ok(());
    }

    let mut any = false;
    for (id, timeline, idx) in turns {
        let Some(tl) = TimelineId::from_raw(timeline) else {
            continue;
        };
        let Some(blob) = substrate.projection_events_blob(tl, TurnIndex(idx)) else {
            continue;
        };
        let events = decode_events(blob);
        if events.is_empty() {
            continue;
        }
        any = true;
        println!(
            "\n══ turn {timeline}#{idx}  (stream {})  — {} projection event(s)",
            stream_hex(id.0),
            events.len()
        );
        for (i, ev) in events.iter().enumerate() {
            print_projection_event(i, ev);
        }
    }
    if !any {
        println!(
            "(no projection events recorded — these are written per decoded dialogue turn; \
             section / utility ingests don't emit them)"
        );
    }
    Ok(())
}

/// One `ProjectionEvent` — a projection selected at a POINT in the decode
/// (`start_token` is the generated-token position; it governs everything forward
/// until the next event). Prints the point, its token buckets, and the
/// per-section selected/skipped breakdown.
fn print_projection_event(i: usize, ev: &ProjectionEvent) {
    println!(
        "  proj #{i}: @token {} (t={:.2}s)   materialized {} / substrate {} tokens",
        ev.start_token, ev.seconds, ev.materialized_tokens, ev.substrate_tokens,
    );
    if !ev.buckets.is_empty() {
        let parts: Vec<String> = ev
            .buckets
            .iter()
            .map(|b| format!("{}={}", b.label, b.tokens))
            .collect();
        println!("      buckets: {}", parts.join("  "));
    }
    for item in &ev.selection.system {
        match item {
            SystemItem::Glue { name, tokens, .. } => {
                println!("      [glue]    {name} ({tokens} tok)")
            }
            SystemItem::Section { name, tokens } => {
                println!("      [section] {name} ({tokens} tok)")
            }
            SystemItem::Collection { name, sections } => {
                let sel = sections.iter().filter(|s| s.selected).count();
                println!(
                    "      [collect] {name}: {sel}/{} sections selected",
                    sections.len()
                );
                for s in sections {
                    println!(
                        "          {} {} ({} tok)",
                        if s.selected { "[x]" } else { "[ ]" },
                        s.name,
                        s.tokens
                    );
                }
            }
        }
    }
    if !ev.selection.turns.is_empty() {
        println!("      turns selected:  (src = raw turn vs SUMMARY node; why = selection origin)");
        for t in &ev.selection.turns {
            // The decisive column: was this slot filled with a real turn, or a
            // summary node standing in for the turns beneath it?
            let src = match t.kind {
                TurnKind::Normal => "turn",
                TurnKind::SummaryOfTurns => "SUMMARY(SoT)",
                TurnKind::SummaryOfSummaries => "SUMMARY(SoS)",
            };
            let why = t
                .reason
                .map(|r| format!("{r:?}"))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "          {}/{} #{} {} {:<12} ({} tok)  why={}",
                t.layer, t.group, t.index, t.role, src, t.tokens, why
            );
        }
    }
}

/// Render the per-timeline summary tree reconstructed from `TreeMetadata`
/// records. With `with_text`, decode each summary node (and a SoT's source
/// turn) using the tokenizer sidecar so faithfulness is visible.
fn tree(
    log: &mut LogFile,
    log_path: &std::path::Path,
    only_timeline: Option<u64>,
    with_text: bool,
) -> Result<()> {
    let substrate = build_substrate(log)?;
    let tok = if with_text {
        load_log_tokenizer(log_path)?
    } else {
        None
    };

    // A raw walker pass populates `streams` + the tree metadata (via
    // `set_tree_meta`), but NOT the per-timeline turn registry (that needs
    // `register_timeline`, whose layer/group map isn't in the log). So enumerate
    // turns from the turn streams, query the tree meta off the substrate, and
    // grab each turn's Tokens-record location for on-demand decode.
    struct NodeInfo {
        idx: u32,
        kind: TurnKind,
        children: Vec<u32>,
        height: u8,
    }
    let mut by_tl: std::collections::BTreeMap<u64, Vec<NodeInfo>> =
        std::collections::BTreeMap::new();
    // Peak set (orphan summary nodes = window entry points) per timeline.
    let mut peaks: std::collections::BTreeMap<u64, std::collections::BTreeSet<u32>> =
        std::collections::BTreeMap::new();
    let mut tokens_loc: std::collections::HashMap<(u64, u32), (u64, u64)> =
        std::collections::HashMap::new();

    for (_sid, entry) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &entry.decl else {
            continue;
        };
        let Some(tl) = TimelineId::from_raw(t.timeline_id) else {
            continue;
        };
        let (kind, children, height) = match substrate.tree_meta_of(tl, TurnIndex(t.turn_index)) {
            Some(m) => (
                m.kind,
                m.children.iter().map(|c| c.0).collect(),
                m.tree_height,
            ),
            None => (TurnKind::Normal, Vec::new(), 0),
        };
        by_tl.entry(t.timeline_id).or_default().push(NodeInfo {
            idx: t.turn_index,
            kind,
            children,
            height,
        });
        peaks.entry(t.timeline_id).or_insert_with(|| {
            substrate
                .peaks_of(tl)
                .into_iter()
                .map(|(idx, _)| idx.0)
                .collect()
        });
        if let Some(l) = entry.tokens {
            tokens_loc.insert((t.timeline_id, t.turn_index), (l.offset, l.record_size));
        }
    }
    drop(substrate);

    if by_tl.is_empty() {
        println!("(no turn streams in substrate)");
        return Ok(());
    }

    for (tl_raw, mut nodes) in by_tl {
        if let Some(want) = only_timeline {
            if tl_raw != want {
                continue;
            }
        }
        nodes.sort_by_key(|n| n.idx);
        let n_normal = nodes
            .iter()
            .filter(|n| matches!(n.kind, TurnKind::Normal))
            .count();
        let n_sot = nodes
            .iter()
            .filter(|n| matches!(n.kind, TurnKind::SummaryOfTurns))
            .count();
        let n_sos = nodes
            .iter()
            .filter(|n| matches!(n.kind, TurnKind::SummaryOfSummaries))
            .count();
        let tl_peaks = peaks.get(&tl_raw).cloned().unwrap_or_default();
        println!(
            "\n══ timeline {tl_raw} ── {} turns: {n_normal} normal, {n_sot} SoT, {n_sos} SoS  peaks={}",
            nodes.len(),
            tl_peaks.len(),
        );

        for n in &nodes {
            if matches!(n.kind, TurnKind::Normal) {
                continue;
            }
            let kind = match n.kind {
                TurnKind::SummaryOfTurns => "SoT",
                TurnKind::SummaryOfSummaries => "SoS",
                TurnKind::Normal => "?",
            };
            let mark = if tl_peaks.contains(&n.idx) {
                "  [PEAK]"
            } else {
                ""
            };
            println!(
                "  #{:<4} {kind}  h={}  children={:?}{mark}",
                n.idx, n.height, n.children
            );
            if let Some(t) = &tok {
                if let Some(text) = decode_turn(log, &tokens_loc, tl_raw, n.idx, t)? {
                    println!("        summary: {}", trunc(&text, 320));
                }
                if matches!(n.kind, TurnKind::SummaryOfTurns) {
                    for &c in &n.children {
                        if let Some(ctext) = decode_turn(log, &tokens_loc, tl_raw, c, t)? {
                            println!("        source #{c}: {}", trunc(&ctext, 320));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Per-turn structural audit (see [`Cmd::TurnAudit`]).  The decisive column is
/// `kv_tok` (summed chunk `token_count` = the real sealed-KV length) vs `n_tok`
/// (the persisted token_ids length).  When `kv_tok < n_tok` the arena is missing
/// the leading tokens; when `kv_tok ≈ n_tok − assistant_content_start` those
/// missing tokens are exactly the USER half — the turn was sealed assistant-only,
/// so reprojection re-injects a turn the model reads as having no user message.
fn turn_audit(
    log: &mut LogFile,
    log_path: &std::path::Path,
    only_timeline: Option<u64>,
    with_text: bool,
) -> Result<()> {
    let substrate = build_substrate(log)?;
    let first_seen = first_seen_offsets(log)?;
    let tok = load_log_tokenizer(log_path)?;

    struct TurnRec {
        id: StreamId,
        decl: TurnDecl,
        chunk_locs: Vec<(u64, u64)>, // (record offset, record size), chunk-index order
        tokens_loc: Option<(u64, u64)>,
    }
    let mut turns: Vec<TurnRec> = Vec::new();
    for (id, entry) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &entry.decl else {
            continue;
        };
        if only_timeline.is_some_and(|o| t.timeline_id != o) {
            continue;
        }
        turns.push(TurnRec {
            id,
            decl: t.clone(),
            chunk_locs: entry
                .chunks
                .values()
                .map(|l| (l.offset, l.record_size))
                .collect(),
            tokens_loc: entry.tokens.map(|l| (l.offset, l.record_size)),
        });
    }
    drop(substrate);
    turns.sort_by_key(|t| first_seen.get(&t.id).copied().unwrap_or(u64::MAX));

    if turns.is_empty() {
        println!("(no turn streams matching that filter)");
        return Ok(());
    }

    let role_name = |r: u8| match r {
        0 => "sys",
        1 => "user",
        2 => "asst",
        _ => "?",
    };
    println!(
        "{:<22} {:>4} {:>6} {:>6} {:>5} {:>5} {:>14} {:>10}   flags",
        "timeline#idx", "role", "n_tok", "kv_tok", "chnk", "blks", "uc[start..end)", "asst@"
    );
    for t in &turns {
        let d = &t.decl;
        let layout = candle_conversation::turn_layout::TurnLayout::new(d.segments.clone());
        let n_tok = match t.tokens_loc {
            Some((off, sz)) => decode_token_ids(&read_record_at(log, off, sz)?.payload)?.len(),
            None => 0,
        } as u64;
        let mut kv_tok = 0u64;
        for &(off, sz) in &t.chunk_locs {
            kv_tok += read_record_at(log, off, sz)?.header.token_count;
        }
        let blks = d.block_end.saturating_sub(d.block_start);
        // Chunks are stored per (block × layer); `kv_tok` sums all layers, so the
        // real KV length is per-layer.  `n_layers = chunks / blks`.
        let n_chunks = t.chunk_locs.len() as u64;
        let n_layers = if blks > 0 { n_chunks / blks } else { 0 };
        let kv_per_layer = if n_layers > 0 {
            kv_tok / n_layers
        } else {
            kv_tok
        };
        let asst = layout.assistant_content_start() as u64;
        let assistant_body = n_tok.saturating_sub(asst);

        let mut flags: Vec<String> = Vec::new();
        if kv_per_layer + 1 < n_tok {
            flags.push(format!("KV<token_ids by {}", n_tok - kv_per_layer));
        }
        // The smoking-gun check: per-layer sealed KV equals the assistant body
        // length, i.e. the user half ([0..assistant_content_start)) was NOT sealed.
        if asst > 0 && kv_per_layer + 1 < n_tok && kv_per_layer.abs_diff(assistant_body) <= 1 {
            flags.push("** ASSISTANT-ONLY: user half not sealed **".to_string());
        }
        if asst == 0 {
            flags.push("user-region-empty(asst@0)".to_string());
        }
        if n_layers > 0 && n_chunks % blks != 0 {
            flags.push(format!("chunks {n_chunks} not /blks {blks}"));
        }
        if flags.is_empty() {
            flags.push("ok".to_string());
        }

        println!(
            "{:<22} {:>4} {:>6} {:>7} {:>5} {:>5} {:>4} {:>13} {:>6}   {}",
            format!("{}#{}", d.timeline_id, d.turn_index),
            role_name(d.role),
            n_tok,
            kv_per_layer,
            t.chunk_locs.len(),
            blks,
            n_layers,
            format!(
                "[{}..{})",
                layout.user_content_start(),
                layout.user_content_end()
            ),
            asst,
            flags.join("; "),
        );
        if with_text {
            println!(
                "    view (selected context turns): {:?}   anchored_prefix={}",
                d.view,
                d.anchored_prefix.len()
            );
            let user_text = layout.user_text();
            let assistant_text = layout.assistant_text().unwrap_or_default();
            let utxt = trunc(user_text, 200);
            let atxt = trunc(&assistant_text, 500);
            println!("    user_text({:>3}): {utxt}", user_text.chars().count());
            println!(
                "    asst_text({:>3}): {atxt}",
                assistant_text.chars().count()
            );
            // Decode the leading `kv_tok` tokens to show what the SEALED K/V
            // actually starts with (does it open with the user message or the
            // assistant body?).
            if let (Some(t0), Some((off, sz))) = (tok.as_ref(), t.tokens_loc) {
                let ids = decode_token_ids(&read_record_at(log, off, sz)?.payload)?;
                let head: Vec<u32> = ids.iter().take(24).copied().collect();
                let head_txt = t0.decode(&head, false).unwrap_or_default();
                println!("    token_ids head: {}", trunc(&head_txt, 110));
            }
        }
    }
    Ok(())
}

/// Combined linear dump of a conversation — see [`Cmd::Dump`]. One metadata
/// scan builds the substrate; then each turn is printed in append order with its
/// forest kind, `no_think`, token/KV counts, decoded text, and projection
/// events. Replaces stitching `streams` + `tree` + `turn-audit` + `tokens` +
/// `projections` together across several slow full-log passes.
fn dump(
    log: &mut LogFile,
    only_timeline: Option<u64>,
    full: bool,
) -> Result<()> {
    let substrate = build_substrate(log)?;
    let first_seen = first_seen_offsets(log)?;

    struct TurnRec {
        id: StreamId,
        decl: TurnDecl,
        chunk_locs: Vec<(u64, u64)>,
        tokens_loc: Option<(u64, u64)>,
        kind: TurnKind,
        children: Vec<u32>,
        proj: Option<Vec<u8>>,
    }
    let mut turns: Vec<TurnRec> = Vec::new();
    for (id, entry) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &entry.decl else {
            continue;
        };
        if only_timeline.is_some_and(|o| t.timeline_id != o) {
            continue;
        }
        let tl = TimelineId::from_raw(t.timeline_id);
        let (kind, children) =
            match tl.and_then(|tl| substrate.tree_meta_of(tl, TurnIndex(t.turn_index))) {
                Some(m) => (m.kind, m.children.iter().map(|c| c.0).collect()),
                None => (TurnKind::Normal, Vec::new()),
            };
        let proj = tl
            .and_then(|tl| substrate.projection_events_blob(tl, TurnIndex(t.turn_index)))
            .map(|b| b.to_vec());
        turns.push(TurnRec {
            id,
            decl: t.clone(),
            chunk_locs: entry
                .chunks
                .values()
                .map(|l| (l.offset, l.record_size))
                .collect(),
            tokens_loc: entry.tokens.map(|l| (l.offset, l.record_size)),
            kind,
            children,
            proj,
        });
    }
    drop(substrate);
    turns.sort_by_key(|t| first_seen.get(&t.id).copied().unwrap_or(u64::MAX));

    if turns.is_empty() {
        println!("(no turn streams matching that filter)");
        return Ok(());
    }

    let role_name = |r: u8| match r {
        0 => "sys",
        1 => "user",
        2 => "asst",
        _ => "?",
    };
    let kind_str = |k: TurnKind, ch: &[u32]| match k {
        TurnKind::Normal => "NORMAL".to_string(),
        TurnKind::SummaryOfTurns => format!("SoT ←{ch:?}"),
        TurnKind::SummaryOfSummaries => format!("SoS ←{ch:?}"),
    };

    let mut cur_tl: Option<u64> = None;
    for t in &turns {
        let d = &t.decl;
        if cur_tl != Some(d.timeline_id) {
            cur_tl = Some(d.timeline_id);
            let n = turns
                .iter()
                .filter(|x| x.decl.timeline_id == d.timeline_id)
                .count();
            println!(
                "\n════════ conversation timeline {}  ({} turns) ════════",
                d.timeline_id, n
            );
        }
        let layout = candle_conversation::turn_layout::TurnLayout::new(d.segments.clone());
        let ids = match t.tokens_loc {
            Some((off, sz)) => decode_token_ids(&read_record_at(log, off, sz)?.payload)?,
            None => Vec::new(),
        };
        let n_tok = ids.len();
        let mut kv_tok = 0u64;
        for &(off, sz) in &t.chunk_locs {
            kv_tok += read_record_at(log, off, sz)?.header.token_count;
        }
        let blks = d.block_end.saturating_sub(d.block_start);
        let n_chunks = t.chunk_locs.len() as u64;
        let n_layers = if blks > 0 { n_chunks / blks } else { 0 };
        let kv_per_layer = if n_layers > 0 { kv_tok / n_layers } else { kv_tok };
        let events = t.proj.as_deref().map(decode_events).unwrap_or_default();

        println!(
            "\n── #{:<3} {:<12} {}  no_think={}  n_tok={} kv/layer={} chunks={}({}blk×{}L) proj={}",
            d.turn_index,
            kind_str(t.kind, &t.children),
            role_name(d.role),
            layout.no_think(),
            n_tok,
            kv_per_layer,
            n_chunks,
            blks,
            n_layers,
            events.len(),
        );
        let user_text = layout.user_text();
        let asst_text = layout.assistant_text().unwrap_or_default();
        let (umax, amax) = if full {
            (usize::MAX, usize::MAX)
        } else {
            (240, 500)
        };
        println!(
            "   user({:>3}): {}",
            user_text.chars().count(),
            trunc(user_text, umax)
        );
        println!(
            "   asst({:>3}): {}",
            asst_text.chars().count(),
            trunc(&asst_text, amax)
        );
        if full {
            for (i, ev) in events.iter().enumerate() {
                print_projection_event(i, ev);
            }
        } else {
            for (i, ev) in events.iter().enumerate() {
                let sel: Vec<String> = ev
                    .selection
                    .turns
                    .iter()
                    .map(|st| {
                        let tag = match st.kind {
                            TurnKind::Normal => "",
                            TurnKind::SummaryOfTurns => "(SoT)",
                            TurnKind::SummaryOfSummaries => "(SoS)",
                        };
                        format!("#{}{tag}", st.index)
                    })
                    .collect();
                println!(
                    "   proj #{i} @tok{} t={:.2}s  mat={}/sub={}  sel=[{}]",
                    ev.start_token,
                    ev.seconds,
                    ev.materialized_tokens,
                    ev.substrate_tokens,
                    sel.join(", ")
                );
            }
        }
    }
    Ok(())
}

/// Decode a turn's text from its Tokens record (looked up by `(timeline, idx)`).
fn decode_turn(
    log: &mut LogFile,
    tokens_loc: &std::collections::HashMap<(u64, u32), (u64, u64)>,
    timeline: u64,
    idx: u32,
    tok: &Tokenizer,
) -> Result<Option<String>> {
    let Some(&(off, size)) = tokens_loc.get(&(timeline, idx)) else {
        return Ok(None);
    };
    let rec = read_record_at(log, off, size)?;
    let ids = decode_token_ids(&rec.payload)?;
    let text = tok
        .decode(&ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
    Ok(Some(text))
}

/// Trim + truncate to `max` chars with an ellipsis.
fn trunc(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

// ── Views ───────────────────────────────────────────────────────────────────

fn summary(path: &std::path::Path, log: &mut LogFile) -> Result<()> {
    let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let sb = log.superblock();
    let (entries, _) = walker::collect(log, SUPERBLOCK_SIZE)?;

    // Record-type histogram. `type_index` maps each discriminant to
    // `rt as usize - 1`, so `Unknown` (the highest discriminant) maps to the
    // largest index; sizing off it keeps the array in step as record types are
    // added without re-counting by hand.
    let mut counts = [0usize; RecordType::Unknown as usize];
    let mut payload_bytes = 0u64;
    for e in &entries {
        counts[type_index(e.record.header.record_type)] += 1;
        payload_bytes += e.record.header.payload_len;
    }

    let manifest = build_manifest(log)?;
    let substrate = build_substrate(log)?;
    let (turns, sections) = stream_kind_counts(&substrate);
    let live_chunks: usize = substrate.all_streams().map(|(_, s)| s.chunks.len()).sum();
    let dead = compaction::dead_record_ratio(log, &manifest, &substrate)?;

    println!("file              {}", path.display());
    println!(
        "size              {file_len} bytes ({} KiB)",
        file_len / 1024
    );
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
    println!(
        "records           {} ({} payload bytes)",
        entries.len(),
        payload_bytes
    );
    for rt in ALL_TYPES {
        let n = counts[type_index(rt)];
        if n > 0 {
            println!("  {:<12} {n}", format!("{rt:?}"));
        }
    }
    println!();
    println!(
        "streams           {} ({turns} turn, {sections} prompt-section)",
        substrate.all_streams().count()
    );
    println!("live chunks       {live_chunks}");
    println!(
        "model spec        {}",
        present(manifest.model_spec.is_some())
    );
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
    println!(
        "dead-record ratio {:.1}% {}",
        dead * 100.0,
        compaction_hint(dead)
    );
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
    let substrate = build_substrate(log)?;
    let mut streams: Vec<(StreamId, &StreamRuntime)> = substrate.all_streams().collect();
    if streams.is_empty() {
        println!("(no streams)");
        return Ok(());
    }
    // The substrate's stream map is keyed by the stream-id hash, which is
    // unrelated to time. Order by append order instead — the offset of each
    // stream's first record in the log — so the listing reads oldest-first.
    let first_seen = first_seen_offsets(log)?;
    streams.sort_by_key(|(id, _)| first_seen.get(id).copied().unwrap_or(u64::MAX));

    for (n, (id, entry)) in streams.into_iter().enumerate() {
        let tok = if entry.tokens.is_some() { "y" } else { "n" };
        let sig = if entry.signatures.is_some() { "y" } else { "n" };
        let detail = match &entry.decl {
            Some(StreamDecl::Turn(t)) => {
                let no_think =
                    candle_conversation::turn_layout::TurnLayout::new(t.segments.clone())
                        .no_think();
                format!(
                    "exchange  idx={} chunks={} tok={} sig={}  no_think={}  conv={}",
                    t.turn_index,
                    entry.chunks.len(),
                    tok,
                    sig,
                    no_think,
                    t.timeline_id,
                )
            }
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
    // Headers only — first_seen needs each record's `(stream_id, offset)`, never
    // its payload, so skip every payload read.
    let (entries, _) = walker::collect_filtered(log, SUPERBLOCK_SIZE, |_| false)?;
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
    let substrate = build_substrate(log)?;
    let entry = substrate
        .stream_of(stream_id)
        .with_context(|| format!("no stream {} in the log", stream_hex(stream_id.0)))?;
    if entry.chunks.is_empty() {
        println!("stream {} has no chunk records", stream_hex(stream_id.0));
        return Ok(());
    }
    let locs: Vec<(u64, u64)> = entry
        .chunks
        .values()
        .map(|l| (l.offset, l.record_size))
        .collect();
    let n_chunks = locs.len();

    // Accumulate the per-sub-band format distribution across every chunk, plus
    // a few coarse totals — the compressed view of how the turn is quantized.
    let mut k_dist: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut v_dist: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut total_kv = 0usize;
    let mut sub_bands = 0usize;
    let (mut full, mut partial) = (0usize, 0usize);
    let mut first_kv: Option<Vec<u8>> = None;
    for (i, &(offset, record_size)) in locs.iter().enumerate() {
        let rec = read_record_at(log, offset, record_size)?;
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

/// Prompt sections grouped by name, each variant with its content address and a
/// KV fingerprint over its persisted chunks.
///
/// A section-tree's branch variants share a name and `section_hash` (identical
/// tokens) but were sealed under different prefixes, so they differ in
/// `prefix_hash` — and therefore in their actual K/V.  This view proves that
/// straight from the persisted bytes: it reads every variant's chunk records and
/// hashes the quantized `kv_bytes`, then flags whether the branches' content
/// matches while their prefix + KV diverge.
fn sections(log: &mut LogFile, log_path: &std::path::Path) -> Result<()> {
    let substrate = build_substrate(log)?;
    // Decode a short content preview per group so each branched node is
    // identifiable (`frame`, `thinking_effort.deep`, …) — the persisted
    // debug_name is only `section_<id>`.
    let tok = load_log_tokenizer(log_path)?;

    // Collect every prompt-section stream with its content address. Section
    // NAMES are per-layer (titler, repo_map, dialogue all declare a `frame`), so
    // the robust key for "branch variants of one node" is the SECTION HASH: same
    // tokens sealed under different prefixes. Group by that.
    struct Sec {
        id: StreamId,
        name: String,
        addr: ContentAddress,
    }
    let mut all: Vec<Sec> = Vec::new();
    for (id, entry) in substrate.all_streams() {
        if let Some(StreamDecl::PromptSection(s)) = &entry.decl {
            all.push(Sec {
                id,
                name: s.debug_name.clone(),
                addr: s.address,
            });
        }
    }
    if all.is_empty() {
        println!("(no prompt-section streams)");
        return Ok(());
    }

    let mut by_content: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (i, s) in all.iter().enumerate() {
        by_content
            .entry((s.addr.section_hash.hi, s.addr.section_hash.lo))
            .or_default()
            .push(i);
    }
    let branched = by_content.values().filter(|v| v.len() > 1).count();
    println!(
        "{} prompt-section streams, {} distinct contents, {} branched \
         (same content sealed under multiple prefixes)\n",
        all.len(),
        by_content.len(),
        branched,
    );

    // Detail every branched section — these are the section-tree nodes.
    for idxs in by_content.values() {
        if idxs.len() < 2 {
            continue;
        }
        let preview = stream_preview(log, &substrate, all[idxs[0]].id, &tok);
        println!(
            "content {}  ({} branch variants)   {preview}",
            hash_hex(all[idxs[0]].addr.section_hash),
            idxs.len(),
        );
        let mut kv_hashes = Vec::new();
        let mut prefixes = Vec::new();
        for &i in idxs {
            let s = &all[i];
            let entry = substrate.stream_of(s.id).expect("listed stream is present");
            let (kv_hash, chunks, kv_bytes) = fingerprint_stream(log, entry)?;
            kv_hashes.push(kv_hash);
            prefixes.push(s.addr.prefix_hash);
            println!(
                "   {:<14} {}  prefix={}  chunks={chunks:>3}  kv={kv_bytes:>8}B  kvhash=0x{kv_hash:016x}",
                format!("\"{}\"", s.name),
                stream_hex(s.id.0),
                hash_hex(s.addr.prefix_hash),
            );
        }
        let distinct_prefix = all_distinct(&prefixes);
        let distinct_kv = all_distinct(&kv_hashes);
        let verdict = if distinct_prefix && distinct_kv {
            "✓ branches sealed independently (distinct prefix ⇒ distinct K/V)"
        } else {
            "✗ unexpected — branches may have collapsed"
        };
        println!(
            "   → distinct prefix: {}   distinct KV: {}   {verdict}\n",
            yn(distinct_prefix),
            yn(distinct_kv),
        );
    }
    if branched == 0 {
        println!("(no branched sections — every prompt section has a single prefix)");
    }
    Ok(())
}

/// Decode a stream's first non-blank content line (truncated) from its `Tokens`
/// record, for an at-a-glance "which section is this".  Falls back to a token
/// count when no tokenizer sidecar is present.
fn stream_preview(
    log: &mut LogFile,
    substrate: &Substrate,
    sid: StreamId,
    tok: &Option<Tokenizer>,
) -> String {
    let Some(loc) = substrate.stream_of(sid).and_then(|s| s.tokens) else {
        return String::new();
    };
    let ids = match read_record_at(log, loc.offset, loc.record_size) {
        Ok(rec) => match decode_token_ids(&rec.payload) {
            Ok(ids) => ids,
            Err(_) => return String::new(),
        },
        Err(_) => return String::new(),
    };
    let Some(t) = tok else {
        return format!("({} tokens)", ids.len());
    };
    let text = t.decode(&ids, false).unwrap_or_default();
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let truncated: String = line.chars().take(64).collect();
    if line.chars().count() > 64 {
        format!("\"{truncated}…\"")
    } else {
        format!("\"{truncated}\"")
    }
}

/// FNV-1a over a stream's persisted chunk records — the quantized `kv_bytes`
/// plus palettes + scales, walked in chunk-index order.  Returns `(hash,
/// chunk_count, kv_byte_total)`.
fn fingerprint_stream(log: &mut LogFile, entry: &StreamRuntime) -> Result<(u64, usize, usize)> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64 offset basis
    let mut chunk_count = 0usize;
    let mut kv_total = 0usize;
    for loc in entry.chunks.values() {
        let rec = read_record_at(log, loc.offset, loc.record_size)?;
        let p = ChunkPayload::decode(&rec.payload)?;
        fnv_mix(&mut h, &p.kv_bytes);
        fnv_mix(&mut h, &p.k_pal);
        fnv_mix(&mut h, &p.v_pal);
        for s in &p.k_scale {
            fnv_mix(&mut h, &s.to_le_bytes());
        }
        for s in &p.v_scale {
            fnv_mix(&mut h, &s.to_le_bytes());
        }
        chunk_count += 1;
        kv_total += p.kv_bytes.len();
    }
    Ok((h, chunk_count, kv_total))
}

fn fnv_mix(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// 128-bit content hash as full `hi:lo` hex.
fn hash_hex(h: ContentHash) -> String {
    format!("{:016x}{:016x}", h.hi, h.lo)
}

fn all_distinct<T: Eq + std::hash::Hash + Clone>(xs: &[T]) -> bool {
    xs.iter()
        .cloned()
        .collect::<std::collections::HashSet<T>>()
        .len()
        == xs.len()
}

fn yn(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "NO"
    }
}

fn tokens(
    log: &mut LogFile,
    log_path: &std::path::Path,
    stream_id: StreamId,
    as_ids: bool,
) -> Result<()> {
    let substrate = build_substrate(log)?;
    let loc = substrate
        .stream_of(stream_id)
        .and_then(|s| s.tokens)
        .with_context(|| format!("stream {} has no Tokens record", stream_hex(stream_id.0)))?;
    let rec = read_record_at(log, loc.offset, loc.record_size)?;
    let ids = decode_token_ids(&rec.payload)?;
    println!("stream {}  ({} tokens)", stream_hex(stream_id.0), ids.len());

    if as_ids {
        println!("{ids:?}");
        return Ok(());
    }

    // Default: decode to text using the tokenizer sidecar next to the log.
    // Special tokens are kept so chat-format markers (`<|im_start|>` …) stay
    // visible — useful for inspecting a turn.
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
    print_singleton(
        log,
        "model spec",
        manifest.model_spec.map(|l| (l.offset, l.record_size)),
    )?;
    print_singleton(
        log,
        "template",
        manifest.template.map(|l| (l.offset, l.record_size)),
    )?;
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
    // The checkpoint payload carries only singleton offsets; streams are
    // rebuilt by replaying the tail through the substrate sink.
    let mut substrate = Substrate::new();
    let recovered = checkpoint::recover_with_sink(log, hint, |e| substrate.apply_walker_entry(e))?;
    let (turns, sections) = stream_kind_counts(&substrate);
    println!(
        "recovers to: {} streams ({turns} turn, {sections} section), torn-tail={}",
        substrate.all_streams().count(),
        recovered.torn,
    );
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const ALL_TYPES: [RecordType; 17] = [
    RecordType::ModelSpec,
    RecordType::Template,
    RecordType::StreamDecl,
    RecordType::Chunk,
    RecordType::Tokens,
    RecordType::Signatures,
    RecordType::Commit,
    RecordType::Checkpoint,
    RecordType::Tokenizer,
    RecordType::Label,
    RecordType::ConvState,
    RecordType::TreeMetadata,
    RecordType::DebugId,
    RecordType::Tombstone,
    RecordType::ProjectionEvents,
    RecordType::ToolSummary,
    RecordType::Unknown,
];

fn type_index(rt: RecordType) -> usize {
    rt as usize - 1
}

/// Decode and print the cached tool-catalog summary (the `ToolSummary`
/// singleton): its catalog hash and the full generated text.
fn tool_summary(log: &mut LogFile) -> Result<()> {
    let manifest = build_manifest(log)?;
    match manifest.tool_summary {
        None => {
            println!("tool summary  (none — not generated yet; restart the daemon to create it)");
        }
        Some(loc) => {
            let rec = read_record_at(log, loc.offset, loc.record_size)?;
            let payload = ToolSummaryPayload::decode(&rec.payload)
                .map_err(|e| anyhow::anyhow!("decode ToolSummary: {e}"))?;
            for (label, entry) in [
                ("comprehensive", &payload.comprehensive),
                ("restricted", &payload.restricted),
            ] {
                println!("──── {label} ────");
                match entry {
                    None => println!("(none — not generated)\n"),
                    Some(e) => {
                        println!("catalog hash  {:032x}", e.catalog_hash);
                        println!("text          {} bytes\n", e.summary.len());
                        println!("{}\n", e.summary);
                    }
                }
            }
        }
    }
    Ok(())
}

fn build_manifest(log: &mut LogFile) -> Result<Manifest> {
    Ok(Manifest::build_from_walk(log, SUPERBLOCK_SIZE)?.0)
}

/// Rebuild the in-RAM substrate from the log — the authoritative per-stream
/// index. The manifest only carries singleton offsets (model spec, template,
/// tokenizer, checkpoint); streams, chunks, tokens, and signatures live in the
/// substrate, populated by the same walker pass `open_in_with_substrate` uses.
fn build_substrate(log: &mut LogFile) -> Result<Substrate> {
    let mut substrate = Substrate::new();
    // Fast metadata scan: skip the large reference-stored payloads (KV chunks,
    // token blobs, signatures). `apply_walker_entry` keeps only their
    // `(offset, len)` — never the bytes — and the inspector reads specific
    // payloads on demand, so reading them here would scan the whole multi-GB log
    // for nothing. This is the single biggest cost in every command.
    let (entries, _) = walker::collect_filtered(log, SUPERBLOCK_SIZE, |rt| {
        !matches!(
            rt,
            RecordType::Chunk | RecordType::Tokens | RecordType::Signatures
        )
    })?;
    for e in &entries {
        substrate.apply_walker_entry(e);
    }
    Ok(substrate)
}

fn stream_kind_counts(substrate: &Substrate) -> (usize, usize) {
    let mut turns = 0;
    let mut sections = 0;
    for (_, s) in substrate.all_streams() {
        match &s.decl {
            Some(StreamDecl::Turn(_)) => turns += 1,
            Some(StreamDecl::PromptSection(_)) => sections += 1,
            None => {}
        }
    }
    (turns, sections)
}

fn print_singleton(log: &mut LogFile, label: &str, loc: Option<(u64, u64)>) -> Result<()> {
    match loc {
        None => println!("{label:<11} (none)"),
        Some((off, size)) => {
            let rec = read_record_at(log, off, size)?;
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
