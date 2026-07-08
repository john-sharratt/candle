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
use candle_conversation::persistence::record::{ChunkPayload, Record, RecordType};
use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{ContentAddress, StreamDecl, StreamId, TurnDecl};
use candle_conversation::persistence::walker;
use candle_conversation::projection::{
    decode_events, ProjectionEvent, SelectedSection, SystemItem, TimelineId, TurnIndex,
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
    /// Score one turn's stored wide-Q signature against the tag-scoped tool
    /// gallery, offline (no model / live gather). Isolates whether the belief
    /// scoring discriminates on persisted data — the top slots should be the
    /// probe turn's own tool.
    BeliefProbe {
        /// Probe turn stream id, decimal or `0x`-hex.
        stream_id: String,
        /// Gallery tag scope (matches the collection policy's `tags`).
        #[arg(long, default_value = "tool")]
        tag: String,
    },
    /// §80 tool-selection accuracy: leave-one-out over every tagged turn — each
    /// is scored against the gallery of all the others — reporting Top-1 / Top-5
    /// / MRR ranking accuracy plus the selection policy's recall and set size.
    BeliefEval {
        /// Gallery tag scope (matches the collection policy's `tags`).
        #[arg(long, default_value = "tool")]
        tag: String,
        /// `min_score` gate for the selection metrics (committed_tool_scope = 35).
        #[arg(long, default_value_t = 35.0)]
        min_score: f32,
        /// `budget.max` for the selection metrics (committed_tool_scope = 3).
        #[arg(long, default_value_t = 3)]
        max_budget: usize,
        /// Scorer: `fused` (shipped z-late-fusion), `margin` (per-token margin
        /// vote, all groups), or `margin-id` (margin over identity groups only,
        /// skipping the noise group L0–45).
        #[arg(long, default_value = "fused")]
        scorer: String,
        /// Truncate each probe to its last N tokens (0 = full turn) to match the
        /// live reproject window.
        #[arg(long, default_value_t = 0)]
        probe_tokens: usize,
    },
    /// §80.3 production-faithful replay: unlike `belief-eval` (which scores ONE
    /// window per turn), this replays each turn's recorded reprojection sequence
    /// through the online belief exactly as production does — sliding
    /// `probe_tokens` window at each reprojection point, `score_slots` → RelLeak
    /// `belief_step` with the `CommittedToolScope` policy (β0.40, min 1000 / evict
    /// 750, budget 1..3), belief carried across projections.
    /// Reports whether the true tool ends the turn in the committed selected set —
    /// the metric a single-shot ranking can't see (an early pin survives the faded
    /// tail). Leave-one-out over the tagged corpus.
    BeliefReplay {
        /// Gallery tag scope (matches the collection policy's `tags`).
        #[arg(long, default_value = "tool")]
        tag: String,
        /// Probe window cap per reprojection (`reproject_max_probe_tokens` = 256).
        #[arg(long, default_value_t = 256)]
        probe_tokens: usize,
        /// Reprojection cadence in tokens (`reproject_every_n_tokens` = 64).
        #[arg(long, default_value_t = 64)]
        cadence: usize,
        /// Stream id(s) to print a per-reprojection belief trace for (repeatable).
        #[arg(long)]
        trace: Vec<String>,
    },
    /// §80.2 threshold derivation: compute the leave-one-out score matrix once,
    /// then sweep `min_score` × `budget.max` over it — reporting recall (hit
    /// rate), mean set size, mean/max false positives, and exact-1 — to find the
    /// gate that holds 100% recall with the fewest false positives.
    BeliefSweep {
        /// Gallery tag scope (matches the collection policy's `tags`).
        #[arg(long, default_value = "tool")]
        tag: String,
        /// Scorer to derive thresholds for: `fused`, `margin`, `margin-id`, `hybrid`.
        #[arg(long, default_value = "hybrid")]
        scorer: String,
        /// Truncate each probe to its last N tokens to match the live reproject
        /// window (`reproject_max_probe_tokens` = 64). 0 = full turn. Scores sum
        /// over probe tokens, so this sets the threshold's scale.
        #[arg(long, default_value_t = 64)]
        probe_tokens: usize,
    },
    /// §82 — model a conditional-decay (adaptive) probe window: walk the window
    /// backward in chunks, accumulate a weighted belief, and let each chunk's
    /// weight decay by the accumulated confidence so a confident probe aborts
    /// early instead of reaching into stale context. Sweeps decay α × abort
    /// threshold, reporting retained hit rate vs mean tokens actually used.
    BeliefDecay {
        /// Gallery tag scope (matches the collection policy's `tags`).
        #[arg(long, default_value = "tool")]
        tag: String,
        /// Full probe window before decay (matches `reproject_max_probe_tokens`).
        #[arg(long, default_value_t = 256)]
        window: usize,
        /// Chunk size to walk backward in.
        #[arg(long, default_value_t = 64)]
        chunk: usize,
    },
    /// §81 — deep-dive the scoring of ONE probe against its rivals, broken down
    /// per fold layer-group (L0–45 / L46 / L47) and per token. Isolates whether a
    /// tool-identity signal survives in an upper layer but is drowned by the
    /// generic lower-layer group in the late-fusion. Built for the lone Top-5
    /// miss (`tcp_session_list` vs the session-list family).
    BeliefDissect {
        /// Probe turn stream id, decimal or `0x`-hex.
        stream_id: String,
        /// Gallery tag scope (matches the collection policy's `tags`).
        #[arg(long, default_value = "tool")]
        tag: String,
        /// How many discriminative tokens to list.
        #[arg(long, default_value_t = 20)]
        tokens: usize,
    },
    /// Calibration-quality audit: decode every tagged turn and flag any whose
    /// assistant response never emitted a completed `</tool_call>` — a prompt
    /// that made the model deliberate/refuse instead of calling its tool poisons
    /// that tool's reference signature.
    CalibCheck {
        /// Gallery tag scope (the calibration tag).
        #[arg(long, default_value = "tool")]
        tag: String,
    },
    /// Dump a diffable per-turn calibration baseline, keyed by `tool|prompt`
    /// (stable across rebuilds — stream ids are not), so a decode-built substrate
    /// can be compared token-for-token and sig-for-sig against a prefill-built one
    /// after a rebuild. Each row: key, token count + hash, wide-Q token count +
    /// blob hash + mean popcount. Identical hashes ⇒ identical reproduction.
    CalibBaseline {
        /// Gallery tag scope (the calibration tag).
        #[arg(long, default_value = "tool")]
        tag: String,
        /// Write the TSV baseline to this path (default: stdout).
        #[arg(long)]
        out: Option<String>,
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
        Cmd::Checkpoint => checkpoint_view(&mut log)?,
        Cmd::BeliefProbe { stream_id, tag } => {
            belief_probe(&mut log, parse_stream_id(&stream_id)?, &tag)?
        }
        Cmd::BeliefEval {
            tag,
            min_score,
            max_budget,
            scorer,
            probe_tokens,
        } => belief_eval(&mut log, &tag, min_score, max_budget, &scorer, probe_tokens)?,
        Cmd::BeliefReplay {
            tag,
            probe_tokens,
            cadence,
            trace,
        } => {
            let trace_ids = trace
                .iter()
                .map(|s| parse_stream_id(s))
                .collect::<Result<Vec<_>>>()?;
            belief_replay(&mut log, &tag, probe_tokens, cadence, &trace_ids)?
        }
        Cmd::BeliefDissect {
            stream_id,
            tag,
            tokens,
        } => belief_dissect(&mut log, parse_stream_id(&stream_id)?, &tag, tokens)?,
        Cmd::CalibCheck { tag } => calib_check(&mut log, &log_path, &tag)?,
        Cmd::CalibBaseline { tag, out } => calib_baseline(&mut log, &log_path, &tag, out)?,
        Cmd::BeliefSweep {
            tag,
            scorer,
            probe_tokens,
        } => belief_sweep(&mut log, &tag, &scorer, probe_tokens)?,
        Cmd::BeliefDecay { tag, window, chunk } => belief_decay(&mut log, &tag, window, chunk)?,
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
                // Rank by belief score so the scoring's shape is visible: if the
                // top scores are all ~equal (or all zero), selection is degenerate.
                let mut ranked: Vec<&SelectedSection> = sections.iter().collect();
                ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
                for s in ranked {
                    println!(
                        "          {} {:<28} score {:>8.3}  ({} tok)",
                        if s.selected { "[x]" } else { "[ ]" },
                        s.name,
                        s.score,
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

/// Score one turn's stored wide-Q signature against the tag-scoped gallery,
/// offline. Mirrors `Conversation::belief_gallery` + `score_slots`: the gallery
/// is every turn whose tags intersect `tag`, each mapped to a slot keyed by its
/// non-`tag` tag (the tool name); the probe is the given turn's own signature.
fn belief_probe(log: &mut LogFile, probe_id: StreamId, tag: &str) -> Result<()> {
    use candle_conversation::provenance::{decode_wide_sigs, score_slots, WideQSig};

    let substrate = build_substrate(log)?;

    let probe_window = substrate
        .stream_of(probe_id)
        .and_then(|e| e.wide_q_sigs.as_ref())
        .and_then(|b| decode_wide_sigs(b))
        .with_context(|| format!("probe stream {} has no wide-Q signature", probe_id.0))?;
    let probe_tags: Vec<String> = substrate
        .stream_of(probe_id)
        .and_then(|e| match &e.decl {
            Some(StreamDecl::Turn(t)) => Some(t.tags.clone()),
            _ => None,
        })
        .unwrap_or_default();

    // Slot table: distinct tool names (the non-scope tag) in first-seen order.
    let mut slot_names: Vec<String> = Vec::new();
    let mut windows: Vec<Vec<WideQSig>> = Vec::new();
    let mut slots: Vec<usize> = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if sid == probe_id {
            continue; // never let the probe match itself
        }
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        let Some(window) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if window.is_empty() {
            continue;
        }
        let slot = slot_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| {
                slot_names.push(name.clone());
                slot_names.len() - 1
            });
        windows.push(window);
        slots.push(slot);
    }

    if windows.is_empty() {
        println!("gallery is EMPTY for tag {tag:?} — no tagged turns carry wide-Q. This alone forces degenerate selection.");
        return Ok(());
    }

    let wref: Vec<&[WideQSig]> = windows.iter().map(|w| w.as_slice()).collect();
    let fresh = score_slots(&probe_window, &wref, &slots, slot_names.len());

    println!(
        "probe stream {}  tags={:?}  ({} probe tokens)",
        probe_id.0,
        probe_tags,
        probe_window.len()
    );
    println!(
        "gallery: {} windows over {} slots (tag scope {:?})\n",
        windows.len(),
        slot_names.len(),
        tag
    );

    let mut ranked: Vec<(usize, f32)> = fresh.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let ground_truth = probe_tags.iter().find(|x| x.as_str() != tag);
    for (rank, (slot, score)) in ranked.iter().enumerate().take(15) {
        let name = &slot_names[*slot];
        let mark = if Some(name) == ground_truth {
            " ← ground truth"
        } else {
            ""
        };
        println!("  #{rank:<2} {name:<28} {score:>9.3}{mark}");
    }
    Ok(())
}

/// Margin-weighted per-token scorer. For each probe token and each requested
/// fold group, find the best agreement **per tool** and vote the top tool's
/// lead over the runner-up (`best − second_best`). A token where one tool
/// sharply wins (an identity token) dominates; a token where the family ties
/// (a generic "list sessions" token) contributes ~nothing. Complements the
/// shipped z-fusion, which self-mutes non-discriminative *groups* but not
/// non-discriminative *tokens*.
fn score_slots_margin(
    probe: &[candle_conversation::provenance::WideQSig],
    gallery_windows: &[&[candle_conversation::provenance::WideQSig]],
    gallery_slot: &[usize],
    n_slots: usize,
    gw: usize,
    groups: &[usize],
) -> Vec<f32> {
    let mut votes = vec![0f32; n_slots];
    let mut case_max = vec![0u32; n_slots];
    for q in probe {
        for &g in groups {
            let base = g * gw;
            if q.words.len() < base + gw {
                continue;
            }
            let qg = &q.words[base..base + gw];
            for m in case_max.iter_mut() {
                *m = 0;
            }
            for (wi, w) in gallery_windows.iter().enumerate() {
                let c = gallery_slot[wi];
                for cand in w.iter() {
                    if cand.words.len() >= base + gw {
                        let ag = word_agreement(qg, &cand.words[base..base + gw]);
                        if ag > case_max[c] {
                            case_max[c] = ag;
                        }
                    }
                }
            }
            // Top-1 and top-2 tool agreements → margin vote for the leader.
            let (mut top1, mut top1c, mut top2) = (0u32, usize::MAX, 0u32);
            for (c, &m) in case_max.iter().enumerate() {
                if m > top1 {
                    top2 = top1;
                    top1 = m;
                    top1c = c;
                } else if m > top2 {
                    top2 = m;
                }
            }
            if top1c != usize::MAX {
                votes[top1c] += top1.saturating_sub(top2) as f32;
            }
        }
    }
    votes
}

/// Hybrid scorer: per token and group, vote `z × margin` for the leading tool —
/// combining the shipped z-fusion's group self-muting (an outlier vs the group's
/// whole agreement distribution) with the margin's token self-muting (the
/// leader's lead over the runner-up tool). Stays on the z-scale (margin is a
/// unitless multiplier only where a token is discriminative). The noise group
/// L0–45 is auto-muted by the near-zero margin, so all groups can be passed.
fn score_slots_hybrid(
    probe: &[candle_conversation::provenance::WideQSig],
    gallery_windows: &[&[candle_conversation::provenance::WideQSig]],
    gallery_slot: &[usize],
    n_slots: usize,
    gw: usize,
    groups: &[usize],
) -> Vec<f32> {
    let mut votes = vec![0f32; n_slots];
    let mut case_max = vec![0u32; n_slots];
    for q in probe {
        for &g in groups {
            let base = g * gw;
            if q.words.len() < base + gw {
                continue;
            }
            let qg = &q.words[base..base + gw];
            for m in case_max.iter_mut() {
                *m = 0;
            }
            let (mut sum, mut sumsq, mut count) = (0u64, 0u64, 0u64);
            for (wi, w) in gallery_windows.iter().enumerate() {
                let c = gallery_slot[wi];
                for cand in w.iter() {
                    if cand.words.len() >= base + gw {
                        let ag = word_agreement(qg, &cand.words[base..base + gw]);
                        if ag > case_max[c] {
                            case_max[c] = ag;
                        }
                        sum += ag as u64;
                        sumsq += (ag as u64) * (ag as u64);
                        count += 1;
                    }
                }
            }
            if count == 0 {
                continue;
            }
            let (mut top1, mut top1c, mut top2) = (0u32, usize::MAX, 0u32);
            for (c, &m) in case_max.iter().enumerate() {
                if m > top1 {
                    top2 = top1;
                    top1 = m;
                    top1c = c;
                } else if m > top2 {
                    top2 = m;
                }
            }
            if top1c != usize::MAX {
                let n = count as f32;
                let mean = sum as f32 / n;
                let var = (sumsq as f32 / n - mean * mean).max(1e-6);
                let z = ((top1 as f32 - mean) / var.sqrt()).max(0.0);
                let margin = top1.saturating_sub(top2) as f32;
                votes[top1c] += z * margin;
            }
        }
    }
    votes
}

/// §80 tool-selection accuracy over the whole tagged corpus. For every tagged
/// turn, score its stored signature against the gallery of *all the others*
/// (leave-one-out) and record where its true tool ranked; then apply the
/// selection policy (`min_score` gate + `max_budget`, min-fill 1) to measure the
/// projected set. Ranking metrics are policy-independent; selection metrics show
/// what the shipped `committed_tool_scope` actually admits.
fn belief_eval(
    log: &mut LogFile,
    tag: &str,
    min_score: f32,
    max_budget: usize,
    scorer: &str,
    probe_tokens: usize,
) -> Result<()> {
    use candle_conversation::provenance::wide_sig::PROV_HEADS_PER_LAYER;
    use candle_conversation::provenance::{decode_wide_sigs, score_slots, WideQSig};
    use rayon::prelude::*;

    let substrate = build_substrate(log)?;

    // Corpus: every tagged turn with a wide-Q window → (stream id, tool slot, window).
    let mut slot_names: Vec<String> = Vec::new();
    let mut corpus: Vec<(StreamId, usize, Vec<WideQSig>)> = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        let Some(window) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if window.is_empty() {
            continue;
        }
        let slot = slot_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| {
                slot_names.push(name.clone());
                slot_names.len() - 1
            });
        corpus.push((sid, slot, window));
    }

    let n_slots = slot_names.len();
    if corpus.len() < 2 {
        println!(
            "corpus has {} tagged turn(s) — need at least 2 for leave-one-out.",
            corpus.len()
        );
        return Ok(());
    }

    // Fold geometry for the margin scorers.
    let shape = &corpus[0].2[0];
    let gw = PROV_HEADS_PER_LAYER * shape.words_per_head();
    let n_groups = shape.n_heads as usize / PROV_HEADS_PER_LAYER;
    let groups: Vec<usize> = match scorer {
        "margin" | "hybrid" => (0..n_groups).collect(),
        "margin-id" => (1..n_groups).collect(), // skip the noise group L0–45
        _ => Vec::new(),
    };

    struct Trial {
        sid: StreamId,
        gt_slot: usize,
        rank: usize,
        hit: bool, // ground truth scored > 0 (a real match, not an all-zero tie)
        gt_score: f32,
        best_slot: usize,
        best_score: f32,
        selected: Vec<usize>,
    }

    // Leave-one-out, parallel over probes. Each rebuilds its gallery from all
    // corpus windows except its own — the probe never matches itself.
    let trials: Vec<Trial> = (0..corpus.len())
        .into_par_iter()
        .map(|pi| {
            let (sid, gt_slot, full) = &corpus[pi];
            // Match the live reproject window when requested: the last N tokens.
            let probe: &[WideQSig] = if probe_tokens > 0 && full.len() > probe_tokens {
                &full[full.len() - probe_tokens..]
            } else {
                full.as_slice()
            };
            let mut gwin: Vec<&[WideQSig]> = Vec::with_capacity(corpus.len() - 1);
            let mut gslot: Vec<usize> = Vec::with_capacity(corpus.len() - 1);
            for (j, (_, s, w)) in corpus.iter().enumerate() {
                if j != pi {
                    gwin.push(w.as_slice());
                    gslot.push(*s);
                }
            }
            let fresh = match scorer {
                "margin" | "margin-id" => {
                    score_slots_margin(probe, &gwin, &gslot, n_slots, gw, &groups)
                }
                "hybrid" => score_slots_hybrid(probe, &gwin, &gslot, n_slots, gw, &groups),
                _ => score_slots(probe, &gwin, &gslot, n_slots),
            };
            let gt_score = fresh[*gt_slot];
            let rank = 1 + fresh.iter().filter(|&&s| s > gt_score).count();

            let mut ranked: Vec<(usize, f32)> = fresh.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let (best_slot, best_score) = ranked[0];
            let mut selected: Vec<usize> = ranked
                .iter()
                .filter(|(_, s)| *s >= min_score)
                .take(max_budget)
                .map(|(i, _)| *i)
                .collect();
            if selected.is_empty() {
                selected.push(ranked[0].0); // budget min = 1: force-fill the top
            }

            Trial {
                sid: *sid,
                gt_slot: *gt_slot,
                rank,
                hit: gt_score > 0.0,
                gt_score,
                best_slot,
                best_score,
                selected,
            }
        })
        .collect();

    let n = trials.len();
    let pct = |k: usize| 100.0 * k as f64 / n as f64;
    let top1 = trials.iter().filter(|t| t.rank == 1 && t.hit).count();
    let top3 = trials.iter().filter(|t| t.rank <= 3 && t.hit).count();
    let top5 = trials.iter().filter(|t| t.rank <= 5 && t.hit).count();
    let misses = trials.iter().filter(|t| !t.hit).count();
    let mrr: f64 = trials
        .iter()
        .map(|t| if t.hit { 1.0 / t.rank as f64 } else { 0.0 })
        .sum::<f64>()
        / n as f64;

    let recall = trials
        .iter()
        .filter(|t| t.selected.contains(&t.gt_slot))
        .count();
    let mean_sz: f64 = trials.iter().map(|t| t.selected.len()).sum::<usize>() as f64 / n as f64;
    let exact1 = trials
        .iter()
        .filter(|t| t.selected.len() == 1 && t.selected[0] == t.gt_slot)
        .count();

    println!("\n══ §80 tool-selection eval (leave-one-out over current substrate) ══\n");
    println!("scorer:  {scorer}");
    println!("corpus:  {n} tagged turns over {n_slots} tools   (tag scope {tag:?})",);
    println!(
        "gallery: leave-one-out — each probe scored against the other {} turns\n",
        n - 1
    );

    println!("Ranking accuracy (belief argmax, policy-independent):");
    println!("  Tool-1 : {:>5.1}%   ({top1}/{n})", pct(top1));
    println!("  Tool-3 : {:>5.1}%   ({top3}/{n})", pct(top3));
    println!("  Tool-5 : {:>5.1}%   ({top5}/{n})", pct(top5));
    println!("  MRR    : {mrr:>6.3}");
    if misses > 0 {
        println!(
            "  misses : {:>5.1}%   ({misses}/{n} scored 0 for their own tool)",
            pct(misses)
        );
    }

    println!("\nSelection policy (min_score {min_score} / budget 1..{max_budget}, min-fill 1):");
    println!(
        "  recall (true tool in set) : {:>5.1}%   ({recall}/{n})",
        pct(recall)
    );
    println!(
        "  exact-1 (only the right tool) : {:>5.1}%   ({exact1}/{n})",
        pct(exact1)
    );
    println!("  mean selected-set size    : {mean_sz:>5.2}");

    // The genuinely hard probes: those whose true tool fell outside Top-3 (the
    // shipped `budget.max`). Listed so a Tool-3 regression names its offenders,
    // not just its count. A `‖T5` marker flags the ones also outside Top-5.
    let mut hardest: Vec<&Trial> = trials.iter().filter(|t| t.rank > 3 || !t.hit).collect();
    hardest.sort_by(|a, b| b.rank.cmp(&a.rank));
    println!(
        "\nHardest probes (true tool outside Top-3): {}",
        hardest.len()
    );
    for t in &hardest {
        let t5 = if t.rank > 5 || !t.hit { " ‖T5" } else { "" };
        println!(
            "  stream {:#018x}  tool {:<22} rank #{:<3} score {:>7.3}   beaten by {} ({:.3}){t5}",
            t.sid.0,
            slot_names[t.gt_slot],
            t.rank,
            t.gt_score,
            slot_names[t.best_slot],
            t.best_score,
        );
    }

    // Per-tool Tool-1, worst first — surfaces which tools are confusable.
    let mut per_tool: Vec<(usize, usize)> = vec![(0, 0); n_slots]; // (correct, total)
    for t in &trials {
        per_tool[t.gt_slot].1 += 1;
        if t.rank == 1 && t.hit {
            per_tool[t.gt_slot].0 += 1;
        }
    }
    let mut rows: Vec<(String, usize, usize)> = per_tool
        .iter()
        .enumerate()
        .filter(|(_, (_, total))| *total > 0)
        .map(|(i, (c, total))| (slot_names[i].clone(), *c, *total))
        .collect();
    rows.sort_by(|a, b| {
        let ra = a.1 as f64 / a.2 as f64;
        let rb = b.1 as f64 / b.2 as f64;
        ra.total_cmp(&rb).then(b.2.cmp(&a.2))
    });
    println!("\nPer-tool Tool-1 (worst first, showing up to 15):");
    for (name, correct, total) in rows.iter().take(15) {
        println!(
            "  {name:<30} {:>5.1}%   ({correct}/{total})",
            100.0 * *correct as f64 / *total as f64
        );
    }
    Ok(())
}

/// §80.3 — replay each tagged turn's recorded reprojection sequence through the
/// online belief exactly as production does, leave-one-out. See
/// [`Cmd::BeliefReplay`].
fn belief_replay(
    log: &mut LogFile,
    tag: &str,
    probe_tokens: usize,
    cadence: usize,
    trace_ids: &[StreamId],
) -> Result<()> {
    use candle_conversation::provenance::{
        belief_step, decode_wide_sigs, score_slots, GroupBudget, SectionPolicy, WideQSig,
    };
    use rayon::prelude::*;

    let substrate = build_substrate(log)?;

    // Corpus: every tagged turn with a wide-Q window, its tool slot, and its full
    // per-token signature. The prefill-built calibration turns carry only a single
    // degenerate projection event, so we cannot replay a *recorded* reprojection
    // sequence — instead we synthesise production's cadence (a reprojection every
    // `cadence` generated tokens) over the full signature, which is exactly what a
    // live decode of the same trajectory would fire.
    struct Turn {
        sid: StreamId,
        slot: usize,
        sigs: Vec<WideQSig>,
    }
    let mut slot_names: Vec<String> = Vec::new();
    let mut corpus: Vec<Turn> = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        let Some(sigs) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if sigs.is_empty() {
            continue;
        }
        let slot = slot_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| {
                slot_names.push(name.clone());
                slot_names.len() - 1
            });
        corpus.push(Turn { sid, slot, sigs });
    }

    let n_slots = slot_names.len();
    let n = corpus.len();
    if n < 2 {
        println!("corpus has {n} tagged turn(s) — need at least 2 for leave-one-out.");
        return Ok(());
    }

    // The shipped `tools` collection policy — `CommittedToolScope` (policy.rs).
    let policy = SectionPolicy {
        group: 0,
        beta: 0.40,
        min_score: 1000.0,
        evict_score: 750.0,
    };
    let budget = GroupBudget { min: 1, max: 3 };
    let trace_set: std::collections::HashSet<u64> = trace_ids.iter().map(|s| s.0).collect();

    struct Outcome {
        sid: StreamId,
        slot: usize,
        committed: bool,         // true tool in the final (at-tool_call) selected set
        ever: bool,              // true tool selected at any reprojection
        first_pt: Option<usize>, // first generated-token point where it was selected
        final_rank: usize,       // rank of the true tool by accumulated belief at the end
        n_points: usize,
        trace: Vec<String>,
    }

    let outcomes: Vec<Outcome> = (0..n)
        .into_par_iter()
        .map(|pi| {
            let owner = &corpus[pi];
            let mut gwin: Vec<&[WideQSig]> = Vec::with_capacity(n - 1);
            let mut gslot: Vec<usize> = Vec::with_capacity(n - 1);
            for (j, tj) in corpus.iter().enumerate() {
                if j != pi {
                    gwin.push(tj.sigs.as_slice());
                    gslot.push(tj.slot);
                }
            }
            // Synthesised production cadence: a reprojection every `cadence`
            // generated tokens (the live decode's `reproject_every_n_tokens`),
            // plus a final one at the tool_call. Each fires on the tokens generated
            // *so far* — so early points see the discriminative head, late ones the
            // faded tail, exactly as a real decode would.
            let len = owner.sigs.len();
            let step = cadence.max(1);
            let mut points: Vec<usize> = (step..len).step_by(step).collect();
            points.push(len);
            let do_trace = trace_set.contains(&owner.sid.0);
            let mut trace: Vec<String> = Vec::new();

            let mut prior_scores: Vec<f32> = Vec::new();
            let mut prior_selected: Vec<bool> = Vec::new();
            let mut ever = false;
            let mut first_pt = None;
            for &p in &points {
                let end_idx = p.min(owner.sigs.len());
                let lo = end_idx.saturating_sub(probe_tokens);
                let probe = &owner.sigs[lo..end_idx];
                if probe.is_empty() {
                    continue;
                }
                let fresh = score_slots(probe, &gwin, &gslot, n_slots);
                let beliefs = belief_step(&fresh, &prior_scores, &prior_selected, policy, budget);
                prior_scores = beliefs.iter().map(|b| b.score).collect();
                prior_selected = beliefs.iter().map(|b| b.selected).collect();
                let sel = prior_selected.get(owner.slot).copied().unwrap_or(false);
                if sel {
                    ever = true;
                    if first_pt.is_none() {
                        first_pt = Some(p);
                    }
                }
                if do_trace {
                    let (bi, bs) = prior_scores.iter().enumerate().fold(
                        (0usize, f32::MIN),
                        |(ai, as_), (i, &s)| if s > as_ { (i, s) } else { (ai, as_) },
                    );
                    trace.push(format!(
                        "  tok@{p:<4} win[{lo}..{end_idx}] len {:>3}  true={:>9.1}{}  top={} ({bs:.1})  selected={:?}",
                        probe.len(),
                        prior_scores.get(owner.slot).copied().unwrap_or(0.0),
                        if sel { " *SELECTED*" } else { "" },
                        slot_names[bi],
                        prior_selected
                            .iter()
                            .enumerate()
                            .filter(|(_, &s)| s)
                            .map(|(i, _)| slot_names[i].as_str())
                            .collect::<Vec<_>>(),
                    ));
                }
            }
            let gt = prior_scores.get(owner.slot).copied().unwrap_or(0.0);
            let final_rank = 1 + prior_scores.iter().filter(|&&s| s > gt).count();
            let committed = prior_selected.get(owner.slot).copied().unwrap_or(false);
            Outcome {
                sid: owner.sid,
                slot: owner.slot,
                committed,
                ever,
                first_pt,
                final_rank,
                n_points: points.len(),
                trace,
            }
        })
        .collect();

    let pct = |k: usize| 100.0 * k as f64 / n as f64;
    let committed = outcomes.iter().filter(|o| o.committed).count();
    let ever = outcomes.iter().filter(|o| o.ever).count();
    let top1 = outcomes.iter().filter(|o| o.final_rank == 1).count();
    let top3 = outcomes.iter().filter(|o| o.final_rank <= 3).count();
    let mean_pts: f64 = outcomes.iter().map(|o| o.n_points).sum::<usize>() as f64 / n as f64;

    println!("\n══ §80.3 production-faithful reprojection replay (leave-one-out) ══\n");
    println!("corpus:  {n} tagged turns over {n_slots} tools   (tag scope {tag:?})");
    println!("policy:  CommittedToolScope — β0.40, min 1000 / evict 750, budget 1..3");
    println!(
        "probe:   sliding {probe_tokens}-token window, reprojecting every {cadence} tokens (synthesised cadence)"
    );
    println!(
        "gallery: leave-one-out — each turn scored against the other {}\n",
        n - 1
    );

    println!("mean reprojections per turn : {mean_pts:>5.2}");
    let first_pts: Vec<usize> = outcomes.iter().filter_map(|o| o.first_pt).collect();
    if !first_pts.is_empty() {
        let mean_first = first_pts.iter().sum::<usize>() as f64 / first_pts.len() as f64;
        println!(
            "mean tokens to first lock-on: {mean_first:>5.1}   (when the true tool is first selected + pinned)"
        );
    }
    println!("\nProduction outcome (what the model actually gets offered):");
    println!(
        "  committed (true tool in the selected set at tool_call) : {:>5.1}%   ({committed}/{n})",
        pct(committed)
    );
    println!(
        "  ever selected during the turn                          : {:>5.1}%   ({ever}/{n})",
        pct(ever)
    );
    println!("\nFor reference — ranking by *accumulated* belief at turn end:");
    println!("  Tool-1 : {:>5.1}%   ({top1}/{n})", pct(top1));
    println!("  Tool-3 : {:>5.1}%   ({top3}/{n})", pct(top3));

    let mut missed: Vec<&Outcome> = outcomes.iter().filter(|o| !o.committed).collect();
    missed.sort_by(|a, b| b.final_rank.cmp(&a.final_rank));
    println!(
        "\nProduction misses (true tool NOT committed at tool_call): {}",
        missed.len()
    );
    for o in &missed {
        println!(
            "  stream {:#018x}  tool {:<22} final-rank #{:<3} ever-selected={}",
            o.sid.0, slot_names[o.slot], o.final_rank, o.ever,
        );
    }

    // Detailed per-reprojection traces for the requested streams.
    for o in &outcomes {
        if o.trace.is_empty() {
            continue;
        }
        println!(
            "\n── trace {}  tool {} (slot {})  committed={} ever={} ──",
            stream_hex(o.sid.0),
            slot_names[o.slot],
            o.slot,
            o.committed,
            o.ever,
        );
        for line in &o.trace {
            println!("{line}");
        }
    }

    Ok(())
}

/// One probe's selection outcome under a `(min_score, budget)` policy: whether
/// the true tool made the set, and the set's size / false-positive count.
fn evaluate_selection(
    scores: &[f32],
    gt_slot: usize,
    min_score: f32,
    budget: usize,
) -> (bool, usize, usize) {
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut selected: Vec<usize> = ranked
        .iter()
        .filter(|(_, s)| *s >= min_score)
        .take(budget)
        .map(|(i, _)| *i)
        .collect();
    if selected.is_empty() {
        selected.push(ranked[0].0); // budget min = 1: force-fill the top
    }
    let hit = selected.contains(&gt_slot);
    let size = selected.len();
    let fp = size - usize::from(hit);
    (hit, size, fp)
}

/// Per-token hybrid votes: for each probe token, the list of `(tool, z×margin)`
/// contributions (one per fold group). Summing all tokens reproduces
/// `score_slots_hybrid`; keeping them per-token lets the adaptive window
/// re-weight and truncate by position without re-scanning the gallery.
fn per_token_hybrid_votes(
    probe: &[candle_conversation::provenance::WideQSig],
    gallery_windows: &[&[candle_conversation::provenance::WideQSig]],
    gallery_slot: &[usize],
    n_slots: usize,
    gw: usize,
    groups: &[usize],
) -> Vec<Vec<(usize, f32)>> {
    let mut out = Vec::with_capacity(probe.len());
    let mut case_max = vec![0u32; n_slots];
    for q in probe {
        let mut votes: Vec<(usize, f32)> = Vec::new();
        for &g in groups {
            let base = g * gw;
            if q.words.len() < base + gw {
                continue;
            }
            let qg = &q.words[base..base + gw];
            for m in case_max.iter_mut() {
                *m = 0;
            }
            let (mut sum, mut sumsq, mut count) = (0u64, 0u64, 0u64);
            for (wi, w) in gallery_windows.iter().enumerate() {
                let c = gallery_slot[wi];
                for cand in w.iter() {
                    if cand.words.len() >= base + gw {
                        let ag = word_agreement(qg, &cand.words[base..base + gw]);
                        if ag > case_max[c] {
                            case_max[c] = ag;
                        }
                        sum += ag as u64;
                        sumsq += (ag as u64) * (ag as u64);
                        count += 1;
                    }
                }
            }
            if count == 0 {
                continue;
            }
            let (mut t1, mut t1c, mut t2) = (0u32, usize::MAX, 0u32);
            for (c, &m) in case_max.iter().enumerate() {
                if m > t1 {
                    t2 = t1;
                    t1 = m;
                    t1c = c;
                } else if m > t2 {
                    t2 = m;
                }
            }
            if t1c != usize::MAX {
                let mean = sum as f32 / count as f32;
                let var = (sumsq as f32 / count as f32 - mean * mean).max(1e-6);
                let z = ((t1 as f32 - mean) / var.sqrt()).max(0.0);
                votes.push((t1c, z * t1.saturating_sub(t2) as f32));
            }
        }
        out.push(votes);
    }
    out
}

/// §82 — conditional-decay adaptive probe window. Walk `window` backward in
/// `chunk`-token steps (most recent first); accumulate a weighted belief; after
/// each chunk decay the next chunk's weight by the accumulated confidence
/// `(top1−top2)/top1`; abort when the weight falls below the threshold. Sweeps
/// α × abort and reports retained hit rate vs mean tokens actually consumed.
fn belief_decay(log: &mut LogFile, tag: &str, window: usize, chunk: usize) -> Result<()> {
    use candle_conversation::provenance::wide_sig::PROV_HEADS_PER_LAYER;
    use candle_conversation::provenance::{decode_wide_sigs, WideQSig};
    use rayon::prelude::*;

    let substrate = build_substrate(log)?;

    let mut slot_names: Vec<String> = Vec::new();
    let mut corpus: Vec<(usize, Vec<WideQSig>)> = Vec::new();
    for (_sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        let Some(win) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if win.is_empty() {
            continue;
        }
        let slot = slot_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| {
                slot_names.push(name.clone());
                slot_names.len() - 1
            });
        corpus.push((slot, win));
    }
    let n_slots = slot_names.len();
    if corpus.len() < 2 {
        println!("corpus too small");
        return Ok(());
    }
    let shape = &corpus[0].1[0];
    let gw = PROV_HEADS_PER_LAYER * shape.words_per_head();
    let n_groups = shape.n_heads as usize / PROV_HEADS_PER_LAYER;
    let groups: Vec<usize> = (0..n_groups).collect();

    // Phase 1 (expensive, once): per probe, the per-token votes over the last
    // `window` tokens, leave-one-out.
    let cached: Vec<(usize, Vec<Vec<(usize, f32)>>)> = (0..corpus.len())
        .into_par_iter()
        .map(|pi| {
            let (gt, full) = &corpus[pi];
            let probe: &[WideQSig] = if full.len() > window {
                &full[full.len() - window..]
            } else {
                full.as_slice()
            };
            let mut gwin: Vec<&[WideQSig]> = Vec::new();
            let mut gslot: Vec<usize> = Vec::new();
            for (j, (s, w)) in corpus.iter().enumerate() {
                if j != pi {
                    gwin.push(w.as_slice());
                    gslot.push(*s);
                }
            }
            (
                *gt,
                per_token_hybrid_votes(probe, &gwin, &gslot, n_slots, gw, &groups),
            )
        })
        .collect();

    let n = cached.len();
    let pct = |k: usize| 100.0 * k as f64 / n as f64;

    // Phase 2 (cheap): apply the adaptive window for a given (α, abort).
    let run = |alpha: f32, abort: f32| -> (usize, usize, usize, f64, f64) {
        // returns (tool1, tool3, tool5, mean_tokens, mean_chunks)
        let (mut t1, mut t3, mut t5, mut tok_sum, mut chunk_sum) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for (gt, toks) in &cached {
            let tlen = toks.len();
            let mut acc = vec![0f32; n_slots];
            let mut w = 1.0f32;
            let n_chunks = tlen.div_ceil(chunk);
            let mut used_tokens = 0usize;
            let mut used_chunks = 0usize;
            for k in 0..n_chunks {
                if w < abort {
                    break;
                }
                let hi = tlen.saturating_sub(k * chunk);
                let lo = tlen.saturating_sub((k + 1) * chunk);
                for t in lo..hi {
                    for &(slot, v) in &toks[t] {
                        acc[slot] += w * v;
                    }
                }
                used_tokens += hi - lo;
                used_chunks += 1;
                // Accumulated confidence → decay the next chunk's weight.
                let (mut top1, mut top2) = (0f32, 0f32);
                for &s in &acc {
                    if s > top1 {
                        top2 = top1;
                        top1 = s;
                    } else if s > top2 {
                        top2 = s;
                    }
                }
                let conf = if top1 > 0.0 {
                    (top1 - top2) / top1
                } else {
                    0.0
                };
                w *= 1.0 - alpha * conf;
            }
            let rank = 1 + acc.iter().filter(|&&s| s > acc[*gt]).count();
            let hit = acc[*gt] > 0.0;
            if hit && rank == 1 {
                t1 += 1;
            }
            if hit && rank <= 3 {
                t3 += 1;
            }
            if hit && rank <= 5 {
                t5 += 1;
            }
            tok_sum += used_tokens;
            chunk_sum += used_chunks;
        }
        (
            t1,
            t3,
            t5,
            tok_sum as f64 / n as f64,
            chunk_sum as f64 / n as f64,
        )
    };

    println!("\n══ §82 conditional-decay probe window  (window {window}, chunk {chunk}) ══\n");
    println!("corpus: {n} tagged turns over {n_slots} tools\n");
    println!(
        "  {:>6} {:>6}  {:>7} {:>7} {:>7}  {:>9} {:>8}",
        "alpha", "abort", "Tool-1", "Tool-3", "Tool-5", "mean_tok", "chunks"
    );
    // Baseline: no decay (full window).
    let (b1, b3, b5, bt, bc) = run(0.0, 0.0);
    println!(
        "  {:>6} {:>6}  {:>6.1}% {:>6.1}% {:>6.1}%  {:>9.1} {:>8.2}   ← baseline (full window)",
        "0.00",
        "—",
        pct(b1),
        pct(b3),
        pct(b5),
        bt,
        bc
    );
    for &alpha in &[0.5f32, 0.7, 0.9, 1.0] {
        for &abort in &[0.10f32, 0.25, 0.40] {
            let (a1, a3, a5, at, ac) = run(alpha, abort);
            let flag = if a3 == b3 { "" } else { "  ← Tool-3 dropped" };
            println!(
                "  {alpha:>6.2} {abort:>6.2}  {:>6.1}% {:>6.1}% {:>6.1}%  {:>9.1} {:>8.2}{flag}",
                pct(a1),
                pct(a3),
                pct(a5),
                at,
                ac
            );
        }
    }
    println!("\n(Tool-3 = recall at budget 3; mean_tok = tokens actually consumed before abort.)");

    // Position-independent alternative: weight each chunk by its OWN confidence
    // `(top1−top2)/top1` raised to γ, then sum. A sharp chunk (a needle, near or
    // far) keeps full weight; a diffuse chunk (generic/stale, near or far) is
    // muted. γ=0 ⇒ uniform (baseline); higher γ ⇒ sharper focus.
    let run_quality = |gamma: f32| -> (usize, usize, usize) {
        let (mut t1, mut t3, mut t5) = (0usize, 0usize, 0usize);
        for (gt, toks) in &cached {
            let tlen = toks.len();
            let n_chunks = tlen.div_ceil(chunk);
            let mut belief = vec![0f32; n_slots];
            for k in 0..n_chunks {
                let hi = tlen.saturating_sub(k * chunk);
                let lo = tlen.saturating_sub((k + 1) * chunk);
                let mut cs = vec![0f32; n_slots];
                for t in lo..hi {
                    for &(slot, v) in &toks[t] {
                        cs[slot] += v;
                    }
                }
                let (mut top1, mut top2) = (0f32, 0f32);
                for &s in &cs {
                    if s > top1 {
                        top2 = top1;
                        top1 = s;
                    } else if s > top2 {
                        top2 = s;
                    }
                }
                let conf = if top1 > 0.0 {
                    (top1 - top2) / top1
                } else {
                    0.0
                };
                let wt = conf.powf(gamma);
                for (b, c) in belief.iter_mut().zip(&cs) {
                    *b += wt * c;
                }
            }
            let rank = 1 + belief.iter().filter(|&&s| s > belief[*gt]).count();
            let hit = belief[*gt] > 0.0;
            if hit && rank == 1 {
                t1 += 1;
            }
            if hit && rank <= 3 {
                t3 += 1;
            }
            if hit && rank <= 5 {
                t5 += 1;
            }
        }
        (t1, t3, t5)
    };

    println!("\n── per-chunk quality weighting (position-independent, no abort) ──");
    println!(
        "  {:>6}  {:>7} {:>7} {:>7}",
        "gamma", "Tool-1", "Tool-3", "Tool-5"
    );
    for &g in &[0.0f32, 0.5, 1.0, 2.0, 3.0] {
        let (q1, q3, q5) = run_quality(g);
        let tag = if (g - 0.0).abs() < 1e-9 {
            "  ← uniform baseline"
        } else {
            ""
        };
        println!(
            "  {g:>6.2}  {:>6.1}% {:>6.1}% {:>6.1}%{tag}",
            pct(q1),
            pct(q3),
            pct(q5)
        );
    }

    // Per-token needle gate: keep only the top fraction of tokens by vote
    // magnitude (the needles), drop the rest (the haystack), position-independent.
    // If recall holds while keeping few tokens, the signal is sparse and the bulk
    // of the window is droppable — the true defensive lever.
    let run_gate = |frac: f32| -> (usize, usize, usize, f64) {
        let (mut t1, mut t3, mut t5, mut kept_sum) = (0usize, 0usize, 0usize, 0usize);
        for (gt, toks) in &cached {
            let tlen = toks.len();
            let mag: Vec<f32> = toks
                .iter()
                .map(|vs| vs.iter().map(|(_, v)| *v).sum())
                .collect();
            let mut sorted = mag.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let keep_n = ((frac * tlen as f32).ceil() as usize).clamp(1, tlen);
            let thresh = sorted[keep_n - 1];
            let mut belief = vec![0f32; n_slots];
            let mut kept = 0usize;
            for (t, vs) in toks.iter().enumerate() {
                if mag[t] >= thresh {
                    for &(slot, v) in vs {
                        belief[slot] += v;
                    }
                    kept += 1;
                }
            }
            let rank = 1 + belief.iter().filter(|&&s| s > belief[*gt]).count();
            let hit = belief[*gt] > 0.0;
            if hit && rank == 1 {
                t1 += 1;
            }
            if hit && rank <= 3 {
                t3 += 1;
            }
            if hit && rank <= 5 {
                t5 += 1;
            }
            kept_sum += kept;
        }
        (t1, t3, t5, kept_sum as f64 / n as f64)
    };

    println!(
        "\n── per-token needle gate (keep top-frac by vote magnitude, position-independent) ──"
    );
    println!(
        "  {:>6}  {:>7} {:>7} {:>7}  {:>9}",
        "frac", "Tool-1", "Tool-3", "Tool-5", "mean_kept"
    );
    for &f in &[1.0f32, 0.5, 0.25, 0.1, 0.05, 0.02] {
        let (g1, g3, g5, mk) = run_gate(f);
        let tag = if (f - 1.0).abs() < 1e-9 {
            "  ← keep all (baseline)"
        } else {
            ""
        };
        println!(
            "  {f:>6.2}  {:>6.1}% {:>6.1}% {:>6.1}%  {:>9.1}{tag}",
            pct(g1),
            pct(g3),
            pct(g5),
            mk
        );
    }
    Ok(())
}

/// §80.2 threshold sweep: cache the leave-one-out score matrix, then sweep
/// `min_score` × `budget.max` to chart the recall / false-positive frontier and
/// pick the tightest gate that still holds 100% recall.
fn belief_sweep(log: &mut LogFile, tag: &str, scorer: &str, probe_tokens: usize) -> Result<()> {
    use candle_conversation::provenance::wide_sig::PROV_HEADS_PER_LAYER;
    use candle_conversation::provenance::{decode_wide_sigs, score_slots, WideQSig};
    use rayon::prelude::*;

    let substrate = build_substrate(log)?;

    let mut slot_names: Vec<String> = Vec::new();
    let mut corpus: Vec<(usize, Vec<WideQSig>)> = Vec::new();
    for (_sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        let Some(window) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if window.is_empty() {
            continue;
        }
        let slot = slot_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| {
                slot_names.push(name.clone());
                slot_names.len() - 1
            });
        corpus.push((slot, window));
    }
    let n_slots = slot_names.len();
    if corpus.len() < 2 {
        println!("corpus too small");
        return Ok(());
    }

    let shape = &corpus[0].1[0];
    let gw = PROV_HEADS_PER_LAYER * shape.words_per_head();
    let n_groups = shape.n_heads as usize / PROV_HEADS_PER_LAYER;
    let groups: Vec<usize> = match scorer {
        "margin" | "hybrid" => (0..n_groups).collect(),
        "margin-id" => (1..n_groups).collect(),
        _ => Vec::new(),
    };

    // Leave-one-out score matrix (probes × tools) + each probe's true slot.
    let matrix: Vec<(usize, Vec<f32>)> = (0..corpus.len())
        .into_par_iter()
        .map(|pi| {
            let (gt_slot, full) = &corpus[pi];
            // Match the live reproject window: the last `probe_tokens` tokens.
            let probe: &[WideQSig] = if probe_tokens > 0 && full.len() > probe_tokens {
                &full[full.len() - probe_tokens..]
            } else {
                full.as_slice()
            };
            let mut gwin: Vec<&[WideQSig]> = Vec::with_capacity(corpus.len() - 1);
            let mut gslot: Vec<usize> = Vec::with_capacity(corpus.len() - 1);
            for (j, (s, w)) in corpus.iter().enumerate() {
                if j != pi {
                    gwin.push(w.as_slice());
                    gslot.push(*s);
                }
            }
            let fresh = match scorer {
                "margin" | "margin-id" => {
                    score_slots_margin(probe, &gwin, &gslot, n_slots, gw, &groups)
                }
                "hybrid" => score_slots_hybrid(probe, &gwin, &gslot, n_slots, gw, &groups),
                _ => score_slots(probe, &gwin, &gslot, n_slots),
            };
            (*gt_slot, fresh)
        })
        .collect();

    let n = matrix.len();
    let pct = |k: usize| 100.0 * k as f64 / n as f64;

    // Ranking: how deep the budget must reach (Tool-k).
    let ranks: Vec<usize> = matrix
        .iter()
        .map(|(gt, s)| 1 + s.iter().filter(|&&x| x > s[*gt]).count())
        .collect();
    let tool_k = |k: usize| ranks.iter().filter(|&&r| r <= k).count();

    // Ground-truth score distribution — the 100%-recall floor.
    let mut gt_scores: Vec<f32> = matrix.iter().map(|(gt, s)| s[*gt]).collect();
    gt_scores.sort_by(f32::total_cmp);
    let q = |frac: f64| gt_scores[((frac * (n - 1) as f64).round() as usize).min(n - 1)];

    let probe_desc = if probe_tokens == 0 {
        "full turn".to_string()
    } else {
        format!("last {probe_tokens} tokens (live reproject window)")
    };
    println!("\n══ §80.2 threshold sweep  (scorer {scorer}, leave-one-out) ══\n");
    println!("matrix: {n} probes × {n_slots} tools   probe: {probe_desc}\n");
    println!("Recall ceiling by budget (min_score 0 ⇒ recall = Tool-k):");
    for k in 1..=8 {
        println!("  budget {k}: {:.1}%", pct(tool_k(k)));
    }
    println!("\nTrue-tool score distribution (min = 100%-recall ceiling):");
    println!(
        "  min {:.0}   p1 {:.0}   p5 {:.0}   p10 {:.0}   p25 {:.0}   p50 {:.0}   max {:.0}",
        gt_scores[0],
        q(0.01),
        q(0.05),
        q(0.10),
        q(0.25),
        q(0.50),
        gt_scores[n - 1],
    );

    // Threshold grid: 0, then the sorted true-tool scores at a spread of ranks
    // (each is where one more probe's true tool drops below the gate), plus p50.
    let idxs = [0usize, 1, 2, 3, 4, 6, 9, 14, 24, 49, (n / 4).max(1), n / 2];
    let mut grid: Vec<f32> = vec![0.0];
    for &k in &idxs {
        if k < n {
            grid.push(gt_scores[k]);
        }
    }
    grid.dedup();

    for budget in [3usize, 4, 5, 6] {
        println!("\n── budget max = {budget} ──");
        println!(
            "  {:>10}  {:>7}  {:>9}  {:>8}  {:>7}  {:>8}",
            "min_score", "recall", "mean_sz", "mean_fp", "max_fp", "exact-1"
        );
        for &ms in &grid {
            let mut hits = 0usize;
            let (mut sz_sum, mut fp_sum, mut max_fp, mut exact1) = (0usize, 0usize, 0usize, 0usize);
            for (gt, s) in &matrix {
                let (hit, size, fp) = evaluate_selection(s, *gt, ms, budget);
                if hit {
                    hits += 1;
                    if size == 1 {
                        exact1 += 1;
                    }
                }
                sz_sum += size;
                fp_sum += fp;
                max_fp = max_fp.max(fp);
            }
            let recall = pct(hits);
            let flag = if (recall - 100.0).abs() < 1e-9 {
                ""
            } else {
                "  ← <100%"
            };
            println!(
                "  {ms:>10.0}  {recall:>6.1}%  {:>9.2}  {:>8.2}  {max_fp:>7}  {:>7.1}%{flag}",
                sz_sum as f64 / n as f64,
                fp_sum as f64 / n as f64,
                pct(exact1),
            );
        }
    }

    // Recommendation: smallest budget whose Tool-k = 100% (min-fill can't rescue
    // a rank-3 truth), then fine-scan for the highest min_score still at 100%
    // recall — min-fill rescues rank-1 probes, so this ceiling sits above the raw
    // true-tool floor.
    let budget_floor = (1..=5).find(|&k| tool_k(k) == n).unwrap_or(5);
    let recall_at = |ms: f32, b: usize| -> bool {
        matrix
            .iter()
            .all(|(gt, s)| evaluate_selection(s, *gt, ms, b).0)
    };
    let hi = q(0.15);
    let steps = 300usize;
    let mut ceiling = 0.0f32;
    for i in 0..=steps {
        let ms = hi * i as f32 / steps as f32;
        if recall_at(ms, budget_floor) {
            ceiling = ms;
        }
    }
    let stats_at = |ms: f32, b: usize| -> (f64, f64, f64) {
        let (mut sz, mut fp, mut ex) = (0usize, 0usize, 0usize);
        for (gt, s) in &matrix {
            let (hit, size, f) = evaluate_selection(s, *gt, ms, b);
            sz += size;
            fp += f;
            if hit && size == 1 {
                ex += 1;
            }
        }
        (
            sz as f64 / n as f64,
            fp as f64 / n as f64,
            100.0 * ex as f64 / n as f64,
        )
    };
    // Ship a hair below the ceiling (10% margin) so a slightly weaker future
    // probe doesn't tip recall under 100%.
    let robust = ceiling * 0.90;
    let (c_sz, c_fp, c_ex) = stats_at(ceiling, budget_floor);
    let (r_sz, r_fp, r_ex) = stats_at(robust, budget_floor);
    let evict = gt_scores[0] * 0.75; // below the weakest true-tool score → never evict a correct tool

    println!("\n── recommended for 100% recall, minimal FP ──");
    println!("  budget 1..{budget_floor}   (Tool-{budget_floor} = 100%, so the truth is always reachable)");
    println!(
        "  ceiling  min_score {ceiling:.0}  → recall 100%, mean set {c_sz:.2}, mean FP {c_fp:.2}, exact-1 {c_ex:.1}%"
    );
    println!(
        "  robust   min_score {robust:.0}  (10% margin) → recall 100%, mean set {r_sz:.2}, mean FP {r_fp:.2}, exact-1 {r_ex:.1}%"
    );
    println!("  evict_score ≈ {evict:.0}  (below the weakest true-tool score {:.0} so a correct pick is never evicted)", gt_scores[0]);
    Ok(())
}

/// Popcount of XNOR agreement between two equal-length word slices.
fn word_agreement(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (!(x ^ y)).count_ones()).sum()
}

/// §81 — dissect a probe's scoring against the tag-scoped gallery, per fold
/// layer-group and per token, to locate (or rule out) a tool-identity signal
/// that the full-signature late-fusion loses.
fn belief_dissect(log: &mut LogFile, probe_id: StreamId, tag: &str, n_tokens: usize) -> Result<()> {
    use candle_conversation::provenance::wide_sig::{PROV_FOLD_SIZES, PROV_HEADS_PER_LAYER};
    use candle_conversation::provenance::{decode_wide_sigs, WideQSig};

    let substrate = build_substrate(log)?;

    let probe: Vec<WideQSig> = substrate
        .stream_of(probe_id)
        .and_then(|e| e.wide_q_sigs.as_ref())
        .and_then(|b| decode_wide_sigs(b))
        .with_context(|| format!("probe stream {} has no wide-Q signature", probe_id.0))?;
    let gt_name = substrate
        .stream_of(probe_id)
        .and_then(|e| match &e.decl {
            Some(StreamDecl::Turn(t)) => t.tags.iter().find(|x| x.as_str() != tag).cloned(),
            _ => None,
        })
        .with_context(|| "probe is not a tagged turn")?;

    // Gallery flattened to tokens, each carrying its tool slot (case). Excludes
    // the probe's own turn.
    let mut slot_names: Vec<String> = Vec::new();
    let mut gtoks: Vec<WideQSig> = Vec::new();
    let mut gcase: Vec<usize> = Vec::new();
    for (sid, e) in substrate.all_streams() {
        if sid == probe_id {
            continue;
        }
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        let Some(win) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        let slot = slot_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| {
                slot_names.push(name.clone());
                slot_names.len() - 1
            });
        for tokn in win {
            gtoks.push(tokn);
            gcase.push(slot);
        }
    }
    let n_slots = slot_names.len();
    let gt_slot = slot_names.iter().position(|n| n == &gt_name).unwrap();

    let shape = probe.first().or(gtoks.first()).context("no signatures")?;
    let wph = shape.words_per_head();
    let gw = PROV_HEADS_PER_LAYER * wph; // words per layer-group
    let n_groups = shape.n_heads as usize / PROV_HEADS_PER_LAYER;

    // Human labels for the fold groups from PROV_FOLD_SIZES ([46,1,1] → L0–45 / L46 / L47).
    let group_label = |g: usize| -> String {
        let mut lo = 0usize;
        for (i, &sz) in PROV_FOLD_SIZES.iter().enumerate() {
            if i == g {
                return if sz == 1 {
                    format!("L{lo}")
                } else {
                    format!("L{lo}\u{2013}{}", lo + sz - 1)
                };
            }
            lo += sz;
        }
        format!("g{g}")
    };

    // z-score late-fusion restricted to one group → per-tool votes.
    let group_scores = |g: usize| -> Vec<f32> {
        let base = g * gw;
        let n = gtoks.len() as f32;
        let mut votes = vec![0f32; n_slots];
        for q in &probe {
            if q.words.len() < base + gw {
                continue;
            }
            let qg = &q.words[base..base + gw];
            let (mut best_ag, mut best_case) = (0u32, usize::MAX);
            let (mut sum, mut sumsq) = (0u64, 0u64);
            for (j, cand) in gtoks.iter().enumerate() {
                if cand.words.len() < base + gw {
                    continue;
                }
                let ag = word_agreement(qg, &cand.words[base..base + gw]);
                if ag > best_ag {
                    best_ag = ag;
                    best_case = gcase[j];
                }
                sum += ag as u64;
                sumsq += (ag as u64) * (ag as u64);
            }
            if best_case != usize::MAX {
                let mean = sum as f32 / n;
                let var = (sumsq as f32 / n - mean * mean).max(1e-6);
                let z = ((best_ag as f32 - mean) / var.sqrt()).max(0.0);
                votes[best_case] += z;
            }
        }
        votes
    };

    let rank_of =
        |scores: &[f32], slot: usize| 1 + scores.iter().filter(|&&s| s > scores[slot]).count();
    let print_top = |scores: &[f32], k: usize| {
        let mut r: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (rank, (slot, sc)) in r.iter().enumerate().take(k) {
            let mark = if *slot == gt_slot {
                "  ← ground truth"
            } else {
                ""
            };
            println!("      #{rank:<2} {:<24} {sc:>8.2}{mark}", slot_names[*slot]);
        }
    };

    println!(
        "\n══ §81 dissect — probe {:#018x}  tool {gt_name:?}  ({} tokens) ══",
        probe_id.0,
        probe.len()
    );
    println!(
        "gallery: {} tokens over {n_slots} tools ({} groups × {PROV_HEADS_PER_LAYER} heads)\n",
        gtoks.len(),
        n_groups
    );

    // Fused (all groups) — the shipped scorer — for the baseline rank.
    let fused: Vec<f32> =
        (0..n_groups)
            .map(group_scores)
            .fold(vec![0f32; n_slots], |mut acc, g| {
                for (a, v) in acc.iter_mut().zip(&g) {
                    *a += v;
                }
                acc
            });
    println!(
        "FUSED (all layer-groups)  —  ground truth at #{}",
        rank_of(&fused, gt_slot)
    );
    print_top(&fused, 8);

    // Per-group rankings: does any single layer-group rank the true tool higher?
    for g in 0..n_groups {
        let s = group_scores(g);
        println!(
            "\ngroup {g} ({})  —  ground truth at #{}",
            group_label(g),
            rank_of(&s, gt_slot)
        );
        print_top(&s, 8);
    }

    // Token-level: the top rival that beats the truth in the fused score. For
    // each probe token, per group, the best agreement against the truth's tokens
    // vs the rival's tokens — tokens where truth wins in some group are the
    // discriminative signal we'd want to amplify.
    let rival_slot = {
        let mut r: Vec<(usize, f32)> = fused.iter().copied().enumerate().collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1));
        r.iter().map(|(s, _)| *s).find(|&s| s != gt_slot).unwrap()
    };
    println!(
        "\nPer-token discrimination — truth {gt_name:?} vs top rival {:?}:",
        slot_names[rival_slot]
    );
    println!(
        "  (agreement out of {} bits per group; ‘*’ = truth wins that group)",
        gw * 64
    );

    // Pre-split gallery tokens by the two tools of interest.
    let truth_toks: Vec<&WideQSig> = gtoks
        .iter()
        .zip(&gcase)
        .filter(|(_, c)| **c == gt_slot)
        .map(|(t, _)| t)
        .collect();
    let rival_toks: Vec<&WideQSig> = gtoks
        .iter()
        .zip(&gcase)
        .filter(|(_, c)| **c == rival_slot)
        .map(|(t, _)| t)
        .collect();
    let best_ag = |qg: &[u64], set: &[&WideQSig], base: usize| -> u32 {
        set.iter()
            .filter(|c| c.words.len() >= base + gw)
            .map(|c| word_agreement(qg, &c.words[base..base + gw]))
            .max()
            .unwrap_or(0)
    };

    // Rank probe tokens by how strongly the truth beats the rival in the identity
    // groups (1 + 2), surfacing the most discriminative first.
    let mut scored: Vec<(usize, [(u32, u32); 8])> = Vec::new(); // (token, per-group (truth, rival))
    for (ti, q) in probe.iter().enumerate() {
        let mut per_g = [(0u32, 0u32); 8];
        for g in 0..n_groups.min(8) {
            let base = g * gw;
            if q.words.len() < base + gw {
                continue;
            }
            let qg = &q.words[base..base + gw];
            per_g[g] = (
                best_ag(qg, &truth_toks, base),
                best_ag(qg, &rival_toks, base),
            );
        }
        scored.push((ti, per_g));
    }
    // Sort by summed truth-minus-rival margin over the identity groups (1..n).
    scored.sort_by(|a, b| {
        let m = |x: &[(u32, u32); 8]| -> i64 {
            (1..n_groups).map(|g| x[g].0 as i64 - x[g].1 as i64).sum()
        };
        m(&b.1).cmp(&m(&a.1))
    });

    print!("  {:>5}", "tok");
    for g in 0..n_groups {
        print!("   {:>14}", format!("{} (t/r)", group_label(g)));
    }
    println!();
    for (ti, per_g) in scored.iter().take(n_tokens) {
        print!("  {ti:>5}");
        for g in 0..n_groups {
            let (t, r) = per_g[g];
            let win = if t > r { "*" } else { " " };
            print!("   {t:>3}/{r:<3}{win:>7}");
        }
        println!();
    }
    Ok(())
}

/// Decode every tagged turn and flag any that never produced a completed
/// `</tool_call>` — the calibration decode was supposed to force a tool call, so
/// a turn without one is a prompt that made the model deliberate or refuse, and
/// its stored signature is off-tool (poisons that tool's reference set and fails
/// as a probe). Groups the offenders by tool and shows the user prompt.
fn calib_check(log: &mut LogFile, log_path: &std::path::Path, tag: &str) -> Result<()> {
    let tok = load_log_tokenizer(log_path)?
        .context("no tokenizer.json sidecar next to the log — cannot decode turns")?;
    let substrate = build_substrate(log)?;

    // Collect tagged turns with a Tokens record: (stream, tool, loc).
    let mut turns: Vec<(
        StreamId,
        String,
        candle_conversation::persistence::manifest::RecordLoc,
    )> = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        if let Some(loc) = e.tokens {
            turns.push((sid, name.clone(), loc));
        }
    }

    // Decode each and classify: no `<tool_call>` at all (a refusal/deliberation),
    // or an opening tag with no closing `</tool_call>` (truncated).
    let mut no_call: Vec<(String, StreamId, String, usize)> = Vec::new();
    let mut truncated: Vec<(String, StreamId, String, usize)> = Vec::new();
    for (sid, name, loc) in &turns {
        let rec = read_record_at(log, loc.offset, loc.record_size)?;
        let ids = decode_token_ids(&rec.payload)?;
        let text = tok.decode(&ids, false).unwrap_or_default();
        let prompt = text
            .split("<|im_end|>")
            .next()
            .unwrap_or("")
            .trim()
            .replace('\n', " ");
        if !text.contains("<tool_call>") {
            no_call.push((name.clone(), *sid, prompt, ids.len()));
        } else if !text.contains("</tool_call>") {
            truncated.push((name.clone(), *sid, prompt, ids.len()));
        }
    }
    no_call.sort();
    truncated.sort();

    println!("\n══ calibration tool-call audit  (tag {tag:?}) ══\n");
    println!("scanned {} tagged turns", turns.len());
    println!(
        "  {} with a completed tool call, {} missing any <tool_call>, {} truncated (no </tool_call>)\n",
        turns.len() - no_call.len() - truncated.len(),
        no_call.len(),
        truncated.len()
    );

    if !no_call.is_empty() {
        println!(
            "── NO tool call (model deliberated/refused) — these poison the tool's signature ──"
        );
        for (name, sid, prompt, ntok) in &no_call {
            println!("  {name:<26} {:#018x}  ({ntok} tok)  “{prompt}”", sid.0);
        }
    }
    if !truncated.is_empty() {
        println!("\n── truncated (opened <tool_call> but no close) — hit the decode cap ──");
        for (name, sid, prompt, ntok) in &truncated {
            println!("  {name:<26} {:#018x}  ({ntok} tok)  “{prompt}”", sid.0);
        }
    }
    if no_call.is_empty() && truncated.is_empty() {
        println!("All tagged turns produced a completed tool call. ✓");
    }
    Ok(())
}

/// FNV-1a over bytes — a small, stable (version-independent) hash so baseline
/// files diff identically across separate `substrate_inspect` builds.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv1a_ids(ids: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Dump a diffable per-turn calibration baseline keyed by `tool|prompt`.
fn calib_baseline(
    log: &mut LogFile,
    log_path: &std::path::Path,
    tag: &str,
    out: Option<String>,
) -> Result<()> {
    use candle_conversation::provenance::decode_wide_sigs;

    let tok = load_log_tokenizer(log_path)?
        .context("no tokenizer.json sidecar next to the log — cannot decode turns")?;
    let substrate = build_substrate(log)?;

    // (stream, tool, tokens_loc, wide_q_blob) for every tagged turn.
    struct Src {
        tool: String,
        loc: candle_conversation::persistence::manifest::RecordLoc,
        wq: Option<Vec<u8>>,
    }
    let mut srcs: Vec<Src> = Vec::new();
    for (_sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        if !t.tags.iter().any(|x| x == tag) {
            continue;
        }
        let Some(name) = t.tags.iter().find(|x| x.as_str() != tag) else {
            continue;
        };
        let Some(loc) = e.tokens else { continue };
        srcs.push(Src {
            tool: name.clone(),
            loc,
            wq: e.wide_q_sigs.clone(),
        });
    }

    // Decode each and compute the stable per-turn fingerprint.
    let mut rows: Vec<(String, usize, u64, usize, u64, f64)> = Vec::new();
    let mut without_sig = 0usize;
    for s in &srcs {
        let rec = read_record_at(log, s.loc.offset, s.loc.record_size)?;
        let ids = decode_token_ids(&rec.payload)?;
        let text = tok.decode(&ids, false).unwrap_or_default();
        // The prompt prefix (before the first `<|im_end|>`) is the authored
        // example — stable across rebuilds, so it keys the row.
        let prompt = text
            .split("<|im_end|>")
            .next()
            .unwrap_or("")
            .trim()
            .replace('\n', " ");
        let key = format!("{}|{}", s.tool, prompt);
        let tok_hash = fnv1a_ids(&ids);
        let (n_sig, wq_hash, mean_pop) = match &s.wq {
            Some(b) => {
                let wq_hash = fnv1a(b);
                match decode_wide_sigs(b) {
                    Some(w) if !w.is_empty() => {
                        let mp =
                            w.iter().map(|x| x.popcount() as f64).sum::<f64>() / w.len() as f64;
                        (w.len(), wq_hash, mp)
                    }
                    _ => (0, wq_hash, 0.0),
                }
            }
            None => {
                without_sig += 1;
                (0, 0, 0.0)
            }
        };
        rows.push((key, ids.len(), tok_hash, n_sig, wq_hash, mean_pop));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    // Emit TSV: a header comment, then one row per turn. Hashes are the compare
    // keys; token/sig counts and mean popcount aid diagnosis when a hash moves.
    let mut buf = String::new();
    buf.push_str(&format!(
        "# calib-baseline  tag={tag}  turns={}  ({} without wide-Q)\n",
        rows.len(),
        without_sig
    ));
    buf.push_str("# key\tn_tok\ttok_hash\tn_sig\twq_hash\tmean_pop\n");
    for (key, n_tok, th, n_sig, wh, mp) in &rows {
        buf.push_str(&format!(
            "{key}\t{n_tok}\t{th:016x}\t{n_sig}\t{wh:016x}\t{mp:.1}\n"
        ));
    }

    match out {
        Some(path) => {
            std::fs::write(&path, &buf).with_context(|| format!("writing baseline to {path}"))?;
            println!(
                "wrote calibration baseline: {} turns → {path}  ({} without wide-Q)",
                rows.len(),
                without_sig
            );
        }
        None => print!("{buf}"),
    }
    Ok(())
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
fn dump(log: &mut LogFile, only_timeline: Option<u64>, full: bool) -> Result<()> {
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
        let kv_per_layer = if n_layers > 0 {
            kv_tok / n_layers
        } else {
            kv_tok
        };
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

/// Summarise a turn's stored wide-Q signature window for the stream listing:
/// `wsig=<tokens>tok×<heads>h pop=<mean set bits>/<total bits>`. The mean popcount
/// should sit near half the total bits — that's the signal the signs are real.
fn wide_sig_summary(bytes: Option<&[u8]>) -> String {
    match bytes {
        None => "wsig=n".to_string(),
        Some(b) => match candle_conversation::provenance::decode_wide_sigs(b) {
            Some(w) if !w.is_empty() => {
                let toks = w.len();
                let heads = w[0].n_heads as usize;
                let total_bits = heads * w[0].words_per_head() * 64;
                let mean_pop = w.iter().map(|s| s.popcount() as f64).sum::<f64>() / toks as f64;
                format!("wsig={toks}tok×{heads}h pop={mean_pop:.0}/{total_bits}")
            }
            _ => "wsig=bad".to_string(),
        },
    }
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
        let wsig = wide_sig_summary(entry.wide_q_sigs.as_deref());
        let detail = match &entry.decl {
            Some(StreamDecl::Turn(t)) => {
                let no_think =
                    candle_conversation::turn_layout::TurnLayout::new(t.segments.clone())
                        .no_think();
                format!(
                    "exchange  idx={} chunks={} tok={} {}  no_think={}  conv={}  tags={:?}",
                    t.turn_index,
                    entry.chunks.len(),
                    tok,
                    wsig,
                    no_think,
                    t.timeline_id,
                    t.tags,
                )
            }
            Some(StreamDecl::PromptSection(s)) => format!(
                "section \"{}\"  chunks={} tok={}",
                s.debug_name,
                entry.chunks.len(),
                tok,
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
    let entry = substrate
        .stream_of(stream_id)
        .with_context(|| format!("stream {} not found", stream_hex(stream_id.0)))?;
    let tokens_loc = entry.tokens;
    // A distilled turn's `Tokens` record is reclaimed, but its `StreamDecl` still
    // carries the verbatim user/assistant text — surface that instead of failing.
    let decl_text = match &entry.decl {
        Some(StreamDecl::Turn(t)) => {
            let layout = candle_conversation::turn_layout::TurnLayout::new(t.segments.clone());
            Some((
                layout.user_text().to_string(),
                layout.assistant_text().unwrap_or_default(),
            ))
        }
        _ => None,
    };
    let loc = match tokens_loc {
        Some(loc) => loc,
        None => match decl_text {
            Some((user, assistant)) => {
                println!(
                    "stream {}  (no Tokens record — distilled; decl text below)\n",
                    stream_hex(stream_id.0)
                );
                println!("── user ──\n{user}\n");
                println!("── assistant ──\n{assistant}");
                return Ok(());
            }
            None => anyhow::bail!(
                "stream {} has no Tokens record and no turn decl",
                stream_hex(stream_id.0)
            ),
        },
    };
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
    RecordType::WideQSig,
    RecordType::Commit,
    RecordType::Checkpoint,
    RecordType::Tokenizer,
    RecordType::Label,
    RecordType::ConvState,
    RecordType::TreeMetadata,
    RecordType::DebugId,
    RecordType::Tombstone,
    RecordType::Distilled,
    RecordType::ProjectionEvents,
    RecordType::Unknown,
];

fn type_index(rt: RecordType) -> usize {
    rt as usize - 1
}

fn build_manifest(log: &mut LogFile) -> Result<Manifest> {
    Ok(Manifest::build_from_walk(log, SUPERBLOCK_SIZE)?.0)
}

/// Rebuild the in-RAM substrate from the log — the authoritative per-stream
/// index. The manifest only carries singleton offsets (model spec, template,
/// tokenizer, checkpoint); streams, chunks, and tokens live in the substrate,
/// populated by the same walker pass `open_in_with_substrate` uses.
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
