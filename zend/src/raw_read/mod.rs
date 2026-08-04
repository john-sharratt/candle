//! Raw ChatML ingestion — the "raw" loading mode.
//!
//! Reads every file under a content folder, parses its ChatML records, and
//! prefills them into a projection layer as turns — with **no** grouping,
//! carving, or summarisation. Where the deterministic modes (`repo_scan`,
//! `code_read`) derive structure from source code, raw mode holds conversation
//! records verbatim: a file of `<|im_start|>role … <|im_end|>` blocks becomes a
//! run of `(user, assistant)` turns on the layer's owning sequence.
//!
//! Refresh mirrors `repo_scan`: the folder's per-file content hashes are the
//! change record ([`RawState`]); when they move, a fresh timeline is minted,
//! re-prefilled, and swapped in while the old one is tombstoned.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use ignore::WalkBuilder;
use sha2::{Digest, Sha256};

use candle_conversation::projection::{self, TimelineId};
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};

use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan::utility_config;
use crate::repo_scan::walk::MAX_FILE_BYTES;
use crate::turn_sink::{InsertTurnSink, SequenceTurnSink};
use crate::types::Role;

/// Per-file content-hash record for a raw layer — the change detector the
/// refresh path consults so a filesystem event rebuilds only when a record
/// file's content actually moved.
#[derive(Debug, Clone, Default)]
pub struct RawState {
    /// Folder-relative path → content hash.
    pub file_hashes: BTreeMap<String, String>,
}

impl RawState {
    fn equivalent_to(&self, other: &RawState) -> bool {
        self.file_hashes == other.file_hashes
    }
}

/// Outcome of an atomic [`refresh_raw`] call — same contract as
/// [`crate::repo_scan::RefreshOutcome`].
#[allow(clippy::large_enum_variant)]
pub enum RefreshOutcome {
    NoOp,
    Replaced { sequence: Sequence, state: RawState },
}

/// Parse ChatML records out of `content`.
///
/// Canonical form is `<|im_start|>role\n…<|im_end|>` blocks; each becomes one
/// `(role, text)` segment. If the content carries no ChatML delimiters at all it
/// is treated as the *decoded* marker-line form (role words on their own lines)
/// via [`crate::chatml::split_turn`], so both a raw model transcript and a
/// human-authored `user:` / `assistant:` file load the same way.
pub fn parse_chatml_records(content: &str) -> Vec<(Role, String)> {
    const OPEN: &str = "<|im_start|>";
    const CLOSE: &str = "<|im_end|>";

    let mut out = Vec::new();
    let mut rest = content;
    while let Some(i) = rest.find(OPEN) {
        rest = &rest[i + OPEN.len()..];
        let (header, body) = rest.split_once('\n').unwrap_or((rest, ""));
        let role = role_from(header.trim());
        let (text, after) = match body.find(CLOSE) {
            Some(e) => (&body[..e], &body[e + CLOSE.len()..]),
            None => (body, ""),
        };
        let text = text.trim();
        if !text.is_empty() {
            out.push((role, text.to_string()));
        }
        rest = after;
    }
    if out.is_empty() && !content.trim().is_empty() {
        return crate::chatml::split_turn(Role::User, content);
    }
    out
}

fn role_from(header: &str) -> Role {
    match header {
        "assistant" => Role::Assistant,
        "system" => Role::System,
        _ => Role::User,
    }
}

/// Fold parsed ChatML records into `(user, assistant)` turns — the substrate's
/// turn shape. Consecutive `system`/`user` records accumulate into the next
/// turn's user side; an `assistant` record closes the turn. A trailing user with
/// no reply is held as a turn with an empty assistant side.
pub fn records_to_turns(records: Vec<(Role, String)>) -> Vec<(String, String)> {
    let mut turns = Vec::new();
    let mut user_buf: Vec<String> = Vec::new();
    for (role, text) in records {
        match role {
            Role::Assistant => {
                turns.push((user_buf.join("\n\n"), text));
                user_buf.clear();
            }
            // system + user accumulate as the user side of the next turn.
            _ => user_buf.push(text),
        }
    }
    if !user_buf.is_empty() {
        turns.push((user_buf.join("\n\n"), String::new()));
    }
    turns
}

fn content_hash(path: &str, bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(path.as_bytes());
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Read every UTF-8 file under `root` as a raw record file, folder-relative path
/// paired with its content. Honours the same ignore rules as the workspace walk
/// (`.gitignore`, hidden files, size cap); non-UTF-8 (binary) files are skipped.
fn read_raw_files(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let walker = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true)
        .require_git(false)
        .follow_links(false)
        .build();
    for dent in walker.flatten() {
        let path = dent.path();
        if !dent.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        if dent.metadata().map(|m| m.len()).unwrap_or(0) > MAX_FILE_BYTES {
            continue;
        }
        let Ok(bytes) = fs::read(path) else {
            continue;
        };
        let Ok(content) = String::from_utf8(bytes) else {
            continue; // binary — not a ChatML record file
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        out.push((rel, content));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Prefill each record file's ChatML turns onto `sink`, recording the per-file
/// content hashes into a fresh [`RawState`]. Shared by ingest + refresh.
fn ingest_raw_into_sink<S: InsertTurnSink>(
    sink: &mut S,
    root: &Path,
    progress: &LoadProgress,
) -> anyhow::Result<RawState> {
    let files = read_raw_files(root);
    let total = files.len();
    let mut state = RawState::default();
    for (i, (rel, content)) in files.iter().enumerate() {
        let turns = records_to_turns(parse_chatml_records(content));
        for (user, assistant) in &turns {
            sink.insert_prefill_turn(user, assistant, vec!["raw".to_string(), rel.clone()])?;
        }
        state
            .file_hashes
            .insert(rel.clone(), content_hash(rel, content.as_bytes()));
        progress.set_step_progress((i + 1) as u64, total as u64);
    }
    Ok(state)
}

/// Local mirror of the deterministic modes' `layer_system_prompt`: pull the
/// layer's authored sections out of the schema and wrap them in the dialect's
/// system-prompt markers. A raw layer only projects from the layer it targets.
fn layer_system_prompt(
    builder: &projection::Builder,
    layer_name: &str,
    config: &SequenceConfig,
) -> String {
    use projection::SystemPromptItem;
    debug_assert!(
        builder.schema().layers.iter().any(|l| l.name == layer_name),
        "projection schema missing '{layer_name}' layer"
    );
    // Every ingest conversation frames on the single shared system prompt.
    let mut body = String::new();
    for item in &builder.schema().system_prompt.items {
        if let SystemPromptItem::Section(s) = item {
            body.push_str(&s.content);
        }
    }
    config.dialect.format_system_prompt(&body)
}

/// Top-level raw ingestion — creates the layer's owning [`Sequence`] and prefills
/// every ChatML record under `root` into it. Returns the sequence (held by the
/// daemon so its sealed K/V stays reachable by dialogue retrieval) and the
/// per-file [`RawState`] for the refresh path.
pub fn ingest_raw(
    engine: &Mutex<ConversationEngine>,
    proj_builder: projection::Builder,
    root: &Path,
    config: SequenceConfig,
    progress: &LoadProgress,
    layer_name: &str,
    group_name: &str,
) -> anyhow::Result<(Sequence, RawState)> {
    let layer = proj_builder
        .id_for_layer(layer_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
    let group = proj_builder
        .id_for_group(group_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{group_name}' group"))?;
    let system_prompt = layer_system_prompt(&proj_builder, layer_name, &config);
    // Lock only to mint the sequence; the prefill below runs on the sequence's
    // own handle, lock-free, so concurrent engine consumers keep running.
    let mut sequence = {
        let engine = engine.lock().unwrap();
        engine
            .new_conversation_with_projection(
                &system_prompt,
                proj_builder,
                layer,
                group,
                utility_config(config),
            )
            .map_err(|e| anyhow::anyhow!("{layer_name} conv create: {e}"))?
    };

    let mut sink = SequenceTurnSink::new(&mut sequence);
    let state = ingest_raw_into_sink(&mut sink, root, progress)?;
    Ok((sequence, state))
}

/// Atomic refresh of a raw layer when any record file's content hash changed.
/// Mirrors [`crate::repo_scan::refresh_repo_map`]: re-read the folder, and on a
/// hash change mint a fresh timeline, re-prefill every record, then tombstone the
/// old timeline. Stale-better-than-missing holds throughout.
pub fn refresh_raw(
    ctx: &RefreshContext<'_>,
    root: &Path,
    prior: &RawState,
    old_timeline: TimelineId,
    progress: &LoadProgress,
    layer_name: &str,
    group_name: &str,
) -> anyhow::Result<RefreshOutcome> {
    let files = read_raw_files(root);
    let mut next = RawState::default();
    for (rel, content) in &files {
        next.file_hashes
            .insert(rel.clone(), content_hash(rel, content.as_bytes()));
    }
    if prior.equivalent_to(&next) {
        tracing::trace!("raw refresh: no record hash changed, skipping");
        return Ok(RefreshOutcome::NoOp);
    }
    tracing::info!(
        n_records = files.len(),
        "raw refresh: minting new timeline and re-prefilling records",
    );

    let layer = ctx
        .proj_builder
        .id_for_layer(layer_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
    let group = ctx
        .proj_builder
        .id_for_group(group_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{group_name}' group"))?;
    let system_prompt = layer_system_prompt(&ctx.proj_builder, layer_name, &ctx.config);

    let mut new_sequence = {
        let engine = ctx.engine.lock().unwrap();
        engine
            .new_conversation_with_projection(
                &system_prompt,
                ctx.proj_builder.clone(),
                layer,
                group,
                utility_config(ctx.config.clone()),
            )
            .map_err(|e| anyhow::anyhow!("raw refresh: new conv create: {e}"))?
    };

    {
        let mut sink = SequenceTurnSink::new(&mut new_sequence);
        let total = files.len();
        for (i, (rel, content)) in files.iter().enumerate() {
            let turns = records_to_turns(parse_chatml_records(content));
            for (user, assistant) in &turns {
                sink.insert_prefill_turn(user, assistant, vec!["raw".to_string(), rel.clone()])?;
            }
            progress.set_step_progress((i + 1) as u64, total as u64);
        }
    }

    {
        let engine = ctx.engine.lock().unwrap();
        engine
            .tombstone_timeline(old_timeline)
            .map_err(|e| anyhow::anyhow!("raw refresh: tombstone old timeline: {e}"))?;
    }

    Ok(RefreshOutcome::Replaced {
        sequence: new_sequence,
        state: next,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn_sink::RecordingTurnSink;

    #[test]
    fn parses_canonical_chatml_blocks() {
        let src = "<|im_start|>system\nBe kind<|im_end|>\n\
                   <|im_start|>user\nHi<|im_end|>\n\
                   <|im_start|>assistant\nHello<|im_end|>";
        assert_eq!(
            parse_chatml_records(src),
            vec![
                (Role::System, "Be kind".to_string()),
                (Role::User, "Hi".to_string()),
                (Role::Assistant, "Hello".to_string()),
            ]
        );
    }

    #[test]
    fn falls_back_to_marker_lines_without_delimiters() {
        // No <|im_start|> markers → decoded marker-line form via split_turn.
        let src = "user\nHi there\nassistant\nHello back";
        assert_eq!(
            parse_chatml_records(src),
            vec![
                (Role::User, "Hi there".to_string()),
                (Role::Assistant, "Hello back".to_string()),
            ]
        );
    }

    #[test]
    fn folds_system_and_user_into_the_next_turn() {
        let records = vec![
            (Role::System, "Be kind".to_string()),
            (Role::User, "Hi".to_string()),
            (Role::Assistant, "Hello".to_string()),
        ];
        assert_eq!(
            records_to_turns(records),
            vec![("Be kind\n\nHi".to_string(), "Hello".to_string())]
        );
    }

    #[test]
    fn trailing_user_without_reply_is_held_with_empty_assistant() {
        let records = vec![
            (Role::User, "q1".to_string()),
            (Role::Assistant, "a1".to_string()),
            (Role::User, "q2".to_string()),
        ];
        assert_eq!(
            records_to_turns(records),
            vec![
                ("q1".to_string(), "a1".to_string()),
                ("q2".to_string(), String::new()),
            ]
        );
    }

    #[test]
    fn reads_and_prefills_records_from_a_folder() {
        let dir = std::env::temp_dir().join(format!("zend_raw_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("a.chatml"),
            "<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\npong<|im_end|>",
        )
        .unwrap();
        fs::write(dir.join("b.chatml"), "user\nhi\nassistant\nyo").unwrap();

        let mut sink = RecordingTurnSink::new();
        let progress = LoadProgress::silent();
        let state = ingest_raw_into_sink(&mut sink, &dir, &progress).unwrap();

        // Two files → two record turns, sorted by path (a before b), each tagged
        // with its folder-relative path.
        assert_eq!(sink.turns.len(), 2);
        assert_eq!(sink.turns[0].0, "ping");
        assert_eq!(sink.turns[0].1, "pong");
        assert_eq!(
            sink.turns[0].2,
            vec!["raw".to_string(), "a.chatml".to_string()]
        );
        assert_eq!(sink.turns[1].0, "hi");
        assert_eq!(sink.turns[1].1, "yo");
        assert_eq!(state.file_hashes.len(), 2);

        let _ = fs::remove_dir_all(&dir);
    }
}
