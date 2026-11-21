//! Maps a client's conversation id to its timeline and names ingested units for
//! the layer views.
//!
//! A timeline is a stable hash of the conversation id, so a client that
//! reconnects after a restart reaches the timeline the substrate reload
//! recovered. A passthrough id carries its own timeline. An ingested unit is
//! keyed by its workspace-relative path, and the view names it by its
//! repository and the path inside it. A title falls back from the sidebar label
//! to the conversation id to the path a file-scoped ingest recorded, so those
//! conversations do not read as untitled.

use super::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// The timeline a sidebar `id` names: a passthrough conversation's id carries
/// its timeline ([`passthrough::timeline_of`]); any other is [`timeline_for`].
pub(super) fn conversation_timeline(conv_id: &str) -> projection::TimelineId {
    passthrough::timeline_of(conv_id).unwrap_or_else(|| timeline_for(conv_id))
}

/// Close the tool sessions `conv_id` opened, now that it will run no more
/// rounds (`docs/mcp_file_tools.md` §4.2), and say so when it held any.
pub(super) fn close_tool_sessions(state: &InferenceState, conv_id: &str, why: &str) {
    let closed = state.tool_host.close_sessions(conv_id);
    if closed > 0 {
        tracing::info!(conv_id, closed, "tool sessions closed: conversation {why}");
    }
}

/// The [`projection::TimelineId`] a client `conv_id` maps to — a stable hash
/// of the id. Deterministic across daemon restarts, so a reconnecting client
/// resolves to the same timeline the substrate reload recovered (§16.12).
pub fn timeline_for(conv_id: &str) -> projection::TimelineId {
    let h = content_hash::hash_bytes(conv_id.as_bytes());
    projection::TimelineId::from_raw(h.lo.max(1)).expect("timeline id is non-zero")
}

/// What a committed ingested unit is keyed by, workspace-relative: a file's
/// reading by its path, a folder's by its folder, a layer's unit by its
/// subject — the file or folder it analyses. `None` for a conversation that is
/// no committed unit.
pub(super) fn committed_unit_key(meta: &POrdMap<String, String>) -> Option<&String> {
    meta.get(PATH_KEY)
        .or_else(|| meta.get(DIR_KEY))
        .or_else(|| meta.get(SUBJECT_KEY))
        .filter(|_| meta.contains_key(CONTENT_KEY))
}

/// How zend names one of its conversations in a layer's list. An ingested unit
/// — the internal content layers (repo_map, code_reading, the analysis layers)
/// — the way the tools address it: its repository, the path inside it, and the
/// branches or commit that tell one version from another.
pub(super) fn unit_of(s: &Substrate, tl: TimelineId) -> Unit {
    // Title precedence: the sidebar label, then the conv_id, then the
    // conversation's `path` metadata — code_reading (and any file-scoped
    // ingest) records the file path there at creation, so those otherwise
    // conv_id-less timelines read as the file they cover instead of
    // "(untitled)".
    let custom = s.custom_of(tl);
    let label = title(s, tl)
        .or_else(|| {
            custom
                .and_then(|m| m.get(PATH_KEY).or_else(|| m.get(SUBJECT_KEY)))
                .filter(|p| !p.is_empty())
                .cloned()
        })
        .unwrap_or_default();
    let branches = custom
        .and_then(|m| m.get(BRANCHES_KEY))
        .map(|v| {
            v.split(',')
                .filter(|b| !b.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let commit = custom
        .and_then(|m| m.get(COMMIT_KEY))
        .cloned()
        .unwrap_or_default();
    // An ingested unit is keyed workspace-relative (`candle/AGENTS.md`); the
    // view shows it the way the tools address it — its repository, and the
    // path inside it.
    let (repo, label) = match custom.and_then(committed_unit_key) {
        Some(key) => {
            let (repo, inner) = shown_unit(key);
            (repo.to_string(), inner)
        }
        None => (String::new(), label),
    };
    Unit {
        label,
        repo,
        branches,
        commit,
    }
}
