//! What every Zen tool does to what lasts (`docs/private_mode.md` §4.3): the
//! one table, for every server behind the aggregator, of which tools only
//! read, which write, and which are diaries'. A tool the table does not name
//! is a write. Each server's tests hold every tool it lists to being named
//! here, so a tool added without a line here fails the build.

use serde_json::Value;

/// What a tool does to what lasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// It only reads: allowed in a private conversation.
    Read,
    /// It writes something that lasts, or tells someone: refused in a
    /// private conversation.
    Write,
    /// A diary's: allowed only in a private conversation (§5).
    Diary,
}

use Access::{Diary, Read, Write};

/// Every tool, by name, and what it does.
pub const TOOLS: &[(&str, Access)] = &[
    // The aggregator's own.
    ("guide_list", Read),
    ("guide_read", Read),
    // zend: the memory.
    ("get_user_context", Read),
    ("journal_read", Read),
    ("journal_grep", Read),
    ("journal_note", Write),
    ("journal_focus", Write),
    ("journal_edit", Write),
    ("journal_forget", Write),
    ("journal_create", Write),
    ("journal_link", Write),
    ("journal_accept", Write),
    ("group_create", Write),
    ("group_describe", Write),
    ("group_add", Write),
    ("group_remove", Write),
    ("group_default", Write),
    // zend: the aggregator's own calls on it, refused from a client. A ref's
    // clock is kept for a private conversation as for any other (§3.2).
    ("user_episode_note", Write),
    ("conversation_ref_seen", Read),
    // zend: diaries.
    ("diary_open", Diary),
    ("diary_create", Diary),
    ("diary_list", Diary),
    ("diary_read", Diary),
    ("diary_write", Diary),
    ("diary_delete", Diary),
    ("diary_rotate", Diary),
    ("diary_close", Diary),
    // zend: threads, tasks and chats.
    ("thread_attach", Write),
    ("thread_list", Read),
    ("thread_archive", Write),
    ("repo_task_start", Write),
    ("repo_task_reply", Write),
    ("repo_task_nudge", Write),
    ("zen_chat_send", Write),
    ("zen_chat_read", Read),
    ("zen_chat_close", Write),
    // zend: files.
    ("file_read", Read),
    ("file_list", Read),
    ("file_search", Read),
    ("file_edit_and_commit", Write),
    ("file_download_and_commit", Write),
    ("file_upload_and_commit", Write),
    // zend: jobs and operations.
    ("job_list", Read),
    ("job_status", Read),
    ("job_read", Read),
    ("job_grep", Read),
    ("job_cancel", Write),
    ("cluster_status", Read),
    ("cluster_audit", Read),
    ("member_status", Read),
    ("member_memory", Read),
    ("log_list", Read),
    ("log_read", Read),
    ("log_grep", Read),
    ("cluster_command", Write),
    ("gateway_command", Write),
    ("approval_grant", Write),
    ("repo_job_start", Write),
    ("service_deploy", Write),
    ("service_status", Read),
    ("service_control", Write),
    ("layer_read", Read),
    ("layer_toggle", Write),
    ("timeline_read", Read),
    ("timeline_delete", Write),
    // npcd: bots.
    ("bot_list", Read),
    ("bot_mission_read", Read),
    ("bot_message_read", Read),
    ("bot_channel_read", Read),
    ("bot_discord_read", Read),
    ("bot_thread_read", Read),
    ("bot_mission_give", Write),
    ("bot_mission_step", Write),
    ("bot_mission_cancel", Write),
    ("bot_message_send", Write),
    ("bot_channel_post", Write),
    // npcd: images — a prompt is kept with what it drew.
    ("image_generate", Write),
    ("image_edit", Write),
    ("image_upload", Write),
    // zenj: the DJ and the house.
    ("zenj_now", Read),
    ("zenj_events", Read),
    ("zenj_queue", Write),
    ("zenj_request", Write),
    ("zenj_hands", Write),
    ("zenj_rekordbox", Write),
    ("zenj_moment", Write),
    ("zenj_sfx", Write),
    ("zenj_said", Write),
    ("zenj_midnight", Write),
    ("zenj_energy", Write),
    ("lights_scene", Write),
    ("lights_house", Write),
];

/// What a refused write is answered in a private conversation.
pub const PRIVATE: &str = "not available in private mode";

/// What a diary call is answered outside one.
pub const DIARIES_PRIVATE: &str = "diaries are available only in private mode: begin a \
    conversation with get_user_context {private_mode: true} to use them";

/// What `tool` does, when the table names it.
pub fn classified(tool: &str) -> Option<Access> {
    TOOLS.iter().find(|(name, _)| *name == tool).map(|(_, a)| *a)
}

/// What `tool` does: a tool the table does not name is a write.
pub fn access(tool: &str) -> Access {
    classified(tool).unwrap_or(Write)
}

/// Whether `tool` may be called in a conversation that is private or not;
/// why not, when it may not.
pub fn permits(tool: &str, private: bool) -> Result<(), &'static str> {
    match (access(tool), private) {
        (Write, true) => Err(PRIVATE),
        (Diary, false) => Err(DIARIES_PRIVATE),
        _ => Ok(()),
    }
}

/// The tools of `listed` — a server's `tools/list` — the table does not
/// name: what each server's tests hold empty.
pub fn unclassified(listed: &Value) -> Vec<String> {
    listed
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t.get("name").and_then(Value::as_str))
        .filter(|name| classified(name).is_none())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::json;

    use super::*;

    #[test]
    fn every_tool_is_named_once() {
        let mut seen = BTreeSet::new();
        for (name, _) in TOOLS {
            assert!(seen.insert(*name), "{name} is named twice");
        }
    }

    /// **A private conversation reads and keeps diaries; a normal one does
    /// everything but diaries**; an unnamed tool is a write.
    #[test]
    fn private_refuses_writes_and_only_private_keeps_diaries() {
        assert_eq!(permits("journal_read", true), Ok(()));
        assert_eq!(permits("journal_note", true), Err(PRIVATE));
        assert_eq!(permits("journal_note", false), Ok(()));
        assert_eq!(permits("diary_read", true), Ok(()));
        assert_eq!(permits("diary_read", false), Err(DIARIES_PRIVATE));
        assert_eq!(access("a_tool_nobody_named"), Write);
        assert_eq!(permits("a_tool_nobody_named", true), Err(PRIVATE));
        assert_eq!(classified("a_tool_nobody_named"), None);
    }

    #[test]
    fn a_listing_names_the_tools_the_table_does_not() {
        let listed = json!([{"name": "journal_read"}, {"name": "brand_new"}, {}]);
        assert_eq!(unclassified(&listed), ["brand_new"]);
    }
}
