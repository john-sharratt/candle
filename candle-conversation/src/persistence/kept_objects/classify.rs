//! Which custom objects a move-aside keeps.
//!
//! Every live object is kept but those whose meaning is a model-derived
//! record, which the move-aside destroys. The one list of them is
//! [`LEFT_OUT`]; a key under none of its prefixes is kept.

/// The prefixes whose objects are left out of an export:
///
/// - `cluster/` — the cluster's own state, rebuilt when the app starts over;
/// - `recurrent/` — recurrent reply rows, model state;
/// - `working_set:` — marks on turns that no longer exist; the locks they
///   rebuild go with the dialogue;
/// - `channels/` — sections npcd submits to a character's conversation,
///   submitted again when the character's conversation is rebuilt.
pub const LEFT_OUT: &[&str] = &["cluster/", "recurrent/", "working_set:", "channels/"];

/// Whether the object keyed `key` is kept.
pub fn is_kept(key: &str) -> bool {
    !LEFT_OUT.iter().any(|p| key.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_left_out_prefixes_are_left_out() {
        assert!(!is_kept("cluster/members/a"));
        assert!(!is_kept("recurrent/abc"));
        assert!(!is_kept("working_set:12:3"));
        assert!(!is_kept("channels/w/npc"));
        assert!(is_kept("accounts/google:1"));
        assert!(is_kept("mcp/conversation/x"));
        assert!(is_kept("clusterish"), "a prefix is matched whole");
    }
}
