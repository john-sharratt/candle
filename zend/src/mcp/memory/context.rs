//! `get_user_context` (§6.4, §6.5): the frame for this conversation — its
//! own journal, each journal above it, the person, each group it frames — within the
//! channel's budget, and how to keep and forget something.

use journals::path::JournalPath;
use journals::scope::Channel;
use journals::slug::Slug;
use journals::time::Stamp;
use journals::tree::Tree;
use journals::view::frame::{frame, Budget};

/// What a channel is given: a voice channel its digests, a chat channel its frame.
pub fn budget(channel: Channel) -> Budget {
    match channel {
        Channel::Grok => Budget::Voice,
        Channel::ClaudeCode | Channel::Other => Budget::Chat,
    }
}

/// The tools' own words, after the frame.
pub const HOW: &str =
    "To keep something for next time: journal_note {journal: \"~\", text} — or a \
     group's journal, `@group/topics/…`, which every member's conversations read. To forget one: \
     journal_forget {journal, says}. journal_read and journal_grep reach everything they read.";

pub fn render(
    tree: &Tree,
    person: &Slug,
    conversation: Option<&JournalPath>,
    channel: Channel,
    now: Stamp,
) -> String {
    let mut out = frame(tree, person, conversation, &[], budget(channel), now);
    out.push('\n');
    out.push_str(HOW);
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use journals::tree::load::load;
    use journals::view::NOT_INSTRUCTIONS;

    use super::*;

    #[test]
    fn a_voice_channel_gets_digests_and_a_chat_channel_the_frame() {
        assert_eq!(budget(Channel::Grok), Budget::Voice);
        assert_eq!(budget(Channel::ClaudeCode), Budget::Chat);
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([(
            "persons/john/person.yaml".to_string(),
            b"person: john\nname: \"John\"\naccounts: []\ngroups: []\n".to_vec(),
        )]);
        let t = load(&files);
        let text = render(
            &t,
            &Slug::parse("john").unwrap(),
            None,
            Channel::ClaudeCode,
            Stamp::from_secs(0),
        );
        assert!(text.starts_with(NOT_INSTRUCTIONS), "{text}");
        assert!(text.ends_with(HOW));
    }
}
