//! `get_user_context` (§6.4–§6.6): the frame for this conversation — the
//! group's master, who it could be talking to, the conversation and its
//! focus, the journals the focus names, and the cues below — within the
//! channel's budget, and how to work with the journals.

use journals::path::JournalPath;
use journals::scope::Channel;
use journals::slug::Slug;
use journals::time::Stamp;
use journals::tree::Tree;
use journals::view::frame::{frame, Budget};

/// What a channel is given: a voice channel its spoken frame, a chat channel
/// its frame.
pub fn budget(channel: Channel) -> Budget {
    match channel {
        Channel::Grok => Budget::Voice,
        Channel::ClaudeCode | Channel::Other => Budget::Chat,
    }
}

/// How to work with the journals, after a chat frame. A voice frame opens
/// with its own, spoken.
pub const HOW: &str = "Work out who is speaking — never assume the account's owner — and \
what this conversation is about, walking the cues above; say both with journal_focus as each \
becomes clear. Keep what you learn with journal_note on the journal it is about: a person's on \
people/<name>, a subject's on its topic, this conversation's open tasks on `here`, and only what \
lasts on the master. journal_create adds a topic and journal_link a way to it; restructuring the \
tree is a coding task for Zen on the journals repository.";

/// What a guest is told after its frame.
pub const GUEST_HOW: &str = "This account is mapped to no one here, so it reads only @guest and \
nothing is kept for it: an estate admin maps it to a person with group_add.";

pub fn render(
    tree: &Tree,
    group: &Slug,
    conversation: Option<&JournalPath>,
    channel: Channel,
    guest: bool,
    now: Stamp,
) -> String {
    let budget = budget(channel);
    let mut out = frame(tree, group, conversation, &[], budget, now);
    let how = match (guest, budget) {
        (true, _) => GUEST_HOW,
        (false, Budget::Voice) => return out,
        (false, _) => HOW,
    };
    out.push('\n');
    out.push_str(how);
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use journals::tree::load::load;
    use journals::view::frame::VOICE_SPINE;
    use journals::view::NOT_INSTRUCTIONS;

    use super::*;

    fn tree() -> Tree {
        load(&BTreeMap::from([
            (
                "persons/john/person.yaml".to_string(),
                b"person: john\nname: \"John\"\naccounts: []\ngroups: [zen]\ndefault_group: zen\n".to_vec(),
            ),
            (
                "groups/zen/group.yaml".to_string(),
                b"group: zen\nname: \"Zen\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n".to_vec(),
            ),
        ]))
    }

    #[test]
    fn a_voice_channel_gets_its_spoken_frame_and_a_chat_channel_the_frame_and_how() {
        assert_eq!(budget(Channel::Grok), Budget::Voice);
        assert_eq!(budget(Channel::ClaudeCode), Budget::Chat);
        let t = tree();
        let zen = Slug::parse("zen").unwrap();
        let text = render(
            &t,
            &zen,
            None,
            Channel::ClaudeCode,
            false,
            Stamp::from_secs(0),
        );
        assert!(text.starts_with(NOT_INSTRUCTIONS), "{text}");
        assert!(text.ends_with(HOW));
        let spoken = render(&t, &zen, None, Channel::Grok, false, Stamp::from_secs(0));
        assert!(
            spoken.contains(VOICE_SPINE) && !spoken.contains(HOW),
            "{spoken}"
        );
        let guest = render(&t, &zen, None, Channel::Other, true, Stamp::from_secs(0));
        assert!(guest.ends_with(GUEST_HOW));
    }
}
