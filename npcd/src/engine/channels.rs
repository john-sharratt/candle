//! The channels a character is on, as its system prompt carries them
//! (`docs/npcd_discord.md` §3.5).
//!
//! One section per channel — the world's own channel, each Discord channel
//! that stands as one — submitted to the character's own conversation under
//! `channels/<thread id>` and selected under `channels_intro`, as a journal's
//! entries are. Each says what the channel is called, which is the name
//! `message` takes for it, and what it is for, which is its description: the
//! character reads the list to decide where to say a thing. A channel whose
//! name or description changes reseals its section alone; one the character
//! is no longer on leaves the prompt with its section.

use crate::sim::phone::Threads;

/// Where a channel's section sits in the `channels` collection.
const SECTION_PREFIX: &str = "channels/";

/// One channel, as the prompt carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelSection {
    /// `channels/<thread id>`: one per channel, kept by the thread's id so a
    /// rename reseals it rather than carrying two.
    pub name: String,
    pub text: String,
}

/// The channels `me` is on, each as its section; none for a character with
/// no handset, which can send on none of them.
pub fn sections(threads: &Threads, me: &str, has_phone: bool) -> Vec<ChannelSection> {
    if !has_phone {
        return Vec::new();
    }
    threads
        .channels_for(me)
        .into_iter()
        .map(|t| ChannelSection {
            name: format!("{SECTION_PREFIX}{}", t.id),
            text: line(&t.name, t.topic.as_deref()),
        })
        .collect()
}

/// A channel's line: its name, and what it is for when it says.
fn line(name: &str, topic: Option<&str>) -> String {
    match topic {
        Some(topic) => format!("- {name}: {topic}\n"),
        None => format!("- {name}\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::phone::CHANNEL;

    /// **Each channel the character is on is one section**, its line its
    /// name and what it is for; a group, a direct thread, a channel it is not
    /// on and a closed one are not channels it reads, and without a handset
    /// it reads none.
    #[test]
    fn each_channel_is_a_section_with_what_it_is_for() {
        let mut p = Threads::new();
        let open = p.join_channel(CHANNEL, "Zen");
        let research = p.open_standing("#research", &["Zen".to_string()]);
        p.describe(&research, Some("Findings and open questions"));
        let elsewhere = p.open_standing("#ops", &["Wick".to_string()]);
        let gone = p.open_standing("#old", &["Zen".to_string()]);
        p.close(&gone);
        p.open_group("the sweep", &["Zen".to_string(), "Wick".to_string()]);
        p.reach("Zen", "Wick");

        let mut got = sections(&p, "Zen", true);
        got.sort_by(|a, b| a.name.cmp(&b.name));
        let mut want = vec![
            ChannelSection {
                name: format!("channels/{open}"),
                text: "- the channel\n".into(),
            },
            ChannelSection {
                name: format!("channels/{research}"),
                text: "- #research: Findings and open questions\n".into(),
            },
        ];
        want.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(got, want);
        assert!(!got.iter().any(|s| s.name.ends_with(&elsewhere)));
        assert_eq!(sections(&p, "Zen", false), []);
    }
}
