//! `WorldOp::ChannelMessage`, made on the world's authority
//! (`docs/npcd_discord.md` §5.2): a message written in a server channel,
//! said into the thread that mirrors it for the character.
//!
//! Its author is brought onto the thread at its end if this is their first
//! message there, as an invitation brings somebody, and the message carries
//! its Discord origin so the mirror never posts it back, and the character it
//! was said to, if any, so that character is told it urgently. A message
//! whose origin the thread already holds is not said twice.

use super::record::Channel;
use crate::sim::phone::Origin;
use crate::world::Hosted;

/// Say `text`, from `from`, on the thread of character `npc`'s server
/// channel `channel`, to member `to` when it was said to one: whether the
/// thread holds it now.
pub fn channel_message(
    hosted: &Hosted,
    npc: u64,
    channel: u64,
    from: &str,
    text: &str,
    origin: Option<Origin>,
    to: Option<&str>,
) -> bool {
    let Some(Channel { thread, .. }) = hosted.servers(|s| s.channel(npc, channel)) else {
        return false;
    };
    hosted.with_sim(|s| {
        let Some(t) = s.threads.get(&thread).filter(|t| t.open) else {
            return false;
        };
        if origin.is_some() && t.messages.iter().any(|m| m.origin == origin) {
            return true;
        }
        s.threads.join(&thread, from);
        s.threads.send_to(&thread, from, text, origin, to).is_ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");

    /// **The author joins at the thread's end and is heard on it, once**;
    /// a channel with no record, or a closed thread, takes nothing.
    #[test]
    fn a_channel_message_joins_its_author_and_is_said_once() {
        let hosted = Hosted::load("creators-vault", ROOMS).unwrap();
        let thread = hosted.with_sim(|s| {
            let id = s
                .threads
                .open_standing("zenworld #general", &["Zen".to_string()]);
            s.threads.send_on(&id, "Zen", "earlier", None).unwrap();
            id
        });
        hosted.with_servers(|s| {
            s.put_channel(
                42,
                610,
                Channel {
                    guild_id: 600,
                    thread: thread.clone(),
                    mirrored_through: 1,
                    last_seen_discord: 900,
                },
            )
        });
        let origin = Some(Origin::Discord {
            message: 901,
            channel: 610,
        });
        let said = |text: &str, origin: Option<Origin>, to: Option<&str>| {
            channel_message(&hosted, 42, 610, "Wren (@wren)", text, origin, to)
        };
        assert!(said("hi", origin, None));
        assert!(said("hi", origin, None));
        assert!(said("Zen, you there?", None, Some("Zen")));
        let (members, said, waiting) = hosted.sim(|s| {
            let t = s.threads.get(&thread).unwrap();
            (
                t.members.clone(),
                t.messages
                    .iter()
                    .map(|m| {
                        (
                            m.seq,
                            m.from.clone(),
                            m.intent.clone(),
                            m.origin,
                            m.to.clone(),
                        )
                    })
                    .collect::<Vec<_>>(),
                t.unread_for("Wren (@wren)"),
            )
        });
        assert_eq!(members, ["Zen", "Wren (@wren)"]);
        assert_eq!(
            said,
            [
                (1, "Zen".to_string(), "earlier".to_string(), None, None),
                (
                    2,
                    "Wren (@wren)".to_string(),
                    "hi".to_string(),
                    origin,
                    None
                ),
                (
                    3,
                    "Wren (@wren)".to_string(),
                    "Zen, you there?".to_string(),
                    None,
                    Some("Zen".to_string())
                ),
            ]
        );
        assert_eq!(waiting, 0, "a newcomer was handed the backlog");

        assert!(!channel_message(
            &hosted,
            42,
            611,
            "Wren (@wren)",
            "x",
            None,
            None
        ));
        hosted.with_sim(|s| s.threads.close(&thread));
        assert!(!channel_message(
            &hosted,
            42,
            610,
            "Wren (@wren)",
            "x",
            None,
            None
        ));
    }
}
