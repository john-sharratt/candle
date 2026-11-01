//! `WorldOp::Mirrored`, made on the world's authority (`docs/npcd_discord.md`
//! §5.2): a message written in a channel of the world's server, said on the
//! thread it mirrors.
//!
//! Its author is brought onto the thread at its end if this is their first
//! message there, as an invitation brings somebody — a person writing in a
//! channel a person made, which the whole cast is on and nobody joined by
//! hand; on a group the inbound side has already refused anyone not on it.
//! The message carries its Discord origin, so the mirror never posts it back,
//! and the character it was said to, if any, so that character is told it
//! urgently. A message whose origin the thread already holds is not said
//! twice.

use crate::sim::phone::Origin;
use crate::world::Hosted;

/// Say `text`, from `from`, on mirrored thread `thread` of `hosted`, to
/// member `to` when it was said to one: whether the thread holds it now.
pub fn mirrored_message(
    hosted: &Hosted,
    thread: &str,
    from: &str,
    text: &str,
    origin: Origin,
    to: Option<&str>,
) -> bool {
    hosted.with_sim(|s| {
        let Some(t) = s.threads.get(thread).filter(|t| t.open) else {
            return false;
        };
        if t.messages.iter().any(|m| m.origin == Some(origin)) {
            return true;
        }
        s.threads.join(thread, from);
        s.threads
            .send_to(thread, from, text, Some(origin), to)
            .is_ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");

    /// **The author joins at the thread's end and is heard on it, once, said
    /// to the character it names**; a closed thread takes nothing.
    #[test]
    fn a_mirrored_message_joins_its_author_and_is_said_once() {
        let hosted = Hosted::load("creators-vault", ROOMS).unwrap();
        let thread = hosted.with_sim(|s| {
            let id = s
                .threads
                .open_standing("#research", &["Zen".to_string(), "Wick".to_string()]);
            s.threads.send_on(&id, "Wick", "earlier", None).unwrap();
            id
        });
        let origin = Origin::Discord {
            message: 901,
            channel: 7300,
        };
        assert!(mirrored_message(
            &hosted,
            &thread,
            "John",
            "Zen, look",
            origin,
            Some("Zen")
        ));
        assert!(mirrored_message(
            &hosted,
            &thread,
            "John",
            "Zen, look",
            origin,
            Some("Zen")
        ));
        let (members, said, waiting) = hosted.sim(|s| {
            let t = s.threads.get(&thread).unwrap();
            (
                t.members.clone(),
                t.messages
                    .iter()
                    .map(|m| (m.seq, m.from.clone(), m.origin, m.to.clone()))
                    .collect::<Vec<_>>(),
                t.unread_for("John"),
            )
        });
        assert_eq!(members, ["Zen", "Wick", "John"]);
        assert_eq!(
            said,
            [
                (1, "Wick".to_string(), None, None),
                (2, "John".to_string(), Some(origin), Some("Zen".to_string())),
            ]
        );
        assert_eq!(waiting, 0, "a newcomer was handed the backlog");

        hosted.with_sim(|s| s.threads.close(&thread));
        let later = Origin::Discord {
            message: 902,
            channel: 7300,
        };
        assert!(!mirrored_message(
            &hosted, &thread, "John", "x", later, None
        ));
    }
}
