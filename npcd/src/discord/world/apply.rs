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

use crate::sim::phone::{Origin, Said, Who};
use crate::world::Hosted;

/// Say `text`, from `from`, on mirrored thread `thread` of `hosted`, to the
/// member `to` (by id) when it was said to one: whether the thread holds it
/// now.
pub fn mirrored_message(
    hosted: &Hosted,
    thread: &str,
    from: &Who,
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
        let said = Said {
            text: text.to_string(),
            origin: Some(origin),
            to: to.map(str::to_string),
            account: Some(from.id.clone()),
            ..Said::default()
        };
        s.threads.send_on(thread, &from.id, said).is_ok()
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
        let zen = Who::new("npc-1", "Zen");
        let wick = Who::new("npc-2", "Wick");
        let john = Who::new("u_john", "John");
        let thread = hosted.with_sim(|s| {
            let id = s.threads.open_standing("#research", &[zen, wick]);
            s.threads
                .send_on(&id, "npc-2", Said::words("earlier"))
                .unwrap();
            id
        });
        let origin = Origin::Discord {
            message: 901,
            channel: 7300,
        };
        for _ in 0..2 {
            assert!(mirrored_message(
                &hosted,
                &thread,
                &john,
                "Zen, look",
                origin,
                Some("npc-1")
            ));
        }
        let (members, said, waiting) = hosted.sim(|s| {
            let t = s.threads.get(&thread).unwrap();
            (
                t.members.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
                t.messages
                    .iter()
                    .map(|m| {
                        (
                            m.seq,
                            m.from.clone(),
                            m.origin,
                            m.to.clone(),
                            m.account.clone(),
                        )
                    })
                    .collect::<Vec<_>>(),
                t.unread_for("u_john"),
            )
        });
        assert_eq!(members, ["npc-1", "npc-2", "u_john"]);
        assert_eq!(
            said,
            [
                (1, "npc-2".to_string(), None, None, None),
                (
                    2,
                    "u_john".to_string(),
                    Some(origin),
                    Some("npc-1".to_string()),
                    Some("u_john".to_string())
                ),
            ]
        );
        assert_eq!(waiting, 0, "a newcomer was handed the backlog");

        hosted.with_sim(|s| s.threads.close(&thread));
        let later = Origin::Discord {
            message: 902,
            channel: 7300,
        };
        assert!(!mirrored_message(&hosted, &thread, &john, "x", later, None));
    }
}
