//! The channels people made in the world's server, as channels of the world
//! (`docs/npcd_discord.md` §3, §5.3).
//!
//! The server is the world's, and what its people make in it is the world's
//! too: every text channel the mirror did not make itself — not
//! `#the-channel`, a group's, or the channel private threads are made under —
//! is a **room**, a standing channel of the world named `#<channel>`, with
//! the whole cast on it, as the cast is on the world's own channel. Its
//! description is the one Discord gives the channel, and it is what tells a
//! character what the room is for: each character reads its channels with
//! their descriptions in its prompt, and says something on the one that fits.
//!
//! Read on every reconcile, and on the beat after Discord says a channel of
//! the server was made, changed or deleted — one read of the server's
//! channels, on the listener's token:
//!
//! - a text channel nobody has recorded is opened as a room, posted from its
//!   end and read from now, so neither side's history crosses;
//! - every mirrored thread is described as its channel now is, and a room is
//!   named as its channel now is and has the cast on it — a character who
//!   came into the world since is brought on at its end;
//! - a room whose channel is gone is closed and forgotten: nobody is offered
//!   it, and what was said there is kept, as for a closed group;
//! - a channel private threads were made under that is gone is forgotten, so
//!   the next private thread makes it again.

use std::collections::{BTreeMap, BTreeSet};

use super::channels::{channel_thread, last_seq};
use super::record::{Guild, Kind, Mirrored};
use crate::api::{now_ms, Authored};
use crate::discord::client::GuildChannel;
use crate::discord::message::snowflake_at;
use crate::discord::online::Online;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};
use crate::world::Hosted;

/// What a room is called in the world: its channel's name, as Discord shows
/// it.
pub fn room_name(channel: &str) -> String {
    format!("#{channel}")
}

/// Who a room has on it: everybody on the world's own channel — the cast,
/// and the people enrolled in the world.
fn cast(hosted: &Hosted) -> Vec<String> {
    let Some(thread) = channel_thread(hosted) else {
        return Vec::new();
    };
    hosted.sim(|s| {
        s.threads
            .get(&thread)
            .map(|t| t.members.clone())
            .unwrap_or_default()
    })
}

/// Read `guild`'s channels as `bot`, its listener, and keep `hosted`'s rooms,
/// descriptions and parents as they stand.
pub async fn reconcile(a: &Authored, bot: &Online, hosted: &Hosted, guild: &Guild) {
    let channels = match a.discord.guild_channels(&bot.token, guild.guild_id).await {
        Ok(channels) => channels,
        Err(e) => {
            failed(
                a,
                bot.npc,
                Note::new(
                    now_ms(),
                    Direction::Out,
                    "mirror",
                    &format!("channels not read: {e}"),
                )
                .guild(guild.guild_id),
            );
            return;
        }
    };
    let standing: BTreeMap<u64, &GuildChannel> = channels
        .iter()
        .filter(|c| c.is_text())
        .map(|c| (c.id, c))
        .collect();
    let cast = cast(hosted);
    for (opened, closed) in keep(hosted, &standing, &cast) {
        let outcome = match closed {
            false => "room_opened",
            true => "room_closed",
        };
        note(
            a,
            bot.npc,
            Note::new(now_ms(), Direction::Out, "mirror", outcome)
                .guild(guild.guild_id)
                .channel(opened.1)
                .thread(&opened.0),
        );
    }
}

/// Keep `hosted`'s mirror as the server's text channels `standing` are, with
/// `cast` on every room: each `((thread, channel), closed)` a room opened, or
/// one closed.
fn keep(
    hosted: &Hosted,
    standing: &BTreeMap<u64, &GuildChannel>,
    cast: &[String],
) -> Vec<((String, u64), bool)> {
    let mut changed = Vec::new();
    for (name, channel_id) in hosted.mirror(|m| m.parents()) {
        if !standing.contains_key(&channel_id) {
            hosted.with_mirror(|m| m.remove_parent(&name));
        }
    }
    for (thread, mirrored) in hosted.mirror(|m| m.threads()) {
        match (standing.get(&mirrored.channel_id), mirrored.kind) {
            (Some(c), kind) => hosted.with_sim(|s| {
                s.threads.describe(&thread, c.topic.as_deref());
                if kind == Kind::Room {
                    s.threads.rename(&thread, &room_name(&c.name));
                    for who in cast {
                        s.threads.join(&thread, who);
                    }
                }
            }),
            (None, Kind::Room) => {
                hosted.with_sim(|s| s.threads.close(&thread));
                hosted.with_mirror(|m| m.remove_thread(&thread));
                changed.push(((thread, mirrored.channel_id), true));
            }
            // A private thread is no channel of the server's, and a group's
            // channel deleted by hand is the group's to answer for.
            (None, _) => {}
        }
    }
    let parents: BTreeSet<u64> = hosted
        .mirror(|m| m.parents())
        .into_iter()
        .map(|(_, id)| id)
        .collect();
    for c in standing.values() {
        if parents.contains(&c.id) || hosted.mirror(|m| m.by_channel(c.id)).is_some() {
            continue;
        }
        let thread = hosted.with_sim(|s| {
            let id = s.threads.open_standing(&room_name(&c.name), cast);
            s.threads.describe(&id, c.topic.as_deref());
            id
        });
        let mirrored = Mirrored::new(
            c.id,
            Kind::Room,
            last_seq(hosted, &thread),
            snowflake_at(now_ms()),
        );
        hosted.with_mirror(|m| m.put_thread(&thread, mirrored));
        changed.push(((thread, c.id), false));
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::phone::CHANNEL;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");

    fn channel(id: u64, name: &str, topic: Option<&str>) -> GuildChannel {
        GuildChannel {
            id,
            kind: 0,
            name: name.into(),
            topic: topic.map(str::to_string),
        }
    }

    /// A world with Zen and Wick on its channel, `#the-channel` recorded as
    /// channel 7000 and `#dms` as 7001.
    fn world() -> Hosted {
        let hosted = Hosted::load("creators-vault", ROOMS).unwrap();
        let open = hosted.with_sim(|s| {
            s.threads.join_channel(CHANNEL, "Zen");
            s.threads.join_channel(CHANNEL, "Wick")
        });
        hosted.with_mirror(|m| {
            m.put_thread(&open, Mirrored::new(7000, Kind::Channel, 0, 1));
            m.put_parent("dms", 7001);
        });
        hosted
    }

    /// A room as a test reads it: its thread, name, description, members and
    /// whether it is open.
    type Room = (String, String, Option<String>, Vec<String>, bool);

    fn room_of(hosted: &Hosted, channel_id: u64) -> Option<Room> {
        let (thread, mirrored) = hosted.mirror(|m| m.by_channel(channel_id))?;
        assert_eq!(mirrored.kind, Kind::Room);
        hosted.sim(|s| {
            let t = s.threads.get(&thread)?;
            Some((
                thread.clone(),
                t.name.clone(),
                t.topic.clone(),
                t.members.clone(),
                t.open,
            ))
        })
    }

    /// **A channel a person made is a room of the world**, named for it,
    /// described as it is, with the cast on it; the mirror's own channels are
    /// not, and the world's channel takes its description.
    #[test]
    fn a_channel_a_person_made_is_a_room_with_the_cast_on_it() {
        let hosted = world();
        let made = [
            channel(7000, "the-channel", Some("Anything for everyone")),
            channel(7001, "dms", None),
            channel(7300, "research", Some("Findings, sources, open questions")),
            channel(7301, "ops", None),
        ];
        let standing: BTreeMap<u64, &GuildChannel> = made.iter().map(|c| (c.id, c)).collect();
        let changed = keep(&hosted, &standing, &cast(&hosted));
        assert_eq!(changed.len(), 2, "{changed:?}");
        let (_, name, topic, members, open) = room_of(&hosted, 7300).unwrap();
        assert_eq!(name, "#research");
        assert_eq!(topic.as_deref(), Some("Findings, sources, open questions"));
        assert_eq!(members, ["Zen", "Wick"]);
        assert!(open);
        assert_eq!(room_of(&hosted, 7301).unwrap().2, None);
        assert!(hosted.mirror(|m| m.by_channel(7001)).is_none());
        let open = channel_thread(&hosted).unwrap();
        assert_eq!(
            hosted
                .sim(|s| s.threads.get(&open).unwrap().topic.clone())
                .as_deref(),
            Some("Anything for everyone")
        );

        // Read again: nothing new, and nothing opened twice.
        assert!(keep(&hosted, &standing, &cast(&hosted)).is_empty());
        assert_eq!(hosted.mirror(|m| m.threads()).len(), 3);
    }

    /// **A room follows its channel**: renamed and described again with it,
    /// a newcomer to the world brought on at its end, and closed and
    /// forgotten when the channel goes — as is a parent whose channel went.
    #[test]
    fn a_room_follows_its_channel_and_closes_with_it() {
        let hosted = world();
        let first = [
            channel(7300, "research", Some("Findings")),
            channel(7001, "dms", None),
        ];
        let standing: BTreeMap<u64, &GuildChannel> = first.iter().map(|c| (c.id, c)).collect();
        keep(&hosted, &standing, &cast(&hosted));
        let (thread, ..) = room_of(&hosted, 7300).unwrap();

        hosted.with_sim(|s| s.threads.join_channel(CHANNEL, "Jitter"));
        let renamed = [channel(7300, "lab", Some("  ")), channel(7001, "dms", None)];
        let standing: BTreeMap<u64, &GuildChannel> = renamed.iter().map(|c| (c.id, c)).collect();
        assert!(keep(&hosted, &standing, &cast(&hosted)).is_empty());
        let (_, name, topic, members, _) = room_of(&hosted, 7300).unwrap();
        assert_eq!((name.as_str(), topic), ("#lab", None));
        assert_eq!(members, ["Zen", "Wick", "Jitter"]);

        let gone: BTreeMap<u64, &GuildChannel> = BTreeMap::new();
        assert_eq!(
            keep(&hosted, &gone, &cast(&hosted)),
            [((thread.clone(), 7300), true)]
        );
        assert!(hosted.mirror(|m| m.by_channel(7300)).is_none());
        assert!(!hosted.sim(|s| s.threads.get(&thread).unwrap().open));
        assert!(hosted.mirror(|m| m.parent("dms")).is_none());
    }
}
