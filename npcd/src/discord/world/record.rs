//! A world's Discord server and what is mirrored into it, as the world keeps
//! them (`docs/npcd_discord.md` §4.2, §4.4).
//!
//! Records under the world, written by its authority with the world's other
//! records each beat, and followed and reloaded with them:
//!
//! - `discord/guild` — the world's server, its name, the listener (the
//!   character whose bot reads it) and whether character-to-character DMs are
//!   mirrored under `#backstage`;
//! - `discord/thread/<thid>` — a thread of the world mirrored into the server:
//!   the Discord channel it is, what kind of conversation it is, how far it has
//!   been posted there (`mirrored_through`, a
//!   [`crate::sim::phone::Message::seq`]) and the last Discord message taken in
//!   from it (`last_seen_discord`);
//! - `discord/webhook/<channel_id>` — the persona webhook of one of those
//!   channels, which a character without a bot of its own speaks through. Its
//!   token is sealed for the record ([`crate::discord::seal::Sealer::seal_for`]):
//!   whoever holds it posts in that channel under any name.
//!
//! Setting another server, or none, forgets every thread and webhook of the
//! last: they are its channels.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::world_state::Records;

/// The world's server's record.
pub const GUILD: &str = "discord/guild";

/// Where the mirrored threads are among a world's records.
pub const THREAD_PREFIX: &str = "discord/thread/";

/// Where the webhooks are among a world's records.
pub const HOOK_PREFIX: &str = "discord/webhook/";

/// Where the channels private threads are made under are among a world's
/// records: `#dms` and `#backstage`, by name.
pub const PARENT_PREFIX: &str = "discord/parent/";

/// A channel private threads are made under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parent {
    pub channel_id: u64,
}

/// A world's Discord server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Guild {
    pub guild_id: u64,
    /// What the server is called, as Discord last said.
    pub name: String,
    /// The character whose bot reads the server's channels.
    pub listener: u64,
    /// Whether DMs between two characters are mirrored for their owners.
    pub backstage: bool,
}

/// What kind of conversation a mirrored thread is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// The world's open channel, `#the-channel`.
    Channel,
    /// A named group: a text channel only its members see.
    Group,
    /// A person and a character without a bot of its own: a private thread
    /// under `#dms`.
    Dm,
    /// Two characters: a private thread under `#backstage`, read by their
    /// owners.
    Backstage,
}

/// One thread of the world, mirrored into a channel of its server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mirrored {
    /// The channel, or the private thread, it is posted in.
    pub channel_id: u64,
    pub kind: Kind,
    /// The seq of the last message of the thread posted to the channel.
    pub mirrored_through: u64,
    /// The id of the last Discord message taken in from the channel.
    pub last_seen_discord: u64,
    /// For a private thread, the channel it is under — whose webhook posts
    /// into it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u64>,
    /// A group nobody remains on: kept read-only, and posted to no more.
    #[serde(default, skip_serializing_if = "is_false")]
    pub archived: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl Mirrored {
    /// A channel or thread of `kind`, posted from after seq `from` and read
    /// from after Discord message `seen`.
    pub fn new(channel_id: u64, kind: Kind, from: u64, seen: u64) -> Self {
        Self {
            channel_id,
            kind,
            mirrored_through: from,
            last_seen_discord: seen,
            parent: None,
            archived: false,
        }
    }

    /// The channel whose webhook posts here: its parent, for a private
    /// thread, and itself otherwise.
    pub fn hook_channel(&self) -> u64 {
        self.parent.unwrap_or(self.channel_id)
    }
}

/// A channel's persona webhook.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hook {
    pub webhook_id: u64,
    /// Its token, sealed for this record.
    pub token: String,
}

/// A world's server and everything mirrored into it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Mirror {
    guild: Option<Guild>,
    threads: BTreeMap<String, Mirrored>,
    hooks: BTreeMap<u64, Hook>,
    parents: BTreeMap<String, Parent>,
}

/// The record name of thread `thread`'s mirror.
pub fn thread_name(thread: &str) -> String {
    format!("{THREAD_PREFIX}{thread}")
}

/// The record name of channel `channel_id`'s webhook — what its token is
/// sealed for.
pub fn hook_name(channel_id: u64) -> String {
    format!("{HOOK_PREFIX}{channel_id}")
}

/// The records under `prefix` that decode as `T`, by what follows the prefix.
fn decoded<T: DeserializeOwned>(records: &Records, prefix: &str) -> BTreeMap<String, T> {
    records
        .iter()
        .filter_map(|(k, v): (&String, &Value)| {
            let at = k.strip_prefix(prefix).filter(|at| !at.is_empty())?;
            match serde_json::from_value::<T>(v.clone()) {
                Ok(record) => Some((at.to_string(), record)),
                Err(e) => {
                    tracing::error!("discord: the record {k} would not decode: {e}");
                    None
                }
            }
        })
        .collect()
}

impl Mirror {
    /// The world's server, when it has one.
    pub fn guild(&self) -> Option<&Guild> {
        self.guild.as_ref()
    }

    /// Set the world's server, or take it away. Another server than the one
    /// set forgets every mirrored thread and webhook: they were the last one's
    /// channels.
    pub fn set_guild(&mut self, guild: Option<Guild>) {
        let same = matches!(
            (&self.guild, &guild),
            (Some(was), Some(now)) if was.guild_id == now.guild_id
        );
        if !same {
            self.threads.clear();
            self.hooks.clear();
            self.parents.clear();
        }
        self.guild = guild;
    }

    /// The channel called `name` that private threads are made under.
    pub fn parent(&self, name: &str) -> Option<u64> {
        self.parents.get(name).map(|p| p.channel_id)
    }

    /// Record channel `channel_id` as the one called `name` that private
    /// threads are made under.
    pub fn put_parent(&mut self, name: &str, channel_id: u64) {
        self.parents.insert(name.to_string(), Parent { channel_id });
    }

    /// Thread `thread`'s mirror.
    pub fn thread(&self, thread: &str) -> Option<Mirrored> {
        self.threads.get(thread).copied()
    }

    /// Write thread `thread`'s mirror.
    pub fn put_thread(&mut self, thread: &str, mirrored: Mirrored) {
        self.threads.insert(thread.to_string(), mirrored);
    }

    /// Every mirrored thread, by thread id.
    pub fn threads(&self) -> Vec<(String, Mirrored)> {
        self.threads
            .iter()
            .map(|(id, m)| (id.clone(), *m))
            .collect()
    }

    /// The thread mirrored into Discord channel `channel_id`.
    pub fn by_channel(&self, channel_id: u64) -> Option<(String, Mirrored)> {
        self.threads
            .iter()
            .find(|(_, m)| m.channel_id == channel_id)
            .map(|(id, m)| (id.clone(), *m))
    }

    /// Channel `channel_id`'s webhook.
    pub fn hook(&self, channel_id: u64) -> Option<Hook> {
        self.hooks.get(&channel_id).cloned()
    }

    /// Write channel `channel_id`'s webhook.
    pub fn put_hook(&mut self, channel_id: u64, hook: Hook) {
        self.hooks.insert(channel_id, hook);
    }

    /// Forget channel `channel_id`'s webhook: Discord no longer has it.
    pub fn remove_hook(&mut self, channel_id: u64) {
        self.hooks.remove(&channel_id);
    }

    /// The server, its threads and its webhooks as world records.
    pub fn records(&self) -> Records {
        let guild = self.guild.iter().map(|g| {
            (
                GUILD.to_string(),
                serde_json::to_value(g).expect("a guild serialises"),
            )
        });
        let threads = self.threads.iter().map(|(id, m)| {
            (
                thread_name(id),
                serde_json::to_value(m).expect("a mirrored thread serialises"),
            )
        });
        let hooks = self.hooks.iter().map(|(id, h)| {
            (
                hook_name(*id),
                serde_json::to_value(h).expect("a webhook serialises"),
            )
        });
        let parents = self.parents.iter().map(|(name, p)| {
            (
                format!("{PARENT_PREFIX}{name}"),
                serde_json::to_value(p).expect("a parent serialises"),
            )
        });
        guild.chain(threads).chain(hooks).chain(parents).collect()
    }

    /// The server, threads and webhooks `records` hold, in place of these:
    /// what a world is hosted, followed and reloaded from.
    pub fn take_in(&mut self, records: &Records) {
        self.guild =
            records
                .get(GUILD)
                .and_then(|v| match serde_json::from_value::<Guild>(v.clone()) {
                    Ok(g) => Some(g),
                    Err(e) => {
                        tracing::error!("discord: the record {GUILD} would not decode: {e}");
                        None
                    }
                });
        self.threads = decoded(records, THREAD_PREFIX);
        self.hooks = decoded::<Hook>(records, HOOK_PREFIX)
            .into_iter()
            .filter_map(|(id, h)| Some((id.parse().ok()?, h)))
            .collect();
        self.parents = decoded(records, PARENT_PREFIX);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn guild(guild_id: u64) -> Guild {
        Guild {
            guild_id,
            name: "Life".into(),
            listener: 42,
            backstage: true,
        }
    }

    fn mirrored(channel_id: u64) -> Mirrored {
        Mirrored::new(channel_id, Kind::Channel, 3, 900)
    }

    /// **A private thread names its parent, and an archived group says so**;
    /// neither field is written when it says nothing.
    #[test]
    fn a_private_threads_parent_and_an_archive_are_written_only_when_set() {
        let mut dm = Mirrored::new(7100, Kind::Dm, 0, 1);
        dm.parent = Some(7001);
        let mut group = Mirrored::new(7200, Kind::Group, 2, 3);
        group.archived = true;
        assert_eq!(
            serde_json::to_value(dm).unwrap(),
            json!({ "channel_id": 7100, "kind": "dm", "mirrored_through": 0,
                    "last_seen_discord": 1, "parent": 7001 })
        );
        assert_eq!(
            serde_json::to_value(group).unwrap(),
            json!({ "channel_id": 7200, "kind": "group", "mirrored_through": 2,
                    "last_seen_discord": 3, "archived": true })
        );
        assert_eq!((dm.hook_channel(), group.hook_channel()), (7001, 7200));
    }

    /// **The server is `discord/guild`, a thread `discord/thread/<thid>` and
    /// a webhook `discord/webhook/<channel_id>`**, in exactly these shapes.
    #[test]
    fn the_server_its_threads_and_webhooks_are_records_of_these_shapes() {
        let mut m = Mirror::default();
        m.set_guild(Some(guild(600)));
        m.put_thread("th1", mirrored(7000));
        m.put_hook(
            7000,
            Hook {
                webhook_id: 107000,
                token: "c2VhbGVk".into(),
            },
        );
        assert_eq!(
            m.records(),
            [
                (
                    "discord/guild".to_string(),
                    json!({ "guild_id": 600, "name": "Life", "listener": 42, "backstage": true })
                ),
                (
                    "discord/thread/th1".to_string(),
                    json!({
                        "channel_id": 7000,
                        "kind": "channel",
                        "mirrored_through": 3,
                        "last_seen_discord": 900,
                    })
                ),
                (
                    "discord/webhook/7000".to_string(),
                    json!({ "webhook_id": 107000, "token": "c2VhbGVk" })
                ),
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(
            m.by_channel(7000),
            Some(("th1".to_string(), mirrored(7000)))
        );
        assert_eq!(m.by_channel(7001), None);
    }

    /// Records taken in are the mirror again, and nothing else among a
    /// world's records is taken for part of it.
    #[test]
    fn records_taken_in_are_the_mirror_again() {
        let mut m = Mirror::default();
        m.set_guild(Some(guild(600)));
        m.put_thread("th1", mirrored(7000));
        m.put_hook(
            7000,
            Hook {
                webhook_id: 1,
                token: "t".into(),
            },
        );
        let mut records = m.records();
        records.insert("discord/channel/42/610".into(), json!({}));
        records.insert(
            "discord/webhook/notanumber".into(),
            json!({"webhook_id": 2, "token": "u"}),
        );
        let mut back = Mirror::default();
        back.put_thread("stale", mirrored(1));
        back.take_in(&records);
        assert_eq!(back, m);

        m.put_parent("dms", 7001);
        assert_eq!(
            m.records().get("discord/parent/dms"),
            Some(&json!({ "channel_id": 7001 }))
        );
        back.take_in(&m.records());
        assert_eq!(back.parent("dms"), Some(7001));
    }

    /// **Another server forgets the last one's channels**, and the same one
    /// set again — a rename, a new listener — keeps them.
    #[test]
    fn another_server_forgets_the_last_ones_channels() {
        let mut m = Mirror::default();
        m.set_guild(Some(guild(600)));
        m.put_thread("th1", mirrored(7000));
        m.put_hook(
            7000,
            Hook {
                webhook_id: 1,
                token: "t".into(),
            },
        );
        let mut renamed = guild(600);
        renamed.name = "Moon".into();
        renamed.listener = 43;
        m.set_guild(Some(renamed.clone()));
        assert_eq!(m.guild(), Some(&renamed));
        assert_eq!(m.threads().len(), 1);
        assert!(m.hook(7000).is_some());

        m.set_guild(Some(guild(601)));
        assert!(m.threads().is_empty());
        assert!(m.hook(7000).is_none());
        m.put_thread("th1", mirrored(8000));
        m.set_guild(None);
        assert_eq!(m, Mirror::default());
        assert!(m.records().is_empty());
    }
}
