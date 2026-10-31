//! A bot's servers and their channels, as the world keeps them
//! (`docs/npcd_discord.md` §4.2).
//!
//! Two records under the world, written by its authority with the world's
//! other records each beat, and followed and reloaded with them:
//!
//! - `discord/channel/<npc_id>/<channel_id>` — a text channel of a server
//!   the character's bot is in: its server, the group thread that mirrors it,
//!   how far the thread has been posted there (`mirrored_through`, a
//!   [`crate::sim::phone::Message::seq`]) and the last Discord message taken
//!   in from it (`last_seen_discord`);
//! - `discord/server/<npc_id>/<guild_id>` — a server the bot is in, by name,
//!   so a bot that comes online can tell its character of the servers it
//!   joined and left while it was offline.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::world_state::Records;

/// Where the channel records are among a world's records.
pub const CHANNEL_PREFIX: &str = "discord/channel/";

/// Where the server records are among a world's records.
pub const SERVER_PREFIX: &str = "discord/server/";

/// A server channel a character's bot is in, and its thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Channel {
    /// The server it is in.
    pub guild_id: u64,
    /// The id of the group thread that mirrors it.
    pub thread: String,
    /// The seq of the last message of the thread posted to the channel.
    pub mirrored_through: u64,
    /// The id of the last Discord message taken in from the channel.
    pub last_seen_discord: u64,
}

/// A server a character's bot is in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Server {
    /// What the server is called, as it was last heard.
    pub name: String,
}

/// Every server and channel of a world's bots, by character.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Servers {
    channels: BTreeMap<(u64, u64), Channel>,
    servers: BTreeMap<(u64, u64), Server>,
}

/// The record name of character `npc_id`'s channel `channel_id`.
pub fn channel_name(npc_id: u64, channel_id: u64) -> String {
    format!("{CHANNEL_PREFIX}{npc_id}/{channel_id}")
}

/// The record name of character `npc_id`'s server `guild_id`.
pub fn server_name(npc_id: u64, guild_id: u64) -> String {
    format!("{SERVER_PREFIX}{npc_id}/{guild_id}")
}

/// The character and Discord id a record name under `prefix` names.
fn parse(name: &str, prefix: &str) -> Option<(u64, u64)> {
    let (npc, id) = name.strip_prefix(prefix)?.split_once('/')?;
    Some((npc.parse().ok()?, id.parse().ok()?))
}

/// The records under `prefix` that decode as `T`, by character and id.
fn decoded<T: DeserializeOwned>(records: &Records, prefix: &str) -> BTreeMap<(u64, u64), T> {
    records
        .iter()
        .filter_map(|(k, v): (&String, &Value)| {
            let at = parse(k, prefix)?;
            match serde_json::from_value::<T>(v.clone()) {
                Ok(record) => Some((at, record)),
                Err(e) => {
                    tracing::error!("discord: the record {k} would not decode: {e}");
                    None
                }
            }
        })
        .collect()
}

impl Servers {
    /// Character `npc_id`'s channel `channel_id`.
    pub fn channel(&self, npc_id: u64, channel_id: u64) -> Option<Channel> {
        self.channels.get(&(npc_id, channel_id)).cloned()
    }

    /// Write character `npc_id`'s channel `channel_id`.
    pub fn put_channel(&mut self, npc_id: u64, channel_id: u64, channel: Channel) {
        self.channels.insert((npc_id, channel_id), channel);
    }

    /// Forget character `npc_id`'s channel `channel_id`: it is gone.
    pub fn remove_channel(&mut self, npc_id: u64, channel_id: u64) -> Option<Channel> {
        self.channels.remove(&(npc_id, channel_id))
    }

    /// Every channel of character `npc_id`'s bot, by channel id.
    pub fn channels_of(&self, npc_id: u64) -> Vec<(u64, Channel)> {
        self.channels
            .range((npc_id, 0)..=(npc_id, u64::MAX))
            .map(|((_, id), c)| (*id, c.clone()))
            .collect()
    }

    /// Character `npc_id`'s channels in server `guild_id`, by channel id.
    pub fn channels_in(&self, npc_id: u64, guild_id: u64) -> Vec<(u64, Channel)> {
        self.channels_of(npc_id)
            .into_iter()
            .filter(|(_, c)| c.guild_id == guild_id)
            .collect()
    }

    /// Character `npc_id`'s server `guild_id`.
    pub fn server(&self, npc_id: u64, guild_id: u64) -> Option<Server> {
        self.servers.get(&(npc_id, guild_id)).cloned()
    }

    /// Write character `npc_id`'s server `guild_id`.
    pub fn put_server(&mut self, npc_id: u64, guild_id: u64, server: Server) {
        self.servers.insert((npc_id, guild_id), server);
    }

    /// Forget character `npc_id`'s server `guild_id`: the bot left it.
    pub fn remove_server(&mut self, npc_id: u64, guild_id: u64) -> Option<Server> {
        self.servers.remove(&(npc_id, guild_id))
    }

    /// Every server character `npc_id`'s bot is in, by id.
    pub fn servers_of(&self, npc_id: u64) -> Vec<(u64, Server)> {
        self.servers
            .range((npc_id, 0)..=(npc_id, u64::MAX))
            .map(|((_, id), s)| (*id, s.clone()))
            .collect()
    }

    /// The servers and channels as world records.
    pub fn records(&self) -> Records {
        let channels = self.channels.iter().map(|((npc, id), c)| {
            (
                channel_name(*npc, *id),
                serde_json::to_value(c).expect("a channel serialises"),
            )
        });
        let servers = self.servers.iter().map(|((npc, id), s)| {
            (
                server_name(*npc, *id),
                serde_json::to_value(s).expect("a server serialises"),
            )
        });
        channels.chain(servers).collect()
    }

    /// The servers and channels `records` hold, in place of these: what a
    /// world is hosted, followed and reloaded from.
    pub fn take_in(&mut self, records: &Records) {
        self.channels = decoded(records, CHANNEL_PREFIX);
        self.servers = decoded(records, SERVER_PREFIX);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn channel(guild_id: u64, thread: &str) -> Channel {
        Channel {
            guild_id,
            thread: thread.into(),
            mirrored_through: 3,
            last_seen_discord: 900,
        }
    }

    /// **A channel is `discord/channel/<npc_id>/<channel_id>` and a server
    /// `discord/server/<npc_id>/<guild_id>`**, in exactly these shapes.
    #[test]
    fn channels_and_servers_are_world_records_of_these_shapes() {
        let mut s = Servers::default();
        s.put_channel(42, 610, channel(600, "th7"));
        s.put_server(
            42,
            600,
            Server {
                name: "zenworld".into(),
            },
        );
        assert_eq!(
            s.records(),
            [
                (
                    "discord/channel/42/610".to_string(),
                    json!({
                        "guild_id": 600,
                        "thread": "th7",
                        "mirrored_through": 3,
                        "last_seen_discord": 900,
                    })
                ),
                (
                    "discord/server/42/600".to_string(),
                    json!({ "name": "zenworld" })
                ),
            ]
            .into_iter()
            .collect()
        );
    }

    /// Records taken in are the servers and channels again, by character
    /// and by server, and nothing else among a world's records is taken for
    /// one.
    #[test]
    fn records_taken_in_are_the_servers_and_channels_again() {
        let mut s = Servers::default();
        s.put_channel(42, 610, channel(600, "th7"));
        s.put_channel(42, 611, channel(600, "th8"));
        s.put_channel(42, 710, channel(700, "th9"));
        s.put_channel(43, 610, channel(600, "th10"));
        s.put_server(42, 600, Server { name: "a".into() });
        s.put_server(43, 600, Server { name: "a".into() });
        let mut records = s.records();
        records.insert("discord/dm/42/0a1b".into(), json!({}));
        records.insert("discord/channel/42/x".into(), json!({}));
        let mut back = Servers::default();
        back.put_server(
            1,
            1,
            Server {
                name: "stale".into(),
            },
        );
        back.take_in(&records);
        assert_eq!(back, s);
        assert_eq!(
            back.channels_in(42, 600),
            [(610, channel(600, "th7")), (611, channel(600, "th8"))]
        );
        assert_eq!(back.channels_of(43), [(610, channel(600, "th10"))]);
        assert_eq!(back.servers_of(42), [(600, Server { name: "a".into() })]);
        assert_eq!(back.remove_channel(42, 611), Some(channel(600, "th8")));
        assert_eq!(back.channel(42, 611), None);
        assert_eq!(
            back.remove_server(42, 600),
            Some(Server { name: "a".into() })
        );
        assert_eq!(back.server(42, 600), None);
    }
}
