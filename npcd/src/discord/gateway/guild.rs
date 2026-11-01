//! A server and its channels, as the gateway's `GUILD_CREATE`,
//! `GUILD_UPDATE` and `CHANNEL_` events carry them — only the fields a bot
//! in a server's channels reads (`docs/npcd_discord.md` §3.4).

use serde::Deserialize;

use crate::discord::message::{maybe_snowflake, snowflake};

/// The channel types a bot talks in: a text channel (`0`) and an
/// announcement channel (`5`). Voice, categories, threads and forums are not
/// mirrored.
pub const TEXT_TYPES: [u8; 2] = [0, 5];

/// A server, as `GUILD_CREATE` hands it over.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Guild {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub channels: Vec<GuildChannel>,
    /// A server Discord cannot serve right now: an outage, not a leave.
    #[serde(default)]
    pub unavailable: bool,
}

/// One of a server's channels.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GuildChannel {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    #[serde(rename = "type")]
    pub kind: u8,
    #[serde(default)]
    pub name: String,
    /// What the channel is for, as its server describes it: `null` or absent
    /// for one with no description.
    #[serde(default)]
    pub topic: Option<String>,
    /// The server it is in: present on a `CHANNEL_` event, absent inside a
    /// `GUILD_CREATE`, which is the server's own.
    #[serde(default, deserialize_with = "maybe_snowflake")]
    pub guild_id: Option<u64>,
}

impl GuildChannel {
    /// Whether it is a channel people write in: one a bot talks in.
    pub fn is_text(&self) -> bool {
        TEXT_TYPES.contains(&self.kind)
    }
}

impl Guild {
    /// Its text channels, in the order Discord gave them.
    pub fn text_channels(&self) -> impl Iterator<Item = &GuildChannel> {
        self.channels.iter().filter(|c| c.is_text())
    }
}

/// What `GUILD_UPDATE` and `GUILD_DELETE` name: a server's id, and whether
/// it is only unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GuildRef {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub unavailable: bool,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A `GUILD_CREATE` reads as Discord sends it**, and its text channels
    /// are its types 0 and 5 only.
    #[test]
    fn a_server_reads_with_its_text_channels() {
        let g: Guild = serde_json::from_value(json!({
            "id": "600",
            "name": "zenworld",
            "member_count": 4,
            "channels": [
                {"id": "601", "type": 0, "name": "general", "position": 0},
                {"id": "602", "type": 2, "name": "Voice"},
                {"id": "603", "type": 5, "name": "news"},
                {"id": "604", "type": 4, "name": "Text Channels"},
            ],
        }))
        .unwrap();
        assert_eq!(
            (g.id, g.name.as_str(), g.unavailable),
            (600, "zenworld", false)
        );
        assert_eq!(
            g.text_channels()
                .map(|c| (c.id, c.name.as_str()))
                .collect::<Vec<_>>(),
            [(601, "general"), (603, "news")]
        );
    }

    /// A `CHANNEL_CREATE` names its server and its description, `null` read
    /// as none; a `GUILD_DELETE` says whether it was an outage.
    #[test]
    fn a_channel_event_names_its_server_and_a_delete_its_outage() {
        let c: GuildChannel = serde_json::from_value(json!({
            "id": "605", "type": 0, "name": "lore", "guild_id": "600",
            "topic": "The world's history, era by era",
        }))
        .unwrap();
        assert_eq!(
            c,
            GuildChannel {
                id: 605,
                kind: 0,
                name: "lore".into(),
                topic: Some("The world's history, era by era".into()),
                guild_id: Some(600),
            }
        );
        let bare: GuildChannel = serde_json::from_value(json!({
            "id": "606", "type": 0, "name": "ops", "topic": null,
        }))
        .unwrap();
        assert_eq!(bare.topic, None);
        let gone: GuildRef =
            serde_json::from_value(json!({"id": "600", "unavailable": true})).unwrap();
        assert!(gone.unavailable);
        let left: GuildRef = serde_json::from_value(json!({"id": "600"})).unwrap();
        assert_eq!((left.id, left.unavailable), (600, false));
    }
}
