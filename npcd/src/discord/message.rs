//! A Discord message, as the gateway's `MESSAGE_CREATE` and
//! `GET /channels/{id}/messages` both carry it — only the fields the bridge
//! reads.
//!
//! Discord writes every id (a *snowflake*) as a decimal string, so the JSON
//! stays exact in JavaScript; here each is read into the `u64` it is.

use serde::de::{self, Deserializer};
use serde::Deserialize;

/// A Discord message.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DiscordMessage {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    #[serde(deserialize_with = "snowflake")]
    pub channel_id: u64,
    /// The server it was written in; absent for a DM.
    #[serde(default, deserialize_with = "maybe_snowflake")]
    pub guild_id: Option<u64>,
    pub author: Author,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    /// The webhook that posted it, for a webhook's message.
    #[serde(default, deserialize_with = "maybe_snowflake")]
    pub webhook_id: Option<u64>,
    /// The author as a member of the server, for a server's message: what
    /// the server calls them.
    #[serde(default)]
    pub member: Option<Member>,
    /// Everybody the message mentions, so a `<@id>` in it can be written out
    /// by name.
    #[serde(default)]
    pub mentions: Vec<Mentioned>,
}

/// Who wrote a message.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Author {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    pub username: String,
    /// The name they chose to be shown by, across Discord.
    #[serde(default)]
    pub global_name: Option<String>,
    /// Absent on a person's account, which is the same as `false`.
    #[serde(default)]
    pub bot: bool,
}

/// Somebody as a member of one server.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct Member {
    /// Their nickname in the server, when they have one.
    #[serde(default)]
    pub nick: Option<String>,
}

/// A user a message mentions.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Mentioned {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    pub username: String,
    #[serde(default)]
    pub global_name: Option<String>,
    /// Them as a member of the server the message was written in.
    #[serde(default)]
    pub member: Option<Member>,
}

/// What a server shows somebody as: their nickname there, else the name
/// they chose across Discord, else their username.
pub fn display_name(member: Option<&Member>, global_name: Option<&str>, username: &str) -> String {
    member
        .and_then(|m| m.nick.as_deref())
        .or(global_name)
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .unwrap_or(username)
        .to_string()
}

/// A file attached to a message.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Attachment {
    pub url: String,
}

impl DiscordMessage {
    /// What the message says, as a line of the world's thread: its text, then
    /// a `[image: <url>]` line for each attachment.
    pub fn text(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        let said = self.content.trim();
        if !said.is_empty() {
            lines.push(said.to_string());
        }
        lines.extend(
            self.attachments
                .iter()
                .map(|a| format!("[image: {}]", a.url)),
        );
        lines.join("\n")
    }
}

/// Discord's epoch, the first second of 2015, in Unix milliseconds: what a
/// snowflake's time counts from.
pub const DISCORD_EPOCH_MS: u64 = 1_420_070_400_000;

/// The lowest snowflake Discord could give anything made at `ms`: every
/// message written since is above it.
pub fn snowflake_at(ms: u64) -> u64 {
    ms.saturating_sub(DISCORD_EPOCH_MS) << 22
}

/// A snowflake, from its decimal string.
pub fn snowflake<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    let s = String::deserialize(d)?;
    s.parse()
        .map_err(|_| de::Error::custom(format!("`{s}` is not a snowflake")))
}

/// A snowflake that may be absent or `null`.
pub fn maybe_snowflake<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    match Option::<String>::deserialize(d)? {
        None => Ok(None),
        Some(s) => s
            .parse()
            .map(Some)
            .map_err(|_| de::Error::custom(format!("`{s}` is not a snowflake"))),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A DM reads as Discord sends it**: string ids to numbers, no server,
    /// and an author with no `bot` field a person.
    #[test]
    fn a_dm_reads_as_discord_sends_it() {
        let m: DiscordMessage = serde_json::from_value(json!({
            "id": "1300000000000000011",
            "channel_id": "1300000000000000012",
            "author": {"id": "1300000000000000013", "username": "wren", "global_name": "Wren"},
            "content": " hello there ",
            "attachments": [{"id": "5", "url": "https://cdn.discordapp.com/a/b.png", "filename": "b.png"}],
            "embeds": [],
            "type": 0,
        }))
        .unwrap();
        assert_eq!(
            m,
            DiscordMessage {
                id: 1_300_000_000_000_000_011,
                channel_id: 1_300_000_000_000_000_012,
                guild_id: None,
                author: Author {
                    id: 1_300_000_000_000_000_013,
                    username: "wren".into(),
                    global_name: Some("Wren".into()),
                    bot: false,
                },
                content: " hello there ".into(),
                attachments: vec![Attachment {
                    url: "https://cdn.discordapp.com/a/b.png".into()
                }],
                webhook_id: None,
                member: None,
                mentions: vec![],
            }
        );
        assert_eq!(
            m.text(),
            "hello there\n[image: https://cdn.discordapp.com/a/b.png]"
        );
    }

    /// Discord documents `175928847299117063` as made at `1462015105796`.
    #[test]
    fn a_time_is_the_lowest_snowflake_of_its_millisecond() {
        assert_eq!(
            snowflake_at(1_462_015_105_796),
            175_928_847_299_117_063 & !((1 << 22) - 1)
        );
        assert_eq!(snowflake_at(0), 0);
    }

    #[test]
    fn a_server_message_names_its_server_and_a_webhook_its_hook() {
        let m: DiscordMessage = serde_json::from_value(json!({
            "id": "1", "channel_id": "2", "guild_id": "3", "webhook_id": "4",
            "author": {"id": "5", "username": "hook", "bot": true},
        }))
        .unwrap();
        assert_eq!(
            (m.guild_id, m.webhook_id, m.author.bot),
            (Some(3), Some(4), true)
        );
        assert_eq!(m.text(), "");
        assert!(serde_json::from_value::<DiscordMessage>(json!({
            "id": "x1", "channel_id": "2", "author": {"id": "5", "username": "a"},
        }))
        .is_err());
    }

    /// **A server message carries its author as a member and who it
    /// mentions**, as Discord sends them.
    #[test]
    fn a_server_message_reads_its_member_and_mentions() {
        let m: DiscordMessage = serde_json::from_value(json!({
            "id": "1", "channel_id": "2", "guild_id": "3",
            "author": {"id": "5", "username": "wren", "global_name": null},
            "member": {"nick": "Wrennie", "roles": []},
            "content": "<@222> hi <@!6>",
            "mentions": [
                {"id": "222", "username": "Zen", "bot": true},
                {"id": "6", "username": "kit", "global_name": "Kit", "member": {"nick": null}},
            ],
        }))
        .unwrap();
        assert_eq!(
            m.member,
            Some(Member {
                nick: Some("Wrennie".into())
            })
        );
        assert_eq!(
            m.mentions,
            [
                Mentioned {
                    id: 222,
                    username: "Zen".into(),
                    global_name: None,
                    member: None,
                },
                Mentioned {
                    id: 6,
                    username: "kit".into(),
                    global_name: Some("Kit".into()),
                    member: Some(Member { nick: None }),
                },
            ]
        );
    }

    /// **A server shows its nickname first, then the global name, then the
    /// username**; a blank one is no name.
    #[test]
    fn a_display_name_is_the_nick_then_the_global_name_then_the_username() {
        let nick = Member {
            nick: Some("Wrennie".into()),
        };
        assert_eq!(display_name(Some(&nick), Some("Wren"), "wren"), "Wrennie");
        assert_eq!(display_name(None, Some("Wren"), "wren"), "Wren");
        assert_eq!(
            display_name(Some(&Member { nick: None }), None, "wren"),
            "wren"
        );
        assert_eq!(display_name(None, Some("  "), "wren"), "wren");
    }
}
