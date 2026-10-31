//! Who wrote a server message, and who it mentions, as the world hears them
//! (`docs/npcd_discord.md` §3.4).
//!
//! A linked author is heard by the name their account speaks by, as in a DM.
//! Anyone else is `<display name> (@<username>)` — the server's nickname,
//! else their global name, else their username, and then their username —
//! which no character's or account's name can be, so a stranger on Discord
//! is never taken for somebody in the world. A mention, `<@id>` or `<@!id>`,
//! is written out as `@<name>` from the message's `mentions`; the bot's own
//! as `@<character>`, so the character reads that it was addressed.

use crate::discord::message::{display_name, DiscordMessage, Mentioned};

/// What an author linked to no account is called in the world.
pub fn unlinked_name(m: &DiscordMessage) -> String {
    let shown = display_name(
        m.member.as_ref(),
        m.author.global_name.as_deref(),
        &m.author.username,
    );
    format!("{shown} (@{})", m.author.username)
}

/// `text` with every mention of `mentions` written out by name: the bot's
/// own (`bot_user`) as `@<character>`, anyone else's as `@<display name>`. A
/// mention the message does not list is left as it is.
pub fn written_out(text: &str, mentions: &[Mentioned], bot_user: u64, character: &str) -> String {
    let mut out = text.to_string();
    for m in mentions {
        let name = match m.id == bot_user {
            true => character.to_string(),
            false => display_name(m.member.as_ref(), m.global_name.as_deref(), &m.username),
        };
        let at = format!("@{name}");
        for form in [format!("<@{}>", m.id), format!("<@!{}>", m.id)] {
            out = out.replace(&form, &at);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    fn message(author: Value, member: Value) -> DiscordMessage {
        serde_json::from_value(json!({
            "id": "1", "channel_id": "2", "guild_id": "3",
            "author": author, "member": member, "content": "",
        }))
        .unwrap()
    }

    /// **An unlinked author is their display name and their username** —
    /// the nickname first, then the global name, then the username alone.
    #[test]
    fn an_unlinked_author_is_named_by_display_name_and_username() {
        let both = json!({"id": "5", "username": "wren", "global_name": "Wren"});
        assert_eq!(
            unlinked_name(&message(both.clone(), json!({"nick": "Wrennie"}))),
            "Wrennie (@wren)"
        );
        assert_eq!(
            unlinked_name(&message(both, json!({"nick": null}))),
            "Wren (@wren)"
        );
        assert_eq!(
            unlinked_name(&message(
                json!({"id": "5", "username": "wren"}),
                Value::Null
            )),
            "wren (@wren)"
        );
    }

    /// **Mentions are written out by name**, both forms; the bot's own is
    /// the character; one the message does not list is left alone.
    #[test]
    fn mentions_are_written_out_by_name() {
        let mentions: Vec<Mentioned> = serde_json::from_value(json!([
            {"id": "222", "username": "Zen", "bot": true},
            {"id": "6", "username": "kit", "global_name": "Kit", "member": {"nick": "Kitty"}},
            {"id": "7", "username": "orin"},
        ]))
        .unwrap();
        assert_eq!(
            written_out(
                "<@222> ask <@!6> and <@7>, not <@8>",
                &mentions,
                222,
                "Zen the Builder"
            ),
            "@Zen the Builder ask @Kitty and @orin, not <@8>"
        );
    }
}
