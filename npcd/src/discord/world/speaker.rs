//! Who posts a message of the world into its server, and as whom
//! (`docs/npcd_discord.md` §3.1).
//!
//! A character with its own bot online, in the server, posts as that bot.
//! Everyone else — a character without one, a person writing in the console
//! — is posted by the channel's persona webhook under their own name; a
//! character with its portrait, through a signed link to it that Discord can
//! fetch without signing in (`crate::image_links`).

use std::collections::HashMap;

use crate::api::{now_ms, Authored};
use crate::discord::online::Online;
use crate::images::Kept;
use crate::tenant::HOST;
use crate::world::Hosted;

/// The longest name Discord takes on a webhook's post.
const NAME_MAX: usize = 80;

/// Who posts a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Speaker<'a> {
    /// The speaker's own bot.
    Bot(&'a Online),
    /// The channel's webhook, under this name and picture.
    Hook {
        username: String,
        avatar_url: Option<String>,
    },
}

/// Who posts what `from` said, in `guild_id`, among the bots `online`.
pub fn of<'a>(
    a: &Authored,
    hosted: &Hosted,
    guild_id: u64,
    from: &str,
    online: &HashMap<u64, &'a Online>,
) -> Speaker<'a> {
    let npc = character(a, hosted, from);
    if let Some(bot) = npc.and_then(|npc| online.get(&npc)) {
        if hosted.servers(|s| s.server(bot.npc, guild_id)).is_some() {
            return Speaker::Bot(bot);
        }
    }
    Speaker::Hook {
        username: webhook_name(from),
        avatar_url: npc.and_then(|npc| portrait_url(a, npc)),
    }
}

/// The character called `from` in `hosted`, when one is.
fn character(a: &Authored, hosted: &Hosted, from: &str) -> Option<u64> {
    let body = hosted.read(|w| w.actors().find(|x| x.name == from).map(|x| x.id.clone()))?;
    a.runtime.as_ref()?.bodies.mind_of(hosted.id(), &body)
}

/// A link Discord can fetch character `npc`'s portrait at, when it has one.
fn portrait_url(a: &Authored, npc: u64) -> Option<String> {
    let image = a.npcs.view().payload(npc)?.portrait_image_id.clone()?;
    let path = a.image_links.path(&image, now_ms() / 1000, Kept::Drawn);
    Some(format!("https://{HOST}{path}"))
}

/// `name`, as a webhook's post may carry it: at most [`NAME_MAX`]
/// characters, and never the words Discord keeps for itself.
pub fn webhook_name(name: &str) -> String {
    let mut out: String = name.trim().chars().take(NAME_MAX).collect();
    for kept in ["discord", "clyde"] {
        // ASCII-lowered, so a byte offset in it is one in `out`.
        while let Some(at) = out.to_ascii_lowercase().find(kept) {
            out.replace_range(at..at + kept.len(), &"*".repeat(kept.len()));
        }
    }
    match out.is_empty() {
        true => "Someone".to_string(),
        false => out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A webhook's name is the speaker's, cut to Discord's 80 and with the
    /// words Discord keeps starred out** — in any case — and never empty.
    #[test]
    fn a_webhook_name_is_the_speakers_as_discord_takes_it() {
        assert_eq!(webhook_name("Wick"), "Wick");
        assert_eq!(webhook_name("  Wren S (console) "), "Wren S (console)");
        assert_eq!(webhook_name(&"x".repeat(100)).chars().count(), 80);
        assert_eq!(webhook_name("Discord Dan"), "******* Dan");
        assert_eq!(webhook_name("my CLYDE"), "my *****");
        assert_eq!(webhook_name("   "), "Someone");
    }
}
