//! A character's bot in its servers' channels (`docs/npcd_discord.md` §3.4):
//! every text channel of every server the bot is in is a group thread on
//! the character's handset, heard and answered as any group is.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`record`] | `discord/channel/…` and `discord/server/…`, as the world keeps them |
//! | [`notice`] | what a character is told of a join or a leave, and a thread's name |
//! | [`thread`] | a channel's thread, opened, renamed and closed with it |
//! | [`guild`] | a server joined, replayed, renamed or left, and its channels' changes |
//! | [`author`] | who wrote a server message and who it mentions, by name |
//! | [`inbound`] | a server message heard in the world |
//! | [`apply`] | `WorldOp::ChannelMessage`, made on the world's authority |
//! | [`outbound`] | what is said on a channel's thread, posted to the channel |
//! | [`gap`] | what was written while no member ran the bot |

pub mod apply;
pub mod author;
pub mod gap;
pub mod guild;
pub mod inbound;
pub mod notice;
pub mod outbound;
pub mod record;
pub mod thread;

use std::sync::Arc;

use crate::api::Authored;
use crate::authority::ops::WorldOp;
use crate::discord::online::Online;
use crate::world::Hosted;

/// The world `bot`'s character has its body in — the bot's own — and what
/// the character is called there.
pub fn embodied(a: &Authored, bot: &Online) -> Option<(Arc<Hosted>, String)> {
    let (hosted, body) = a
        .runtime
        .as_ref()?
        .body_of(bot.npc)
        .filter(|(hosted, _)| hosted.id() == bot.world)?;
    let character = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()))?;
    Some((hosted, character))
}

/// Tell `bot`'s character `text`, as something that happened to it
/// (`WorldOp::Notice`).
pub async fn tell(a: &Authored, bot: &Online, text: String) {
    let Some(rt) = a.runtime.as_ref() else {
        return;
    };
    let op = WorldOp::Notice { npc: bot.npc, text };
    if let Err(e) = rt.world_write(&bot.world, op).await {
        tracing::warn!(
            npc = bot.npc,
            "discord: the character was not told of its server: {e:#}"
        );
    }
}
