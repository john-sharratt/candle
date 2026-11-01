//! Who the members of a world's thread are on Discord
//! (`docs/npcd_discord.md` §3), and what a channel is opened to them with.
//!
//! A character is its bot's user, when it has a bot. A person is the
//! Discord account linked to theirs: the one whose account speaks by the
//! name the thread holds. Anyone else — a character without a bot, a person
//! who has linked nothing — is not on Discord, and is posted for by the
//! persona webhook.

use std::collections::BTreeMap;

use crate::accounts::public_id;
use crate::api::Authored;
use crate::discord::client::Overwrite;
use crate::discord::link::index;
use crate::discord::store;
use crate::engine::speaking_as;
use crate::world::Hosted;

/// Discord's permission bits, as an overwrite carries them.
pub const VIEW: u64 = 1 << 10;
pub const SEND: u64 = 1 << 11;
pub const HISTORY: u64 = 1 << 16;
pub const MANAGE_WEBHOOKS: u64 = 1 << 29;
pub const CREATE_PRIVATE_THREADS: u64 = 1 << 36;
pub const SEND_IN_THREADS: u64 = 1 << 38;

/// What a member of a channel is let do in it: see and read it, and write.
pub const MEMBER: u64 = VIEW | SEND | HISTORY;

/// What the listener is let do in a channel it mirrors into: a member's,
/// and its webhooks and private threads.
pub const LISTENER: u64 = MEMBER | MANAGE_WEBHOOKS | CREATE_PRIVATE_THREADS | SEND_IN_THREADS;

/// An overwrite for one member.
pub fn member(user_id: u64, allow: u64, deny: u64) -> Overwrite {
    Overwrite {
        id: user_id,
        kind: 1,
        allow,
        deny,
    }
}

/// The overwrite for `@everyone` — the role whose id is the server's.
pub fn everyone(guild_id: u64, allow: u64, deny: u64) -> Overwrite {
    Overwrite {
        id: guild_id,
        kind: 0,
        allow,
        deny,
    }
}

/// The character called `name` in `hosted`, when one is.
pub fn character(a: &Authored, hosted: &Hosted, name: &str) -> Option<u64> {
    let body = hosted.read(|w| w.actors().find(|x| x.name == name).map(|x| x.id.clone()))?;
    a.runtime.as_ref()?.bodies.mind_of(hosted.id(), &body)
}

/// Character `npc`'s bot's Discord user, when it has a bot.
pub fn bot_user(a: &Authored, npc: u64) -> Option<u64> {
    store::get(&a.npcs.substrate(), npc)?.user_id.parse().ok()
}

/// Every linked person's Discord user, by the name their account speaks by.
pub async fn linked_people(a: &Authored) -> BTreeMap<String, u64> {
    let accounts = a.accounts.read().await;
    index::all(&a.npcs.substrate())
        .into_iter()
        .filter_map(|(discord_id, account)| {
            let identity = accounts.identity(&account)?;
            let name = speaking_as(&identity, &public_id(&account), &a.roles);
            Some((name, discord_id.parse().ok()?))
        })
        .collect()
}

/// The Discord users of `names` who are on Discord, among `people`: each
/// character's bot, and each linked person.
pub fn discord_users(
    a: &Authored,
    hosted: &Hosted,
    names: &[String],
    people: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    names
        .iter()
        .filter_map(|name| {
            let user = match character(a, hosted, name) {
                Some(npc) => bot_user(a, npc)?,
                None => *people.get(name)?,
            };
            Some((name.clone(), user))
        })
        .collect()
}
