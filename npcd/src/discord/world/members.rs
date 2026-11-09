//! Who the members of a world's thread are on Discord
//! (`docs/npcd_discord.md` §3), and what a channel is opened to them with.
//!
//! Members are ids. A character — a body id — is its bot's user, when it has
//! a bot. A person — an account's public handle — is the Discord account
//! linked to theirs. Anyone else — a character without a bot, a person who
//! has linked nothing — is not on Discord, and is posted for by the persona
//! webhook.

use std::collections::BTreeMap;

use crate::accounts::public_id;
use crate::api::Authored;
use crate::discord::client::Overwrite;
use crate::discord::link::index;
use crate::discord::store;
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

/// The character whose body is `id` in `hosted`, when it is one.
pub fn character(a: &Authored, hosted: &Hosted, id: &str) -> Option<u64> {
    a.runtime.as_ref()?.bodies.mind_of(hosted.id(), id)
}

/// Character `npc`'s bot's Discord user, when it has a bot.
pub fn bot_user(a: &Authored, npc: u64) -> Option<u64> {
    store::get(&a.npcs.substrate(), npc)?.user_id.parse().ok()
}

/// Every linked person's Discord user, by their account's public handle.
pub fn linked_people(a: &Authored) -> BTreeMap<String, u64> {
    index::all(&a.npcs.substrate())
        .into_iter()
        .filter_map(|(discord_id, account)| Some((public_id(&account), discord_id.parse().ok()?)))
        .collect()
}

/// The Discord users of the members `ids` who are on Discord, among
/// `people`: each character's bot, and each linked person; by id.
pub fn discord_users(
    a: &Authored,
    hosted: &Hosted,
    ids: &[String],
    people: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    ids.iter()
        .filter_map(|id| {
            let user = match character(a, hosted, id) {
                Some(npc) => bot_user(a, npc)?,
                None => *people.get(id)?,
            };
            Some((id.clone(), user))
        })
        .collect()
}
