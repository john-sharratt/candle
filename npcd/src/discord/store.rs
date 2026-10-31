//! The record of a character's bot.
//!
//! One `CustomObject discord/bot/<npc_id>` per character that has a bot, in
//! the substrate (`docs/replicated_substrate.md` §10.3), so every member reads
//! the same bots and whichever holds a world can connect its characters. It is
//! never in the character's own record, which `/v1/npcs/export` hands out
//! whole; and the token in it is sealed ([`crate::discord::seal`]), never
//! returned by a route and never logged. Disconnecting writes the key's
//! tombstone, which replicates like the record did.

use std::fmt::{self, Debug, Formatter};

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use serde::{Deserialize, Serialize};

/// Where bots live among the substrate's custom objects.
const PREFIX: &str = "discord/bot/";

/// The record key of character `npc_id`'s bot.
pub fn key(npc_id: u64) -> String {
    format!("{PREFIX}{npc_id}")
}

/// A character's bot, as its record holds it.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bot {
    /// The Discord application the bot belongs to: what an invite names.
    pub application_id: String,
    /// The bot's own Discord user.
    pub user_id: String,
    /// What the bot is called on Discord, as of the connect.
    pub username: String,
    /// The bot token, sealed for this record.
    pub token: String,
    /// When the owner connected it, in real milliseconds.
    pub connected_ms: u64,
}

impl Debug for Bot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bot")
            .field("application_id", &self.application_id)
            .field("user_id", &self.user_id)
            .field("username", &self.username)
            .field("token", &"…")
            .field("connected_ms", &self.connected_ms)
            .finish()
    }
}

/// Character `npc_id`'s bot, if it has one.
pub fn get(shared: &SharedSubstrate, npc_id: u64) -> Option<Bot> {
    shared
        .custom_object(&key(npc_id))
        .filter(|o| !o.ceased)
        .and_then(|o| serde_json::from_value(o.blob).ok())
}

/// Write character `npc_id`'s bot.
pub fn put(shared: &SharedSubstrate, npc_id: u64, bot: &Bot) -> anyhow::Result<()> {
    let body = serde_json::to_value(bot).expect("a bot serialises");
    shared
        .put_custom_object(CustomObjectPayload::new(key(npc_id), body))
        .map_err(|e| anyhow::anyhow!("the bot of {npc_id} was not written: {e:?}"))
}

/// Remove character `npc_id`'s bot: its key's tombstone.
pub fn remove(shared: &SharedSubstrate, npc_id: u64) -> anyhow::Result<()> {
    shared
        .put_custom_object(CustomObjectPayload::ceased(key(npc_id)))
        .map_err(|e| anyhow::anyhow!("the bot of {npc_id} was not removed: {e:?}"))
}

#[cfg(test)]
mod tests {
    use std::env::temp_dir;
    use std::fs::{create_dir_all, remove_dir_all};
    use std::path::PathBuf;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::json;

    use super::*;
    use crate::npcs::Npcs;

    fn tmp() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = temp_dir().join(format!(
            "npcd-discord-store-{}-{}",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = remove_dir_all(&p);
        create_dir_all(&p).unwrap();
        p
    }

    fn bot() -> Bot {
        Bot {
            application_id: "111".into(),
            user_id: "222".into(),
            username: "Zen".into(),
            token: "c2VhbGVk".into(),
            connected_ms: 1_800_000_000_000,
        }
    }

    /// **A bot is the record `discord/bot/<npc_id>`**, in exactly this shape,
    /// and survives a restart.
    #[test]
    fn a_bot_is_a_custom_object_record_that_survives_a_restart() {
        let dir = tmp();
        let shared = Npcs::load(&dir).unwrap().substrate();
        put(&shared, 42, &bot()).unwrap();
        assert_eq!(
            shared.custom_object("discord/bot/42").unwrap().blob,
            json!({
                "application_id": "111",
                "user_id": "222",
                "username": "Zen",
                "token": "c2VhbGVk",
                "connected_ms": 1_800_000_000_000u64,
            })
        );
        drop(shared);
        let again = Npcs::load(&dir).unwrap().substrate();
        assert_eq!(get(&again, 42), Some(bot()));
        assert_eq!(get(&again, 43), None);
    }

    #[test]
    fn a_removed_bot_is_gone() {
        let dir = tmp();
        let shared = Npcs::load(&dir).unwrap().substrate();
        put(&shared, 42, &bot()).unwrap();
        remove(&shared, 42).unwrap();
        assert_eq!(get(&shared, 42), None);
        drop(shared);
        assert_eq!(get(&Npcs::load(&dir).unwrap().substrate(), 42), None);
    }

    #[test]
    fn a_bot_never_prints_its_token() {
        let printed = format!("{:?}", bot());
        assert!(!printed.contains("c2VhbGVk"), "{printed}");
        assert!(printed.contains("\"Zen\""), "{printed}");
    }
}
