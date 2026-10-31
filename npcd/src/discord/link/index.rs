//! Which account a Discord user is linked to (`docs/npcd_discord.md` §3.2).
//!
//! The custom object `discord/user/<discord user id>` → `{ "account": <key> }`,
//! replicated, so a message arriving on any member's bot finds its author's
//! account without reading every account. The account's own `discord` field
//! ([`crate::accounts::DiscordUser`]) is the other half; [`super::linking`]
//! writes the two together.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use serde_json::json;

/// Where the index lives among the substrate's custom objects.
pub const PREFIX: &str = "discord/user/";

/// The record key of Discord user `discord_id`'s link.
pub fn key(discord_id: &str) -> String {
    format!("{PREFIX}{discord_id}")
}

/// The account Discord user `discord_id` is linked to.
pub fn account_of(shared: &SharedSubstrate, discord_id: &str) -> Option<String> {
    shared
        .custom_object(&key(discord_id))?
        .blob
        .get("account")?
        .as_str()
        .map(str::to_string)
}

/// Every link, as `(discord user id, account key)`.
pub fn all(shared: &SharedSubstrate) -> Vec<(String, String)> {
    shared
        .custom_objects_under(PREFIX)
        .into_iter()
        .filter_map(|(o, _)| {
            let account = o.blob.get("account")?.as_str()?.to_string();
            Some((o.key[PREFIX.len()..].to_string(), account))
        })
        .collect()
}

/// Link Discord user `discord_id` to `account`.
pub fn put(shared: &SharedSubstrate, discord_id: &str, account: &str) -> anyhow::Result<()> {
    shared
        .put_custom_object(CustomObjectPayload::new(
            key(discord_id),
            json!({ "account": account }),
        ))
        .map_err(|e| {
            anyhow::anyhow!("the link of Discord user {discord_id} was not written: {e:?}")
        })
}

/// Unlink Discord user `discord_id`: its key's tombstone.
pub fn remove(shared: &SharedSubstrate, discord_id: &str) -> anyhow::Result<()> {
    shared
        .put_custom_object(CustomObjectPayload::ceased(key(discord_id)))
        .map_err(|e| {
            anyhow::anyhow!("the link of Discord user {discord_id} was not removed: {e:?}")
        })
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    /// **A link is the record `discord/user/<id>` → `{account}`**, and a
    /// removed one is gone.
    #[test]
    fn a_link_is_a_custom_object_naming_the_account() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        put(&shared, "1300000000000000013", "0a1b").unwrap();
        assert_eq!(
            shared
                .custom_object("discord/user/1300000000000000013")
                .unwrap()
                .blob,
            json!({ "account": "0a1b" })
        );
        assert_eq!(
            account_of(&shared, "1300000000000000013").as_deref(),
            Some("0a1b")
        );
        assert_eq!(
            all(&shared),
            [("1300000000000000013".to_string(), "0a1b".to_string())]
        );
        remove(&shared, "1300000000000000013").unwrap();
        assert_eq!(account_of(&shared, "1300000000000000013"), None);
        assert_eq!(all(&shared), []);
    }
}
