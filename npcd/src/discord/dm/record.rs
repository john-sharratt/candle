//! A bot's DM with a linked person, as the world keeps it
//! (`docs/npcd_discord.md` §4.2).
//!
//! One record per character and person, `discord/dm/<npc_id>/<account>` under
//! the world: the DM channel Discord gave the pair, the person's Discord user,
//! how far the thread has been posted there (`mirrored_through`, a
//! [`crate::sim::phone::Message::seq`]) and the last Discord message taken in
//! (`last_seen_discord`). It is a world record because the world's authority
//! is the one member that posts and hears the DM: it writes the record with
//! the world's other records each beat, every other member holds it as they
//! replicate, and the member that takes the world over next goes on from
//! where the record says.
//!
//! The person is named by their account key, not their display name, which
//! they may change.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::world_state::Records;

/// Where the DM records are among a world's records.
pub const PREFIX: &str = "discord/dm/";

/// One DM between a character's bot and a person.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dm {
    /// The DM channel Discord opened for the bot and the person.
    pub channel_id: u64,
    /// The person's Discord user.
    pub user_id: u64,
    /// The seq of the last message of the thread posted to the channel, or
    /// taken in from it.
    pub mirrored_through: u64,
    /// The id of the last Discord message taken in from the channel; `0` for
    /// none.
    pub last_seen_discord: u64,
}

/// Every DM of a world's bots, by character and account.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dms {
    by: BTreeMap<(u64, String), Dm>,
}

/// The record name of character `npc_id`'s DM with account `account`.
pub fn name(npc_id: u64, account: &str) -> String {
    format!("{PREFIX}{npc_id}/{account}")
}

/// The character and account a record name names, if it is a DM's.
fn parse(name: &str) -> Option<(u64, String)> {
    let (npc, account) = name.strip_prefix(PREFIX)?.split_once('/')?;
    if account.is_empty() || account.contains('/') {
        return None;
    }
    Some((npc.parse().ok()?, account.to_string()))
}

impl Dms {
    /// Character `npc_id`'s DM with `account`.
    pub fn get(&self, npc_id: u64, account: &str) -> Option<Dm> {
        self.by.get(&(npc_id, account.to_string())).copied()
    }

    /// Write character `npc_id`'s DM with `account`.
    pub fn put(&mut self, npc_id: u64, account: &str, dm: Dm) {
        self.by.insert((npc_id, account.to_string()), dm);
    }

    /// Every DM of character `npc_id`'s bot, by account.
    pub fn of(&self, npc_id: u64) -> Vec<(String, Dm)> {
        self.by
            .range((npc_id, String::new())..)
            .take_while(|((npc, _), _)| *npc == npc_id)
            .map(|((_, account), dm)| (account.clone(), *dm))
            .collect()
    }

    /// The DMs as world records.
    pub fn records(&self) -> Records {
        self.by
            .iter()
            .map(|((npc, account), dm)| {
                (
                    name(*npc, account),
                    serde_json::to_value(dm).expect("a DM serialises"),
                )
            })
            .collect()
    }

    /// The DMs `records` hold, in place of these: what a world is hosted,
    /// followed and reloaded from.
    pub fn take_in(&mut self, records: &Records) {
        self.by = records
            .iter()
            .filter_map(|(k, v): (&String, &Value)| {
                let at = parse(k)?;
                match serde_json::from_value::<Dm>(v.clone()) {
                    Ok(dm) => Some((at, dm)),
                    Err(e) => {
                        tracing::error!("discord: the DM record {k} would not decode: {e}");
                        None
                    }
                }
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn dm(channel_id: u64) -> Dm {
        Dm {
            channel_id,
            user_id: 77,
            mirrored_through: 4,
            last_seen_discord: 900,
        }
    }

    /// **A DM is the record `discord/dm/<npc_id>/<account>`**, in exactly this
    /// shape.
    #[test]
    fn a_dm_is_a_world_record_of_this_shape() {
        let mut dms = Dms::default();
        dms.put(42, "0a1b", dm(31));
        let records = dms.records();
        assert_eq!(
            records,
            [(
                "discord/dm/42/0a1b".to_string(),
                json!({
                    "channel_id": 31,
                    "user_id": 77,
                    "mirrored_through": 4,
                    "last_seen_discord": 900,
                })
            )]
            .into_iter()
            .collect()
        );
    }

    /// Records taken in are the DMs again, and nothing else among a world's
    /// records is mistaken for one.
    #[test]
    fn records_taken_in_are_the_dms_again() {
        let mut dms = Dms::default();
        dms.put(42, "0a1b", dm(31));
        dms.put(42, "ffee", dm(32));
        dms.put(43, "0a1b", dm(33));
        let mut records = dms.records();
        records.insert("sim/meta".into(), json!({"table_open": true}));
        records.insert("discord/dm/notanumber/x".into(), json!({}));
        let mut back = Dms::default();
        back.put(1, "stale", dm(1));
        back.take_in(&records);
        assert_eq!(back, dms);
        assert_eq!(
            back.of(42),
            [("0a1b".to_string(), dm(31)), ("ffee".to_string(), dm(32))]
        );
        assert_eq!(back.of(43), [("0a1b".to_string(), dm(33))]);
        assert_eq!(back.of(44), []);
        assert_eq!(back.get(43, "0a1b"), Some(dm(33)));
        assert_eq!(back.get(43, "ffee"), None);
    }
}
