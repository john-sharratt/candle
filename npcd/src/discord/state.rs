//! Whether a character's bot is on Discord (`docs/npcd_discord.md` §3.3).
//!
//! The custom object `discord/state/<npc_id>`, written by the member running
//! the bot's shard when the bot's state changes — online once Discord answers
//! `READY`, refused when Discord refuses its token — and tombstoned by that
//! member when it stops the shard for any other reason. A refusal stays, so
//! the console can say so, until a new token is connected.
//!
//! A record no longer in force reads as offline: one older than its world's
//! authority, which is a member that stopped running the bot without saying
//! so; and a refusal of a token since replaced.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use serde::{Deserialize, Serialize};

use crate::discord::store::Bot;
use crate::world_state::store as world_store;

/// Where bots' states live among the substrate's custom objects.
const PREFIX: &str = "discord/state/";

/// The record key of character `npc_id`'s bot's state.
pub fn key(npc_id: u64) -> String {
    format!("{PREFIX}{npc_id}")
}

/// What a running shard last said of its bot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// The shard is open and Discord answered `READY`.
    Online,
    /// Discord refused the token (close code 4004, or a 401), and the shard
    /// was stopped.
    Refused,
}

/// A bot's state record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BotState {
    pub state: State,
    /// When it took the state, in real milliseconds.
    pub since_ms: u64,
    /// The world whose authority runs the bot.
    pub world: String,
    /// The `connected_ms` of the bot record whose token it ran: a refusal is
    /// of that token, and a newer one is not refused.
    pub connected_ms: u64,
}

/// What the console says of a bot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Online { world: String },
    Offline,
    Refused,
}

impl Status {
    /// Its name on the wire: `online`, `offline` or `refused`.
    pub fn name(&self) -> &'static str {
        match self {
            Status::Online { .. } => "online",
            Status::Offline => "offline",
            Status::Refused => "refused",
        }
    }
}

/// Character `npc_id`'s bot's state record, if it has one.
pub fn get(shared: &SharedSubstrate, npc_id: u64) -> Option<BotState> {
    shared
        .custom_object(&key(npc_id))
        .filter(|o| !o.ceased)
        .and_then(|o| serde_json::from_value(o.blob).ok())
}

/// Write character `npc_id`'s bot's state.
pub fn put(shared: &SharedSubstrate, npc_id: u64, state: &BotState) -> anyhow::Result<()> {
    let body = serde_json::to_value(state).expect("a bot state serialises");
    shared
        .put_custom_object(CustomObjectPayload::new(key(npc_id), body))
        .map_err(|e| anyhow::anyhow!("the state of bot {npc_id} was not written: {e:?}"))
}

/// Tombstone character `npc_id`'s bot's state: no member runs it.
pub fn remove(shared: &SharedSubstrate, npc_id: u64) -> anyhow::Result<()> {
    shared
        .put_custom_object(CustomObjectPayload::ceased(key(npc_id)))
        .map_err(|e| anyhow::anyhow!("the state of bot {npc_id} was not removed: {e:?}"))
}

/// Whether `bot`'s token is one Discord refused: the record says so of this
/// very token.
pub fn refused(shared: &SharedSubstrate, npc_id: u64, bot: &Bot) -> bool {
    get(shared, npc_id)
        .is_some_and(|s| s.state == State::Refused && s.connected_ms == bot.connected_ms)
}

/// What character `npc_id`'s `bot` is, as its state record and its world's
/// records say.
pub fn status(shared: &SharedSubstrate, npc_id: u64, bot: &Bot) -> Status {
    let Some(s) = get(shared, npc_id) else {
        return Status::Offline;
    };
    match s.state {
        State::Refused if s.connected_ms == bot.connected_ms => Status::Refused,
        State::Online
            if s.connected_ms == bot.connected_ms
                && s.since_ms >= world_store::highest_epoch(shared, &s.world) =>
        {
            Status::Online { world: s.world }
        }
        _ => Status::Offline,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;
    use crate::world_state::{diff, Records};

    fn bot(connected_ms: u64) -> Bot {
        Bot {
            application_id: "111".into(),
            user_id: "222".into(),
            username: "Zen".into(),
            token: "c2VhbGVk".into(),
            connected_ms,
        }
    }

    fn online(since_ms: u64, connected_ms: u64) -> BotState {
        BotState {
            state: State::Online,
            since_ms,
            world: "life".into(),
            connected_ms,
        }
    }

    /// **A bot's state is the record `discord/state/<npc_id>`**, in exactly
    /// this shape.
    #[test]
    fn a_state_is_a_custom_object_of_this_shape() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        put(&shared, 42, &online(1_800_000_000_500, 1_800_000_000_000)).unwrap();
        assert_eq!(
            shared.custom_object("discord/state/42").unwrap().blob,
            json!({
                "state": "online",
                "since_ms": 1_800_000_000_500u64,
                "world": "life",
                "connected_ms": 1_800_000_000_000u64,
            })
        );
        remove(&shared, 42).unwrap();
        assert_eq!(get(&shared, 42), None);
    }

    /// **Online while its record is the world's current authority's**;
    /// offline with no record, with one older than the world's epoch, or for
    /// a token since replaced; refused only for the token refused.
    #[test]
    fn a_status_is_what_the_record_says_while_it_is_in_force() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        assert_eq!(status(&shared, 42, &bot(10)), Status::Offline);

        let records: Records = [("sim/meta".to_string(), json!({}))].into_iter().collect();
        world_store::write(&shared, "life", &diff(&Records::new(), &records), 1_000).unwrap();
        put(&shared, 42, &online(1_500, 10)).unwrap();
        assert_eq!(
            status(&shared, 42, &bot(10)),
            Status::Online {
                world: "life".into()
            }
        );
        assert_eq!(
            status(&shared, 42, &bot(11)),
            Status::Offline,
            "a new token"
        );

        // The world taken over since, by a member that has not run the bot.
        let moved: Records = [("sim/meta".to_string(), json!({"moved": true}))]
            .into_iter()
            .collect();
        world_store::write(&shared, "life", &diff(&records, &moved), 2_000).unwrap();
        assert_eq!(status(&shared, 42, &bot(10)), Status::Offline);

        put(
            &shared,
            42,
            &BotState {
                state: State::Refused,
                since_ms: 2_500,
                world: "life".into(),
                connected_ms: 10,
            },
        )
        .unwrap();
        assert_eq!(status(&shared, 42, &bot(10)), Status::Refused);
        assert!(refused(&shared, 42, &bot(10)));
        assert_eq!(status(&shared, 42, &bot(11)), Status::Offline);
        assert!(!refused(&shared, 42, &bot(11)));
        assert_eq!(
            [Status::Refused.name(), Status::Offline.name()],
            ["refused", "offline"]
        );
    }
}
