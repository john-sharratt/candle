//! A bot shown typing in a DM a person is waiting on
//! (`docs/npcd_discord.md` §8 step 13).
//!
//! From when a person's DM from Discord is the last thing said to the
//! character until the character answers — its answer is then the last
//! thing said — and for two minutes at most, its bot shows typing in that
//! DM. Discord shows typing for ten seconds from each ask, so it is asked
//! again every eight while the DM waits.
//!
//! Stateless but for when each DM was last asked: whether a DM is waiting is
//! read off its thread each beat, so a restart, a move, or an answer posted
//! from the console all end it the same way.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::outbound::speaking_name;
use crate::api::{now_ms, Authored};
use crate::discord::online::Online;
use crate::discord::rejected::Rejected;
use crate::discord::server::embodied;

/// The longest a bot is shown typing for one message.
pub const TYPING_FOR: Duration = Duration::from_secs(2 * 60);

/// How often typing is asked for again while a DM waits.
pub const RENEW: Duration = Duration::from_secs(8);

/// When each DM channel was last shown typing.
pub type Typed = HashMap<u64, Instant>;

/// Show `bot` typing in each of its DMs a person is waiting on.
pub async fn show(a: &Arc<Authored>, bot: &Online, typed: &mut Typed) -> Result<(), Rejected> {
    let Some((hosted, character)) = embodied(a, bot) else {
        return Ok(());
    };
    for (account, dm) in hosted.dms(|d| d.of(bot.npc)) {
        let Some(person) = speaking_name(a, &account).await else {
            continue;
        };
        let waiting = hosted.sim(|s| {
            s.threads
                .direct_between(&person, &character)
                .and_then(|t| t.messages.last())
                .is_some_and(|m| {
                    m.from == person
                        && m.origin.is_some()
                        && now_ms().saturating_sub(m.at_ms) < TYPING_FOR.as_millis() as u64
                })
        });
        if !waiting {
            typed.remove(&dm.channel_id);
            continue;
        }
        if typed
            .get(&dm.channel_id)
            .is_some_and(|at| at.elapsed() < RENEW)
        {
            continue;
        }
        match a.discord.trigger_typing(&bot.token, dm.channel_id).await {
            Ok(()) => {
                typed.insert(dm.channel_id, Instant::now());
            }
            Err(e) => Rejected::or_logged(bot.npc, "typing", e)?,
        }
    }
    Ok(())
}
