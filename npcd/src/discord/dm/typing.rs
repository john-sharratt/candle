//! A bot shown typing in a DM a person is waiting on
//! (`docs/npcd_discord.md` §8 step 13).
//!
//! While a person's DM from Discord sits above the character's read cursor
//! on their thread, and for two minutes at most, its bot shows typing in that
//! DM. The character's conversation reads through the line it took up —
//! whether it answered or wrote it down (`docs/npc_conversations.md` §8) — so
//! typing stops when the turn ends, whatever it ended in. Discord shows typing
//! for ten seconds from each ask, so it is asked again every eight while the
//! DM waits.
//!
//! Stateless but for when each DM was last asked: whether a DM is waiting is
//! read off its thread each beat, so a restart, a move, or an answer posted
//! from the console all end it the same way.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::accounts::public_id;
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
        let person = public_id(&account);
        let waiting = hosted.sim(|s| {
            s.threads.direct_waits_on(&character.id, &person)
                && s.threads
                    .direct_between(&character.id, &person)
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
        // Stamped whatever Discord answers, so a refused DM is asked again
        // after a renewal, not on the next beat; a refusal is not asked
        // again for as long as typing could last.
        let result = a.discord.trigger_typing(&bot.token, dm.channel_id).await;
        let next = match result {
            Ok(()) => Instant::now(),
            Err(_) => Instant::now() + TYPING_FOR,
        };
        typed.insert(dm.channel_id, next);
        if let Err(e) = result {
            Rejected::or_logged(bot.npc, "typing", e)?;
        }
    }
    Ok(())
}
