//! Every character's bot this member runs (`docs/npcd_discord.md` §3.3, §6):
//! a gateway shard per bot, on the authority of its character's world.
//!
//! The bots have a beat of their own, at the world's ([`BEAT`]), on the
//! member that is the world's authority. Each beat:
//!
//! 1. **The shards follow the authority** ([`wanted`]): a bot of a character
//!    with a body in a world this member is the authority of is connected,
//!    one that is not — the world lost, the character moved, the bot's record
//!    tombstoned or changed — is closed and its state tombstoned. A bot whose
//!    token Discord refused, or that the gateway will not serve, stays closed
//!    until a new token is connected.
//! 2. **What the shards heard is taken in.** `READY` writes the bot's state
//!    online, with its world, and reads what its DMs missed ([`gap`]); a DM
//!    is heard in the world ([`inbound`]); a refusal (`4004`, or a `401` on
//!    any call) stops the shard and writes the bot's state refused.
//! 3. **What the characters wrote is posted** to their DMs ([`outbound`]).
//!
//! A shard never writes the world: it hands what it hears to this beat's
//! queue, and the beat — on the authority — makes the world's writes. The
//! beat is the bots' own rather than the world's because posting waits on
//! Discord, and a world's beat waits on nothing.
//!
//! [`gap`]: crate::discord::dm::gap
//! [`inbound`]: crate::discord::dm::inbound
//! [`outbound`]: crate::discord::dm::outbound

#[cfg(test)]
mod tests;
pub mod wanted;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Mutex};
use tokio::time::{interval, MissedTickBehavior};

use crate::api::{now_ms, Authored};
use crate::discord::dm::online::Online;
use crate::discord::dm::outbound::{self, Unreachable};
use crate::discord::dm::told::Told;
use crate::discord::dm::{gap, inbound, Rejected};
use crate::discord::gateway::protocol::{self, DIRECT_MESSAGES};
use crate::discord::gateway::shard::{Config, Event, Shard};
use crate::discord::state::{self, BotState, State};
use crate::engine::driver::EVERY as BEAT;
use crate::tenant::HOST;

use self::wanted::wanted;

/// How long a shard first waits to connect again in production.
const RETRY: Duration = Duration::from_secs(1);

/// The bots this member runs.
pub struct Bots {
    /// The gateway's base: Discord's, or a test's.
    gateway: String,
    /// The console's public address, which an unlinked author is sent to.
    console: String,
    retry: Duration,
    events: mpsc::UnboundedSender<(u64, Event)>,
    held: Mutex<Held>,
}

/// What the beat holds between beats.
struct Held {
    heard: mpsc::UnboundedReceiver<(u64, Event)>,
    running: HashMap<u64, Running>,
    told: HashMap<u64, Told>,
    unreachable: HashMap<u64, Unreachable>,
    /// For each character, the `connected_ms` of a bot the gateway will not
    /// serve or whose token does not open: not tried again until another is
    /// connected.
    halted: HashMap<u64, u64>,
}

/// One bot's shard.
struct Running {
    shard: Shard,
    world: String,
    connected_ms: u64,
    token: String,
    online: Option<Online>,
}

impl Bots {
    /// Bots that connect to `gateway`, send an unlinked author to the
    /// console at `console`, and first wait `retry` to reconnect.
    pub fn new(gateway: &str, console: &str, retry: Duration) -> Self {
        let (events, heard) = mpsc::unbounded_channel();
        Self {
            gateway: gateway.to_string(),
            console: console.trim_end_matches('/').to_string(),
            retry,
            events,
            held: Mutex::new(Held {
                heard,
                running: HashMap::new(),
                told: HashMap::new(),
                unreachable: HashMap::new(),
                halted: HashMap::new(),
            }),
        }
    }

    /// Bots on Discord's own gateway, sending people to npcd's console.
    pub fn production() -> Self {
        Self::new(protocol::PRODUCTION, &format!("https://{HOST}"), RETRY)
    }

    /// The console's public address.
    pub fn console(&self) -> &str {
        &self.console
    }

    /// The characters whose bots are online here.
    pub async fn online(&self) -> Vec<u64> {
        let held = self.held.lock().await;
        let mut online: Vec<u64> = held
            .running
            .iter()
            .filter(|(_, r)| r.online.is_some())
            .map(|(npc, _)| *npc)
            .collect();
        online.sort();
        online
    }

    /// The characters whose bots have a shard here, online or not yet.
    pub async fn running(&self) -> Vec<u64> {
        let mut running: Vec<u64> = self.held.lock().await.running.keys().copied().collect();
        running.sort();
        running
    }

    /// One beat: the shards to the authority, what they heard taken in, what
    /// the characters wrote posted.
    pub async fn tick(&self, a: &Arc<Authored>) {
        let mut held = self.held.lock().await;
        self.follow(a, &mut held).await;
        self.take_in(a, &mut held).await;
        let online: Vec<Online> = held
            .running
            .values()
            .filter_map(|r| r.online.clone())
            .collect();
        for bot in online {
            let unreachable = held.unreachable.entry(bot.npc).or_default();
            if outbound::post(a, &bot, unreachable).await.is_err() {
                refuse(a, &mut held, bot.npc).await;
            }
        }
    }

    /// Close every shard, tombstoning each bot's state: this member is
    /// stopping.
    pub async fn close_all(&self, a: &Authored) {
        let mut held = self.held.lock().await;
        let shared = a.npcs.substrate();
        for (npc, r) in held.running.drain() {
            r.shard.close().await;
            if let Err(e) = state::remove(&shared, npc) {
                tracing::error!(npc, "discord: {e:#}");
            }
        }
    }

    /// Close the shards this member should not run, and open the ones it
    /// should.
    async fn follow(&self, a: &Authored, held: &mut Held) {
        let shared = a.npcs.substrate();
        let wanted = wanted(a);
        let stale: Vec<u64> = held
            .running
            .iter()
            // A shard that ended on its own said why, and [`Self::take_in`]
            // reads it: a refusal is to be written, not tombstoned.
            .filter(|(npc, r)| {
                !wanted
                    .get(npc)
                    .is_some_and(|w| w.world == r.world && w.bot.connected_ms == r.connected_ms)
            })
            .map(|(npc, _)| *npc)
            .collect();
        for npc in stale {
            let Some(r) = held.running.remove(&npc) else {
                continue;
            };
            r.shard.close().await;
            if let Err(e) = state::remove(&shared, npc) {
                tracing::error!(npc, "discord: {e:#}");
            }
            tracing::info!(npc, world = %r.world, "discord: the bot's shard is closed here");
        }
        for (npc, w) in wanted {
            if held.running.contains_key(&npc) || held.halted.get(&npc) == Some(&w.bot.connected_ms)
            {
                continue;
            }
            let token = match a.discord_tokens.open(npc, &w.bot.token) {
                Ok(token) => token,
                Err(why) => {
                    tracing::warn!(
                        npc,
                        "discord: the bot's token does not open here ({why:?}); connect it again"
                    );
                    held.halted.insert(npc, w.bot.connected_ms);
                    continue;
                }
            };
            let world_name = a
                .worlds
                .read()
                .await
                .get(&w.world)
                .and_then(|r| r.body["name"].as_str().map(str::to_string))
                .unwrap_or_else(|| w.world.clone());
            let config = Config {
                npc,
                gateway: self.gateway.clone(),
                token: token.clone(),
                intents: DIRECT_MESSAGES,
                activity: format!("in {world_name}"),
                retry: self.retry,
            };
            tracing::info!(npc, world = %w.world, "discord: opening the bot's shard");
            held.running.insert(
                npc,
                Running {
                    shard: Shard::spawn(config, self.events.clone()),
                    world: w.world,
                    connected_ms: w.bot.connected_ms,
                    token,
                    online: None,
                },
            );
        }
    }

    /// Take in what the shards heard since the last beat.
    async fn take_in(&self, a: &Arc<Authored>, held: &mut Held) {
        let shared = a.npcs.substrate();
        while let Ok((npc, event)) = held.heard.try_recv() {
            let Some(r) = held.running.get_mut(&npc) else {
                // A shard closed since: what it heard is the next holder's.
                continue;
            };
            let refused = match event {
                Event::Ready { user_id } => {
                    let bot = Online {
                        npc,
                        world: r.world.clone(),
                        token: r.token.clone(),
                        user_id,
                        since_ms: now_ms(),
                    };
                    r.online = Some(bot.clone());
                    let online = BotState {
                        state: State::Online,
                        since_ms: bot.since_ms,
                        world: bot.world.clone(),
                        connected_ms: r.connected_ms,
                    };
                    if let Err(e) = state::put(&shared, npc, &online) {
                        tracing::error!(npc, "discord: {e:#}");
                    }
                    let told = held.told.entry(npc).or_default();
                    gap::fetch(a, &bot, told).await.is_err()
                }
                Event::Message(m) => match r.online.clone() {
                    Some(bot) => {
                        let told = held.told.entry(npc).or_default();
                        match inbound::hear(a, &bot, told, &m).await {
                            Ok(heard) => {
                                tracing::debug!(npc, message = m.id, "discord: {heard:?}");
                                false
                            }
                            Err(Rejected) => true,
                        }
                    }
                    None => false,
                },
                Event::Refused => true,
                Event::Stopped(why) => {
                    let connected_ms = r.connected_ms;
                    held.running.remove(&npc);
                    held.halted.insert(npc, connected_ms);
                    if let Err(e) = state::remove(&shared, npc) {
                        tracing::error!(npc, "discord: {e:#}");
                    }
                    tracing::error!(npc, "discord: the bot is stopped: {why}");
                    false
                }
            };
            if refused {
                refuse(a, held, npc).await;
            }
        }
    }
}

/// Discord refused character `npc`'s bot's token: its shard stops, and its
/// state says so until a new token is connected.
async fn refuse(a: &Authored, held: &mut Held, npc: u64) {
    let Some(r) = held.running.remove(&npc) else {
        return;
    };
    r.shard.close().await;
    let refused = BotState {
        state: State::Refused,
        since_ms: now_ms(),
        world: r.world,
        connected_ms: r.connected_ms,
    };
    if let Err(e) = state::put(&a.npcs.substrate(), npc, &refused) {
        tracing::error!(npc, "discord: {e:#}");
    }
    tracing::warn!(
        npc,
        "discord: the bot's token was refused; connect a new one"
    );
}

/// Run the bots' beat until the daemon stops, then close every shard. Holds
/// the state weakly, so the task ends with it.
pub fn start(a: &Arc<Authored>) {
    let weak = Arc::downgrade(a);
    tokio::spawn(async move {
        let mut every = interval(BEAT);
        every.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            every.tick().await;
            let Some(a) = weak.upgrade() else {
                return;
            };
            if a.runtime.as_ref().is_some_and(|rt| rt.stopping()) {
                a.bots.close_all(&a).await;
                return;
            }
            a.bots.tick(&a).await;
        }
    });
}
