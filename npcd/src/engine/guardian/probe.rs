//! The guardian's window onto the engine.
//!
//! A trait, so the runner can be exercised against a scripted cast; the one
//! implementation that matters reads and steers the live runtime.

use std::future::Future;
use std::sync::Arc;

use crate::effector::router::address_of;
use crate::effector::station;
use crate::engine::event::{EventKind, Salience};
use crate::engine::guardian::view::{MissionView, Question, Reply, Station, Step};
use crate::engine::journal::state::Waiting;
use crate::engine::mission::{Mission, StepOutcome};
use crate::engine::runtime::Runtime;

/// How many of the scheduler's most recent ticks, across the whole cast, a
/// character's recent acts are read from.
const RECENT_TICKS: usize = 512;

pub trait Probe: Send + Sync {
    /// Every character in the cast.
    fn npcs(&self) -> Vec<u64>;

    /// Whether the engine is loaded and the cast is thinking.
    fn ready(&self) -> bool;

    fn stopping(&self) -> bool;

    /// The open mission a character carries, if any.
    fn mission(&self, npc_id: u64) -> Option<MissionView>;

    /// What a character can work where it stands.
    fn stations(&self, npc_id: u64) -> Vec<Station>;

    /// The last `limit` acts a character took, rendered, oldest first.
    fn recent_acts(&self, npc_id: u64, limit: usize) -> Vec<String>;

    /// The stretch of a character's life its journal does not cover yet, once
    /// there is enough of it to be worth asking about.
    fn journal(&self, npc_id: u64) -> Option<Waiting>;

    /// Put a question to a character; what it said.
    fn ask(
        &self,
        npc_id: u64,
        question: &Question,
    ) -> impl Future<Output = anyhow::Result<Reply>> + Send;

    /// Act on the character's answer about its journal: close the stretch when
    /// it is not worth an entry, start writing it up when it is. Whether a
    /// stretch was still there to act on.
    fn decide_journal(&self, npc_id: u64, worth: bool, reply: &Reply) -> bool;

    /// Put a thought into a character's head, as its own.
    fn think(&self, npc_id: u64, text: String) -> impl Future<Output = bool> + Send;

    /// Put an instruction in front of a character as something it was told.
    fn rouse(&self, npc_id: u64, text: String) -> impl Future<Output = bool> + Send;

    /// Sign a step off the open mission with how it turned out; whether one
    /// matched.
    fn tick_step(&self, npc_id: u64, step: &str, outcome: StepOutcome) -> bool;
}

/// A [`Probe`] over the live runtime.
pub struct RuntimeProbe(pub Arc<Runtime>);

fn view_of(mission: &Mission) -> MissionView {
    MissionView {
        prompt: mission.mission_text(),
        steps: mission
            .todo
            .iter()
            .map(|t| Step {
                text: t.text.clone(),
                done: t.done,
                reports: t.reports,
            })
            .collect(),
        standing: mission.standing_text(),
    }
}

impl Probe for RuntimeProbe {
    fn npcs(&self) -> Vec<u64> {
        self.0
            .scheduler
            .census()
            .into_iter()
            .map(|c| c.npc_id)
            .collect()
    }

    fn ready(&self) -> bool {
        self.0.is_ready()
    }

    fn stopping(&self) -> bool {
        self.0.stopping()
    }

    fn mission(&self, npc_id: u64) -> Option<MissionView> {
        let (hosted, body) = self.0.body_of(npc_id)?;
        hosted.sim(|s| s.missions.active(&body).map(view_of))
    }

    fn stations(&self, npc_id: u64) -> Vec<Station> {
        let Some((hosted, body)) = self.0.body_of(npc_id) else {
            return Vec::new();
        };
        hosted.read(|w| {
            let Some(at) = w.actor(&body).map(|a| a.at.clone()) else {
                return Vec::new();
            };
            w.map()
                .instances_at(&at)
                .into_iter()
                .filter_map(|inst| {
                    let address = address_of(&inst)?;
                    let verbs = station::verbs_at(inst.part_id())
                        .into_iter()
                        .map(|(verb, _)| verb)
                        .collect();
                    Some(Station {
                        name: inst.name().to_string(),
                        address,
                        verbs,
                    })
                })
                .collect()
        })
    }

    fn recent_acts(&self, npc_id: u64, limit: usize) -> Vec<String> {
        let acts: Vec<String> = self
            .0
            .scheduler
            .recent(RECENT_TICKS)
            .into_iter()
            .filter(|t| t.npc_id == npc_id)
            .flat_map(|t| t.acts)
            .collect();
        let skip = acts.len().saturating_sub(limit);
        acts.into_iter().skip(skip).collect()
    }

    fn journal(&self, npc_id: u64) -> Option<Waiting> {
        self.0.scheduler.journal_waiting(npc_id)
    }

    async fn ask(&self, npc_id: u64, question: &Question) -> anyhow::Result<Reply> {
        let answer = self
            .0
            .ask(npc_id, &question.text, &question.choices)
            .await?;
        Ok(Reply {
            answer: answer.answer,
            reason: answer.reason.unwrap_or_default(),
            ms: u64::try_from(answer.ms).unwrap_or(u64::MAX),
        })
    }

    fn decide_journal(&self, npc_id: u64, worth: bool, reply: &Reply) -> bool {
        self.0.decide_journal(npc_id, worth, reply)
    }

    async fn think(&self, npc_id: u64, text: String) -> bool {
        let world_ms = self.0.world_ms_async(npc_id).await;
        self.0.scheduler.deliver(
            npc_id,
            world_ms,
            Salience::URGENT,
            EventKind::MindControl { text },
        )
    }

    async fn rouse(&self, npc_id: u64, text: String) -> bool {
        let world_ms = self.0.world_ms_async(npc_id).await;
        self.0.scheduler.deliver(
            npc_id,
            world_ms,
            Salience::URGENT,
            EventKind::Nudge { text },
        )
    }

    fn tick_step(&self, npc_id: u64, step: &str, outcome: StepOutcome) -> bool {
        let Some((hosted, body)) = self.0.body_of(npc_id) else {
            return false;
        };
        hosted.with_sim(|s| s.missions.check_off(&body, step, outcome))
    }
}
