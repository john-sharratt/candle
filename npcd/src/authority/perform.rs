//! World writes made where they belong (`docs/replicated_substrate.md` §9.3):
//! in place when this daemon runs the world — its authority, or cut off from
//! it — and otherwise sent to the authority as `POST /v1/world/:wid/op`, which
//! makes them through [`Runtime::make`], the same path.

use std::sync::Arc;

use axum::http::Method;
use host::cluster::up::request_up;
use npc_map::world::Where;

use super::forward::{act_of, call_line};
use super::npc_lease::{hand_back, leases_of, released};
use super::ops::{Decided, OpReply, WorldOp};
use crate::api::now_ms;
use crate::engine::act::Act;
use crate::engine::body;
use crate::engine::event::{EventKind, Salience};
use crate::engine::mission::{Mission, StepOutcome};
use crate::engine::runtime::{Recorded, Runtime, REFUSED};

/// An act made, for the feed and the character, and whether it moved the
/// character's mission.
#[derive(Debug, Clone)]
pub struct Performed {
    pub recorded: Recorded,
    pub moved: bool,
}

/// What the world opens a character's turn with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Opening {
    /// What its arrival signed off of its mission, and what comes next.
    pub arrival: Option<String>,
    /// The mission compass, when it has changed since last told.
    pub compass: Option<String>,
    /// What the table it stands at hands it, every turn it stands there.
    pub summons: Option<String>,
}

impl Runtime {
    /// Make `op` in `world` and say what came of it, wherever the world runs.
    pub async fn world_write(&self, world: &str, op: WorldOp) -> anyhow::Result<OpReply> {
        if self.standing.writes(world) {
            return self.make(world, op, self.standing.node()).await;
        }
        let member = self.standing.member().ok_or_else(|| {
            anyhow::anyhow!("world `{world}` is run elsewhere and no cluster is joined")
        })?;
        let (status, body) = request_up(
            member,
            Method::POST,
            &format!("/v1/world/{world}/op"),
            serde_json::to_vec(&op)?,
        )
        .await?;
        if status != 200 {
            anyhow::bail!(
                "world `{world}`'s authority answered {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        Ok(serde_json::from_slice(&body)?)
    }

    /// [`Self::world_write`] for a beat that cannot wait on it: sent, and a
    /// failure logged.
    pub(crate) fn send_op(self: &Arc<Self>, world: &str, op: WorldOp) {
        let Some(handle) = self.spawner() else {
            return;
        };
        let rt = Arc::clone(self);
        let world = world.to_string();
        handle.spawn(async move {
            if let Err(e) = rt.world_write(&world, op).await {
                tracing::warn!("world `{world}`: a write did not reach its authority: {e:#}");
            }
        });
    }

    /// Make `op` in `world`, which this daemon runs; `from` is the member it
    /// came from.
    pub async fn make(
        &self,
        world: &str,
        op: WorldOp,
        from: Option<u64>,
    ) -> anyhow::Result<OpReply> {
        Ok(match op {
            // In this order: the compass reads where the mission stands once
            // the arrival has signed its steps off, and before the table hands
            // over the next.
            WorldOp::Begin { npc } => {
                let arrival = self.mark_arrival(npc);
                let compass = self.compass_news(npc);
                let summons = self.at_table_summons(npc);
                OpReply::Began {
                    arrival,
                    compass,
                    summons,
                }
            }
            WorldOp::Act {
                npc,
                call,
                decided,
                pinned_at,
            } => {
                let act = act_of(&call).map_err(|why| anyhow::anyhow!("{why}"))?;
                let decided = decided.and_then(|d| Some((Where::parse(&d.at)?, d.company)));
                let body = self.body_of(npc);
                if let (Some((hosted, body)), Some((at, company))) = (&body, decided) {
                    hosted.with(|w| w.restore_decision(body, at, company));
                }
                let pinned = pinned_at.as_deref().and_then(Where::parse);
                let done = self.perform_here(npc, &act, pinned.as_ref()).await;
                if let Some((hosted, body)) = &body {
                    hosted.with(|w| w.end_decision(body));
                }
                let r = done.recorded;
                OpReply::Performed {
                    feed: r.feed,
                    answer: r.answer,
                    landed: r.landed,
                    departing: r.departing,
                    moved: done.moved,
                }
            }
            WorldOp::Message { npc, from, text } => OpReply::Sent {
                sent: self.message_npc(npc, &from, &text),
            },
            WorldOp::EnterBeside { npc, visitor, name } => OpReply::Placed {
                at: self
                    .enter_world_beside(npc, &visitor, &name)
                    .map(|at| at.to_string()),
            },
            WorldOp::Leave { visitor } => OpReply::Done {
                ok: self.leave_world(&visitor),
            },
            WorldOp::Place { visitor, at } => {
                let ok = match (self.hosted.get(world), Where::parse(&at)) {
                    (Some(hosted), Some(at)) => hosted.with(|w| w.place(&visitor, at).is_ok()),
                    _ => false,
                };
                OpReply::Done { ok }
            }
            WorldOp::Lodge {
                npc,
                mission,
                start,
            } => OpReply::Lodged {
                now_text: self.lodge_mission(npc, *mission, start),
            },
            WorldOp::Step {
                npc,
                done,
                outcome,
                add,
            } => {
                let (ticked, added) =
                    self.step_mission(npc, done.as_deref(), outcome, add.as_deref());
                OpReply::Stepped { ticked, added }
            }
            WorldOp::Cancel { npc } => OpReply::Done {
                ok: self.cancel_mission(npc, self.world_ms(npc)),
            },
            WorldOp::Channel { from, text } => OpReply::Heard {
                by: self.say_on_channel(world, &from, &text),
            },
            WorldOp::Posting { at, on, by, text } => OpReply::Posted {
                lines: self.post_in_world(world, &at, &on, &by, &text),
            },
            // **Leases are the authority's alone to change** (§9.3): a member
            // running the world cut off answers neither, and the character
            // stays where it is until the world's authority takes it.
            WorldOp::HandBack { .. } | WorldOp::Release { .. }
                if !self.standing.role(world).is_authority() =>
            {
                OpReply::Done { ok: false }
            }
            WorldOp::HandBack { npc, through } => {
                let lease = from.and_then(|from| {
                    let current = leases_of(&self.standing.written(world));
                    hand_back(current.get(&npc)?, from, through, now_ms())
                });
                let ok = lease.is_some();
                if let Some(lease) = lease {
                    self.standing.handed_back(world, npc, lease);
                }
                OpReply::Done { ok }
            }
            WorldOp::Release { npc } => {
                self.standing.handed_back(world, npc, released());
                tracing::warn!(npc, "an admin freed this character's lease");
                OpReply::Done { ok: true }
            }
        })
    }

    /// Make one act of a character's turn in the world it stands in, and
    /// say what came of it — the feed's line, the character's, and whether its
    /// mission moved. The world may run elsewhere: then the act goes to its
    /// authority with the decision it was made in, and a world that does not
    /// answer is told to the character as a refusal.
    pub async fn perform(&self, npc: u64, act: &Act, pinned: Option<&Where>) -> Performed {
        let Some((hosted, body)) = self.body_of(npc) else {
            return self.perform_here(npc, act, pinned).await;
        };
        let world = hosted.id().to_string();
        // An act that happens inside a head touches no world, wherever the
        // world runs.
        let of_the_world =
            body::is_of_the_body(act.tool) || body::is_device(act.tool) || body::composes(act.tool);
        if self.standing.writes(&world) || !of_the_world {
            return self.perform_here(npc, act, pinned).await;
        }
        let decided = hosted
            .read(|w| w.deliberation(&body))
            .map(|(at, company)| Decided {
                at: at.to_string(),
                company,
            });
        let op = WorldOp::Act {
            npc,
            call: call_line(act),
            decided,
            pinned_at: pinned.map(|at| at.to_string()),
        };
        match self.world_write(&world, op).await {
            Ok(OpReply::Performed {
                feed,
                answer,
                landed,
                departing,
                moved,
            }) => {
                if landed {
                    self.cooldowns.took_act(npc, act);
                }
                Performed {
                    recorded: Recorded {
                        feed,
                        answer,
                        landed,
                        departing,
                    },
                    moved,
                }
            }
            Ok(other) => self.unanswered(act, &format!("the world answered {other:?}")),
            Err(e) => self.unanswered(act, &format!("{e:#}")),
        }
    }

    fn unanswered(&self, act: &Act, why: &str) -> Performed {
        tracing::warn!(tool = act.tool, "an act did not reach its world: {why}");
        Performed {
            recorded: Recorded {
                feed: format!("{} {REFUSED} the world did not answer", act.summary()),
                answer: "Nothing happened: the world did not answer. Try again in a moment."
                    .to_string(),
                landed: false,
                departing: false,
            },
            moved: false,
        }
    }

    /// One act made in a world this daemon runs.
    pub(crate) async fn perform_here(
        &self,
        npc: u64,
        act: &Act,
        pinned: Option<&Where>,
    ) -> Performed {
        let before = self.mission_mark(npc);
        let recorded = if body::is_device(act.tool) {
            self.enact_device(npc, act, pinned).await
        } else if body::composes(act.tool) {
            self.compose(npc, act).await
        } else {
            self.record_act(npc, act)
        };
        let moved = self.mission_mark(npc) != before;
        // A machine that refused is refused in front of the room, so a
        // colleague can see what went wrong rather than hear what it was taken
        // for.
        if !recorded.landed {
            self.show_refusal(npc, act, &recorded.answer);
        }
        Performed { recorded, moved }
    }

    /// What a character's turn opens with from the world: what its arrival
    /// signed off, the mission compass's news, and what the table it stands at
    /// hands it.
    pub async fn begin_turn(&self, npc: u64) -> Opening {
        let Some((hosted, _)) = self.body_of(npc) else {
            return Opening::default();
        };
        match self.world_write(hosted.id(), WorldOp::Begin { npc }).await {
            Ok(OpReply::Began {
                arrival,
                compass,
                summons,
            }) => Opening {
                arrival,
                compass,
                summons,
            },
            Ok(other) => {
                tracing::warn!(npc, "a turn's opening came back as {other:?}");
                Opening::default()
            }
            Err(e) => {
                tracing::warn!(npc, "a turn's opening did not reach its world: {e:#}");
                Opening::default()
            }
        }
    }

    /// Lodge a mission for a character, or put it in its hands at once when
    /// `start`; the standing task to rouse it with, when carried at once.
    pub(crate) fn lodge_mission(&self, npc: u64, mission: Mission, start: bool) -> Option<String> {
        let (hosted, character) = self.body_of(npc)?;
        let now_text = hosted.with_sim(|sim| {
            if start {
                sim.missions.assign(&character, mission);
                sim.missions.active(&character).map(|m| m.standing_text())
            } else {
                sim.missions.lodge(&character, mission);
                None
            }
        });
        if let Some(text) = &now_text {
            self.tell(
                npc,
                self.world_ms(npc),
                Salience::URGENT,
                EventKind::Nudge { text: text.clone() },
            );
        }
        now_text
    }

    /// Tick off `done` on a character's open mission, and add `add`.
    pub(crate) fn step_mission(
        &self,
        npc: u64,
        done: Option<&str>,
        outcome: Option<StepOutcome>,
        add: Option<&str>,
    ) -> (Option<bool>, Option<bool>) {
        let Some((hosted, character)) = self.body_of(npc) else {
            return (None, None);
        };
        hosted.with_sim(|sim| {
            let ticked = done.map(|step| {
                let outcome = outcome.unwrap_or(StepOutcome::Achieved);
                sim.missions.check_off(&character, step, outcome)
            });
            let added = add.map(|step| sim.missions.add_todo(&character, step));
            (ticked, added)
        })
    }
}
