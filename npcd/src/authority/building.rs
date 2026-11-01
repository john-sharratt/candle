//! The bridge and building acts, enacted where the world runs
//! (`docs/zen_moon_mind.md` §13–§15): awaited, because a bridge reaches the
//! network and a building act keeps what it changes.

use crate::bridge;
use crate::effector::reshape::keep_for;
use crate::engine::act::Act;
use crate::engine::build;
use crate::engine::build::edit::moved;
use crate::engine::mission::{Origin, StepOutcome};
use crate::engine::runtime::{Recorded, Runtime, LANDED, REFUSED};
use crate::world::Hosted;

impl Runtime {
    /// A bridge act — the survey, a Sonos player — made for `npc` in the world
    /// it stands in.
    pub(crate) async fn bridge_act(&self, npc: u64, act: &Act) -> Recorded {
        let Some((hosted, body)) = self.body_of(npc) else {
            return refused(act, "You have no body here to do that with.");
        };
        let place = hosted.standpoint_of(&body);
        let done = bridge::enact(&hosted, &body, act.tool, &act.args, self.agents()).await;
        if done.landed {
            hosted.with_sim(|s| {
                s.missions
                    .observe(&body, done.answer.lines().next().unwrap_or(""))
            });
            if act.tool == "survey_network" {
                self.relink(&hosted, npc, &place).await;
                sign_off(&hosted, &body, SURVEY_STEP);
                // A survey that found nothing the world lacks leaves nothing
                // to put in the room.
                if build::open_findings(&hosted, &place).is_empty() {
                    sign_off(&hosted, &body, ONBOARD_STEP);
                }
            }
        }
        recorded(act, done.landed, done.answer)
    }

    /// **A device that moved is followed**: each part standing for a device
    /// the survey of `place` found at a new address is linked there, so the
    /// world reaches it where it is now.
    async fn relink(&self, hosted: &Hosted, npc: u64, place: &str) {
        let found = hosted.sim(|s| s.findings.of(place).to_vec());
        let edits = hosted.read(|w| moved(w.map(), &found));
        if edits.is_empty() {
            return;
        }
        let keep = keep_for(self, hosted, npc).await;
        match hosted.build(&edits, keep) {
            Ok(_) => tracing::info!(
                world = hosted.id(),
                devices = edits.len(),
                "devices that moved were linked where they are"
            ),
            Err(why) => {
                tracing::warn!(world = hosted.id(), %why, "devices that moved could not be linked again")
            }
        }
    }

    /// A building act made for `npc`: planned against the map, laid into the
    /// world whole or not at all, kept in its state and drafted into the
    /// mind's map under its owner.
    pub(crate) async fn build_act(&self, npc: u64, act: &Act) -> Recorded {
        let Some((hosted, body)) = self.body_of(npc) else {
            return refused(act, "You have no body here to do that with.");
        };
        let (edits, line) = match build::plan(&hosted, &body, act.tool, &act.args) {
            Ok(planned) => planned,
            Err(why) => return refused(act, &why),
        };
        let keep = keep_for(self, &hosted, npc).await;
        if let Err(why) = hosted.build(&edits, keep) {
            tracing::warn!(world = hosted.id(), npc, ?edits, %why, "a building act was refused by the map");
            return refused(act, &format!("The room would not take it: {why}"));
        }
        tracing::info!(
            world = hosted.id(),
            npc,
            act = act.tool,
            edits = edits.len(),
            "the world was built"
        );
        hosted.with_sim(|s| s.missions.observe(&body, &line));
        // Every device found is in the world — once there has been a survey to
        // find them.
        let place = hosted.standpoint_of(&body);
        if hosted.sim(|s| s.findings.surveyed_in(&place))
            && build::open_findings(&hosted, &place).is_empty()
        {
            sign_off(&hosted, &body, ONBOARD_STEP);
        }
        recorded(act, true, line)
    }
}

/// A standing mission's step the survey signs off: one beginning thus.
pub const SURVEY_STEP: &str = "survey";

/// A standing mission's step signed off once every device the room's survey
/// found stands in the world: one beginning thus.
pub const ONBOARD_STEP: &str = "put each device found";

/// Sign off the first open step of `body`'s standing mission beginning with
/// `starts` — a standing mission's only: another's steps are its carrier's to
/// report, however they are worded.
fn sign_off(hosted: &Hosted, body: &str, starts: &str) {
    let step = hosted.sim(|s| {
        s.missions.active(body).and_then(|m| {
            if !matches!(m.origin, Origin::Standing { .. }) {
                return None;
            }
            m.todo
                .iter()
                .find(|t| !t.done && t.text.trim_start().starts_with(starts))
                .map(|t| t.text.clone())
        })
    });
    if let Some(step) = step {
        hosted.with_sim(|s| s.missions.check_off(body, &step, StepOutcome::Achieved));
    }
}

fn recorded(act: &Act, landed: bool, answer: String) -> Recorded {
    let first = answer.lines().next().unwrap_or_default().to_string();
    Recorded {
        feed: match landed {
            true => format!("{} {LANDED} {first}", act.summary()),
            false => format!("{} {REFUSED} {first}", act.summary()),
        },
        answer,
        landed,
        departing: false,
    }
}

fn refused(act: &Act, why: &str) -> Recorded {
    recorded(act, false, why.to_string())
}
