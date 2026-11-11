//! A conversation's mission acts, as the world's authority applies them
//! (`docs/npc_conversations.md` §7.1, §7.2).
//!
//! **Checked again where they land.** The lease holder offered the acts
//! because the party had standing when the turn began; the authority asks
//! again — the party's account against the character's owner and the world's
//! authorizers, and the thrash cooldown — because the owner, the authorizers
//! and the last change are the authority's to know, and a turn decided on
//! another member may land after any of them moved.
//!
//! **An ordered body's acts are its own** ([`crate::engine::agenda::ordered`]):
//! `mission_add` gives it the mission at once as lodged by the party's
//! account, calling off what it carried; `mission_edit` keeps every done and
//! critical step; `mission_cancel` withdraws it; `mission_done` is the
//! `report_done` handler, so every gate a report passes applies — a critical
//! step still unsigned among them.

use serde_json::{Map, Value};

use crate::authority::ops::{OpReply, Voiced};
use crate::engine::act::Act;
use crate::engine::agenda::ordered::{self, Order};
use crate::engine::body::{self, Outcome};
use crate::engine::experience::concern::Change;
use crate::engine::experience::line::first_sentence;
use crate::engine::mission::{Mission, Origin, Todo};
use crate::engine::runtime::Runtime;
use crate::engine::talk::authority::{self, Standing};
use crate::engine::talk::key::Party;
use crate::engine::work;
use crate::world::Hosted;

/// Why a party with no standing over the mission is refused it.
pub const NO_STANDING: &str =
    "Only whoever owns you, or an authorizer of this world, can change your mission.";

/// Why a change inside the thrash cooldown is refused.
pub const COOLING: &str =
    "Your mission was changed through a conversation a moment ago; it stays as it is for now.";

impl Runtime {
    /// Apply mission act `act` from `npc_id`'s conversation with `member`,
    /// where the world is written.
    pub(crate) fn talk_mission_act(
        &self,
        npc_id: u64,
        hosted: &Hosted,
        body: &str,
        act: &Act,
        member: &str,
        said: Option<&Voiced>,
    ) -> Outcome {
        let party = Party::of_member(member, |b| self.bodies.mind_of(hosted.id(), b));
        let owner = self
            .npcs()
            .and_then(|n| n.view().payload(npc_id).map(|p| p.owner_id.clone()));
        let authorizers = hosted.sim(|s| s.missions.authorizers().to_vec());
        let standing = authority::standing(&party, owner.as_deref(), &authorizers);
        let world_ms = self.world_ms(npc_id);
        if standing == Standing::Nobody {
            return Outcome::Refused(NO_STANDING.into());
        }
        if !authority::may_change(standing, self.talk_changes.last(npc_id), world_ms) {
            return Outcome::Refused(COOLING.into());
        }
        let done = match hosted.sim(|s| s.missions.is_unordered(body)) {
            true => body::perform_said(hosted, body, act, said),
            false => match (&party, ordered::order(act.tool, &act.args)) {
                (Party::Person(account), Ok(order)) => {
                    let name = hosted.sim(|s| s.threads.name_of(member).map(str::to_string));
                    let name = name.as_deref().unwrap_or(account);
                    self.ordered_act(npc_id, hosted, body, order, account, name)
                }
                (_, Err(why)) => Outcome::Refused(why),
                (Party::Npc(_), Ok(_)) => Outcome::Refused(NO_STANDING.into()),
            },
        };
        if done.happened() {
            self.talk_changes.mark(npc_id, world_ms);
        }
        done
    }

    /// An ordered body's mission changed by `account`, whom it calls `name`.
    fn ordered_act(
        &self,
        npc_id: u64,
        hosted: &Hosted,
        body: &str,
        order: Order,
        account: &str,
        name: &str,
    ) -> Outcome {
        let carries = |mission: &str| {
            hosted.sim(|s| {
                s.missions
                    .active(body)
                    .is_some_and(|m| m.mission_text().eq_ignore_ascii_case(mission.trim()))
            })
        };
        match order {
            Order::Add { brief, steps } => {
                let mission = Mission::new(
                    brief.clone(),
                    steps.into_iter().map(Todo::new).collect(),
                    Origin::Lodged {
                        by: account.to_string(),
                    },
                );
                match self.lodge_mission(npc_id, mission, true) {
                    OpReply::Lodged { refused: false, .. } => Outcome::Did(format!(
                        "Your mission now, from {name}: {}",
                        first_sentence(&brief)
                    )),
                    _ => Outcome::Refused("The mission could not be given to you.".into()),
                }
            }
            Order::Edit {
                mission,
                brief,
                steps,
            } => {
                if !carries(&mission) {
                    return Outcome::Refused(format!("You carry no mission called {mission}."));
                }
                let edited = hosted.with_sim(|s| s.missions.edit_active(body, brief, steps));
                match edited {
                    Ok(()) => {
                        let now = hosted
                            .sim(|s| s.missions.active(body).map(|m| m.mission_text()))
                            .unwrap_or(mission);
                        let line = format!("{name} changed your mission: {}", first_sentence(&now));
                        self.changed(
                            hosted,
                            Change::Mission {
                                body: body.to_string(),
                                line: line.clone(),
                            },
                        );
                        Outcome::Did(line)
                    }
                    Err(why) => Outcome::Refused(why),
                }
            }
            Order::Cancel { mission, why } => {
                if !carries(&mission) {
                    return Outcome::Refused(format!("You carry no mission called {mission}."));
                }
                match self.cancel_mission(npc_id, self.world_ms(npc_id)) {
                    true => Outcome::Did(format!(
                        "{name} called off your mission: {}. {why}",
                        first_sentence(&mission)
                    )),
                    false => Outcome::Refused("You carry no mission to call off.".into()),
                }
            }
            Order::Done { mission, found } => {
                if !carries(&mission) {
                    return Outcome::Refused(format!("You carry no mission called {mission}."));
                }
                let mut args = Map::new();
                args.insert("found".into(), Value::String(found));
                work::perform(
                    hosted,
                    body,
                    &Act {
                        tool: "report_done",
                        args,
                    },
                )
            }
        }
    }
}
