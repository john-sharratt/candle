//! The house's mode, set where the world runs (`docs/house_modes.md` §3):
//! awaited, because a switch waits on the decks and on each device.

use crate::engine::act::Act;
use crate::engine::runtime::{Recorded, Runtime, LANDED, REFUSED};
use crate::house::switch::switch;
use crate::house::Mode;

impl Runtime {
    /// `house_mode`, made for `npc` from where it stands.
    pub(crate) async fn house_act(&self, npc: u64, act: &Act) -> Recorded {
        let Some((hosted, body)) = self.body_of(npc) else {
            return recorded(act, false, "You have no body here to do that with.".into());
        };
        let asked = act.args.get("mode").and_then(|v| v.as_str()).unwrap_or("");
        let Some(mode) = Mode::parse(asked) else {
            return recorded(
                act,
                false,
                format!("`{asked}` is no mode of the house: away, asleep, home, listening, watching or party."),
            );
        };
        let member = self.standing.member().cloned();
        let (landed, answer) = switch(&hosted, &body, mode, self.agents(), member).await;
        if landed {
            hosted.with_sim(|s| {
                s.missions
                    .observe(&body, answer.lines().next().unwrap_or(""))
            });
        }
        recorded(act, landed, answer)
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
