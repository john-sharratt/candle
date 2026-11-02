//! The order table from the side of the character that gives orders
//! (`docs/zen_moon_mind.md` §8): post a mission for those who take orders, and
//! read what came back.
//!
//! The acts are [`ORDER_POST`](crate::engine::mission_acts::ORDER_POST) and
//! [`ORDER_RESULTS`](crate::engine::mission_acts::ORDER_RESULTS), offered at the
//! table only to a character that takes no orders; the work they post is carried
//! by the others through `collect_mission` and `report_done`, as any mission is.

use serde_json::{Map, Value};

use crate::engine::body::Outcome;
use crate::engine::mission::{Mission, Origin, Outcome as Verdict, Todo};
use crate::world::Hosted;

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// Whether `body` is somebody a world gives orders to: a character's body —
/// not a visitor's — that takes them.
fn takes_orders(hosted: &Hosted, body: &str) -> bool {
    !body.starts_with("visitor:") && !hosted.sim(|s| s.missions.is_unordered(body))
}

/// `order_post`: put a mission on the table, for somebody by name or for
/// whoever comes for work first — its own brief and steps, or one of the
/// world's standing jobs whole.
///
/// **A mission has a step to do.** Posted with none, its only step was the
/// report back, so whoever took it was told at once that everything but the
/// report was done — and reported having done nothing.
pub fn post(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let job = text(args, "job");
    let (brief, steps) = match &job {
        Some(id) => match hosted.sim(|s| s.missions.routine(id).cloned()) {
            Some(r) => (r.brief, r.steps),
            None => {
                return Outcome::Refused(format!(
                    "There is no standing job called \"{id}\" to post."
                ))
            }
        },
        None => {
            let Some(brief) = text(args, "brief") else {
                return Outcome::Refused(
                    "You meant to post a mission, but did not say what is to be done.".into(),
                );
            };
            let steps: Vec<String> = text(args, "steps")
                .map(|s| {
                    s.lines()
                        .map(|l| l.trim().trim_start_matches(['-', '*']).trim().to_string())
                        .filter(|l| !l.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            if steps.is_empty() {
                return Outcome::Refused(
                    "Give it at least one step, one to a line, that whoever carries it can be \
                     seen to do — \"go to the ward room\", \"scan the sensor panel and read what \
                     state it is in\"."
                        .into(),
                );
            }
            (brief, steps)
        }
    };
    let steps: Vec<Todo> = steps.into_iter().map(Todo::new).collect();
    let (me, room) = hosted
        .read(|w| {
            let actor = w.actor(body)?;
            Some((actor.name.clone(), w.node(&actor.at)?.name.clone()))
        })
        .unwrap_or_default();
    let to = match text(args, "for") {
        None => None,
        Some(name) => {
            let found = hosted.read(|w| {
                w.actors()
                    .find(|a| a.name.eq_ignore_ascii_case(&name))
                    .map(|a| (a.id.clone(), a.name.clone()))
            });
            match found {
                Some((id, named)) if takes_orders(hosted, &id) && id != body => Some((id, named)),
                Some((_, named)) => {
                    return Outcome::Refused(format!(
                        "{named} takes no orders: there is nobody to set it for there."
                    ))
                }
                None => {
                    return Outcome::Refused(format!(
                        "There is nobody called \"{name}\" to give it to. {}",
                        who_takes_orders(hosted)
                    ))
                }
            }
        }
    };
    let mut todo = steps;
    todo.push(Todo::report(format!(
        "go back to the table in {room} and report it"
    )));
    let mission = Mission::new(
        brief.as_str(),
        todo,
        Origin::Ordered {
            by: me,
            poster: body.to_string(),
            job,
        },
    );
    let for_whom = match &to {
        Some((_, named)) => format!("for {named}"),
        None => "for whoever comes for work first".to_string(),
    };
    let id = to.as_ref().map(|(id, _)| id.as_str());
    hosted.with_sim(|s| s.missions.post(mission, id));
    Outcome::Did(format!(
        "Posted on the table, {for_whom}: {brief} You will be told when it is reported."
    ))
}

/// The characters who take orders here, named for a refusal that has to say
/// whom a mission could be for.
fn who_takes_orders(hosted: &Hosted) -> String {
    let names: Vec<String> = hosted
        .read(|w| {
            w.actors()
                .map(|a| (a.id.clone(), a.name.clone()))
                .collect::<Vec<_>>()
        })
        .into_iter()
        .filter(|(id, _)| takes_orders(hosted, id))
        .map(|(_, name)| name)
        .collect();
    match names.is_empty() {
        true => "Nobody here takes orders.".to_string(),
        false => format!("Those who take your orders: {}.", names.join(", ")),
    }
}

/// `order_results`: the reports on what `body` posted that it has not read,
/// newest first — read now — what is still out, and, with the table empty, the
/// standing jobs it may post.
pub fn results(hosted: &Hosted, body: &str) -> Outcome {
    let filed = hosted.with_sim(|s| s.missions.read_by(body));
    let (out, empty, jobs) = hosted.sim(|s| {
        (
            s.missions.out_for(body),
            s.missions.table_empty(),
            s.missions
                .routines()
                .iter()
                .map(|r| format!("- {}: {}", r.id, r.brief))
                .collect::<Vec<_>>(),
        )
    });
    let name = |b: &str| {
        hosted
            .read(|w| w.actor(b).map(|a| a.name.clone()))
            .unwrap_or_else(|| b.to_string())
    };
    let mut lines = Vec::new();
    match filed.is_empty() {
        true => lines.push("Nothing new has been reported.".to_string()),
        false => {
            lines.push("Reported since you last read, newest first:".to_string());
            for f in &filed {
                let how = match f.outcome {
                    Verdict::Pass => "done",
                    Verdict::Fail => "could not finish",
                };
                lines.push(format!(
                    "- {} {how}: {} — {}",
                    name(&f.carrier),
                    f.brief,
                    f.notes
                ));
            }
        }
    }
    if !out.is_empty() {
        lines.push("Still out:".to_string());
        for (brief, with) in &out {
            let state = match with {
                Some(b) => match hosted.sim(|s| s.missions.is_on_mission(b)) {
                    true => format!("carried by {}", name(b)),
                    false => format!("set for {}", name(b)),
                },
                None => "waiting to be taken".to_string(),
            };
            lines.push(format!("- {brief} — {state}"));
        }
    }
    if empty {
        lines.push("The table is empty.".to_string());
        if !jobs.is_empty() {
            lines.push("Standing jobs you may post:".to_string());
            lines.extend(jobs);
        }
    }
    Outcome::Did(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use npc_map::world::Where;
    use serde_json::json;

    use super::*;
    use crate::engine::mission::bank::Facts;
    use crate::engine::mission_gen::world::{Draw, Routine};

    fn vault() -> Hosted {
        let h = Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load");
        let room = Where::new("vault-command", "command-room");
        h.with(|w| {
            w.enter("zen", "Zen", room.clone()).unwrap();
            w.enter("wick", "Wick", room.clone()).unwrap();
            w.enter("jitter", "Jitter", room).unwrap();
        });
        h.with_sim(|s| s.missions.set_unordered("zen", true));
        h
    }

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    /// **A mission posted for whoever comes is on the table for the first to
    /// collect**, with its steps and the report back to the table it was
    /// posted at, and names its poster.
    #[test]
    fn a_post_for_whoever_comes_is_collected_by_the_first() {
        let h = vault();
        let posted = post(
            &h,
            "zen",
            &args(json!({
                "brief": "Read the breaker panel.",
                "steps": "go to the plant room\n- scan the breaker panel and read what state it is in\n"
            })),
        );
        assert_eq!(
            posted,
            Outcome::Did(
                "Posted on the table, for whoever comes for work first: Read the breaker panel. \
                 You will be told when it is reported."
                    .into()
            )
        );
        let took = h.with_sim(|s| s.missions.collect("wick", &Facts::default()).cloned());
        let m = took.unwrap();
        assert_eq!(m.prompt, "Read the breaker panel.");
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the plant room",
                "scan the breaker panel and read what state it is in",
                "go back to the table in the command room and report it"
            ]
        );
        assert_eq!(
            m.origin,
            Origin::Ordered {
                by: "Zen".into(),
                poster: "zen".into(),
                job: None,
            }
        );
    }

    /// **A mission posted for one by name is theirs alone**; nobody who takes
    /// no orders, nor anybody absent, can be named, and a mission with no step
    /// to do is not posted.
    #[test]
    fn a_post_for_one_by_name_is_theirs() {
        let h = vault();
        let watch =
            json!({"brief": "Watch Earth.", "steps": "go to the observatory", "for": "jitter"});
        let posted = post(&h, "zen", &args(watch));
        assert!(posted.happened(), "{posted:?}");
        assert!(!h.sim(|s| s.missions.has_work_for("wick")));
        assert!(h.sim(|s| s.missions.has_work_for("jitter")));
        assert_eq!(
            post(
                &h,
                "zen",
                &args(json!({"brief": "x", "steps": "go", "for": "Zen"}))
            ),
            Outcome::Refused("Zen takes no orders: there is nobody to set it for there.".into())
        );
        assert_eq!(
            post(
                &h,
                "zen",
                &args(json!({"brief": "x", "steps": "go", "for": "Umbra"}))
            ),
            Outcome::Refused(
                "There is nobody called \"Umbra\" to give it to. Those who take your orders: \
                 Jitter, Wick."
                    .into()
            )
        );
        assert!(!post(&h, "zen", &args(json!({"steps": "go"}))).happened());
        assert!(!post(&h, "zen", &args(json!({"brief": "Watch Earth."}))).happened());
        assert!(!post(&h, "zen", &args(json!({"brief": "x", "steps": "\n - \n"}))).happened());
    }

    fn ward_shift() -> Routine {
        Routine {
            id: "ward-shift".into(),
            brief: "Take a shift on the ward.".into(),
            steps: vec!["go to the ward room".into()],
            ..Default::default()
        }
    }

    /// **A standing job is posted whole**, by its name, and is a routine still:
    /// carrying it turns no chapter.
    #[test]
    fn a_standing_job_is_posted_whole() {
        let h = vault();
        h.with_sim(|s| s.missions.set_routines(vec![ward_shift()], Draw::None));
        assert!(post(&h, "zen", &args(json!({"job": "ward-shift"}))).happened());
        let m = h
            .with_sim(|s| s.missions.collect("wick", &Facts::default()).cloned())
            .unwrap();
        assert_eq!(m.prompt, "Take a shift on the ward.");
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the ward room",
                "go back to the table in the command room and report it"
            ]
        );
        h.with_sim(|s| {
            s.missions
                .report("wick", Verdict::Pass, "all quiet on the ward", None)
        });
        assert_eq!(h.sim(|s| s.missions.chapter("wick")), 0);
        assert_eq!(
            post(&h, "zen", &args(json!({"job": "hatchery"}))),
            Outcome::Refused("There is no standing job called \"hatchery\" to post.".into())
        );
    }

    /// **What came back is read at the table, once**: the reports since the
    /// last reading, newest first, what is still out and with whom — and, the
    /// table empty, the standing jobs.
    #[test]
    fn the_results_say_what_came_back_and_what_is_out() {
        let h = vault();
        h.with_sim(|s| s.missions.set_routines(vec![ward_shift()], Draw::None));
        assert_eq!(
            results(&h, "zen"),
            Outcome::Did(
                "Nothing new has been reported.\nThe table is empty.\nStanding jobs you may \
                 post:\n- ward-shift: Take a shift on the ward."
                    .into()
            )
        );
        post(
            &h,
            "zen",
            &args(json!({"brief": "Read the board.", "steps": "go to the dispatch room"})),
        );
        post(
            &h,
            "zen",
            &args(
                json!({"brief": "Watch Earth.", "steps": "go to the observatory", "for": "Jitter"}),
            ),
        );
        h.with_sim(|s| {
            s.missions.collect("wick", &Facts::default()).unwrap();
            s.missions.report(
                "wick",
                Verdict::Pass,
                "the board shows two jobs stalled",
                None,
            );
        });
        assert!(h.sim(|s| s.missions.unread_for("zen")));
        assert_eq!(
            results(&h, "zen"),
            Outcome::Did(
                "Reported since you last read, newest first:\n- Wick done: Read the board. — the \
                 board shows two jobs stalled\nStill out:\n- Watch Earth. — set for Jitter"
                    .into()
            )
        );
        assert!(!h.sim(|s| s.missions.unread_for("zen")), "read once");
    }

    /// **A posted mission that ends unreported comes back to its poster** —
    /// called off, or set for somebody who is gone — and is no longer out.
    #[test]
    fn a_post_called_off_or_left_by_the_gone_is_told_to_its_poster() {
        let h = vault();
        let go = |brief: &str, to: Option<&str>| {
            let mut a = json!({"brief": brief, "steps": "go to the dispatch room"});
            if let Some(to) = to {
                a["for"] = json!(to);
            }
            post(&h, "zen", &args(a));
        };
        go("Read the board.", None);
        go("Watch Earth.", Some("Jitter"));
        h.with_sim(|s| {
            s.missions.collect("wick", &Facts::default()).unwrap();
            assert!(s.missions.cancel("wick"));
            s.missions.forget("jitter");
            s.missions.take_notices();
        });
        assert!(h.sim(|s| s.missions.out_for("zen").is_empty()));
        assert!(h.sim(|s| s.missions.table_empty()));
        let Outcome::Did(read) = results(&h, "zen") else {
            panic!("a reading");
        };
        assert!(
            read.starts_with(
                "Reported since you last read, newest first:\n- Jitter could not finish: Watch \
                 Earth. — Whoever it was for is gone, and it was never done.\n- Wick could not \
                 finish: Read the board. — It was called off before it was done."
            ),
            "{read}"
        );
    }
}
