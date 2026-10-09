//! `scan` with no place named: what in this room a character can work, and what it
//! carries, in prose.
//!
//! A survey is the station data compressed the way a character thinks: one line
//! per station — what it is, its state in a few words, and the exact address and
//! fields to `invoke` — generated from the one source the grammar's `invoke` set
//! is built from ([`station::verbs_at`], [`address_of`]), so the address a line
//! names is always one the grammar offers. The lines come from a [`Survey`], which
//! is also what the API serves, so what an operator reads is what the character
//! was told.

use npc_map::text;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::effector::router::address_of;
use crate::effector::station;
use crate::engine::body::Outcome;
use crate::engine::mission::{progress_line, Mission};
use crate::engine::mission_acts;
use crate::engine::on_you;
use crate::engine::tools::Tool;
use crate::world::Hosted;

/// Survey the room the body stands in and what it carries, as the character reads it.
///
/// A scan is also how a machine is read, so a mission step to read one that
/// stands here is signed off by it, and the character is told what to do next
/// beneath what it read.
pub fn here(hosted: &Hosted, body: &str) -> Outcome {
    let survey = Survey::of(hosted, body);
    let mut said = survey.prose();
    let seen: Vec<String> = survey.stations.iter().map(|e| e.name.clone()).collect();
    let room = hosted.read(|w| {
        let at = &w.actor(body)?.at;
        let level = w.map().get(&at.area).map(|a| a.name.clone())?;
        Some((w.node(at)?.name.clone(), level))
    });
    let next = hosted.with_sim(|s| {
        // A scan is made standing in the room, so it is there too.
        if let Some((room, level)) = &room {
            s.missions.arrived_in(body, room, level);
        }
        if !s.missions.read_off(body, &seen) {
            return None;
        }
        let next = s.missions.active(body).and_then(Mission::next_step);
        tracing::info!(npc = %body, "mission reading signed off by a scan");
        Some(progress_line("You have read it", next, &seen))
    });
    if let Some(line) = next {
        said.push('\n');
        said.push_str(&line);
    }
    Outcome::Did(said)
}

/// A parameter a station verb takes, as the character is told it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Need {
    pub name: &'static str,
    pub ty: &'static str,
    pub required: bool,
    pub about: &'static str,
}

/// One address a character can `invoke`, and what it takes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Verb {
    pub address: String,
    pub verb: String,
    pub needs: Vec<Need>,
}

/// One station in the room: what it is, how it stands, and how to work it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Entry {
    pub name: String,
    pub address: String,
    pub state: Vec<String>,
    pub verbs: Vec<Verb>,
    pub can: Vec<String>,
    pub reading: Map<String, Value>,
}

/// Everything a scan with no place named tells a character, before it is put into
/// words.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Survey {
    pub stations: Vec<Entry>,
    pub threads: Vec<String>,
    pub lately: Option<String>,
}

impl Survey {
    /// Survey the room the body stands in, and what it carries.
    pub fn of(hosted: &Hosted, body: &str) -> Survey {
        let (stations, threads, lately) = hosted.with_both(|world, sim| {
            let Some(actor) = world.actor(body) else {
                return (Vec::new(), Vec::new(), None);
            };
            let at = actor.at.clone();
            let me = actor.name.clone();
            let threads: Vec<String> = match sim.has_phone(body) {
                true => sim
                    .threads
                    .of(&me)
                    .into_iter()
                    .map(|t| on_you::thread_line(t, &me))
                    .collect(),
                false => Vec::new(),
            };
            let lately = on_you::lately(world, body);
            let place = format!("{}/{}", at.area, at.node);
            let who = |b: &str| {
                world
                    .actor(b)
                    .map_or_else(|| b.to_string(), |a| a.name.clone())
            };
            let stations = world
                .map()
                .instances_at(&at)
                .into_iter()
                .filter_map(|inst| match address_of(&inst) {
                    Some(url) => {
                        let mut verbs = station::verbs_at(inst.part_id());
                        // Only what this body can do here — see
                        // [`mission_acts::offered`].
                        let carrying = mission_acts::carrying(sim, body);
                        verbs.retain(|(_, t)| mission_acts::offered(t.name, carrying));
                        let acts: Vec<&str> = verbs.iter().map(|(_, t)| t.name).collect();
                        let read = sim.reading(&inst.id(), &place, &acts, &who);
                        Some(Entry::new(inst.name(), &url, &verbs, &read))
                    }
                    // **A machine with no controls still has a state to read.** A
                    // coolant valve or a breaker panel affords no act, so no route
                    // serves it and it has no address — but it is standing here in
                    // a state, and "go and read the coolant valve" is a mission. It
                    // is listed by the name the sim knows it by, with its reading
                    // and nothing to `invoke`.
                    None => {
                        let device = sim.station(&inst.id())?;
                        let read = sim.reading(&inst.id(), &place, &[], &who);
                        Some(Entry::new(&device.name, "", &[], &read))
                    }
                })
                .collect::<Vec<_>>();
            (stations, threads, lately)
        });
        Survey {
            stations,
            threads,
            lately,
        }
    }

    /// The whole survey as the character reads it.
    pub fn prose(&self) -> String {
        let lines: Vec<String> = self.stations.iter().map(Entry::prose).collect();
        let mut said = say(&lines);
        if let Some(mine) = on_you::say(&self.threads, self.lately.as_deref()) {
            said.push('\n');
            said.push_str(&mine);
        }
        said
    }
}

impl Entry {
    /// A station from its address, its verbs and its reading.
    pub fn new(
        name: &str,
        url: &str,
        verbs: &[(String, &'static Tool)],
        read: &Map<String, Value>,
    ) -> Entry {
        let can = read
            .get("tower")
            .and_then(|t| t["can"].as_array())
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Entry {
            name: name.to_string(),
            address: url.to_string(),
            state: gist(read),
            verbs: verbs
                .iter()
                .map(|(verb, tool)| Verb {
                    address: format!("{url}/{verb}"),
                    verb: verb.clone(),
                    needs: tool
                        .params
                        .iter()
                        .map(|p| Need {
                            name: p.name,
                            ty: p.ty,
                            required: p.required,
                            about: p.description,
                        })
                        .collect(),
                })
                .collect(),
            can,
            reading: read.clone(),
        }
    }

    /// One station: `Name: state. You can `invoke` addr (needs x, y); addr2.`
    pub fn prose(&self) -> String {
        let what = self
            .verbs
            .iter()
            .map(|v| {
                let needs: Vec<&str> = v
                    .needs
                    .iter()
                    .filter(|n| n.required)
                    .map(|n| n.name)
                    .collect();
                match needs.is_empty() {
                    true => format!("`{}`", v.address),
                    false => format!("`{}` (needs {})", v.address, needs.join(", ")),
                }
            })
            .collect::<Vec<_>>()
            .join("; ");
        let mut out = match self.state.is_empty() {
            true => format!("{}.", capital(&self.name)),
            false => format!("{}: {}.", capital(&self.name), self.state.join(", ")),
        };
        if self.verbs.is_empty() {
            // A machine with no address is worked with `operate`, not `invoke`;
            // say into what, from its own states, so the line is complete.
            let current = self.reading.get("mode").and_then(Value::as_str);
            let others: Vec<String> = self
                .reading
                .get("modes")
                .and_then(Value::as_array)
                .map(|m| {
                    m.iter()
                        .filter_map(Value::as_str)
                        .filter(|m| Some(*m) != current)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            if !others.is_empty() {
                out.push_str(&format!(
                    " It can be put to {} with `operate`.",
                    text::list_or(&others)
                ));
            }
            return out;
        }
        out.push_str(&format!(" You can `invoke` {what}"));
        if !self.can.is_empty() {
            out.push_str(&format!(" — the tower can {}", self.can.join(", ")));
        }
        out.push('.');
        out
    }
}

/// The station lines as the character reads them.
pub fn say(lines: &[String]) -> String {
    if lines.is_empty() {
        return "Nothing in this room answers to you. To see somewhere else, name a place.".into();
    }
    format!(
        "Here you can work:\n{}\nTo act on one, `invoke` its address.",
        lines
            .iter()
            .map(|l| format!("- {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

/// A station's reading in a few words.
fn gist(read: &Map<String, Value>) -> Vec<String> {
    let mut bits: Vec<String> = Vec::new();
    if let Some(tower) = read.get("tower") {
        match (tower["besieging"].as_str(), tower["posture"].as_str()) {
            (Some(target), _) => bits.push(format!("besieging {target}")),
            (None, Some(posture)) => bits.push(posture.replace('_', " ")),
            (None, None) => {}
        }
        bits.push(match tower["shields"].as_bool() {
            Some(true) => "shields up".into(),
            _ => "shields down".into(),
        });
        if let Some(minutes) = tower["contact"]["minutes_out"].as_u64() {
            bits.push(format!("a contact {minutes} minutes out"));
        }
        let open = tower["decisions"]
            .as_array()
            .map_or(0, |d| d.iter().filter(|d| d["held_by"].is_null()).count());
        if open > 0 {
            bits.push(format!(
                "{open} open decision{}",
                if open == 1 { "" } else { "s" }
            ));
        }
    }
    if let Some(mode) = read.get("mode").and_then(Value::as_str) {
        bits.push(mode.to_string());
        if read.get("working") == Some(&Value::Bool(false)) {
            bits.push("not working".into());
        }
    }
    if let Some(free) = read.get("free_queues").and_then(Value::as_array) {
        bits.push(format!(
            "{} queue{} free",
            free.len(),
            if free.len() == 1 { "" } else { "s" }
        ));
    }
    if let Some(queued) = read.get("queued").and_then(Value::as_array) {
        if !queued.is_empty() {
            bits.push(format!("{} batch running", queued.len()));
        }
    }
    bits
}

fn capital(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn verbs(part: &str) -> Vec<(String, &'static Tool)> {
        station::verbs_at(part)
    }

    fn read(v: Value) -> Map<String, Value> {
        v.as_object().cloned().expect("an object")
    }

    #[test]
    fn the_console_reads_as_one_prose_line_with_its_exact_address() {
        let r = read(json!({
            "tower": {
                "posture": "besieging",
                "besieging": "the eastern ledger",
                "shields": false,
                "can": ["relocate", "lift siege", "raise shields"],
                "contact": null,
                "decisions": [{ "what": "contact 1", "held_by": null }],
            }
        }));
        let said = Entry::new(
            "bridge console",
            "http://local/tower/bridge-console~0",
            &verbs("bridge-console"),
            &r,
        )
        .prose();
        assert_eq!(
            said,
            "Bridge console: besieging the eastern ledger, shields down, 1 open \
             decision. You can `invoke` `http://local/tower/bridge-console~0/command_tower` \
             (needs action) — the tower can relocate, lift siege, raise shields."
        );
    }

    #[test]
    fn an_entry_carries_every_parameter_with_its_meaning_for_the_operator() {
        let e = Entry::new(
            "bridge console",
            "http://local/tower/bridge-console~0",
            &verbs("bridge-console"),
            &Map::new(),
        );
        assert_eq!(e.address, "http://local/tower/bridge-console~0");
        let v = &e.verbs[0];
        assert_eq!(v.verb, "command_tower");
        assert_eq!(
            v.address,
            "http://local/tower/bridge-console~0/command_tower"
        );
        let action = v.needs.iter().find(|n| n.name == "action").unwrap();
        assert!(action.required);
        assert!(!action.about.is_empty());
        assert!(v.needs.iter().any(|n| !n.required), "{:?}", v.needs);
    }

    #[test]
    fn a_decision_somebody_holds_is_not_open() {
        let r = read(json!({
            "tower": { "posture": "standing", "shields": true, "can": [],
                "decisions": [{ "what": "x", "held_by": "Pax" }] }
        }));
        assert_eq!(gist(&r).join(", "), "standing, shields up");
    }

    #[test]
    fn a_fabricator_reads_its_mode_and_queues_but_not_everything_it_makes() {
        let r = read(json!({
            "mode": "idle", "working": true,
            "free_queues": ["1", "2"], "queued": [],
            "can_make": ["a", "b", "c", "d", "e"],
        }));
        assert_eq!(gist(&r).join(", "), "idle, 2 queues free");
    }

    #[test]
    fn a_machine_that_is_not_working_says_so() {
        let r = read(json!({ "mode": "idle", "working": false }));
        assert_eq!(gist(&r).join(", "), "idle, not working");
    }

    #[test]
    fn an_empty_reading_leaves_only_the_name_and_the_address() {
        let said = Entry::new(
            "terminal",
            "http://local/x/y",
            &verbs("fabricator"),
            &Map::new(),
        )
        .prose();
        assert!(
            said.starts_with("Terminal. You can `invoke` `http://local/x/y/"),
            "{said}"
        );
    }

    /// **A machine with no controls is read, not worked.** The plant room's
    /// coolant valve and breaker panel afford no act, and a mission to read one
    /// was impossible while the survey left them out: the character found only
    /// the plant panel and raised a fault on it instead.
    #[test]
    fn a_machine_with_no_controls_reads_its_state_and_offers_nothing_to_invoke() {
        let r = read(json!({ "mode": "open", "modes": ["open", "shut", "half"], "working": true }));
        assert_eq!(
            Entry::new("the coolant valve", "", &[], &r).prose(),
            "The coolant valve: open. It can be put to shut or half with `operate`."
        );
        let broken = read(json!({ "mode": "tripped", "modes": ["tripped"], "working": false }));
        assert_eq!(
            Entry::new("the breaker panel", "", &[], &broken).prose(),
            "The breaker panel: tripped, not working."
        );
    }

    #[test]
    fn an_empty_room_says_where_to_look_instead() {
        assert_eq!(
            say(&[]),
            "Nothing in this room answers to you. To see somewhere else, name a place."
        );
    }

    #[test]
    fn the_survey_lists_each_station_and_closes_on_how_to_act() {
        let said = say(&["One.".into(), "Two.".into()]);
        assert_eq!(
            said,
            "Here you can work:\n- One.\n- Two.\nTo act on one, `invoke` its address."
        );
    }
}
