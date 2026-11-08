//! The round (`docs/zen_moon_mind.md` §17.3): what the order table gives a body
//! that takes no orders when its agenda is empty, one item at a time, in order,
//! for ever.
//!
//! One file, `rounds/<world>.yaml`. Each item is a mission's brief and steps, with when it is due:
//! only while a condition of the world holds (`when`), and not again until a
//! period has passed since it was last done (`every`). The item given is the
//! first due one after the one given last, so an item just done is given again
//! only when nothing else is due — and with nothing due, nothing is given.

use std::collections::BTreeSet;

use serde::Deserialize;

use crate::engine::mission::{Mission, Origin, Todo};
use crate::engine::mission_gen::standing::{period, RETRY_MS};
use crate::mind::files::MindFiles;
use crate::sim::round::RoundState;
use crate::sim::Sim;

/// Where world `world`'s round stands in the mind: a file of its own, apart
/// from the world's template library (`missions/<world>/`) and its standing
/// missions (`standing/<world>/`).
pub fn path(world: &str) -> String {
    format!("rounds/{world}.yaml")
}

/// How long a round mission may be carried without every step signed off
/// before it is called off as failed: an hour, so a step that cannot be done
/// holds the round up for that long and no longer.
pub const STALE_MS: u64 = 3_600_000;

/// What must be true of the world for an item to be due.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum When {
    /// Always.
    Always,
    /// A report on what the body posted waits unread.
    Reports,
    /// Nothing waits on the order table.
    TableEmpty,
}

/// What holds of the world for one body, as the round's `when` asks it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Holds {
    pub reports: bool,
    pub table_empty: bool,
}

impl Holds {
    /// What holds for `body` in `sim`.
    pub fn of(sim: &Sim, body: &str) -> Holds {
        Holds {
            reports: sim.missions.unread_for(body),
            table_empty: sim.missions.table_empty(),
        }
    }
}

impl When {
    fn holds(self, h: Holds) -> bool {
        match self {
            When::Always => true,
            When::Reports => h.reports,
            When::TableEmpty => h.table_empty,
        }
    }
}

/// One item of the round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub id: String,
    pub brief: String,
    pub when: When,
    /// The least time between one doing of it and the next.
    pub every: Option<u64>,
    pub steps: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    round: Vec<ItemFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ItemFile {
    id: String,
    brief: String,
    #[serde(default)]
    when: Option<String>,
    #[serde(default)]
    every: Option<String>,
    steps: Vec<String>,
}

/// Parse and check a round's file, whole: one fault refuses it all.
pub fn parse(file: &str, yaml: &str) -> Result<Vec<Item>, String> {
    let f: File = serde_yaml::from_str(yaml).map_err(|e| format!("{file}: {e}"))?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for i in f.round {
        let kebab = !i.id.is_empty()
            && i.id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !kebab {
            return Err(format!("{file}: the id `{}` is not kebab-case", i.id));
        }
        if !seen.insert(i.id.clone()) {
            return Err(format!("{file}: `{}` is in the round twice", i.id));
        }
        if i.brief.trim().is_empty()
            || i.steps.is_empty()
            || i.steps.iter().any(|s| s.trim().is_empty())
        {
            return Err(format!(
                "{file}: `{}` needs a brief and at least one step",
                i.id
            ));
        }
        let when = match i.when.as_deref().map(str::trim) {
            None => When::Always,
            Some("reports") => When::Reports,
            Some("table-empty") => When::TableEmpty,
            Some(other) => {
                return Err(format!(
                    "{file}: `{}` is due `when: {other}`, which is neither `reports` nor \
                     `table-empty`",
                    i.id
                ))
            }
        };
        let every = match &i.every {
            None => None,
            Some(p) => Some(period(p).ok_or(format!(
                "{file}: `{}`'s `every` is a period — `30m`, `2h`, `1d`",
                i.id
            ))?),
        };
        out.push(Item {
            id: i.id,
            brief: i.brief.trim().to_string(),
            when,
            every,
            steps: i.steps.into_iter().map(|s| s.trim().to_string()).collect(),
        });
    }
    Ok(out)
}

/// World `world`'s round, from `rounds/<world>.yaml` — empty when it has
/// none.
pub fn load(mind: &MindFiles, world: &str) -> Result<Vec<Item>, String> {
    let file = path(world);
    match mind.read(&file) {
        Some(text) => parse(&file, &text),
        None if mind.is_file(&file) => Err(format!("{file} is not UTF-8")),
        None => Ok(Vec::new()),
    }
}

impl Item {
    /// Whether this item is due at `now_ms`: its `when` holds, its `every`
    /// has passed since it was last done, and it is not resting from a
    /// failure.
    fn due(&self, state: &RoundState, holds: Holds, now_ms: u64) -> bool {
        let ready = match state.done_at(&self.id) {
            None => true,
            // Done just now, not yet stamped.
            Some(None) => false,
            Some(Some(at)) => self.every.is_none_or(|p| now_ms >= at.saturating_add(p)),
        };
        let resting = match state.failed_at(&self.id) {
            None => false,
            Some(None) => true,
            Some(Some(at)) => now_ms < at.saturating_add(RETRY_MS),
        };
        self.when.holds(holds) && ready && !resting
    }

    /// The mission this item sets.
    pub fn mission(&self) -> Mission {
        Mission::new(
            self.brief.clone(),
            self.steps.iter().map(Todo::new).collect(),
            Origin::Round {
                id: self.id.clone(),
            },
        )
    }
}

impl Item {
    /// What the item is named by on an agenda: its id, in words.
    pub fn title(&self) -> String {
        self.id.replace('-', " ")
    }
}

/// Whether the round has an item due for `body` at `now_ms` — what offers it
/// `collect_mission` with an empty agenda, and what wakes it.
pub fn due_for(sim: &Sim, body: &str, now_ms: u64) -> bool {
    next(
        sim.missions.round_items(),
        sim.missions.round(),
        Holds::of(sim, body),
        now_ms,
    )
    .is_some()
}

/// Give `body` the round's next due item at `now_ms`, on its agenda at
/// `later`. Returns the mission given, or `None` when nothing is due.
pub fn collect(sim: &mut Sim, body: &str, now_ms: u64) -> Option<Mission> {
    let holds = Holds::of(sim, body);
    let item = next(
        sim.missions.round_items(),
        sim.missions.round(),
        holds,
        now_ms,
    )?
    .clone();
    sim.missions
        .take_round(body, item.mission(), &item.title(), now_ms);
    sim.missions.active(body).cloned()
}

/// The next item to set, or `None` when nothing is due: the first due item
/// after the one set last, going round from the start.
pub fn next<'a>(
    items: &'a [Item],
    state: &RoundState,
    holds: Holds,
    now_ms: u64,
) -> Option<&'a Item> {
    let from = state
        .last()
        .and_then(|last| items.iter().position(|i| i.id == last))
        .map_or(0, |at| at + 1);
    (0..items.len())
        .map(|k| &items[(from + k) % items.len()])
        .find(|i| i.due(state, holds, now_ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission::Outcome;
    use crate::sim::agenda::Priority;

    const ROUND: &str = "round:\n  - id: read-the-reports\n    when: reports\n    brief: Read what came back.\n    steps:\n      - read what came back at the order table\n  - id: set-the-work\n    when: table-empty\n    brief: Set the work.\n    steps:\n      - post work on the order table\n  - id: watch-earth\n    every: 2h\n    brief: Watch Earth.\n    steps:\n      - go to the watch room\n      - scan the sensor panel and read what state it is in\n";

    fn round() -> Vec<Item> {
        parse("missions/life/round.yaml", ROUND).unwrap()
    }

    const ALL: Holds = Holds {
        reports: true,
        table_empty: true,
    };

    /// **A round reads as its items, in order**, each with when it is due and
    /// its steps; a round that names an unknown `when`, a bad period, an item
    /// twice, or an item with no step is refused whole.
    #[test]
    fn a_round_reads_and_is_checked() {
        let r = round();
        let ids: Vec<&str> = r.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["read-the-reports", "set-the-work", "watch-earth"]);
        assert_eq!(
            (r[0].when, r[1].when, r[2].when),
            (When::Reports, When::TableEmpty, When::Always)
        );
        assert_eq!(r[2].every, Some(7_200_000));
        let m = r[2].mission();
        assert_eq!(m.prompt, "Watch Earth.");
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the watch room",
                "scan the sensor panel and read what state it is in"
            ],
            "no report step: the engine closes it"
        );
        assert_eq!(
            m.origin,
            Origin::Round {
                id: "watch-earth".into()
            }
        );
        for (bad, why) in [
            (ROUND.replace("when: reports", "when: often"), "neither"),
            (ROUND.replace("when: reports", "when: messages"), "neither"),
            (ROUND.replace("every: 2h", "every: soon"), "is a period"),
            (
                ROUND.replace("id: set-the-work", "id: read-the-reports"),
                "twice",
            ),
            (ROUND.replace("id: watch-earth", "id: Watch"), "kebab-case"),
            (
                ROUND.replace(
                    "    steps:\n      - post work on the order table\n",
                    "    steps: []\n",
                ),
                "at least one step",
            ),
        ] {
            let e = parse("f", &bad).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
    }

    /// **A round is read from its own file**, `rounds/<world>.yaml`, apart from
    /// the world's template library and its standing missions: a world with
    /// none has an empty round, and a bad one is refused naming the file.
    #[test]
    fn a_round_is_read_from_its_own_file() {
        let tmp = tempfile::tempdir().unwrap();
        let mind = MindFiles::folder(tmp.path());
        assert_eq!(load(&mind, "life"), Ok(Vec::new()));
        std::fs::create_dir_all(tmp.path().join("rounds")).unwrap();
        std::fs::write(tmp.path().join(path("life")), ROUND).unwrap();
        assert_eq!(load(&mind, "life").unwrap(), round());
        std::fs::write(tmp.path().join(path("life")), "round: [x]\n").unwrap();
        assert!(load(&mind, "life")
            .unwrap_err()
            .starts_with("rounds/life.yaml: "));
    }

    /// **In order, from where it left off, and never the same twice running
    /// while another is due**: each set goes on to the next due item, round
    /// from the start; an item whose `when` does not hold is passed over; with
    /// only one due, it is set again.
    #[test]
    fn the_round_goes_on_from_the_item_set_last() {
        let r = round();
        let mut s = RoundState::default();
        let id = |s: &RoundState, h: Holds| next(&r, s, h, 0).map(|i| i.id.clone());
        assert_eq!(id(&s, ALL).as_deref(), Some("read-the-reports"));
        s.set("zen", "read-the-reports", 0);
        assert_eq!(id(&s, ALL).as_deref(), Some("set-the-work"));
        s.set("zen", "set-the-work", 0);
        assert_eq!(id(&s, ALL).as_deref(), Some("watch-earth"));
        s.set("zen", "watch-earth", 0);
        assert_eq!(
            id(&s, ALL).as_deref(),
            Some("read-the-reports"),
            "round again"
        );

        let quiet = Holds::default();
        assert_eq!(
            id(&s, quiet).as_deref(),
            Some("watch-earth"),
            "only one holds"
        );
        s.set("zen", "watch-earth", 0);
        assert_eq!(
            id(&s, quiet).as_deref(),
            Some("watch-earth"),
            "the only one due is set again"
        );
    }

    /// **An item is not due again until its `every` has passed since it was
    /// done, nor while a failure rests** — and with nothing due, nothing is
    /// set.
    #[test]
    fn an_item_rests_after_it_is_done_or_fails() {
        let r = round();
        let quiet = Holds::default();
        let mut s = RoundState::default();
        s.set("zen", "watch-earth", 0);
        s.closed("zen", "watch-earth", Outcome::Pass);
        assert!(next(&r, &s, quiet, 0).is_none(), "done, not yet stamped");
        s.stamp(1_000);
        assert!(
            next(&r, &s, quiet, 1_000 + 7_200_000 - 1).is_none(),
            "resting"
        );
        assert!(
            next(&r, &s, quiet, 1_000 + 7_200_000).is_some(),
            "due again"
        );

        s.set("zen", "watch-earth", 0);
        s.closed("zen", "watch-earth", Outcome::Fail);
        s.stamp(10_000_000);
        assert!(
            next(&r, &s, quiet, 10_000_000 + RETRY_MS - 1).is_none(),
            "failure rests"
        );
        assert!(next(&r, &s, quiet, 10_000_000 + RETRY_MS).is_some());
        assert!(next(&[], &s, ALL, 0).is_none(), "no round, nothing set");
    }

    /// **Zen's round, end to end in the world**: collected at the table, the
    /// first due item — the table is empty, so setting the work — comes onto
    /// Zen's agenda at `later` under its title; the post it makes signs the
    /// step off as it lands; the engine closes the mission done, with no
    /// report and no chapter turned; and the next item collected is the one
    /// after it.
    #[test]
    fn zen_collects_its_round_and_the_engine_closes_what_it_saw_done() {
        use serde_json::json;

        use crate::engine::order_table;
        use crate::world::Hosted;

        let h = Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load");
        let room = npc_map::world::Where::new("vault-command", "command-room");
        h.with(|w| {
            w.enter("zen", "Zen", room.clone()).unwrap();
            w.enter("wick", "Wick", room).unwrap();
        });
        h.with_sim(|s| {
            s.missions.set_unordered("zen", true);
            s.missions.set_round(round());
        });

        assert!(h.sim(|s| due_for(s, "zen", 0)));
        let got = h.with_sim(|s| collect(s, "zen", 0)).unwrap();
        assert_eq!(got.prompt, "Set the work.");
        assert_eq!(
            got.agendum.as_ref().map(|a| (a.title.as_str(), a.priority)),
            Some(("set the work", Priority::Later))
        );

        let post = json!({"brief": "Read the board.", "steps": "go to the dispatch room"});
        assert!(order_table::post(&h, "zen", post.as_object().unwrap()).happened());
        assert!(h.sim(|s| s
            .missions
            .active("zen")
            .is_some_and(Mission::all_todos_done)));

        let closed = h.with_sim(|s| s.missions.settle_agendas(0));
        assert_eq!(
            closed,
            [("zen".to_string(), "set the work".to_string(), Outcome::Pass)]
        );
        assert!(!h.sim(|s| s.missions.is_on_mission("zen")));
        assert_eq!(
            h.sim(|s| s.missions.chapter("zen")),
            0,
            "a routine, not a chapter"
        );

        h.with_sim(|s| s.missions.stamp_round(1));
        let got = h.with_sim(|s| collect(s, "zen", 1)).unwrap();
        assert_eq!(got.prompt, "Watch Earth.");
    }

    /// **A round mission carried an hour without every step done is called
    /// off as failed**, and rests as a failure does.
    #[test]
    fn a_round_mission_left_undone_an_hour_is_called_off() {
        use crate::sim::missions::Missions;

        let mut m = Missions::default();
        m.set_unordered("zen", true);
        m.set_round(round());
        m.take_round("zen", round()[2].mission(), "watch earth", 0);
        assert!(m.settle_agendas(STALE_MS - 1).is_empty());
        let closed = m.settle_agendas(STALE_MS);
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].2, Outcome::Fail);
        assert_eq!(m.round().failed_at("watch-earth"), Some(None));
    }

    /// **A round mission whose character was suspended longer than the hour
    /// is not called off for it**: not while it is held, and on its resume the
    /// hour picks up where it stood, so only time it could work counts.
    #[test]
    fn a_suspension_longer_than_the_hour_does_not_call_a_mission_off() {
        use std::collections::BTreeSet;

        use crate::sim::missions::Missions;

        let held = BTreeSet::from(["zen".to_string()]);
        let mut m = Missions::default();
        m.set_unordered("zen", true);
        m.set_round(round());
        m.take_round("zen", round()[2].mission(), "watch earth", 0);
        m.hold_round(&held, 1_000);
        assert!(
            m.settle_agendas(3 * STALE_MS).is_empty(),
            "called off while it was suspended"
        );
        m.hold_round(&BTreeSet::new(), 3 * STALE_MS);
        assert!(
            m.settle_agendas(3 * STALE_MS).is_empty(),
            "the suspension counted against it"
        );
        // 1,000 ms of its hour ran before it was held; the rest runs now.
        assert!(m.settle_agendas(4 * STALE_MS - 1_001).is_empty());
        let closed = m.settle_agendas(4 * STALE_MS - 1_000);
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].2, Outcome::Fail);
    }
}
