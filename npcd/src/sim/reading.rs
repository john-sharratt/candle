//! What a station tells you when you look at it.
//!
//! A character that asks "which fabricators are running?" or "what is in the
//! stockpile?" is asking the world, and the world has an answer: the machine's
//! mode, the tower's stock, which queues hold a batch. Without a reading those
//! questions have no answer anywhere a character can reach, so it asks its
//! neighbours — who know no more — and the question circulates. A reading is
//! assembled here from the sim's own state, so it is the same truth the grammar
//! and the refusals draw on and cannot drift from them.
//!
//! The reading is keyed on what the station *does* (the acts it affords), not on
//! what it is called, so a new station that produces or commands the tower
//! reads honestly the moment the catalogue says it does.

use serde_json::{json, Map, Value};

use crate::sim::Sim;

impl Sim {
    /// The live state of one placed station: the machine's own mode and whether
    /// it answers, and — for the acts it affords — what the tower behind them
    /// holds right now: its draw, how long the energy lasts, what is closing on
    /// it, and who is deciding what to do about each. `who` turns the body
    /// holding a decision into the name a character knows it by. Empty when the
    /// sim knows nothing about it.
    pub fn reading(
        &self,
        instance_id: &str,
        place: &str,
        acts: &[&str],
        who: &dyn Fn(&str) -> String,
    ) -> Map<String, Value> {
        let mut out = Map::new();
        if let Some(device) = self.station(instance_id) {
            out.insert("mode".into(), json!(device.mode));
            out.insert("modes".into(), json!(device.modes));
            out.insert("working".into(), json!(device.working));
        }
        let Some(tower) = self.tower.as_ref() else {
            return out;
        };
        if acts.contains(&"produce") && self.makes_here(place) {
            let stockpile: Map<String, Value> = tower
                .stockpile()
                .into_iter()
                .map(|(name, amount)| (name.to_string(), json!(amount)))
                .collect();
            out.insert("stockpile".into(), Value::Object(stockpile));
            out.insert("can_make".into(), json!(tower.makeable()));
            out.insert("free_queues".into(), json!(tower.free_queues()));
            let queued: Vec<Value> = tower
                .queued()
                .iter()
                .map(|b| {
                    let what = tower
                        .recipe_by_name(&b.recipe)
                        .map_or(b.recipe.clone(), |r| r.name.clone());
                    json!({ "what": what, "count": b.count, "queue": b.queue })
                })
                .collect();
            out.insert("queued".into(), Value::Array(queued));
        }
        if acts.contains(&"command_tower") && self.commands_here(place) {
            let contact = tower
                .contact
                .as_ref()
                .map(|c| json!({ "id": c.id, "kind": c.kind, "minutes_out": c.minutes_out() }));
            let decisions: Vec<Value> = self
                .tower_decisions()
                .into_iter()
                .map(|d| json!({ "what": d.what, "held_by": d.held_by.as_deref().map(who) }))
                .collect();
            out.insert(
                "tower".into(),
                json!({
                    "posture": tower.posture,
                    "depth": tower.depth,
                    "shields": tower.shields,
                    "besieging": tower.besieging,
                    "can": tower.actions(),
                    "energy_per_minute": tower.draw_per_minute(),
                    "energy_from_the_ground_per_minute": tower.tap_per_minute(),
                    "minutes_of_energy": tower.minutes_of_energy(),
                    "contact": contact,
                    "decisions": decisions,
                }),
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use npc_map::load::MapSet;
    use npc_map::schema::Where;
    use serde_json::json;

    use crate::sim::field::Resource;
    use crate::sim::seed::battle_cities;
    use crate::sim::tower::Posture;
    use crate::sim::upkeep::{
        BASE_DRAW, CONTACT_CLOSES_IN, CONTACT_EVERY, RESERVE, SHIELD_DRAW, TAP_YIELD,
    };
    use crate::sim::Sim;

    const FOUNDRY: &str = "tower-redoubt/foundry";

    fn named(body: &str) -> String {
        format!("name-of-{body}")
    }

    fn console(sim: &Sim) -> String {
        sim.part_tools
            .iter()
            .find(|(_, tools)| tools.iter().any(|t| t == "command_tower"))
            .map(|(place, _)| place.clone())
            .expect("a station commands the tower")
    }

    fn maps() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped maps must load")
    }

    fn a_fabricator(map: &MapSet) -> String {
        map.instances_at(&Where::new("tower-redoubt", "foundry"))
            .into_iter()
            .find(|i| i.part_id() == "fabricator")
            .expect("a fabricator stands in the foundry")
            .id()
    }

    /// **A fabricator reads its own state and the stockpile behind it.** The mode
    /// it is in, what the tower holds, what can be made from it and which queues
    /// are free — the answers a character was otherwise asking its neighbours for.
    #[test]
    fn a_fabricator_reads_its_mode_and_the_stockpile() {
        let map = maps();
        let sim = battle_cities(Some(&map));
        let id = a_fabricator(&map);

        let read = sim.reading(&id, FOUNDRY, &["produce"], &named);

        assert_eq!(read["mode"], json!("idle"), "{read:?}");
        assert_eq!(read["working"], json!(true), "{read:?}");
        assert_eq!(read["stockpile"]["nanobots"], json!(12), "{read:?}");
        assert!(
            read["stockpile"]["energy"].as_u64().unwrap() > 0,
            "{read:?}"
        );
        let can_make = read["can_make"].as_array().expect("a can_make list");
        assert!(can_make.contains(&json!("bolt rounds")), "{read:?}");
        assert!(!can_make.contains(&json!("a companion")), "{read:?}");
        assert_eq!(read["free_queues"].as_array().unwrap().len(), 8, "{read:?}");
        assert_eq!(read["queued"], json!([]), "{read:?}");
    }

    /// **A batch on a queue shows in the reading by the name a character knows
    /// it by**, and takes its queue out of the free ones.
    #[test]
    fn a_queued_batch_is_read_by_name_and_holds_its_queue() {
        let map = maps();
        let mut sim = battle_cities(Some(&map));
        let id = a_fabricator(&map);
        sim.tower
            .as_mut()
            .unwrap()
            .queue("bolt rounds", 2, 3)
            .expect("the stockpile covers two batches");

        let read = sim.reading(&id, FOUNDRY, &["produce"], &named);

        assert_eq!(
            read["queued"],
            json!([{ "what": "bolt rounds", "count": 2, "queue": 3 }]),
            "{read:?}"
        );
        assert!(
            !read["free_queues"]
                .as_array()
                .unwrap()
                .contains(&json!("3")),
            "{read:?}"
        );
    }

    /// **A station that does not produce does not read like one.** Stock is the
    /// foundry's to report; a station that affords nothing of the kind says only
    /// what its machine is.
    #[test]
    fn a_station_that_does_not_produce_reports_no_stockpile() {
        let map = maps();
        let sim = battle_cities(Some(&map));
        let id = a_fabricator(&map);

        let read = sim.reading(&id, FOUNDRY, &[], &named);

        assert!(read.get("stockpile").is_none(), "{read:?}");
        assert!(read.get("can_make").is_none(), "{read:?}");
        assert!(read.contains_key("mode"), "{read:?}");
    }

    /// **The tower reads as itself where it can be commanded from**, and what it
    /// can afford is what the grammar would offer.
    #[test]
    fn the_tower_reads_where_it_can_be_commanded() {
        let map = maps();
        let sim = battle_cities(Some(&map));
        let place = console(&sim);

        let read = sim.reading("no-such-device", &place, &["command_tower"], &named);

        assert_eq!(read["tower"]["posture"], json!("standing"), "{read:?}");
        assert_eq!(read["tower"]["shields"], json!(false), "{read:?}");
        assert_eq!(read["tower"]["can"], json!(sim.tower_actions()), "{read:?}");
        assert!(read.get("mode").is_none(), "{read:?}");
    }

    /// **The tower reads what it costs to keep up and how long that lasts**, so
    /// "is the stockpile draining?" has a number for an answer rather than a
    /// guess.
    #[test]
    fn the_tower_reads_its_draw_and_how_long_the_energy_lasts() {
        let map = maps();
        let mut sim = battle_cities(Some(&map));
        let place = console(&sim);
        let energy = sim.tower.as_ref().unwrap().stock_of(Resource::Energy);

        let read = sim.reading("x", &place, &["command_tower"], &named);
        assert_eq!(
            read["tower"]["energy_per_minute"],
            json!(BASE_DRAW),
            "{read:?}"
        );
        assert_eq!(
            read["tower"]["minutes_of_energy"],
            json!((energy - RESERVE) / BASE_DRAW),
            "{read:?}"
        );

        sim.tower.as_mut().unwrap().shields = true;
        let shielded = sim.reading("x", &place, &["command_tower"], &named);
        assert_eq!(
            shielded["tower"]["energy_per_minute"],
            json!(BASE_DRAW + SHIELD_DRAW),
            "{shielded:?}"
        );
    }

    /// **A buried tower reads what the ground gives it, and that it will not run
    /// out** when the ground gives more than it draws.
    #[test]
    fn a_buried_tower_reads_what_the_ground_gives_it() {
        let map = maps();
        let mut sim = battle_cities(Some(&map));
        let place = console(&sim);

        let standing = sim.reading("x", &place, &["command_tower"], &named);
        assert_eq!(
            standing["tower"]["energy_from_the_ground_per_minute"],
            json!(0),
            "{standing:?}"
        );

        sim.tower.as_mut().unwrap().posture = Posture::DugIn;
        let buried = sim.reading("x", &place, &["command_tower"], &named);
        assert_eq!(
            buried["tower"]["energy_from_the_ground_per_minute"],
            json!(TAP_YIELD),
            "{buried:?}"
        );
        assert_eq!(
            buried["tower"]["minutes_of_energy"],
            json!(null),
            "{buried:?}"
        );
    }

    /// **A contact is readable, with how long until it is here.**
    #[test]
    fn the_tower_reads_what_is_closing_on_it() {
        let map = maps();
        let mut sim = battle_cities(Some(&map));
        let place = console(&sim);

        let quiet = sim.reading("x", &place, &["command_tower"], &named);
        assert_eq!(quiet["tower"]["contact"], json!(null), "{quiet:?}");

        sim.tower.as_mut().unwrap().advance(CONTACT_EVERY);
        let read = sim.reading("x", &place, &["command_tower"], &named);
        assert_eq!(read["tower"]["contact"]["id"], json!(1), "{read:?}");
        assert_eq!(
            read["tower"]["contact"]["minutes_out"],
            json!(CONTACT_CLOSES_IN.as_secs() / 60),
            "{read:?}"
        );
    }

    /// **Who is deciding a decision is readable, by name**, so a character that
    /// wants to know whether it is its to make can see without asking anybody.
    #[test]
    fn the_tower_reads_who_is_deciding_what() {
        let map = maps();
        let mut sim = battle_cities(Some(&map));
        let place = console(&sim);
        sim.tower.as_mut().unwrap().advance(CONTACT_EVERY);
        sim.watch_tower();

        let open = sim.reading("x", &place, &["command_tower"], &named);
        let decisions = open["tower"]["decisions"].as_array().unwrap();
        assert_eq!(decisions.len(), 1, "{open:?}");
        assert!(decisions[0]["what"].as_str().unwrap().contains("contact 1"));
        assert_eq!(decisions[0]["held_by"], json!(null), "{open:?}");

        let what = decisions[0]["what"].as_str().unwrap().to_string();
        sim.ledger.take_order(&what, "c1").unwrap();
        let held = sim.reading("x", &place, &["command_tower"], &named);
        assert_eq!(
            held["tower"]["decisions"][0]["held_by"],
            json!("name-of-c1"),
            "{held:?}"
        );
    }

    /// **A world with no tower reads only its machine.** The vault has fabricator
    /// language in its orders and none in its sim, and must not invent a stock.
    #[test]
    fn a_world_without_a_tower_reads_no_stock() {
        let map = maps();
        let sim = crate::sim::seed::vault(Some(&map));

        let read = sim.reading(
            "nothing",
            "vault-chronicle/early-range",
            &["produce"],
            &named,
        );

        assert!(read.is_empty(), "{read:?}");
    }
}
