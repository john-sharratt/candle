//! Building a world's state, deterministically.
//!
//! # A world gets what it is, and nothing else
//!
//! There is no capability flag anywhere in this file. A seed puts things into a
//! world, and the tools that need those things become reachable because their
//! live sets stop being empty. The vault gets terminals and an order board and
//! no hostiles, so `engage` is unreachable there — not disabled, unreachable,
//! which is a stronger guarantee and one nothing can get out of step with.
//!
//! # Deterministic, because the tests assert live sets
//!
//! Nothing here reads a clock or a random source, every collection is ordered,
//! and every quantity is written down. Two calls to the same seed produce
//! byte-identical state. That is what lets a tool test say *these are the four
//! things you may name* rather than *something plausible was offered*.

use npc_map::part::{Part, PartKind};
use npc_map::MapSet;

use crate::sim::device::{Device, Devices, Kind as DeviceKind};
use crate::sim::field::{Breed, Deposit, Field, Hostile, Resource};
use crate::sim::item::{Item as PackItem, Kind as ItemKind};
use crate::sim::phone;
use crate::sim::posting;
use crate::sim::record::{Item, Kind as RecordKind, State as RecordState};
use crate::sim::tower::{Coord, Recipe, Tower};
use crate::sim::Sim;

/// What sort of machine a part is, from what it is called and what it does.
///
/// The kind decides nothing mechanical — the modes do that — so a part whose
/// name matches nothing in particular is still a working device. It is used for
/// grouping and description, and guessing wrong costs a category and not a
/// capability.
fn kind_of(part: &Part) -> DeviceKind {
    let id = part.id.as_str();
    if id.contains("door") {
        DeviceKind::Portal
    } else if id.contains("turret") {
        DeviceKind::Turret
    } else if id.contains("fabricator") {
        DeviceKind::Fabricator
    } else if id.contains("carrier") || id.contains("vehicle") {
        DeviceKind::Vehicle
    } else if id.contains("panel") {
        DeviceKind::Panel
    } else {
        DeviceKind::Terminal
    }
}

/// Every machine in a loaded map, derived from the parts standing in its rooms.
///
/// **The map is the only place a device is written down.** A part that declares
/// modes is a thing a body works, so it becomes a device wherever it stands and
/// as many times as it stands there — six terminals in a room are six things to
/// claim, which is what makes a level able to hold a crew.
///
/// The alternative was a second list in this file, free to disagree with the
/// map about what a room contains. It would have disagreed within the week.
pub fn devices_from_map(map: &MapSet) -> Devices {
    devices_and_tools(map).0
}

/// Every machine in a map, and what every room's parts let a body do there.
///
/// Two answers from one walk, because they come from the same parts and a
/// second walk could disagree. A part becomes a *device* only if it declares
/// modes; it contributes its *tools* either way, which is the distinction that
/// matters for a station whose whole job is one act — a bridge console is
/// spoken to, not switched.
pub fn devices_and_tools(map: &MapSet) -> (Devices, Vec<(String, Vec<String>)>) {
    let mut tools: Vec<(String, Vec<String>)> = Vec::new();
    let mut out = Devices::new();
    for area in map.areas() {
        for node in &area.nodes {
            let at = format!("{}/{}", area.id, node.id);
            // **Which acts a room offers is asked of the catalogue, not of the
            // map.** Each act names the parts it attaches to, so a room's
            // surface is every act whose `at` mentions something standing here.
            //
            // The map used to carry the list, which put this engine's
            // vocabulary inside a crate that only describes buildings, made one
            // act reaching six stations six edits in six files, and turned a
            // typo into an act that quietly attached to nothing.
            let mut here: Vec<String> = Vec::new();
            for (part, _) in map.parts_at(node) {
                for t in crate::engine::tools::CATALOG.iter() {
                    if t.at.contains(&part.id.as_str()) && !here.iter().any(|n| n == t.name) {
                        here.push(t.name.to_string());
                    }
                }
            }
            if !here.is_empty() {
                tools.push((at.clone(), here));
            }
            for (part, count) in map.parts_at(node) {
                if !part.is_machine() {
                    continue;
                }
                let modes: Vec<&str> = part.modes.iter().map(String::as_str).collect();
                // A counted part is several machines, each claimable on its own.
                // Numbering starts at one, because "the second terminal" is what
                // a body would say and "terminal 0" is not.
                for n in 1..=count {
                    let id = format!("{}/{}#{n}", at, part.id);
                    // A part authored as "the catalogue" already carries its
                    // article; one authored as "fabricator" does not. Prefixing
                    // blindly gives "the the catalogue", which a character
                    // would then have to name back exactly.
                    let name = if count == 1 {
                        if part.name.starts_with("the ") {
                            part.name.clone()
                        } else {
                            format!("the {}", part.name)
                        }
                    } else {
                        format!("{} {n}", part.name.trim_start_matches("the "))
                    };
                    let mut device = Device::new(id, name, kind_of(part), &at, &modes);
                    // A station is claimed while it is worked; a fixture is not.
                    device.claimable = part.kind == PartKind::Station;
                    out.install(device);
                }
            }
        }
    }
    (out, tools)
}

/// Every surface in a map that words can be left on.
///
/// **A fixture, by the map's own definition, is "something read or consulted,
/// fixed in place"** — see [`npc_map::part::PartKind::Fixture`]. That is a
/// readable surface described in as many words, and until now nothing in the
/// engine treated it as one: a fixture declares no modes, so it never became a
/// [`Device`], so the boards in the vault were furniture that could be walked
/// past and nothing else. Meanwhile `read` was bound to the *machines*, which
/// hold no text at all.
///
/// So each fixture is stood up as an empty [`posting::Postings`] entry. Empty is
/// the right starting state and costs nothing: a board with nothing on it is
/// offered to nobody, because `readable_at` asks for what a body has not read
/// and there is nothing to have not read. It becomes real the moment anybody
/// writes on it.
fn postings_from_map(map: &MapSet) -> posting::Postings {
    let mut out = posting::Postings::new();
    for area in map.areas() {
        for node in &area.nodes {
            let at = format!("{}/{}", area.id, node.id);
            for (part, _) in map.parts_at(node) {
                if part.kind != PartKind::Fixture {
                    continue;
                }
                // The article the part carries, or one supplied — the same rule
                // a device's name follows, because a character has to name it
                // back exactly as it is written.
                let name = match part.name.starts_with("the ") {
                    true => part.name.clone(),
                    false => format!("the {}", part.name),
                };
                out.stand_up(&at, &name);
            }
        }
    }
    out
}

/// Fill a world's map-derived facts: what stands in each room, what its parts
/// let a body do there, and where a recall lands.
fn from_map(sim: &mut Sim, map: &MapSet, home_area: &str) {
    let (devices, tools) = devices_and_tools(map);
    sim.devices = devices;
    sim.postings = postings_from_map(map);
    for (at, ts) in tools {
        sim.set_part_tools(at, ts);
    }
    if let Some(home) = map.teleport_to(home_area) {
        sim.set_home(format!("{}/{}", home.area, home.node));
    }
}

/// The vault: terminals to work at, an order board, and nothing to shoot.
///
/// Its machines come from the map, so `operate` at a chronicle terminal is the
/// same act as `operate` at a blast door and the editing surface needs no
/// vocabulary of its own.
pub fn vault(map: Option<&MapSet>) -> Sim {
    let mut sim = Sim::new();
    if let Some(map) = map {
        // The vault musters from the room orders are given in, which is where a
        // Maker's day starts and where it is called back to.
        from_map(&mut sim, map, "creators-vault");
    }

    // The board a Maker takes work from. Two standing orders, so `claim` has
    // something to offer on the first tick of a fresh world.
    sim.ledger
        .set_order("close the longest silence in the record", "the vault", None);
    sim.ledger
        .set_order("settle the boundary nobody has settled", "the vault", None);

    // **Something for the stations to be about.** A chronicle terminal with no
    // era to hold offers a tool whose argument has nothing in it, which takes
    // the act out of the grammar — so a vault seeded with no record is a vault
    // where none of the station acts are reachable, and the whole level reads
    // as broken rather than as empty.
    //
    // Two eras that meet, so a boundary can be settled; a silence, so something
    // can be written; a face and a place nobody has done, so drawing and
    // surveying have a subject; and an intake, so custody has something whose
    // origin is not yet written down.
    for item in [
        Item::new("era_third", "the third era", RecordKind::Era).in_state(RecordState::Filed),
        Item::new("era_fourth", "the fourth era", RecordKind::Era).in_state(RecordState::Filed),
        Item::new("gap_third", "the third silence", RecordKind::Gap),
        Item::new(
            "face_unmade",
            "a face nobody has drawn",
            RecordKind::Portrait,
        ),
        Item::new("place_flats", "the eastern flats", RecordKind::Place),
        Item::new("char_courier", "the courier", RecordKind::Character),
        Item::new("intake_west", "the western intake", RecordKind::Accession),
        Item::new(
            "enq_gate",
            "the question about the eastern gate",
            RecordKind::Enquiry,
        ),
    ] {
        sim.record.put(item);
    }

    sim
}

/// The tower and the waste around it: everything Battle Cities needs.
pub fn battle_cities(map: Option<&MapSet>) -> Sim {
    let mut sim = Sim::new();
    if let Some(map) = map {
        from_map(&mut sim, map, "tower-redoubt");
    }

    // ---- the tower ----
    let mut tower = Tower::new("the Redoubt", Coord::new(120, -48));
    tower.put(Resource::Energy, 4_000);
    tower.put(Resource::Metal, 900);
    tower.put(Resource::Ore, 300);
    tower.put(Resource::Gold, 150);
    tower.put(Resource::Crystals, 40);
    tower.put(Resource::Nanobots, 12);

    for r in [
        Recipe::new("bolt", "bolt rounds", &[(Resource::Metal, 20)], 30),
        Recipe::new("stimpak", "stimpaks", &[(Resource::Metal, 15)], 4),
        Recipe::new(
            "missile",
            "missiles",
            &[(Resource::Metal, 60), (Resource::Crystals, 5)],
            6,
        ),
        Recipe::new(
            "companion",
            "a companion",
            &[(Resource::Nanobots, 50), (Resource::Metal, 200)],
            1,
        ),
    ] {
        tower.learn(r);
    }
    sim.tower = Some(tower);

    // ---- what is outside ----
    let mut f = Field::new();
    f.seed_deposit(Deposit::new(
        "east_seam",
        "the ore seam",
        "the-waste/east-ridge",
        Resource::Ore,
        60,
    ));
    f.seed_deposit(Deposit::new(
        "crystal_bloom",
        "the crystal bloom",
        "the-waste/glass-flats",
        Resource::Crystals,
        20,
    ));
    f.seed_deposit(Deposit::new(
        "burnt_carrier",
        "the burnt-out carrier",
        "the-waste/ruins",
        Resource::Metal,
        30,
    ));

    f.seed_hostile(Hostile::new(
        "drone_a",
        "a drone",
        Breed::Drone,
        "the-waste/east-ridge",
    ));
    f.seed_hostile(Hostile::new(
        "drone_b",
        "a second drone",
        Breed::Drone,
        "the-waste/east-ridge",
    ));
    let mut shooter = Hostile::new("shooter_a", "a shooter", Breed::Shooter, "the-waste/ruins");
    shooter.firing = true;
    f.seed_hostile(shooter);
    f.seed_hostile(Hostile::new(
        "mech_a",
        "a mech",
        Breed::Mech,
        "the-waste/ruins",
    ));
    sim.field = f;

    sim
}

/// Issue a handset, put somebody on the roster a phone can reach, and join them
/// to the world's standing channel.
///
/// Three halves of one thing: carrying the handset is what makes the messaging
/// acts reachable, being on the roster is what makes *you* reachable, and being
/// on the channel is what makes you reachable by everybody at once. A character
/// with a phone and an empty roster has nobody to call; one on the roster with
/// no phone can be called and cannot answer.
///
/// **The channel is joined here rather than chosen**, which is the whole of
/// what makes it a capability rather than an option. See [`phone::CHANNEL`] for
/// why a cast that had to opt in would spend its isolated hours not opting in.
///
/// Idempotent in all three, because this runs on every arrival and a character
/// re-entering a world it was already in is the ordinary case, not an error.
pub fn issue_handset(sim: &mut Sim, body: &str, name: &str) {
    sim.pack_mut(body)
        .add(PackItem::new(phone::PHONE, "handset", ItemKind::Gear, 1));
    sim.enrol(name);
}

/// Kit a body out. What a companion walks out of the tower carrying.
pub fn outfit(sim: &mut Sim, body: &str) {
    let pack = sim.pack_mut(body);
    pack.add(PackItem::new(
        "mono_sword",
        "mono sword",
        ItemKind::Weapon,
        1,
    ));
    pack.add(PackItem::new(
        "plasma_rifle",
        "plasma rifle",
        ItemKind::Weapon,
        1,
    ));
    pack.add(PackItem::new(
        "combat_armour",
        "combat armour",
        ItemKind::Armour,
        1,
    ));
    pack.add(PackItem::new(
        "bolt",
        "bolt rounds",
        ItemKind::Ammunition,
        60,
    ));
    pack.add(PackItem::new("stimpak", "stimpak", ItemKind::Consumable, 3));
    pack.add(PackItem::new(
        "scanner",
        "advanced scanner",
        ItemKind::Gear,
        1,
    ));
}

/// The state a world starts in, chosen by its id.
///
/// An unknown world gets an empty [`Sim`], which is the honest answer: it has no
/// items, no machines, no ground worth digging and no tower, so it is offered
/// speech, movement and attention and nothing else. A world nobody has described
/// should not be able to shoot.
pub fn for_world(id: &str, map: Option<&MapSet>) -> Sim {
    match id {
        "creators-vault" => vault(map),
        "battle-cities" => battle_cities(map),
        _ => {
            // Even an undescribed world gets its machines, because those come
            // from its own map rather than from this file. What it does not get
            // is a tower, a stockpile, or anything to shoot.
            let mut sim = Sim::new();
            if let Some(map) = map {
                sim.devices = devices_from_map(map);
            }
            sim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The maps this repository ships, which every seed test loads so that what
    /// it asserts about a room is what the map actually says is in it.
    fn maps() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped maps must load")
    }

    #[test]
    fn the_same_seed_twice_is_the_same_world_to_the_byte() {
        let m = maps();
        assert_eq!(vault(Some(&m)), vault(Some(&m)));
        assert_eq!(battle_cities(Some(&m)), battle_cities(Some(&m)));
    }

    #[test]
    fn the_vault_has_nothing_to_shoot_and_nowhere_to_dig() {
        // The mechanism that keeps Battle Cities' vocabulary out of the vault:
        // not a flag, an empty live set.
        let s = vault(Some(&maps()));
        assert!(s.field.is_empty());
        assert!(s.hostiles("vault-chronicle/early-range").is_empty());
        assert!(s.extractable("vault-chronicle/early-range").is_empty());
        assert!(s.tower.is_none(), "the vault grew a tower");
        assert!(s.tower_actions().is_empty());
        // It *does* have somewhere to port back to — the room orders are given
        // in. A muster point and a tower are different things, and conflating
        // them is what made `recall` look like a battlefield act.
        assert!(s.has_home(), "the vault lost its own teleport point");
    }

    #[test]
    fn the_vault_has_terminals_to_work_at_and_orders_to_take() {
        let s = vault(Some(&maps()));
        // Six terminals stand in the early range, so six things can be claimed
        // there — the count is what makes a level able to hold a crew.
        assert_eq!(s.operable("vault-chronicle/early-range").len(), 6);
        assert_eq!(
            s.modes_of("vault-chronicle/early-range", "world history terminal 1"),
            vec!["reading", "working", "offered"]
        );
        assert_eq!(s.unheld_orders().len(), 2);
    }

    #[test]
    fn a_counted_part_becomes_that_many_machines_and_a_lone_one_is_the_only_one() {
        let s = vault(Some(&maps()));
        let one = s.operable("vault-command/plant");
        assert_eq!(
            one.len(),
            0,
            "the plant panel declares no modes, so it is read"
        );

        let many = s.operable("vault-chronicle/catalogue-room");
        // The catalogue plus two terminals, all three of them machines.
        assert_eq!(many.len(), 3, "{many:?}");
        assert!(many.contains(&"the catalogue".to_string()));
    }

    #[test]
    fn battle_cities_has_a_tower_that_can_pay_for_what_it_offers() {
        let s = battle_cities(Some(&maps()));
        let acts = s.tower_actions();
        assert!(acts.contains(&"relocate".to_string()));
        assert!(acts.contains(&"siege".to_string()));
        assert!(acts.contains(&"drill down".to_string()));
        assert!(acts.contains(&"raise shields".to_string()));
        assert!(s.has_home());
    }

    #[test]
    fn only_what_the_stockpile_covers_can_be_made() {
        let s = battle_cities(Some(&maps()));
        let makeable = s.makeable();
        // Twelve nanobots against a companion's fifty.
        assert!(!makeable.contains(&"a companion".to_string()));
        assert!(makeable.contains(&"bolt rounds".to_string()));
        assert!(makeable.contains(&"missiles".to_string()));
    }

    #[test]
    fn the_waste_holds_different_things_in_different_places() {
        let s = battle_cities(Some(&maps()));
        assert_eq!(
            s.extractable("the-waste/east-ridge"),
            vec!["the ore seam".to_string()]
        );
        assert_eq!(s.hostiles("the-waste/east-ridge").len(), 2);
        assert_eq!(s.hostiles("the-waste/ruins").len(), 2);
        assert!(
            s.hostiles("tower-redoubt/bridge").is_empty(),
            "the tower interior had hostiles standing in it"
        );
    }

    #[test]
    fn a_turret_and_a_door_offer_their_own_modes_and_not_each_others() {
        let s = battle_cities(Some(&maps()));
        let door = s.modes_of("tower-redoubt/gatehouse", "the blast door");
        let gun = s.modes_of("tower-redoubt/rampart", "wall turret 1");
        assert_eq!(door, vec!["open", "closed", "locked"]);
        assert!(gun.contains(&"air only".to_string()), "{gun:?}");
        assert!(!door.contains(&"air only".to_string()));
    }

    #[test]
    fn the_rampart_holds_four_turrets_and_the_gatehouse_one_door() {
        let s = battle_cities(Some(&maps()));
        assert_eq!(s.operable("tower-redoubt/rampart").len(), 4);
        assert_eq!(s.operable("tower-redoubt/gatehouse").len(), 1);
        assert_eq!(s.operable("tower-redoubt/foundry").len(), 8, "eight queues");
    }

    #[test]
    fn an_outfitted_body_can_equip_use_and_give_and_a_bare_one_cannot() {
        let mut s = battle_cities(Some(&maps()));
        assert!(s.carried("c1").is_empty());
        outfit(&mut s, "c1");
        assert_eq!(s.equippable("c1").len(), 4, "sword, rifle, armour, scanner");
        assert_eq!(s.usable("c1"), vec!["advanced scanner", "stimpak"]);
        assert!(s.carried("c1").contains(&"bolt rounds".to_string()));
    }

    #[test]
    fn a_world_nobody_has_described_gets_nothing_it_could_hurt_anyone_with() {
        let s = for_world("somewhere-else", None);
        assert!(s.field.is_empty());
        assert!(s.devices.is_empty());
        assert!(!s.has_home());
        assert!(!s.has_packs());
        assert!(s.unheld_orders().is_empty());
    }

    #[test]
    fn the_shipped_worlds_are_reachable_by_their_own_ids() {
        let m = maps();
        assert!(!for_world("creators-vault", Some(&m)).devices.is_empty());
        assert!(for_world("battle-cities", Some(&m)).has_home());
    }

    #[test]
    fn a_station_is_claimed_while_it_is_worked_and_a_fixture_is_not() {
        let s = battle_cities(Some(&maps()));
        // A fabricator binds one production run, so two bodies cannot queue
        // over each other.
        let bay = s
            .devices
            .by_name_at("tower-redoubt/foundry", "fabricator 1")
            .expect("the foundry holds eight bays");
        assert!(bay.claimable, "a station that binds a run was not claimed");

        let door = s
            .devices
            .by_name_at("tower-redoubt/gatehouse", "the blast door")
            .expect("the gatehouse holds a door");
        assert!(
            !door.claimable,
            "a claimed door is a door nobody else can shut behind them"
        );
    }
}
