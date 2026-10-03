//! What the room timer has to be.
//!
//! The interesting assertions here are all about *where* and *when*, because
//! those are the two things a single building for the whole world would get
//! wrong: the same rat in two rooms at once, and a slow burn that ran itself
//! out against an empty floor before anybody walked in.

use std::time::Duration;

use npc_map::salience::{weight, Weight};
use npc_map::witness::{narrate, since};
use npc_map::world::{Happening, Where, World};
use npc_map::MapSet;

use crate::engine::event::Salience;
use crate::engine::rooms::{rung, Rooms, WAIT};
use crate::engine::stir::Stirring;
use crate::sim::{seed, Sim};

fn shipped() -> MapSet {
    MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
        .expect("the shipped vault must load")
}

/// The vault's world and the sim its objects are devices in.
fn vault() -> (World, Sim) {
    let map = shipped();
    let sim = seed::vault(Some(&map));
    (World::new(map), sim)
}

fn at(node: &str) -> Where {
    Where::new("vault-casting", node)
}

/// Long enough that every room has certainly been asked at least once.
fn past_the_wait(n: u32) -> Duration {
    WAIT.end * n
}

/// Every stirring in the log, with where it happened.
fn stirrings(w: &World) -> Vec<(Where, String)> {
    w.log()
        .iter()
        .filter_map(|e| match &e.what {
            Happening::Stirred { text, .. } => Some((e.place.clone(), text.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn a_room_nobody_is_standing_in_is_never_asked() {
    // Not only an economy. A room advanced while empty burns its whole slow
    // burn against a floor nobody was on, so the first Maker to walk into the
    // coolant gallery finds it pooling for reasons nobody witnessed.
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    let mut rooms = Rooms::new();

    for i in 1..=20 {
        rooms.stir_at(&mut w, &mut sim, past_the_wait(i));
    }
    assert_eq!(rooms.len(), 1, "a room with nobody in it was fitted");
    for (place, _) in stirrings(&w) {
        assert_eq!(
            place,
            at("green-room"),
            "the building spoke into an empty room"
        );
    }
}

#[test]
fn two_rooms_are_two_buildings() {
    // The reason this is per room rather than per world: a rat is in a room. One
    // building for the whole vault would have the same rat bolt across two
    // floors at once.
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.enter("m2", "Maker-02", at("band-one")).unwrap();
    let mut rooms = Rooms::new();

    for i in 1..=30 {
        rooms.stir_at(&mut w, &mut sim, past_the_wait(i));
    }
    assert_eq!(rooms.len(), 2);

    let said = stirrings(&w);
    let green: Vec<&String> = said
        .iter()
        .filter(|(p, _)| *p == at("green-room"))
        .map(|(_, t)| t)
        .collect();
    let band: Vec<&String> = said
        .iter()
        .filter(|(p, _)| *p == at("band-one"))
        .map(|(_, t)| t)
        .collect();
    assert!(!green.is_empty() && !band.is_empty(), "{said:?}");
    assert_ne!(
        green, band,
        "the two rooms ran identically, so they are one building in two places"
    );
}

#[test]
fn a_room_waits_between_looks() {
    // The metronome runs at 2 Hz. Without the per-room timer every room would
    // speak twice a second, which is the four-second treadmill this whole
    // design was built to remove, in better prose.
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    let mut rooms = Rooms::new();

    let mut spoke = 0;
    // Ten minutes of world, sampled every half second, exactly as the daemon
    // does it.
    let run = Duration::from_secs(600);
    for beat in 0..1200u32 {
        spoke += rooms.stir_at(&mut w, &mut sim, Duration::from_millis(500) * beat);
    }

    // **Derived from `WAIT`, not written down twice.** The rate is a dial, and
    // a test that hard-codes today's setting fails on the next turn of it for
    // the wrong reason — saying "the room is not waiting" when the room is
    // waiting exactly as long as it was told to.
    let most = run.as_secs() / WAIT.start.as_secs();
    let fewest = run.as_secs() / WAIT.end.as_secs();
    assert!(
        (fewest..=most).contains(&(spoke as u64)),
        "{spoke} stirrings in ten minutes, outside the {fewest}–{most} a \
         {WAIT:?} timer admits"
    );
}

#[test]
fn the_first_look_is_a_wait_away_and_not_on_arrival() {
    // A Maker walking into a room should meet the room, not a fan changing note
    // in the same instant it arrives.
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    let mut rooms = Rooms::new();
    assert_eq!(rooms.stir_at(&mut w, &mut sim, Duration::ZERO), 0);
    assert_eq!(
        rooms.stir_at(&mut w, &mut sim, WAIT.start - Duration::from_secs(1)),
        0
    );
}

#[test]
fn a_stirring_is_read_only_in_the_room_it_happened_in() {
    // A building noise that carried through a doorway would put the same rat in
    // two rooms, and a character that walked next door to look would find the
    // one place it certainly is not.
    let (mut w, _) = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.enter("m2", "Maker-02", at("band-one")).unwrap();
    w.mark_seen("m1");
    w.mark_seen("m2");

    w.stir(
        &at("green-room"),
        "The lights in the ceiling flicker and steady.",
        Weight::Note,
    )
    .unwrap();

    let here = since(&w, "m1");
    assert_eq!(here.len(), 1, "{here:?}");
    assert!(here[0].here);
    assert!(
        since(&w, "m2").is_empty(),
        "it carried into the next room: {:?}",
        since(&w, "m2")
    );
}

#[test]
fn a_stirring_reads_as_a_sentence_with_nobody_attached() {
    // The rule the whole `stir` module is written to: the line names its own
    // subject, so `narrate` must stand it on its own rather than putting it in
    // somebody's hands.
    let (mut w, _) = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.enter("m2", "Maker-02", at("green-room")).unwrap();
    w.mark_seen("m1");

    w.say("m2", "somebody should look at the redoubt").unwrap();
    w.stir(
        &at("green-room"),
        "The lights in the ceiling flicker and steady.",
        Weight::Note,
    )
    .unwrap();

    let seen = since(&w, "m1");
    let text = narrate(&w, &seen).expect("something happened");
    assert!(
        text.contains("The lights in the ceiling flicker and steady."),
        "{text}"
    );
    assert!(
        !text.contains("Maker-02 The lights"),
        "the building's doing was attributed to somebody: {text}"
    );
}

#[test]
fn what_the_building_does_is_worth_what_it_says_it_is() {
    // Every other happening's weight follows from its kind. A fan changing note
    // and a breaker going are the same kind of thing, so the weight has to ride
    // on the event.
    let (mut w, _) = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.mark_seen("m1");
    w.stir(
        &at("green-room"),
        "A breaker goes somewhere in the vault.",
        Weight::Preempt,
    )
    .unwrap();
    w.stir(
        &at("green-room"),
        "A pen rolls to the edge of the table.",
        Weight::Ambient,
    )
    .unwrap();

    let seen = since(&w, "m1");
    assert_eq!(weight(&seen[0]), Weight::Preempt);
    assert_eq!(weight(&seen[1]), Weight::Ambient);
}

#[test]
fn the_two_scales_meet_at_the_maps_own_bars() {
    // `stir` reasons in floats, the map in named rungs. The mapping is a lookup
    // against the bars the map already enforces, not a judgement — anything
    // that would preempt does, and anything that would end a standing wait
    // wakes.
    let s = |v: Salience| rung(&Stirring::new("t", "A thing happened in the room.", v));
    assert_eq!(s(Salience::URGENT), Weight::Preempt);
    assert_eq!(s(Salience::new(Salience::PREEMPT_AT)), Weight::Preempt);
    assert_eq!(s(Salience::new(Salience::ROUSES_AT)), Weight::Wake);
    assert_eq!(s(Salience::NORMAL), Weight::Note);
    assert_eq!(s(Salience::IDLE), Weight::Ambient);
}

#[test]
fn a_room_hears_the_recordings_its_building_holds() {
    // The tannoy's words come off the map, and the map puts them on the
    // *building* — so a room three levels down still hears them without the
    // level repeating the list.
    let (w, _) = vault();
    let said = w.map().announcements_for("vault-casting");
    assert!(
        said.len() >= 200,
        "the casting floor cannot hear the vault's tannoy: {} recordings",
        said.len()
    );
    assert_eq!(
        said,
        w.map().announcements_for("creators-vault"),
        "a level heard something different from its building"
    );
}

fn command(node: &str) -> Where {
    Where::new("vault-command", node)
}

/// The place string a device or a posting is filed under.
fn place(w: &Where) -> String {
    format!("{}/{}", w.area, w.node)
}

const PANEL: &str = "the breaker panel";
const BOARD: &str = "the status board";

fn panel_mode(sim: &Sim) -> String {
    sim.devices
        .by_name_at(&place(&command("plant")), PANEL)
        .expect("the plant room holds a breaker panel")
        .mode
        .clone()
}

fn put_panel(sim: &mut Sim, mode: &str) {
    let device = sim
        .devices
        .by_name_at_mut(&place(&command("plant")), PANEL)
        .expect("the plant room holds a breaker panel");
    assert!(device.set(mode), "{mode} is not a mode of the panel");
}

#[test]
fn a_room_is_fitted_with_what_it_has_an_object_for() {
    // The plant room holds a breaker panel and so has a power supply to speak of;
    // the pressure-door room holds none, so a building fitted there has no
    // supply to dip.
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("plant")).unwrap();
    w.enter("m2", "Maker-02", command("enquiry")).unwrap();
    let mut rooms = Rooms::new();
    rooms.stir_at(&mut w, &mut sim, Duration::ZERO);

    let plant = rooms.fitted_ids(&command("plant"));
    let enquiry = rooms.fitted_ids(&command("enquiry"));
    assert!(plant.contains(&"power"), "{plant:?}");
    assert!(!enquiry.contains(&"power"), "{enquiry:?}");
    assert!(enquiry.contains(&"structure"), "{enquiry:?}");
}

#[test]
fn a_room_without_the_object_never_names_the_system() {
    // The enquiry room has a pressure door and nothing else to fail. Hours of it
    // never mention a breaker panel, because there is none to mention.
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("enquiry")).unwrap();
    let mut rooms = Rooms::new();

    for beat in 0..(6 * 3600 * 2u32) {
        rooms.stir_at(&mut w, &mut sim, Duration::from_millis(500) * beat);
    }

    let said = stirrings(&w);
    assert!(!said.is_empty(), "the room never spoke at all");
    for (_, text) in said {
        assert!(
            !text.to_lowercase().contains("breaker panel"),
            "a room with no breaker panel named one: {text}"
        );
    }
}

#[test]
fn a_character_who_trips_the_panel_changes_what_the_room_does() {
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("plant")).unwrap();
    let mut rooms = Rooms::new();
    rooms.stir_at(&mut w, &mut sim, Duration::ZERO);
    assert_eq!(panel_mode(&sim), "closed");

    put_panel(&mut sim, "tripped");
    rooms.stir_at(&mut w, &mut sim, Duration::from_secs(1));
    let said = stirrings(&w);
    assert!(
        said.iter()
            .any(|(p, t)| *p == command("plant") && t.contains("clunks over")),
        "nothing in the room answered the breaker being thrown: {said:?}"
    );
    assert_eq!(
        panel_mode(&sim),
        "tripped",
        "the room undid the character's own work"
    );
}

#[test]
fn a_character_who_resets_the_panel_is_answered_and_not_overwritten() {
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("plant")).unwrap();
    let mut rooms = Rooms::new();
    rooms.stir_at(&mut w, &mut sim, Duration::ZERO);
    put_panel(&mut sim, "tripped");
    rooms.stir_at(&mut w, &mut sim, Duration::from_secs(1));

    put_panel(&mut sim, "closed");
    rooms.stir_at(&mut w, &mut sim, Duration::from_secs(2));
    let said = stirrings(&w);
    assert!(
        said.iter().any(|(_, t)| t.contains("thrown back in")),
        "the reset was not noticed: {said:?}"
    );
    assert_eq!(panel_mode(&sim), "closed");

    // Nothing else changes it: the supply was mended by hand and stays mended
    // until something new goes wrong.
    rooms.stir_at(&mut w, &mut sim, Duration::from_secs(3));
    assert_eq!(panel_mode(&sim), "closed");
}

#[test]
fn a_fault_the_building_raises_puts_its_object_in_the_fault_mode() {
    // The reverse direction: a character walking in finds the breaker already
    // tripped, because the supply failed, rather than being told so.
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("plant")).unwrap();
    let mut rooms = Rooms::new();

    // Nobody touches the panel, so the only way it reads tripped is the supply
    // having been written to it.
    let tripped = (0..(24 * 3600 * 2u32)).any(|beat| {
        rooms.stir_at(&mut w, &mut sim, Duration::from_millis(500) * beat);
        panel_mode(&sim) == "tripped"
    });
    assert!(tripped, "the supply never failed in a day of running");
}

#[test]
fn the_status_board_writes_up_the_fault_that_really_stands() {
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("plant")).unwrap();
    w.enter("m2", "Maker-02", command("dispatch")).unwrap();
    let mut rooms = Rooms::new();
    rooms.stir_at(&mut w, &mut sim, Duration::ZERO);

    // Keep the panel tripped until the board next looks, so the test does not
    // depend on which of the two timers happens to fall first.
    let mut entry = None;
    for beat in 1..4000u32 {
        if panel_mode(&sim) == "closed" {
            put_panel(&mut sim, "tripped");
        }
        rooms.stir_at(&mut w, &mut sim, Duration::from_millis(500) * beat);
        if let Some(p) = sim.postings.by_name_at(&place(&command("dispatch")), BOARD) {
            if let Some(line) = p.lines.last() {
                entry = Some(line.clone());
                break;
            }
        }
    }

    let line = entry.expect("the board never wrote anything up");
    assert_eq!(line.by, "the building");
    assert_eq!(
        line.text,
        "Fault against the main supply bus: a breaker has tripped, at the breaker panel in \
         the plant room."
    );
}

#[test]
fn the_status_board_takes_the_entry_down_when_the_fault_is_mended() {
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("plant")).unwrap();
    w.enter("m2", "Maker-02", command("dispatch")).unwrap();
    let mut rooms = Rooms::new();
    rooms.stir_at(&mut w, &mut sim, Duration::ZERO);

    let mut beat = 1u32;
    let written = |sim: &Sim, starting: &str| {
        sim.postings
            .by_name_at(&place(&command("dispatch")), BOARD)
            .map(|p| p.lines.iter().any(|l| l.text.starts_with(starting)))
            .unwrap_or(false)
    };
    while !written(&sim, "Fault against") {
        assert!(beat < 4000, "the board never wrote the fault up");
        if panel_mode(&sim) == "closed" {
            put_panel(&mut sim, "tripped");
        }
        rooms.stir_at(&mut w, &mut sim, Duration::from_millis(500) * beat);
        beat += 1;
    }

    put_panel(&mut sim, "closed");
    let mended = beat;
    while !written(&sim, "Cleared") {
        assert!(beat < mended + 4000, "the board never took it down");
        rooms.stir_at(&mut w, &mut sim, Duration::from_millis(500) * beat);
        beat += 1;
    }
}

#[test]
fn a_board_with_nothing_wrong_writes_no_fault() {
    let (mut w, mut sim) = vault();
    w.enter("m1", "Maker-01", command("dispatch")).unwrap();
    let mut rooms = Rooms::new();

    for beat in 0..(6 * 3600 * 2u32) {
        rooms.stir_at(&mut w, &mut sim, Duration::from_millis(500) * beat);
    }

    let lines = sim
        .postings
        .by_name_at(&place(&command("dispatch")), BOARD)
        .map(|p| p.lines.clone())
        .unwrap_or_default();
    assert!(
        lines.iter().all(|l| !l.text.starts_with("Fault against")),
        "the board listed a fault nothing has: {lines:?}"
    );
}

#[test]
fn a_world_with_no_recordings_still_runs() {
    // The engine holds no announcements of its own. A map that declares none is
    // a building nobody left a message in, not a bug.
    let (w, _) = vault();
    assert!(w.map().announcements_for("nowhere-at-all").is_empty());
}
