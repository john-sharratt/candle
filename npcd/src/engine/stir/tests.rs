//! What the building has to be, and a look at it running.
//!
//! Two kinds of test live here. The first kind holds the rules every line must
//! obey, because these strings are read by a language model and a malformed one
//! costs a wrong inference rather than a wrong pixel. The second kind pins the
//! *couplings* — each one drives two fixtures against each other and asserts
//! that the second reacted to the first, with a control run proving it would not
//! have reacted on its own.
//!
//! `a_shift_in_the_vault` prints a timeline. Run it with
//!
//! ```text
//! cargo test -p npcd --lib engine::stir -- --nocapture
//! ```

use std::time::{Duration, UNIX_EPOCH};

use crate::engine::event::Salience;
use crate::engine::stir::{
    air::AirHandling, ambient::Ambient, boards::Boards, broadcast::Broadcast, chime::Chime,
    coolant::CoolantLoop, growth::Growth, lights::Lighting, rat::Rat, stores::Stores, Building,
    Cond, Fixture, Stirring, Watch,
};

/// Stand-in recordings, so the tests are not reading the shipped world.
///
/// The engine holds no announcements of its own — the vault's real ones are
/// authored on the building in `<mind>/map/battle-cities/creators-vault.yaml`.
/// These are four throwaway lines that exist only to prove the mechanism, and
/// deliberately do not read like the shipped ones.
fn recordings() -> Vec<String> {
    [
        "This is the first test recording.",
        "This is the second test recording.",
        "This is the third test recording.",
        "This is the fourth test recording.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// A building at `t`, with `conds` true of it.
fn watch(t: u64, conds: &[Cond]) -> Watch {
    Watch {
        since_start: Duration::from_secs(t),
        // Fixed rather than `now()`, so the two fixtures that read the wall
        // clock behave the same in January as in June. Half past nine in the
        // evening, far from any hour boundary.
        wall: UNIX_EPOCH + Duration::from_secs(70_200 + t),
        conds: conds.to_vec(),
    }
}

/// A thing another fixture did, for feeding to [`Fixture::notice`].
fn did(from: &'static str, tags: &[Cond], salience: Salience) -> Stirring {
    Stirring::new(from, "Something happened in the building.", salience).tagged(tags)
}

/// Drive one fixture over `steps` of `every`, collecting what it said.
fn drive(f: &mut dyn Fixture, conds: &[Cond], every: u64, steps: u64) -> Vec<String> {
    (0..steps)
        .filter_map(|i| f.consider(&watch(i * every, conds)).map(|s| s.text))
        .collect()
}

// ---------------------------------------------------------------------------
// The rules every line obeys
// ---------------------------------------------------------------------------

/// Words that make a sentence lean on whatever came before it.
const ANAPHORA: &[&str] = &[
    "it", "its", "he", "his", "she", "her", "they", "them", "their", "this", "that", "these",
    "those",
];

/// Everything the building can say, over a long run and several seeds.
fn everything() -> Vec<Stirring> {
    (0..24u64)
        .flat_map(|seed| {
            Building::new(seed * 7919 + 3, recordings())
                .run(Duration::ZERO, Duration::from_secs(25), 1200)
                .into_iter()
                .map(|(_, s)| s)
        })
        .collect()
}

#[test]
fn every_line_names_its_own_subject() {
    // Several of these run together in a room's prose. A line opening with a
    // bare pronoun attaches itself to whichever subject came last, so the
    // character reasons about the wrong thing having happened — which is worse
    // than being told nothing at all.
    for s in everything() {
        let text = &s.text;
        let first = text.split_whitespace().next().unwrap_or_default();

        assert!(
            first.chars().next().is_some_and(char::is_uppercase),
            "{}: does not start with a capital: {text:?}",
            s.from
        );
        // A recording ends inside its quotes, which is still a finished
        // sentence — the stop is there, it just has a quote mark after it.
        assert!(
            text.ends_with('.') || text.ends_with(".\""),
            "{}: does not end in a full stop: {text:?}",
            s.from
        );
        assert!(
            text.split_whitespace().count() >= 4,
            "{}: too short to be a sentence: {text:?}",
            s.from
        );
        let opener = first
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_lowercase();
        assert!(
            !ANAPHORA.contains(&opener.as_str()),
            "{}: opens on a pronoun, so it will attach to the previous line: {text:?}",
            s.from
        );
    }
}

#[test]
fn most_of_what_the_building_does_can_be_ignored() {
    // The whole reason idle ticks were removed is that a character which is
    // interrupted constantly ends up talking to itself. A building that
    // preempted often would be that treadmill in better prose.
    let all = everything();
    let loud = all.iter().filter(|s| s.salience.preempts()).count();
    let share = loud as f32 / all.len() as f32;
    assert!(
        share < 0.12,
        "{loud} of {} stirrings preempt ({:.1}%) — the building interrupts too much",
        all.len(),
        share * 100.0
    );
}

#[test]
fn the_building_does_not_repeat_itself_for_a_long_time() {
    // The failure this module exists to fix was a character emitting the same
    // sentence over and over. A wide spread of distinct lines is the property
    // that stops it coming back.
    let lines: std::collections::BTreeSet<String> =
        everything().into_iter().map(|s| s.text).collect();
    assert!(
        lines.len() >= 60,
        "only {} distinct lines across every fixture — too narrow",
        lines.len()
    );
}

#[test]
fn most_of_the_building_is_nothing_happening() {
    // The first version of this module was implausible in a way no single
    // fixture could see: a dozen individually reasonable mechanisms all
    // reporting into one room is a base where something goes wrong every twenty
    // seconds. `AT_REST` is the dial that fixes it and this is what holds it.
    let all = everything();
    let calm = all.iter().filter(|s| s.from == "ambient").count();
    let share = calm as f32 / all.len() as f32;
    assert!(
        (0.6..0.9).contains(&share),
        "{calm} of {} stirrings are the building at rest ({:.0}%) — outside the range that reads \
         as a working base rather than one being evacuated",
        all.len(),
        share * 100.0
    );
}

#[test]
fn the_quiet_does_not_repeat_itself_inside_a_conversation() {
    // A cold building has only a handful of cold things to notice, and drawing
    // from that pool directly put frost on the window twice inside five
    // minutes. Nothing else in the module would have caught it: both lines are
    // well formed, and the fault is only visible in the sequence.
    let mut calm = Ambient::new(5);
    let cold_dark = [Cond::Cold, Cond::Dark, Cond::Quiet];
    let said: Vec<String> = (0..40)
        .filter_map(|i| calm.consider(&watch(i * 30, &cold_dark)).map(|s| s.text))
        .collect();

    for (i, line) in said.iter().enumerate() {
        let back = i.saturating_sub(20);
        assert!(
            !said[back..i].contains(line),
            "said twice inside twenty lines: {line:?}"
        );
    }
}

#[test]
fn the_quiet_notices_the_building_it_is_in() {
    // Idle is not the same as generic. A remark drawn while the building is
    // cold should sometimes be about the cold, or the pools are decoration.
    let mut calm = Ambient::new(9);
    let said = drive(&mut calm, &[Cond::Cold], 60, 200);
    assert!(
        said.iter()
            .any(|l| l.contains("cold") || l.contains("chill") || l.contains("Frost")),
        "two hundred remarks in a cold room, none of them about the cold"
    );

    let mut warm = Ambient::new(9);
    let said = drive(&mut warm, &[], 60, 200);
    assert!(
        !said.iter().any(|l| l.contains("Frost has formed")),
        "a room that is not cold noticed frost on the window"
    );
}

#[test]
fn every_fixture_gets_a_turn() {
    // A fixture that never speaks is dead weight, and a round-robin cursor that
    // stopped rotating would silently starve the ones at the back.
    let mut b = Building::new(11, recordings());
    let spoke: std::collections::BTreeSet<&str> = b
        .run(Duration::ZERO, Duration::from_secs(20), 3000)
        .into_iter()
        .map(|(_, s)| s.from)
        .collect();
    assert_eq!(
        spoke.len(),
        b.len(),
        "only {spoke:?} of {} fixtures ever spoke",
        b.len()
    );
}

// ---------------------------------------------------------------------------
// The couplings
// ---------------------------------------------------------------------------

#[test]
fn a_gust_sends_whatever_is_in_the_walls_bolting() {
    // The incident this module was designed around: the air handling purges,
    // and the thing living behind the panelling goes across the floor over
    // whatever is in the way. Neither fixture knows the other exists.
    let mut rat = Rat::new(4);
    let quiet_dark = [Cond::Dark, Cond::Quiet];

    // Get it out of hiding first — a rat nobody could have startled is not a
    // test of startling.
    let mut t = 0;
    let mut out = false;
    while t < 40_000 && !out {
        out = rat.consider(&watch(t, &quiet_dark)).is_some();
        t += 60;
    }
    assert!(out, "the rat never left cover in eleven hours");

    rat.notice(
        &did("air", &[Cond::Gusting, Cond::Loud], Salience::NORMAL),
        &watch(t, &quiet_dark),
    );

    // The fright jumps the queue: it does not wait for the rat's own timer.
    let bolt = rat
        .consider(&watch(t, &quiet_dark))
        .expect("a startled rat says something immediately");
    assert!(
        bolt.text.contains("bolts") || bolt.text.contains("at speed"),
        "expected a bolt, got {:?}",
        bolt.text
    );
}

#[test]
fn nothing_startles_a_rat_that_is_not_out() {
    // The control for the test above. A gust into an empty room is just a gust.
    let mut rat = Rat::new(4);
    let w = watch(0, &[Cond::Gusting]);
    rat.notice(&did("air", &[Cond::Gusting], Salience::NORMAL), &w);
    assert!(
        rat.consider(&w).is_none(),
        "a rat still in cover reacted to a gust it could not have heard"
    );
}

#[test]
fn an_unstable_supply_shows_in_the_lights_before_its_own_timer() {
    // A light bank is the cheapest instrument in the vault. This is what makes
    // the cascade read as one incident rather than three coincidences.
    let mut alone = Lighting::new(9);
    assert!(
        alone.consider(&watch(20, &[])).is_none(),
        "the lights spoke inside 20s with nothing prompting them"
    );

    let mut watched = Lighting::new(9);
    let w = watch(0, &[Cond::Unstable]);
    watched.notice(&did("power", &[Cond::Unstable], Salience::NORMAL), &w);
    assert!(
        watched.consider(&watch(20, &[Cond::Unstable])).is_some(),
        "the lights ignored an unstable supply"
    );
}

#[test]
fn what_leaks_for_long_enough_grows_something() {
    // The payoff at the end of the coolant loop's slow burn: a character that
    // ignored "the coolant pressure drops off its mark" meets, an hour later, a
    // bloom on the wall.
    let mut wet = Growth::new(21);
    let said = drive(&mut wet, &[Cond::Damp, Cond::Dark], 400, 60);
    assert!(
        said.iter()
            .any(|l| l.contains("bloom") || l.contains("growth")),
        "nothing grew in a damp dark building over six hours: {said:?}"
    );

    // And the control: a dry building stays clean however long it runs.
    let mut dry = Growth::new(21);
    let said = drive(&mut dry, &[], 400, 60);
    assert!(
        !said
            .iter()
            .any(|l| l.contains("bloom") || l.contains("growth")),
        "something grew in a dry building: {said:?}"
    );
}

#[test]
fn the_coolant_only_goes_wrong_under_strain() {
    // A building nobody is working in should not fall apart on its own — the
    // slow burn has to be *caused*, or it is just a timer.
    let mut steady = CoolantLoop::new(5);
    let said = drive(&mut steady, &[], 300, 80);
    assert!(
        !said
            .iter()
            .any(|l| l.contains("weep") || l.contains("pooled")),
        "the loop failed with nothing straining it: {said:?}"
    );

    let mut strained = CoolantLoop::new(5);
    let said = drive(&mut strained, &[Cond::Unstable, Cond::Working], 300, 80);
    assert!(
        said.iter()
            .any(|l| l.contains("under its mark") || l.contains("hammers")),
        "the loop held up under sustained strain: {said:?}"
    );
}

#[test]
fn what_the_rat_does_in_the_stores_is_found_later() {
    // Consequence arriving after the fact, in a room the rat left twenty
    // minutes ago. This is the coupling that makes the world feel like it kept
    // running while nobody was looking.
    let mut stores = Stores::new(13);
    let w = watch(0, &[]);
    stores.notice(
        &Stirring::new("rat", "Something is at work in the stores.", Salience::IDLE),
        &w,
    );
    let said = drive(&mut stores, &[], 300, 40);
    assert!(
        said.iter()
            .any(|l| l.contains("turns out to have") || l.contains("been got at")),
        "the damage was never found: {said:?}"
    );

    let mut untouched = Stores::new(13);
    let said = drive(&mut untouched, &[], 300, 40);
    assert!(
        !said
            .iter()
            .any(|l| l.contains("turns out to have") || l.contains("been got at")),
        "the stores found damage nothing had done: {said:?}"
    );
}

#[test]
fn the_board_only_names_a_gallery_that_something_went_wrong_in() {
    // The board reads the building's history rather than its state, and there
    // is no path by which it can name a system that has not actually failed.
    let mut b = Boards::new(31);
    let w = watch(0, &[]);
    b.notice(&did("coolant", &[Cond::Leaking], Salience::NORMAL), &w);
    let said = drive(&mut b, &[], 120, 30);
    assert!(
        said.iter()
            .any(|l| l.contains("against the coolant gallery")),
        "the fault never reached the board: {said:?}"
    );
    assert!(
        !said
            .iter()
            .any(|l| l.contains("against the main supply bus")),
        "the board invented a fault in a system that was fine: {said:?}"
    );
}

#[test]
fn a_board_with_nothing_wrong_files_nothing() {
    let mut b = Boards::new(31);
    let w = watch(0, &[]);
    // An idle vent noise is not a fault, and a board that logged it would be as
    // useless here as it is in a real building.
    b.notice(&did("air", &[Cond::Quiet], Salience::IDLE), &w);
    let said = drive(&mut b, &[], 120, 30);
    assert!(
        !said.iter().any(|l| l.contains("fault entry has come up")),
        "the board filed an idle noise as a fault: {said:?}"
    );
}

#[test]
fn a_building_nobody_left_a_message_in_says_nothing() {
    // The engine holds no announcements of its own. Given none, the tannoy is
    // silent — it does not fall back on words that would then be the engine's
    // opinion about somebody else's world.
    let mut quiet = Broadcast::new(2, Vec::new());
    assert_eq!(quiet.loaded(), 0);
    assert!(
        drive(&mut quiet, &[], 600, 200).is_empty(),
        "an unloaded tannoy invented something to say"
    );
}

#[test]
fn every_recording_plays_before_any_plays_twice() {
    // Drawing at random would repeat within a handful of plays however long
    // the list. Dealing from a shuffled bag is what buys a day between
    // repeats, and it is the whole reason the bag exists.
    let said: Vec<String> = (0..8).map(|i| format!("Recording number {i}.")).collect();
    let mut b = Broadcast::new(6, said.clone());
    let played = drive(&mut b, &[], 300, 400);
    assert!(
        played.len() >= said.len(),
        "only {} plays in a day and a half — too rare to test",
        played.len()
    );

    let first_round: std::collections::BTreeSet<&String> = played[..said.len()].iter().collect();
    assert_eq!(
        first_round.len(),
        said.len(),
        "a recording came round twice before the rest had played once: {:?}",
        &played[..said.len()]
    );
}

#[test]
fn a_recording_quotes_the_world_and_nothing_else() {
    // The words are the world's, verbatim. An engine that paraphrased them
    // would be authoring content it has no business authoring.
    let mut b = Broadcast::new(7, vec!["Mind the gap in the floor plates.".to_string()]);
    let played = drive(&mut b, &[], 400, 60);
    let first = played.first().expect("the tannoy never played anything");
    assert!(
        first.contains("\"Mind the gap in the floor plates.\""),
        "the recording was not quoted as given: {first:?}"
    );
}

#[test]
fn an_alarm_pushes_a_recording_back_and_never_pulls_it_forward() {
    // A standing notice about tidying your bench, played under an evacuation
    // tone, is funny exactly once — so an alarm defers the tannoy.
    //
    // The second half of the name is the bug this guards: deferring with
    // `Due::hold` *sets* the deadline, so a recording that was not due for
    // another forty minutes would have been dragged forward to meet the alarm,
    // which is the exact opposite of what was asked for.
    let when = |b: &mut Broadcast| (0..900u64).find(|i| b.consider(&watch(i * 10, &[])).is_some());

    for seed in 0..24u64 {
        let quiet = when(&mut Broadcast::new(seed, recordings()))
            .expect("the tannoy never played in two and a half hours");

        // The alarm lands in the ten seconds before the recording was going to
        // play, which is the case that matters and the only one in which a
        // deferral is observable at all.
        let mut under = Broadcast::new(seed, recordings());
        under.notice(
            &did("announce", &[Cond::Alarmed], Salience::URGENT),
            &watch(quiet.saturating_sub(1) * 10, &[]),
        );
        let alarmed = when(&mut under).expect("the tannoy went silent for good after an alarm");

        assert!(
            alarmed > quiet,
            "seed {seed}: the recording played over the alarm anyway, at {alarmed}"
        );
    }
}

#[test]
fn a_noisy_building_does_not_silence_the_tannoy() {
    // The other half of that deferral, and a bug it actually had: pushing back
    // by a full gap each time meant every deferral landed past the next one, so
    // a vault with anything wrong in it played four recordings in a whole day.
    // Letting the noise finish is all it has to do.
    let mut b = Broadcast::new(4, recordings());
    let mut played = 0;
    // A day, with something loud going off every ten minutes throughout.
    for i in 0..8640u64 {
        let t = i * 10;
        let w = watch(t, &[]);
        if t % 600 == 0 {
            b.notice(&did("power", &[Cond::Unstable], Salience::URGENT), &w);
        }
        played += usize::from(b.consider(&w).is_some());
    }
    assert!(
        played >= 20,
        "only {played} recordings played in a day of a noisy building"
    );
}

#[test]
fn the_air_follows_the_compute_floor() {
    // The air does not know what compute is. It knows that something published
    // `Working`, which is the whole of the coupling.
    let mut air = AirHandling::new(77);
    let w = watch(0, &[]);
    air.notice(&did("compute", &[Cond::Working], Salience::IDLE), &w);
    let said = drive(&mut air, &[Cond::Working], 100, 40);
    assert!(
        said.iter()
            .any(|l| l.contains("steps up") || l.contains("running hard")),
        "the air never picked up under load: {said:?}"
    );
}

#[test]
fn the_clock_strikes_each_hour_exactly_once() {
    // The one fixture where a repeat would be a bug rather than merely dull.
    let mut chime = Chime::new(3);
    let mut struck = 0;
    // Ten past midnight, then a minute at a time for a little over three hours.
    for m in 0..200u64 {
        let secs = 600 + m * 60;
        let w = Watch {
            since_start: Duration::from_secs(m * 60),
            wall: UNIX_EPOCH + Duration::from_secs(secs),
            conds: Vec::new(),
        };
        if let Some(s) = chime.consider(&w) {
            if s.text.contains("strikes") {
                struck += 1;
            }
        }
    }
    assert_eq!(
        struck, 3,
        "three hours went by and the clock struck {struck} times"
    );
}

// ---------------------------------------------------------------------------
// A look at it running
// ---------------------------------------------------------------------------

fn clock(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

fn mark(s: &Stirring) -> &'static str {
    match (s.salience.preempts(), s.salience >= Salience::NORMAL) {
        (true, _) => "!!",
        (_, true) => " *",
        _ => "  ",
    }
}

/// A day of the tannoy, playing the vault's own recordings.
///
/// Every other printed run above uses throwaway lines, because the engine's
/// tests have no business depending on the shipped world. This one loads the
/// real map and plays what the Creator actually left, which is the only way to
/// see the generalisation working end to end: the fixture is the same fixture,
/// and all of the character comes from the file.
#[test]
fn what_the_creator_left_on_the_tannoy() {
    let set = npc_map::MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
        .expect("the shipped vault must load");
    let said = set
        .get("creators-vault")
        .expect("the building")
        .announcements
        .clone();

    println!(
        "\n=== the vault's {} standing recordings, over a day ===",
        said.len()
    );
    let mut b = Building::new(2027, said.clone());
    let mut heard = 0;
    for (t, s) in b.run(Duration::ZERO, Duration::from_secs(20), 4320) {
        if s.from == "broadcast" {
            heard += 1;
            println!("{} {}", clock(t), s.text);
        }
    }
    println!("--- {heard} played, out of {} loaded", said.len());
}

/// Six hours in the vault, printed.
///
/// Not an assertion — this is the thing you read to decide whether the building
/// feels like a place. Everything it prints came out of the fixtures reacting to
/// each other; none of it is a scripted sequence.
#[test]
fn a_shift_in_the_vault() {
    for seed in [1u64, 2, 3] {
        let mut b = Building::new(seed, recordings());
        let run = b.run(Duration::ZERO, Duration::from_secs(20), 1080);

        println!("\n=== seed {seed} — six hours, {} fixtures ===", b.len());
        for (t, s) in &run {
            println!("{} {} [{:<9}] {}", clock(*t), mark(s), s.from, s.text);
        }

        let distinct: std::collections::BTreeSet<&str> =
            run.iter().map(|(_, s)| s.text.as_str()).collect();
        println!(
            "--- {} stirrings, {} distinct, {} loud enough to interrupt",
            run.len(),
            distinct.len(),
            run.iter().filter(|(_, s)| s.salience.preempts()).count()
        );
    }
}

/// The incidents, hunted out of a long run and printed on their own.
///
/// A coupling is only worth having if you can see it happen. This walks a day
/// of building looking for the two chains the module was designed around and
/// prints each with what led into it.
#[test]
fn the_incidents() {
    let mut b = Building::new(1789, recordings());
    let run = b.run(Duration::ZERO, Duration::from_secs(15), 5760);

    println!("\n=== incidents found in twenty-four hours ===");
    let mut found = 0;
    for (i, (t, s)) in run.iter().enumerate() {
        // The rat incident: something frightened it, and it went over
        // something on its way out.
        let bolted = s.from == "rat" && s.text.contains("bolts");
        // The cascade: an alarm the address system raised off two standing
        // faults it never had to be told about.
        let alarm = s.from == "announce" && s.text.contains("alarm goes off");
        if !(bolted || alarm) {
            continue;
        }
        found += 1;
        let name = match bolted {
            true => "the rat incident",
            false => "the cascade",
        };
        println!("\n-- {name}, at {}", clock(*t));
        for (at, before) in run[i.saturating_sub(4)..=i].iter() {
            println!("   {} [{:<9}] {}", clock(*at), before.from, before.text);
        }
    }
    println!("\n{found} incidents in twenty-four hours");
}
