//! The `/` command catalogue and the act vocabulary as the console reads them,
//! served by the daemon so the console cannot drift from it.
//!
//! A console with its own list would offer commands the daemon rejects. The act
//! table is composed here from the host's tool list rather than serialised raw,
//! because the raw form has none of the modes, source, calibration or parameters
//! the console shows. Each act's condition names its stations from the loaded map,
//! and an act reaching many stations is counted rather than listed, with the full
//! list kept for the detail view where it fits.

use super::*;

/// `GET /v1/commands` — the `/` vocabulary, for the console's autocomplete.
///
/// Served rather than duplicated in the console, so the two cannot drift. A
/// console with its own list would offer a command the daemon rejects, and the
/// operator would read that as the character ignoring them.
pub async fn commands() -> Response {
    Json(json!({
        "commands": slash::CATALOG,
        "default": "say",
        "preempt_at": Salience::PREEMPT_AT,
    }))
    .into_response()
}

/// The act vocabulary, with its calibration examples, in the shape the console
/// reads it (the host's `GET /v1/tools`, through [`catalog`]).
///
/// **Not the catalog serialised raw.** It used to be, and three of the table's
/// five columns were therefore blank: `Tool` has no `modes`, no `source` and no
/// `calibrated` field, and the parameter modal read `t.parameters`, which did
/// not exist either — so the one view of the vocabulary an operator has showed a
/// name, a description, and three empty cells over an empty schema.
///
/// Composed here rather than added to `Tool` because these are facts about how
/// the daemon *offers* an act, not about the act: which modes reach it is
/// `Availability` resolved against each mode, and whether it is calibrated is
/// whether anybody wrote it examples. Putting them on the struct would be
/// caching four derived values next to the thing they derive from.
/// What each station is called, by the id an act attaches to.
///
/// Read off whatever worlds are loaded rather than restated in this file, so
/// renaming a part in the map renames it in the console and the two cannot
/// drift. Empty before any world is open, which is the honest answer: with no
/// map, nothing here knows what a station is called.
pub(super) type PartNames = std::collections::BTreeMap<String, String>;

/// "a" or "an", by the sound the name starts with.
///
/// A vowel letter is the rule almost all the time, and the exceptions in
/// English are about *sound* rather than spelling — "a universal", "an hour".
/// Neither shape occurs among the parts, so the letter is the whole rule here;
/// a part that needed otherwise would be authored with its own article, which
/// the caller already honours.
pub(super) fn article_for(name: &str) -> &'static str {
    match name.chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

/// How many stations a condition names before it stops naming them.
///
/// Two fits a line and is worth reading. Six does not: it wraps, it pushes
/// every other column out of shape, and by the third name the reader has
/// stopped taking any of them in. Past this the summary gives the count, and
/// the full list goes where the detail view has room for it.
pub(super) const NAME_AT_MOST: usize = 2;

/// Each station an act attaches to, in the words the map uses.
pub(super) fn named_stations(at: &[&str], names: &PartNames) -> Vec<String> {
    at.iter()
        .map(|id| match names.get(*id) {
            // The map's own name, with the article a reader expects. A part
            // authored with its own article keeps it — "the appraisal bench"
            // is one bench and saying "a the appraisal bench" would be worse
            // than saying nothing.
            Some(n) if n.starts_with("the ") => n.clone(),
            Some(n) => format!("{} {n}", article_for(n)),
            // An id no loaded world places. Said plainly rather than dressed
            // up: it means the act is unreachable, and somebody should notice.
            None => format!("`{id}` (nothing places one)"),
        })
        .collect()
}

/// The one-line answer to *where do I have to be*.
///
/// **Named when naming helps, counted when it does not.** "Standing at the
/// station that carries it" was true of every station act at once and told an
/// operator hunting a missing one nothing — so a condition names its station.
/// But an act reaching six of them produced a sentence longer than the column
/// it sat in, and a reader made to parse six noun phrases to learn "a
/// workstation" has been given less rather than more.
///
/// The full list is not lost: it goes in `at_named`, which the detail view
/// renders and which has the room for it.
pub(super) fn where_it_is(at: &[&str], names: &PartNames) -> Option<String> {
    if at.is_empty() {
        return None;
    }
    let named = named_stations(at, names);
    Some(match named.len() > NAME_AT_MOST {
        false => format!("standing at {}", npc_map::text::list_or(&named)),
        true => format!("standing at any of {} stations", named.len()),
    })
}

/// The act vocabulary as the console reads it, with each station named from
/// the worlds this daemon hosts — npcd's share of the host's `/v1/tools`.
pub fn catalog(s: &Authored) -> Value {
    let mut part_names = PartNames::new();
    if let Some(rt) = s.runtime.as_ref() {
        for world_id in rt.hosted.ids() {
            if let Some(hosted) = rt.hosted.get(&world_id) {
                hosted.read(|w| {
                    for area in w.map().areas() {
                        for node in &area.nodes {
                            for (part, _) in w.map().parts_at(node) {
                                part_names
                                    .entry(part.id.clone())
                                    .or_insert_with(|| part.name.clone());
                            }
                        }
                    }
                });
            }
        }
    }
    describe_catalog(&part_names)
}

/// The catalogue as the console reads it.
///
/// Separated from the route because it is the part with the decisions in it and
/// takes only the station names — so a test asserts what an operator will
/// actually see, against a real name map, without standing a daemon up.
pub(super) fn describe_catalog(part_names: &PartNames) -> Value {
    use crate::engine::tools::{self, Availability, Mode};

    const MODES: [Mode; 2] = [Mode::Physical, Mode::InstantMessage];

    let described: Vec<Value> = tools::CATALOG
        .iter()
        .map(|t| {
            // **Which channels the act can reach at all**, which is not the
            // same question `for_mode` answers. That one builds the *prompt*, so
            // it drops everything conditional on the situation — a `Nearby` act
            // is absent from it in every mode, and reading modes off it reported
            // that `tell` works nowhere.
            //
            // A condition that is not about the channel does not narrow the
            // channel: needing company, or a body, is true down a phone line as
            // much as face to face. Those conditions are reported in `needs`,
            // where an operator can act on them.
            let modes: Vec<&str> = MODES
                .iter()
                .filter(|m| match t.availability {
                    // Where a body is standing, like who is standing with it,
                    // is not a fact about the channel.
                    Availability::Always
                    | Availability::Nearby
                    | Availability::PersonNearby
                    | Availability::PersonAmongOthers
                    | Availability::PersonInEarshot
                    | Availability::Embodied
                    | Availability::AwayFromHome
                    // What the hands hold is where the body is, not the channel.
                    | Availability::Holding
                    // Being at the lift, or in it, is where the body is standing,
                    // not a fact about the channel — said in `needs` instead.
                    | Availability::AtLift
                    | Availability::AtLanding
                    // A grant is the carried mission's, not the channel's.
                    | Availability::Granted
                    // What a personality reaches is the character's.
                    | Availability::Reached
                    // A conversation's own act is the thread's, whatever its channel.
                    | Availability::OnThread => true,
                    Availability::PhysicalOnly => **m == Mode::Physical,
                    Availability::MessagingOnly => m.remote(),
                    Availability::Pictorial => m.carries_pictures(),
                    // Standing next to the right thing is not a fact about the
                    // channel, so it narrows nothing here and is said in
                    // `needs` instead.
                    Availability::AtPart => true,
                })
                .map(|m| m.as_wire())
                .collect();

            // The schema the model actually sees, in the JSON Schema shape the
            // modal renders. Built from the same `params` the grammar compiles
            // from, so the two cannot disagree about what an act takes.
            let properties: serde_json::Map<String, Value> = t
                .params
                .iter()
                .map(|p| {
                    let mut prop = tools::wire_type(p.ty);
                    prop["description"] = json!(p.description);
                    if let Some(vs) = tools::fixed_values(t.name, p.name) {
                        prop["enum"] = json!(vs);
                    } else if let Some(c) = tools::live_choice(t.name, p.name) {
                        // A live set has no fixed membership to publish — what
                        // it admits is a fact about a room, and saying so is
                        // more use to an operator than a list that would be
                        // wrong everywhere except where it was taken.
                        prop["x-bound-to"] = json!(format!("{c:?}"));
                    }
                    (p.name.to_string(), prop)
                })
                .collect();
            let required: Vec<&str> = t
                .params
                .iter()
                .filter(|p| p.required)
                .map(|p| p.name)
                .collect();

            json!({
                "name": t.name,
                "category": t.category,
                "description": t.description,
                "plane": t.plane,
                "modes": modes,
                "needs": match t.availability {
                    Availability::Always => Value::Null,
                    Availability::Nearby => json!("somebody else here"),
                    Availability::PersonNearby => json!("a person here"),
                    Availability::PersonAmongOthers => json!("a person here, and somebody else"),
                    Availability::PersonInEarshot => {
                        json!("a person within earshot, or the field")
                    }
                    Availability::Embodied => json!("a body"),
                    Availability::AwayFromHome => json!("being somewhere that is not home"),
                    Availability::Holding => json!("holding an order or a station"),
                    Availability::PhysicalOnly => json!("being present"),
                    Availability::MessagingOnly => json!("being at a distance"),
                    Availability::Pictorial => json!("a channel that carries pictures"),
                    // Named, not generic. "Standing at the station that carries
                    // it" is true of every station act and tells an operator
                    // hunting a missing one exactly nothing.
                    Availability::AtPart => match where_it_is(t.at, part_names) {
                        Some(where_) => json!(where_),
                        // An `AtPart` act naming no station is a contradiction
                        // the catalogue's own tests refuse; said here rather
                        // than unwrapped so a route never panics over it.
                        None => json!("standing at a station"),
                    },
                    Availability::AtLift => json!("at the lift, with it on another floor"),
                    Availability::AtLanding => json!("at the lift, wherever the car is"),
                    Availability::Granted => {
                        json!("a mission that grants it, in that mission's room in a mirrored place")
                    }
                    Availability::Reached => json!("a personality that reaches it"),
                    Availability::OnThread => json!("a conversation's decision"),
                },
                // Every station, named, however many there are. The `needs`
                // line above summarises for the table; this is what the detail
                // view shows, where a list of six costs nothing.
                "at_named": named_stations(t.at, part_names),
                "source": "builtin",
                // Calibration is examples: a tool with none selects measurably
                // worse, which is exactly what the column is for.
                "calibrated": !t.examples.is_empty(),
                "examples": t.examples.len(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        })
        .collect();

    let uncalibrated = tools::CATALOG
        .iter()
        .filter(|t| t.examples.is_empty())
        .count();

    json!({
        "tools": described,
        "uncalibrated": uncalibrated,
        "acts_per_turn": tools::ACTS_PER_TURN,
        "examples": tools::CATALOG.iter().map(|t| t.examples.len()).sum::<usize>(),
    })
}
