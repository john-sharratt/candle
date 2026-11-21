//! Reads and changes the pieces of craft that everyone in the world reads: the
//! library fields, the character sheets and the likenesses.
//!
//! Identifiers and categories are left out of what a Maker may change, since
//! everything else finds a piece through them. A personality's identity fields are
//! spliced into its YAML sheet in place, so the comments a person wrote survive. A
//! likeness is changed through its art direction, and the picture follows the
//! words. A mission handed in from a past year is brought back to the world's
//! present.

use super::*;

/// Reading and changing a piece of the craft.
///
/// Kept out of the main dispatch because these are the only acts addressed by
/// two arguments rather than a subject and a preposition — and because what
/// they change is not a document about the world but *how everybody in it
/// reads*, which is worth having in one place somebody can find.
/// What a Maker handing in `mission` is told of its time: back in the world's
/// present when the mission was worked in a year of the past — the year is the
/// mission's, and leaves with it.
pub(super) fn back_to_now(mission: &Mission) -> String {
    match mission.year {
        Some(year) => format!(
            "\nYou are back in the world's present; {year} and what came before it are behind \
             you again."
        ),
        None => String::new(),
    }
}

pub(super) fn library(
    hosted: &Hosted,
    body: &str,
    tool: &str,
    args: &Map<String, Value>,
) -> Outcome {
    let path = match library_path(args) {
        Ok(p) => p,
        Err(why) => return Outcome::Refused(why),
    };
    let shelf = hosted.shelf_of(body);
    if tool == "library_read" {
        return hosted.sim(
            |s| match s.bench.read_field(body, &path, &["template"], &*shelf) {
                Ok(t) => Outcome::Did(format!("{path}:\n{t}")),
                Err(why) => Outcome::Refused(why),
            },
        );
    }
    let (Some(field), Some(t)) = (text(args, "field"), verbatim(args, "text")) else {
        return Outcome::Refused(
            "Changing a piece of the craft is a field and what it says. Say both.".into(),
        );
    };
    if !LIBRARY_FIELDS.contains(&field.as_str()) {
        return Outcome::Refused(format!(
            "There is no {field} to change. What can be changed is {}.",
            LIBRARY_FIELDS.join(" or ")
        ));
    }
    hosted.with_sim(|s| {
        match s
            .bench
            .write_field(body, &path, &path, &[&field], &t, &*shelf)
        {
            Ok(p) => Outcome::Did(format!(
                "The {field} of {p} says what you wrote. It is how everybody here reads, once you \
             commit it."
            )),
            Err(why) => Outcome::Refused(why),
        }
    })
}

/// Where a likeness's art direction lives inside a personality.
pub(super) const PORTRAIT_PROMPT: &[&str] = &[crate::personality_portrait::FIELD, "prompt"];

/// The fields of a craft piece a Maker may change.
///
/// `id` and `category` are deliberately absent: an id is how every other
/// document refers to this one and a category is how the engine selects it, so
/// changing either from inside the world silently unhooks the piece from
/// everything that points at it.
pub(super) const LIBRARY_FIELDS: &[&str] = &["description", "template"];

/// The document a personality is.
pub(super) fn personality_path(who: &str) -> String {
    format!("personalities/{}.yaml", slug_of(who))
}

/// Write one field of a character's founding sheet, in place.
///
/// The `portrait_draw` pattern for the two identity fields — `anchor` (who they
/// are) and `wants` (what they are after). The sheet is a YAML document and the
/// field is spliced into it through [`crate::sim::bench::Benches::write_field`],
/// so the reasoning a person wrote around it in comments survives where a
/// round-trip would lose it. A world with no mind folder has no sheet to edit
/// and keeps the in-RAM draft the shipped handler always kept.
pub(super) fn author_sheet(
    hosted: &Hosted,
    body: &str,
    args: &Map<String, Value>,
    at: &[&str],
) -> Outcome {
    let (Some(who), Some(t)) = (text(args, "of"), text(args, "what")) else {
        return Outcome::Refused(
            "You meant to write who somebody is, but not whose, or not what.".into(),
        );
    };
    let shelf = hosted.shelf_of(body);
    hosted.with_sim(|s| {
        if !s.bench.has_root() {
            return match s.record.write(&who, body, &t) {
                Ok(n) => Outcome::Did(format!("You write into {n}: {t}")),
                Err(why) => Outcome::Refused(why),
            };
        }
        let path = personality_path(&who);
        match s.bench.write_field(body, &who, &path, at, &t, &*shelf) {
            Ok(p) => Outcome::Did(format!(
                "{who} is written into {p}: {t}. Nobody else sees it until you commit."
            )),
            Err(why) => Outcome::Refused(why),
        }
    })
}

/// The document a piece of the craft is, from the two halves that address it.
pub(super) fn library_path(args: &Map<String, Value>) -> Result<String, String> {
    let (Some(kind), Some(id)) = (text(args, "kind"), text(args, "id")) else {
        return Err("A piece of the craft is named by its library and its id. Say both.".into());
    };
    let dir = match kind.trim().to_lowercase().as_str() {
        "mood" | "moods" => "moods",
        "response" | "responses" => "responses",
        other => {
            return Err(format!(
                "There is no {other} library. There is `mood` and there is `response`."
            ))
        }
    };
    Ok(format!("{dir}/{}.yaml", slug_of(&id)))
}
