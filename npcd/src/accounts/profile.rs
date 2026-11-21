//! Defines the shape of an author profile, keeps stored profiles in that shape, and
//! checks a patch before it is merged.
//!
//! Only the listed fields exist, so a stored key that the list does not name is
//! removed on the next write. A patch is checked for type before any of it is
//! merged, because normalising a wrong-typed field blanks it, which would answer a
//! write with success while erasing the author's text. The gender field holds a
//! closed set, since a character writes prose from it. Superseded revisions are
//! capped at a fixed number, because the whole account record is resident on every
//! member.

use super::*;

/// Every field a profile has. There are no others.
///
/// The registry preserves unknown keys on purpose — an authored world is
/// hand-edited and must survive a round trip through a program that has not
/// heard of half of it. A profile is the opposite: its shape is defined here,
/// so a key on disk that this list does not name is not a field, it is
/// residue, and [`normalise_profile`] removes it on the next write.
pub(super) const PROFILE_FIELDS: [&str; 5] =
    ["description", "gender", "history", "turn_index", "revision"];

/// Force a stored profile into the shape above: drop what is not a field, fill
/// in what is missing, and blank a `gender` that is not a value it may hold.
///
/// Called on every sign-in, so a record written by an older shape corrects
/// itself the next time its owner appears rather than needing anyone to go and
/// find it.
pub(super) fn normalise_profile(profile: &mut Value) {
    let Some(map) = profile.as_object_mut() else {
        *profile = json!({});
        return normalise_profile(profile);
    };

    map.retain(|k, _| PROFILE_FIELDS.contains(&k.as_str()));

    for text in ["description", "gender", "history"] {
        if !map.get(text).is_some_and(Value::is_string) {
            map.insert(text.into(), json!(""));
        }
    }
    for count in ["turn_index", "revision"] {
        if !map.get(count).is_some_and(Value::is_u64) {
            map.insert(count.into(), json!(0));
        }
    }
    if !map["gender"].as_str().is_some_and(gender_ok) {
        map.insert("gender".into(), json!(""));
    }
}

/// The values `gender` may hold, plus blank for not-yet-chosen.
///
/// A closed set rather than free text because a character reads this field and
/// writes prose from it — "she" or "he" follows from it directly. Free text
/// would make that a guess, and a guess is the one thing an NPC should not be
/// doing about the person it is talking to.
pub const GENDERS: [&str; 2] = ["Male", "Female"];

/// Whether a submitted `gender` is one this profile can hold.
pub fn gender_ok(v: &str) -> bool {
    v.is_empty() || GENDERS.contains(&v)
}

/// The profile fields a caller writes, all of them free text.
pub(super) const AUTHORED_TEXT: [&str; 2] = ["description", "history"];

/// How many superseded revisions are kept.
///
/// Bounded because the whole account record is resident, on every member, and
/// every save appends a full copy of the outgoing profile — paragraphs of it.
/// Unbounded, an author who edits often eventually carries their entire writing
/// history resident, forever, for a feature that reaches back a few steps.
///
/// **Twenty, not two hundred.** The bound was set generously on the reasoning
/// that it should never be the thing a person notices — but the cost is not
/// linear in it. Every save writes the account whole, and the history is the
/// largest thing in it, so a deeper history makes *every* save more expensive
/// rather than only the rare walk back through one. Twenty is still far past
/// any real use of undo on a profile description, and it is the difference
/// between a save that is instant and one that is not.
pub(super) const KEEP_REVISIONS: usize = 20;

/// One line of a revision, for choosing between them.
///
/// Enough to recognise which edit this was, not to read it. Cut on a character
/// boundary — `&s[..N]` panics mid-codepoint, and a profile is the one place
/// somebody is most likely to have written their name in their own script.
pub(super) fn preview(s: &str) -> String {
    const MAX: usize = 90;
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(MAX) {
        None => flat,
        Some((cut, _)) => format!("{}…", flat[..cut].trim_end()),
    }
}

/// Why a profile patch was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum PatchError {
    /// A field the caller may write was sent as something other than a string.
    NotText(&'static str),
    /// `gender` was a string, but not one of the values it may hold.
    BadGender,
}

/// Check a profile patch before any of it is merged.
///
/// It runs over the whole patch rather than one field, because the failure it
/// prevents is the same in every case and is silent: [`normalise_profile`]
/// coerces a wrong-typed field to `""` on its way to disk, so `{"gender": 3}`
/// or `{"description": 42}` would answer `200 OK` while destroying whatever the
/// author had written there. Repairing a record read off disk is what that
/// coercion is for; it must never be how a live request is handled.
pub fn check_patch(patch: &Map<String, Value>) -> Result<(), PatchError> {
    for field in AUTHORED_TEXT {
        if patch.get(field).is_some_and(|v| !v.is_string()) {
            return Err(PatchError::NotText(field));
        }
    }
    match patch.get("gender") {
        None => Ok(()),
        // Checked for type first: `null` and `3` are not "an invalid gender",
        // they are the wrong kind of thing, and saying so is the difference
        // between a caller fixing the value and a caller fixing the request.
        Some(v) => match v.as_str() {
            None => Err(PatchError::NotText("gender")),
            Some(g) if !gender_ok(g) => Err(PatchError::BadGender),
            Some(_) => Ok(()),
        },
    }
}
