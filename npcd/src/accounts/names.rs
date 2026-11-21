//! Validates the one name an author is addressed by, and suggests a first one.
//!
//! A unique name is typed into a tool call, so it is held to a narrow shape: ASCII
//! only, no whitespace, no punctuation that would need quoting, and no leading or
//! trailing separator, which keeps one author from impersonating another. The
//! suggestion is the local part of the author's email, tidied and padded where it
//! would otherwise fail that check, so a new account is never born with a name its
//! own setter would refuse.

use super::*;

/// Why a `unique_name` was refused.
///
/// The three are separated because the console does different things with them:
/// a shape complaint is text to put under the field, `Taken` asks for a
/// different name, and the rest is a failure the author cannot act on.
#[derive(Debug)]
pub enum NameError {
    Shape(&'static str),
    Taken,
    NoAccount,
    Io(anyhow::Error),
}

/// Bounds on the one string an NPC can be made to address.
///
/// Deliberately narrower than a display name. It is typed by a person into a
/// tool call, so it excludes whitespace and punctuation that would need
/// quoting, and it excludes the confusables that make one author's address
/// impersonable as another's — no leading or trailing separator, and nothing
/// outside ASCII.
pub(super) fn check_unique_name(name: &str) -> Result<(), NameError> {
    if name.len() < 2 {
        return Err(NameError::Shape("at least 2 characters"));
    }
    if name.len() > 24 {
        return Err(NameError::Shape("at most 24 characters"));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(NameError::Shape(
            "letters, digits, hyphen and underscore only",
        ));
    }
    let edge = |c: char| c == '-' || c == '_';
    if name.starts_with(edge) || name.ends_with(edge) {
        return Err(NameError::Shape(
            "cannot start or end with a hyphen or underscore",
        ));
    }
    Ok(())
}

/// A first suggestion for the name an author is known by here — the local part
/// of their email, tidied. Theirs to change immediately; the point is only that
/// a new account is not born blank.
pub(super) fn default_unique_name(id: &Identity) -> String {
    let base = id
        .email
        .split('@')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(&id.name);
    let cleaned: String = base
        .chars()
        .filter(|c| c.is_alphanumeric())
        .take(24)
        .collect();
    // A suggestion the setter would then refuse is worse than no suggestion —
    // the author sees a name in the field, presses Save, and is told it is
    // invalid without having typed anything. A one-letter local part is enough
    // to hit that, so pad rather than emit something `check_unique_name` bounces.
    match cleaned.len() {
        0 => "author".into(),
        1 => format!("{cleaned}-1"),
        _ => cleaned,
    }
}
