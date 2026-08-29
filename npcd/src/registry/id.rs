//! What may become a file name.
//!
//! This is the whole security boundary of the registry, so it is deliberately
//! the most restrictive thing in the crate: an id is lowercase ASCII letters,
//! digits and hyphens, and nothing else exists.
//!
//! The reason it is an **allowlist** rather than a list of things to reject is
//! that the rejection list is not knowable. `..` and `/` are the obvious two;
//! then it is `\` on Windows, `:` for NTFS alternate data streams, a trailing
//! dot or space that Win32 silently strips (so `x ` and `x` are the same file),
//! NUL bytes truncating a path in the syscall layer, Unicode that normalises
//! onto an existing name, and right-to-left overrides that make a name render
//! as something it is not. An allowlist of 37 characters answers all of them at
//! once and keeps answering them for the ones nobody has thought of yet.
//!
//! Reads never reach this code at all — a URL id is a key into the in-memory
//! map, and an unknown key is a 404 rather than a path. This runs only when
//! something is about to be **written**, which is the one moment an id becomes
//! a file name.

use std::fmt;

/// Longest id we will write. Well inside every filesystem's limit even after
/// the extension, and long enough that no real world or personality needs to be
/// abbreviated.
const MAX_LEN: usize = 64;

/// Names Win32 resolves to devices rather than files, in any directory and
/// **with any extension** — `con.yaml` opens the console, it does not create a
/// file. Every one of them is pure ASCII alphanumeric, so the allowlist above
/// lets them straight through; this is the one class it cannot catch.
///
/// Checked on every platform, not just Windows. A registry written on Linux and
/// read on Windows would otherwise carry a landmine across, and the cost of
/// refusing a world called `aux` everywhere is nil.
const RESERVED: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdError {
    Empty,
    TooLong(usize),
    BadChar(char),
    EdgeHyphen,
    Reserved,
}

impl fmt::Display for IdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdError::Empty => write!(f, "an id cannot be empty"),
            IdError::TooLong(n) => write!(f, "an id may be at most {MAX_LEN} characters, got {n}"),
            IdError::BadChar(c) => write!(
                f,
                "`{c}` is not allowed in an id — use lowercase letters, digits and hyphens"
            ),
            IdError::EdgeHyphen => write!(f, "an id cannot start or end with a hyphen"),
            IdError::Reserved => write!(
                f,
                "that name is reserved by the operating system and cannot be a file"
            ),
        }
    }
}

impl std::error::Error for IdError {}

/// Validate an id that is about to become `<id>.yaml` inside the registry
/// directory.
pub fn check(id: &str) -> Result<(), IdError> {
    if id.is_empty() {
        return Err(IdError::Empty);
    }
    if id.len() > MAX_LEN {
        return Err(IdError::TooLong(id.len()));
    }
    if let Some(c) = id
        .chars()
        .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-'))
    {
        return Err(IdError::BadChar(c));
    }
    // A leading hyphen makes an id look like a flag to anything that later
    // shells out; a trailing one is invisible in a listing.
    if id.starts_with('-') || id.ends_with('-') {
        return Err(IdError::EdgeHyphen);
    }
    if RESERVED.contains(&id) {
        return Err(IdError::Reserved);
    }
    Ok(())
}

// There was a `from_name` here, turning a typed display name into a candidate
// id. It existed for the "+ New world" and "+ New personality" buttons, and
// those are gone: worlds and personalities are files an author writes into the
// mind, so an id is chosen by naming a file rather than derived from a form.
// Nothing suggests ids any more, so nothing needs to.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_ids_are_accepted() {
        for ok in ["ardh", "hill-villages", "world-2", "a", "x9"] {
            assert_eq!(check(ok), Ok(()), "{ok}");
        }
    }

    /// The traversal cases, spelled out. None of these can reach the filesystem
    /// through the read path at all — they are here because the write path is
    /// the one place an id becomes a name, and this is that gate.
    #[test]
    fn nothing_can_escape_the_directory() {
        for bad in [
            "..",
            "../etc/passwd",
            "..\\..\\windows",
            "/etc/passwd",
            "a/b",
            "a\\b",
            "c:",
            "c:/x",
            ".hidden",
            "x.yaml",
            "x.",
            "x ",
            " x",
        ] {
            assert!(check(bad).is_err(), "accepted `{bad}`");
        }
    }

    /// NTFS alternate data streams, NUL truncation, and names that render as
    /// something other than what they are.
    #[test]
    fn the_exotic_filesystem_tricks_are_shut_out_too() {
        for bad in [
            "x:stream",           // NTFS ADS — writes beside the file, invisibly
            "x\u{0}y",            // NUL truncates in the syscall layer
            "x\u{202E}lmth.evil", // RTL override: renders reversed
            "café",               // non-ASCII: normalisation collisions
            "Ardh",               // uppercase: case-insensitive FS collision
            "x\ny",
            "x\ty",
        ] {
            assert!(check(bad).is_err(), "accepted `{}`", bad.escape_debug());
        }
    }

    /// The one class the allowlist genuinely cannot catch, because every one of
    /// these is already lowercase alphanumeric.
    #[test]
    fn windows_device_names_are_refused_on_every_platform() {
        for bad in ["con", "prn", "aux", "nul", "com1", "com9", "lpt1", "lpt9"] {
            assert_eq!(check(bad), Err(IdError::Reserved), "accepted `{bad}`");
        }
        // Only the exact name is a device; these are ordinary files.
        for ok in ["console", "connor", "com0", "com10", "nullify", "auxiliary"] {
            assert_eq!(check(ok), Ok(()), "refused `{ok}`");
        }
    }

    #[test]
    fn hyphens_may_not_sit_on_the_edges() {
        for bad in ["-x", "x-", "-", "--"] {
            assert_eq!(check(bad), Err(IdError::EdgeHyphen), "accepted `{bad}`");
        }
        assert_eq!(check("a-b-c"), Ok(()));
    }

    #[test]
    fn length_is_bounded() {
        assert_eq!(check(&"a".repeat(MAX_LEN)), Ok(()));
        assert_eq!(
            check(&"a".repeat(MAX_LEN + 1)),
            Err(IdError::TooLong(MAX_LEN + 1))
        );
    }

    /// The ids the mind actually holds all pass, which is the case that matters
    /// most: a file already on disk whose name this gate would refuse is a
    /// document the console can read and never save back.
    #[test]
    fn every_authored_id_in_use_is_writable() {
        for id in [
            "battle-cities",
            "earth",
            "sandbox",
            "commander",
            "loyal-soldier",
            "anchor-the-protector",
            "cindy-tan",
            "babel-the-polyglot-parrot",
        ] {
            assert_eq!(check(id), Ok(()), "`{id}` is on disk and unsaveable");
        }
    }
}
