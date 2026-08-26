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
/// the extension, and long enough that no real world or archetype needs to be
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

/// Turn a human name into a candidate id. Lossy on purpose — the caller shows
/// the result and lets the author correct it, rather than the author guessing
/// what survived.
pub fn from_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_hyphen = true; // suppresses a leading hyphen
    for c in name.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            last_hyphen = false;
        } else if !last_hyphen {
            out.push('-');
            last_hyphen = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out.truncate(MAX_LEN);
    while out.ends_with('-') {
        out.pop();
    }
    // A world called "Con" is a perfectly reasonable thing to want, and the
    // author should not have to learn why Win32 disagrees. Disambiguate rather
    // than hand back a suggestion that the save path will refuse.
    if RESERVED.contains(&out.as_str()) {
        out.push_str("-1");
    }
    out
}

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

    #[test]
    fn a_name_becomes_a_usable_id() {
        assert_eq!(from_name("Ardh"), "ardh");
        assert_eq!(from_name("Hill Villages"), "hill-villages");
        assert_eq!(from_name("  The North!  "), "the-north");
        assert_eq!(from_name("a/b\\c"), "a-b-c");
        assert_eq!(from_name("café"), "caf");
        // Reserved names are disambiguated rather than handed back to be
        // rejected later.
        assert_eq!(from_name("CON"), "con-1");
    }

    /// Whatever `from_name` produces must pass `check`, or the GUI can suggest
    /// an id its own save will reject.
    #[test]
    fn a_derived_id_always_passes_the_gate() {
        let long = "a".repeat(200);
        for name in [
            "Ardh",
            "Hill Villages",
            "  spaces  ",
            "!!!weird!!!",
            "CON",
            "Nul",
            "com1",
            long.as_str(),
            "---",
            "",
            "🙂",
        ] {
            let id = from_name(name);
            if id.is_empty() {
                continue; // caller must ask for a name; nothing to write
            }
            assert!(
                check(&id).is_ok(),
                "`{name}` produced `{id}`, which the gate rejects: {:?}",
                check(&id)
            );
        }
    }
}
