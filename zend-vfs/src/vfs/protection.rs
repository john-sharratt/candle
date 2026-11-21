//! The directories the file tools may not touch.
//!
//! A repository's `secrets/` directory is protected by name at any depth, so a
//! new one is protected the day it is created. Its `.git` database is protected
//! as well: its config holds remote URLs with their credentials, which the git
//! layer redacts and a file read would not, and its refs are a database that
//! hand-parsing answers wrongly. Matching is case-insensitive, with trailing dots
//! and spaces dropped, so each spelling that Windows would open is refused.

use super::*;

// 4 MiB

/// Path segment marking a directory the tools may not touch.
///
/// A repository's secrets live in a `secrets/` directory — `web/secrets/auth.yaml`
/// for the gateway's sign-in config, `npcd/secrets/` for npcd's. One name, matched at any depth, so a new secrets directory is
/// protected the day it is created rather than the day someone remembers to add
/// it to a list.
pub const PROTECTED_SEGMENT: &str = "secrets";

/// A repository's git database, protected for two separate reasons.
///
/// **It leaks what the git layer redacts.** `.git/config` holds a remote's URL
/// verbatim, credentials and all — `url = https://user:token@host` — and
/// the git layer redacts exactly that before any remote reaches a tool response.
/// A `file_read` of the same file hands the token over whole, so leaving
/// `.git` readable makes the redaction decorative.
///
/// **And reading it gives wrong answers.** Measured on a live turn: asked how
/// many branches and tags a repository had, the model bypassed `git_refs`,
/// read `.git/packed-refs` and `.git/refs/heads` itself, and answered 53 and
/// 23 where the truth was 56 and 24 — loose refs, packed refs and symbolic
/// refs are a database, not a list, and hand-parsing them is wrong in ways
/// that look plausible. The git tools exist to answer those questions; this
/// closes the shortcut around them.
pub const PROTECTED_GIT_DIR: &str = ".git";

/// Every path segment the tools may not touch, matched at any depth.
pub const PROTECTED_SEGMENTS: [&str; 2] = [PROTECTED_SEGMENT, PROTECTED_GIT_DIR];

impl VfsStore {
    /// Whether a normalised key names something under a protected directory.
    ///
    /// Compared the way Windows resolves a name — case-insensitively, with
    /// trailing dots and spaces dropped — so `Secrets/`, `SECRETS/` and
    /// `secrets./` are the directory they open, not three unprotected ones.
    pub fn is_protected(norm: &str) -> bool {
        norm.split('/').any(|s| {
            let s = s.trim_end_matches(['.', ' ']);
            PROTECTED_SEGMENTS.iter().any(|p| s.eq_ignore_ascii_case(p))
        })
    }

    /// `Err(Forbidden)` for a protected key, `Ok(())` otherwise.
    pub(super) fn guard(norm: &str) -> Result<(), VfsError> {
        if Self::is_protected(norm) {
            return Err(VfsError::Forbidden(norm.to_string()));
        }
        Ok(())
    }
}
