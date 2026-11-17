//! A push, as GitHub's webhook tells it (`docs/battlecities_app.md` §4.1).
//!
//! GitHub signs each delivery's body with the secret the webhook was made
//! with, as `X-Hub-Signature-256: sha256=<hex of HMAC-SHA256>`, and names what
//! happened in `X-GitHub-Event`. A delivery is believed only when its
//! signature is the body's under this gateway's copy of the secret, and it is
//! a push only when the event is `push` and it moved a branch to a commit —
//! not a tag, and not a branch's deletion.

use std::io::ErrorKind;
use std::path::Path;

use anyhow::{bail, Context};
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;

/// The header that carries a delivery's signature.
pub const SIGNATURE: &str = "x-hub-signature-256";

/// The header that names what a delivery is about.
pub const EVENT: &str = "x-github-event";

/// A branch of a repository moved to a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pushed {
    /// The repository's name, without its owner: `battle-cities`.
    pub repo: String,
    pub branch: String,
    pub commit: String,
}

/// The webhook's secret, from the file `cluster.deploy_on_push` names: its
/// bytes without the line end an editor leaves. A file that is not there is
/// no secret, and no delivery is believed; one that is there and empty is a
/// mistake, and stops the start.
pub fn secret(path: &Path) -> anyhow::Result<Option<Vec<u8>>> {
    let mut read = match std::fs::read(path) {
        Ok(read) => read,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            tracing::warn!(
                "hook: {} is not there, so no webhook delivery is believed",
                path.display()
            );
            return Ok(None);
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    while read.last().is_some_and(u8::is_ascii_whitespace) {
        read.pop();
    }
    if read.is_empty() {
        bail!("{} holds no secret", path.display());
    }
    Ok(Some(read))
}

/// The bytes `hex` spells; `None` when it is not hex of whole bytes.
fn unhex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) || !hex.is_ascii() {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).ok())
        .collect()
}

/// Whether `signature` — the [`SIGNATURE`] header — is `body`'s under
/// `secret`. Compared in constant time.
pub fn signed(secret: &[u8], body: &[u8], signature: &str) -> bool {
    let Some(want) = signature.trim().strip_prefix("sha256=").and_then(unhex) else {
        return false;
    };
    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(secret) else {
        return false;
    };
    mac.update(body);
    mac.verify_slice(&want).is_ok()
}

#[derive(Deserialize)]
struct Delivery {
    #[serde(rename = "ref")]
    moved: String,
    after: String,
    #[serde(default)]
    deleted: bool,
    repository: Repository,
}

#[derive(Deserialize)]
struct Repository {
    name: String,
}

/// The push `body` tells of, when `event` — the [`EVENT`] header — is a push
/// of a branch to a commit.
pub fn pushed(event: &str, body: &[u8]) -> Option<Pushed> {
    if event != "push" {
        return None;
    }
    let delivery: Delivery = serde_json::from_slice(body).ok()?;
    let branch = delivery.moved.strip_prefix("refs/heads/")?;
    let is_commit = delivery.after.len() == 40
        && delivery.after.bytes().all(|b| b.is_ascii_hexdigit())
        && delivery.after.bytes().any(|b| b != b'0');
    if delivery.deleted || !is_commit || branch.is_empty() {
        return None;
    }
    Some(Pushed {
        repo: delivery.repository.name,
        branch: branch.to_string(),
        commit: delivery.after,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const COMMIT: &str = "a744abc8d1f0c4b7e9a3552f6c1d8e0b9a7f6e5d";

    /// **The signature is GitHub's**: the secret, the body and the header of
    /// the example GitHub's own documentation gives.
    #[test]
    fn a_delivery_is_believed_by_its_signature() {
        let secret = b"It's a Secret to Everybody";
        let body = b"Hello, World!";
        let header = "sha256=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17";
        assert!(signed(secret, body, header));
        assert!(!signed(b"another secret", body, header));
        assert!(!signed(secret, b"Hello, World?", header));
        for bad in [
            "757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17",
            "sha1=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17",
            "sha256=757107ea",
            "sha256=zz",
            "sha256=7",
            "",
        ] {
            assert!(!signed(secret, body, bad), "{bad}");
        }
    }

    /// The secret is the file's bytes without its line end; a missing file
    /// is no secret, and an empty one is refused.
    #[test]
    fn the_secret_is_read_from_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("github-hook.key");
        assert_eq!(secret(&file).unwrap(), None);
        std::fs::write(&file, b"s3cret with spaces\r\n").unwrap();
        assert_eq!(secret(&file).unwrap(), Some(b"s3cret with spaces".to_vec()));
        std::fs::write(&file, b"\n").unwrap();
        let empty = secret(&file).unwrap_err();
        assert!(format!("{empty:#}").contains("holds no secret"));
    }

    fn delivery(moved: &str, after: &str, deleted: bool) -> Vec<u8> {
        json!({"ref": moved, "after": after, "deleted": deleted, "before": "0",
               "repository": {"name": "battle-cities", "full_name": "john-sharratt/battle-cities"}})
        .to_string()
        .into_bytes()
    }

    #[test]
    fn a_push_is_a_branch_moved_to_a_commit() {
        assert_eq!(
            pushed("push", &delivery("refs/heads/main", COMMIT, false)),
            Some(Pushed {
                repo: "battle-cities".into(),
                branch: "main".into(),
                commit: COMMIT.into(),
            })
        );
        let on_a_branch = pushed("push", &delivery("refs/heads/zen/x", COMMIT, false));
        assert_eq!(on_a_branch.unwrap().branch, "zen/x");
    }

    /// A tag, a deleted branch, another event and a body that is not a
    /// delivery are no push.
    #[test]
    fn what_is_not_a_push_of_a_branch_is_nothing() {
        let gone = "0000000000000000000000000000000000000000";
        assert_eq!(
            pushed("push", &delivery("refs/tags/v1", COMMIT, false)),
            None
        );
        assert_eq!(
            pushed("push", &delivery("refs/heads/main", gone, true)),
            None
        );
        assert_eq!(
            pushed("push", &delivery("refs/heads/main", gone, false)),
            None
        );
        assert_eq!(
            pushed("push", &delivery("refs/heads/main", "a744abc8", false)),
            None
        );
        assert_eq!(
            pushed("ping", &delivery("refs/heads/main", COMMIT, false)),
            None
        );
        assert_eq!(pushed("push", b"{}"), None);
        assert_eq!(pushed("push", b"not json"), None);
    }
}
