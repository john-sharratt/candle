//! Accounts — durable like worlds, publishable like nothing.
//!
//! An account is the local record of somebody the gateway has vouched for: who
//! they are, what they have chosen to be called here, and the author profile
//! their characters read. It survives a substrate wipe for the same reason a
//! world does, and by the same mechanism — a file read once at start and
//! written back on change.
//!
//! It is **not** in the repository, and that is the one thing about this module
//! worth being loud about. `npcd/.gitignore` excludes `accounts/`, because the
//! repository is public and an account carries a real email address and the
//! identity provider's stable subject id. Git history is the one place you
//! cannot quietly remove something from afterwards, so the decision has to hold
//! before the first write rather than after the first user.
//!
//! # Nothing here decides who you are
//!
//! Identity arrives already established, from [`crate::identity`], which got it
//! by verifying a signature. This module maps a verified subject to a record
//! and creates one the first time it sees a new subject. It never reads a
//! header and never takes a caller's word for a `sub`.

use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use web::auth::session::Identity;

use crate::registry::Registry;

/// The provider's subject id, as a file name.
///
/// Hashed rather than used directly, for two reasons that are both about not
/// depending on what a provider happens to emit. Google's `sub` is digits;
/// another provider's may be a UUID, base64 with `_` and uppercase, or an email
/// address — none of which survive the registry's deliberately narrow id
/// allowlist. A SHA-256 prefix is always `[0-9a-f]`, so every provider that
/// will ever exist already fits.
///
/// It also keeps the raw provider id out of a directory listing, which costs
/// nothing and is one less place for it to be read from.
fn file_id(sub: &str) -> String {
    let digest = Sha256::digest(sub.as_bytes());
    // 128 bits of a 256-bit digest. Collisions are not a security boundary here
    // — the subject inside the file is authoritative and is checked on load —
    // so this is about a name of workable length.
    digest[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// The account store.
pub struct Accounts {
    reg: Registry,
}

impl Accounts {
    pub fn load(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        Ok(Self {
            reg: Registry::load("account", dir)?,
        })
    }

    pub fn len(&self) -> usize {
        self.reg.len()
    }

    /// The record for a verified identity, created on first sight.
    ///
    /// Fields divide into two kinds and they are treated differently on every
    /// call. **The provider owns** `email`, `display` and `avatar_url` — they
    /// are refreshed from the assertion each time, because the provider is the
    /// authority on them and a stale email here would be a lie the daemon told
    /// itself. **The author owns** `unique_name` and `profile` — they are
    /// written here and never overwritten by a sign-in, or a name someone chose
    /// would be silently reverted every time they logged in.
    pub fn upsert(&mut self, id: &Identity, now_ms: u64) -> anyhow::Result<Value> {
        let key = file_id(&id.sub);

        let existing = self.reg.get(&key).map(|r| r.body.clone());
        let mut body = existing.clone().unwrap_or_else(|| {
            json!({
                "sub": id.sub,
                "provider": "google",
                "created_ms": now_ms,
                // A display name the author can change without touching the
                // one the provider gave them.
                "unique_name": default_unique_name(id),
                "profile": {
                    "description": "",
                    "gender": "—",
                    "pronouns": "",
                    "history": "",
                    "turn_index": 0,
                    "revision": 0
                }
            })
        });

        if let Some(map) = body.as_object_mut() {
            // The provider's fields, refreshed every time.
            map.insert("email".into(), json!(id.email));
            map.insert("display".into(), json!(id.name));
            map.insert(
                "avatar_url".into(),
                if id.picture.is_empty() {
                    Value::Null
                } else {
                    json!(id.picture)
                },
            );
            // The subject is the identity of the record. A file whose `sub` has
            // drifted from its name is a corrupted record, not a rename.
            map.insert("sub".into(), json!(id.sub));
        }

        // Only write when something actually changed. A GET of `/v1/me` on
        // every page load must not rewrite a file and dirty a working tree.
        if existing.as_ref() != Some(&body) {
            self.reg.put(&key, body.clone())?;
        }
        Ok(with_public_id(&key, &body))
    }

    /// Revise the profile: append a new turn, tombstone the previous one.
    ///
    /// It does not rewrite, and that is the whole point. An NPC that gathered
    /// your profile last month attended over what it said *then*, and its
    /// memory of the conversation cites that text. Overwriting would leave the
    /// citation pointing at words you never had, so the superseded revision
    /// stays readable in `profile_history` and only stops being *current*.
    ///
    /// Only the profile sub-object is touched: a `PUT /v1/me/profile` that
    /// could reach `sub` or `email` would be a way to become somebody else.
    pub fn put_profile(
        &mut self,
        sub: &str,
        patch: &Value,
        now_ms: u64,
    ) -> anyhow::Result<Option<Value>> {
        let key = file_id(sub);
        let Some(mut body) = self.reg.get(&key).map(|r| r.body.clone()) else {
            return Ok(None);
        };

        {
            let Some(map) = body.as_object_mut() else {
                anyhow::bail!("account {key} is not an object");
            };
            let mut profile = map.get("profile").cloned().unwrap_or_else(|| json!({}));

            // The outgoing revision, stamped with the moment it stopped being
            // the answer. A reader with a turn index can then tell which text
            // was live when a given turn was gathered.
            let mut retired = profile.clone();
            if let Some(old) = retired.as_object_mut() {
                old.insert("tombstoned_ms".into(), json!(now_ms));
            }

            if let (Some(dst), Some(src)) = (profile.as_object_mut(), patch.as_object()) {
                for (k, v) in src {
                    // The store owns both counters; a caller that could set
                    // `revision` could make two turns claim the same number.
                    if k != "revision" && k != "tombstoned_ms" {
                        dst.insert(k.clone(), v.clone());
                    }
                }
                let next = dst.get("revision").and_then(|r| r.as_u64()).unwrap_or(0) + 1;
                dst.insert("revision".into(), json!(next));
                // `turn_index` names the live profile turn. The profile's turns
                // are the revisions in this file, so it tracks the revision
                // rather than being a second, independently-drifting counter.
                dst.insert("turn_index".into(), json!(next));
            }

            let mut history = match map.remove("profile_history") {
                Some(Value::Array(v)) => v,
                _ => Vec::new(),
            };
            history.push(retired);
            map.insert("profile_history".into(), Value::Array(history));
            map.insert("profile".into(), profile);
        }

        self.reg.put(&key, body.clone())?;
        Ok(Some(with_public_id(&key, &body)))
    }

    /// Every revision of the profile, newest first, live one included.
    ///
    /// The live revision is marked and carries no `tombstoned_ms`; that is the
    /// only difference between it and the rest, because a tombstoned turn is
    /// still readable — it is superseded, not deleted.
    pub fn profile_history(&self, sub: &str) -> Option<Vec<Value>> {
        let body = &self.reg.get(&file_id(sub))?.body;

        let mut out = Vec::new();
        if let Some(live) = body.get("profile") {
            let mut v = live.clone();
            if let Some(m) = v.as_object_mut() {
                m.insert("live".into(), json!(true));
            }
            out.push(v);
        }
        if let Some(Value::Array(past)) = body.get("profile_history") {
            for v in past.iter().rev() {
                let mut v = v.clone();
                if let Some(m) = v.as_object_mut() {
                    m.insert("live".into(), json!(false));
                }
                out.push(v);
            }
        }
        Some(out)
    }

    /// Change the name characters know this author by.
    ///
    /// This is the one author-owned field that is not a matter of taste. The
    /// profile is prose an NPC reads; `unique_name` is an *address* — a tool
    /// that sends something to a person names them by it — so two accounts
    /// sharing one would make a target ambiguous rather than merely confusing.
    /// It is checked for shape and for uniqueness, and a clash is the caller's
    /// to resolve, not this function's to paper over by appending a digit.
    pub fn put_unique_name(&mut self, sub: &str, name: &str) -> Result<Value, NameError> {
        let name = name.trim();
        check_unique_name(name)?;

        let key = file_id(sub);
        // Uniqueness is case-insensitive: two authors called `Wren` and `wren`
        // are the same address to anyone typing it.
        let folded = name.to_ascii_lowercase();
        let taken = self.reg.iter().any(|r| {
            r.id != key
                && r.body
                    .get("unique_name")
                    .and_then(|v| v.as_str())
                    .is_some_and(|n| n.to_ascii_lowercase() == folded)
        });
        if taken {
            return Err(NameError::Taken);
        }

        let mut body = self
            .reg
            .get(&key)
            .map(|r| r.body.clone())
            .ok_or(NameError::NoAccount)?;
        let Some(map) = body.as_object_mut() else {
            return Err(NameError::Io(anyhow::anyhow!(
                "account {key} is not an object"
            )));
        };
        map.insert("unique_name".into(), json!(name));

        self.reg.put(&key, body.clone()).map_err(NameError::Io)?;
        Ok(with_public_id(&key, &body))
    }

    /// What the console sees for a verified identity, without creating anything.
    pub fn get(&self, sub: &str) -> Option<Value> {
        let key = file_id(sub);
        self.reg.get(&key).map(|r| with_public_id(&key, &r.body))
    }
}

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
fn check_unique_name(name: &str) -> Result<(), NameError> {
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

/// The wire shape.
///
/// `user_id` is the file key, not the provider's subject — the console needs a
/// stable handle for a user and has no business holding the provider's id for
/// one. The `sub` never leaves this process.
fn with_public_id(key: &str, body: &Value) -> Value {
    let mut out = body.clone();
    if let Some(map) = out.as_object_mut() {
        map.remove("sub");
        // Superseded revisions have their own endpoint. Shipping the whole
        // history on every `/v1/me` would grow the page-load payload by one
        // profile per edit the author has ever made, to render a chip.
        map.remove("profile_history");
        map.insert("user_id".into(), json!(format!("u_{}", &key[..8])));
    }
    out
}

/// A first suggestion for the name an author is known by here — the local part
/// of their email, tidied. Theirs to change immediately; the point is only that
/// a new account is not born blank.
fn default_unique_name(id: &Identity) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "npcd-accounts-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn ident(sub: &str, email: &str, name: &str) -> Identity {
        Identity {
            sub: sub.into(),
            email: email.into(),
            name: name.into(),
            picture: String::new(),
            exp: 9_999_999_999,
        }
    }

    #[test]
    fn a_first_sight_creates_an_account_that_survives_a_restart() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        let me = a
            .upsert(&ident("google-1", "wren@example.com", "Wren S"), 1_000)
            .unwrap();
        assert_eq!(me["email"], "wren@example.com");
        assert_eq!(me["unique_name"], "wren");

        let again = Accounts::load(&dir).unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(again.get("google-1").unwrap()["display"], "Wren S");
    }

    /// The provider is the authority on email and display name.
    #[test]
    fn provider_fields_refresh_on_each_sign_in() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "old@example.com", "Old Name"), 1)
            .unwrap();
        let me = a
            .upsert(&ident("g1", "new@example.com", "New Name"), 2)
            .unwrap();
        assert_eq!(me["email"], "new@example.com");
        assert_eq!(me["display"], "New Name");
    }

    /// And the author is the authority on theirs — a sign-in must not revert a
    /// name somebody chose.
    #[test]
    fn author_owned_fields_are_not_overwritten_by_signing_in_again() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "wren@example.com", "Wren"), 1)
            .unwrap();
        a.put_profile("g1", &json!({"description": "Ex-surveyor."}), 2)
            .unwrap();

        let me = a
            .upsert(&ident("g1", "wren@example.com", "Wren"), 2)
            .unwrap();
        assert_eq!(me["profile"]["description"], "Ex-surveyor.");
        assert_eq!(me["unique_name"], "wren");
    }

    /// A `/v1/me` on every page load must not dirty the working tree.
    #[test]
    fn an_unchanged_sign_in_does_not_rewrite_the_file() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        let id = ident("g1", "wren@example.com", "Wren");
        a.upsert(&id, 1).unwrap();

        let path = dir.join(format!("{}.yaml", file_id("g1")));
        let first = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        a.upsert(&id, 2).unwrap();
        let second = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(first, second, "an unchanged upsert rewrote the file");
    }

    /// A profile edit must not be a way to become somebody else.
    #[test]
    fn a_profile_edit_cannot_reach_identity_fields() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "wren@example.com", "Wren"), 1)
            .unwrap();
        let me = a
            .put_profile(
                "g1",
                &json!({"description": "x", "email": "attacker@evil.com", "sub": "someone-else"}),
                2,
            )
            .unwrap()
            .unwrap();

        assert_eq!(me["email"], "wren@example.com", "email was reachable");
        assert!(me.get("sub").is_none(), "the subject leaked to the wire");
        // The stray keys landed in the profile, which is inert, not on the record.
        assert_eq!(me["profile"]["description"], "x");
    }

    #[test]
    fn the_revision_is_the_stores_to_bump() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let one = a
            .put_profile("g1", &json!({"description": "a"}), 2)
            .unwrap()
            .unwrap();
        assert_eq!(one["profile"]["revision"], 1);
        // A caller cannot set it.
        let two = a
            .put_profile("g1", &json!({"description": "b", "revision": 99}), 3)
            .unwrap()
            .unwrap();
        assert_eq!(two["profile"]["revision"], 2);
    }

    /// The claim the console makes on the Save button, as a test: a revision is
    /// superseded, never erased. An NPC that cited the old text must still be
    /// able to find it.
    #[test]
    fn a_revision_is_tombstoned_rather_than_overwritten() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        a.put_profile("g1", &json!({"description": "Surveyor."}), 100)
            .unwrap();
        a.put_profile("g1", &json!({"description": "Ex-surveyor."}), 200)
            .unwrap();

        let h = a.profile_history("g1").unwrap();
        assert_eq!(h.len(), 3, "live revision plus its two predecessors");

        // Newest first, and exactly one is live.
        assert_eq!(h[0]["description"], "Ex-surveyor.");
        assert_eq!(h[0]["live"], true);
        assert!(h[0].get("tombstoned_ms").is_none());

        assert_eq!(h[1]["description"], "Surveyor.");
        assert_eq!(h[1]["live"], false);
        assert_eq!(h[1]["tombstoned_ms"], 200);

        // The empty profile the account was born with.
        assert_eq!(h[2]["description"], "");
        assert_eq!(h[2]["tombstoned_ms"], 100);

        // And it survives a restart, because the file carries it.
        let again = Accounts::load(&dir).unwrap();
        assert_eq!(again.profile_history("g1").unwrap().len(), 3);
    }

    /// The history is the larger half of the record and is asked for rarely.
    #[test]
    fn the_history_does_not_ride_along_on_every_page_load() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let me = a
            .put_profile("g1", &json!({"description": "a"}), 2)
            .unwrap()
            .unwrap();
        assert!(me.get("profile_history").is_none());
        assert!(a.get("g1").unwrap().get("profile_history").is_none());
    }

    /// Every provider's subject shape has to become a usable file name.
    #[test]
    fn any_provider_subject_becomes_a_valid_id() {
        for sub in [
            "107839274652839471028",                // Google: digits
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890", // UUID
            "AbCdEf_-123==",                        // base64ish, uppercase, padding
            "user@example.com",                     // an email as a subject
            "日本語",
        ] {
            let key = file_id(sub);
            assert_eq!(
                crate::registry::id::check(&key),
                Ok(()),
                "`{sub}` → `{key}`"
            );
        }
    }

    /// The store must never suggest a name its own setter would refuse — the
    /// author would open the page, press Save without typing, and be told the
    /// name they were given is invalid.
    #[test]
    fn every_suggested_name_passes_the_gate_that_guards_it() {
        for (email, name) in [
            ("wren@example.com", "Wren S"),
            ("a@example.com", "A"),
            ("@example.com", "Solitary"),
            ("....@example.com", "Punctuation Only"),
            ("averyveryverylonglocalpartindeed@example.com", "Long"),
        ] {
            let suggested = default_unique_name(&ident("g", email, name));
            assert!(
                check_unique_name(&suggested).is_ok(),
                "`{email}` → `{suggested}`, which the setter refuses"
            );
        }
    }

    #[test]
    fn a_chosen_name_persists_across_a_restart() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "wren@example.com", "Wren S"), 1)
            .unwrap();
        a.put_unique_name("g1", "  ridge-walker  ").unwrap();

        let again = Accounts::load(&dir).unwrap();
        // Trimmed on the way in, so the file holds the address and not the
        // author's stray whitespace.
        assert_eq!(again.get("g1").unwrap()["unique_name"], "ridge-walker");
    }

    #[test]
    fn the_provider_subject_never_reaches_the_wire() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        let me = a
            .upsert(&ident("secret-subject-id", "w@e.com", "W"), 1)
            .unwrap();
        let text = serde_json::to_string(&me).unwrap();
        assert!(!text.contains("secret-subject-id"), "{text}");
        assert!(text.contains("user_id"));
    }
}
