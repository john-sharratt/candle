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

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use web::auth::session::Identity;

use crate::registry::Registry;

/// The account key: **issuer and subject**, as a file name.
///
/// A subject is unique per issuer, not globally. Keying on the subject alone
/// worked only because exactly one provider was configured; the day a second
/// arrives, two issuers could emit the same subject string and two people would
/// share one account — silently, and with no way to separate them afterwards
/// because the records would already be merged. Both halves go into the hash so
/// that day is a configuration change instead.
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
fn file_id(provider: &str, sub: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(provider.as_bytes());
    // A NUL separator, because it cannot occur in either half. Concatenating
    // them plainly would make `("goog", "le1")` and `("google", "1")` the same
    // account — an unlikely pair today and a real one across providers whose id
    // formats nobody controls.
    hasher.update([0u8]);
    hasher.update(sub.as_bytes());
    let digest = hasher.finalize();
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
        let key = file_id(&id.provider, &id.sub);

        let existing = self.reg.get(&key).map(|r| r.body.clone());
        let mut body = existing.clone().unwrap_or_else(|| {
            json!({
                "sub": id.sub,
                // From the identity, not a literal. It is half of the key this
                // record is filed under, so a hardcoded value here would be a
                // second answer to the same question, free to disagree with the
                // file name the moment a second provider exists.
                "provider": id.provider,
                "created_ms": now_ms,
                // A display name the author can change without touching the
                // one the provider gave them.
                "unique_name": default_unique_name(id),
                "profile": {
                    "description": "",
                    // Blank until the author picks one. An account is created
                    // by the act of signing in, before anybody has been asked
                    // anything, so every field here starts unstated.
                    "gender": "",
                    "history": "",
                    "turn_index": 0,
                    "revision": 0
                }
            })
        });

        if let Some(map) = body.as_object_mut() {
            // The provider's fields, refreshed every time — but only from a
            // value that actually arrived.
            //
            // An absent header is "the gateway did not forward this", not "the
            // provider says it is empty". `identify` defaults a missing one to
            // `""`, and the gateway itself skips any field whose value will not
            // fit in a header — so overwriting unconditionally means one
            // oversized display name blanks the stored one, permanently, on a
            // sign-in that otherwise succeeded. Keeping the last known good
            // value is the right answer to not being told.
            for (field, value) in [
                ("email", id.email.as_str()),
                ("display", id.name.as_str()),
                ("avatar_url", id.picture.as_str()),
            ] {
                if !value.trim().is_empty() {
                    map.insert(field.into(), json!(value));
                } else {
                    // Present but empty stays present-and-empty rather than
                    // becoming absent, so the shape of the record is stable.
                    map.entry(field.to_string()).or_insert(Value::Null);
                }
            }
            // The subject is the identity of the record. A file whose `sub` has
            // drifted from its name is a corrupted record, not a rename.
            map.insert("sub".into(), json!(id.sub));

            // The live profile is held to its current shape. A record written
            // when the shape was different is corrected here — the comparison
            // below then sees a change and writes it back, so the correction is
            // durable rather than applied afresh on every read.
            let mut profile = map.remove("profile").unwrap_or_else(|| json!({}));
            normalise_profile(&mut profile);
            map.insert("profile".into(), profile);
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
    /// The patch is a `Map` rather than a `Value` so a body that is not an
    /// object cannot reach here at all. It used to be a `Value`, and a `PUT`
    /// carrying `[]` or `"x"` fell through the merge — which is guarded — into
    /// the tombstone and the write, which are not: the live revision was filed
    /// into history, `revision` never advanced, and two turns ended up claiming
    /// the same number while the file grew on every repeat.
    pub fn put_profile(
        &mut self,
        id: &Identity,
        patch: &Map<String, Value>,
        now_ms: u64,
    ) -> anyhow::Result<Option<Value>> {
        let key = file_id(&id.provider, &id.sub);
        let Some(mut body) = self.reg.get(&key).map(|r| r.body.clone()) else {
            return Ok(None);
        };

        {
            let Some(map) = body.as_object_mut() else {
                anyhow::bail!("account {key} is not an object");
            };
            let mut profile = map.get("profile").cloned().unwrap_or_else(|| json!({}));
            normalise_profile(&mut profile);

            // The outgoing revision, stamped with the moment it stopped being
            // the answer. A reader with a turn index can then tell which text
            // was live when a given turn was gathered.
            let mut retired = profile.clone();
            if let Some(old) = retired.as_object_mut() {
                old.insert("tombstoned_ms".into(), json!(now_ms));
            }

            // `normalise_profile` above guarantees an object, so the merge and
            // the revision bump are unconditional — as the tombstone below
            // already was. There is no longer a shape of input that files a
            // revision into history without advancing the counter.
            let dst = profile
                .as_object_mut()
                .expect("normalise_profile leaves an object");
            for (k, v) in patch {
                // The store owns both counters; a caller that could set
                // `revision` could make two turns claim the same number.
                if k != "revision" && k != "tombstoned_ms" {
                    dst.insert(k.clone(), v.clone());
                }
            }
            let next = dst.get("revision").and_then(|r| r.as_u64()).unwrap_or(0) + 1;
            dst.insert("revision".into(), json!(next));
            // `turn_index` names the live profile turn. The profile's turns are
            // the revisions in this file, so it tracks the revision rather than
            // being a second, independently-drifting counter.
            dst.insert("turn_index".into(), json!(next));
            // A patch cannot introduce a field either: the shape is the
            // store's, not the caller's, so an unrecognised key is discarded
            // rather than stored where it would look like part of the profile.
            normalise_profile(&mut profile);

            let mut history = match map.remove("profile_history") {
                Some(Value::Array(v)) => v,
                _ => Vec::new(),
            };
            history.push(retired);
            // Oldest first, so dropping the front drops the least useful. The
            // file is read into memory whole at start, and every entry is a
            // full copy of a profile.
            if history.len() > KEEP_REVISIONS {
                history.drain(..history.len() - KEEP_REVISIONS);
            }
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
    /// An index of every revision — enough to choose one, not to read it.
    ///
    /// Summaries rather than whole revisions, because this is the payload a
    /// chooser needs and there can be hundreds of them: an author who edits
    /// often would otherwise download every paragraph they have ever written
    /// each time they open the page, to render a list of dates. The full text
    /// arrives from [`profile_revision`] when one is actually picked.
    pub fn profile_history(&self, id: &Identity) -> Option<Vec<Value>> {
        let body = &self.reg.get(&file_id(&id.provider, &id.sub))?.body;

        let summarise = |v: &Value, live: bool| {
            json!({
                "revision": v.get("revision").and_then(Value::as_u64).unwrap_or(0),
                "tombstoned_ms": v.get("tombstoned_ms").cloned().unwrap_or(Value::Null),
                "live": live,
                "preview": preview(v.get("description").and_then(Value::as_str).unwrap_or("")),
            })
        };

        let mut out = Vec::new();
        if let Some(live) = body.get("profile") {
            out.push(summarise(live, true));
        }
        if let Some(Value::Array(past)) = body.get("profile_history") {
            out.extend(past.iter().rev().map(|v| summarise(v, false)));
        }
        Some(out)
    }

    /// One revision in full, live or superseded.
    pub fn profile_revision(&self, id: &Identity, revision: u64) -> Option<Value> {
        let body = &self.reg.get(&file_id(&id.provider, &id.sub))?.body;
        let at = |v: &Value| v.get("revision").and_then(Value::as_u64) == Some(revision);

        if body.get("profile").is_some_and(at) {
            return body.get("profile").cloned();
        }
        match body.get("profile_history") {
            Some(Value::Array(past)) => past.iter().find(|v| at(v)).cloned(),
            _ => None,
        }
    }

    /// Bring a superseded revision back as the live one.
    ///
    /// An append, not a rewind: the text returns as a *new* revision and the
    /// one it replaced is tombstoned like any other edit. Rewinding the counter
    /// would leave two different profiles claiming the same revision number,
    /// and an NPC citing the earlier one would be pointing at text it never
    /// read. Restoring is just another way of saying what you want to say now.
    pub fn restore_profile(
        &mut self,
        id: &Identity,
        revision: u64,
        now_ms: u64,
    ) -> anyhow::Result<Option<Value>> {
        let Some(old) = self.profile_revision(id, revision) else {
            return Ok(None);
        };
        // Only the authored text comes back. The counters belong to the store,
        // and `tombstoned_ms` describes when *that* revision died — carrying it
        // forward would mark the new live profile as already superseded.
        let mut patch = Map::new();
        for field in AUTHORED_TEXT.iter().chain(std::iter::once(&"gender")) {
            if let Some(v) = old.get(*field) {
                patch.insert((*field).to_string(), v.clone());
            }
        }
        self.put_profile(id, &patch, now_ms)
    }

    /// Change the name characters know this author by.
    ///
    /// This is the one author-owned field that is not a matter of taste. The
    /// profile is prose an NPC reads; `unique_name` is an *address* — a tool
    /// that sends something to a person names them by it — so two accounts
    /// sharing one would make a target ambiguous rather than merely confusing.
    /// It is checked for shape and for uniqueness, and a clash is the caller's
    /// to resolve, not this function's to paper over by appending a digit.
    pub fn put_unique_name(&mut self, id: &Identity, name: &str) -> Result<Value, NameError> {
        let name = name.trim();
        check_unique_name(name)?;

        let key = file_id(&id.provider, &id.sub);
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

        self.reg
            .put(&key, body.clone())
            .map_err(|e| NameError::Io(anyhow::Error::new(e)))?;
        Ok(with_public_id(&key, &body))
    }

    /// What the console sees for a verified identity, without creating anything.
    pub fn get(&self, id: &Identity) -> Option<Value> {
        let key = file_id(&id.provider, &id.sub);
        self.reg.get(&key).map(|r| with_public_id(&key, &r.body))
    }
}

/// Every field a profile has. There are no others.
///
/// The registry preserves unknown keys on purpose — an authored world is
/// hand-edited and must survive a round trip through a program that has not
/// heard of half of it. A profile is the opposite: its shape is defined here,
/// so a key on disk that this list does not name is not a field, it is
/// residue, and [`normalise_profile`] removes it on the next write.
const PROFILE_FIELDS: [&str; 5] = ["description", "gender", "history", "turn_index", "revision"];

/// Force a stored profile into the shape above: drop what is not a field, fill
/// in what is missing, and blank a `gender` that is not a value it may hold.
///
/// Called on every sign-in, so a record written by an older shape corrects
/// itself the next time its owner appears rather than needing anyone to go and
/// find it.
fn normalise_profile(profile: &mut Value) {
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
const AUTHORED_TEXT: [&str; 2] = ["description", "history"];

/// How many superseded revisions are kept.
///
/// Bounded because the whole account file is read into memory at start, and
/// every save appends a full copy of the outgoing profile — paragraphs of it.
/// Unbounded, an author who edits often eventually carries their entire writing
/// history resident, forever, for a feature that reaches back a few steps.
///
/// Generous on purpose: two hundred is far past any real use of *undo*, so the
/// bound should never be the thing a person notices.
const KEEP_REVISIONS: usize = 200;

/// One line of a revision, for choosing between them.
///
/// Enough to recognise which edit this was, not to read it. Cut on a character
/// boundary — `&s[..N]` panics mid-codepoint, and a profile is the one place
/// somebody is most likely to have written their name in their own script.
fn preview(s: &str) -> String {
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

    /// A directory of this test's own.
    ///
    /// Counted rather than timestamped. `SystemTime::now()` is only as fine as
    /// the platform clock — around 15ms on Windows — so tests starting together
    /// were handed the *same* nanosecond and therefore the same directory, and
    /// then wrote over each other's accounts. It failed as an account that had
    /// just been saved coming back missing, which reads as a bug in the store.
    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-accounts-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // A previous run of the same PID could have left one behind.
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// `json!` builds a `Value`; the store takes a `Map`, because a patch that
    /// is not an object is not a patch. This is the seam the API does for real.
    fn patch(v: Value) -> Map<String, Value> {
        v.as_object().expect("a test patch is an object").clone()
    }

    fn ident(sub: &str, email: &str, name: &str) -> Identity {
        Identity {
            provider: "google".into(),
            sub: sub.into(),
            email: email.into(),
            name: name.into(),
            picture: String::new(),
            exp: 9_999_999_999,
        }
    }

    /// A subject, as the identity the account methods now take.
    ///
    /// They take a whole `Identity` rather than a `sub` because the account key
    /// is issuer *and* subject: two arguments that must agree are two arguments
    /// that can disagree, and the identity is the thing that already holds a
    /// matched pair.
    fn who(sub: &str) -> Identity {
        ident(sub, "a@b.c", "A B")
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
        assert_eq!(again.get(&who("google-1")).unwrap()["display"], "Wren S");
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
        a.put_profile(
            &who("g1"),
            &patch(json!({"description": "Ex-surveyor."})),
            2,
        )
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

        let path = dir.join(format!("{}.yaml", file_id("google", "g1")));
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
                &who("g1"),
                &patch(json!({"description": "x", "email": "attacker@evil.com", "sub": "someone-else"})),
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
            .put_profile(&who("g1"), &patch(json!({"description": "a"})), 2)
            .unwrap()
            .unwrap();
        assert_eq!(one["profile"]["revision"], 1);
        // A caller cannot set it.
        let two = a
            .put_profile(
                &who("g1"),
                &patch(json!({"description": "b", "revision": 99})),
                3,
            )
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
        a.put_profile(&who("g1"), &patch(json!({"description": "Surveyor."})), 100)
            .unwrap();
        a.put_profile(
            &who("g1"),
            &patch(json!({"description": "Ex-surveyor."})),
            200,
        )
        .unwrap();

        // The index is summaries — enough to choose one, not to read it.
        let h = a.profile_history(&who("g1")).unwrap();
        assert_eq!(h.len(), 3, "live revision plus its two predecessors");

        // Newest first, and exactly one is live.
        assert_eq!(h[0]["preview"], "Ex-surveyor.");
        assert_eq!(h[0]["revision"], 2);
        assert_eq!(h[0]["live"], true);
        assert_eq!(h[0]["tombstoned_ms"], Value::Null);

        assert_eq!(h[1]["preview"], "Surveyor.");
        assert_eq!(h[1]["revision"], 1);
        assert_eq!(h[1]["live"], false);
        assert_eq!(h[1]["tombstoned_ms"], 200);

        // The empty profile the account was born with.
        assert_eq!(h[2]["preview"], "");
        assert_eq!(h[2]["revision"], 0);
        assert_eq!(h[2]["tombstoned_ms"], 100);

        // The index carries no prose — that is the point of it being an index.
        assert!(h[1].get("description").is_none(), "{:?}", h[1]);
        assert!(h[1].get("history").is_none(), "{:?}", h[1]);
        // And the full text is one fetch away.
        let full = a.profile_revision(&who("g1"), 1).unwrap();
        assert_eq!(full["description"], "Surveyor.");

        // And it survives a restart, because the file carries it.
        let again = Accounts::load(&dir).unwrap();
        assert_eq!(again.profile_history(&who("g1")).unwrap().len(), 3);
    }

    /// History is bounded, because the whole file is resident.
    ///
    /// Every save appends a full copy of the outgoing profile — paragraphs of
    /// it — and `Registry::load` reads every account into memory at start.
    /// Unbounded, an author who edits often carries their entire writing
    /// history forever to support an undo that reaches back a few steps.
    #[test]
    fn the_kept_history_is_bounded_and_keeps_the_newest() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();

        let n = KEEP_REVISIONS + 25;
        for i in 0..n {
            a.put_profile(
                &who("g1"),
                &patch(json!({ "description": format!("v{i}") })),
                i as u64,
            )
            .unwrap();
        }

        // The live one plus the cap, and no more.
        let h = a.profile_history(&who("g1")).unwrap();
        assert_eq!(h.len(), KEEP_REVISIONS + 1);

        // What survives is the newest. The most recent edit is live; the oldest
        // kept is far enough back to be a real undo and no further.
        assert_eq!(h[0]["preview"], format!("v{}", n - 1));
        assert_eq!(h[0]["live"], true);
        assert_eq!(h[h.len() - 1]["revision"], (n - KEEP_REVISIONS) as u64);

        // The dropped ones are gone from lookup too, not merely hidden.
        assert!(a.profile_revision(&who("g1"), 0).is_none());
        assert!(a.profile_revision(&who("g1"), n as u64).is_some());
    }

    /// A preview is one line, and cutting it must not split a character.
    #[test]
    fn a_preview_is_short_and_never_splits_a_character() {
        assert_eq!(preview(""), "");
        assert_eq!(preview("short"), "short");
        // Newlines and runs of spaces collapse — this is one line in a chooser.
        assert_eq!(preview("two\n\nlines   here"), "two lines here");

        // Multi-byte throughout: `&s[..90]` would land mid-codepoint and panic.
        let wide = "日".repeat(300);
        let cut = preview(&wide);
        assert!(cut.ends_with('…'));
        assert_eq!(cut.chars().count(), 91, "90 characters plus the ellipsis");

        // A boundary case either side of the limit.
        for n in [89, 90, 91] {
            let s = "é".repeat(n);
            let p = preview(&s);
            assert!(
                p.chars().count() <= 91,
                "{n} produced {}",
                p.chars().count()
            );
        }
    }

    /// The history is the larger half of the record and is asked for rarely.
    #[test]
    fn the_history_does_not_ride_along_on_every_page_load() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let me = a
            .put_profile(&who("g1"), &patch(json!({"description": "a"})), 2)
            .unwrap()
            .unwrap();
        assert!(me.get("profile_history").is_none());
        assert!(a.get(&who("g1")).unwrap().get("profile_history").is_none());
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
            let key = file_id("google", sub);
            assert_eq!(
                crate::registry::id::check(&key),
                Ok(()),
                "`{sub}` → `{key}`"
            );
        }
    }

    /// A field the gateway could not forward must not blank the stored one.
    ///
    /// `identify` defaults a missing header to `""`, and the gateway skips any
    /// field whose value will not fit in one — so an overlong display name
    /// would have wiped the account's, permanently, on a sign-in that otherwise
    /// worked. Not being told something is not being told it is empty.
    #[test]
    fn a_field_the_gateway_did_not_send_keeps_its_stored_value() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        let full = ident("g1", "wren@example.com", "Wren S");
        a.upsert(&full, 1_000).unwrap();

        // The same person, signing in through a gateway that could not forward
        // the descriptive fields.
        let bare = ident("g1", "", "");
        let me = a.upsert(&bare, 2_000).unwrap();
        assert_eq!(me["email"], "wren@example.com", "the email was blanked");
        assert_eq!(me["display"], "Wren S", "the display name was blanked");

        // And it is durable, not just what this call returned.
        let again = Accounts::load(&dir).unwrap();
        assert_eq!(again.get(&full).unwrap()["display"], "Wren S");

        // A real change still lands.
        let renamed = ident("g1", "wren@example.com", "Wren Sharratt");
        let me = a.upsert(&renamed, 3_000).unwrap();
        assert_eq!(me["display"], "Wren Sharratt");
    }

    /// **The account key is issuer AND subject.**
    ///
    /// A subject is unique per issuer, not globally. Keyed on the subject
    /// alone, two providers emitting the same string would land on one account
    /// — silently, and unseparably, because by then the records would already
    /// be merged.
    #[test]
    fn the_same_subject_from_two_providers_is_two_accounts() {
        let a = file_id("google", "1234");
        let b = file_id("github", "1234");
        assert_ne!(a, b, "two issuers collided on one account");

        // And the separator does its job: without it, `("goog", "le1234")` and
        // `("google", "1234")` would hash the same bytes and be one account.
        assert_ne!(file_id("goog", "le1234"), file_id("google", "1234"));
        assert_ne!(file_id("a", "bc"), file_id("ab", "c"));
    }

    /// An account is reached only by the pair that created it. Keying on one
    /// half is what the previous design did, so this pins the other.
    #[test]
    fn an_account_is_not_reachable_from_another_provider() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        let mut google = ident("1234", "wren@example.com", "Wren");
        google.provider = "google".into();
        a.upsert(&google, 1_000).unwrap();

        let mut elsewhere = google.clone();
        elsewhere.provider = "github".into();
        assert!(
            a.get(&elsewhere).is_none(),
            "the same subject from another issuer reached this account"
        );
        assert!(a.get(&google).is_some());
    }

    /// A record written under an older profile shape corrects itself the next
    /// time its owner signs in — no field survives that the profile does not
    /// have, and a `gender` outside the allowed set is cleared rather than
    /// carried.
    #[test]
    fn an_older_shape_is_corrected_on_the_next_sign_in() {
        let dir = tmp();
        let key = file_id("google", "g1");
        std::fs::write(
            dir.join(format!("{key}.yaml")),
            serde_yaml::to_string(&json!({
                "sub": "g1",
                "provider": "google",
                "created_ms": 1,
                "unique_name": "wren",
                "profile": {
                    "description": "Ex-surveyor.",
                    "gender": "—",
                    "pronouns": "they/them",
                    "history": "",
                    "turn_index": 0,
                    "revision": 0
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut a = Accounts::load(&dir).unwrap();
        let me = a
            .upsert(&ident("g1", "wren@example.com", "Wren"), 2)
            .unwrap();

        assert!(me["profile"].get("pronouns").is_none(), "{me}");
        assert_eq!(me["profile"]["gender"], "", "an invalid gender was kept");
        // What the profile does have is untouched.
        assert_eq!(me["profile"]["description"], "Ex-surveyor.");

        // Durable, not re-derived on every read.
        let again = Accounts::load(&dir).unwrap();
        let stored = &again.get(&who("g1")).unwrap()["profile"];
        assert!(stored.get("pronouns").is_none(), "{stored}");
        assert_eq!(stored["gender"], "");
    }

    /// Everything a caller may write is checked before any of it is merged.
    ///
    /// The failure this guards is silent rather than loud: `normalise_profile`
    /// blanks a wrong-typed field on its way to disk, which is right for
    /// repairing a record and catastrophic for a live write — a `200 OK` that
    /// erased the author's prose. `null` and `3` are refused as the wrong
    /// *kind* of thing, separately from a string that is not an allowed gender,
    /// because those are two different mistakes to fix.
    #[test]
    fn a_wrong_typed_field_is_refused_rather_than_blanked() {
        for bad in [
            json!({"description": 42}),
            json!({"history": null}),
            json!({"description": ["a"]}),
            json!({"gender": null}),
            json!({"gender": 3}),
            json!({"gender": {"v": "Male"}}),
        ] {
            let m = patch(bad.clone());
            assert!(
                matches!(check_patch(&m), Err(PatchError::NotText(_))),
                "{bad} was accepted as text"
            );
        }

        // A string that is simply not one of the values is its own answer.
        assert_eq!(
            check_patch(&patch(json!({"gender": "Other"}))),
            Err(PatchError::BadGender)
        );

        for good in [
            json!({}),
            json!({"description": ""}),
            json!({"gender": ""}),
            json!({"gender": "Female", "history": "x"}),
            // Unknown keys are the store's to discard, not a reason to refuse.
            json!({"nickname": 7}),
        ] {
            assert_eq!(check_patch(&patch(good.clone())), Ok(()), "{good} refused");
        }
    }

    /// A patch that is not an object cannot reach the store at all, so it
    /// cannot file a revision into history without advancing the counter.
    ///
    /// It used to be able to: the merge was guarded by `patch.as_object()` but
    /// the tombstone and the write below it were not, so `PUT []` retired the
    /// live profile, left `revision` where it was, and grew the file on every
    /// repeat until two turns claimed the same number. The signature is the fix
    /// — `Map` rather than `Value` — and this is the assertion that the shape
    /// really is unrepresentable.
    #[test]
    fn a_body_that_is_not_an_object_is_not_a_patch() {
        for not_a_patch in [json!([]), json!("x"), json!(3), json!(null)] {
            assert!(
                not_a_patch.as_object().is_none(),
                "{not_a_patch} would still reach the store"
            );
        }

        // And a legitimate empty patch still advances exactly one revision.
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let me = a
            .put_profile(&who("g1"), &patch(json!({})), 2)
            .unwrap()
            .unwrap();
        assert_eq!(me["profile"]["revision"], 1);
        assert_eq!(a.profile_history(&who("g1")).unwrap().len(), 2);
    }

    /// A caller cannot add one back, either.
    #[test]
    fn a_patch_cannot_introduce_a_field_the_profile_does_not_have() {
        let dir = tmp();
        let mut a = Accounts::load(&dir).unwrap();
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let me = a
            .put_profile(
                &who("g1"),
                &patch(json!({"description": "x", "pronouns": "they/them", "nickname": "boss"})),
                2,
            )
            .unwrap()
            .unwrap();

        assert_eq!(me["profile"]["description"], "x");
        assert!(me["profile"].get("pronouns").is_none(), "{me}");
        assert!(me["profile"].get("nickname").is_none(), "{me}");
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
        a.put_unique_name(&who("g1"), "  ridge-walker  ").unwrap();

        let again = Accounts::load(&dir).unwrap();
        // Trimmed on the way in, so the file holds the address and not the
        // author's stray whitespace.
        assert_eq!(
            again.get(&who("g1")).unwrap()["unique_name"],
            "ridge-walker"
        );
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
