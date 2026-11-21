//! Stores each signed-in person's account as a replicated record that every member
//! of the cluster reads.
//!
//! An account is a `CustomObject` record in the substrate, and never a file in the
//! mind repository, because a repository can be public and an account carries a
//! real email address and the provider's subject. Identity arrives already
//! verified, and this module maps the verified subject to a record, creating one
//! on first sight. The provider owns the email, display name and avatar and
//! refreshes them on each sign-in. The author owns the chosen name and the profile,
//! which a sign-in never overwrites.
//!
//! - `identity`: the account key and the public identity.
//! - `names`: the validated unique name.
//! - `profile`: the profile's shape, checks and revisions.
//!
//! - `tests_identity`, `tests_names`, `tests_profile`, `tests_store`: tests of the
//!   account store.

use std::path::Path;

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use web::auth::session::Identity;

use crate::registry::Registry;

/// Where accounts live among the substrate's custom objects.
const PREFIX: &str = "account/";

/// A person's Discord account, linked to their account here
/// (`docs/npcd_discord.md` §3.2): the account's `discord` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscordUser {
    /// Their Discord user id, as Discord writes it.
    pub id: String,
    /// Their Discord username, as of the link.
    pub username: String,
    /// When they linked it, in real milliseconds.
    pub linked_ms: u64,
}

/// The account store: the substrate's `account/` records.
pub struct Accounts {
    shared: SharedSubstrate,
}

impl Accounts {
    /// The accounts `shared` holds — taking in, once, the account files a
    /// daemon kept in `legacy` before accounts were records, and removing the
    /// folder after.
    pub fn open(shared: SharedSubstrate, legacy: &Path) -> anyhow::Result<Self> {
        let accounts = Self { shared };
        if legacy.is_dir() {
            let files = Registry::load("account", legacy)?;
            for file in files.iter() {
                if accounts.body(&file.id).is_none() {
                    accounts.put(&file.id, file.body.clone())?;
                }
            }
            std::fs::remove_dir_all(legacy)?;
            tracing::info!(
                "accounts: {} taken in from {}",
                files.len(),
                legacy.display()
            );
        }
        Ok(accounts)
    }

    /// Account `key`'s record.
    fn body(&self, key: &str) -> Option<Value> {
        self.shared
            .custom_object(&format!("{PREFIX}{key}"))
            .filter(|o| !o.ceased)
            .map(|o| o.blob)
    }

    /// Write account `key`'s record.
    fn put(&self, key: &str, body: Value) -> anyhow::Result<()> {
        self.shared
            .put_custom_object(CustomObjectPayload::new(format!("{PREFIX}{key}"), body))
            .map_err(|e| anyhow::anyhow!("account {key} was not written: {e:?}"))
    }

    /// Every account, by key.
    fn all(&self) -> Vec<(String, Value)> {
        self.shared
            .custom_objects_under(PREFIX)
            .into_iter()
            .map(|(o, _)| (o.key[PREFIX.len()..].to_string(), o.blob))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.all().len()
    }

    /// Whether nobody has ever signed in. A fresh daemon, not a broken one.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        let (key, existing, body) = self.merged(id, now_ms);
        // Only write when something actually changed. A GET of `/v1/me` on
        // every page load must not write a record every member then applies.
        if existing.as_ref() != Some(&body) {
            self.put(&key, body.clone())?;
        }
        Ok(with_public_id(&key, &body))
    }

    /// The public record for `id` when signing in now would change nothing —
    /// what [`Self::upsert`] answers, read without writing — or `None` when it
    /// would: a new identity, or one the provider has told something new.
    /// What lets a request take the accounts' lock for reading.
    pub fn current(&self, id: &Identity, now_ms: u64) -> Option<Value> {
        let (key, existing, body) = self.merged(id, now_ms);
        (existing.as_ref() == Some(&body)).then(|| with_public_id(&key, &body))
    }

    /// `id`'s record as it stands, and as signing in now would leave it: the
    /// one merge [`Self::upsert`] writes and [`Self::current`] compares.
    fn merged(&self, id: &Identity, now_ms: u64) -> (String, Option<Value>, Value) {
        let key = file_id(&id.provider, &id.sub);

        let existing = self.body(&key);
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
        (key, existing, body)
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
        let Some(mut body) = self.body(&key) else {
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
            // record is resident whole, and every entry is a full copy of a
            // profile.
            if history.len() > KEEP_REVISIONS {
                history.drain(..history.len() - KEEP_REVISIONS);
            }
            map.insert("profile_history".into(), Value::Array(history));
            map.insert("profile".into(), profile);
        }

        self.put(&key, body.clone())?;
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
        let body = self.body(&file_id(&id.provider, &id.sub))?;

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
        let body = self.body(&file_id(&id.provider, &id.sub))?;
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
        let taken = self.all().into_iter().any(|(other, body)| {
            other != key
                && body
                    .get("unique_name")
                    .and_then(|v| v.as_str())
                    .is_some_and(|n| n.to_ascii_lowercase() == folded)
        });
        if taken {
            return Err(NameError::Taken);
        }

        let mut body = self.body(&key).ok_or(NameError::NoAccount)?;
        let Some(map) = body.as_object_mut() else {
            return Err(NameError::Io(anyhow::anyhow!(
                "account {key} is not an object"
            )));
        };
        map.insert("unique_name".into(), json!(name));

        self.put(&key, body.clone()).map_err(NameError::Io)?;
        Ok(with_public_id(&key, &body))
    }

    /// Every account by the names it is known by — its public id, its chosen
    /// name, its display name and email — for an admin choosing whom to hand a
    /// character to. Never the provider's subject.
    pub fn known(&self) -> Vec<Value> {
        let mut rows: Vec<Value> = self
            .all()
            .into_iter()
            .map(|(key, body)| {
                let field = |f: &str| body.get(f).cloned().unwrap_or(json!(""));
                json!({
                    "user_id": public_id(&key),
                    "unique_name": field("unique_name"),
                    "display": field("display"),
                    "email": field("email"),
                })
            })
            .collect();
        rows.sort_by(|a, b| {
            let name = |v: &Value| v["unique_name"].as_str().unwrap_or_default().to_lowercase();
            name(a).cmp(&name(b))
        });
        rows
    }

    /// What the console sees for a verified identity, without creating anything.
    pub fn get(&self, id: &Identity) -> Option<Value> {
        let key = file_id(&id.provider, &id.sub);
        self.body(&key).map(|body| with_public_id(&key, &body))
    }

    /// The identity account `key` was made for, as its record keeps it: who
    /// the person is when they arrive by another way than signing in — a
    /// Discord DM. `None` with no such account.
    pub fn identity(&self, key: &str) -> Option<Identity> {
        let body = self.body(key)?;
        let field = |f: &str| body[f].as_str().unwrap_or_default().to_string();
        Some(Identity {
            provider: field("provider"),
            sub: field("sub"),
            email: field("email"),
            name: field("display"),
            picture: field("avatar_url"),
            exp: 0,
        })
    }

    /// The Discord account linked to account `key`, if any.
    pub fn discord(&self, key: &str) -> Option<DiscordUser> {
        let body = self.body(key)?;
        serde_json::from_value(body.get("discord")?.clone()).ok()
    }

    /// Link `user` to account `key`, or unlink whatever is linked with
    /// `None`. `false` with no such account. Writes nothing when nothing
    /// changes.
    pub fn set_discord(&self, key: &str, user: Option<&DiscordUser>) -> anyhow::Result<bool> {
        let Some(mut body) = self.body(key) else {
            return Ok(false);
        };
        let Some(map) = body.as_object_mut() else {
            anyhow::bail!("account {key} is not an object");
        };
        let was = map.get("discord").cloned();
        match user {
            Some(user) => {
                map.insert(
                    "discord".into(),
                    serde_json::to_value(user).expect("a Discord user serialises"),
                );
            }
            None => {
                map.remove("discord");
            }
        }
        if map.get("discord").cloned() != was {
            self.put(key, body)?;
        }
        Ok(true)
    }
}

mod identity;
mod names;
mod profile;
#[cfg(test)]
mod tests_identity;
#[cfg(test)]
mod tests_names;
#[cfg(test)]
mod tests_profile;
#[cfg(test)]
mod tests_store;

pub use identity::*;
pub use names::*;
pub use profile::*;
