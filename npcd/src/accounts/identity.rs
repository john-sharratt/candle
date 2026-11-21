//! Derives the stored account key from a verified identity's issuer and subject.
//!
//! The key hashes the issuer and the subject together, with a separator that cannot
//! occur in either, so the same subject from two providers is two accounts. The
//! key is a hash prefix rather than the raw subject, so it is always a safe name
//! whatever a provider emits, and the raw id stays out of directory listings. The
//! subject stored in the record is authoritative and is set on sign-in. The sign-in
//! path reads the account under a shared lock when nothing would change, and takes
//! the write lock only when something would.

use super::*;

/// The account key: **issuer and subject**, hashed into a record name.
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
pub(super) fn file_id(provider: &str, sub: &str) -> String {
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
    // — the subject inside the record is authoritative and is set on sign-in —
    // so this is about a name of workable length.
    digest[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// The public id of an account filed under `key`: what its characters are
/// owned by, and its author's store over the mind (§10.3).
pub fn public_id(key: &str) -> String {
    format!("u_{}", &key[..8])
}

/// The public id of `id`'s account.
pub fn user_id_of(id: &Identity) -> String {
    public_id(&key_of(id))
}

/// The key `id`'s account is filed under: what names the account to the
/// records that point at it — a Discord link, a bot's DM with the person.
pub fn key_of(id: &Identity) -> String {
    file_id(&id.provider, &id.sub)
}

/// `id`'s account, created or refreshed as a sign-in does it, through the
/// shared store `accounts`: under its read lock when signing in changes
/// nothing — every request but the first, and the first after the provider
/// says something new — and under its write lock otherwise, where
/// [`Accounts::upsert`] merges again, so no update is lost to a write that
/// landed in between.
pub async fn account_of(
    accounts: &RwLock<Accounts>,
    id: &Identity,
    now_ms: u64,
) -> anyhow::Result<Value> {
    // Bound before the write lock is asked for: a read guard alive across it
    // would wait on itself, the lock not being reentrant.
    let current = accounts.read().await.current(id, now_ms);
    if let Some(me) = current {
        return Ok(me);
    }
    accounts.write().await.upsert(id, now_ms)
}

/// The wire shape.
///
/// `user_id` is the record key, not the provider's subject — the console needs a
/// stable handle for a user and has no business holding the provider's id for
/// one. The `sub` never leaves this process.
pub(super) fn with_public_id(key: &str, body: &Value) -> Value {
    let mut out = body.clone();
    if let Some(map) = out.as_object_mut() {
        map.remove("sub");
        // Superseded revisions have their own endpoint. Shipping the whole
        // history on every `/v1/me` would grow the page-load payload by one
        // profile per edit the author has ever made, to render a chip.
        map.remove("profile_history");
        map.insert("user_id".into(), json!(public_id(key)));
    }
    out
}
