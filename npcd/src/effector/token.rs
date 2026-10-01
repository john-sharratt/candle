//! The effector device's identity: one opaque token per character.
//!
//! A character reaches the world through `http://local` (see the effector
//! design), and every such call carries a bearer token that resolves to the
//! body making it. The token is the whole of the device surface's auth — no
//! roles, no ownership, no `x-tokera-*` (see [`crate::effector::auth`]) — so it
//! must be a *real secret*: minted from the OS entropy source, never derived
//! from the deterministic `body_id` (`npc-{id}`), which anyone could guess.
//!
//! # It is durable, and it is not in the repository
//!
//! Tokens persist through a [`Registry`], one file per character keyed by
//! `npc_id`, exactly the way accounts do ([`crate::accounts`]) — an external
//! client's token has to survive a restart, because the world it drives does
//! not re-hand it one. And like accounts it is **git-ignored**: a token is a
//! credential, git history is the one place you cannot quietly remove something
//! from later, so the exclusion (`npcd/.gitignore`, `tokens/`) holds before the
//! first write rather than after the first leak.
//!
//! # Two indices, one under a lock
//!
//! [`Tokens`] answers the hot question — *whose token is this?* — from an
//! in-memory `HashMap` in O(1), the check the auth layer runs on every call.
//! The durable per-character record lives beside it in the registry, and both
//! sit under one [`Mutex`] so the store can be shared as a single `Arc` and
//! both read (resolve) and written (mint) through it: the auth layer resolves
//! while startup seeding mints, and a second handle to one registry is the bug
//! this daemon already learned the cost of (`npcs.rs`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use base64::Engine;
use rand::TryRngCore;
use serde::{Deserialize, Serialize};
use serde_json::{from_value, json};

use crate::registry::Registry;

/// What a token lets its holder reach.
///
/// The scope is checked *after* the one lookup that resolves a token to a body,
/// and it decides proximity rather than permission (effector design §8.3):
///
/// - [`Scope::AsNpc`] — the character's own token, and the scope every
///   in-fiction call uses. Proximity-gated: it reaches only what is within the
///   body's standpoint, so an id not near the body is not addressable.
/// - [`Scope::Direct`] — for an operator or embedder driving the world from
///   outside. Addresses any item by id regardless of where the body stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Scope {
    AsNpc,
    Direct,
}

/// What a token resolves to: the body it acts through, and how far it reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grant {
    pub npc_id: u64,
    pub scope: Scope,
}

/// Every character's device token, resolvable by token and by character.
pub struct Tokens {
    inner: Mutex<Inner>,
}

struct Inner {
    /// The O(1) auth index: the secret a caller presents → the body it grants.
    /// Rebuilt from the registry on load, and the one map the hot path reads.
    by_token: HashMap<String, Grant>,
    /// The durable per-character record — `npc_id` → `{token, scope}` — one
    /// file each, so a token outlives the process that minted it.
    reg: Registry,
}

impl Tokens {
    /// Read every persisted token into the resolve index.
    ///
    /// A missing directory is an empty store, not an error — a fresh daemon has
    /// seeded nobody yet. `load_generated` because this daemon writes these
    /// files and no human ever comments one, so a save serialises whole rather
    /// than splicing (the accounts reasoning, [`crate::accounts`]).
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let reg = Registry::load_generated("token", dir)?;
        let mut by_token = HashMap::new();
        for record in reg.iter() {
            // The record id is the character's numeric id. A file whose name is
            // not one is residue, not a token — skip it rather than fail the
            // whole store for one stray file.
            let Ok(npc_id) = record.id.parse::<u64>() else {
                tracing::warn!(id = %record.id, "token: file name is not an npc id, skipping");
                continue;
            };
            let (Some(token), Ok(scope)) = (
                record.body.get("token").and_then(|v| v.as_str()),
                from_value::<Scope>(record.body.get("scope").cloned().unwrap_or(json!("as-npc"))),
            ) else {
                tracing::warn!(
                    id = %record.id,
                    "token: record is missing a token or scope, skipping"
                );
                continue;
            };
            by_token.insert(token.to_string(), Grant { npc_id, scope });
        }
        tracing::info!("tokens: {} loaded", by_token.len());
        Ok(Self {
            inner: Mutex::new(Inner { by_token, reg }),
        })
    }

    /// Mint a fresh secret for a character and persist it.
    ///
    /// Always a new secret, straight from the OS — so a caller that already has
    /// a token and wants to keep it checks [`Self::token_of`] first (startup
    /// seeding does). The write lands before the map is updated, so a token this
    /// returns is one a restart will still resolve.
    pub fn mint(&self, npc_id: u64, scope: Scope) -> Result<String> {
        let token = random_token();
        let mut inner = self.inner.lock().expect("tokens lock");

        // Rotate rather than accumulate: a previous secret for this character
        // stops resolving, so a re-mint invalidates the old token instead of
        // leaving it valid beside the new one.
        let previous = inner
            .reg
            .get(&npc_id.to_string())
            .and_then(|r| r.body.get("token"))
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        // Disk before RAM: persist the new secret first, so a `put` failure
        // leaves both stores unchanged (the old token still resolves in memory
        // and still stands on disk) rather than a half-rotation where the old
        // token stops resolving in RAM but survives a restart.
        inner
            .reg
            .put(
                &npc_id.to_string(),
                json!({ "token": token, "scope": scope }),
            )
            .with_context(|| format!("persisting the token for npc {npc_id}"))?;
        if let Some(old) = previous {
            inner.by_token.remove(&old);
        }
        inner
            .by_token
            .insert(token.clone(), Grant { npc_id, scope });
        Ok(token)
    }

    /// Resolve a presented token to the body it grants. The O(1) auth check.
    pub fn resolve(&self, token: &str) -> Option<Grant> {
        self.inner
            .lock()
            .expect("tokens lock")
            .by_token
            .get(token)
            .copied()
    }

    /// The token already minted for a character, if any.
    ///
    /// What the in-process fast path stamps ([`crate::engine::runtime::Runtime`])
    /// and what startup seeding checks so it mints only for a character that has
    /// no token yet, rather than invalidating a persisted one on every boot.
    pub fn token_of(&self, npc_id: u64) -> Option<String> {
        self.inner
            .lock()
            .expect("tokens lock")
            .reg
            .get(&npc_id.to_string())
            .and_then(|r| r.body.get("token").and_then(|v| v.as_str()))
            .map(str::to_owned)
    }

    /// The character's token, minting one at the given scope if it has none.
    ///
    /// The idempotent form the fast path and startup both want: a character's
    /// device token is stable across the run and across a restart, and asking
    /// for it twice returns the same secret rather than a second one.
    pub fn ensure(&self, npc_id: u64, scope: Scope) -> Result<String> {
        match self.token_of(npc_id) {
            Some(token) => Ok(token),
            None => self.mint(npc_id, scope),
        }
    }

    /// The character's own **as-npc** token, for the in-process fast path.
    ///
    /// The in-fiction call always acts as the body itself, so it must carry the
    /// `AsNpc` scope — never a `Direct` one, which bypasses proximity (§8.3).
    /// [`Self::ensure`] hands back whatever scope was stored, so the fast path
    /// uses this instead: it returns the character's token only when it is
    /// already `AsNpc`, and mints a fresh `AsNpc` one otherwise. A `Direct` token
    /// under a character's own id is an operator's, not the body's, and must not
    /// become the in-fiction identity — so it is rotated out here rather than
    /// carried into a character's own turn.
    pub fn as_npc(&self, npc_id: u64) -> Result<String> {
        let stored = {
            let inner = self.inner.lock().expect("tokens lock");
            inner.reg.get(&npc_id.to_string()).and_then(|r| {
                let token = r.body.get("token").and_then(|v| v.as_str())?.to_owned();
                let scope =
                    from_value::<Scope>(r.body.get("scope").cloned().unwrap_or(json!("as-npc")))
                        .ok()?;
                Some((token, scope))
            })
        };
        match stored {
            Some((token, Scope::AsNpc)) => Ok(token),
            // No token, or one at the wrong scope: mint a fresh as-npc secret.
            _ => self.mint(npc_id, Scope::AsNpc),
        }
    }
}

/// 256 bits from the OS, URL-safe so it rides in a header unescaped.
///
/// From the OS entropy source and a hard failure if it will not answer — a
/// guessable device token is the whole of the surface's auth defeated, so there
/// is no seeded fallback worth having (the reasoning `web`'s `random_token`
/// records for its OAuth secrets).
fn random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut bytes)
        .expect("the OS must provide entropy");
    B64.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of this test's own, counted rather than timestamped so two
    /// tests starting in the same clock tick do not share one (the accounts
    /// tests learned this on Windows' ~15ms clock).
    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-tokens-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The round trip the auth layer depends on: a minted token resolves to the
    /// body it was minted for, and to that body's scope.
    #[test]
    fn a_minted_token_resolves_to_its_body() {
        let t = Tokens::load(tmp()).unwrap();
        let token = t.mint(7, Scope::AsNpc).unwrap();
        assert_eq!(
            t.resolve(&token),
            Some(Grant {
                npc_id: 7,
                scope: Scope::AsNpc
            })
        );
    }

    /// The refusal half: a token nobody minted resolves to nothing, which is
    /// what turns into the 401 the device surface answers.
    #[test]
    fn an_unknown_token_resolves_to_nothing() {
        let t = Tokens::load(tmp()).unwrap();
        t.mint(1, Scope::AsNpc).unwrap();
        assert_eq!(t.resolve("not-a-real-token"), None);
        assert_eq!(t.resolve(""), None);
    }

    /// **A token is a real secret, so two characters never share one.** Derived
    /// from `npc_id` it would be guessable (the whole reason §8.3 forbids it);
    /// random, a collision is astronomically unlikely, and the two also grant
    /// their own bodies rather than each other's.
    #[test]
    fn tokens_differ_per_character() {
        let t = Tokens::load(tmp()).unwrap();
        let a = t.mint(1, Scope::AsNpc).unwrap();
        let b = t.mint(2, Scope::AsNpc).unwrap();
        assert_ne!(a, b);
        // And neither is the deterministic body id nor contains the npc number
        // in any obvious way — it is entropy, not a formatting of the id.
        assert_ne!(a, "npc-1");
        assert_eq!(t.resolve(&a).unwrap().npc_id, 1);
        assert_eq!(t.resolve(&b).unwrap().npc_id, 2);
    }

    /// A token outlives the process: a fresh store over the same directory
    /// resolves what a previous one minted, so an external client's token
    /// survives a restart.
    #[test]
    fn tokens_survive_a_reload() {
        let dir = tmp();
        let token = {
            let t = Tokens::load(&dir).unwrap();
            t.mint(42, Scope::Direct).unwrap()
        };

        let again = Tokens::load(&dir).unwrap();
        assert_eq!(
            again.resolve(&token),
            Some(Grant {
                npc_id: 42,
                scope: Scope::Direct
            })
        );
    }

    /// Minting again for a character rotates the secret: the new one resolves
    /// and the old one no longer does, so a re-mint is a revocation.
    #[test]
    fn re_minting_rotates_and_invalidates_the_old_token() {
        let t = Tokens::load(tmp()).unwrap();
        let first = t.mint(4, Scope::AsNpc).unwrap();
        let second = t.mint(4, Scope::AsNpc).unwrap();
        assert_ne!(first, second);
        assert_eq!(t.resolve(&second).map(|g| g.npc_id), Some(4));
        assert_eq!(t.resolve(&first), None, "the rotated token still worked");
    }

    /// `ensure` is idempotent — the character's token is stable across the run,
    /// so a second ask returns the same secret rather than minting another.
    #[test]
    fn ensure_returns_the_same_token_twice() {
        let t = Tokens::load(tmp()).unwrap();
        let first = t.ensure(3, Scope::AsNpc).unwrap();
        let second = t.ensure(3, Scope::AsNpc).unwrap();
        assert_eq!(first, second);
        assert_eq!(t.token_of(3), Some(first));
        assert_eq!(t.token_of(99), None);
    }

    /// **The fast path forces the as-npc scope.** An as-npc token is handed back
    /// unchanged; a `Direct` token under a character's own id is rotated out for
    /// a fresh as-npc one, so the in-fiction call can never carry `Direct`; and a
    /// character with no token gets a fresh as-npc secret.
    #[test]
    fn as_npc_forces_the_as_npc_scope() {
        let t = Tokens::load(tmp()).unwrap();

        let own = t.mint(1, Scope::AsNpc).unwrap();
        assert_eq!(
            t.as_npc(1).unwrap(),
            own,
            "an as-npc token is returned as-is"
        );

        let direct = t.mint(2, Scope::Direct).unwrap();
        let forced = t.as_npc(2).unwrap();
        assert_ne!(
            forced, direct,
            "a Direct token was carried into the fast path"
        );
        assert_eq!(t.resolve(&forced).unwrap().scope, Scope::AsNpc);
        assert_eq!(
            t.resolve(&direct),
            None,
            "the Direct token was not rotated out"
        );

        let fresh = t.as_npc(3).unwrap();
        assert_eq!(t.resolve(&fresh).unwrap().scope, Scope::AsNpc);
    }

    /// The scope is kebab-case on disk, because that is how it reads in a file
    /// a person may open, and it round-trips through the registry unchanged.
    #[test]
    fn scope_is_kebab_case_and_round_trips() {
        assert_eq!(serde_json::to_value(Scope::AsNpc).unwrap(), json!("as-npc"));
        assert_eq!(
            serde_json::to_value(Scope::Direct).unwrap(),
            json!("direct")
        );

        let dir = tmp();
        {
            Tokens::load(&dir).unwrap().mint(5, Scope::AsNpc).unwrap();
        }
        assert_eq!(
            Tokens::load(&dir)
                .unwrap()
                .resolve(&Tokens::load(&dir).unwrap().token_of(5).unwrap())
                .unwrap()
                .scope,
            Scope::AsNpc
        );
    }
}
