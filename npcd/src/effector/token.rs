//! The bearer tokens the device is reached with.
//!
//! A token is an opaque secret drawn from the operating system's entropy and
//! stored beside what it opens. It is **never derived from a body id**: a body
//! id is public (`npc-7`), so a token computed from one would be guessable by
//! anyone who could read a name. The table is the only thing that connects a
//! token to a body.
//!
//! Each token carries a [`Scope`]:
//!
//! - [`Scope::AsNpc`] — the in-fiction identity. Every route is gated on what
//!   the body can reach from where it stands; an address out of reach is a `404`.
//! - [`Scope::Direct`] — an operator's identity for a body: any item by id, and
//!   the world-shaping routes.
//!
//! The table is persisted under the daemon's `tokens/` directory so a token a
//! client holds survives a restart.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rand::rngs::OsRng;
use rand::TryRngCore;
use serde::{Deserialize, Serialize};

/// How far a token reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    AsNpc,
    Direct,
}

/// Whose a token is and what it may do — what a valid bearer resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Caller {
    pub npc_id: u64,
    pub scope: Scope,
}

/// The persisted token table.
pub struct Tokens {
    file: PathBuf,
    table: Mutex<HashMap<String, Caller>>,
}

impl Tokens {
    const FILE: &'static str = "tokens.json";

    /// Open the store in `dir`, creating the directory and reading the table
    /// back when one was saved.
    pub fn load(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let file = dir.join(Self::FILE);
        let table = if file.exists() {
            let text = fs::read_to_string(&file)
                .with_context(|| format!("reading {}", file.display()))?;
            serde_json::from_str(&text).with_context(|| format!("parsing {}", file.display()))?
        } else {
            HashMap::new()
        };
        Ok(Self {
            file,
            table: Mutex::new(table),
        })
    }

    /// The body a bearer token opens, `None` for a token this store never issued.
    pub fn resolve(&self, token: &str) -> Option<Caller> {
        self.table.lock().unwrap().get(token).copied()
    }

    /// The in-fiction token for a body, issued on first use. Always an
    /// [`Scope::AsNpc`] token, whatever else is stored for that body.
    pub fn as_npc(&self, npc_id: u64) -> Result<String> {
        self.ensure(npc_id, Scope::AsNpc)
    }

    /// The operator token for a body, issued on first use.
    pub fn direct(&self, npc_id: u64) -> Result<String> {
        self.ensure(npc_id, Scope::Direct)
    }

    /// The token for `(npc_id, scope)`: the one already issued, else a fresh
    /// one, saved before it is returned so a token that exists is a token that
    /// survives.
    pub fn ensure(&self, npc_id: u64, scope: Scope) -> Result<String> {
        let wanted = Caller { npc_id, scope };
        let mut table = self.table.lock().unwrap();
        if let Some((token, _)) = table.iter().find(|(_, caller)| **caller == wanted) {
            return Ok(token.clone());
        }
        let token = mint()?;
        table.insert(token.clone(), wanted);
        if let Err(e) = self.save(&table) {
            table.remove(&token);
            return Err(e);
        }
        Ok(token)
    }

    fn save(&self, table: &HashMap<String, Caller>) -> Result<()> {
        let mut tmp = self.file.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        let text = serde_json::to_string_pretty(table)?;
        fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &self.file)
            .with_context(|| format!("replacing {}", self.file.display()))
    }
}

/// 256 bits from the operating system, in lowercase hex.
fn mint() -> Result<String> {
    let mut bytes = [0u8; 32];
    OsRng
        .try_fill_bytes(&mut bytes)
        .context("the operating system would not give entropy")?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn store() -> (TempDir, Tokens) {
        let dir = TempDir::new().unwrap();
        let tokens = Tokens::load(dir.path().join("tokens")).unwrap();
        (dir, tokens)
    }

    #[test]
    fn a_token_is_256_bits_of_hex() {
        let (_dir, tokens) = store();
        let token = tokens.as_npc(1).unwrap();
        assert_eq!(token.len(), 64);
        assert!(token.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    }

    #[test]
    fn a_body_keeps_its_token_and_two_bodies_never_share_one() {
        let (_dir, tokens) = store();
        let one = tokens.as_npc(1).unwrap();
        assert_eq!(tokens.as_npc(1).unwrap(), one);
        assert_ne!(tokens.as_npc(2).unwrap(), one);
    }

    #[test]
    fn the_two_scopes_are_two_tokens_that_resolve_to_their_own_scope() {
        let (_dir, tokens) = store();
        let npc = tokens.as_npc(5).unwrap();
        let direct = tokens.direct(5).unwrap();
        assert_ne!(npc, direct);
        assert_eq!(
            tokens.resolve(&npc),
            Some(Caller { npc_id: 5, scope: Scope::AsNpc })
        );
        assert_eq!(
            tokens.resolve(&direct),
            Some(Caller { npc_id: 5, scope: Scope::Direct })
        );
    }

    #[test]
    fn a_stored_direct_scope_never_becomes_the_in_fiction_identity() {
        let (_dir, tokens) = store();
        let direct = tokens.direct(9).unwrap();
        let npc = tokens.as_npc(9).unwrap();
        assert_ne!(direct, npc);
        assert_eq!(tokens.resolve(&npc).unwrap().scope, Scope::AsNpc);
    }

    #[test]
    fn an_unissued_token_opens_nothing() {
        let (_dir, tokens) = store();
        tokens.as_npc(1).unwrap();
        assert_eq!(tokens.resolve("npc-1"), None);
        assert_eq!(tokens.resolve(""), None);
    }

    #[test]
    fn the_table_survives_a_restart() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tokens");
        let issued = {
            let tokens = Tokens::load(path.clone()).unwrap();
            tokens.as_npc(3).unwrap()
        };
        let reopened = Tokens::load(path).unwrap();
        assert_eq!(
            reopened.resolve(&issued),
            Some(Caller { npc_id: 3, scope: Scope::AsNpc })
        );
        assert_eq!(reopened.as_npc(3).unwrap(), issued);
    }

    #[test]
    fn the_saved_table_names_the_scope_in_snake_case() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tokens");
        let tokens = Tokens::load(path.clone()).unwrap();
        let token = tokens.as_npc(4).unwrap();
        let text = fs::read_to_string(path.join("tokens.json")).unwrap();
        let saved: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(saved[&token], serde_json::json!({ "npc_id": 4, "scope": "as_npc" }));
    }
}
