//! Sessions for a server that holds nothing on them: each is carried whole in
//! its own id, so every member of a cluster reads the same session from a
//! handshake any one of them answered, and nothing is kept anywhere.
//!
//! An id is not a secret and is not signed. It names its owner, and a request
//! on it from anyone else is refused as on any session; one made up by its
//! own owner is only another session of theirs, which holds nothing.

use web::auth::oidc::random_token;

use crate::client::Client;
use crate::session::{Session, Store};

/// Sessions that live in their ids.
#[derive(Debug, Default)]
pub struct SealedStore;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

/// The id that carries `session`: a random part, so two sessions of one
/// person and client differ, and the session itself.
fn seal(session: &Session) -> anyhow::Result<String> {
    Ok(format!(
        "{}-{}",
        random_token(),
        hex(&serde_json::to_vec(session)?)
    ))
}

/// The session `id` carries.
fn unseal(id: &str) -> Option<Session> {
    let (_, sealed) = id.rsplit_once('-')?;
    serde_json::from_slice(&unhex(sealed)?).ok()
}

impl Store for SealedStore {
    fn open(
        &self,
        owner: &str,
        client: Client,
        protocol: &str,
        now: u64,
    ) -> anyhow::Result<String> {
        seal(&Session::new(owner, client, protocol, now))
    }

    fn get(&self, id: &str, _now: u64) -> Option<Session> {
        unseal(id)
    }

    fn ensure(
        &self,
        _id: &str,
        owner: &str,
        client: Client,
        protocol: &str,
        now: u64,
    ) -> anyhow::Result<Session> {
        Ok(Session::new(owner, client, protocol, now))
    }

    fn hold(&self, _id: &str, key: &str, _value: &str, _now: u64) -> anyhow::Result<()> {
        anyhow::bail!("this server holds nothing on a session, so `{key}` cannot be kept")
    }

    /// Nothing is kept, so there is nothing to end.
    fn close(&self, _id: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_is_read_back_from_its_id_by_any_store() {
        let id = SealedStore
            .open("google/1", Client::ClaudeCode, "2025-11-25", 10)
            .unwrap();
        assert!(id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'));
        assert_eq!(
            SealedStore.get(&id, 99),
            Some(Session::new(
                "google/1",
                Client::ClaudeCode,
                "2025-11-25",
                10
            ))
        );
        assert_ne!(
            id,
            SealedStore
                .open("google/1", Client::ClaudeCode, "2025-11-25", 10)
                .unwrap()
        );
    }

    #[test]
    fn an_id_that_carries_nothing_names_no_session_and_nothing_is_held() {
        assert_eq!(SealedStore.get("abc", 0), None);
        assert_eq!(SealedStore.get("abc-zz", 0), None);
        assert_eq!(SealedStore.get("abc-7b7d", 0), None);
        assert!(SealedStore.hold("x", "attached", "t", 0).is_err());
    }
}
