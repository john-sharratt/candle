//! A run's overlay as the wire carries it: the base its changes are made on,
//! and each changed path's chain named by its digest. The chains themselves
//! travel only when the agent lacks them (`docs/zen_coding_challenge_fixes.md`
//! Cause 5).
//!
//! A member sends the names first. An agent holding every chain named runs
//! the job; one that lacks some answers `409` with the digests it lacks
//! ([`Missing`]), and the member sends the job again carrying those chains
//! alone. A long session's unchanged files are sent once, not on every run.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zend_vfs::{SavedBase, SavedChain, Snapshot};

/// The `error` a `409` names when the agent lacks chains an overlay names.
pub const MISSING: &str = "missing";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Overlay {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<SavedBase>,
    /// Each changed path's chain, by its [`digest`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub chains: BTreeMap<String, String>,
    /// Chains carried with the job, by digest: those the agent said it lacks,
    /// or every one for an agent in the member's own process.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub carried: BTreeMap<String, SavedChain>,
}

/// The `409` body of an agent lacking chains an overlay names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Missing {
    pub error: String,
    pub missing: Vec<String>,
}

impl Missing {
    pub fn of(missing: Vec<String>) -> Missing {
        Missing {
            error: MISSING.to_string(),
            missing,
        }
    }
}

/// A chain's name: the SHA-256 of its JSON, in lower-case hex. A chain is
/// named by what it holds, so two runs naming one digest name one chain.
pub fn digest(chain: &SavedChain) -> String {
    let bytes = serde_json::to_vec(chain).expect("a chain serialises");
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl Overlay {
    /// `snapshot` named and carried whole.
    pub fn of(snapshot: &Snapshot) -> Overlay {
        let mut chains = BTreeMap::new();
        let mut carried = BTreeMap::new();
        for (path, chain) in snapshot.chains() {
            let name = digest(chain);
            chains.insert(path.clone(), name.clone());
            carried.insert(name, chain.clone());
        }
        Overlay {
            base: snapshot.base().cloned(),
            chains,
            carried,
        }
    }

    /// The overlay by its names alone: what a member sends first.
    pub fn named(&self) -> Overlay {
        Overlay {
            carried: BTreeMap::new(),
            ..self.clone()
        }
    }

    /// The overlay by its names, carrying the chains of `missing` it holds:
    /// what a member sends once an agent has said what it lacks.
    pub fn carrying(&self, missing: &[String]) -> Overlay {
        Overlay {
            carried: missing
                .iter()
                .filter_map(|d| self.carried.get(d).map(|c| (d.clone(), c.clone())))
                .collect(),
            ..self.clone()
        }
    }

    /// How many changed paths it names.
    pub fn files(&self) -> usize {
        self.chains.len()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use zend_vfs::VfsStore;

    use super::*;

    /// A chain that leaves a file at 3 bytes and holds no deltas.
    fn chain() -> SavedChain {
        SavedChain {
            deltas: Vec::new(),
            size: Some(3),
            conflict: false,
        }
    }

    /// **A chain's digest is the SHA-256 of its JSON**, `{"deltas":[],"size":3}`
    /// — exactly these bytes.
    #[test]
    fn a_chain_is_named_by_the_sha256_of_its_json() {
        assert_eq!(
            serde_json::to_string(&chain()).unwrap(),
            r#"{"deltas":[],"size":3}"#
        );
        assert_eq!(
            digest(&chain()),
            "fe53f3228f499ab02532ea88dc0b54008fe1a3df4b5cf2d6487acf7a41fff862"
        );
    }

    /// **An overlay names every chain and carries them; its names alone go
    /// first, and a resend carries only what was missing.**
    #[test]
    fn an_overlay_is_sent_by_name_then_by_what_is_missing() {
        let store = VfsStore::new();
        store.write("a.txt", "one\n".into()).unwrap();
        store.write("b.txt", "two\n".into()).unwrap();
        let snapshot = store.snapshot();
        let overlay = Overlay::of(&snapshot);
        assert_eq!(overlay.files(), 2);
        assert_eq!(overlay.carried.len(), 2);
        let a = overlay.chains["a.txt"].clone();
        assert_eq!(a, digest(&snapshot.chains()["a.txt"]));

        let named = overlay.named();
        assert!(named.carried.is_empty());
        assert_eq!(named.chains, overlay.chains);

        let resent = overlay.carrying(&[a.clone(), "f".repeat(64)]);
        assert_eq!(resent.carried.keys().collect::<Vec<_>>(), [&a]);
        assert_eq!(resent.chains, overlay.chains);
    }

    /// **An overlay with nothing in it is `{}`**, and a `409`'s body names the
    /// digests the agent lacks — exactly these bytes.
    #[test]
    fn an_empty_overlay_and_a_missing_answer_are_their_json() {
        assert_eq!(serde_json::to_value(Overlay::default()).unwrap(), json!({}));
        assert_eq!(
            serde_json::to_value(Missing::of(vec!["ab".into()])).unwrap(),
            json!({"error": "missing", "missing": ["ab"]})
        );
    }
}
