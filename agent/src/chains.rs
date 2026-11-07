//! The chains an agent has been sent, by digest: what a run's overlay names
//! instead of carrying (`overlay`). Held in memory, least recently used out
//! first, up to [`HELD_BYTES`]; an agent that restarts holds none, and asks
//! for each again.

use std::collections::{BTreeMap, HashMap, VecDeque};

use zend_vfs::{SavedChain, Snapshot};

use crate::overlay::{digest, Overlay};

/// The most chain bytes an agent holds.
pub const HELD_BYTES: usize = 256 * 1024 * 1024;

/// Why an overlay does not resolve into a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unresolved {
    /// The chains it names that are neither carried nor held, by digest.
    Missing(Vec<String>),
    /// A carried chain is not the chain its digest names.
    Misnamed(String),
}

pub struct Chains {
    held: HashMap<String, (SavedChain, usize)>,
    /// Digests, least recently used first.
    order: VecDeque<String>,
    bytes: usize,
    cap: usize,
}

impl Default for Chains {
    fn default() -> Self {
        Chains::new(HELD_BYTES)
    }
}

impl Chains {
    pub fn new(cap: usize) -> Chains {
        Chains {
            held: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            cap,
        }
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether the chain named `name` is held.
    pub fn holds(&self, name: &str) -> bool {
        self.held.contains_key(name)
    }

    fn touch(&mut self, name: &str) {
        if let Some(at) = self.order.iter().position(|d| d == name) {
            self.order.remove(at);
        }
        self.order.push_back(name.to_string());
    }

    fn hold(&mut self, name: String, chain: SavedChain) {
        let size = serde_json::to_vec(&chain).map_or(0, |b| b.len());
        if let Some((_, was)) = self.held.insert(name.clone(), (chain, size)) {
            self.bytes -= was;
        }
        self.bytes += size;
        self.touch(&name);
        while self.bytes > self.cap && self.order.len() > 1 {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some((_, size)) = self.held.remove(&old) {
                self.bytes -= size;
            }
        }
    }

    /// `overlay` as the snapshot its run lays down: each path's chain from
    /// those it carries — each checked against its name, then held — or from
    /// those held already.
    pub fn resolve(&mut self, overlay: &Overlay) -> Result<Snapshot, Unresolved> {
        for (name, chain) in &overlay.carried {
            if digest(chain) != *name {
                return Err(Unresolved::Misnamed(name.clone()));
            }
        }
        let mut chains = BTreeMap::new();
        let mut missing = Vec::new();
        for (path, name) in &overlay.chains {
            let chain = overlay
                .carried
                .get(name)
                .or_else(|| self.held.get(name).map(|(c, _)| c));
            match chain {
                Some(c) => {
                    chains.insert(path.clone(), c.clone());
                }
                None => missing.push(name.clone()),
            }
        }
        for (name, chain) in &overlay.carried {
            self.hold(name.clone(), chain.clone());
        }
        if !missing.is_empty() {
            missing.sort();
            missing.dedup();
            return Err(Unresolved::Missing(missing));
        }
        for name in overlay.chains.values() {
            self.touch(name);
        }
        Ok(Snapshot::new(overlay.base.clone(), chains))
    }
}

#[cfg(test)]
mod tests {
    use zend_vfs::VfsStore;

    use super::*;

    fn overlay() -> Overlay {
        let store = VfsStore::new();
        store.write("a.txt", "one\n".into()).unwrap();
        store.write("b.txt", "two\n".into()).unwrap();
        Overlay::of(&store.snapshot())
    }

    /// **Names alone resolve once the chains are held**: the first send by
    /// name is answered with every digest, a send carrying them resolves to
    /// the snapshot they came from, and the names alone then resolve too.
    #[test]
    fn names_resolve_once_their_chains_are_held() {
        let store = VfsStore::new();
        store.write("a.txt", "one\n".into()).unwrap();
        let snapshot = store.snapshot();
        let full = Overlay::of(&snapshot);
        let mut chains = Chains::default();
        let name = full.chains["a.txt"].clone();
        assert_eq!(
            chains.resolve(&full.named()),
            Err(Unresolved::Missing(vec![name.clone()]))
        );
        assert_eq!(chains.resolve(&full.carrying(&[name])).unwrap(), snapshot);
        assert_eq!(chains.resolve(&full.named()).unwrap(), snapshot);
    }

    /// **A carried chain that is not what its name says is refused**, and
    /// nothing it carried is held.
    #[test]
    fn a_misnamed_chain_is_refused() {
        let mut full = overlay();
        let a = full.chains["a.txt"].clone();
        let b = full.chains["b.txt"].clone();
        let other = full.carried[&b].clone();
        full.carried.insert(a.clone(), other);
        assert_eq!(chains_of(&full), Err(Unresolved::Misnamed(a)));
    }

    fn chains_of(o: &Overlay) -> Result<Snapshot, Unresolved> {
        let mut chains = Chains::default();
        let got = chains.resolve(o);
        assert_eq!(chains.bytes(), 0);
        got
    }

    /// **The least recently used chain goes first** once the bytes held pass
    /// the cap; one just used stays.
    #[test]
    fn the_least_recently_used_chain_goes_first() {
        let full = overlay();
        let (a, b) = (full.chains["a.txt"].clone(), full.chains["b.txt"].clone());
        let one = serde_json::to_vec(&full.carried[&a]).unwrap().len();
        let mut chains = Chains::new(one + one / 2);
        let only = |name: &str| Overlay {
            chains: BTreeMap::from([("x".to_string(), name.to_string())]),
            ..full.carrying(&[name.to_string()])
        };
        chains.resolve(&only(&a)).unwrap();
        chains.resolve(&only(&b)).unwrap();
        assert!(!chains.holds(&a), "the older chain went");
        assert!(chains.holds(&b));
    }
    /// **A chain named again outlives one not named**: touching a chain on
    /// resolve makes it the most recently used, so the next eviction takes the
    /// one that was never named again.
    #[test]
    fn a_chain_named_again_outlives_one_not_named() {
        let store = VfsStore::new();
        store.write("a.txt", "one\n".into()).unwrap();
        store.write("b.txt", "two\n".into()).unwrap();
        store.write("c.txt", "x\n".into()).unwrap();
        let snapshot = store.snapshot();
        let full = Overlay::of(&snapshot);
        let (a, b, c) = (
            full.chains["a.txt"].clone(),
            full.chains["b.txt"].clone(),
            full.chains["c.txt"].clone(),
        );
        let size_a = serde_json::to_vec(&full.carried[&a]).unwrap().len();
        let size_b = serde_json::to_vec(&full.carried[&b]).unwrap().len();
        let mut chains = Chains::new(size_a + size_b);
        let only = |name: &str| Overlay {
            chains: BTreeMap::from([("x".to_string(), name.to_string())]),
            ..full.carrying(&[name.to_string()])
        };
        chains.resolve(&only(&a)).unwrap();
        chains.resolve(&only(&b)).unwrap();
        // Name A alone — no carried — so A is touched but not re-held.
        let named_a = Overlay {
            chains: BTreeMap::from([("x".to_string(), a.clone())]),
            ..Default::default()
        };
        chains.resolve(&named_a).unwrap();
        // Hold C; the cap fits two chains, so B (the LRU) goes.
        chains.resolve(&only(&c)).unwrap();
        assert!(chains.holds(&a), "A was named again and should stay");
        assert!(!chains.holds(&b), "B was never named again and should go");
        assert!(chains.holds(&c), "C was just held");
    }
}
