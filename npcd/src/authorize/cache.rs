//! What this member knows of prompts, by purpose and prompt, for an
//! authorization's lifetime (`docs/guest_draw_orchestration.md` §2): the
//! authorizations it holds, so a retry of the same prompt — a `no_room` retry,
//! a redraw at a new seed — asks no one; and the prompts it saw refused, so a
//! declined prompt sent again is answered at once instead of drawn again
//! before its refusal comes back.

use std::collections::HashMap;
use std::sync::Mutex;

use super::token::{prompt_sha256, Authorization, LIFETIME};

/// The most entries kept of each kind. Past it the soonest to expire goes.
const CAPACITY: usize = 512;

/// A prompt and the purpose it was asked for, by the prompt's digest.
type Key = (String, String);

/// Authorizations and refusals by `(purpose, prompt_sha256)`.
#[derive(Debug, Default)]
pub struct Cache {
    held: Mutex<HashMap<Key, Authorization>>,
    /// When each refusal stops counting, Unix milliseconds.
    refused: Mutex<HashMap<Key, u64>>,
}

impl Cache {
    /// Keep `auth`, dropping any that expired by `now_ms`, and the soonest to
    /// expire when full.
    pub fn put(&self, auth: Authorization, now_ms: u64) {
        let key = (auth.purpose.clone(), auth.prompt_sha256.clone());
        let mut held = self.held.lock().unwrap();
        held.retain(|_, a| a.exp_ms > now_ms);
        if !held.contains_key(&key) && held.len() >= CAPACITY {
            if let Some(first) = held
                .iter()
                .min_by_key(|(_, a)| a.exp_ms)
                .map(|(k, _)| k.clone())
            {
                held.remove(&first);
            }
        }
        held.insert(key, auth);
    }

    /// The authorization kept for `prompt` and `purpose`, while it counts at
    /// `now_ms`.
    pub fn get(&self, purpose: &str, prompt: &str, now_ms: u64) -> Option<Authorization> {
        let key = (purpose.to_string(), prompt_sha256(prompt));
        let mut held = self.held.lock().unwrap();
        match held.get(&key) {
            Some(a) if a.exp_ms > now_ms => Some(a.clone()),
            Some(_) => {
                held.remove(&key);
                None
            }
            None => None,
        }
    }

    /// Remember that `prompt` was refused for `purpose` at `now_ms`.
    pub fn refuse(&self, purpose: &str, prompt: &str, now_ms: u64) {
        let key = (purpose.to_string(), prompt_sha256(prompt));
        let mut refused = self.refused.lock().unwrap();
        refused.retain(|_, exp| *exp > now_ms);
        if !refused.contains_key(&key) && refused.len() >= CAPACITY {
            if let Some(first) = refused
                .iter()
                .min_by_key(|(_, exp)| **exp)
                .map(|(k, _)| k.clone())
            {
                refused.remove(&first);
            }
        }
        refused.insert(key, now_ms + LIFETIME.as_millis() as u64);
    }

    /// Whether `prompt` was refused for `purpose` within an authorization's
    /// lifetime of `now_ms`.
    pub fn refused(&self, purpose: &str, prompt: &str, now_ms: u64) -> bool {
        let key = (purpose.to_string(), prompt_sha256(prompt));
        self.refused
            .lock()
            .unwrap()
            .get(&key)
            .is_some_and(|exp| *exp > now_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorize::token::{Signer, IMAGE};

    /// **Kept for its prompt and purpose until it expires.**
    #[test]
    fn an_authorization_is_kept_until_it_expires() {
        let s = Signer::new([2; 32]);
        let cache = Cache::default();
        let a = s.sign(IMAGE, "a fox", 1_000);
        cache.put(a.clone(), 1_000);
        assert_eq!(cache.get(IMAGE, "a fox", 2_000), Some(a.clone()));
        assert_eq!(cache.get(IMAGE, "a wolf", 2_000), None);
        assert_eq!(cache.get("portrait", "a fox", 2_000), None);
        assert_eq!(cache.get(IMAGE, "a fox", a.exp_ms), None, "expired");
    }

    /// **Bounded**: past capacity the soonest to expire goes — and putting a
    /// prompt it already holds takes nothing else out.
    #[test]
    fn it_holds_at_most_its_capacity() {
        let s = Signer::new([2; 32]);
        let cache = Cache::default();
        for i in 0..=CAPACITY as u64 {
            cache.put(s.sign(IMAGE, &format!("p{i}"), i), 0);
        }
        assert_eq!(cache.held.lock().unwrap().len(), CAPACITY);
        assert_eq!(
            cache.get(IMAGE, "p0", 0),
            None,
            "the soonest to expire went"
        );
        assert!(cache.get(IMAGE, &format!("p{CAPACITY}"), 0).is_some());

        cache.put(s.sign(IMAGE, "p1", 5), 0);
        assert_eq!(cache.held.lock().unwrap().len(), CAPACITY);
        assert!(
            cache.get(IMAGE, "p2", 0).is_some(),
            "a prompt put again evicted another"
        );
    }

    /// **A refusal is remembered for its prompt and purpose for an
    /// authorization's lifetime**, and then forgotten.
    #[test]
    fn a_refusal_is_remembered_for_a_while() {
        let cache = Cache::default();
        cache.refuse(IMAGE, "a fox", 1_000);
        assert!(cache.refused(IMAGE, "a fox", 2_000));
        assert!(!cache.refused(IMAGE, "a wolf", 2_000));
        assert!(!cache.refused("portrait", "a fox", 2_000));
        assert!(!cache.refused(IMAGE, "a fox", 1_000 + LIFETIME.as_millis() as u64));
    }
}
