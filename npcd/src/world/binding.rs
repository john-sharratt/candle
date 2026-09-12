//! Which mind is which body.
//!
//! A character in this daemon is a number; a body in a world is a name in a
//! map. Binding is the correspondence, and it is deliberately its own thing
//! rather than a field on either side — a character can exist without a world
//! (most do), a world can hold bodies nothing is thinking for (a scripted
//! extra), and neither should have to carry a hole shaped like the other.
//!
//! # It is one to one, in both directions, and enforced
//!
//! One mind cannot be two bodies: it would perceive from two places at once and
//! the percept is a point in time, singular. One body cannot be two minds: they
//! would take turns acting through the same hands, and each would read the
//! other's acts as its own.
//!
//! Both are refused rather than overwritten. A silent rebind is the worst
//! version: everything keeps working, one character quietly stops perceiving,
//! and nothing anywhere says which.
//!
//! # What a binding buys, and what it costs
//!
//! A bound character draws its perception from the world it stands in, which
//! means the push endpoint and the narrating environment simulator both become
//! *wrong* for it rather than merely unused — two sources of truth for where a
//! body is standing is the same bug as two world simulations. [`Bindings::bound`]
//! is what those refusals are checked against.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Mutex;

/// Where a mind's body is.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bound {
    /// The world it stands in, by the id it was hosted under.
    pub world: String,
    /// The body it is, by the id the map knows.
    pub body: String,
}

/// Why a binding was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Taken {
    /// This mind is already somewhere.
    MindIsElsewhere { npc_id: u64, at: Bound },
    /// This body already has a mind.
    BodyHasAMind { at: Bound, npc_id: u64 },
}

impl fmt::Display for Taken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Taken::MindIsElsewhere { npc_id, at } => {
                write!(f, "{npc_id} is already {}/{}", at.world, at.body)
            }
            Taken::BodyHasAMind { at, npc_id } => {
                write!(f, "{}/{} is already {npc_id}", at.world, at.body)
            }
        }
    }
}

impl std::error::Error for Taken {}

/// Every mind that has a body, and every body that has a mind.
#[derive(Default)]
pub struct Bindings {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    by_mind: BTreeMap<u64, Bound>,
    /// The reverse index, kept rather than derived. Refusing a second mind for
    /// a body is a check on every bind, and a scan of every binding to answer
    /// it would make binding a population-sized operation.
    by_body: BTreeMap<Bound, u64>,
}

impl Bindings {
    pub fn new() -> Bindings {
        Bindings::default()
    }

    /// Give a mind a body. Refused if either side is already spoken for.
    ///
    /// Binding a mind to the body it already has succeeds and changes nothing,
    /// so a caller that cannot easily tell whether it has already bound is not
    /// forced to find out first.
    pub fn bind(
        &self,
        npc_id: u64,
        world: impl Into<String>,
        body: impl Into<String>,
    ) -> Result<Bound, Taken> {
        let want = Bound {
            world: world.into(),
            body: body.into(),
        };
        let mut inner = self.inner.lock().expect("bindings lock");

        if let Some(at) = inner.by_mind.get(&npc_id) {
            return if at == &want {
                Ok(want)
            } else {
                Err(Taken::MindIsElsewhere {
                    npc_id,
                    at: at.clone(),
                })
            };
        }
        if let Some(other) = inner.by_body.get(&want) {
            return Err(Taken::BodyHasAMind {
                at: want,
                npc_id: *other,
            });
        }

        inner.by_mind.insert(npc_id, want.clone());
        inner.by_body.insert(want.clone(), npc_id);
        Ok(want)
    }

    /// Where a mind's body is, if it has one.
    pub fn bound(&self, npc_id: u64) -> Option<Bound> {
        self.inner
            .lock()
            .expect("bindings lock")
            .by_mind
            .get(&npc_id)
            .cloned()
    }

    /// Whether this mind draws its perception from a world.
    ///
    /// What the push endpoint and the narrating simulator are refused against.
    pub fn is_bound(&self, npc_id: u64) -> bool {
        self.bound(npc_id).is_some()
    }

    /// Whose body this is, if anyone's.
    pub fn mind_of(&self, world: &str, body: &str) -> Option<u64> {
        let key = Bound {
            world: world.to_string(),
            body: body.to_string(),
        };
        self.inner
            .lock()
            .expect("bindings lock")
            .by_body
            .get(&key)
            .copied()
    }

    /// Every mind with a body in this world, and which body it is.
    ///
    /// In body order, so a sweep over a world visits its minds the same way
    /// twice — a report that reshuffles between two reads is a report nobody
    /// can diff.
    pub fn in_world(&self, world: &str) -> Vec<(u64, String)> {
        self.inner
            .lock()
            .expect("bindings lock")
            .by_body
            .iter()
            .filter(|(at, _)| at.world == world)
            .map(|(at, npc)| (*npc, at.body.clone()))
            .collect()
    }

    /// Take a mind's body away. `false` if it had none.
    pub fn unbind(&self, npc_id: u64) -> bool {
        let mut inner = self.inner.lock().expect("bindings lock");
        let Some(at) = inner.by_mind.remove(&npc_id) else {
            return false;
        };
        inner.by_body.remove(&at);
        true
    }

    /// Unbind every mind in a world. What releasing a world means for the minds
    /// that were standing in it.
    pub fn release_world(&self, world: &str) -> usize {
        let mut inner = self.inner.lock().expect("bindings lock");
        let gone: Vec<Bound> = inner
            .by_body
            .keys()
            .filter(|at| at.world == world)
            .cloned()
            .collect();
        for at in &gone {
            if let Some(npc) = inner.by_body.remove(at) {
                inner.by_mind.remove(&npc);
            }
        }
        gone.len()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("bindings lock").by_mind.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(world: &str, body: &str) -> Bound {
        Bound {
            world: world.into(),
            body: body.into(),
        }
    }

    #[test]
    fn a_mind_with_a_body_can_be_found_from_either_side() {
        let b = Bindings::new();
        b.bind(1, "vault", "m1").unwrap();

        assert_eq!(b.bound(1), Some(bound("vault", "m1")));
        assert_eq!(b.mind_of("vault", "m1"), Some(1));
        assert!(b.is_bound(1));
        assert!(!b.is_bound(2));
    }

    #[test]
    fn a_mind_cannot_be_two_bodies() {
        // It would perceive from two places at once, and a percept is a point
        // in time — singular.
        let b = Bindings::new();
        b.bind(1, "vault", "m1").unwrap();
        assert_eq!(
            b.bind(1, "vault", "m2"),
            Err(Taken::MindIsElsewhere {
                npc_id: 1,
                at: bound("vault", "m1")
            })
        );
        assert_eq!(b.bound(1), Some(bound("vault", "m1")), "it moved anyway");
        assert!(b.mind_of("vault", "m2").is_none());
    }

    #[test]
    fn a_body_cannot_be_two_minds() {
        // They would act through the same hands and read each other's acts as
        // their own.
        let b = Bindings::new();
        b.bind(1, "vault", "m1").unwrap();
        assert_eq!(
            b.bind(2, "vault", "m1"),
            Err(Taken::BodyHasAMind {
                at: bound("vault", "m1"),
                npc_id: 1
            })
        );
        assert!(!b.is_bound(2));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn binding_a_mind_to_the_body_it_already_has_changes_nothing() {
        let b = Bindings::new();
        b.bind(1, "vault", "m1").unwrap();
        assert_eq!(b.bind(1, "vault", "m1"), Ok(bound("vault", "m1")));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn the_same_body_id_in_two_worlds_is_two_bodies() {
        let b = Bindings::new();
        b.bind(1, "vault", "m1").unwrap();
        b.bind(2, "harbour", "m1").expect("a different world");
        assert_eq!(b.mind_of("vault", "m1"), Some(1));
        assert_eq!(b.mind_of("harbour", "m1"), Some(2));
    }

    #[test]
    fn a_refusal_names_both_sides_of_the_clash() {
        // An operator reading this has to know which mind and which body,
        // because the fix is to detach one of them.
        let b = Bindings::new();
        b.bind(7, "vault", "m1").unwrap();

        let mine = b.bind(7, "vault", "m2").unwrap_err().to_string();
        assert!(mine.contains('7') && mine.contains("vault/m1"), "{mine}");

        let theirs = b.bind(9, "vault", "m1").unwrap_err().to_string();
        assert!(
            theirs.contains('7') && theirs.contains("vault/m1"),
            "{theirs}"
        );
    }

    #[test]
    fn unbinding_frees_both_sides() {
        let b = Bindings::new();
        b.bind(1, "vault", "m1").unwrap();
        assert!(b.unbind(1));
        assert!(!b.unbind(1), "unbound twice");
        assert!(b.is_empty());

        // Both halves of the index were cleared, so either can be rebound.
        b.bind(2, "vault", "m1").expect("the body was freed");
        b.bind(1, "vault", "m2").expect("the mind was freed");
    }

    #[test]
    fn a_world_lists_its_minds_in_a_stable_order() {
        let b = Bindings::new();
        for (npc, body) in [(3, "m3"), (1, "m1"), (2, "m2")] {
            b.bind(npc, "vault", body).unwrap();
        }
        b.bind(9, "harbour", "ana").unwrap();

        let here = b.in_world("vault");
        assert_eq!(
            here,
            vec![(1, "m1".into()), (2, "m2".into()), (3, "m3".into())]
        );
        assert_eq!(b.in_world("harbour"), vec![(9, "ana".into())]);
        assert!(b.in_world("nowhere").is_empty());
    }

    #[test]
    fn releasing_a_world_unbinds_its_minds_and_leaves_the_others() {
        let b = Bindings::new();
        b.bind(1, "vault", "m1").unwrap();
        b.bind(2, "vault", "m2").unwrap();
        b.bind(9, "harbour", "ana").unwrap();

        assert_eq!(b.release_world("vault"), 2);
        assert!(!b.is_bound(1));
        assert!(!b.is_bound(2));
        assert!(b.is_bound(9), "another world's minds were released");
        assert_eq!(b.release_world("vault"), 0);

        // And the reverse index went with them.
        assert!(b.mind_of("vault", "m1").is_none());
        b.bind(5, "vault", "m1").expect("the body was freed");
    }
}
