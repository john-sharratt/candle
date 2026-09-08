//! The worlds this daemon hosts, and the bodies standing in them.
//!
//! A world is a directory of map files loaded into one [`npc_map::World`]: who
//! is where, what is claimed, what just happened. This module owns them and
//! nothing else — it does not know that anything perceives them, which is why
//! it has no dependency on the engine and can be exercised without one.
//!
//! # One lock, one world
//!
//! Every mutation in `npc_map` is a handful of map operations, so a mutex
//! around the whole world costs nothing worth measuring and buys the invariant
//! that is otherwise impossible: there is exactly one answer to *who has that
//! character*. Sixteen Makers reading and writing one building is the case the
//! world was written for, not one it was adapted to.
//!
//! The reader bookkeeping ([`npc_map::Attention`]) sits under the same lock. It
//! has to: composing a delta reads the world and advances what that reader was
//! last shown, and two locks taken in two orders by two threads is the one bug
//! this design would otherwise have.

pub mod binding;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use npc_map::delta::{Attention, Delta};
use npc_map::world::World;
use npc_map::MapSet;

/// One world, and everything about who has been told what in it.
pub struct Hosted {
    id: String,
    state: Mutex<State>,
}

struct State {
    world: World,
    attention: Attention,
}

impl Hosted {
    /// Load a world from a directory of map files.
    ///
    /// Validation happens at load, in `npc_map`: a map that does not hold
    /// together cannot be walked, and finding that out when a body tries to
    /// leave a room is finding out far too late.
    pub fn load(id: impl Into<String>, dir: impl AsRef<Path>) -> Result<Hosted> {
        let id = id.into();
        let dir = dir.as_ref();
        let map = MapSet::load_dir(dir)
            .with_context(|| format!("world `{id}` from {}", dir.display()))?;
        Ok(Hosted {
            id,
            state: Mutex::new(State {
                world: World::new(map),
                attention: Attention::new(),
            }),
        })
    }

    /// Host a world already in memory. What a generated world arrives as, and
    /// what a test uses.
    pub fn of(id: impl Into<String>, world: World) -> Hosted {
        Hosted {
            id: id.into(),
            state: Mutex::new(State {
                world,
                attention: Attention::new(),
            }),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Do something to the world, under the lock.
    ///
    /// The one way in. Every act a body takes — walking, speaking, sitting down
    /// — comes through here, so there is no path that reads the world without
    /// holding it still.
    pub fn with<T>(&self, f: impl FnOnce(&mut World) -> T) -> T {
        let mut state = self.state.lock().expect("world lock");
        f(&mut state.world)
    }

    /// Look at the world without changing it.
    pub fn read<T>(&self, f: impl FnOnce(&World) -> T) -> T {
        let state = self.state.lock().expect("world lock");
        f(&state.world)
    }

    /// Advance one moment: everybody on their way covers a leg.
    ///
    /// Returns how many bodies moved. Nobody walking is the common case and
    /// costs a lock and a scan of the actor map.
    pub fn tick(&self) -> usize {
        let mut state = self.state.lock().expect("world lock");
        state.world.tick()
    }

    /// What every body has to be told, and nothing for the ones with nothing.
    ///
    /// One pass under one lock, because the alternative — a lock per body — is
    /// sixteen acquisitions to answer one question and leaves the world free to
    /// move between two bodies' readings of the same moment.
    pub fn sweep(&self) -> Vec<Delta> {
        let mut state = self.state.lock().expect("world lock");
        let State { world, attention } = &mut *state;
        attention.sweep(world)
    }

    /// What one body has to be told, marking it as delivered.
    pub fn delta(&self, body: &str) -> Delta {
        let mut state = self.state.lock().expect("world lock");
        let State { world, attention } = &mut *state;
        attention.take(world, body)
    }

    /// What one body would be told, without delivering it.
    pub fn peek(&self, body: &str) -> Delta {
        let state = self.state.lock().expect("world lock");
        state.attention.peek(&state.world, body)
    }

    /// Forget what a body was last shown, so its next delta grounds it again.
    /// For a body leaving, or a mind whose context was rebuilt beneath it.
    pub fn forget(&self, body: &str) {
        let mut state = self.state.lock().expect("world lock");
        state.attention.forget(body);
    }
}

impl std::fmt::Debug for Hosted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The world behind the lock is large and taking it to format is a
        // deadlock waiting for a debug print inside a `with`.
        f.debug_struct("Hosted").field("id", &self.id).finish()
    }
}

/// Every world the daemon has loaded.
#[derive(Default)]
pub struct Worlds {
    worlds: Mutex<BTreeMap<String, Arc<Hosted>>>,
}

impl Worlds {
    pub fn new() -> Worlds {
        Worlds::default()
    }

    /// Load a world and keep it. Replaces one already loaded under that id —
    /// which is what a map edit during authoring means, and the bodies in the
    /// old one go with it.
    pub fn load(&self, id: impl Into<String>, dir: impl AsRef<Path>) -> Result<Arc<Hosted>> {
        let hosted = Arc::new(Hosted::load(id, dir)?);
        self.keep(hosted.clone());
        Ok(hosted)
    }

    /// Keep a world that was built rather than loaded.
    pub fn keep(&self, hosted: Arc<Hosted>) -> Arc<Hosted> {
        let mut worlds = self.worlds.lock().expect("worlds lock");
        worlds.insert(hosted.id.clone(), hosted.clone());
        hosted
    }

    pub fn get(&self, id: &str) -> Option<Arc<Hosted>> {
        self.worlds.lock().expect("worlds lock").get(id).cloned()
    }

    pub fn ids(&self) -> Vec<String> {
        self.worlds
            .lock()
            .expect("worlds lock")
            .keys()
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.worlds.lock().expect("worlds lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stop hosting a world. `false` if there was none.
    ///
    /// Anything still holding an `Arc` keeps its copy alive and working — which
    /// is correct: a tick already in flight finishes against the world it
    /// started on rather than panicking half way through a moment.
    pub fn release(&self, id: &str) -> bool {
        self.worlds
            .lock()
            .expect("worlds lock")
            .remove(id)
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use npc_map::world::Where;

    fn vault() -> Hosted {
        Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load")
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    #[test]
    fn a_world_loads_from_its_map_files_and_knows_its_own_id() {
        let h = vault();
        assert_eq!(h.id(), "creators-vault");
        assert!(h.read(|w| w.map().get("vault-casting").is_some()));
    }

    #[test]
    fn a_directory_that_is_not_a_world_says_which_one_it_was() {
        // The error has to name the world, or an operator with six loaded is
        // told only that "a" map failed.
        let err = Hosted::load("harbour", concat!(env!("CARGO_MANIFEST_DIR"), "/src"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("harbour"), "{err}");
    }

    #[test]
    fn acts_go_through_the_lock_and_are_seen_by_the_next_reader() {
        let h = vault();
        h.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        h.with(|w| w.take("m1", Some("cindy")).unwrap());
        assert!(h.read(|w| w.holder_of("cindy").is_some()));
    }

    #[test]
    fn a_sweep_grounds_everybody_once_and_then_goes_quiet() {
        let h = vault();
        h.with(|w| {
            for i in 1..=4 {
                w.enter(format!("m{i}"), format!("Maker-{i:02}"), at("core"))
                    .unwrap();
            }
        });
        assert_eq!(h.sweep().len(), 4);
        assert!(h.sweep().is_empty(), "an idle world cost something");
    }

    #[test]
    fn a_tick_moves_the_bodies_on_their_way_and_nobody_else() {
        let h = vault();
        h.with(|w| {
            w.enter("walker", "Maker-01", at("band-one")).unwrap();
            w.enter("sitter", "Maker-02", at("band-one")).unwrap();
            w.set_off("walker", at("green-room")).unwrap();
        });
        assert_eq!(h.tick(), 1);
        assert_eq!(h.tick(), 0, "a settled world kept moving");
        assert!(h.read(|w| w.actor("walker").unwrap().at == at("green-room")));
        assert!(h.read(|w| w.actor("sitter").unwrap().at == at("band-one")));
    }

    #[test]
    fn peeking_does_not_spend_what_a_delta_would_carry() {
        let h = vault();
        h.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        let first = h.peek("m1");
        assert_eq!(h.peek("m1"), first, "peeking twice differed");
        assert_eq!(h.delta("m1"), first);
        assert!(h.delta("m1").is_empty(), "taking spent nothing");
    }

    #[test]
    fn forgetting_a_body_grounds_it_again() {
        let h = vault();
        h.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        h.delta("m1");
        assert!(h.delta("m1").is_empty());
        h.forget("m1");
        assert!(h.delta("m1").percept.is_some());
    }

    #[test]
    fn a_world_is_shared_by_reference_and_seen_the_same_way_by_everyone() {
        // The whole point of hosting: two handles are one world, not two.
        let worlds = Worlds::new();
        let a = worlds
            .load(
                "creators-vault",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
            )
            .unwrap();
        let b = worlds.get("creators-vault").expect("kept");

        a.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        assert!(b.read(|w| w.actor("m1").is_some()), "two worlds, not one");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn a_released_world_stops_being_listed_and_keeps_working_for_its_holders() {
        let worlds = Worlds::new();
        let held = worlds
            .load(
                "creators-vault",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
            )
            .unwrap();
        assert_eq!(worlds.ids(), vec!["creators-vault"]);

        assert!(worlds.release("creators-vault"));
        assert!(!worlds.release("creators-vault"), "released twice");
        assert!(worlds.is_empty());
        assert!(worlds.get("creators-vault").is_none());

        // A tick already in flight finishes against the world it started on.
        held.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        assert!(held.read(|w| w.actor("m1").is_some()));
    }

    #[test]
    fn loading_over_a_world_replaces_it_and_the_bodies_in_it() {
        let worlds = Worlds::new();
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
        let first = worlds.load("creators-vault", dir).unwrap();
        first.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());

        let second = worlds.load("creators-vault", dir).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(second.read(|w| w.actor("m1").is_none()), "a body survived");
        assert_eq!(worlds.len(), 1);
    }
}
