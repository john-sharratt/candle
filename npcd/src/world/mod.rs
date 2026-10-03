//! The worlds this daemon hosts, and the bodies standing in them.
//!
//! A world is a directory of map files loaded into one [`npc_map::World`]: who
//! is where, what is claimed, what just happened. This module owns them, and it
//! still knows nothing about anything *perceiving* them — no scheduler, no
//! minds, no windows — so it can be exercised without an engine.
//!
//! It does own the building's own doings ([`crate::engine::rooms`]), because a
//! stirring is written into the world log and producing one is therefore a
//! world mutation like any other. Holding it anywhere else would mean a second
//! lock, or a route that advances the world without the building noticing.
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
pub mod mapstore;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use npc_map::delta::{Attention, Delta};
use npc_map::world::World;
use npc_map::{MapEdit, MapSet};

use crate::engine::rooms::Rooms;
use crate::engine::watch::Watch;
use crate::sim::{seed, Sim};

/// One world, and everything about who has been told what in it.
pub struct Hosted {
    id: String,
    state: Mutex<State>,
    /// The directory this world's map was loaded from, when it was loaded from
    /// disk — where a runtime reshape ([`Hosted::reshape`]) writes the change
    /// back so it survives a restart. `None` for an in-memory world
    /// ([`Hosted::of`], a generated world or a test), whose reshapes are
    /// ephemeral by construction: there is no authored file to keep them in.
    map_dir: Option<PathBuf>,
    /// Serialises the write-back half of [`Hosted::reshape`] — deliberately a
    /// second lock, separate from `state`, because that write is deliberately
    /// off the world lock for latency. Without it, two reshapes landing on the
    /// same area at once would both call [`mapstore::persist`] and both write
    /// the identical `<area>.yaml.tmp` path at once, with nothing stopping the
    /// two `File::create`/`write_all` calls from interleaving before either
    /// `rename` lands.
    writeback: Mutex<()>,
}

struct State {
    world: World,
    attention: Attention,
    /// Everything about the world that is not its shape — see [`crate::sim`].
    ///
    /// **Under the same lock as the world, deliberately.** Composing a situation
    /// reads where a body is standing *and* what is within reach of it there,
    /// and two locks taken in two orders by two threads is the one bug this
    /// design would otherwise have. It is the same argument the attention
    /// bookkeeping is here for.
    sim: Sim,
    /// What the building itself is doing, room by room — see
    /// [`crate::engine::rooms`].
    ///
    /// Under the world's lock for the plainest of the reasons: a stirring is
    /// written into the world log, so producing one *is* a world mutation and
    /// has to be serialised with every other.
    rooms: Rooms,
    /// The tower's clock — see [`crate::engine::watch`]. Under the world's lock
    /// because what it says is stirred into the world log.
    watch: Watch,
}

/// What a successful reshape did — who it displaced, and whether it was kept.
#[derive(Debug, Clone)]
pub struct Reshaped {
    /// The ids of bodies relocated because the room they stood in was drowned,
    /// in a stable order. Empty when the reshape displaced nobody.
    pub relocated: Vec<String>,
    /// Whether the change was written back to the authored map.
    pub durability: Durability,
}

/// Whether a reshape survives a restart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Durability {
    /// Written back to the authored YAML — it will still be there after a reboot.
    Written,
    /// The change is live but was not written back (the write failed). The reason
    /// is carried so an operator learns the reshape is not yet durable.
    Failed(String),
    /// This world has no authored directory, so the change holds for the run only.
    Ephemeral,
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
        // Seeded from the map it was just loaded from, so what stands in a room
        // is written down in exactly one place. A second list here would be
        // free to disagree with the building, and would.
        let sim = seed::for_world(&id, Some(&map));
        Ok(Hosted {
            id,
            state: Mutex::new(State {
                world: World::new(map),
                attention: Attention::new(),
                sim,
                rooms: Rooms::new(),
                watch: Watch::new(),
            }),
            map_dir: Some(dir.to_path_buf()),
            writeback: Mutex::new(()),
        })
    }

    /// Host a world already in memory. What a generated world arrives as, and
    /// what a test uses.
    pub fn of(id: impl Into<String>, world: World) -> Hosted {
        let id = id.into();
        let sim = seed::for_world(&id, Some(world.map()));
        Hosted {
            id,
            state: Mutex::new(State {
                world,
                attention: Attention::new(),
                sim,
                rooms: Rooms::new(),
                watch: Watch::new(),
            }),
            // An in-memory world has no authored directory, so its reshapes are
            // ephemeral: nothing to write them back to.
            map_dir: None,
            writeback: Mutex::new(()),
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

    /// Change what the world holds — packs, machines, ground, the tower.
    pub fn with_sim<T>(&self, f: impl FnOnce(&mut Sim) -> T) -> T {
        let mut state = self.state.lock().expect("world lock");
        f(&mut state.sim)
    }

    /// Read what the world holds.
    pub fn sim<T>(&self, f: impl FnOnce(&Sim) -> T) -> T {
        let state = self.state.lock().expect("world lock");
        f(&state.sim)
    }

    /// Both halves at once, under one acquisition.
    ///
    /// The one an act needs: performing `gather` reads where a body is standing
    /// and writes what it now carries, and taking the lock twice would let the
    /// world move between the two.
    pub fn with_both<T>(&self, f: impl FnOnce(&mut World, &mut Sim) -> T) -> T {
        let mut state = self.state.lock().expect("world lock");
        let State { world, sim, .. } = &mut *state;
        f(world, sim)
    }

    /// Reshape this world's walkable map while it runs, and keep the change
    /// (effector design Appendix F).
    ///
    /// **Two steps, in order: mutate, then persist.** The in-RAM mutation runs
    /// under the world lock ([`World::reshape`]) — swap the map, re-derive the
    /// lift shaft, relocate anyone standing where a room was drowned — and refuses
    /// without touching the world if the edit would not validate. Only once that
    /// has landed is the change written back to the authored YAML, so the swap is
    /// the source of truth and the file follows it.
    ///
    /// **The write is off the lock, and best-effort.** The map is snapshotted
    /// under the lock and serialised outside it, so a disk stall never freezes
    /// ticks; and a write that fails is *reported* ([`Durability::Failed`]) rather
    /// than un-happening a reshape that already took — the operator is told the
    /// change is live but not yet durable, which is the truth, instead of a lie in
    /// either direction. A world with no authored directory ([`Hosted::of`]) is
    /// [`Durability::Ephemeral`]: the reshape holds for the run and is gone on
    /// restart, because there is nowhere to keep it.
    ///
    /// **The write-back itself is serialised** against every other reshape's
    /// write-back on this world ([`Self::writeback`]), so two edits landing on
    /// the same area in quick succession still write it one at a time rather
    /// than racing on the same temp file.
    ///
    /// `Err(reason)` is the world's own words on why the edit was refused, with
    /// nothing changed.
    pub fn reshape(&self, edit: &MapEdit) -> Result<Reshaped, String> {
        let relocated = self.with(|w| w.reshape(edit))?;
        let durability = match &self.map_dir {
            None => Durability::Ephemeral,
            Some(dir) => {
                // Held for the snapshot and the write together, so no other
                // reshape's write-back can interleave with this one's.
                let _writeback = self.writeback.lock().expect("writeback lock");
                let map = self.read(|w| w.map().clone());
                match mapstore::persist(dir, edit, &map) {
                    Ok(()) => Durability::Written,
                    Err(e) => {
                        tracing::error!(
                            world = %self.id,
                            area = %edit.area(),
                            error = %e,
                            "reshape landed in memory but could not be written back"
                        );
                        Durability::Failed(e.to_string())
                    }
                }
            }
        };
        Ok(Reshaped {
            relocated,
            durability,
        })
    }

    /// Point this world's benches at the documents they work on.
    pub fn set_bench_root(&self, root: impl Into<std::path::PathBuf>) {
        self.with_sim(|s| s.set_bench_root(root));
    }

    /// Where a body is standing, as the `area/node` string the sim keys on.
    ///
    /// Empty for a body that is not in the world, which is the honest answer:
    /// nothing is within reach of somewhere that is not a place.
    pub fn place_of(&self, body: &str) -> String {
        self.read(|w| {
            w.actor(body)
                .map(|a| format!("{}/{}", a.at.area, a.at.node))
                .unwrap_or_default()
        })
    }

    /// The room an act is about, as the `area/node` string the sim keys on: the
    /// one the body began deciding in, while it is deciding, and where it stands
    /// otherwise.
    ///
    /// **The grammar names what is in the room the body read, and the decode
    /// outlasts a step of the metronome.** A moving body can be a leg on by the
    /// time its call lands, and an act that looked its target up in the room it
    /// had reached would refuse a name the grammar had just offered. Acts that
    /// name something in the room read it here, so what was offered is what is
    /// acted on.
    pub fn standpoint_of(&self, body: &str) -> String {
        self.read(|w| {
            w.decided_at(body)
                .or_else(|| w.actor(body).map(|a| &a.at))
                .map(|at| format!("{}/{}", at.area, at.node))
                .unwrap_or_default()
        })
    }

    /// Advance one moment: everybody on their way covers a leg.
    ///
    /// Returns how many bodies moved. Nobody walking is the common case and
    /// costs a lock and a scan of the actor map.
    pub fn tick(&self) -> usize {
        let mut state = self.state.lock().expect("world lock");
        state.world.tick()
    }

    /// Let the building have its say in every room somebody is standing in.
    ///
    /// Returns how many rooms stirred, which is nearly always none — each room
    /// waits minutes between looks, and this runs twice a second.
    pub fn stir(&self) -> usize {
        let mut state = self.state.lock().expect("world lock");
        let State {
            world, sim, rooms, ..
        } = &mut *state;
        rooms.stir(world, sim)
    }

    /// Let the tower have the time since the last beat, and tell the crew what
    /// that did to it. Returns how many lines were said.
    pub fn watch_tower(&self) -> usize {
        let mut state = self.state.lock().expect("world lock");
        let State {
            world, sim, watch, ..
        } = &mut *state;
        watch.watch(world, sim)
    }

    /// How many rooms have a building running in them. A room is fitted the
    /// first time anybody stands in it and is never asked while empty.
    pub fn rooms_running(&self) -> usize {
        self.state.lock().expect("world lock").rooms.len()
    }

    /// What every body has to be told, and nothing for the ones with nothing.
    ///
    /// One pass under one lock, because the alternative — a lock per body — is
    /// sixteen acquisitions to answer one question and leaves the world free to
    /// move between two bodies' readings of the same moment.
    pub fn sweep(&self) -> Vec<Delta> {
        let mut state = self.state.lock().expect("world lock");
        let State {
            world, attention, ..
        } = &mut *state;
        attention.sweep(world)
    }

    /// What one body has to be told, marking it as delivered.
    pub fn delta(&self, body: &str) -> Delta {
        let mut state = self.state.lock().expect("world lock");
        let State {
            world, attention, ..
        } = &mut *state;
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
    /// Where the documents a bench edits live, for every world hosted here.
    ///
    /// Held once rather than passed to each `load`, because it is one fact
    /// about this daemon — where its authored content is — and a per-call
    /// argument would be a chance for two worlds to disagree about it. `None`
    /// in a daemon started without a mind, and then every bench act refuses.
    bench_root: Mutex<Option<PathBuf>>,
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

    /// Name where the documents a bench edits live.
    ///
    /// Applies to the worlds already hosted as well as the ones still to come,
    /// so the order of startup — mind resolved before or after the first world
    /// is loaded — cannot leave a world without its documents.
    pub fn set_bench_root(&self, root: impl Into<PathBuf>) {
        let root = root.into();
        for hosted in self.worlds.lock().expect("worlds lock").values() {
            hosted.set_bench_root(root.clone());
        }
        *self.bench_root.lock().expect("bench root lock") = Some(root);
    }

    /// Keep a world that was built rather than loaded.
    pub fn keep(&self, hosted: Arc<Hosted>) -> Arc<Hosted> {
        if let Some(root) = self.bench_root.lock().expect("bench root lock").clone() {
            hosted.set_bench_root(root);
        }
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

    /// **Concurrent reshapes to the same area do not corrupt the write-back.**
    /// Eight threads each add their own room off `core` at once; the write-back
    /// mutex ([`Hosted::writeback`]) must serialise their write-backs to
    /// `a.yaml.tmp` so the file on disk is never a torn write of two of them at
    /// once, and every room they added survives to the reloaded map.
    #[test]
    fn concurrent_reshapes_to_the_same_area_do_not_corrupt_the_written_file() {
        use npc_map::schema::{Node, NodeKind};
        use std::sync::atomic::{AtomicU64, Ordering};

        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "npcd-world-race-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.yaml"),
            "id: a\nkind: level\nname: A\nsummary: a place\nnodes:\n  - id: core\n    kind: core\n    name: core\n",
        )
        .unwrap();

        let hosted = Arc::new(Hosted::load("race", &dir).expect("the base map loads"));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let hosted = Arc::clone(&hosted);
                std::thread::spawn(move || {
                    let edit = MapEdit::AddNode {
                        area: "a".into(),
                        node: Box::new(Node {
                            id: format!("room-{i}"),
                            kind: NodeKind::Social,
                            name: format!("room {i}"),
                            plural: false,
                            stand: None,
                            off: vec!["core".into()],
                            character: None,
                            parts: vec![],
                            ground: vec![],
                            habit: None,
                            sees: vec![],
                            exits: vec![],
                            visible: vec![],
                        }),
                    };
                    hosted.reshape(&edit).expect("each add lands")
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Still valid YAML — a torn write would fail to parse here — and every
        // room every thread added is in it.
        let reloaded = MapSet::load_dir(&dir).expect("the written map still parses");
        let area = reloaded.get("a").expect("the area survives");
        for i in 0..8 {
            assert!(
                area.node(&format!("room-{i}")).is_some(),
                "room-{i} did not survive concurrent write-back"
            );
        }
    }
}
