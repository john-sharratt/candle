//! The cast — every character this daemon knows, held in memory and backed by
//! the substrate's redo log.
//!
//! # Memory is the authority; the log is the durable projection
//!
//! This is not a cache in front of a database. The substrate holds no opinion
//! about a character: it stores [`NpcPayload`] records and hands them back on
//! the load walk, and that is the whole of its involvement. Every read here is
//! a map lookup. The log is read exactly once, at [`Npcs::load`].
//!
//! That is not an optimisation, it is the substrate's contract for this record
//! class. Compaction does not copy payload-keyed records forward — it
//! re-synthesises them from live state — so a registry that did not hold the
//! cast in memory would have no way to survive one.
//!
//! # One handle to the substrate, shared with the engine
//!
//! [`Npcs::load`] performs the process's **only** open of `--data/substrate/`,
//! and the engine adopts that same handle via [`Npcs::substrate`]. A second
//! `SubstratePersistence` over one directory is a second unlocked append cursor
//! and a second record index, and the compactor carries forward only what its
//! own handle walked — so characters created after the engine's open were
//! silently dropped at the next compaction. See [`Npcs::substrate`].
//!
//! # Editing supersedes; deleting is an edit
//!
//! Every write appends one record keyed by `npc_id`. The newest wins on replay
//! and every earlier copy is dead weight the compactor reclaims — an implicit
//! tombstone, with no delete record to write and none to replay. Deleting a
//! character sets `state: "tombstoned"`, which is still just another
//! superseding record: the id stays taken, because the acts it already
//! committed still name it.
//!
//! # No lock
//!
//! A read takes the published [`Cast`] ([`Npcs::view`]) and never waits. A
//! write runs on the persistence owner's thread: it builds its records from the
//! current cast, appends and fsyncs them, and only then publishes the cast with
//! them in (`docs/npcd_cast_lock_free.md`).

mod cast;

use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;
use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use candle_conversation::models::identity::LegacySpec;
use candle_conversation::persistence::identity::RecordIdentity;
use candle_conversation::persistence::record::{
    AuthoredBelief, AuthoredRelationship, AuthoredStrategy, Modulation, NpcPayload, RecordType,
};
use candle_conversation::persistence::{
    Admit, OpenProgress, SharedSubstrate, SubstratePersistence, SUBSTRATE_DIR,
};
use candle_conversation::substrate::Substrate;
use host::MODEL;
use serde_json::{json, Value};
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::task::block_in_place;

use web::auth::session::Identity;

use crate::registry::id;

pub use cast::Cast;

/// States a character may be in, mirroring §10 of the design doc. A state off
/// this list is a typo or a version skew, and either way not something to write
/// into a durable record.
pub const STATES: [&str; 5] = ["active", "idle", "asleep", "suspended", "tombstoned"];

/// The state a new character starts in. Idle, not active: a character that has
/// never been ticked is not doing anything, and saying `active` would put it in
/// every "who is thinking" count before it has thought.
const INITIAL_STATE: &str = "idle";

/// Default idle metabolism. Salience raises it; this is the resting rate.
const DEFAULT_HEARTBEAT_MS: u64 = 30_000;
const DEFAULT_SALIENCE_GATE: f32 = 0.42;

/// A name has to fit in a record and in a sentence. Long enough for a title
/// ("The Toll-keeper of the North Gate"), short enough not to be prose.
const MAX_NAME: usize = 120;
const MAX_PERSONA: usize = 8_000;
const MAX_TAGS: usize = 32;
const MAX_TAG: usize = 48;

/// How many entries each authoring-plane layer may hold.
///
/// **These are the only unbounded collections on the record, and the record is
/// rewritten whole on every edit.** Each layer is keyed by an id, so revising an
/// entry replaces it — but a *new* id appended, and nothing stopped that. A
/// character accumulating beliefs one save at a time grows a record that every
/// subsequent write copies, every compaction carries forward, and every
/// maintenance pass relocates.
///
/// The figure is operator scale, which is what this plane is for: §16's
/// authoring layers are "tens of entries, not the thousands an engine would
/// accumulate". A hundred and twenty-eight beliefs is far past any character
/// somebody has actually written and still bounds the record.
///
/// The cap is on *appending*. An entry that already exists can always be
/// revised or deleted, so reaching the limit never leaves a character stuck
/// with content it cannot edit its way out of.
const MAX_BELIEFS: usize = 128;
const MAX_RELATIONSHIPS: usize = 128;
const MAX_STRATEGIES: usize = 128;

/// What the roster's filter bar asks for.
///
/// Applied here rather than in the browser because the page fetches a listing
/// and would otherwise narrow only what it happens to hold — and because a
/// filter control that silently does nothing is worse than no control at all:
/// the reader believes they have excluded something.
#[derive(Debug, Default)]
pub struct Filter<'a> {
    /// Exact tag match. Tags are chosen from a small authored set, so a
    /// substring match here would be surprising rather than helpful.
    pub tag: Option<&'a str>,
    /// One of [`STATES`]. `any` (or absent) means no state filter.
    pub state: Option<&'a str>,
    pub world_id: Option<&'a str>,
    /// Free text over the name and the persona — case-insensitive substring.
    pub q: Option<&'a str>,
    pub include_hidden: bool,
}

impl Filter<'_> {
    fn matches(&self, n: &NpcPayload) -> bool {
        if let Some(t) = self.tag.filter(|t| !t.is_empty()) {
            if !n.tags.iter().any(|x| x == t) {
                return false;
            }
        }
        if let Some(s) = self.state.filter(|s| !s.is_empty() && *s != "any") {
            if n.state != s {
                return false;
            }
        }
        if let Some(w) = self.world_id.filter(|w| !w.is_empty()) {
            if n.world_id != w {
                return false;
            }
        }
        if let Some(q) = self.q.map(str::trim).filter(|q| !q.is_empty()) {
            // Lowercased on both sides: somebody searching "varek" means the
            // character called "Varek".
            let q = q.to_lowercase();
            let hit = n.name.to_lowercase().contains(&q)
                || n.persona_description.to_lowercase().contains(&q)
                || n.tags.iter().any(|t| t.to_lowercase().contains(&q));
            if !hit {
                return false;
            }
        }
        true
    }
}

/// One character to bring back up after a restart.
///
/// The world and the room travel with the id because waking a character and
/// putting it back in its body are one step: a character woken without its world
/// would think for a while about a place it is not standing in, and one woken
/// without its room would think from the front door about a conversation it was
/// having three floors up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Casting {
    pub npc_id: u64,
    pub world_id: String,
    /// Where its record said it was last standing, as `area/node`, from before
    /// a world kept its bodies' places as its own records
    /// (`docs/replicated_substrate.md` §9.1). Read once, for a world taken in
    /// from what was kept then; every other world places its bodies itself.
    pub at: Option<String>,
    /// The register it was last in, by mood id. `None` for one that has never
    /// said — which is a different fact from being calm, and is left absent
    /// rather than defaulted.
    pub mood: Option<String>,
}

#[derive(Debug)]
pub enum NpcError {
    /// The caller is not this character's owner. Deliberately indistinguishable
    /// from "no such character" at the API boundary — see [`Npcs::visible_to`].
    NotFound,
    Invalid(&'static str),
    Persist(String),
}

/// Which strategy an agency write lands on.
#[derive(Clone, Copy, Debug)]
pub enum StrategyTarget<'a> {
    /// The strategy with this id — created, or replaced.
    Id(&'a str),
    /// A new child of this strategy, its id minted from the record the write
    /// supersedes (`docs/npcd_cast_lock_free.md` §2.1).
    ChildOf(&'a str),
}

/// The cast, and the one handle to the substrate it is written through.
pub struct Npcs {
    /// The published cast. Replaced only on the persistence owner's thread,
    /// after the records that make it different are durable.
    current: ArcSwap<Cast>,
    /// **The daemon's only handle to the substrate**, shared with the engine
    /// rather than opened twice. See [`Npcs::substrate`].
    shared: SharedSubstrate,
}

/// A record the log would not take, as the error every write returns.
fn persist(e: impl Display) -> NpcError {
    NpcError::Persist(e.to_string())
}

/// Run `f`, which blocks on the persistence owner, without holding an async
/// runtime's worker: on a multi-threaded runtime the worker's other tasks move
/// to another thread for the length of it. A write waits for its own append
/// and fsync, behind whatever the owner is serving, and a cast written by many
/// tasks at once would otherwise park a worker per writer.
fn off_the_runtime<R>(f: impl FnOnce() -> R) -> R {
    match Handle::try_current() {
        Ok(h) if h.runtime_flavor() == RuntimeFlavor::MultiThread => block_in_place(f),
        _ => f(),
    }
}

/// One superseding record and what the caller is shown of it.
fn one(npc: NpcPayload, viewer: &str) -> (Vec<NpcPayload>, Value) {
    let view = wire(&npc, viewer);
    (vec![npc], view)
}

/// `npc` with its revision moved on — every authored edit's record.
fn bumped(mut npc: NpcPayload, now_ms: u64) -> NpcPayload {
    npc.revision += 1;
    npc.updated_ms = now_ms;
    npc
}

impl Npcs {
    /// Open the substrate at `dir/substrate/` and rebuild the cast from it.
    ///
    /// The one read of the log. Records arrive in append order, so inserting
    /// each into the map *is* last-writer-wins — no ordering pass, no revision
    /// comparison.
    ///
    /// **This is the process's one open of that directory.** The engine adopts
    /// the same handle through [`Self::substrate`] instead of opening its own —
    /// see that method for what the second handle destroyed.
    pub fn load(dir: &Path) -> Result<Self, NpcError> {
        Self::load_reporting(dir, &|_, _, _| {})
    }

    /// [`Self::load`], telling `progress` how the open goes — a store written
    /// before identities is rewritten whole before it returns, which on a large
    /// one takes minutes.
    pub fn load_reporting(dir: &Path, progress: OpenProgress<'_>) -> Result<Self, NpcError> {
        let mut substrate = Substrate::new();
        let mut by_id: BTreeMap<u64, NpcPayload> = BTreeMap::new();

        // Opened for the model this process runs, named by its spec: the pack
        // that would give its key is resolved later, beside the engine, and a
        // store that model would refuse is refused here before anything in it
        // is rewritten.
        let model = LegacySpec::of(&MODEL.spec());
        let persistence = SubstratePersistence::open_in_with_substrate_and_sink(
            dir,
            &mut substrate,
            Admit::Spec(&model),
            |entry| {
                if entry.record.header.record_type != RecordType::Npc {
                    return;
                }
                match NpcPayload::decode(&entry.record.payload) {
                    Ok(p) => {
                        by_id.insert(p.npc_id, p);
                    }
                    // A record this build cannot read is skipped, not fatal. The
                    // alternative is a daemon that will not start because one
                    // character out of a thousand was written by a newer build.
                    Err(e) => tracing::warn!("skipping undecodable NPC record: {e}"),
                }
            },
            progress,
        )
        .map_err(|e| NpcError::Persist(e.to_string()))?;

        let live = by_id.values().filter(|n| !n.is_tombstoned()).count();
        tracing::info!(
            "cast: {live} characters ({} records, {} tombstoned) from {}",
            by_id.len(),
            by_id.len() - live,
            dir.join(SUBSTRATE_DIR).display()
        );

        let read_as = by_id
            .keys()
            .filter_map(|id| Some((*id, persistence.npc_locs().identity(id)?)))
            .collect();
        let cast = Cast {
            by_id: by_id.into_iter().collect(),
            read_as,
        };
        Ok(Self {
            current: ArcSwap::from_pointee(cast),
            shared: SharedSubstrate::new(substrate, persistence),
        })
    }

    /// The cast as last published. Never waits; the snapshot stays whole for as
    /// long as it is held, whatever is written after.
    pub fn view(&self) -> Arc<Cast> {
        self.current.load_full()
    }

    /// Build records from the current cast with `edit`, append and fsync them,
    /// then publish the cast with them in — one step on the persistence owner's
    /// thread, so every write is ordered after the one before it and computed
    /// from what that one published (`docs/npcd_cast_lock_free.md` §2).
    ///
    /// Nothing is published unless it is durable, and everything durable is
    /// published: a refused edit leaves the cast exactly as it was, and when an
    /// append fails partway through a batch, the records staged before it are
    /// committed and published and the error returned — so the cast never
    /// disagrees with the log. Each published record is the one its character
    /// is read as, so a refresh does not read it again. An edit that returns no
    /// records writes nothing. `edit` runs on the owner's thread, so it must be
    /// pure over the cast it is handed (§2.3).
    fn write<R: Send>(
        &self,
        edit: impl FnOnce(&Cast) -> Result<(Vec<NpcPayload>, R), NpcError> + Send,
    ) -> Result<R, NpcError> {
        let current = &self.current;
        off_the_runtime(|| {
            self.shared.persistence.call(|p| {
                let cast = current.load_full();
                let (records, out) = edit(&cast)?;
                if records.is_empty() {
                    return Ok(out);
                }
                let mut staged = Vec::with_capacity(records.len());
                let mut refused = None;
                for npc in records {
                    match p.write_npc(&npc) {
                        Ok(identity) => staged.push((npc, identity)),
                        Err(e) => {
                            refused = Some(persist(e));
                            break;
                        }
                    }
                }
                // Flush and fsync before publishing. `write_npc` only *stages*
                // the record, and a staged record is lost on a crash — which for
                // a character somebody just created means the API said "created"
                // about something that never existed.
                p.commit().map_err(persist)?;
                if !staged.is_empty() {
                    let mut next = Cast::clone(&cast);
                    for (npc, identity) in staged {
                        next.read_as.insert(npc.npc_id, identity);
                        next.by_id.insert(npc.npc_id, npc);
                    }
                    current.store(Arc::new(next));
                }
                match refused {
                    Some(e) => Err(e),
                    None => Ok(out),
                }
            })
        })
    }

    /// Take in characters written since this registry last looked — created,
    /// edited or deleted on another member, and replicated here
    /// (`docs/replicated_substrate.md` §9.3). Each character whose current
    /// record is not the one it was read from is read again. Returns the ids
    /// whose record changed.
    ///
    /// The identities are read, the moved records read again, and the cast
    /// published in one step on the owner's thread, so no write lands between
    /// what was compared and what is published (`docs/npcd_cast_lock_free.md`
    /// §2.2).
    pub fn refresh(&self) -> Vec<u64> {
        let current = &self.current;
        off_the_runtime(|| self.refresh_on(current))
    }

    /// [`Self::refresh`]'s one step on the owner's thread.
    fn refresh_on(&self, current: &ArcSwap<Cast>) -> Vec<u64> {
        self.shared.persistence.call(|p| {
            let cast = current.load_full();
            let moved: Vec<(u64, RecordIdentity)> = p
                .npc_locs()
                .iter()
                .filter_map(|(id, _)| Some((*id, p.npc_locs().identity(id)?)))
                .filter(|(id, identity)| cast.read_as.get(id) != Some(identity))
                .collect();
            if moved.is_empty() {
                return Vec::new();
            }
            let mut next = Cast::clone(&cast);
            let mut changed = Vec::new();
            for (id, identity) in moved {
                match p.read_npc(id) {
                    Ok(Some(npc)) => {
                        if next.by_id.get(&id) != Some(&npc) {
                            next.by_id.insert(id, npc);
                            changed.push(id);
                        }
                        next.read_as.insert(id, identity);
                    }
                    Ok(None) => {}
                    Err(e) => tracing::warn!("character {id}'s record would not read: {e}"),
                }
            }
            current.store(Arc::new(next));
            changed
        })
    }

    /// Record the register a character has named for itself.
    ///
    /// Returns whether a record was written. A mood is named only by a
    /// character stopping to say so — at most once every couple of minutes, and
    /// usually the same word as last time — so the change check alone is the
    /// whole of the cost control.
    ///
    /// **Not an authoring act: no revision bump and no `updated_ms`.** Those
    /// describe authored change — what somebody edited, and what the console
    /// shows as the last time this character was worked on — and a roster
    /// sorted by "recently edited" would otherwise report whoever happened to
    /// be feeling something.
    pub fn remember_mood(&self, npc_id: u64, mood: &str) -> bool {
        let write = self.write(|cast| {
            let Some(npc) = cast.payload(npc_id) else {
                return Ok((Vec::new(), false));
            };
            if npc.mood.as_deref() == Some(mood) {
                return Ok((Vec::new(), false));
            }
            let mut npc = npc.clone();
            npc.mood = Some(mood.to_string());
            Ok((vec![npc], true))
        });
        match write {
            Ok(wrote) => wrote,
            Err(e) => {
                // Not fatal and not propagated, for the reason a place
                // checkpoint is not: soft state on a driver thread with nobody
                // to report to. The character goes on feeling it; the next
                // time it says so, this tries again.
                tracing::warn!("could not record how {npc_id} feels: {e:?}");
                false
            }
        }
    }

    /// The substrate handle, for the engine to adopt.
    ///
    /// **The engine must take this rather than open `--data` itself.** One
    /// `substrate/` admits exactly one writable handle per process: the log
    /// file is opened read-write and unlocked, so a second
    /// `SubstratePersistence` is a second append cursor *and* a second view of
    /// which character records exist. Compaction carries forward only what the
    /// compacting handle's own walk saw.
    ///
    /// That is not a hypothetical. This daemon opened one handle here and let
    /// the engine open another for conversation turns, and every character
    /// created *while the daemon ran* was dropped by the engine's next
    /// compaction — invisible to it, because it had walked the log before that
    /// character existed. Characters that predated both opens survived, so the
    /// loss looked intermittent instead of mechanical. Two Makers were lost
    /// this way before the cause was found.
    pub fn substrate(&self) -> SharedSubstrate {
        self.shared.clone()
    }

    /// Create a character owned by the caller.
    pub fn create(
        &self,
        id: &Identity,
        owner: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let name = clean_name(body.get("name"))?;
        let world_id = ref_id(body.get("world_id")).ok_or(NpcError::Invalid("world_id"))?;
        let personality_id =
            ref_id(body.get("personality_id")).ok_or(NpcError::Invalid("personality_id"))?;
        let mut npc = NpcPayload {
            // Minted against the cast the write lands on, below.
            npc_id: 0,
            owner_id: owner.to_string(),
            revision: 1,
            created_ms: now_ms,
            updated_ms: now_ms,
            state: INITIAL_STATE.to_string(),
            name,
            world_id,
            personality_id,
            hidden: body.get("hidden").and_then(Value::as_bool).unwrap_or(false),
            heartbeat_ms: DEFAULT_HEARTBEAT_MS,
            salience_gate: DEFAULT_SALIENCE_GATE,
            tags: clean_tags(body.get("tags"))?,
            persona_description: clean_persona(body.get("persona_description"))?,
            persona_origin: "authored".to_string(),
            portrait_image_id: None,
            portrait_origin: None,
            // Nowhere yet. The runtime embodies the character straight after
            // this and the first checkpoint records where it was put — writing
            // an arrival door here would be this registry guessing at a map it
            // has never read.
            at: None,
            // Nobody has asked it how it feels yet, which is a different fact
            // from it being calm.
            mood: None,
            // The authoring plane starts empty. A character nobody has written
            // beliefs for holds none — which is different from one whose
            // beliefs could not be read, and is what the console shows.
            beliefs: Vec::new(),
            relationships: Vec::new(),
            agency: Vec::new(),
            modulation: Modulation::default(),
        };
        self.write(|cast| {
            npc.npc_id = cast.mint_id(id, &npc.name, now_ms);
            Ok(one(npc, owner))
        })
    }

    /// Apply a partial edit. Absent fields are left alone; a field present and
    /// invalid is an error rather than a silent default, because a durable
    /// record written from a typo is worse than a refused write.
    pub fn patch(
        &self,
        npc_id: u64,
        owner: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        self.write(|cast| {
            let npc = cast
                .visible_to(npc_id, owner)
                .ok_or(NpcError::NotFound)?
                .clone();
            Ok(one(bumped(Self::patched(npc, body)?, now_ms), owner))
        })
    }

    /// `npc` with a partial edit's fields applied.
    fn patched(mut npc: NpcPayload, body: &Value) -> Result<NpcPayload, NpcError> {
        if let Some(v) = body.get("name") {
            npc.name = clean_name(Some(v))?;
        }
        if let Some(v) = body.get("persona_description") {
            npc.persona_description = clean_persona(Some(v))?;
        }
        if let Some(v) = body.get("tags") {
            npc.tags = clean_tags(Some(v))?;
        }
        if let Some(v) = body.get("hidden") {
            npc.hidden = v.as_bool().ok_or(NpcError::Invalid("hidden"))?;
        }
        if let Some(v) = body.get("state") {
            let s = v.as_str().ok_or(NpcError::Invalid("state"))?;
            // `tombstoned` is not settable here: deletion goes through `delete`,
            // so there is one path that removes a character and one place the
            // decision is made.
            if s == "tombstoned" || !STATES.contains(&s) {
                return Err(NpcError::Invalid("state"));
            }
            npc.state = s.to_string();
        }
        if let Some(v) = body.get("heartbeat_ms") {
            let ms = v.as_u64().ok_or(NpcError::Invalid("heartbeat_ms"))?;
            // A sub-second metabolism is a busy-loop, not a character.
            if !(1_000..=86_400_000).contains(&ms) {
                return Err(NpcError::Invalid("heartbeat_ms"));
            }
            npc.heartbeat_ms = ms;
        }
        if let Some(v) = body.get("salience_gate") {
            let g = v.as_f64().ok_or(NpcError::Invalid("salience_gate"))?;
            if !(0.0..=1.0).contains(&g) {
                return Err(NpcError::Invalid("salience_gate"));
            }
            npc.salience_gate = g as f32;
        }
        Ok(npc)
    }

    /* ── the authoring plane (§16) ──────────────────────────────────────────
     *
     * What an operator says a character believes, who they know, what they are
     * trying to do, and where their affect sits. Every one of these is a write
     * to the character's record and supersedes it, exactly as an edit to their
     * name does — one write path, one supersession rule, one place a change is
     * durable.
     *
     * Each is an upsert keyed by the caller's own id, so a `PUT` is idempotent
     * and the console can save a row without knowing whether it exists. */

    /// State an operator's belief. Replaces the one with that id, or adds it.
    pub fn put_belief(
        &self,
        npc_id: u64,
        owner: &str,
        belief_id: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        self.write(|cast| {
            let mut npc = cast.owned(npc_id, owner)?;
            let existing = npc.beliefs.iter().position(|b| b.belief_id == belief_id);
            let mut belief = existing
                .map(|i| npc.beliefs[i].clone())
                .unwrap_or(AuthoredBelief {
                    belief_id: belief_id.to_string(),
                    statement: String::new(),
                    confidence: 0.5,
                    threshold: 0.5,
                });
            if let Some(v) = body.get("statement") {
                belief.statement = clean_line(v, "statement")?;
            }
            belief.confidence = unit(body, "confidence", belief.confidence)?;
            belief.threshold = unit(body, "threshold", belief.threshold)?;
            // A belief with nothing said in it is not a belief.
            if belief.statement.trim().is_empty() {
                return Err(NpcError::Invalid("statement"));
            }
            match existing {
                Some(i) => npc.beliefs[i] = belief,
                None => {
                    if npc.beliefs.len() >= MAX_BELIEFS {
                        return Err(NpcError::Invalid("beliefs"));
                    }
                    npc.beliefs.push(belief);
                }
            }
            Ok(one(bumped(npc, now_ms), owner))
        })
    }

    pub fn delete_belief(
        &self,
        npc_id: u64,
        owner: &str,
        belief_id: &str,
        now_ms: u64,
    ) -> Result<bool, NpcError> {
        self.write(|cast| {
            let mut npc = cast.owned(npc_id, owner)?;
            let before = npc.beliefs.len();
            npc.beliefs.retain(|b| b.belief_id != belief_id);
            if npc.beliefs.len() == before {
                return Ok((Vec::new(), false));
            }
            Ok((vec![bumped(npc, now_ms)], true))
        })
    }

    /// Set how this character holds somebody.
    pub fn put_relationship(
        &self,
        npc_id: u64,
        owner: &str,
        entity_id: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        self.write(|cast| {
            let mut npc = cast.owned(npc_id, owner)?;
            let existing = npc
                .relationships
                .iter()
                .position(|r| r.entity_id == entity_id);
            let mut rel =
                existing
                    .map(|i| npc.relationships[i].clone())
                    .unwrap_or(AuthoredRelationship {
                        entity_id: entity_id.to_string(),
                        display: entity_id.to_string(),
                        trust: 0.0,
                        affect: 0.0,
                        familiarity: 0.0,
                        notes: String::new(),
                    });
            if let Some(v) = body.get("display") {
                rel.display = clean_line(v, "display")?;
            }
            if let Some(v) = body.get("notes") {
                rel.notes = clean_line(v, "notes")?;
            }
            // Trust and affect run −1..1; familiarity only accumulates.
            rel.trust = signed(body, "trust", rel.trust)?;
            rel.affect = signed(body, "affect", rel.affect)?;
            rel.familiarity = unit(body, "familiarity", rel.familiarity)?;
            match existing {
                Some(i) => npc.relationships[i] = rel,
                None => {
                    if npc.relationships.len() >= MAX_RELATIONSHIPS {
                        return Err(NpcError::Invalid("relationships"));
                    }
                    npc.relationships.push(rel);
                }
            }
            Ok(one(bumped(npc, now_ms), owner))
        })
    }

    /// State a strategy, optionally under another.
    pub fn put_strategy(
        &self,
        npc_id: u64,
        owner: &str,
        strategy_id: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        self.write(|cast| {
            let npc = Self::strategized(cast.owned(npc_id, owner)?, strategy_id, body)?;
            Ok(one(bumped(npc, now_ms), owner))
        })
    }

    /// `npc` with the strategy `strategy_id` created or replaced from `body`.
    fn strategized(
        mut npc: NpcPayload,
        strategy_id: &str,
        body: &Value,
    ) -> Result<NpcPayload, NpcError> {
        let existing = npc.agency.iter().position(|a| a.strategy_id == strategy_id);
        let mut st = existing
            .map(|i| npc.agency[i].clone())
            .unwrap_or(AuthoredStrategy {
                strategy_id: strategy_id.to_string(),
                statement: String::new(),
                parent_id: None,
                state: "active".to_string(),
            });
        if let Some(v) = body.get("statement") {
            st.statement = clean_line(v, "statement")?;
        }
        if let Some(v) = body.get("state") {
            let s = v.as_str().ok_or(NpcError::Invalid("state"))?;
            if !["active", "finished", "abandoned"].contains(&s) {
                return Err(NpcError::Invalid("state"));
            }
            st.state = s.to_string();
        }
        if let Some(v) = body.get("parent_id") {
            st.parent_id = match v {
                Value::Null => None,
                v => {
                    let p = v.as_str().ok_or(NpcError::Invalid("parent_id"))?;
                    // A strategy cannot be its own parent, and a parent has to
                    // exist — a tree with a dangling edge renders as a root,
                    // which silently loses the child.
                    if p == strategy_id || !npc.agency.iter().any(|a| a.strategy_id == p) {
                        return Err(NpcError::Invalid("parent_id"));
                    }
                    Some(p.to_string())
                }
            };
        }
        if st.statement.trim().is_empty() {
            return Err(NpcError::Invalid("statement"));
        }
        match existing {
            Some(i) => npc.agency[i] = st,
            None => {
                if npc.agency.len() >= MAX_STRATEGIES {
                    return Err(NpcError::Invalid("agency"));
                }
                npc.agency.push(st);
            }
        }
        Ok(npc)
    }

    /// State a strategy on a character's **own** agency, owner-blind, and say
    /// which strategy id it landed on.
    ///
    /// The effector device acts as the character's own body and carries no
    /// operator owner to authorise the write, so this resolves the owner from
    /// the record itself ([`Cast::payload`]) and then performs the ordinary
    /// [`Self::put_strategy`] write. A [`StrategyTarget::ChildOf`] mints the
    /// child's id from the record this write supersedes, in the same step that
    /// writes it. That is the whole of the effector's *projected
    /// store* (effector design §9.2): a `plan_*`/`orders_*` call the character
    /// makes lands in the same `agency` layer the operator API writes, supersedes
    /// the record, and projects into the next turn's prompt through the `agency`
    /// collection and [`crate::engine::persona`].
    ///
    /// **Agency only.** This writes a character's own goals; the belief-write
    /// invariant is untouched — a belief is never a tool's silent edit (§9.2),
    /// and there is deliberately no `put_belief_self`.
    pub fn put_strategy_self(
        &self,
        npc_id: u64,
        target: StrategyTarget<'_>,
        body: &Value,
        now_ms: u64,
    ) -> Result<(Value, String), NpcError> {
        self.write(|cast| {
            // A tombstoned or unknown character reads as absent, the same as
            // every other read path.
            let npc = cast.payload(npc_id).ok_or(NpcError::NotFound)?.clone();
            let strategy_id = match target {
                StrategyTarget::Id(id) => id.to_string(),
                StrategyTarget::ChildOf(parent) => cast::child_id(&npc, parent),
            };
            let owner = npc.owner_id.clone();
            let npc = bumped(Self::strategized(npc, &strategy_id, body)?, now_ms);
            let (records, view) = one(npc, &owner);
            Ok((records, (view, strategy_id)))
        })
    }

    /// Attach an uploaded portrait.
    ///
    /// Deliberately **not** part of [`Self::patch`]. That takes the fields a
    /// person edits in a form, and an image id is not one of them: it is minted
    /// by the daemon from the bytes it just stored. Accepting one through
    /// `PATCH /v1/npc/:id` would let a caller point their character at an id
    /// they did not upload — every id in the store is a valid one, so there
    /// would be nothing to reject.
    pub fn set_portrait(
        &self,
        npc_id: u64,
        owner: &str,
        image_id: String,
        origin: &str,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        self.write(|cast| {
            let mut npc = cast.owned(npc_id, owner)?;
            npc.portrait_image_id = Some(image_id);
            npc.portrait_origin = Some(origin.to_string());
            Ok(one(bumped(npc, now_ms), owner))
        })
    }

    /// Every character whose portrait names an image `renames` has a new name
    /// for, written again naming it. Returns how many were. One fsync covers
    /// them all.
    pub fn repoint_portraits(
        &self,
        renames: &HashMap<String, String>,
        now_ms: u64,
    ) -> Result<usize, NpcError> {
        self.write(|cast| {
            let moved: Vec<NpcPayload> = cast
                .by_id
                .values()
                .filter(|n| {
                    n.portrait_image_id
                        .as_ref()
                        .is_some_and(|id| renames.contains_key(id))
                })
                .cloned()
                .map(|mut npc| {
                    let id = npc.portrait_image_id.take().unwrap_or_default();
                    npc.portrait_image_id = renames.get(&id).cloned();
                    bumped(npc, now_ms)
                })
                .collect();
            let n = moved.len();
            Ok((moved, n))
        })
    }

    /// Set the affect dials.
    pub fn put_modulation(
        &self,
        npc_id: u64,
        owner: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        self.write(|cast| {
            let mut npc = cast.owned(npc_id, owner)?;
            npc.modulation.affect = signed(body, "affect", npc.modulation.affect)?;
            npc.modulation.threat = unit(body, "threat", npc.modulation.threat)?;
            npc.modulation.curiosity = unit(body, "curiosity", npc.modulation.curiosity)?;
            Ok(one(bumped(npc, now_ms), owner))
        })
    }

    /// Delete a character: one superseding record with `state: "tombstoned"`.
    /// The record stays, so the id stays taken.
    pub fn delete(&self, npc_id: u64, owner: &str, now_ms: u64) -> Result<(), NpcError> {
        self.write(|cast| {
            let mut npc = cast
                .visible_to(npc_id, owner)
                .ok_or(NpcError::NotFound)?
                .clone();
            npc.state = "tombstoned".to_string();
            Ok((vec![bumped(npc, now_ms)], ()))
        })
    }

    /// Hand character `npc_id` to the account `to` (its public id) — an admin's
    /// act, and the one write that crosses ownership: every other write starts
    /// from [`Cast::visible_to`]. Ownership is what every read and write of the
    /// character is checked against, so this is how a character made under one
    /// account comes to be another's to watch and author.
    pub fn reassign(&self, npc_id: u64, to: &str, now_ms: u64) -> Result<Value, NpcError> {
        self.write(|cast| {
            let mut npc = cast.payload(npc_id).ok_or(NpcError::NotFound)?.clone();
            if npc.owner_id == to {
                return Ok((Vec::new(), wire(&npc, to)));
            }
            npc.owner_id = to.to_string();
            Ok(one(bumped(npc, now_ms), to))
        })
    }

    /// Restore backed-up character records into a fresh substrate.
    ///
    /// Each payload is written verbatim — its id, owner, revision and last place
    /// are preserved — so a restored cast is the same characters, only with the
    /// conversation/belief/dream history the wipe cleared. One fsync covers the
    /// batch. Returns how many were written.
    pub fn import(&self, payloads: Vec<NpcPayload>) -> Result<usize, NpcError> {
        let n = payloads.len();
        self.write(|_| Ok((payloads, n)))
    }
}

/// Write a character id in base-36 (`0-9a-z`), lowercase.
///
/// The same reason and the same alphabet as a world instance id's ordinal
/// (`npc_map::instance::base36`): this id is bandied around every page that
/// lists a character and every route that names one, so it is spent in a
/// human's read and a console's URL far more than the twenty digits a minted
/// hash needs in decimal — a real id is `6817662845163923144`, base-36 is
/// `1gtpgs1gobgkw`. It is still carried as a JSON **string**, never a bare
/// number: a `u64` this wide already sits well past 2^53, so a bare JSON
/// number would round in any JavaScript reader before the encoding even
/// mattered — see [`wire`] and [`npc_id_of_wire`].
pub fn npc_id_wire(mut n: u64) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".to_string();
    }
    let mut buf = Vec::new();
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    buf.reverse();
    // Every byte is an ASCII digit from `DIGITS`, so this is always valid UTF-8.
    String::from_utf8(buf).expect("base-36 digits are ASCII")
}

/// The inverse of [`npc_id_wire`]: read a character id back off the wire.
///
/// `None` for anything that is not a valid base-36 numeral — an empty string,
/// a character outside `0-9a-z` (read case-insensitively, so a hand-typed
/// uppercase id still resolves), or digits that overflow `u64`. Every route
/// that used to treat a failed decimal parse as "no such character" treats a
/// `None` here the same way.
///
/// **There is no version tag distinguishing this from the decimal `to_string`
/// this format replaced**, because every decimal numeral is also a valid
/// base-36 one — a caller holding a pre-migration decimal id decodes it here
/// as a base-36 number instead of getting a clean parse failure. No shim
/// carries that old format forward (`CLAUDE.md`: no backward compatibility in
/// this pre-publication codebase), so the residual risk is bounded by how the
/// id itself is minted rather than guarded against here: character ids are
/// drawn across the full `u64` range, so a real id's base-36 form is 13
/// characters wide and the odds every one of them lands on a digit — the only
/// way it would still read as a plausible decimal number — are
/// `(10/36)^13`, on the order of one in ten million. A short, hand-authored
/// test id (`npc_id: 5`) does not carry that guarantee, so treat any
/// deliberately small id as decimal-shaped and pick a large one instead.
pub fn npc_id_of_wire(s: &str) -> Option<u64> {
    u64::from_str_radix(s, 36).ok()
}

/// A `#[serde(serialize_with = "npcs::npc_id_wire_serde")]` for a struct field
/// that is a character id — [`npc_id_wire`] in the shape `serde` wants of a
/// field serializer, for a wire type derived rather than built by hand.
pub fn npc_id_wire_serde<S: serde::Serializer>(id: &u64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&npc_id_wire(*id))
}

/// One character as §10 defines it on the wire.
///
/// Ids cross as base-36 **strings** ([`npc_id_wire`]): they are `u64`, and a
/// JSON number above 2^53 is silently rounded by every browser that parses
/// it, so the string is mandatory with or without the encoding underneath it.
fn wire(n: &NpcPayload, caller: &str) -> Value {
    json!({
        "npc_id": npc_id_wire(n.npc_id),
        "name": n.name,
        "world_id": n.world_id,
        "personality_id": n.personality_id,
        "state": n.state,
        "tick": {
            "heartbeat_ms": n.heartbeat_ms,
            // Rounded on the way out. `json!` widens the stored `f32` to `f64`,
            // and 0.42f32 widens to 0.41999998688697815 — a gate somebody typed
            // as "0.42" coming back as noise, which looks like the daemon
            // changed it.
            "salience_gate": (f64::from(n.salience_gate) * 1_000.0).round() / 1_000.0,
            // Live values, and this daemon runs no engine, so they are absent
            // rather than zero — a character with `pending_events: 0` reads as
            // measured and idle, which is a claim nothing here can make.
            "last_tick_ms": Value::Null,
            "pending_events": Value::Null,
        },
        // Same reason: the monitor is an engine measurement.
        "monitor": Value::Null,
        // The register it last said it was in, by mood id.
        //
        // **Null, never a default register.** A character nobody has asked yet
        // and one that is calm are different facts, and only one of them is a
        // measurement — the same rule the live tick figures above follow.
        "mood": n.mood,
        "modulation": {
            "affect": round3(n.modulation.affect),
            "threat": round3(n.modulation.threat),
            "curiosity": round3(n.modulation.curiosity),
        },
        "owner_id": n.owner_id,
        "access": if n.owner_id == caller { "owner" } else { "viewer" },
        "hidden": n.hidden,
        "tags": n.tags,
        "portrait": n.portrait_image_id.as_ref().map(|id| json!({
            "image_id": id,
            "origin": n.portrait_origin.clone().unwrap_or_else(|| "generated".to_string()),
        })),
        "persona": { "description": n.persona_description, "origin": n.persona_origin },
        "created_ms": n.created_ms,
        "updated_ms": n.updated_ms,
        "revision": n.revision,
    })
}

/// An `f32` on the wire, without the widening noise.
///
/// `json!` widens to `f64`, and `0.42f32` widens to `0.41999998688697815` — a
/// dial somebody typed as 0.42 coming back as noise, which reads as the daemon
/// having changed it.
fn round3(v: f32) -> f64 {
    (f64::from(v) * 1_000.0).round() / 1_000.0
}

/// The authoring plane, as the console reads it.
///
/// The engine's measurements are **absent**, not zero. A belief has no
/// `disconfirmation` until something has weighed evidence against it, and a
/// strategy has no `salience` until something has scored it — reporting either
/// as 0 would be a measurement this daemon has not made.
pub fn beliefs_wire(n: &NpcPayload) -> Value {
    json!({ "beliefs": n.beliefs.iter().map(|b| json!({
        "belief_id": b.belief_id,
        "statement": b.statement,
        "confidence": round3(b.confidence),
        "threshold": round3(b.threshold),
        "origin": "authored",
        "disconfirmation": Value::Null,
        "under_pressure": Value::Null,
        "history": Value::Null,
    })).collect::<Vec<_>>() })
}

pub fn relationships_wire(n: &NpcPayload) -> Value {
    json!({ "relationships": n.relationships.iter().map(|r| json!({
        "entity_id": r.entity_id,
        "display": r.display,
        "trust": round3(r.trust),
        "affect": round3(r.affect),
        "familiarity": round3(r.familiarity),
        "notes": r.notes,
        "origin": "authored",
    })).collect::<Vec<_>>() })
}

pub fn agency_wire(n: &NpcPayload) -> Value {
    json!({ "agency": n.agency.iter().map(|a| json!({
        "strategy_id": a.strategy_id,
        "statement": a.statement,
        "parent_id": a.parent_id,
        "state": a.state,
        "origin": "authored",
        // Scored by the engine against what is happening; nothing here has.
        "salience": Value::Null,
        "progress_notes": Value::Null,
    })).collect::<Vec<_>>() })
}

pub fn modulation_wire(n: &NpcPayload) -> Value {
    json!({
        "affect": round3(n.modulation.affect),
        "threat": round3(n.modulation.threat),
        "curiosity": round3(n.modulation.curiosity),
    })
}

/// A field that must be 0..1, or the value it already had.
///
/// Absent means unchanged, never zero: a `PUT` that sets one dial must not
/// silently reset the other two, which is the whole reason these take the
/// current value rather than a default.
fn unit(body: &Value, key: &str, current: f32) -> Result<f32, NpcError> {
    bounded(body, key, current, 0.0, 1.0)
}

/// A field that must be −1..1, on the same terms.
fn signed(body: &Value, key: &str, current: f32) -> Result<f32, NpcError> {
    bounded(body, key, current, -1.0, 1.0)
}

fn bounded(body: &Value, key: &str, current: f32, lo: f64, hi: f64) -> Result<f32, NpcError> {
    let Some(v) = body.get(key) else {
        return Ok(current);
    };
    let n = v.as_f64().ok_or(NpcError::Invalid(leak(key)))?;
    if !n.is_finite() || !(lo..=hi).contains(&n) {
        return Err(NpcError::Invalid(leak(key)));
    }
    Ok(n as f32)
}

/// The longest one line of authored text may be. Generous for a statement or a
/// note, short of a way to fill a disk one save at a time.
///
/// This was `MAX_PROMPT_CHARS`, belonging to the simulated environment's system
/// prompt and borrowed by the helper below. That feature is gone; the bound is
/// still needed here, so it is named for what it actually guards.
const MAX_LINE_CHARS: usize = 8_000;

/// One line of authored text, trimmed and bounded.
fn clean_line(v: &Value, key: &'static str) -> Result<String, NpcError> {
    let s = v.as_str().ok_or(NpcError::Invalid(key))?.trim();
    if s.chars().count() > MAX_LINE_CHARS {
        return Err(NpcError::Invalid(key));
    }
    Ok(s.to_string())
}

/// `NpcError::Invalid` names the field in a `&'static str`, and these come from
/// a runtime key. The set is closed and small, so it is matched rather than
/// leaked — a `Box::leak` here would grow the binary's heap by one string per
/// bad request, for ever.
fn leak(key: &str) -> &'static str {
    match key {
        "confidence" => "confidence",
        "threshold" => "threshold",
        "trust" => "trust",
        "affect" => "affect",
        "familiarity" => "familiarity",
        "threat" => "threat",
        "curiosity" => "curiosity",
        _ => "value",
    }
}

/// Ids arrive as decimal strings; accept a number too, since a hand-written
/// request is the common way to hit this.
/// A reference to an authored world or personality: the slug that is its file
/// name.
///
/// Validated with the registry's own gate rather than a looser rule of its own.
/// The two must agree, because the point of storing the slug is that it always
/// resolves to a file — a reference this accepted and `registry::id::check`
/// would refuse is a durable record naming something that can never exist.
///
/// Whether the file is actually *there* is a different question, answered where
/// the registries are (`api::create_npc`); this is the shape check.
fn ref_id(v: Option<&Value>) -> Option<String> {
    let s = v?.as_str()?;
    id::check(s).ok().map(|()| s.to_string())
}

fn clean_name(v: Option<&Value>) -> Result<String, NpcError> {
    let s = v
        .and_then(Value::as_str)
        .ok_or(NpcError::Invalid("name"))?
        .trim();
    if s.is_empty() || s.chars().count() > MAX_NAME {
        return Err(NpcError::Invalid("name"));
    }
    // Control characters would survive into a record and out onto a page.
    if s.chars().any(|c| c.is_control()) {
        return Err(NpcError::Invalid("name"));
    }
    Ok(s.to_string())
}

fn clean_persona(v: Option<&Value>) -> Result<String, NpcError> {
    let s = v.and_then(Value::as_str).unwrap_or("").trim();
    if s.chars().count() > MAX_PERSONA {
        return Err(NpcError::Invalid("persona_description"));
    }
    Ok(s.to_string())
}

fn clean_tags(v: Option<&Value>) -> Result<Vec<String>, NpcError> {
    let Some(arr) = v.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    if arr.len() > MAX_TAGS {
        return Err(NpcError::Invalid("tags"));
    }
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let s = t.as_str().ok_or(NpcError::Invalid("tags"))?.trim();
        if s.is_empty() || s.chars().count() > MAX_TAG || s.chars().any(|c| c.is_control()) {
            return Err(NpcError::Invalid("tags"));
        }
        // Duplicates are dropped rather than refused: a repeated tag is a
        // slip, not an error worth losing an edit over.
        if !out.contains(&s.to_string()) {
            out.push(s.to_string());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::engine::bodies::NpcList;

    use super::*;

    /// Two accounts, in the shape `accounts::with_public_id` mints: a `u_`
    /// prefix over the account file's key. Not a number — see `NpcPayload`.
    const ME: &str = "u_1a2b3c4d";
    const OTHER: &str = "u_99887766";

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-npcs-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn ident(sub: &str) -> Identity {
        Identity {
            provider: "google".to_string(),
            sub: sub.to_string(),
            email: String::new(),
            name: String::new(),
            picture: String::new(),
            exp: 0,
        }
    }

    fn body(name: &str) -> Value {
        json!({ "name": name, "world_id": "battle-cities", "personality_id": "commander" })
    }

    /// A wire record's `npc_id`, decoded back to the `u64` the tests key on.
    fn nid(v: &Value) -> u64 {
        npc_id_of_wire(v["npc_id"].as_str().unwrap()).unwrap()
    }

    /// The property the whole module exists for: a character outlives the
    /// process. Written by one registry, read back by another over the same
    /// directory, with no engine in between.
    #[test]
    fn a_character_survives_a_restart() {
        let dir = tmp();
        let created = {
            let n = Npcs::load(&dir).unwrap();
            n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap()
        };
        let npc_id: u64 = nid(&created);

        // A second registry over the same log — a restart, in effect.
        let reopened = Npcs::load(&dir).unwrap().view();
        let back = reopened.get(npc_id, ME).unwrap();
        assert_eq!(back["name"], "Varek");
        assert_eq!(back["owner_id"], ME);
        assert_eq!(back["revision"], 1);
        assert_eq!(reopened.list(ME, &Filter::default()).len(), 1);
    }

    /// **A character handed to another account is theirs, and stays theirs.**
    /// The new owner reads and lists it, the old one no longer does, the owners
    /// listing says so, and a restart keeps it.
    #[test]
    fn a_reassigned_character_belongs_to_its_new_owner() {
        let dir = tmp();
        let npc_id = {
            let n = Npcs::load(&dir).unwrap();
            let npc_id = nid(&n.create(&ident("u1"), OTHER, &body("Ilsa"), 1_000).unwrap());
            let handed = n.reassign(npc_id, ME, 2_000).unwrap();
            assert_eq!(handed["owner_id"], ME);
            assert_eq!(handed["access"], "owner");
            assert_eq!(handed["revision"], 2);
            assert_eq!(n.view().owners()[0]["owner_id"], ME);
            assert_eq!(n.view().owners()[0]["name"], "Ilsa");
            assert_eq!(
                n.reassign(npc_id, ME, 3_000).unwrap()["revision"],
                2,
                "handing it to its owner writes nothing"
            );
            assert!(matches!(
                n.reassign(npc_id + 1, ME, 3_000),
                Err(NpcError::NotFound)
            ));
            npc_id
        };
        let reopened = Npcs::load(&dir).unwrap().view();
        assert_eq!(reopened.get(npc_id, ME).unwrap()["name"], "Ilsa");
        assert!(reopened.get(npc_id, OTHER).is_err());
        assert_eq!(reopened.list(OTHER, &Filter::default()).len(), 0);
    }

    /// **A character created while the daemon runs survives the engine's
    /// maintenance.**
    ///
    /// The regression guard for the bug that lost Makers overnight. The engine
    /// used to open `--data/substrate/` for itself, so it held a second
    /// writable handle whose record index had been built before the character
    /// existed — and its next compaction carried forward only what that index
    /// knew, dropping every character created since. Characters present at boot
    /// survived, so it read as an intermittent fault rather than a mechanical
    /// one.
    ///
    /// Here the engine takes the registry's handle, which is the whole fix: the
    /// compaction runs against the same view the character was written into.
    #[test]
    fn a_character_created_at_runtime_survives_an_engine_compaction() {
        let dir = tmp();
        let npc_id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            // Whatever the engine does to this log, it does through this handle
            // — the one the registry is writing through.
            let engine = n.substrate();

            let created = n.create(&ident("u1"), ME, &body("Wyneth"), 1_000).unwrap();

            // The engine's routine compaction, on the handle it was handed.
            let substrate = &engine.substrate;
            engine
                .persistence
                .call(|p| p.compact_shared(substrate, None))
                .unwrap();
            nid(&created)
        };

        // A restart.
        let reopened = Npcs::load(&dir).unwrap().view();
        assert_eq!(
            reopened.get(npc_id, ME).unwrap()["name"],
            "Wyneth",
            "a character created at runtime must outlive a compaction"
        );
    }

    /// **Each authoring layer is bounded, and reaching the bound still leaves
    /// the character editable.**
    ///
    /// These are the only collections on the record that grow, and the record is
    /// rewritten whole on every write, carried forward by every compaction and
    /// relocated by every maintenance pass. An uncapped layer is a character
    /// that gets more expensive to keep every time somebody saves.
    ///
    /// The cap is on appending. Revising or deleting an existing entry works at
    /// the limit — a full character that could not be edited down would be a
    /// worse trap than the growth.
    #[test]
    fn an_authoring_layer_stops_growing_but_stays_editable() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());

        for i in 0..MAX_BELIEFS {
            let b = json!({ "statement": format!("belief {i}") });
            n.put_belief(id, ME, &format!("b{i}"), &b, 2_000)
                .unwrap_or_else(|e| panic!("belief {i} refused: {e:?}"));
        }
        assert!(
            matches!(
                n.put_belief(id, ME, "one-too-many", &json!({ "statement": "no" }), 3_000),
                Err(NpcError::Invalid("beliefs"))
            ),
            "the layer grew past its cap"
        );

        // At the cap, an existing belief still revises...
        n.put_belief(id, ME, "b0", &json!({ "statement": "revised" }), 4_000)
            .expect("an existing entry must stay editable at the cap");
        let view = n.view();
        let held = view.payload(id).expect("still there");
        assert_eq!(
            held.beliefs.len(),
            MAX_BELIEFS,
            "revising must not have appended"
        );
        assert_eq!(
            held.beliefs
                .iter()
                .find(|b| b.belief_id == "b0")
                .map(|b| b.statement.as_str()),
            Some("revised")
        );
        // ...and deleting one makes room again.
        assert!(n.delete_belief(id, ME, "b0", 5_000).unwrap());
        n.put_belief(id, ME, "fresh", &json!({ "statement": "yes" }), 6_000)
            .expect("a deletion frees a slot");
    }

    /// **`put_strategy_self` writes the character's own agency owner-blind.**
    ///
    /// The effector device is the character's own body and carries no operator
    /// owner, so this resolves the owner from the record and lands the write in
    /// the same `agency` layer the owner-scoped `put_strategy` writes — the
    /// bridge the projected store depends on (effector design §9.2).
    #[test]
    fn put_strategy_self_resolves_the_owner_and_writes_the_agency() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());

        // No owner passed — the caller does not have one.
        let (_, landed) = n
            .put_strategy_self(
                id,
                StrategyTarget::Id("hold-the-gate"),
                &json!({ "statement": "Hold the toll gate.", "state": "active" }),
                2_000,
            )
            .expect("the owner-blind write lands");
        assert_eq!(landed, "hold-the-gate");

        let view = n.view();
        let held = view.payload(id).expect("still there");
        let strat = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "hold-the-gate")
            .expect("the strategy is in the agency layer");
        assert_eq!(strat.statement, "Hold the toll gate.");
        assert_eq!(strat.state, "active");

        // A character nobody has is absent, the same as every other read path —
        // never a panic, never a write under a mislaid owner.
        assert!(matches!(
            n.put_strategy_self(
                id + 1,
                StrategyTarget::Id("x"),
                &json!({ "statement": "no" }),
                3_000
            ),
            Err(NpcError::NotFound)
        ));
    }

    /// **How a character feels survives a restart.**
    ///
    /// A mood is what a character *is* between one thought and the next, and it
    /// otherwise lives only in the window — which a restart empties. Without
    /// this, one that had spent the afternoon getting angrier came back in no
    /// register at all while its own transcript said otherwise, which is the
    /// same wrong as returning it to the arrival door.
    #[test]
    fn how_a_character_feels_survives_a_restart() {
        let dir = tmp();
        let id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());
            // Absent, not calm. Nobody has asked it yet, and those are
            // different facts about a character.
            assert_eq!(n.view().mood_of(id), None);

            assert!(n.remember_mood(id, "wary"));
            assert_eq!(n.view().mood_of(id), Some("wary"));
            // Feeling the same thing again costs nothing, however often it is
            // said — which is the whole of the write gate, because a register
            // is named at most once a pause and is usually last time's word.
            assert!(!n.remember_mood(id, "wary"));
            assert!(n.remember_mood(id, "cornered"));
            id
        };

        let back = Npcs::load(&dir).unwrap().view();
        assert_eq!(
            back.mood_of(id),
            Some("cornered"),
            "the register did not survive the restart"
        );
        // And it rides along with the cast, which is what the loader reads to
        // put the character back into it.
        let casting = back.cast().into_iter().find(|c| c.npc_id == id).unwrap();
        assert_eq!(casting.mood.as_deref(), Some("cornered"));
    }

    /// Feeling something is not an authoring act: a revision bump and an
    /// `updated_ms` describe what somebody *edited*, and a roster that sorted by
    /// "recently edited" would report whoever happened to be feeling something.
    #[test]
    fn feeling_something_is_not_an_edit() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let made = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = nid(&made);
        let before = n.view().payload(id).unwrap().clone();

        assert!(n.remember_mood(id, "bitter"));
        let view = n.view();
        let after = view.payload(id).unwrap();
        assert_eq!(
            after.revision, before.revision,
            "a mood bumped the revision"
        );
        assert_eq!(
            after.updated_ms, before.updated_ms,
            "a mood counted as an edit"
        );
    }

    /// An edit supersedes rather than accumulating: the reopened registry sees
    /// exactly one character, at the newest revision.
    #[test]
    fn an_edit_supersedes_the_previous_record() {
        let dir = tmp();
        let npc_id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
            let id: u64 = nid(&c);
            n.patch(id, ME, &json!({ "name": "Varek the Elder" }), 2_000)
                .unwrap();
            n.patch(id, ME, &json!({ "state": "active" }), 3_000)
                .unwrap();
            id
        };

        let reopened = Npcs::load(&dir).unwrap().view();
        let list = reopened.list(ME, &Filter::default());
        assert_eq!(list.len(), 1, "three records, one character");
        assert_eq!(list[0]["name"], "Varek the Elder");
        assert_eq!(list[0]["state"], "active");
        assert_eq!(list[0]["revision"], 3);
        assert_eq!(reopened.get(npc_id, ME).unwrap()["updated_ms"], 3_000);
    }

    /// Deleting writes a tombstoned record. It disappears from every read path
    /// but the id stays taken — the acts it committed still name it.
    #[test]
    fn deleting_hides_the_character_but_keeps_its_id() {
        let dir = tmp();
        let npc_id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
            let id: u64 = nid(&c);
            n.delete(id, ME, 2_000).unwrap();
            assert!(n.view().list(ME, &Filter::default()).is_empty());
            assert!(matches!(n.view().get(id, ME), Err(NpcError::NotFound)));
            id
        };

        let reopened = Npcs::load(&dir).unwrap().view();
        assert!(reopened.list(ME, &Filter::default()).is_empty());
        assert!(
            reopened.by_id.contains_key(&npc_id),
            "the record survives so the id cannot be reused"
        );
    }

    /// Ownership is authorization, and a stranger's character reads as absent
    /// rather than forbidden — a 403 would confirm the id exists.
    #[test]
    fn another_accounts_character_is_invisible_not_forbidden() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = nid(&c);

        assert!(matches!(n.view().get(id, OTHER), Err(NpcError::NotFound)));
        assert!(n.view().list(OTHER, &Filter::default()).is_empty());
        assert!(matches!(
            n.patch(id, OTHER, &json!({ "name": "Stolen" }), 2_000),
            Err(NpcError::NotFound)
        ));
        assert!(matches!(
            n.delete(id, OTHER, 2_000),
            Err(NpcError::NotFound)
        ));
        // And the original is untouched by the attempts.
        assert_eq!(n.view().get(id, ME).unwrap()["name"], "Varek");
    }

    /// Hidden characters stay out of the default listing (§8.3).
    #[test]
    fn hidden_characters_are_opt_in() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        n.create(&ident("u1"), ME, &body("Seen"), 1_000).unwrap();
        let mut hidden_body = body("Unseen");
        hidden_body["hidden"] = json!(true);
        n.create(&ident("u1"), ME, &hidden_body, 1_001).unwrap();

        let n = n.view();
        assert_eq!(n.list(ME, &Filter::default()).len(), 1);
        let all = n.list(
            ME,
            &Filter {
                include_hidden: true,
                ..Default::default()
            },
        );
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn filters_narrow_and_do_not_silently_pass_everything() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let mut a = body("Varek");
        a["tags"] = json!(["north", "campaign-2"]);
        n.create(&ident("u1"), ME, &a, 1_000).unwrap();
        let mut b = json!({ "name": "Ilse", "world_id": "earth", "personality_id": "commander" });
        b["tags"] = json!(["market"]);
        n.create(&ident("u1"), ME, &b, 1_001).unwrap();
        let n = n.view();

        let by_tag = |t| {
            n.list(
                ME,
                &Filter {
                    tag: Some(t),
                    ..Default::default()
                },
            )
        };
        assert_eq!(by_tag("north").len(), 1);
        assert_eq!(by_tag("market").len(), 1);
        assert_eq!(by_tag("nowhere").len(), 0);

        // Free text reaches the name and the tags, case-insensitively.
        let q = |s| {
            n.list(
                ME,
                &Filter {
                    q: Some(s),
                    ..Default::default()
                },
            )
        };
        assert_eq!(q("VAREK").len(), 1);
        assert_eq!(q("mar").len(), 1);
        assert_eq!(q("zzz").len(), 0);

        let w = |id| {
            n.list(
                ME,
                &Filter {
                    world_id: Some(id),
                    ..Default::default()
                },
            )
        };
        assert_eq!(w("battle-cities").len(), 1);
        assert_eq!(w("earth").len(), 1);
        assert_eq!(w("sandbox").len(), 0);
    }

    /// Ids must not collide, and must not be sequential — a sequential id says
    /// how many characters exist and in what order, which is the enumeration
    /// leak by another route.
    #[test]
    fn minted_ids_are_unique_and_not_sequential() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let mut ids = Vec::new();
        for i in 0..24 {
            let c = n
                .create(&ident("u1"), ME, &body(&format!("N{i}")), 1_000 + i)
                .unwrap();
            ids.push(nid(&c));
        }
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "ids collided");
        assert!(ids.iter().all(|&v| v != 0));
        // Not a counter: consecutive ids differ by more than 1 essentially
        // always. One accidental neighbour would be astronomically unlikely.
        assert!(
            ids.windows(2).filter(|w| w[1].abs_diff(w[0]) == 1).count() == 0,
            "ids look sequential"
        );
    }

    /// A durable record written from a typo is worse than a refused write.
    #[test]
    fn invalid_fields_are_refused_rather_than_defaulted() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        assert!(matches!(
            n.create(
                &ident("u1"),
                ME,
                &json!({ "world_id": "1", "personality_id": "1" }),
                1
            ),
            Err(NpcError::Invalid("name"))
        ));
        assert!(matches!(
            n.create(
                &ident("u1"),
                ME,
                &json!({ "name": "  ", "world_id": "1", "personality_id": "1" }),
                1
            ),
            Err(NpcError::Invalid("name"))
        ));
        assert!(matches!(
            n.create(
                &ident("u1"),
                ME,
                &json!({ "name": "A", "personality_id": "1" }),
                1
            ),
            Err(NpcError::Invalid("world_id"))
        ));

        let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = nid(&c);
        for bad in [
            json!({ "state": "melting" }),
            json!({ "state": "tombstoned" }),
            json!({ "heartbeat_ms": 10 }),
            json!({ "salience_gate": 1.5 }),
            json!({ "hidden": "yes" }),
        ] {
            assert!(
                matches!(n.patch(id, ME, &bad, 2_000), Err(NpcError::Invalid(_))),
                "accepted {bad}"
            );
        }
        // And none of the refusals bumped the revision.
        assert_eq!(n.view().get(id, ME).unwrap()["revision"], 1);
    }

    /// A reference is a file name. Anything that could not become one is
    /// refused here rather than written and discovered at spawn, and the rule
    /// is the registry's own — `world_id: "../etc"` names nothing, but a record
    /// carrying it is a durable reference to a path.
    #[test]
    fn a_reference_that_could_not_be_a_file_name_is_refused() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        for bad in [
            "../etc/passwd",
            "Battle-Cities", // uppercase: the registry is lowercase-only
            "battle cities",
            "-leading",
            "con",
            "",
        ] {
            let b = json!({ "name": "A", "world_id": bad, "personality_id": "commander" });
            assert!(
                matches!(
                    n.create(&ident("u1"), ME, &b, 1),
                    Err(NpcError::Invalid("world_id"))
                ),
                "accepted world_id `{bad}`"
            );
        }
        // A number is no longer a reference: these are slugs, and `4` was the
        // shape the previous `u64` ids took on the wire.
        let numeric = json!({ "name": "A", "world_id": 4, "personality_id": "commander" });
        assert!(matches!(
            n.create(&ident("u1"), ME, &numeric, 1),
            Err(NpcError::Invalid("world_id"))
        ));
    }

    /// The figure the personalities and worlds listings decorate themselves
    /// with. Hidden characters are **included** on purpose: a count that moved
    /// when one was hidden would say so to anybody polling it, which is the
    /// §8.3 leak with a smaller denominator than the roster's.
    #[test]
    fn hiding_a_character_does_not_move_a_count() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let mut b = json!({ "name": "Ilse", "world_id": "earth", "personality_id": "drifter" });
        b["hidden"] = json!(true);
        n.create(&ident("u1"), ME, &b, 1_001).unwrap();
        // Another owner's character still counts: doctrine reaches every
        // character of a personality, not only the publisher's.
        let mut c =
            json!({ "name": "Toll-keeper", "world_id": "earth", "personality_id": "commander" });
        c["tags"] = json!([]);
        n.create(&ident("u2"), "u_other", &c, 1_002).unwrap();

        let view = n.view();
        let worlds = view.counts_by(|x| x.world_id.as_str());
        assert_eq!(worlds.get("battle-cities"), Some(&1));
        assert_eq!(worlds.get("earth"), Some(&2), "the hidden one still counts");

        let people = view.counts_by(|x| x.personality_id.as_str());
        assert_eq!(people.get("commander"), Some(&2), "across owners");
        assert_eq!(people.get("drifter"), Some(&1), "hidden, and still counted");

        // The property that matters: hiding a character moves nothing. An
        // observer polling this cannot tell that anything happened.
        let owned = |view: &Cast| -> BTreeMap<String, usize> {
            view.counts_by(|x| x.personality_id.as_str())
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        };
        let before = owned(&view);
        let ilse: u64 = view
            .list(
                ME,
                &Filter {
                    include_hidden: true,
                    ..Default::default()
                },
            )
            .iter()
            .find(|v| v["name"] == "Ilse")
            .map(nid)
            .unwrap();
        n.patch(ilse, ME, &json!({ "hidden": false }), 1_500)
            .unwrap();
        assert_eq!(
            owned(&n.view()),
            before,
            "un-hiding moved the count, so hiding would too"
        );

        // A deleted character stops counting.
        let id: u64 = n
            .view()
            .list(ME, &Filter::default())
            .iter()
            .find(|v| v["name"] == "Varek")
            .map(nid)
            .unwrap();
        n.delete(id, ME, 2_000).unwrap();
        assert_eq!(
            n.view()
                .counts_by(|x| x.world_id.as_str())
                .get("battle-cities"),
            None
        );
    }

    /// Engine-derived values are absent, not zero. A character nothing has run
    /// has no pending count and no monitor band, and saying `0`/`healthy` would
    /// be a measurement nobody took.
    #[test]
    fn the_wire_shape_reports_engine_values_as_absent() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let v = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();

        assert_eq!(v["monitor"], Value::Null);
        assert_eq!(v["tick"]["pending_events"], Value::Null);
        assert_eq!(v["tick"]["last_tick_ms"], Value::Null);
        // Authored configuration IS present.
        assert_eq!(v["tick"]["heartbeat_ms"], DEFAULT_HEARTBEAT_MS);
        // Ids cross as strings — a u64 above 2^53 would be rounded as a number.
        assert!(v["npc_id"].is_string());
        assert!(v["world_id"].is_string());
        assert!(v["owner_id"].is_string());
    }

    /// **The roster reads as the MCP tools read it** (`engine::bodies::NpcList`):
    /// what `GET /v1/npc` lists is the shape `npc_list` names a character by.
    #[test]
    fn the_roster_reads_as_the_mcp_tools_npc_list() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let v = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let listed: NpcList =
            serde_json::from_value(json!({ "items": n.view().list(ME, &Filter::default()) }))
                .unwrap();
        assert_eq!(listed.items.len(), 1);
        let npc = &listed.items[0];
        assert_eq!(npc.npc_id, v["npc_id"].as_str().unwrap());
        assert_eq!(npc.name, "Varek");
        assert_eq!(npc.state, v["state"].as_str().unwrap());
        assert_eq!(npc.mood, None, "never asked is absent");
    }

    /// **A refused write publishes nothing and appends nothing**
    /// (`docs/npcd_cast_lock_free.md` §2): the cast a reader loads afterwards is
    /// the very one it loaded before, and a restart reads the same.
    #[test]
    fn a_refused_write_publishes_nothing() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());
        let before = n.view();

        assert!(matches!(
            n.put_belief(id, ME, "b", &json!({ "statement": "  " }), 2_000),
            Err(NpcError::Invalid("statement"))
        ));
        assert!(
            Arc::ptr_eq(&before, &n.view()),
            "a refused write published a cast"
        );

        // The one writer lets go before the restart opens the log again.
        drop(n);
        let reopened = Npcs::load(&dir).unwrap().view();
        assert_eq!(reopened.get(id, ME).unwrap()["revision"], 1);
    }

    /// **A snapshot a reader holds is not moved by a later write**: it answers
    /// as the cast was when it was taken, while a fresh view sees the write.
    #[test]
    fn a_snapshot_is_not_moved_by_a_write() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());
        let held = n.view();

        n.patch(id, ME, &json!({ "name": "Varek the Elder" }), 2_000)
            .unwrap();

        assert_eq!(held.get(id, ME).unwrap()["name"], "Varek");
        assert_eq!(held.get(id, ME).unwrap()["revision"], 1);
        assert_eq!(n.view().get(id, ME).unwrap()["name"], "Varek the Elder");
        assert_eq!(n.view().get(id, ME).unwrap()["revision"], 2);
    }

    /// **Concurrent writes to one character lose nothing**: each is computed
    /// from what the one before it published, so eight threads each adding
    /// their own belief leave all eight, at revision 1 + 8 — in memory and
    /// after a restart.
    #[test]
    fn concurrent_writes_to_one_character_all_land() {
        let dir = tmp();
        let n = Arc::new(Npcs::load(&dir).unwrap());
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());

        let writers: Vec<_> = (0..8)
            .map(|i| {
                let n = Arc::clone(&n);
                std::thread::spawn(move || {
                    let b = json!({ "statement": format!("belief {i}") });
                    n.put_belief(id, ME, &format!("b{i}"), &b, 2_000 + i)
                        .unwrap_or_else(|e| panic!("belief {i} refused: {e:?}"));
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }

        let check = |cast: &Cast| {
            let held = cast.payload(id).expect("still there");
            let mut ids: Vec<&str> = held.beliefs.iter().map(|b| b.belief_id.as_str()).collect();
            ids.sort_unstable();
            assert_eq!(ids, ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]);
            assert_eq!(held.revision, 9);
        };
        check(&n.view());
        drop(n);
        check(&Npcs::load(&dir).unwrap().view());
    }
}
