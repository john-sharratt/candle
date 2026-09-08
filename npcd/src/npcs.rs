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
//! [`Npcs::load`] performs the process's **only** open of `--data/.substrate/`,
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

use std::collections::BTreeMap;
use std::path::Path;

use candle_conversation::persistence::record::{
    AuthoredBelief, AuthoredRelationship, AuthoredStrategy, Modulation, NpcPayload, RecordType,
};
use candle_conversation::persistence::{SharedSubstrate, SubstratePersistence};
use candle_conversation::substrate::Substrate;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use web::auth::session::Identity;

use crate::registry::id;

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

/// The shortest interval between two durable records of where a character is.
///
/// **Not a tuning knob — the difference between a bounded write rate and an
/// unbounded one.** A world's metronome runs at `driver::EVERY` (500 ms), and a
/// walking body covers a leg per tick, so a character crossing the building
/// changes room twice a second. Writing the record on each of those would append
/// the whole payload — persona description and all, up to several kilobytes —
/// twice a second per moving character, and supersede the previous copy each
/// time. That is megabytes a minute of dead weight for a cast of two, to record
/// something that is soft state.
///
/// So position is *checkpointed*: written when it has actually changed and at
/// most this often. The cost is that a hard kill can lose up to this much
/// movement and a character reconstructs a room or two behind — which is
/// acceptable for state whose whole purpose is to avoid starting everybody at
/// the front door, and which is the truth about where they were within the last
/// half minute.
const PLACE_CHECKPOINT_MS: u64 = 30_000;

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
    /// Where it was last standing, as `area/node`. `None` for a character that
    /// has never been embodied — it starts at the way in.
    pub at: Option<String>,
}

#[derive(Debug)]
pub enum NpcError {
    /// The caller is not this character's owner. Deliberately indistinguishable
    /// from "no such character" at the API boundary — see [`Npcs::visible_to`].
    NotFound,
    Invalid(&'static str),
    Persist(String),
}

/// The whole cast, including tombstoned characters.
///
/// Tombstoned ones are kept because their ids must stay taken; every read path
/// filters them out.
pub struct Npcs {
    by_id: BTreeMap<u64, NpcPayload>,
    /// **The daemon's only handle to the substrate**, shared with the engine
    /// rather than opened twice. See [`Npcs::substrate`].
    shared: SharedSubstrate,
    /// When each character's place was last written durably.
    ///
    /// In memory only, and deliberately: it is the debounce for
    /// [`PLACE_CHECKPOINT_MS`], not a fact about the character. A restart
    /// forgetting it means the first move after boot checkpoints immediately,
    /// which is the behaviour you want anyway.
    place_written_ms: BTreeMap<u64, u64>,
}

impl Npcs {
    /// Open the substrate at `dir/.substrate/` and rebuild the cast from it.
    ///
    /// The one read of the log. Records arrive in append order, so inserting
    /// each into the map *is* last-writer-wins — no ordering pass, no revision
    /// comparison.
    ///
    /// **This is the process's one open of that directory.** The engine adopts
    /// the same handle through [`Self::substrate`] instead of opening its own —
    /// see that method for what the second handle destroyed.
    pub fn load(dir: &Path) -> Result<Self, NpcError> {
        let mut substrate = Substrate::new();
        let mut by_id: BTreeMap<u64, NpcPayload> = BTreeMap::new();

        let persistence =
            SubstratePersistence::open_in_with_substrate_and_sink(dir, &mut substrate, |entry| {
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
            })
            .map_err(|e| NpcError::Persist(e.to_string()))?;

        let live = by_id.values().filter(|n| !n.is_tombstoned()).count();
        tracing::info!(
            "cast: {live} characters ({} records, {} tombstoned) from {}",
            by_id.len(),
            by_id.len() - live,
            dir.join(".substrate").display()
        );

        Ok(Self {
            by_id,
            shared: SharedSubstrate::new(substrate, persistence),
            place_written_ms: BTreeMap::new(),
        })
    }

    /// Where a character was last standing, as `area/node`.
    ///
    /// What the runtime puts a body back at on a restart, in place of the
    /// world's arrival door. `None` for a character that has never been
    /// embodied — or whose recorded room the map no longer has, which the
    /// caller resolves, not this.
    pub fn place_of(&self, npc_id: u64) -> Option<&str> {
        self.by_id
            .get(&npc_id)
            .filter(|n| !n.is_tombstoned())
            .and_then(|n| n.at.as_deref())
    }

    /// Record where a character is, if it has moved and enough time has passed.
    ///
    /// Returns whether a record was written. Called from the tick driver every
    /// moment a body moves, so **the two gates are what make it affordable**:
    /// nothing is written for a character standing still, and a character
    /// walking across the building writes at most once per
    /// [`PLACE_CHECKPOINT_MS`] rather than twice a second.
    ///
    /// Fsynced like any other write, and affordable **because** of the gates
    /// above: at one record per thirty seconds per moving character, the cost of
    /// making the checkpoint durable is nothing, and a staged one would be worth
    /// very little. The failure this exists to prevent is a daemon restart, and
    /// a restart that took the log's un-flushed tail with it would put the
    /// character back at the door — the exact outcome the record is for.
    pub fn remember_place(&mut self, npc_id: u64, at: &str, now_ms: u64) -> bool {
        let Some(npc) = self.by_id.get(&npc_id).filter(|n| !n.is_tombstoned()) else {
            return false;
        };
        if npc.at.as_deref() == Some(at) {
            return false;
        }
        // A character with no place yet is checkpointed at once: it has just
        // been embodied, and the whole point is that a restart finds it there.
        if npc.at.is_some() {
            let last = self.place_written_ms.get(&npc_id).copied().unwrap_or(0);
            if now_ms.saturating_sub(last) < PLACE_CHECKPOINT_MS {
                return false;
            }
        }

        let mut npc = npc.clone();
        npc.at = Some(at.to_string());
        // **No revision bump and no `updated_ms`.** Those describe authored
        // change — what somebody edited, and what the console shows as the last
        // time this character was worked on. A body walking through a door is
        // neither, and counting it would make every roster sort by "recently
        // edited" report whoever happens to be walking.
        let write = {
            let mut p = self
                .shared
                .persistence
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            p.write_npc(&npc).and_then(|()| p.commit())
        };
        if let Err(e) = write {
            // Not fatal and not propagated: this is a checkpoint of soft state
            // on a driver thread with nobody to report to. The character keeps
            // walking; the next checkpoint tries again.
            tracing::warn!("could not record where {npc_id} is: {e}");
            return false;
        }
        self.place_written_ms.insert(npc_id, now_ms);
        self.by_id.insert(npc_id, npc);
        true
    }

    /// The substrate handle, for the engine to adopt.
    ///
    /// **The engine must take this rather than open `--data` itself.** One
    /// `.substrate/` admits exactly one writable handle per process: the log
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

    /// Every character this caller may see, newest first.
    ///
    /// Ownership is the filter (§8.2). Hidden characters are excluded unless
    /// asked for, and no total is ever produced — a count of everything you own
    /// is the one figure that gives a hidden character away (§8.3).
    pub fn list(&self, owner: &str, f: &Filter<'_>) -> Vec<Value> {
        let mut rows: Vec<&NpcPayload> = self
            .by_id
            .values()
            .filter(|n| n.owner_id == owner && !n.is_tombstoned())
            .filter(|n| f.include_hidden || !n.hidden)
            .filter(|n| f.matches(n))
            .collect();
        rows.sort_unstable_by(|a, b| {
            b.updated_ms
                .cmp(&a.updated_ms)
                .then(a.npc_id.cmp(&b.npc_id))
        });
        rows.into_iter().map(|n| wire(n, owner)).collect()
    }

    /// How many living characters each authored document has, keyed by slug.
    ///
    /// `pick` selects which reference to count by, so one walk serves both
    /// registries. Every owner's cast is included, because the figure this
    /// answers is a global one: publishing doctrine reaches every character of
    /// that personality, not only the publisher's.
    ///
    /// Hidden characters **are** counted, which is the opposite of what it
    /// first looks like it should be.
    ///
    /// §8.3 says a hidden character must not be deducible. *Excluding* them is
    /// what breaches it: the figure would drop the moment one was hidden, so
    /// anybody polling learns that a character was just hidden and under which
    /// personality — a sharper signal than the roster gives, because the
    /// denominator is smaller. Including them makes hiding invisible here,
    /// which is the whole point of hiding.
    ///
    /// What remains is a global aggregate over every owner. It answers "how
    /// many of these exist" and never "how many do *you* have" — the per-owner
    /// total is the one §8.3 forbids, and [`Npcs::list`] still refuses to
    /// produce it.
    pub fn counts_by<'a>(
        &'a self,
        pick: fn(&'a NpcPayload) -> &'a str,
    ) -> BTreeMap<&'a str, usize> {
        let mut out = BTreeMap::new();
        for n in self.by_id.values() {
            if n.is_tombstoned() {
                continue;
            }
            *out.entry(pick(n)).or_insert(0) += 1;
        }
        out
    }

    /// One character, if this caller may see it.
    ///
    /// A character owned by somebody else reads as absent rather than
    /// forbidden. A 403 would confirm the id exists, which is enough to
    /// enumerate a stranger's cast one guess at a time.
    pub fn visible_to(&self, npc_id: u64, owner: &str) -> Option<&NpcPayload> {
        self.by_id
            .get(&npc_id)
            .filter(|n| n.owner_id == owner && !n.is_tombstoned())
    }

    pub fn get(&self, npc_id: u64, owner: &str) -> Result<Value, NpcError> {
        self.visible_to(npc_id, owner)
            .map(|n| wire(n, owner))
            .ok_or(NpcError::NotFound)
    }

    /// Every living character, with the world it belongs to.
    ///
    /// **Deliberately not filtered by owner.** The tick scheduler runs the whole
    /// cast — the quartermaster counts sacks whether or not the person who
    /// authored him is signed in — so this is the one read that crosses
    /// ownership. Every route that reaches a character on a user's behalf still
    /// goes through [`Self::visible_to`]; this is for the engine, which serves
    /// the world rather than a caller.
    pub fn cast(&self) -> Vec<Casting> {
        self.by_id
            .values()
            .filter(|n| !n.is_tombstoned())
            .map(|n| Casting {
                npc_id: n.npc_id,
                world_id: n.world_id.clone(),
                at: n.at.clone(),
            })
            .collect()
    }

    /// The world a character belongs to, whoever owns it. Used by the tick
    /// driver to resolve which clock a character lives on.
    pub fn world_of(&self, npc_id: u64) -> Option<&str> {
        self.by_id
            .get(&npc_id)
            .filter(|n| !n.is_tombstoned())
            .map(|n| n.world_id.as_str())
    }

    /// The ids of every living character this account owns.
    ///
    /// A set, because the caller is testing membership per tick record and a
    /// linear scan over a large cast per row is the kind of cost that only shows
    /// up once somebody has two hundred characters.
    pub fn owned_by(&self, owner: &str) -> std::collections::HashSet<u64> {
        self.by_id
            .values()
            .filter(|n| !n.is_tombstoned() && n.owner_id == owner)
            .map(|n| n.npc_id)
            .collect()
    }

    /// A living character's record, whoever owns it.
    ///
    /// Ownership-blind for the same reason [`Self::cast`] is: the engine serves
    /// the world rather than a caller. Every route that reaches a character on a
    /// user's behalf still goes through [`Self::visible_to`].
    pub fn payload(&self, npc_id: u64) -> Option<&NpcPayload> {
        self.by_id.get(&npc_id).filter(|n| !n.is_tombstoned())
    }

    /// Create a character owned by the caller.
    pub fn create(
        &mut self,
        id: &Identity,
        owner: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let name = clean_name(body.get("name"))?;
        let world_id = ref_id(body.get("world_id")).ok_or(NpcError::Invalid("world_id"))?;
        let personality_id =
            ref_id(body.get("personality_id")).ok_or(NpcError::Invalid("personality_id"))?;

        let npc_id = self.mint_id(id, &name, now_ms);
        let npc = NpcPayload {
            npc_id,
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
            // The authoring plane starts empty. A character nobody has written
            // beliefs for holds none — which is different from one whose
            // beliefs could not be read, and is what the console shows.
            beliefs: Vec::new(),
            relationships: Vec::new(),
            agency: Vec::new(),
            modulation: Modulation::default(),
        };
        self.commit(npc, owner)
    }

    /// Apply a partial edit. Absent fields are left alone; a field present and
    /// invalid is an error rather than a silent default, because a durable
    /// record written from a typo is worse than a refused write.
    pub fn patch(
        &mut self,
        npc_id: u64,
        owner: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let mut npc = self
            .visible_to(npc_id, owner)
            .ok_or(NpcError::NotFound)?
            .clone();

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

        npc.revision += 1;
        npc.updated_ms = now_ms;
        self.commit(npc, owner)
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
        &mut self,
        npc_id: u64,
        owner: &str,
        belief_id: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let mut npc = self.owned(npc_id, owner)?;
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
        self.bump(npc, owner, now_ms)
    }

    pub fn delete_belief(
        &mut self,
        npc_id: u64,
        owner: &str,
        belief_id: &str,
        now_ms: u64,
    ) -> Result<bool, NpcError> {
        let mut npc = self.owned(npc_id, owner)?;
        let before = npc.beliefs.len();
        npc.beliefs.retain(|b| b.belief_id != belief_id);
        if npc.beliefs.len() == before {
            return Ok(false);
        }
        self.bump(npc, owner, now_ms)?;
        Ok(true)
    }

    /// Set how this character holds somebody.
    pub fn put_relationship(
        &mut self,
        npc_id: u64,
        owner: &str,
        entity_id: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let mut npc = self.owned(npc_id, owner)?;
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
        self.bump(npc, owner, now_ms)
    }

    /// State a strategy, optionally under another.
    pub fn put_strategy(
        &mut self,
        npc_id: u64,
        owner: &str,
        strategy_id: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let mut npc = self.owned(npc_id, owner)?;
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
        self.bump(npc, owner, now_ms)
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
        &mut self,
        npc_id: u64,
        owner: &str,
        image_id: String,
        origin: &str,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let mut npc = self.owned(npc_id, owner)?;
        npc.portrait_image_id = Some(image_id);
        npc.portrait_origin = Some(origin.to_string());
        self.bump(npc, owner, now_ms)
    }

    /// Set the affect dials.
    pub fn put_modulation(
        &mut self,
        npc_id: u64,
        owner: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let mut npc = self.owned(npc_id, owner)?;
        npc.modulation.affect = signed(body, "affect", npc.modulation.affect)?;
        npc.modulation.threat = unit(body, "threat", npc.modulation.threat)?;
        npc.modulation.curiosity = unit(body, "curiosity", npc.modulation.curiosity)?;
        self.bump(npc, owner, now_ms)
    }

    /// The character, if the caller owns it. Every authoring write starts here:
    /// ownership is authorization (§8.2), and a role cannot express "yours".
    fn owned(&self, npc_id: u64, owner: &str) -> Result<NpcPayload, NpcError> {
        let npc = self.visible_to(npc_id, owner).ok_or(NpcError::NotFound)?;
        if npc.owner_id != owner {
            return Err(NpcError::NotFound);
        }
        Ok(npc.clone())
    }

    /// One superseding record, with the revision moved on.
    fn bump(&mut self, mut npc: NpcPayload, owner: &str, now_ms: u64) -> Result<Value, NpcError> {
        npc.revision += 1;
        npc.updated_ms = now_ms;
        self.commit(npc, owner)
    }

    /// Delete a character: one superseding record with `state: "tombstoned"`.
    /// The record stays, so the id stays taken.
    pub fn delete(&mut self, npc_id: u64, owner: &str, now_ms: u64) -> Result<(), NpcError> {
        let mut npc = self
            .visible_to(npc_id, owner)
            .ok_or(NpcError::NotFound)?
            .clone();
        npc.state = "tombstoned".to_string();
        npc.revision += 1;
        npc.updated_ms = now_ms;
        self.commit(npc, owner)?;
        Ok(())
    }

    /// Write the record, then update memory — in that order.
    ///
    /// If the append fails the map is untouched, so the daemon's view still
    /// matches the log. The other order would leave a character that exists in
    /// memory, vanishes on restart, and is never written again because nothing
    /// knows it is missing.
    fn commit(&mut self, npc: NpcPayload, owner: &str) -> Result<Value, NpcError> {
        {
            // The engine holds this same lock for its own appends, which is the
            // point: one cursor, one `npc_locs`, so a compaction on the engine's
            // side sees this character. Held across both calls so nothing
            // interleaves between the append and its fsync.
            let mut p = self
                .shared
                .persistence
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            p.write_npc(&npc)
                .map_err(|e| NpcError::Persist(e.to_string()))?;
            // Flush and fsync before returning. `write_npc` only *stages* the
            // record, and a staged record is lost on a crash — which for a
            // character somebody just created means the API said "created" about
            // something that never existed.
            //
            // Group-committing instead would be the right call for a hot write
            // path; this one is a person pressing save, so the fsync is both
            // affordable and what they are entitled to assume happened.
            p.commit().map_err(|e| NpcError::Persist(e.to_string()))?;
        }
        let view = wire(&npc, owner);
        self.by_id.insert(npc.npc_id, npc);
        Ok(view)
    }

    /// A fresh id, unique across the whole cast including tombstoned ones.
    ///
    /// Derived rather than sequential: a sequential id leaks how many
    /// characters exist and in what order they were made, which is the §8.3
    /// enumeration problem by another route. The hash is not a secret — the
    /// subject and name are known to their owner — it is only a spread.
    fn mint_id(&self, id: &Identity, name: &str, now_ms: u64) -> u64 {
        for salt in 0u32.. {
            let mut h = Sha256::new();
            h.update(id.sub.as_bytes());
            h.update(name.as_bytes());
            h.update(now_ms.to_le_bytes());
            h.update(salt.to_le_bytes());
            let d = h.finalize();
            let v = u64::from_le_bytes(d[..8].try_into().expect("32-byte digest"));
            // Never zero: zero is the header's "no stream" value, and an id that
            // collides with an absence is a debugging session nobody needs.
            if v != 0 && !self.by_id.contains_key(&v) {
                return v;
            }
        }
        unreachable!("u32 salts exhausted against a u64 keyspace")
    }
}

/// One character as §10 defines it on the wire.
///
/// Ids cross as decimal **strings**: they are `u64`, and a JSON number above
/// 2^53 is silently rounded by every browser that parses it.
fn wire(n: &NpcPayload, caller: &str) -> Value {
    json!({
        "npc_id": n.npc_id.to_string(),
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

    /// The property the whole module exists for: a character outlives the
    /// process. Written by one registry, read back by another over the same
    /// directory, with no engine in between.
    #[test]
    fn a_character_survives_a_restart() {
        let dir = tmp();
        let created = {
            let mut n = Npcs::load(&dir).unwrap();
            n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap()
        };
        let npc_id: u64 = created["npc_id"].as_str().unwrap().parse().unwrap();

        // A second registry over the same log — a restart, in effect.
        let reopened = Npcs::load(&dir).unwrap();
        let back = reopened.get(npc_id, ME).unwrap();
        assert_eq!(back["name"], "Varek");
        assert_eq!(back["owner_id"], ME);
        assert_eq!(back["revision"], 1);
        assert_eq!(reopened.list(ME, &Filter::default()).len(), 1);
    }

    /// **A character created while the daemon runs survives the engine's
    /// maintenance.**
    ///
    /// The regression guard for the bug that lost Makers overnight. The engine
    /// used to open `--data/.substrate/` for itself, so it held a second
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
            let mut n = Npcs::load(&dir).unwrap();
            // Whatever the engine does to this log, it does through this handle
            // — the one the registry is writing through.
            let engine = n.substrate();

            let created = n.create(&ident("u1"), ME, &body("Wyneth"), 1_000).unwrap();

            // The engine's routine compaction, on the handle it was handed.
            {
                let mut p = engine.persistence.lock().unwrap();
                let mut s = engine.substrate.write().unwrap();
                p.compact(&mut s, None).unwrap();
            }
            created["npc_id"].as_str().unwrap().parse().unwrap()
        };

        // A restart.
        let reopened = Npcs::load(&dir).unwrap();
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
        let mut n = Npcs::load(&dir).unwrap();
        let id: u64 = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap()["npc_id"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();

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
        let held = n.payload(id).expect("still there");
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

    /// **Where a character is survives a restart, so the world can be rebuilt
    /// from the cast.**
    ///
    /// The world is not persisted — who stands where lives in RAM and goes with
    /// the process — so this record is what stops everybody re-entering at the
    /// arrival door. Two Makers who had spent an hour finding each other were
    /// returned to the front room as strangers, with their own transcripts
    /// saying otherwise.
    #[test]
    fn where_a_character_is_survives_a_restart() {
        let dir = tmp();
        let id: u64 = {
            let mut n = Npcs::load(&dir).unwrap();
            let id: u64 = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap()["npc_id"]
                .as_str()
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(n.place_of(id), None, "a new character stands nowhere yet");

            assert!(
                n.remember_place(id, "vault-casting/green-room", 10_000),
                "the first place is recorded at once, not after a delay"
            );
            assert_eq!(n.place_of(id), Some("vault-casting/green-room"));
            id
        };

        let back = Npcs::load(&dir).unwrap();
        assert_eq!(
            back.place_of(id),
            Some("vault-casting/green-room"),
            "the room did not survive the restart"
        );
        // And it rides along with the cast, which is what the loader reads.
        let casting = back.cast().into_iter().find(|c| c.npc_id == id).unwrap();
        assert_eq!(casting.at.as_deref(), Some("vault-casting/green-room"));
    }

    /// **A moving body is checkpointed, not transcribed.**
    ///
    /// A world's metronome runs at 500 ms and a walking body covers a leg per
    /// tick, so recording every room change would append the whole record —
    /// persona description and all — twice a second per moving character, and
    /// supersede the previous copy each time. The two gates are what make the
    /// driver able to call this every moment: unchanged is free, and changed is
    /// bounded by [`PLACE_CHECKPOINT_MS`].
    #[test]
    fn a_moving_body_is_checkpointed_rather_than_written_every_step() {
        let dir = tmp();
        let mut n = Npcs::load(&dir).unwrap();
        let id: u64 = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap()["npc_id"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();

        assert!(n.remember_place(id, "vault-command/command-room", 1_000));
        // Standing still costs nothing, however often it is asked.
        assert!(!n.remember_place(id, "vault-command/command-room", 99_000));
        // Moving again inside the window is held back...
        assert!(!n.remember_place(id, "vault-command/anteroom", 2_000));
        assert!(!n.remember_place(id, "vault-command/dispatch-room", 3_000));
        assert_eq!(
            n.place_of(id),
            Some("vault-command/command-room"),
            "a checkpoint inside the window must not have been written"
        );
        // ...and the next one past it records wherever the body ended up.
        assert!(n.remember_place(id, "vault-casting/green-room", 1_000 + PLACE_CHECKPOINT_MS));
        assert_eq!(n.place_of(id), Some("vault-casting/green-room"));
    }

    /// A checkpoint is not an edit. `revision` and `updated_ms` describe what
    /// somebody authored, and a body walking through a door is neither — a
    /// roster sorted by "recently edited" would otherwise rank whoever happens
    /// to be walking.
    #[test]
    fn walking_does_not_count_as_editing_the_character() {
        let dir = tmp();
        let mut n = Npcs::load(&dir).unwrap();
        let made = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = made["npc_id"].as_str().unwrap().parse().unwrap();

        n.remember_place(id, "vault-casting/green-room", 50_000);

        let after = n.get(id, ME).unwrap();
        assert_eq!(after["revision"], made["revision"]);
        assert_eq!(after["updated_ms"], made["updated_ms"]);
    }

    /// An edit supersedes rather than accumulating: the reopened registry sees
    /// exactly one character, at the newest revision.
    #[test]
    fn an_edit_supersedes_the_previous_record() {
        let dir = tmp();
        let npc_id: u64 = {
            let mut n = Npcs::load(&dir).unwrap();
            let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
            let id: u64 = c["npc_id"].as_str().unwrap().parse().unwrap();
            n.patch(id, ME, &json!({ "name": "Varek the Elder" }), 2_000)
                .unwrap();
            n.patch(id, ME, &json!({ "state": "active" }), 3_000)
                .unwrap();
            id
        };

        let reopened = Npcs::load(&dir).unwrap();
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
            let mut n = Npcs::load(&dir).unwrap();
            let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
            let id: u64 = c["npc_id"].as_str().unwrap().parse().unwrap();
            n.delete(id, ME, 2_000).unwrap();
            assert!(n.list(ME, &Filter::default()).is_empty());
            assert!(matches!(n.get(id, ME), Err(NpcError::NotFound)));
            id
        };

        let reopened = Npcs::load(&dir).unwrap();
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
        let mut n = Npcs::load(&dir).unwrap();
        let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = c["npc_id"].as_str().unwrap().parse().unwrap();

        assert!(matches!(n.get(id, OTHER), Err(NpcError::NotFound)));
        assert!(n.list(OTHER, &Filter::default()).is_empty());
        assert!(matches!(
            n.patch(id, OTHER, &json!({ "name": "Stolen" }), 2_000),
            Err(NpcError::NotFound)
        ));
        assert!(matches!(
            n.delete(id, OTHER, 2_000),
            Err(NpcError::NotFound)
        ));
        // And the original is untouched by the attempts.
        assert_eq!(n.get(id, ME).unwrap()["name"], "Varek");
    }

    /// Hidden characters stay out of the default listing (§8.3).
    #[test]
    fn hidden_characters_are_opt_in() {
        let dir = tmp();
        let mut n = Npcs::load(&dir).unwrap();
        n.create(&ident("u1"), ME, &body("Seen"), 1_000).unwrap();
        let mut hidden_body = body("Unseen");
        hidden_body["hidden"] = json!(true);
        n.create(&ident("u1"), ME, &hidden_body, 1_001).unwrap();

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
        let mut n = Npcs::load(&dir).unwrap();
        let mut a = body("Varek");
        a["tags"] = json!(["north", "campaign-2"]);
        n.create(&ident("u1"), ME, &a, 1_000).unwrap();
        let mut b = json!({ "name": "Ilse", "world_id": "earth", "personality_id": "commander" });
        b["tags"] = json!(["market"]);
        n.create(&ident("u1"), ME, &b, 1_001).unwrap();

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
        let mut n = Npcs::load(&dir).unwrap();
        let mut ids = Vec::new();
        for i in 0..24 {
            let c = n
                .create(&ident("u1"), ME, &body(&format!("N{i}")), 1_000 + i)
                .unwrap();
            ids.push(c["npc_id"].as_str().unwrap().parse::<u64>().unwrap());
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
        let mut n = Npcs::load(&dir).unwrap();
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
        let id: u64 = c["npc_id"].as_str().unwrap().parse().unwrap();
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
        assert_eq!(n.get(id, ME).unwrap()["revision"], 1);
    }

    /// A reference is a file name. Anything that could not become one is
    /// refused here rather than written and discovered at spawn, and the rule
    /// is the registry's own — `world_id: "../etc"` names nothing, but a record
    /// carrying it is a durable reference to a path.
    #[test]
    fn a_reference_that_could_not_be_a_file_name_is_refused() {
        let dir = tmp();
        let mut n = Npcs::load(&dir).unwrap();
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
        let mut n = Npcs::load(&dir).unwrap();
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

        let worlds = n.counts_by(|x| x.world_id.as_str());
        assert_eq!(worlds.get("battle-cities"), Some(&1));
        assert_eq!(worlds.get("earth"), Some(&2), "the hidden one still counts");

        let people = n.counts_by(|x| x.personality_id.as_str());
        assert_eq!(people.get("commander"), Some(&2), "across owners");
        assert_eq!(people.get("drifter"), Some(&1), "hidden, and still counted");

        // The property that matters: hiding a character moves nothing. An
        // observer polling this cannot tell that anything happened.
        // Owned, so the map does not keep borrowing `n` across the edit.
        let before: BTreeMap<String, usize> = n
            .counts_by(|x| x.personality_id.as_str())
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let ilse: u64 = n
            .list(
                ME,
                &Filter {
                    include_hidden: true,
                    ..Default::default()
                },
            )
            .iter()
            .find(|v| v["name"] == "Ilse")
            .and_then(|v| v["npc_id"].as_str())
            .unwrap()
            .parse()
            .unwrap();
        n.patch(ilse, ME, &json!({ "hidden": false }), 1_500)
            .unwrap();
        let after: BTreeMap<String, usize> = n
            .counts_by(|x| x.personality_id.as_str())
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        assert_eq!(
            after, before,
            "un-hiding moved the count, so hiding would too"
        );

        // A deleted character stops counting.
        let id: u64 = n
            .list(ME, &Filter::default())
            .iter()
            .find(|v| v["name"] == "Varek")
            .and_then(|v| v["npc_id"].as_str())
            .unwrap()
            .parse()
            .unwrap();
        n.delete(id, ME, 2_000).unwrap();
        assert_eq!(
            n.counts_by(|x| x.world_id.as_str()).get("battle-cities"),
            None
        );
    }

    /// Engine-derived values are absent, not zero. A character nothing has run
    /// has no pending count and no monitor band, and saying `0`/`healthy` would
    /// be a measurement nobody took.
    #[test]
    fn the_wire_shape_reports_engine_values_as_absent() {
        let dir = tmp();
        let mut n = Npcs::load(&dir).unwrap();
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
}
