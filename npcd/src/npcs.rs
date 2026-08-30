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
use candle_conversation::persistence::SubstratePersistence;
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
    persistence: SubstratePersistence,
    /// Held only because `SubstratePersistence` needs one to walk with. This
    /// daemon runs no inference, so nothing else ever touches it.
    _substrate: Substrate,
}

impl Npcs {
    /// Open the substrate at `dir/.substrate/` and rebuild the cast from it.
    ///
    /// The one read of the log. Records arrive in append order, so inserting
    /// each into the map *is* last-writer-wins — no ordering pass, no revision
    /// comparison.
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
            persistence,
            _substrate: substrate,
        })
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
            environment_enabled: body
                .get("environment_enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            heartbeat_ms: DEFAULT_HEARTBEAT_MS,
            salience_gate: DEFAULT_SALIENCE_GATE,
            tags: clean_tags(body.get("tags"))?,
            persona_description: clean_persona(body.get("persona_description"))?,
            persona_origin: "authored".to_string(),
            portrait_image_id: None,
            portrait_origin: None,
            // The authoring plane starts empty. A character nobody has written
            // beliefs for holds none — which is different from one whose
            // beliefs could not be read, and is what the console shows.
            beliefs: Vec::new(),
            relationships: Vec::new(),
            agency: Vec::new(),
            modulation: Modulation::default(),
            environment_prompt: String::new(),
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
        if let Some(v) = body.get("environment_enabled") {
            npc.environment_enabled = v
                .as_bool()
                .ok_or(NpcError::Invalid("environment_enabled"))?;
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
            None => npc.beliefs.push(belief),
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
            None => npc.relationships.push(rel),
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
            None => npc.agency.push(st),
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

    /// Set the simulated environment: whether it runs, and what it says.
    pub fn put_environment(
        &mut self,
        npc_id: u64,
        owner: &str,
        body: &Value,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        let mut npc = self.owned(npc_id, owner)?;
        if let Some(v) = body.get("enabled") {
            npc.environment_enabled = v.as_bool().ok_or(NpcError::Invalid("enabled"))?;
        }
        if let Some(v) = body.get("system_prompt") {
            let s = v.as_str().ok_or(NpcError::Invalid("system_prompt"))?;
            if s.chars().count() > MAX_PROMPT_CHARS {
                return Err(NpcError::Invalid("system_prompt"));
            }
            npc.environment_prompt = s.to_string();
        }
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
        self.persistence
            .write_npc(&npc)
            .map_err(|e| NpcError::Persist(e.to_string()))?;
        // Flush and fsync before returning. `write_npc` only *stages* the
        // record, and a staged record is lost on a crash — which for a
        // character somebody just created means the API said "created" about
        // something that never existed.
        //
        // Group-committing instead would be the right call for a hot write
        // path; this one is a person pressing save, so the fsync is both
        // affordable and what they are entitled to assume happened.
        self.persistence
            .commit()
            .map_err(|e| NpcError::Persist(e.to_string()))?;
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
        "environment_enabled": n.environment_enabled,
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

pub fn environment_wire(n: &NpcPayload) -> Value {
    json!({
        "enabled": n.environment_enabled,
        "system_prompt": n.environment_prompt,
        // What the simulated environment has actually done. It has not run.
        "last_event_ms": Value::Null,
        "events": Value::Null,
    })
}

/// The longest a simulated environment's instructions may be. Generous for
/// prose, short of a way to fill a disk one save at a time.
const MAX_PROMPT_CHARS: usize = 8_000;

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

/// One line of authored text, trimmed and bounded.
fn clean_line(v: &Value, key: &'static str) -> Result<String, NpcError> {
    let s = v.as_str().ok_or(NpcError::Invalid(key))?.trim();
    if s.chars().count() > MAX_PROMPT_CHARS {
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
