//! Reads and writes a character's record, publishing a change only once it is
//! durable.
//!
//! The registry is loaded from the substrate log once and changed only through
//! `write`, which builds records from the current cast, appends and fsyncs
//! them, and then publishes the cast with them in. A write that is not durable
//! is never published, so the cast never disagrees with the log. `refresh`
//! takes in characters another member replicated here, comparing each identity
//! against the record it was read from. Records are read by their newest
//! stamp, not by replay order, because a replicated older record can follow a
//! newer one in the log.

use super::*;

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

/// `cast` caught up with the records `p` holds that it was not read from —
/// written on another member and replicated here — and the ids whose record
/// changed; `None` when nothing moved.
pub(super) fn caught_up(p: &mut SubstratePersistence, cast: &Cast) -> Option<(Cast, Vec<u64>)> {
    let moved: Vec<(u64, RecordIdentity)> = p
        .npc_locs()
        .iter()
        .filter_map(|(id, _)| Some((*id, p.npc_locs().identity(id)?)))
        .filter(|(id, identity)| cast.read_as.get(id) != Some(identity))
        .collect();
    if moved.is_empty() {
        return None;
    }
    let mut next = Cast::clone(cast);
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
    Some((next, changed))
}

/// A record the log would not take, as the error every write returns.
pub(super) fn persist(e: impl Display) -> NpcError {
    NpcError::Persist(e.to_string())
}

/// Run `f`, which blocks on the persistence owner, without holding an async
/// runtime's worker: on a multi-threaded runtime the worker's other tasks move
/// to another thread for the length of it. A write waits for its own append
/// and fsync, behind whatever the owner is serving, and a cast written by many
/// tasks at once would otherwise park a worker per writer.
pub(super) fn off_the_runtime<R>(f: impl FnOnce() -> R) -> R {
    match Handle::try_current() {
        Ok(h) if h.runtime_flavor() == RuntimeFlavor::MultiThread => block_in_place(f),
        _ => f(),
    }
}

/// One superseding record and what the caller is shown of it.
pub(super) fn one(npc: NpcPayload, viewer: &str) -> (Vec<NpcPayload>, Value) {
    let view = wire(&npc, viewer);
    (vec![npc], view)
}

/// `npc` with its revision moved on — every authored edit's record.
pub(super) fn bumped(mut npc: NpcPayload, now_ms: u64) -> NpcPayload {
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
        let mut persistence = SubstratePersistence::open_in_with_substrate_and_sink(
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

        // Each character is read from the record its index names, the one a
        // refresh compares against — not the last one replay passed. A record
        // replicated in carries its writer's stamp, so a log can hold an older
        // one after a newer: taking replay's last kept the older content under
        // the newer one's identity, and no refresh ever read it again.
        let replayed = Cast {
            by_id: by_id.into_iter().collect(),
            read_as: Default::default(),
        };
        let cast = caught_up(&mut persistence, &replayed).map_or(replayed, |(cast, _)| cast);
        Ok(Self {
            current: ArcSwap::from_pointee(cast),
            shared: SharedSubstrate::new(substrate, persistence),
            taken_in: Mutex::new(Vec::new()),
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
    /// **Computed from every record the log holds**, those replicated since
    /// the last refresh included: the cast is caught up first, in the same
    /// step. An edit clones the whole record it changes, so one computed from
    /// a cast behind the log would write the older record back over the newer
    /// — a mood named from the copy before another member's suspension lands
    /// as the character's record, unsuspended, on every member.
    ///
    /// Nothing is published unless it is durable, and everything durable is
    /// published: a refused edit leaves the cast exactly as it was, and when an
    /// append fails partway through a batch, the records staged before it are
    /// committed and published and the error returned — so the cast never
    /// disagrees with the log. Each published record is the one its character
    /// is read as, so a refresh does not read it again. An edit that returns no
    /// records writes nothing. `edit` runs on the owner's thread, so it must be
    /// pure over the cast it is handed (§2.3).
    pub(super) fn write<R: Send>(
        &self,
        edit: impl FnOnce(&Cast) -> Result<(Vec<NpcPayload>, R), NpcError> + Send,
    ) -> Result<R, NpcError> {
        let current = &self.current;
        off_the_runtime(|| {
            self.shared.persistence.call(|p| {
                let mut cast = current.load_full();
                if let Some((next, changed)) = caught_up(p, &cast) {
                    let next = Arc::new(next);
                    current.store(Arc::clone(&next));
                    cast = next;
                    self.taken_in.lock().unwrap().extend(changed);
                }
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
    pub(super) fn refresh_on(&self, current: &ArcSwap<Cast>) -> Vec<u64> {
        self.shared.persistence.call(|p| {
            let mut changed = mem::take(&mut *self.taken_in.lock().unwrap());
            if let Some((next, more)) = caught_up(p, &current.load_full()) {
                current.store(Arc::new(next));
                changed.extend(more);
            }
            changed.sort_unstable();
            changed.dedup();
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
            // No suspension of the whole cast holds it.
            suspended_from: None,
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
    pub(super) fn patched(mut npc: NpcPayload, body: &Value) -> Result<NpcPayload, NpcError> {
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
            // The owner has said where it stands, which resuming the whole
            // cast must not then overrule.
            npc.suspended_from = None;
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
    pub(super) fn strategized(
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

    /// Move every character of world `from` to world `to` — the world renamed
    /// (`docs/world_rename.md` §3): each record keeps everything but its
    /// `world_id`. Returns how many moved; none once they all have.
    pub fn rehome(&self, from: &str, to: &str, now_ms: u64) -> Result<usize, NpcError> {
        self.write(|cast| {
            let moved: Vec<NpcPayload> = cast
                .export_all()
                .into_iter()
                .filter(|n| n.world_id == from)
                .map(|mut n| {
                    n.world_id = to.to_string();
                    bumped(n, now_ms)
                })
                .collect();
            let n = moved.len();
            Ok((moved, n))
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
