//! Loading worlds, placing characters in them, and taking them out again.
//!
//! A world is hosted under the id its authored document has, so a character's
//! world id finds the rooms it can stand in. A body's id is derived from the
//! character it belongs to, so a restart finds the same body without a table that
//! could disagree. A character returns to the room it last stood in, or to the
//! door when that room is no longer on the map. A world's metronome holds only a
//! weak handle back to the runtime, because a strong one would keep the daemon
//! from ever dropping.

use super::*;

impl Runtime {
    /// The runtime over `mind_handle`, its state kept under `data`. Refused
    /// when the mind's repository cannot be read for its operations' stores.
    pub fn new(mind_handle: Mind, data: &Path) -> anyhow::Result<Arc<Self>> {
        // **The benches are pointed at the mind this daemon was given**, before
        // any world is hosted. Derived from the handle rather than set on a
        // hosting call: a world reached through `host` and one reached through
        // `host_authored` are the same world, and making only the second of
        // them able to edit documents was a difference nothing in the fiction
        // justifies and nothing in the signature announces.
        let hosted = Worlds::new();
        if mind_handle.is_present() {
            hosted.set_bench_root(mind_handle.clone());
        }
        let ops = MindStores::for_mind(mind_handle.repo(), data)?;
        hosted.set_op_stores(Arc::clone(&ops));
        let rt = Arc::new(Self {
            scheduler: Arc::new(Scheduler::default()),
            progress: Arc::new(LoadProgress::new()),
            data: data.to_path_buf(),

            mind_handle,
            journals: Journals::new(),
            talks: Arc::new(Talks::new()),
            talk_changes: Changes::new(),
            days: Days::new(),
            mind_repos: OnceLock::new(),
            agents: OnceLock::new(),
            machines: RwLock::new(Vec::new()),
            relaunch: OnceLock::new(),
            mind_arrived: RwLock::new(None),
            // Read from disk: a document already in the substrate must not be
            // prefilled again, and at four seconds each that is the difference
            // between a twelve-second boot and half an hour of one.
            ledger: Arc::new(Ledger::open(data)),
            clock: RwLock::new(None),
            guardian_log: RwLock::new(None),
            substrate: RwLock::new(None),
            npcs: OnceLock::new(),
            minds: RwLock::new(None),
            persona: RwLock::new(None),
            mood_sink: RwLock::new(None),
            tokens: RwLock::new(None),
            effector: RwLock::new(None),
            hosted,
            interactions: crate::engine::interaction::Interactions::new(),
            bodies: Bindings::new(),
            metronomes: Mutex::new(BTreeMap::new()),
            places: Mutex::new(BTreeMap::new()),
            standing: Standing::new(false),
            compass_told: Mutex::new(HashMap::new()),
            chapters: Mutex::new(HashMap::new()),
            dreaming: Mutex::new(HashSet::new()),
            reflect_domain: AtomicUsize::new(0),
            feelings: RwLock::new(Vec::new()),
            idles: RwLock::new(HashMap::new()),
            reaches: RwLock::new(HashMap::new()),
            tools: Tools::default(),
            tool_secrets: RwLock::new(Arc::new(Secrets::empty())),
            cooldowns: crate::engine::cooldown::Cooldowns::new(),
            loop_guards: crate::engine::loopguard::LoopGuards::new(),
            stopping: AtomicBool::new(false),
            spawner: Handle::try_current().ok(),
            tasks: Mutex::new(BTreeMap::new()),
            started: Instant::now(),
            ops,
        });
        // A world run disconnected keeps its operations' stores in memory,
        // writing no event (§9.4, §10.3).
        let standing = Arc::downgrade(&rt);
        rt.ops.set_quiet(Box::new(move |world| {
            standing
                .upgrade()
                .is_some_and(|rt| rt.standing.role(world) == Role::Disconnected)
        }));
        // A conversation's lines and purposes go to its runner, never to the
        // mind's inbox (`docs/npc_conversations.md` §9).
        let talking = Arc::downgrade(&rt);
        rt.scheduler.set_talk(Box::new(move |npc, kind| {
            if let Some(rt) = talking.upgrade() {
                rt.talk_arrived(npc, kind);
            }
        }));
        rt.start_talking();
        Ok(rt)
    }

    /// Host every authored world that has rooms to stand in.
    ///
    /// Driven by the world registry rather than by what is on disk, so a map
    /// directory nothing authored is *not* silently hosted under an id no
    /// character can name. Each world that does have one is hosted under its own
    /// id, which is what lets a character's `world_id` find the places it can be.
    ///
    /// Reports every world it tried, and what came of it. A map that does not
    /// hold together is an authoring mistake to fix, not a reason for the rest
    /// of the cast to have nowhere to stand.
    pub fn host_authored<'a>(
        self: &Arc<Self>,
        authored: &Path,
        worlds: impl IntoIterator<Item = &'a str>,
    ) -> Vec<(String, anyhow::Result<Arc<Hosted>>)> {
        // The mind's maps when it has a mind: its files, at the commit the log
        // names. The data directory's folder only for a daemon with none.
        if let Some(files) = self.mind_files() {
            return worlds
                .into_iter()
                .filter(|id| files.is_dir(&format!("{MAPS}/{id}")))
                .map(|id| (id.to_string(), self.host_from_mind(id, &files)))
                .collect();
        }
        let maps = authored.join(MAPS);
        worlds
            .into_iter()
            .filter(|id| maps.join(id).is_dir())
            .map(|id| (id.to_string(), self.host(id, &maps.join(id))))
            .collect()
    }

    /// Load one world and start it moving.
    ///
    /// The metronome holds only a [`Weak`] handle back here. A strong one would
    /// be a cycle — the runtime owns the metronome and the metronome would own
    /// the runtime — and the daemon would never drop, which reads as a clean
    /// shutdown that never finishes.
    pub fn host(self: &Arc<Self>, id: &str, dir: &Path) -> anyhow::Result<Arc<Hosted>> {
        let world = self.hosted.load(id, dir)?;
        self.start_hosted(id, world)
    }

    /// Load one world from the mind's files — its map at `map/<id>` — and
    /// start it moving (`docs/substrate_git.md` §8.1).
    pub fn host_from_mind(
        self: &Arc<Self>,
        id: &str,
        mind: &MindFiles,
    ) -> anyhow::Result<Arc<Hosted>> {
        let world = Arc::new(Hosted::load_mind(id, mind, &format!("{MAPS}/{id}"))?);
        let world = self.hosted.keep(world);
        self.start_hosted(id, world)
    }

    /// A world just loaded: its saved state restored, and its metronome
    /// started.
    pub(super) fn start_hosted(
        self: &Arc<Self>,
        id: &str,
        world: Arc<Hosted>,
    ) -> anyhow::Result<Arc<Hosted>> {
        let Some(spawner) = self.spawner.clone() else {
            anyhow::bail!("no async runtime to beat `{id}`'s metronome on");
        };
        self.restore(&world);
        let back = Arc::downgrade(self);
        let name = id.to_string();
        let moving = world.clone();
        let beat = Metronome::start(&spawner, driver::EVERY, move || {
            let Some(rt) = back.upgrade() else {
                return;
            };
            if rt.stopping.load(Ordering::Relaxed) {
                return;
            }
            rt.beat(&moving);
        });
        self.metronomes.lock().unwrap().insert(name, beat);
        Ok(world)
    }

    /// Stop a world: its metronome, its minds' bodies, and the world itself.
    ///
    /// All three, in that order. Stopping the metronome first means nothing is
    /// mid-moment while the bindings go, and unbinding before releasing means
    /// no character is left pointing at a world that is no longer there.
    pub fn unhost(&self, id: &str) -> bool {
        if let Some(beat) = self.metronomes.lock().unwrap().remove(id) {
            beat.stop();
        }
        self.bodies.release_world(id);
        self.hosted.release(id)
    }

    /// Give a character a body in a hosted world, and set it to a working pace.
    ///
    /// A character with a body has somewhere to be and something in front of
    /// it, so it never goes as quiet as one that is only reacting — see
    /// [`Pace`]. It is also grounded at once rather than at the world's next
    /// moment: a character that can act before it has been told where it is
    /// would act blind.
    pub fn embody(&self, npc_id: u64, world: &str, body: &str, now_ms: u64) -> anyhow::Result<()> {
        let Some(hosted) = self.hosted.get(world) else {
            anyhow::bail!("no world `{world}` is running");
        };
        if hosted.read(|w| w.actor(body).is_none()) {
            anyhow::bail!("`{world}` has no body `{body}`");
        }
        self.bodies.bind(npc_id, world, body)?;
        self.scheduler.set_pace(npc_id, Pace::WORKING, now_ms);
        environment::push_one(&hosted, &self.scheduler, npc_id, body, &|id| {
            self.world_ms(id)
        });
        Ok(())
    }

    /// The body id a character acts through.
    ///
    /// Derived rather than stored, so a restart reconstructs the same
    /// correspondence without a table — the same reason a timeline id is
    /// derived. A body a character is not in is a body nothing can address.
    pub fn body_id(npc_id: u64) -> String {
        format!("npc-{npc_id}")
    }

    /// Put a new character into its world, at the way in.
    ///
    /// Called when a character is created, and again on a restart for every
    /// character whose world is hosted — both are the same operation, because
    /// entering a body that is already there is not an error and binding one
    /// that is already bound changes nothing.
    ///
    /// Silent when the character's world has no map: most worlds have none, and
    /// a character with lore and no body is a character, not a failure.
    pub fn embody_in_world(
        &self,
        npc_id: u64,
        world_id: &str,
        kind: Kind<'_>,
        name: &str,
        remembered: Option<&str>,
        now_ms: u64,
    ) -> anyhow::Result<bool> {
        let Some(hosted) = self.hosted.get(world_id) else {
            return Ok(false);
        };
        let Kind {
            home,
            unordered,
            attends,
            idle,
            reaches,
        } = kind;
        let body = Self::body_id(npc_id);
        let placed = hosted.with(|w| {
            if w.actor(&body).is_some() {
                return Ok(());
            }
            // **Back where it was, if the world still has that room.**
            //
            // The world itself is not persisted — who is standing where lives in
            // RAM and goes with the process — so without this every character
            // re-entered at the arrival door on every restart, however far it
            // had walked. Two Makers who had spent an hour finding each other
            // were returned to the front room as strangers while their own
            // transcripts said otherwise.
            //
            // A remembered room the map no longer has is not an error: maps are
            // authored and rooms get renamed. Fall through to the door, which is
            // exactly what a character whose room was demolished should do.
            let recalled = remembered
                .and_then(Where::parse)
                .filter(|at| w.map().node_at(at).is_some());
            let at = match recalled {
                Some(at) => at,
                // Where in the world this kind of character belongs. A world is
                // bigger than any one character's part of it — a Maker starts in
                // the vault and a soldier starts in a city, and they are the same
                // world — so the door is the home area's, not the world's.
                None => match home {
                    Some(home) => w.map().arrival_in(home).ok_or_else(|| {
                        anyhow::anyhow!("`{world_id}` has no `{home}` to arrive in")
                    })?,
                    None => w.map().arrival().ok_or_else(|| {
                        anyhow::anyhow!("world `{world_id}` has nowhere to arrive")
                    })?,
                },
            };
            w.enter(&body, name, at).map_err(anyhow::Error::from)
        });
        placed?;
        // Whether the world may order it, set every time it is embodied so the
        // personality's word is the one that stands — by the world's authority,
        // whose missions these are; a member following the world takes the mark
        // in with the rest of its state.
        if self.standing.writes(world_id) {
            hosted.with_sim(|s| {
                s.set_unordered(&body, unordered);
                s.set_attends(&body, attends);
            });
        }
        self.set_idle(npc_id, idle);
        self.set_reaches(npc_id, reaches);
        // **A handset on arrival, and a line on the roster.** Two halves of one
        // thing: carrying the phone is what makes the messaging acts reachable,
        // and being on the roster is what makes this character reachable *by*
        // them. Issued here rather than in the seed because the cast is not
        // known when a world is built — a character that arrives an hour later
        // has to be callable too.
        let display = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()));
        if let Some(display) = display {
            hosted.with_sim(|s| crate::sim::seed::issue_handset(s, &body, &display));
        }
        self.embody(npc_id, world_id, &body, now_ms)?;
        Ok(true)
    }

    /// Take a deleted character out of everything still holding it here: its
    /// running mind, what its body was waiting on, the questions it was part of,
    /// and its body, which leaves this member's copy of the world.
    ///
    /// One step, for both ways a deletion arrives — the member it was deleted
    /// on, and every other taking it in from the cast (`follow_cast`). Each
    /// member holds every body in its own copy of the world, so each takes its
    /// own copy's out; a body left behind stood in its room for ever, on the
    /// pulse page and in every percept of the characters around it.
    ///
    /// The questions are the world's to forget, so only its authority does.
    pub fn forget_character(&self, npc_id: u64) {
        // The task first: stopping it drops whatever turn it was awaiting, and
        // retiring wakes a task that was parked so it finds its inbox gone.
        self.stop_character(npc_id);
        self.scheduler.retire(npc_id);
        self.cooldowns.forget(npc_id);
        if let Some((hosted, body)) = self.body_of(npc_id) {
            if self.standing.writes(hosted.id()) {
                hosted.with_sim(|s| {
                    s.ledger.forget_questions(&body);
                    s.missions.forget(&body);
                    s.set_attends(&body, false);
                });
            }
            hosted.with(|w| {
                let _ = w.leave(&body);
            });
        }
        self.bodies.unbind(npc_id);
        self.set_idle(npc_id, None);
        self.set_reaches(npc_id, &[]);
        self.tools.forget(npc_id);
    }

    /// Take a character's body away, and let it settle back to reacting.
    pub fn disembody(&self, npc_id: u64, now_ms: u64) -> bool {
        if !self.bodies.unbind(npc_id) {
            return false;
        }
        self.scheduler.set_pace(npc_id, Pace::AMBIENT, now_ms);
        true
    }

    /// Hold a world still, or let it go again. `false` if no such world.
    pub fn hold_world(&self, id: &str, still: bool) -> bool {
        let beats = self.metronomes.lock().unwrap();
        let Some(beat) = beats.get(id) else {
            return false;
        };
        if still {
            beat.pause();
        } else {
            beat.resume();
        }
        true
    }

    /// How many moments each hosted world has taken. What a health check reads:
    /// a world whose count has stopped rising has lost its metronome.
    pub fn moments(&self) -> Vec<(String, u64, bool)> {
        self.metronomes
            .lock()
            .unwrap()
            .iter()
            .map(|(id, beat)| (id.clone(), beat.moments(), beat.is_paused()))
            .collect()
    }
}
