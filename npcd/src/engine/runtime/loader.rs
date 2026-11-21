//! Building the engine, substrate, cluster, mind and cast in the order each
//! depends on.
//!
//! The load runs as phases, and each is reported to the loading page under the
//! line it belongs to, so a slow start shows which phase owns the time. The engine
//! is published on the runtime before the layers are ingested, so routes that
//! need it answer during the ingest. The acts and the rest of the system prompt
//! are installed before any document is written, because a document written under
//! an incomplete prompt captures a signature no character will think under. A
//! mind that arrives after start is taken in before the schema is read, so nothing
//! is built from a mind that was not yet there.

use super::*;

pub(super) fn load(
    rt: &Arc<Runtime>,
    plan: &LoadPlan,
    rt_handle: Option<Handle>,
) -> anyhow::Result<()> {
    let p = &rt.progress;
    let _ = rt.relaunch.set(Arc::clone(&plan.stopping));
    let spec = model::spec();

    // ── the model ──────────────────────────────────────────────────────────
    p.set_step(LoadStep::Model);
    p.set_detail(format!("{} {}", spec.name, spec.quant));
    tracing::info!(
        "loading {} {} ({:.1} GB) from {}",
        spec.name,
        spec.quant,
        spec.bytes as f64 / 1e9,
        spec.repo
    );

    let device = candle::Device::new_cuda(0)
        .map_err(|e| anyhow::anyhow!("no CUDA device: {e}. npcd needs a GPU to run an engine."))?;
    // A prefill forward carries 8 192 tokens here, not the 2 048 default.
    //
    // A wave's fixed cost is paid per slab regardless of width. The layer
    // ingest's documents average ~1 900 tokens, so at the default budget every
    // document was its own slab paying the whole fixed sweep — measured 1.9 s a
    // document with the GPU pinned at 100%, which is a 7× gap to the
    // batched-forward gate's rate for work that is pure prefill.
    //
    // The default is conservative because interactive serving rarely has more
    // than a turn queued and a starved forward wastes the same fixed cost on
    // fewer tokens. This daemon is the other case: the ingest queues a windowful
    // of documents at a time, and a cast of characters ticking queues many
    // small prefills at once. Both reliably keep a wide forward fed.
    // The co-resident models, if this deployment configured any. Read before the
    // engine is built and **before the long checkpoint load**, so a wrong path
    // is a refusal at second zero rather than a five-minute load followed by
    // one. A daemon with no `guests.yaml` gets an empty registry, which costs
    // one atomic load per scheduler pass and nothing else.
    let guests = crate::guests::load(&rt.data).map_err(|e| anyhow::anyhow!("guests.yaml: {e}"))?;
    if !guests.is_empty() {
        tracing::info!(
            "co-resident guests configured: {:?} — normal inference stops between waves to \
             serve them",
            guests.configured()
        );
    }

    // **The engine adopts the daemon's substrate; it does not open one.**
    //
    // The character registry opened `--data/substrate/` at startup and goes on
    // appending character records to it. One `substrate/` admits exactly one
    // writable handle per process — the log is opened read-write and unlocked,
    // so a second is a second append cursor and a second record index, and a
    // compaction carries forward only what its own handle walked. This daemon
    // ran that way and lost every character created while it was up.
    //
    // Naming a path here instead would also revive an older failure: unset, the
    // conversation layer falls back to the *process working directory*, so a
    // daemon launched from the repo root wrote its redo log to
    // `candle/substrate` while the character store sat correctly under
    // `--data`. The storage view reports on `--data` and showed 65 MB
    // while the real log reached **190 GB** across a morning of re-ingests. It
    // filled the disk, and surfaced as a linker error.
    //
    // A missing handle is a refusal rather than a fallback: both failures above
    // are silent, and neither is worth risking to save a startup error.
    let shared = rt.substrate.read().unwrap().clone().ok_or_else(|| {
        anyhow::anyhow!(
            "the engine was started before the substrate was handed to it — call \
             `Runtime::set_substrate` with `Npcs::substrate()` first"
        )
    })?;
    // The mind's real projection, parsed once. Everything written from here on
    // is written under it — which is what gives a document something to gather
    // from and therefore a provenance signature. `None` falls back to a
    // synthetic schema that cannot gather, and says so.
    //
    // Parsed before the engine is built, because the engine takes each layer's
    // corrupt-turn policy from it and applies them in its startup reload. A
    // mind pending at start has none to give yet: both are read again once it
    // arrives, below.
    let mut mind_files = rt.mind_files();
    let mut projection = schema::build(mind_files.as_deref(), "battle-cities");
    if projection.is_none() && mind_files.is_some() {
        tracing::warn!(
            "no usable projection schema — documents will be written but will produce \
             no provenance signatures, so nothing will gather them"
        );
    }
    let corrupt_turn = projection
        .as_ref()
        .map(|p| schema::corrupt_turn_policies(&p.builder))
        .unwrap_or_default();
    // The model pack, through the host's download — the same resolution every
    // tenant uses, so the pack built and the model loaded are one. Its fetches
    // are async and the load is not: a local current-thread runtime drives them
    // and is dropped before the load, since a nested `block_on` inside the
    // ambient runtime would panic. Each status line reaches the loading page
    // from a thread of its own, which ends when the resolution drops its sender.
    let pack = {
        let download = RuntimeBuilder::new_current_thread().enable_all().build()?;
        let (status, mut said) = watch::channel(String::new());
        let resolved = std::thread::scope(|s| {
            s.spawn(|| {
                while futures::executor::block_on(said.changed()).is_ok() {
                    p.set_detail(said.borrow().clone());
                }
            });
            let got = ensure_model_pack(&MODEL, &device, &download, &status);
            drop(status);
            got
        });
        drop(download);
        resolved?
    };
    let mut builder = engine_asks(MODEL.builder())
        .model_path(pack)
        .guests(guests)
        .corrupt_turn_policies(corrupt_turn)
        .substrate(shared.clone());
    // The checkpoint's own dialect and sampling. Captured before the builder is
    // consumed, and handed to every conversation — a second opinion about either
    // would run the model outside the settings it was tuned under.
    // Sampling is the checkpoint's, taken from `SamplingConfig::for_gguf_architecture`.
    //
    // **This daemon used to state Qwen3.5's published numbers here, and that was a
    // workaround for a bug one layer down.** The auto-detect reported
    // `temp=0.7, top_k=0, top_p=0.9`, which reads like bad checkpoint metadata and is
    // not: it is `for_gguf_architecture`'s *unknown architecture* fallback, verbatim.
    // `qwen35` was missing from that table while the whole lineage loaded through it.
    //
    // Restating the four visible numbers here fixed what could be seen and hid what
    // could not. `sampling()` replaces the config wholesale, and the literal was built
    // from `SamplingConfig::default()` — so every steering field came back zeroed,
    // including `force_segment_close_after`, the hard cap that rewrites the next token
    // to `</think>`. A checkpoint that stops closing its own reasoning block then runs
    // to `max_response_tokens` and the caller discards the entire decode as
    // all-reasoning: an empty document, reported as a success. A stock checkpoint hides
    // it by always self-closing; three abliterated ones did not, and were blamed for it.
    //
    // The numbers themselves were right and now live where they belong, beside the
    // architecture that publishes them.
    let conv_config = builder.conversation_config();

    // Back into the ambient runtime — after the download's own runtime is gone
    // — so anything spawned from here has a context.
    let _guard = rt_handle.as_ref().map(|h| h.enter());

    // The engine, from the process's slot: built here with every tenant's
    // requests merged, or this substrate attached to the one a higher-ranked
    // tenant built.
    let engine = plan.engine.ticket().engine_with_progress(
        &mut builder,
        &device,
        // Each phase of the load names the line under the bar and what its
        // counts are, so the part of startup that takes the time says which
        // part it is.
        Some(&|phase: LoadPhase, done, total| {
            p.set_detail(phase.label());
            p.set_unit(phase.unit());
            p.set_progress(done as u64, total as u64);
        }),
        &|tenants| p.set_waiting_for(tenants),
    )?;

    // ── the substrate ──────────────────────────────────────────────────────
    //
    // The engine's own construction replays the redo log, so by the time we
    // have an engine the substrate is already open. The step is still shown,
    // because on a large log it is where the time goes and an operator watching
    // a slow start needs to see which phase owns it.
    p.set_step(LoadStep::Substrate);
    let (done, total, _) = engine.substrate_reload_status().snapshot();
    p.set_progress(done as u64, total as u64);
    tracing::info!("substrate: {done} of {total} turns restored");

    // **The engine goes on the runtime here, not at the end of startup.**
    //
    // It used to be published only after every layer had been ingested and
    // every character woken, so on a real mind — twenty minutes of ingest —
    // every engine-backed route answered 503 for the whole of it. That is
    // exactly the window in which somebody watching a slow start wants to ask
    // the daemon something: what the guests are, how the ingest is going,
    // whether the model is the one they expected.
    //
    // Safe because the engine is shared rather than held (see
    // [`SharedEngine`]): the ingest creating a window's conversations does not
    // stand in front of a guest submission. The routes that need the *cast*
    // still report an empty one until the wake phase, which is true rather
    // than a refusal.
    let engine: SharedEngine = Arc::new(engine);
    let minds_over = |files: Option<&MindFiles>| {
        // Retention is read from the mind's own schema, so a deployment that
        // wants whole transcripts omits the block and nothing changes for it.
        let keep_turns = crate::engine::schema::turn_retention(files);
        // The reflection's two user turns, from the same schema. Read here
        // rather than per request: they are authored content and a reflection
        // sends them verbatim, so re-reading the file mid-run would let two
        // reflections in the same minute be asked different questions.
        let reflection = crate::engine::schema::reflection(files);
        Arc::new(
            Minds::new(Arc::clone(&engine), conv_config.clone())
                .keeping_turns(keep_turns)
                .asking(reflection),
        )
    };
    // Published now unless the mind is still to be pulled, when they wait for
    // it below (the routes report no engine meanwhile, which is true).
    if plan.mind_repo.is_none() || mind_files.is_some() {
        *rt.minds.write().unwrap() = Some(minds_over(mind_files.as_deref()));
    }

    // ── the cluster ────────────────────────────────────────────────────────
    //
    // With `--cluster`, joined now that the substrate — and the node it carries
    // — is open, and before the load writes anything the group would share
    // (`docs/replicated_substrate.md` §7.6). A gateway that refuses this build
    // for good holds the load here, the reason on the loading page, until the
    // process stops.
    if let Some(join) = &plan.cluster {
        p.set_step(LoadStep::Connecting);
        let (node, model, writer, store) = {
            let e = &engine;
            let conv = e.conversation();
            let reloader = e.reloader();
            let store = ConversationStore::new(conv.clone(), e.kv_layers(), move || {
                reloader.reload_and_wait()
            });
            (conv.node(), model_id(e.model_key()), conv.clone(), store)
        };
        let report = |views: &[ClusterView]| {
            p.set_progress(admitted(views), views.len() as u64);
            p.set_detail(connecting_detail(views));
        };
        let stopping = || rt.stopping();
        let link = join.connect(
            node,
            Some(model),
            store.replication()?,
            move || writer.last_oseq(),
            stopping,
            report,
        );
        if let Link::Stopped(_) = link {
            join.hold(stopping, report);
        }
        if link == Link::Joined {
            // Ready only once caught up on what the group wrote (§7.6).
            p.set_step(LoadStep::CatchUp);
            join.catch_up(stopping, |catchups| {
                let (done, total) = catchup_progress(catchups);
                p.set_progress(done, total);
                p.set_detail(catchup_detail(catchups));
            });
        }
        if rt.stopping() {
            anyhow::bail!("the process stopped while connecting to the cluster");
        }
        // The worlds' leases, their pulses and the characters lent here are
        // followed from now on (§9.2–§9.4).
        if let Some(member) = join.link().member() {
            rt.join_cluster(Arc::clone(member));
        }
    }

    // The mind repository's fetches and pushes go through an agent holding it —
    // the cluster's, reaching this process's own while cut off (§10.3, §11.5),
    // or with no cluster the one in this process alone. Its packs and branches
    // are records in the substrate, loaded whole now — pulled through an agent
    // first when no member has — and followed from then on
    // (`docs/substrate_git.md` §8, §9).
    let mut read_again: Option<Arrived> = None;
    if let Some(source) = &plan.mind_repo {
        let handle = rt_handle
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no async runtime for the agents' jobs to run on"))?;
        // A member of a cluster never acts as a daemon alone: one whose member
        // never started, or has no replication to hold the import lease
        // through, fails here rather than importing beside the others.
        let (agents, replicator) = match &plan.cluster {
            Some(join) => {
                let member = join.link().member().ok_or_else(|| {
                    anyhow::anyhow!(
                        "the mind is kept through the cluster, whose member never started"
                    )
                })?;
                let replicator = member.replicator().cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "the mind's import lease is held through the cluster's replication, and \
                         this member has none"
                    )
                })?;
                (
                    Agents::new(Arc::clone(member), join.in_process_agent()),
                    Some(replicator),
                )
            }
            None => {
                let jobs = plan.agent.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "the mind is read through an agent, and there is none: give the process \
                         an `agent:` file naming a checkout of `{}`",
                        source.name
                    )
                })?;
                (Agents::in_process(jobs), None)
            }
        };
        // The same agent client the mind's git jobs go through also carries a
        // device-bridge job to an in-network agent (`docs/zen_moon_mind.md`
        // §15), so an NPC's effector verb and the operator route can reach it.
        rt.set_agents(agents.clone());
        rt.ops.set_pusher(AgentPush::shared(
            agents.clone(),
            handle.clone(),
            &source.name,
        ));
        rt.ops.set_fetcher(AgentFetch::shared(
            agents.clone(),
            handle.clone(),
            &source.name,
            Arc::new(AtomicUsize::new(0)),
        ));
        let access = Access {
            agents,
            runtime: handle,
            replicator,
        };
        if load_mind(rt, source, &shared, access)? {
            // **Taken in here, not restarted onto.** In one process with zend
            // a restart stops zend too, mid-load. Nothing has been written
            // under the schema yet — the acts, the identities and the layers
            // all come after this — so what was built from no mind is built
            // again from the one that arrived: the projection, each layer's
            // corrupt-turn policy, the minds, and the authored state the
            // tenant read before.
            mind_files = rt.mind_files();
            projection = schema::build(mind_files.as_deref(), "battle-cities");
            if let Some(p) = &projection {
                let conv = engine.conversation();
                for (layer, policy) in schema::corrupt_turn_policies(&p.builder) {
                    conv.set_layer_corrupt_turn_policy(layer, policy);
                }
            }
            if projection.is_none() {
                tracing::warn!(
                    "no usable projection schema in the mind that arrived — documents will be \
                     written but will produce no provenance signatures, so nothing will gather \
                     them"
                );
            }
            read_again = rt.read_again().transpose()?;
        }
    }
    // A mind pending at start publishes its minds only now, over the mind that
    // arrived: one published before the pull, and replaced here, would orphan
    // every session a guest or pulse opened on it meanwhile.
    if rt.minds.read().unwrap().is_none() {
        *rt.minds.write().unwrap() = Some(minds_over(mind_files.as_deref()));
    }
    // What the tenant handed the load, or what it read again from the mind
    // that arrived.
    let cast = read_again.as_ref().map_or(&plan.cast, |a| &a.cast);
    let characters = read_again
        .as_ref()
        .map_or(&plan.characters, |a| &a.characters);
    let authored = read_again.as_ref().map_or(&plan.authored, |a| &a.authored);

    // ── the mind's layers ──────────────────────────────────────────────────
    //
    // Every layer that declares an `ingest_unit:` and has a folder becomes turns
    // on a conversation of its own — one per layer for shared ground, and one
    // per *character* where the layer is theirs. That split is not cosmetic: a
    // life story is per-character, and three biographies in one shared
    // conversation is every character able to recall every other character's
    // childhood.
    //
    // In dependency order, lowest first, memory last — see `ingest::ORDER`.
    //
    // This phase used to walk the tree, count what it found and write nothing,
    // under a label that said "Ingesting mind layers" over a world whose
    // sixty-six documents no character could ever reach.

    // ── the acts ───────────────────────────────────────────────────────────
    //
    // Before the layers, deliberately. Every layer frames on the shared system
    // prompt and the acts are part of it, so a document prefilled while they are
    // still absent captures its KV — and the wide-Q signature the gather matches
    // against — under a prompt no character will ever think under. Nothing
    // would fail; retrieval would simply be worse than it should be, for the
    // life of that substrate.
    //
    // **Every act, installed once; each turn shows the ones it can take.** The
    // world's catalog and the reflection's two answers go into one collection,
    // and an acting turn names its members from the same `specs_within` its
    // grammar is compiled from — so the list a character reads and the mask it
    // decodes under are one computation and cannot disagree.
    p.set_step(LoadStep::Tools);
    let acts: Vec<&Tool> = tools::CATALOG
        .iter()
        .chain(ORDERED_ACTS)
        .chain(reflect::ASKED)
        .collect();
    let total = acts.len() as u64;
    p.set_progress(0, total);
    if let Some(built) = projection.as_mut() {
        let installed = tools::install(&mut built.builder, acts, |n, t| {
            p.set_detail(t.name);
            p.set_progress(n as u64, total);
        })?;
        tracing::info!(
            "tools: {installed} installed into the prompt, each shown on the turns that can \
             take it"
        );
    }

    // ── what the acts are ranked by ────────────────────────────────────────
    //
    // A turn shows the acts its room offers that provenance ranks highest, and
    // an act is ranked by the turns that carry its tag. Those are written here,
    // from each act's authored examples and questions, by the phase zend runs
    // over its tool files. On its own hidden layer, so it reads nothing of the
    // mind and nothing of the mind reads it but the `tools` collection.
    p.set_step(LoadStep::Calibrating);
    if projection.is_some() {
        let acts: Vec<&Tool> = tools::CATALOG
            .iter()
            .chain(ORDERED_ACTS)
            .chain(reflect::ASKED)
            .collect();
        let defs = calibrate::defs(&acts, &conv_config.dialect);
        let report = calibration::run(
            &calibration::Phase {
                engine: &engine,
                config: &conv_config,
                frame: &calibrate::frame(&conv_config.dialect),
                progress: &|done, total| p.set_progress(done, total),
                stopped: &|| rt.stopping(),
            },
            &defs,
        )?;
        if report.stopped {
            anyhow::bail!("the process stopped while calibrating the acts");
        }
    }

    // ── the prompt a character thinks under ────────────────────────────────
    //
    // **Everything goes into the schema before anything is written under it.**
    // The ordering rule `LoadStep::Tools` states applies to all of it: a layer
    // document prefilled while the system prompt is still incomplete captures
    // its signature — and the wide-Q the gather matches against — under a
    // prompt no character will ever think under.
    //
    // Who everybody is. A collection, so each member seals once and is selected
    // per turn: one copy of the vault for the world rather than one per Maker
    // standing in it.
    let identities = match projection.as_mut() {
        Some(p) => {
            // Two members per building of every world: one for a character
            // standing in it, pinned to the one its body is in, and one for a
            // mind with no body that is the place itself — see
            // [`Runtime::building_of`] and [`Runtime::place_persona`].
            let places: BTreeMap<String, String> = rt
                .hosted
                .ids()
                .into_iter()
                .filter_map(|id| rt.hosted.get(&id).map(|hosted| (id, hosted)))
                .flat_map(|(id, hosted)| {
                    rt.places_of(&hosted)
                        .into_iter()
                        .flat_map(move |(part, text)| {
                            let key = identity::building_key(&id, &part);
                            [
                                (identity::whole_key(&key), prompt::being(&text)),
                                (key, prompt::building(&text)),
                            ]
                        })
                })
                .collect();
            identity::install(&mut p.builder, authored, &places)?
        }
        None => identity::Installed::default(),
    };
    // The talk templates a conversation's words are shaped by, and the
    // `response` collection made one each turn offers from.
    let templates = match projection.as_mut() {
        Some(p) => templates::install(&mut p.builder, mind_files.as_deref())?,
        None => templates::Installed::default(),
    };

    if let (Some(p), Some(minds)) = (projection.as_ref(), rt.minds.read().unwrap().as_ref()) {
        minds.set_projection(Projected {
            prompt: p.prelude.clone(),
            builder: p.builder.clone(),
            layer: p.layer,
            group: p.group,
            talk: p.talk,
            identities: identities.clone(),
            templates,
            // After everything is installed, so the acts and identities are in
            // it: a conversation written under a different set is superseded on
            // its next open rather than rejoined.
            frame: frame_fingerprint(&schema::frame(&p.builder)),
        });
    }

    // ── a wiped substrate, when asked for ──────────────────────────────────
    //
    // Before the layers, because the ingest decides what to write from the
    // ledger: every conversation goes, the ledger is emptied with them, and the
    // mind and every life are then written afresh on this same start.
    if plan.wipe_conversations {
        let retired = wipe_conversations(&engine);
        rt.ledger.forget_all();
        tracing::info!(
            "substrate: {retired} conversation(s) retired and the ingest ledger cleared — the \
             mind and every life re-ingest now; the cast itself is kept"
        );
    }

    p.set_step(LoadStep::Layers);
    // The writers below await their prefill drains, and this is a plain
    // loader thread — so it drives them itself. `futures::executor::block_on`
    // rather than the ambient runtime's `block_on`, because these futures wake
    // on the engine's own channels and need no timer or I/O driver, and a
    // current-thread executor here cannot collide with the ambient runtime
    // this thread has entered.
    match &mind_files {
        Some(files) => {
            // Every document path is a mind path, relative to the mind's root.
            let dir: &Path = Path::new("");
            // ── what the substrate already holds is not owed again ────────
            //
            // First, so everything below reads one ledger: the documents
            // another member wrote — or this one, before its ledger was lost —
            // are already in the substrate at a recorded version, and writing
            // them again is how every document came to be held twice.
            let seeded = mind_record::seed_ledger(&engine, &rt.ledger);
            if seeded > 0 {
                tracing::info!(
                    "mind: {seeded} document(s) already in the substrate, taken into the \
                     ingest ledger"
                );
            }
            // ── what has left the record leaves memory ────────────────────
            //
            // Before anything is ingested: a document gone from disk is
            // retired from the substrate, and the ledger forgets it. The mind
            // folder holds only published work — an operation's drafts are in
            // its store until its publish step lands them (§10.3) — so
            // everything on disk is memory's to take.
            //
            // A member whose ingest is off writes none of this: another
            // member's retirements and documents reach it by replication.
            if !plan.ingest {
                tracing::info!(
                    "ingest: off — the mind's documents and lives are written by another member \
                     and read here as they replicate"
                );
            }
            let out_of_record: Vec<PathBuf> = if plan.ingest {
                rt.ledger.missing(files)
            } else {
                Vec::new()
            };
            if !out_of_record.is_empty() {
                let e = &engine;
                let retired: usize = out_of_record
                    .iter()
                    .filter_map(|p| mind_record::rel(dir, p))
                    .map(|r| mind_record::retire(e, &r))
                    .sum();
                for p in &out_of_record {
                    rt.ledger.forget(p);
                }
                rt.ledger.flush();
                tracing::info!(
                    "mind: {} document(s) out of the record — {retired} conversation(s) retired",
                    out_of_record.len()
                );
            }
            let mind_root: &MindFiles = files;

            let units = crate::projection::ingest_units(&rt.mind_handle);
            let sources = ingest::sources(files, &units, characters);
            if sources.is_empty() {
                tracing::info!("mind: no layer declares content to ingest");
            }
            let mut shared_groups: Vec<(String, GroupId)> = Vec::new();
            for source in &sources {
                // A shared layer's documents are written into its own group,
                // and a projection targeting that layer reads only the document
                // it is writing — as zend writes a file onto its ingest layer.
                if let Some((p, (layer, group))) = projection
                    .as_ref()
                    .and_then(|p| Some((p, shared_target(p, source)?)))
                {
                    engine.mark_layer_append_only(layer);
                    let moved = if plan.ingest {
                        retire_misplaced(&engine, p.group, source, &rt.ledger)
                    } else {
                        0
                    };
                    if moved > 0 {
                        tracing::info!(
                            "layer {}: {moved} document(s) were in the live group, where no \
                             character could reach them — retired, to be written into the \
                             layer's own group",
                            source.label()
                        );
                    }
                    shared_groups.push((source.name.clone(), group));
                }
                if !plan.ingest {
                    continue;
                }
                let (turns, mut report) = ingest::pending(files, source, &rt.ledger)?;
                let total = turns.len();
                ingest::announce(p, source, 0, total);

                if total > 0 {
                    let failed = block_on(ingest_layer(
                        &engine,
                        Record {
                            mind: dir,
                            ledger: &rt.ledger,
                        },
                        &conv_config,
                        projection.as_ref(),
                        source,
                        &turns,
                        p,
                    ))?;
                    report.written -= failed;
                    report.failed += failed;
                }
                // A final flush at the layer boundary. The windows inside
                // `ingest_layer` have been flushing as they seal — see the note
                // there — so this only catches a layer that ingested nothing.
                rt.ledger.flush();
                tracing::info!(
                    "layer {}: {} written, {} unchanged, {} failed ({})",
                    source.label(),
                    report.written,
                    report.unchanged,
                    report.failed,
                    source.unit
                );
            }

            // The shared layers are not warmed by self-match. Every document is
            // one exchange, so a group's documents compete as one pool, and a
            // document probed by its own signature is a ceiling no query
            // reaches (`docs/provenance_score_normalization.md` §2). Their hit
            // levels are learned from the characters' turns, at each seal.

            // ── the lives ──────────────────────────────────────────────────
            //
            // After the world, because a life is lived against it: an episode's
            // provenance is captured relative to what is already in the
            // substrate, so the world has to be there first for the signatures
            // to attach to anything.
            //
            // Each character's episodes run in date order, and each one's
            // `<tool_call>` blocks are executed as it lands — so a belief is
            // formed at the point in the life where it formed, by the same
            // mechanism a belief would be formed at run time.
            let lives = if plan.ingest {
                life::lives(files, characters)
            } else {
                Vec::new()
            };
            for (who, dir) in lives {
                let (episodes, rejected) = life::episodes(files, &dir, &who);
                for r in &rejected {
                    // A life with a hole in it is still a life, but the author
                    // has to be told which document fell out.
                    tracing::warn!("life {who}: {}", r.message());
                }
                if episodes.is_empty() {
                    continue;
                }
                p.set_unit("episodes");
                p.set_detail(format!("life · {who}"));
                p.set_progress(0, episodes.len() as u64);
                match block_on(run_life(
                    &engine,
                    mind_root,
                    &conv_config,
                    projection.as_ref(),
                    &who,
                    &episodes,
                    &rt.ledger,
                    p,
                )) {
                    Ok(lived) => tracing::info!(
                        "life {who}: {} episode(s), {} belief(s), {} relationship(s), \
                         {} intention(s){}",
                        lived.episodes,
                        lived.beliefs,
                        lived.relationships,
                        lived.intents,
                        if lived.rejected == 0 {
                            String::new()
                        } else {
                            format!(" — {} call(s) refused", lived.rejected)
                        }
                    ),
                    Err(e) => tracing::warn!("life {who}: not run — {e:#}"),
                }
                rt.ledger.flush();
            }

            // ── what is read and never continued keeps no recurrent state ──
            //
            // After the documents and the lives, so everything they wrote this
            // boot is marked. A store from before the mark existed kept a whole
            // recurrent state (~112 MiB) as the tail of every document and
            // dream — hundreds of gigabytes for a mind of a few thousand
            // documents; this retires those tails, and compaction reclaims them.
            {
                // A document held twice keeps one copy — the duplicates' tails
                // then go with their timelines rather than being retired one
                // by one below.
                let copies = mind_record::retire_duplicate_documents(&engine);
                if copies > 0 {
                    tracing::info!(
                        copies,
                        "retired the extra copies of documents written more than once"
                    );
                }
                let retired = mind_record::retire_unresumed_snapshots(&engine);
                if retired > 0 {
                    tracing::info!(
                        retired,
                        "retired the recurrent snapshots of documents and dreams, which are \
                         read and never continued"
                    );
                }
            }

            // ── what each bench draws on ───────────────────────────────────
            //
            // After the lives, which are written against the whole of the
            // world: from here on a shared group offers a conversation only what
            // the bearing it stands at draws on. See `bearings`.
            match bearings::Bearings::read(files) {
                Ok(Some(b)) if !shared_groups.is_empty() => {
                    let e = &engine;
                    let shelves: Vec<bearings::Shelf> = shared_groups
                        .iter()
                        .map(|(name, group)| bearings::Shelf::read(e, name, *group))
                        .collect();
                    let scopes = bearings::Scopes::new(b, shelves);
                    for group in scopes.groups() {
                        e.mark_group_scoped(group);
                    }
                    if let Some(minds) = rt.minds.read().unwrap().as_ref() {
                        minds.set_scopes(scopes);
                    }
                    tracing::info!(
                        "bearings: {} shared layer(s) drawn on by what stands where a character is",
                        shared_groups.len()
                    );
                }
                Ok(_) => tracing::info!(
                    "bearings: none declared — every shared layer is drawn on everywhere"
                ),
                Err(e) => tracing::warn!(
                    "bearings: {} could not be read ({e:#}) — every shared layer is drawn on \
                     everywhere",
                    bearings::FILE
                ),
            }
        }
        None => tracing::info!("mind: none — nothing to ingest"),
    }

    // ── a clean slate, when asked for ──────────────────────────────────────
    //
    // Before the cast wakes, because a character opens its conversation on its
    // first thought: retired here, nothing is left for it to rejoin, and the
    // open it makes next is a fresh one.
    if plan.forget_conversations {
        if let Some(minds) = rt.minds.read().unwrap().as_ref() {
            let retired: usize = cast
                .iter()
                .map(|c| minds.forget_conversations(c.npc_id))
                .sum();
            tracing::info!(
                "conversations: {retired} retired across {} character(s) — every character \
                 starts a fresh conversation",
                cast.len()
            );
        }
    }
    if plan.forget_dreams {
        if let Some(minds) = rt.minds.read().unwrap().as_ref() {
            let retired: usize = cast.iter().map(|c| minds.forget_dreams(c.npc_id)).sum();
            tracing::info!(
                "dreams: {retired} retired across {} character(s) — every character starts \
                 with nothing dreamt",
                cast.len()
            );
        }
    }
    if plan.forget_journal {
        if let Some(minds) = rt.minds.read().unwrap().as_ref() {
            let retired: usize = cast.iter().map(|c| minds.forget_journal(c.npc_id)).sum();
            tracing::info!(
                "journal: {retired} conversation record(s) emptied across {} character(s) — \
                 every character starts with an empty journal",
                cast.len()
            );
        }
    }

    // ── the cast ───────────────────────────────────────────────────────────
    p.set_step(LoadStep::Waking);
    p.set_progress(0, cast.len() as u64);
    let now = 0;
    let mut embodied = 0;
    let mut recalled = 0;
    let mut felt = 0;
    for (i, casting) in cast.iter().enumerate() {
        let id = casting.npc_id;
        let world = &casting.world_id;
        let world_ms = rt.world_ms(id);
        // **Every character has a body in every member's copy of its world;
        // only the ones this member is lent think here** (§9.3). The rest
        // are woken where their lease is held, and here if it moves.
        let runs = rt.standing.runs(id);
        if runs {
            rt.scheduler.wake(id, now, world_ms);
        }
        // Back into the body it had. Entering one that is already there is not
        // an error, so this is the same call the create path makes and the two
        // do not have to agree about anything beyond the id.
        let persona = rt.persona_of(id);
        let name = persona
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| Runtime::body_id(id));
        // Where the world's records last saw its body standing — or, in a
        // world taken in this boot from the objects kept before world records,
        // where its own record said, once.
        let remembered = place_of(&rt.standing.written(world).records, &Runtime::body_id(id))
            .or_else(|| {
                rt.standing
                    .from_legacy(world)
                    .then(|| casting.at.clone())
                    .flatten()
            });
        // Where its personality belongs, and whether the world may order it —
        // what every path that embodies a character reads from the persona. The
        // room it was last in is consulted first, so the home is only reached
        // for one that has none.
        let kind = persona.as_ref().map(OwnedPersona::kind).unwrap_or_default();
        match rt.embody_in_world(id, world, kind, &name, remembered.as_deref(), now) {
            Ok(true) => {
                embodied += 1;
                if remembered.is_some() {
                    recalled += 1;
                }
            }
            Ok(false) => {}
            Err(e) => tracing::error!("npc {id}: no body in `{world}` — {e:#}"),
        }
        if !runs {
            p.set_progress(i as u64 + 1, cast.len() as u64);
            continue;
        }
        // **And back into the register it was in.** A mood is what a character
        // *is* between one thought and the next, and it lives in the window —
        // which a restart empties. Without this a character that had spent the
        // afternoon getting angrier came back with no sign of it, while its
        // conversation history said otherwise, which is the same wrong as
        // returning it to the arrival door.
        //
        // Delivered as an arrival rather than written into the prompt: the
        // prompt is rendered once when a conversation opens, so a register in
        // it would go on asserting an afternoon's mood for ever.
        if let Some(mood) = &casting.mood {
            felt += 1;
            rt.scheduler.deliver(
                id,
                world_ms,
                Salience::IDLE,
                EventKind::Description {
                    text: format!(
                        "What you were feeling, when you last stopped to notice, was {mood}."
                    ),
                },
            );
        }
        p.set_progress(i as u64 + 1, cast.len() as u64);
    }
    tracing::info!(
        "cast: {} character(s) awake, {embodied} standing in a world \
         ({recalled} back where they were, {felt} back in the register they were in)",
        rt.scheduler.population()
    );

    // The engine was published at the substrate step — see there. What changes
    // here is only that the load is over.
    p.mark_ready();
    tracing::info!("engine ready in {:?}", rt.uptime());

    // The command table's generator, once there is a cast to set work for, and
    // the mind kept at the record by the substrate.
    if rt_handle.is_some() {
        mission_gen::run::spawn(Arc::clone(rt));
    }

    supervise(rt);
    Ok(())
}
