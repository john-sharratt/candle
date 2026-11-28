//! Brings npcd up on the host: reads the mind and the cast, builds the engine,
//! and starts the routes while the model loads.
//!
//! The mind is read through an agent, either the cluster's or the one in this
//! process, and with no cluster this daemon is every world's authority. The cast
//! is rebuilt from the substrate's redo log in one read, after which every request
//! is answered from memory. A mind not yet in the substrate is read again when it
//! arrives, so the worlds, personalities and libraries reflect what was taken in,
//! and a schema that cannot be read is fatal because characters would have no
//! registers to decode under. The runtime is built before the state and given its
//! resolvers afterwards, which breaks the cycle between them. The loader runs on
//! its own thread so the server can bind while the model is still fetched, and
//! the loading screen stays reachable during that wait. Reloads follow the mind
//! folder whenever it moves to a newer record.

use zend_tools::state::launch as secrets;

use super::*;

pub(super) async fn start(t: NpcdTenant, ctx: TenantContext) -> anyhow::Result<Running> {
    let NpcdTenant {
        args,
        schema,
        data,
        fell_back,
        repo,
    } = t;
    // The mind is read through an agent: the cluster's, or the one in this
    // process (`docs/substrate_git.md` §9).
    if repo.is_some() && ctx.cluster.is_none() && ctx.agent.is_none() {
        anyhow::bail!(
            "--mind-repo is read through an agent, and there is none: give the process an \
             `agent:` file naming a checkout of it, or a --cluster"
        );
    }
    // With no cluster this daemon is every world's authority and runs every
    // character: there is no other member for them to run on.
    if !args.characters && ctx.cluster.is_none() {
        anyhow::bail!(
            "--no-characters keeps a member out of the leases its cluster lends, and there is \
             no cluster: give it one, or let it run characters"
        );
    }
    if !args.ingest && ctx.cluster.is_none() {
        anyhow::bail!(
            "--no-ingest leaves the mind's writing to another member of the cluster, and there \
             is no cluster: give it one, or let it ingest"
        );
    }
    match &repo {
        // A repository an agent holds: read from the substrate once it opens.
        Some(r) => tracing::info!(
            "the mind is the repository `{}` on `{}`, read from the substrate",
            r.name,
            r.branch
        ),
        // A warning, not a note: a daemon with no mind starts, serves the
        // console and looks healthy, while every character in it has no world,
        // no personality and no libraries to think with.
        None => tracing::warn!(
            "no --mind-repo given: running on the {} schema's layers with no content \
             libraries, no worlds and no personalities. Pass --mind-repo <name> to run a real \
             one",
            schema.label
        ),
    }
    if fell_back {
        tracing::warn!(
            "the compiled-in data directory {} is not on this machine; falling back to the \
             working directory {} — pass --data to be sure",
            env!("CARGO_MANIFEST_DIR"),
            data.display()
        );
    }
    tracing::info!("data: {} (substrate and accounts)", data.display());

    // The cast, rebuilt from the substrate's redo log. The one read of that
    // log — every request after this is answered from memory. Shown on the
    // loading page as it runs: a store written before identities is rewritten
    // whole here, once, and on a large one that is minutes.
    let starting = Arc::clone(&ctx.starting);
    starting.set_step("opening", "Opening the substrate");
    let stage_shown: Cell<Option<OpenStage>> = Cell::new(None);
    let npcs = Npcs::load_reporting(&data, &|stage, done, total| {
        if stage_shown.get() != Some(stage) {
            stage_shown.set(Some(stage));
            starting.set_detail(stage.label());
        }
        starting.set_progress(done as u64, total as u64, stage.unit());
    })
    .map_err(|e| anyhow::anyhow!("opening the substrate at {}: {e:?}", data.display()))?;
    starting.set_step("starting", "Starting");
    let retention = Duration::from_secs(args.tombstone_retention.saturating_mul(SECONDS_PER_DAY));
    npcs.substrate()
        .persistence
        .call(|p| p.set_tombstone_retention(retention));

    // Accounts are records in the substrate, replicated but never published:
    // one carries a real email address and a provider subject id.
    let accounts = Accounts::open(npcs.substrate(), &data.join("accounts"))?;
    tracing::info!("accounts: {} known", accounts.len());

    let roles = ctx.roles.clone();
    if roles.names_no_admin() {
        tracing::warn!(
            "roles: no admin or creator configured — worlds and personalities are read-only to everyone"
        );
    }

    // The mind. A repository the agents hold is read from the substrate, now
    // that it is open: the files of the commit the log names for its branch
    // (`docs/substrate_git.md` §8). One the log does not hold yet is pending:
    // the load pulls it through an agent once joined, and takes it in before
    // it builds anything from it — what is read from it here is read again
    // then (`read_again_on_arrival`). A `--mind` folder is read as it stands.
    let (schema, mind) = match &repo {
        Some(source) => {
            let (db, files) = mind_store::from_log(source, &npcs.substrate())?;
            match files {
                Some(files) => {
                    let label = mind_label(source, &files);
                    let schema = projection::Source::from_files(Arc::clone(&files), label)?;
                    (schema, Mind::of(files, MindRepo::Substrate(db)))
                }
                None => {
                    tracing::warn!(
                        "mind: the substrate holds no `{}` on `{}` yet — starting with no mind; \
                         once joined it is pulled through an agent and taken in before the \
                         engine builds anything from it",
                        source.name,
                        source.branch
                    );
                    (schema, Mind::pending(db))
                }
            }
        }
        None => {
            let mind = Mind::new(schema.dir.clone());
            (schema, mind)
        }
    };
    tracing::info!("projection schema: {}", schema.label);
    // Read before the mind moves into the API state: whether it still has to
    // arrive, for the hook that takes it in (below).
    let mind_pending = !mind.is_present();

    // Authored content — from the MIND when there is one, from the data
    // directory otherwise. Worlds and personalities are written by a person;
    // the data directory holds what the ENGINE wrote.
    let authored_dir = schema.dir.clone().unwrap_or_else(|| data.clone());
    let (worlds, personalities) = match mind.is_present() || repo.is_some() {
        true => (
            Registry::load_mind("world", &mind, WORLDS)?,
            Registry::load_mind("personality", &mind, PERSONALITIES)?,
        ),
        false => (
            Registry::load("world", authored_dir.join(WORLDS))?,
            Registry::load("personality", authored_dir.join(PERSONALITIES))?,
        ),
    };
    tracing::info!(
        "authored content: {} worlds, {} personalities",
        worlds.len(),
        personalities.len(),
    );

    let (libraries, feelings) = read_libraries(&schema);

    // The editor and the engine read the same schema through the same `Mind`.
    // Only a `--mind` folder is watched: a mind read from the substrate moves
    // when its branch does, not when a file does.
    let mind_for_engine = mind.clone();
    match mind.files() {
        Some(files) => tracing::info!("mind: {files:?}"),
        None => {
            tracing::info!("mind: none yet — the file editor will report it has nothing to edit")
        }
    }

    // Portraits, as blobs in the substrate; a character naming an image kept
    // as a file before then is pointed at the blob it became.
    let (images, renames) = Images::open(npcs.substrate(), &data.join("images"))
        .map_err(|e| anyhow::anyhow!("taking in the images at {}: {e}", data.display()))?;
    if !renames.is_empty() {
        let moved = npcs
            .repoint_portraits(&renames, api::now_ms())
            .map_err(|e| anyhow::anyhow!("repointing portraits: {e:?}"))?;
        tracing::info!("images: {moved} character portraits repointed");
    }
    let personality_portraits =
        personality_portrait::ingest(&personalities, personalities.dir(), &images);
    tracing::info!(
        "authored portraits: {} of {} personalities carry one",
        personality_portraits.len(),
        personalities.len(),
    );

    let mut authored = Authored::new(
        worlds,
        personalities,
        accounts,
        npcs,
        roles.clone(),
        libraries,
        mind,
        images,
        personality_portraits,
    );
    // Every bot's calls through one count of the invalid answers, at this
    // member's share of the address's lines (`discord::limit`).
    authored.discord = DiscordClient::limited(
        discord_client::PRODUCTION,
        InvalidRequests::new(
            args.discord_invalid_throttle,
            args.discord_invalid_ceiling,
            limit::WINDOW,
        ),
    );
    // A link one member signs, another member may be asked to open.
    if let Some(join) = &ctx.cluster {
        authored.image_links = ImageLinks::new(join.derive_key(image_links::PURPOSE));
        authored.authorizations = Signer::new(join.derive_key(authorize::token::KEY_PURPOSE));
        // A bot token one member seals, whichever member holds the world opens.
        authored.discord_tokens = Sealer::new(join.derive_key(seal::PURPOSE));
        // Said before the engine load joins: the gateway takes a member's
        // weight from its admission. The runtime refuses leases on its own
        // account too (`Standing::characters`), so one lent before this is
        // handed back.
        join.link().set_leaseholder(args.characters);
        match (args.characters, args.host_worlds) {
            (true, _) => {}
            (false, true) => tracing::info!(
                "characters: off, host_worlds: on — this member takes no character lease, and \
                 hosts a world while no member that runs characters can"
            ),
            (false, false) => {
                tracing::info!("characters: off — this member takes no world or character lease")
            }
        }
    }

    // ── the engine ─────────────────────────────────────────────────────────
    //
    // The runtime is built first — it needs only the mind directory — then
    // handed to the state, then given the resolvers that read that state back.
    // That order breaks the cycle between them.
    let runtime = Runtime::new(mind_for_engine, &data)?;
    // npcd's own checkout of the journals repository, when this member keeps
    // one: the worlds that name a group write their bots' experience there.
    if let Some(dir) = &args.journals_dir {
        runtime.journals.start(dir.clone());
    }

    // The engine adopts the substrate the cast was just loaded from: one
    // `substrate/` takes one writable handle. See `Npcs::substrate`.
    runtime.set_substrate(authored.npcs.substrate());
    runtime.set_npcs(authored.npcs.clone());
    // What the characters' acts beyond the world present — the search key
    // among them — from the same file zend reads (`docs/zen_moon_mind.md`
    // §16). Never handed over, every `web_search` answered "no
    // tavily_api_key configured".
    runtime.set_tool_secrets(secrets::load(None)?);
    let authored = authored.with_runtime(runtime.clone());

    // The effector device: one durable token per character, and the `local`
    // router the engine's fast path drives in-process.
    let tokens = Arc::new(Tokens::load(data.join("tokens"))?);
    runtime.set_tokens(tokens.clone());
    let local_router = effector_router(Local::new(tokens.clone(), &runtime));
    runtime.set_effector_router(local_router.clone());

    install_sinks(&runtime, &authored, feelings);
    reload_on_move(&runtime, &authored);
    retire_kept_pictures(&authored);
    warm_content_check(&authored);
    // The characters' Discord bots, on the authority of each one's world
    // (`docs/npcd_discord.md` §3.3).
    bots::start(&authored);
    // Only for a mind still pending: the hook holds the authored state, which
    // holds the runtime, and the load takes it back out once it has run.
    if let Some(source) = repo.as_ref().filter(|_| mind_pending) {
        read_again_on_arrival(&runtime, &authored, source.clone(), authored_dir.clone());
    }
    // A member follows each world it hosts until its lease says otherwise
    // (`docs/replicated_substrate.md` §9.2): known before the first is hosted.
    if ctx.cluster.is_some() {
        runtime.standing.set_clustered();
    }
    runtime.standing.set_characters(args.characters);
    runtime.standing.set_host_worlds(args.host_worlds);
    host_worlds(&runtime, &authored, &authored_dir).await;

    // ── the routes ─────────────────────────────────────────────────────────
    let routes = routes(
        &authored,
        &runtime,
        roles.clone(),
        local_router,
        args.log_identity,
    )?;

    // ── start the engine ───────────────────────────────────────────────────
    //
    // The loader runs on its own OS thread and returns immediately, so the
    // server binds while the model is still being fetched — the loading screen
    // has to be reachable during the load it is describing.
    let cast: Vec<Casting> = authored.npcs.view().cast();
    tracing::info!("engine: loading, {} character(s) to wake", cast.len());
    let characters = authored
        .personalities
        .read()
        .await
        .iter()
        .map(|r| r.id.clone())
        .collect();
    let identities = identities(&authored).await;
    start_engine(
        runtime.clone(),
        LoadPlan {
            engine: ctx.engine,
            cluster: ctx.cluster,
            forget_conversations: args.forget_conversations,
            forget_dreams: args.forget_dreams,
            forget_journal: args.forget_journal,
            wipe_conversations: args.wipe_conversations,
            ingest: args.ingest,
            cast,
            characters,
            authored: identities,
            mind_repo: repo,
            agent: ctx.agent.clone(),
            stopping: ctx.stopping.clone(),
        },
    );

    // The guardian, when `guardian.yaml` asks for one.
    let guardian = match GuardianConfig::load(&data)?.filter(|c| c.enabled) {
        Some(config) => {
            let guardian = GuardianBuilder::from_config(&config).build()?;
            runtime.set_guardian_log(guardian.log());
            tracing::info!("guardian on: {} modules", config.modules.len());
            Some(tokio::spawn(
                guardian.run(Arc::new(RuntimeProbe(runtime.clone()))),
            ))
        }
        None => None,
    };

    let stopping = runtime.clone();
    Ok(Running {
        routes,
        reports: Arc::new(NpcdReports::new(authored, data)),
        // Its characters are handed back at their turn boundaries by the drain
        // itself (`docs/replicated_substrate.md` §9.3): nothing runs past it.
        leave: Box::new(|| Box::pin(async {})),
        stop: Box::new(move || Box::pin(stop(stopping, guardian))),
    })
}

/// Read the worlds and personalities again each time the mind folder moves to
/// a newer record — a publish on this member, or another member's that the
/// folder followed (`docs/replicated_substrate.md` §10.3). Holds the state
/// weakly, so the task ends with it.
pub(super) fn reload_on_move(runtime: &Arc<Runtime>, authored: &Arc<Authored>) {
    let mut moves = runtime.ops.moves();
    let authored = Arc::downgrade(authored);
    tokio::spawn(async move {
        while moves.changed().await.is_ok() {
            let Some(authored) = authored.upgrade() else {
                return;
            };
            if authored.runtime.as_ref().is_some_and(|rt| rt.stopping()) {
                return;
            }
            authored.reload_authored().await;
            tracing::info!("authored: read again after the mind moved to a newer record");
        }
    });
}

/// Read again what was read from a mind pending at start, once the load has
/// taken it in (`engine::runtime::MindArrived`): the worlds and personalities
/// with their portraits, the libraries and the registers, the worlds to host
/// — and the cast, caught up from the cluster meanwhile, with who everybody
/// is, handed back to the load for the identities it installs and the
/// characters it wakes.
pub(super) fn read_again_on_arrival(
    runtime: &Arc<Runtime>,
    authored: &Arc<Authored>,
    source: MindSource,
    authored_dir: PathBuf,
) {
    let state = authored.clone();
    runtime.set_mind_arrived(Arc::new(
        move || -> BoxFuture<'static, anyhow::Result<Arrived>> {
            let (state, source, authored_dir) =
                (state.clone(), source.clone(), authored_dir.clone());
            Box::pin(async move {
                state.reload_authored().await;
                if let Some(files) = state.mind.files() {
                    let label = mind_label(&source, &files);
                    // Fatal here as it is at start: a schema that cannot be read
                    // leaves characters with no registers to decode under.
                    let schema = projection::Source::from_files(files, label).map_err(|e| {
                        anyhow::anyhow!("mind: the schema that arrived could not be read: {e}")
                    })?;
                    tracing::info!("projection schema: {}", schema.label);
                    let (libraries, feelings) = read_libraries(&schema);
                    state.set_libraries(libraries);
                    if let Some(rt) = &state.runtime {
                        rt.set_feelings(feelings);
                    }
                }
                if let Some(rt) = &state.runtime {
                    host_worlds(rt, &state, &authored_dir).await;
                }
                state.npcs.refresh();
                if let Some(rt) = &state.runtime {
                    rt.follow_suspended();
                }
                let cast = state.npcs.view().cast();
                let characters: Vec<String> = state
                    .personalities
                    .read()
                    .await
                    .iter()
                    .map(|r| r.id.clone())
                    .collect();
                let worlds = state.worlds.read().await.len();
                tracing::info!(
                "authored content, read again: {worlds} worlds, {} personalities, {} character(s) \
                 to wake",
                characters.len(),
                cast.len(),
            );
                Ok(Arrived {
                    cast,
                    characters,
                    authored: identities(&state).await,
                })
            })
        },
    ));
}

/// What a mind read from the substrate is called in the log: its repository,
/// commit and branch.
pub(super) fn mind_label(source: &MindSource, files: &MindFiles) -> String {
    format!(
        "{}@{} ({})",
        source.name,
        files.commit().map(|c| c.to_string()).unwrap_or_default(),
        source.branch
    )
}

/// The response and mood libraries, beside the schema — ingested untagged
/// and shared by every world — and the registers a character may say it is
/// in, the moods' ids, which a turn's decoding grammar is built over.
pub(super) fn read_libraries(schema: &projection::Source) -> (Libraries, Vec<String>) {
    let libraries = Libraries::load(schema);
    tracing::info!(
        "libraries: {} responses ({} with examples), {} moods ({} with examples)",
        libraries.responses.len(),
        libraries.responses.with_examples(),
        libraries.moods.len(),
        libraries.moods.with_examples(),
    );
    let feelings = libraries
        .moods
        .sections
        .iter()
        .map(|s| s.id.clone())
        .collect();
    (libraries, feelings)
}
