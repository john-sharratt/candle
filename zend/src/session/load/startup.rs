//! Builds an `InferenceState` from the workspace on the loader thread, one ordered
//! step at a time.
//!
//! Each step depends on the ones before it. The tool catalogue and identities are
//! installed into the schema before the base conversation is built, the redo log
//! is replayed before section prefill begins, and ingest passes run only once the
//! schema they read is final. Tool sections are content addressed, so an unchanged
//! catalogue costs no prefill on the next boot. A stop asked at any step boundary
//! drains what was sealed and returns without an engine, which is a stop rather
//! than a failure. Each step reports its progress against the loading bar.

use super::*;
use crate::reprojection_settings::ReprojectionSettings;
use candle_conversation::index_page_cache;

impl InferenceState {
    // Every argument is a distinct piece of deployment config read from the CLI
    // / config file at startup, used once here and never carried together
    // afterwards, so there is no struct for them to already belong to.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn load(
        mut proj_builder: Builder,
        model: Model,
        engine_source: EngineSource,
        device: Device,
        pack: PathBuf,
        workspace: Workspace,
        secrets: Arc<Secrets>,
        disabled_layers: HashSet<String>,
        wiped_layers: HashSet<String>,
        ingest_dirs: HashMap<String, String>,
        max_depth: Option<usize>,
        ignore: Ignore,
        host_runtime: Option<Handle>,
        agent: Option<Arc<Jobs>>,
        compact_substrate: bool,
        self_check: SelfCheck,
        read_only_substrate: bool,
        qsa_selection_budget: Option<usize>,
        qsa_strata: StrataTokens,
        summarize: bool,
        reprojection: ReprojectionSettings,
        tombstone_retention: Duration,
        cluster: Option<Join>,
        status_tx: watch::Sender<String>,
        progress: Arc<LoadProgress>,
        loading_substrate: Arc<OnceLock<Conversation>>,
        ops: Arc<dyn OpsRunner>,
        memory: Arc<Memory>,
    ) -> anyhow::Result<Option<Arc<Self>>> {
        // The workspace folder: where the substrate, the schema override and a
        // mind's section folders live. Repositories are reached through
        // `workspace` itself.
        let root = workspace.root().to_path_buf();

        // The pack is found; the steps from here move as the work does — the
        // substrate open's hook enters `Opening`, the first weight block
        // mounted enters `Model` (see the builder below). The engine ctor
        // also reloads the substrate internally, so the visible boundary
        // between Model and Substrate below is best-effort — the substrate
        // has already loaded by the time its step is announced.

        // The card's headroom at startup, beside the file-ingest pool width
        // that will spend it, so a CUDA OOM during code_read is traceable to
        // both figures rather than an opaque exit-1.
        if let Ok((free, total)) = device.mem_get_info() {
            let free_gib = (free as f64) / (1024.0 * 1024.0 * 1024.0);
            let total_gib = (total as f64) / (1024.0 * 1024.0 * 1024.0);
            let n_workers = crate::code_read::CODE_READ_PARALLELISM;
            tracing::info!(
                "VRAM {free_gib:.1}/{total_gib:.1} GiB free · code_read workers={n_workers}"
            );
        }

        // Resolve the dialogue layer / primary group up front.  The
        // tool-catalog injection adds sections to that layer.
        let dialogue_layer = proj_builder
            .id_for_layer("dialogue")
            .ok_or_else(|| anyhow::anyhow!("projection schema missing 'dialogue' layer"))?;
        let primary_group = proj_builder
            .id_for_group("primary_conversation")
            .ok_or_else(|| {
                anyhow::anyhow!("projection schema missing 'primary_conversation' group")
            })?;
        let working_set = dialogue_config(proj_builder.schema(), dialogue_layer);
        // (layer, group) are passed through to
        // `new_conversation_with_projection`, which mints a fresh
        // `TimelineId` internally.

        // Install the tool catalog into the schema before building the
        // base conversation.  Each tool gets a section in dialogue's
        // system_prompt; the projection layer's `tools` collection
        // governs `top_k` so only the K most relevant tools survive
        // into any single projection.
        //
        // Sections are content-addressed in the redo log: a boot restores
        // every section whose stream is already persisted as a cold marker
        // (the next projection that needs it lifts it hot) and prefills only
        // the rest, whose seals the persistence thread then writes. An
        // unchanged catalog therefore costs no prefill and adds no records on
        // the next boot (`tools_integration::a_second_boot_restores_every_
        // prompt_section`).
        // Resolve the effective tool catalog before any consumer touches it: a
        // `<workspace>/tools/` folder (a mind/game's own tools) overrides the
        // bundled built-ins; absent it, the built-in coding-assistant catalog.
        // Every `repo` parameter is constrained to the workspace's repositories.
        crate::tool_def::init(&workspace);
        let tool_sections = install_tool_catalog(&mut proj_builder)
            .map_err(|e| anyhow::anyhow!("tool catalog install: {e}"))?;
        tracing::info!(
            n_tools = tool_sections.len(),
            "tool catalog installed (top_k governed by `tools` collection in projection.yaml)",
        );
        // The journal collection selects each conversation's own sections by
        // name; scoped before the base conversation is built, so no projection
        // ever reads it any other way.
        let journaling = carry::install(&mut proj_builder)
            .map_err(|e| anyhow::anyhow!("journal install: {e}"))?;
        tracing::info!(journaling, "journal collection installed");

        // Structure-derived section-collection population. Each empty section
        // collection (e.g. `response`, `mood`) is filled from its content folder
        // — `<collection>s/*.yaml` — as authored sections BEFORE base-conv build,
        // so each template's KV seals into the frame (baseline provenance from its
        // own prefill, exactly like a tool's JSON line). The folder and the fact
        // that it needs loading are DERIVED from the declared schema (see
        // `crate::ingest::section_sinks`); nothing is annotated in projection.yaml.
        // Each section's `examples` are parsed and validated here; selection
        // provenance currently comes from each template's own prefill (the
        // calibration phase below calibrates tool selection only). For the
        // coding-assistant schema there are no such collections, so this is a no-op.
        let identity = crate::response_section::Identity::load(&root);
        for sink in crate::ingest::section_sinks(proj_builder.schema()) {
            if disabled_layers.contains(&sink.collection) {
                tracing::info!(collection = %sink.collection, "--disable-layer: section collection suppressed");
                continue;
            }
            let sections =
                crate::response_section::load_sections(&root.join(&sink.folder), &identity);
            if sections.is_empty() {
                continue;
            }
            let installed = crate::response_section::install_sections(
                &mut proj_builder,
                &sink.collection,
                &sections,
            )
            .map_err(|e| anyhow::anyhow!("install '{}' sections: {e}", sink.collection))?;
            tracing::info!(
                collection = %sink.collection,
                folder = %sink.folder,
                n = installed.len(),
                "installed calibrated section collection",
            );
        }

        // Identity sections: the two-level `identities/<name>/*.yaml` tree, one
        // sub-folder per identity (its `anchor.yaml` + detail facets). Members are
        // namespaced `<name>::<stem>` and installed into the `identity_anchor`
        // (always-visible) and `identity` (top-k) collections so every identity's
        // KV seals once here; a conversation scopes to its own identity at
        // projection time (see `IdentityBuilders`). No-op when the schema declares
        // neither collection (the coding-assistant schema) or `identities/` is
        // absent.
        let identity_sections =
            crate::response_section::load_identity_sections(&root.join("identities"), &identity);
        for (collection, sections) in [
            ("identity_anchor", &identity_sections.anchors),
            ("identity", &identity_sections.facets),
        ] {
            if sections.is_empty() {
                continue;
            }
            // Only install into a collection the schema actually declares — a
            // workspace with an `identities/` folder but a schema that doesn't
            // declare these collections (e.g. the coding-assistant schema) is a
            // no-op, not a startup failure.
            if proj_builder.id_for_system_collection(collection).is_none() {
                tracing::warn!(
                    collection,
                    n = sections.len(),
                    "identities/ present but schema declares no '{collection}' collection — skipping",
                );
                continue;
            }
            let installed =
                crate::response_section::install_sections(&mut proj_builder, collection, sections)
                    .map_err(|e| anyhow::anyhow!("install '{collection}' sections: {e}"))?;
            tracing::info!(
                collection,
                n = installed.len(),
                "installed identity collection",
            );
        }
        let default_identity = identity.default_identity.clone();

        // `--disable-layer` takes its layer OUT OF THE PROVENANCE GATHER, and the
        // flag has to be set HERE — while there is still exactly one builder.
        // Every schema that resolves a projection later is a copy of this one:
        // `base_conv` is built from `proj_builder` itself, and the clone below
        // feeds the ingest passes, the per-mode turn builders
        // (`ModeBuilders::build`) and the ingest worker's `refresh_builder`. Setting
        // `gathered` after the clone would leave the base conversation — the very
        // schema the live dialogue projects against — still gathering the layer,
        // which is the half that mattered.
        //
        // `gathered` is runtime-only and deliberately not a YAML field: the
        // schema declares what a layer IS, and whether this boot lets it compete
        // is an operator decision for this process only. Nothing is deleted, so
        // dropping the flag restores the layer's turns to the gather intact.
        //
        // A name that is a section collection rather than a layer (`response`,
        // `mood`) matches nothing here; its suppression is the `section_sinks`
        // loop above, so a `false` return is expected, not an error.
        for name in &disabled_layers {
            if proj_builder.set_layer_gathered(name, false) {
                tracing::info!(
                    layer = %name,
                    "--disable-layer: layer excluded from the provenance gather",
                );
            }
        }
        // The dialogue layer's `system_prompt.items` start with a static
        // prelude (mode/frame/grounding/tools_intro) →
        // then the `tools` collection (90+ tool sections, top_k=5) →
        // then `tools_outro`.  The pre-collection prelude is what we
        // pass as the engine's `system_prompt` so it gets ChatML-wrapped
        // for the dialect; everything after the first Collection is
        // expanded by `preemptive_prefill` itself.
        let before_text: String = pre_collection_prelude(&proj_builder);

        // The model pack, from the model cache: shared by every workspace on
        // this model, it survives `--wipe-substrate`, and carries the tokenizer.
        // What every unit this process ingests says it was computed under: a
        // duplicate a better-quantized member wrote replaces this one's.
        crate::branch_ingest::keys::set_model_bits(
            candle_conversation::guest::checkpoint::weight_bits(&pack),
        );
        // zend's engine-wide asks come from `engine_asks`, the one place a
        // host that runs without zend also takes them from.
        let mut builder = engine_asks::ask(
            model
                .builder()
                .system_prompt(&before_text)
                .model_path(pack)
                .workspace_path(root.clone())
                .read_only_substrate(read_only_substrate)
                .tombstone_retention(tombstone_retention)
                .cancel_token(crate::ingest_cancel::token().clone()),
            &proj_builder,
            qsa_selection_budget,
            qsa_strata,
        )
        // Dialogue turns compress at C5 (moderate adaptive quantization).
        // Paired with the removed uniform-K pin (see `ModelBuilder::engine`),
        // so K is adaptive too.
        .compression_level(5)
        // Per-layer corrupt-turn policy (from the projection schema): the
        // startup reload drops the whole conversation for ingest layers and
        // only the corrupt turn for dialogue.
        .corrupt_turn_policies(
            proj_builder
                .schema()
                .layers
                .iter()
                .map(|l| (l.id, l.on_corrupt_turn))
                .collect(),
        )
        // The substrate open is its own step: a store written before
        // identities is stamped here, record by record, and that ran for a
        // minute and a half under "Loading model" at 0%. The stage names
        // the line under the bar; the records fill it.
        .on_substrate_open({
            let progress = Arc::clone(&progress);
            let status = status_tx.clone();
            let shown: Arc<Mutex<Option<OpenStage>>> = Arc::new(Mutex::new(None));
            OpenProgressHook(Arc::new(
                move |stage: OpenStage, done: usize, total: usize| {
                    progress.set_step(LoadStep::Opening);
                    progress.set_step_unit(stage.unit());
                    progress.set_step_progress(done as u64, total as u64);
                    let mut last = shown.lock().unwrap_or_else(|e| e.into_inner());
                    if *last != Some(stage) {
                        *last = Some(stage);
                        status.send(stage.label().to_string()).ok();
                    }
                },
            ))
        });
        // The model load, phase by phase: the builder says when it begins —
        // before the weights, and before the substrate opens — and each phase
        // names the line under the bar and what its counts are.
        let model_progress = Arc::clone(&progress);
        let model_status = status_tx.clone();
        let model_phase: Mutex<Option<LoadPhase>> = Mutex::new(None);
        let model_hook = move |phase: LoadPhase, done: usize, total: usize| {
            if model_progress.current_step() != Some(LoadStep::Model) {
                model_progress.set_step(LoadStep::Model);
            }
            let mut shown = model_phase.lock().unwrap_or_else(|e| e.into_inner());
            if *shown != Some(phase) {
                *shown = Some(phase);
                model_status.send(phase.label().to_string()).ok();
                model_progress.set_step_unit(phase.unit());
            }
            model_progress.set_step_progress(done as u64, total as u64);
        };
        // The engine comes from the process's slot: built here, with every
        // tenant's requests merged, when zend ranks highest of the tenants that
        // arrive; otherwise zend's substrate is bound and attached to the engine
        // the higher one built. Building loads the model weights (blocking,
        // per-layer hook above), then opens zend's substrate (the `Opening`
        // step), then spawns the scheduler thread, which kicks off the
        // substrate reload (redo-log replay) in the background. So this timing
        // is the weight load plus the open; the reload's cost shows up as the
        // `LoadStep::Substrate` wait below.
        let t_engine = Instant::now();
        let waiting_progress = Arc::clone(&progress);
        let engine = engine_source
            .ticket()
            .engine_with_progress(&mut builder, &device, Some(&model_hook), &|tenants| {
                waiting_progress.set_waiting_for(tenants)
            })
            .map_err(|e| anyhow::anyhow!("engine build: {e}"))?;
        tracing::info!(
            elapsed_ms = t_engine.elapsed().as_millis() as u64,
            "load timing: model weights loaded (engine built; substrate reload now running in background)",
        );

        // The index page cache's slab, taken once per process and only now:
        // the model has loaded, so the expert tier has pinned its pool and the
        // cache takes what the host partition left it
        // (`docs/hit_driven_projection.md` §7.6).
        let host_total = vram::total_physical_ram().unwrap_or(0);
        index_page_cache::install(vram::index_cache_target(host_total), &device);

        // AFTER the engine build, never before. The builder resolves the
        // `<think>`/`</think>` ids into its sampling config while loading the
        // tokenizer, so a config taken earlier is a snapshot with both ids still
        // at their `-1` placeholder — and every tier of the think-block close
        // budget is gated on them. Unresolved, `update_segment_state` returns at
        // its first guard, `in_segment` never opens, `segment_len` never counts,
        // and the graceful ramp, the force cutoff and the closer script are all
        // silently unreachable: think blocks then run to the stencil's runaway
        // span cap and get amputated mid-word instead of closing on a clause.
        let mut conv_config = builder.conversation_config();
        // zend decodes code, not prose: every conversation it opens — dialogue,
        // passthrough, ingest — derives from this config. See `coding_sampling`.
        coding_sampling::apply(&mut conv_config.sampling);
        // The same: every conversation decodes under the daemon's reprojection
        // settings.
        reprojection.apply(&mut conv_config);
        tracing::info!(
            probe_tokens = conv_config.reproject_max_probe_tokens,
            in_thinking = conv_config.reproject_in_thinking,
            "reprojection settings",
        );
        // Every conversation — dialogue, titler, passthrough, ingest — derives
        // from this config, so this one switch covers them all. See
        // `DaemonConfig::summarize` for why summaries are opt-in.
        if !summarize {
            conv_config.tree.disable_summarization();
        }
        tracing::info!(
            enabled = conv_config.tree.summarizes(),
            "background tree summaries (--summarize)",
        );
        if conv_config.sampling.segment_close_token_id < 0
            || conv_config.sampling.segment_open_token_id < 0
        {
            // Loud, because the failure mode is invisible: generation still
            // works, it just ignores every thinking budget.
            anyhow::bail!(
                "thinking token ids unresolved in the dialogue sampling config \
                 (open={}, close={}) — the think-block close budget would be inert",
                conv_config.sampling.segment_open_token_id,
                conv_config.sampling.segment_close_token_id,
            );
        }

        // Compile each tools mode's catalog into its own constrained-decoding
        // stencil, keyed by the `<tool_call>` trigger, so any tool call a turn
        // starts is forced to the exact shape of a tool its mode offers. Only
        // arm a stencil for tools that were actually installed into the
        // dialogue prompt. A tool-free projection, or a mode that offers
        // nothing, gets an empty `TriggerRegistry` (never fires) rather than a
        // compiled stencil, which rejects an empty catalog.
        let t_compile = Instant::now();
        let compile = |mode: ToolMode| -> anyhow::Result<Arc<TriggerRegistry>> {
            let specs = crate::tools::catalog_for(mode);
            if tool_sections.is_empty() || specs.is_empty() {
                return Ok(Arc::new(TriggerRegistry::new()));
            }
            // A tool a turn calls once is held out of every later call of the
            // turn, by every registry held from this one.
            engine
                .compile_tool_stencil(&specs, MAX_TOOL_CALLS_PER_TURN)
                .map(|triggers| engine.never_twice(&triggers, &crate::tools::never_twice_names()))
                .map_err(|e| anyhow::anyhow!("{} tool stencil compile: {e}", mode.id()))
        };
        let [none, restricted, comprehensive] = ToolMode::ALL.map(compile);
        let tool_stencils = [none?, restricted?, comprehensive?];
        // The thinking-block steering trees (one per non-off effort dial),
        // compiled once and reused across turns alongside the tool-call base.
        let think_steering = engine
            .compile_think_steering()
            .map_err(|e| anyhow::anyhow!("think steering compile: {e}"))?;
        tracing::info!(
            elapsed_ms = t_compile.elapsed().as_millis() as u64,
            "load timing: tool-stencil + think-steering compiled",
        );

        // The substrate reload (redo-log replay) runs on the scheduler thread
        // spawned by the engine ctor. As the substrate grows this is no longer
        // instantaneous, so surface real progress and block here until it
        // finishes before advancing to section prefill (which would otherwise
        // stall against the busy scheduler under a misleading step label).
        //
        // This wait's elapsed time is the reload's OVERHANG past the model-weight
        // load — i.e. how much longer the redo-log replay took than loading the
        // weights it ran alongside. It grows with substrate size, so it's the
        // number to watch as the corpus (and its segment count) scales.
        progress.set_step(LoadStep::Substrate);
        let t_reload_wait = Instant::now();
        {
            let reload = engine.substrate_reload_status();
            loop {
                let (done, total, finished) = reload.snapshot();
                if total > 0 {
                    progress.set_step_progress(done as u64, total as u64);
                }
                if finished {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        tracing::info!(
            elapsed_ms = t_reload_wait.elapsed().as_millis() as u64,
            "load timing: substrate reload overhang (redo-log replay past the weight load)",
        );
        // Replayed: the substrate views can read it from here, while the rest of
        // the load runs (`ZendSession::substrate_for_views`).
        let _ = loading_substrate.set(engine.conversation());

        // Reclaim is normally fully background: the segmented log's
        // persistence-thread maintenance pass drops / compacts / combines
        // segments incrementally, so a
        // startup pays no whole-store rewrite. `--compact-substrate` forces the
        // eager path instead — a whole-store rewrite here, after the reload (so
        // the live set is known) and before serving. It always runs when the
        // flag is set (no reclaimable-marker gate): the operator asked for it.
        // A read-only substrate is never rewritten.
        if compact_substrate && !read_only_substrate {
            progress.set_step(LoadStep::Compacting);
            let cprog = Arc::clone(&progress);
            let cb = move |done: usize, total: usize| {
                cprog.set_step_progress(done as u64, total as u64);
            };
            match engine.compact_substrate(Some(&cb)) {
                Ok(()) => tracing::info!("substrate compaction complete"),
                Err(e) => tracing::warn!("substrate compaction failed: {e:#}"),
            }
            // Compaction rewrote the log — re-reconstruct the substrate so the
            // scheduler-side view (KV residence + offsets) matches the new log.
            let reload = engine.reload_substrate();
            loop {
                let (done, total, finished) = reload.snapshot();
                if total > 0 {
                    progress.set_step_progress(done as u64, total as u64);
                }
                if finished {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }

        if load_stopped(&engine, "the substrate reload") {
            return Ok(None);
        }

        // With `--cluster`, join the gateways now that the substrate — and the
        // node it carries — is open (`docs/replicated_substrate.md` §7.6). A
        // gateway that refuses this build for good leaves the daemon here, on
        // its loading page with the reason, until it is stopped.
        if let Some(join) = &cluster {
            progress.set_step(LoadStep::Connecting);
            let conv = engine.conversation();
            let writer = conv.clone();
            let reloader = engine.reloader();
            let store = ConversationStore::new(conv.clone(), engine.kv_layers(), move || {
                reloader.reload_and_wait()
            });
            let replication = store.replication()?;
            join.link().set_repos(repo_ads(&workspace));
            // The ingest units still to do, which the gateway's plan moves
            // characters off this card for (`docs/cooperative_admission.md` §4.4).
            join.link().set_bulk(|| {
                crate::ingest_backlog::snapshot().map_or(0, |b| {
                    b.total.saturating_sub(b.processed).min(u64::from(u32::MAX)) as u32
                })
            });
            let report = |views: &[ClusterView]| {
                progress.set_step_progress(admitted(views), views.len() as u64);
                status_tx.send(connecting_detail(views)).ok();
            };
            let link = join.connect(
                conv.node(),
                Some(model_id(engine.model_key())),
                replication,
                move || writer.last_oseq(),
                crate::ingest_cancel::cancelled,
                report,
            );
            if let Link::Stopped(_) = link {
                join.hold(crate::ingest_cancel::cancelled, report);
            }
            // A conversation another member lets go is this one's to take once
            // it holds it complete, and this one's own releases name their
            // conversations' newest commits (`handover`).
            if let Some(member) = join.link().member() {
                crate::handover::install(
                    member.clone(),
                    &store,
                    conv.clone(),
                    conversation_timeline,
                );
            }
            if link == Link::Joined {
                // Serving from a substrate that is behind is the thing to
                // avoid (§7.6): ready only once caught up.
                progress.set_step(LoadStep::CatchUp);
                join.catch_up(crate::ingest_cancel::cancelled, |catchups| {
                    let (done, total) = catchup_progress(catchups);
                    progress.set_step_progress(done, total);
                    status_tx.send(catchup_detail(catchups)).ok();
                });
            }
            if crate::ingest_cancel::cancelled() {
                tracing::warn!("shutdown while connecting to the cluster — draining the engine");
                if let Err(e) = engine.shutdown() {
                    tracing::error!("shutdown-while-connecting drain failed: {e}");
                }
                return Ok(None);
            }
            if let Some(replicator) = join
                .link()
                .member()
                .and_then(|member| member.replicator().cloned())
            {
                // Readings lease a unit before minting its conversation, so a
                // second member that planned the same tree does not read it too.
                ingest_lease::install(replicator);
            }
            status_tx.send(String::new()).ok();
        }

        // Repositories read from the substrate (`docs/substrate_git.md` §7):
        // every store brought up to the log before anything reads a branch,
        // then kept there — and with origin — on a thread of its own, which
        // `ZendSession::shutdown` stops. A workspace naming no repository
        // has nothing to load, and needs no agent.
        let repositories = {
            let names: Vec<String> = workspace
                .repos()
                .iter()
                .map(|r| r.name.clone())
                .filter(|name| name != UPLOADS_REPO)
                .collect();
            if names.is_empty() {
                None
            } else {
                let runtime = host_runtime.clone().ok_or_else(|| {
                    anyhow::anyhow!("no async runtime for the agents' jobs to run on")
                })?;
                let access = substrate_git::access(cluster.as_ref(), agent.as_ref(), runtime)?;
                Some(substrate_git::load(
                    &workspace,
                    &names,
                    engine.conversation().shared(),
                    access,
                    &progress,
                    &status_tx,
                )?)
            }
        };
        // A stop ends the repositories' load between rounds, incomplete — and
        // nothing after it should read a branch they have not brought up. The
        // follower goes with `repositories`, whose drop stops it.
        if load_stopped(&engine, "loading repositories") {
            return Ok(None);
        }
        // The seed, named from the repositories' roots as the knowledge base's
        // mandatory files before any conversation is built — and only once the
        // repositories are up to the log, or a fresh substrate's seed reads no
        // repository at all (docs/placed_section_layout.md §3.5b).
        let (seed_repos, _) = crate::branch_ingest::record_branches(&workspace);
        let seed_read = crate::files::seed::install(&engine, &proj_builder, &seed_repos)?;

        let formatted_prompt = builder.format_system_prompt();
        let decoder = engine.token_decoder();

        // Tokenise every `kind: template` item in the projection schema
        // using the engine's tokenizer.  The projection engine emits a
        // `Generated` segment for each template at apply time carrying
        // these tokens; the assembler then runs (or caches) a live
        // prefill against them under the current runtime left context.
        let tokenizer = engine.tokenizer();
        proj_builder.tokenize_templates::<anyhow::Error, _>(|s| {
            let encoded = tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
            Ok(encoded.get_ids().to_vec())
        })?;

        progress.set_step(LoadStep::Sections);
        // Per-section progress callback — bytes ingested out of total
        // section bytes (cheap proxy for tokens, smooth across uneven
        // sections without a pre-tokenise pass). The "Prefilling tool
        // sections" step is split in two: section prefill fills the first
        // half (0..50%), the tool-catalog summary decode fills the second
        // (50..100%) — both reported against a fixed 10_000-unit total so the
        // two phases compose into one continuous bar.
        let section_progress = Arc::clone(&progress);
        let section_hook = move |done: u64, total: u64| {
            // Cheap (one `GlobalMemoryStatusEx`/`/proc/meminfo` read) at section
            // granularity, not per token — see `vram::sample_available_low_water`.
            // Section prefill is the first of the two boot phases whose combined
            // footprint (tool corpus + the expert warm tier sized against
            // launch-time RAM) has crashed a from-empty-substrate boot with a raw
            // host allocator abort; this is the trough this phase leaves behind.
            vram::sample_available_low_water();
            let scaled = (done * 5_000).checked_div(total).unwrap_or(0);
            section_progress.set_step_progress(scaled, 10_000);
        };
        // A clone of the (tools-installed + templates-tokenised) projection
        // builder is kept for the schema-driven ingest passes (one cheap clone
        // per ingest layer below — schemas are `Arc`-backed) and stays on
        // `InferenceState` afterwards for the ingest worker's passes to mint
        // fresh ingest-layer timelines.
        let proj_builder_refresh = proj_builder.clone();
        let base_conv = engine.new_conversation_with_projection_progress(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config.clone(),
            Some(&section_hook),
        );
        // The section prefill checks the ingest-cancel latch between sections,
        // and a stop asked during it ends the build `IngestCancelled`: the stop,
        // not a failed load.
        if matches!(
            base_conv,
            Err(candle_conversation::ConversationError::IngestCancelled)
        ) && load_stopped(&engine, "prefilling tool sections")
        {
            return Ok(None);
        }
        let mut base_conv = base_conv.map_err(|e| anyhow::anyhow!("base conv create: {e}"))?;

        // Section ingestion (prelude + parallel tool-section forks +
        // outro) happens eagerly inside
        // `new_conversation_with_projection`: the schema's declared
        // sections / collections are pinned into the workspace
        // substrate at construction time via `insert_section` /
        // `insert_section_collection`.  Each tool section's BDP sigs come
        // from that single prefill of its JSON definition; JSON
        // structural tokens are excluded from the reprojection probe
        // (`build_reproject_probe_filter_token_ids`) so they don't add
        // shared-structure noise to section scoring.
        tracing::info!(
            n_tool_sections = tool_sections.len(),
            free_low_water_mib = vram::available_low_water().map(|b| b / (1024 * 1024)),
            "base conversation ready (prelude + tool catalog + outro pinned at init)",
        );
        // A stop asked as the last section sealed lands here.
        if load_stopped(&engine, "prefilling tool sections") {
            return Ok(None);
        }

        // Tool-catalog summaries: one per tools mode that projects tools, each
        // listing exactly the tools that mode offers. Each is assembled
        // **deterministically** from the catalog metadata — every tool grouped
        // under its category — so there is no model call and nothing to cache:
        // the text is rebuilt and its section prefilled on every startup, exactly
        // like the tool sections themselves.
        {
            let prelude = pre_tools_prefix(&proj_builder_refresh, base_conv.native_context());
            let summary_for = |mode: ToolMode| {
                let offered = crate::tools::offered_tool_names(mode);
                let sections: Vec<crate::tool_summary::InstalledTool> = tool_sections
                    .iter()
                    .filter(|(name, _, _)| offered.contains(name))
                    .cloned()
                    .collect();
                crate::tool_summary::build_tool_summary(&sections)
            };
            let summaries: Vec<(String, Reserved, &str)> = ToolMode::ALL
                .into_iter()
                .filter_map(|mode| {
                    Some((summary_for(mode), tool_summary_section(mode)?, mode.id()))
                })
                .collect();
            // Assembly is instant — the step's second half completes immediately.
            progress.set_step_progress(10_000, 10_000);

            // Seal each mode's summary under its reserved section id, prefilled with
            // the same pre-tools prefix so its KV is position-correct for "just
            // before the tools". Each mode's projection points the tools collection
            // at its own summary ([`tool_summary_section`]); None emits none.
            for (text, reserved, label) in summaries {
                let sid = SectionId::reserved(reserved);
                match base_conv.insert_section_with_prefix(sid, &text, &prelude) {
                    Ok(()) => tracing::info!(
                        section = sid.raw(),
                        label,
                        "tool summary section sealed (prefilled before tools)",
                    ),
                    Err(e) => tracing::warn!(label, "tool summary section seal failed: {e}"),
                }
            }

            // The lines saying a tool is out for now, each sealed after the
            // summary they follow, so a round picks any set of them for nothing.
            let mut after_summary = prelude;
            after_summary.extend(
                tool_summary_section(ToolMode::Comprehensive)
                    .map(|r| PrefixItem::Section(SectionId::reserved(r))),
            );
            for (sid, line) in crate::tool_availability::line_sections() {
                let text = format!("\n{line}");
                if let Err(e) = base_conv.insert_section_with_prefix(sid, &text, &after_summary) {
                    tracing::warn!(line, "tool availability line seal failed: {e}");
                }
            }
        }

        // The titler: what each title job opens its own conversation on
        // (`titles::Titler`). The timelines a titler conversation that lived
        // across jobs left behind hold every title it ever wrote, which each
        // title's projection scored: they go.
        let titler = Titler::new(
            conv_config
                .dialect
                .format_system_prompt(TITLER_SYSTEM_PROMPT),
            conv_config.clone(),
        );
        let retired = retire_titler_timelines(&engine);
        let tokenizer = Arc::new(engine.tokenizer().clone());
        tracing::info!(retired, "titler ready");

        // ── Calibrating sections ──────────────────────────────────────────────
        // Prefill each registered tool's authored trajectories and questions into
        // the hidden `Reserved::Calibration` layer, capturing each think→call
        // trajectory (and its per-reprojection wide-Q windows) and each question.
        //
        // All cases share ONE projection: the whole catalog as a name-keyed `tools`
        // collection governed by `SelectionRule::Named`. Each run pins exactly its
        // tool by name (`TurnOptions::selection`), so the model sees a single
        // available tool — the condition under which a free decode reliably calls
        // the right one. The tool sections are sealed once (the first run); every
        // later run reuses them via the idempotent section ingest. One throwaway
        // conversation per (tool, example); a failing case is logged and skipped so
        // it can never break daemon load.
        if load_stopped(&engine, "the tool summaries") {
            return Ok(None);
        }
        progress.set_step(LoadStep::CalibratingSections);
        // Skip entirely for a tool-free projection — nothing to calibrate — and
        // on a read-only substrate, whose calibration corpus is whatever the
        // daemon that writes it has already sealed.
        if !tool_sections.is_empty() && !read_only_substrate {
            // The phase itself is the host's, run by every tenant that offers
            // tools; this daemon gives it the coding assistant's frame, its
            // tool files, and the load's own progress and stop.
            let frame = calibration_frame(&conv_config.dialect);
            calibration::run(
                &calibration::Phase {
                    engine: &engine,
                    config: &conv_config,
                    frame: &frame,
                    progress: &|done, total| progress.set_step_progress(done, total),
                    stopped: &ingest_cancel::cancelled,
                },
                tool_def::all(),
            )
            .map_err(|e| anyhow::anyhow!("calibrating sections: {e}"))?;
            // Stopped mid-phase: the drain persists what sealed, and the rest
            // is the next load's.
            if load_stopped(&engine, "calibrating sections") {
                return Ok(None);
            }
        }

        // Every ingest pass takes the shared `&ConversationEngine`, so the loop
        // below is uniform across injection types. The substrate handle status
        // reads comes out first.
        let substrate = engine.conversation();

        // Structure-derived ingestion. The load plan is derived from the declared
        // schema (see `crate::ingest`), not annotated in it: each turn-sink layer
        // is prepared here, in schema order, skipping any named by
        // `--disable-layer`. The folder and file layers are ingested from every
        // repository's branches (or the `--ingest-dir` scope inside one) by the
        // background worker; a raw layer reads its content folder under the
        // workspace.
        let ingest_layers =
            crate::ingest::ingest_layers(proj_builder_refresh.schema(), &root, &ingest_dirs);
        // Mark EVERY enabled ingest layer append-only UP FRONT — deterministically,
        // before any ingest AND before `warm_ingest_normalization` runs — using
        // the SAME builder (`proj_builder_refresh`) whose layer ids the warm-up
        // checks. Marking via any other builder's id leaves the warm seeing 0
        // append-only layers, every code scope COLD, and the most promiscuous
        // file topping every query at an un-normalized score (the observed 0%
        // retrieval after a re-ingest). Marking here, once, from the warm's own
        // builder keeps the append-only set consistent on every load path.
        //
        // That consistency is what the mark buys, and it is expensive to lose: it
        // is what `warm_ingest_normalization` recognises a layer by, and without it
        // every code scope stays unwarmed — a cold scope once put a repo_map file
        // listing, the most promiscuous member in the corpus, at the top of an
        // unrelated dialogue's context at a score of 13,315,007 against the live
        // conversation's 0.
        //
        // `--disable-layer` means "this layer is out of service". No mark, so no
        // normalization warm; `gathered = false` (set before the builder was
        // cloned, far above), so no gather; and no sweep, because sweeping
        // tombstones turns, which is a MUTATION — the one thing an out-of-service
        // layer must not suffer. Its substrate is left exactly as it stands, so
        // dropping the flag restores the layer whole.
        //
        // `base_conv`'s counterpart for ingestion — one prefilled template per
        // `Folders`/`Files` layer, built here exactly as `base_conv` itself was
        // built above, so `repo_scan`/`code_read` can fork a unit's conversation
        // off it the same way a dialogue turn forks off `base_conv`, instead of
        // each unit independently re-running the schema's "eager section
        // ingestion". See `InferenceState::ingest_bases`.
        //
        // Built unconditionally, even for a `--disable-layer` layer: the upload
        // path (`ZendSession::ingest_uploaded_files`) bypasses the disabled
        // layers' passes entirely and still needs a `Files` layer's base to fork
        // from on the first upload. `Raw` is excluded — it
        // only ever prefills records and never decodes, so it has no reasoning
        // to protect and stays on `raw_read::ingest_raw`'s existing construction.
        let mut ingest_bases: HashMap<String, Sequence> = HashMap::new();
        for il in &ingest_layers {
            if il.mode == IngestMode::Raw {
                continue;
            }
            let layer = proj_builder_refresh
                .id_for_layer(&il.name)
                .ok_or_else(|| anyhow::anyhow!("projection schema missing '{}' layer", il.name))?;
            let group = proj_builder_refresh
                .id_for_group(&il.group)
                .ok_or_else(|| anyhow::anyhow!("projection schema missing '{}' group", il.group))?;
            let base = engine
                .new_conversation_with_projection(
                    &formatted_prompt,
                    proj_builder_refresh.clone(),
                    layer,
                    group,
                    // Append-only ingest: no mid-decode reprojection. Each turn
                    // still assembles its slot from the substrate — slots are
                    // emptied at every seal — so the priming chain, inherited
                    // through `forked_from`, is in every turn's context.
                    utility_config(conv_config.clone()),
                )
                .map_err(|e| anyhow::anyhow!("{} ingest base create: {e}", il.name))?;
            ingest_bases.insert(il.name.clone(), base);
        }
        if load_stopped(&engine, "the ingest bases") {
            return Ok(None);
        }

        // `--self-check`: every stored conversation asked whether it is intact,
        // before the retire sweep, the priming chain and every ingest — so an
        // ingest unit it tombstones is simply read again by them. Each is
        // written out as a record and judged through the dialogue base, which
        // has no history of its own to mix into the record.
        if self_check != SelfCheck::Off {
            progress.set_step(LoadStep::SelfCheck);
            let mut layers: HashSet<LayerId> = ingest_layers
                .iter()
                .filter(|il| ingest_bases.contains_key(&il.name))
                .filter_map(|il| proj_builder_refresh.id_for_layer(&il.name))
                .collect();
            layers.insert(dialogue_layer);
            self_check::run(
                &engine,
                &conv_config,
                &layers,
                base_conv.unsealed_asker(),
                self_check,
                &|done, total| progress.set_step_progress(done, total),
            )?;
        }

        crate::layers::removed::sweep(&engine);
        for il in &ingest_layers {
            if disabled_layers.contains(&il.name) {
                continue;
            }
            if let Some(layer_id) = proj_builder_refresh.id_for_layer(&il.name) {
                engine.mark_layer_append_only(layer_id);
            }
            // Retire an analysis layer's crashed partials before any pool
            // starts. Nothing is in flight at this point, which is why the sweep
            // lives here and not inside the ingest passes, which can overlap a
            // pool mid-flight. A file or folder record is written whole, and a
            // raw record's turns and content hash commit together, so neither
            // has a half-built state to retire.
            match il.mode {
                IngestMode::Folders | IngestMode::Files | IngestMode::Raw => {}
                IngestMode::Derived => {
                    crate::layers::run::retire_crashed_partials(&engine, &il.name)
                }
            }
            // `--wipe-layer <name>`: tombstone EVERY conversation in this layer,
            // not just crashed partials — a targeted alternative to
            // `--wipe-substrate` for exercising the background ingest worker
            // against a real, full-layer backlog without touching anything else
            // (dialogue, the other ingest layers, uploads). Runs before the ingest
            // worker's first pass, which then finds the layer empty.
            // `--disable-layer` still wins: a disabled layer gets no cleanup of
            // any kind, wipe included (see the `continue` above).
            if wiped_layers.contains(&il.name) {
                match il.mode {
                    IngestMode::Folders => crate::repo_scan::wipe_layer(&engine),
                    IngestMode::Files => crate::code_read::wipe_layer(&engine),
                    IngestMode::Derived => {
                        let e = &engine;
                        let all = crate::layers::run::all_of(e, &il.name);
                        let wiped = all
                            .iter()
                            .filter(|tl| e.tombstone_timeline(**tl).is_ok())
                            .count();
                        tracing::info!(layer = %il.name, wiped, "--wipe-layer: every unit of the layer tombstoned");
                    }
                    IngestMode::Raw => tracing::warn!(
                        layer = %il.name,
                        "--wipe-layer: raw layers are not wipeable this way — use \
                         --wipe-substrate, or remove the layer's content folder by hand",
                    ),
                }
            }
        }
        // The raw-layer registry starts empty: a raw layer's records are read
        // only when a mind's folder is read for them.
        let ingest_convs: HashMap<String, IngestConv> = HashMap::new();

        // The branch layers this boot ingests, each with the scope its walk runs
        // under, and each one's group SCOPED: a conversation may retrieve only
        // the units its own base holds (`docs/zend_branch_ingest.md` §8.2), set
        // before each of its turns by `retrieval_scope`. Marked for a disabled
        // layer too — it is out of the gather already, and scoped it could
        // never offer another branch's content if it came back.
        // Every working-set group's share of the budget, as the schema declares
        // it, so the working set can price what it admits: a conversation in
        // one of these groups draws on its group's share, one in no such group
        // is never a member.
        for group in proj_builder_refresh
            .schema()
            .layers
            .iter()
            .flat_map(|l| l.groups.iter())
        {
            if let SelectionRule::WorkingSet { share } = group.selection {
                engine.set_working_set_share(group.id, share);
            }
        }
        // The types of document the dialogue's working set caps at a share of
        // its budget, read from the path a `code_reading` unit is of
        // (`docs/zend_working_set.md` §4.10). One dialogue layer declares the
        // working set.
        if let Some(ws) = proj_builder_refresh
            .schema()
            .layers
            .iter()
            .find_map(|l| l.working_set.as_ref())
        {
            engine.set_working_set_types(ws.type_shares.clone(), PATH_KEY);
        }
        // Each group that draws files out of another's conversations by path
        // (`docs/guidance_layer.md` §1): scoped like the group it draws from,
        // with the files it draws in its scope rather than that group's.
        let mut drawn_groups = Vec::new();
        for group in proj_builder_refresh
            .schema()
            .layers
            .iter()
            .flat_map(|l| l.groups.iter())
        {
            if let Some(draws) = &group.draws {
                engine.set_group_draws(group.id, draws.from);
                engine.mark_group_scoped(group.id);
                drawn_groups.push((group.id, draws.clone()));
            }
        }

        let mut branch_layers: Vec<(IngestLayer, IngestScope)> = Vec::new();
        let mut files_group = None;
        let mut folders_group = None;
        let mut derived_groups = Vec::new();
        for il in &ingest_layers {
            if il.mode == IngestMode::Raw {
                continue;
            }
            let group = proj_builder_refresh
                .id_for_group(&il.group)
                .ok_or_else(|| anyhow::anyhow!("projection schema missing '{}' group", il.group))?;
            // A day layer's units are of no branch, so its group is not scoped.
            let of_files = il
                .spec
                .as_ref()
                .is_none_or(|s| s.unit == crate::layers::UnitKind::File);
            if of_files {
                engine.mark_group_scoped(group);
            }
            if disabled_layers.contains(&il.name) {
                continue;
            }
            let scope = IngestScope::new(&il.folder, max_depth).ignoring(ignore.clone());
            match il.mode {
                IngestMode::Files => files_group = Some(group),
                // Retrieval derives folders at full depth: the layer retains
                // a folder past the bound while a branch holds it, and a held
                // folder must stay retrievable. A key found for a folder never
                // ingested simply matches nothing.
                IngestMode::Folders => folders_group = Some((group, scope.full_depth())),
                IngestMode::Derived if of_files => derived_groups.push(group),
                IngestMode::Raw | IngestMode::Derived => {}
            }
            branch_layers.push((il.clone(), scope));
        }
        // Files are what a drawing group draws by path; one drawing from any
        // other group is scoped and given nothing.
        drawn_groups.retain(|(_, draws)| Some(draws.from) == files_group);
        let retrieval_scope = RetrievalScope::new(files_group, folders_group)
            .with_derived(derived_groups)
            .with_drawn(drawn_groups);
        // Each layer of units compounds on the layers of units below it: their
        // groups are what its units retrieve across conversations
        // (`Substrate::set_ingest_sees`), the scoped ones on the unit's own
        // branches (`layer_sees`).
        let layer_sees = layer_sees(&proj_builder_refresh, &ingest_layers, &disabled_layers);
        for (name, sees) in &layer_sees {
            let (Some(layer), groups) = (
                proj_builder_refresh.id_for_layer(name),
                sees.iter()
                    .map(|(_, g, _)| *g)
                    .collect::<HashSet<GroupId>>(),
            ) else {
                continue;
            };
            engine.set_ingest_sees(layer, groups);
        }
        let layer_sees: HashMap<String, Vec<(String, GroupId)>> = layer_sees
            .into_iter()
            .map(|(name, sees)| {
                let scoped = sees
                    .into_iter()
                    .filter(|(_, _, scoped)| *scoped)
                    .map(|(l, g, _)| (l, g))
                    .collect();
                (name, scoped)
            })
            .collect();
        retrieval_scope.refresh(&engine);

        // Cooperative shutdown during the (long) startup ingest: if a Ctrl-C
        // arrived while ingesting, the loop above broke early. Don't finish
        // building the InferenceState or mark ready — drain the engine HERE, on
        // this loader thread that owns it, exactly as `engine.shutdown()` does at
        // ready (scheduler join → persistence hot→warm→cold full drain → off-thread
        // writer flush). Then return `None`: the loader thread exits without
        // publishing an engine, and `ZendSession::shutdown` (joining this thread)
        // sees `inference` still `None` and knows the drain already happened. The
        // turns ingested so far are durable; the rest re-ingest on the next start.
        if crate::ingest_cancel::cancelled() {
            tracing::warn!(
                "shutdown during startup ingest — draining the engine on the loader thread"
            );
            if let Err(e) = engine.shutdown() {
                tracing::error!("shutdown-during-ingest drain failed: {e}");
            }
            return Ok(None);
        }

        // The titler runs on a single dedicated task fed by this queue. The
        // task owns the titler `Sequence` exclusively (no shared mutex, so
        // title generation never serialises against the request path), and is
        // awaited on shutdown so its in-flight turn unwinds.
        let (titler_tx, titler_rx) = mpsc::channel(TITLER_QUEUE_DEPTH);
        // Build the per-tools-mode projection builders once, up front, so each
        // turn only pays a cheap `Arc` clone instead of re-cloning the
        // ~93-section schema (see `ModeBuilders`).
        let mode_builders = ModeBuilders::build(&proj_builder_refresh)
            .map_err(|e| anyhow::anyhow!("tools-mode projection builders: {e}"))?;
        // Identity scoping rides the tools-mode builders (see `IdentityBuilders`).
        // Empty when the schema has no `identities/` content — then it's a no-op
        // and conversations use the plain tools-mode projection.
        let identity_builders = IdentityBuilders::new(&identity_sections, default_identity);
        tracing::info!(
            n_identities = identity_builders.keeps.len(),
            default = ?identity_builders.default_identity,
            "identity projection scoping ready",
        );
        let think_closer_phrase = tokenizer
            .encode(THINK_CLOSER_PHRASE, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        // Every conversation keeps its own branch per repository; one with none
        // yet starts on the repository's base.
        let base_branches = conv_branches::base_branches(&workspace);
        let seeded = conv_branches::seed_live(&engine, &base_branches);
        tracing::info!(seeded, "conversations given their base branches");
        let state = Arc::new(Self {
            decoder,
            engine,
            substrate,
            conversations: Resident::default(),
            tool_modes: Mutex::new(HashMap::new()),
            target_layer: base_conv.target_layer_name(),
            base_conv,
            ingest_bases,
            titler_tx,
            retitles: PendingRetitles::default(),
            titler_worker: Mutex::new(None),
            memory_writer: Mutex::new(None),
            memory_stop: CancelToken::new(),
            shutting_down: AtomicBool::new(false),
            ingest_convs: Mutex::new(ingest_convs),
            ingest_layers,
            branch_layers,
            layer_sees,
            branch_ingest: BranchIngest::default(),
            retrieval_scope,
            ingest_wake: OnceLock::new(),
            learning_exclusions: AtomicBool::new(false),
            exclusion_learners: Mutex::new(HashMap::new()),
            working_set,
            refresh_builder: proj_builder_refresh,
            refresh_config: conv_config.clone(),
            formatted_prompt,
            mode_builders,
            identity_builders,
            think_closer_phrase,
            tokenizer,
            tool_host: ToolHost::deployed(&workspace, secrets, cluster.as_ref(), agent.as_ref())?
                .running_ops(ops)
                .guarding_defaults(&base_branches),
            repositories,
            workspace,
            base_branches,
            tool_stencils,
            held_stencils: ArcSwap::default(),
            think_steering,
            passthrough: PassthroughCache::new(),
            deletions: Deletions::default(),
            leaving: Leaving::default(),
            resuming: resume::Resuming::default(),
            journaling,
            journals: Journals::default(),
            prewarm: Prewarm::default(),
            memory,
        });
        // A conversation is this member's for one turn at a time, and what it
        // keeps of one goes when the conversation does.
        if let Some(join) = cluster.as_ref() {
            if let Some(member) = join.link().member() {
                let held = Arc::downgrade(&state);
                turn_claims::install(Arc::clone(member), move |key| {
                    if let Some(state) = held.upgrade() {
                        state.forget_conversation(key);
                    }
                });
                // An unlocked conversation's diary calls are held to the
                // member holding what it unlocked (`docs/private_mode.md`
                // §5.4).
                diary_placement::install(Arc::clone(member));
                // A worker's turn moves to a markedly faster member at a round
                // boundary, signed so the member it moves to knows a member
                // sent it (`worker_hop`).
                worker_hop::install(Arc::clone(member), join.derive_key(worker_hop::PURPOSE));
            }
        }
        state.branch_ingest.set_seed_read(seed_read);
        let worker_state = Arc::clone(&state);
        // Spawned on the daemon's runtime — the loader thread holds an enter
        // guard for exactly this kind of arming (see `start_loading`).
        let worker = tokio::spawn(titler_task(worker_state, titler, titler_rx));
        *state.titler_worker.lock().unwrap() = Some(worker);
        // The idle sweep, on the same runtime. It holds the state weakly and
        // ends with it.
        let swept = Arc::downgrade(&state);
        tokio::spawn(async move {
            let mut every = tokio::time::interval(CONV_SWEEP_EVERY);
            loop {
                every.tick().await;
                let Some(state) = swept.upgrade() else {
                    return;
                };
                let released = state.sweep_idle_conversations(Instant::now());
                if released > 0 {
                    tracing::info!(released, "released idle conversations");
                }
            }
        });
        // Kept images no conversation uses, retired after their grace.
        spawn_image_sweep(Arc::downgrade(&state));
        // The shared journal runs a new conversation reads, sealed ahead of
        // it (`docs/shared_journal_sections.md` §7).
        if state.journaling {
            state.prewarm.seed(&state.engine.known_conversations());
            tokio::spawn(prewarm::run(Arc::downgrade(&state)));
        }
        // The journals' calendar: every open conversation folds the periods
        // that have ended since it last looked — a day turning over while
        // nobody talks to it included.
        if state.journaling {
            let folding = Arc::downgrade(&state);
            tokio::spawn(async move {
                let mut every = tokio::time::interval(FOLD_SWEEP_EVERY);
                loop {
                    every.tick().await;
                    let Some(state) = folding.upgrade() else {
                        return;
                    };
                    for conv_id in state.journals.open_ids() {
                        begin_journal_folds(&state, &conv_id, conversation_timeline(&conv_id));
                    }
                }
            });
        }
        Ok(Some(state))
    }
}
