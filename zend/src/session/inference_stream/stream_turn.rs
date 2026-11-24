//! Runs one chat turn as an async task: decoding, tool rounds, sealing, and
//! streaming the reply to the client.
//!
//! The task is async throughout, so it parks rather than pinning a runtime thread.
//! The per-conversation guard it holds across the decode is what stops a second
//! submit for the same conversation from running alongside it. A client that
//! disconnects does not end the turn, so no half turn is sealed on a socket's say.
//! Each round seals its call before its tools run, so a round cut off by a restart
//! is resumed, and the tool loop is bounded by a cap that closes it with a final
//! answer. The stream hides an empty think block, and holds back a tool-call object
//! so the client can render it as a card.

use super::*;

// The submit parameter list, threaded from the HTTP handler through
// `submit`/`submit_with_sampling` to here. Bundling it into a request struct is
// worth doing, but it is a change to the request path rather than a lint fix.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn run_inference_stream(
    state: Arc<InferenceState>,
    conv_id: String,
    user_message: String,
    max_tokens: Option<usize>,
    sampling: Option<candle_conversation::SamplingConfig>,
    force_hires: Option<String>,
    assistant_prefill: Option<String>,
    lossless_kv: bool,
    tools_mode: ToolMode,
    identity: Option<String>,
    selection: candle_conversation::SelectionState,
    // Not from the HTTP handler: how the turn begins. When it resumes a sealed
    // call turn — the boot scan's, or a worker's handed here — `user_message`
    // is unused: the turn's message is the tool output, produced below.
    mut start: TurnStart,
) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
    let (tx, rx) = mpsc::channel::<anyhow::Result<StreamItem>>(64);

    // An async task, not a blocking one: every wait in the body is an await —
    // the turn's event stream, the client channel's backpressure — and the
    // per-conversation lock it holds across the decode is async precisely so
    // this task parks instead of pinning a thread. A client that disconnects
    // drops `rx`; the next send fails, the task returns, and dropping the
    // in-flight `TurnHandle` stops the decode at its next step.
    tokio::spawn(async move {
        // Counted until the task ends, so a leave waits for what the turn
        // writes on its way out (`crate::leaving`).
        let _running = state.leaving.running();
        // A member leaving starts no turn: one begun now would be cut at
        // once, and the gateway places the next ask on another member.
        if state.leaving.is_leaving() {
            let _ = tx
                .send(Ok(StreamItem::Error(LEFT_STATUS.to_string())))
                .await;
            return;
        }
        let msg_preview: String = user_message.chars().take(60).collect();
        tracing::info!(
            conv_id = %conv_id,
            msg_len = user_message.len(),
            images = start.images.len(),
            "inference start  \"{}{}\"",
            msg_preview,
            if user_message.len() > 60 { "…" } else { "" },
        );
        // The message's images, read before its first round: from the
        // substrate when an earlier turn read the same image, by the tower
        // otherwise — a guest job between waves, which this waits out on the
        // blocking pool.
        let mut turn_images = Vec::new();
        if !start.images.is_empty() {
            let _ = tx
                .send(Ok(StreamItem::Status("Reading the images…".into())))
                .await;
            let (reader, refs) = (Arc::clone(&state), std::mem::take(&mut start.images));
            let using = conversation_timeline(&conv_id);
            // Each image the conversation reads is kept while it lasts
            // (`vision::tenancy`), noted before it is read so a delete of
            // another conversation meanwhile never finds it unused.
            let read = tokio::task::spawn_blocking(move || {
                let substrate = reader.engine.conversation().shared();
                for source in &refs {
                    note_use(&substrate, &source.id(), using)?;
                }
                images::resolve(&reader.engine, &substrate, refs)
            })
            .await;
            match read {
                Ok(Ok(read)) => turn_images = read,
                Ok(Err(e)) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!("the message's images: {e}")))
                        .await;
                    return;
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!(
                            "reading the message's images panicked: {e}"
                        )))
                        .await;
                    return;
                }
            }
        }

        let timeline = timeline_for(&conv_id);
        // Only a freshly-forked conv uses the stored identity; a reused conv
        // keeps its in-memory identity.
        let stored_identity = state
            .engine
            .conversation_metadata_value(timeline, "identity");
        // What the fork that mints this conversation hands back beside it: the
        // per-conversation exclusivity guard, and the file stores and mirror
        // that are restored from its events once it is open.
        //
        // The guard is taken inside the fork, before `conversations` hands the
        // new state to anyone: a second request for the same, not-yet-resident
        // conv_id awaits the fork and then parks on this guard. Without it, that
        // request could run a full turn against a `ConvState` whose files are
        // not yet restored from the substrate (`upper.base` still unset) — and
        // then have that turn's writes silently overwritten when this request's
        // own restore, below, replaces the store's contents wholesale. See the
        // note where the guard is picked back up, and
        // `zend_run_iteration_traps.md` for the trace that found this.
        type Minted = (
            OwnedMutexGuard<ConvState>,
            (Arc<RepoFiles>, Arc<Mutex<Mirror>>),
        );
        let (minted_tx, mut minted_rx) = mpsc::unbounded_channel::<Minted>();
        let forked = state
            .conversations
            .get_or_try_init(&conv_id, Instant::now(), || {
                let (state, minted_tx) = (Arc::clone(&state), minted_tx.clone());
                let (conv_id, identity) = (conv_id.clone(), stored_identity.clone());
                async move {
                    tracing::info!(conv_id = %conv_id, "forking new conv from base");
                    // Fork onto the timeline derived from `conv_id` — a stable
                    // hash, so a daemon restart reconnects the client to the
                    // turns the substrate reload recovered for this
                    // conversation (§16.12). An unknown conv_id simply forks
                    // empty. On a blocking thread: the fork is a scheduler
                    // round-trip.
                    let fork_state = Arc::clone(&state);
                    let conv = tokio::task::spawn_blocking(move || {
                        fork_state.base_conv.fork_resuming(timeline)
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("fork task: {e}"))?
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let files = state.tool_host.conversation_files();
                    let mirror = Arc::new(Mutex::new(Mirror::unread()));
                    let arc = Arc::new(ConvLock::new(ConvState {
                        conv,
                        identity,
                        files: Arc::clone(&files),
                        mirror: Arc::clone(&mirror),
                        // An earlier run may have left this conversation's
                        // sections: the first reconcile looks for them.
                        journal: Carried::new(false),
                        seen: FilesSeen::default(),
                    }));
                    let guard = Arc::clone(&arc)
                        .try_lock_owned()
                        .expect("a conversation nothing has been handed yet is locked by nothing");
                    let _ = minted_tx.send((guard, (files, mirror)));
                    Ok::<_, anyhow::Error>(arc)
                }
            })
            .await;
        // The last fork this request ran is the one it minted (an earlier one
        // was dropped for an entry that won the key).
        let mut minted_by_us: Option<Minted> = None;
        while let Ok(m) = minted_rx.try_recv() {
            minted_by_us = Some(m);
        }
        let (forked, minted) = match forked {
            Ok((arc, minted)) => {
                if !minted {
                    tracing::debug!(conv_id = %conv_id, "reusing existing conv");
                }
                (Ok(arc), minted)
            }
            Err(e) => (Err(e), false),
        };
        let (minted_guard, minted_files) = match minted_by_us.filter(|_| minted) {
            Some((guard, files)) => (Some(guard), Some(files)),
            None => (None, None),
        };
        // Record the fork's lineage: the seed alias (`Substrate::SEED_PARENT`),
        // so the priming chain's anchor documents open a brand new
        // conversation — nothing is copied, and the pointer is persisted
        // metadata. The alias is resolved where the lineage is read, to the
        // chain the serving member holds now: a timeline id here named one
        // boot's chain on one member, and the conversation lost its seed on
        // every other member and after the chain was rebuilt (2026-11-11).
        //
        // Best-effort: a conversation that answers without its inherited
        // context is far better than one that refuses to open.
        if minted {
            if let Err(e) = state.engine.set_forked_from_seed(timeline) {
                tracing::warn!(
                    conv_id = %conv_id,
                    "recording fork lineage failed, conversation starts unprimed: {e}",
                );
            }
        }
        let conv_arc = match forked {
            Ok(arc) => arc,
            Err(e) => {
                tracing::error!(conv_id = %conv_id, "fork failed: {e}");
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        // Held across every decode await below, deliberately: this guard IS
        // the per-conversation in-flight exclusivity — a second submit for
        // the same conv_id parks here until this turn loop finishes.
        //
        // For a minted conversation this is `minted_guard`, taken inside the
        // fork before `state.conversations` handed the state to anyone (see the
        // note there) — never re-acquired here, only picked back up, so there
        // is no gap between "findable" and "exclusive" for anything freshly
        // minted. A reused conversation had no reason to hold anything
        // earlier, so it acquires its own guard here — which is, as it always
        // was, the first moment this codepath asks for one.
        let mut cs = match minted_guard {
            Some(guard) => guard,
            None => conv_arc.clone().lock_owned().await,
        };
        // The slot was seeded from this (fresh) timeline before the lineage
        // above existed; ask again so the conversation opens with the priming
        // chain's recurrent memory behind it rather than from nothing.
        //
        // On a blocking thread: the seed is a scheduler round-trip that waits
        // on the queue behind whatever wave is running (measured in the
        // hundreds of ms for its sibling `InstallRecurrentState`), and this
        // task's contract is that every wait in it is an await — see the note
        // where it is spawned. Blocking here would pin a runtime worker and
        // stall every other stream sharing it.
        //
        // `cs` is moved into the closure and handed back rather than
        // re-locked through `conv_arc`: it is already held, and
        // `OwnedMutexGuard` is not reentrant — a second lock through the same
        // Arc from this same logical task would deadlock against itself.
        if minted {
            let cid = conv_id.clone();
            cs = tokio::task::spawn_blocking(move || {
                if let Err(e) = cs.conv.seed_recurrent_from_lineage() {
                    tracing::warn!(
                        conv_id = %cid,
                        "seeding recurrent memory from the priming chain failed: {e}",
                    );
                }
                cs
            })
            .await
            .expect("seed_recurrent_from_lineage task panicked");
        }
        // What a draft of this conversation's journal is put through, so one can
        // start without its lock: the asker its questions fork from.
        if minted && state.journaling {
            let open = Open {
                asker: cs.conv.unsealed_asker(),
            };
            state
                .journals
                .opened(&conv_id, open, || journal_record(&state, timeline));
        }

        // Persist the conv_id ↔ timeline mapping *after* `fork_resuming`
        // has registered the timeline in the substrate — otherwise
        // `set_conv_id` no-ops in-RAM and the sidebar wouldn't see this
        // conversation until the next daemon restart. Idempotent on
        // repeat calls (no-op when the substrate already has it).
        {
            let engine = &state.engine;
            if let Err(e) = engine.set_conversation_conv_id(timeline, &conv_id) {
                tracing::warn!(conv_id = %conv_id, "persist conv_id failed: {e}");
            }
            // A new conversation — or one unarchived since startup — starts on
            // each repository's base; one already working somewhere keeps it,
            // and then this writes nothing.
            conv_branches::seed(engine, timeline, &state.base_branches);
            // A turn is a use: the sidebar lists this conversation first.
            engine.touch_conversation(timeline);
        }

        if let Some((files, mirror)) = minted_files {
            // A conversation built again — after an eviction, after a restart
            // — reads each repository through its own branch, and gets back
            // the changes its tool rounds made there.
            conv_overlay::restore(&state.engine, timeline, &files, &mirror);
            // A task a conversation started begins on its starter's branches
            // and changes, once.
            if let Some(from) = conv_inherit::starter(&state.engine, timeline) {
                let theirs = state.saved_files(&from);
                let took = conv_inherit::take(&state.engine, timeline, &files, &mirror, &theirs);
                tracing::info!(conv_id = %conv_id, %from, took, "a task begins on its starter's work");
            }
            // Its working set is in-memory and did not survive. The seeds
            // resolve again on its own branch and enter provenance at the
            // seeding momentum, and the rest of the momentum rebuilds from
            // the scan.
            if let Some(config) = state.working_set.clone() {
                let open_state = Arc::clone(&state);
                let open_conv = conv_id.clone();
                // The turn's seed dial: off, the conversation opens on no seeds
                // and its working set is what provenance alone brings in.
                let seeding = selection.seeds();
                if let Err(e) = tokio::task::spawn_blocking(move || {
                    let t_open = Instant::now();
                    open_state.engine.working_set_clear(timeline);
                    let clear_ms = t_open.elapsed().as_millis() as u64;
                    let t_scope = Instant::now();
                    // Scope it first. A seed is admitted only as the
                    // conversation's scope offers it, and the scope is
                    // otherwise first set right before its first submit — after
                    // this — so every seed was refused: measured 2026-10-24, no
                    // dialogue had carried a seed since scoping landed.
                    open_state.retrieval_scope.apply(
                        &open_state.engine,
                        timeline,
                        &open_conv,
                        &files,
                    );
                    let scope_ms = t_scope.elapsed().as_millis() as u64;
                    let t_seed = Instant::now();
                    let seeded = if seeding {
                        seed_working_set(&open_state, timeline, &open_conv, &files, &config)
                    } else {
                        0
                    };
                    let seed_ms = t_seed.elapsed().as_millis() as u64;
                    tracing::info!(
                        target: "zend::working_set",
                        timeline = timeline.raw(),
                        seeded,
                        clear_ms,
                        scope_ms,
                        seed_ms,
                        "working set opened: seeds in provenance",
                    );
                })
                .await
                {
                    tracing::warn!(conv_id = %conv_id, "opening the working set panicked: {e}");
                }
            }
        }

        // An explicit request `identity` overrides and is persisted, so later
        // turns (and a daemon restart) keep it. Only writes when it actually
        // changes, to avoid a substrate write every turn.
        if let Some(req_id) = identity {
            // `cs` is already held (see above) — no reason to re-lock through
            // `conv_arc` for what is the same guard under a different name.
            let changed = if cs.identity.as_deref() == Some(req_id.as_str()) {
                false
            } else {
                cs.identity = Some(req_id.clone());
                true
            };
            if changed {
                if let Err(e) = state
                    .engine
                    .set_conversation_metadata(timeline, "identity", &req_id)
                {
                    tracing::warn!(conv_id = %conv_id, "persist identity failed: {e}");
                }
            }
        }

        // The dials this turn ran under, persisted so both a resumed turn and
        // the GUI reopening the conversation use the settings it was last set
        // to rather than the defaults. Written only when they change, as
        // `identity` is — otherwise every turn costs a metadata record.
        {
            let dials = dials_metadata(&selection, tools_mode);
            let engine = &state.engine;
            let differs = engine
                .conversation_metadata_with(timeline, |stored| {
                    dials.iter().any(|(k, v)| stored.get(k) != Some(v))
                })
                .unwrap_or(!dials.is_empty());
            if differs {
                if let Err(e) = engine.set_conversation_metadata_many(timeline, &dials) {
                    tracing::warn!(conv_id = %conv_id, "persist dials failed: {e}");
                }
            }
        }

        // Lossless capture: seal this conversation's turns WITHOUT KV
        // quantization — K/V persist in native R16/F16 so the provenance work
        // gets full-resolution keys. Set before the first turn's residence is
        // allocated so the hot→warm migration inherits it.
        if lossless_kv {
            state.engine.set_timeline_compression(
                timeline,
                Some(candle_conversation::substrate::ConvCompression {
                    lossless: true,
                    level: None,
                    disable_k_override: false,
                    force_k: None,
                    force_v: None,
                }),
            );
        }

        // Hand the titler off to its worker, now that the timeline is
        // registered and the conv_id is in-RAM (so the titler's
        // `set_conversation_label` write composes correctly with the conv_id
        // field of the same Label record). The decode runs in the background,
        // batched alongside the main decode. Enqueue is non-blocking: a full
        // queue or in-progress shutdown drops the job (labels are best-effort).
        let already_labelled = state.engine.conversation_label_of(timeline).is_some();
        if !already_labelled
            && !user_message.is_empty()
            && !state.shutting_down.load(Ordering::Relaxed)
        {
            let job = TitleJob::Title {
                timeline,
                message: user_message.clone(),
            };
            if let Err(e) = state.titler_tx.try_send(job) {
                tracing::debug!("titler queue full or closed — skipping label: {e}");
            }
        }

        // Per-conversation projection swap. Both the prefill projection and the
        // reprojection read the swapped builder, so neither re-introduces a
        // filtered section mid-conversation.
        //
        // `force_hires` is the zend capture aid (force one collection to
        // AllVisible) and wins when present; otherwise apply the composer's
        // tools mode — Comprehensive restores the full catalog (so switching a
        // conversation back from Restricted/None works), Restricted drops the
        // high-risk tools, None drops the whole catalog.
        // `tool_modes` records the mode actually applied, so the projection
        // panel shows the matching tool summary.
        if let Some(ref coll) = force_hires {
            match build_hires_projection(&state, coll, cs.identity.as_deref()) {
                Ok(b) => {
                    cs.conv.set_projection(b);
                    // The capture override forces the full catalog (AllVisible),
                    // so the panel should show the comprehensive summary.
                    state
                        .tool_modes
                        .lock()
                        .unwrap()
                        .insert(conv_id.clone(), ToolMode::Comprehensive);
                    tracing::info!(conv_id = %conv_id, collection = %coll,
                        "force-high-resolution: collection forced to AllVisible");
                }
                Err(e) => {
                    tracing::warn!(conv_id = %conv_id, "force-high-resolution failed: {e}")
                }
            }
        } else {
            // Applied every turn so a mid-conversation dial change takes effect,
            // and both projection and reprojection use it.
            let proj = turn_projection(&state, &conv_id, cs.identity.as_deref(), tools_mode);
            cs.conv.set_projection(proj);
            state
                .tool_modes
                .lock()
                .unwrap()
                .insert(conv_id.clone(), tools_mode);
        }

        let original_user_message = user_message.clone();
        let mut current_message = TurnText::from(user_message);

        // A tool round this daemon never finished. Run its tools now and enter
        // the loop at a tool round rather than at a fresh question, so the
        // response turn projects as it would have: without the worked
        // demonstration, and without re-applying the caller's prefill.
        // Every `cs.conv` touch below is its own synchronous statement. A
        // `&Sequence` is not `Send` — a conversation owns its background
        // cognitive tasks — so one held across the await would make this whole
        // task non-`Send`, and the spawn would not compile.
        let mut start_iteration = 0usize;
        if let Some(Resume {
            call_turn,
            dispatch,
            iteration: resumed_at,
        }) = start.resume
        {
            let calls = resumed_calls(&cs.conv, &conv_id, timeline, call_turn);
            let n_calls = calls.len();
            if calls.is_empty() {
                resume::clear_call_turn(&state.engine, timeline);
                return;
            }
            // The round is finished over the conversation's own files, under
            // the grants of the mode it was held at.
            let ctx = state.tool_host.dialogue_context(
                tools_mode,
                &cs.files,
                &conv_id,
                journal_shelf(&state, &conv_id, timeline),
            );
            let dispatched =
                tokio::task::spawn_blocking(move || run_tool_calls(&ctx, calls, dispatch)).await;
            conv_overlay::save(&state.engine, timeline, &cs.files, &cs.mirror);
            let results = match dispatched {
                Ok(results) => results,
                Err(e) => {
                    tracing::error!(
                        conv_id = %conv_id,
                        call_turn,
                        "resume: tool dispatch panicked: {e}",
                    );
                    resume::clear_call_turn(&state.engine, timeline);
                    return;
                }
            };
            let text = format_tool_responses(&results);
            if text.is_blank() {
                tracing::warn!(
                    conv_id = %conv_id,
                    call_turn,
                    n_calls,
                    "resume: the re-issued round produced no response text",
                );
                resume::clear_call_turn(&state.engine, timeline);
                return;
            }
            // The one window in which the coupling is knowable: the tools have
            // returned and the response turn is certain to follow. Nothing
            // recomputes it afterwards, so without this the call turn and its
            // response never become one exchange.
            if let Err(e) = cs.conv.couple_turn(call_turn) {
                tracing::warn!(conv_id = %conv_id, call_turn, "resume: couple_turn: {e}");
            }
            tracing::info!(
                conv_id = %conv_id,
                call_turn,
                n_calls,
                handed = dispatch == Dispatch::Live,
                "resume: finishing a tool round begun on another run or member",
            );
            current_message = text;
            start_iteration = resumed_at + 1;
        }
        // A fresh request's message is durable before its first round is
        // submitted, so a round cut off before it seals is asked again on the
        // member that resumes it (`crate::resume`, `docs/turn_handover.md` §3).
        // A resumed round's input is its call turn's, named by its own entry.
        // Its images go in the record as the turn's text holds them — a marker
        // line each ahead of the message — and were kept when they were read,
        // so the member that asks it again reads them back by their ids.
        let recorded: String = turn_images
            .iter()
            .map(|i| format!("{}\n", marker(&i.id, i.width, i.height)))
            .chain(std::iter::once(original_user_message.clone()))
            .collect();
        let pending_message = start_iteration == 0 && !recorded.is_empty();
        if pending_message {
            resume::mark_message(&state.engine, timeline, &recorded);
        }
        // Set when the turn stops at a round boundary to move to a faster
        // member: that member finishes the round, so its entry stays.
        let mut handed = false;
        // Set when the member leaves under the turn: the member that takes the
        // conversation resumes it, so its entries stay (`crate::leaving`).
        let mut left = false;
        let mut leaving = state.leaving.watch();

        // The reflection-marker suppression ceiling is per-dial: derive the think
        // mode once, materialise the turn's sampling config (the conversation
        // default carries the resolved marker family), and bake in the penalty —
        // Quick/Balanced suppress the "Wait"/"Hmm"/… family in-block, Deep/
        // Exhaustive leave it 0 (reconsideration is wanted there).
        let think_mode = think_mode_from_selection(&selection);
        // Vary the RNG seed per turn from real entropy. The default base seed is
        // a fixed constant, which makes a whole conversation a deterministic
        // replay — the same context always samples the same tokens, so a turn
        // that lands in a bad attractor can never sample its way out. A fresh
        // per-turn seed restores genuine run-to-run variation. (The per-token
        // `rng_offset` still advances within the turn.)
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or_else(|_| cs.conv.default_sampling().seed);
        let sampling = turn_sampling(
            &cs.conv,
            sampling,
            &selection,
            &state.think_closer_phrase,
            seed,
        );

        // The tool loop runs until the model stops emitting tool calls (i.e.
        // produces a final answer). A workflow is as long as it needs to be, so
        // the only bound is [`MAX_TOOL_ROUNDS`], which sits far above any real
        // one — see the note there for why a bound has to exist at all now that
        // a disconnect no longer ends the turn.
        let mut repeat_guard = RepeatGuard::new();
        // Set once the repeat guard closes the loop: the turn that answers the
        // closing round is the last, and no call it makes runs.
        let mut closing = false;
        // What the catalog offers this turn, and what is out of it for now.
        let offered = crate::tools::offered_tool_names(tools_mode);
        // Where this conversation stands in the nesting of tasks
        // (`docs/zen_tool_reach.md` §6.3): a nested task's rounds' grammar and
        // prompt hold out the task tools.
        let level = level_of(&state.engine.conversation().shared(), &conv_id);
        // The guidance this turn placed and read, logged once it ends.
        let mut guidance_tally = guidance::Tally::default();
        // A new question starts a new loop: the files the last one read may
        // be placed again, since its reads may be summarised away by now. A
        // resumed round goes on with the loop it belongs to.
        if start_iteration == 0 && state.working_set.is_some() {
            let engine_state = Arc::clone(&state);
            if let Err(e) = tokio::task::spawn_blocking(move || {
                engine_state.engine.working_set_clear_withheld(timeline);
            })
            .await
            {
                tracing::warn!(conv_id = %conv_id, "clearing the withheld files panicked: {e}");
            }
        }
        for iteration in start_iteration.. {
            // The closing turn the cap gives runs below it ([`closes_at_cap`]);
            // a turn resumed at or past the cap ends here.
            if iteration >= MAX_TOOL_ROUNDS {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    "tool loop resumed past its round cap — ending the turn on the answer already streamed",
                );
                break;
            }
            // A conversation deleted while its reply runs ends the reply at the
            // next round. A turn runs on with no one listening so a client that
            // reconnects can follow it, but a deleted conversation has no one to
            // come back: its K/V is dead to the compactor, which unlinks the
            // segments holding it, and every further round would reproject turns
            // whose chunks are gone.
            if state.engine.is_timeline_tombstoned(timeline) {
                tracing::info!(
                    conv_id = %conv_id,
                    iteration,
                    "conversation deleted while its reply ran — ending the turn",
                );
                break;
            }
            // The member is leaving: no further round starts here. The round
            // whose tools just ran is named by its resume entry, unanswered,
            // so the member that takes the conversation runs them again and
            // goes on from there.
            if *leaving.borrow() {
                left = true;
                break;
            }
            tracing::debug!(conv_id = %conv_id, iteration, "submitting turn");
            // Collect this turn's projection events (reprojections + decode-end)
            // so they survive a browser reload (served back on hydrate).
            let mut turn_events: Vec<ProjectionEventOut> = Vec::new();
            // The turn that answers the repeat guard's closing round may not
            // open a tool call: its results said none would run, and a call
            // written anyway would be sealed with no answer.
            // An explicit caller config may carry no resolved id; the
            // conversation's own config always does.
            let mut this_turn = sampling.clone();
            if closing {
                let open = match this_turn.tool_call_open_token_id {
                    id if id >= 0 => id,
                    _ => cs.conv.default_sampling().tool_call_open_token_id,
                };
                repeat_guard::closing_answer(&mut this_turn, open);
            }
            // A tool round's user message is the previous turn's results;
            // it projects without the worked demonstration, which there
            // becomes the question the model answers (`tool_round_selection`).
            let mut round_selection = if iteration == 0 {
                selection.clone()
            } else {
                tool_round_selection(&selection)
            };
            // The round's head, timed by part: what stands between a message
            // and its first token before the engine sees the turn
            // (`round head` below).
            let t_head = Instant::now();
            let mut withheld = unavailable(&state.tool_host.availability_facts(), &offered);
            withhold_by_level(&mut withheld.tools, &offered, level);
            // The round's calls to the files it has read or edited are held to
            // what those files hold now (`docs/stencil_tree.md` §9.2), and a
            // task's answer ends in the `Steps:` section of the steps it was
            // given (`answer_steps`). Reading the files and building the
            // grammar is blocking work, and none of it reads the journal: it
            // runs on the blocking pool while the journal is carried, and
            // `triggers_ms` is what is left of it once the journal is.
            let base = round_triggers(&state, think_mode, tools_mode, &withheld);
            let (hold_state, seen, held_files) =
                (Arc::clone(&state), cs.seen.clone(), Arc::clone(&cs.files));
            let holding = tokio::task::spawn_blocking(move || {
                let engine = &hold_state.engine;
                let held = seen.triggers(engine, &base, &held_files);
                engine.ending_with_steps(&held, &answer_steps::of(engine, timeline))
            });
            if state.journaling {
                carry_journal(&state, &conv_id, timeline, &mut cs, &mut round_selection).await;
            }
            let journal_ms = t_head.elapsed().as_millis() as u64;
            // The prompt shows what the grammar lets the turn call, and no more.
            round_selection.withhold_members(withheld.tools.iter().copied());
            if let Some(summary) = tool_summary_section(tools_mode) {
                round_selection.note_summary(SectionId::reserved(summary), withheld.lines.clone());
            }
            let t_triggers = Instant::now();
            let triggers = match holding.await {
                Ok(triggers) => triggers,
                Err(e) => {
                    tracing::warn!(conv_id = %conv_id, "holding a round's values panicked: {e}");
                    round_triggers(&state, think_mode, tools_mode, &withheld)
                }
            };
            let triggers_ms = t_triggers.elapsed().as_millis() as u64;
            let options = candle_conversation::TurnOptions {
                max_tokens,
                sampling: Some(this_turn),
                // Apply the caller's assistant prefill only on the first tool
                // iteration — re-prefilling it on every chained iteration would
                // prevent the model ever reaching a final answer.
                assistant_prefill: if iteration == 0 {
                    assistant_prefill.clone()
                } else {
                    None
                },
                selection: round_selection,
                triggers,
                // The message's images belong to its own turn; a tool round's
                // message is the results.
                images: if iteration == 0 {
                    std::mem::take(&mut turn_images)
                } else {
                    Vec::new()
                },
                ..Default::default()
            };
            // What this turn may retrieve from the branch layers: the units its
            // own base holds. Set before every submit — a tool round can move
            // the base (a commit, a merge, a switch) or change a file. Git reads
            // on a base not yet listed, so on the blocking pool.
            let t_scope = Instant::now();
            let scope_state = Arc::clone(&state);
            let scope_files = Arc::clone(&cs.files);
            let scope_conv = conv_id.clone();
            if let Err(e) = tokio::task::spawn_blocking(move || {
                scope_state.retrieval_scope.apply(
                    &scope_state.engine,
                    timeline,
                    &scope_conv,
                    &scope_files,
                )
            })
            .await
            {
                tracing::warn!(conv_id = %conv_id, "retrieval scope panicked: {e}");
            }
            let scope_ms = t_scope.elapsed().as_millis() as u64;
            let submitted = Instant::now();
            // Logged once, at the round's first token.
            let mut head_logged = false;
            let handle = match cs
                .conv
                .submit_turn_with_options(current_message.clone(), options)
            {
                Ok(h) => h,
                Err(e) => {
                    tracing::error!(conv_id = %conv_id, iteration, "submit_turn failed: {e}");
                    let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
                    return;
                }
            };

            // Stream tokens to the client as they arrive, the round's text
            // decoded token by token (`TextStream`). We track emitted_len (byte
            // offset into the text) so incremental deltas are correct across
            // BPE byte-fallback sequences. Fragments that still contain U+FFFD
            // are held back until the sequence completes.
            let mut text_stream = state.decoder.stream();
            let mut emitted_len: usize = 0;
            // Where the answer starts: past the last `</think>` decoded so far.
            let mut answer_at: usize = 0;
            // Byte offset where the answer's tool-call object begins: once set, the
            // object is held back (not streamed live) and re-emitted wrapped in
            // <tool_call> tags at the flush, so the GUI renders a card instead of
            // showing the bare JSON the model emits when it drops the wrapper.
            let mut hold_from: Option<usize> = None;
            // **A collapsed `<think></think>` must not reach the client.**
            //
            // The engine strips empty think blocks from the text it STORES
            // (`strip_empty_think_blocks`, in `build_turn_layout`), but the
            // stream is assembled from the raw token events and nothing filtered
            // it — so an empty block went out verbatim and rendered as leaked
            // markup. It became the common case with thinking-span projection:
            // older turns have their reasoning windowed out of the K/V, so the
            // model reads a history with no visible reasoning and answers with a
            // collapsed block of its own.
            //
            // Classifying a block needs its `</think>`, which is why this holds
            // rather than filters — but only while the block is still
            // WHITESPACE-ONLY. Real reasoning trips `resolved` on its first
            // non-blank token and streams from then on, so a thinking turn is
            // never held back waiting for its own close.
            let mut think_resolved = false;
            let mut think_progress = ThinkProgress::default();
            let mut done_resp = None;
            // The finished turn's report, sent once its last text is out.
            let mut turn_end: Option<StreamItem> = None;
            let mut turn_error: Option<anyhow::Error> = None;
            // Whether the client stopped reading. Observational only: a turn is
            // not abandoned because nobody is listening. It decodes, runs its
            // tools and seals exactly as it would with the browser attached, so
            // a reload finds the work finished rather than lost — and the
            // substrate never holds half a turn the client's socket decided the
            // length of.
            let mut listener_gone = false;
            // The last "waiting for the GPU" status sent, so a wait reported
            // every second reaches the client only when its text changes.
            let mut last_queued: Option<String> = None;
            // Set when the conversation is deleted while this round runs: the
            // round ends where it stands and the handle goes with it, so the
            // engine stops spending on a reply nobody can come back to.
            let mut deleted = false;
            let mut deletions = state.deletions.watch();

            // Scoped so the stream's borrow of `handle` ends with the loop —
            // `finish_turn` consumes the handle below.
            {
                let mut events_in = std::pin::pin!(handle.stream_async());
                loop {
                    let event = tokio::select! {
                        event = events_in.next() => event,
                        Ok(()) = deletions.changed() => {
                            if state
                                .engine
                                .is_timeline_tombstoned(timeline)
                            {
                                deleted = true;
                                break;
                            }
                            continue;
                        }
                        // The member leaves mid-round: the round stops where it
                        // stands, nothing of it sealed (`crate::leaving`).
                        Ok(()) = leaving.changed() => {
                            if *leaving.borrow() {
                                left = true;
                                break;
                            }
                            continue;
                        }
                    };
                    let Some(event) = event else {
                        break;
                    };
                    match event {
                        TurnEvent::Token(id) => {
                            if !head_logged {
                                head_logged = true;
                                // Message to first token, by what it waited on:
                                // the journal's sections, the round's grammar,
                                // the retrieval scope, then the engine — its
                                // admission, projection, prefill and the first
                                // reprojection.
                                tracing::info!(
                                    conv_id = %conv_id,
                                    iteration,
                                    journal_ms,
                                    triggers_ms,
                                    scope_ms,
                                    engine_ms = submitted.elapsed().as_millis() as u64,
                                    first_token_ms = t_head.elapsed().as_millis() as u64,
                                    "round head"
                                );
                            }
                            if turn_error.is_none() {
                                let from = text_stream.push(id);
                                let text = text_stream.text();
                                if let Some(update) =
                                    think_progress.observe(text, text_stream.tokens())
                                {
                                    let (count, done) = match update {
                                        ThinkUpdate::Running(n) => (n, false),
                                        ThinkUpdate::Done(n) => (n, true),
                                    };
                                    let progress = StreamItem::Think {
                                        tokens: count,
                                        done,
                                    };
                                    if tx.send(Ok(progress)).await.is_err() {
                                        listener_gone = true;
                                    }
                                }
                                // Once the post-</think> answer opens with `{`, it's a
                                // (usually un-tagged) tool-call object: stop streaming
                                // here and hold the rest, so the flush can wrap it.
                                if hold_from.is_none() {
                                    // Only the new bytes, and the tail of the old
                                    // a `</think>` may straddle, are searched; the
                                    // last one found moves the answer's start.
                                    answer_at = answer_start(text, from, answer_at);
                                    let rest = text[answer_at..].trim_start();
                                    if rest.starts_with('{') {
                                        hold_from = Some(text.len() - rest.len());
                                    }
                                }
                                // Resolve a leading think block before anything of it
                                // is emitted. Three outcomes, checked in order: the
                                // block closed while empty (skip it entirely), it has
                                // real content (release and never look again), or it
                                // is still opening (hold — at most a few whitespace
                                // tokens).
                                if !think_resolved {
                                    const OPEN: &str = "<think>";
                                    const CLOSE: &str = "</think>";
                                    match text.find(OPEN) {
                                        None => think_resolved = !text.trim().is_empty(),
                                        Some(open) => {
                                            let inner_at = open + OPEN.len();
                                            match text[inner_at..].find(CLOSE) {
                                                Some(rel) => {
                                                    let inner = &text[inner_at..inner_at + rel];
                                                    if inner.trim().is_empty() {
                                                        // Skip the block and the blank
                                                        // run after it, so the answer
                                                        // does not open on a gap.
                                                        let after = inner_at + rel + CLOSE.len();
                                                        let tail = text[after..].trim_start();
                                                        emitted_len = text.len() - tail.len();
                                                    }
                                                    think_resolved = true;
                                                }
                                                // Open but unclosed: real reasoning the
                                                // moment it is not just whitespace.
                                                None => {
                                                    think_resolved =
                                                        !text[inner_at..].trim().is_empty()
                                                }
                                            }
                                        }
                                    }
                                    if !think_resolved {
                                        continue;
                                    }
                                }
                                let emit_to = hold_from.unwrap_or(text.len());
                                if emit_to > emitted_len
                                    && text.is_char_boundary(emitted_len)
                                    && text.is_char_boundary(emit_to)
                                {
                                    let new_part = &text[emitted_len..emit_to];
                                    if !new_part.contains('\u{FFFD}') {
                                        if tx
                                            .send(Ok(StreamItem::Token(new_part.to_string())))
                                            .await
                                            .is_err()
                                        {
                                            // The client closed the connection. The
                                            // decode carries on: breaking here would
                                            // drop the handle on return, stop the
                                            // scheduler at its next token, and seal a
                                            // truncated answer whose length was decided
                                            // by a socket.
                                            listener_gone = true;
                                        }
                                        emitted_len = emit_to;
                                    }
                                }
                            }
                        }
                        TurnEvent::Done(resp) => {
                            tracing::info!(
                                conv_id = %conv_id,
                                iteration,
                                tokens = resp.stats.tokens_generated,
                                tps    = resp.stats.tokens_per_second as u32,
                                prefill_ms = resp.stats.prefill_ms as u32,
                                "turn complete",
                            );
                            // The finished turn — usage + finish_reason — goes to
                            // the client after the turn's last text: the held
                            // tail and a cut-off call's close are still to come,
                            // and a reader that ends the turn on this item would
                            // take them for the next turn's.
                            turn_end = Some(StreamItem::TurnEnd {
                                usage: Usage::for_turn(
                                    resp.stats.context_tokens,
                                    resp.stats.tokens_generated,
                                ),
                                finish: resp.stats.finish,
                            });
                            done_resp = Some(resp);
                        }
                        TurnEvent::Error(e) => {
                            let msg = format!("{e}");
                            tracing::error!(conv_id = %conv_id, iteration, "scheduler error: {msg}");
                            // A per-layer KV divergence the engine has already
                            // decided it cannot trust a repair for
                            // (`heal_tail_divergence`/`assert_sealed_layers_aligned`
                            // in candle-nn) is not a transient hiccup — the
                            // sequence's K/V is inconsistent across layers and
                            // every future turn on it fails identically. Rather
                            // than surface the internal diagnostic as if the
                            // assistant said it, tell the user plainly and stop:
                            // no retry here would do anything but repeat it.
                            let shown = if msg.contains("no repair from here can be trusted") {
                                "This conversation's memory has become corrupted and can't \
                                 be recovered. Please start a new conversation."
                                    .to_string()
                            } else {
                                msg.clone()
                            };
                            // Its own item, so the client shows it beside the
                            // reply rather than dropping the connection.
                            let _ = tx.send(Ok(StreamItem::Error(shown.clone()))).await;
                            turn_error = Some(anyhow::anyhow!("{shown}"));
                            // Do not return — drain the iterator so the channel
                            // closes cleanly before we decide what to do with the
                            // conversation state.
                        }
                        TurnEvent::HealthWarning(msg) => {
                            tracing::warn!(conv_id = %conv_id, "decode health: {msg}");
                        }
                        // A mid-decode reprojection — stream it straight to the GUI
                        // timeline as it happens (a dot per reprojection).
                        TurnEvent::Projection(event) => {
                            let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let out = ProjectionEventOut::answer(seq, event);
                            turn_events.push(out.clone());
                            let _ = tx.send(Ok(StreamItem::Projection(out))).await;
                        }
                        TurnEvent::PrefillProgress {
                            tokens_done,
                            tokens_total,
                        } => {
                            let progress = StreamItem::Prefill {
                                done: tokens_done,
                                total: tokens_total,
                            };
                            if tx.send(Ok(progress)).await.is_err() {
                                listener_gone = true;
                            }
                        }
                        // The prefill is waiting on the GPU: say so, and for how
                        // long, instead of a silent stream.
                        TurnEvent::PrefillQueued {
                            ahead_tokens,
                            ahead_turns,
                            waiting_ms,
                            ..
                        } => {
                            let status = queued_status(ahead_tokens, ahead_turns, waiting_ms);
                            if last_queued.as_deref() != Some(status.as_str()) {
                                last_queued = Some(status.clone());
                                if tx.send(Ok(StreamItem::Status(status))).await.is_err() {
                                    listener_gone = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            if deleted {
                tracing::info!(
                    conv_id = %conv_id,
                    iteration,
                    "conversation deleted mid-round — ending the turn now",
                );
                // As a turn that ended without its Done: the scope it was given
                // goes with it.
                if let Some(end) = turn_end.take() {
                    let _ = tx.send(Ok(end)).await;
                }
                state.engine.clear_retrieval_scope(timeline);
                break;
            }
            // Dropping the round's handle, which the break does, withdraws it
            // from the engine: its reply is never sealed.
            if left {
                state.engine.clear_retrieval_scope(timeline);
                break;
            }

            if listener_gone {
                tracing::info!(
                    conv_id = %conv_id,
                    iteration,
                    "client went away mid-stream — the turn runs to completion",
                );
            }

            let resp = match done_resp {
                Some(r) => r,
                None => {
                    if turn_error.is_none() {
                        // Scheduler closed the channel without Done — the
                        // conversation's turn_in_flight flag is stuck.
                        tracing::error!(
                            conv_id = %conv_id,
                            iteration,
                            "turn ended without Done or Error — evicting conversation",
                        );
                        let _ = tx
                            .send(Ok(StreamItem::Error(
                                "Generation ended unexpectedly. Your next message will start fresh."
                                    .to_string(),
                            )))
                            .await;
                    }
                    // Evict the conversation so the next request forks fresh
                    // rather than hitting the in-flight guard.  Dropping `handle`
                    // here (via return) closes event_rx and stops the scheduler.
                    // The scope it was given goes with it: the next request
                    // sets its own before its first turn.
                    drop(cs);
                    state.conversations.remove(&conv_id);
                    state.journals.closed(&conv_id);
                    state.engine.clear_retrieval_scope(timeline);
                    return;
                }
            };

            // Flush the tail.  If we held back the answer's tool-call object, emit
            // it wrapped in <tool_call> tags so the GUI renders a card (the model
            // dropped the wrapper); a held object that isn't a recognized call is
            // flushed verbatim by `render_held_tail`.  Otherwise flush the raw tail
            // the byte-fallback guard held back.
            if let Some(hf) = hold_from {
                if resp.text.is_char_boundary(hf) {
                    if let Some(out) = render_held_tail(&resp.text[hf..]) {
                        let _ = tx.send(Ok(StreamItem::Token(out))).await;
                    }
                }
            } else if resp.text.len() > emitted_len && resp.text.is_char_boundary(emitted_len) {
                let tail = &resp.text[emitted_len..];
                if !tail.is_empty() {
                    let _ = tx.send(Ok(StreamItem::Token(tail.to_string()))).await;
                }
            }

            // Every call the answer made, in order — including any that could
            // not be read, which are answered with an error rather than dropped
            // (see `tool_round`), so a broken call is a failed round the model
            // and the GUI both see, not a turn that silently reads as final.
            let round = tool_round::plan(&resp.answer);
            // Force-high-resolution is a capture mode: seal the first turn (the
            // tool invocation) into the substrate as the dataset baseline, but
            // do NOT execute the tools — capture-only, so `code_run` / network
            // tools have no real side effects.
            //
            // A turn answering the repeat guard's closing round is final too:
            // its results told the model no further call runs.
            if closing && !round.is_empty() {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    n_calls = round.len(),
                    "the turn after the repeat guard closed the loop made tool calls — not run",
                );
            }
            // A turn that failed ends the reply too: what it wrote is cut off
            // wherever the error landed, and a call in it was never finished.
            let is_final =
                round.is_empty() || force_hires.is_some() || closing || turn_error.is_some();
            // A call the turn stopped writing has no `</tool_call>` in the
            // stream, and the GUI reads everything after an unclosed opener —
            // the rounds that follow included — as that call's JSON. Close it
            // in the stream so its card ends where the call did.
            if !is_final && tool_round::ends_inside_a_call(&resp.answer) {
                let _ = tx
                    .send(Ok(StreamItem::Token("\n</tool_call>".to_string())))
                    .await;
            }
            if let Some(end) = turn_end.take() {
                let _ = tx.send(Ok(end)).await;
            }

            // If tools will run, tell the GUI *now* — before sealing/persisting
            // this turn — so the in-flight tool cards show their spinner across
            // the whole execution window (seal + persist + dispatch), not just
            // the instant before dispatch (which an instant tool coalesces away,
            // making the card jump straight to a result).
            let tool_names: Vec<String> = if is_final {
                Vec::new()
            } else {
                round.iter().map(|s| s.name().to_string()).collect()
            };
            if !is_final {
                let _ = tx
                    .send(Ok(StreamItem::Tool(ToolStatusOut {
                        phase: "running",
                        tools: tool_names.clone(),
                        results: Vec::new(),
                        tokens: Vec::new(),
                    })))
                    .await;
            }

            if let Err(e) = cs.conv.finish_turn(handle, &resp) {
                tracing::warn!(conv_id = %conv_id, "finish_turn error: {e}");
            }
            // Where the round's time went, under its turn: what a slow task is
            // read by before anything is changed to speed it up.
            if let Some(turn) = resp.seal.as_ref().and_then(|s| s.turn_index) {
                let timing = RoundTiming::of(
                    submitted.elapsed().as_millis() as u64,
                    resp.stats.turn_prefill_tokens as u64,
                    resp.stats.prefill_ms as u64,
                    resp.stats.tokens_generated as u64,
                    resp.stats.decode_ms as u64,
                );
                if let Err(err) = state.engine.set_conversation_metadata(
                    timeline,
                    &round_timing::key(turn),
                    &timing.value(),
                ) {
                    tracing::warn!(conv_id = %conv_id, "recording the round's timing failed: {err}");
                }
            }
            // A failed turn is kept — what it wrote is what the user watched
            // arrive — and so is why it stopped, under the turn it belongs to,
            // for the history to show beside it after a reload.
            if let (Some(e), Some(turn)) =
                (&turn_error, resp.seal.as_ref().and_then(|s| s.turn_index))
            {
                if let Err(err) = state.engine.set_conversation_metadata(
                    timeline,
                    &turn_error_key(turn),
                    &e.to_string(),
                ) {
                    tracing::warn!(conv_id = %conv_id, "recording the turn's error failed: {err}");
                }
            }

            // One projection event per decode: the scored reprojection now
            // reflects what provenance materialized for this turn (finish_turn
            // refreshed the BDP scores). Stream it so the GUI drops a timeline
            // dot the moment the turn seals — no separate fetch needed.
            // Which of this conversation's turns it selected is also what tells
            // the journal whether the dialogue has started losing sight of turns
            // no entry covers.
            let mut selected_turns: Option<Vec<u32>> = None;
            if let Some(event) = cs.conv.projection_event(&resp.stats) {
                selected_turns = Some(pressure::selected_turns(&event, timeline));
                let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let out = ProjectionEventOut::answer(seq, event);
                turn_events.push(out.clone());
                let _ = tx.send(Ok(StreamItem::Projection(out))).await;
            }
            // Persist this turn's events to the substrate redo log so the
            // timeline survives a browser reload AND a daemon restart. Keyed to
            // the just-sealed turn; served back on hydrate via the substrate.
            if !turn_events.is_empty() {
                let events: Vec<candle_conversation::ProjectionEvent> =
                    turn_events.into_iter().map(|o| o.event).collect();
                if let Err(e) = cs.conv.persist_projection_events(&events) {
                    tracing::warn!(conv_id = %conv_id, "persist projection events: {e}");
                }
            }

            // This turn asked for tools that have not run yet. Name it before
            // the commit below, so one group-commit makes the call turn and the
            // note of it durable together: an entry still in the writer queue
            // while the tools run would be lost by the very crash it exists to
            // survive. `crate::resume` finds it on the next boot and finishes
            // the round.
            if !is_final {
                if let Some(call_idx) = resp.seal.as_ref().and_then(|s| s.turn_index) {
                    resume::mark_call_turn(&state.engine, timeline, call_idx);
                }
            }
            // The request's first round sealed: its message is in the substrate
            // now, so the pending message goes in the same commit — handed to
            // the resume entry above when the round called tools, and to
            // nothing when it was the final response.
            if pending_message && iteration == start_iteration {
                resume::clear_message(&state.engine, timeline);
            }

            // Group-commit the substrate redo log: the just-sealed turn is
            // now durable on disk, so a crash or restart resumes it intact.
            // The commit waits for the writes queued before it, and nothing
            // else waits on the commit.
            let conversation = state.engine.conversation();
            if let Err(e) = conversation.commit_persistence() {
                tracing::warn!(conv_id = %conv_id, "persistence commit error: {e}");
            }

            // A worker's turn moves to a markedly faster member here: its call
            // turn is sealed and its commit queued above, and its tools have not
            // run, so the member it moves to runs them (`worker_hop`). Before
            // the draft below, so the member it moves to drafts this stretch
            // and nothing here runs on after the move. A turn the guard or the
            // cap is closing answers here instead.
            // Nor a round the cap is about to close: the member it moves to
            // finishes it without the cap's checks.
            if start.may_hand && !closing && !is_final && !closes_at_cap(iteration) {
                let call_idx = resp.seal.as_ref().and_then(|s| s.turn_index);
                if let (Some(to), Some(call_turn)) = (worker_hop::target(), call_idx) {
                    tracing::info!(
                        conv_id = %conv_id,
                        iteration,
                        call_turn,
                        to = %format!("{to:016x}"),
                        "a worker's turn moves to a markedly faster member",
                    );
                    handed = true;
                    let _ = tx
                        .send(Ok(StreamItem::Handed {
                            to,
                            call_turn,
                            iteration,
                        }))
                        .await;
                    break;
                }
            }

            // The sealed turn may make a draft due: always when it ends the
            // request, and within a long one by the cadence or the selection
            // pressure. The draft runs on its own, on forks of this
            // conversation at `Normal`; this reply does not wait for it.
            if state.journaling {
                if let Some(newest) = resp.seal.as_ref().and_then(|s| s.turn_index) {
                    // Each round seals one turn, so the request opened at the
                    // turn `iteration` rounds back — a resumed round's too.
                    let rounds = u32::try_from(iteration).unwrap_or(u32::MAX);
                    // The reply's notes are the request's next steps, carried
                    // from the next round's head; a final reply clears them
                    // for the entry that replaces them (`journal::steps`).
                    state
                        .journals
                        .of(&conv_id, || journal_record(&state, timeline))
                        .lock()
                        .unwrap()
                        .steps_mut()
                        .sealed(
                            newest.saturating_sub(rounds),
                            own_notes(&resp.text),
                            is_final,
                        );
                    let cadence = Cadence::Due {
                        newest,
                        selected: selected_turns.as_deref(),
                        request: Request {
                            from: newest.saturating_sub(rounds),
                            finished: is_final,
                        },
                    };
                    match begin_journal_draft(&state, &conv_id, timeline, cadence) {
                        Ok(span) => tracing::info!(
                            target: "zend::journal",
                            conv_id = %conv_id,
                            from_part = span.from_turn,
                            to_part = span.to_turn,
                            parts = span.turns.len(),
                            "journal draft started",
                        ),
                        // No draft holds the journal: a period that ended
                        // since the last round folds now.
                        Err(_) => {
                            begin_journal_folds(&state, &conv_id, timeline);
                        }
                    }
                }
            }

            if is_final {
                // The request turn has ended: title the conversation again from
                // this exchange, in the background (`titles`).
                request_retitle(&state, timeline, &original_user_message, &resp.text);
                break;
            }

            tracing::info!(
                conv_id = %conv_id,
                iteration,
                n_calls = round.len(),
                "dispatching tool calls",
            );
            // The "running" notice was already sent above (before the seal).  Run
            // the tools — file and process work, so on the blocking pool while
            // this task stays parked — then the "done" notice clears the spinner
            // and carries each result so the cards resolve immediately, before
            // the post-stream hydrate.
            let n_calls = round.len();
            let ctx = state.tool_host.dialogue_context(
                tools_mode,
                &cs.files,
                &conv_id,
                journal_shelf(&state, &conv_id, timeline),
            );
            // A program it runs is told on the stream as it goes: a watcher's
            // line, dropped when the stream is full and never the model's.
            let progress = tx.clone();
            let watch: RunWatch = Arc::new(move |line| {
                let _ = progress.try_send(Ok(StreamItem::Tool(ToolStatusOut {
                    phase: "progress",
                    tools: vec![line],
                    results: Vec::new(),
                    tokens: Vec::new(),
                })));
            });
            let ran = tokio::task::spawn_blocking(move || {
                run_watch::watching(watch, || tool_round::run(&ctx, round))
            })
            .await;
            // Whatever the round changed outlives this conversation's state.
            conv_overlay::save(&state.engine, timeline, &cs.files, &cs.mirror);
            let mut results = match ran {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(conv_id = %conv_id, iteration, "tool dispatch panicked: {e}");
                    let _ = tx
                        .send(Ok(StreamItem::Token(format!(
                            "\n\n⚠ tool dispatch failed: {e}"
                        ))))
                        .await;
                    break;
                }
            };
            // A round that published to origin or fetched from it moved the
            // tracking refs itself, so no origin probe will see a difference:
            // wake the ingest worker, which reads branches from those refs.
            if results.iter().any(|r| moves_branches(&r.call.name)) {
                if let Some(wake) = state.ingest_wake.get() {
                    crate::ingest_worker::request(wake);
                }
            }
            withhold_read_files(&state, timeline, &cs.files, &results).await;
            let guidance_reads = drawn_reads(&state, &results);
            guidance_tally.place(placed_guidance(&state, timeline));
            for read in &guidance_reads {
                guidance_tally.read(read.path());
            }
            calibrate_guidance_reads(
                &state,
                &cs.files,
                guidance_reads,
                &original_user_message,
                &resp.answer,
            );
            cs.seen.note(&results);
            let now_out = unavailable(&state.tool_host.availability_facts(), &offered);
            announce_returned(&mut results, &withheld, &now_out);
            if repeat_guard.screen(&mut results) == Verdict::Close {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    "{} rounds running made only calls already made {} times with the same \
                     result — the next turn answers without tools",
                    repeat_guard::STOP_AFTER,
                    repeat_guard::NOTICE_FROM - 1,
                );
                closing = true;
                let _ = tx.send(Ok(StreamItem::Closing(CLOSED_BY_REPEATS))).await;
            }
            if !closing && closes_at_cap(iteration) {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    "the next turn is the last the round cap allows — it answers without tools",
                );
                repeat_guard::cap_notice(&mut results);
                closing = true;
                let _ = tx.send(Ok(StreamItem::Closing(CLOSED_BY_CAP))).await;
            }
            // A worker `/mcp` started reads what Claude sent it — a stop, a
            // nudge — at this boundary, never mid-decode.
            let from_claude = owner_of(&conv_id).map(|_| state.engine.conversation().shared());
            // Claude stopping it closes the loop here, as the cap does. A
            // round the guard or the cap closed too is still Claude's stop,
            // so the worker settles stopped. A round that ran nothing has no
            // results to carry it and ends below; the stop is then reported
            // unreached when the worker settles.
            if let Some(shared) = from_claude.as_ref().filter(|_| !results.is_empty()) {
                if let Some(reason) = stops::take(shared, &conv_id) {
                    tracing::info!(
                        conv_id = %conv_id,
                        iteration,
                        "Claude stopped the task — the next turn answers without tools",
                    );
                    stops::notice(&mut results, &reason);
                    closing = true;
                    let _ = tx.send(Ok(StreamItem::Closing(CLOSED_BY_CLAUDE))).await;
                }
            }
            // Each result measured as its own block of the turn it is about to
            // become — the length its card shows.
            let tokens = results
                .iter()
                .map(|r| {
                    cs.conv
                        .encoded_len(&format_tool_responses(std::slice::from_ref(r)))
                })
                .collect();
            let _ = tx
                .send(Ok(StreamItem::Tool(ToolStatusOut {
                    phase: "done",
                    tools: tool_names,
                    results: results.iter().map(|r| r.response.clone()).collect(),
                    tokens,
                })))
                .await;
            current_message = format_tool_responses(&results);
            // A tool round that produces no response text must NOT spawn a
            // follow-up turn. `format_tool_responses` wraps every real result in
            // `<tool_response>…</tool_response>`, so an empty `current_message`
            // means no tool actually ran (the model emitted a tool call that was
            // filtered/failed). Submitting it would seal a phantom turn whose grid
            // is nothing but boundary glue (no user content), which the model then
            // "answers" with a generic greeting — derailing the conversation. The
            // answer already streamed for this turn stands; end the loop here.
            if current_message.is_blank() {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    n_calls,
                    "tool round yielded no response text; ending turn loop instead \
                     of submitting an empty follow-up turn",
                );
                break;
            }
            // Guidance Claude queued for a worker `/mcp` started joins real
            // results only — after the check above, so a nudge never turns an
            // empty round into a turn, and one left queued when the loop ends
            // is reported by the worker as undelivered.
            if let Some(shared) = &from_claude {
                current_message = nudges::follow(current_message, shared, &conv_id);
            }

            // The round-trip is now certain — the tools returned real output and
            // the follow-up turn is guaranteed to be submitted below — so couple
            // the just-sealed call turn to it. This is the only window where that
            // is knowable: at the call turn's seal the tools had not run, and by
            // the response turn's seal the call turn's record is already durable.
            // Couple the call turn by its OWN sealed index (`resp.seal.turn_index`),
            // never by "the last turn": the async summariser can append a summary
            // turn in this window, so the newest index may not be the call turn.
            if let Some(call_idx) = resp.seal.as_ref().and_then(|s| s.turn_index) {
                if let Err(e) = cs.conv.couple_turn(call_idx) {
                    tracing::warn!(conv_id = %conv_id, "couple_turn: {e}");
                }
            } else {
                tracing::warn!(
                    conv_id = %conv_id,
                    "tool call turn sealed without an index — cannot couple its response"
                );
            }
        }

        // The round is over, however it ended — a final answer, a tool round
        // that yielded nothing, a dispatch that panicked. The entry only ever
        // means "tools are in flight", so it goes now — unless the turn moved,
        // when the member finishing the round clears it. A crash before this
        // reaches disk is covered by the scan's own check: an entry whose turn
        // arrived is tombstoned rather than acted on.
        //
        // A turn the member left under keeps its entries — the call turn's,
        // the pending message — for the member that takes the conversation,
        // and the conversation is marked handed so that member resumes it now
        // (`docs/turn_handover.md` §4). Committed here: the leave waits for
        // this task, and the drain after it sends what is committed.
        if left {
            // A worker's own record says it was handed on (`mcp::workers`).
            if conv_names(&conv_id).is_none() {
                resume::mark_handed(&state.engine, timeline, workers::now_ms());
            }
            if let Err(e) = state.engine.conversation().commit_persistence() {
                tracing::warn!(conv_id = %conv_id, "persistence commit error: {e}");
            }
            tracing::info!(
                conv_id = %conv_id,
                "the member leaves: the turn stopped where it stood, for another member to resume",
            );
            let _ = tx.send(Ok(StreamItem::Left)).await;
            return;
        }
        if !handed {
            resume::clear_call_turn(&state.engine, timeline);
        }

        // The answering round's placements count too: what was placed as the
        // turn answered without reading it is relevant or idle.
        guidance_tally.place(placed_guidance(&state, timeline));
        if let Some(ws) = &state.working_set {
            log_guidance_tally(&conv_id, &guidance_tally, ws.min_momentum);
        }

        // Title generation has already been fired in parallel from
        // `submit_with_sampling` — it doesn't depend on the main
        // convo's state, so we don't repeat it here.
        let _ = original_user_message;
    });

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}
