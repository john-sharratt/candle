//! Writing the mind's documents into the substrate a window at a time.
//!
//! Documents are created a window at a time, every turn in the window is
//! submitted before any is awaited, and each is drained and sealed afterward, so
//! the window runs as one prefill forward rather than one per document. A
//! document's ledger entry is written only when its turn seals, so an interrupted
//! run is retried on the next boot instead of being written a second time. Each
//! window gives its arena back before the next one begins, so the ingest does not
//! slow down as it goes.

use super::*;

/// The marker a prefilled turn's projection is recorded against.
///
/// The same string zend uses (`zend/src/tool_def.rs`), duplicated rather than
/// shared because it is a wire constant of the conversation layer that neither
/// daemon owns — and taking a dependency on the other daemon to reach it would
/// be far stranger than the fourteen bytes.
pub(crate) const PROJECTION_MARKER: &str = "<|projection|>";

/// Drain a turn to `Done`, keeping the projection events it streams.
///
/// **The events are the provenance signature.** Each one records what a
/// projection selected out of what was already in the substrate at that moment —
/// so an episode of a life carries the world documents it happened against, and
/// a later scan over that memory has a hook rather than having to rediscover the
/// relationship from surface text.
///
/// Borrows the handle, because `finish_turn` consumes it: the stream has to be
/// fully drained before the turn can be sealed.
pub(crate) async fn drain(
    handle: &TurnHandle,
    addr: &str,
) -> (Option<TurnResponse>, Vec<ProjectionEvent>) {
    use tokio_stream::StreamExt;

    let mut events = Vec::new();
    let mut response = None;
    let mut stream = std::pin::pin!(handle.stream_async());
    while let Some(ev) = stream.next().await {
        match ev {
            TurnEvent::Projection(e) => events.push(e),
            TurnEvent::Done(r) => {
                response = Some(r);
                break;
            }
            TurnEvent::Error(e) => {
                tracing::warn!("{addr}: prefill failed — {e}");
                break;
            }
            _ => {}
        }
    }
    (response, events)
}

/// Persist a turn's provenance signature, after the turn is sealed.
///
/// The seal-time projection is appended to whatever streamed during the turn:
/// the streamed events are the composition as it was being built, and this is
/// what it settled at. zend persists the same pair for the same reason.
///
/// Failure is logged, never propagated — a document whose signature did not
/// persist is still a document in the substrate, and refusing the whole ingest
/// over a lost hook would trade something for nothing.
pub(crate) fn persist_signatures(
    conv: &mut Sequence,
    response: &TurnResponse,
    events: &mut Vec<ProjectionEvent>,
    addr: &str,
) {
    if let Some(e) = conv.projection_event(&response.stats) {
        events.push(e);
    }
    if events.is_empty() {
        return;
    }
    if let Err(e) = conv.persist_projection_events(events) {
        tracing::warn!("{addr}: provenance signature not persisted — {e:?}");
    }
}

/// Write one layer's documents into the substrate, a windowful at a time.
///
/// Returns how many failed. The shape is zend's calibration ingest and the
/// reasoning is in [`crate::engine::ingest`]: **one conversation per document**,
/// created a windowful at a time in a single pipelined call, every turn submitted
/// before any is awaited, so the whole window co-batches into one prefill
/// forward.
///
/// The order of operations is the whole point and is easy to undo by accident:
///
/// 1. create the batch — one round trip for all of them;
/// 2. submit *every* turn, collecting handles — **nothing is awaited here**;
/// 3. only then drain each handle and seal it.
///
/// Awaiting inside step 2 — the obvious way to write it — serialises the window
/// back into one forward per document, which is exactly the 3.5-seconds-each
/// this replaced. The sequences are dropped after sealing: the turns are in the
/// substrate by then and the gather reaches them across conversations, so
/// holding the sequence would only hold its K/V.
/// **The engine, shared rather than borrowed, and never locked.**
///
/// The ingest used to hold a `&ConversationEngine` for the whole of startup,
/// and the shared handle was only published once every layer had been ingested
/// and every character woken. So for the twenty minutes a real mind takes to
/// load, every engine-backed route answered 503 — including the guest routes,
/// which is exactly when somebody wants to look at the daemon.
///
/// No mutex around it, because the engine synchronises itself — every method
/// takes `&self` — and a lock out here only serialised its callers. Opening a
/// conversation waits on the scheduler for a slot, which answers between waves,
/// so behind a lock every opening on the daemon queued a wave apart: a draw's
/// four compliance questions took a minute to open and one step to answer.
pub(super) type SharedEngine = Arc<ConversationEngine>;

/// Retire every live conversation in the substrate — `--wipe-conversations`.
///
/// Every kind goes: the characters' own conversations, their dreams, every
/// mind-layer document, every life episode and the beliefs, relationships and
/// intentions it left. Listed from the substrate's timelines rather than its
/// named conversations, because an ingested document carries no `conv_id`.
///
/// The characters are not conversations — each is an `Npc` record in the same
/// log — so the cast survives it untouched.
///
/// Returns how many it retired.
pub(super) fn wipe_conversations(engine: &SharedEngine) -> usize {
    let mut retired = 0;
    for timeline in engine.live_conversations() {
        match engine.tombstone_timeline(timeline) {
            Ok(()) => retired += 1,
            Err(e) => {
                tracing::warn!(
                    "conversation {timeline} could not be retired: {e:?} — it stays live"
                )
            }
        }
    }
    retired
}

/// Where an ingest records what it wrote: the mind's root, so each document's
/// conversations are marked with its path (see `mind_record`), and the ledger.
pub(super) struct Record<'a> {
    pub(super) mind: &'a Path,
    // Each document's hash is recorded **as its turn seals**, so a document that never
    // reached the substrate is not claimed to be in it. See `ingest::Pending`.
    pub(super) ledger: &'a Ledger,
}

/// The shared layer group a source's documents are written to — see
/// [`schema::document_target`]. `None` for a character's own content and for a
/// layer with no group of its own, which keep the live target.
pub(super) fn shared_target(
    p: &schema::Projection,
    source: &ingest::LayerSource,
) -> Option<(LayerId, GroupId)> {
    match source.owner {
        Some(_) => None,
        None => schema::document_target(&p.builder, &source.name),
    }
}

/// Retire `source`'s documents still standing in the live group, and have the
/// ledger forget the source's files so they are written again — into their own
/// layer's group. Answers how many conversations were retired.
///
/// **A timeline's group is fixed when it is created**, so a document written to
/// the wrong group is moved by writing it again. Found by its tags, which every
/// document's turn carries and no character's does.
pub(super) fn retire_misplaced(
    engine: &SharedEngine,
    live: GroupId,
    source: &ingest::LayerSource,
    ledger: &Ledger,
) -> usize {
    let e = &engine;
    let misplaced: Vec<TimelineId> = e
        .group_conversations(live)
        .into_iter()
        .filter(|t| {
            e.turn_tag_lists(*t)
                .first()
                .is_some_and(|tags| source.wrote(tags))
        })
        .collect();
    let mut retired = 0;
    for t in misplaced {
        match e.tombstone_timeline(t) {
            Ok(()) => retired += 1,
            Err(err) => {
                tracing::warn!("conversation {t} not retired from the live group — {err:?}")
            }
        }
    }
    if retired > 0 {
        ledger.forget_under(&source.dir);
        ledger.flush();
    }
    retired
}

pub(super) async fn ingest_layer(
    engine: &SharedEngine,
    record: Record<'_>,
    base_config: &SequenceConfig,
    proj: Option<&schema::Projection>,
    source: &ingest::LayerSource,
    turns: &[ingest::Pending],
    p: &LoadProgress,
) -> anyhow::Result<usize> {
    let Record { mind, ledger } = record;
    let mut cfg = base_config.clone();
    // A layer document is content, not dialogue: there is no recent-turn tail to
    // carry, and the gather is the only way back to it.
    cfg.context_window_turns = 0;

    let tags = source.tags();
    // **The mind's real projection, not a synthetic one.**
    //
    // A projection runs against the conversation's own schema, and
    // `Builder::for_plain_prompt` declares one layer holding one section — so a
    // document written under it has nothing to gather from and produces no
    // signature. That is how 1,818 documents came to be written and unreachable.
    let synthetic = match proj {
        Some(_) => None,
        None => {
            let prompt = source.prompt();
            // Its frame id comes from its text, so a layer's documents are
            // written under the layer's own prompt rather than under whichever
            // plain prompt sealed a shared id first.
            let frame = engine.plain_prompt_section(&prompt)?;
            let b = Builder::for_plain_prompt(&prompt, frame);
            let l = &b.schema().layers[0];
            let (layer, group) = (l.id, l.groups[0].id);
            Some((prompt, b, layer, group))
        }
    };
    let (prompt, builder, layer_id, group_id) = match (proj, &synthetic) {
        (Some(p), _) => {
            let (layer, group) = shared_target(p, source).unwrap_or((p.layer, p.group));
            (p.prelude.clone(), &p.builder, layer, group)
        }
        (None, Some((prompt, b, l, g))) => (prompt.clone(), b, *l, *g),
        (None, None) => unreachable!("synthetic is built exactly when proj is None"),
    };

    let mut failed = 0usize;
    let mut done = 0usize;
    let mut next = 0usize;
    let started = Instant::now();

    while next < turns.len() {
        let want = ingest::refill(0, turns.len() - next);
        let slice = &turns[next..next + want];
        next += want;

        // One round trip for the whole window. Creating them one at a time is
        // what starved zend's own batch to 2–4 wide before it was pipelined.
        //
        let convs = engine.new_conversations_with_projection_batch(
            slice.len(),
            &prompt,
            builder,
            layer_id,
            group_id,
            &cfg,
        );

        // Submit every turn first, await none. This is the step that batches.
        let mut inflight = Vec::with_capacity(slice.len());
        for (doc, conv) in slice.iter().zip(convs) {
            let (addr, body) = (&doc.addr, &doc.body);
            let mut conv = match conv {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("{addr}: conversation not created — {e:?}");
                    failed += 1;
                    done += 1;
                    continue;
                }
            };
            // A changed document's old conversation leaves memory before its
            // new text enters it, so the gather never holds both versions.
            if matches!(
                ledger.inspect(&doc.path, Some(&doc.body)),
                Reconcile::Changed
            ) {
                if let Some(r) = mind_record::rel(mind, &doc.path) {
                    let gone = mind_record::retire(engine, &r);
                    tracing::info!("{addr}: changed — {gone} old conversation(s) retired");
                }
            }
            // Prefilled, not decoded: both halves are written verbatim in one
            // batched forward, because the text is already on disk.
            match conv.submit_prefilled_turn(
                addr,
                body,
                PROJECTION_MARKER,
                SelectionState::new(),
                tags.clone(),
            ) {
                Ok(handle) => inflight.push((conv.timeline_id(), conv, handle, doc)),
                Err(e) => {
                    tracing::warn!("{addr}: not submitted — {e:?}");
                    failed += 1;
                    done += 1;
                }
            }
        }

        // Now drain. Each seal is what puts the turn in the substrate; the
        // sequence is dropped immediately after, freeing its K/V.
        let mut sealed: Vec<_> = Vec::with_capacity(inflight.len());
        for (timeline, mut conv, handle, doc) in inflight {
            let addr = &doc.addr;
            // Where the prefill forward is actually awaited. For a co-batched
            // window the first drain absorbs the slab and the rest return
            // quickly — measured as roughly four documents per slab, which is
            // what an 8 192-token budget over ~1 900-token documents predicts.
            //
            // **The projection events are collected here, not discarded.** They
            // are the provenance signature: what this document's projection
            // selected out of the layers already in the substrate. Dropping them
            // — which is what `_ => {}` did — leaves the document written and
            // unlinked, so a later scan has nothing to pull on.
            let (response, mut events) = drain(&handle, addr).await;
            match response {
                Some(r) => match conv.finish_turn(handle, &r) {
                    Ok(_) => {
                        // **The tokens this document actually cost the GPU.**
                        //
                        // Documents are the wrong unit for watching an ingest:
                        // they differ by an order of magnitude in length, so a
                        // count of them says nothing about the rate. Tokens per
                        // second is the number that compares directly against
                        // the forward-batched gate's prefill rate on the same
                        // card — which is how "this is taking a while" becomes
                        // "this is running at a tenth of what the card can do".
                        //
                        // Counted at the seal, so it only ever includes work
                        // that reached the substrate.
                        p.add_prefill_tokens(r.stats.turn_prefill_tokens as u64);
                        persist_signatures(&mut conv, &r, &mut events, addr);
                        if let Some(rel) = mind_record::rel(mind, &doc.path) {
                            mind_record::mark_version(engine, timeline, &rel, &doc.body);
                        }
                        // **The seal is what earns the ledger entry.** Recorded here, at the
                        // one point the document is provably a turn in the substrate, so a
                        // failure on any path above leaves it un-recorded and therefore
                        // retried on the next boot.
                        ledger.reconcile(&doc.path, Some(&doc.body));
                        sealed.push(timeline);
                    }
                    Err(e) => {
                        tracing::warn!("{addr}: not sealed — {e:?}");
                        failed += 1;
                    }
                },
                None => failed += 1,
            }
            done += 1;
            ingest::announce(p, source, done, turns.len());
        }

        // **The ledger is written at the window boundary, not the layer's.**
        //
        // It used to flush once per layer, on the reasoning that a crash
        // mid-layer "costs one layer rather than the whole mind". That holds for
        // a mind of many small layers and fails completely for a real one: this
        // estate's `world/` layer is 1,267 of its 1,823 documents, so one layer
        // *is* the whole mind, and a daemon stopped anywhere inside it wrote no
        // ledger at all.
        //
        // The cost of that is not a repeated ingest — it is a **duplicated**
        // one. A document with no ledger entry reads as `Added` rather than
        // `Changed`, so nothing tombstones the turn the interrupted run already
        // sealed, and the next boot writes a second copy that the gather can
        // surface alongside the first. Two runs of 641 documents left 84 GB of
        // world history with no way to tell the copies apart.
        //
        // A window is ~4 documents, so this is a few hundred small writes across
        // a half-hour load, and the most an interruption can cost is the window
        // in flight.
        ledger.flush();

        // Give the arena back before the next window.
        //
        // Sealing writes the turn to the substrate; it does **not** reclaim the
        // hot tier, and nothing else was going to. Without this the ingest
        // degraded monotonically — a measured 385 → 602 → 760 → 891 ms per
        // document across four windows — because every document's K/V stayed
        // resident and each successive forward had less arena to work in.
        //
        // Flushing, unlike zend's incremental sweep. zend can demote its older
        // cases while newer ones are still in flight, because its window is a
        // continuous pipeline with a later boundary sweep to catch stragglers.
        // This drains completely between windows, so there is no later sweep and
        // a not-yet-warm timeline would simply not be reclaimed.
        if !sealed.is_empty() {
            if let Err(e) = engine.demote_timelines_hot(&sealed, true) {
                tracing::warn!("hot→warm demote failed: {e} — the arena will fill");
            }
        }
    }

    // One line per layer, not per window. The per-window attribution that found
    // all of this — `wait_first` against `wait_rest`, which is the only thing
    // that distinguishes a batching window from a serial one — did its job and
    // came out again; a line per window is noise once the shape is settled.
    // `docs/npc_engine_design.md` records what it measured.
    let secs = started.elapsed().as_secs_f64();
    tracing::info!(
        "layer {} ingested: {} document(s) in {:.1}s ({:.0} doc/min)",
        source.label(),
        turns.len(),
        secs,
        if secs > 0.0 {
            turns.len() as f64 * 60.0 / secs
        } else {
            0.0
        }
    );
    Ok(failed)
}
