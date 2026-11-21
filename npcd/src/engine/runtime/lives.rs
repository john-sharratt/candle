//! Running a character's life: its episodes as dated conversations, then the
//! beliefs they form.
//!
//! Each episode is its own conversation, so its date and title are the
//! conversation's metadata and the substrate knows it as one named thing. An
//! episode is sealed into the substrate before the calls it contains run, so a
//! belief always has the episode it refers to already present. The hash of an
//! episode is recorded only at its seal, so an episode that fails part-way is
//! retried on the next boot rather than skipped. Each belief is written as a
//! tagged record that names its episode and what the character held when it formed.

use super::*;

/// What running one character's life produced.
#[derive(Debug, Default)]
pub struct Lived {
    pub episodes: usize,
    pub beliefs: usize,
    pub relationships: usize,
    pub intents: usize,
    /// Calls an author wrote that could not be executed. Counted and logged,
    /// never silent — a belief that did not form leaves a character missing a
    /// conviction with nothing to say why.
    pub rejected: usize,
}

/// Run one character's life: prefill each episode in date order and execute the
/// calls it contains.
///
/// # Why the episodes are separate conversations
///
/// Same reason the layer documents are: independent sequences co-batch, an
/// appended chain cannot. But there is a second reason here that matters more —
/// each episode carries its **date and title as conversation metadata**, and a
/// conversation holds one title. An episode is a named, dated thing in the
/// character's history, and the substrate should know it as one.
///
/// # Why the calls run after the prose is sealed
///
/// The belief is a consequence of the episode, so the episode has to be in the
/// substrate before the belief that refers to it. Running the calls first would
/// write a conviction whose origin did not yet exist — which is exactly the
/// orphaned-belief shape this whole design exists to end.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_life(
    engine: &SharedEngine,
    mind: &MindFiles,
    base_config: &SequenceConfig,
    proj: Option<&schema::Projection>,
    who: &str,
    episodes: &[life::Episode],
    ledger: &Ledger,
    p: &LoadProgress,
) -> anyhow::Result<Lived> {
    let mut lived = Lived::default();
    let mut cfg = base_config.clone();
    cfg.context_window_turns = 0;

    for (i, ep) in episodes.iter().enumerate() {
        let Some(raw) = mind.read(&ep.path.to_string_lossy().replace('\\', "/")) else {
            tracing::warn!("life {who}: {} unreadable", ep.path.display());
            continue;
        };
        // The ledger covers episodes too, so a restart re-lives nothing. `inspect` asks the
        // question; the answer is recorded at the seal below, so an episode that errors on the
        // way there is re-lived next boot instead of being silently lost with every belief,
        // relationship and intention it would have formed.
        let rel = mind_record::rel(Path::new(""), &ep.path);
        match ledger.inspect(&ep.path, Some(&raw)) {
            Reconcile::Added => {}
            // The episode as it was, and every belief, relationship and
            // intention it formed, leave memory before it is lived again.
            Reconcile::Changed => {
                if let Some(r) = &rel {
                    let gone = mind_record::retire(engine, r);
                    tracing::info!(
                        "life {who}: {} changed — {gone} conversation(s) retired",
                        ep.title
                    );
                }
            }
            Reconcile::Unchanged | Reconcile::Removed => {
                lived.episodes += 1;
                p.set_progress(i as u64 + 1, episodes.len() as u64);
                continue;
            }
        }

        let parsed = authoring::parse(&raw);
        lived.rejected += parsed.rejected.len();
        for r in &parsed.rejected {
            tracing::warn!("life {who}: {} — {}", ep.title, r.message());
        }

        // The episode itself: prose only, with the call machinery stripped. The
        // character's history is written in its own voice, not in tool syntax.
        // Under the real projection where there is one — an episode written on a
        // synthetic schema has no world to project against and produces no
        // signature, which is the whole point of running the life against the
        // world rather than beside it.
        let mut conv = match proj {
            Some(pr) => engine.new_conversation_with_projection(
                &pr.prelude,
                pr.builder.clone(),
                pr.layer,
                pr.group,
                cfg.clone(),
            )?,
            None => engine
                .new_conversation(&format!("An episode from the life of {who}."), cfg.clone())?,
        };
        let address = life::address(ep);
        let handle = conv.submit_prefilled_turn(
            &address,
            &parsed.prose,
            PROJECTION_MARKER,
            SelectionState::new(),
            life::tags(ep),
        )?;
        // **This is the world projection.** The episode's projection selects out
        // of the layers already in the substrate — the world first among them,
        // which is why the world is ingested before any life runs — and
        // persisting it writes the link from this moment to the world content it
        // happened against.
        let (response, mut events) = drain(&handle, &address).await;
        // A silent `continue` here left the episode un-lived *and* un-retryable: the hash was
        // already banked above, so the next boot skipped it. Now the hash is written at the
        // seal, so this path simply leaves it unrecorded — but it still has to say so and
        // still has to advance the bar, which stalled for that character otherwise.
        let Some(r) = response else {
            tracing::warn!(
                "life {who}: {address} produced no response — not lived, and left unrecorded \
                 so the next boot retries it"
            );
            p.set_progress(i as u64 + 1, episodes.len() as u64);
            continue;
        };
        conv.finish_turn(handle, &r)?;
        persist_signatures(&mut conv, &r, &mut events, &address);
        // Date and title as conversation metadata, so the substrate knows when
        // this happened and what it was called — which is what a projection
        // needs to reach an episode by *when* rather than only by what it said.
        let timeline = conv.timeline_id();
        if let Some(r) = &rel {
            mind_record::mark_version(engine, timeline, r, &raw);
        }
        for (key, value) in [
            ("life.date", ep.date.as_str()),
            ("life.title", ep.title.as_str()),
        ] {
            if let Err(e) = engine.set_conversation_metadata(timeline, key, value) {
                tracing::debug!("life {who}: metadata {key} not set — {e:?}");
            }
        }
        // **The episode is in the substrate; record it.** Everything above this line can fail
        // and leave the hash unwritten, which is what makes the next boot retry rather than
        // skip. See `ingest::Pending`.
        ledger.reconcile(&ep.path, Some(&raw));
        lived.episodes += 1;

        // Now the consequences, on their own records, after the episode they
        // came from is durable.
        for call in &parsed.calls {
            match write_consequence(engine, &cfg, proj, who, ep, rel.as_deref(), call).await {
                Ok(()) => match call.tool {
                    "form_belief" => lived.beliefs += 1,
                    "form_relationship" | "revise_relationship" => lived.relationships += 1,
                    "leave_intent" => lived.intents += 1,
                    _ => {}
                },
                Err(e) => {
                    tracing::warn!(
                        "life {who}: {} — {} not written: {e:#}",
                        ep.title,
                        call.tool
                    );
                    lived.rejected += 1;
                }
            }
        }

        // Give the arena back — the same reason the layer ingest does.
        if let Err(e) = engine.demote_timelines_hot(&[timeline], true) {
            tracing::debug!("life {who}: demote failed — {e}");
        }
        p.set_progress(i as u64 + 1, episodes.len() as u64);
    }
    Ok(lived)
}

/// Write one authoring call as a tagged substrate record.
///
/// **A belief is a record in the substrate, not a file on disk.** That is the
/// design's central move: it makes a belief retrievable by the same tag-filtered
/// gather as any other layer content, editable from the console, and — because
/// it names the episode that produced it — arguable. A belief you can trace to a
/// day is one you can disagree with; a belief simply asserted is not.
pub(super) async fn write_consequence(
    engine: &SharedEngine,
    cfg: &SequenceConfig,
    proj: Option<&schema::Projection>,
    who: &str,
    ep: &life::Episode,
    // The episode's mind path, so the consequence leaves memory with it.
    rel: Option<&str>,
    call: &authoring::Call,
) -> anyhow::Result<()> {
    let layer = authoring::by_name(call.tool)
        .map(|t| t.writes)
        .unwrap_or("beliefs");

    // The record reads as the character's own, in the register the rest of the
    // layers use, and carries its origin. The origin is the point: it is what
    // turns a conviction into something with a history.
    let body = render_consequence(call);
    let address = format!(
        "{layer}/{who}/{} {}",
        ep.date,
        call.args
            .get("statement")
            .or_else(|| call.args.get("entity_id"))
            .or_else(|| call.args.get("intent"))
            .and_then(|v| v.as_str())
            .unwrap_or(call.tool)
    );

    let mut conv = match proj {
        Some(pr) => engine.new_conversation_with_projection(
            &pr.prelude,
            pr.builder.clone(),
            pr.layer,
            pr.group,
            cfg.clone(),
        )?,
        None => engine.new_conversation(
            &format!("What {who}'s life left them holding."),
            cfg.clone(),
        )?,
    };
    let handle = conv.submit_prefilled_turn(
        &address,
        &body,
        PROJECTION_MARKER,
        SelectionState::new(),
        vec![
            layer.to_string(),
            format!("{layer}:{who}"),
            // The episode it came from, so the origin is reachable from the
            // belief and not only the other way round.
            format!("from:{}", ep.date),
        ],
    )?;
    let (response, mut events) = drain(&handle, &address).await;
    let r = response.ok_or_else(|| anyhow::anyhow!("no response"))?;
    let timeline = conv.timeline_id();
    conv.finish_turn(handle, &r)?;
    // A belief carries a signature too: what the character already held when it
    // formed. That is what makes a later contradiction reachable from the belief
    // it contradicts.
    persist_signatures(&mut conv, &r, &mut events, &address);
    {
        let e = &engine;
        let _ = e.set_conversation_metadata(timeline, "from.date", &ep.date);
        let _ = e.set_conversation_metadata(timeline, "from.title", &ep.title);
        if let Some(r) = rel {
            mind_record::mark(e, timeline, r);
        }
        let _ = e.demote_timelines_hot(&[timeline], true);
    }
    Ok(())
}

/// A call as the character would hold it.
pub(super) fn render_consequence(call: &authoring::Call) -> String {
    let s = |k: &str| call.args.get(k).and_then(|v| v.as_str()).unwrap_or("");
    match call.tool {
        "form_belief" => s("statement").to_string(),
        "form_relationship" | "revise_relationship" => {
            let who = if s("display").is_empty() {
                s("entity_id")
            } else {
                s("display")
            };
            let notes = s("notes");
            if notes.is_empty() {
                who.to_string()
            } else {
                format!("{who}. {notes}")
            }
        }
        "leave_intent" => {
            let until = s("until");
            if until.is_empty() {
                s("intent").to_string()
            } else {
                format!("{}. Until: {until}", s("intent"))
            }
        }
        _ => String::new(),
    }
}
