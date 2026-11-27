//! Builds a conversation's system-prompt state and installs its recurrent
//! checkpoint, grouping conversations that share a prompt branch.
//!
//! Each conversation's sections are ingested into the substrate before its first
//! turn. Conversations on one prompt branch install identical recurrent state,
//! and the scheduler wait per request dominates the cost, so the installer
//! resolves each branch once and sends one request per branch. A transplanted
//! state is never recomputed, so a non-finite one is named at the install that
//! brought it in, and a refused install is reported rather than leaving the
//! recurrent layers silently at zero.

use super::*;
use crate::projection::{
    collection_of, node_options_tokens, section_slot, summary_points, tree_prefix, unpack_selection,
};

/// Warn of a dynamic section, or a tree node, whose content outgrew the
/// `min_tokens` it declares (`docs/placed_section_layout.md` §3.1c): its slot
/// grew, everything after it moved, and what was sealed behind the old slot is
/// stale.
fn warn_outgrown(name: &str, tokens: usize, min_tokens: usize) {
    if min_tokens > 0 && tokens > min_tokens {
        tracing::warn!(
            section = name,
            tokens,
            min_tokens,
            "a section outgrows its min_tokens: everything after it moved, and the \
             substrate sealed behind its old slot is stale"
        );
    }
}

/// Whether a dialed first turn has to swap the prompt checkpoint: only when the
/// branch its selection names is not the one already installed.
pub(super) fn needs_branch_swap(installed: Option<ContentHash>, selected: ContentHash) -> bool {
    installed != Some(selected)
}

/// A freshly built conversation's prompt-branch state, still to be installed
/// onto its slot. Returned by [`Sequence::new_with_projection`] instead of being
/// installed there, so a caller building several conversations can install them
/// as one group — see [`install_branch_states`] for why that matters.
pub(crate) struct PendingBranchState {
    pub(super) sequence_id: SequenceId,
    pub(super) prefix: ContentHash,
    /// The payload when the caller already holds it (just computed, or a memo
    /// hit); `None` sends the installer to the substrate for it.
    pub(super) payload: Option<Arc<BranchCheckpointPayload>>,
}

/// Install each conversation's prompt-branch recurrent state onto its slot,
/// one scheduler request per distinct branch.
///
/// Conversations born on the same prompt branch install byte-identical state,
/// and the install's cost is overwhelmingly the wait for the scheduler to drain
/// the request — measured at 157 ms of a 175 ms round-trip, against 17 ms of
/// device scatter per slot and 8 ms to materialise the layers. So the layers are
/// built ONCE per branch and every slot on that branch rides one request. A
/// caller creating a window of conversations therefore pays one queue wait, not
/// one per conversation.
///
/// `fresh` is the payload when the caller just computed or memo-hit it; `None`
/// falls back to the substrate read. A branch with no checkpoint is an ordinary
/// outcome (logged), not an error: the conversation starts with its prompt in
/// K/V and nothing in the recurrent layers, and reads perfectly while doing so.
pub(crate) fn install_branch_states(
    scheduler_tx: &Sender<SchedulerRequest>,
    pending: &[PendingBranchState],
    substrate: &Conversation,
) -> crate::Result<()> {
    // Slots are grouped by branch FIRST, and the payload resolved per branch
    // afterwards. Resolving inline while grouping means an entry whose own
    // `payload` is `None` and whose substrate read misses is skipped without
    // registering its slot — so a later entry on the SAME branch that does carry
    // a payload builds a group excluding it, and that conversation silently
    // starts with no recurrent state while an identical payload sits one
    // iteration away. Grouping first makes the payload a property of the branch,
    // which is what it actually is.
    let mut groups: Vec<(
        ContentHash,
        Option<Arc<BranchCheckpointPayload>>,
        Vec<SequenceId>,
    )> = Vec::new();
    for PendingBranchState {
        sequence_id: id,
        prefix,
        payload: fresh,
    } in pending
    {
        match groups.iter_mut().find(|(p, _, _)| p == prefix) {
            Some(g) => {
                g.2.push(*id);
                // Any entry's payload serves the whole branch — take the first
                // one offered rather than depending on which slot carried it.
                if g.1.is_none() {
                    g.1 = fresh.as_ref().map(Arc::clone);
                }
            }
            None => groups.push((*prefix, fresh.as_ref().map(Arc::clone), vec![*id])),
        }
    }

    // Resolve each branch's payload once, falling back to the substrate only for
    // a branch no caller had in hand. Just computed ⇒ already there; only a
    // branch this process did not compute needs the disk, and by then the record
    // is durable.
    let groups: Vec<(ContentHash, Arc<BranchCheckpointPayload>, Vec<SequenceId>)> = groups
        .into_iter()
        .filter_map(|(prefix, payload, ids)| {
            let payload = match payload {
                Some(p) => p,
                None => match substrate.read_branch_checkpoint(prefix) {
                    Ok(Some(p)) => p,
                    Ok(None) => {
                        tracing::warn!(
                            slots = ids.len(),
                            "no branch checkpoint for this prompt branch — the first turn \
                             will run with the system prompt in K/V and nothing in the \
                             recurrent layers, and will read perfectly while doing so"
                        );
                        return None;
                    }
                    Err(e) => {
                        tracing::warn!(
                            slots = ids.len(),
                            "branch checkpoint unreadable ({e}) — starting from zero"
                        );
                        return None;
                    }
                },
            };
            Some((prefix, payload, ids))
        })
        .collect();

    for (prefix, payload, sequence_ids) in groups {
        // The scheduler installs from a shared slice, so the layers are
        // materialised once for the branch — held for every later group on it —
        // rather than per slot or per install.
        let state = shared_held(scheduler_tx, prefix, &payload);
        let layers = Arc::clone(&state.layers);
        let n_layers = layers.len();
        let n_slots = sequence_ids.len();
        // A transplanted state is never recomputed by the slot that receives it,
        // so a non-finite one is invisible from the outside: the geometry checks
        // pass, the install succeeds, and every subsequent forward on that
        // sequence returns NaN logits while the daemon reports a healthy turn.
        // Name it at the boundary that introduces it — the alternative is
        // diagnosing it thirty layers downstream from a row of `!`.
        if let Some((layer, kind)) = first_non_finite_layer(&layers) {
            tracing::error!(
                layer_index = layer,
                buffer = kind,
                layers = n_layers,
                slots = n_slots,
                "BRANCH CHECKPOINT IS NON-FINITE — refusing to install it. Installing \
                 would give every one of these slots NaN logits for the rest of the \
                 conversation. They start with no memory of their system prompt instead."
            );
            continue;
        }
        let (tx, rx) = flume::bounded(1);
        scheduler_tx
            .send(SchedulerRequest::InstallRecurrentState {
                sequence_ids,
                state,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        match replied(&rx).map_err(|_| ConversationError::SchedulerGone)? {
            Ok(true) => {
                for _ in 0..n_slots {
                    note_branch_checkpoint_installed();
                }
                note("installed prompt branch checkpoint", || {
                    format!("layers={n_layers} slots={n_slots}")
                });
            }
            // The scheduler reports `false` when no slot took the state — a
            // refused import, which leaves the recurrent layers at zero while
            // the K/V holds the whole system prompt. That conversation then
            // reads perfectly while remembering nothing of its own prompt, so
            // it has to be said out loud; silence here is what let it look like
            // a successful install.
            Ok(false) => tracing::error!(
                layers = n_layers,
                slots = n_slots,
                "BRANCH CHECKPOINT NOT INSTALLED — no slot accepted the state (schedule \
                 hash or geometry mismatch). These slots start with the system prompt in \
                 K/V and nothing in the recurrent layers."
            ),
            Err(e) => tracing::warn!(
                "BRANCH CHECKPOINT REFUSED (hash or geometry mismatch): {e} — the first \
                 turn starts with no memory of its system prompt"
            ),
        }
    }
    Ok(())
}

/// The first layer of a checkpoint carrying a non-finite `S` or conv tail, with
/// the name of the offending buffer.
///
/// `state` and `conv_tail` are raw little-endian `f32` bytes. A trailing partial
/// element cannot be interpreted, so it is reported rather than silently
/// ignored — a length that is not a multiple of four means the payload is
/// already malformed.
pub(super) fn first_non_finite_layer(layers: &[ExportedLayerState]) -> Option<(u32, &'static str)> {
    fn bad(bytes: &[u8]) -> bool {
        if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
            return true;
        }
        bytes
            .as_chunks::<{ std::mem::size_of::<f32>() }>()
            .0
            .iter()
            .any(|c| !f32::from_le_bytes(*c).is_finite())
    }
    layers.iter().find_map(|l| {
        if bad(&l.state) {
            Some((l.layer_index, "state"))
        } else if bad(&l.conv_tail) {
            Some((l.layer_index, "conv_tail"))
        } else {
            None
        }
    })
}

impl Sequence {
    /// Create a new conversation backed by a full projection [`Builder`].
    ///
    /// `system_prompt` must already be formatted for the model's chat template
    /// (e.g. ChatML-wrapped).  `target` names the `(layer, group)` this
    /// conversation is for — turns are appended into `target.group`, and
    /// `target` is what gets passed to `projection.project()`.
    ///
    /// Eagerly ingests the conversation's plain `system_prompt` and the
    /// schema's declared sections / collections into the workspace
    /// substrate before returning, so the first `submit_turn` finds
    /// every static system-side section already pinned.
    pub(crate) fn new_with_projection(
        scheduler_tx: Sender<SchedulerRequest>,
        sequence_id: SequenceId,
        tokenizer: Arc<tokenizers::Tokenizer>,
        literal_tokenizer: Arc<tokenizers::Tokenizer>,
        system_prompt: &str,
        projection: Builder,
        target: ProjectionTarget,
        config: SequenceConfig,
        chunk_size: usize,
        model_core: ModelCoreProperties,
        vision: Option<&'static VisionTowerSource>,
        substrate: Conversation,
        section_progress: Option<&dyn Fn(u64, u64)>,
    ) -> crate::Result<(Self, Option<PendingBranchState>)> {
        // Persistence is now a property of the workspace `Conversation`
        // (the substrate handle), wired in by the engine via
        // `Conversation::open(path)`.  The Sequence has nothing to do
        // with on-disk records anymore; turn_counter is informational
        // only — exposed via `Sequence::turn_count` for tests.  Count
        // the system prompt as turn 0 when present, matching the
        // pre-substrate behaviour (system + user + assistant = 3).
        let prior_turns = substrate.read().turn_count(target.timeline) as u64;
        let initial_turn_counter: u64 = if system_prompt.is_empty() {
            prior_turns
        } else {
            prior_turns + 1
        };

        let tree_config = config.tree.clone();
        let projection = Arc::new(projection);
        let conv = Self {
            submitter: TurnSubmitter {
                scheduler_tx,
                tokenizer,
                literal_tokenizer,
                config,
            },
            ask_state: Arc::new(Mutex::new(AskState {
                projection: Arc::clone(&projection),
                selection: SelectionState::default(),
            })),
            id: sequence_id,
            tree: ConversationTree::with_config(system_prompt, tree_config),
            selection: SelectionState::default(),
            pending_user: None,
            turn_counter: initial_turn_counter,
            turn_in_flight: false,
            freed: false,
            current_blocks: BlockCount(0),
            chunk_size,
            model_core,
            vision,
            base_projection: Arc::clone(&projection),
            projection,
            submitted: SubmittedSections::default(),
            substrate,
            target,
            birth_prefix: Arc::new(Vec::new()),
            birth_ranges: Vec::new(),
            frame: None,
            frame_window: None,
            branch_spans: Vec::new(),
            state_is_prompt_only: false,
            installed_branch: None,
        };

        // Set the in-memory tree's system prompt tokens so the tree's
        // own per-turn formatting can reference them.  Persistence
        // for the system prompt has moved to the substrate side via
        // `insert_section` below — no need for a separate cold-store
        // write.
        let t_prompt = std::time::Instant::now();
        let mut conv = conv;
        if !system_prompt.is_empty() {
            let text = conv.tree.system_prompt_text().to_string();
            let token_ids = conv.tokenize(&text).unwrap_or_default();
            conv.tree.set_system_prompt_tokens(token_ids);
            let _ = text;
        }
        let prompt_ms = t_prompt.elapsed();

        // Eagerly seed every static system-side section into the
        // workspace substrate.  The slot itself stays empty — the
        // first `submit_turn` runs `apply_projection` which
        // materialises the relevant sections onto it from the
        // substrate's CPU-pinned bytes via the upload cache.
        //
        // The schema's declared sections + collections for this
        // target's layer.  Each `Section` becomes one substrate
        // section under its declared id; each `Collection` runs
        // all its sections in parallel via
        // `insert_section_collection`.  The whole system prompt is
        // composed from these schema items — no separate monolithic
        // "system_section_id" pre-pinning, which used to double the
        // system content with an unwrapped fragment copy.
        let t_items = std::time::Instant::now();
        let layer_items: Vec<SystemPromptItem> =
            conv.projection.schema().system_prompt.items.clone();
        let items_ms = t_items.elapsed();
        let t_ingest = std::time::Instant::now();

        // Cumulative-prefix ingest builds each content section's K/V
        // conditioned on the chain of previously-ingested content
        // sections.  Templates do not appear in this chain — their K/V
        // is live-prefilled at projection-apply time under the actual
        // runtime context, so substrate carries no entries for them
        // and they contribute nothing to the prefix here.
        let mut linear_prefix: Vec<PrefixItem> = Vec::new();
        // Mirrors `linear_prefix` but excludes Collection members and
        // `depends_on`-gated sections — the birth prefix: the empty-collection
        // layout an append-only conversation's first turn is framed by.
        let mut fixed_prefix: Vec<PrefixItem> = Vec::new();
        // Where each reserved range's hole stands in `fixed_prefix`.
        let mut birth_ranges: Vec<BirthRange> = Vec::new();
        // The engine's trained window, which picks each range's size.
        let native = conv.substrate.read().trained_window();
        // Where each section tree's emitted ids land inside `fixed_prefix`, as
        // `(start, len)` — the windows the branch-checkpoint pass rewrites per
        // leaf, one per declared tree in item order. Empty when the layer
        // declares no tree, which makes the prompt single-branch and the pass a
        // single checkpoint.
        let mut branch_spans: Vec<(usize, usize)> = Vec::new();

        // ── Progress accounting ────────────────────────────────────────
        // Byte totals across every section we're about to ingest. The
        // frontend's "Sections" step progress fills as we walk the
        // ingest loop below, so the load screen actually moves between
        // 0 % and 100 % instead of sitting at 0 % until the final
        // `set_step` transition.
        // Templates are filtered from the ingest loop and contribute
        // nothing to the byte-count progress — they don't enter the
        // substrate, so there's no work to count for them.
        let total_bytes: u64 = layer_items
            .iter()
            .map(|item| match item {
                SystemPromptItem::Section(s) => {
                    if s.is_template {
                        0
                    } else {
                        s.content.len() as u64
                    }
                }
                SystemPromptItem::Collection(c) => c
                    .sections
                    .iter()
                    .filter(|s| !s.is_template)
                    .map(|s| s.content.len() as u64)
                    .sum::<u64>(),
                // Tree nodes are sealed content; the ingest loop seals every
                // option's branch variants — count content × variant-count over
                // every option of every node.
                SystemPromptItem::SectionTree(t) => t
                    .nodes
                    .iter()
                    .map(|n| {
                        let opt_bytes: u64 = n
                            .options
                            .iter()
                            .map(|o| o.content.len() as u64 * o.variants.len() as u64)
                            .sum();
                        // A collection node seals each member ×branch.
                        let coll_bytes: u64 = n.collection.as_ref().map_or(0, |tc| {
                            tc.collection
                                .sections
                                .iter()
                                .zip(tc.variants.iter())
                                .map(|(s, vs)| s.content.len() as u64 * vs.len() as u64)
                                .sum()
                        });
                        opt_bytes + coll_bytes
                    })
                    .sum::<u64>(),
            })
            .sum::<u64>();
        let mut done_bytes: u64 = 0;
        let report = |done: u64| {
            if let Some(cb) = section_progress {
                cb(done, total_bytes);
            }
        };
        // Initial 0 / total tick so the bar appears immediately —
        // otherwise the UI shows nothing until the first section
        // finishes, which can be seconds for a heavy schema.
        report(0);

        // Cumulative ingest: walk schema items in declaration order,
        // ingesting each non-template `Section`-kind item with all
        // previously-ingested content sections Arc-injected onto the
        // scratch slot as its prefix.  The prefill forward pass for
        // section N therefore attends to content sections 0..N-1 just
        // like it would at projection time, producing K_raw values
        // conditioned on the real preceding content context.
        //
        // Template-kind sections (`is_template = true`) ingest here too, and
        // that is a change: their K/V used to be live-prefilled at
        // projection-apply time under the actual runtime left context, so the
        // substrate never saw them.
        //
        // They are dialect structural text — `<|im_start|>system\n`,
        // `<tools>`, `</tools>`, the system close — a handful of tokens whose
        // *content* is fixed and whose position in the prompt is fixed. What
        // varied was their left context, because a `depends_on` template emits
        // only when its collection materialises. Sealing them therefore accepts
        // the same approximation the collection path already accepts: a
        // section's K/V is not a strict function of which members ingested.
        //
        // The reason to accept it is that a live-prefilled run is a glue
        // **island**, and an island is a hole in the middle of the sequence
        // that the engine must gap-fill. A recurrence cannot compute the output
        // of a token inserted mid-sequence, so on the hybrid lineage the
        // alternative to an approximate bake is not a more accurate answer —
        // it is a refused wave. See `docs/deltanet_state_persistence.md` §4.7d.
        //
        // `Collection` items: every member ingests in parallel with
        // the *same* pre-collection prefix (members do not attend to
        // each other). Sections declared after a Collection attend to
        // the full set of members during ingest — the maximal-context
        // approximation for runtime selection.
        // Every item takes its slot in both prefixes, as every projection lays
        // it down (`docs/placed_section_layout.md` §3.1c): a catalog summary's
        // range is a hole here, nothing being summarised yet; a section is
        // padded to its slot, and one the birth prefix leaves out is a hole of
        // it; each tree node is padded to its slot, inside the tree as after it.
        let summary_slots = summary_points(&layer_items);
        let substrate = conv.substrate.clone();
        let tokens_of = |id: SectionId| substrate.read().section_tokens_of(id).len();
        for (at, item) in layer_items.iter().enumerate() {
            for &cid in summary_slots.get(&at).into_iter().flatten() {
                if let Some(range) = collection_of(&layer_items, cid).and_then(|c| c.summary_range)
                {
                    let positions = range.positions(native);
                    linear_prefix.push(PrefixItem::Hole(positions));
                    fixed_prefix.push(PrefixItem::Hole(positions));
                }
            }
            match item {
                SystemPromptItem::Section(s) => {
                    conv.insert_section_with_prefix(s.id, s.content.as_str(), &linear_prefix)?;
                    let tokens = tokens_of(s.id);
                    let slot = section_slot(s, tokens);
                    warn_outgrown(&s.name, tokens, s.min_tokens);
                    linear_prefix.push(PrefixItem::Section(s.id));
                    if slot > tokens {
                        linear_prefix.push(PrefixItem::Hole(slot - tokens));
                    }
                    // Sections with `depends_on` are conditional — they
                    // only emit when the named collection materialises —
                    // so they do not enter the birth prefix `fixed_prefix`:
                    // their slot does, as a hole, so what follows stands
                    // where it does whether they emit or not. A section with
                    // no content is named by neither.
                    if s.depends_on.is_none() && !s.content.is_empty() {
                        fixed_prefix.push(PrefixItem::Section(s.id));
                        if slot > tokens {
                            fixed_prefix.push(PrefixItem::Hole(slot - tokens));
                        }
                    } else if slot > 0 {
                        fixed_prefix.push(PrefixItem::Hole(slot));
                    }
                    done_bytes += s.content.len() as u64;
                    report(done_bytes);
                }
                SystemPromptItem::Collection(coll) => {
                    // A reserved range: everything after it is sealed behind it
                    // left empty; a range filled from the top seals each member
                    // ending at its top, one filled from the bottom straight
                    // after the prompt before it.
                    let range = coll.range.map(|r| r.positions(native));
                    let sealed_at_top = coll
                        .range
                        .filter(|r| r.fill == RangeFill::Top)
                        .map(|r| r.positions(native));
                    let batch: Vec<(SectionId, &str)> = coll
                        .sections
                        .iter()
                        .map(|sec| (sec.id, sec.content.as_str()))
                        .collect();
                    if !batch.is_empty() {
                        // Per-section progress: tick `done_bytes`
                        // and call the outer report() as each member
                        // completes its ingest / restore / skip.
                        // Collection members complete in submission
                        // order at the wait-phase boundary inside
                        // `insert_section_collection_with_progress`,
                        // so the bar advances ~once per ~prefill of
                        // a single tool description rather than
                        // jumping at the end of the whole collection.
                        let done_bytes_ref = &mut done_bytes;
                        let report_ref = &report;
                        conv.insert_section_collection_with_progress(
                            &batch,
                            &linear_prefix,
                            sealed_at_top,
                            // A schema collection's members, at its level.
                            SectionSeal::Member {
                                level: coll.kv_level,
                            },
                            |_sid, content_len| {
                                *done_bytes_ref += content_len as u64;
                                report_ref(*done_bytes_ref);
                            },
                        )?;
                    }
                    if let Some(positions) = range {
                        // The range, empty, in both prefixes: the sections
                        // after it are sealed and read past it, whatever a
                        // projection places in it.
                        birth_ranges.push(BirthRange {
                            collection: coll.id,
                            at: fixed_prefix.len(),
                            positions,
                            fill: coll.range.map_or(RangeFill::Top, |r| r.fill),
                        });
                        linear_prefix.push(PrefixItem::Hole(positions));
                        fixed_prefix.push(PrefixItem::Hole(positions));
                        continue;
                    }
                    // Without a range, extend the live linear_prefix with
                    // every collection member so subsequent sections see
                    // them at projection time, but do NOT add them
                    // to fixed_prefix — the birth prefix must be
                    // coherent for the empty-collection case.
                    //
                    // This is also why these members cannot be offloaded as the
                    // section tree's embedded collections are: those never enter
                    // `linear_prefix`, so nothing prefills against them once
                    // their branch is sealed. These do. Offloading them here
                    // costs every later section its prefix — `prepare_section_
                    // ingest` finds no sealed substrate entry and skips it,
                    // building a silently wrong system prompt. The cold-boot hot
                    // set is therefore bounded by the whole collection span, not
                    // by one collection, and the KV reservation must cover it.
                    for sec in &coll.sections {
                        if sec.is_template {
                            continue;
                        }
                        linear_prefix.push(PrefixItem::Section(sec.id));
                    }
                }
                SystemPromptItem::SectionTree(tree) => {
                    // Seal EVERY option's branch variants up front — the full
                    // cross-product prefill that makes selector switching free.
                    // Each variant is sealed against `[outer content prefix |
                    // that branch's in-tree ancestor variants]`, every earlier
                    // node in its slot (`tree_prefix`).  Sealing in (node order,
                    // option order, variant order) is dependency-safe: a
                    // variant's in-tree prefix references only earlier nodes'
                    // variants, which are already sealed, so their slots are
                    // known.
                    for (ni, node) in tree.nodes.iter().enumerate() {
                        // The slotted in-tree prefix a variant of this node
                        // keyed `ancestors` seals behind.
                        let in_tree = |ancestors: u32| {
                            let selection = unpack_selection(tree, ancestors, node.ancestor_dims);
                            tree_prefix(tree, ni, &selection, tokens_of)
                        };
                        // A prefix-transparent embedded collection: seal each
                        // member ONCE PER ancestor branch (the ×outer-selector
                        // fan-out), batching all members of a branch under that
                        // branch's prefix with the aggressive collection quantize
                        // policy.  It never extends `linear_prefix` — nodes below
                        // anchor on the next mandatory node, not these members.
                        if let Some(tc) = &node.collection {
                            let members = &tc.collection.sections;
                            // The branch set is identical across members; read it
                            // off member 0's per-branch variant list.
                            if let Some(branch_list) = tc.variants.first() {
                                for (bi, bvar) in branch_list.iter().enumerate() {
                                    let mut prefix = linear_prefix.clone();
                                    prefix.extend(in_tree(bvar.ancestors));
                                    let batch: Vec<(SectionId, &str)> = members
                                        .iter()
                                        .enumerate()
                                        .map(|(mi, sec)| {
                                            (tc.variants[mi][bi].id, sec.content.as_str())
                                        })
                                        .collect();
                                    if !batch.is_empty() {
                                        let done_bytes_ref = &mut done_bytes;
                                        let report_ref = &report;
                                        conv.insert_section_collection_with_progress(
                                            &batch,
                                            &prefix,
                                            None,
                                            SectionSeal::Member {
                                                level: tc.collection.kv_level,
                                            },
                                            |_sid, content_len| {
                                                *done_bytes_ref += content_len as u64;
                                                report_ref(*done_bytes_ref);
                                            },
                                        )?;
                                        // Offload-as-we-go: quantize + persist +
                                        // evict THIS branch's members before the
                                        // next branch prefills, so the native
                                        // catalog never exceeds one branch.  Safe
                                        // because members are prefix-transparent —
                                        // nothing attends back over them in the
                                        // build — and the per-turn elevate reloads
                                        // the projection's top-k on demand.
                                        let (tx, rx) = flume::bounded(1);
                                        conv.scheduler_tx
                                            .send(SchedulerRequest::OffloadCollectionMembers {
                                                conversation: conv.substrate.clone(),
                                                response_tx: tx,
                                            })
                                            .map_err(|_| ConversationError::SchedulerGone)?;
                                        rx.recv()
                                            .map_err(|_| ConversationError::SchedulerGone)??;
                                    }
                                }
                            }
                            continue;
                        }
                        for option in &node.options {
                            for v in &option.variants {
                                let mut prefix = linear_prefix.clone();
                                prefix.extend(in_tree(v.ancestors));
                                conv.insert_section_with_prefix(
                                    v.id,
                                    option.content.as_str(),
                                    &prefix,
                                )?;
                                done_bytes += option.content.len() as u64;
                                report(done_bytes);
                            }
                        }
                        warn_outgrown(
                            &node.name,
                            node_options_tokens(node, tokens_of),
                            node.min_tokens,
                        );
                    }
                    // Sections declared after the tree (and the birth
                    // prefix) attend to the default branch — the fan-out is
                    // bounded to the tree itself.
                    //
                    // Remember where this tree's branch ids and its pad sit in
                    // `fixed_prefix`, so the checkpoint pass below can swap the
                    // default branch for any other leaf, padded to the same
                    // slot, without re-walking the schema. Pushed rather than
                    // assigned: a schema may declare several trees, and each
                    // owns its own window.
                    let start = fixed_prefix.len();
                    let branch =
                        tree_prefix(tree, tree.nodes.len(), &tree.default_selection, tokens_of);
                    linear_prefix.extend(branch.iter().cloned());
                    fixed_prefix.extend(branch);
                    branch_spans.push((start, fixed_prefix.len() - start));
                }
            }
        }

        // Compute one recurrent checkpoint per prompt branch.
        //
        // A new conversation Arc-injects its sealed prompt K/V, so the wave
        // never sees those tokens: on a model whose memory is a recurrence the
        // state would enter the first user turn at zero while the attention
        // layers hold the entire prompt. This is the only place a conversation
        // needs a state it did not compute (§4.6), and the only way to get it is
        // to run the tokens — a recurrence cannot be Arc-injected.
        //
        // Nothing here runs on a plain transformer: `carries_recurrent_state`
        // is false, so the loop is skipped and no forward is paid.
        let ingest_ms = t_ingest.elapsed();
        let t_ckpt = std::time::Instant::now();
        let branch_checkpoint = if conv.model_core.carries_recurrent_state {
            conv.build_branch_checkpoint(&fixed_prefix, &branch_spans)?
        } else {
            None
        };
        let ckpt_ms = t_ckpt.elapsed();
        // Retained for the first-turn branch re-key: the composer dials arrive
        // with the first `TurnOptions.selection`, which this create cannot
        // know, so the checkpoint installed below is the DEFAULT branch's. If
        // that first turn selects a different branch, `submit_turn_with_options`
        // rebuilds from exactly these inputs.
        conv.birth_prefix = Arc::new(fixed_prefix);
        conv.birth_ranges = birth_ranges;
        conv.frame = conv
            .submitter
            .config
            .disable_reprojection
            .then(|| Arc::clone(&conv.birth_prefix));
        conv.branch_spans = branch_spans;
        // Only a checkpoint that exists is installed: a failed pass returns the
        // prefix without a payload, and a first turn on that branch should try again.
        conv.installed_branch = branch_checkpoint
            .as_ref()
            .and_then(|(prefix, payload)| payload.as_ref().map(|_| *prefix));

        // The slot is left empty: the first `submit_turn`'s `apply_projection`
        // lays the system prompt down as part of that turn's prefill, in the
        // scheduler's ordinary waves, rather than in a request of its own that
        // would hold the wave thread for every other conversation.
        //
        // The recurrent layers' state that goes with the prompt is NOT
        // installed here: the install is a scheduler round-trip that is almost
        // all queue wait, and that wait is per-request, not per-slot. The
        // caller installs — one group per batch of conversations — via
        // `install_branch_states`, before the first turn is submitted.
        let pending = branch_checkpoint.map(|(prefix, payload)| PendingBranchState {
            sequence_id: conv.id,
            prefix,
            payload,
        });
        // Per-conversation, so an event: a phase that opens hundreds of them
        // would otherwise write hundreds of lines. The parts are split
        // because they have very different characters — `ingest` is this
        // thread's own walk of the schema and `ckpt` is a memo hit or a forward.
        note("sequence construction", || {
            format!(
                "prompt_ms={} items_ms={} ingest_ms={} ckpt_ms={}",
                prompt_ms.as_millis(),
                items_ms.as_millis(),
                ingest_ms.as_millis(),
                ckpt_ms.as_millis(),
            )
        });
        // From here until the first turn advances it, this conversation's
        // state is exactly the installed prompt checkpoint — the window in
        // which a dial-selected branch may swap it.
        conv.state_is_prompt_only = conv.model_core.carries_recurrent_state;

        Ok((conv, pending))
    }

    /// The ordered sections this conversation's system prompt is built from,
    /// and the content prefix that identifies them.
    ///
    /// `birth` is the birth-time projection — the collections-as-empty section
    /// list — and `branch_spans` are the windows inside it that the section
    /// trees filled, one per tree in item order, each a branch's sections and
    /// the hole padding them to the tree's slot. Each window is
    /// **re-derived** from its tree and this conversation's own selection
    /// rather than taken as-is, because the walk that built `birth`
    /// had only the authored defaults to go on: `SectionTree::branch_prefix_ids`
    /// is what turns a selection into the branch's sealed sections, so a
    /// conversation opened on a non-default assignment names the branch it will
    /// actually run.
    ///
    /// A schema declaring several trees gets all of them spliced. Re-deriving
    /// only the first would splice that tree's ids into whichever window the
    /// spans happened to name — the two disagree the moment a second tree
    /// exists, which zend's bundled schema is.
    ///
    /// Collection members are skipped, for the same reason the section-ingest
    /// chain skips them: projection picks a subset at runtime, so a prefix that
    /// counted them would change whenever the catalog did.
    ///
    /// A reserved range is a hole in `birth` and stays one: the prompt's tokens
    /// come back as runs, each after the hole ahead of it, and the content
    /// prefix names the holes as well as the sections.
    pub(super) fn prompt_branch(
        &self,
        birth: &[PrefixItem],
        branch_spans: &[(usize, usize)],
    ) -> (ContentHash, Vec<TokenRun>) {
        let schema = self.projection.schema();
        let trees: Vec<&SectionTree> = schema.system_prompt.section_trees().collect();
        let mut ids: Vec<PrefixItem> = Vec::with_capacity(birth.len());
        // Walk the windows in order, copying the untouched runs between them
        // and re-deriving each tree's own run. `cursor` is where the last
        // window ended, so an out-of-order or out-of-range span is caught
        // rather than producing a scrambled prefix.
        let mut cursor = 0usize;
        let mut spliced = trees.len() == branch_spans.len();
        if spliced {
            for (tree, &(start, len)) in trees.iter().zip(branch_spans) {
                if start < cursor || start + len > birth.len() {
                    spliced = false;
                    break;
                }
                ids.extend_from_slice(&birth[cursor..start]);
                let selection = tree.selection(|id| self.selection.get(id));
                // The branch, every node in its slot as the default's were
                // (`docs/placed_section_layout.md` §3.1c).
                let tokens_of = |id: SectionId| self.substrate.read().section_tokens_of(id).len();
                ids.extend(tree_prefix(tree, tree.nodes.len(), &selection, tokens_of));
                cursor = start + len;
            }
        }
        if spliced {
            ids.extend_from_slice(&birth[cursor..]);
        } else {
            // No trees, or spans that no longer index `birth`: the birth list
            // is the prompt.
            ids = birth.to_vec();
        }

        ids.retain(|item| {
            item.section()
                .is_none_or(|id| !schema.system_prompt.is_collection_member(id))
        });
        chain_of(&self.substrate, &ids)
    }

    /// Compute and persist this conversation's prompt branch checkpoint (§4.6).
    ///
    /// A checkpoint is valid for an *ordered prefix* and nothing weaker: state
    /// does not compose, so `state([A][B])` is not any function of `state([A])`
    /// and `state([B])`. It is keyed by the branch's content prefix, which is
    /// what the sealed K/V is already addressed by, so the two go stale
    /// together.
    ///
    /// **One branch, not the cross-product.** §4.6 describes pre-sealing every
    /// leaf the way the K/V tree does, and measurement made that obviously
    /// wrong: the live tree has 200 branches and a conversation uses one, so
    /// the eager version put 199 unused full-prompt prefills in front of the
    /// first turn. Each branch is computed the first time some conversation
    /// selects it and persisted, so the cost is one prefill per distinct branch
    /// *ever used*, paid once across every conversation and restart.
    ///
    /// Returns `(prefix, freshly computed payload)`. The payload comes back in
    /// memory rather than being re-read, and that is correctness rather than
    /// economy: the write is fire-and-forget onto the persistence thread, so a
    /// read issued immediately after finds nothing. The first version of this
    /// restored from disk and installed **zero** checkpoints out of two hundred
    /// computed, silently, because a miss is a warning and not an error.
    ///
    /// Failures are logged, never fatal. A missing checkpoint costs the first
    /// turn its prompt memory, which is a degradation; refusing to open the
    /// conversation is an outage.
    pub(super) fn build_branch_checkpoint(
        &self,
        birth: &[PrefixItem],
        branch_spans: &[(usize, usize)],
    ) -> crate::Result<Option<(ContentHash, Option<Arc<BranchCheckpointPayload>>)>> {
        let (prefix, runs) = self.prompt_branch(birth, branch_spans);
        if runs_are_empty(&runs) {
            return Ok(None);
        }
        // An already-durable checkpoint comes back decoded rather than being
        // dropped: `read_branch_checkpoint` reads the whole ~63 MiB record off
        // the log, and returning `None` here made `restore_branch_checkpoint`
        // read and decode the identical record a moment later.
        match checkpoint(&self.scheduler_tx, &self.substrate, prefix, runs) {
            Ok(Some((payload, computed))) => {
                if computed {
                    note_branch_checkpoint_computed();
                    tracing::info!("computed the prompt branch checkpoint");
                }
                Ok(Some((prefix, Some(payload))))
            }
            // The model carries no recurrent state after all.
            Ok(None) => Ok(None),
            Err(ConversationError::SchedulerGone) => Err(ConversationError::SchedulerGone),
            Err(e) => {
                tracing::warn!(
                    "branch checkpoint pass failed ({e}) — the first turn will start \
                     without its prompt state, and will read perfectly while doing so"
                );
                Ok(Some((prefix, None)))
            }
        }
    }

    /// Install this conversation's branch checkpoint onto its slot (P6.4).
    ///
    /// A branch checkpoint describes the recurrent state after the prompt. It
    /// is installed before the first turn, whose projection lays the prompt's
    /// K/V onto the slot: placing K/V restores K/V and nothing else, so the
    /// installed state survives it and the first prefill reads both together.
    ///
    /// A miss is a degradation, not a failure: the conversation starts with its
    /// prompt in K/V and zero in the recurrent layers, exactly as it did before
    /// any of this existed. It is logged at WARN because that state reads
    /// perfectly.
    pub(super) fn restore_branch_checkpoint(
        &self,
        prefix: ContentHash,
        fresh: Option<Arc<BranchCheckpointPayload>>,
    ) -> crate::Result<()> {
        install_branch_states(
            &self.scheduler_tx,
            &[PendingBranchState {
                sequence_id: self.id,
                prefix,
                payload: fresh,
            }],
            &self.substrate,
        )
    }

    /// Install an observer channel for cognitive task events (summarization,
    /// daydream, etc.). When set, streaming events — `Token`, `Prefill`,
    /// `PrefillProgress`, `HealthWarning` — are forwarded to `tx` in real
    /// time as the task generates tokens. This allows callers to monitor
    /// background inference (e.g. print tokens as they stream).
    ///
    /// Pass `None` to remove the observer.
    pub fn set_task_observer(&mut self, tx: Option<Sender<TurnEvent>>) {
        self.tree.task_event_observer = tx;
    }
}
