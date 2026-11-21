//! Builds the token-level triggers that constrain a turn's tool calls and its
//! thinking block.
//!
//! A turn's `<tool_call>` trigger forces any call the model emits into the exact
//! JSON shape of a tool the mode offers. The think block is steered per effort
//! dial by replacing the think trigger. Each round holds its opening call to a
//! journal note once it has reasoned long enough to have a conclusion to keep.
//! Tools taken out for a round are removed from its grammar. A held tool call is
//! wrapped for the client when it is a registered call, and otherwise streamed
//! verbatim as text.

use super::*;

/// A [`round_triggers`] held: its mode's level, its dial, the tools out.
pub(super) type HeldKey = (u8, String, Vec<&'static str>);

/// Every [`round_triggers`] held so far, swapped whole so a round never waits.
pub(super) type HeldStencils = HashMap<HeldKey, Arc<TriggerRegistry>>;

/// How many tokens of reasoning a round runs before its opening call is held
/// to `journal_note` (`docs/zend_journal.md` §5.2): below it there is no
/// conclusion to keep, and the round goes straight to its calls.
pub(super) const NOTE_AFTER_REASONING: u32 = 200;

/// [`turn_triggers`] for one round of a conversation, every one of which keeps
/// a journal: once the round has reasoned for [`NOTE_AFTER_REASONING`] tokens,
/// its opening call held to `journal_note` and a call after it held to come,
/// so a round that acts first notes what its thinking concluded and then acts
/// on it (`docs/zend_journal.md` §5.2); and a call to any of `unavailable`'s
/// tools taken out of the grammar. Held once
/// per mode, dial and set: a set changes only as sessions open and close, and
/// holding decodes the catalog's name arms. The base is what a round falls
/// back to, as a tool the grammar leaves in is only refused.
pub(super) fn round_triggers(
    state: &InferenceState,
    think_mode: ThinkMode,
    tools_mode: ToolMode,
    unavailable: &Unavailable,
) -> Arc<TriggerRegistry> {
    let base = turn_triggers(state, think_mode, tools_mode);
    let offered = crate::tools::offered_tool_names(tools_mode);
    let noting = offered.contains(JOURNAL_NOTE_TOOL);
    if unavailable.tools.is_empty() && !noting {
        return base;
    }
    let names: Vec<&'static str> = unavailable.tools.iter().copied().collect();
    let key = (tools_mode.level(), format!("{think_mode:?}"), names);
    if let Some(triggers) = state.held_stencils.load().get(&key) {
        return Arc::clone(triggers);
    }
    let allowed: Vec<String> = offered
        .into_iter()
        .filter(|name| !unavailable.tools.contains(name.as_str()))
        .collect();
    let first = noting.then_some((JOURNAL_NOTE_TOOL, NOTE_AFTER_REASONING));
    match state.engine.holding_tools(&base, &allowed, first) {
        Ok(triggers) => {
            state.held_stencils.rcu(|held| {
                let mut held = HeldStencils::clone(held);
                held.insert(key.clone(), Arc::clone(&triggers));
                held
            });
            triggers
        }
        Err(e) => {
            tracing::warn!("tools out of a round were left in its grammar: {e}");
            base
        }
    }
}

/// The triggers a chat turn decodes under: any `<tool_call>` the model emits is
/// forced to the exact JSON shape of a tool `tools_mode` offers (name ∈ that
/// mode's catalog, required params in order, valid JSON), and atop that the `<think>` block is steered per the
/// effort dial, replacing the `<think>` trigger atomically.
pub(super) fn turn_triggers(
    state: &InferenceState,
    think_mode: ThinkMode,
    tools_mode: ToolMode,
) -> Arc<TriggerRegistry> {
    let stencil = &state.tool_stencils[tools_mode.level() as usize];
    match &state.think_steering {
        Some(ts) => ts.registry_for(stencil, think_mode),
        None => Arc::clone(stencil),
    }
}

/// What each layer of units compounds on: every other layer of units below
/// it in the schema's stack, by name, with its group and whether that group is
/// scoped by branch (a layer of files) or not (a layer of days). A disabled
/// layer is out of service: it compounds on nothing and nothing on it.
pub(super) type LayerSees = Vec<(String, Vec<(String, GroupId, bool)>)>;

pub(super) fn layer_sees(
    builder: &Builder,
    layers: &[IngestLayer],
    disabled: &HashSet<String>,
) -> LayerSees {
    let schema = builder.schema();
    let rank = |name: &str| {
        schema
            .layers
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.rank)
    };
    let derived: Vec<(&IngestLayer, i32, GroupId, bool)> = layers
        .iter()
        .filter(|il| il.mode == IngestMode::Derived && !disabled.contains(&il.name))
        .filter_map(|il| {
            let scoped = il
                .spec
                .as_ref()
                .is_some_and(|s| s.unit == crate::layers::UnitKind::File);
            Some((
                il,
                rank(&il.name)?,
                builder.id_for_group(&il.group)?,
                scoped,
            ))
        })
        .collect();
    derived
        .iter()
        .map(|(il, r, _, _)| {
            let below = derived
                .iter()
                .filter(|(other, other_rank, _, _)| other.name != il.name && other_rank < r)
                .map(|(other, _, group, scoped)| (other.name.clone(), *group, *scoped))
                .collect();
            (il.name.clone(), below)
        })
        .collect()
}

/// Render the answer's held tool-call object for the client stream.  The GUI
/// renders tool cards from `<tool_call>…</tool_call>` tags, but Qwen3 frequently
/// emits the JSON object bare (no wrapper).  When `held` is a recognized tool call
/// (via the registry-gated [`extract_tool_calls`]), wrap it so the card renders;
/// otherwise return it verbatim (a bare object that isn't a call — e.g. a JSON
/// answer — streams as plain text).  `None` when there's nothing to emit.
pub(super) fn render_held_tail(held: &str) -> Option<String> {
    let held = held.trim();
    if held.is_empty() {
        return None;
    }
    if extract_tool_calls(held).is_empty() {
        Some(held.to_string())
    } else {
        Some(format!("<tool_call>\n{held}\n</tool_call>"))
    }
}

/// Closer phrase the sampler plays (followed by `</think>`) when the think
/// block's HARD token cap fires mid-sentence — the em-dash lead-in reads as
/// a deliberate self-interruption after any dangling fragment, and the
/// commitment ("know what to do") primes the answer that follows. Tokenized
/// once at startup into `InferenceState::think_closer_phrase`.
pub(super) const THINK_CLOSER_PHRASE: &str =
    " — actually, I've reasoned enough and know what to do.";
