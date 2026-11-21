//! Records the composer dials a conversation ran under and maps them to the
//! settings a turn runs with.
//!
//! The record stores the projection's own option ids rather than the GUI's
//! levels, so a stored id keeps its meaning if the GUI changes its scale, and a
//! resumed turn re-selects the same settings. Thinking effort becomes the
//! steering think mode, and response length the answer budget. A `/no_think`
//! turn drops the thinking-temperature boost, which is meant for the reasoning
//! span. A conversation that predates the record reports no levels rather than
//! invented ones.

use super::*;

/// Metadata keys carrying the composer dials a conversation last ran under.
///
/// The values are the projection's own option ids (`balanced`, `standard`, …),
/// not the GUI's levels: the daemon speaks in ids, the ids are what a resumed
/// turn re-selects, and a stored id keeps its meaning if the GUI ever changes
/// its scale. [`ZendSession::conversation_dials`] maps them back to levels for
/// the client.
pub(super) const DIAL_EFFORT_KEY: &str = "dial_effort";

pub(super) const DIAL_LENGTH_KEY: &str = "dial_response_length";

pub(super) const DIAL_THINK_KEY: &str = "dial_think";

pub(super) const DIAL_TOOLS_KEY: &str = "dial_tools";

/// The dials `selection` and `tools_mode` express, as the metadata bag stores
/// them. A selector the turn left unset is stored empty, so "ran without this
/// dial" stays distinguishable from "never recorded".
pub(super) fn dials_metadata(
    selection: &candle_conversation::SelectionState,
    tools_mode: ToolMode,
) -> BTreeMap<String, String> {
    let think = match selection.optional(candle_conversation::NO_THINK_SELECTOR) {
        // The `/no_think` node present means the turn suppressed thinking.
        Some(candle_conversation::OptionalState::Present) => "off",
        Some(candle_conversation::OptionalState::Absent) => "on",
        None => "",
    };
    BTreeMap::from([
        (
            DIAL_EFFORT_KEY.to_string(),
            selection
                .get("thinking_effort")
                .unwrap_or_default()
                .to_string(),
        ),
        (
            DIAL_LENGTH_KEY.to_string(),
            selection
                .get("response_length")
                .unwrap_or_default()
                .to_string(),
        ),
        (DIAL_THINK_KEY.to_string(), think.to_string()),
        (DIAL_TOOLS_KEY.to_string(), tools_mode.id().to_string()),
    ])
}

/// The composer dials a conversation last ran under, as the levels the GUI's
/// dials are set from.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ConversationDials {
    pub effort: u8,
    pub verbosity: u8,
    pub think: bool,
    pub tools: u8,
}

/// Map a turn's `thinking_effort` dial to the steering [`ThinkMode`].  Mirrors
/// `dial_selection` in `api/chat.rs`: effort 0 → `off`, whose tree closes the
/// block on the token after `<think>`; an unset dial defaults to the
/// projection's `balanced` (free flow).
pub(super) fn think_mode_from_selection(
    selection: &candle_conversation::SelectionState,
) -> ThinkMode {
    match selection.get("thinking_effort") {
        Some("off") => ThinkMode::Off,
        Some("quick") => ThinkMode::Quick,
        Some("deep") => ThinkMode::Deep,
        Some("exhaustive") => ThinkMode::Exhaustive,
        // Explicit `balanced`, or no dial set (projection default) → free flow.
        _ => ThinkMode::Balanced,
    }
}

/// Maps the composer's `response_length` dial (terse/concise/standard/detailed/
/// comprehensive = 0..4, the verbosity toggle) to the answer's token budget — the
/// room reserved after the think block, which the total-length EOS ramp covers
/// (`ThinkMode::eos_budget`).  These are generous hard caps (the typical answer is
/// shorter); the EOS boost/failsafe makes them the turn-ender backstop.
pub(super) fn response_budget_from_selection(
    selection: &candle_conversation::SelectionState,
) -> i32 {
    match selection.get("response_length") {
        Some("terse") => 512,
        Some("concise") => 1024,
        Some("detailed") => 4096,
        Some("comprehensive") => 7168,
        // Explicit `standard`, or no dial set (projection default).
        _ => 2048,
    }
}

/// A chat turn's sampling config. A caller's explicit config (e.g. a test's
/// `argmax()`) is honoured verbatim; the conversation default gets the per-turn
/// adjustments and `seed`. Either way the dials' thinking and answer budget is
/// steered in: the effort level sets the think block's close thresholds, and
/// `response_length` the answer's room above them (see `think_budget::steer`).
pub(super) fn turn_sampling(
    conv: &Sequence,
    explicit: Option<SamplingConfig>,
    selection: &SelectionState,
    think_closer: &[u32],
    seed: u64,
) -> SamplingConfig {
    let think_mode = think_mode_from_selection(selection);
    let mut sampling = match explicit {
        Some(explicit) => explicit,
        None => {
            let mut sampling = conv.default_sampling();
            // A `/no_think` turn is a direct response, not reasoning: strip the
            // thinking-temperature boost, which is meant for the `<think>` span
            // and has no business heating a plain answer.
            if think_mode == ThinkMode::Off {
                sampling.segment_temp_boost = 0.0;
            }
            sampling.seed = seed;
            sampling
        }
    };
    think_budget::steer(
        &mut sampling,
        think_mode,
        response_budget_from_selection(selection),
        think_closer,
    );
    sampling
}

impl ZendSession {
    pub fn conversation_dials(&self, conv_id: &str) -> Option<ConversationDials> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let dials = |meta: &POrdMap<String, String>| -> Option<ConversationDials> {
            let level = |key: &str, options: &[&str]| -> Option<u8> {
                let id = meta.get(key)?;
                options.iter().position(|o| o == id).map(|i| i as u8)
            };
            let effort = level(DIAL_EFFORT_KEY, &EFFORT_OPTIONS);
            let verbosity = level(DIAL_LENGTH_KEY, &RESPONSE_LENGTH_OPTIONS);
            let tool_ids = ToolMode::ALL.map(ToolMode::id);
            let tools = level(DIAL_TOOLS_KEY, &tool_ids);
            let think = meta.get(DIAL_THINK_KEY).map(|v| v != "off");
            // Nothing recorded at all: the conversation predates the dial record, or
            // has never taken a turn. Say so rather than inventing levels.
            //
            // The test is whether any dial key is present, NOT whether effort parses
            // to a level. `dials_metadata` writes all four keys on every turn and
            // stores `""` for a selector the turn left unset, so a request carrying
            // a tools dial but no thinking effort records `dial_effort = ""` — a
            // record that plainly exists while parsing to no level. Gating on effort
            // dropped the whole record there, and a resumed turn then ran the
            // defaults rather than the tools mode the conversation was held at.
            let recorded = [
                DIAL_EFFORT_KEY,
                DIAL_LENGTH_KEY,
                DIAL_THINK_KEY,
                DIAL_TOOLS_KEY,
            ]
            .iter()
            .any(|key| meta.contains_key(*key));
            if !recorded {
                return None;
            }
            Some(ConversationDials {
                effort: effort.unwrap_or(2),
                verbosity: verbosity.unwrap_or(2),
                think: think.unwrap_or(true),
                tools: tools.unwrap_or(ToolMode::Comprehensive.level()),
            })
        };
        state.engine.conversation_metadata_with(timeline, dials)?
    }
}
