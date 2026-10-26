//! Whose working sets hold the hot K/V no drain may move — reported once a
//! minute beside the learned rates.
//!
//! The pin set is the union of every live slot's working set
//! (`Scheduler::publish_working_set_pins`), and a pinned residence is skipped
//! by the hot→warm drain and every automatic hot-drop. When it grows, the
//! expert zone shrinks under it, and `pinned_undrainable_bytes` says how much
//! but not whose. This line names the timelines, how many slots' working sets
//! name each, and whether each is still live — the questions a growing pin set
//! raises.

use super::Scheduler;

/// Timelines the line names, largest first.
const PINNED_TOP: usize = 8;

impl Scheduler {
    /// Log, per attached substrate, the hot K/V its pins hold and the
    /// timelines holding most of it.
    pub(super) fn log_pinned_working_sets(&self) {
        let slots = self.slot_projection_state.len();
        for conversation in &self.attached_substrates {
            let sub = conversation.read();
            let pinned = sub.pinned_hot_by_timeline(PINNED_TOP);
            if pinned.pins == 0 {
                continue;
            }
            let top: Vec<String> = pinned
                .by_timeline
                .iter()
                .map(|&(timeline, turns, bytes)| {
                    let naming = self
                        .slot_projection_state
                        .values()
                        .filter(|st| st.working_set.turns.iter().any(|k| k.timeline == timeline))
                        .count();
                    let layer = sub
                        .timeline_target(timeline)
                        .map_or_else(|| "?".to_string(), |(l, _)| format!("{l:?}"));
                    let custom = sub.custom_of(timeline);
                    let field = |key: &str| {
                        custom
                            .and_then(|c| c.get(key))
                            .map_or("-", String::as_str)
                            .to_string()
                    };
                    format!(
                        "{} {layer} kind={} subject={} turns={turns} mib={} slots={naming}{}",
                        timeline.raw(),
                        field("kind"),
                        field("subject"),
                        bytes >> 20,
                        if sub.is_tombstoned(timeline) {
                            " TOMBSTONED"
                        } else {
                            ""
                        },
                    )
                })
                .collect();
            tracing::info!(
                target: "candle_conversation::scheduler::pins",
                slots,
                pins = pinned.pins,
                pinned_mib = pinned.pinned_bytes >> 20,
                turn_mib = pinned.turn_bytes >> 20,
                top = %top.join(" | "),
                "working-set pins hold hot K/V"
            );
        }
    }
}
