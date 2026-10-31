//! Layers that have left the schema, and the sweep that retires what they held.
//!
//! A unit's conversation records its layer by name, so it can be found after
//! the schema stops declaring the layer. Tombstones are ordinary writes, so a
//! sweep reaches every member through the log. It runs at every boot, on every
//! member: a member still on an older build goes on writing units until it
//! upgrades, and what it wrote is retired when it does.

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

use super::run::all_of;

/// Per-file analyses that could not keep up with the corpus: thousands of
/// units at minutes each, holding the forwards live work waits for.
pub const REMOVED_LAYERS: [&str; 3] = ["dependency_analysis", "code_review", "security_review"];

/// Retire every conversation of a removed layer, reporting how many went.
pub fn sweep(engine: &ConversationEngine) -> usize {
    REMOVED_LAYERS
        .iter()
        .map(|&layer| {
            let units = all_of(engine, layer);
            let retired = retire(&units, |tl| engine.tombstone_timeline(tl).is_ok());
            if retired > 0 {
                tracing::info!(
                    target: "zend::layers",
                    layer,
                    retired,
                    of = units.len(),
                    "retired the units of a removed layer",
                );
            }
            retired
        })
        .sum()
}

/// How many of `units` `tombstone` retired. One that fails does not stop the
/// rest; the next boot finds it again.
fn retire(units: &[TimelineId], mut tombstone: impl FnMut(TimelineId) -> bool) -> usize {
    units.iter().filter(|&&tl| tombstone(tl)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timeline(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    #[test]
    fn a_unit_that_fails_does_not_stop_the_rest() {
        let units = [timeline(1), timeline(2), timeline(3)];
        let mut asked = Vec::new();
        let retired = retire(&units, |tl| {
            asked.push(tl.raw());
            tl.raw() != 2
        });
        assert_eq!(retired, 2);
        assert_eq!(asked, vec![1, 2, 3]);
    }

    /// A removed layer stays removed: the schema declaring it again would
    /// have its units retired at every boot.
    #[test]
    fn the_schema_declares_no_removed_layer() {
        let yaml = include_str!("../prompts/projection.yaml");
        for layer in REMOVED_LAYERS {
            assert!(
                !yaml.contains(&format!("- name: {layer}")),
                "{layer} is declared and swept"
            );
        }
    }
}
