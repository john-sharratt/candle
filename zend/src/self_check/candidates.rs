//! Which stored conversations the self-check asks.
//!
//! A conversation is checked when it lives in a layer the check can fork a
//! reader for (the dialogue, and the `repo_map` / `code_reading` ingest
//! layers), has at least one turn to judge, and is not a passthrough record —
//! a passthrough conversation is a client's own transcript replayed verbatim,
//! not something this daemon wrote.

use std::collections::{BTreeMap, HashSet};

use candle_conversation::projection::{LayerId, TimelineId};

use crate::code_read::PATH_KEY;
use crate::passthrough::CONV_ID_PREFIX;
use crate::repo_scan::DIR_KEY;

/// What the substrate says about one live timeline.
#[derive(Debug, Clone)]
pub struct Entry {
    pub timeline: TimelineId,
    pub layer: Option<LayerId>,
    pub turns: u64,
    pub conv_id: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

/// A conversation to ask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub timeline: TimelineId,
    pub layer: LayerId,
    /// How a report names it: its conversation id, its ingested path, or its
    /// folder, whichever it has.
    pub label: String,
}

/// The entries to check, in timeline order so a report reads the same from
/// run to run.
pub fn select(
    entries: impl IntoIterator<Item = Entry>,
    layers: &HashSet<LayerId>,
) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = entries
        .into_iter()
        .filter(|e| e.turns > 0)
        .filter(|e| {
            !e.conv_id
                .as_deref()
                .is_some_and(|id| id.starts_with(CONV_ID_PREFIX))
        })
        .filter_map(|e| {
            let layer = e.layer.filter(|l| layers.contains(l))?;
            let label = e
                .conv_id
                .clone()
                .or_else(|| e.metadata.get(PATH_KEY).cloned())
                .or_else(|| e.metadata.get(DIR_KEY).cloned())
                .unwrap_or_else(|| "-".to_string());
            Some(Candidate {
                timeline: e.timeline,
                layer,
                label,
            })
        })
        .collect();
    out.sort_by_key(|c| c.timeline.raw());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(raw: u64, layer: u32, turns: u64) -> Entry {
        Entry {
            timeline: TimelineId::for_test(raw),
            layer: Some(LayerId::for_test(layer)),
            turns,
            conv_id: None,
            metadata: BTreeMap::new(),
        }
    }

    fn layers(ids: &[u32]) -> HashSet<LayerId> {
        ids.iter().map(|&l| LayerId::for_test(l)).collect()
    }

    #[test]
    fn only_checked_layers_with_turns_are_asked() {
        let picked = select(
            [
                entry(3, 1, 2),
                entry(1, 1, 0),
                entry(2, 9, 4),
                entry(4, 2, 1),
            ],
            &layers(&[1, 2]),
        );
        let ids: Vec<u64> = picked.iter().map(|c| c.timeline.raw()).collect();
        assert_eq!(ids, [3, 4]);
    }

    #[test]
    fn a_passthrough_transcript_is_not_asked() {
        let mut e = entry(1, 1, 3);
        e.conv_id = Some(format!("{CONV_ID_PREFIX}abc"));
        assert!(select([e], &layers(&[1])).is_empty());
    }

    #[test]
    fn a_timeline_with_no_layer_is_not_asked() {
        let mut e = entry(1, 1, 3);
        e.layer = None;
        assert!(select([e], &layers(&[1])).is_empty());
    }

    #[test]
    fn the_label_is_the_conv_id_then_the_path_then_the_folder() {
        let mut a = entry(1, 1, 1);
        a.conv_id = Some("chat-7".into());
        a.metadata.insert(PATH_KEY.into(), "x.rs".into());
        let mut b = entry(2, 1, 1);
        b.metadata.insert(PATH_KEY.into(), "candle/a.rs".into());
        b.metadata.insert(DIR_KEY.into(), "candle".into());
        let mut c = entry(3, 1, 1);
        c.metadata.insert(DIR_KEY.into(), "candle/src".into());
        let d = entry(4, 1, 1);
        let labels: Vec<String> = select([a, b, c, d], &layers(&[1]))
            .into_iter()
            .map(|c| c.label)
            .collect();
        assert_eq!(labels, ["chat-7", "candle/a.rs", "candle/src", "-"]);
    }
}
