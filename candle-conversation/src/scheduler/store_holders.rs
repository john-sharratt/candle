//! Every recurrent store on the device, by the sequence that holds it, and why
//! it is still there.
//!
//! The memory report's slot rows say whose slots exist; they do not say which
//! of them hold a store, what each is doing, or why an idle one was not parked.
//! A store is ~200 MB of ground the expert zone gives up, and on WhiteWorld
//! (2026-11-23) 10–14 slots stood idle for minutes beside 18 live stores and
//! none parked — a question the slot rows could not answer. So each store is
//! listed with the facts the parking decision is made from
//! (`recurrent_park`): what runs or waits on its sequence, how long it has been
//! idle, and which condition keeps it.
//!
//! The model is asked which sequences hold a store (`recurrent_holders`)
//! rather than the scheduler's slots being asked whether they do, so a store
//! no slot accounts for any more is listed too.

use std::time::Instant;

use serde::Serialize;

use super::state_census::GroupNames;
use super::{Scheduler, SequenceId};
use crate::projection::SubstrateRef;

/// One recurrent store on the device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoreHolder {
    /// The sequence holding it.
    pub seq: u64,
    /// `slot`, a conversation's own sequence; `view`, a turn's view forked
    /// from one (`parent` names it); or `unaccounted`, a sequence the
    /// scheduler holds no slot or view for.
    pub role: &'static str,
    /// A view's parent sequence.
    pub parent: Option<u64>,
    /// The conversation it serves — its `conv_id`, else its label, else its
    /// timeline — or empty for a sequence with no projection target.
    pub holder: String,
    pub layer: String,
    pub group: String,
    /// What runs on or waits for the sequence now; empty when idle.
    pub doing: Vec<&'static str>,
    /// How long the sequence has stood idle, as the last rate pricing saw it.
    pub idle_secs: Option<u64>,
    /// Why the store has not been parked.
    pub kept_because: String,
}

impl Scheduler {
    /// Every store on the device, by sequence.
    pub(super) fn store_holders(
        &self,
        names: &[(SubstrateRef, GroupNames)],
        now: Instant,
    ) -> Vec<StoreHolder> {
        self.model
            .recurrent_holders()
            .into_iter()
            .map(|raw| self.store_holder(SequenceId(raw), names, now))
            .collect()
    }

    fn store_holder(
        &self,
        seq: SequenceId,
        names: &[(SubstrateRef, GroupNames)],
        now: Instant,
    ) -> StoreHolder {
        let view_of = self.turn_views.get(&seq).map(|v| v.parent_id);
        let owner = view_of.unwrap_or(seq);
        let (role, accounted) = match view_of {
            Some(_) => ("view", true),
            None if self.slot_conversations.contains_key(&seq) => ("slot", true),
            None => ("unaccounted", false),
        };
        let (holder, layer, group) = self.slot_names(owner, names);
        let doing = self.slot_doing(seq);
        let idle_secs = self
            .idle_since
            .get(&seq)
            .map(|t| now.saturating_duration_since(*t).as_secs());
        let kept_because = if !accounted {
            "no slot or view accounts for this sequence".to_string()
        } else if view_of.is_some() {
            "a turn's view: freed when its turn finishes".to_string()
        } else if !doing.is_empty() {
            "in use".to_string()
        } else if let Some((timeline, turn)) = self.sealed_snapshots.get(&seq) {
            format!(
                "idle, waiting for its seal of turn {turn} on timeline {} to be durable",
                timeline.raw()
            )
        } else {
            "idle, and no seal since its store was built: nothing parks it".to_string()
        };
        StoreHolder {
            seq: seq.0 as u64,
            role,
            parent: view_of.map(|p| p.0 as u64),
            holder,
            layer,
            group,
            doing,
            idle_secs,
            kept_because,
        }
    }

    /// What runs on or waits for `slot` — each condition `slot_idle` checks,
    /// by name.
    pub(super) fn slot_doing(&self, slot: SequenceId) -> Vec<&'static str> {
        let mut doing = Vec::new();
        if self.turn_views.values().any(|v| v.parent_id == slot) {
            doing.push("parent of a turn's view");
        }
        if self.active_decodes.contains_key(&slot) {
            doing.push("decoding");
        }
        if self
            .active_prefills
            .iter()
            .any(|p| p.work.sequence_id == slot)
        {
            doing.push("prefilling");
        }
        if self.prefill_queue.iter().any(|w| w.sequence_id == slot) {
            doing.push("prefill queued");
        }
        if self.section_queue.iter().any(|s| s.sequence_id == slot) {
            doing.push("section queued");
        }
        if self
            .active_section_ingests
            .iter()
            .any(|s| s.sequence_id == slot)
        {
            doing.push("ingesting a section");
        }
        if self.pending_submits.iter().any(|p| p.parent() == slot) {
            doing.push("a submit waiting on it");
        }
        if self.fork_waits_on(slot) {
            doing.push("a fork waiting on it");
        }
        doing
    }

    /// `slot`'s conversation, layer and group, as the slot rows name them:
    /// empty for a sequence with no projection target.
    pub(super) fn slot_names(
        &self,
        slot: SequenceId,
        names: &[(SubstrateRef, GroupNames)],
    ) -> (String, String, String) {
        let Some(target) = self.slot_targets.get(&slot) else {
            return (String::new(), String::new(), String::new());
        };
        let conv = self.slot_conversations.get(&slot);
        let (layer, group) = conv
            .and_then(|conv| names.iter().find(|(r, _)| r.is(conv)))
            .and_then(|(_, n)| n.get(&target.group))
            .cloned()
            .unwrap_or_default();
        let holder = conv
            .and_then(|c| {
                c.conv_id_of(target.timeline)
                    .or_else(|| c.label_of(target.timeline))
            })
            .unwrap_or_else(|| format!("timeline {}", target.timeline.raw()));
        (holder, layer, group)
    }
}
