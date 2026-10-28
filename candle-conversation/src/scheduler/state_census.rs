//! The span's "state" band, broken down by whose ground it is.
//!
//! The performance page's VRAM chart shows the span's tenants — recurrent
//! state, the provenance gallery, the QSA index caches, a creep's residuals —
//! as one band (`candle_nn::kv_cache::VramBands::state`), and the memory
//! report's `span_tenants` splits it by tenant. Neither says *whose* the ground
//! is: which layer's turns the gallery holds, how much of it belongs to
//! conversations already tombstoned, how many sequences the recurrent state is.
//! This section does.
//!
//! - **The gallery** ([`GalleryCensus`]), per resident turn: its timeline live,
//!   tombstoned, or not in any substrate at all, who it is held for (every
//!   conversation, or its own — `docs/gallery_residency.md`), and the layer and
//!   group it was written to — summed into rows. Reading every resident turn's stream is a
//!   walk of the residency map and a lookup per turn, so it is taken at most
//!   once per [`CENSUS_EVERY`] and published as it stood then; a substrate
//!   whose lock a writer holds is skipped and counted rather than waited on,
//!   so the census never stalls a wave.
//! - **Recurrent state** ([`RecurrentBreakdown`]): stores live and parked, what
//!   one costs, and the live slots by the layer and group they serve. Taken at
//!   every publish, naming each slot's group from its substrate's names as the
//!   last census found them.
//!
//! **Names come from every schema captured for a substrate.** A substrate's
//! conversations do not share one builder — each layer's, and each
//! conversation's submitted sections, are their own — so a group is named by
//! whichever captured builder declares it ([`GroupNames`]). Group ids are
//! small integers local to a substrate's schema, so names are never looked up
//! across substrates: two daemons on one engine reuse the same ids.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use super::turn_status::{turn_in, TurnStatus};
use super::Scheduler;
use crate::projection::{
    Builder, GroupId, Reserved, Schema, Substrate, SubstrateRead, SubstrateRef,
};
use crate::provenance::gallery_arena::{HolderKind, ResidentRun};
use crate::provenance::GalleryArena;

/// How often the census is taken again.
pub const CENSUS_EVERY: Duration = Duration::from_secs(60);

/// A substrate's groups by id: each one's layer and group name.
pub type GroupNames = HashMap<GroupId, (String, String)>;

/// The engine's reserved groups, named by their kind. A reserved conversation
/// projects under a synthetic schema whose one layer and group are called
/// `dialogue` / `primary_conversation` whatever the kind, so named from its
/// builder every calibration question reads as a dialogue turn.
const RESERVED_GROUPS: [(Reserved, &str); 6] = [
    (Reserved::Titler, "titler"),
    (Reserved::ToolSummary, "tool_summary"),
    (Reserved::ToolSummaryRestricted, "tool_summary_restricted"),
    (Reserved::Calibration, "calibration"),
    (Reserved::Prose, "prose"),
    (Reserved::Passthrough, "passthrough"),
];

/// The census as it stood when last taken.
pub struct CensusCache {
    at: Instant,
    gallery: Option<GalleryCensus>,
    /// Each substrate the census could read, with its group names.
    names: Vec<(SubstrateRef, GroupNames)>,
}

/// The "state" band's breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct StateSection {
    /// The gallery's last census; `None` without a gallery arena.
    pub gallery: Option<GalleryCensus>,
    pub recurrent: RecurrentBreakdown,
}

/// What a resident turn is: its timeline's standing, who it is held for, and
/// the layer and group it was written to (empty when unknown).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnClass {
    pub status: TurnStatus,
    pub holder: HolderKind,
    pub layer: String,
    pub group: String,
}

/// Every resident turn of one class, summed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GalleryRow {
    pub status: TurnStatus,
    pub holder: HolderKind,
    pub layer: String,
    pub group: String,
    pub turns: usize,
    /// Bytes of the pages its turns hold.
    pub bytes: u64,
}

/// The gallery, by whose turns it holds.
#[derive(Debug, Clone, Serialize)]
pub struct GalleryCensus {
    /// Milliseconds since the Unix epoch when it was taken.
    pub taken_unix_ms: u64,
    /// Bytes of the pages resident turns hold — what the rows sum to.
    pub page_bytes: u64,
    /// Whole regions the gallery's arenas stand on: more than `page_bytes` by
    /// every arena's free slots and unused tail, since a region goes back only
    /// when its arena's last page does.
    pub region_bytes: u64,
    /// How often a turn must be scanned to keep its ground; `None` keeps
    /// every turn whatever its use.
    pub keep_within_secs: Option<u64>,
    /// Turns, and their bytes, idle for longer than `keep_within_secs` — what
    /// the next keep trade gives back.
    pub idle_past_keep_turns: usize,
    pub idle_past_keep_bytes: u64,
    /// Substrates whose lock a writer held when the census looked: their turns
    /// are counted [`TurnStatus::Absent`].
    pub substrates_skipped: usize,
    /// Largest first.
    pub rows: Vec<GalleryRow>,
}

/// Live sequences of one layer and group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SlotRow {
    pub layer: String,
    pub group: String,
    pub slots: usize,
}

/// Recurrent state, by the stores that hold it.
#[derive(Debug, Clone, Serialize)]
pub struct RecurrentBreakdown {
    /// Bytes its arenas hold (the census's `recurrent state` row).
    pub held_bytes: u64,
    /// What admitting one sequence costs: its store and its prefill scratch.
    /// A decoding sequence holds less — the scratch goes back once it decodes.
    pub store_bytes: u64,
    /// Stores of sequences with a slot.
    pub live_stores: usize,
    /// Stores parked for sequences between turns.
    pub parked_stores: usize,
    /// The engine's slots by the layer and group they serve, most first.
    pub slots: Vec<SlotRow>,
}

/// `runs` — the resident turns — summed by `class_of` each turn, largest
/// first.
pub fn tally(
    runs: &[ResidentRun],
    class_of: impl Fn(&ResidentRun) -> TurnClass,
) -> Vec<GalleryRow> {
    type Key = (TurnStatus, HolderKind, String, String);
    let mut by: HashMap<Key, (usize, u64)> = HashMap::new();
    for run in runs {
        let c = class_of(run);
        let row = by
            .entry((c.status, c.holder, c.layer, c.group))
            .or_default();
        row.0 += 1;
        row.1 += run.bytes;
    }
    let mut rows: Vec<GalleryRow> = by
        .into_iter()
        .map(
            |((status, holder, layer, group), (turns, bytes))| GalleryRow {
                status,
                holder,
                layer,
                group,
                turns,
                bytes,
            },
        )
        .collect();
    let key = |r: &GalleryRow| (r.status, r.holder, r.layer.clone(), r.group.clone());
    rows.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| key(a).cmp(&key(b))));
    rows
}

/// `rows` of slots, most first, ties by name.
fn slot_rows(counts: HashMap<(String, String), usize>) -> Vec<SlotRow> {
    let mut rows: Vec<SlotRow> = counts
        .into_iter()
        .map(|((layer, group), slots)| SlotRow {
            layer,
            group,
            slots,
        })
        .collect();
    rows.sort_by(|a, b| {
        b.slots
            .cmp(&a.slots)
            .then_with(|| (&a.layer, &a.group).cmp(&(&b.layer, &b.group)))
    });
    rows
}

/// `groups` — `(id, layer, group)` as each schema declares them — by id, the
/// first declaration of an id naming it.
pub fn group_names<'a>(
    groups: impl IntoIterator<Item = (GroupId, &'a str, &'a str)>,
) -> GroupNames {
    let mut out = GroupNames::new();
    for (id, layer, group) in groups {
        out.entry(id)
            .or_insert_with(|| (layer.to_string(), group.to_string()));
    }
    out
}

/// Every group `schema` declares, as `(id, layer, group)`.
fn schema_groups(schema: &Schema) -> impl Iterator<Item = (GroupId, &str, &str)> {
    schema.layers.iter().flat_map(|l| {
        l.groups
            .iter()
            .map(move |g| (g.id, l.name.as_str(), g.name.as_str()))
    })
}

/// The gallery's census over `reads`, each with the group names of its
/// substrate's schemas; `skipped` substrates could not be read.
fn census_gallery(
    arena: &GalleryArena,
    reads: &[(SubstrateRead<'_>, &GroupNames)],
    skipped: usize,
) -> GalleryCensus {
    let runs = arena.resident_runs();
    let keep = arena.keep_within();
    let (idle_past_keep_turns, idle_past_keep_bytes) = match keep {
        Some(k) => runs
            .iter()
            .filter(|r| r.idle > k)
            .fold((0, 0), |(n, b), r| (n + 1, b + r.bytes)),
        None => (0, 0),
    };
    let class_of = |run: &ResidentRun| {
        for (read, names) in reads {
            let Some((status, group)) = turn_in(read, run.sid) else {
                continue;
            };
            let (layer, group) = group
                .and_then(|g| names.get(&g))
                .cloned()
                .unwrap_or_default();
            return TurnClass {
                status,
                holder: run.holder.kind(),
                layer,
                group,
            };
        }
        TurnClass {
            status: TurnStatus::Absent,
            holder: run.holder.kind(),
            layer: String::new(),
            group: String::new(),
        }
    };
    GalleryCensus {
        taken_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64),
        page_bytes: runs.iter().map(|r| r.bytes).sum(),
        region_bytes: arena.resident_bytes(),
        keep_within_secs: keep.map(|k| k.as_secs()),
        idle_past_keep_turns,
        idle_past_keep_bytes,
        substrates_skipped: skipped,
        rows: tally(&runs, class_of),
    }
}

impl Scheduler {
    /// The "state" band's breakdown, from a census at most [`CENSUS_EVERY`]
    /// old. `recurrent_held` is the census's `recurrent state` row, taken in
    /// the same breath as the report's.
    pub(super) fn state_section(&mut self, recurrent_held: u64) -> StateSection {
        let stale = match &self.state_census {
            Some(cache) => cache.at.elapsed() >= CENSUS_EVERY,
            None => true,
        };
        if stale {
            self.state_census = Some(self.take_census());
        }
        let cache = self.state_census.as_ref().expect("census taken above");
        StateSection {
            gallery: cache.gallery.clone(),
            recurrent: self.recurrent_breakdown(recurrent_held, &cache.names),
        }
    }

    fn take_census(&self) -> CensusCache {
        let substrates = self.distinct_substrates();
        let mut skipped = 0usize;
        let mut reads = Vec::new();
        let mut names = Vec::new();
        for conv in &substrates {
            let Some(read) = conv.try_read() else {
                skipped += 1;
                continue;
            };
            names.push((conv.substrate_ref(), self.group_names_of(&read)));
            reads.push(read);
        }
        let gallery = self.gallery_arena.as_ref().map(|arena| {
            let paired: Vec<(SubstrateRead<'_>, &GroupNames)> = reads
                .into_iter()
                .zip(names.iter().map(|(_, n)| n))
                .collect();
            census_gallery(arena, &paired, skipped)
        });
        CensusCache {
            at: Instant::now(),
            gallery,
            names,
        }
    }

    /// The group names of every builder captured for one of `substrate`'s
    /// timelines, the reserved groups named by their kind first.
    fn group_names_of(&self, substrate: &Substrate) -> GroupNames {
        let mut seen: HashSet<*const Builder> = HashSet::new();
        group_names(
            RESERVED_GROUPS
                .iter()
                .map(|&(kind, name)| (GroupId::reserved(kind), name, name))
                .chain(
                    self.timeline_projections
                        .iter()
                        .filter(|(tl, _)| substrate.custom_of(**tl).is_some())
                        .filter(|(_, b)| seen.insert(Arc::as_ptr(b)))
                        .flat_map(|(_, b)| schema_groups(b.schema())),
                ),
        )
    }

    fn recurrent_breakdown(
        &self,
        held_bytes: u64,
        names: &[(SubstrateRef, GroupNames)],
    ) -> RecurrentBreakdown {
        let (live_stores, parked_stores) = self.model.recurrent_census();
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for (slot, target) in &self.slot_targets {
            let named = self
                .slot_conversations
                .get(slot)
                .and_then(|conv| names.iter().find(|(r, _)| r.is(conv)))
                .and_then(|(_, n)| n.get(&target.group))
                .cloned()
                .unwrap_or_default();
            *counts.entry(named).or_default() += 1;
        }
        let unbound = self
            .slot_conversations
            .keys()
            .filter(|s| !self.slot_targets.contains_key(s))
            .count();
        if unbound > 0 {
            *counts
                .entry(("(none)".to_string(), "(no projection target)".to_string()))
                .or_default() += unbound;
        }
        RecurrentBreakdown {
            held_bytes,
            store_bytes: self.model.recurrent_store_bytes() as u64,
            live_stores,
            parked_stores,
            slots: slot_rows(counts),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::streams::StreamId;
    use crate::projection::TimelineId;
    use crate::provenance::gallery_arena::Holder;

    fn class(run: &ResidentRun, status: TurnStatus, layer: &str, group: &str) -> TurnClass {
        TurnClass {
            status,
            holder: run.holder.kind(),
            layer: layer.into(),
            group: group.into(),
        }
    }

    fn run(sid: u64, bytes: u64, holder: Holder) -> ResidentRun {
        ResidentRun {
            sid: StreamId(sid),
            bytes,
            idle: Duration::ZERO,
            holder,
        }
    }

    fn row(
        status: TurnStatus,
        holder: HolderKind,
        layer: &str,
        group: &str,
        turns: usize,
        bytes: u64,
    ) -> GalleryRow {
        GalleryRow {
            status,
            holder,
            layer: layer.into(),
            group: group.into(),
            turns,
            bytes,
        }
    }

    /// **The gallery is summed by whose turns it holds**, largest first: two
    /// live code_reading turns are one row, a tombstoned turn of the same group
    /// is a row of its own, a dialogue turn held for its conversation is one
    /// more, and a stream no substrate declares is absent.
    #[test]
    fn resident_turns_are_summed_by_status_holder_layer_and_group() {
        let chat = Holder::Conversation(TimelineId::from_raw(9).unwrap());
        let runs = [
            run(1, 6144, Holder::Shared),
            run(2, 12288, Holder::Shared),
            run(3, 6144, Holder::Shared),
            run(4, 30720, Holder::Shared),
            run(5, 24576, chat),
        ];
        let rows = tally(&runs, |r| match r.sid.0 {
            1 | 2 => class(r, TurnStatus::Live, "code_reading", "files"),
            3 => class(r, TurnStatus::Tombstoned, "code_reading", "files"),
            5 => class(r, TurnStatus::Live, "dialogue", "chat"),
            _ => class(r, TurnStatus::Absent, "", ""),
        });
        use HolderKind::{Conversation, Shared};
        use TurnStatus::{Absent, Live, Tombstoned};
        assert_eq!(
            rows,
            vec![
                row(Absent, Shared, "", "", 1, 30720),
                row(Live, Conversation, "dialogue", "chat", 1, 24576),
                row(Live, Shared, "code_reading", "files", 2, 18432),
                row(Tombstoned, Shared, "code_reading", "files", 1, 6144),
            ]
        );
    }

    #[test]
    fn an_empty_gallery_has_no_rows() {
        assert!(tally(&[], |r| class(r, TurnStatus::Live, "", "")).is_empty());
    }

    /// **A substrate's groups are named by every schema captured for it**: a
    /// group only the second schema declares is still named, and an id two
    /// schemas both declare keeps the first one's names.
    #[test]
    fn groups_are_named_by_whichever_schema_declares_them() {
        let id = |raw| GroupId::from_raw(raw).expect("group id");
        let names = group_names([
            (id(1), "code_reading", "scopes"),
            (id(2), "repo_map", "structure"),
            (id(2), "dialogue", "conversation"),
            (id(3), "dependency_analysis", "dependencies"),
        ]);
        let mut got: Vec<(u32, &str, &str)> = names
            .iter()
            .map(|(g, (l, n))| (g.raw(), l.as_str(), n.as_str()))
            .collect();
        got.sort();
        assert_eq!(
            got,
            [
                (1, "code_reading", "scopes"),
                (2, "repo_map", "structure"),
                (3, "dependency_analysis", "dependencies"),
            ]
        );
    }

    /// Slots are listed most first, a tie broken by name.
    #[test]
    fn slots_are_listed_most_first() {
        let mut counts = HashMap::new();
        counts.insert(("code_reading".to_string(), "files".to_string()), 3);
        counts.insert(("code_review".to_string(), "code_reviews".to_string()), 16);
        counts.insert(("dialogue".to_string(), "main".to_string()), 3);
        let rows = slot_rows(counts);
        let order: Vec<(&str, usize)> = rows.iter().map(|r| (r.layer.as_str(), r.slots)).collect();
        assert_eq!(
            order,
            [("code_review", 16), ("code_reading", 3), ("dialogue", 3)]
        );
    }
}
