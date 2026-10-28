//! Giving up the gallery's turns that no scan can read any more
//! (`docs/gallery_residency.md` §3): a conversation's own turns once it is no
//! longer loaded, a tombstoned timeline's or turn's, and a turn no attached
//! substrate declares.
//!
//! The keep trade gives up turns that are *idle*; this gives up turns that are
//! *dead*, whatever their use. It runs between forwards, beside the keep
//! trade, and only when something it judges by has changed: a slot was freed
//! (a conversation may have unloaded), or a substrate's tombstone epoch moved.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use super::turn_status::{turn_in, TurnStatus};
use super::Scheduler;
use crate::persistence::streams::StreamId;
use crate::projection::{SubstrateRef, TimelineId};
use crate::provenance::gallery_arena::{Holder, ResidentRun};

/// The least time between two sweeps, and between two looks at whether one is
/// due: a burst of freed slots (a code_read scope's forks churn by the dozen)
/// is one walk of the residency map, not one per slot, and a wave that changes
/// nothing costs one clock read.
pub const SWEEP_EVERY: Duration = Duration::from_secs(2);

/// Why a resident turn goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doom {
    /// Held for one conversation, and nothing targets it.
    Unloaded,
    /// Its timeline, or the turn itself, is tombstoned.
    Tombstoned,
    /// No attached substrate declares it.
    Absent,
}

/// What the sweep has seen, and what it owes.
#[derive(Default)]
pub struct SweepState {
    /// A slot was freed, or a doomed turn was pinned, since the last sweep.
    owed: bool,
    /// Each substrate's tombstone epoch when a sweep last read it.
    tombstones_seen: Vec<(SubstrateRef, u64)>,
    last: Option<Instant>,
}

impl SweepState {
    /// A slot was freed: the conversation it served may have unloaded.
    pub fn slot_freed(&mut self) {
        self.owed = true;
    }
}

/// Which of `runs` go, and why. `loaded` is every timeline a live slot
/// targets. `status_of`, when the substrates were read, says where each turn
/// stands; `judge_absent` is whether every substrate could be read, so that a
/// turn no substrate declares is genuinely in none.
pub fn doomed(
    runs: &[ResidentRun],
    loaded: &HashSet<TimelineId>,
    status_of: Option<&dyn Fn(StreamId) -> TurnStatus>,
    judge_absent: bool,
) -> Vec<(StreamId, Doom)> {
    runs.iter()
        .filter_map(|run| {
            if let Holder::Conversation(tl) = run.holder {
                if !loaded.contains(&tl) {
                    return Some((run.sid, Doom::Unloaded));
                }
            }
            match status_of.map(|of| of(run.sid)) {
                Some(TurnStatus::Tombstoned) => Some((run.sid, Doom::Tombstoned)),
                Some(TurnStatus::Absent) if judge_absent => Some((run.sid, Doom::Absent)),
                _ => None,
            }
        })
        .collect()
}

impl Scheduler {
    /// Give up the gallery's dead turns, if anything they are judged by has
    /// changed and [`SWEEP_EVERY`] has passed since the last sweep.
    pub(super) fn sweep_gallery(&mut self) {
        let Some(arena) = self.gallery_arena.clone() else {
            return;
        };
        if self
            .gallery_sweep
            .last
            .is_some_and(|at| at.elapsed() < SWEEP_EVERY)
        {
            return;
        }
        // Even the look costs a lock attempt per substrate, so it is paced too.
        self.gallery_sweep.last = Some(Instant::now());
        // Which substrates have tombstoned something since the last read. A
        // substrate whose lock a writer holds is skipped now and read later.
        let substrates = self.distinct_substrates();
        let mut moved = false;
        let mut epochs = Vec::with_capacity(substrates.len());
        for conv in &substrates {
            let Some(read) = conv.try_read() else {
                moved = true;
                continue;
            };
            let epoch = read.tombstone_epoch();
            let seen = self
                .gallery_sweep
                .tombstones_seen
                .iter()
                .find(|(r, _)| r.is(conv))
                .map(|(_, e)| *e);
            moved |= seen != Some(epoch);
            epochs.push((conv.substrate_ref(), epoch));
        }
        if !self.gallery_sweep.owed && !moved {
            return;
        }
        let runs = arena.resident_runs();
        let loaded: HashSet<TimelineId> = self.slot_targets.values().map(|t| t.timeline).collect();
        let doom = if moved {
            let reads: Vec<_> = substrates.iter().filter_map(|c| c.try_read()).collect();
            let judge_absent = reads.len() == substrates.len();
            let status_of = |sid: StreamId| {
                reads
                    .iter()
                    .find_map(|read| turn_in(read, sid))
                    .map_or(TurnStatus::Absent, |(status, _)| status)
            };
            let doom = doomed(&runs, &loaded, Some(&status_of), judge_absent);
            // Seen only what every substrate was read for: one skipped is
            // looked at again on the next sweep.
            if judge_absent {
                self.gallery_sweep.tombstones_seen = epochs;
            }
            doom
        } else {
            doomed(&runs, &loaded, None, false)
        };
        let sids: Vec<StreamId> = doom.iter().map(|(sid, _)| *sid).collect();
        let (turns, bytes) = arena.drop_turns(&sids);
        // A pinned turn stays for this sweep and is owed to the next.
        self.gallery_sweep.owed = turns < sids.len();
        if turns > 0 {
            let count = |why: Doom| doom.iter().filter(|(_, d)| *d == why).count();
            tracing::info!(
                target: "candle_conversation::provenance",
                turns,
                freed_mib = bytes >> 20,
                unloaded = count(Doom::Unloaded),
                tombstoned = count(Doom::Tombstoned),
                absent = count(Doom::Absent),
                kept = runs.len() - turns,
                "gallery gave up the turns no scan can read"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn run(sid: u64, holder: Holder) -> ResidentRun {
        ResidentRun {
            sid: StreamId(sid),
            bytes: 6144,
            idle: Duration::ZERO,
            holder,
        }
    }

    /// **A conversation's own turns go once it is not loaded; a shared turn
    /// stays whoever is loaded.** Without the substrates read, nothing else is
    /// judged.
    #[test]
    fn an_unloaded_conversations_turns_go_and_shared_ones_stay() {
        let runs = [
            run(1, Holder::Conversation(tl(5))),
            run(2, Holder::Conversation(tl(6))),
            run(3, Holder::Shared),
        ];
        let loaded: HashSet<TimelineId> = [tl(5)].into();
        assert_eq!(
            doomed(&runs, &loaded, None, true),
            vec![(StreamId(2), Doom::Unloaded)]
        );
    }

    /// **A tombstoned turn goes whoever holds it; an absent one only when
    /// every substrate was read.** A loaded conversation's tombstoned turn goes
    /// too: nothing scans it.
    #[test]
    fn tombstoned_turns_go_and_absent_ones_only_when_every_substrate_was_read() {
        let runs = [
            run(1, Holder::Shared),
            run(2, Holder::Conversation(tl(5))),
            run(3, Holder::Shared),
            run(4, Holder::Shared),
        ];
        let loaded: HashSet<TimelineId> = [tl(5)].into();
        let status = |sid: StreamId| match sid.0 {
            1 | 2 => TurnStatus::Tombstoned,
            3 => TurnStatus::Absent,
            _ => TurnStatus::Live,
        };
        assert_eq!(
            doomed(&runs, &loaded, Some(&status), true),
            vec![
                (StreamId(1), Doom::Tombstoned),
                (StreamId(2), Doom::Tombstoned),
                (StreamId(3), Doom::Absent),
            ]
        );
        assert_eq!(
            doomed(&runs, &loaded, Some(&status), false),
            vec![
                (StreamId(1), Doom::Tombstoned),
                (StreamId(2), Doom::Tombstoned)
            ],
            "a substrate skipped: an undeclared turn may be in it"
        );
    }
}
