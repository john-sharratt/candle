//! A belief turn group's assembled scan, kept between scans.
//!
//! Before a group is scored, every conversation it offers the target is
//! gathered, each file's gallery looked up, its chunk exclusion cut and, under
//! `kind_z`, its kind read from its path ([`super::resolver`],
//! `score_belief_groups`). None of that depends on the probe. On BlackWorld
//! (2026-11-18) the code-reading groups offered 7,599 files, and gathering them
//! took about 70 ms of every reprojection (scope 8, assembly 36, exclusion 16,
//! kinds 6), though nothing they read had changed since the reprojection
//! before.
//!
//! So the assembly is kept per group and whose it is — one target's, when the
//! scan is masked to it or leaves its lineage out, and otherwise every
//! target's whose scope of the group is the same ([`Whose`]), so the
//! conversations reprojecting together share it and their probes ride one
//! launch — stamped with what it was built from:
//!
//! - [`Substrate::group_epoch`] of the group — a stream declared, a signature
//!   or projection-event blob written, a turn or timeline tombstoned in a
//!   conversation the group reads, or in a calibration of one of its files,
//!   which is every change to its files' turns, windows and calibrations'
//!   signatures — and, for a target's own assembly, of the target's group;
//! - [`Substrate::retrieval_version`] — a group's scoping, a target's scope, a
//!   chunk exclusion, an archived flag, a timeline's metadata (its path, the
//!   unit it calibrates);
//!
//! and the policy it was assembled under: the group's tags and whether it reads
//! kinds. A scan reuses it while all of them stand, and assembles afresh when
//! any moved.

use std::sync::Arc;

use super::belief_files::FileScan;
use super::ids::{GroupId, TimelineId};
use crate::substrate::Substrate;

/// Whose assembly: one target's alone, or every target's whose scope of the
/// group names the same conversations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Whose {
    /// A scan masked to the target's own timeline, or one that leaves the
    /// target's lineage out: no other target's assembly is the same.
    Target(TimelineId),
    /// Shared: the group's scope digest (`Substrate::scope_digest`), `None`
    /// for a group that is not scoped. Targets that share one share its
    /// files, and their probes ride one launch.
    Scope(Option<u64>),
}

/// Whose assembly of which group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AssemblyKey {
    pub group: GroupId,
    pub whose: Whose,
}

/// What an assembly was built from (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AssemblyStamp {
    /// The group's content epoch.
    pub group_epoch: u64,
    /// The target's group's content epoch, for an assembly that is the
    /// target's own ([`Whose::Target`]); zero for a shared one.
    pub target_epoch: u64,
    pub retrieval: u64,
}

impl AssemblyStamp {
    /// `key`'s assembly as `sub` stands now, for a target in `target_group`.
    pub fn of(sub: &Substrate, key: AssemblyKey, target_group: GroupId) -> Self {
        Self {
            group_epoch: sub.group_epoch(key.group),
            target_epoch: match key.whose {
                Whose::Target(_) => sub.group_epoch(target_group),
                Whose::Scope(_) => 0,
            },
            retrieval: sub.retrieval_version(),
        }
    }
}

/// One group's files as a scan reads them: gathered, excluded and, when the
/// group reads them by kind, each file's kind.
#[derive(Clone)]
pub(crate) struct GroupAssembly {
    pub stamp: AssemblyStamp,
    /// The group's tags it was gathered under.
    pub tags: Vec<String>,
    /// Each file's kind, when the group reads by kind.
    pub kinds: Option<Vec<String>>,
    pub files: Vec<FileScan>,
    /// Conversations offered before the files with no window were left out.
    pub n_timelines: usize,
    /// Turns walked to build galleries that were not held yet.
    pub turns_walked: usize,
}

impl GroupAssembly {
    /// Whether this assembly is what a scan would assemble now: built at
    /// `stamp`, under `tags`, reading kinds iff `by_kind`.
    pub fn serves(&self, stamp: AssemblyStamp, tags: &[String], by_kind: bool) -> bool {
        self.stamp == stamp && self.tags == tags && self.kinds.is_some() == by_kind
    }
}

/// The assemblies kept, by [`AssemblyKey`].
pub(crate) type Assemblies = papaya::HashMap<AssemblyKey, Arc<GroupAssembly>>;

/// The assembly `key` holds when it serves `stamp`, `tags` and `by_kind`.
pub(crate) fn held(
    assemblies: &Assemblies,
    key: AssemblyKey,
    stamp: AssemblyStamp,
    tags: &[String],
    by_kind: bool,
) -> Option<Arc<GroupAssembly>> {
    assemblies
        .pin()
        .get(&key)
        .filter(|a| a.serves(stamp, tags, by_kind))
        .cloned()
}

/// Keep `assembly` for `key`, replacing what it held. Assemblies of targets
/// that have gone are dropped once more than [`MAX_KEPT`] are held: each is a
/// few handles per file, and a busy member opens many conversations.
pub(crate) fn keep(assemblies: &Assemblies, key: AssemblyKey, assembly: Arc<GroupAssembly>) {
    let map = assemblies.pin();
    if map.len() >= MAX_KEPT && !map.contains_key(&key) {
        map.clear();
    }
    map.insert(key, assembly);
}

/// How many assemblies are held at most before the rest are let go.
pub(crate) const MAX_KEPT: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    fn assembly(stamp: AssemblyStamp, tags: &[&str], by_kind: bool) -> GroupAssembly {
        GroupAssembly {
            stamp,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            kinds: by_kind.then(Vec::new),
            files: Vec::new(),
            n_timelines: 3,
            turns_walked: 0,
        }
    }

    fn key(target: u64) -> AssemblyKey {
        AssemblyKey {
            group: GroupId::from_raw(7).unwrap(),
            whose: Whose::Target(TimelineId::from_raw(target).unwrap()),
        }
    }

    /// Targets with the same scope digest share one key, and so one
    /// assembly; another digest, or a target's own, is another.
    #[test]
    fn targets_with_one_scope_share_a_key() {
        let shared = |digest| AssemblyKey {
            group: GroupId::from_raw(7).unwrap(),
            whose: Whose::Scope(digest),
        };
        assert_eq!(shared(Some(9)), shared(Some(9)));
        assert_ne!(shared(Some(9)), shared(Some(10)));
        assert_ne!(shared(None), shared(Some(9)));
        assert_ne!(shared(Some(9)), key(9));
    }

    const AT: AssemblyStamp = AssemblyStamp {
        group_epoch: 10,
        target_epoch: 0,
        retrieval: 4,
    };

    /// **An assembly serves only the stamp, tags and kind reading it was built
    /// under**: a moved epoch, a moved retrieval version, other tags or a
    /// changed kind reading each send the scan to assemble afresh.
    #[test]
    fn an_assembly_serves_only_what_it_was_built_from() {
        let a = assembly(AT, &["code"], true);
        assert!(a.serves(AT, &["code".into()], true));
        let later_epoch = AssemblyStamp {
            group_epoch: 11,
            ..AT
        };
        assert!(!a.serves(later_epoch, &["code".into()], true));
        let later_target = AssemblyStamp {
            target_epoch: 1,
            ..AT
        };
        assert!(!a.serves(later_target, &["code".into()], true));
        let later_scope = AssemblyStamp { retrieval: 5, ..AT };
        assert!(!a.serves(later_scope, &["code".into()], true));
        assert!(!a.serves(AT, &["docs".into()], true));
        assert!(!a.serves(AT, &["code".into()], false));
    }

    /// A kept assembly is handed back while it serves and not after; a newer
    /// one replaces it under the same key.
    #[test]
    fn a_kept_assembly_is_reused_until_it_is_stale() {
        let map = Assemblies::new();
        keep(&map, key(1), Arc::new(assembly(AT, &[], false)));
        assert!(held(&map, key(1), AT, &[], false).is_some());
        assert!(
            held(&map, key(2), AT, &[], false).is_none(),
            "another target"
        );
        let moved = AssemblyStamp {
            group_epoch: 12,
            ..AT
        };
        assert!(held(&map, key(1), moved, &[], false).is_none());
        keep(&map, key(1), Arc::new(assembly(moved, &[], false)));
        assert!(held(&map, key(1), moved, &[], false).is_some());
        assert_eq!(map.pin().len(), 1);
    }

    /// The map stays bounded: past [`MAX_KEPT`] targets it lets the old go and
    /// keeps the newest.
    #[test]
    fn the_kept_assemblies_stay_bounded() {
        let map = Assemblies::new();
        for t in 1..=(MAX_KEPT as u64 + 1) {
            keep(&map, key(t), Arc::new(assembly(AT, &[], false)));
        }
        assert!(map.pin().len() <= MAX_KEPT);
        assert!(held(&map, key(MAX_KEPT as u64 + 1), AT, &[], false).is_some());
    }
}
