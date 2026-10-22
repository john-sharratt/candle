//! What one pass does to one layer that builds on another, decided from keys
//! alone: which of its units still stand, which are stale and go now, and
//! which are read.
//!
//! **A unit stands while what it was built on does.** Each records the key of
//! the unit below it ([`super::keys::AFTER_KEY`]) and the version of every file
//! its own depends on ([`super::keys::DEPS_KEY`]). It is stale the moment the
//! unit below is no longer current — the file moved on, or the unit below was
//! itself found stale — or a file it depends on moved on along a branch it is
//! on. A stale unit is tombstoned at once, not kept until its replacement
//! commits: the layer briefly has no unit for the file, rather than one that
//! describes another version of it. (The layers that read the branches keep a
//! stale version until the new one commits, because a version a branch held is
//! still true of that branch; an analysis of a version is not.)
//!
//! **A unit is read only once what it builds on stands**, so the layers below
//! are always done first: a file whose `code_reading` is being read again has
//! no current unit below it in any higher layer until that read commits.

use std::collections::HashMap;

use candle_conversation::persistence::identity::VersionStamp;
use candle_conversation::projection::TimelineId;

/// A unit of the layer below that stands this pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Below {
    /// Its content key — what a unit built on it records as
    /// [`super::keys::AFTER_KEY`].
    pub key: String,
    /// What it is of: a file's workspace-relative path, or a day.
    pub subject: String,
    /// Its conversation — what a unit built on it continues. `None` for a day
    /// read from the dialogue itself, which no conversation holds.
    pub timeline: Option<TimelineId>,
    /// The `code_reading` key of its file; `None` for a day.
    pub file_key: Option<String>,
    /// Every branch whose tip holds its file at this version; empty for a day.
    pub branches: Vec<String>,
    /// The commit its file was read at, when it was.
    pub at: Option<String>,
}

/// A committed unit of the layer being planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Done {
    pub timeline: TimelineId,
    /// Its content key — what a unit built on it records as
    /// [`super::keys::AFTER_KEY`].
    pub key: String,
    pub subject: String,
    /// The key of the unit it was built on.
    pub after: String,
    /// Each file it depends on, at the version it was built against.
    pub deps: Vec<(String, String)>,
    pub stamp: VersionStamp,
    /// The spec it was built under ([`super::keys::SPEC_KEY`]); `None` for one
    /// built before that was recorded.
    pub spec: Option<String>,
    /// The model's bits per weight, when recorded.
    pub quant: Option<u32>,
}

/// Every version of a file some branch holds: path → `(key, branches)`.
pub type LiveFiles = HashMap<String, Vec<(String, Vec<String>)>>;

/// A unit that stands: the unit below it, its own conversation, and the
/// branches it stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Held {
    /// Index into the units below.
    pub below: usize,
    pub timeline: TimelineId,
    /// Its content key.
    pub key: String,
    /// The branches of the unit below it on which every file it depends on is
    /// the version it was built against — every branch of the unit below, for
    /// a unit that depends on nothing. Empty for a unit of no branch.
    pub branches: Vec<String>,
}

/// A unit to build: the unit below it, and the branches it is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Queued {
    /// Index into the units below.
    pub below: usize,
    /// The branches of the unit below no standing unit covers.
    pub branches: Vec<String>,
}

/// One pass over one layer.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// The units that stand, each for its own branches.
    pub held: Vec<Held>,
    /// Conversations to tombstone now: stale ones, and every one standing for
    /// no branch another does not already stand for.
    pub tombstone: Vec<TimelineId>,
    /// What to build, in the order of the units below.
    pub queued: Vec<Queued>,
}

/// Plan the layer: `below` is what stands one layer down, `committed` what
/// this layer holds, `live` every file version a branch holds, and `ingest`
/// whether a unit below is in the scope this pass reads (within the depth
/// bound) — a unit outside it is kept while it stands, never read.
///
/// **A unit stands for the branches its dependencies are current on.** One
/// version of a file is on several branches, and the files it depends on need
/// not be: a change to a dependency on one branch leaves the unit right for
/// the others and wrong for that one. So each unit is judged branch by
/// branch, the one there first taking every branch it is current on, the next
/// the branches left, and a branch no unit is current on is built a unit of
/// its own — the analyses split where, and only where, what they depend on
/// differs. A unit that depends on nothing stands for every branch.
pub fn plan(
    below: &[Below],
    committed: &[Done],
    live: &LiveFiles,
    ingest: &dyn Fn(&Below) -> bool,
    spec: &str,
) -> Plan {
    let by_key: HashMap<&str, usize> = below
        .iter()
        .enumerate()
        .map(|(i, b)| (b.key.as_str(), i))
        .collect();
    let mut tombstone = Vec::new();
    let mut on: HashMap<usize, Vec<&Done>> = HashMap::new();
    for done in committed {
        // Built under another spec — another question, other tools, other
        // thinking — the unit answers something the layer no longer asks.
        match by_key.get(done.after.as_str()) {
            Some(&i) if done.spec.as_deref() == Some(spec) => on.entry(i).or_default().push(done),
            _ => tombstone.push(done.timeline),
        }
    }
    let mut held = Vec::new();
    let mut queued = Vec::new();
    for (i, b) in below.iter().enumerate() {
        let mut candidates = on.remove(&i).unwrap_or_default();
        let all_quant = candidates_quants(&candidates);
        candidates.sort_by_key(|d| holder_rank(d, all_quant));
        if b.branches.is_empty() {
            // Of no branch: a dependency is judged at the version any branch
            // holds, and the first unit current at it stands.
            let mut holder = None;
            for d in candidates {
                if holder.is_none() && deps_current(d, &[], live) {
                    holder = Some(d);
                } else {
                    tombstone.push(d.timeline);
                }
            }
            match holder {
                Some(d) => held.push(Held {
                    below: i,
                    timeline: d.timeline,
                    key: d.key.clone(),
                    branches: Vec::new(),
                }),
                None if ingest(b) => queued.push(Queued {
                    below: i,
                    branches: Vec::new(),
                }),
                None => {}
            }
            continue;
        }
        let mut left: Vec<String> = b.branches.clone();
        for d in candidates {
            let ok: Vec<String> = left
                .iter()
                .filter(|br| deps_current(d, std::slice::from_ref(*br), live))
                .cloned()
                .collect();
            if ok.is_empty() {
                tombstone.push(d.timeline);
                continue;
            }
            left.retain(|br| !ok.contains(br));
            held.push(Held {
                below: i,
                timeline: d.timeline,
                key: d.key.clone(),
                branches: ok,
            });
        }
        if !left.is_empty() && ingest(b) {
            queued.push(Queued {
                below: i,
                branches: left,
            });
        }
    }
    tombstone.sort_by_key(|t| t.raw());
    Plan {
        held,
        tombstone,
        queued,
    }
}

/// Whether every file `done` depends on is, on each of `branches`, the
/// version it was built against — on any branch when `branches` is empty. A
/// dependency a branch no longer holds has moved on there too.
///
/// **Judged by the same rule that chose it** ([`dep_version`]): a dependency
/// is current when the version the unit would take now is the one it took. A
/// separate test once required the version to be on one of the unit's own
/// branches, while the choice falls back to any version when none of them
/// holds the file — a unit depending on such a file was stale the moment it
/// committed, and read again every pass.
fn deps_current(done: &Done, branches: &[String], live: &LiveFiles) -> bool {
    done.deps
        .iter()
        .all(|(path, key)| dep_version(live, path, branches).as_deref() == Some(key))
}

/// Whether every unit records its model's quantization — the holder rule's
/// switch ([`holder_rank`]).
fn candidates_quants(all: &[&Done]) -> bool {
    all.iter().all(|d| d.quant.is_some())
}

/// The order units built on one unit below hold in: the one there first,
/// unless a better model wrote one — the rule `code_reading` keeps
/// (`branch_ingest::plan`), so every member keeps the same one.
fn holder_rank(d: &Done, all_quant: bool) -> (std::cmp::Reverse<u32>, VersionStamp, u64) {
    let quant = if all_quant { d.quant.unwrap_or(0) } else { 0 };
    (std::cmp::Reverse(quant), d.stamp, d.timeline.raw())
}

/// The version of `path` a unit on `branches` depends on: the one a branch it
/// is on holds, else any — `None` when no branch holds the path.
pub fn dep_version(live: &LiveFiles, path: &str, branches: &[String]) -> Option<String> {
    let versions = live.get(path)?;
    versions
        .iter()
        .find(|(_, on)| on.iter().any(|b| branches.contains(b)))
        .or_else(|| versions.first())
        .map(|(k, _)| k.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn below(key: &str, subject: &str, n: u64) -> Below {
        Below {
            key: key.into(),
            subject: subject.into(),
            timeline: Some(tl(n)),
            file_key: Some(key.into()),
            branches: vec!["main".into()],
            at: None,
        }
    }

    fn done(n: u64, after: &str, deps: &[(&str, &str)]) -> Done {
        Done {
            timeline: tl(n),
            key: format!("u{n}"),
            subject: "s".into(),
            after: after.into(),
            deps: deps
                .iter()
                .map(|(p, k)| (p.to_string(), k.to_string()))
                .collect(),
            stamp: VersionStamp {
                gen: 0,
                epoch: 0,
                hlc: n,
                origin: 0,
            },
            spec: Some("s".into()),
            quant: None,
        }
    }

    fn live(files: &[(&str, &str, &[&str])]) -> LiveFiles {
        let mut out = LiveFiles::new();
        for (path, key, branches) in files {
            out.entry(path.to_string()).or_default().push((
                key.to_string(),
                branches.iter().map(|b| b.to_string()).collect(),
            ));
        }
        out
    }

    const ALL: &dyn Fn(&Below) -> bool = &|_| true;

    fn queued(p: &Plan) -> Vec<usize> {
        p.queued.iter().map(|q| q.below).collect()
    }

    /// A unit built on what stands, with its dependencies where they were,
    /// stands; nothing is read again.
    #[test]
    fn a_unit_on_what_stands_is_held() {
        let b = [below("a@1", "a", 1)];
        let p = plan(
            &b,
            &[done(10, "a@1", &[("d", "d@1")])],
            &live(&[("d", "d@1", &["main"])]),
            ALL,
            "s",
        );
        assert_eq!(
            p.held,
            [Held {
                below: 0,
                timeline: tl(10),
                key: "u10".into(),
                branches: vec!["main".into()],
            }]
        );
        assert!(p.tombstone.is_empty() && p.queued.is_empty());
    }

    /// **The file moved on: its unit goes now, not once the new one commits.**
    /// The unit below for the new version stands, so the new unit is read.
    #[test]
    fn a_unit_on_a_version_that_moved_on_goes_at_once() {
        let b = [below("a@2", "a", 2)];
        let p = plan(&b, &[done(10, "a@1", &[])], &LiveFiles::new(), ALL, "s");
        assert!(p.held.is_empty());
        assert_eq!(p.tombstone, [tl(10)]);
        assert_eq!(queued(&p), [0]);
    }

    /// **A dependency moved on: its dependent is stale too.** The file itself
    /// did not change; the version it was analysed against did.
    #[test]
    fn a_unit_whose_dependency_moved_on_is_stale() {
        let b = [below("a@1", "a", 1)];
        let p = plan(
            &b,
            &[done(10, "a@1", &[("d", "d@1")])],
            &live(&[("d", "d@2", &["main"])]),
            ALL,
            "s",
        );
        assert_eq!(p.tombstone, [tl(10)]);
        assert_eq!(queued(&p), [0]);
    }

    /// A dependency that moved on along ANOTHER branch leaves the unit
    /// standing: the version it was built against is still on its own.
    #[test]
    fn a_dependency_moving_on_another_branch_leaves_the_unit() {
        let b = [below("a@1", "a", 1)];
        let p = plan(
            &b,
            &[done(10, "a@1", &[("d", "d@1")])],
            &live(&[("d", "d@1", &["main"]), ("d", "d@2", &["topic"])]),
            ALL,
            "s",
        );
        assert_eq!(p.held.len(), 1);
        assert!(p.tombstone.is_empty());
    }

    /// **Nothing below, nothing here.** A unit whose file's reading is being
    /// redone has no unit below it this pass, so it goes and none is read.
    #[test]
    fn with_nothing_below_the_unit_goes_and_none_is_read() {
        let p = plan(&[], &[done(10, "a@1", &[])], &LiveFiles::new(), ALL, "s");
        assert_eq!(p.tombstone, [tl(10)]);
        assert!(p.queued.is_empty());
    }

    /// Two units on one unit below: the one there first holds, the other goes.
    #[test]
    fn of_two_units_on_one_the_first_holds() {
        let b = [below("a@1", "a", 1)];
        let p = plan(
            &b,
            &[done(12, "a@1", &[]), done(11, "a@1", &[])],
            &LiveFiles::new(),
            ALL,
            "s",
        );
        assert_eq!(p.held[0].timeline, tl(11));
        assert_eq!(p.tombstone, [tl(12)]);
    }

    /// **A dependency no branch of the unit holds stands as chosen.** The
    /// version taken for it was the one the choice falls back to; judged by a
    /// stricter test it read as stale the moment it committed.
    #[test]
    fn a_dependency_off_the_units_branches_stands_as_chosen() {
        let b = [below("a@1", "a", 1)];
        let l = live(&[("d", "d@9", &["topic"])]);
        let chosen = dep_version(&l, "d", &b[0].branches).unwrap();
        let p = plan(&b, &[done(10, "a@1", &[("d", &chosen)])], &l, ALL, "s");
        assert_eq!(p.held.len(), 1);
        assert!(p.tombstone.is_empty() && p.queued.is_empty());
    }

    /// **A dependency that moved on one branch splits the unit there.** The
    /// file is the same version on `main` and `topic`; its dependency changed
    /// only on `topic`. The unit stands for `main`, and `topic` is built a unit
    /// of its own.
    #[test]
    fn a_dependency_moving_on_one_branch_splits_the_unit_there() {
        let mut b = below("a@1", "a", 1);
        b.branches = vec!["main".into(), "topic".into()];
        let l = live(&[("d", "d@1", &["main"]), ("d", "d@2", &["topic"])]);
        let p = plan(&[b], &[done(10, "a@1", &[("d", "d@1")])], &l, ALL, "s");
        assert_eq!(p.held.len(), 1);
        assert_eq!(p.held[0].branches, ["main".to_string()]);
        assert!(p.tombstone.is_empty());
        assert_eq!(
            p.queued,
            [Queued {
                below: 0,
                branches: vec!["topic".into()]
            }]
        );
        // Once `topic`'s own unit commits, both stand and nothing is read.
        let both = [
            done(10, "a@1", &[("d", "d@1")]),
            done(11, "a@1", &[("d", "d@2")]),
        ];
        let mut b2 = below("a@1", "a", 1);
        b2.branches = vec!["main".into(), "topic".into()];
        let p = plan(&[b2], &both, &l, ALL, "s");
        assert_eq!(p.held.len(), 2);
        assert!(p.queued.is_empty() && p.tombstone.is_empty());
    }

    /// **A layer whose question changes is asked again.** A unit built under
    /// another spec — another prompt, other tools, other thinking — goes, and
    /// the unit below it is read again; one with no spec recorded goes too.
    #[test]
    fn a_unit_built_under_another_spec_goes() {
        let b = [below("a@1", "a", 1)];
        let mut old = done(10, "a@1", &[]);
        old.spec = Some("before".into());
        let mut none = done(11, "a@1", &[]);
        none.spec = None;
        let p = plan(&b, &[old, none], &LiveFiles::new(), ALL, "s");
        assert!(p.held.is_empty());
        assert_eq!(p.tombstone, [tl(10), tl(11)]);
        assert_eq!(queued(&p), [0]);
    }

    /// A unit current on no branch left to it goes: one there first already
    /// stands for every branch it could.
    #[test]
    fn a_second_unit_for_the_same_branches_goes() {
        let b = [below("a@1", "a", 1)];
        let p = plan(
            &b,
            &[done(11, "a@1", &[]), done(12, "a@1", &[])],
            &LiveFiles::new(),
            ALL,
            "s",
        );
        assert_eq!(p.held.len(), 1);
        assert_eq!(p.held[0].timeline, tl(11));
        assert_eq!(p.tombstone, [tl(12)]);
    }

    /// A unit below past the depth bound is held while it stands, never read.
    #[test]
    fn past_the_depth_bound_a_unit_is_kept_but_never_read() {
        let b = [below("a@1", "a", 1), below("b@1", "b", 2)];
        let in_scope: &dyn Fn(&Below) -> bool = &|b| b.subject == "a";
        let p = plan(
            &b,
            &[done(10, "b@1", &[])],
            &LiveFiles::new(),
            in_scope,
            "s",
        );
        assert_eq!(p.held[0].below, 1);
        assert_eq!(queued(&p), [0]);
    }

    #[test]
    fn a_dependency_is_taken_at_the_version_its_branch_holds() {
        let l = live(&[("d", "d@1", &["main"]), ("d", "d@2", &["topic"])]);
        assert_eq!(
            dep_version(&l, "d", &["topic".into()]).as_deref(),
            Some("d@2")
        );
        assert_eq!(
            dep_version(&l, "d", &["other".into()]).as_deref(),
            Some("d@1")
        );
        assert_eq!(dep_version(&l, "missing", &[]), None);
    }
}
