//! What one pass does to one layer, decided from keys alone
//! (`docs/zend_branch_ingest.md` §7.1–§7.2): which units to ingest, and which
//! conversations to tombstone.

use std::collections::{HashMap, HashSet};

use candle_conversation::persistence::identity::VersionStamp;
use candle_conversation::projection::TimelineId;

use super::keys::{key_version, KEY_VERSION};

/// A unit on some branch: its key, and the path or folder it is of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Live<'a> {
    pub key: &'a str,
    pub subject: &'a str,
}

/// A committed conversation of the layer: its key, the path or folder it is
/// of, and the stamp of its first turn's declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Committed {
    pub timeline: TimelineId,
    pub key: String,
    pub subject: String,
    pub stamp: VersionStamp,
    /// The bits per weight of the model that wrote it ([`super::keys::QUANT_KEY`]),
    /// `None` for a unit written before that was recorded.
    pub quant: Option<u32>,
}

/// One pass over one layer.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Indices into the units to ingest of those no conversation holds, in
    /// the order given.
    pub queued: Vec<usize>,
    /// Conversations to tombstone now: of a key not among the units to
    /// retain, whose path or folder has nothing queued to replace it — and
    /// every conversation of a key but the one that holds it.
    pub tombstone: Vec<TimelineId>,
}

/// Plan a pass: `ingest` is what the layer reads — the corpus within the
/// depth bound — `retain` every key it keeps, the corpus at full depth, and
/// `committed` what the layer holds.
///
/// What is queued comes from `ingest`; what is tombstoned is judged against
/// `retain`, so a unit past the depth bound is kept for as long as a branch
/// holds it. A dead key whose path or folder has a queued key stays until
/// that replacement commits — its ingest tombstones it then — so a path is
/// never missing from the layer while it is being read again.
///
/// **A key is held by its conversation with the lowest stamp, unless a better
/// model wrote one** (`docs/replicated_substrate.md` §10.4, [`holder_of`]).
/// Two members can ingest one unit before either sees the other's; both
/// conversations arrive everywhere, and every member keeps the same one and
/// tombstones the other, whatever order they arrived in.
///
/// **A conversation under a newer [`KEY_VERSION`] is not this member's to
/// judge.** A peer upgraded first writes units this build cannot key, so none
/// of them is ever in `retain`, and planned as this member's they would all be
/// tombstoned — while the peer, planning the same layer, tombstones this
/// member's older ones, and each re-ingests what the other just removed. Left
/// alone, the newer units stand until this member upgrades and plans them as
/// its own.
pub fn plan(ingest: &[Live<'_>], retain: &HashSet<&str>, committed: &[Committed]) -> Plan {
    let committed: Vec<&Committed> = committed
        .iter()
        .filter(|c| key_version(&c.key) <= KEY_VERSION)
        .collect();
    // Decided per key over all its conversations, never pair by pair: "better
    // quantization, else the lower stamp" between pairs is no order once a
    // conversation with no quantization is among them (a Q4 beats a Q3, the
    // Q3 an unrecorded one by stamp, that one the Q4 by stamp), and members
    // seeing them in different orders would keep different holders.
    let mut by_key: HashMap<&str, Vec<&Committed>> = HashMap::new();
    for &c in &committed {
        by_key.entry(c.key.as_str()).or_default().push(c);
    }
    let holder: HashMap<&str, &Committed> = by_key
        .into_iter()
        .filter_map(|(key, all)| holder_of(&all).map(|h| (key, h)))
        .collect();
    let mut tombstone: Vec<TimelineId> = committed
        .iter()
        .filter(|c| holder.get(c.key.as_str()).map(|h| h.timeline) != Some(c.timeline))
        .map(|c| c.timeline)
        .collect();
    let holder: HashMap<&str, TimelineId> =
        holder.into_iter().map(|(k, c)| (k, c.timeline)).collect();
    let queued: Vec<usize> = ingest
        .iter()
        .enumerate()
        .filter(|(_, l)| !holder.contains_key(l.key))
        .map(|(i, _)| i)
        .collect();
    let replacing: HashSet<&str> = queued.iter().map(|&i| ingest[i].subject).collect();
    for c in committed {
        let held_here = holder.get(c.key.as_str()) == Some(&c.timeline);
        if held_here && !retain.contains(c.key.as_str()) && !replacing.contains(c.subject.as_str())
        {
            tombstone.push(c.timeline);
        }
    }
    Plan { queued, tombstone }
}

/// The conversation that holds a key, of every one committed under it.
///
/// **The one there first, unless a better model wrote one.** When every
/// conversation records its model's quantization ([`Committed::quant`]), the
/// highest holds — a member on a Q4 pack replaces what a Q3 member read, and
/// the Q3 copy is tombstoned whichever arrived first — the lowest stamp
/// breaking a tie. When any does not (one written before the quantization was
/// recorded), the lowest stamp holds, as it always has: the existing record
/// is kept, and a duplicate arriving from a member that joined is the one let
/// go.
fn holder_of<'a>(all: &[&'a Committed]) -> Option<&'a Committed> {
    let earliest = |c: &&&Committed| (c.stamp, c.timeline.raw());
    match all.iter().map(|c| c.quant).collect::<Option<Vec<u32>>>() {
        Some(_) => all
            .iter()
            .min_by_key(|c| (std::cmp::Reverse(c.quant), c.stamp, c.timeline.raw()))
            .copied(),
        None => all.iter().min_by_key(earliest).copied(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn committed(n: u64, key: &str, subject: &str) -> Committed {
        stamped(n, key, subject, n)
    }

    /// A conversation whose first turn was declared at `hlc`.
    fn stamped(n: u64, key: &str, subject: &str, hlc: u64) -> Committed {
        Committed {
            timeline: tl(n),
            key: key.to_string(),
            subject: subject.to_string(),
            stamp: VersionStamp {
                hlc,
                ..VersionStamp::default()
            },
            quant: None,
        }
    }

    fn live<'a>(key: &'a str, subject: &'a str) -> Live<'a> {
        Live { key, subject }
    }

    /// A pass with no depth bound: everything ingested is everything retained.
    fn unbounded(ingest: &[Live<'_>], committed: &[Committed]) -> Plan {
        let retain: HashSet<&str> = ingest.iter().map(|l| l.key).collect();
        plan(ingest, &retain, committed)
    }

    /// **What no conversation holds is queued; what a conversation holds is
    /// left alone** — however many branches carry it.
    #[test]
    fn what_is_not_held_is_queued() {
        let p = unbounded(
            &[live("a@1", "a"), live("b@1", "b"), live("a@2", "a")],
            &[committed(1, "a@1", "a")],
        );
        assert_eq!(p.queued, [1, 2]);
        assert!(p.tombstone.is_empty(), "a@1 is still on a branch");
    }

    /// **A path gone from every branch goes at once.**
    #[test]
    fn a_path_on_no_branch_is_tombstoned() {
        let p = unbounded(
            &[live("a@1", "a")],
            &[committed(1, "a@1", "a"), committed(2, "gone@1", "gone")],
        );
        assert!(p.queued.is_empty());
        assert_eq!(p.tombstone, [tl(2)]);
    }

    /// **A version no branch holds stays while its path's new version is
    /// queued** — its ingest retires it — and goes once nothing replaces it.
    #[test]
    fn a_replaced_version_waits_for_its_replacement() {
        let old = [committed(1, "a@1", "a")];
        let waiting = unbounded(&[live("a@2", "a")], &old);
        assert_eq!(waiting.queued, [0]);
        assert!(waiting.tombstone.is_empty());

        let replaced = unbounded(
            &[live("a@2", "a")],
            &[committed(1, "a@1", "a"), committed(2, "a@2", "a")],
        );
        assert!(replaced.queued.is_empty());
        assert_eq!(replaced.tombstone, [tl(1)]);
    }

    /// **Two conversations of one key are one too many**: the one whose first
    /// turn has the lower stamp stays, whatever order they are listed in —
    /// so every member keeps the same one (§10.4).
    #[test]
    fn a_duplicate_of_a_held_key_is_tombstoned_by_its_stamp() {
        let earlier = stamped(2, "a@1", "a", 10);
        let later = stamped(1, "a@1", "a", 20);
        for committed in [[earlier.clone(), later.clone()], [later, earlier]] {
            let p = unbounded(&[live("a@1", "a")], &committed);
            assert!(p.queued.is_empty());
            assert_eq!(p.tombstone, [tl(1)], "the later ingest goes");
        }
        // The same stamp — two writes of one origin never tie — falls to the
        // timeline.
        let p = unbounded(
            &[live("b@1", "b")],
            &[stamped(4, "b@1", "b", 5), stamped(3, "b@1", "b", 5)],
        );
        assert_eq!(p.tombstone, [tl(4)]);
    }

    /// A conversation stamped at `hlc`, written under a `bits`-per-weight model.
    fn quantized(n: u64, key: &str, hlc: u64, bits: Option<u32>) -> Committed {
        Committed {
            quant: bits,
            ..stamped(n, key, key, hlc)
        }
    }

    /// **The better quantization holds a key**, the earlier one tombstoned
    /// whichever arrived first; at one quantization the lower stamp holds.
    #[test]
    fn a_better_quantized_duplicate_replaces_the_existing_one() {
        let q3 = quantized(1, "a@1", 10, Some(3));
        let q4 = quantized(2, "a@1", 20, Some(4));
        for committed in [[q3.clone(), q4.clone()], [q4, q3]] {
            let p = unbounded(&[live("a@1", "a@1")], &committed);
            assert_eq!(
                p.tombstone,
                [tl(1)],
                "the Q3 copy goes, though it was first"
            );
        }
        let p = unbounded(
            &[live("b@1", "b@1")],
            &[
                quantized(3, "b@1", 10, Some(4)),
                quantized(4, "b@1", 20, Some(4)),
            ],
        );
        assert_eq!(
            p.tombstone,
            [tl(4)],
            "the same quantization: the later goes"
        );
    }

    /// **A key any copy of which records no quantization is judged by stamp
    /// alone**: the existing record is kept. And the rule is one per key, not
    /// per pair — a Q4, a Q3 and an unrecorded copy pick one holder in every
    /// order they are seen.
    #[test]
    fn an_unrecorded_quantization_keeps_the_existing_record() {
        let old = quantized(1, "a@1", 10, None);
        let new = quantized(2, "a@1", 20, Some(8));
        let p = unbounded(&[live("a@1", "a@1")], &[new.clone(), old.clone()]);
        assert_eq!(
            p.tombstone,
            [tl(2)],
            "the existing record, written before the tag, stays"
        );

        let q4 = quantized(3, "c@1", 30, Some(4));
        let q3 = quantized(4, "c@1", 10, Some(3));
        let none = quantized(5, "c@1", 20, None);
        let orders = [
            [q4.clone(), q3.clone(), none.clone()],
            [q3.clone(), none.clone(), q4.clone()],
            [none.clone(), q4.clone(), q3.clone()],
        ];
        for committed in orders {
            let p = unbounded(&[live("c@1", "c@1")], &committed);
            let mut gone = p.tombstone.clone();
            gone.sort_by_key(|t| t.raw());
            assert_eq!(gone, [tl(3), tl(5)], "the lowest stamp holds: q3 at 10");
        }
    }
    /// **A unit past the depth bound is retained, not ingested**: held, it
    /// stays for as long as a branch holds it; not held, nothing reads it.
    #[test]
    fn a_unit_past_the_bound_is_retained_and_not_queued() {
        let retain: HashSet<&str> = ["top@1", "deep@1", "unread@1"].into();
        let p = plan(
            &[live("top@1", "top")],
            &retain,
            &[committed(1, "top@1", "top"), committed(2, "deep@1", "deep")],
        );
        assert!(p.queued.is_empty(), "unread@1 is past the bound");
        assert!(
            p.tombstone.is_empty(),
            "deep@1 is on a branch at full depth"
        );
    }

    /// **Past the bound, only what no branch holds goes**: a deep file gone
    /// from every branch, and a deep file's old version — its new one is past
    /// the bound too, so nothing is queued to replace it.
    #[test]
    fn past_the_bound_what_no_branch_holds_goes() {
        let retain: HashSet<&str> = ["top@1", "deep@2"].into();
        let p = plan(
            &[live("top@1", "top")],
            &retain,
            &[
                committed(1, "top@1", "top"),
                committed(2, "gone@1", "gone"),
                committed(3, "deep@1", "deep"),
            ],
        );
        assert!(p.queued.is_empty());
        assert_eq!(p.tombstone, [tl(2), tl(3)]);
    }

    /// **A key version tombstones every unit read under the old one** — each
    /// once its replacement has committed, so no path is missing from the
    /// layer while it is read again, and one past the depth bound at once,
    /// since nothing re-reads it there. Old keys are the unversioned
    /// `path@blob` form the layer held before `KEY_VERSION`.
    #[test]
    fn a_key_version_tombstones_every_unit_read_under_the_old_one() {
        use super::super::keys::file_key;
        use zend_vfs::Oid;
        let blob = Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap();
        let old_a = format!("r/a.rs@{blob}");
        let old_b = format!("r/b.rs@{blob}");
        let old_deep = format!("r/deep.rs@{blob}");
        let new_a = file_key("r/a.rs", &blob);
        let new_b = file_key("r/b.rs", &blob);
        let new_deep = file_key("r/deep.rs", &blob);
        assert_ne!(old_a, new_a, "the version is in the key");

        let ingest = [live(&new_a, "r/a.rs"), live(&new_b, "r/b.rs")];
        let retain: HashSet<&str> = [new_a.as_str(), new_b.as_str(), new_deep.as_str()].into();
        let held = [
            committed(1, &old_a, "r/a.rs"),
            committed(2, &old_b, "r/b.rs"),
            committed(3, &old_deep, "r/deep.rs"),
        ];

        // The first pass after the change: every unit within the bound is
        // read again, its old conversation kept until the new one commits.
        let first = plan(&ingest, &retain, &held);
        assert_eq!(first.queued, [0, 1]);
        assert_eq!(
            first.tombstone,
            [tl(3)],
            "past the bound, nothing replaces it"
        );

        // Once both replacements have committed, both old conversations go.
        let mut after = held[..2].to_vec();
        after.push(committed(4, &new_a, "r/a.rs"));
        after.push(committed(5, &new_b, "r/b.rs"));
        let second = plan(&ingest, &retain, &after);
        assert!(second.queued.is_empty());
        assert_eq!(second.tombstone, [tl(1), tl(2)]);
    }

    /// **A unit under a newer key version is a peer's, and left alone** — not a
    /// holder, never tombstoned — while one under an older version beside it is
    /// replaced as usual. Two members on different versions would otherwise each
    /// tombstone the other's units and re-ingest them, forever.
    #[test]
    fn a_unit_under_a_newer_key_version_is_left_alone() {
        let newer = format!("v{}:r/a.rs@1", KEY_VERSION + 1);
        let ours = format!("v{KEY_VERSION}:r/a.rs@1");
        let p = unbounded(
            &[live(&ours, "r/a.rs")],
            &[
                committed(1, &newer, "r/a.rs"),
                committed(2, "r/a.rs@1", "r/a.rs"),
            ],
        );
        // Ours is queued — the newer conversation does not hold its key — and
        // the older one waits for it; the newer one is never tombstoned.
        assert_eq!(p.queued, [0]);
        assert!(p.tombstone.is_empty(), "{:?}", p.tombstone);

        let after = unbounded(
            &[live(&ours, "r/a.rs")],
            &[
                committed(1, &newer, "r/a.rs"),
                committed(2, "r/a.rs@1", "r/a.rs"),
                committed(3, &ours, "r/a.rs"),
            ],
        );
        assert!(after.queued.is_empty());
        assert_eq!(after.tombstone, [tl(2)], "only the older version goes");
    }

    #[test]
    fn nothing_live_and_nothing_held_plans_nothing() {
        assert_eq!(unbounded(&[], &[]), Plan::default());
    }
}
