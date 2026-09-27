//! A conversation's file events read back into what its stores restore, and
//! the mirror of them (`docs/zend_vfs_events.md` §5).

use std::collections::BTreeMap;

use candle_conversation::persistence::vfs::VfsEventPayload;
use zend_vfs::{SavedChain, Snapshot};

use super::event::FileEvent;
use super::mirror::{absorb, RepoMirror};

/// One repository's events, replayed: what its store restores, and what the
/// substrate holds of it.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Replayed {
    pub snapshot: Snapshot,
    pub mirror: RepoMirror,
}

/// Replay `events` — one repository's, in `(key, seq)` order — by the rules
/// the mirror records its own writes by. Deltas a torn save left with no
/// state to commit them are dropped ([`RepoMirror::settle`]): each path is
/// its last committed chain, and a path no state ever committed is not in
/// the snapshot at all. `Err` says why the events do not make a store's
/// state: a body that is no event, or an event under a key it cannot be
/// under.
pub(super) fn replay_repo(events: &[VfsEventPayload]) -> Result<Replayed, String> {
    let mut mirror = RepoMirror::default();
    for e in events {
        let event = FileEvent::from_value(&e.body)
            .map_err(|why| format!("event {} of {:?}: {why}", e.seq, e.key))?;
        absorb(&mut mirror, &e.key, e.seq, event)?;
    }
    mirror.settle();
    let chains = mirror
        .paths
        .iter()
        .filter(|(_, at)| at.stated)
        .map(|(path, at)| {
            (
                path.clone(),
                SavedChain {
                    deltas: at.deltas.clone(),
                    size: at.size,
                    conflict: at.conflict,
                },
            )
        })
        .collect();
    Ok(Replayed {
        snapshot: Snapshot::new(mirror.base.clone(), chains),
        mirror,
    })
}

/// `events` grouped by repository, each group still in `(key, seq)` order.
pub(super) fn by_repo(events: Vec<VfsEventPayload>) -> BTreeMap<String, Vec<VfsEventPayload>> {
    let mut out: BTreeMap<String, Vec<VfsEventPayload>> = BTreeMap::new();
    for e in events {
        out.entry(e.repo.clone()).or_default().push(e);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    fn event(seq: u64, key: &str, body: serde_json::Value) -> VfsEventPayload {
        VfsEventPayload {
            timeline_id: 7,
            seq,
            repo: "candle".into(),
            key: key.into(),
            body,
        }
    }

    fn delta(at_ns: i64, content: &str, start: bool) -> serde_json::Value {
        json!({"kind": "delta", "delta": {"at_ns": at_ns, "kind": "replace", "content": content}, "start": start})
    }

    /// **Events replay into the snapshot a store saved** — the latest base,
    /// each path's deltas in order and its last state — and the mirror names
    /// every live event behind each.
    #[test]
    fn events_replay_into_the_snapshot_saved() {
        let events = vec![
            event(
                0,
                "",
                json!({"kind": "base", "base": {"tree": B, "parents": [A]}}),
            ),
            event(1, "a.rs", delta(1, "x", true)),
            event(
                2,
                "a.rs",
                json!({"kind": "state", "size": 1, "conflict": false}),
            ),
            event(5, "a.rs", delta(2, "xy", false)),
            event(
                6,
                "a.rs",
                json!({"kind": "state", "size": 2, "conflict": true}),
            ),
            event(
                3,
                "gone.rs",
                json!({"kind": "delta", "delta": {"at_ns": 3, "kind": "delete"}, "start": true}),
            ),
            event(
                4,
                "gone.rs",
                json!({"kind": "state", "size": null, "conflict": false}),
            ),
        ];
        let replayed = replay_repo(&events).unwrap();
        let expected: Snapshot = serde_json::from_value(json!({
            "base": {"tree": B, "parents": [A]},
            "chains": {
                "a.rs": {"deltas": [
                    {"at_ns": 1, "kind": "replace", "content": "x"},
                    {"at_ns": 2, "kind": "replace", "content": "xy"}
                ], "size": 2, "conflict": true},
                "gone.rs": {"deltas": [{"at_ns": 3, "kind": "delete"}], "size": null}
            }
        }))
        .unwrap();
        assert_eq!(replayed.snapshot, expected);
        assert_eq!(replayed.mirror.base_seqs, vec![0]);
        assert_eq!(replayed.mirror.paths["a.rs"].seqs, vec![1, 2, 5, 6]);
        assert_eq!(replayed.mirror.paths["gone.rs"].seqs, vec![3, 4]);
    }

    /// **A batch cut short between its events and its tombstones replays
    /// right**: the chain a `start` begins again supersedes the old one, and
    /// the later base wins — while the old events stay named, to be killed.
    #[test]
    fn a_batch_cut_short_replays_its_new_chain() {
        let events = vec![
            event(
                0,
                "",
                json!({"kind": "base", "base": {"tree": B, "parents": [A]}}),
            ),
            event(1, "a.rs", delta(1, "old", true)),
            event(
                2,
                "a.rs",
                json!({"kind": "state", "size": 3, "conflict": false}),
            ),
            event(
                3,
                "",
                json!({"kind": "base", "base": {"tree": B, "parents": [B]}}),
            ),
            event(4, "a.rs", delta(2, "merged", true)),
            event(
                5,
                "a.rs",
                json!({"kind": "state", "size": 6, "conflict": true}),
            ),
        ];
        let replayed = replay_repo(&events).unwrap();
        let chain = &replayed.snapshot.chains()["a.rs"];
        assert_eq!(chain.deltas.len(), 1);
        assert_eq!((chain.size, chain.conflict), (Some(6), true));
        assert_eq!(
            serde_json::to_value(replayed.snapshot.base()).unwrap(),
            json!({"tree": B, "parents": [B]})
        );
        assert_eq!(replayed.mirror.paths["a.rs"].seqs, vec![1, 2, 4, 5]);
        assert_eq!(replayed.mirror.base_seqs, vec![0, 3]);
    }

    fn state(size: Option<usize>) -> serde_json::Value {
        json!({"kind": "state", "size": size, "conflict": false})
    }

    fn committed(at_ns: i64, content: &str) -> Vec<VfsEventPayload> {
        vec![
            event(1, "a.rs", delta(at_ns, content, true)),
            event(2, "a.rs", state(Some(content.len()))),
        ]
    }

    /// **A save torn inside a rewrite leaves the path as it was committed.**
    /// The new chain's deltas never had their state, so they are dropped; the
    /// old chain is intact — its tombstones come after the events and never
    /// landed — and the path is marked torn, its every event still named.
    #[test]
    fn a_rewrite_torn_before_its_state_keeps_the_committed_chain() {
        let mut events = committed(1, "old");
        events.push(event(3, "a.rs", delta(2, "new", true)));
        let replayed = replay_repo(&events).unwrap();
        let chain = &replayed.snapshot.chains()["a.rs"];
        assert_eq!(chain.deltas, replayed.mirror.paths["a.rs"].deltas);
        assert_eq!(
            serde_json::to_value(&chain.deltas).unwrap(),
            json!([{"at_ns": 1, "kind": "replace", "content": "old"}])
        );
        assert_eq!(chain.size, Some(3));
        let at = &replayed.mirror.paths["a.rs"];
        assert!(at.torn);
        assert_eq!(at.seqs, vec![1, 2, 3]);
    }

    /// **A save torn inside an append leaves the file as it was committed** —
    /// a trailing delete without its state neither deletes the file nor
    /// leaves the chain saying it exists while its deltas say it does not.
    #[test]
    fn an_append_torn_before_its_state_keeps_the_committed_chain() {
        let mut events = committed(1, "old");
        events.push(event(
            3,
            "a.rs",
            json!({"kind": "delta", "delta": {"at_ns": 2, "kind": "delete"}}),
        ));
        let replayed = replay_repo(&events).unwrap();
        let chain = &replayed.snapshot.chains()["a.rs"];
        assert_eq!(chain.deltas.len(), 1);
        assert_eq!(chain.size, Some(3), "still there");
        assert!(replayed.mirror.paths["a.rs"].torn);
    }

    /// **A new file whose save was torn is not there at all** — and the rest
    /// of the repository replays: one torn path never takes the others with
    /// it.
    #[test]
    fn a_new_file_torn_before_its_state_is_absent_and_the_rest_replays() {
        let mut events = committed(1, "kept");
        events.push(event(3, "b.rs", delta(2, "half", true)));
        let replayed = replay_repo(&events).unwrap();
        assert!(!replayed.snapshot.chains().contains_key("b.rs"));
        assert!(replayed.snapshot.chains().contains_key("a.rs"));
        let b = &replayed.mirror.paths["b.rs"];
        assert!(b.torn && !b.stated);
        assert_eq!(b.seqs, vec![3], "named, so the next save kills it");
    }

    /// **A chain begun again after a torn save commits cleanly**: its `start`
    /// supersedes the torn deltas, whether or not their tombstone landed.
    #[test]
    fn a_chain_begun_again_after_a_torn_save_supersedes_it() {
        let mut events = committed(1, "old");
        events.push(event(
            3,
            "a.rs",
            json!({"kind": "delta", "delta": {"at_ns": 2, "kind": "delete"}}),
        ));
        events.push(event(4, "a.rs", delta(3, "fresh", true)));
        events.push(event(5, "a.rs", state(Some(5))));
        let replayed = replay_repo(&events).unwrap();
        let at = &replayed.mirror.paths["a.rs"];
        assert!(!at.torn);
        assert_eq!(
            serde_json::to_value(&at.deltas).unwrap(),
            json!([{"at_ns": 3, "kind": "replace", "content": "fresh"}])
        );
        assert_eq!(at.size, Some(5));
    }

    /// **Events that do not make a store's state are refused, saying why** —
    /// never replayed into a wrong one.
    #[test]
    fn events_that_make_no_state_are_refused() {
        let wrong_key = vec![event(
            1,
            "",
            json!({"kind": "state", "size": 1, "conflict": false}),
        )];
        assert!(replay_repo(&wrong_key).unwrap_err().contains("wrong key"));
        let garbage = vec![event(1, "a.rs", json!({"kind": "rename"}))];
        assert!(replay_repo(&garbage)
            .unwrap_err()
            .contains("not a file event"));
    }
}
