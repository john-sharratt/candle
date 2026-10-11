use super::*;
use serde_json::json;
use zend_vfs::FileDelta;

const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

fn base(commit: &str) -> SavedBase {
    serde_json::from_value(json!({"tree": B, "parents": [commit]})).unwrap()
}

fn replace(at_ns: i64, content: &str) -> TimedDelta {
    TimedDelta {
        at_ns,
        delta: FileDelta::Replace {
            content: content.into(),
        },
    }
}

fn chain(deltas: Vec<TimedDelta>, size: Option<usize>, conflict: bool) -> SavedChain {
    SavedChain {
        deltas,
        size,
        conflict,
    }
}

fn snapshot(commit: &str, chains: Vec<(&str, SavedChain)>) -> Snapshot {
    Snapshot::new(
        Some(base(commit)),
        chains
            .into_iter()
            .map(|(p, c)| (p.to_string(), c))
            .collect(),
    )
}

fn one(repo: &str, s: Snapshot) -> BTreeMap<String, Snapshot> {
    [(repo.to_string(), s)].into()
}

type Shape = (Vec<(String, serde_json::Value)>, Vec<(String, Vec<u64>)>);

/// One writer's event `seq` — every event in these tests is one writer's.
fn ev(seq: u64) -> VfsEventId {
    VfsEventId { origin: 0xA, seq }
}

fn seqs(ids: &[VfsEventId]) -> Vec<u64> {
    ids.iter().map(|id| id.seq).collect()
}

/// `(key, body)` of every event, and `(key, kills)` of every tombstone.
fn shape(write: &VfsWrite) -> Shape {
    (
        write
            .events
            .iter()
            .map(|e| (e.key.clone(), e.body.clone()))
            .collect(),
        write
            .tombstones
            .iter()
            .map(|t| (t.key.clone(), seqs(&t.kills)))
            .collect(),
    )
}

/// Write `write` into `mirror`, its events taking sequence numbers from
/// `next`.
fn land(mirror: &mut Mirror, write: &VfsWrite, next: &mut u64) {
    let ids: Vec<VfsEventId> = (0..write.events.len() as u64)
        .map(|i| ev(*next + i))
        .collect();
    *next += write.events.len() as u64 + write.tombstones.len() as u64;
    mirror.apply(write, &ids);
}

fn delta_json(at_ns: i64, content: &str, start: bool) -> serde_json::Value {
    let mut v =
        json!({"kind": "delta", "delta": {"at_ns": at_ns, "kind": "replace", "content": content}});
    if start {
        v["start"] = json!(true);
    }
    v
}

/// **An unread mirror writes nothing**, whatever the stores hold.
#[test]
fn an_unread_mirror_writes_nothing() {
    let s = one(
        "candle",
        snapshot(
            A,
            vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
        ),
    );
    assert!(Mirror::unread().diff(&s).is_empty());
}

/// **A new repository writes its base, and each path a started chain
/// then its state**; written again, nothing.
#[test]
fn a_new_repository_is_written_whole_once() {
    let s = one(
        "candle",
        snapshot(
            A,
            vec![(
                "a.rs",
                chain(vec![replace(1, "x"), replace(2, "yz")], Some(2), false),
            )],
        ),
    );
    let mut mirror = Mirror::of(BTreeMap::new());
    let write = mirror.diff(&s);
    let (events, tombstones) = shape(&write);
    assert!(tombstones.is_empty());
    assert_eq!(
        events,
        vec![
            (
                "".into(),
                json!({"kind": "base", "base": {"tree": B, "parents": [A]}})
            ),
            ("a.rs".into(), delta_json(1, "x", true)),
            ("a.rs".into(), delta_json(2, "yz", false)),
            (
                "a.rs".into(),
                json!({"kind": "state", "size": 2, "conflict": false})
            ),
        ]
    );
    land(&mut mirror, &write, &mut 0);
    assert!(mirror.diff(&s).is_empty(), "nothing changed");
}

/// **A chain that only grew writes its new deltas, and its state when
/// that changed.**
#[test]
fn a_grown_chain_appends() {
    let before = one(
        "candle",
        snapshot(
            A,
            vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
        ),
    );
    let mut mirror = Mirror::of(BTreeMap::new());
    let first = mirror.diff(&before);
    land(&mut mirror, &first, &mut 0);
    let after = one(
        "candle",
        snapshot(
            A,
            vec![(
                "a.rs",
                chain(vec![replace(1, "x"), replace(2, "xy")], Some(2), false),
            )],
        ),
    );
    let (events, tombstones) = shape(&mirror.diff(&after));
    assert!(tombstones.is_empty());
    assert_eq!(
        events,
        vec![
            ("a.rs".into(), delta_json(2, "xy", false)),
            (
                "a.rs".into(),
                json!({"kind": "state", "size": 2, "conflict": false})
            ),
        ]
    );
    let flagged = one(
        "candle",
        snapshot(
            A,
            vec![("a.rs", chain(vec![replace(1, "x")], Some(1), true))],
        ),
    );
    let (events, _) = shape(&mirror.diff(&flagged));
    assert_eq!(
        events,
        vec![(
            "a.rs".into(),
            json!({"kind": "state", "size": 1, "conflict": true})
        )],
        "only the flag changed"
    );
}

/// **A grown chain always ends with its state**, even one that did not
/// change: the state is what commits the deltas before it, so a save
/// torn after them leaves them uncommitted rather than half-applied.
#[test]
fn a_grown_chain_always_ends_with_its_state() {
    let before = one(
        "candle",
        snapshot(
            A,
            vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
        ),
    );
    let mut mirror = Mirror::of(BTreeMap::new());
    let mut next = 0;
    let first = mirror.diff(&before);
    land(&mut mirror, &first, &mut next);
    let after = one(
        "candle",
        snapshot(
            A,
            vec![(
                "a.rs",
                chain(vec![replace(1, "x"), replace(2, "y")], Some(1), false),
            )],
        ),
    );
    let write = mirror.diff(&after);
    let (events, tombstones) = shape(&write);
    assert!(tombstones.is_empty());
    assert_eq!(
        events,
        vec![
            ("a.rs".into(), delta_json(2, "y", false)),
            (
                "a.rs".into(),
                json!({"kind": "state", "size": 1, "conflict": false})
            ),
        ]
    );
    land(&mut mirror, &write, &mut next);
    assert!(mirror.diff(&after).is_empty());
}

/// **A path a torn save left is written whole next time**, `start`ed,
/// and every event it had tombstoned — an append would be read as
/// following the dropped deltas. A torn file no state ever committed is
/// tombstoned outright.
#[test]
fn a_torn_path_is_written_whole_and_its_events_killed() {
    let delta = |at_ns, content, start| FileEvent::Delta {
        delta: replace(at_ns, content),
        start,
    };
    let mut repo = RepoMirror::default();
    absorb(
        &mut repo,
        BASE_KEY,
        ev(0),
        FileEvent::Base { base: base(A) },
    )
    .unwrap();
    absorb(&mut repo, "a.rs", ev(1), delta(1, "x", true)).unwrap();
    let committed = FileEvent::State {
        size: Some(1),
        conflict: false,
        start: false,
    };
    absorb(&mut repo, "a.rs", ev(2), committed).unwrap();
    absorb(&mut repo, "a.rs", ev(3), delta(2, "xy", false)).unwrap();
    absorb(&mut repo, "b.rs", ev(4), delta(3, "b", true)).unwrap();
    repo.settle();
    assert!(repo.paths["a.rs"].torn && repo.paths["b.rs"].torn);
    let mirror = Mirror::of([("candle".to_string(), repo)].into());

    // The store restored the committed chain and grew it by the same
    // delta the torn save had begun to write.
    let s = one(
        "candle",
        snapshot(
            A,
            vec![(
                "a.rs",
                chain(vec![replace(1, "x"), replace(2, "xy")], Some(2), false),
            )],
        ),
    );
    let (events, tombstones) = shape(&mirror.diff(&s));
    assert_eq!(
        events,
        vec![
            ("a.rs".into(), delta_json(1, "x", true)),
            ("a.rs".into(), delta_json(2, "xy", false)),
            (
                "a.rs".into(),
                json!({"kind": "state", "size": 2, "conflict": false})
            ),
        ]
    );
    assert_eq!(
        tombstones,
        vec![("a.rs".into(), vec![1, 2, 3]), ("b.rs".into(), vec![4])]
    );
}

/// **A rewritten chain is written whole, started, and its old events
/// tombstoned; a path the store let go of is tombstoned.** Events come
/// before tombstones, so a batch cut short never kills a chain with
/// nothing in its place.
#[test]
fn a_rewritten_or_dropped_path_is_tombstoned() {
    let before = one(
        "candle",
        snapshot(
            A,
            vec![
                ("a.rs", chain(vec![replace(1, "x")], Some(1), false)),
                ("b.rs", chain(vec![replace(2, "b")], Some(1), false)),
            ],
        ),
    );
    // base 0; a.rs delta 1, state 2; b.rs delta 3, state 4.
    let mut mirror = Mirror::of(BTreeMap::new());
    let mut next = 0;
    let first = mirror.diff(&before);
    land(&mut mirror, &first, &mut next);
    let after = one(
        "candle",
        snapshot(
            A,
            vec![("a.rs", chain(vec![replace(9, "merged")], Some(6), true))],
        ),
    );
    let write = mirror.diff(&after);
    let (events, tombstones) = shape(&write);
    assert_eq!(
        events,
        vec![
            ("a.rs".into(), delta_json(9, "merged", true)),
            (
                "a.rs".into(),
                json!({"kind": "state", "size": 6, "conflict": true})
            ),
        ]
    );
    assert_eq!(
        tombstones,
        vec![("a.rs".into(), vec![1, 2]), ("b.rs".into(), vec![3, 4])]
    );
    land(&mut mirror, &write, &mut next);
    assert!(mirror.diff(&after).is_empty());
    assert_eq!(seqs(&mirror.repos["candle"].paths["a.rs"].ids), vec![5, 6]);
}

/// **A commit moves the base and lets go of what it took**: the new base
/// written, the old and the committed path tombstoned.
#[test]
fn a_commit_moves_the_base_and_tombstones_what_it_took() {
    let before = one(
        "candle",
        snapshot(
            A,
            vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
        ),
    );
    let mut mirror = Mirror::of(BTreeMap::new());
    let mut next = 0;
    let first = mirror.diff(&before);
    land(&mut mirror, &first, &mut next);
    let after = one("candle", snapshot(B, vec![]));
    let write = mirror.diff(&after);
    let (events, tombstones) = shape(&write);
    assert_eq!(
        events,
        vec![(
            "".into(),
            json!({"kind": "base", "base": {"tree": B, "parents": [B]}})
        )]
    );
    assert_eq!(
        tombstones,
        vec![("".into(), vec![0]), ("a.rs".into(), vec![1, 2])]
    );
    land(&mut mirror, &write, &mut next);
    assert!(mirror.diff(&after).is_empty());
    assert_eq!(seqs(&mirror.repos["candle"].base_ids), vec![3]);
}

/// **A repository with no changes left is forgotten whole.**
#[test]
fn a_repository_gone_is_forgotten() {
    let before = one(
        "candle",
        snapshot(
            A,
            vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
        ),
    );
    let mut mirror = Mirror::of(BTreeMap::new());
    let mut next = 0;
    let first = mirror.diff(&before);
    land(&mut mirror, &first, &mut next);
    let write = mirror.diff(&BTreeMap::new());
    let (events, tombstones) = shape(&write);
    assert!(events.is_empty());
    assert_eq!(
        tombstones,
        vec![("".into(), vec![0]), ("a.rs".into(), vec![1, 2])]
    );
    land(&mut mirror, &write, &mut next);
    assert!(mirror.repos.is_empty());
}

/// **A conflict kept exactly as the base holds it — no delta at all —
/// is written as a started state.**
#[test]
fn a_chain_with_no_delta_starts_with_its_state() {
    let s = one(
        "candle",
        snapshot(A, vec![("a.rs", chain(vec![], Some(3), true))]),
    );
    let (events, _) = shape(&Mirror::of(BTreeMap::new()).diff(&s));
    assert_eq!(
        events[1],
        (
            "a.rs".into(),
            json!({"kind": "state", "size": 3, "conflict": true, "start": true})
        )
    );
}
