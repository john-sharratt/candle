use std::cell::RefCell;

use zend_vfs::{RepoSpec, Workspace};

use super::*;

fn workspace(dir: &std::path::Path) -> Workspace {
    for repo in ["a", "b"] {
        std::fs::create_dir_all(dir.join(repo)).unwrap();
        std::fs::write(dir.join(repo).join("base.txt"), "base\n").unwrap();
    }
    Workspace::new(dir, vec![RepoSpec::named("a"), RepoSpec::named("b")]).unwrap()
}

/// The events a sequence of writes leaves live, as the log would hand
/// them back: every event, less every one a tombstone killed, in
/// `(repo, key, seq)` order — one writer's, so its stamps run in `seq`
/// order too.
fn live(writes: &[(VfsWrite, Vec<VfsEventId>)]) -> Vec<VfsEventPayload> {
    let mut events = Vec::new();
    let mut killed = BTreeSet::new();
    for (write, ids) in writes {
        for (append, id) in write.events.iter().zip(ids) {
            events.push(VfsEventPayload {
                timeline_id: 1,
                origin: id.origin,
                seq: id.seq,
                repo: append.repo.clone(),
                key: append.key.clone(),
                body: append.body.clone(),
            });
        }
        for tomb in &write.tombstones {
            killed.extend(tomb.kills.iter().copied());
        }
    }
    events.retain(|e| !killed.contains(&e.id()));
    events.sort_by(|a, b| (&a.repo, &a.key, a.seq).cmp(&(&b.repo, &b.key, b.seq)));
    events
}

/// A log in memory: every write kept, read back live.
#[derive(Default)]
struct Memory {
    writes: RefCell<Vec<(VfsWrite, Vec<VfsEventId>)>>,
    next: RefCell<u64>,
}

impl FileLog for Memory {
    fn write_files(&self, _: TimelineId, write: &VfsWrite) -> Result<Vec<VfsEventId>, String> {
        let mut next = self.next.borrow_mut();
        let ids: Vec<VfsEventId> = (0..write.events.len() as u64)
            .map(|i| VfsEventId {
                origin: 0xA,
                seq: *next + i,
            })
            .collect();
        *next += (write.events.len() + write.tombstones.len()) as u64;
        self.writes.borrow_mut().push((write.clone(), ids.clone()));
        Ok(ids)
    }

    fn read_files(&self, _: TimelineId) -> Result<Vec<VfsEventPayload>, String> {
        Ok(live(&self.writes.borrow()))
    }
}

fn timeline() -> TimelineId {
    TimelineId::from_raw(7).unwrap()
}

/// **Changes survive being written as events and replayed into a fresh
/// set** — written, edited and deleted files alike, across rounds that
/// grow, rewrite and drop chains — and the mirror a replay builds writes
/// nothing more.
#[test]
fn changes_survive_as_events() {
    let dir = tempfile::tempdir().unwrap();
    let files = RepoFiles::overlay(workspace(dir.path()));
    let log = Memory::default();
    let mirror = Mutex::new(Mirror::of(BTreeMap::new()));

    let a = files.repo("a").unwrap();
    a.write("new.txt", "new\n".into()).unwrap();
    a.edit("base.txt", "base\nedited\n".into()).unwrap();
    files.repo("b").unwrap().delete("base.txt");
    save(&log, timeline(), &files, &mirror);
    a.edit("base.txt", "base\nedited\nagain\n".into()).unwrap();
    a.write("new.txt", "rewritten\n".into()).unwrap();
    save(&log, timeline(), &files, &mirror);
    a.delete("new.txt");
    save(&log, timeline(), &files, &mirror);

    let fresh = files.fresh();
    let restored = Mutex::new(Mirror::unread());
    restore(&log, timeline(), &fresh, &restored);
    let (fa, fb) = (fresh.repo("a").unwrap(), fresh.repo("b").unwrap());
    assert_eq!(fa.read("new.txt").unwrap(), None);
    assert_eq!(
        fa.read("base.txt").unwrap().as_deref(),
        Some("base\nedited\nagain\n")
    );
    assert_eq!(fb.read("base.txt").unwrap(), None);
    assert_eq!(fa.deltas("base.txt"), a.deltas("base.txt"));
    assert_eq!(fresh.snapshots(), files.snapshots());
    assert!(
        restored.lock().unwrap().diff(&fresh.snapshots()).is_empty(),
        "the replayed mirror holds exactly what the stores do"
    );
}

/// **An unread mirror saves nothing**, and a restore that cannot read the
/// log leaves it unread.
#[test]
fn an_unread_set_saves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let files = RepoFiles::overlay(workspace(dir.path()));
    files
        .repo("a")
        .unwrap()
        .write("x.txt", "x\n".into())
        .unwrap();
    let log = Memory::default();
    save(&log, timeline(), &files, &Mutex::new(Mirror::unread()));
    assert!(log.writes.borrow().is_empty());
}

/// **Unrestorable work is kept under a name of its own**, numbered past
/// every name taken.
#[test]
fn unrestorable_work_is_kept_under_a_name_of_its_own() {
    let taken: BTreeSet<String> = ["a".to_string(), format!("a{UNRESTORED}-0")].into();
    assert_eq!(kept_name("a", &taken), format!("a{UNRESTORED}-1"));
    assert_eq!(kept_name("b", &taken), format!("b{UNRESTORED}-0"));
}
