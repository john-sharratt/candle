//! Checks that a published snapshot stays fixed and that concurrent writes all
//! land.
//!
//! A reader that holds a cast keeps seeing it as it was, whatever is written
//! after. Writes are computed from the cast the previous write published, so
//! writes to one character from many threads at once must not drop one
//! another's beliefs, in memory or after a restart.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_lifecycle::tests::{body, ident, nid, tmp, ME};

    /// **A snapshot a reader holds is not moved by a later write**: it answers
    /// as the cast was when it was taken, while a fresh view sees the write.
    #[test]
    fn a_snapshot_is_not_moved_by_a_write() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());
        let held = n.view();

        n.patch(id, ME, &json!({ "name": "Varek the Elder" }), 2_000)
            .unwrap();

        assert_eq!(held.get(id, ME).unwrap()["name"], "Varek");
        assert_eq!(held.get(id, ME).unwrap()["revision"], 1);
        assert_eq!(n.view().get(id, ME).unwrap()["name"], "Varek the Elder");
        assert_eq!(n.view().get(id, ME).unwrap()["revision"], 2);
    }

    /// **Concurrent writes to one character lose nothing**: each is computed
    /// from what the one before it published, so eight threads each adding
    /// their own belief leave all eight, at revision 1 + 8 — in memory and
    /// after a restart.
    #[test]
    fn concurrent_writes_to_one_character_all_land() {
        let dir = tmp();
        let n = Arc::new(Npcs::load(&dir).unwrap());
        let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());

        let writers: Vec<_> = (0..8)
            .map(|i| {
                let n = Arc::clone(&n);
                std::thread::spawn(move || {
                    let b = json!({ "statement": format!("belief {i}") });
                    n.put_belief(id, ME, &format!("b{i}"), &b, 2_000 + i)
                        .unwrap_or_else(|e| panic!("belief {i} refused: {e:?}"));
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }

        let check = |cast: &Cast| {
            let held = cast.payload(id).expect("still there");
            let mut ids: Vec<&str> = held.beliefs.iter().map(|b| b.belief_id.as_str()).collect();
            ids.sort_unstable();
            assert_eq!(ids, ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]);
            assert_eq!(held.revision, 9);
        };
        check(&n.view());
        drop(n);
        check(&Npcs::load(&dir).unwrap().view());
    }
}
