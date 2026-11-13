//! A union written into a substrate.
//!
//! Each blob first, then each object, in the union's order, under the
//! importing member's node with fresh stamps — so whatever is ordered by stamp
//! comes back in its original order, and replicates to the other members like
//! any write. A key already live is left as it is and named in the report.

use serde::Serialize;

use crate::persistence::record::CustomObjectPayload;

use super::union::Union;

/// What an import did.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct ImportReport {
    pub objects_written: usize,
    pub blobs_written: usize,
    /// Keys whose winner was a tombstone: never written.
    pub ceased: usize,
    /// Keys already live, left as they were.
    pub left_live: Vec<String>,
}

/// Write `union` through `put_blob` and `put_object`, leaving every key
/// `is_live` names as it is.
pub fn import(
    union: Union,
    is_live: impl Fn(&str) -> bool,
    mut put_blob: impl FnMut(Vec<u8>) -> anyhow::Result<[u8; 32]>,
    mut put_object: impl FnMut(CustomObjectPayload) -> anyhow::Result<()>,
) -> anyhow::Result<ImportReport> {
    let mut report = ImportReport {
        ceased: union.ceased,
        ..Default::default()
    };
    for (name, bytes) in union.blobs {
        let written = put_blob(bytes)?;
        if written != name {
            anyhow::bail!("an imported blob's bytes do not hash to its name");
        }
        report.blobs_written += 1;
    }
    for obj in union.objects {
        if is_live(&obj.key) {
            report.left_live.push(obj.key);
            continue;
        }
        put_object(obj)?;
        report.objects_written += 1;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::super::{export, union};
    use super::*;
    use crate::persistence::blob::hex;
    use crate::persistence::identity::RecordIdentity;

    /// A substrate's custom objects in memory: each key's winning object and
    /// its record, and the blobs by name, with stamps minted in write order.
    #[derive(Default)]
    struct Store {
        objects: RefCell<BTreeMap<String, (CustomObjectPayload, RecordIdentity)>>,
        blobs: RefCell<BTreeMap<[u8; 32], Vec<u8>>>,
        hlc: RefCell<u64>,
    }

    impl Store {
        fn put(&self, obj: CustomObjectPayload) {
            let mut hlc = self.hlc.borrow_mut();
            *hlc += 1;
            let id = RecordIdentity {
                origin: 9,
                oseq: *hlc,
                gen: 1,
                epoch: 0,
                hlc: *hlc,
            };
            self.objects.borrow_mut().insert(obj.key.clone(), (obj, id));
        }

        fn put_blob(&self, bytes: Vec<u8>) -> [u8; 32] {
            let name: [u8; 32] = Sha256::digest(&bytes).into();
            self.blobs.borrow_mut().insert(name, bytes);
            name
        }

        fn export(&self, at: u64) -> String {
            let held = self.objects.borrow().values().cloned().collect();
            export(
                held,
                |n| Ok(self.blobs.borrow().get(n).cloned()),
                "zend",
                at,
            )
            .unwrap()
        }

        fn live(&self, key: &str) -> bool {
            self.objects
                .borrow()
                .get(key)
                .is_some_and(|(o, _)| !o.ceased)
        }

        fn import(&self, exports: &str) -> ImportReport {
            import(
                union(exports, "zend").unwrap(),
                |k| self.live(k),
                |b| Ok(self.put_blob(b)),
                |o| {
                    self.put(o);
                    Ok(())
                },
            )
            .unwrap()
        }
    }

    /// **An export imported into an empty substrate exports again as it
    /// was**, but for the header's time and the stamps, which the import mints
    /// fresh: the kept objects in their original order, a tombstone left out
    /// of the new substrate, `cluster/` objects in neither, the blob a kept
    /// object names carried with it; and an import over a live key leaves it.
    #[test]
    fn an_export_round_trips_through_an_import() {
        let old = Store::default();
        let portrait = old.put_blob(b"a portrait".to_vec());
        old.put(CustomObjectPayload::new("accounts/1", json!({"name": "a"})));
        old.put(CustomObjectPayload::new("cluster/members/x", json!({})));
        old.put(CustomObjectPayload::new(
            "images/p",
            json!({"sha256": hex(&portrait)}),
        ));
        old.put(CustomObjectPayload::new("mcp/s/1", json!(1)));
        old.put(CustomObjectPayload::ceased("mcp/s/1"));
        let first = old.export(100);
        assert!(!first.contains("cluster/"), "cluster state is left out");
        assert!(first.contains(r#""ceased":true"#), "tombstones travel");

        let new = Store::default();
        let report = new.import(&first);
        assert_eq!(
            report,
            ImportReport {
                objects_written: 2,
                blobs_written: 1,
                ceased: 1,
                left_live: vec![],
            }
        );
        assert!(!new.live("mcp/s/1"), "a deleted key is not written");
        let strip = |text: &str| -> Vec<String> {
            text.lines()
                .skip(1)
                .map(|l| {
                    let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
                    if let Some(m) = v.as_object_mut() {
                        m.remove("stamp");
                    }
                    v.to_string()
                })
                .collect()
        };
        let again = new.export(200);
        let live_first: Vec<String> = strip(&first)
            .into_iter()
            .filter(|l| !l.contains(r#""ceased":true"#))
            .collect();
        assert_eq!(strip(&again), live_first);

        // A live key is left as it was.
        new.put(CustomObjectPayload::new("accounts/1", json!({"name": "b"})));
        let report = new.import(&first);
        assert_eq!(
            report.left_live,
            vec!["accounts/1".to_string(), "images/p".into()]
        );
        assert_eq!(
            new.objects.borrow()["accounts/1"].0.blob,
            json!({"name": "b"})
        );
    }

    /// **The union takes each key's latest original stamp, a tombstone
    /// included**: a key one member deleted after another last wrote it stays
    /// deleted; a key one member wrote after another deleted it lives; an
    /// object only one member held is not lost.
    #[test]
    fn the_union_takes_each_keys_latest_stamp() {
        let a = Store::default();
        let b = Store::default();
        a.put(CustomObjectPayload::new("k/deleted", json!(1)));
        a.put(CustomObjectPayload::ceased("k/revived"));
        a.put(CustomObjectPayload::new("k/only_a", json!(1)));
        // b's stamps run later than a's.
        *b.hlc.borrow_mut() = 10;
        b.put(CustomObjectPayload::ceased("k/deleted"));
        b.put(CustomObjectPayload::new("k/revived", json!(2)));
        let (ea, eb) = (a.export(1), b.export(1));
        let u = union(&format!("{ea}{eb}"), "zend").unwrap();
        let keys: Vec<&str> = u.objects.iter().map(|o| o.key.as_str()).collect();
        assert_eq!(keys, vec!["k/only_a", "k/revived"]);
        assert_eq!(u.ceased, 1);
        assert_eq!(
            union(&format!("{eb}{ea}"), "zend").unwrap(),
            u,
            "the order of exports is not"
        );
        assert!(
            union(&ea, "npcd").is_err(),
            "another daemon's export is refused"
        );
        assert!(union("", "zend").is_err(), "no export is no import");
    }
}
