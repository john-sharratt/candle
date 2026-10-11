//! A world's state records in the substrate (`docs/replicated_substrate.md`
//! §9.1): read under its prefix, written by its authority under its lease's
//! epoch — a changed entity as its value, a ceased one as its tombstone.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::{Result as PersistenceResult, SharedSubstrate};

use super::diff::Changes;
use super::keys::{entity, key, world_prefix};
use super::snapshot::Records;

/// Every live record of world `wid`, by its entity name.
pub fn read(shared: &SharedSubstrate, wid: &str) -> Records {
    shared
        .custom_objects_under(&world_prefix(wid))
        .into_iter()
        .filter_map(|(obj, _)| Some((entity(wid, &obj.key)?.to_string(), obj.blob)))
        .collect()
}

/// The highest lease epoch any live record of world `wid` was written under:
/// what a new authority's epoch must rise above (§9.2).
pub fn highest_epoch(shared: &SharedSubstrate, wid: &str) -> u64 {
    shared
        .custom_objects_under(&world_prefix(wid))
        .into_iter()
        .map(|(_, stamp)| stamp.epoch)
        .max()
        .unwrap_or(0)
}

/// Write a beat's `changes` to world `wid`, stamped with `epoch`, in one
/// commit.
pub fn write(
    shared: &SharedSubstrate,
    wid: &str,
    changes: &Changes,
    epoch: u64,
) -> PersistenceResult<()> {
    let objs: Vec<CustomObjectPayload> = changes
        .put
        .iter()
        .map(|(name, v)| CustomObjectPayload::new(key(wid, name), v.clone()))
        .chain(
            changes
                .ceased
                .iter()
                .map(|name| CustomObjectPayload::ceased(key(wid, name))),
        )
        .collect();
    shared.put_custom_objects_under(objs, epoch)
}

/// Tombstone every record of world `wid`, in one commit, stamped with `epoch`:
/// the world is removed, and no write under an earlier epoch brings an entity
/// of it back.
pub fn cease_world(shared: &SharedSubstrate, wid: &str, epoch: u64) -> PersistenceResult<usize> {
    let objs: Vec<CustomObjectPayload> = shared
        .custom_objects_under(&world_prefix(wid))
        .into_iter()
        .map(|(obj, _)| CustomObjectPayload::ceased(obj.key))
        .collect();
    let n = objs.len();
    shared.put_custom_objects_under(objs, epoch)?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use candle_conversation::persistence::SubstratePersistence;
    use candle_conversation::substrate::Substrate;
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::world_state::diff::diff;

    fn shared(dir: &TempDir) -> SharedSubstrate {
        let mut substrate = Substrate::new();
        let persistence =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        SharedSubstrate::new(substrate, persistence)
    }

    /// Written changes read back as the world's records; a ceased entity is
    /// gone from them, and another world's records are not among them.
    #[test]
    fn changes_written_read_back_as_the_records() {
        let dir = TempDir::new().unwrap();
        let s = shared(&dir);
        let first: Records = [
            ("body/a".to_string(), json!({"at": "hall/door"})),
            ("table".to_string(), json!({"open": true})),
        ]
        .into_iter()
        .collect();
        write(&s, "w1", &diff(&Records::new(), &first), 5).unwrap();
        write(&s, "w2", &diff(&Records::new(), &first), 5).unwrap();
        assert_eq!(read(&s, "w1"), first);

        let mut second = first.clone();
        second.remove("body/a");
        write(&s, "w1", &diff(&first, &second), 5).unwrap();
        assert_eq!(read(&s, "w1"), second);
        assert_eq!(read(&s, "w2"), first);
        assert_eq!(highest_epoch(&s, "w1"), 5);
        assert_eq!(highest_epoch(&s, "w3"), 0);
    }

    /// A removed world's records are all tombstoned, above every epoch they
    /// were written under: a former authority's late write does not bring one
    /// back, and another world is untouched.
    #[test]
    fn a_removed_world_leaves_no_record_behind() {
        let dir = TempDir::new().unwrap();
        let s = shared(&dir);
        let records: Records = [
            ("body/a".to_string(), json!({"at": "hall/door"})),
            ("sim/meta".to_string(), json!({"table_open": true})),
        ]
        .into_iter()
        .collect();
        write(&s, "w1", &diff(&Records::new(), &records), 5).unwrap();
        write(&s, "w2", &diff(&Records::new(), &records), 5).unwrap();
        assert_eq!(cease_world(&s, "w1", 6).unwrap(), 2);
        assert!(read(&s, "w1").is_empty());
        assert_eq!(read(&s, "w2"), records);
        let late: Records = [("body/a".to_string(), json!({"at": "hall/window"}))]
            .into_iter()
            .collect();
        write(&s, "w1", &diff(&Records::new(), &late), 5).unwrap();
        assert!(read(&s, "w1").is_empty(), "an earlier epoch stays dead");
    }
}
