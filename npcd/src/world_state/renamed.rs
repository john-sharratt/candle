//! A world renamed (`docs/world_rename.md`): the world under its new id takes
//! in the records its former id kept, once, and the former's are then
//! tombstoned by its authority's first write — the shape of the take-in of a
//! world kept before records ([`super::legacy`]).
//!
//! A world names the ids it had in its file, `formerly: [<old>, …]`. A former
//! id is never a world of its own again: it is not hosted, and a world file
//! under it gone from the mind is not ceased, since its records are the new
//! world's to take in.

use std::collections::{BTreeMap, BTreeSet};

use candle_conversation::persistence::{Result as PersistenceResult, SharedSubstrate};

use super::store;
use crate::registry::Registry;
use crate::world::Hosted;

/// Each world that names former ids, with them, in the order its file gives.
pub fn formers(worlds: &Registry) -> BTreeMap<String, Vec<String>> {
    worlds
        .iter()
        .filter_map(|r| {
            let ids: Vec<String> = r
                .body
                .get("formerly")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty() && *s != r.id)
                .map(str::to_string)
                .collect();
            (!ids.is_empty()).then(|| (r.id.clone(), ids))
        })
        .collect()
}

/// Every id some world names as former.
pub fn former_ids(formers: &BTreeMap<String, Vec<String>>) -> BTreeSet<String> {
    formers.values().flatten().cloned().collect()
}

/// The world that names `id` as former, if one does.
pub fn renamed_to<'a>(formers: &'a BTreeMap<String, Vec<String>>, id: &str) -> Option<&'a str> {
    formers
        .iter()
        .find(|(_, olds)| olds.iter().any(|o| o == id))
        .map(|(new, _)| new.as_str())
}

/// Put the records of the first of `formerly` that has any into `hosted`, the
/// world under its new id — which has none of its own. Returns which former
/// it took in, or `None` when none had records.
pub fn take_in(shared: &SharedSubstrate, hosted: &Hosted, formerly: &[String]) -> Option<String> {
    formerly.iter().find_map(|old| {
        let records = store::read(shared, old);
        if records.is_empty() {
            return None;
        }
        super::apply(hosted, &records);
        Some(old.clone())
    })
}

/// Tombstone every record of former world `old`, once the world under its new
/// id has written its own: above every epoch they were written under, so no
/// late write of the former brings one back.
pub fn retire(shared: &SharedSubstrate, old: &str, epoch: u64) -> PersistenceResult<usize> {
    store::cease_world(shared, old, epoch)
}

#[cfg(test)]
mod tests {
    use candle_conversation::persistence::SubstratePersistence;
    use candle_conversation::substrate::Substrate;
    use tempfile::TempDir;

    use super::super::snapshot::tests::vault;
    use super::*;
    use crate::world_state::{diff, records_of, Records};

    fn shared(dir: &TempDir) -> SharedSubstrate {
        let mut substrate = Substrate::new();
        let persistence =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        SharedSubstrate::new(substrate, persistence)
    }

    /// **A world's file names the ids it had**; one naming none, or only
    /// itself, is no rename.
    #[test]
    fn a_world_names_the_ids_it_had() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("moon.yaml"),
            "name: Moon\nformerly: [life]\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("vault.yaml"),
            "name: Vault\nformerly: [vault]\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("earth.yaml"), "name: Earth\n").unwrap();
        let worlds = Registry::load("world", dir.path()).unwrap();
        let f = formers(&worlds);
        assert_eq!(
            f,
            BTreeMap::from([("moon".to_string(), vec!["life".to_string()])])
        );
        assert_eq!(former_ids(&f), BTreeSet::from(["life".to_string()]));
        assert_eq!(renamed_to(&f, "life"), Some("moon"));
        assert_eq!(renamed_to(&f, "earth"), None);
    }

    /// **The world under its new id takes in its former's records, byte for
    /// byte, and they are retired after**: the new world comes up as the old
    /// one was; the former's records are then all tombstoned, and a late write
    /// under an earlier epoch stays dead. A former with no records is passed
    /// over.
    #[test]
    fn a_renamed_world_takes_in_its_former_and_retires_it() {
        let dir = TempDir::new().unwrap();
        let s = shared(&dir);
        let new = vault();
        let seeded = new.sim(|s| s.table_open);
        let old = vault();
        old.with_sim(|s| s.table_open = !seeded);
        let kept = records_of(&old);
        store::write(&s, "life", diff(&Records::new(), &kept), 5).unwrap();
        assert_ne!(records_of(&new), kept, "the seed is not what was kept");
        assert_eq!(
            take_in(&s, &new, &["gone".to_string(), "life".to_string()]),
            Some("life".to_string())
        );
        assert_eq!(records_of(&new), kept, "the same world, record for record");

        assert_eq!(retire(&s, "life", 6).unwrap(), kept.len());
        assert!(store::read(&s, "life").is_empty());
        store::write(&s, "life", diff(&Records::new(), &kept), 5).unwrap();
        assert!(
            store::read(&s, "life").is_empty(),
            "an earlier epoch stays dead"
        );
        assert_eq!(take_in(&s, &vault(), &["life".to_string()]), None);
    }
}
