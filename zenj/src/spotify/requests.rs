//! Requests from the phones (§4.5): the shared *ZenJ Requests* playlist,
//! watched every few seconds. Only a moved `snapshot_id` costs a read of its
//! items; each item not seen before is a request, from whoever added it.

use std::collections::{BTreeMap, HashSet};

use super::api::Item;

/// One request as it arrived.
#[derive(Debug, Clone, PartialEq)]
pub struct Asked {
    pub spotify_id: String,
    pub title: String,
    pub artist: String,
    pub isrc: Option<String>,
    pub duration_ms: u64,
    /// Who added it, by the name ZenJ knows them by.
    pub by: String,
    pub at: String,
}

/// What the watcher remembers between reads.
#[derive(Debug, Default)]
pub struct Watch {
    snapshot: Option<String>,
    seen: HashSet<(String, String)>,
    /// Spotify user id → person.
    people: BTreeMap<String, String>,
}

impl Watch {
    pub fn new(people: BTreeMap<String, String>) -> Watch {
        Watch {
            people,
            ..Watch::default()
        }
    }

    /// Whether `snapshot` is news: the items need reading.
    pub fn moved(&self, snapshot: &str) -> bool {
        self.snapshot.as_deref() != Some(snapshot)
    }

    /// The requests among `items`, read at `snapshot`, not seen before. The
    /// first read only learns what is there: what was asked before ZenJ
    /// started is not asked again.
    pub fn read(&mut self, snapshot: &str, items: &[Item]) -> Vec<Asked> {
        let first = self.snapshot.is_none();
        self.snapshot = Some(snapshot.to_string());
        let mut out = Vec::new();
        for item in items {
            let Some(t) = &item.track else { continue };
            let Some(id) = &t.id else { continue };
            let at = item.added_at.clone().unwrap_or_default();
            if !self.seen.insert((id.clone(), at.clone())) || first {
                continue;
            }
            let by = item
                .added_by
                .as_ref()
                .map(|u| self.people.get(&u.id).cloned().unwrap_or_else(|| u.id.clone()))
                .unwrap_or_else(|| "someone".into());
            out.push(Asked {
                spotify_id: id.clone(),
                title: t.name.clone(),
                artist: t.artists.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", "),
                isrc: t.external_ids.isrc.clone(),
                duration_ms: t.duration_ms,
                by,
                at,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::api::{Artist, ExternalIds, Track, User};
    use super::*;

    fn item(id: &str, at: &str, by: &str) -> Item {
        Item {
            track: Some(Track {
                id: Some(id.into()),
                name: format!("{id} song"),
                artists: vec![Artist { name: "A".into() }, Artist { name: "B".into() }],
                duration_ms: 200_000,
                external_ids: ExternalIds {
                    isrc: Some(format!("ISRC{id}")),
                },
                album: Default::default(),
            }),
            added_at: Some(at.into()),
            added_by: Some(User {
                id: by.into(),
                display_name: None,
            }),
        }
    }

    /// **What was there at start is not asked again; each new item is a
    /// request from whoever added it**, and the same track added again later
    /// is asked again.
    #[test]
    fn each_new_item_is_a_request_from_its_adder() {
        let mut w = Watch::new([("spot_yen".to_string(), "yen".to_string())].into());
        assert!(w.moved("s1"));
        assert!(w.read("s1", &[item("old", "t0", "spot_yen")]).is_empty());
        assert!(!w.moved("s1"));
        let asked = w.read(
            "s2",
            &[item("old", "t0", "spot_yen"), item("new", "t1", "spot_yen"), item("x", "t2", "stranger")],
        );
        assert_eq!(asked.len(), 2);
        assert_eq!(
            asked[0],
            Asked {
                spotify_id: "new".into(),
                title: "new song".into(),
                artist: "A, B".into(),
                isrc: Some("ISRCnew".into()),
                duration_ms: 200_000,
                by: "yen".into(),
                at: "t1".into(),
            }
        );
        assert_eq!(asked[1].by, "stranger");
        let again = w.read("s3", &[item("old", "t9", "spot_yen")]);
        assert_eq!(again.len(), 1, "added again later");
    }
}
