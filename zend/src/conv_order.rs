//! The order the sidebar lists conversations in: most recently used first.
//!
//! Each conversation carries two ranks from the substrate — `active`, stamped
//! on every use (`TimelineEntry::active`), and `order`, stamped once at
//! creation (`TimelineEntry::order`). Neither is a clock. A conversation used
//! since the last-use rank was kept outranks every one that has not been, and
//! conversations with no use recorded fall back to creation order among
//! themselves.
//!
//! The wire carries the result as `updated_ms`: the entry's position counted
//! from the bottom, so the top entry holds the highest value. The client only
//! ever sorts on it, and seats a conversation it has just used at the current
//! top plus one — which is where the server will put it too.

use crate::session::ConvEntry;

/// A conversation's ranks: `(active, order)`.
pub type Ranks = (u64, u64);

/// `entries` most recently used first, each stamped with its position.
pub fn by_last_use(mut entries: Vec<(ConvEntry, Ranks)>) -> Vec<ConvEntry> {
    entries.sort_by_key(|(_, ranks)| std::cmp::Reverse(*ranks));
    let n = entries.len() as u64;
    entries
        .into_iter()
        .enumerate()
        .map(|(i, (mut e, _))| {
            e.updated_ms = n - i as u64;
            e
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str) -> ConvEntry {
        ConvEntry {
            id: id.to_string(),
            label: String::new(),
            turn_count: 1,
            archived: false,
            updated_ms: 0,
        }
    }

    /// **The last one used leads, however old it is**, and conversations with
    /// no use recorded follow in creation order, newest first.
    #[test]
    fn the_most_recently_used_leads_then_creation_order() {
        let listed = by_last_use(vec![
            (entry("oldest-used-last"), (9, 1)),
            (entry("never-used-new"), (0, 5)),
            (entry("used-earlier"), (4, 3)),
            (entry("never-used-old"), (0, 2)),
        ]);
        let ids: Vec<(&str, u64)> = listed
            .iter()
            .map(|e| (e.id.as_str(), e.updated_ms))
            .collect();
        assert_eq!(
            ids,
            vec![
                ("oldest-used-last", 4),
                ("used-earlier", 3),
                ("never-used-new", 2),
                ("never-used-old", 1),
            ]
        );
    }
}
