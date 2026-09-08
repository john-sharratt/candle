//! Moving one class of a sequence's carried state from one slot to another.
//!
//! A sequence carries several independent classes of state — the GDN store,
//! the PLE window, the QSA index — and a slot can hold any subset of them: a
//! wave populates the store and the PLE window, but an index also arrives by
//! injection into a slot that has never decoded a token. Moving them together
//! behind one "does the source have a store" test is how an index-only source
//! lost its index, so each class moves on its own, through this.

use std::collections::HashMap;
use std::sync::RwLock;

use candle::Result;

/// Move `from`'s entry in `map` to `to`, reporting whether there was one.
///
/// A source with an entry replaces the destination's: a move is not a merge,
/// and what the source held is what the destination's state becomes. **A
/// source with no entry moves nothing and disturbs nothing** — the destination
/// keeps whatever it held, which is right for a view that never ran a wave
/// handing its parent back an unchanged state.
///
/// `class` names the map in the poisoned-lock error.
pub(super) fn move_entry<V>(
    map: &RwLock<HashMap<usize, V>>,
    class: &str,
    from: usize,
    to: usize,
) -> Result<bool> {
    let mut m = map
        .write()
        .map_err(|_| candle::Error::Msg(format!("qwen4exp: {class} lock poisoned")))?;
    match m.remove(&from) {
        Some(v) => {
            m.insert(to, v);
            Ok(true)
        }
        None => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::move_entry;
    use std::collections::HashMap;
    use std::sync::RwLock;

    fn map(entries: &[(usize, &'static str)]) -> RwLock<HashMap<usize, &'static str>> {
        RwLock::new(entries.iter().copied().collect())
    }

    #[test]
    fn a_held_entry_moves_and_replaces_the_destinations() {
        let m = map(&[(1, "view"), (2, "parent")]);
        assert!(move_entry(&m, "test", 1, 2).unwrap());
        let m = m.into_inner().unwrap();
        assert_eq!(m.get(&2), Some(&"view"));
        assert_eq!(m.get(&1), None, "the source holds nothing after a move");
    }

    #[test]
    fn a_missing_entry_leaves_the_destination_as_it_was() {
        let m = map(&[(2, "parent")]);
        assert!(!move_entry(&m, "test", 1, 2).unwrap());
        assert_eq!(m.into_inner().unwrap().get(&2), Some(&"parent"));
    }

    /// **The shape that lost the index.** A source holding an index and no
    /// recurrent store — an injected prefix that never decoded — must still hand
    /// its index over; the classes are independent, so the empty store says
    /// nothing about whether there is an index to move.
    #[test]
    fn classes_move_independently_of_one_another() {
        let store = map(&[]);
        let index = map(&[(7, "injected pages")]);
        assert!(!move_entry(&store, "recurrent", 7, 9).unwrap());
        assert!(move_entry(&index, "index", 7, 9).unwrap());
        assert_eq!(index.into_inner().unwrap().get(&9), Some(&"injected pages"));
        assert!(store.into_inner().unwrap().is_empty());
    }
}
