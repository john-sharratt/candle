//! What a journal entry says that the journal already says — taken out before
//! the entry is kept.
//!
//! **A journal that repeats itself is a loop the character reads back.** The
//! check in [`super::verify`] dedups typed claims within one entry and nothing
//! else, so "Did: I have written the draft" and "the record is absent" were kept
//! entry after entry, and every turn's prompt carried the same line four times —
//! the character read its own repetition as the state of the world. A claim or
//! intention that says nearly what a held entry (or an earlier line of the same
//! entry) says is dropped, and so is a new open item that says what one still
//! open says. "Nearly" is word overlap: the share of the two lines' distinct
//! words they have in common.

use std::collections::BTreeSet;

use super::entry::{Entry, Item};

/// The word overlap at or above which two lines say the same thing.
const SAME: f32 = 0.7;

/// The words a line is compared on: lower case, letters and digits only, the
/// short connective words left out.
fn words(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_lowercase)
        .collect()
}

/// Whether `a` and `b` say the same thing.
fn same(a: &BTreeSet<String>, b: &BTreeSet<String>) -> bool {
    if a.is_empty() || b.is_empty() {
        return a == b;
    }
    let shared = a.intersection(b).count() as f32;
    let all = a.union(b).count() as f32;
    shared / all >= SAME
}

/// `entry` without what `held` (the journal's entries) and `open` (its open
/// items) already say, and how many lines were dropped.
pub fn drop_repeats(mut entry: Entry, held: &[Entry], open: &[Item]) -> (Entry, usize) {
    let before = entry.claims.len() + entry.intend.len() + entry.opened.len();

    let mut said: Vec<BTreeSet<String>> = held
        .iter()
        .flat_map(|e| e.claims.iter().map(|c| words(&c.text)))
        .collect();
    entry.claims.retain(|c| {
        let w = words(&c.text);
        let fresh = !said.iter().any(|s| same(s, &w));
        if fresh {
            said.push(w);
        }
        fresh
    });

    let mut meant: Vec<BTreeSet<String>> = held
        .iter()
        .flat_map(|e| e.intend.iter().map(|i| words(i)))
        .collect();
    entry.intend.retain(|i| {
        let w = words(i);
        let fresh = !meant.iter().any(|s| same(s, &w));
        if fresh {
            meant.push(w);
        }
        fresh
    });

    // A restated item keeps its id and is the character saying where it stands
    // now; only a new one that is already open is a repeat.
    let mut standing: Vec<BTreeSet<String>> = open.iter().map(|i| words(&i.text)).collect();
    entry.opened.retain(|i| {
        if i.id != 0 {
            return true;
        }
        let w = words(&i.text);
        let fresh = !standing.iter().any(|s| same(s, &w));
        if fresh {
            standing.push(w);
        }
        fresh
    });

    let after = entry.claims.len() + entry.intend.len() + entry.opened.len();
    (entry, before - after)
}

/// Whether `entry` has anything left to keep.
pub fn says_anything(entry: &Entry) -> bool {
    !(entry.claims.is_empty()
        && entry.intend.is_empty()
        && entry.opened.is_empty()
        && entry.resolved.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::journal::entry::{Claim, Kind};

    fn claim(text: &str) -> Claim {
        Claim {
            text: text.into(),
            cite: vec![1],
            kind: Kind::Did,
            perishable: false,
            typed: None,
            corrected: false,
        }
    }

    fn entry(claims: &[&str], intend: &[&str], opened: &[&str]) -> Entry {
        Entry {
            id: 0,
            from_turn: 1,
            to_turn: 2,
            from_ms: 0,
            to_ms: 0,
            claims: claims.iter().map(|c| claim(c)).collect(),
            intend: intend.iter().map(|s| s.to_string()).collect(),
            opened: opened
                .iter()
                .map(|t| Item {
                    id: 0,
                    text: t.to_string(),
                })
                .collect(),
            resolved: Vec::new(),
        }
    }

    /// **What the journal already says is not said again** — against the
    /// entries it holds, the items still open, and the entry's own earlier
    /// lines — and what is new stands.
    #[test]
    fn a_repeat_is_dropped_and_what_is_new_stands() {
        let held = [entry(
            &["I have written the full draft of The First Breath of the Anchor."],
            &["Report the draft at the command table."],
            &[],
        )];
        let open = [Item {
            id: 3,
            text: "The record for Ione Valtiere is absent".into(),
        }];
        let fresh = entry(
            &[
                "I have written the full draft of The First Breath of the Anchor",
                "The gate refused the draft: it is in the first person.",
                "The gate refused the draft — it is in the first person!",
            ],
            &[
                "Report the draft at the command table",
                "Rewrite it in the third person.",
            ],
            &[
                "the record for Ione Valtiere is absent.",
                "Find who wrote the Concord.",
            ],
        );
        let (kept, dropped) = drop_repeats(fresh, &held, &open);
        let claims: Vec<&str> = kept.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            claims,
            ["The gate refused the draft: it is in the first person."]
        );
        assert_eq!(kept.intend, ["Rewrite it in the third person."]);
        assert_eq!(
            kept.opened
                .iter()
                .map(|i| i.text.as_str())
                .collect::<Vec<_>>(),
            ["Find who wrote the Concord."]
        );
        assert_eq!(dropped, 4);
        assert!(says_anything(&kept));
    }

    #[test]
    fn an_entry_of_nothing_but_repeats_says_nothing() {
        let held = [entry(
            &["Nothing has changed in the command room."],
            &[],
            &[],
        )];
        let (kept, dropped) = drop_repeats(
            entry(&["nothing has changed in the command room"], &[], &[]),
            &held,
            &[],
        );
        assert_eq!(dropped, 1);
        assert!(!says_anything(&kept));
    }
}
