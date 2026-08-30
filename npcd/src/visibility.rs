//! Hidden authored content, and the one thing that reveals it.
//!
//! # What this is for
//!
//! Some authored content is not for whoever happens to be looking. The `earth`
//! world is the case here: it is the world that admits the adult response
//! categories, and it should not appear in a list somebody is shown over a
//! shoulder or on a screen share.
//!
//! # Hidden, not secret
//!
//! A document with `hidden: true` is left out of listings. It is revealed by
//! typing a **whole word** of its name into the filter — `earth` reveals
//! `earth`; `ear` does not.
//!
//! Whole-word is the whole point. A prefix match would let anybody find hidden
//! content by typing one letter and watching what appears, which is exactly the
//! browsing this prevents. Requiring a complete word means you cannot discover
//! what is hidden — you can only ask for something you already know the name
//! of, and then you get it.
//!
//! It is **not** an access control and this module does not pretend otherwise.
//! Anybody who knows the id can `GET /v1/world/earth` and read it, which is the
//! same thing typing the word does. What it buys is that the content is not
//! *offered*: it does not turn up in a listing, an autocomplete, or a
//! screenshot. That is the actual requirement — the public does not need to
//! know it is there — and the honest name for it is discretion rather than
//! security.
//!
//! It is also deliberately **role-independent**. An admin sees the same listing
//! as anybody else, because the moment that matters is a demo, and during a
//! demo the person at the keyboard is signed in as an admin.

use serde_json::Value;

/// Whether an authored document asks to be kept out of listings.
///
/// Absent means visible. A document that says nothing is a document nobody
/// thought about, and defaulting *that* to hidden would quietly remove content
/// from a console for no stated reason.
pub fn is_hidden(body: &Value) -> bool {
    body.get("hidden").and_then(Value::as_bool).unwrap_or(false)
}

/// Whether `query` names this document by a whole word.
///
/// `tokens` are the document's own words — its id and name, split on the
/// separators an id may contain. A query word matches when it *equals* one of
/// them, case-insensitively.
pub fn revealed_by(query: &str, tokens: &[String]) -> bool {
    words(query).any(|w| tokens.iter().any(|t| t.eq_ignore_ascii_case(w)))
}

/// The words of an id and a name, together.
///
/// `battle-cities` yields `battle`, `cities` **and** `battle-cities`: the whole
/// id is a word somebody may reasonably type, and splitting it away would mean
/// pasting the exact id failed to find the thing it names.
pub fn tokens_of(id: &str, body: &Value) -> Vec<String> {
    let name = body.get("name").and_then(Value::as_str).unwrap_or_default();
    let mut out: Vec<String> = words(id).chain(words(name)).map(str::to_owned).collect();
    if !id.is_empty() {
        out.push(id.to_owned());
    }
    out
}

/// Split on everything that is not alphanumeric, dropping empties.
///
/// Ids are lowercase-and-hyphens and names are prose, so one rule covers both:
/// `cindy-tan` and `Cindy Tan` produce the same two words.
fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
}

/// Whether this document belongs in a listing filtered by `query`.
///
/// Two different rules, on purpose:
///
/// - **Visible** documents are narrowed by an ordinary substring, because that
///   is what a person expects a filter box to do — `comm` finds `commander`.
/// - **Hidden** ones need a whole word, because a substring filter would reveal
///   them one letter at a time, which is the browsing this exists to prevent.
///
/// An empty query lists everything visible and nothing hidden.
///
/// Both rules live here rather than in the browser because a hidden document is
/// never sent: a client-side filter would have nothing to reveal however
/// completely it was typed, so the narrowing has to happen where the hiding
/// does or the two disagree.
pub fn listable(id: &str, body: &Value, query: &str) -> bool {
    let q = query.trim();
    if is_hidden(body) {
        return !q.is_empty() && revealed_by(q, &tokens_of(id, body));
    }
    q.is_empty() || matches_loosely(q, id, body)
}

/// A visible document's ordinary filter match: case-insensitive substring over
/// its id and its name.
fn matches_loosely(query: &str, id: &str, body: &Value) -> bool {
    let name = body.get("name").and_then(Value::as_str).unwrap_or_default();
    let q = query.to_lowercase();
    id.to_lowercase().contains(&q) || name.to_lowercase().contains(&q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn earth() -> Value {
        json!({ "name": "Earth", "hidden": true })
    }

    #[test]
    fn a_document_that_says_nothing_is_visible() {
        assert!(!is_hidden(&json!({ "name": "Battle Cities" })));
        assert!(!is_hidden(&json!({ "hidden": false })));
        assert!(is_hidden(&json!({ "hidden": true })));
        // Not a bool is not a claim to be hidden.
        assert!(!is_hidden(&json!({ "hidden": "yes" })));
    }

    /// **The property this module exists for.** A prefix must not reveal, or
    /// anybody can find hidden content by typing one letter and watching.
    #[test]
    fn a_prefix_never_reveals_and_a_whole_word_does() {
        let t = tokens_of("earth", &earth());
        for no in ["", "e", "ea", "ear", "eart", "art", "rth", "earths"] {
            assert!(!revealed_by(no, &t), "`{no}` revealed a hidden world");
        }
        for yes in ["earth", "Earth", "EARTH", "  earth  "] {
            assert!(revealed_by(yes, &t), "`{yes}` did not reveal it");
        }
    }

    /// A hyphenated id is findable by either half or by the whole thing —
    /// pasting the exact id must work.
    #[test]
    fn a_hyphenated_id_is_found_by_a_part_or_the_whole() {
        let b = json!({ "name": "Battle Cities", "hidden": true });
        let t = tokens_of("battle-cities", &b);
        for yes in ["battle", "cities", "battle-cities", "Battle Cities"] {
            assert!(revealed_by(yes, &t), "`{yes}` did not reveal it");
        }
        for no in ["batt", "citi", "battlecities"] {
            assert!(!revealed_by(no, &t), "`{no}` revealed it");
        }
    }

    /// A query with several words reveals if *any* of them names the document,
    /// so a filter that also narrows on something else still finds it.
    #[test]
    fn any_word_of_the_query_is_enough() {
        let t = tokens_of("earth", &earth());
        assert!(revealed_by("sydney earth", &t));
        assert!(revealed_by("earth, singapore", &t));
        assert!(!revealed_by("sydney singapore", &t));
    }

    /// A visible document filters the way a person expects a filter box to
    /// work: a substring, on the id or the name.
    #[test]
    fn a_visible_document_narrows_on_a_substring() {
        let v = json!({ "name": "Battle Cities" });
        for yes in [
            "",
            "b",
            "bat",
            "battle",
            "cit",
            "Battle Cities",
            "battle-cit",
        ] {
            assert!(listable("battle-cities", &v, yes), "`{yes}` hid it");
        }
        for no in ["zzz", "earth", "sandbox"] {
            assert!(!listable("battle-cities", &v, no), "`{no}` matched it");
        }
    }

    /// And a hidden one does not, however close the substring gets.
    #[test]
    fn a_hidden_document_is_listed_only_when_named_in_full() {
        assert!(!listable("earth", &earth(), ""));
        assert!(!listable("earth", &earth(), "world"));
        // The cases a substring filter would have matched.
        for near in ["e", "ea", "ear", "eart", "art", "rt"] {
            assert!(!listable("earth", &earth(), near), "`{near}` revealed it");
        }
        assert!(listable("earth", &earth(), "earth"));
        assert!(listable("earth", &earth(), "Earth"));
    }

    /// The two rules together: filtering for one thing does not drag a hidden
    /// document along because it happened to share letters.
    #[test]
    fn filtering_narrows_the_visible_without_revealing_the_hidden() {
        let vis = json!({ "name": "Sandbox" });
        let hid = json!({ "name": "Sandbox Annexe", "hidden": true });
        // `sand` is a substring of both. Only the visible one answers.
        assert!(listable("sandbox", &vis, "sand"));
        assert!(!listable("sandbox-annexe", &hid, "sand"));
        // The whole word reaches both, which is the point of knowing it.
        assert!(listable("sandbox", &vis, "sandbox"));
        assert!(listable("sandbox-annexe", &hid, "sandbox"));
    }

    /// The name is a way in too — somebody typing what they see on screen.
    #[test]
    fn the_name_reveals_as_well_as_the_id() {
        let doc = json!({ "name": "Low Fen", "hidden": true });
        let t = tokens_of("lf-2", &doc);
        assert!(revealed_by("fen", &t));
        assert!(revealed_by("low", &t));
        assert!(!revealed_by("lo", &t));
    }
}
