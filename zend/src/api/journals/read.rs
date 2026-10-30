//! The read routes (`docs/memory_backbone.md` §6.7):
//!
//! `GET /v1/journals?journal=&lines=` — one journal opened (the caller's
//!   group's master when none is named): its summary, focus, references,
//!   sub-journals, newest `lines` (50 by default), what is archived and who
//!   reads it; or, for `<path>#<line id>`, that one line.
//! `GET /v1/journals/tree?under=` — every journal under `under`, or under
//!   each of the caller's groups' masters, folded ones too.
//! `GET /v1/journals/grep?pattern=&under=&max=` — the lines, archives and
//!   summaries that hold `pattern`, newest first, at most `max` (20 by
//!   default, 50 at most, as `journal_grep` answers).
//!
//! A member answers from the tree it last fetched, so a line written through
//! another member is read here after this one's next sync, within seconds.

use std::sync::Arc;

use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use journals::path::JournalPath;
use journals::scope::Channel;
use journals::time::Stamp;
use journals::tree::WINDOW;
use journals::view::grep::grep as search;
use serde::Deserialize;
use serde_json::json;

use super::{blocking, caller, malformed, refusal, view, Answer};
use crate::mcp::tools::mem::{Mem, Miss};
use crate::session::ZendSession;

/// The most hits one search answers, as `journal_grep` allows.
pub const MAX_HITS: usize = 50;
const DEFAULT_HITS: usize = 20;

fn invalid(e: &str) -> Answer {
    refusal(StatusCode::BAD_REQUEST, "invalid", e)
}

fn not_found(e: &str) -> Answer {
    refusal(StatusCode::NOT_FOUND, "not_found", e)
}

/// A path that missed: malformed, or naming nothing the caller reads.
fn missed(m: Miss) -> Answer {
    match m {
        Miss::Malformed(e) => invalid(&e),
        Miss::Absent(e) => not_found(&e),
    }
}

/// The journal or line `journal` names, opened for `m`'s caller: their
/// group's master when it names none.
pub fn opened(m: &Mem, journal: Option<&str>, lines: Option<usize>, now: Stamp) -> Answer {
    let master = m.group.as_ref().map(|g| format!("@{g}"));
    let Some(journal) = journal.or(master.as_deref()) else {
        return invalid("you are in no group: name the `journal`, `@group/…`");
    };
    if journal.contains('#') {
        let address = match m.find_address(journal) {
            Ok(a) => a,
            Err(e) => return missed(e),
        };
        return match view::readable(m, &address.path) {
            Ok(j) => match j.line(&address.id) {
                Some(l) => (StatusCode::OK, view::one_line(&address.path, l)),
                None => not_found(&format!("{address} is no line")),
            },
            Err(e) => not_found(&e),
        };
    }
    let path = match m.find(journal) {
        Ok(p) => p,
        Err(e) => return missed(e),
    };
    let lines = lines.unwrap_or(WINDOW).clamp(1, WINDOW);
    match view::readable(m, &path) {
        Ok(j) => (StatusCode::OK, view::journal(m, &path, j, lines, now)),
        Err(e) => not_found(&e),
    }
}

/// Every journal under `under`, or under each root `m`'s caller reads.
pub fn listed(m: &Mem, under: Option<&str>) -> Answer {
    let roots: Vec<JournalPath> = match under {
        None => view::roots(m),
        Some(u) => match m.find(u) {
            Ok(p) => vec![p],
            Err(e) => return missed(e),
        },
    };
    let mut rows = Vec::new();
    for root in &roots {
        match view::readable(m, root) {
            Ok(j) => rows.extend(view::subtree(root, j)),
            Err(e) => return not_found(&e),
        }
    }
    (StatusCode::OK, json!({ "journals": rows }))
}

/// What holds `pattern`, under `under` or anywhere `m`'s caller reads.
pub fn searched(m: &Mem, pattern: &str, under: Option<&str>, max: Option<usize>) -> Answer {
    let under = match under {
        None => None,
        Some(u) => match m.find(u) {
            Ok(p) => match view::readable(m, &p) {
                Ok(_) => Some(p),
                Err(e) => return not_found(&e),
            },
            Err(e) => return missed(e),
        },
    };
    let max = max.unwrap_or(DEFAULT_HITS).clamp(1, MAX_HITS);
    let (hits, total) = search(&m.tree, &m.actor(), pattern, under.as_ref(), max);
    (StatusCode::OK, view::hits(&hits, total))
}

#[derive(Deserialize)]
pub struct ReadQuery {
    journal: Option<String>,
    lines: Option<usize>,
}

pub async fn read(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    q: Result<Query<ReadQuery>, QueryRejection>,
) -> Response {
    let Query(q) = match q {
        Ok(q) => q,
        Err(e) => return malformed(e.body_text()),
    };
    blocking(move || match caller(&session, &headers, Channel::Other) {
        Ok(m) => opened(&m, q.journal.as_deref(), q.lines, Stamp::now()),
        Err(a) => a,
    })
    .await
}

#[derive(Deserialize)]
pub struct TreeQuery {
    under: Option<String>,
}

pub async fn tree(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    q: Result<Query<TreeQuery>, QueryRejection>,
) -> Response {
    let Query(q) = match q {
        Ok(q) => q,
        Err(e) => return malformed(e.body_text()),
    };
    blocking(move || match caller(&session, &headers, Channel::Other) {
        Ok(m) => listed(&m, q.under.as_deref()),
        Err(a) => a,
    })
    .await
}

#[derive(Deserialize)]
pub struct GrepQuery {
    pattern: String,
    under: Option<String>,
    max: Option<usize>,
}

pub async fn grep(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    q: Result<Query<GrepQuery>, QueryRejection>,
) -> Response {
    let Query(q) = match q {
        Ok(q) => q,
        Err(e) => return malformed(e.body_text()),
    };
    blocking(move || match caller(&session, &headers, Channel::Other) {
        Ok(m) => searched(&m, &q.pattern, q.under.as_deref(), q.max),
        Err(a) => a,
    })
    .await
}

#[cfg(test)]
mod tests {
    use journals::ops::{apply, OnTaken, Op};
    use journals::slug::Slug;

    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    fn at(s: &str) -> Stamp {
        Stamp::parse(s).unwrap()
    }

    /// `person`'s view of Zen with a topic and a note on the master, and the
    /// note's address.
    fn written(person: &str) -> (Mem, String) {
        let mut m = mem(person);
        let mut tree = (*m.tree).clone();
        let john = mem("john");
        apply(
            &mut tree,
            &john.actor(),
            &Op::Create {
                under: john.path("topics").unwrap(),
                name: Slug::parse("trip").unwrap(),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse,
            },
            at("2026-10-29T10:00:00Z"),
        )
        .unwrap();
        let master = john.path("@zen").unwrap();
        apply(
            &mut tree,
            &john.actor(),
            &Op::Note {
                path: master.clone(),
                text: "Everyone drinks tea.".into(),
                speaker: None,
            },
            at("2026-10-29T10:01:00Z"),
        )
        .unwrap();
        let id = tree.get(&master).unwrap().window[0].id.clone();
        m.tree = tree.into();
        (m, format!("@zen#{}", id.as_str()))
    }

    #[test]
    fn a_journal_is_opened_and_a_line_read_by_its_address() {
        let (m, address) = written("john");
        let (status, body) = opened(&m, None, None, at("2026-10-30T00:00:00Z"));
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["journal"], "@zen", "the master when none is named");
        assert_eq!(body["lines"][0]["line"], address.as_str());
        let (status, body) = opened(&m, Some(&address), None, at("2026-10-30T00:00:00Z"));
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["line"]["text"], "Everyone drinks tea.");
        let missing = "@zen#2026-01-01-0000-0000";
        assert_eq!(
            opened(&m, Some(missing), None, at("2026-10-30T00:00:00Z")),
            (
                StatusCode::NOT_FOUND,
                json!({"error": "not_found", "message": "@zen#2026-01-01-0000-0000 is no line"})
            )
        );
        assert_eq!(
            opened(&m, Some("nowhere"), None, at("2026-10-30T00:00:00Z")),
            invalid(&m.path("nowhere").unwrap_err())
        );
        assert_eq!(
            opened(&m, Some("topics/none"), None, at("2026-10-30T00:00:00Z")),
            not_found(&m.path("topics/none").unwrap_err())
        );
        assert_eq!(
            opened(&mem("solo"), None, None, at("2026-10-30T00:00:00Z")),
            invalid("you are in no group: name the `journal`, `@group/…`")
        );
    }

    /// **A group the caller is outside is never read**: a guest finds no
    /// line of Zen's, by address or by search.
    #[test]
    fn a_line_of_a_group_the_caller_is_outside_is_never_read() {
        let (guest, address) = written("guest");
        assert_eq!(
            opened(&guest, Some(&address), None, at("2026-10-30T00:00:00Z")),
            not_found("there is no journal @zen")
        );
        let (_, body) = searched(&guest, "tea", None, None);
        assert_eq!(body["total"], 0);
    }

    #[test]
    fn the_tree_lists_the_callers_groups_or_one_subtree() {
        let (m, _) = written("john");
        let (status, body) = listed(&m, None);
        assert_eq!(status, StatusCode::OK);
        let names: Vec<&str> = body["journals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["journal"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "@zen",
                "@zen/bots",
                "@zen/conversations",
                "@zen/people",
                "@zen/threads",
                "@zen/topics",
                "@zen/topics/trip",
            ]
        );
        let (_, body) = listed(&m, Some("topics"));
        assert_eq!(body["journals"].as_array().unwrap().len(), 2);
        assert_eq!(
            listed(&m, Some("@nobody")),
            not_found("there is no journal @nobody")
        );
    }

    #[test]
    fn a_search_finds_what_the_caller_reads_within_its_bounds() {
        let (m, address) = written("john");
        let (status, body) = searched(&m, "TEA", None, None);
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total"], 1);
        assert_eq!(body["hits"][0]["line"], address.as_str());
        let (_, body) = searched(&m, "tea", Some("topics"), None);
        assert_eq!(body["total"], 0);
        assert_eq!(
            searched(&m, "tea", Some("@nobody"), None),
            not_found("there is no journal @nobody")
        );
        assert_eq!(
            searched(&m, "tea", Some("nowhere"), None),
            invalid(&m.path("nowhere").unwrap_err())
        );
    }
}
