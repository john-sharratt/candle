//! The memory's journals as JSON, for the routes of [`super`]: a journal
//! opened, one line, the shape of a subtree, a search's hits. Every path is
//! written as its reader writes it (`~/…`, `@group/…`), every line by its
//! address, so whatever one answer names another request can take. Every
//! answer that carries what was written carries `framing` too: notes, not
//! instructions (§7.2).

use journals::files::Summary;
use journals::line::{Folds, Line};
use journals::path::JournalPath;
use journals::time::Stamp;
use journals::tree::fingerprint::is_dirty;
use journals::tree::Journal;
use journals::view::grep::{Found, Hit};
use journals::view::open::open;
use journals::view::NOT_INSTRUCTIONS;
use serde_json::{json, Map, Value};

use crate::mcp::memory::Memory;
use crate::mcp::tools::mem::Mem;

/// One line, under its address. A fold line names what it folds: a period
/// of the journal's own, or the child journal by its path.
pub fn line(path: &JournalPath, l: &Line) -> Value {
    json!({
        "line": format!("{path}#{}", l.id.as_str()),
        "at": l.at.to_string(),
        "kind": l.kind.name(),
        "by": l.by.name(),
        "person": l.person.as_ref().map(|p| p.as_str()),
        "channel": l.channel.map(|c| c.name()),
        "folds": l.folds.as_ref().map(|f| match f {
            Folds::Period(p) => json!({"period": p.to_string()}),
            Folds::Journal(j) => json!({
                "journal": path.child(j.clone()).expect("a folded child is at a place").show(),
            }),
        }),
        "proposed": l.proposed,
        "text": l.text,
    })
}

/// One line, read on its own: framed as what it is.
pub fn one_line(path: &JournalPath, l: &Line) -> Value {
    json!({ "framing": NOT_INSTRUCTIONS, "line": line(path, l) })
}

fn summary(s: &Summary, current: bool) -> Value {
    json!({
        "text": s.text(),
        "sentences": s.sentences.iter().map(|x| json!({
            "text": x.text,
            "cites": x.cites.iter().map(ToString::to_string).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "digest": s.digest,
        "built": s.built.to_string(),
        "stale": s.stale,
        "current": current,
    })
}

/// What a journal's summary is: current, behind its journal, or never built.
fn summary_state(j: &Journal) -> &'static str {
    match j.summary() {
        None => "none",
        Some(_) if is_dirty(j) => "behind",
        Some(_) => "current",
    }
}

/// `j` at `path`, opened for `m`'s caller: its summary, its sub-journals, the
/// newest `lines` of its window, what is archived, and the text a reader is
/// shown.
pub fn journal(m: &Mem, path: &JournalPath, j: &Journal, lines: usize, now: Stamp) -> Value {
    let live: Vec<Value> = j
        .live()
        .map(|(name, child)| {
            let p = path.child(name.clone()).expect("a child is at a place");
            json!({
                "journal": p.show(),
                "cue": child.cue(),
                "lines": child.all_lines().count(),
                "sub_journals": child.live_count(),
                "summary": summary_state(child),
            })
        })
        .collect();
    let folded: Vec<String> = j
        .children
        .iter()
        .filter(|(_, c)| c.is_folded())
        .map(|(name, _)| {
            path.child(name.clone())
                .expect("a child is at a place")
                .show()
        })
        .collect();
    let archived: Map<String, Value> = j
        .archives
        .iter()
        .map(|(p, held)| (p.to_string(), json!(held.len())))
        .collect();
    let shown = &j.window[j.window.len() - j.window.len().min(lines)..];
    json!({
        "framing": NOT_INSTRUCTIONS,
        "journal": path.show(),
        "cue": j.cue(),
        "readers": m.readers(path),
        "summary": j.summary().map(|s| summary(s, !is_dirty(j))),
        "sub_journals": live,
        "folded": folded,
        "window": j.window.len(),
        "lines": shown.iter().map(|l| line(path, l)).collect::<Vec<_>>(),
        "archived": archived,
        "closed": j.is_closed(),
        "problems": j.problems.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "text": open(&m.tree, path, j, lines, now),
    })
}

/// Every journal at `root` and below, folded ones too, depth first: where it
/// is, its cue, and how much it holds.
pub fn subtree(root: &JournalPath, j: &Journal) -> Vec<Value> {
    j.walk()
        .into_iter()
        .map(|(names, child)| {
            let path = names.into_iter().fold(root.clone(), |p, n| {
                p.child(n).expect("a child is at a place")
            });
            json!({
                "journal": path.show(),
                "cue": child.cue(),
                "lines": child.all_lines().count(),
                "sub_journals": child.live_count(),
                "folded": child.is_folded(),
                "closed": child.is_closed(),
                "summary": summary_state(child),
                "unreadable": !child.problems.is_empty(),
            })
        })
        .collect()
}

/// A search's hits, newest first, and how many there were before the cut.
pub fn hits(hits: &[Hit], total: usize) -> Value {
    json!({
        "framing": NOT_INSTRUCTIONS,
        "total": total,
        "hits": hits.iter().map(|h| json!({
            "journal": h.path.show(),
            "line": h.id.as_ref().map(|id| format!("{}#{}", h.path, id.as_str())),
            "found": match &h.found {
                Found::Window => "window".to_string(),
                Found::Archive(p) => format!("archive {p}"),
                Found::Summary => "summary".to_string(),
            },
            "at": h.at.to_string(),
            "shown": h.shown,
        })).collect::<Vec<_>>(),
    })
}

/// The roots `m`'s caller reads: `~`, then each of their groups.
pub fn roots(m: &Mem) -> Vec<JournalPath> {
    let mut out = vec![m.path("~").expect("`~` is a path")];
    out.extend(
        m.groups()
            .into_iter()
            .filter_map(|f| m.path(&format!("@{f}")).ok()),
    );
    out
}

/// The journal at `path`, if `m`'s caller may read it.
pub fn readable<'a>(m: &'a Mem, path: &JournalPath) -> Result<&'a Journal, String> {
    Memory::readable(&m.tree, &m.actor(), path)
}

#[cfg(test)]
mod tests {
    use journals::ops::{apply, OnTaken, Op};
    use journals::period::Period;
    use journals::slug::Slug;

    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    fn at(s: &str) -> Stamp {
        Stamp::parse(s).unwrap()
    }

    /// A tree with one topic and two notes on it, written by John via Grok.
    fn written() -> (Mem, JournalPath) {
        let mut m = mem("john");
        let mut tree = (*m.tree).clone();
        let topics = m.path("topics").unwrap();
        let actor = m.actor();
        apply(
            &mut tree,
            &actor,
            &Op::Create {
                under: topics,
                name: Slug::parse("trip").unwrap(),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse,
            },
            at("2026-10-29T10:00:00Z"),
        )
        .unwrap();
        let trip = m.path("topics/trip").unwrap();
        for (t, text) in [
            ("2026-10-29T10:01:00Z", "Flights booked."),
            ("2026-10-29T10:02:00Z", "Hotel too."),
        ] {
            apply(
                &mut tree,
                &actor,
                &Op::Note {
                    path: trip.clone(),
                    text: text.into(),
                },
                at(t),
            )
            .unwrap();
        }
        m.tree = tree.into();
        (m, trip)
    }

    #[test]
    fn a_line_is_named_by_its_address_and_says_who_wrote_it() {
        let (m, trip) = written();
        let j = m.tree.get(&trip).unwrap();
        let v = line(&trip, &j.window[0]);
        assert_eq!(
            v,
            json!({
                "line": format!("@sharratt/topics/trip#{}", j.window[0].id.as_str()),
                "at": "2026-10-29T10:01:00Z",
                "kind": "note",
                "by": "grok",
                "person": "john",
                "channel": null,
                "folds": null,
                "proposed": false,
                "text": "Flights booked.",
            })
        );
        assert_eq!(one_line(&trip, &j.window[0])["line"], v);
        assert_eq!(one_line(&trip, &j.window[0])["framing"], NOT_INSTRUCTIONS);
    }

    /// **A fold names what it folds as an input takes it**: a child by its
    /// path, which `unfold` reads back; a period by its name.
    #[test]
    fn a_fold_line_names_its_child_by_path_and_its_period_by_name() {
        let (m, trip) = written();
        let topics = m.path("topics").unwrap();
        let mut l = m.tree.get(&trip).unwrap().window[0].clone();
        l.folds = Some(Folds::Journal(Slug::parse("trip").unwrap()));
        assert_eq!(
            line(&topics, &l)["folds"],
            json!({"journal": "@sharratt/topics/trip"})
        );
        assert_eq!(m.path("@sharratt/topics/trip").unwrap(), trip);
        l.folds = Some(Folds::Period(Period::parse("2026-10-29").unwrap()));
        assert_eq!(line(&topics, &l)["folds"], json!({"period": "2026-10-29"}));
    }

    #[test]
    fn an_opened_journal_holds_its_newest_lines_and_who_reads_it() {
        let (m, trip) = written();
        let j = m.tree.get(&trip).unwrap();
        let v = journal(&m, &trip, j, 1, at("2026-10-30T00:00:00Z"));
        assert_eq!(v["journal"], "@sharratt/topics/trip");
        assert_eq!(v["cue"], "open for the trip");
        assert_eq!(v["readers"], "shared with Yen and Thanh Long");
        assert_eq!(v["summary"], Value::Null);
        assert_eq!(v["window"], 2);
        assert_eq!(v["lines"].as_array().unwrap().len(), 1);
        assert_eq!(v["lines"][0]["text"], "Hotel too.");
        assert_eq!(v["sub_journals"], json!([]));
        assert_eq!(v["folded"], json!([]));
        assert_eq!(v["archived"], json!({}));
        assert_eq!(v["problems"], json!([]));
        assert_eq!(v["framing"], NOT_INSTRUCTIONS);
        assert_eq!(
            v["text"],
            open(&m.tree, &trip, j, 1, at("2026-10-30T00:00:00Z"))
        );
        let topics = m.path("topics").unwrap();
        let t = journal(
            &m,
            &topics,
            m.tree.get(&topics).unwrap(),
            50,
            at("2026-10-30T00:00:00Z"),
        );
        assert_eq!(
            t["sub_journals"],
            json!([{
                "journal": "@sharratt/topics/trip",
                "cue": "open for the trip",
                "lines": 2,
                "sub_journals": 0,
                "summary": "none",
            }])
        );
    }

    #[test]
    fn a_subtree_lists_every_journal_below_its_root() {
        let (m, _) = written();
        let fam = m.path("@sharratt").unwrap();
        let rows = subtree(&fam, m.tree.get(&fam).unwrap());
        let names: Vec<&str> = rows
            .iter()
            .map(|r| r["journal"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "@sharratt",
                "@sharratt/bots",
                "@sharratt/projects",
                "@sharratt/threads",
                "@sharratt/topics",
                "@sharratt/topics/trip",
            ]
        );
        assert_eq!(rows[5]["lines"], 2);
        assert_eq!(rows[5]["folded"], false);
        assert_eq!(
            roots(&m).iter().map(JournalPath::show).collect::<Vec<_>>(),
            ["~", "@sharratt"]
        );
    }

    #[test]
    fn a_hit_names_its_line_and_where_it_was_found() {
        let (m, trip) = written();
        let id = m.tree.get(&trip).unwrap().window[1].id.clone();
        let h = Hit {
            path: trip.clone(),
            found: Found::Window,
            id: Some(id.clone()),
            at: at("2026-10-29T10:02:00Z"),
            shown: "shown".into(),
        };
        let v = hits(&[h], 3);
        assert_eq!(v["framing"], NOT_INSTRUCTIONS);
        assert_eq!(v["total"], 3);
        assert_eq!(
            v["hits"][0],
            json!({
                "journal": "@sharratt/topics/trip",
                "line": format!("@sharratt/topics/trip#{}", id.as_str()),
                "found": "window",
                "at": "2026-10-29T10:02:00Z",
                "shown": "shown",
            })
        );
    }
}
