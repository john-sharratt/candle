//! A journal's index (§5.2): its live children, one line each — at most ten,
//! so a parent's index is always small enough to read whole — and its
//! references, each a way to another journal (§2.5).

use crate::path::JournalPath;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::fingerprint::is_dirty;
use crate::tree::Journal;
use crate::view::who::ago;

/// `j`'s live children with something written in or below them, by name.
pub(crate) fn written(j: &Journal) -> Vec<(&Slug, &Journal)> {
    j.live().filter(|(_, c)| !c.is_blank()).collect()
}

/// `path`'s line in its parent's index.
pub fn index_line(path: &JournalPath, j: &Journal, now: Stamp) -> String {
    let mut out = path.show();
    if let Some(cue) = j.cue() {
        out.push_str(" — ");
        out.push_str(&cue);
    }
    out.push_str(". ");
    if let Some(p) = j.problems.first() {
        out.push_str(&format!("Unreadable: {p}"));
        return out;
    }
    let lines = j.all_lines().count();
    // Named two levels down, so a reader goes deeper by name rather than by
    // a guess or a grep; at most ten a level, so the line stays bounded.
    let below: Vec<(&str, Vec<&str>)> = written(j)
        .into_iter()
        .map(|(n, c)| {
            let under = written(c).into_iter().map(|(m, _)| m.as_str()).collect();
            (n.as_str(), under)
        })
        .collect();
    let mut facts = vec![format!("{lines} line{}", if lines == 1 { "" } else { "s" })];
    if !below.is_empty() {
        let deep = below.iter().any(|(_, under)| !under.is_empty());
        let listed: Vec<String> = below
            .iter()
            .map(|(n, under)| match under.is_empty() {
                true => n.to_string(),
                false => format!("{n}: {}", under.join(", ")),
            })
            .collect();
        facts.push(format!(
            "{} sub-journal{} ({})",
            below.len(),
            if below.len() == 1 { "" } else { "s" },
            listed.join(if deep { "; " } else { ", " })
        ));
    }
    let live = below.len();
    let state = match j.summary() {
        None if lines == 0 && live == 0 => "empty".to_string(),
        None => "no summary yet".to_string(),
        Some(s) => {
            let newer = j.window.iter().filter(|l| l.at > s.built).count();
            let mut st = if is_dirty(j) {
                format!("summary {}", ago(s.built, now))
            } else {
                "current".to_string()
            };
            if newer > 0 {
                st.push_str(&format!(" (+{newer})"));
            }
            if s.stale {
                st.push_str(", stale");
            }
            st
        }
    };
    out.push_str(&format!("{}; {state}", facts.join(", ")));
    if j.is_closed() {
        out.push_str("; closed");
    }
    out
}

/// `j`'s references, one line each: `→ wasmer (topics/wasmer) — open for …`.
pub fn refs_lines(j: &Journal) -> Vec<String> {
    j.meta
        .iter()
        .flat_map(|m| &m.refs)
        .map(|r| format!("→ {} ({}) — {}", r.name, r.to, r.cue))
        .collect()
}

/// The index of `j` at `path`: each live child, then the fixed ones nothing
/// is written in yet — named on one line, so a reader knows they are there
/// to write in — then how many are folded.
pub fn index(path: &JournalPath, j: &Journal, now: Stamp) -> Vec<String> {
    let blank = |c: &Journal| c.is_fixed() && c.is_blank();
    let mut out: Vec<String> = j
        .live()
        .filter(|(_, child)| !blank(child))
        .map(|(name, child)| {
            let child_path = path.child(name.clone()).expect("a child is at a place");
            index_line(&child_path, child, now)
        })
        .collect();
    let empty: Vec<&str> = j
        .live()
        .filter(|(_, child)| blank(child))
        .map(|(name, _)| name.as_str())
        .collect();
    if !empty.is_empty() {
        out.push(format!("{}: nothing written yet", empty.join(", ")));
    }
    let folded = j.children.len() - j.live_count();
    if folded > 0 {
        out.push(format!("… and {folded} folded"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::error::Problem;
    use crate::files::summary::{Cite, Sentence};
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::fold::fold_child;
    use crate::ops::{apply, OnTaken, Op};
    use crate::testing::slug;
    use crate::tree::fingerprint::fingerprint;

    /// **An index line names a journal's sub-journals**, so a reader opens
    /// the next level by name.
    #[test]
    fn an_index_line_names_its_sub_journals() {
        let mut t = zen();
        for name in ["trip", "budget"] {
            apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: p("topics"),
                    name: slug(name),
                    cue: format!("open for the {name}"),
                    on_taken: OnTaken::Refuse,
                },
                now(),
            )
            .unwrap();
        }
        let root = p("@zen");
        assert_eq!(
            index(&root, t.get(&root).unwrap(), now()),
            [
                "@zen/topics — open for everything durable the group talks about. 0 lines, 2 \
                 sub-journals (budget, trip); no summary yet",
                "bots, conversations, people, threads: nothing written yet",
            ]
        );
        // A level further down is named under its parent.
        for name in ["flights", "hotels"] {
            apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: p("topics/trip"),
                    name: slug(name),
                    cue: format!("open for the {name}"),
                    on_taken: OnTaken::Refuse,
                },
                now(),
            )
            .unwrap();
        }
        assert_eq!(
            index(&root, t.get(&root).unwrap(), now())[0],
            "@zen/topics — open for everything durable the group talks about. 0 lines, 2 \
             sub-journals (budget; trip: flights, hotels); no summary yet"
        );
    }

    #[test]
    fn index_lines_say_what_is_there_and_how_fresh() {
        let mut t = zen();
        let topics = p("topics");
        let trip = apply(
            &mut t,
            &john(),
            &Op::Create {
                under: topics.clone(),
                name: slug("trip"),
                cue: "open for the Da Nang trip".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        assert_eq!(
            index(&topics, t.get(&topics).unwrap(), now()),
            ["@zen/topics/trip — open for the Da Nang trip. 0 lines; empty"]
        );
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: trip.clone(),
                text: "Flights booked.".into(),
                speaker: None,
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        assert!(
            index(&topics, t.get(&topics).unwrap(), now())[0].ends_with("1 line; no summary yet")
        );
        let over = fingerprint(t.get(&trip).unwrap());
        let s = Sentence {
            text: "Booked.".into(),
            cites: vec![Cite::Line(id)],
        };
        apply(
            &mut t,
            &Actor::Zend,
            &Op::Summarize {
                path: trip.clone(),
                over,
                sentences: vec![s],
                digest: None,
                cue: None,
            },
            now(),
        )
        .unwrap();
        assert!(index(&topics, t.get(&topics).unwrap(), now())[0].ends_with("1 line; current"));
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: trip.clone(),
                text: "Hotel too.".into(),
                speaker: None,
            },
            now().plus(7_200),
        )
        .unwrap();
        assert!(
            index(&topics, t.get(&topics).unwrap(), now().plus(7_300))[0]
                .ends_with("2 lines; summary 2 h ago (+1)")
        );
        let parent = t.get_mut(&topics).unwrap();
        fold_child(parent, &topics, trip.last().unwrap(), now()).unwrap();
        assert_eq!(
            index(&topics, t.get(&topics).unwrap(), now()),
            ["… and 1 folded"]
        );
    }

    #[test]
    fn a_broken_journal_reads_as_unreadable() {
        let mut t = zen();
        let path = p("conversations");
        t.get_mut(&path).unwrap().problems.push(Problem::new(
            "groups/zen/conversations/lines.yaml",
            Some(3),
            "`text` is missing",
        ));
        assert_eq!(
            index_line(&path, t.get(&path).unwrap(), now()),
            "@zen/conversations — open for each conversation, from any client. Unreadable: groups/zen/conversations/lines.yaml:3: `text` is missing"
        );
    }
}
