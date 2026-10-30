//! A journal's index (§5.2): its live children, one line each — at most ten,
//! so a parent's index is always small enough to read whole; and for a
//! person's root, the groups they belong to.

use crate::actor::{framed_groups, role_in, viewer};
use crate::path::JournalPath;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::fingerprint::is_dirty;
use crate::tree::{Journal, Tree};
use crate::view::who::ago;

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
    // Named, so a reader goes a level deeper by name rather than by a guess
    // or a grep; at most ten, so the line stays short.
    let names: Vec<&str> = j
        .live()
        .filter(|(_, c)| !c.is_blank())
        .map(|(n, _)| n.as_str())
        .collect();
    let mut facts = vec![format!("{lines} line{}", if lines == 1 { "" } else { "s" })];
    if !names.is_empty() {
        facts.push(format!(
            "{} sub-journal{} ({})",
            names.len(),
            if names.len() == 1 { "" } else { "s" },
            names.join(", ")
        ));
    }
    let live = names.len();
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

/// Every group `person` belongs to, one line each, as `~`'s index lists them
/// after its sections: so a group that does not open their conversations is
/// still one they know to open.
pub fn group_lines(tree: &Tree, person: &Slug) -> Vec<String> {
    let Some(file) = tree.person_file(person) else {
        return Vec::new();
    };
    let framed = framed_groups(tree, person);
    let default = viewer(tree, person).default_group;
    file.groups
        .iter()
        .filter(|g| role_in(tree, person, g).is_some())
        .map(|g| {
            let (name, kind) = tree.group_file(g).map_or((g.to_string(), "group"), |f| {
                (f.name.clone(), f.kind.name())
            });
            let opens = if default.as_ref() == Some(g) {
                "their default, in every conversation"
            } else if framed.contains(g) {
                "in every conversation"
            } else {
                "opened when needed"
            };
            format!("@{g} — {name}, a {kind}; {opens}")
        })
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
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, OnTaken, Op};
    use crate::testing::slug;
    use crate::tree::fingerprint::fingerprint;

    /// **An index line names a journal's sub-journals**, so a reader opens
    /// the next level by name.
    #[test]
    fn an_index_line_names_its_sub_journals() {
        let mut t = sharratt();
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
        let root = p("@sharratt");
        assert_eq!(
            index(&root, t.get(&root).unwrap(), now()),
            [
                "@sharratt/topics — open for themes that span conversations. 0 lines, 2 \
                 sub-journals (budget, trip); no summary yet",
                "bots, projects, threads: nothing written yet",
            ]
        );
    }

    #[test]
    fn index_lines_say_what_is_there_and_how_fresh() {
        let mut t = sharratt();
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
            ["@sharratt/topics/trip — open for the Da Nang trip. 0 lines; empty"]
        );
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: trip.clone(),
                text: "Flights booked.".into(),
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
            },
            now().plus(7_200),
        )
        .unwrap();
        assert!(
            index(&topics, t.get(&topics).unwrap(), now().plus(7_300))[0]
                .ends_with("2 lines; summary 2 h ago (+1)")
        );
        apply(&mut t, &john(), &Op::Fold { path: trip }, now()).unwrap();
        assert_eq!(
            index(&topics, t.get(&topics).unwrap(), now()),
            ["… and 1 folded"]
        );
    }

    #[test]
    fn a_broken_journal_reads_as_unreadable() {
        let mut t = sharratt();
        let path = p("~/channels/grok");
        t.get_mut(&path).unwrap().problems.push(Problem::new(
            "persons/john/channels/grok/lines.yaml",
            Some(3),
            "`text` is missing",
        ));
        assert_eq!(
            index_line(&path, t.get(&path).unwrap(), now()),
            "~/channels/grok — open for their conversations with Grok, in its apps and the car. Unreadable: persons/john/channels/grok/lines.yaml:3: `text` is missing"
        );
    }
}
