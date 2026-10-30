//! Grep (§6.5 `journal_grep`): a case-insensitive phrase over every line,
//! archive and summary a person may read — their own folder and their
//! groups' — or one subtree of it. Folded journals are searched too:
//! folding changes what is shown by default, never what is found.

use crate::actor::may_read;
use crate::line::LineId;
use crate::path::JournalPath;
use crate::period::Period;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::Tree;
use crate::view::who::render_line;

/// Where a hit is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Found {
    Window,
    Archive(Period),
    Summary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub path: JournalPath,
    pub found: Found,
    pub id: Option<LineId>,
    pub at: Stamp,
    /// The line as a reader sees it, or the summary's text.
    pub shown: String,
}

/// Every hit for `phrase` that `person` may read, under `under` when given,
/// newest first, and the count before `limit` cut it.
pub fn grep(
    tree: &Tree,
    person: &Slug,
    phrase: &str,
    under: Option<&JournalPath>,
    limit: usize,
) -> (Vec<Hit>, usize) {
    let want = phrase.trim().to_lowercase();
    if want.is_empty() {
        return (Vec::new(), 0);
    }
    let mut hits = Vec::new();
    for (path, j) in tree.journals() {
        if !may_read(tree, person, path.owner()) || under.is_some_and(|u| !u.holds(&path)) {
            continue;
        }
        if !j.problems.is_empty() {
            continue;
        }
        for l in &j.window {
            if l.text.to_lowercase().contains(&want) {
                hits.push(Hit {
                    path: path.clone(),
                    found: Found::Window,
                    id: Some(l.id.clone()),
                    at: l.at,
                    shown: render_line(tree, l),
                });
            }
        }
        for (period, lines) in &j.archives {
            for l in lines {
                if l.text.to_lowercase().contains(&want) {
                    hits.push(Hit {
                        path: path.clone(),
                        found: Found::Archive(*period),
                        id: Some(l.id.clone()),
                        at: l.at,
                        shown: render_line(tree, l),
                    });
                }
            }
        }
        if let Some(s) = j.summary() {
            let text = s.text();
            if text.to_lowercase().contains(&want)
                || s.digest
                    .as_deref()
                    .is_some_and(|d| d.to_lowercase().contains(&want))
            {
                hits.push(Hit {
                    path: path.clone(),
                    found: Found::Summary,
                    id: None,
                    at: s.built,
                    shown: text,
                });
            }
        }
    }
    hits.sort_by(|a, b| b.at.cmp(&a.at).then_with(|| a.path.cmp(&b.path)));
    let total = hits.len();
    hits.truncate(limit);
    (hits, total)
}

/// Hits as a reader sees them.
pub fn render(hits: &[Hit], total: usize) -> String {
    if hits.is_empty() {
        return "Nothing found.".into();
    }
    let mut out: Vec<String> = hits
        .iter()
        .map(|h| match &h.found {
            Found::Window => format!("{} {}", h.path, h.shown),
            Found::Archive(p) => format!("{} (archive {p}) {}", h.path, h.shown),
            Found::Summary => format!("{} (summary) {}", h.path, h.shown),
        })
        .collect();
    if total > hits.len() {
        out.push(format!(
            "… and {} more; narrow the phrase or search under one journal",
            total - hits.len()
        ));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{john, now, p, p_as, sharratt, yen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::testing::{note, slug};

    fn seeded() -> Tree {
        let mut t = sharratt();
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("~"),
                text: "John's permit worry.".into(),
            },
            now(),
        )
        .unwrap();
        apply(
            &mut t,
            &yen(),
            &Op::Note {
                path: p_as("yen", "~"),
                text: "Yen's private permit note.".into(),
            },
            now(),
        )
        .unwrap();
        apply(
            &mut t,
            &yen(),
            &Op::Note {
                path: p("topics"),
                text: "Permit lodged with council.".into(),
            },
            now().plus(60),
        )
        .unwrap();
        let trip = apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("old"),
                cue: "x".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let j = t.get_mut(&trip).unwrap();
        j.archives.insert(
            Period::parse("2026-09-01").unwrap(),
            vec![note("2026-09-01T10:00:00Z", "An old PERMIT.")],
        );
        apply(&mut t, &john(), &Op::Fold { path: trip }, now()).unwrap();
        t
    }

    #[test]
    fn grep_finds_what_a_person_reads_and_nothing_else() {
        let t = seeded();
        let (hits, total) = grep(&t, &slug("john"), "permit", None, 50);
        let shown: Vec<String> = hits
            .iter()
            .map(|h| format!("{} {:?}", h.path, h.found))
            .collect();
        assert_eq!(total, 3);
        assert_eq!(
            shown,
            [
                "@sharratt/topics Window",
                "~ Window",
                "@sharratt/topics/old Archive(Day { year: 2026, month: 9, day: 1 })",
            ]
        );
        let (yens, _) = grep(&t, &slug("yen"), "private", None, 50);
        assert_eq!(yens.len(), 1);
        let (johns, _) = grep(&t, &slug("john"), "private", None, 50);
        assert!(
            johns.is_empty(),
            "Yen's own folder is never John's to search"
        );
    }

    #[test]
    fn grep_narrows_to_a_subtree_and_cuts_at_its_limit() {
        let t = seeded();
        let (hits, total) = grep(&t, &slug("john"), "permit", Some(&p("topics")), 1);
        assert_eq!((hits.len(), total), (1, 2));
        let text = render(&hits, total);
        assert!(
            text.ends_with("… and 1 more; narrow the phrase or search under one journal"),
            "{text}"
        );
        assert_eq!(render(&[], 0), "Nothing found.");
        assert_eq!(grep(&t, &slug("john"), "  ", None, 5).1, 0);
    }
}
