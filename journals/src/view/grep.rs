//! Grep (§6.5 `journal_grep`): a case-insensitive phrase over every line,
//! archive and summary a person may read — their own folder and their
//! groups' — or one subtree of it. Folded journals are searched too:
//! folding changes what is shown by default, never what is found.

use crate::actor::may_read;
use crate::line::{Line, LineId};
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

/// The most of a line's text a hit shows: around the match, so a reader
/// scans many hits and opens the one they want by its address.
const EXCERPT: usize = 260;
/// How much of an excerpt comes before the match.
const LEAD: usize = 80;

/// Each char lowercased to one char, so a place in the lowercase is the
/// same place in the text.
fn fold(s: &str) -> Vec<char> {
    s.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect()
}

/// `text` cut to about [`EXCERPT`] characters around the first place
/// `phrase` is found, ignoring case, `…` marking what is cut.
fn excerpt(text: &str, phrase: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= EXCERPT {
        return text.to_string();
    }
    // Text and phrase folded alike, char by char: the match is found where
    // it is, else the excerpt starts at the text's start.
    let lower = fold(text);
    let want = fold(phrase.trim());
    let at = (0..=lower.len().saturating_sub(want.len()))
        .find(|&i| lower[i..].starts_with(&want))
        .unwrap_or(0);
    let start = at.saturating_sub(LEAD);
    let end = (start + EXCERPT).min(chars.len());
    let start = end.saturating_sub(EXCERPT).min(start);
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.extend(&chars[start..end]);
    if end < chars.len() {
        out.push('…');
    }
    out
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
        let shown = |l: &Line| {
            let mut cut = l.clone();
            cut.text = excerpt(&l.text, phrase);
            render_line(tree, &cut)
        };
        for l in &j.window {
            if l.text.to_lowercase().contains(&want) {
                hits.push(Hit {
                    path: path.clone(),
                    found: Found::Window,
                    id: Some(l.id.clone()),
                    at: l.at,
                    shown: shown(l),
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
                        shown: shown(l),
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
                    shown: excerpt(&text, phrase),
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

    /// **A hit shows its text around the match**, cut where it is long.
    #[test]
    fn a_long_hit_is_an_excerpt_around_its_match() {
        assert_eq!(excerpt("short line", "line"), "short line");
        let long = format!(
            "{} the PERMIT was granted {}",
            "a".repeat(300),
            "b".repeat(300)
        );
        let cut = excerpt(&long, "permit");
        assert_eq!(cut.chars().count(), EXCERPT + 2, "both ends marked");
        assert!(cut.starts_with('…') && cut.ends_with('…'));
        assert!(cut.contains("the PERMIT was granted"), "{cut}");
        let head = format!("permit {}", "x".repeat(400));
        let cut = excerpt(&head, "permit");
        assert!(cut.starts_with("permit ") && cut.ends_with('…'));
        // Text whose lowercase is longer than itself is still cut at the
        // match, and so is a phrase that is.
        let odd = format!("{} PERMİT granted {}", "İ".repeat(300), "x".repeat(300));
        let cut = excerpt(&odd, "permİt");
        assert!(cut.contains("PERMİT granted"), "{cut}");
        assert_eq!(cut.chars().count(), EXCERPT + 2);
        // A match near the end keeps a full excerpt, cut at its start only.
        let tail = format!("{} the end PERMIT", "y".repeat(400));
        let cut = excerpt(&tail, "permit");
        assert!(
            cut.starts_with('…') && cut.ends_with("the end PERMIT"),
            "{cut}"
        );
        assert_eq!(cut.chars().count(), EXCERPT + 1);
    }

    /// **A grep hit on a long line shows its excerpt**, with the address
    /// that opens the whole line.
    #[test]
    fn a_long_lines_hit_is_shown_as_its_excerpt() {
        let mut t = sharratt();
        let long = format!(
            "{} the builder starts Monday {}",
            "a".repeat(300),
            "b".repeat(300)
        );
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("topics"),
                text: long,
            },
            now(),
        )
        .unwrap();
        let (hits, _) = grep(&t, &slug("john"), "builder", None, 10);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].shown.contains("…") && hits[0].shown.contains("the builder starts Monday"));
        assert!(hits[0].shown.chars().count() < 400, "{}", hits[0].shown);
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
