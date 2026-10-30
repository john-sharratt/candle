//! Grep (§6.5 `journal_grep`): a case-insensitive phrase over every named
//! journal's own name and written cue, and every line, archive and summary
//! the caller may read — their groups, a bot's grants, `@guest` — or one
//! subtree of it. Folded journals are searched too: folding changes what is
//! shown by default, never what is found. References are not followed, so
//! each line is found once, at its real path.

use crate::actor::{need_read, Actor};
use crate::line::{Line, LineId};
use crate::path::JournalPath;
use crate::period::Period;
use crate::time::Stamp;
use crate::tree::{Journal, Tree};
use crate::view::who::render_line;

/// Where a hit is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Found {
    /// The journal itself, by its name or its cue.
    Journal,
    Window,
    Archive(Period),
    Summary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub path: JournalPath,
    pub found: Found,
    pub id: Option<LineId>,
    /// When it was written; [`Stamp::MIN`] for a journal found by its name
    /// or cue.
    pub at: Stamp,
    /// The line as a reader sees it, the summary's text, or the journal's
    /// cue.
    pub shown: String,
}

/// The most of a line's text a hit shows. Nearly every line is shorter and
/// is shown whole, so a hit answers without a second read — a dense line of
/// definitions, or a rule with its reason, runs to 900 characters, and the
/// clause a search is after is often its last; a longer one is cut around
/// the match, and its address opens the rest.
const EXCERPT: usize = 1_000;
/// How much of an excerpt comes before the match.
const LEAD: usize = 250;

/// A named journal whose own name or written cue holds `want`, a folded
/// phrase. Only its last segment counts, and never a fixed journal's cue,
/// which every folder shares: a phrase naming a parent, an owner or a
/// section finds no journal beneath it.
fn names(path: &JournalPath, j: &Journal, want: &[char]) -> bool {
    if j.is_fixed() {
        return false;
    }
    path.last().is_some_and(|s| holds(s.as_str(), want))
        || j.cue().is_some_and(|cue| holds(&cue, want))
}

/// Each char lowercased to one char, and a hyphen read as a space, so a
/// place in the fold is the same place in the text, and "cpu experts" finds
/// `cpu-experts` and "environment variable" finds "environment-variable".
fn fold(s: &str) -> Vec<char> {
    s.chars()
        .map(|c| match c {
            '-' => ' ',
            c => c.to_lowercase().next().unwrap_or(c),
        })
        .collect()
}

/// What `phrase` asks for: folded, then trimmed, so a phrase of hyphens is
/// as empty as one of spaces.
fn wanted(phrase: &str) -> Vec<char> {
    let folded = fold(phrase);
    let first = folded.iter().position(|c| !c.is_whitespace());
    let last = folded.iter().rposition(|c| !c.is_whitespace());
    match (first, last) {
        (Some(a), Some(b)) => folded[a..=b].to_vec(),
        _ => Vec::new(),
    }
}

/// `text`, folded, holds `want`, a folded phrase that is not empty.
fn holds(text: &str, want: &[char]) -> bool {
    fold(text).windows(want.len()).any(|w| w == want)
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
    let want = wanted(phrase);
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

/// Every hit for `phrase` that `actor` may read, under `under` when given —
/// the journals it names first, then lines and summaries newest first — and
/// the count before `limit` cut it.
pub fn grep(
    tree: &Tree,
    actor: &Actor,
    phrase: &str,
    under: Option<&JournalPath>,
    limit: usize,
) -> (Vec<Hit>, usize) {
    let want = wanted(phrase);
    if want.is_empty() {
        return (Vec::new(), 0);
    }
    let mut hits = Vec::new();
    for (path, j) in tree.journals() {
        if need_read(tree, actor, &path).is_err() || under.is_some_and(|u| !u.holds(&path)) {
            continue;
        }
        if !j.problems.is_empty() {
            continue;
        }
        if names(&path, j, &want) {
            hits.push(Hit {
                path: path.clone(),
                found: Found::Journal,
                id: None,
                at: Stamp::MIN,
                shown: j.cue().unwrap_or_default(),
            });
        }
        let shown = |l: &Line| {
            let mut cut = l.clone();
            cut.text = excerpt(&l.text, phrase);
            render_line(tree, &cut)
        };
        for l in &j.window {
            if holds(&l.text, &want) {
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
                if holds(&l.text, &want) {
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
            if holds(&text, &want) || s.digest.as_deref().is_some_and(|d| holds(d, &want)) {
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
    // Journals the phrase names come first, by path, then what was written,
    // newest first.
    let line = |h: &Hit| h.found != Found::Journal;
    hits.sort_by(|a, b| {
        line(a)
            .cmp(&line(b))
            .then_with(|| b.at.cmp(&a.at))
            .then_with(|| a.path.cmp(&b.path))
    });
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
            Found::Journal => format!("{} (journal) {}", h.path, h.shown),
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
    use crate::ops::fixture::{bot, guest, john, now, p, stranger, yen, zen};
    use crate::ops::fold::fold_child;
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Section;
    use crate::testing::{note, slug};

    fn noted(t: &mut Tree, actor: &Actor, path: &str, text: &str, at: Stamp) {
        apply(
            t,
            actor,
            &Op::Note {
                path: p(path),
                text: text.into(),
                speaker: None,
            },
            at,
        )
        .unwrap();
    }

    fn seeded() -> Tree {
        let mut t = zen();
        noted(&mut t, &john(), "people", "John's permit worry.", now());
        noted(&mut t, &yen(), "@zen", "Yen's permit note.", now());
        noted(
            &mut t,
            &yen(),
            "topics",
            "Permit lodged with council.",
            now().plus(60),
        );
        noted(
            &mut t,
            &john(),
            "@guest",
            "A public permit page.",
            now().plus(30),
        );
        let old = apply(
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
        let j = t.get_mut(&old).unwrap();
        j.archives.insert(
            Period::parse("2026-09-01").unwrap(),
            vec![note("2026-09-01T10:00:00Z", "An old PERMIT.")],
        );
        let topics = p("topics");
        fold_child(t.get_mut(&topics).unwrap(), &topics, &slug("old"), now()).unwrap();
        t
    }

    /// **A hit shows its text around the match**, cut where it is long.
    #[test]
    fn a_long_hit_is_an_excerpt_around_its_match() {
        assert_eq!(excerpt("short line", "line"), "short line");
        let whole = format!("{} permit", "a".repeat(EXCERPT - 7));
        assert_eq!(
            excerpt(&whole, "permit"),
            whole,
            "a line that fits is whole"
        );
        let long = format!(
            "{} the PERMIT was granted {}",
            "a".repeat(800),
            "b".repeat(800)
        );
        let cut = excerpt(&long, "permit");
        assert_eq!(cut.chars().count(), EXCERPT + 2, "both ends marked");
        assert!(cut.starts_with('…') && cut.ends_with('…'));
        assert!(cut.contains("the PERMIT was granted"), "{cut}");
        let head = format!("permit {}", "x".repeat(1_200));
        let cut = excerpt(&head, "permit");
        assert!(cut.starts_with("permit ") && cut.ends_with('…'));
        // Text whose lowercase is longer than itself is still cut at the
        // match, and so is a phrase that is.
        let odd = format!("{} PERMİT granted {}", "İ".repeat(1_200), "x".repeat(1_200));
        let cut = excerpt(&odd, "permİt");
        assert!(cut.contains("PERMİT granted"), "{cut}");
        assert_eq!(cut.chars().count(), EXCERPT + 2);
        // A match near the end keeps a full excerpt, cut at its start only.
        let tail = format!("{} the end PERMIT", "y".repeat(1_200));
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
        let mut t = zen();
        let long = format!(
            "{} the builder starts Monday {}",
            "a".repeat(800),
            "b".repeat(800)
        );
        noted(&mut t, &john(), "topics", &long, now());
        let (hits, _) = grep(&t, &john(), "builder", None, 10);
        assert_eq!(hits.len(), 1);
        let cut = excerpt(&long, "builder");
        assert_eq!(cut.chars().count(), EXCERPT + 2, "both ends marked");
        assert!(cut.contains("the builder starts Monday"), "{cut}");
        assert!(hits[0].shown.ends_with(&cut), "{}", hits[0].shown);
    }

    /// **A journal is found by its name or its cue**, ahead of every line,
    /// its name's hyphens read as spaces.
    #[test]
    fn a_journal_is_found_by_its_name_and_its_cue() {
        let mut t = seeded();
        apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("cpu-experts"),
                cue: "experts computed by the host pool".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        noted(
            &mut t,
            &john(),
            "topics",
            "The cpu experts pool is on by default.",
            now().plus(120),
        );
        let (hits, total) = grep(&t, &john(), "CPU experts", None, 10);
        assert_eq!(total, 2);
        assert_eq!(
            render(&hits, total),
            "@zen/topics/cpu-experts (journal) experts computed by the host pool\n\
             @zen/topics #2026-10-30-1202-b082 2026-10-30 12:02 note, john via Claude Code: \
             The cpu experts pool is on by default."
        );
        let (by_cue, _) = grep(&t, &john(), "host pool", None, 10);
        assert_eq!(by_cue[0].found, Found::Journal);
        // A word naming its group, its section or a fixed journal's cue
        // finds no journal: only a named journal's own name and cue count.
        let journals = |phrase| {
            grep(&t, &john(), phrase, None, 50)
                .0
                .into_iter()
                .filter(|h| h.found == Found::Journal)
                .count()
        };
        assert_eq!(journals("zen"), 0);
        assert_eq!(journals("topics"), 0);
        assert_eq!(journals(Section::Topics.cue()), 0);
    }

    /// **A reference is not walked**: the journal it leads to is found once,
    /// at its real path.
    #[test]
    fn a_reference_is_not_walked() {
        let mut t = zen();
        for name in ["candle", "wasmer"] {
            apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: p("topics"),
                    name: slug(name),
                    cue: format!("open for {name}"),
                    on_taken: OnTaken::Refuse,
                },
                now(),
            )
            .unwrap();
        }
        apply(
            &mut t,
            &john(),
            &Op::Link {
                under: p("topics/candle"),
                name: slug("wasmer"),
                to: p("topics/wasmer"),
                cue: "open for the work it grew from".into(),
            },
            now(),
        )
        .unwrap();
        noted(&mut t, &john(), "topics/wasmer", "WASIX came first.", now());
        let (hits, total) = grep(&t, &john(), "wasix", None, 10);
        assert_eq!(total, 1);
        assert_eq!(hits[0].path, p("topics/wasmer"));
    }

    /// **A hyphen and a space match each other**, either way round.
    #[test]
    fn a_hyphen_matches_a_space() {
        let mut t = zen();
        noted(
            &mut t,
            &john(),
            "topics",
            "No environment-variable feature flags.",
            now(),
        );
        for phrase in [
            "environment variable",
            "ENVIRONMENT-VARIABLE",
            "feature-flags",
        ] {
            assert_eq!(grep(&t, &john(), phrase, None, 10).1, 1, "{phrase}");
        }
        assert_eq!(grep(&t, &john(), "environmentvariable", None, 10).1, 0);
        for empty in ["-", " - ", "--"] {
            assert_eq!(grep(&t, &john(), empty, None, 10).1, 0, "{empty:?}");
        }
        assert_eq!(
            excerpt(
                &format!("{} a-b {}", "x".repeat(800), "y".repeat(800)),
                "A B"
            ),
            format!(
                "…{} a-b {}…",
                "x".repeat(LEAD - 1),
                "y".repeat(EXCERPT - LEAD - 4)
            )
        );
    }

    /// **Grep finds what the caller reads and nothing else**: a member all
    /// of their groups, a guest `@guest`, a bot its grants, a stranger
    /// nothing.
    #[test]
    fn grep_finds_what_the_caller_reads_and_nothing_else() {
        let t = seeded();
        let shown = |actor: &Actor| -> Vec<String> {
            grep(&t, actor, "permit", None, 50)
                .0
                .iter()
                .map(|h| format!("{} {:?}", h.path, h.found))
                .collect()
        };
        assert_eq!(
            shown(&yen()),
            [
                "@zen/topics Window",
                "@zen Window",
                "@zen/people Window",
                "@zen/topics/old Archive(Day { year: 2026, month: 9, day: 1 })"
            ],
            "Yen reads John's lines too: every member reads the group"
        );
        assert_eq!(shown(&john()).len(), 5, "and @guest, which John owns");
        assert_eq!(shown(&guest()), ["@guest Window"]);
        assert!(
            shown(&bot()).is_empty(),
            "its grant, topics/game, holds no permit"
        );
        assert!(shown(&stranger()).is_empty());
    }

    #[test]
    fn grep_narrows_to_a_subtree_and_cuts_at_its_limit() {
        let t = seeded();
        let (hits, total) = grep(&t, &john(), "permit", Some(&p("topics")), 1);
        assert_eq!((hits.len(), total), (1, 2));
        let text = render(&hits, total);
        assert!(
            text.ends_with("… and 1 more; narrow the phrase or search under one journal"),
            "{text}"
        );
        assert_eq!(render(&[], 0), "Nothing found.");
        assert_eq!(grep(&t, &john(), "  ", None, 5).1, 0);
    }
}
