//! Grep (§6.5 `journal_grep`): a case-insensitive phrase over every named
//! journal's own name and written cue, and every line, archive and summary
//! the caller may read — their groups, a bot's grants, `@guest` — or one
//! subtree of it. Folded journals are searched too: folding changes what is
//! shown by default, never what is found. References are not followed, so
//! each line is found once, at its real path.

use std::cmp::Reverse;
use std::fmt;

use crate::actor::{need_read, Actor};
use crate::line::{Kind, Line, LineId};
use crate::path::JournalPath;
use crate::period::Period;
use crate::slug::Slug;
use crate::terms::{is_current, replaced};
use crate::time::Stamp;
use crate::tree::{Journal, Tree};
use crate::view::room::{fitting, CURSOR_ROOM};
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

/// Which hits a search keeps: lines of one kind, lines from one speaker, and
/// whatever comes after the last hit of the page before, so a long answer
/// is read page by page, no hit on two pages. A filter on lines finds no
/// journal by its name, and no summary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Filter {
    pub only: Option<Only>,
    pub speaker: Option<Slug>,
    /// The hit the page before ended on.
    pub after: Option<After>,
}

/// A hit a page ended on, as its answer names it: `<path>` for a journal
/// found by its name or cue, `<path>#<line id>` for a line, `<path>#summary`
/// for a summary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum After {
    Journal(JournalPath),
    Line(JournalPath, LineId),
    Summary(JournalPath),
}

/// What a cursor names a summary by, after its journal's `#`.
pub const SUMMARY: &str = "summary";

impl fmt::Display for After {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            After::Journal(p) => write!(f, "{p}"),
            After::Line(p, id) => write!(f, "{p}#{id}"),
            After::Summary(p) => write!(f, "{p}#{SUMMARY}"),
        }
    }
}

impl Hit {
    /// The cursor that pages on from this hit.
    pub fn after(&self) -> After {
        match (&self.found, &self.id) {
            (Found::Journal, _) => After::Journal(self.path.clone()),
            (Found::Summary, _) | (_, None) => After::Summary(self.path.clone()),
            (Found::Window | Found::Archive(_), Some(id)) => {
                After::Line(self.path.clone(), id.clone())
            }
        }
    }
}

/// Where a hit sorts: journals the phrase names first, by path, then what
/// was written, newest first.
type Place<'a> = (bool, Reverse<Stamp>, &'a JournalPath, Option<&'a LineId>);

fn place(h: &Hit) -> Place<'_> {
    (
        h.found != Found::Journal,
        Reverse(h.at),
        &h.path,
        h.id.as_ref(),
    )
}

/// Where `after` sorts, read from the tree: a line or a summary sorts by
/// when it was written, so the page goes on from its place even when it no
/// longer matches, is hidden, or was filtered out. `None` when it is gone.
fn place_of<'a>(tree: &Tree, after: &'a After) -> Option<Place<'a>> {
    Some(match after {
        After::Journal(p) => (false, Reverse(Stamp::MIN), p, None),
        After::Line(p, id) => (true, Reverse(tree.get(p)?.line(id)?.at), p, Some(id)),
        After::Summary(p) => (true, Reverse(tree.get(p)?.summary()?.built), p, None),
    })
}

/// The kind of line a search keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Only {
    Notes,
    Folds,
    /// Notes a worker proposed, not yet accepted.
    Proposals,
}

impl Only {
    pub fn parse(s: &str) -> Result<Only, String> {
        match s {
            "note" => Ok(Only::Notes),
            "fold" => Ok(Only::Folds),
            "proposal" => Ok(Only::Proposals),
            _ => Err(format!("`kind` is `note`, `fold` or `proposal`, not `{s}`")),
        }
    }

    fn keeps(self, l: &Line) -> bool {
        match self {
            Only::Notes => l.kind == Kind::Note && !l.proposed,
            Only::Folds => l.kind == Kind::Fold,
            Only::Proposals => l.proposed,
        }
    }
}

impl Filter {
    /// Whether it narrows to lines alone.
    fn lines_only(&self) -> bool {
        self.only.is_some() || self.speaker.is_some()
    }

    fn keeps(&self, l: &Line) -> bool {
        self.only.is_none_or(|o| o.keeps(l))
            && self
                .speaker
                .as_ref()
                .is_none_or(|s| l.speaker.as_ref() == Some(s) || l.account.as_ref() == Some(s))
    }
}

/// Every hit for `phrase` that `actor` may read, under `under` when given —
/// the journals it names first, then lines and summaries newest first — and
/// the count before `limit` cut it.
pub fn grep(
    now: Stamp,
    tree: &Tree,
    actor: &Actor,
    phrase: &str,
    under: Option<&JournalPath>,
    limit: usize,
) -> (Vec<Hit>, usize) {
    let mut hits = found(now, tree, actor, phrase, under, &Filter::default());
    let total = hits.len();
    hits.truncate(limit);
    (hits, total)
}

/// [`grep`], keeping only the hits `filter` keeps; or, when the hit its
/// `after` names is gone from the tree, why the page cannot go on.
pub fn grep_by(
    now: Stamp,
    tree: &Tree,
    actor: &Actor,
    phrase: &str,
    under: Option<&JournalPath>,
    filter: &Filter,
    limit: usize,
) -> Result<(Vec<Hit>, usize), String> {
    let mut hits = found(now, tree, actor, phrase, under, filter);
    // The page after the hit the last one ended on, from that hit's place.
    if let Some(after) = &filter.after {
        let Some(from) = place_of(tree, after) else {
            return Err(format!(
                "{after} is gone from the memory, so the search cannot go on after it: \
                 search again without `after`"
            ));
        };
        hits.retain(|h| place(h) > from);
    }
    let total = hits.len();
    hits.truncate(limit);
    Ok((hits, total))
}

/// Every hit `filter` keeps, sorted by [`place`].
fn found(
    now: Stamp,
    tree: &Tree,
    actor: &Actor,
    phrase: &str,
    under: Option<&JournalPath>,
    filter: &Filter,
) -> Vec<Hit> {
    let want = wanted(phrase);
    if want.is_empty() {
        return Vec::new();
    }
    let lines_only = filter.lines_only();
    let mut hits = Vec::new();
    for (path, j) in tree.journals() {
        if need_read(tree, actor, &path).is_err() || under.is_some_and(|u| !u.holds(&path)) {
            continue;
        }
        if !j.problems.is_empty() {
            continue;
        }
        if !lines_only && names(&path, j, &want) {
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
        // A line is hidden wherever it is kept: an archived one past its
        // expiry, or replaced by a line still in the window, is found by no
        // search.
        let replaced = replaced(j.all_lines());
        let visible = |l: &Line| !replaced.contains(&l.id) && is_current(l, now);
        for l in j.window.iter().filter(|l| visible(l)) {
            if filter.keeps(l) && holds(&l.text, &want) {
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
            for l in lines.iter().filter(|l| visible(l)) {
                if filter.keeps(l) && holds(&l.text, &want) {
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
        if let Some(s) = j.summary().filter(|_| !lines_only) {
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
    hits.sort_by(|a, b| place(a).cmp(&place(b)));
    hits
}

/// Hits as a reader sees them, in an answer of at most `fit` characters: the
/// first of them that fit, then how many more there are and the cursor that
/// pages on from the last one shown.
pub fn render(hits: &[Hit], total: usize, fit: usize) -> String {
    if hits.is_empty() {
        return "Nothing found.".into();
    }
    let lines: Vec<String> = hits
        .iter()
        .map(|h| match &h.found {
            Found::Journal => format!("{} (journal) {}", h.path, h.shown),
            Found::Window => format!("{} {}", h.path, h.shown),
            Found::Archive(p) => format!("{} (archive {p}) {}", h.path, h.shown),
            Found::Summary => format!("{} (summary) {}", h.path, h.shown),
        })
        .collect();
    let shown = fitting(&lines, fit.saturating_sub(CURSOR_ROOM));
    let mut out = lines[..shown].to_vec();
    if total > shown {
        out.push(format!(
            "… and {} more; narrow the phrase, search under one journal, or page on with \
             `after: {}`",
            total - shown,
            hits[shown - 1].after()
        ));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ops::fixture::{bot, guest, john, now, p, stranger, yen, zen};
    use crate::ops::fold::fold_child;
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Section;
    use crate::terms::Terms;
    use crate::testing::{note, slug};

    fn noted(t: &mut Tree, actor: &Actor, path: &str, text: &str, at: Stamp) {
        apply(
            t,
            actor,
            &Op::Note {
                path: p(path),
                text: text.into(),
                speaker: None,
                terms: Terms::default(),
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
            Arc::new(vec![note("2026-09-01T10:00:00Z", "An old PERMIT.")]),
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
        let (hits, _) = grep(now(), &t, &john(), "builder", None, 10);
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
        let (hits, total) = grep(now(), &t, &john(), "CPU experts", None, 10);
        assert_eq!(total, 2);
        assert_eq!(
            render(&hits, total, usize::MAX),
            "@zen/topics/cpu-experts (journal) experts computed by the host pool\n\
             @zen/topics #2026-10-30-1202-b082 2026-10-30 12:02 note, john via Claude Code: \
             The cpu experts pool is on by default."
        );
        let (by_cue, _) = grep(now(), &t, &john(), "host pool", None, 10);
        assert_eq!(by_cue[0].found, Found::Journal);
        // A word naming its group, its section or a fixed journal's cue
        // finds no journal: only a named journal's own name and cue count.
        let journals = |phrase| {
            grep(now(), &t, &john(), phrase, None, 50)
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
        let (hits, total) = grep(now(), &t, &john(), "wasix", None, 10);
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
            assert_eq!(grep(now(), &t, &john(), phrase, None, 10).1, 1, "{phrase}");
        }
        assert_eq!(
            grep(now(), &t, &john(), "environmentvariable", None, 10).1,
            0
        );
        for empty in ["-", " - ", "--"] {
            assert_eq!(grep(now(), &t, &john(), empty, None, 10).1, 0, "{empty:?}");
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
            grep(now(), &t, actor, "permit", None, 50)
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
        let (hits, total) = grep(now(), &t, &john(), "permit", Some(&p("topics")), 1);
        assert_eq!((hits.len(), total), (1, 2));
        let text = render(&hits, total, usize::MAX);
        assert!(
            text.ends_with(&format!(
                "… and 1 more; narrow the phrase, search under one journal, or page on with \
                 `after: @zen/topics#{}`",
                hits[0].id.as_ref().unwrap()
            )),
            "{text}"
        );
        assert_eq!(render(&[], 0, usize::MAX), "Nothing found.");
        assert_eq!(grep(now(), &t, &john(), "  ", None, 5).1, 0);
    }

    /// **A filter keeps lines alone — of a kind, from a speaker, before a
    /// moment — and pages newest first**: each page's last hit is where the
    /// next page begins, and no hit is on two pages.
    #[test]
    fn a_filter_keeps_lines_and_pages_newest_first() {
        let t = seeded();
        let lines = |filter: &Filter, limit| {
            grep_by(now(), &t, &john(), "permit", None, filter, limit).unwrap()
        };
        let all = lines(&Filter::default(), 50).0;
        let notes = Filter {
            only: Some(Only::Notes),
            ..Filter::default()
        };
        let (every, total) = lines(&notes, 50);
        assert!(!every.is_empty());
        assert!(every
            .iter()
            .all(|h| h.found != Found::Journal && h.found != Found::Summary));
        assert!(every.len() <= all.len());
        let first = lines(&notes, 1).0;
        let after = first[0].after();
        let rest = lines(
            &Filter {
                after: Some(after),
                ..notes.clone()
            },
            50,
        );
        assert_eq!(
            first.len() + rest.1,
            total,
            "the pages hold every line once"
        );
        assert!(!rest.0.contains(&first[0]));
        assert_eq!(
            &every[1..],
            rest.0.as_slice(),
            "the next page goes on in order"
        );
        let yen = Filter {
            speaker: Some(slug("yen")),
            ..notes
        };
        assert!(lines(&yen, 50)
            .0
            .iter()
            .all(|h| h.shown.contains("yen") || h.shown.contains("Yen")));
        assert_eq!(
            Only::parse("episode").unwrap_err(),
            "`kind` is `note`, `fold` or `proposal`, not `episode`"
        );
    }

    /// **A page goes on from its cursor's place even when the cursor's line
    /// no longer matches**: the next page is what came after it, never the
    /// first page again. A cursor whose line is gone says so.
    #[test]
    fn a_cursor_that_no_longer_matches_still_pages_on() {
        let mut t = seeded();
        let (all, total) = grep(now(), &t, &john(), "permit", None, 50);
        let after = all[1].after();
        let After::Line(path, id) = &after else {
            panic!("{after:?}")
        };
        t.get_mut(path).unwrap().line_mut(id).unwrap().text = "Edited away.".into();
        let filter = Filter {
            after: Some(after.clone()),
            ..Filter::default()
        };
        let (rest, left) = grep_by(now(), &t, &john(), "permit", None, &filter, 50).unwrap();
        assert_eq!(rest.as_slice(), &all[2..]);
        assert_eq!(left, total - 2);
        let gone = Filter {
            after: Some(After::Line(
                path.clone(),
                LineId::parse("2026-01-01-0000-0000").unwrap(),
            )),
            ..Filter::default()
        };
        assert!(grep_by(now(), &t, &john(), "permit", None, &gone, 50)
            .unwrap_err()
            .contains("search again without `after`"));
    }

    /// **A page that ends on a journal or a summary names a cursor too**, and
    /// the next page goes on after it.
    #[test]
    fn a_page_ending_on_a_journal_pages_on() {
        let mut t = seeded();
        apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("permits"),
                cue: "open for council permits".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        let (first, total) = grep(now(), &t, &john(), "permit", None, 1);
        assert_eq!(first[0].found, Found::Journal);
        let text = render(&first, total, usize::MAX);
        assert!(
            text.ends_with("page on with `after: @zen/topics/permits`"),
            "{text}"
        );
        let filter = Filter {
            after: Some(first[0].after()),
            ..Filter::default()
        };
        let (rest, left) = grep_by(now(), &t, &john(), "permit", None, &filter, 50).unwrap();
        assert_eq!(left, total - 1);
        assert!(rest.iter().all(|h| h.found != Found::Journal));
        assert_eq!(
            After::Summary(p("topics")).to_string(),
            "@zen/topics#summary"
        );
    }

    /// **An archived line is hidden as a window line is**: past its expiry,
    /// or replaced, no search finds it.
    #[test]
    fn a_hidden_archived_line_is_found_by_no_search() {
        let mut t = zen();
        let mut expired = note("2026-09-01T10:00:00Z", "the old permit reminder");
        expired.terms.expires = Some(Stamp::parse("2026-09-02T00:00:00Z").unwrap());
        let superseded = note("2026-09-01T11:00:00Z", "the permit is version 2");
        let kept = note("2026-09-01T12:00:00Z", "the archived permit fact");
        let mut replacing = note("2026-10-01T12:00:00Z", "the permit is version 3");
        replacing.terms.replaces = Some(superseded.id.clone());
        let j = t.get_mut(&p("topics")).unwrap();
        j.archives.insert(
            Period::parse("2026-09-01").unwrap(),
            Arc::new(vec![expired, superseded, kept]),
        );
        j.window.push(replacing);
        let texts: Vec<String> = grep(now(), &t, &john(), "permit", None, 50)
            .0
            .into_iter()
            .map(|h| h.shown)
            .collect();
        assert_eq!(texts.len(), 2, "{texts:?}");
        assert!(texts[0].ends_with("the permit is version 3"));
        assert!(texts[1].ends_with("the archived permit fact"));
    }

    /// **A page that would not fit its answer shows the hits that do**, and
    /// the cursor pages on from the last one shown.
    #[test]
    fn a_page_is_cut_to_its_answer_and_keeps_its_cursor() {
        let mut t = zen();
        for i in 0..10 {
            noted(
                &mut t,
                &john(),
                "topics",
                &format!("permit {i} {}", "x".repeat(900)),
                now().plus(60 * i),
            );
        }
        let (hits, total) = grep(now(), &t, &john(), "permit", None, 10);
        let text = render(&hits, total, 4_000);
        assert!(text.chars().count() <= 4_000, "{}", text.chars().count());
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");
        assert!(lines[3].ends_with(&format!(
            "… and 7 more; narrow the phrase, search under one journal, or page on with \
             `after: @zen/topics#{}`",
            hits[2].id.as_ref().unwrap()
        )));
    }
}
