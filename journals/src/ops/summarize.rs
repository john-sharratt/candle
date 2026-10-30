//! A rebuilt summary (§5), kept only if it was built over the journal as it
//! stands, cites only what it was built from, says as many sentences as its
//! inputs ask for, fits its cap, and did not lose half of itself with nothing
//! gone to explain it.

use journal_core::tools::half_floor;

use crate::error::{Error, Result};
use crate::files::summary::{Cite, Fingerprint, Sentence, Summary};
use crate::files::Meta;
use crate::ops::create::clean_cue;
use crate::ops::Applied;
use crate::path::{JournalPath, Place};
use crate::scope::Section;
use crate::time::Stamp;
use crate::tree::check::{digest_cap, summary_cap};
use crate::tree::fingerprint::{fingerprint, has_inputs, speaks};
use crate::tree::{Journal, Tree};

/// The name a journal's `journal.yaml` says: a master's its group's, a
/// section's its own, a named journal's its folder's.
pub(crate) fn fixed_name(path: &JournalPath, j: &Journal) -> String {
    match j.place {
        Place::Root => path.group().to_string(),
        Place::Section(s) => s.name().into(),
        Place::Named { .. } => path.last().map(|s| s.to_string()).unwrap_or_default(),
    }
}

/// The most sentences a summary's floor asks for, however many its inputs.
pub const SENTENCE_FLOOR_MAX: usize = 3;

/// `j`'s live children with something to say.
fn speaking(j: &Journal) -> usize {
    j.live().filter(|(_, c)| speaks(c)).count()
}

/// Whether `c` names one of `j`'s inputs: a window line, or a live child
/// with something to say.
fn is_input(j: &Journal, c: &Cite) -> bool {
    match c {
        Cite::Line(id) => j.has_id(id),
        Cite::Child(name) => j.live().any(|(n, c)| n == name && speaks(c)),
    }
}

/// The fewest sentences a summary of `j` holds; none when it has no inputs.
/// A conversation is one exchange and says it in one. Any other journal
/// holds one for every two inputs — window lines and children with
/// something to say — since a sentence cites at most two, up to
/// [`SENTENCE_FLOOR_MAX`]: a rebuild asked for fewer closes its list after
/// one subject.
pub fn sentence_floor(j: &Journal) -> usize {
    let inputs = j.window.len() + speaking(j);
    let conversation = matches!(
        j.place,
        Place::Named {
            section: Section::Conversations,
            ..
        }
    );
    match inputs {
        0 => 0,
        _ if conversation => 1,
        n => half_floor(n, SENTENCE_FLOOR_MAX),
    }
}

/// How long `j`'s summary is and the fewest characters a rebuild may come
/// to — half of it — while nothing it was built over has gone: no fewer
/// lines or children, and every input it cites still there. `None` when a
/// rebuild may be any length.
pub fn shortest(j: &Journal) -> Option<(usize, usize)> {
    let old = j.summary()?;
    let nothing_gone = j.window.len() >= old.lines
        && speaking(j) >= old.children
        && old
            .sentences
            .iter()
            .flat_map(|s| &s.cites)
            .all(|c| is_input(j, c));
    let len = old.len();
    (nothing_gone && len > 0).then(|| (len, len.div_ceil(2)))
}

fn check_sentences(
    j: &Journal,
    path: &JournalPath,
    sentences: &[Sentence],
) -> Result<Vec<Sentence>> {
    let mut out = Vec::with_capacity(sentences.len());
    for (i, s) in sentences.iter().enumerate() {
        let text = s.text.trim();
        if text.is_empty() || text.contains('\n') {
            return Err(Error::invalid(format!(
                "sentence {} is not one line of text",
                i + 1
            )));
        }
        if s.cites.is_empty() {
            return Err(Error::invalid(format!(
                "sentence {} cites nothing: every sentence cites the lines or children it rests on",
                i + 1
            )));
        }
        for c in &s.cites {
            if !is_input(j, c) {
                return Err(Error::invalid(format!(
                    "sentence {} cites {c}, which is not one of {path}'s inputs",
                    i + 1
                )));
            }
        }
        out.push(Sentence {
            text: text.to_string(),
            cites: s.cites.clone(),
        });
    }
    if has_inputs(j) && out.is_empty() {
        return Err(Error::invalid(format!(
            "{path} has lines or children: its summary says something"
        )));
    }
    let floor = sentence_floor(j);
    if out.len() < floor {
        return Err(Error::invalid(format!(
            "{path}'s inputs ask for at least {floor} sentences in its summary, not {}",
            out.len()
        )));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn summarize(
    tree: &mut Tree,
    path: &JournalPath,
    over: &Fingerprint,
    sentences: &[Sentence],
    digest: Option<&str>,
    cue: Option<&str>,
    now: Stamp,
) -> Result<Applied> {
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    let current = fingerprint(j);
    if &current != over {
        return Err(Error::conflict(format!(
            "{path} changed while its summary was built; it is still dirty and is rebuilt next pass"
        )));
    }
    // Two rebuilds over the same journal: the first kept wins, and the
    // second — by another agent, replayed after it — is dropped.
    if j.summary().is_some_and(|s| s.over == current) {
        return Err(Error::conflict(format!(
            "{path}'s summary is current already"
        )));
    }
    let sentences = check_sentences(j, path, sentences)?;
    let summary = Summary {
        over: current,
        built: now,
        lines: j.window.len(),
        children: speaking(j),
        stale: false,
        digest: match (digest.map(str::trim), digest_cap(j.place)) {
            (None, _) => None,
            (Some(""), _) => return Err(Error::invalid("the digest is empty")),
            (Some(_), None) => {
                return Err(Error::invalid(format!(
                    "{path} has no voice digest: only the master and a person's own journal do"
                )))
            }
            (Some(d), Some(cap)) if d.chars().count() > cap => {
                return Err(Error::refused(format!(
                    "the digest is {} characters, over its cap of {cap}",
                    d.chars().count()
                )))
            }
            (Some(d), Some(_)) => Some(d.to_string()),
        },
        sentences,
    };
    let cap = summary_cap(j.place);
    if summary.len() > cap {
        return Err(Error::refused(format!(
            "the summary is {} characters, over {path}'s cap of {cap}",
            summary.len()
        )));
    }
    if let Some((was, least)) = shortest(j) {
        if summary.len() < least {
            return Err(Error::refused(format!(
                "the summary shrinks from {was} to {} characters with nothing gone from {path}",
                summary.len()
            )));
        }
    }
    let cue = match cue {
        Some(_) if j.is_fixed() => return Err(Error::invalid(format!("{path}'s cue is fixed"))),
        Some(c) => Some(clean_cue(c)?),
        None => None,
    };
    let name = fixed_name(path, j);
    let meta = j.meta.get_or_insert_with(|| Meta::new(name, None, now));
    if let Some(c) = cue {
        meta.cue = Some(c);
    }
    meta.summary = Some(summary);
    Ok(Applied::touching(
        format!("Summary of {path}"),
        vec![path.clone()],
    ))
}

/// The most of a refusal a stale summary's commit subject keeps.
const WHY_CHARS: usize = 240;

/// Marks `path`'s summary stale, its commit naming `why` — the refusal that
/// left it standing — on one line of at most [`WHY_CHARS`].
pub fn mark_stale(tree: &mut Tree, path: &JournalPath, why: &str) -> Result<Applied> {
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    match j.meta.as_mut().and_then(|m| m.summary.as_mut()) {
        Some(s) if s.stale => Err(Error::refused(format!("{path}'s summary is stale already"))),
        Some(s) => {
            s.stale = true;
            let why = why.split_whitespace().collect::<Vec<_>>().join(" ");
            let mut subject = format!("Stale summary of {path}");
            if !why.is_empty() {
                subject.push_str(": ");
                subject.extend(why.chars().take(WHY_CHARS));
            }
            Ok(Applied::touching(subject, vec![path.clone()]))
        }
        None => Err(Error::refused(format!(
            "{path} has no summary to mark stale"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::line::LineId;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, OnTaken, Op, Which};
    use crate::scope::Channel;
    use crate::testing::slug;
    use crate::tree::fingerprint::is_dirty;

    fn s(text: &str, cites: Vec<Cite>) -> Sentence {
        Sentence {
            text: text.into(),
            cites,
        }
    }

    fn noted(t: &mut Tree, path: &str, text: &str) -> (JournalPath, Cite) {
        let path = p(path);
        let id = apply(
            t,
            &john(),
            &Op::Note {
                path: path.clone(),
                text: text.into(),
                speaker: None,
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        (path, Cite::Line(id))
    }

    fn summarise(
        t: &mut Tree,
        path: &JournalPath,
        sentences: Vec<Sentence>,
        digest: Option<&str>,
    ) -> Result<Applied> {
        let over = fingerprint(t.get(path).unwrap());
        apply(
            t,
            &Actor::Zend,
            &Op::Summarize {
                path: path.clone(),
                over,
                sentences,
                digest: digest.map(Into::into),
                cue: None,
            },
            now(),
        )
    }

    #[test]
    fn a_summary_built_over_the_journal_as_it_stands_is_kept() {
        let mut t = zen();
        let (path, cite) = noted(&mut t, "@zen", "Zen is the household's memory.");
        assert!(is_dirty(t.get(&path).unwrap()));
        let a = summarise(
            &mut t,
            &path,
            vec![s("Zen remembers for the household.", vec![cite])],
            Some("The household's memory."),
        )
        .unwrap();
        assert_eq!(a.subject, "Summary of @zen");
        let j = t.get(&path).unwrap();
        assert!(!is_dirty(j));
        let meta = j.meta.as_ref().unwrap();
        assert_eq!(
            meta.name, "zen",
            "a fixed journal's file is made with its fixed name"
        );
        let sum = meta.summary.as_ref().unwrap();
        // Its five fixed sections are live children, but say nothing yet.
        assert_eq!((sum.lines, sum.children, sum.built), (1, 0, now()));
        assert_eq!(sum.digest.as_deref(), Some("The household's memory."));
    }

    /// **A child gone silent is a child gone**: once its lines are forgotten
    /// and its summary emptied, the parent that rested on it alone is emptied
    /// too, not refused as a shrink with nothing gone.
    #[test]
    fn a_parent_over_a_child_gone_silent_is_emptied() {
        let mut t = zen();
        let trip = apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("trip"),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let (_, cite) = noted(&mut t, "topics/trip", "Flights to Da Nang.");
        summarise(&mut t, &trip, vec![s("Flights booked.", vec![cite])], None).unwrap();
        let topics = p("topics");
        summarise(
            &mut t,
            &topics,
            vec![s(
                "A trip to Da Nang is planned.",
                vec![Cite::Child(slug("trip"))],
            )],
            None,
        )
        .unwrap();
        assert_eq!(t.get(&topics).unwrap().summary().unwrap().children, 1);
        apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: trip.clone(),
                which: Which::Says("Flights to Da Nang".into()),
            },
            now(),
        )
        .unwrap();
        summarise(&mut t, &trip, vec![], None).unwrap();
        summarise(&mut t, &topics, vec![], None).unwrap();
        let j = t.get(&topics).unwrap();
        assert!(!is_dirty(j) && j.summary().unwrap().is_empty());
    }

    #[test]
    fn a_summary_built_over_what_has_changed_is_a_conflict() {
        let mut t = zen();
        let (path, cite) = noted(&mut t, "people", "Prefers tea.");
        let over = fingerprint(t.get(&path).unwrap());
        noted(&mut t, "people", "Now coffee.");
        let op = Op::Summarize {
            path: path.clone(),
            over,
            sentences: vec![s("Tea.", vec![cite])],
            digest: None,
            cue: None,
        };
        assert!(matches!(
            apply(&mut t, &Actor::Zend, &op, now()),
            Err(Error::Conflict(_))
        ));
        assert!(t.get(&path).unwrap().summary().is_none());
    }

    #[test]
    fn sentences_cite_only_what_is_there() {
        let mut t = zen();
        let (path, cite) = noted(&mut t, "topics", "x");
        // A live child with nothing to say yet is no input either.
        apply(
            &mut t,
            &john(),
            &Op::Create {
                under: path.clone(),
                name: slug("trip"),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        let bogus = Cite::Line(LineId::parse("2020-01-01-0000-0000").unwrap());
        for (sentences, says) in [
            (vec![s("A.", vec![bogus])], "which is not one of"),
            (
                vec![s("A.", vec![Cite::Child(slug("nobody"))])],
                "which is not one of",
            ),
            (
                vec![s("A.", vec![Cite::Child(slug("trip"))])],
                "which is not one of @zen/topics's inputs",
            ),
            (vec![s("A.", vec![])], "cites nothing"),
            (vec![s("  ", vec![cite.clone()])], "not one line"),
            (vec![s("A\nB", vec![cite.clone()])], "not one line"),
            (vec![], "says something"),
        ] {
            let err = summarise(&mut t, &path, sentences, None)
                .unwrap_err()
                .to_string();
            assert!(err.contains(says), "{err}");
        }
    }

    /// **A summary says as many sentences as its inputs ask for**, whoever
    /// writes it: one for every two inputs, up to three.
    #[test]
    fn a_summary_below_its_sentence_floor_is_refused() {
        let mut t = zen();
        let (path, a) = noted(&mut t, "topics", "a");
        let (_, b) = noted(&mut t, "topics", "b");
        let (_, c) = noted(&mut t, "topics", "c");
        assert_eq!(sentence_floor(t.get(&path).unwrap()), 2);
        let err = summarise(&mut t, &path, vec![s("A.", vec![a.clone(), b])], None).unwrap_err();
        assert_eq!(
            err.to_string(),
            Error::invalid(
                "@zen/topics's inputs ask for at least 2 sentences in its summary, not 1"
            )
            .to_string()
        );
        summarise(
            &mut t,
            &path,
            vec![s("A.", vec![a]), s("C.", vec![c])],
            None,
        )
        .unwrap();
        assert_eq!(sentence_floor(t.get(&p("threads")).unwrap()), 0);
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("conversations"),
                name: slug("2026-10-30-1200-grok"),
                cue: "open for a drive".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap();
        let drive = "conversations/2026-10-30-1200-grok";
        let (conversation, _) = noted(&mut t, drive, "a");
        noted(&mut t, drive, "b");
        noted(&mut t, drive, "c");
        assert_eq!(
            sentence_floor(t.get(&conversation).unwrap()),
            1,
            "a conversation says its exchange in one"
        );
    }

    /// **A line its summary cited, forgotten, is something gone** however
    /// many lines replaced it: the rebuild may shrink.
    #[test]
    fn a_cited_line_forgotten_lets_the_summary_shrink() {
        let mut t = zen();
        let (path, a) = noted(&mut t, "topics", "a");
        summarise(&mut t, &path, vec![s(&"long ".repeat(40), vec![a])], None).unwrap();
        assert_eq!(shortest(t.get(&path).unwrap()), Some((199, 100)));
        let first = t.get(&path).unwrap().window[0].id.clone();
        apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: path.clone(),
                which: Which::Id(first),
            },
            now(),
        )
        .unwrap();
        let (_, b) = noted(&mut t, "topics", "b");
        assert_eq!(shortest(t.get(&path).unwrap()), None);
        summarise(&mut t, &path, vec![s("short", vec![b])], None).unwrap();
    }

    #[test]
    fn caps_digests_and_cues_are_held() {
        let mut t = zen();
        let (path, cite) = noted(&mut t, "topics", "x");
        let long = "w".repeat(1001);
        assert!(
            summarise(&mut t, &path, vec![s(&long, vec![cite.clone()])], None)
                .unwrap_err()
                .to_string()
                .contains("cap of 1000")
        );
        assert!(
            summarise(&mut t, &path, vec![s("A.", vec![cite.clone()])], Some("d"))
                .unwrap_err()
                .to_string()
                .contains("only the master and a person's own journal do")
        );
        let (root, rcite) = noted(&mut t, "@zen", "family");
        assert!(summarise(
            &mut t,
            &root,
            vec![s("A.", vec![rcite.clone()])],
            Some(&"d".repeat(401))
        )
        .unwrap_err()
        .to_string()
        .contains("cap of 400"));
        let over = fingerprint(t.get(&path).unwrap());
        let op = Op::Summarize {
            path: path.clone(),
            over,
            sentences: vec![s("A.", vec![cite])],
            digest: None,
            cue: Some("new cue".into()),
        };
        assert!(apply(&mut t, &Actor::Zend, &op, now())
            .unwrap_err()
            .to_string()
            .contains("cue is fixed"));
    }

    #[test]
    fn a_second_rebuild_over_the_same_journal_is_dropped() {
        let mut t = zen();
        let (path, cite) = noted(&mut t, "topics", "x");
        summarise(&mut t, &path, vec![s("First.", vec![cite.clone()])], None).unwrap();
        let err = summarise(&mut t, &path, vec![s("Second.", vec![cite])], None).unwrap_err();
        assert!(matches!(err, Error::Conflict(_)), "{err}");
        assert_eq!(t.get(&path).unwrap().summary().unwrap().text(), "First.");
    }

    #[test]
    fn a_named_journal_can_take_a_new_cue_with_its_summary() {
        let mut t = zen();
        let path = apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("trip"),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: path.clone(),
                text: "Flights to Da Nang.".into(),
                speaker: None,
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let over = fingerprint(t.get(&path).unwrap());
        let op = Op::Summarize {
            path: path.clone(),
            over,
            sentences: vec![s("Flying to Da Nang.", vec![Cite::Line(id)])],
            digest: None,
            cue: Some("open for the Da Nang trip".into()),
        };
        apply(&mut t, &Actor::Zend, &op, now()).unwrap();
        assert_eq!(
            t.get(&path).unwrap().cue().as_deref(),
            Some("open for the Da Nang trip")
        );
    }

    #[test]
    fn a_summary_that_halves_with_nothing_gone_is_refused_and_marked_stale() {
        let mut t = zen();
        let (path, cite) = noted(&mut t, "topics", "x");
        summarise(
            &mut t,
            &path,
            vec![s(&"long ".repeat(40), vec![cite.clone()])],
            None,
        )
        .unwrap();
        noted(&mut t, "topics", "y");
        let err = summarise(&mut t, &path, vec![s("short", vec![cite])], None).unwrap_err();
        assert!(err.to_string().contains("shrinks from 199 to 5"), "{err}");
        let stale = |path: &JournalPath| Op::MarkStale {
            path: path.clone(),
            why: format!("{err}\n  (asked again when it changes)"),
        };
        let a = apply(&mut t, &Actor::Zend, &stale(&path), now()).unwrap();
        assert_eq!(
            a.subject,
            "Stale summary of @zen/topics: the summary shrinks from 199 to 5 characters \
             with nothing gone from @zen/topics (asked again when it changes)",
            "the refusal, on one line"
        );
        assert!(t.get(&path).unwrap().summary().unwrap().stale);
        // An empty refusal leaves the subject bare; a long one is cut at
        // 240 characters, counted in characters.
        let mut fresh = zen();
        let (q, qc) = noted(&mut fresh, "topics", "q");
        summarise(&mut fresh, &q, vec![s("Q.", vec![qc])], None).unwrap();
        let bare = Op::MarkStale {
            path: q.clone(),
            why: "  \n ".into(),
        };
        assert_eq!(
            apply(&mut fresh.clone(), &Actor::Zend, &bare, now())
                .unwrap()
                .subject,
            "Stale summary of @zen/topics"
        );
        let long = Op::MarkStale {
            path: q,
            why: "é".repeat(300),
        };
        assert_eq!(
            apply(&mut fresh, &Actor::Zend, &long, now())
                .unwrap()
                .subject,
            format!("Stale summary of @zen/topics: {}", "é".repeat(240))
        );
        assert!(apply(&mut t, &Actor::Zend, &stale(&path), now())
            .unwrap_err()
            .to_string()
            .contains("stale already"));
        assert!(apply(&mut t, &Actor::Zend, &stale(&p("threads")), now())
            .unwrap_err()
            .to_string()
            .contains("no summary"));
    }

    #[test]
    fn a_summary_may_shrink_when_lines_are_gone() {
        let mut t = zen();
        let (path, a) = noted(&mut t, "topics", "a");
        let (_, b) = noted(&mut t, "topics", "b");
        summarise(
            &mut t,
            &path,
            vec![s(&"long ".repeat(40), vec![a, b.clone()])],
            None,
        )
        .unwrap();
        let first = t.get(&path).unwrap().window[0].id.clone();
        apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: path.clone(),
                which: Which::Id(first),
            },
            now(),
        )
        .unwrap();
        summarise(&mut t, &path, vec![s("short", vec![b])], None).unwrap();
    }
}
