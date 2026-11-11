//! What a set of journals says after a point in time
//! (`docs/npc_conversations.md` §6.1): a conversation's catch-up on what its
//! NPC heard, said and did since it last spoke with its party.
//!
//! **Read from a little before the point.** A line can be stamped before the
//! point and arrive after it — written on another member and synced across a
//! few seconds later — so the read starts [`OVERLAP`] earlier and leaves out
//! the lines the caller says it has seen.
//!
//! **Fitted to a budget, what matters first.** When the lines do not all fit,
//! what the NPC said, wrote, promised and what changed for it are kept first,
//! then what it heard, then its own acts ([`ExperienceKind::rank`]), newest
//! first within each. A fold stands in for the archived lines of its period
//! that did not fit.

use std::cmp::Reverse;
use std::collections::BTreeSet;

use crate::actor::{need_read, Actor};
use crate::error::Result;
use crate::experience::ExperienceKind;
use crate::line::{Folds, Kind, Line, LineId};
use crate::path::JournalPath;
use crate::period::Period;
use crate::time::Stamp;
use crate::tree::Tree;

/// How far before the point the read starts, in seconds.
pub const OVERLAP: i64 = 60;

/// What a line costs of the budget beyond its text: its id, time and writer
/// as a reader is shown them.
const LINE_COST: usize = 48;

/// One line read, and the journal it is in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Read {
    pub path: JournalPath,
    pub line: Line,
}

/// What [`since`] read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Since {
    /// Oldest first.
    pub lines: Vec<Read>,
    /// Lines after the point that did not fit, and no fold stood in for.
    pub left_out: usize,
}

/// A line's place in the order the budget is filled in: by its experience's
/// rank, a line with none ranking with what was heard.
fn rank(l: &Line) -> u8 {
    l.experience
        .as_ref()
        .map_or(ExperienceKind::Heard.rank(), |e| e.kind.rank())
}

fn cost(l: &Line) -> usize {
    l.text.chars().count() + LINE_COST
}

/// A folded period's archived lines after the point, and the fold that
/// stands for them.
struct Folded {
    lines: Vec<Read>,
    fold: Option<Read>,
    period: Period,
}

/// The lines of `paths` after `after`, read as `actor`, leaving out `seen`,
/// fitted to `budget` characters; oldest first.
///
/// A window's own lines are fitted first, by rank. Then each folded period
/// the read reaches, newest first, is read whole when its archived lines all
/// fit, and as its fold when they do not.
pub fn since(
    tree: &Tree,
    actor: &Actor,
    paths: &[JournalPath],
    after: Stamp,
    seen: &BTreeSet<LineId>,
    budget: usize,
) -> Result<Since> {
    let start = after.plus(-OVERLAP);
    let wanted = |l: &Line| l.kind != Kind::Fold && l.at >= start && !seen.contains(&l.id);
    let mut window: Vec<Read> = Vec::new();
    let mut folded: Vec<Folded> = Vec::new();
    for path in paths {
        need_read(tree, actor, path)?;
        // A journal not made yet has nothing to say: a bot's first turn.
        let Some(j) = tree.get(path) else { continue };
        let read = |l: &Line| Read {
            path: path.clone(),
            line: l.clone(),
        };
        window.extend(j.window.iter().filter(|l| wanted(l)).map(read));
        for (period, archived) in &j.archives {
            if period.closes() <= start {
                continue;
            }
            let lines: Vec<Read> = archived.iter().filter(|l| wanted(l)).map(read).collect();
            if lines.is_empty() {
                continue;
            }
            let fold = j
                .all_lines()
                .find(|l| l.folds == Some(Folds::Period(*period)))
                .map(read);
            folded.push(Folded {
                lines,
                fold,
                period: *period,
            });
        }
    }
    window.sort_by_key(|r| (rank(&r.line), Reverse(r.line.at), r.line.id.clone()));
    let mut left = budget;
    let mut kept = Vec::new();
    let mut left_out = 0;
    for r in window {
        let c = cost(&r.line);
        if c <= left {
            left -= c;
            kept.push(r);
        } else {
            left_out += 1;
        }
    }
    folded.sort_by_key(|f| Reverse(f.period.closes()));
    for f in folded {
        let whole: usize = f.lines.iter().map(|r| cost(&r.line)).sum();
        if whole <= left {
            left -= whole;
            kept.extend(f.lines);
            continue;
        }
        match f.fold {
            Some(fold) if cost(&fold.line) <= left => {
                left -= cost(&fold.line);
                kept.push(fold);
            }
            _ => left_out += f.lines.len(),
        }
    }
    kept.sort_by_key(|r| (r.line.at, r.line.id.clone()));
    Ok(Since {
        lines: kept,
        left_out,
    })
}

#[cfg(test)]
mod tests {
    use std::slice;

    use super::*;
    use crate::experience::Experience;
    use crate::ops::fixture::{p, zen};
    use crate::ops::period_fold::lines_over;
    use crate::ops::{apply, Lived, OnTaken, Op};
    use crate::testing::{at, slug};
    use crate::tree::window::next_fold;

    fn npcd() -> Actor {
        Actor::Npcd { group: slug("zen") }
    }

    /// `bots/zen`, holding `lines`.
    fn bot(lines: Vec<(Stamp, &str, ExperienceKind)>) -> Tree {
        let mut t = zen();
        apply(
            &mut t,
            &npcd(),
            &Op::Create {
                under: p("bots"),
                name: slug("zen"),
                cue: "open for Zen".into(),
                on_taken: OnTaken::Refuse,
            },
            at("2026-10-01T00:00:00Z"),
        )
        .unwrap();
        apply(
            &mut t,
            &npcd(),
            &Op::Episodes {
                path: p("bots/zen"),
                lines: lines
                    .into_iter()
                    .map(|(at, text, kind)| Lived {
                        at,
                        text: text.into(),
                        experience: Experience::of(kind),
                    })
                    .collect(),
            },
            at("2026-10-01T00:00:00Z"),
        )
        .unwrap();
        t
    }

    fn texts(s: &Since) -> Vec<&str> {
        s.lines.iter().map(|r| r.line.text.as_str()).collect()
    }

    /// **A line stamped just before the point but arriving after it is still
    /// read, once**: the read starts a minute early and leaves out what was
    /// seen.
    #[test]
    fn the_overlap_catches_a_late_arrival_and_seen_lines_are_left_out() {
        let after = at("2026-10-01T12:00:00Z");
        let t = bot(vec![
            (after.plus(-120), "long before", ExperienceKind::Heard),
            (after.plus(-30), "arrived late", ExperienceKind::Heard),
            (after.plus(-10), "seen already", ExperienceKind::Said),
            (after.plus(5), "after", ExperienceKind::Said),
        ]);
        let j = t.get(&p("bots/zen")).unwrap();
        let seen: BTreeSet<LineId> = j
            .window
            .iter()
            .filter(|l| l.text == "seen already")
            .map(|l| l.id.clone())
            .collect();
        let s = since(&t, &npcd(), &[p("bots/zen")], after, &seen, 12_000).unwrap();
        assert_eq!(texts(&s), ["arrived late", "after"]);
        assert_eq!(s.left_out, 0);
    }

    /// **A short budget keeps what matters to a conversation first**: what was
    /// said and promised, then what was heard, then the NPC's own acts.
    #[test]
    fn a_short_budget_keeps_what_was_said_over_what_was_done() {
        let after = at("2026-10-01T12:00:00Z");
        let t = bot(vec![
            (after.plus(10), "did one", ExperienceKind::Did),
            (after.plus(20), "heard one", ExperienceKind::Heard),
            (after.plus(30), "promised one", ExperienceKind::Promised),
            (after.plus(40), "did two", ExperienceKind::Did),
        ]);
        let two = 2 * (LINE_COST + "promised one".len());
        let s = since(&t, &npcd(), &[p("bots/zen")], after, &BTreeSet::new(), two).unwrap();
        assert_eq!(texts(&s), ["heard one", "promised one"]);
        assert_eq!(s.left_out, 2);
    }

    /// **A fold stands in for its period's lines that did not fit.**
    #[test]
    fn a_fold_stands_in_for_its_archived_lines() {
        let t0 = at("2026-10-01T00:00:00Z");
        let lines: Vec<(Stamp, String)> = (0..=50)
            .map(|i| (t0.plus(3600 * i), format!("did {i:02}")))
            .collect();
        let mut t = bot(lines
            .iter()
            .map(|(a, s)| (*a, s.as_str(), ExperienceKind::Did))
            .collect());
        let now = at("2026-10-04T00:00:00Z");
        let path = p("bots/zen");
        let plan = next_fold(t.get(&path).unwrap(), now).unwrap();
        let over = lines_over(t.get(&path).unwrap(), &plan.ids);
        apply(
            &mut t,
            &Actor::Zend,
            &Op::FoldPeriod {
                over,
                path: path.clone(),
                period: plan.period,
                ids: plan.ids,
                text: "A day of chores.".into(),
            },
            now,
        )
        .unwrap();
        // Room for the 27 lines still in the window and the fold, not for the
        // 24 the fold archived; read from the start of the 1st.
        let line = LINE_COST + "did 00".len();
        let budget = 27 * line + LINE_COST + "A day of chores.".len();
        let s = since(
            &t,
            &npcd(),
            slice::from_ref(&path),
            t0.plus(OVERLAP),
            &BTreeSet::new(),
            budget,
        )
        .unwrap();
        let read = texts(&s);
        assert_eq!(read.len(), 28, "{read:?}");
        assert_eq!(
            read[0], "A day of chores.",
            "the fold, at its day's last line"
        );
        assert!(read.contains(&"did 50"));
        assert!(
            !read.contains(&"did 00"),
            "the archived originals did not fit"
        );
        assert_eq!(s.left_out, 0, "the fold stood in for them");
        // With room for everything, the originals are read instead of it.
        let all = since(
            &t,
            &npcd(),
            &[path],
            t0.plus(OVERLAP),
            &BTreeSet::new(),
            51 * line,
        )
        .unwrap();
        assert_eq!(all.lines.len(), 51);
        assert_eq!(texts(&all)[0], "did 00");
    }

    /// **npcd reads only what it may**: a journal outside its reach refuses
    /// the whole read, and one not made yet reads as nothing.
    #[test]
    fn reach_is_checked_and_a_missing_journal_is_empty() {
        let t = zen();
        let after = at("2026-10-01T12:00:00Z");
        let none = since(&t, &npcd(), &[p("bots/zen")], after, &BTreeSet::new(), 100).unwrap();
        assert_eq!(none, Since::default());
        assert!(since(&t, &npcd(), &[p("topics")], after, &BTreeSet::new(), 100).is_err());
    }
}
