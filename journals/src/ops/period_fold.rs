//! Folding a closed period of a window into its archive (§4): the lines' files
//! move to `archive/<period>/`, and one `fold` line, zend's account of them,
//! takes their place at the period's last moment.

use std::mem;

use crate::error::{Error, Result};
use crate::files::{in_order, Fingerprint};
use crate::line::{Folds, Kind, Line, LineId, Writer};
use crate::ops::append::clean;
use crate::ops::{mint_keeping, Applied};
use crate::path::JournalPath;
use crate::period::Period;
use crate::time::Stamp;
use crate::tree::window::inside;
use crate::tree::{Journal, Tree};

/// What a fold of the window lines `ids` is written from — each line as the
/// fold's question shows it: its time, kind, writer, the person and channel it
/// was written for, whether it is still proposed, and its text, in time order.
/// Ids are left out, so a replay that renames a colliding id still matches;
/// which lines are folded is checked by id on its own. The lines are taken
/// as a set — each one's fields, the set sorted — so neither the window's
/// order nor a tie in time can move it.
pub fn lines_over(j: &Journal, ids: &[LineId]) -> Fingerprint {
    let mut lines: Vec<Vec<u8>> = j
        .window
        .iter()
        .filter(|l| ids.contains(&l.id))
        .map(line_bytes)
        .collect();
    lines.sort();
    Fingerprint::of_bytes(&lines.concat())
}

/// One line's fields as [`lines_over`] reads them, each closed by a NUL.
fn line_bytes(l: &Line) -> Vec<u8> {
    let mut bytes = Vec::new();
    for field in [
        l.at.to_string().as_str(),
        l.kind.name(),
        l.by.name(),
        l.person.as_ref().map_or("", |p| p.as_str()),
        l.channel.map_or("", |c| c.name()),
        if l.proposed { "proposed" } else { "" },
        l.text.as_str(),
    ] {
        bytes.extend_from_slice(field.as_bytes());
        bytes.push(0);
    }
    bytes
}

#[allow(clippy::too_many_arguments)]
pub fn fold_period(
    tree: &mut Tree,
    path: &JournalPath,
    period: Period,
    ids: &[LineId],
    over: &Fingerprint,
    text: &str,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    let text = clean(text)?;
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    if !period.is_closed(now) {
        return Err(Error::refused(format!("{period} is not over yet")));
    }
    let mut want = inside(j, period);
    if want.is_empty() {
        return Err(Error::refused(format!(
            "{path} has no lines in {period} to fold"
        )));
    }
    let mut given = ids.to_vec();
    want.sort();
    given.sort();
    if want != given || &lines_over(j, &want) != over {
        return Err(Error::conflict(format!(
            "{path}'s lines in {period} changed while the fold was written; it is folded next pass"
        )));
    }
    let (moved, kept): (Vec<Line>, Vec<Line>) = mem::take(&mut j.window)
        .into_iter()
        .partition(|l| want.binary_search(&l.id).is_ok());
    let at = moved.iter().map(|l| l.at).max().expect("not empty");
    let held = j.archives.entry(period).or_default();
    held.extend(moved);
    in_order(held);
    j.window = kept;
    let fold = Line {
        id: mint_keeping(j, keep, at, Writer::Zend, &text),
        at,
        kind: Kind::Fold,
        by: Writer::Zend,
        person: None,
        channel: None,
        folds: Some(Folds::Period(period)),
        proposed: false,
        text,
    };
    let id = fold.id.clone();
    j.window.push(fold);
    in_order(&mut j.window);
    Ok(Applied {
        line: Some(id),
        ..Applied::touching(format!("Fold {period} of {path}"), vec![path.clone()])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{john, p, sharratt};
    use crate::ops::{apply, Op};
    use crate::path::Address;
    use crate::testing::{at, slug};
    use crate::tree::render::own_files;
    use crate::tree::window::next_fold;
    use crate::tree::WINDOW;

    fn filled(t: &mut Tree) -> JournalPath {
        let path = p("~");
        let t0 = at("2026-10-01T00:00:00Z");
        for i in 0..=WINDOW {
            apply(
                t,
                &john(),
                &Op::Note {
                    path: path.clone(),
                    text: format!("n{i}"),
                },
                t0.plus(3600 * i as i64),
            )
            .unwrap();
        }
        path
    }

    #[test]
    fn a_closed_day_moves_to_its_archive_behind_one_fold_line() {
        let mut t = sharratt();
        let path = filled(&mut t);
        let now = at("2026-10-03T12:00:00Z");
        let plan = next_fold(t.get(&path).unwrap(), now).unwrap();
        assert_eq!(plan.period.to_string(), "2026-10-01");
        let op = Op::FoldPeriod {
            path: path.clone(),
            period: plan.period,
            ids: plan.ids.clone(),
            over: lines_over(t.get(&path).unwrap(), &plan.ids),
            text: "Settled in on the 1st.".into(),
        };
        let a = apply(&mut t, &Actor::Zend, &op, now).unwrap();
        assert_eq!(a.subject, "Fold 2026-10-01 of ~");
        let j = t.get(&path).unwrap();
        assert_eq!(j.window.len(), WINDOW + 1 - 24 + 1);
        let fold = &j.window[0];
        assert_eq!(
            (fold.kind, fold.at),
            (Kind::Fold, at("2026-10-01T23:00:00Z"))
        );
        assert_eq!(fold.folds, Some(Folds::Period(plan.period)));
        assert_eq!(j.archives[&plan.period].len(), 24);
        assert_eq!(j.window[1].text, "n24");
        let files = own_files(&path, j);
        let archived = files
            .keys()
            .filter(|f| f.starts_with("persons/john/archive/2026-10-01/"))
            .count();
        assert_eq!(
            archived, 24,
            "a file per folded line, in its period's folder"
        );
        // Every folded line is still found, by id, in its archive.
        assert!(plan.ids.iter().all(|id| j.has_id(id)));
    }

    #[test]
    fn an_open_period_or_a_changed_set_is_refused() {
        let mut t = sharratt();
        let path = filled(&mut t);
        let plan = next_fold(t.get(&path).unwrap(), at("2026-10-03T12:00:00Z")).unwrap();
        let over = lines_over(t.get(&path).unwrap(), &plan.ids);
        let fold = |ids: Vec<LineId>| Op::FoldPeriod {
            path: path.clone(),
            period: plan.period,
            ids,
            over: over.clone(),
            text: "x".into(),
        };
        let early = apply(
            &mut t,
            &Actor::Zend,
            &fold(plan.ids.clone()),
            at("2026-10-01T23:59:59Z"),
        );
        assert!(early.unwrap_err().to_string().contains("not over yet"));
        let fewer = plan.ids[1..].to_vec();
        assert!(matches!(
            apply(
                &mut t,
                &Actor::Zend,
                &fold(fewer),
                at("2026-10-03T12:00:00Z")
            ),
            Err(Error::Conflict(_))
        ));
        let empty = Op::FoldPeriod {
            path: path.clone(),
            period: Period::parse("2025-01-01").unwrap(),
            ids: vec![],
            over: lines_over(t.get(&path).unwrap(), &[]),
            text: "x".into(),
        };
        assert!(
            apply(&mut t, &Actor::Zend, &empty, at("2026-10-03T12:00:00Z"))
                .unwrap_err()
                .to_string()
                .contains("no lines in")
        );
    }

    /// **A fold is written from the lines as they were shown**: one of them
    /// edited while the fold was being written makes it a conflict, though
    /// the same lines are there by id.
    #[test]
    fn a_line_edited_while_the_fold_was_written_is_a_conflict() {
        let mut t = sharratt();
        let path = filled(&mut t);
        let now = at("2026-10-03T12:00:00Z");
        let plan = next_fold(t.get(&path).unwrap(), now).unwrap();
        let op = Op::FoldPeriod {
            path: path.clone(),
            period: plan.period,
            ids: plan.ids.clone(),
            over: lines_over(t.get(&path).unwrap(), &plan.ids),
            text: "The first day.".into(),
        };
        apply(
            &mut t,
            &john(),
            &Op::Edit {
                address: Address {
                    path: path.clone(),
                    id: plan.ids[0].clone(),
                },
                text: "n0, corrected".into(),
            },
            now,
        )
        .unwrap();
        assert!(matches!(
            apply(&mut t, &Actor::Zend, &op, now),
            Err(Error::Conflict(_))
        ));
    }

    /// The fingerprint reads what the lines say, not what they are called.
    #[test]
    fn what_a_fold_is_written_from_leaves_ids_out() {
        let mut t = sharratt();
        let path = filled(&mut t);
        let j = t.get(&path).unwrap().clone();
        let ids: Vec<LineId> = j.window[..2].iter().map(|l| l.id.clone()).collect();
        let mut renamed = j.clone();
        renamed.window[0].id = LineId::parse("2026-10-01-0000-ffff").unwrap();
        let renamed_ids = vec![renamed.window[0].id.clone(), ids[1].clone()];
        assert_eq!(lines_over(&j, &ids), lines_over(&renamed, &renamed_ids));
        let mut edited = j.clone();
        edited.window[1].text = "other".into();
        assert_ne!(lines_over(&j, &ids), lines_over(&edited, &ids));
        let mut accepted = j.clone();
        accepted.window[1].proposed = !accepted.window[1].proposed;
        assert_ne!(lines_over(&j, &ids), lines_over(&accepted, &ids));
        let mut reattributed = j.clone();
        reattributed.window[1].person = match &j.window[1].person {
            Some(_) => None,
            None => Some(slug("yen")),
        };
        assert_ne!(lines_over(&j, &ids), lines_over(&reattributed, &ids));
        assert_ne!(lines_over(&j, &ids), lines_over(&j, &ids[..1]));
        let mut reordered = j.clone();
        reordered.window.swap(0, 1);
        assert_eq!(
            lines_over(&j, &ids),
            lines_over(&reordered, &ids),
            "the window's order is not what the fold read"
        );
    }
}
