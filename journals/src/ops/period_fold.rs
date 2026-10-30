//! Folding a closed period of a window into its archive (§4): the lines move
//! to `archive/<period>.yaml`, and one `fold` line, zend's account of them,
//! takes their place where the first of them stood.

use crate::error::{Error, Result};
use crate::line::{Folds, Kind, Line, LineId, Writer};
use crate::ops::append::clean;
use crate::ops::{mint_keeping, Applied};
use crate::path::JournalPath;
use crate::period::Period;
use crate::time::Stamp;
use crate::tree::window::inside;
use crate::tree::Tree;

pub fn fold_period(
    tree: &mut Tree,
    path: &JournalPath,
    period: Period,
    ids: &[LineId],
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
    if want != given {
        return Err(Error::conflict(format!(
            "{path}'s lines in {period} changed while the fold was written; it is folded next pass"
        )));
    }
    let first = j
        .window
        .iter()
        .position(|l| want.binary_search(&l.id).is_ok())
        .expect("the period holds lines of the window");
    let (moved, kept): (Vec<Line>, Vec<Line>) = std::mem::take(&mut j.window)
        .into_iter()
        .partition(|l| want.binary_search(&l.id).is_ok());
    let at = moved.iter().map(|l| l.at).max().expect("not empty");
    let held = j.archives.entry(period).or_default();
    held.extend(moved);
    held.sort_by_key(|l| l.at);
    j.window = kept;
    let fold = Line {
        id: mint_keeping(j, keep, at, Writer::Zend, &text),
        at,
        kind: Kind::Fold,
        by: Writer::Zend,
        person: None,
        client: None,
        folds: Some(Folds::Period(period)),
        proposed: false,
        text,
    };
    let id = fold.id.clone();
    j.window.insert(first.min(j.window.len()), fold);
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
    use crate::testing::at;
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
        assert!(files.contains_key("persons/john/archive/2026-10-01.yaml"));
        // Every folded line is still found, by id, in its archive.
        assert!(plan.ids.iter().all(|id| j.has_id(id)));
    }

    #[test]
    fn an_open_period_or_a_changed_set_is_refused() {
        let mut t = sharratt();
        let path = filled(&mut t);
        let plan = next_fold(t.get(&path).unwrap(), at("2026-10-03T12:00:00Z")).unwrap();
        let fold = |ids: Vec<LineId>| Op::FoldPeriod {
            path: path.clone(),
            period: plan.period,
            ids,
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
            text: "x".into(),
        };
        assert!(
            apply(&mut t, &Actor::Zend, &empty, at("2026-10-03T12:00:00Z"))
                .unwrap_err()
                .to_string()
                .contains("no lines in")
        );
    }
}
