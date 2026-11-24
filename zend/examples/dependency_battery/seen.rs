//! Recall split by whether each expected item was in view as the answer
//! began (`view.rs`): the table that tells a selection failure — the item
//! never reached the model — from a reading failure — it was there and the
//! answer did not name it. Written into `summary.md` and
//! `table_9_12_in_view.csv`.
//!
//! Every `(answer, item)` pair is one observation, pooled over an arm's
//! answered lines that recorded a view.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::arms::{Arm, ARMS};
use crate::report::{csv, pct};
use crate::run::Line;
use crate::stats::mean;

/// One arm's items, split by what was in view.
#[derive(Debug, Clone, PartialEq)]
pub struct SeenRow {
    pub arm: Arm,
    pub answers: usize,
    pub items: usize,
    /// The share of items in an answer file's text in view.
    pub in_files: f64,
    /// The share of those the answer named.
    pub named_in_files: f64,
    /// The share of the items not there that the answer named.
    pub named_not_in_files: f64,
    /// The share of items anywhere in view.
    pub anywhere: f64,
    /// The share of those the answer named.
    pub named_anywhere: f64,
    /// The mean share of an answer file's reading turns placed, over the
    /// answer files whose reading the point weighed; NaN for R1.
    pub turns_placed: f64,
}

/// The share of `xs` that are true; NaN when empty.
fn share(xs: &[bool]) -> f64 {
    mean(&xs.iter().map(|&x| x as u8 as f64).collect::<Vec<_>>())
}

/// `arm`'s row over its answered lines that recorded a view; `None` when
/// none did.
pub fn seen_row(lines: &[Line], arm: Arm) -> Option<SeenRow> {
    let ls: Vec<&Line> = lines
        .iter()
        .filter(|l| l.error.is_none() && l.arm == arm.id() && l.view.is_some())
        .collect();
    if ls.is_empty() {
        return None;
    }
    // (in an answer file, anywhere, named) per item.
    let mut obs: Vec<(bool, bool, bool)> = Vec::new();
    let mut placed_shares: Vec<f64> = Vec::new();
    for l in &ls {
        let v = l.view.as_ref().expect("filtered on a view");
        obs.extend(
            v.in_files
                .iter()
                .zip(&v.anywhere)
                .zip(&v.named)
                .map(|((&f, &a), &n)| (f, a, n)),
        );
        placed_shares.extend(
            v.files
                .iter()
                .filter(|f| f.turns > 0)
                .map(|f| f.placed as f64 / f.turns as f64),
        );
    }
    let named_where = |pick: &dyn Fn(&(bool, bool, bool)) -> bool| {
        share(
            &obs.iter()
                .filter(|o| pick(o))
                .map(|o| o.2)
                .collect::<Vec<_>>(),
        )
    };
    Some(SeenRow {
        arm,
        answers: ls.len(),
        items: obs.len(),
        in_files: share(&obs.iter().map(|o| o.0).collect::<Vec<_>>()),
        named_in_files: named_where(&|o| o.0),
        named_not_in_files: named_where(&|o| !o.0),
        anywhere: share(&obs.iter().map(|o| o.1).collect::<Vec<_>>()),
        named_anywhere: named_where(&|o| o.1),
        turns_placed: mean(&placed_shares),
    })
}

/// Write the section into `md` and its CSV into `dir`.
pub fn write(md: &mut String, lines: &[Line], dir: &Path) -> anyhow::Result<()> {
    let rows: Vec<SeenRow> = ARMS.iter().filter_map(|&a| seen_row(lines, a)).collect();
    writeln!(
        md,
        "\n## Table 9.12 — an expected item in view as the answer began\n\n\
         Each expected item of each answer, looked for in the text the model had at its \
         answer's first token: in an answer file's text (the injected turns of its reading, \
         or R1's block of it), or anywhere it was given.\n"
    )?;
    if rows.is_empty() {
        writeln!(md, "No answer recorded a view.")?;
    } else {
        writeln!(
            md,
            "| Arm | Answers | Items | In an answer file's text | Named, when there | Named, \
             when not | Anywhere in view | Named, when anywhere | Answer-file turns placed |"
        )?;
        writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")?;
        for r in &rows {
            writeln!(
                md,
                "| {:?} | {} | {} | {}% | {}% | {}% | {}% | {}% | {}% |",
                r.arm,
                r.answers,
                r.items,
                pct(r.in_files),
                pct(r.named_in_files),
                pct(r.named_not_in_files),
                pct(r.anywhere),
                pct(r.named_anywhere),
                pct(r.turns_placed)
            )?;
        }
    }
    fs::write(
        dir.join("table_9_12_in_view.csv"),
        csv(
            &[
                "arm",
                "answers",
                "items",
                "in_files",
                "named_in_files",
                "named_not_in_files",
                "anywhere",
                "named_anywhere",
                "turns_placed",
            ],
            rows.iter().map(|r| {
                vec![
                    format!("{:?}", r.arm),
                    r.answers.to_string(),
                    r.items.to_string(),
                    r.in_files.to_string(),
                    r.named_in_files.to_string(),
                    r.named_not_in_files.to_string(),
                    r.anywhere.to_string(),
                    r.named_anywhere.to_string(),
                    r.turns_placed.to_string(),
                ]
            }),
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::battery::Question;
    use crate::placement::Placement;
    use crate::run::scored;
    use crate::view::{FileShare, View};

    fn line(arm: Arm, view: Option<View>) -> Line {
        let q = Question {
            id: "d1".into(),
            category: "direct".into(),
            question: "Where?".into(),
            expect: vec!["a".into(), "b".into()],
            files: vec!["x.rs".into()],
            names_file: false,
        };
        let mut l = scored(
            &q,
            arm,
            1,
            "a",
            &HashSet::new(),
            1,
            None,
            Placement::default(),
            1.0,
        );
        l.view = view;
        l
    }

    fn view(in_files: [bool; 2], anywhere: [bool; 2], named: [bool; 2], placed: usize) -> View {
        View {
            in_files: in_files.to_vec(),
            anywhere: anywhere.to_vec(),
            named: named.to_vec(),
            files: vec![FileShare { turns: 4, placed }],
        }
    }

    /// **Items pool across answers**, split by whether each was in an answer
    /// file's text: three of four were there and two of those were named; the
    /// one that was not there was not named. A line with no view counts for
    /// nothing.
    #[test]
    fn items_are_split_by_what_was_in_view() {
        let lines = vec![
            line(
                Arm::C1,
                Some(view([true, true], [true, true], [true, false], 1)),
            ),
            line(
                Arm::C1,
                Some(view([true, false], [true, true], [true, false], 3)),
            ),
            line(Arm::C1, None),
            line(Arm::R1, None),
        ];
        let r = seen_row(&lines, Arm::C1).unwrap();
        assert_eq!((r.answers, r.items), (2, 4));
        assert_eq!(r.in_files, 0.75);
        assert_eq!(r.named_in_files, 2.0 / 3.0);
        assert_eq!(r.named_not_in_files, 0.0);
        assert_eq!((r.anywhere, r.named_anywhere), (1.0, 0.5));
        assert_eq!(r.turns_placed, 0.5);
        assert_eq!(seen_row(&lines, Arm::R1), None);
    }
}
