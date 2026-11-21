//! A plan checked against the file: every item goes to exactly one file.
//!
//! An item named whole (`N`) goes where it is named. A container may instead
//! have its members named one by one (`N.M`) — then every member must be, and
//! each of the files they go to gets the container again around its share. A
//! `use` inside a `mod { }` goes with every share and is never named.

use std::collections::BTreeMap;

use crate::layout::Layout;
use crate::plan::{Id, PlanFile};

/// Where one item goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// Whole, to the plan file at this index.
    Whole(usize),
    /// Member by member: the plan file each member goes to, `None` for a
    /// `use` that goes with every share.
    Members(Vec<Option<usize>>),
}

/// Every item's placement, in item order, or every way the plan is wrong.
pub fn assign(layout: &Layout, plan: &[PlanFile]) -> Result<Vec<Placement>, String> {
    let mut whole: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut member: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    let mut errors: Vec<String> = Vec::new();
    for (f, file) in plan.iter().enumerate() {
        for id in &file.ids {
            match *id {
                Id::Whole(n) => whole.entry(n).or_default().push(f),
                Id::Member(n, m) => member.entry((n, m)).or_default().push(f),
            }
        }
    }
    let n_items = layout.items.len();
    for (&n, files) in &whole {
        if n == 0 || n > n_items {
            errors.push(format!("{n} is not an item (1..={n_items})"));
        } else if files.len() > 1 {
            errors.push(format!("{n} is named {} times", files.len()));
        }
    }
    let mut out = Vec::with_capacity(n_items);
    for (i, item) in layout.items.iter().enumerate() {
        let n = i + 1;
        let named_whole = whole.get(&n).and_then(|f| f.first().copied());
        let members = item
            .body
            .as_ref()
            .map(|b| b.members.as_slice())
            .unwrap_or(&[]);
        // An item that cannot be divided (a trait impl, an ordinary
        // `mod { }`) but is named by members goes whole to the file that
        // names most of them, the first such file on a tie.
        if item.body.is_none() && named_whole.is_none() {
            let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
            for (_, files) in member.range((n, 0)..(n + 1, 0)) {
                for &f in files {
                    *votes.entry(f).or_default() += 1;
                }
            }
            let most = votes.values().copied().max();
            if let Some(f) = votes
                .iter()
                .find(|&(_, &v)| Some(v) == most)
                .map(|(&f, _)| f)
            {
                out.push(Placement::Whole(f));
                continue;
            }
        }
        let named_members: Vec<Option<usize>> = (1..=members.len())
            .map(|m| member.get(&(n, m)).and_then(|f| f.first().copied()))
            .collect();
        for (m, files) in member.range((n, 0)..(n + 1, 0)) {
            if m.1 == 0 || m.1 > members.len() {
                errors.push(format!(
                    "{n}.{} is not a member of {n} ({})",
                    m.1, item.name
                ));
            } else if files.len() > 1 {
                errors.push(format!("{n}.{} is named {} times", m.1, files.len()));
            }
        }
        let any_member = named_members.iter().any(Option::is_some);
        match (named_whole, any_member) {
            (Some(f), false) => out.push(Placement::Whole(f)),
            (Some(_), true) => {
                errors.push(format!("{n} ({}) is named whole and by member", item.name));
            }
            (None, false) => errors.push(format!("{n} ({}) is not placed", item.name)),
            (None, true) => {
                let mut placed = Vec::with_capacity(members.len());
                for (j, (m, f)) in members.iter().zip(&named_members).enumerate() {
                    match (m.is_use, f) {
                        (true, Some(_)) => errors.push(format!(
                            "{n}.{} is a `use` or an inner attribute, which goes with every share: do not name it",
                            j + 1
                        )),
                        (true, None) => placed.push(None),
                        (false, Some(f)) => placed.push(Some(*f)),
                        (false, None) => {
                            errors.push(format!("{n}.{} ({}) is not placed", j + 1, m.name))
                        }
                    }
                }
                out.push(Placement::Members(placed));
            }
        }
    }
    for &(n, _) in member.keys() {
        if n == 0 || n > n_items {
            errors.push(format!("{n} is not an item (1..={n_items})"));
        }
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        errors.dedup();
        Err(errors.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::layout;
    use crate::plan::parse;

    const SRC: &str = "struct A;\n\nimpl A {\n    fn x() {}\n    fn y() {}\n}\n\n\
                       #[cfg(test)]\nmod tests {\n    use super::*;\n    fn t() {}\n}\n";

    #[test]
    fn whole_items_and_members_are_placed() {
        let l = layout(SRC).unwrap();
        let p = parse("file a: A.\n1 2.1\nfile b: B.\n2.2 3\n").unwrap();
        assert_eq!(
            assign(&l, &p).unwrap(),
            vec![
                Placement::Whole(0),
                Placement::Members(vec![Some(0), Some(1)]),
                Placement::Whole(1),
            ]
        );
        // A test module's `use` goes with every share, unnamed.
        let p = parse("file a: A.\n1 2 3.2\n").unwrap();
        assert_eq!(
            assign(&l, &p).unwrap()[2],
            Placement::Members(vec![None, Some(0)])
        );
    }

    /// **Every mistake is reported**, each saying which id.
    #[test]
    fn every_misplacement_is_named() {
        let l = layout(SRC).unwrap();
        let p = parse("file a: A.\n1 1 2 2.1 9 3.1\n").unwrap();
        assert_eq!(
            assign(&l, &p).unwrap_err(),
            "1 is named 2 times\n9 is not an item (1..=3)\n\
             2 (impl A) is named whole and by member\n\
             3.1 is a `use` or an inner attribute, which goes with every share: do not name it\n\
             3.2 (fn t) is not placed"
        );
        let p = parse("file a: A.\n1 2.1\n").unwrap();
        assert_eq!(
            assign(&l, &p).unwrap_err(),
            "2.2 (fn y) is not placed\n3 (mod tests) is not placed"
        );
    }

    /// **An indivisible item named by members goes whole** to the file that
    /// names most of them — a trait impl cannot be spread over files.
    #[test]
    fn an_indivisible_item_named_by_members_goes_whole() {
        let src = "struct A;\n\nimpl Clone for A {\n    fn clone(&self) -> A { A }\n}\n";
        let l = layout(src).unwrap();
        let p = parse("file a: A.\n1 2.1\nfile b: B.\n2.2 2.3\n").unwrap();
        assert_eq!(
            assign(&l, &p).unwrap(),
            vec![Placement::Whole(0), Placement::Whole(1)]
        );
    }
}
