//! A changed tree against the one it changed (`journals check --against`,
//! `docs/zen_memory_and_guides.md` Phase 7): no line lost, and how far the
//! tree's shape moved, held to a budget. A restructuring or a maintenance run
//! moves lines and journals; it never drops a line it did not name.

use std::collections::{BTreeMap, BTreeSet};

use crate::line::LineId;
use crate::tree::Tree;

/// How far a change moved a tree's shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Shape {
    pub created: usize,
    pub removed: usize,
    pub lines_moved: usize,
    pub summaries_rewritten: usize,
}

/// The most a change may move each part of the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget(pub Shape);

/// Where each line of `tree` is, by id: its journal's address.
fn line_homes(tree: &Tree) -> BTreeMap<LineId, String> {
    tree.journals()
        .into_iter()
        .flat_map(|(path, j)| {
            let at = path.show();
            j.all_lines()
                .map(move |l| (l.id.clone(), at.clone()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Every line of `old` that `new` holds nowhere, unless `forgets` names it:
/// each as a problem naming the line and where it was.
pub fn lost(old: &Tree, new: &Tree, forgets: &BTreeSet<LineId>) -> Vec<String> {
    let kept = line_homes(new);
    line_homes(old)
        .into_iter()
        .filter(|(id, _)| !kept.contains_key(id) && !forgets.contains(id))
        .map(|(id, at)| {
            format!("{at}#{id} is gone: a change keeps every line it does not forget by name")
        })
        .collect()
}

/// How far `new` moved from `old`.
pub fn shape(old: &Tree, new: &Tree) -> Shape {
    let journals = |t: &Tree| -> BTreeMap<String, Option<String>> {
        t.journals()
            .into_iter()
            .map(|(p, j)| (p.show(), j.summary().map(|s| s.text())))
            .collect()
    };
    let (before, after) = (journals(old), journals(new));
    let (was, is) = (line_homes(old), line_homes(new));
    Shape {
        created: after.keys().filter(|p| !before.contains_key(*p)).count(),
        removed: before.keys().filter(|p| !after.contains_key(*p)).count(),
        lines_moved: was
            .iter()
            .filter(|(id, at)| is.get(*id).is_some_and(|now| now != *at))
            .count(),
        summaries_rewritten: before
            .iter()
            .filter(|(p, s)| after.get(*p).is_some_and(|now| now != *s))
            .count(),
    }
}

impl Budget {
    /// A budget as its file writes it: `created: 2`, one part to a line;
    /// a part left out may not move at all.
    pub fn parse(text: &str) -> Result<Budget, String> {
        let mut b = Shape::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once(':')
                .ok_or_else(|| format!("line {}: `{line}` is not `part: n`", n + 1))?;
            let value: usize = value
                .trim()
                .parse()
                .map_err(|_| format!("line {}: `{}` is not a count", n + 1, value.trim()))?;
            match key.trim() {
                "created" => b.created = value,
                "removed" => b.removed = value,
                "lines_moved" => b.lines_moved = value,
                "summaries_rewritten" => b.summaries_rewritten = value,
                other => {
                    return Err(format!(
                        "line {}: `{other}` is no part of a shape: created, removed, \
                         lines_moved or summaries_rewritten",
                        n + 1
                    ))
                }
            }
        }
        Ok(Budget(b))
    }

    /// Each part of `shape` past this budget, as a problem.
    pub fn over(&self, shape: Shape) -> Vec<String> {
        [
            ("journals created", shape.created, self.0.created),
            ("journals removed", shape.removed, self.0.removed),
            ("lines moved", shape.lines_moved, self.0.lines_moved),
            (
                "summaries rewritten",
                shape.summaries_rewritten,
                self.0.summaries_rewritten,
            ),
        ]
        .into_iter()
        .filter(|(_, n, most)| n > most)
        .map(|(what, n, most)| format!("{n} {what}, over the budget of {most}"))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::tree::load::load;

    const ZEN: &str = "group: zen\nname: \"Zen\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n";
    const LINE: &str =
        "at: 2026-10-30T09:12:40Z\nkind: note\nby: claude-code\naccount: john\ntext: a\n";

    /// A tree holding the journals `at`, each `name:` folder with its lines.
    fn tree(at: &[(&str, &[&str])]) -> Tree {
        let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            ("groups/zen/group.yaml".to_string(), ZEN.as_bytes().to_vec()),
            (
                "persons/john/person.yaml".to_string(),
                b"person: john\nname: \"John\"\naccounts: []\ngroups: [zen]\n".to_vec(),
            ),
        ]);
        for (folder, lines) in at {
            files.insert(
                format!("groups/zen/{folder}/journal.yaml"),
                format!(
                    "name: {}\ncue: open for it\ncreated: 2026-10-30T00:00:00Z\nfolded: false\n",
                    folder.rsplit('/').next().unwrap()
                )
                .into_bytes(),
            );
            for id in *lines {
                files.insert(
                    format!("groups/zen/{folder}/{id}.yaml"),
                    LINE.as_bytes().to_vec(),
                );
            }
        }
        load(&files)
    }

    const A: &str = "2026-10-30-0912-7e3a";
    const B: &str = "2026-10-30-0912-7e3b";

    /// **A moved line is kept, a dropped one is lost unless it is forgotten
    /// by name**, and the shape counts what moved.
    #[test]
    fn a_dropped_line_is_lost_unless_forgotten_and_a_moved_one_counts() {
        let old = tree(&[("topics/a", &[A, B])]);
        let moved = tree(&[("topics/a", &[A]), ("topics/b", &[B])]);
        assert!(lost(&old, &moved, &BTreeSet::new()).is_empty());
        assert_eq!(
            shape(&old, &moved),
            Shape {
                created: 1,
                removed: 0,
                lines_moved: 1,
                summaries_rewritten: 0
            }
        );
        let dropped = tree(&[("topics/a", &[A])]);
        assert_eq!(
            lost(&old, &dropped, &BTreeSet::new()),
            [format!(
                "@zen/topics/a#{B} is gone: a change keeps every line it does not forget by name"
            )]
        );
        let forgets = BTreeSet::from([LineId::parse(B).unwrap()]);
        assert!(lost(&old, &dropped, &forgets).is_empty());
    }

    /// **A budget is read part by part, and a shape past it is named part by
    /// part**; a part it leaves out may not move.
    #[test]
    fn a_budget_is_read_and_held() {
        let b = Budget::parse("# a run's budget\ncreated: 1\nlines_moved: 10\n").unwrap();
        assert_eq!(
            b.over(Shape {
                created: 2,
                removed: 1,
                lines_moved: 10,
                summaries_rewritten: 0
            }),
            [
                "2 journals created, over the budget of 1",
                "1 journals removed, over the budget of 0"
            ]
        );
        assert!(Budget::parse("moved: 3")
            .unwrap_err()
            .contains("no part of a shape"));
        assert!(Budget::parse("created: many")
            .unwrap_err()
            .contains("not a count"));
    }
}
