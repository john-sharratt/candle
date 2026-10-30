//! Moving a subtree: each journal in it takes the place its new path gives,
//! and every path in it is listed so the store rewrites their files.

use crate::error::Result;
use crate::path::JournalPath;
use crate::tree::Journal;

/// Give `j` and everything below it the places `path` gives them. Refuses a
/// subtree that would reach deeper than the tree allows.
pub fn re_place(j: &mut Journal, path: &JournalPath) -> Result<()> {
    j.place = path.place()?;
    let names: Vec<_> = j.children.keys().cloned().collect();
    for name in names {
        let child_path = path.child(name.clone())?;
        re_place(
            j.children.get_mut(&name).expect("listed above"),
            &child_path,
        )?;
    }
    Ok(())
}

/// `path` and every journal below it.
pub fn subtree_paths(path: &JournalPath, j: &Journal) -> Vec<JournalPath> {
    j.walk()
        .into_iter()
        .map(|(names, _)| {
            let mut all = path.names().to_vec();
            all.extend(names);
            JournalPath::new(path.owner().clone(), all).expect("a placed subtree is at places")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::Meta;
    use crate::ops::fixture::p;
    use crate::path::Place;
    use crate::scope::Section;
    use crate::testing::{at, slug};

    fn named(name: &str) -> Journal {
        Journal::new(
            Place::Named {
                section: Section::Topics,
                depth: 1,
            },
            Some(Meta::new(
                name,
                Some("x".into()),
                at("2026-10-01T00:00:00Z"),
            )),
        )
    }

    #[test]
    fn a_moved_subtree_takes_its_new_places() {
        let mut a = named("a");
        a.children.insert(slug("b"), named("b"));
        let to = p("projects/x/a");
        re_place(&mut a, &to).unwrap();
        assert_eq!(
            a.place,
            Place::Named {
                section: Section::Projects,
                depth: 2
            }
        );
        assert_eq!(
            a.children[&slug("b")].place,
            Place::Named {
                section: Section::Projects,
                depth: 3
            }
        );
        let paths: Vec<String> = subtree_paths(&to, &a).iter().map(|p| p.show()).collect();
        assert_eq!(
            paths,
            ["@sharratt/projects/x/a", "@sharratt/projects/x/a/b"]
        );
    }

    #[test]
    fn a_subtree_moved_too_deep_is_refused() {
        let mut a = named("a");
        a.children.insert(slug("b"), named("b"));
        assert!(
            re_place(&mut a, &p("topics/1/2/3/4/a")).is_ok(),
            "b lands at depth 6"
        );
        assert!(
            re_place(&mut a, &p("topics/1/2/3/4/5/a")).is_err(),
            "b would land at depth 7"
        );
    }
}
