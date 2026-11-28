//! A layer whose units are paths — the knowledge base's files and folders —
//! one folder at a time.
//!
//! The layer's units are members of the system prompt's collection, not
//! conversations, so the layer's list of conversations is empty however much
//! it holds. What it holds is paths, nested under the folders they lie in. A
//! knowledge base is thousands of them, so the view is one level: a folder's
//! own listing, its child folders each with a count of what lies below it, and
//! its files each with the versions held. A child folder's level is fetched
//! when it is opened.

use std::collections::BTreeMap;

use serde::Serialize;

/// One unit as the view shows it: a version of a path, sealed as one or more
/// sections.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PathUnitView {
    /// The version it read: a file's git blob, a folder's listing hash.
    pub blob: String,
    /// The branches whose tip holds that version.
    pub branches: Vec<String>,
    /// A file's line count; a folder's entries.
    pub lines: u32,
    /// The sections it is sealed as.
    pub sections: usize,
    pub tokens: usize,
    /// Whether this process holds its sections, so the fill can place it.
    pub placed: bool,
}

/// A unit and the workspace path it was read at, repository first; a folder's
/// path ends with `/`.
pub struct PathUnit {
    pub path: String,
    pub unit: PathUnitView,
}

/// One child of a level.
#[derive(Debug, PartialEq, Serialize)]
pub struct PathNode {
    /// The last component, a folder's with its `/`.
    pub name: String,
    /// The whole path, as [`PathUnit::path`] spells it — what its own level is
    /// fetched by.
    pub path: String,
    pub folder: bool,
    /// Units at this path and below it.
    pub unit_count: usize,
    /// Their tokens.
    pub tokens: usize,
    /// A file's versions. Empty for a folder, whose units are its own level's.
    pub units: Vec<PathUnitView>,
}

/// One folder of the tree: what is read at it, and its children.
#[derive(Debug, PartialEq, Serialize)]
pub struct PathLevel {
    /// The folder, `""` for the top, whose children are the repositories.
    pub path: String,
    /// The folder's own listing, one per version held.
    pub units: Vec<PathUnitView>,
    /// Folders first, then files, each by name.
    pub children: Vec<PathNode>,
}

/// The level of `units` at the folder `at`: units at other paths than `at` and
/// below it are left out, so a caller may hand over more than the folder holds.
pub fn path_level(at: &str, units: impl IntoIterator<Item = PathUnit>) -> PathLevel {
    let mut own = Vec::new();
    let mut children: BTreeMap<String, PathNode> = BTreeMap::new();
    for PathUnit { path, unit } in units {
        let Some(rest) = path.strip_prefix(at) else {
            continue;
        };
        if rest.is_empty() {
            own.push(unit);
            continue;
        }
        let (name, folder) = match rest.find('/') {
            Some(end) => (&rest[..=end], true),
            None => (rest, false),
        };
        let child = children
            .entry(name.to_string())
            .or_insert_with(|| PathNode {
                name: name.to_string(),
                path: format!("{at}{name}"),
                folder,
                unit_count: 0,
                tokens: 0,
                units: Vec::new(),
            });
        child.unit_count += 1;
        child.tokens += unit.tokens;
        if !folder {
            child.units.push(unit);
        }
    }
    let mut children: Vec<PathNode> = children.into_values().collect();
    children.sort_by(|a, b| (!a.folder, &a.name).cmp(&(!b.folder, &b.name)));
    PathLevel {
        path: at.to_string(),
        units: own,
        children,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn unit(path: &str, blob: &str, tokens: usize) -> PathUnit {
        PathUnit {
            path: path.into(),
            unit: PathUnitView {
                blob: blob.into(),
                branches: vec!["main".into()],
                lines: 3,
                sections: 1,
                tokens,
                placed: true,
            },
        }
    }

    fn all() -> Vec<PathUnit> {
        vec![
            unit("candle/src/lib.rs", "a1", 100),
            unit("candle/README.md", "b2", 50),
            unit("candle/src/", "c3", 20),
            unit("candle/src/x/y.rs", "d4", 7),
            unit("yenbot/main.rs", "e5", 9),
        ]
    }

    /// **The top level is the repositories**, each counting every unit below
    /// it and holding none inline.
    #[test]
    fn the_top_level_is_the_repositories() {
        let level = path_level("", all());
        assert_eq!(
            serde_json::to_value(&level).unwrap(),
            json!({
                "path": "",
                "units": [],
                "children": [
                    {"name": "candle/", "path": "candle/", "folder": true,
                     "unit_count": 4, "tokens": 177, "units": []},
                    {"name": "yenbot/", "path": "yenbot/", "folder": true,
                     "unit_count": 1, "tokens": 9, "units": []},
                ],
            })
        );
    }

    /// **A folder's level is its own listing, its folders with their counts,
    /// then its files with their versions** — and nothing of another folder.
    #[test]
    fn a_folders_level_holds_its_listing_folders_and_files() {
        let level = path_level("candle/src/", all());
        assert_eq!(
            serde_json::to_value(&level).unwrap(),
            json!({
                "path": "candle/src/",
                "units": [{"blob": "c3", "branches": ["main"], "lines": 3, "sections": 1,
                           "tokens": 20, "placed": true}],
                "children": [
                    {"name": "x/", "path": "candle/src/x/", "folder": true,
                     "unit_count": 1, "tokens": 7, "units": []},
                    {"name": "lib.rs", "path": "candle/src/lib.rs", "folder": false,
                     "unit_count": 1, "tokens": 100,
                     "units": [{"blob": "a1", "branches": ["main"], "lines": 3,
                                "sections": 1, "tokens": 100, "placed": true}]},
                ],
            })
        );
    }

    /// **Two versions of one file are one child with both**, and a sibling
    /// whose name only starts with the folder's is not inside it.
    #[test]
    fn versions_of_a_file_share_its_child() {
        let level = path_level(
            "candle/",
            vec![
                unit("candle/a.rs", "old", 10),
                unit("candle/a.rs", "new", 12),
                unit("candle-extra/b.rs", "x", 1),
            ],
        );
        assert_eq!(level.children.len(), 1);
        let a = &level.children[0];
        let blobs: Vec<&str> = a.units.iter().map(|u| u.blob.as_str()).collect();
        assert_eq!((a.unit_count, a.tokens, blobs), (2, 22, vec!["old", "new"]));
    }

    /// No units, an empty level.
    #[test]
    fn no_units_make_an_empty_level() {
        let level = path_level("", Vec::new());
        assert!(level.units.is_empty() && level.children.is_empty());
    }
}
