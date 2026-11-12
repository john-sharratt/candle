//! What a round holds its file calls' repositories and paths to
//! (`docs/stencil_tree.md` §9.2): the workspace's repositories, each one's
//! files and directories as the conversation's store holds them now, its own
//! writes included.
//!
//! A path the store does not hold is a refused call, and one class of them was
//! not a model's slip but the tools' own teaching: every read fence names a
//! file `<repo>/<path>`, so a first directory that is, or starts with, a
//! repository's name (`candle-conversation/…`, `journals/…` in `candle`) was
//! written without it, as though it were the prefix. Held, a call can name only
//! what exists. A path the tool will create — `write`, `file_download`, the
//! destination of `file_move` and `file_image` — names nothing yet, and is not
//! held; neither is a git tool's, which names a file at a revision.

use std::collections::{BTreeSet, HashMap};
use std::iter;

use candle_conversation::stencil::FieldHold;
use zend_tools::registry;
use zend_vfs::RepoFiles;

/// The tools whose `repo` names one repository of the workspace.
const ONE_REPO: [&str; 10] = [
    "file_read",
    "file_edit",
    "file_delete",
    "file_move",
    "file_upload",
    "file_image",
    "file_list",
    "file_present",
    "write",
    "file_download",
];

/// The tools whose `repo` names a repository or `*`, every one of them.
const ANY_REPO: [&str; 2] = ["file_grep", "file_search"];

/// What `repo` takes in [`ANY_REPO`] to cover every repository.
const EVERY: &str = "*";

/// The tools whose `path` names a file that exists.
const EXISTING_FILE: [&str; 6] = [
    "file_read",
    "file_edit",
    "file_delete",
    "file_move",
    "file_upload",
    "file_image",
];

/// `tools` and every alias each answers to.
fn names_of(tools: &[&str]) -> Vec<String> {
    tools
        .iter()
        .flat_map(|tool| iter::once(*tool).chain(registry::aliases(tool).iter().copied()))
        .map(str::to_string)
        .collect()
}

/// Every directory `files` lie in, by its path: `a/b/c.rs` lies in `a` and
/// `a/b`.
fn dirs_of(files: &[String]) -> BTreeSet<String> {
    let mut dirs = BTreeSet::new();
    for file in files {
        let mut at = 0;
        while let Some(slash) = file[at..].find('/') {
            at += slash;
            dirs.insert(file[..at].to_string());
            at += 1;
        }
    }
    dirs
}

/// The holds of a workspace whose repositories hold `repos`' files, each
/// `(repository, its files)`.
pub fn holds_of(repos: &[(String, Vec<String>)]) -> Vec<FieldHold> {
    let names: Vec<String> = repos.iter().map(|(name, _)| name.clone()).collect();
    let mut files = HashMap::new();
    let mut listed = HashMap::new();
    let mut prefixes = HashMap::new();
    let mut every_prefix = BTreeSet::new();
    for (repo, paths) in repos {
        let key = vec![repo.clone()];
        let dirs = dirs_of(paths);
        // A listing names a directory, or `""` for the repository's root.
        listed.insert(
            key.clone(),
            iter::once(String::new())
                .chain(dirs.iter().cloned())
                .collect(),
        );
        // A search prefix is a directory, with or without its closing `/`.
        let prefix: BTreeSet<String> = dirs
            .iter()
            .flat_map(|d| [d.clone(), format!("{d}/")])
            .collect();
        every_prefix.extend(prefix.iter().cloned());
        prefixes.insert(key.clone(), prefix.into_iter().collect());
        files.insert(key, paths.clone());
    }
    prefixes.insert(vec![EVERY.to_string()], every_prefix.into_iter().collect());
    let unkeyed = |values: Vec<String>| HashMap::from([(Vec::new(), values)]);
    let any_repo: Vec<String> = names.iter().cloned().chain([EVERY.to_string()]).collect();
    let hold = |tools: &[&str], field: &str, on: &[&str], table| FieldHold {
        tools: names_of(tools),
        field: field.to_string(),
        on: on.iter().map(|f| f.to_string()).collect(),
        table,
    };
    vec![
        hold(&ONE_REPO, "repo", &[], unkeyed(names)),
        hold(&ANY_REPO, "repo", &[], unkeyed(any_repo)),
        hold(&EXISTING_FILE, "path", &["repo"], files),
        hold(&["file_list"], "path", &["repo"], listed),
        hold(&ANY_REPO, "prefix", &["repo"], prefixes),
    ]
}

/// The holds of `files` as the conversation's store holds them now.
pub fn holds(files: &RepoFiles) -> Vec<FieldHold> {
    let repos: Vec<(String, Vec<String>)> = files
        .all()
        .into_iter()
        .map(|(name, store)| (name, store.paths("")))
        .collect();
    holds_of(&repos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repos() -> Vec<(String, Vec<String>)> {
        vec![
            (
                "candle".to_string(),
                vec![
                    "README.md".to_string(),
                    "candle-conversation/src/turn_layout.rs".to_string(),
                    "candle-conversation/build.rs".to_string(),
                ],
            ),
            (
                "journals".to_string(),
                vec!["groups/zen/journal.yaml".to_string()],
            ),
        ]
    }

    fn table<'a>(
        holds: &'a [FieldHold],
        tool: &str,
        field: &str,
    ) -> &'a HashMap<Vec<String>, Vec<String>> {
        &holds
            .iter()
            .find(|h| h.field == field && h.tools.iter().any(|t| t == tool))
            .unwrap_or_else(|| panic!("{tool}.{field} is not held"))
            .table
    }

    fn key(repo: &str) -> Vec<String> {
        vec![repo.to_string()]
    }

    /// **A file call names a repository the workspace has**, and a search may
    /// name every one.
    #[test]
    fn a_repo_is_one_of_the_workspaces() {
        let holds = holds_of(&repos());
        assert_eq!(
            table(&holds, "file_read", "repo")[&Vec::new()],
            ["candle", "journals"]
        );
        assert_eq!(
            table(&holds, "write", "repo")[&Vec::new()],
            ["candle", "journals"]
        );
        assert_eq!(
            table(&holds, "file_grep", "repo")[&Vec::new()],
            ["candle", "journals", "*"]
        );
    }

    /// **A file a tool works on is one the repository named holds**: a path
    /// whose first directory begins with the repository's name keeps it, and
    /// a tool that creates its file is not held.
    #[test]
    fn an_existing_files_path_is_one_of_its_repositorys() {
        let holds = holds_of(&repos());
        for tool in EXISTING_FILE {
            assert_eq!(
                table(&holds, tool, "path")[&key("candle")],
                [
                    "README.md",
                    "candle-conversation/src/turn_layout.rs",
                    "candle-conversation/build.rs"
                ],
                "{tool}"
            );
        }
        assert!(!holds
            .iter()
            .any(|h| h.field == "path" && h.tools.iter().any(|t| t == "write")));
    }

    /// **A listing names a directory or the root; a search prefix a directory,
    /// with or without its `/`, and across every repository for `*`.**
    #[test]
    fn directories_are_held_for_listings_and_searches() {
        let holds = holds_of(&repos());
        assert_eq!(
            table(&holds, "file_list", "path")[&key("candle")],
            ["", "candle-conversation", "candle-conversation/src"]
        );
        assert_eq!(
            table(&holds, "file_search", "prefix")[&key("journals")],
            ["groups", "groups/", "groups/zen", "groups/zen/"]
        );
        let every = &table(&holds, "file_grep", "prefix")[&key("*")];
        assert!(every.contains(&"candle-conversation/src/".to_string()));
        assert!(every.contains(&"groups/zen".to_string()));
    }

    #[test]
    fn a_files_directories_are_every_folder_it_lies_in() {
        let dirs = dirs_of(&[
            "a/b/c.rs".to_string(),
            "a/d.rs".to_string(),
            "e.rs".to_string(),
        ]);
        assert_eq!(dirs.into_iter().collect::<Vec<_>>(), ["a", "a/b"]);
    }
}
