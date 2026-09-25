//! Per-directory ingest units for the `repo_map` layer.
//!
//! One directory, one conversation. The unit carries everything the folder's
//! turns need: the files directly inside it, the one-level listing its
//! `file_list` turn shows, and a content hash driving the resume cache and the
//! refresh decision.
//!
//! **No anchor excerpt.** A folder used to be described by reading its
//! `README`/module doc; the chain is now a single `file_list` round-trip and the
//! summary, so names and paths are the whole evidence.

use std::collections::{BTreeMap, HashMap, HashSet};

use sha2::{Digest, Sha256};
use zend_tools::tools::file::list::LIST_PAGE_ENTRIES;

use super::types::{FileEntry, ModuleHint, RepoMap};

/// One directory's ingest unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirUnit {
    /// Workspace-relative directory path with a trailing slash (`zend/src/`),
    /// or `"."` for the workspace root — the form used as the gather-scope tag
    /// and the resume-cache key.
    pub dir: String,
    /// Files directly inside the directory, in path order.
    pub files: Vec<FileEntry>,
    /// The walked entries of this directory — its own files, plus the NAME of
    /// each subdirectory holding anything, which is exactly the shape
    /// `file_list` returns for one level. Paged, so this keeps the first
    /// [`LIST_PAGE_ENTRIES`] in path order. What the hash covers, and an
    /// approximation of what the turn shows: see [`listed_paths`] for where the
    /// two diverge.
    pub listed: Vec<String>,
    /// SHA-256 over [`Self::listed`]. Hashing the
    /// evidence — rather than a proxy like the directory's own file names — is
    /// what makes the cache exact in both directions: a file added deep in the
    /// subtree changes the listing and must re-ingest, while an edit to a file
    /// that is only named (never shown) leaves the summary accurate and must not.
    ///
    /// Exact only as far as `listed` reaches. A file the walk cannot see — a
    /// `.cu` kernel, anything off the extension allowlist — appears in the
    /// folder's rendered listing but not here, so adding one changes what the
    /// turn shows without moving the hash, and that folder never re-ingests.
    pub content_hash: String,
}

impl DirUnit {
    /// Prefix the folder's `file_list` call uses. The workspace root lists with
    /// an empty prefix, matching how the live tool addresses it.
    pub fn list_prefix(&self) -> &str {
        if self.dir == "." {
            ""
        } else {
            &self.dir
        }
    }

    /// Human-facing directory label for the summarise request.
    pub fn label(&self) -> &str {
        &self.dir
    }

    /// The manifest hint of a file directly inside this directory, if one is a
    /// manifest (`Cargo.toml`, `package.json`, …). Rendered into the summarise
    /// request so a crate/package root announces itself rather than leaving the
    /// model to infer it from a filename in the listing.
    pub fn module_hint(&self) -> Option<&ModuleHint> {
        self.files.iter().find_map(|f| f.module_hint.as_ref())
    }
}

/// Build one unit per directory that holds at least one walked file.
///
/// Deterministic: directories in sorted order, files in walk (path) order, so
/// two runs over an unchanged tree produce byte-identical units and the resume
/// cache hits.
pub fn build_units(map: &RepoMap) -> Vec<DirUnit> {
    let mut by_dir: BTreeMap<String, Vec<&FileEntry>> = BTreeMap::new();
    for file in &map.files {
        by_dir.entry(dir_of(&file.path)).or_default().push(file);
    }

    by_dir
        .into_iter()
        .map(|(dir, files)| {
            let files: Vec<FileEntry> = files.into_iter().cloned().collect();
            let listed = listed_paths(map, &dir);
            let hint = files.iter().find_map(|f| f.module_hint.as_ref());
            let content_hash = hash_unit(&listed, hint);
            DirUnit {
                dir,
                files,
                listed,
                content_hash,
            }
        })
        .collect()
}

/// The walked entries OF the directory — its own files, plus one name per
/// subdirectory — in path order, capped at one page: an approximation of what
/// the folder's `file_list` turn shows.
///
/// **One level, because the turn is one level.** `file_list` lists a single
/// directory and names its subdirectories rather than descending into them, so a
/// hash taken over the whole subtree would answer a question the turn never
/// asks. It did, when the tool matched a plain path prefix: a file added three
/// levels down moved this directory's hash *and* every ancestor's, re-decoding
/// summaries whose listings were byte-identical.
///
/// The two are still NOT the same set. `file_list` walks the workspace itself,
/// while this filters [`walk_workspace`]'s output, which drops files off the
/// extension allowlist (`.cu`, `.cuh`, extensionless files), files above
/// `MAX_FILE_BYTES`, and whole nested git repos / submodules.
/// `candle-kernels/src/paged-decode/` shows eight files in its turn and
/// contributes one here, because the kernels themselves are `.cu`. So this must
/// not be read as the shown evidence: it is the walked evidence, which is what
/// the hash and the refresh compare on. A subdirectory whose every file is off
/// the allowlist is invisible here and named in the turn — the same divergence,
/// now reachable a level up as well.
fn listed_paths(map: &RepoMap, dir: &str) -> Vec<String> {
    let prefix = if dir == "." { "" } else { dir };
    let mut out: Vec<String> = Vec::new();
    let mut subdirs: Vec<String> = Vec::new();
    for f in map.files.iter() {
        let Some(rest) = f.path.strip_prefix(prefix) else {
            continue;
        };
        match rest.split_once('/') {
            // Below this directory: the turn shows the subdirectory's NAME, so
            // that is what the hash covers. Once each, however many files are
            // under it — which is also what stops a file added three levels down
            // from re-summarising every folder above it.
            Some((head, _)) if !head.is_empty() => {
                let name = format!("{prefix}{head}/");
                if !subdirs.contains(&name) {
                    subdirs.push(name);
                }
            }
            _ => out.push(f.path.clone()),
        }
    }
    out.append(&mut subdirs);
    out.sort();
    out.truncate(LIST_PAGE_ENTRIES);
    out
}

/// Directory of a workspace-relative file path, with a trailing slash; `"."` for
/// a file at the workspace root.
fn dir_of(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[..=i].to_string(),
        None => ".".to_string(),
    }
}

/// Hash the walked entries of the directory. A rename, addition or deletion
/// among them moves it (the listing changed); an edit to a file that is only
/// NAMED does not — the unit never showed that content, so re-summarising would
/// decode the same answer at full cost.
///
/// **The anchor's text is no longer part of it**, because the anchor is no
/// longer part of the turn: the chain is one `file_list` round-trip and the
/// summary. Hashing an excerpt the conversation never shows would re-decode
/// every folder whose README was touched, for a listing that did not change —
/// the rule here is that the hash covers what the turn shows, and nothing else.
///
/// **The module hint is part of what the turn shows**, so by that same rule it
/// belongs here: `render_request` splices it into the request the model reads
/// (`Summarize the \`candle-nn/\` folder (crate: candle-nn) …`). Left out, a
/// manifest edit that changes the hint without changing the listing — a crate
/// renamed in its `Cargo.toml`, a `[workspace]` table added — moved the question
/// while the resume cache still reported a hit, so the sealed summary answered a
/// request that is no longer the one this unit renders.
fn hash_unit(listed: &[String], hint: Option<&ModuleHint>) -> String {
    let mut h = Sha256::new();
    for n in listed {
        h.update(n.as_bytes());
        h.update(b"\n");
    }
    if let Some(hint) = hint {
        h.update(hint.render().as_bytes());
        h.update(b"\n");
    }
    let digest = h.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// Per-unit state recorded after a successful ingest — the refresh path compares
/// a fresh walk against this to decide which directories changed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirState {
    pub units: Vec<DirRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirRecord {
    pub dir: String,
    pub content_hash: String,
}

impl DirState {
    /// The state that would result if EVERY unit ingested successfully.
    ///
    /// Tests use this to express an expected outcome. The production path must
    /// not: it builds the state from the walk, so a directory whose ingest
    /// failed still gets its hash recorded as ingested and is never retried.
    /// See `repo_scan::dir_state_from_substrate`, which reads what actually
    /// landed.
    pub fn from_units(units: &[DirUnit]) -> Self {
        Self {
            units: units
                .iter()
                .map(|u| DirRecord {
                    dir: u.dir.clone(),
                    content_hash: u.content_hash.clone(),
                })
                .collect(),
        }
    }

    /// This state without the records `map`'s `--max-depth` bound froze. A frozen
    /// directory's unit stays in the substrate, but the bounded walk never builds
    /// it — so comparing the whole state against the walk would read every
    /// refresh as a change and re-run the pool on each watcher burst.
    pub fn without_frozen(&self, map: &RepoMap) -> Self {
        Self {
            units: self
                .units
                .iter()
                .filter(|r| !map.is_frozen_dir(&r.dir))
                .cloned()
                .collect(),
        }
    }

    /// `true` when a fresh walk produced exactly the same directories and hashes
    /// — the refresh has nothing to do.
    ///
    /// Compared as a set, not pairwise: the state is rebuilt from substrate
    /// metadata, whose enumeration order has nothing to do with the walk's.
    pub fn equivalent_to(&self, units: &[DirUnit]) -> bool {
        if self.units.len() != units.len() {
            return false;
        }
        let prior: HashMap<&str, &str> = self
            .units
            .iter()
            .map(|r| (r.dir.as_str(), r.content_hash.as_str()))
            .collect();
        units
            .iter()
            .all(|u| prior.get(u.dir.as_str()) == Some(&u.content_hash.as_str()))
    }

    /// Directories whose hash differs, plus those that vanished. Informational —
    /// the refresh itself re-ingests only the changed units.
    pub fn changed_dirs(&self, units: &[DirUnit]) -> Vec<String> {
        let prior: HashMap<&str, &str> = self
            .units
            .iter()
            .map(|r| (r.dir.as_str(), r.content_hash.as_str()))
            .collect();
        let mut out: Vec<String> = units
            .iter()
            .filter(|u| prior.get(u.dir.as_str()) != Some(&u.content_hash.as_str()))
            .map(|u| u.dir.clone())
            .collect();
        let fresh: HashSet<&str> = units.iter().map(|u| u.dir.as_str()).collect();
        for r in &self.units {
            if !fresh.contains(r.dir.as_str()) {
                out.push(r.dir.clone());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::types::Language;
    use crate::repo_scan::walk_workspace;

    /// With `--max-depth 2`, a unit AT the bound (`src/deep/`) is frozen: it
    /// leaves the state the refresh compares, so a bounded walk that never
    /// builds it cannot read as a change. Unbounded, the state is untouched.
    #[test]
    fn a_frozen_directory_leaves_the_compared_state() {
        let rec = |dir: &str, hash: &str| DirRecord {
            dir: dir.to_string(),
            content_hash: hash.to_string(),
        };
        let prior = DirState {
            units: vec![rec(".", "a"), rec("src/", "b"), rec("src/deep/", "c")],
        };
        let bounded = RepoMap {
            max_depth: Some(2),
            ..RepoMap::default()
        };
        assert_eq!(
            prior.without_frozen(&bounded),
            DirState {
                units: vec![rec(".", "a"), rec("src/", "b")],
            }
        );
        assert_eq!(prior.without_frozen(&RepoMap::default()), prior);
    }

    fn entry(path: &str) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            line_count: 1,
            language: Language::Rust,
            size_bytes: 1,
            module_hint: None,
        }
    }

    fn map(paths: &[&str]) -> RepoMap {
        RepoMap {
            files: paths.iter().map(|p| entry(p)).collect(),
            ..Default::default()
        }
    }

    fn empty_workspace() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn one_unit_per_directory_in_sorted_order() {
        let m = map(&["b/x.rs", "a/y.rs", "a/z.rs", "top.rs"]);
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let units = build_units(&m);
        let dirs: Vec<&str> = units.iter().map(|u| u.dir.as_str()).collect();
        assert_eq!(dirs, vec![".", "a/", "b/"]);
        assert_eq!(units[1].files.len(), 2, "a/ holds both its files");
    }

    /// The root lists with an empty prefix — the form the live tool takes.
    #[test]
    fn the_root_unit_lists_with_an_empty_prefix() {
        let m = map(&["top.rs"]);
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let units = build_units(&m);
        assert_eq!(units[0].dir, ".");
        assert_eq!(units[0].list_prefix(), "");
    }

    #[test]
    fn a_nested_directory_lists_with_its_own_prefix() {
        let m = map(&["zend/src/x.rs"]);
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let units = build_units(&m);
        assert_eq!(units[0].dir, "zend/src/");
        assert_eq!(units[0].list_prefix(), "zend/src/");
    }

    #[test]
    fn state_round_trips_and_detects_no_change() {
        let m = map(&["a/x.rs"]);
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let units = build_units(&m);
        let st = DirState::from_units(&units);
        assert!(st.equivalent_to(&units));
        assert!(st.changed_dirs(&units).is_empty());
    }

    /// The production state is rebuilt from substrate metadata, whose
    /// enumeration order is unrelated to the walk's. A pairwise comparison
    /// would call that "changed" and re-ingest the whole workspace on every
    /// filesystem event.
    #[test]
    fn state_is_compared_as_a_set_not_pairwise() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let units = build_units(&map(&["a/x.rs", "b/y.rs", "c/z.rs"]));
        let mut st = DirState::from_units(&units);
        st.units.reverse();
        assert!(st.equivalent_to(&units), "{:?}", st.units);
        assert!(st.changed_dirs(&units).is_empty());
    }

    /// A directory missing from the state (its ingest failed, so no hash was
    /// written) must read as changed — that is what gets it retried.
    #[test]
    fn a_directory_absent_from_the_state_is_changed() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let units = build_units(&map(&["a/x.rs", "b/y.rs"]));
        let mut st = DirState::from_units(&units);
        let dropped = st.units.pop().expect("two units").dir;
        assert!(!st.equivalent_to(&units));
        assert!(st.changed_dirs(&units).contains(&dropped), "{dropped}");
    }

    #[test]
    fn adding_a_file_changes_only_the_directories_that_show_it() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let before = build_units(&map(&["a/x.rs", "b/y.rs"]));
        let after = build_units(&map(&["a/x.rs", "a/new.rs", "b/y.rs"]));
        let st = DirState::from_units(&before);
        assert!(!st.equivalent_to(&after));
        assert_eq!(st.changed_dirs(&after), vec!["a/".to_string()]);
    }

    /// **A file added deep below a folder re-summarises that folder and nothing
    /// above it.**
    ///
    /// `file_list` lists ONE directory and names its subdirectories rather than
    /// descending, so `a/`'s turn shows `a/x.rs` and `a/b/` before and after —
    /// byte-identical — and its stored summary is still true. Only `a/b/c/`,
    /// whose own listing gained an entry, is stale.
    ///
    /// It used to move every ancestor's hash, because the tool matched a plain
    /// path prefix and a folder's listing really did span its whole subtree. On
    /// this workspace that made any edit three levels down re-decode every folder
    /// above it, up to and including the root.
    #[test]
    fn a_file_added_deep_re_summarises_its_own_folder_only() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        // `top.rs` gives the workspace root a unit of its own.
        let before = build_units(&map(&["top.rs", "a/x.rs", "a/b/c/y.rs"]));
        let after = build_units(&map(&["top.rs", "a/x.rs", "a/b/c/y.rs", "a/b/c/z.rs"]));
        let st = DirState::from_units(&before);
        let changed = st.changed_dirs(&after);
        assert_eq!(changed, vec!["a/b/c/".to_string()], "{changed:?}");
    }

    /// A NEW subdirectory does move its parent's hash: the parent's listing gains
    /// the subdirectory's name, so what that turn shows really did change.
    #[test]
    fn a_new_subdirectory_moves_the_parents_hash() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let before = build_units(&map(&["top.rs", "a/x.rs"]));
        let after = build_units(&map(&["top.rs", "a/x.rs", "a/b/y.rs"]));
        let st = DirState::from_units(&before);
        let changed = st.changed_dirs(&after);
        assert!(changed.contains(&"a/".to_string()), "{changed:?}");
        // The root is untouched — it showed `top.rs` and `a/` before and after.
        assert!(!changed.contains(&".".to_string()), "{changed:?}");
    }

    /// The listing is PAGED, so only the first page is shown — and only what is
    /// shown is hashed. A file sorting past the page boundary never appears in
    /// the turn, so re-decoding the same summary would be pure cost.
    #[test]
    fn a_file_past_the_shown_page_leaves_the_hash_alone() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let full: Vec<String> = (0..LIST_PAGE_ENTRIES)
            .map(|i| format!("a/f{i:03}.rs"))
            .collect();
        let refs: Vec<&str> = full.iter().map(|s| s.as_str()).collect();
        let before = build_units(&map(&refs));
        assert_eq!(before[0].listed.len(), LIST_PAGE_ENTRIES, "a full page");

        // `zz.rs` sorts after every `f###.rs`, so it lands on page two.
        let mut with_extra = full.clone();
        with_extra.push("a/zz.rs".to_string());
        let refs2: Vec<&str> = with_extra.iter().map(|s| s.as_str()).collect();
        let after = build_units(&map(&refs2));
        assert_eq!(before[0].content_hash, after[0].content_hash);
    }

    /// `listed` is the WALKED evidence, NOT the listing the turn shows.
    /// `walk_workspace` applies an extension allowlist, a size cap, and a
    /// nested-submodule prune that `file_list` does not, so a folder can list
    /// more files in its turn than ever reach its hash.
    ///
    /// Allowlisting `.cu`/`.cuh` closed the case that mattered — the CUDA
    /// kernels — but not the class: an extensionless file still lists and still
    /// never reaches the hash.
    ///
    /// Both halves are asserted against the real walk and the real tool, because
    /// reading `listed` as the shown count once retired
    /// `candle-kernels/src/paged-decode/` and its four siblings from the layer as
    /// "single-file folders with nothing to describe" — a mistake made by
    /// validating the idea against the tool's output while the code read the
    /// walk's.
    #[test]
    fn listed_holds_walked_files_not_the_listing_the_turn_shows() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("k")).unwrap();
        std::fs::write(d.path().join("k/api.rs"), "//! Kernel wrapper.\n").unwrap();
        std::fs::write(d.path().join("k/decode.cu"), "__global__ void d() {}\n").unwrap();
        std::fs::write(d.path().join("k/LICENSE"), "MIT\n").unwrap();

        let walked = walk_workspace(d.path(), None);
        let units = build_units(&walked);
        let k = units.iter().find(|u| u.dir == "k/").expect("k/ has a unit");
        assert_eq!(
            k.listed,
            vec!["k/api.rs".to_string(), "k/decode.cu".to_string()],
            "the kernel now reaches the hash; the extensionless file still does not",
        );

        let ctx = zend_tools::ToolContext::with_workspace(d.path());
        let shown = zend_tools::run(
            "file_list",
            "test",
            &serde_json::json!({"prefix": "k/"}),
            &ctx,
        );
        let listed_by_tool: Vec<&str> = shown["files"]
            .as_array()
            .expect("files array")
            .iter()
            .map(|f| f["path"].as_str().expect("path"))
            .collect();
        assert_eq!(
            listed_by_tool,
            vec!["k/LICENSE", "k/api.rs", "k/decode.cu"],
            "the turn the model reads lists all three — the two sets still differ",
        );
    }

    /// A directory unit exists only where files actually sit — an intermediate
    /// directory that merely contains other directories gets none, so the layer
    /// never carries a folder with nothing of its own to describe.
    #[test]
    fn an_intermediate_directory_with_no_files_gets_no_unit() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let units = build_units(&map(&["a/b/c/y.rs"]));
        let dirs: Vec<&str> = units.iter().map(|u| u.dir.as_str()).collect();
        assert_eq!(dirs, vec!["a/b/c/"]);
    }

    #[test]
    fn a_removed_directory_is_reported() {
        // Held, not used: `build_units` reads the map, but the workspace must
        // outlive the call or the TempDir is removed under it.
        let _d = empty_workspace();
        let before = build_units(&map(&["a/x.rs", "b/y.rs"]));
        let after = build_units(&map(&["a/x.rs"]));
        let st = DirState::from_units(&before);
        assert_eq!(st.changed_dirs(&after), vec!["b/".to_string()]);
    }

    /// **Editing a file's CONTENT leaves the hash alone**, now that the turn
    /// shows only the listing.
    ///
    /// It used to move the hash, because the unit read an anchor file and the
    /// excerpt was hashed with the names. With the read gone the excerpt is not
    /// evidence any more, and re-summarising a folder whose listing is byte-for
    /// -byte identical would decode the same sentence at full cost — on this
    /// workspace, every folder whose README was touched.
    #[test]
    fn editing_a_files_content_leaves_the_hash_alone() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("a")).unwrap();
        std::fs::write(d.path().join("a/mod.rs"), "//! One.\n//! Two.\nfn x() {}\n").unwrap();
        let m = map(&["a/mod.rs"]);
        let before = build_units(&m);

        std::fs::write(
            d.path().join("a/mod.rs"),
            "//! Changed.\n//! Two.\nfn x() {}\n",
        )
        .unwrap();
        let after = build_units(&m);
        assert_eq!(before[0].content_hash, after[0].content_hash);
    }

    /// A rewritten README leaves the hash alone too — it is a file like any
    /// other now, named in the listing and never shown.
    #[test]
    fn rewriting_a_readme_leaves_the_hash_alone() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("a")).unwrap();
        std::fs::write(d.path().join("a/x.rs"), "fn x() {}\n").unwrap();
        std::fs::write(d.path().join("a/README.md"), "# a\n\nHolds the widgets.\n").unwrap();
        let readme = FileEntry {
            language: Language::Markdown,
            ..entry("a/README.md")
        };
        let m = RepoMap {
            files: vec![readme, entry("a/x.rs")],
            ..Default::default()
        };
        let before = build_units(&m);

        std::fs::write(
            d.path().join("a/README.md"),
            "# a\n\nHolds the gadgets now.\n",
        )
        .unwrap();
        let after = build_units(&m);
        assert_eq!(before[0].content_hash, after[0].content_hash);
    }

    /// An edit elsewhere in the directory does not — the unit never showed it,
    /// so its summary is still accurate and re-decoding would cost for nothing.
    #[test]
    fn editing_an_unshown_file_leaves_the_hash_alone() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("a")).unwrap();
        std::fs::write(d.path().join("a/mod.rs"), "//! One.\n//! Two.\nfn x() {}\n").unwrap();
        std::fs::write(d.path().join("a/other.rs"), "fn a() {}\n").unwrap();
        let m = map(&["a/mod.rs", "a/other.rs"]);
        let before = build_units(&m);

        std::fs::write(d.path().join("a/other.rs"), "fn b() {}\n").unwrap();
        let after = build_units(&m);
        assert_eq!(before[0].content_hash, after[0].content_hash);
    }
}
