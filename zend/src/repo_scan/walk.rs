//! `ignore`-driven workspace enumeration plus per-file metadata.
//!
//! `walk_workspace` is a pure function — given a [`Workspace`] it returns a
//! [`RepoMap`] of every repository's files, each keyed by its path relative to
//! the workspace folder (`candle/src/lib.rs`). Only the listed repositories are
//! walked: anything else in the workspace folder — other checkouts, the
//! daemon's `substrate/` — is never visited. The walker respects every ignore
//! file `ripgrep` does (`.gitignore`, `.ignore`, the global git ignore, and
//! `.git/info/exclude`). Hidden files and symlinks are skipped by default.

use std::fs;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use zend_vfs::Workspace;

use super::binary_sniff::is_binary_sample;
use super::types::{FileEntry, Language, ModuleHint, RepoMap};
use crate::code_read::is_upload_path;
use crate::repo_path::split;

/// Hard size ceiling for any single file the walker accepts.  Above
/// this we silently skip; values are bounded by RAM during the read
/// pass and by prefill cost during scope carving.  16 MB
/// accommodates very large source files (e.g. generated parsers,
/// vendored single-file libraries, large markdown / asciidoc
/// reference docs) and substantial design documents without
/// admitting accidentally-checked-in binary blobs (which are
/// typically far larger).
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Walk `workspace`'s repositories and produce a [`RepoMap`]. Never panics —
/// I/O errors during the walk are downgraded to skips and reported via the
/// `files_skipped_*` counters on the returned map.
///
/// `scope` narrows the walk to one workspace-relative folder inside a
/// repository (`--ingest-dir`, e.g. `candle/zend/src`); empty walks every
/// repository. Keys stay workspace-relative either way. A scope outside every
/// repository walks nothing. The daemon's `uploads` repository is never walked
/// — see [`is_upload_path`].
///
/// `max_depth` (`--max-depth`) bounds the walk in path components below each
/// walk's start — a repository's root, or the scope folder: `1` is its own
/// files, `2` adds one folder down. Nothing deeper is visited, and the map
/// records the bound in workspace-relative components so the deleted-path
/// sweeps can tell "not walked" from "gone" ([`RepoMap::is_frozen_file`]).
pub fn walk_workspace(workspace: &Workspace, scope: &str, max_depth: Option<usize>) -> RepoMap {
    let root = workspace.root();
    let scope = scope.trim_matches('/');
    // Each walk's start, and how many workspace-relative components deep it is.
    let starts: Vec<(PathBuf, usize)> = if scope.is_empty() || scope == "." {
        workspace
            .repos()
            .iter()
            .filter(|r| !is_upload_path(&r.name))
            .map(|r| (r.dir.clone(), 1))
            .collect()
    } else {
        let (repo, _) = split(scope);
        match workspace.repo(repo) {
            Some(_) if !is_upload_path(repo) => {
                vec![(root.join(scope), scope.split('/').count())]
            }
            _ => Vec::new(),
        }
    };
    // One offset for the whole map: every start of an unscoped walk is a
    // repository root, one component deep.
    let offset = starts.first().map_or(1, |(_, d)| *d);
    let mut map = RepoMap {
        max_depth: max_depth.map(|d| d + offset),
        ..RepoMap::default()
    };
    for (start, _) in &starts {
        walk_one(root, start, max_depth, &mut map);
    }
    map.files.sort_by(|a, b| a.path.cmp(&b.path));
    map
}

/// Walk `start` (inside `root`) into `map`, keying each file relative to `root`.
fn walk_one(root: &Path, start: &Path, max_depth: Option<usize>, map: &mut RepoMap) {
    let walker = WalkBuilder::new(start)
        .max_depth(max_depth)
        .hidden(true) // skip dotfiles
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true) // honour .ignore
        .require_git(false) // honour .gitignore even outside a git repo
        .follow_links(false)
        // Prune nested git repositories / submodules. Any directory below the
        // walk's start (a repository's own `.git` sits AT the start, depth 0,
        // and is not pruned) that holds a `.git` entry (a submodule uses a `.git` FILE
        // pointing into the superproject's modules dir; a nested clone a `.git`
        // DIR) is a SEPARATE project — its contents are vendored third-party
        // code and generated artifacts (e.g. the cutlass submodule's thousands
        // of Doxygen `.html` files), not part of THIS workspace. The parent's
        // `.gitignore` never lists a tracked submodule, so this is the only gate
        // that stops the walk descending into it. Pruning at the directory skips
        // the whole subtree in one stat, keeping the scan fast and the repo_map
        // free of foreign trees.
        .filter_entry(|entry| {
            if entry.depth() > 0
                && entry.file_type().is_some_and(|t| t.is_dir())
                && entry.path().join(".git").exists()
            {
                tracing::trace!(
                    dir = %entry.path().display(),
                    "repo walk: skipping nested git repo / submodule"
                );
                return false;
            }
            true
        })
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        // Directories themselves don't contribute entries; only files do.
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        map.files_scanned += 1;

        let Some(rel) = path.strip_prefix(root).ok().and_then(|p| p.to_str()) else {
            continue;
        };
        let rel_normalised = rel.replace('\\', "/");

        // Extension allowlist (case-insensitive), with a basename
        // override for files whose meaningful "extension" isn't on
        // the standard list — `go.mod` and `go.sum` are the
        // common ones.
        let basename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let language = match basename {
            "go.mod" | "go.sum" => Some(Language::Go),
            _ => path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .as_deref()
                .and_then(Language::from_extension),
        };
        let Some(language) = language else {
            map.files_skipped_extension += 1;
            continue;
        };

        let metadata = match fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let size_bytes = metadata.len();
        if size_bytes > MAX_FILE_BYTES {
            map.files_skipped_oversize += 1;
            continue;
        }
        // Read the file ONCE — both the content guard and the metadata scan below
        // work from these bytes, so there is no second open / overlapping re-read.
        // A file that stat'd but won't read (a delete/permission race between the
        // two) is skipped this pass and reconsidered on the next walk.
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        // Content guard: an allowlisted extension does NOT guarantee text. A
        // compiled fatbin / object dump checked in as `*.txt` clears both the
        // extension gate (`.txt` ⇒ PlainText) and the size gate (well under
        // `MAX_FILE_BYTES`), then carves into hundreds of garbage scopes that
        // blow the ingest co-batch's VRAM budget. `is_binary_sample` bails at the
        // first few NULs, so a blob is rejected almost immediately and never
        // touches the carve; real source is never mistaken for it.
        if is_binary_sample(&bytes) {
            map.files_skipped_binary += 1;
            continue;
        }

        let (line_count, module_hint) = describe_file(path, &bytes, language);
        map.files.push(FileEntry {
            path: rel_normalised,
            line_count,
            language,
            size_bytes,
            module_hint,
        });
    }
}

/// Count lines + extract any manifest hint from a file's already-read `bytes`.
/// Returns `(line_count, hint)`.
fn describe_file(path: &Path, bytes: &[u8], language: Language) -> (u32, Option<ModuleHint>) {
    let line_count = if bytes.is_empty() {
        0
    } else {
        let nl = bytes.iter().filter(|&&b| b == b'\n').count() as u32;
        // Files without a trailing newline still have one logical line
        // for the last (un-terminated) line.
        if bytes.last() == Some(&b'\n') {
            nl
        } else {
            nl + 1
        }
    };

    let module_hint = manifest_hint(path, bytes, language);
    (line_count, module_hint)
}

/// Pick up workspace / package / module metadata from manifest files.
/// Best-effort — failure to parse is silently treated as "no hint".
fn manifest_hint(path: &Path, bytes: &[u8], language: Language) -> Option<ModuleHint> {
    let file_name = path.file_name().and_then(|n| n.to_str())?;
    let body = std::str::from_utf8(bytes).ok()?;
    match (file_name, language) {
        ("Cargo.toml", Language::Toml) => cargo_hint(body),
        ("package.json", Language::Json) => node_hint(body),
        ("pyproject.toml", Language::Toml) => pyproject_hint(body),
        ("go.mod", _) => go_mod_hint(body),
        _ => None,
    }
}

fn cargo_hint(body: &str) -> Option<ModuleHint> {
    // Workspace detection: a `[workspace]` table whose `members` array
    // we can count.  We don't pull in toml::Value here — a tiny
    // hand-roll keeps the dep tree slim and is sufficient for the
    // common shapes Cargo emits.
    if let Some(members) = parse_workspace_members(body) {
        return Some(ModuleHint::CargoWorkspace { members });
    }
    if let Some(name) = parse_cargo_package_name(body) {
        return Some(ModuleHint::CargoPackage { name });
    }
    None
}

fn parse_workspace_members(body: &str) -> Option<usize> {
    let ws_start = body.find("[workspace]")?;
    let after = &body[ws_start + "[workspace]".len()..];
    let members_idx = after.find("members")?;
    let after_members = &after[members_idx..];
    let array_start = after_members.find('[')?;
    let array_end = after_members[array_start..].find(']')?;
    let inside = &after_members[array_start + 1..array_start + array_end];
    let count = inside
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .filter(|s| !s.is_empty())
        .count();
    Some(count)
}

fn parse_cargo_package_name(body: &str) -> Option<String> {
    let pkg_start = body.find("[package]")?;
    let after = &body[pkg_start + "[package]".len()..];
    // Find a `name = "..."` line before the next `[` table header.
    let next_table = after.find('[').unwrap_or(after.len());
    let section = &after[..next_table];
    for line in section.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("name") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let rest = rest.trim();
                if let Some(stripped) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                    return Some(stripped.to_string());
                }
            }
        }
    }
    None
}

fn node_hint(body: &str) -> Option<ModuleHint> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    Some(ModuleHint::NodePackage { name })
}

fn pyproject_hint(body: &str) -> Option<ModuleHint> {
    let project_idx = body.find("[project]")?;
    let after = &body[project_idx + "[project]".len()..];
    let next_table = after.find('[').unwrap_or(after.len());
    let section = &after[..next_table];
    for line in section.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("name") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let rest = rest.trim();
                if let Some(stripped) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                    return Some(ModuleHint::PythonProject {
                        name: stripped.to_string(),
                    });
                }
            }
        }
    }
    None
}

fn go_mod_hint(body: &str) -> Option<ModuleHint> {
    for line in body.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("module") {
            let name = rest.trim();
            if !name.is_empty() {
                return Some(ModuleHint::GoModule {
                    name: name.to_string(),
                });
            }
        }
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    use zend_vfs::RepoSpec;

    /// The repository every single-repository test below lays its files out in.
    const REPO: &str = "r";

    fn fixture(name: &str) -> tempfile::TempDir {
        let _ = name;
        tempfile::tempdir().expect("tempdir")
    }

    /// `dir` as a workspace holding the listed repositories.
    fn workspace_of(dir: &tempfile::TempDir, repos: &[&str]) -> Workspace {
        for repo in repos {
            fs::create_dir_all(dir.path().join(repo)).unwrap();
        }
        Workspace::new(
            dir.path(),
            repos.iter().map(|r| RepoSpec::named(r)).collect(),
        )
        .unwrap()
    }

    /// Walk the single-repository workspace in `dir`, with each key's `r/`
    /// prefix dropped so a test asserts the path inside the repository. The
    /// prefix itself is asserted by the multi-repository tests.
    fn walk(dir: &tempfile::TempDir, max_depth: Option<usize>) -> RepoMap {
        let mut map = walk_workspace(&workspace_of(dir, &[REPO]), "", max_depth);
        for f in &mut map.files {
            f.path = f
                .path
                .strip_prefix("r/")
                .expect("every key starts with its repository")
                .to_string();
        }
        map
    }

    /// **A repository's git database is never ingested.**
    ///
    /// `.hidden(true)` is what excludes it, which is easy to change for an
    /// unrelated reason and would silently pull the whole object store into
    /// the corpus. Two things make that worth a test of its own rather than
    /// trust in a flag: `.git/config` holds remote URLs with their credentials
    /// intact, and the packed-refs and object files are megabytes of content
    /// that answer no question a developer asks. `VfsStore` refuses the same
    /// paths on an explicit read; this covers the walk.
    #[test]
    fn a_repositorys_git_database_is_never_ingested() {
        let dir = fixture("git_dir");
        let root = dir.path().join(REPO);
        write(&root, "src/lib.rs", b"pub fn ok() {}\n");
        write(
            &root,
            ".git/config",
            b"[remote \"origin\"]\n\turl = https://u:ghp_secrettoken@example.com/a.git\n",
        );
        write(&root, ".git/packed-refs", b"abc refs/heads/main\n");
        write(&root, ".git/objects/ab/cdef", b"binary-ish\n");

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"src/lib.rs"), "{paths:?}");
        for p in &paths {
            assert!(!p.starts_with(".git"), "the git database was ingested: {p}");
        }
        assert!(
            !format!("{map:?}").contains("ghp_secrettoken"),
            "a credential from .git/config reached the repo map"
        );
    }

    /// `--max-depth` counts path components below each repository's root: 1
    /// is the repository's own files, 2 adds one folder down. Nothing past the
    /// bound is walked, and the map records the bound in workspace-relative
    /// components — one more, for the repository segment — so the sweeps can
    /// freeze what lies beyond it.
    #[test]
    fn walk_stops_at_max_depth() {
        let dir = fixture("max_depth");
        let root = dir.path().join(REPO);
        write(&root, "a.rs", b"fn a() {}\n");
        write(&root, "src/b.rs", b"fn b() {}\n");
        write(&root, "src/deep/c.rs", b"fn c() {}\n");

        let paths = |m: &RepoMap| m.files.iter().map(|f| f.path.clone()).collect::<Vec<_>>();
        assert_eq!(paths(&walk(&dir, Some(1))), vec!["a.rs"]);
        let bounded = walk(&dir, Some(2));
        assert_eq!(paths(&bounded), vec!["a.rs", "src/b.rs"]);
        assert_eq!(bounded.max_depth, Some(3));
        assert!(!bounded.is_frozen_file("r/src/b.rs"));
        assert!(bounded.is_frozen_file("r/src/deep/c.rs"));
        let open = walk(&dir, None);
        assert_eq!(paths(&open), vec!["a.rs", "src/b.rs", "src/deep/c.rs"]);
        assert_eq!(open.max_depth, None);
    }

    /// **Every listed repository is walked, and nothing else is.** Keys are
    /// workspace-relative, so the same inner path in two repositories is two
    /// keys; a folder beside the repositories — another checkout, the daemon's
    /// `substrate/` — never appears.
    #[test]
    fn every_listed_repository_is_walked_and_keyed_by_it() {
        let dir = fixture("multi");
        let ws = workspace_of(&dir, &["alpha", "beta"]);
        write(&dir.path().join("alpha"), "src/lib.rs", b"// a\n");
        write(&dir.path().join("beta"), "src/lib.rs", b"// b\n");
        write(dir.path(), "other/src/lib.rs", b"// not listed\n");
        write(dir.path(), "substrate/zend.log", b"x\n");
        write(dir.path(), "top.rs", b"// beside the repositories\n");

        let map = walk_workspace(&ws, "", None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["alpha/src/lib.rs", "beta/src/lib.rs"]);
    }

    /// **A repository's own `.git` does not prune it.** Each repository is
    /// its walk's start, so its `.git` sits at depth 0.
    #[test]
    fn a_repository_that_is_a_git_checkout_is_walked() {
        let dir = fixture("checkout");
        let ws = workspace_of(&dir, &["alpha"]);
        write(
            &dir.path().join("alpha"),
            ".git/HEAD",
            b"ref: refs/heads/main\n",
        );
        write(&dir.path().join("alpha"), "src/lib.rs", b"// keep\n");
        let map = walk_workspace(&ws, "", None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["alpha/src/lib.rs"]);
    }

    /// A scope narrows the walk to one folder and keeps keys workspace-relative;
    /// the depth bound counts from the scope folder.
    #[test]
    fn a_scope_walks_one_folder_inside_a_repository() {
        let dir = fixture("scope");
        let ws = workspace_of(&dir, &["alpha", "beta"]);
        write(&dir.path().join("alpha"), "src/lib.rs", b"// in\n");
        write(&dir.path().join("alpha"), "src/deep/x.rs", b"// deeper\n");
        write(&dir.path().join("alpha"), "docs/a.md", b"out\n");
        write(&dir.path().join("beta"), "src/lib.rs", b"// out\n");

        let map = walk_workspace(&ws, "alpha/src", None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["alpha/src/deep/x.rs", "alpha/src/lib.rs"]);

        let bounded = walk_workspace(&ws, "alpha/src/", Some(1));
        let paths: Vec<&str> = bounded.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["alpha/src/lib.rs"]);
        assert_eq!(bounded.max_depth, Some(3));

        assert!(walk_workspace(&ws, "other/src", None).files.is_empty());
    }

    /// The daemon's `uploads` repository is endpoint-managed and never walked.
    #[test]
    fn the_uploads_repository_is_never_walked() {
        let dir = fixture("uploads_repo");
        let ws = workspace_of(&dir, &["alpha", "uploads"]);
        write(&dir.path().join("alpha"), "a.rs", b"// keep\n");
        write(&dir.path().join("uploads"), "notes.py", b"print(1)\n");
        let map = walk_workspace(&ws, "", None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["alpha/a.rs"]);
        assert!(walk_workspace(&ws, "uploads", None).files.is_empty());
    }

    fn write(root: &Path, rel: &str, body: &[u8]) {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    #[test]
    fn walk_respects_gitignore() {
        let dir = fixture("gitignore");
        let root = dir.path().join(REPO);
        write(&root, ".gitignore", b"target/\nignored.rs\n");
        write(&root, "src/lib.rs", b"// keep\n");
        write(&root, "target/junk.rs", b"// drop\n");
        write(&root, "ignored.rs", b"// drop\n");

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"src/lib.rs"));
        assert!(!paths.iter().any(|p| p.starts_with("target/")));
        assert!(!paths.contains(&"ignored.rs"));
    }

    #[test]
    fn walk_prunes_nested_git_repos_and_submodules() {
        let dir = fixture("submodule");
        let root = dir.path().join(REPO);
        // The repository is itself a git repo (`.git` DIR) — the depth-0
        // guard must NOT prune it, or nothing would scan.
        write(&root, ".git/HEAD", b"ref: refs/heads/main\n");
        // The workspace's own source is kept.
        write(&root, "src/lib.rs", b"// keep\n");
        // A submodule: a nested dir marked by a `.git` FILE (gitlink), holding
        // vendored source and generated docs. None of it must be scanned.
        write(
            &root,
            "vendor/cutlass/.git",
            b"gitdir: ../.git/modules/cutlass\n",
        );
        write(&root, "vendor/cutlass/include/gemm.h", b"// vendored\n");
        write(&root, "vendor/cutlass/docs/index.html", b"<html></html>\n");
        // A nested clone: marked by a `.git` DIR. Also pruned.
        write(&root, "nested/.git/HEAD", b"ref: refs/heads/main\n");
        write(&root, "nested/main.rs", b"// separate project\n");

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"src/lib.rs"));
        assert!(
            !paths.iter().any(|p| p.starts_with("vendor/cutlass/")),
            "submodule subtree must be pruned: {paths:?}"
        );
        assert!(
            !paths.iter().any(|p| p.starts_with("nested/")),
            "nested git repo must be pruned: {paths:?}"
        );
    }

    #[test]
    fn walk_filters_by_extension_allowlist() {
        let dir = fixture("ext");
        let root = dir.path().join(REPO);
        write(&root, "src/lib.rs", b"// keep\n");
        write(&root, "data/blob.bin", b"\x00\x01\x02");
        write(&root, "README.md", b"# title\n");
        write(&root, "shape.svg", b"<svg/>");

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"src/lib.rs"));
        assert!(paths.contains(&"README.md"));
        assert!(!paths.contains(&"data/blob.bin"));
        assert!(!paths.contains(&"shape.svg"));
        assert!(map.files_skipped_extension >= 2);
    }

    #[test]
    fn walk_skips_oversize_files() {
        let dir = fixture("oversize");
        let root = dir.path().join(REPO);
        write(&root, "tiny.rs", b"// small\n");
        // 17 MB > MAX_FILE_BYTES (16 MB).
        let big: Vec<u8> = vec![b'a'; 17 * 1024 * 1024];
        write(&root, "huge.rs", &big);

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"tiny.rs"));
        assert!(!paths.contains(&"huge.rs"));
        assert_eq!(map.files_skipped_oversize, 1);
    }

    #[test]
    fn walk_skips_binary_content_despite_allowlisted_extension() {
        let dir = fixture("binary_txt");
        let root = dir.path().join(REPO);
        // A genuine text file with an allowlisted extension — kept.
        write(&root, "notes.txt", b"plain prose, no NULs here\n");
        // A compiled fatbin dump checked in as `*.txt`: allowlisted extension,
        // well under the size cap, but a NUL in the first bytes ⇒ binary. This
        // is the exact shape of `candle-flash-attn/precompiled/*.txt`.
        write(
            &root,
            "precompiled/hdim64_sass.txt",
            b"\x7fELF\x00\x00fatbin\x00code",
        );

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"notes.txt"), "real text kept: {paths:?}");
        assert!(
            !paths.contains(&"precompiled/hdim64_sass.txt"),
            "binary-as-.txt must be skipped: {paths:?}"
        );
        assert_eq!(map.files_skipped_binary, 1);
    }

    #[test]
    fn walk_accepts_files_just_under_size_cap() {
        let dir = fixture("just_under");
        let root = dir.path().join(REPO);
        // 12 MB — comfortably under the 16 MB cap.  This file should
        // survive the walk; oversize counters stay at zero.
        let body: Vec<u8> = vec![b'a'; 12 * 1024 * 1024];
        write(&root, "doc.md", &body);
        let map = walk(&dir, None);
        assert!(map.files.iter().any(|f| f.path == "doc.md"));
        assert_eq!(map.files_skipped_oversize, 0);
    }

    /// Hidden folders inside a repository — an editor's or a tool's own
    /// state — are never walked.
    #[test]
    fn walk_excludes_hidden_dirs() {
        let dir = fixture("hidden");
        let root = dir.path().join(REPO);
        write(&root, "src/lib.rs", b"// keep\n");
        write(&root, ".zend/config.yaml", b"x: 1\n");

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["src/lib.rs"]);
    }

    /// A folder named `uploads` inside a repository is the repository's own
    /// source, not the daemon's uploads.
    #[test]
    fn an_uploads_folder_inside_a_repository_is_walked() {
        let dir = fixture("uploads");
        let root = dir.path().join(REPO);
        write(&root, "src/main.rs", b"// keep\n");
        write(&root, "uploads/notes.py", b"print(1)\n");
        write(&root, "src/uploads/real.rs", b"// keep\n");

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["src/main.rs", "src/uploads/real.rs", "uploads/notes.py"]
        );
    }

    #[test]
    fn walk_metadata_counts_lines_correctly() {
        let dir = fixture("lines");
        let root = dir.path().join(REPO);
        write(&root, "trailing_nl.rs", b"a\nb\nc\n"); // 3 lines, trailing NL
        write(&root, "no_trailing.rs", b"a\nb\nc"); // 3 lines, no trailing NL
        write(&root, "empty.rs", b""); // 0 lines
        write(&root, "single.rs", b"hello"); // 1 line, no NL

        let map = walk(&dir, None);
        let by_name: std::collections::HashMap<&str, u32> = map
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.line_count))
            .collect();
        assert_eq!(by_name["trailing_nl.rs"], 3);
        assert_eq!(by_name["no_trailing.rs"], 3);
        assert_eq!(by_name["empty.rs"], 0);
        assert_eq!(by_name["single.rs"], 1);
    }

    #[test]
    fn walk_extracts_cargo_workspace_hint() {
        let dir = fixture("cargo_ws");
        let root = dir.path().join(REPO);
        write(
            &root,
            "Cargo.toml",
            br#"[workspace]
members = ["a", "b", "c"]
"#,
        );

        let map = walk(&dir, None);
        let entry = map.files.iter().find(|f| f.path == "Cargo.toml").unwrap();
        assert_eq!(
            entry.module_hint,
            Some(ModuleHint::CargoWorkspace { members: 3 })
        );
    }

    #[test]
    fn walk_extracts_cargo_package_hint() {
        let dir = fixture("cargo_pkg");
        let root = dir.path().join(REPO);
        write(
            &root,
            "Cargo.toml",
            br#"[package]
name = "my-crate"
version = "0.1.0"
"#,
        );

        let map = walk(&dir, None);
        let entry = map.files.iter().find(|f| f.path == "Cargo.toml").unwrap();
        assert_eq!(
            entry.module_hint,
            Some(ModuleHint::CargoPackage {
                name: "my-crate".to_string()
            })
        );
    }

    #[test]
    fn walk_extracts_node_package_hint() {
        let dir = fixture("node");
        let root = dir.path().join(REPO);
        write(
            &root,
            "package.json",
            br#"{"name":"my-app","version":"1.0.0"}"#,
        );

        let map = walk(&dir, None);
        let entry = map.files.iter().find(|f| f.path == "package.json").unwrap();
        assert_eq!(
            entry.module_hint,
            Some(ModuleHint::NodePackage {
                name: "my-app".to_string()
            })
        );
    }

    #[test]
    fn walk_extracts_go_module_hint() {
        let dir = fixture("go");
        let root = dir.path().join(REPO);
        write(&root, "go.mod", b"module example.com/me/widget\ngo 1.22\n");

        let map = walk(&dir, None);
        let entry = map.files.iter().find(|f| f.path == "go.mod").unwrap();
        assert_eq!(
            entry.module_hint,
            Some(ModuleHint::GoModule {
                name: "example.com/me/widget".to_string()
            })
        );
    }

    #[test]
    fn walk_is_sorted_and_deterministic() {
        let dir = fixture("sort");
        let root = dir.path().join(REPO);
        write(&root, "z/last.rs", b"//\n");
        write(&root, "a/first.rs", b"//\n");
        write(&root, "m/middle.rs", b"//\n");

        let map = walk(&dir, None);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["a/first.rs", "m/middle.rs", "z/last.rs"]);
    }
}
