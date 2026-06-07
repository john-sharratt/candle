//! `ignore`-driven workspace enumeration plus per-file metadata.
//!
//! `walk_workspace` is a pure function — given a workspace root it
//! returns a [`RepoMap`].  The walker respects every ignore file
//! `ripgrep` does (`.gitignore`, `.ignore`, the global git ignore, and
//! `.git/info/exclude`).  Hidden files and symlinks are skipped by
//! default.

use std::fs;
use std::path::Path;

use ignore::WalkBuilder;

use super::types::{FileEntry, Language, ModuleHint, RepoMap};

/// Hard size ceiling for any single file the walker accepts.  Above
/// this we silently skip; values are bounded by RAM during the read
/// pass and by prefill cost during scope carving.  16 MB
/// accommodates very large source files (e.g. generated parsers,
/// vendored single-file libraries, large markdown / asciidoc
/// reference docs) and substantial design documents without
/// admitting accidentally-checked-in binary blobs (which are
/// typically far larger).
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Walk `root` and produce a [`RepoMap`].  Never panics — I/O errors
/// during the walk are downgraded to skips and reported via the
/// `files_skipped_*` counters on the returned map.
pub fn walk_workspace(root: &Path) -> RepoMap {
    let mut map = RepoMap::default();

    let walker = WalkBuilder::new(root)
        .hidden(true) // skip dotfiles
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true) // honour .ignore
        .require_git(false) // honour .gitignore even outside a git repo
        .follow_links(false)
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        // Directories themselves don't contribute entries; only files do.
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        // Always exclude the daemon's own outputs.
        if path
            .components()
            .any(|c| c.as_os_str() == ".zend" || c.as_os_str() == ".substrate")
        {
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

        let (line_count, module_hint) = describe_file(path, language, size_bytes);
        map.files.push(FileEntry {
            path: rel_normalised,
            line_count,
            language,
            size_bytes,
            module_hint,
        });
    }

    map.files.sort_by(|a, b| a.path.cmp(&b.path));
    map
}

/// Count lines + extract any manifest hint.  Returns `(line_count, hint)`.
fn describe_file(
    path: &Path,
    language: Language,
    size_bytes: u64,
) -> (u32, Option<ModuleHint>) {
    // Single read — we already know the file is ≤ 256 KB so the
    // allocation is bounded.
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return (0, None),
    };
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

    let module_hint = manifest_hint(path, &bytes, language);
    let _ = size_bytes;
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

    fn fixture(name: &str) -> tempfile::TempDir {
        let _ = name;
        tempfile::tempdir().expect("tempdir")
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
        let root = dir.path().to_path_buf();
        write(&root, ".gitignore", b"target/\nignored.rs\n");
        write(&root, "src/lib.rs", b"// keep\n");
        write(&root, "target/junk.rs", b"// drop\n");
        write(&root, "ignored.rs", b"// drop\n");

        let map = walk_workspace(&root);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"src/lib.rs"));
        assert!(!paths.iter().any(|p| p.starts_with("target/")));
        assert!(!paths.contains(&"ignored.rs"));
    }

    #[test]
    fn walk_filters_by_extension_allowlist() {
        let dir = fixture("ext");
        let root = dir.path().to_path_buf();
        write(&root, "src/lib.rs", b"// keep\n");
        write(&root, "data/blob.bin", b"\x00\x01\x02");
        write(&root, "README.md", b"# title\n");
        write(&root, "shape.svg", b"<svg/>");

        let map = walk_workspace(&root);
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
        let root = dir.path().to_path_buf();
        write(&root, "tiny.rs", b"// small\n");
        // 17 MB > MAX_FILE_BYTES (16 MB).
        let big: Vec<u8> = vec![b'a'; 17 * 1024 * 1024];
        write(&root, "huge.rs", &big);

        let map = walk_workspace(&root);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"tiny.rs"));
        assert!(!paths.contains(&"huge.rs"));
        assert_eq!(map.files_skipped_oversize, 1);
    }

    #[test]
    fn walk_accepts_files_just_under_size_cap() {
        let dir = fixture("just_under");
        let root = dir.path().to_path_buf();
        // 12 MB — comfortably under the 16 MB cap.  This file should
        // survive the walk; oversize counters stay at zero.
        let body: Vec<u8> = vec![b'a'; 12 * 1024 * 1024];
        write(&root, "doc.md", &body);
        let map = walk_workspace(&root);
        assert!(map.files.iter().any(|f| f.path == "doc.md"));
        assert_eq!(map.files_skipped_oversize, 0);
    }

    #[test]
    fn walk_excludes_zend_dir() {
        let dir = fixture("zend");
        let root = dir.path().to_path_buf();
        write(&root, "src/lib.rs", b"// keep\n");
        write(&root, ".zend/config.yaml", b"x: 1\n");
        write(&root, ".substrate/something.log", b"x");

        let map = walk_workspace(&root);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"src/lib.rs"));
        assert!(!paths.iter().any(|p| p.starts_with(".zend/")));
        assert!(!paths.iter().any(|p| p.starts_with(".substrate/")));
    }

    #[test]
    fn walk_metadata_counts_lines_correctly() {
        let dir = fixture("lines");
        let root = dir.path().to_path_buf();
        write(&root, "trailing_nl.rs", b"a\nb\nc\n"); // 3 lines, trailing NL
        write(&root, "no_trailing.rs", b"a\nb\nc"); // 3 lines, no trailing NL
        write(&root, "empty.rs", b""); // 0 lines
        write(&root, "single.rs", b"hello"); // 1 line, no NL

        let map = walk_workspace(&root);
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
        let root = dir.path().to_path_buf();
        write(
            &root,
            "Cargo.toml",
            br#"[workspace]
members = ["a", "b", "c"]
"#,
        );

        let map = walk_workspace(&root);
        let entry = map.files.iter().find(|f| f.path == "Cargo.toml").unwrap();
        assert_eq!(
            entry.module_hint,
            Some(ModuleHint::CargoWorkspace { members: 3 })
        );
    }

    #[test]
    fn walk_extracts_cargo_package_hint() {
        let dir = fixture("cargo_pkg");
        let root = dir.path().to_path_buf();
        write(
            &root,
            "Cargo.toml",
            br#"[package]
name = "my-crate"
version = "0.1.0"
"#,
        );

        let map = walk_workspace(&root);
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
        let root = dir.path().to_path_buf();
        write(&root, "package.json", br#"{"name":"my-app","version":"1.0.0"}"#);

        let map = walk_workspace(&root);
        let entry = map
            .files
            .iter()
            .find(|f| f.path == "package.json")
            .unwrap();
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
        let root = dir.path().to_path_buf();
        write(
            &root,
            "go.mod",
            b"module example.com/me/widget\ngo 1.22\n",
        );

        let map = walk_workspace(&root);
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
        let root = dir.path().to_path_buf();
        write(&root, "z/last.rs", b"//\n");
        write(&root, "a/first.rs", b"//\n");
        write(&root, "m/middle.rs", b"//\n");

        let map = walk_workspace(&root);
        let paths: Vec<&str> = map.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["a/first.rs", "m/middle.rs", "z/last.rs"]);
    }
}
