//! Deterministic construction of `mode: structural` summary nodes
//! (`docs/immutable_summary_forest.md`, the structural pipeline).
//!
//! For directory-tree content (the `repo_map` layer), the structure is fully
//! determined by the input — the model is never needed, at any tree level.
//! Decoding it is pure cost and invites fabrication (the model rambles
//! `/path/to/repo`, emits shell snippets), and a faithful *merge* of children
//! just unions their trees, so the skeleton grows toward the root instead of
//! compressing. So both levels are built here, deterministically, with no decode:
//!
//! - a **leaf** ([`leaf_skeleton`]) strips the trailing `(N lines, …)` / `(crate)` /
//!   `(module)` annotations off its one scan turn, keeping the full skeleton (a leaf
//!   is the base detail, so files are kept);
//! - a **summary-of-summaries** ([`structural_rollup`]) reconstructs the directory
//!   paths from the children — robust to both the full-path-header form
//!   `zend/src/code_read/` and the nested-indent form `zend/` → `  - examples/` —
//!   and **truncates each to a depth set by tree height** (`h=2` keeps three
//!   segments, `h=3` two, `h≥4` only the top-level directory), then deduplicates.
//!   Files drop (they live in the leaves at full fidelity), so a roll-up names
//!   *which directories exist*.
//!
//! The scope (user-half) is the distinct top-level directories. This mirrors the
//! tools categorize→assign split — the deterministic pass owns names — taken to its
//! conclusion: here the model is not consulted at all.

use std::collections::BTreeSet;

/// The skeleton + derived scope for a structural SoS node.
pub struct StructuralRollup {
    /// The user-half (scope): the distinct top-level directories, comma-separated.
    pub scope: String,
    /// The assistant-half (skeleton): the deduplicated directory paths, each
    /// truncated to the height's depth, one per line.
    pub skeleton: String,
}

/// Build a structural SoS from its children's skeletons. `children` are the
/// children's assistant-half skeletons (read from the substrate); `height` is the
/// node's tree height (a summary-of-summaries is always ≥ 2) and sets how deep
/// the directory paths are kept.
pub fn structural_rollup(children: &[String], height: u8) -> StructuralRollup {
    let depth = dir_depth_for_height(height);
    let mut dirs: BTreeSet<String> = BTreeSet::new();
    for child in children {
        for path in extract_dirs(child) {
            let truncated = truncate_path(&path, depth);
            if !truncated.is_empty() {
                dirs.insert(truncated);
            }
        }
    }
    let ordered: Vec<String> = dirs.into_iter().collect();
    let skeleton = ordered
        .iter()
        .map(|d| format!("{d}/"))
        .collect::<Vec<_>>()
        .join("\n");
    let scope = scope_from_dirs(&ordered);
    StructuralRollup { scope, skeleton }
}

/// Build a structural SoT *leaf* from one repository-scan turn. The scan is
/// already a directory listing; "compressing" it is just stripping the size/type
/// annotations each file line carries (`Cargo.toml (33 lines, TOML, …)`), keeping
/// the full skeleton — a leaf is the base detail, so files are kept and only the
/// annotations drop. No model decode. The scope is the distinct top-level
/// directories the scan covers.
pub fn leaf_skeleton(scan: &str) -> StructuralRollup {
    let skeleton = scan
        .lines()
        .map(strip_annotation)
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let scope = scope_from_dirs(&extract_dirs(&skeleton));
    StructuralRollup { scope, skeleton }
}

/// The distinct top-level directories (first path segment) across `dirs`, sorted
/// and comma-separated — the structural scope (user-half).
fn scope_from_dirs(dirs: &[String]) -> String {
    let tops: BTreeSet<&str> = dirs
        .iter()
        .filter_map(|d| d.split('/').next())
        .filter(|s| !s.is_empty())
        .collect();
    tops.into_iter().collect::<Vec<_>>().join(", ")
}

/// Strip a trailing metadata parenthetical from one scan line. The scan annotates
/// files with ` (N lines, …)` and directories with ` (crate)` / ` (module)`, both
/// at the end of the line — so any line ending in `)` has its last ` (…)` removed.
/// A name with no trailing parenthetical passes through untouched.
fn strip_annotation(line: &str) -> &str {
    if line.trim_end().ends_with(')') {
        if let Some(pos) = line.rfind(" (") {
            return line[..pos].trim_end();
        }
    }
    line
}

/// The deepest directory path (in path segments) a SoS at `height` keeps. Each
/// step up the tree drops one segment, so the roll-up is monotonically coarser
/// toward the root: `h=2` keeps three segments, `h=3` two, `h≥4` just the
/// top-level directory. Never below 1, so the top level always survives.
fn dir_depth_for_height(height: u8) -> usize {
    usize::from(5u8.saturating_sub(height)).max(1)
}

/// Reconstruct every directory's full path from one skeleton, robust to the two
/// forms the leaf compression emits: full-path headers (`zend/src/code_read/`)
/// and nested indentation (`zend/` then `  - examples/`). A line is a directory
/// iff its (trimmed) text ends with `/`; files are ignored.
///
/// The `stack` holds `(indent level, full path)` for the *real directory*
/// ancestors of the current line. Non-directory lines — files, and pseudo-roots
/// like `(workspace root)` that carry no trailing slash — are never pushed, so a
/// directory indented under one becomes a root rather than nesting beneath it
/// (the top-level crates listed under `(workspace root)` are themselves roots).
fn extract_dirs(skeleton: &str) -> Vec<String> {
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut dirs: Vec<String> = Vec::new();
    for line in skeleton.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let level = line_depth(line);
        let Some(name) = line_name(line) else {
            continue;
        };
        // Directories carry a trailing slash; files do not.
        if !line.trim_end().ends_with('/') {
            continue;
        }
        // Drop ancestors at this indent level or deeper (siblings, or directories
        // from a previous branch).
        while stack.last().is_some_and(|(l, _)| *l >= level) {
            stack.pop();
        }
        let full = match stack.last() {
            Some((_, parent)) => format!("{parent}/{name}"),
            None => name.to_string(),
        };
        dirs.push(full.clone());
        stack.push((level, full));
    }
    dirs
}

/// Keep the first `depth` non-empty segments of a `/`-separated path.
fn truncate_path(path: &str, depth: usize) -> String {
    path.split('/')
        .filter(|s| !s.is_empty())
        .take(depth)
        .collect::<Vec<_>>()
        .join("/")
}

/// A skeleton line's nesting level: leading indentation in two-space units.
fn line_depth(line: &str) -> usize {
    let indent = line.len() - line.trim_start().len();
    indent / 2
}

/// The name carried by a skeleton line: leading indentation and an optional
/// `- ` list marker stripped, plus a trailing `/`. `None` for a blank line.
fn line_name(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let t = t.strip_prefix("- ").unwrap_or(t);
    let t = t.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.trim_end_matches('/'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_dirs_handles_full_path_headers() {
        // The full-path-header form: each directory is a column-0 path, files
        // indented under it.
        let child = "zend/src/code_read/\n  - carve.rs\n  - mod.rs\nzend/tests/\n  - smoke.rs";
        assert_eq!(
            extract_dirs(child),
            vec!["zend/src/code_read".to_string(), "zend/tests".to_string()]
        );
    }

    #[test]
    fn extract_dirs_handles_nested_indentation() {
        // The nested form: a root directory, then deeper directories indented,
        // their full paths reconstructed from the indentation stack.
        let child =
            "zend/\n  - Cargo.toml\n  - examples/\n    - compress_tools.rs\n  - src/\n    - lib.rs";
        assert_eq!(
            extract_dirs(child),
            vec![
                "zend".to_string(),
                "zend/examples".to_string(),
                "zend/src".to_string(),
            ]
        );
    }

    #[test]
    fn extract_dirs_treats_workspace_root_children_as_roots() {
        // `(workspace root)` carries no trailing slash, so it is not a directory;
        // the top-level crates indented under it are roots, NOT nested beneath
        // each other or under a `(workspace root)` prefix.
        let child = "(workspace root)\n  - CHANGELOG.md\n  - candle-core/\n  - candle-nn/\n  - candle-transformers/";
        assert_eq!(
            extract_dirs(child),
            vec![
                "candle-core".to_string(),
                "candle-nn".to_string(),
                "candle-transformers".to_string(),
            ]
        );
    }

    #[test]
    fn leaf_strips_annotations_keeps_full_tree() {
        // File scan: directory header, files with `(N lines, …)`.
        let scan = "candle-wasm-examples/bert/\n  - Cargo.toml (33 lines, TOML, crate: candle-wasm-example-bert)\n  - README.md (26 lines, Markdown)\ncandle-wasm-examples/bert/src/\n  - lib.rs (12 lines, Rust)";
        let r = leaf_skeleton(scan);
        // Annotations stripped; every file and directory kept (leaf = base detail).
        assert_eq!(
            r.skeleton,
            "candle-wasm-examples/bert/\n  - Cargo.toml\n  - README.md\ncandle-wasm-examples/bert/src/\n  - lib.rs"
        );
        assert_eq!(r.scope, "candle-wasm-examples");
    }

    #[test]
    fn leaf_strips_crate_and_module_annotations() {
        // The workspace-root scan: `(workspace root)` pseudo-root, crates annotated
        // `(crate)`. Stripping makes the crates end in `/`, so they read as dirs.
        let scan =
            "(workspace root)\n  - CHANGELOG.md (113 lines, Markdown)\n  - candle-core/ (crate)\n  - candle-nn/ (crate)";
        let r = leaf_skeleton(scan);
        assert_eq!(
            r.skeleton,
            "(workspace root)\n  - CHANGELOG.md\n  - candle-core/\n  - candle-nn/"
        );
        assert_eq!(r.scope, "candle-core, candle-nn");
    }

    #[test]
    fn strip_annotation_leaves_unannotated_lines() {
        assert_eq!(strip_annotation("candle-core/"), "candle-core/");
        assert_eq!(strip_annotation("  - lib.rs"), "  - lib.rs");
        assert_eq!(strip_annotation("  - x.rs (5 lines, Rust)"), "  - x.rs");
        assert_eq!(
            strip_annotation("  - cpu_backend/ (module)"),
            "  - cpu_backend/"
        );
        // No trailing parenthetical (`.txt` is the line end) → untouched.
        assert_eq!(strip_annotation("  - a (b).txt"), "  - a (b).txt");
    }

    #[test]
    fn truncate_keeps_leading_segments() {
        assert_eq!(truncate_path("a/b/c/d", 2), "a/b");
        assert_eq!(truncate_path("a/b", 3), "a/b");
        assert_eq!(truncate_path("a", 1), "a");
    }

    #[test]
    fn dir_depth_shrinks_with_height() {
        assert_eq!(dir_depth_for_height(2), 3);
        assert_eq!(dir_depth_for_height(3), 2);
        assert_eq!(dir_depth_for_height(4), 1);
        assert_eq!(dir_depth_for_height(6), 1);
    }

    #[test]
    fn rollup_truncates_and_dedupes_by_height() {
        // Two leaf children whose deep trees overlap once truncated.
        let kids = vec![
            "candle-wasm-examples/whisper/\n  - main.js\ncandle-wasm-examples/whisper/src/\n  - app.rs\ncandle-wasm-examples/whisper/src/bin/\n  - m.rs".to_string(),
            "docs/\n  - x.md\ndocs/archived/\n  - y.md".to_string(),
        ];
        // height 2 → depth 3: the `src/bin` path collapses into `src`.
        let r2 = structural_rollup(&kids, 2);
        assert_eq!(
            r2.skeleton,
            "candle-wasm-examples/whisper/\ncandle-wasm-examples/whisper/src/\ndocs/\ndocs/archived/"
        );
        assert_eq!(r2.scope, "candle-wasm-examples, docs");

        // height 3 → depth 2: `whisper/src` collapses into `whisper`.
        let r3 = structural_rollup(&kids, 3);
        assert_eq!(
            r3.skeleton,
            "candle-wasm-examples/whisper/\ndocs/\ndocs/archived/"
        );

        // height 4 → depth 1: only the top-level directories remain.
        let r4 = structural_rollup(&kids, 4);
        assert_eq!(r4.skeleton, "candle-wasm-examples/\ndocs/");
        assert_eq!(r4.scope, "candle-wasm-examples, docs");
    }

    #[test]
    fn rollup_over_already_compact_children_stays_compact() {
        // At h=3 the children are h=2 SoS — already compact directory lists.
        let kids = vec![
            "candle-core/src/\ncandle-core/src/quantized/".to_string(),
            "candle-nn/src/kv_cache/".to_string(),
        ];
        let r = structural_rollup(&kids, 3);
        // depth 2: collapse to two segments, dedup.
        assert_eq!(r.skeleton, "candle-core/src/\ncandle-nn/src/");
        assert_eq!(r.scope, "candle-core, candle-nn");
    }
}
