//! Per-directory clustering of the walked [`RepoMap`].
//!
//! The `repo_map` conversation projects one turn pair per
//! **cluster**.  A cluster is a directory and (recursively) any
//! descendant directories that together fit under a soft token
//! budget — small leaf directories merge upward into their parent
//! so each turn carries a useful chunk of structural context rather
//! than a sliver.
//!
//! Each cluster carries a SHA-256 hash of the basenames it covers
//! so a file rename / add / delete in that subtree changes the
//! hash, letting the refresh path decide whether to re-prefill
//! atomically.

use sha2::{Digest, Sha256};

use super::types::{FileEntry, RepoMap};

/// Soft per-cluster size target.  At ~4 chars/token this is about
/// 300 tokens of listing text.  We don't tokenise the actual model
/// vocabulary here — a char-based heuristic is fast, deterministic,
/// and good enough for clustering: the projection budget is what
/// ultimately caps total layer cost.
pub const TARGET_CLUSTER_BYTES: usize = 1_200;

/// Hard ceiling on cluster listing bytes.  A single directory whose
/// listing on its own exceeds this stays as a standalone cluster (no
/// further merging upward) — splitting one directory across multiple
/// clusters would muddy the "what is in this directory" semantics.
pub const MAX_CLUSTER_BYTES: usize = 8_000;

/// One projection turn pair: covers one or more directories, carries
/// a stable identifier and a content hash for refresh decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    /// Workspace-relative root directory for this cluster
    /// (e.g. `src/auth/`).  The root dir is the cluster's canonical
    /// id — refresh logic keys on it.  Empty string for the repo
    /// root.
    pub root_dir: String,
    /// Every directory absorbed into this cluster, in declaration
    /// order (root first, then merged descendants in lexicographic
    /// order).  Used by the refresh path to know which directories
    /// invalidate this cluster.
    pub covered_dirs: Vec<String>,
    /// SHA-256 hex digest over the cluster's full basename sorted
    /// list — `directory/basename` lines for every file the cluster
    /// covers.  Stable across re-runs; changes only when files are
    /// added, removed, or renamed within `covered_dirs`.
    pub content_hash: String,
    /// User-side prompt for the projection turn — "Tell me what is
    /// in `{root_dir}`?" (or the workspace-root variant).
    pub user_prompt: String,
    /// Assistant-side listing text for the projection turn.  This is
    /// what gets prefilled as the conversation's understanding of
    /// the cluster's contents.
    pub listing: String,
}

/// Build the cluster list for `map`.  Deterministic — the same
/// `RepoMap` always produces the same clusters in the same order.
pub fn build_clusters(map: &RepoMap) -> Vec<Cluster> {
    let tree = build_dir_tree(map);
    let mut out = Vec::new();
    visit(&tree, "", &mut out);
    out
}

// ── Internals ────────────────────────────────────────────────────────────────

/// In-memory directory tree.  Each node owns its direct files and
/// child sub-directories, both in sorted order.
#[derive(Debug, Default)]
struct DirNode<'a> {
    /// `path/` (with trailing slash) of the directory relative to
    /// the workspace root.  Empty string for the root itself.
    rel_path: String,
    /// Files directly inside this directory, in `path` order.
    files: Vec<&'a FileEntry>,
    /// Sub-directories keyed by their basename (sorted).
    children: Vec<DirNode<'a>>,
}

fn build_dir_tree(map: &RepoMap) -> DirNode<'_> {
    let mut root = DirNode {
        rel_path: String::new(),
        ..Default::default()
    };
    for file in &map.files {
        insert_file(&mut root, file);
    }
    // Sort each level's children by basename for deterministic output.
    sort_recursive(&mut root);
    root
}

fn insert_file<'a>(node: &mut DirNode<'a>, file: &'a FileEntry) {
    let components: Vec<&str> = file.path.split('/').collect();
    let (file_name, dir_chain) = match components.split_last() {
        Some(p) => p,
        None => return,
    };
    let _ = file_name;
    insert_at(node, dir_chain, file, 0);
}

fn insert_at<'a>(node: &mut DirNode<'a>, chain: &[&str], file: &'a FileEntry, depth: usize) {
    if depth == chain.len() {
        node.files.push(file);
        return;
    }
    let name = chain[depth];
    let child_pos = match node.children.iter().position(|c| {
        c.rel_path
            .strip_suffix('/')
            .and_then(|s| s.rsplit('/').next())
            == Some(name)
    }) {
        Some(p) => p,
        None => {
            let rel = if node.rel_path.is_empty() {
                format!("{name}/")
            } else {
                format!("{}{name}/", node.rel_path)
            };
            node.children.push(DirNode {
                rel_path: rel,
                ..Default::default()
            });
            node.children.len() - 1
        }
    };
    insert_at(&mut node.children[child_pos], chain, file, depth + 1);
}

fn sort_recursive(node: &mut DirNode<'_>) {
    node.children.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    for c in &mut node.children {
        sort_recursive(c);
    }
}

/// Depth-first cluster emission.  Each call decides whether this
/// node "absorbs" its descendants into a single cluster or whether
/// any descendant exceeds the budget and must be its own cluster.
fn visit(node: &DirNode<'_>, _parent: &str, out: &mut Vec<Cluster>) {
    // Try to absorb the whole subtree under this node into one
    // cluster — that's the desired outcome when the directory is
    // small.  If that overflows, emit per-child clusters instead
    // (each subtree decides for itself recursively).
    let absorbed = render_subtree_listing(node);
    if absorbed.len() <= TARGET_CLUSTER_BYTES {
        // Single cluster covering this node + everything beneath it.
        if listing_is_empty(&absorbed) {
            return;
        }
        let covered = collect_covered_dirs(node);
        let hash = hash_basenames(node);
        out.push(Cluster {
            root_dir: node.rel_path.clone(),
            covered_dirs: covered,
            content_hash: hash,
            user_prompt: user_prompt_for(&node.rel_path),
            listing: absorbed,
        });
        return;
    }
    if node.children.is_empty() {
        // Leaf directory whose own listing exceeds the budget — a
        // flat directory holding thousands of files.  Split the
        // listing into byte-bounded chunks rather than emitting one
        // runaway cluster.  Without this, a single 190 KB listing
        // can land on the prefill path and trigger a multi-hundred-MB
        // hot-tier eviction that stalls the loader.
        emit_split_leaf(node, out);
        return;
    }

    // Subtree too big to absorb wholesale.  Emit a cluster for
    // (this directory's own files + any small leaf children) and
    // recurse into each child sub-directory whose own subtree is
    // still too big.
    let (small_children, big_children): (Vec<_>, Vec<_>) = node
        .children
        .iter()
        .partition(|c| render_subtree_listing(c).len() <= TARGET_CLUSTER_BYTES / 2);

    let mut head_listing = render_node_files(node);

    // If this directory's *own* files alone overflow the cluster
    // budget — a flat directory of thousands of files that also
    // happens to contain subdirectories — split the file list into
    // byte-bounded chunks (same shape as the leaf case) and
    // recurse into the children separately.  Without this, a non-
    // leaf directory with a huge own-file population produces one
    // 190 KB cluster that triggers a multi-hundred-MB hot-tier
    // eviction and stalls the loader.
    if head_listing.len() > MAX_CLUSTER_BYTES {
        emit_split_leaf(node, out);
        for child in &node.children {
            visit(child, &node.rel_path, out);
        }
        return;
    }

    let mut head_covered = vec![node.rel_path.clone()];
    let mut head_basenames: Vec<String> = node
        .files
        .iter()
        .map(|f| format!("{}{}", node.rel_path, basename(&f.path)))
        .collect();

    for small in &small_children {
        let extra = render_subtree_listing(small);
        if head_listing.len() + extra.len() > MAX_CLUSTER_BYTES {
            // Even the "small" subtree would push us past the hard cap
            // — fall back to recursing into it as its own cluster.
            visit(small, &node.rel_path, out);
        } else {
            head_listing.push_str(&extra);
            head_covered.extend(collect_covered_dirs(small));
            collect_basenames_with_prefix(small, &mut head_basenames);
        }
    }

    if !listing_is_empty(&head_listing) {
        head_basenames.sort();
        out.push(Cluster {
            root_dir: node.rel_path.clone(),
            covered_dirs: head_covered,
            content_hash: hash_basename_list(&head_basenames),
            user_prompt: user_prompt_for(&node.rel_path),
            listing: head_listing,
        });
    }

    for big in &big_children {
        visit(big, &node.rel_path, out);
    }
}

fn user_prompt_for(rel_path: &str) -> String {
    if rel_path.is_empty() {
        "Tell me what is in the workspace root.".to_string()
    } else {
        // Drop trailing slash for a more natural prompt.
        let clean = rel_path.trim_end_matches('/');
        format!("Tell me what is in `{clean}`.")
    }
}

/// Split a leaf directory's file list into byte-bounded chunks
/// and emit one cluster per chunk.  Each chunk re-renders the
/// directory header so the listing is self-contained.  Chunk index
/// is appended to `root_dir` so each emitted cluster has a unique
/// state key that the refresh path can hash against.
fn emit_split_leaf(node: &DirNode<'_>, out: &mut Vec<Cluster>) {
    let header_line = if node.rel_path.is_empty() {
        "(workspace root)\n".to_string()
    } else {
        format!("{}\n", node.rel_path)
    };

    // Pre-render every file line so we know its exact byte cost.
    let lines: Vec<String> = node
        .files
        .iter()
        .map(|f| {
            let mut bits = vec![
                format!("{} lines", f.line_count),
                f.language.label().to_string(),
            ];
            if let Some(h) = &f.module_hint {
                bits.push(h.render());
            }
            format!("  - {} ({})\n", basename(&f.path), bits.join(", "))
        })
        .collect();

    // Greedy pack: fill each chunk up to MAX_CLUSTER_BYTES (minus a
    // small safety margin for the trailing blank line) and emit
    // when the next line would overflow.
    let body_budget = MAX_CLUSTER_BYTES.saturating_sub(header_line.len() + 1);
    let mut chunk_idx = 0usize;
    let mut cursor = 0usize;
    while cursor < lines.len() {
        let mut bytes = 0usize;
        let chunk_start = cursor;
        while cursor < lines.len() && bytes + lines[cursor].len() <= body_budget {
            bytes += lines[cursor].len();
            cursor += 1;
        }
        if cursor == chunk_start {
            // A single file line is larger than the whole budget —
            // emit it alone rather than looping forever.  This can
            // only happen if MAX_CLUSTER_BYTES is set lower than the
            // longest single rendered line in the workspace.
            cursor += 1;
        }

        let mut listing = String::with_capacity(header_line.len() + bytes + 1);
        listing.push_str(&header_line);
        for line in &lines[chunk_start..cursor] {
            listing.push_str(line);
        }
        listing.push('\n');

        let basenames: Vec<String> = node.files[chunk_start..cursor]
            .iter()
            .map(|f| format!("{}{}", node.rel_path, basename(&f.path)))
            .collect();

        // Suffix the root_dir with the chunk index so each cluster
        // has a distinct state key.  The covered_dirs list still
        // names the underlying directory verbatim — the suffix is a
        // bookkeeping detail, not a real path.
        let root_dir = if chunk_idx == 0 {
            node.rel_path.clone()
        } else {
            format!("{}#{chunk_idx}", node.rel_path)
        };

        out.push(Cluster {
            root_dir,
            covered_dirs: vec![node.rel_path.clone()],
            content_hash: hash_basename_list(&basenames),
            user_prompt: user_prompt_for(&node.rel_path),
            listing,
        });
        chunk_idx += 1;
    }
}

fn render_subtree_listing(node: &DirNode<'_>) -> String {
    let mut out = String::new();
    render_node_recursive(node, &mut out);
    out
}

fn render_node_recursive(node: &DirNode<'_>, out: &mut String) {
    out.push_str(&render_node_files(node));
    for child in &node.children {
        render_node_recursive(child, out);
    }
}

fn render_node_files(node: &DirNode<'_>) -> String {
    if node.files.is_empty() && node.children.is_empty() {
        return String::new();
    }
    let header = if node.rel_path.is_empty() {
        "(workspace root)".to_string()
    } else {
        node.rel_path.clone()
    };
    let mut out = format!("{header}\n");
    if node.files.is_empty() {
        // Skip rendering an empty directory entirely if it has no
        // direct files — the children will surface their own headers.
        return String::new();
    }
    for f in &node.files {
        let mut bits = vec![
            format!("{} lines", f.line_count),
            f.language.label().to_string(),
        ];
        if let Some(h) = &f.module_hint {
            bits.push(h.render());
        }
        let bn = basename(&f.path);
        out.push_str(&format!("  - {bn} ({})\n", bits.join(", ")));
    }
    out.push('\n');
    out
}

fn listing_is_empty(s: &str) -> bool {
    s.trim().is_empty()
}

fn collect_covered_dirs(node: &DirNode<'_>) -> Vec<String> {
    let mut out = vec![node.rel_path.clone()];
    for c in &node.children {
        out.extend(collect_covered_dirs(c));
    }
    out
}

fn hash_basenames(node: &DirNode<'_>) -> String {
    let mut names = Vec::new();
    collect_basenames_with_prefix(node, &mut names);
    names.sort();
    hash_basename_list(&names)
}

fn collect_basenames_with_prefix(node: &DirNode<'_>, out: &mut Vec<String>) {
    for f in &node.files {
        out.push(format!("{}{}", node.rel_path, basename(&f.path)));
    }
    for c in &node.children {
        collect_basenames_with_prefix(c, out);
    }
}

fn hash_basename_list(names: &[String]) -> String {
    let mut h = Sha256::new();
    for n in names {
        h.update(n.as_bytes());
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

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::types::{FileEntry, Language};

    fn entry(path: &str, lines: u32) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            line_count: lines,
            language: Language::Rust,
            size_bytes: 0,
            module_hint: None,
        }
    }

    fn map(_workspace: &str, files: Vec<FileEntry>) -> RepoMap {
        RepoMap {
            files,
            ..Default::default()
        }
    }

    #[test]
    fn small_workspace_collapses_into_a_single_cluster() {
        let m = map(
            "demo",
            vec![
                entry("src/lib.rs", 5),
                entry("src/main.rs", 10),
                entry("Cargo.toml", 8),
            ],
        );
        let clusters = build_clusters(&m);
        assert_eq!(clusters.len(), 1, "small repo fits one cluster");
        let c = &clusters[0];
        assert_eq!(c.root_dir, "", "root dir is workspace root");
        assert!(c.listing.contains("Cargo.toml"));
        assert!(c.listing.contains("lib.rs"));
        assert!(c.listing.contains("main.rs"));
    }

    #[test]
    fn large_workspace_splits_into_per_subtree_clusters() {
        let mut files = vec![entry("Cargo.toml", 5)];
        // 4 sub-directories each with many files — each individually
        // exceeds the target budget so the cluster builder emits one
        // cluster per sub-directory.
        for sub in &["alpha", "bravo", "charlie", "delta"] {
            for i in 0..40 {
                files.push(entry(&format!("src/{sub}/file_{i:03}.rs"), 50));
            }
        }
        let m = map("big", files);
        let clusters = build_clusters(&m);
        assert!(
            clusters.len() >= 4,
            "large workspace should split into multiple clusters, got {}",
            clusters.len()
        );
        let roots: Vec<&str> = clusters.iter().map(|c| c.root_dir.as_str()).collect();
        for sub in &["src/alpha/", "src/bravo/", "src/charlie/", "src/delta/"] {
            assert!(
                roots.iter().any(|r| r == sub),
                "expected cluster for {sub} in {roots:?}"
            );
        }
    }

    #[test]
    fn cluster_hash_changes_on_file_rename() {
        let m_before = map("rh", vec![entry("src/lib.rs", 5), entry("src/util.rs", 5)]);
        let m_after = map(
            "rh",
            vec![entry("src/lib.rs", 5), entry("src/helper.rs", 5)],
        );
        let h_before = &build_clusters(&m_before)[0].content_hash;
        let h_after = &build_clusters(&m_after)[0].content_hash;
        assert_ne!(h_before, h_after);
    }

    #[test]
    fn cluster_hash_stable_when_file_only_grows() {
        // The hash is over file *names*, not content.  Editing
        // contents (here: changing line counts) leaves the hash
        // untouched so we don't trigger refresh churn.
        let m_before = map("rh", vec![entry("src/lib.rs", 5)]);
        let m_after = map("rh", vec![entry("src/lib.rs", 5000)]);
        assert_eq!(
            build_clusters(&m_before)[0].content_hash,
            build_clusters(&m_after)[0].content_hash,
        );
    }

    #[test]
    fn cluster_user_prompt_names_the_root_dir() {
        let m = map(
            "rh",
            vec![
                entry("Cargo.toml", 1),
                // Push the workspace big enough that src/auth ends up
                // as its own cluster.
                entry("src/auth/handler.rs", 50),
                entry("src/auth/validator.rs", 50),
                entry("src/auth/token.rs", 50),
            ],
        );
        let clusters = build_clusters(&m);
        let _ = clusters;
        // Either workspace fits in one cluster (small case) or
        // src/auth gets a dedicated cluster (larger case); both
        // shapes pass — what we're asserting is the prompt format.
        let m_big = map(
            "rh",
            (0..40)
                .map(|i| entry(&format!("src/auth/file_{i:03}.rs"), 50))
                .chain(std::iter::once(entry("Cargo.toml", 1)))
                .collect(),
        );
        let big_clusters = build_clusters(&m_big);
        let auth_cluster = big_clusters
            .iter()
            .find(|c| c.root_dir == "src/auth/")
            .expect("src/auth cluster present");
        assert!(auth_cluster.user_prompt.contains("src/auth"));
        assert!(auth_cluster.user_prompt.starts_with("Tell me what is in"));
    }

    #[test]
    fn cluster_root_prompt_uses_workspace_root_phrasing() {
        let m = map("rh", vec![entry("Cargo.toml", 1)]);
        let clusters = build_clusters(&m);
        assert!(clusters[0].user_prompt.contains("workspace root"));
    }

    #[test]
    fn cluster_build_is_deterministic() {
        let m = map(
            "rh",
            vec![
                entry("src/lib.rs", 10),
                entry("src/auth/handler.rs", 50),
                entry("docs/guide.md", 30),
            ],
        );
        let a = build_clusters(&m);
        let b = build_clusters(&m);
        assert_eq!(a, b);
    }

    #[test]
    fn flat_leaf_directory_over_max_splits_into_chunked_clusters() {
        // A flat leaf directory whose own listing exceeds
        // `MAX_CLUSTER_BYTES` (here: ~600 files × ~50 chars/line).
        // Without `emit_split_leaf`, this collapses to one giant
        // ~30 KB cluster that triggers a hundreds-of-MB hot-tier
        // eviction at prefill time and stalls the daemon loader.
        // The split path emits multiple chunks, each <= MAX bytes,
        // distinguished by a `#N` suffix on `root_dir`.
        let files: Vec<FileEntry> = (0..600)
            .map(|i| entry(&format!("data/flat/file_{i:04}.rs"), 10))
            .collect();
        let clusters = build_clusters(&map("flat", files));
        let flat_clusters: Vec<&Cluster> = clusters
            .iter()
            .filter(|c| c.root_dir.starts_with("data/flat/"))
            .collect();
        assert!(
            flat_clusters.len() >= 2,
            "flat directory should split into >=2 chunks, got {} (roots: {:?})",
            flat_clusters.len(),
            flat_clusters
                .iter()
                .map(|c| &c.root_dir)
                .collect::<Vec<_>>(),
        );
        for c in &flat_clusters {
            assert!(
                c.listing.len() <= MAX_CLUSTER_BYTES,
                "chunk listing {} bytes exceeded MAX_CLUSTER_BYTES={}",
                c.listing.len(),
                MAX_CLUSTER_BYTES,
            );
        }
        let suffixed: Vec<&Cluster> = flat_clusters
            .iter()
            .filter(|c| c.root_dir.contains('#'))
            .copied()
            .collect();
        assert!(
            !suffixed.is_empty(),
            "expected at least one chunk with a `#N` suffix to disambiguate state keys",
        );
        let hashes: std::collections::HashSet<&str> = flat_clusters
            .iter()
            .map(|c| c.content_hash.as_str())
            .collect();
        assert_eq!(
            hashes.len(),
            flat_clusters.len(),
            "each chunk must hash to a distinct content_hash so the refresh path \
             keys them independently",
        );
    }

    #[test]
    fn cluster_covered_dirs_includes_subtree_for_absorbed() {
        let m = map(
            "rh",
            vec![
                entry("Cargo.toml", 1),
                entry("src/lib.rs", 5),
                entry("src/util.rs", 5),
            ],
        );
        let clusters = build_clusters(&m);
        // Whole workspace absorbs into the root cluster.
        let root = &clusters[0];
        assert!(root.covered_dirs.contains(&"".to_string()));
        assert!(root.covered_dirs.iter().any(|d| d == "src/"));
    }
}
