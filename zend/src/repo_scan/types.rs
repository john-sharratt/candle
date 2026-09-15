//! Data types shared by the workspace walker and the per-directory unit builder.

/// Languages we recognise by extension.  Each variant either maps to
/// a tree-sitter grammar (proper scope-aware carving) or to a
/// header-based / fixed-window fallback (see
/// [`crate::code_read::carve`]).  Adding a language: add the enum
/// variant, plug it into [`Self::label`] / [`Self::from_extension`],
/// and either wire it into the tree-sitter dispatch or rely on the
/// header / fallback tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    C,
    Cpp,
    Java,
    Ruby,
    Php,
    Bash,
    Html,
    Css,
    Markdown,
    Yaml,
    Toml,
    Json,
    PlainText,
}

impl Language {
    /// Markdown code-fence language tag used when rendering scope
    /// prefills.  Returned strings line up with the tag set the
    /// widely-deployed read_file / file-viewer tools emit so coding
    /// models recognise the format from their pretraining data.
    /// Returns an empty string for [`Language::PlainText`] (no fence
    /// tag).
    pub fn fence_tag(self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::TypeScript => "typescript",
            Language::JavaScript => "javascript",
            Language::Go => "go",
            Language::C => "c",
            Language::Cpp => "cpp",
            Language::Java => "java",
            Language::Ruby => "ruby",
            Language::Php => "php",
            Language::Bash => "bash",
            Language::Html => "html",
            Language::Css => "css",
            Language::Markdown => "markdown",
            Language::Yaml => "yaml",
            Language::Toml => "toml",
            Language::Json => "json",
            Language::PlainText => "",
        }
    }

    /// Resolve a file extension (`"rs"`) to a [`Language`].  Returns
    /// `None` for extensions outside the allowlist — those files are
    /// skipped during the walk and never reach the renderer.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Language::Rust),
            "py" | "pyi" => Some(Language::Python),
            "ts" | "tsx" => Some(Language::TypeScript),
            "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
            "go" => Some(Language::Go),
            "c" | "h" => Some(Language::C),
            // CUDA carves as C++. Measured against the real kernels rather than
            // assumed: tree-sitter-cpp names the device functions and their
            // enclosing namespaces (`namespace fused_attn` /
            // `int8_decode_attn_impl()`) and splits the file header cleanly. The
            // constructs it cannot parse — `__global__` declarations, `<<<…>>>`
            // launches in the 37 files that host-launch — degrade to Fallback
            // scopes that still carry the function names in their path, so a
            // question about a kernel by name still retrieves it. A dedicated
            // `Cuda` variant would drop all 293 files to the fallback tier
            // instead, which is strictly less structure.
            "cc" | "cpp" | "cxx" | "hpp" | "hxx" | "hh" | "cu" | "cuh" => Some(Language::Cpp),
            "java" => Some(Language::Java),
            "rb" | "rake" | "ru" | "gemspec" => Some(Language::Ruby),
            "php" | "phtml" => Some(Language::Php),
            "sh" | "bash" | "zsh" => Some(Language::Bash),
            "html" | "htm" => Some(Language::Html),
            "css" | "scss" | "sass" | "less" => Some(Language::Css),
            "md" | "markdown" | "mdx" => Some(Language::Markdown),
            "yaml" | "yml" => Some(Language::Yaml),
            "toml" => Some(Language::Toml),
            "json" | "json5" | "jsonc" => Some(Language::Json),
            "txt" | "rst" | "adoc" | "asciidoc" => Some(Language::PlainText),
            _ => None,
        }
    }
}

/// Optional structural hint surfaced on workspace-manifest files
/// (`Cargo.toml`, `package.json`, `pyproject.toml`, `go.mod`).  Rendered
/// inline next to the file name in the repo-map tree:
///
/// ```text
/// Cargo.toml (workspace: 3 members)
/// package.json (name: my-app)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleHint {
    /// Cargo workspace root — carries the number of member crates.
    CargoWorkspace { members: usize },
    /// Cargo crate manifest with a `[package]` section.
    CargoPackage { name: String },
    /// npm-style manifest.
    NodePackage { name: String },
    /// Python project manifest.
    PythonProject { name: String },
    /// Go module declaration.
    GoModule { name: String },
}

impl ModuleHint {
    /// Short parenthetical rendered into a directory's summarise request
    /// (`Summarize the \`candle-nn/\` folder (crate: candle-nn) …`), so the
    /// model knows a folder is a crate/package root rather than inferring it
    /// from a `Cargo.toml` in the listing.
    pub fn render(&self) -> String {
        match self {
            ModuleHint::CargoWorkspace { members } => format!("workspace: {members} members"),
            ModuleHint::CargoPackage { name } => format!("crate: {name}"),
            ModuleHint::NodePackage { name } => format!("name: {name}"),
            ModuleHint::PythonProject { name } => format!("project: {name}"),
            ModuleHint::GoModule { name } => format!("module: {name}"),
        }
    }
}

/// One file in the repo map.  Path is always relative to the workspace
/// root so the tree renders the same regardless of where the daemon
/// was launched from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Workspace-relative path with `/` separators (normalised on Windows).
    pub path: String,
    /// Newline-count + 1.  Empty files are reported as `0`.
    pub line_count: u32,
    pub language: Language,
    /// File size in bytes — used to filter oversize files before they
    /// reach the carver.
    pub size_bytes: u64,
    /// Module-structure hint for manifest files; `None` for ordinary
    /// source files.
    pub module_hint: Option<ModuleHint>,
}

/// The full set of files surveyed by the workspace walker.
/// Sorted ascending by `path` so the unit builder's output is
/// byte-identical across runs on the same tree.
#[derive(Debug, Clone, Default)]
pub struct RepoMap {
    /// Every retained file in `path` order.
    pub files: Vec<FileEntry>,
    /// Number of files surfaced by the walker before the allowlist /
    /// size-cap filters fired.  Used for the load-progress denominator.
    pub files_scanned: usize,
    /// Files skipped because their size exceeded `MAX_FILE_BYTES`.
    pub files_skipped_oversize: usize,
    /// Files skipped because their extension wasn't allowlisted.
    pub files_skipped_extension: usize,
    /// Files skipped because their content classified as binary (the
    /// statistical NUL-count + non-text-byte-ratio classifier in
    /// [`super::binary_sniff::is_binary_sample`]) despite carrying an
    /// allowlisted extension — e.g. a compiled CUDA fatbin ELF dump checked in
    /// as `*.txt`. Carving one produces hundreds of garbage scopes that blow
    /// the ingest co-batch's VRAM budget.
    pub files_skipped_binary: usize,
    /// The `--max-depth` bound the walk ran under, in path components below the
    /// content root (`None` = unbounded). Nothing deeper was read — which is not
    /// the same as absent: see [`RepoMap::is_frozen_file`].
    pub max_depth: Option<usize>,
}

impl RepoMap {
    /// Whether a root-relative file `path` lies past the depth bound. Its
    /// ingested content is FROZEN: never re-read, and never retired by the
    /// deleted-path sweep either, because the walk not seeing it says nothing
    /// about whether it still exists.
    pub fn is_frozen_file(&self, path: &str) -> bool {
        self.max_depth.is_some_and(|d| path_depth(path) > d)
    }

    /// Whether directory `dir` (`zend/src/`, or `"."` for the root) was cut off
    /// by the depth bound. A directory's own files sit one component deeper
    /// than it, so a directory AT the bound had none walked. Frozen for the same
    /// reason as [`Self::is_frozen_file`].
    pub fn is_frozen_dir(&self, dir: &str) -> bool {
        self.max_depth.is_some_and(|d| path_depth(dir) >= d)
    }
}

/// Path components in a `/`-separated root-relative path; the root (`"."`) is 0.
fn path_depth(path: &str) -> usize {
    path.split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .count()
}

#[cfg(test)]
mod tests {
    use super::{Language, RepoMap};

    /// With `--max-depth 2`: the root's files and one folder down are walked;
    /// anything deeper — and a directory at the bound, whose files are deeper —
    /// is frozen. Unbounded, nothing is.
    #[test]
    fn the_depth_bound_freezes_what_lies_past_it() {
        let bounded = RepoMap {
            max_depth: Some(2),
            ..RepoMap::default()
        };
        assert!(!bounded.is_frozen_file("a.rs"));
        assert!(!bounded.is_frozen_file("src/b.rs"));
        assert!(bounded.is_frozen_file("src/deep/c.rs"));
        assert!(!bounded.is_frozen_dir("."));
        assert!(!bounded.is_frozen_dir("src/"));
        assert!(bounded.is_frozen_dir("src/deep/"));

        let open = RepoMap::default();
        assert!(!open.is_frozen_file("a/b/c/d/e.rs"));
        assert!(!open.is_frozen_dir("a/b/c/d/"));
    }

    /// The kernels are the engine. While `.cu`/`.cuh` were off the allowlist the
    /// walk dropped all 293 of them, which took them out of BOTH layers built
    /// from it: `code_reading` never carved a kernel, and a `repo_map` folder
    /// whose content is CUDA hashed as though it held only its `api.rs` wrapper,
    /// so adding a kernel never re-summarised the folder.
    #[test]
    fn cuda_sources_are_walked_as_cpp() {
        assert_eq!(Language::from_extension("cu"), Some(Language::Cpp));
        assert_eq!(Language::from_extension("cuh"), Some(Language::Cpp));
        assert_eq!(Language::Cpp.fence_tag(), "cpp");
    }
}
