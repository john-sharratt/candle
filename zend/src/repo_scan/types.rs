//! The languages the ingest layers read, and the hint a manifest gives.

/// Languages we recognise by extension — what the walker uses to decide a file
/// is source at all, and what a `code_read` conversation reports as its
/// `lang` tag.  Adding a language: add the enum variant and plug it into
/// [`Self::label`] / [`Self::from_extension`].
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
/// Cargo.toml (Cargo workspace root)
/// package.json (name: my-app)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleHint {
    /// Cargo workspace root.
    ///
    /// **Carries no member count, deliberately.** It used to render
    /// `workspace: 15 members`, and a count is the one thing in this enum that
    /// does not generalize: it is a magnitude of *this* checkout, not a
    /// structural fact, so a sealed ingest turn taught the model a number that
    /// is wrong everywhere else and stale here the moment a crate is added.
    /// It leaked, too — a conversation whose only content was "hello" cited
    /// "a project with 11 members" unprompted (see the `score_threshold` note
    /// in `projection.yaml`'s repo_map group). What the request needs to convey
    /// is the folder's ROLE, which the words alone carry.
    CargoWorkspace,
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
            ModuleHint::CargoWorkspace => "Cargo workspace root".to_string(),
            ModuleHint::CargoPackage { name } => format!("crate: {name}"),
            ModuleHint::NodePackage { name } => format!("name: {name}"),
            ModuleHint::PythonProject { name } => format!("project: {name}"),
            ModuleHint::GoModule { name } => format!("module: {name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Language;

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
