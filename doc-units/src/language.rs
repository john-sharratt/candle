//! Which carve a file gets, from its name.

/// The languages the carver knows. Everything it does not is
/// [`Language::PlainText`], carved into paragraphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    /// TypeScript's JSX dialect — its own grammar.
    Tsx,
    JavaScript,
    Go,
    C,
    /// C++, and CUDA, which carves as C++.
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
    /// The language a path's extension names.
    pub fn of_path(path: &str) -> Language {
        let ext = path
            .rsplit('/')
            .next()
            .and_then(|name| name.rsplit_once('.'))
            .map(|(_, e)| e.to_ascii_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "rs" => Language::Rust,
            "py" | "pyi" => Language::Python,
            "ts" | "mts" | "cts" => Language::TypeScript,
            "tsx" => Language::Tsx,
            "js" | "jsx" | "mjs" | "cjs" => Language::JavaScript,
            "go" => Language::Go,
            "c" | "h" => Language::C,
            "cc" | "cpp" | "cxx" | "hpp" | "hxx" | "hh" | "cu" | "cuh" => Language::Cpp,
            "java" => Language::Java,
            "rb" | "rake" | "ru" | "gemspec" => Language::Ruby,
            "php" | "phtml" => Language::Php,
            "sh" | "bash" | "zsh" => Language::Bash,
            "html" | "htm" => Language::Html,
            "css" | "scss" | "sass" | "less" => Language::Css,
            "md" | "markdown" | "mdx" => Language::Markdown,
            "yaml" | "yml" => Language::Yaml,
            "toml" => Language::Toml,
            "json" | "json5" | "jsonc" => Language::Json,
            _ => Language::PlainText,
        }
    }

    /// Whether the language is prose: its units are paragraphs, and
    /// paragraphs are always kept apart by a blank line.
    pub fn is_prose(self) -> bool {
        matches!(self, Language::Markdown | Language::PlainText)
    }

    /// The name a message gives it.
    pub fn name(self) -> &'static str {
        match self {
            Language::Rust => "Rust",
            Language::Python => "Python",
            Language::TypeScript => "TypeScript",
            Language::Tsx => "TSX",
            Language::JavaScript => "JavaScript",
            Language::Go => "Go",
            Language::C => "C",
            Language::Cpp => "C++",
            Language::Java => "Java",
            Language::Ruby => "Ruby",
            Language::Php => "PHP",
            Language::Bash => "shell",
            Language::Html => "HTML",
            Language::Css => "CSS",
            Language::Markdown => "Markdown",
            Language::Yaml => "YAML",
            Language::Toml => "TOML",
            Language::Json => "JSON",
            Language::PlainText => "text",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extension_names_the_language() {
        assert_eq!(Language::of_path("src/lib.rs"), Language::Rust);
        assert_eq!(Language::of_path("A/B/C.RS"), Language::Rust);
        assert_eq!(Language::of_path("k/paged.cu"), Language::Cpp);
        assert_eq!(Language::of_path("ui/App.tsx"), Language::Tsx);
        assert_eq!(
            Language::of_path("layers/life/bram/2834-11 The Body.md"),
            Language::Markdown
        );
        assert_eq!(Language::of_path("missions.yaml"), Language::Yaml);
        assert_eq!(Language::of_path("Makefile"), Language::PlainText);
        assert_eq!(Language::of_path("notes.txt"), Language::PlainText);
    }
}
