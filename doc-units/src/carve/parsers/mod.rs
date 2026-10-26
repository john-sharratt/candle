//! Per-language scope extractors.
//!
//! Each parser walks a tree-sitter syntax tree and emits a `Vec<Scope>` in
//! declaration order. Carving never panics — a parse that fails outright
//! leaves the file to be carved into paragraphs.

use tree_sitter::Parser;

use crate::Language;

pub mod bash;
pub mod c;
pub mod cpp;
pub mod css;
pub mod go;
pub mod html;
pub mod java;
pub mod javascript;
pub mod php;
pub mod python;
pub mod ruby;
pub mod rust;
pub mod structured_config;
pub mod tree_sitter_util;
pub mod typescript;

/// The grammar a language is parsed by, for the languages that have one.
fn grammar(language: Language) -> Option<tree_sitter::Language> {
    Some(match language {
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        Language::Bash => tree_sitter_bash::LANGUAGE.into(),
        Language::Html => tree_sitter_html::LANGUAGE.into(),
        Language::Css => tree_sitter_css::LANGUAGE.into(),
        _ => return None,
    })
}

/// Whether `source` parses as `language` without a syntax error — `None` for
/// a language with no grammar, where there is nothing to be wrong about.
pub fn parses(source: &[u8], language: Language) -> Option<bool> {
    let mut parser = Parser::new();
    parser.set_language(&grammar(language)?).ok()?;
    let tree = parser.parse(source, None)?;
    Some(!tree.root_node().has_error())
}
