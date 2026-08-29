//! Ruby scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_ruby::LANGUAGE.into())
        .ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label,
    };
    rules.kind_to_chunk.insert("method", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("singleton_method", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("class", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("module", ChunkKind::TypeDefinition);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "method" => field_text(node, "name", source).map(|n| format!("def {n}")),
        "singleton_method" => field_text(node, "name", source).map(|n| format!("def self.{n}")),
        "class" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "module" => field_text(node, "name", source).map(|n| format!("module {n}")),
        _ => None,
    }
}

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "class" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "module" => field_text(node, "name", source).map(|n| format!("module {n}")),
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::Scope;
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Ruby, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Ruby, false)
    }

    // ── Methods + classes + modules ─────────────────────────────────────────

    #[test]
    fn extracts_top_level_method() {
        verify(
            "def hello\n  puts 'hi'\nend\n",
            &[("def hello", "def hello")],
        );
    }

    #[test]
    fn extracts_class_with_methods() {
        verify(
            "class Greeter\n  def greet(name)\n    puts \"hi, #{name}\"\n  end\n  def self.bye\n    puts 'bye'\n  end\nend\n",
            &[
                ("class Greeter", "class Greeter"),
                ("class Greeter > def greet", "def greet"),
                ("class Greeter > def self.bye", "def self.bye"),
            ],
        );
    }

    #[test]
    fn extracts_module_with_methods() {
        verify(
            "module Helpers\n  def self.format(x)\n    x.to_s\n  end\nend\n",
            &[
                ("module Helpers", "module Helpers"),
                ("module Helpers > def self.format", "def self.format"),
            ],
        );
    }

    #[test]
    fn extracts_nested_class() {
        verify(
            "module Outer\n  class Inner\n    def thing; end\n  end\nend\n",
            &[
                ("module Outer > class Inner", "class Inner"),
                ("module Outer > class Inner > def thing", "def thing"),
            ],
        );
    }

    #[test]
    fn extracts_class_with_inheritance() {
        verify(
            "class Animal\n  def speak; 'sound'; end\nend\nclass Dog < Animal\n  def speak; 'woof'; end\nend\n",
            &[
                ("class Animal", "class Animal"),
                ("class Dog", "class Dog < Animal"),
            ],
        );
    }

    // ── Method name variants ─────────────────────────────────────────────────

    #[test]
    fn extracts_predicate_method() {
        verify(
            "def empty?\n  size == 0\nend\n",
            &[("def empty?", "def empty?")],
        );
    }

    #[test]
    fn extracts_bang_method() {
        verify(
            "def save!\n  raise unless valid?\nend\n",
            &[("def save!", "def save!")],
        );
    }

    #[test]
    fn extracts_setter_method() {
        verify(
            "class A\n  def name=(v)\n    @name = v\n  end\nend\n",
            &[("class A > def name=", "def name=")],
        );
    }

    #[test]
    fn extracts_method_with_block_arg() {
        verify(
            "def each_with_index(&block)\n  collection.each_with_index(&block)\nend\n",
            &[("def each_with_index", "def each_with_index")],
        );
    }

    #[test]
    fn extracts_method_with_splat_args() {
        verify(
            "def combine(*args, **kwargs)\n  args + kwargs.values\nend\n",
            &[("def combine", "def combine")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn method_scope_lines_match_def_through_end() {
        let src = "def alpha\n  1\nend\n";
        let scopes = verify(src, &[("def alpha", "def alpha")]);
        let alpha = find_scope(&scopes, "def alpha").unwrap();
        assert_eq!(alpha.start_line, 1);
        assert_eq!(alpha.end_line, 3);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_attr_accessor_class() {
        verify_cov(
            "class Point\n  attr_accessor :x, :y\n  def initialize(x, y); @x = x; @y = y; end\nend\n",
        );
    }

    #[test]
    fn preserves_crlf_endings() {
        verify("def alpha\r\n  1\r\nend\r\n", &[("def alpha", "def alpha")]);
    }

    #[test]
    fn preserves_truncated_method() {
        verify_cov("def broken\n  x = 1\n");
    }

    #[test]
    fn preserves_module_only_file() {
        verify("module Mixin\nend\n", &[("module Mixin", "module Mixin")]);
    }
}
