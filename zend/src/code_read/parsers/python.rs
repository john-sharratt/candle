//! Python scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label,
    };
    rules
        .kind_to_chunk
        .insert("function_definition", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("class_definition", ChunkKind::TypeDefinition);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_definition" => field_text(node, "name", source).map(|n| format!("def {n}")),
        "class_definition" => field_text(node, "name", source).map(|n| format!("class {n}")),
        _ => None,
    }
}

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "class_definition" => field_text(node, "name", source).map(|n| format!("class {n}")),
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
        verify_carve(src, Language::Python, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Python, false)
    }

    // ── Top-level items ──────────────────────────────────────────────────────

    #[test]
    fn extracts_top_level_def_and_class() {
        verify(
            "def alpha():\n    pass\n\nclass Foo:\n    def bar(self):\n        pass\n",
            &[
                ("def alpha", "def alpha"),
                ("class Foo", "class Foo"),
                ("class Foo > def bar", "def bar"),
            ],
        );
    }

    #[test]
    fn extracts_method_with_class_in_path() {
        verify(
            "class AuthHandler:\n    def validate_token(self, token):\n        return True\n",
            &[(
                "class AuthHandler > def validate_token",
                "def validate_token",
            )],
        );
    }

    // ── Decorators ───────────────────────────────────────────────────────────

    #[test]
    fn extracts_decorated_function() {
        verify(
            "@staticmethod\n@cache\ndef helper(x):\n    return x * 2\n",
            &[("def helper", "def helper")],
        );
    }

    #[test]
    fn extracts_decorated_class() {
        verify(
            "@dataclass\nclass Point:\n    x: int\n    y: int\n",
            &[("class Point", "class Point")],
        );
    }

    #[test]
    fn extracts_method_with_classmethod_decorator() {
        verify(
            "class C:\n    @classmethod\n    def from_str(cls, s):\n        return cls()\n",
            &[("class C > def from_str", "def from_str")],
        );
    }

    #[test]
    fn extracts_def_with_property_decorator() {
        let src = "class Box:\n    @property\n    def value(self):\n        return self._v\n    @value.setter\n    def value(self, v):\n        self._v = v\n";
        let scopes = verify(src, &[("class Box", "class Box")]);
        let count = scopes
            .iter()
            .filter(|s| s.qualified_path().ends_with("def value"))
            .count();
        assert!(
            count >= 2,
            "expected ≥2 def value (getter+setter), got {count}"
        );
    }

    #[test]
    fn extracts_class_decorated_with_dataclass() {
        verify(
            "@dataclass(frozen=True, slots=True)\nclass Point:\n    x: int\n    y: int\n    def __repr__(self):\n        return f\"{self.x}\"\n",
            &[
                ("class Point", "class Point"),
                ("class Point > def __repr__", "def __repr__"),
            ],
        );
    }

    // ── Function variants ────────────────────────────────────────────────────

    #[test]
    fn extracts_async_def() {
        verify(
            "async def fetch(url):\n    return await get(url)\n",
            &[("def fetch", "async def fetch")],
        );
    }

    #[test]
    fn extracts_async_method() {
        verify(
            "class API:\n    async def fetch(self):\n        return await self.client.get()\n",
            &[("class API > def fetch", "async def fetch")],
        );
    }

    #[test]
    fn extracts_def_with_type_hints_and_defaults() {
        verify(
            "def make(\n    name: str,\n    *,\n    timeout: float = 1.0,\n    tags: list[str] | None = None,\n) -> dict[str, str]:\n    return {\"name\": name}\n",
            &[("def make", "def make")],
        );
    }

    // ── Class variants ───────────────────────────────────────────────────────

    #[test]
    fn extracts_class_inheritance() {
        verify(
            "class Animal:\n    def speak(self):\n        pass\n\nclass Dog(Animal):\n    def speak(self):\n        return \"woof\"\n",
            &[
                ("class Animal", "class Animal"),
                ("class Dog", "class Dog(Animal)"),
                ("class Dog > def speak", "def speak"),
            ],
        );
    }

    #[test]
    fn extracts_class_with_metaclass() {
        verify(
            "class Meta(type):\n    pass\n\nclass Configured(metaclass=Meta):\n    def go(self):\n        pass\n",
            &[
                ("class Meta", "class Meta(type)"),
                ("class Configured", "class Configured(metaclass=Meta)"),
            ],
        );
    }

    #[test]
    fn extracts_class_with_multiple_inheritance() {
        verify(
            "class A: pass\nclass B: pass\nclass C(A, B): pass\n",
            &[
                ("class A", "class A"),
                ("class B", "class B"),
                ("class C", "class C(A, B)"),
            ],
        );
    }

    #[test]
    fn extracts_nested_classes() {
        verify(
            "class Outer:\n    class Middle:\n        class Inner:\n            def deep(self): pass\n",
            &[(
                "class Outer > class Middle > class Inner",
                "class Inner",
            )],
        );
    }

    #[test]
    fn extracts_class_with_dunder_methods() {
        verify(
            "class Vec:\n    def __init__(self, x, y):\n        self.x = x\n    def __repr__(self):\n        return \"v\"\n",
            &[
                ("class Vec > def __init__", "def __init__"),
                ("class Vec > def __repr__", "def __repr__"),
            ],
        );
    }

    #[test]
    fn extracts_class_with_type_hints() {
        verify(
            "class Container:\n    items: list[int] = []\n    def add(self, x: int) -> None:\n        self.items.append(x)\n",
            &[
                ("class Container", "class Container"),
                ("class Container > def add", "def add"),
            ],
        );
    }

    #[test]
    fn extracts_single_line_class() {
        verify(
            "class Empty: pass\n",
            &[("class Empty", "class Empty: pass")],
        );
    }

    // ── Identifiers ──────────────────────────────────────────────────────────

    #[test]
    fn extracts_def_with_underscore_identifier() {
        verify(
            "def _private():\n    return 1\n",
            &[("def _private", "def _private")],
        );
    }

    #[test]
    fn extracts_def_with_unicode_identifier() {
        let src = "def αβγ():\n    return 1\n";
        // Tree-sitter-python supports Unicode identifiers — the scope
        // is present with the name preserved.
        verify(src, &[("def αβγ", "def αβγ")]);
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn def_scope_starts_at_def_keyword_line() {
        let src = "# header\n\ndef alpha():\n    return 1\n";
        let scopes = verify(src, &[("def alpha", "def alpha")]);
        let alpha = find_scope(&scopes, "def alpha").unwrap();
        assert_eq!(alpha.start_line, 3);
        assert_eq!(alpha.end_line, 4);
    }

    #[test]
    fn class_method_scope_is_method_lines_only() {
        let src = "class Foo:\n    x = 1\n    def bar(self):\n        return self.x\n";
        let scopes = verify(src, &[("class Foo > def bar", "def bar")]);
        let bar = find_scope(&scopes, "class Foo > def bar").unwrap();
        assert_eq!(bar.start_line, 3);
        assert_eq!(bar.end_line, 4);
    }

    #[test]
    fn nested_function_scope_is_inner_only() {
        let src = "def outer():\n    def inner():\n        return 1\n    return inner()\n";
        let scopes = verify(
            src,
            &[("def outer", "def outer"), ("def inner", "def inner")],
        );
        let inner = find_scope(&scopes, "def inner").unwrap();
        assert_eq!(inner.start_line, 2);
        assert_eq!(inner.end_line, 3);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_around_broken_middle_def() {
        verify_cov("def alpha(): return 1\n\ndef beta(:\n    # invalid\n\ndef gamma(): return 3\n");
    }

    #[test]
    fn preserves_crlf_line_endings() {
        verify(
            "def alpha():\r\n    pass\r\n\r\ndef beta():\r\n    pass\r\n",
            &[("def alpha", "def alpha"), ("def beta", "def beta")],
        );
    }

    #[test]
    fn preserves_module_imports_and_globals() {
        verify_cov(
            "import os\nimport sys\nfrom pathlib import Path\n\nX = 42\nY = \"hello\"\n\ndef alpha():\n    return X\n",
        );
    }

    #[test]
    fn preserves_file_with_only_imports() {
        verify_cov("import os\nimport sys\nfrom pathlib import Path\n");
    }

    #[test]
    fn preserves_file_with_only_comments() {
        verify_cov("# header\n# more header\n");
    }

    #[test]
    fn preserves_module_docstring_only() {
        verify_cov("\"\"\"Module docs.\"\"\"\n");
    }

    #[test]
    fn preserves_tab_indented_body() {
        verify("def alpha():\n\treturn 1\n", &[("def alpha", "def alpha")]);
    }

    #[test]
    fn preserves_truncated_def() {
        verify_cov("def broken(\n");
    }
}
