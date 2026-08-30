//! Bash / shell scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_bash::LANGUAGE.into())
        .ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label,
    };
    rules
        .kind_to_chunk
        .insert("function_definition", ChunkKind::Function);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_definition" => field_text(node, "name", source).map(|n| format!("function {n}")),
        _ => None,
    }
}

fn enclosing_label(_node: &Node, _source: &[u8]) -> Option<String> {
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::Scope;
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Bash, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Bash, false)
    }

    #[test]
    fn extracts_posix_function() {
        verify(
            "alpha() {\n    echo hello\n}\n",
            &[("function alpha", "alpha()")],
        );
    }

    #[test]
    fn extracts_function_keyword_form() {
        verify(
            "function beta() {\n    echo world\n}\n",
            &[("function beta", "function beta")],
        );
    }

    #[test]
    fn extracts_function_with_local_vars() {
        verify(
            "compute() {\n    local x=1\n    local y=2\n    echo $((x + y))\n}\n",
            &[("function compute", "compute()")],
        );
    }

    #[test]
    fn extracts_multiple_functions() {
        verify(
            "alpha() {\n    echo a\n}\n\nfunction beta() {\n    echo b\n}\n\ngamma() {\n    echo g\n}\n",
            &[
                ("function alpha", "alpha()"),
                ("function beta", "function beta"),
                ("function gamma", "gamma()"),
            ],
        );
    }

    #[test]
    fn extracts_function_with_underscored_name() {
        verify(
            "_internal_helper() {\n    echo hi\n}\n",
            &[("function _internal_helper", "_internal_helper()")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn function_scope_lines_match_definition() {
        let src = "#!/usr/bin/env bash\n\nhelper() {\n    echo x\n}\n";
        let scopes = verify(src, &[("function helper", "helper()")]);
        let helper = find_scope(&scopes, "function helper").unwrap();
        assert_eq!(helper.start_line, 3);
        assert_eq!(helper.end_line, 5);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_top_level_commands() {
        verify(
            "#!/usr/bin/env bash\nset -euo pipefail\necho \"running\"\n\nhelper() {\n    echo \"x\"\n}\n",
            &[("function helper", "helper()")],
        );
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "alpha() {\r\n    echo x\r\n}\r\n",
            &[("function alpha", "alpha()")],
        );
    }

    #[test]
    fn preserves_truncated_function() {
        verify_cov("broken() {\n    echo x\n");
    }

    #[test]
    fn preserves_script_with_only_commands() {
        verify_cov("#!/bin/bash\nset -e\nrm -rf /tmp/foo\necho done\n");
    }
}
