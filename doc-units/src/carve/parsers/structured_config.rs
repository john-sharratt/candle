//! TOML / JSON carver — splits at TOML tables and top-level keys. A
//! pragmatic line-scanner; no full parser is pulled in for unit boundaries.
//! (YAML is carved line by line instead — see `carve::paragraphs::lines`.)

use crate::carve::types::{ChunkKind, Scope};

/// Carve a structured-config file into top-level key sections.  Each
/// scope is labelled with the key name (e.g. `[dependencies]`,
/// `daemon:`).  Fallback to the whole document when no top-level
/// structure is detected.
pub fn carve(source: &[u8]) -> Vec<Scope> {
    let text = match std::str::from_utf8(source) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut sections: Vec<(usize, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        // TOML table header: `[name]` at column 0.
        if line.starts_with('[') {
            if let Some(end) = trimmed.find(']') {
                let name = trimmed[1..end].trim().to_string();
                if !name.is_empty() {
                    sections.push((i, name));
                    continue;
                }
            }
        }
        // YAML top-level key: at column 0, ends with `:`.
        if !line.is_empty()
            && line
                .chars()
                .next()
                .is_some_and(|c| !c.is_whitespace() && c != '#')
        {
            if let Some(colon) = trimmed.find(':') {
                let key = trimmed[..colon].trim();
                if !key.is_empty()
                    && key
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                {
                    sections.push((i, key.to_string()));
                }
            }
        }
    }

    let mut scopes = Vec::new();
    if sections.is_empty() {
        scopes.push(Scope {
            path: vec!["document".to_string()],
            kind: ChunkKind::HeaderSection,
            start_line: 1,
            end_line: lines.len() as u32,
        });
        return scopes;
    }

    if sections[0].0 > 0 {
        scopes.push(Scope {
            path: vec!["preamble".to_string()],
            kind: ChunkKind::HeaderSection,
            start_line: 1,
            end_line: sections[0].0 as u32,
        });
    }

    for (i, (start, name)) in sections.iter().enumerate() {
        let end = if i + 1 < sections.len() {
            sections[i + 1].0
        } else {
            lines.len()
        };
        scopes.push(Scope {
            path: vec![name.clone()],
            kind: ChunkKind::HeaderSection,
            start_line: (*start as u32) + 1,
            end_line: end as u32,
        });
    }

    scopes
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::carve::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::carve::types::Scope;
    use crate::Language;

    fn verify_toml(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Toml, false, expectations)
    }

    fn verify_cov_toml(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Toml, false)
    }

    // ── TOML ─────────────────────────────────────────────────────────────────

    #[test]
    fn splits_toml_at_table_headers() {
        verify_toml(
            "[package]\nname = \"foo\"\n\n[dependencies]\nbar = \"0.1\"\n",
            &[("package", "[package]"), ("dependencies", "[dependencies]")],
        );
    }

    #[test]
    fn extracts_toml_nested_tables() {
        verify_toml(
            "[package]\nname = \"x\"\n\n[dependencies]\nserde = \"1\"\n\n[dependencies.tokio]\nversion = \"1\"\n",
            &[
                ("package", "[package]"),
                ("dependencies", "[dependencies]"),
                ("dependencies.tokio", "[dependencies.tokio]"),
            ],
        );
    }

    #[test]
    fn extracts_toml_inline_table_syntax() {
        verify_toml(
            "[package]\nname = \"x\"\ninline_table = { a = 1, b = 2 }\n",
            &[("package", "[package]")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn toml_section_lines_match_section() {
        let src = "[package]\nname = \"x\"\nversion = \"1.0\"\n\n[deps]\nx = \"1\"\n";
        let scopes = verify_toml(src, &[("package", "[package]"), ("deps", "[deps]")]);
        let package = find_scope(&scopes, "package").unwrap();
        assert_eq!(package.start_line, 1);
        assert_eq!(package.end_line, 4);
    }

    // ── Preamble + edge cases ────────────────────────────────────────────────

    #[test]
    fn preamble_before_first_section() {
        let src = "# header comment\n# more\n\n[section]\nx = 1\n";
        let scopes = verify_toml(src, &[("preamble", "# header"), ("section", "[section]")]);
        let preamble = find_scope(&scopes, "preamble").unwrap();
        assert_eq!(preamble.start_line, 1);
        assert_eq!(preamble.end_line, 3);
    }

    #[test]
    fn unstructured_input_emits_single_document_scope() {
        let scopes = verify_cov_toml("just plain text\n");
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].path[0], "document");
    }

    #[test]
    fn preserves_crlf_endings() {
        verify_toml(
            "[a]\r\nx = 1\r\n[b]\r\ny = 2\r\n",
            &[("a", "[a]"), ("b", "[b]")],
        );
    }

    #[test]
    fn preserves_invalid_toml() {
        verify_cov_toml("[unclosed\nkey = value\n");
    }

    #[test]
    fn preserves_array_of_tables() {
        verify_cov_toml("[[fruits]]\nname = \"apple\"\n\n[[fruits]]\nname = \"banana\"\n");
    }
}
