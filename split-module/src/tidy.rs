//! Taking the empty modules out of a split's `mod.rs` once `cargo fix` has run.
//!
//! A split test module is declared in `mod.rs` still, gathering its shares'
//! names so that what other modules reach as `<module>::x` they reach as they
//! did (`assemble`). `cargo fix` takes out each re-export nothing reaches, and
//! a module whose re-exports were all unused is left empty: `tidy` removes it,
//! with the attributes and comments written over it. Nothing else changes, and
//! the file is written whole under a temporary name and renamed into place.

use std::fs;
use std::ops::Range;
use std::path::Path;

use tree_sitter::Parser;

/// Each empty inline module of `source`, from its attributes and comments to
/// the end of its line.
pub fn empty_modules(source: &str) -> Result<Vec<Range<usize>>, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|e| format!("the Rust grammar does not load: {e}"))?;
    let tree = parser
        .parse(source, None)
        .ok_or("the file does not parse")?;
    let root = tree.root_node();
    let mut out = Vec::new();
    let mut cursor = root.walk();
    let children: Vec<_> = root.children(&mut cursor).collect();
    for (i, node) in children.iter().enumerate() {
        if node.kind() != "mod_item" {
            continue;
        }
        let Some(body) = node.child_by_field_name("body") else {
            continue;
        };
        let mut c = body.walk();
        if body.children(&mut c).any(|ch| ch.is_named()) {
            continue;
        }
        // Back over the attributes and comments that belong to it.
        let mut start = node.start_byte();
        for prev in children[..i].iter().rev() {
            let gap = &source[prev.end_byte()..start];
            let attached = matches!(
                prev.kind(),
                "attribute_item" | "line_comment" | "block_comment"
            ) && gap.matches('\n').count() <= 1
                && !source[prev.start_byte()..prev.end_byte()].starts_with("//!");
            if !attached {
                break;
            }
            start = prev.start_byte();
        }
        let end = source[node.end_byte()..]
            .find('\n')
            .map_or(source.len(), |nl| node.end_byte() + nl + 1);
        out.push(start..end);
    }
    Ok(out)
}

/// `source` without its empty inline modules.
pub fn tidied(source: &str) -> Result<String, String> {
    let mut out = String::with_capacity(source.len());
    let mut at = 0;
    for r in empty_modules(source)? {
        out.push_str(&source[at..r.start]);
        at = r.end;
    }
    out.push_str(&source[at..]);
    Ok(out)
}

/// Tidy `dir/mod.rs`. Returns the number of modules taken out.
pub fn tidy(dir: &Path) -> Result<usize, String> {
    let path = dir.join("mod.rs");
    let source = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = empty_modules(&source)?.len();
    if n == 0 {
        return Ok(0);
    }
    let text = tidied(&source)?;
    let tmp = path.with_extension("rs.split-tmp");
    fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **An empty module goes with what is written over it**; a module with
    /// anything in it, and the file's own `//!` doc, stay.
    #[test]
    fn empty_modules_are_taken_out() {
        let src = "//! The pool.\n\nuse a::b;\n\n/// Its tests.\n#[cfg(test)]\nmod tests {\n    \n    \n}\n\n\
                   #[cfg(test)]\nmod kept {\n    pub(super) use super::x::*;\n}\n\nmod rows;\n";
        assert_eq!(
            tidied(src).unwrap(),
            "//! The pool.\n\nuse a::b;\n\n\n#[cfg(test)]\nmod kept {\n    pub(super) use super::x::*;\n}\n\nmod rows;\n"
        );
    }
}
