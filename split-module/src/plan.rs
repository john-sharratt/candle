//! A split plan: the files to make, each with its one-line purpose and the map
//! ids it holds.
//!
//! ```text
//! file pool: The block pool and its growth.
//! 1 2 3.4 3.5
//! file mod: What stays in the module file itself.
//! 7
//! ```
//!
//! `file mod` keeps its items in `mod.rs`, at the level they were written at.
//! Every map id must be named exactly once — a whole item by `N`, or every one
//! of a container's members by `N.M` — which [`crate::assign`] checks.

/// A map id: a whole item, or one member of a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Id {
    Whole(usize),
    Member(usize, usize),
}

/// One file of a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanFile {
    /// The module name: `pool` makes `pool.rs`; `mod` is `mod.rs` itself.
    pub name: String,
    /// Its one-line purpose, its `//!` line.
    pub doc: String,
    pub ids: Vec<Id>,
}

/// The name of the file that keeps items in the module itself.
pub const MOD: &str = "mod";

/// Parse a plan.
pub fn parse(text: &str) -> Result<Vec<PlanFile>, String> {
    let mut files: Vec<PlanFile> = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("file ") {
            let (name, doc) = rest
                .split_once(':')
                .ok_or_else(|| format!("line {}: `file <name>: <purpose>`", n + 1))?;
            let name = name.trim().to_string();
            if !is_module_name(&name) {
                return Err(format!("line {}: `{name}` is not a module name", n + 1));
            }
            if files.iter().any(|f| f.name == name) {
                return Err(format!("line {}: file `{name}` named twice", n + 1));
            }
            files.push(PlanFile {
                name,
                doc: doc.trim().to_string(),
                ids: Vec::new(),
            });
            continue;
        }
        let file = files
            .last_mut()
            .ok_or_else(|| format!("line {}: ids before any `file`", n + 1))?;
        for word in line.split(|c: char| c.is_whitespace() || c == ',') {
            if word.is_empty() {
                continue;
            }
            file.ids.push(parse_id(word).ok_or_else(|| {
                format!("line {}: `{word}` is not an id (`N` or `N.M`)", n + 1)
            })?);
        }
    }
    if files.is_empty() {
        return Err("the plan names no file".into());
    }
    Ok(files)
}

fn parse_id(word: &str) -> Option<Id> {
    match word.split_once('.') {
        Some((a, b)) => Some(Id::Member(a.parse().ok()?, b.parse().ok()?)),
        None => Some(Id::Whole(word.parse().ok()?)),
    }
}

/// A plain lowercase Rust module name, not a keyword the split itself uses.
fn is_module_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !matches!(name, "self" | "super" | "crate" | "lib" | "main")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_names_files_and_their_ids() {
        let p = parse(
            "# comment\nfile pool: The pool.\n1 2, 3.4\n3.5\n\nfile mod: Stays.\n7\n",
        )
        .unwrap();
        assert_eq!(
            p,
            vec![
                PlanFile {
                    name: "pool".into(),
                    doc: "The pool.".into(),
                    ids: vec![Id::Whole(1), Id::Whole(2), Id::Member(3, 4), Id::Member(3, 5)],
                },
                PlanFile {
                    name: "mod".into(),
                    doc: "Stays.".into(),
                    ids: vec![Id::Whole(7)],
                },
            ]
        );
    }

    #[test]
    fn a_bad_plan_says_where() {
        assert_eq!(parse("1 2\n").unwrap_err(), "line 1: ids before any `file`");
        assert_eq!(
            parse("file Pool: x\n").unwrap_err(),
            "line 1: `Pool` is not a module name"
        );
        assert_eq!(
            parse("file a: x\nfile a: y\n").unwrap_err(),
            "line 2: file `a` named twice"
        );
        assert_eq!(
            parse("file a: x\n1.x\n").unwrap_err(),
            "line 2: `1.x` is not an id (`N` or `N.M`)"
        );
        assert_eq!(parse("").unwrap_err(), "the plan names no file");
    }
}
