//! A split plan: the files to make, each with its one-line purpose and the map
//! ids it holds.
//!
//! ```text
//! file pool: The block pool and its growth.
//! //! Blocks are handed out from one arena and grown in place when it fills;
//! //! a grow never moves a block a reader holds.
//! 1 2 3.4 3.5
//! file mod: What stays in the module file itself.
//! 7
//! ```
//!
//! The `//!` lines after a `file` line are the rest of that file's module
//! doc, written under its one-line summary as they are.
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
    /// Its one-line purpose, its first `//!` line.
    pub doc: String,
    /// The rest of its module doc, a line each, without the `//!`.
    pub more: Vec<String>,
    pub ids: Vec<Id>,
}

/// The name of the file that keeps items in the module itself.
pub const MOD: &str = "mod";

/// Parse a plan.
pub fn parse(text: &str) -> Result<Vec<PlanFile>, String> {
    let mut files: Vec<PlanFile> = Vec::new();
    // The file the lines under the last `file` line go to.
    let mut current: Option<usize> = None;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        // `=== <map>` heads a plan in the planner's answer; it says nothing
        // about the plan.
        if line.is_empty() || line.starts_with('#') || line.starts_with("===") {
            continue;
        }
        if let Some(doc) = line.strip_prefix("//!") {
            let file = current
                .map(|c| &mut files[c])
                .ok_or_else(|| format!("line {}: a doc line before any `file`", n + 1))?;
            file.more
                .push(doc.strip_prefix(' ').unwrap_or(doc).trim_end().to_string());
            continue;
        }
        // A file line is `file <name>: <purpose>`; `<name>: <purpose>` without
        // the keyword says the same, since no id holds a `:`.
        if let Some(rest) = line.strip_prefix("file ").or_else(|| bare_file_line(line)) {
            let (name, doc) = rest
                .split_once(':')
                .ok_or_else(|| format!("line {}: `file <name>: <purpose>`", n + 1))?;
            // A name written with its extension names the same file.
            let name = name.trim();
            let name = name
                .strip_suffix(".rs")
                .or_else(|| name.strip_suffix(".cuh"))
                .unwrap_or(name)
                .to_string();
            if !is_module_name(&name) {
                return Err(format!("line {}: `{name}` is not a module name", n + 1));
            }
            // `file mod` may come back: what stays in the file is often a few
            // short runs between the parts.
            if let Some(at) = files.iter().position(|f| f.name == name) {
                if name != MOD {
                    return Err(format!("line {}: file `{name}` named twice", n + 1));
                }
                current = Some(at);
                continue;
            }
            files.push(PlanFile {
                name,
                doc: doc.trim().to_string(),
                more: Vec::new(),
                ids: Vec::new(),
            });
            current = Some(files.len() - 1);
            continue;
        }
        let file = current
            .map(|c| &mut files[c])
            .ok_or_else(|| format!("line {}: ids before any `file`", n + 1))?;
        for word in line.split(|c: char| c.is_whitespace() || c == ',') {
            if word.is_empty() {
                continue;
            }
            file.ids.push(
                parse_id(word).ok_or_else(|| {
                    format!("line {}: `{word}` is not an id (`N` or `N.M`)", n + 1)
                })?,
            );
        }
    }
    if files.is_empty() {
        return Err("the plan names no file".into());
    }
    Ok(files)
}

/// `line` itself when it reads `<module name>: <purpose>`.
fn bare_file_line(line: &str) -> Option<&str> {
    let (name, _) = line.split_once(':')?;
    let name = name.trim();
    let name = name
        .strip_suffix(".rs")
        .or_else(|| name.strip_suffix(".cuh"))
        .unwrap_or(name);
    is_module_name(name).then_some(line)
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
            "# comment\nfile pool: The pool.\n//! It grows in place.\n//!\n//! Readers keep their blocks.\n\
             1 2, 3.4\n3.5\n\nfile mod: Stays.\n7\n",
        )
        .unwrap();
        assert_eq!(
            p,
            vec![
                PlanFile {
                    name: "pool".into(),
                    doc: "The pool.".into(),
                    more: vec![
                        "It grows in place.".into(),
                        String::new(),
                        "Readers keep their blocks.".into()
                    ],
                    ids: vec![
                        Id::Whole(1),
                        Id::Whole(2),
                        Id::Member(3, 4),
                        Id::Member(3, 5)
                    ],
                },
                PlanFile {
                    name: "mod".into(),
                    doc: "Stays.".into(),
                    more: Vec::new(),
                    ids: vec![Id::Whole(7)],
                },
            ]
        );
        assert_eq!(
            parse("//! x\n").unwrap_err(),
            "line 1: a doc line before any `file`"
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

    /// **`file mod` can come back**, and what follows it again stays in the
    /// file; any other file named twice is refused.
    #[test]
    fn file_mod_may_come_back() {
        let p = parse("file mod: Stays.\n1\nfile a: A.\n2\nfile mod: Stays.\n3\n").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].ids, vec![Id::Whole(1), Id::Whole(3)]);
        assert_eq!(p[1].ids, vec![Id::Whole(2)]);
    }

    /// **A planner's slips read as it meant them**: the `=== <map>` head of
    /// its answer is skipped, and a file line without `file` is still one.
    #[test]
    fn a_map_head_and_a_bare_file_line_are_read() {
        let p =
            parse("=== maps/a.rs.map\n\nfile mod: Stays.\n1\ntests_rows.rs: The tests.\n2.1 2.2\n")
                .unwrap();
        assert_eq!(
            p.iter()
                .map(|f| (f.name.as_str(), f.doc.as_str(), f.ids.clone()))
                .collect::<Vec<_>>(),
            vec![
                ("mod", "Stays.", vec![Id::Whole(1)]),
                (
                    "tests_rows",
                    "The tests.",
                    vec![Id::Member(2, 1), Id::Member(2, 2)]
                ),
            ]
        );
        assert_eq!(
            parse("file a: x\nNot a name: y\n").unwrap_err(),
            "line 2: `Not` is not an id (`N` or `N.M`)"
        );
    }
}
