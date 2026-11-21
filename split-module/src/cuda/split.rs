//! Splitting a kernel source by a plan: each plan file becomes a header under
//! `<stem>/`, included where its units stood, so the translation unit the
//! compiler sees is the source as it was.
//!
//! A plan file's units must be one run: consecutive, with nothing that stays
//! between them — an `#include`, an opened block's own lines, a unit of another
//! file. `file mod` keeps units where they are. Every byte goes either to the
//! file or to one part, and the split is refused unless putting each part back
//! in place of its `#include` gives back the source byte for byte.

use std::collections::BTreeMap;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use super::layout::Layout;
use super::scan::Kind;
use crate::plan::{Id, PlanFile, MOD};

/// What a stretch of the source becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Owner {
    File,
    Part(usize),
}

/// A split, built: the file as it now reads, and each part's name and text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outputs {
    pub file: String,
    pub parts: Vec<(String, String)>,
}

/// The stretches of `src` in order, each with its owner: the header, every
/// placeable unit as the plan places it, what stays, and the gaps between.
fn stretches(
    src: &str,
    l: &Layout,
    plan: &[PlanFile],
) -> Result<Vec<(Range<usize>, Owner)>, String> {
    let mut place: BTreeMap<Id, usize> = BTreeMap::new();
    let mut errors = Vec::new();
    for (f, file) in plan.iter().enumerate() {
        for id in &file.ids {
            if place.insert(*id, f).is_some() {
                errors.push(format!("{id:?} is named twice"));
            }
        }
    }
    let owner_of = |f: usize| {
        if plan[f].name == MOD {
            Owner::File
        } else {
            Owner::Part(f)
        }
    };
    let mut units: Vec<(Range<usize>, Owner)> = Vec::new();
    let mut placed = 0usize;
    for (i, item) in l.items.iter().enumerate() {
        let n = i + 1;
        let u = &item.unit;
        match &item.members {
            Some(members) => {
                if place.contains_key(&Id::Whole(n)) {
                    errors.push(format!("{n} ({}) is opened: place its members", u.name));
                }
                let body = u.body.clone().expect("an opened item has a body");
                units.push((u.span.start..body.start, Owner::File));
                for (j, m) in members.iter().enumerate() {
                    let id = Id::Member(n, j + 1);
                    if m.kind == Kind::Kept {
                        units.push((m.span.clone(), Owner::File));
                        continue;
                    }
                    match place.get(&id) {
                        Some(&f) => {
                            placed += 1;
                            units.push((m.span.clone(), owner_of(f)));
                        }
                        None => errors.push(format!("{n}.{} ({}) is not placed", j + 1, m.name)),
                    }
                }
                units.push((body.end..u.span.end, Owner::File));
            }
            None if u.kind == Kind::Kept => {
                if place.contains_key(&Id::Whole(n)) {
                    errors.push(format!(
                        "{n} ({}) stays in the file: do not name it",
                        u.name
                    ));
                }
                units.push((u.span.clone(), Owner::File));
            }
            None => match place.get(&Id::Whole(n)) {
                Some(&f) => {
                    placed += 1;
                    units.push((u.span.clone(), owner_of(f)));
                }
                None => errors.push(format!("{n} ({}) is not placed", u.name)),
            },
        }
    }
    if placed != place.len() && errors.is_empty() {
        errors.push("the plan names ids the file does not have".into());
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    // The stretches between, each to whoever owns both sides of it.
    let mut out = vec![(l.header.clone(), Owner::File)];
    let mut at = l.header.end;
    for (k, (r, o)) in units.iter().enumerate() {
        if r.start > at {
            let next_same = units.get(k).map(|(_, o2)| o2);
            let prev = out.last().map(|(_, o)| *o);
            let gap_owner = match (prev, next_same) {
                (Some(Owner::Part(a)), Some(Owner::Part(b))) if a == *b => Owner::Part(a),
                _ => Owner::File,
            };
            out.push((at..r.start, gap_owner));
        }
        out.push((r.clone(), *o));
        at = r.end;
    }
    if at < src.len() {
        out.push((at..src.len(), Owner::File));
    }
    // Each part one run.
    let mut seen: Vec<usize> = Vec::new();
    let mut last: Option<usize> = None;
    for (_, o) in &out {
        let current = match o {
            Owner::Part(f) => Some(*f),
            Owner::File => None,
        };
        if let Some(f) = current {
            if last != Some(f) {
                if seen.contains(&f) {
                    return Err(format!(
                        "file `{}`'s units are not one run: something stays between them",
                        plan[f].name
                    ));
                }
                seen.push(f);
            }
        }
        last = current;
    }
    Ok(out)
}

/// The `#include` line that stands for a part.
fn include_line(stem: &str, name: &str) -> String {
    format!("#include \"{stem}/{name}.cuh\"\n")
}

/// The comment a part opens with, from its plan file: its one-line purpose
/// as a sentence, then the rest, each paragraph wrapped at 80 columns.
fn part_doc(f: &PlanFile) -> String {
    let mut first = f.doc.trim().to_string();
    if let Some(c) = first.chars().next() {
        first.replace_range(..c.len_utf8(), &c.to_uppercase().to_string());
    }
    let mut doc = wrapped(&first);
    if !f.more.is_empty() {
        doc.push_str("//\n");
        for l in &f.more {
            if l.trim().is_empty() {
                doc.push_str("//\n");
            } else {
                doc.push_str(&wrapped(l));
            }
        }
    }
    doc.push('\n');
    doc
}

/// `text` as `//` lines of at most 80 columns, broken between words.
fn wrapped(text: &str) -> String {
    let mut out = String::new();
    let mut line = String::from("//");
    for word in text.split_whitespace() {
        if line.len() > 2 && line.len() + 1 + word.len() > 80 {
            out.push_str(&line);
            out.push('\n');
            line = String::from("//");
        }
        line.push(' ');
        line.push_str(word);
    }
    out.push_str(&line);
    out.push('\n');
    out
}

/// Build the split of `src`, laid out as `l`, by `plan`, for a file whose
/// stem is `stem`; refused unless it gives `src` back.
pub fn split(src: &str, l: &Layout, plan: &[PlanFile], stem: &str) -> Result<Outputs, String> {
    let stretches = stretches(src, l, plan)?;
    let mut file = String::new();
    let mut bodies: BTreeMap<usize, String> = BTreeMap::new();
    let mut last: Option<usize> = None;
    for (r, o) in &stretches {
        match o {
            Owner::File => {
                file.push_str(&src[r.clone()]);
                last = None;
            }
            Owner::Part(f) => {
                if last != Some(*f) {
                    file.push_str(&include_line(stem, &plan[*f].name));
                }
                bodies.entry(*f).or_default().push_str(&src[r.clone()]);
                last = Some(*f);
            }
        }
    }
    let parts: Vec<(String, String)> = bodies
        .iter()
        .map(|(f, body)| {
            (
                plan[*f].name.clone(),
                format!("{}{body}", part_doc(&plan[*f])),
            )
        })
        .collect();
    // Put back, the parts must give the source.
    let mut back = file.clone();
    for (f, body) in &bodies {
        let line = include_line(stem, &plan[*f].name);
        back = back.replacen(&line, body, 1);
    }
    if back != src {
        let at = back
            .bytes()
            .zip(src.bytes())
            .position(|(a, b)| a != b)
            .unwrap_or(back.len().min(src.len()));
        let show = |s: &str| {
            s[at.saturating_sub(40)..(at + 40).min(s.len())]
                .escape_debug()
                .to_string()
        };
        return Err(format!(
            "the parts put back in place do not give the source, from byte {at}: refused\n  source: {}\n  put back: {}",
            show(src),
            show(&back)
        ));
    }
    Ok(Outputs { file, parts })
}

/// Where a split of `original` writes: the parts' folder and its kept copy.
pub fn targets(original: &Path) -> Result<(PathBuf, PathBuf, String), String> {
    let name = original
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("the file has no name")?;
    let stem = name
        .strip_suffix(".cuh")
        .or_else(|| name.strip_suffix(".cu"))
        .ok_or("not a .cu or .cuh file")?
        .to_string();
    Ok((
        original.with_file_name(&stem),
        original.with_file_name(format!("{name}.presplit")),
        stem,
    ))
}

/// Write `out` for `original`; returns each file written and its size.
pub fn write(original: &Path, out: &Outputs) -> Result<Vec<(PathBuf, usize)>, String> {
    let (dir, presplit, _) = targets(original)?;
    if presplit.exists() {
        return Err(format!(
            "{} exists: finish or undo the last split first",
            presplit.display()
        ));
    }
    let mut files: Vec<(PathBuf, &str)> = vec![(original.to_path_buf(), out.file.as_str())];
    for (name, text) in &out.parts {
        let p = dir.join(format!("{name}.cuh"));
        if p.exists() {
            return Err(format!("{} exists already", p.display()));
        }
        files.push((p, text.as_str()));
    }
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let tmp = |p: &Path| p.with_extension("split-tmp");
    for (p, text) in &files {
        fs::write(tmp(p), text).map_err(|e| format!("{}: {e}", tmp(p).display()))?;
    }
    fs::rename(original, &presplit).map_err(|e| format!("keeping the original: {e}"))?;
    let mut written = Vec::new();
    for (p, text) in &files {
        fs::rename(tmp(p), p).map_err(|e| format!("{}: {e}", p.display()))?;
        written.push((p.clone(), text.len()));
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::layout::layout;
    use crate::plan::parse;

    const SRC: &str =
        "// Kernels.\n\n#pragma once\n#include \"a.cuh\"\n\n// One.\nint a;\n\nint b;\n\nint c;\n";

    /// **Each part is included where its run stood**, the file keeps its
    /// header and includes, and the parts put back give the source.
    #[test]
    fn parts_are_included_where_they_stood() {
        let l = layout(SRC).unwrap();
        let plan = parse("file ab: a and b.\n3 4\nfile mod: Stays.\n5\n").unwrap();
        let out = split(SRC, &l, &plan, "k").unwrap();
        assert_eq!(
            out.file,
            "// Kernels.\n\n#pragma once\n#include \"a.cuh\"\n\n#include \"k/ab.cuh\"\n\nint c;\n"
        );
        assert_eq!(
            out.parts,
            vec![(
                "ab".to_string(),
                "// A and b.\n\n// One.\nint a;\n\nint b;\n".to_string()
            )]
        );
    }

    /// **A part's comment is a sentence wrapped at 80 columns.**
    #[test]
    fn a_parts_comment_is_wrapped() {
        let plan = parse(&format!("file p: {}.\n1\n", "the word ".repeat(20).trim())).unwrap();
        let doc = part_doc(&plan[0]);
        assert!(doc.starts_with("// The word the word"));
        assert!(doc.lines().all(|l| l.len() <= 80), "{doc}");
        assert_eq!(doc.lines().filter(|l| l.starts_with("//")).count(), 3);
    }

    /// **A part is one run**, an `#include` is never named, and every unit is
    /// placed.
    #[test]
    fn a_bad_plan_is_refused() {
        let l = layout(SRC).unwrap();
        let split_of = |p: &str| split(SRC, &l, &parse(p).unwrap(), "k").unwrap_err();
        assert!(split_of("file x: X.\n3 5\nfile y: Y.\n4\n").contains("not one run"));
        assert!(split_of("file x: X.\n2 3 4 5\n").contains("stays in the file"));
        assert!(split_of("file x: X.\n3 4\n").contains("5 (c) is not placed"));
    }
}
