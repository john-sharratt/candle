//! The split's outputs: `mod.rs` and one file per plan file, built from the
//! source's own bytes.
//!
//! **Every byte is accounted for before anything is built.** The parts a plan
//! places — what stays in `mod.rs`, each whole item, each container's head,
//! members and tail — must cover the source exactly once apart from the
//! whitespace between them ([`account`]); anything else is refused. Then each
//! output is its parts in source order: a child file's with the edits moving
//! code one module deeper (`rewrite`), `mod.rs`'s untouched.

use std::collections::HashSet;
use std::ops::Range;

use crate::assign::Placement;
use crate::layout::Layout;
use crate::map::{bare_name, identifiers};
use crate::plan::{PlanFile, MOD};
use crate::rewrite::{apply, share_edits, Edit};

/// The edits for the members of a `mod { }` split across files
/// (`rewrite::share_edits`), merged with the move's own.
#[derive(Debug, Clone, Default)]
pub struct Shares {
    /// For a share in a child file: the move's edits and the share edits
    /// seeing as far as the module every share is under.
    pub in_child: Vec<Edit>,
    /// For a share kept in `mod.rs`: the share edits alone.
    pub in_mod: Vec<Edit>,
    /// Nothing, for what stays as it was.
    pub none: Vec<Edit>,
}

impl Shares {
    /// The share edits for `source` beside the move's `edits`.
    pub fn new(source: &str, edits: &[Edit]) -> Result<Self, String> {
        let mut in_child = edits.to_vec();
        in_child.extend(share_edits(source, "pub(in super::super)")?);
        in_child.sort_by_key(|e| (e.range.start, e.range.end));
        Ok(Self {
            in_child,
            in_mod: share_edits(source, "pub(super)")?,
            none: Vec::new(),
        })
    }
}

/// The path from one share of a split `mod { }` to another's: from a share in
/// `mod.rs` (`from_mod`) or in a child file, to the share in file `to`.
fn share_path(to: &str, module: &str, from_mod: bool) -> String {
    match (from_mod, to == MOD) {
        (true, _) => format!("super::{to}::{module}"),
        (false, true) => format!("super::super::{module}"),
        (false, false) => format!("super::super::{to}::{module}"),
    }
}

/// A split, built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outputs {
    pub mod_rs: String,
    /// Each child module's name and text, in plan order.
    pub children: Vec<(String, String)>,
}

/// How far a child's items are seen, for its re-export in `mod.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Seen {
    Module,
    Crate,
    Everywhere,
}

/// Build the split of `source` laid out as `layout`, placed as `placed` by
/// `plan`, with `edits` for what moves.
pub fn assemble(
    source: &str,
    layout: &Layout,
    plan: &[PlanFile],
    placed: &[Placement],
    edits: &[Edit],
    shares: &Shares,
) -> Result<Outputs, String> {
    account(source, layout, placed)?;
    let in_mod = plan.iter().position(|f| f.name == MOD);
    // Each output's pieces, with where they began in the source.
    let mut pieces: Vec<Vec<(usize, String)>> = vec![Vec::new(); plan.len()];
    let mut seen: Vec<Seen> = vec![Seen::Module; plan.len()];
    // Whether everything a file holds is `#[cfg(test)]`.
    let mut test_only: Vec<bool> = vec![true; plan.len()];
    // Whether a file declares anything a glob import of it can name — one
    // holding only methods of an `impl` declares nothing.
    let mut names: Vec<bool> = vec![false; plan.len()];
    for (item, place) in layout.items.iter().zip(placed) {
        match place {
            Placement::Whole(f) => {
                let own = if Some(*f) == in_mod {
                    &shares.none
                } else {
                    edits
                };
                let text = piece(source, item.span.clone(), own)?;
                seen[*f] = seen[*f].max(seen_of(&source[item.span.clone()], &item.name));
                test_only[*f] &= is_test(&source[item.span.clone()]);
                names[*f] |= bare_name(&item.name).is_some();
                pieces[*f].push((item.span.start, text));
            }
            Placement::Members(to) => {
                let body = item
                    .body
                    .as_ref()
                    .expect("members are placed only in a body");
                let module = item.name.strip_prefix("mod ");
                // The files the members go to, and what each share declares
                // and uses — so a share of a `mod { }` imports the shares it
                // calls into.
                let files: Vec<usize> =
                    (0..plan.len()).filter(|f| to.contains(&Some(*f))).collect();
                let declares = |f: usize| -> HashSet<&str> {
                    body.members
                        .iter()
                        .zip(to)
                        .filter(|(_, t)| **t == Some(f))
                        .filter_map(|(m, _)| bare_name(&m.name))
                        .collect()
                };
                let uses = |f: usize| -> HashSet<&str> {
                    body.members
                        .iter()
                        .zip(to)
                        .filter(|(_, t)| **t == Some(f))
                        .flat_map(|(m, _)| identifiers(&source[m.span.clone()]))
                        .collect()
                };
                for &f in &files {
                    let keep = Some(f) == in_mod;
                    let own_edits = match (module, keep) {
                        (Some(_), true) => &shares.in_mod,
                        (Some(_), false) => &shares.in_child,
                        (None, true) => &shares.none,
                        (None, false) => edits,
                    };
                    let frame = if keep { &shares.none } else { edits };
                    test_only[f] &= is_test(&source[body.head.clone()]);
                    let mut text = piece(source, body.head.clone(), frame)?;
                    // A share in a child file is reached by the other shares
                    // through its file: its `mod` is seen by the module they
                    // are all under.
                    if let (Some(module), false) = (module, keep) {
                        let decl = format!("mod {module}");
                        if let Some(at) = text.rfind(&decl) {
                            let before = text[..at].trim_end();
                            if !before.ends_with(')') && !before.ends_with("pub") {
                                text.insert_str(at, "pub(super) ");
                            }
                        }
                    }
                    for (j, m) in body.members.iter().enumerate() {
                        if to[j].is_none() {
                            text.push('\n');
                            text.push_str(&piece(source, m.span.clone(), own_edits)?);
                            text.push('\n');
                        }
                    }
                    if let Some(module) = module {
                        let used = uses(f);
                        for &g in files.iter().filter(|&&g| g != f) {
                            if declares(g).iter().any(|n| used.contains(n)) {
                                let path = share_path(&plan[g].name, module, keep);
                                text.push_str(&format!("\n    use {path}::*;\n"));
                            }
                        }
                    }
                    for (j, m) in body.members.iter().enumerate() {
                        if to[j] == Some(f) {
                            text.push('\n');
                            text.push_str(&piece(source, m.span.clone(), own_edits)?);
                            text.push('\n');
                        }
                    }
                    text.push_str(&piece(source, body.tail.clone(), frame)?);
                    let first = (0..to.len()).find(|&j| to[j] == Some(f)).expect("a share");
                    pieces[f].push((body.members[first].span.start, text));
                }
            }
        }
    }
    let children: Vec<(String, String)> = plan
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name != MOD)
        .map(|(i, f)| {
            let mut out = format!("//! {}\n", f.doc);
            if !f.more.is_empty() {
                out.push_str("//!\n");
                for l in &f.more {
                    if l.is_empty() {
                        out.push_str("//!\n");
                    } else {
                        out.push_str(&format!("//! {l}\n"));
                    }
                }
            }
            out.push_str("\nuse super::*;\n");
            let mut mine = pieces[i].clone();
            mine.sort_by_key(|(at, _)| *at);
            for (_, text) in mine {
                out.push('\n');
                out.push_str(&text);
                out.push('\n');
            }
            (f.name.clone(), out)
        })
        .collect();
    // `mod.rs`: what stayed, and the items `file mod` keeps, in source order,
    // then the children and their re-exports.
    // Each with its end in the source, when it is a kept part whose gap to
    // the next one can be kept as written.
    let mut kept: Vec<(usize, Option<usize>, String)> = layout
        .kept
        .iter()
        .map(|k| {
            (
                k.span.start,
                Some(k.span.end),
                source[k.span.clone()].to_string(),
            )
        })
        .collect();
    if let Some(m) = in_mod {
        kept.extend(pieces[m].iter().map(|(at, t)| (*at, None, t.clone())));
    }
    kept.sort_by_key(|(at, _, _)| *at);
    let mut mod_rs = String::new();
    for (i, (_, end, text)) in kept.iter().enumerate() {
        // Two kept parts the source wrote with only whitespace between them —
        // the lines of one doc comment, a run of `use` lines — keep that gap.
        let gap = match (end, kept.get(i + 1)) {
            (Some(e), Some((next, Some(_), _))) if source[*e..*next].trim().is_empty() => {
                Some(&source[*e..*next])
            }
            _ => None,
        };
        match gap {
            Some(g) => {
                mod_rs.push_str(text);
                mod_rs.push_str(g);
            }
            None => {
                mod_rs.push_str(text.trim_end());
                mod_rs.push_str("\n\n");
            }
        }
    }
    for (i, f) in plan.iter().enumerate().filter(|(_, f)| f.name != MOD) {
        if test_only[i] {
            mod_rs.push_str("#[cfg(test)]\n");
        }
        mod_rs.push_str(&format!("mod {};\n", f.name));
    }
    mod_rs.push('\n');
    // A test-only file is reached by the other shares of its test module,
    // never through `mod.rs`.
    for (i, f) in plan
        .iter()
        .enumerate()
        .filter(|(i, f)| f.name != MOD && !test_only[*i] && names[*i])
    {
        let vis = match seen[i] {
            Seen::Everywhere => "pub ",
            Seen::Crate => "pub(crate) ",
            Seen::Module => "",
        };
        mod_rs.push_str(&format!("{vis}use {}::*;\n", f.name));
    }
    Ok(Outputs { mod_rs, children })
}

/// `source[range]` with `edits` applied — none for what stays as it was.
fn piece(source: &str, range: Range<usize>, edits: &[Edit]) -> Result<String, String> {
    apply(source, range, edits)
}

/// Whether an item, by the attributes over it, is built only for tests.
fn is_test(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .take_while(|l| l.is_empty() || l.starts_with("//") || l.starts_with("#["))
        .any(|l| l.starts_with("#[cfg(test)]"))
}

/// How far an item's own declaration says it is seen. An `impl` or a
/// `mod { }` is re-exported by nothing.
fn seen_of(text: &str, name: &str) -> Seen {
    if name.starts_with("impl ") || name.starts_with("mod ") {
        return Seen::Module;
    }
    let code = text.lines().map(str::trim).find(|l| {
        !l.is_empty()
            && !l.starts_with("//")
            && !l.starts_with("#[")
            && !l.starts_with("/*")
            && !l.starts_with('*')
    });
    match code {
        Some(l) if l.starts_with("pub(crate)") => Seen::Crate,
        Some(l) if l.starts_with("pub ") => Seen::Everywhere,
        _ => Seen::Module,
    }
}

/// Refuse a placement that does not claim every byte of `source` exactly once
/// apart from whitespace between parts.
pub fn account(source: &str, layout: &Layout, placed: &[Placement]) -> Result<(), String> {
    let mut claims: Vec<Range<usize>> = layout.kept.iter().map(|k| k.span.clone()).collect();
    for (item, place) in layout.items.iter().zip(placed) {
        match place {
            Placement::Whole(_) => claims.push(item.span.clone()),
            Placement::Members(_) => {
                let body = item
                    .body
                    .as_ref()
                    .ok_or("a placement by member of an item with no body")?;
                claims.push(body.head.clone());
                claims.extend(body.members.iter().map(|m| m.span.clone()));
                claims.push(body.tail.clone());
            }
        }
    }
    claims.sort_by_key(|r| (r.start, r.end));
    let mut at = 0usize;
    for r in &claims {
        if r.start < at {
            return Err(format!(
                "bytes {r:?} are claimed twice (the last claim ended at {at})"
            ));
        }
        if let Some(off) = source[at..r.start].find(|c: char| !c.is_whitespace()) {
            return Err(format!("byte {} belongs to no part", at + off));
        }
        at = r.end;
    }
    if let Some(off) = source[at..].find(|c: char| !c.is_whitespace()) {
        return Err(format!("byte {} belongs to no part", at + off));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assign::assign;
    use crate::layout::layout;
    use crate::plan::parse;
    use crate::rewrite::edits;

    fn split(src: &str, plan: &str) -> Outputs {
        let l = layout(src).unwrap();
        let p = parse(plan).unwrap();
        let placed = assign(&l, &p).unwrap();
        let e = edits(src).unwrap();
        let shares = Shares::new(src, &e).unwrap();
        assemble(src, &l, &p, &placed, &e, &shares).unwrap()
    }

    /// **A split moves each item's own bytes**, made visible to its old
    /// module, into its file; `mod.rs` keeps the uses, declares the children
    /// and re-exports them as far as their items were seen.
    #[test]
    fn items_move_into_their_files() {
        let src = "//! The pool.\n\nuse std::fmt;\n\npub struct Pool;\n\nfn helper() {}\n";
        let out = split(
            src,
            "file pool: The pool type.\n1\nfile help: Its helper.\n//! Sized for the card.\n2\n",
        );
        assert_eq!(
            out.mod_rs,
            "//! The pool.\n\nuse std::fmt;\n\nmod pool;\nmod help;\n\npub use pool::*;\nuse help::*;\n"
        );
        assert_eq!(
            out.children,
            vec![
                (
                    "pool".to_string(),
                    "//! The pool type.\n\nuse super::*;\n\npub struct Pool;\n".to_string()
                ),
                (
                    "help".to_string(),
                    "//! Its helper.\n//!\n//! Sized for the card.\n\nuse super::*;\n\npub(super) fn helper() {}\n"
                        .to_string()
                ),
            ]
        );
    }

    /// **An impl's methods go to different files**, each inside the impl's
    /// own head and tail; what `file mod` keeps is left untouched in `mod.rs`.
    #[test]
    fn an_impl_splits_by_member() {
        let src = "struct A;\n\nimpl A {\n    fn x(&self) {}\n\n    fn y(&self) {}\n}\n";
        let out = split(src, "file mod: Stays.\n1 2.1\nfile y: The y.\n2.2\n");
        assert_eq!(
            out.mod_rs,
            "struct A;\n\nimpl A {\n    fn x(&self) {}\n}\n\nmod y;\n\n"
        );
        assert_eq!(
            out.children,
            vec![(
                "y".to_string(),
                "//! The y.\n\nuse super::*;\n\nimpl A {\n    pub(super) fn y(&self) {}\n}\n"
                    .to_string()
            )]
        );
    }

    /// **A test module splits by test**: its `use` lines go with every share,
    /// its helpers are seen by every share, and a share imports the shares
    /// whose helpers it calls.
    #[test]
    fn a_test_module_carries_its_uses_into_every_share() {
        let src = "#[cfg(test)]\nmod tests {\n    use super::*;\n\n    fn helper() -> u8 { 1 }\n\n    #[test]\n    fn b() { helper(); }\n}\n";
        let out = split(
            src,
            "file one: The helper.\n1.2\nfile two: The test.\n1.3\n",
        );
        assert_eq!(
            out.children[0].1,
            "//! The helper.\n\nuse super::*;\n\n#[cfg(test)]\npub(super) mod tests {\n    use super::*;\n\n    \
             pub(in super::super) fn helper() -> u8 { 1 }\n}\n"
        );
        assert_eq!(
            out.children[1].1,
            "//! The test.\n\nuse super::*;\n\n#[cfg(test)]\npub(super) mod tests {\n    use super::*;\n\n    \
             use super::super::one::tests::*;\n\n    #[test]\n    pub(in super::super) fn b() { helper(); }\n}\n"
        );
    }

    /// **Nothing is moved that the accounting cannot place**: a claim over
    /// another, or a byte no part holds, is refused.
    #[test]
    fn the_accounting_refuses_a_lost_or_doubled_byte() {
        let src = "fn a() {}\n\nfn b() {}\n";
        let l = layout(src).unwrap();
        assert!(account(src, &l, &[Placement::Whole(0), Placement::Whole(0)]).is_ok());
        let mut doubled = l.clone();
        doubled.items[1].span.start = 3;
        assert_eq!(
            account(src, &doubled, &[Placement::Whole(0), Placement::Whole(0)]).unwrap_err(),
            "bytes 3..20 are claimed twice (the last claim ended at 9)"
        );
        let mut lost = l.clone();
        lost.items[1].span.start = 14;
        assert_eq!(
            account(src, &lost, &[Placement::Whole(0), Placement::Whole(0)]).unwrap_err(),
            "byte 11 belongs to no part"
        );
    }
}
