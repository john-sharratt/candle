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
use crate::layout::{is_cfg_test, Item, Layout};
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
    /// The children that are inline modules hoisted into their own files.
    pub hoisted: Vec<String>,
}

/// How far a child's items are seen, for its re-export in `mod.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Seen {
    Module,
    /// `pub(super)`: by the module above the split one.
    Parent,
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
    // Whether a file declares an item or a whole module of its own, not only
    // shares of a split test module: a test-only file that does is
    // re-exported, under `#[cfg(test)]`, for the tests that reach what it
    // holds through the split module's path.
    let mut test_names: Vec<bool> = vec![false; plan.len()];
    // A file named for the inline `mod { }` placed in it holds that module's
    // body; `mod.rs` keeps its declaration, at its place in the source.
    let mut hoisted: Vec<Option<String>> = vec![None; plan.len()];
    let mut declarations: Vec<(usize, String)> = Vec::new();
    // The modules that gather a split test module's shares, at its place.
    let mut gathers: Vec<(usize, String)> = Vec::new();
    for (item, place) in layout.items.iter().zip(placed) {
        match place {
            Placement::Whole(f) if hoists(item, &plan[*f]) => {
                let inner = item.inline.clone().expect("a hoisted module is inline");
                let head = source[item.span.start..inner.start - 1].trim_end();
                declarations.push((item.span.start, format!("{head};")));
                hoisted[*f] = Some(dedent(&source[inner]));
            }
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
                test_names[*f] |= item.name.starts_with("mod ") || bare_name(&item.name).is_some();
                names[*f] |= item.name.starts_with("mod ") || item.name.starts_with("extern ");
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
                // A `#[test]` function is the harness's alone and stays
                // private, so no other share imports it.
                let declares = |f: usize| -> HashSet<&str> {
                    body.members
                        .iter()
                        .zip(to)
                        .filter(|(_, t)| **t == Some(f))
                        .filter(|(m, _)| !is_test_fn(&source[m.span.clone()]))
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
                // How far a re-export of share `f` reaches: as far as the
                // widest of its items was written to be seen — `pub(crate)`
                // helpers of a test module are reached from other modules
                // of the crate — and no further, since a glob re-export of
                // nothing that reaches so far is refused as a warning.
                let reach = |f: usize| -> &'static str {
                    let seen = body
                        .members
                        .iter()
                        .zip(to)
                        .filter(|(_, t)| **t == Some(f))
                        .map(|(m, _)| seen_of(&source[m.span.clone()], &m.name))
                        .max()
                        .unwrap_or(Seen::Module);
                    match seen {
                        Seen::Everywhere => "pub",
                        Seen::Crate => "pub(crate)",
                        Seen::Parent | Seen::Module => "pub(super)",
                    }
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
                            let mut line = piece(source, m.span.clone(), own_edits)?;
                            // `use self::x::…` names a module another share
                            // declares: from this share it lies in that one.
                            if let Some(module) = module {
                                let trimmed = line.trim_start();
                                if let Some(rest) = trimmed.strip_prefix("use self::") {
                                    let first: String = rest
                                        .chars()
                                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                                        .collect();
                                    let as_mod = format!("mod {first}");
                                    let owner = files.iter().copied().find(|&g| {
                                        body.members
                                            .iter()
                                            .zip(to)
                                            .any(|(m, t)| *t == Some(g) && m.name == as_mod)
                                            || declares(g).contains(first.as_str())
                                    });
                                    if let Some(g) = owner.filter(|&g| g != f) {
                                        let path = share_path(&plan[g].name, module, keep);
                                        let at = line.len() - trimmed.len() + "use ".len();
                                        line.replace_range(at..at + "self".len(), &path);
                                    }
                                }
                            }
                            text.push('\n');
                            text.push_str(&line);
                            text.push('\n');
                        }
                    }
                    if let Some(module) = module {
                        let used = uses(f);
                        let holds_file_module = body.members.iter().zip(to).any(|(m, t)| {
                            *t == Some(f)
                                && m.name.starts_with("mod ")
                                && source[m.span.clone()].trim_end().ends_with(';')
                        });
                        for &g in files.iter().filter(|&&g| g != f) {
                            let path = share_path(&plan[g].name, module, keep);
                            if keep {
                                // The share in `mod.rs` keeps the module's
                                // path whole: everything another share holds
                                // is reached through it, as before the split.
                                text.push_str(&format!("\n    {} use {path}::*;\n", reach(g)));
                            } else {
                                // By name, not by glob: a name the module
                                // declares outranks one its `use super::*`
                                // brings in, and an explicit import keeps that.
                                // A share declaring a module in a file of its
                                // own (`mod golden;`) takes every name: that
                                // file's `use super::*` reads them through it.
                                let mut wanted: Vec<&str> = declares(g)
                                    .into_iter()
                                    .filter(|n| holds_file_module || used.contains(n))
                                    .collect();
                                wanted.sort_unstable();
                                match wanted.as_slice() {
                                    [] => {}
                                    [one] => text.push_str(&format!("\n    use {path}::{one};\n")),
                                    many => text.push_str(&format!(
                                        "\n    use {path}::{{{}}};\n",
                                        many.join(", ")
                                    )),
                                }
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
                // With no share left in `mod.rs`, the module is declared there
                // still, gathering its shares: what other modules reach as
                // `<module>::x` they reach as they did.
                if let (Some(module), None) = (module, in_mod.filter(|m| files.contains(m))) {
                    let mut gather = source[body.head.clone()].to_string();
                    for &g in &files {
                        gather.push_str(&format!(
                            "\n    {} use super::{}::{module}::*;",
                            reach(g),
                            plan[g].name
                        ));
                    }
                    gather.push_str("\n}");
                    gathers.push((item.span.start, gather));
                }
            }
        }
    }
    for (i, f) in plan.iter().enumerate() {
        if hoisted[i].is_some() && !pieces[i].is_empty() {
            return Err(format!(
                "file `{}` holds the module `{}` it is named for, and nothing else can go in it",
                f.name, f.name
            ));
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
            if let Some(body) = &hoisted[i] {
                out.push('\n');
                out.push_str(body);
                return (f.name.clone(), out);
            }
            out.push_str("\nuse super::*;\n");
            // In source order, with the tests after everything else: an item
            // after a test module is one a reader of the file never reaches.
            let mut mine = pieces[i].clone();
            mine.sort_by_key(|(at, text)| (is_test(text), *at));
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
    // What `mod.rs` holds for tests — a test module `file mod` keeps, the
    // module gathering a split one's shares — goes after its other items, in
    // source order.
    let mut tests_last: Vec<(usize, String)> = gathers;
    if let Some(m) = in_mod {
        for (at, t) in &pieces[m] {
            if is_test(t) {
                tests_last.push((*at, t.clone()));
            } else {
                kept.push((*at, None, t.clone()));
            }
        }
    }
    tests_last.sort_by_key(|(at, _)| *at);
    kept.extend(declarations.into_iter().map(|(at, d)| (at, None, d)));
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
    for (i, f) in plan
        .iter()
        .enumerate()
        .filter(|(i, f)| f.name != MOD && hoisted[*i].is_none())
    {
        if test_only[i] {
            mod_rs.push_str("#[cfg(test)]\n");
        }
        mod_rs.push_str(&format!("mod {};\n", f.name));
    }
    mod_rs.push('\n');
    // A test-only file holding only shares of test modules is reached
    // through those modules, never through a glob of `mod.rs`; one declaring
    // helpers or a whole module of its own is re-exported for the tests.
    for (i, f) in plan.iter().enumerate().filter(|(i, f)| {
        f.name != MOD && names[*i] && hoisted[*i].is_none() && (!test_only[*i] || test_names[*i])
    }) {
        let vis = match seen[i] {
            Seen::Everywhere => "pub ",
            Seen::Crate => "pub(crate) ",
            Seen::Parent => "pub(super) ",
            Seen::Module => "",
        };
        if test_only[i] {
            mod_rs.push_str("#[cfg(test)]\n");
        }
        mod_rs.push_str(&format!("{vis}use {}::*;\n", f.name));
    }
    for (_, text) in tests_last {
        mod_rs.push('\n');
        mod_rs.push_str(text.trim_end());
        mod_rs.push('\n');
    }
    let hoisted = plan
        .iter()
        .zip(&hoisted)
        .filter(|(_, h)| h.is_some())
        .map(|(f, _)| f.name.clone())
        .collect();
    Ok(Outputs {
        mod_rs,
        children,
        hoisted,
    })
}

/// Whether `item` is an inline `mod { }` placed whole in the file named for
/// it, which then becomes the module's own file.
fn hoists(item: &Item, file: &PlanFile) -> bool {
    item.inline.is_some() && file.name != MOD && item.name == format!("mod {}", file.name)
}

/// A module body as a file: its bytes as written, from its first line with
/// anything on it through its last. The indentation it had inside the braces
/// stays — `cargo fmt` re-indents code, and taking it off here would also
/// change what a multi-line string literal holds.
fn dedent(body: &str) -> String {
    let start = body
        .find(|c: char| !c.is_whitespace())
        .map(|at| body[..at].rfind('\n').map_or(0, |nl| nl + 1));
    match start {
        Some(start) => format!("{}\n", body[start..].trim_end()),
        None => String::new(),
    }
}

/// `source[range]` with `edits` applied — none for what stays as it was — and
/// its bare section rules left behind ([`drop_bare_rules`]).
fn piece(source: &str, range: Range<usize>, edits: &[Edit]) -> Result<String, String> {
    Ok(drop_bare_rules(&apply(source, range, edits)?))
}

/// `text` without its bare section rules: a `//` line of nothing but rule
/// characters, which separated one part of the original file from the next
/// and separates nothing in a file of one part. A rule with a title — `// ──
/// Per-row bans ──` — names what follows it, and is kept.
fn drop_bare_rules(text: &str) -> String {
    let is_rule = |line: &str| {
        line.trim()
            .strip_prefix("//")
            .map(str::trim)
            .is_some_and(|rest| {
                rest.chars().count() >= 3
                    && rest
                        .chars()
                        .all(|c| matches!(c, '─' | '━' | '═' | '-' | '=' | '*' | '#' | '~' | '_'))
            })
    };
    if !text.lines().any(is_rule) {
        return text.to_string();
    }
    text.split_inclusive('\n').filter(|l| !is_rule(l)).collect()
}

/// Whether a member, by the attributes over it, is a test the harness runs:
/// `#[test]`, `#[tokio::test]` and the like.
fn is_test_fn(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .take_while(|l| l.is_empty() || l.starts_with("//") || l.starts_with("#["))
        .any(|l| l.starts_with("#[") && (l.contains("test]") || l.contains("::test(")))
}

/// Whether an item, by the attributes over it, is built only for tests.
fn is_test(text: &str) -> bool {
    let leading: Vec<&str> = text
        .lines()
        .take_while(|l| {
            let l = l.trim();
            l.is_empty() || l.starts_with("//") || l.starts_with("#[")
        })
        .collect();
    is_cfg_test(leading.join("\n").as_bytes())
}

/// How far an item's own declaration says it is seen. An `impl` is
/// re-exported by nothing; a `mod { }` moved whole is seen as it says.
fn seen_of(text: &str, name: &str) -> Seen {
    if name.starts_with("impl ") {
        return Seen::Module;
    }
    // An `extern { }` block declares its items, each seen as it says: the
    // block's re-export reaches as far as the widest of them.
    if name.starts_with("extern ") {
        let widest = |prefix: &str| text.lines().any(|l| l.trim_start().starts_with(prefix));
        return if widest("pub fn") || widest("pub static") || widest("pub type") {
            Seen::Everywhere
        } else if widest("pub(crate)") {
            Seen::Crate
        } else if widest("pub(super)") {
            Seen::Parent
        } else {
            Seen::Module
        };
    }
    let code = text.lines().map(str::trim).find(|l| {
        !l.is_empty()
            && !l.starts_with("//")
            && !l.starts_with("#[")
            && !l.starts_with("/*")
            && !l.starts_with('*')
    });
    // A glob re-export passes on each item only as far as the item itself is
    // seen, so a wider one than an item's own reach is harmless; `pub(in …)`
    // is re-exported crate-wide for that reason.
    match code {
        Some(l) if l.starts_with("pub(crate)") || l.starts_with("pub(in ") => Seen::Crate,
        Some(l) if l.starts_with("pub(super)") => Seen::Parent,
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

    /// Each module's file as `<path>.rs`.
    fn flat(below: &str) -> String {
        format!("{below}.rs")
    }

    fn split(src: &str, plan: &str) -> Outputs {
        let l = layout(src).unwrap();
        let p = parse(plan).unwrap();
        let placed = assign(&l, &p).unwrap();
        let e = edits(src, &flat).unwrap();
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

    /// **An item seen by the module above is re-exported to it**: a
    /// `pub(super)` item's file is globbed `pub(super)`, so a sibling of the
    /// split module still reaches it through the module's path.
    #[test]
    fn a_parent_visible_item_is_re_exported_to_the_parent() {
        let src = "pub(super) struct D;\n\npub(in crate::a) fn e() {}\n";
        let out = split(src, "file d: The layout.\n1\nfile e: The e.\n2\n");
        assert_eq!(
            out.mod_rs,
            "mod d;\nmod e;\n\npub(super) use d::*;\npub(crate) use e::*;\n"
        );
    }

    /// **An `extern` block's items are re-exported as far as they are seen.**
    #[test]
    fn an_extern_blocks_items_are_re_exported() {
        let src = "extern \"C\" {\n    pub fn run(x: i32);\n}\n";
        let out = split(src, "file ffi: The bindings.\n1\n");
        assert_eq!(out.mod_rs, "mod ffi;\n\npub use ffi::*;\n");
    }

    /// **An inline module placed in the file named for it becomes that file**:
    /// the file holds its body, at the same depth, and `mod.rs` keeps its
    /// declaration where it stood.
    #[test]
    fn an_inline_module_is_hoisted_into_its_file() {
        let src = "fn a() {}\n\n/// Tables.\n#[allow(dead_code)]\npub mod tables {\n    use super::*;\n\n    pub const X: &str = \"a\n  b\";\n}\n";
        let out = split(src, "file mod: Stays.\n1\nfile tables: The tables.\n2\n");
        assert_eq!(
            out.mod_rs,
            "fn a() {}\n\n/// Tables.\n#[allow(dead_code)]\npub mod tables;\n\n\n"
        );
        assert_eq!(
            out.children,
            vec![(
                "tables".to_string(),
                "//! The tables.\n\n    use super::*;\n\n    pub const X: &str = \"a\n  b\";\n"
                    .to_string()
            )]
        );
    }

    /// A file named for a hoisted module holds nothing else.
    #[test]
    fn a_hoisted_module_has_its_file_to_itself() {
        let src = "fn a() {}\n\nmod tables {\n    const X: u8 = 1;\n}\n";
        let l = layout(src).unwrap();
        let plan = parse("file tables: The tables.\n1 2\n").unwrap();
        let placed = assign(&l, &plan).unwrap();
        let edits = edits(src, &flat).unwrap();
        let shares = Shares::new(src, &edits).unwrap();
        assert!(assemble(src, &l, &plan, &placed, &edits, &shares)
            .unwrap_err()
            .contains("holds the module `tables` it is named for"));
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
             use super::super::one::tests::helper;\n\n    #[test]\n    fn b() { helper(); }\n}\n"
        );
        // The module's path still names its helpers: `mod.rs` gathers them.
        assert_eq!(
            out.mod_rs,
            "#[cfg(test)]\nmod one;\n#[cfg(test)]\nmod two;\n\n\n#[cfg(test)]\nmod tests {\n    \
             pub(super) use super::one::tests::*;\n    pub(super) use super::two::tests::*;\n}\n"
        );
    }

    /// **A share's re-export reaches as far as its widest item**: a
    /// `pub(crate)` helper of a test module stays reachable from the crate.
    #[test]
    fn a_shares_re_export_reaches_as_far_as_its_items() {
        let src = "#[cfg(test)]\npub(crate) mod tests {\n    pub(crate) fn gate() {}\n\n    #[test]\n    fn b() {}\n}\n";
        let out = split(src, "file one: The gate.\n1.1\nfile two: The test.\n1.2\n");
        assert!(out.mod_rs.contains(
            "    pub(crate) use super::one::tests::*;\n    pub(super) use super::two::tests::*;\n"
        ));
    }

    /// **A share's `use self::x` names the share that declares `x`**, whose
    /// `mod x;` is seen by every share.
    #[test]
    fn a_use_of_a_sibling_module_names_its_share() {
        let src = "#[cfg(test)]\nmod tests {\n    use self::ledger::L;\n\n    mod ledger;\n\n    #[test]\n    fn b() {}\n}\n";
        let out = split(
            src,
            "file one: The ledger.\n1.2\nfile two: The test.\n1.3\n",
        );
        assert!(out.children[0].1.contains("\n    use self::ledger::L;\n"));
        assert!(out.children[0]
            .1
            .contains("pub(in super::super) mod ledger;"));
        assert!(out.children[1]
            .1
            .contains("\n    use super::super::one::tests::ledger::L;\n"));
    }

    /// **A share declaring a module file imports every other share's names**,
    /// since that file reads them through its `use super::*`.
    #[test]
    fn a_share_with_a_file_module_imports_every_name() {
        let src = "#[cfg(test)]\nmod tests {\n    fn bits() {}\n\n    fn other() {}\n\n    mod golden;\n}\n";
        let out = split(
            src,
            "file one: The helpers.\n1.1 1.2\nfile two: The golden.\n1.3\n",
        );
        assert!(out.children[1]
            .1
            .contains("\n    use super::super::one::tests::{bits, other};\n"));
    }

    /// **A test-only helper declared at the top is re-exported for tests**,
    /// which reach it through the split module's path.
    #[test]
    fn a_top_level_test_helper_is_re_exported_for_tests() {
        let out = split(
            "#[cfg(test)]\npub(crate) fn fixture() {}\n",
            "file support: The fixture.\n1\n",
        );
        assert_eq!(
            out.mod_rs,
            "#[cfg(test)]\nmod support;\n\n#[cfg(test)]\npub(crate) use support::*;\n"
        );
    }

    /// **A bare rule is left behind; a titled one names what follows and
    /// stays.**
    #[test]
    fn bare_section_rules_are_dropped() {
        assert_eq!(
            drop_bare_rules(
                "// ──────\n// ── Bans ──\nfn a() {}\n    // ====\n    // a - b\nfn b() {}\n"
            ),
            "// ── Bans ──\nfn a() {}\n    // a - b\nfn b() {}\n"
        );
        assert_eq!(drop_bare_rules("fn a() {}"), "fn a() {}");
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
