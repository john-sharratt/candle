//! Declared-symbol extraction: the names a folder introduces.
//!
//! This is the raw material for the locational and conceptual probe registers
//! ([`super::Register`]) and the input to the directory-frequency index
//! ([`super::idf`]).
//!
//! Extraction is deliberately **recall-oriented** — every declaration a light
//! line scan can name, regardless of visibility — because precision is not this
//! file's job. [`super::idf`] discards any term occurring across many
//! directories, so `new`, `default`, and `fmt` fall out on their own while
//! `ChunkGid` and `PRODUCTION_K_QREL_THRESHOLDS` survive. Filtering here as well
//! would only cost recall the rarity gate cannot give back.
//!
//! A parser would name these more accurately, and [`crate::code_read::carve`]
//! already runs tree-sitter to do exactly that. It is deliberately not used
//! here: this runs inside the workspace walk, over every allowlisted file in the
//! repository, on bytes the walk has already read, and a tree-sitter parse per
//! file would multiply that pass's cost to win precision the rarity gate
//! supplies for free.

use crate::repo_scan::types::Language;

/// Shortest symbol kept. Two-character names (`fs`, `io`, `db`) carry no
/// retrieval signal and collide across the whole corpus.
const MIN_LEN: usize = 3;

/// Longest symbol kept. Past this a "name" is a mangled or generated identifier,
/// not something a person would put in a question.
const MAX_LEN: usize = 64;

/// Symbols kept per file. A file declaring more than this is generated or
/// vendored, and its tail adds noise rather than signal.
const MAX_PER_FILE: usize = 256;

/// Lines scanned per file. Bounds the walk's cost on a very large file; real
/// source declares what it declares well inside this.
const MAX_LINES: usize = 20_000;

/// Every declared name in `body`, sorted and deduplicated.
///
/// Deterministic: the output feeds a content hash, so two runs over identical
/// bytes must produce byte-identical vectors.
pub fn extract(body: &str, language: Language) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // Test declarations are suppressed, and it is not a nicety. A test name is
    // maximally distinctive — it occurs in exactly one directory, so it tops
    // every ranking the frequency index produces — and it is worthless as
    // retrieval vocabulary, because nobody asks about it. Measured on the first
    // live run: `repo_scan/`'s conceptual probes came out as "What constitutes an
    // 'empty anchor' in the context of `an_empty_anchor_file_yields_nothing`?",
    // a perfectly-formed question about a test function, seeded straight from
    // this extractor.
    // A gate suppresses **the item it applies to**, no more.
    //
    // Suppressing to end-of-file instead was justified by "`#[cfg(test)] mod
    // tests` is conventionally the last block in a Rust file" — true of that
    // one shape and of nothing else. A `#[cfg(test)]` on a single item, which
    // is just as ordinary, then discarded the whole file below it:
    // `zend/src/code_read/mod.rs:34` gates a one-line `pub mod test_util;` and
    // cost the 1,181 lines after it, and `npcd/src/api.rs:27` gates a `use` and
    // cost 4,347. Those directories' term indexes came out empty, which reads
    // downstream as a folder with no distinctive vocabulary rather than as a
    // folder whose symbols were never extracted.
    //
    // So: brace depth says where a gated block ends, and a gated item that
    // opens no block ends on its own line. Depth is counted from the raw line,
    // which cannot see a brace inside a string or a comment; that costs the
    // tail of a file whose test module contains an unbalanced brace in a
    // literal, where the previous rule cost the tail of every file with any
    // gate at all.
    let mut gated_until: Option<i32> = None;
    let mut depth: i32 = 0;
    let mut skip_next_decl = false;
    for line in body.lines().take(MAX_LINES) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let opens = trimmed.matches('{').count() as i32;
        let closes = trimmed.matches('}').count() as i32;
        let depth_before = depth;
        depth += opens - closes;
        if let Some(exit) = gated_until {
            if depth <= exit {
                gated_until = None;
            }
            continue;
        }
        if is_test_gate(trimmed) || is_test_attribute(trimmed) {
            skip_next_decl = true;
            continue;
        }
        if skip_next_decl {
            // Attributes stack; the declaration is the first non-attribute line.
            if trimmed.starts_with("#[") {
                continue;
            }
            skip_next_decl = false;
            // A block item (`mod tests {`) suppresses until it closes; a single
            // item (`pub mod test_util;`) is done with this line.
            if depth > depth_before {
                gated_until = Some(depth_before);
            }
            continue;
        }
        match language {
            Language::Rust => rust(trimmed, &mut out),
            Language::Python => python(trimmed, &mut out),
            Language::TypeScript | Language::JavaScript => script(trimmed, &mut out),
            Language::Go => go(trimmed, &mut out),
            Language::C | Language::Cpp => c_family(trimmed, &mut out),
            Language::Java => java(trimmed, &mut out),
            Language::Ruby => ruby(trimmed, &mut out),
            Language::Php => php(trimmed, &mut out),
            Language::Bash => bash(trimmed, &mut out),
            Language::Markdown => markdown(trimmed, &mut out),
            Language::Yaml => yaml(trimmed, &mut out),
            Language::Toml => toml(trimmed, &mut out),
            Language::Json => json(trimmed, &mut out),
            // A structural scan of markup or prose names nothing a question
            // would use; these folders are described by their prose instead.
            Language::Html | Language::Css | Language::PlainText => {}
        }
    }
    // Deduplicate in DECLARATION order, cap, and only then sort. Sorting first
    // made the cap keep the front of the alphabet rather than the head of the
    // file — the same skew `idf` documents as a measured live failure — so a
    // large file contributed its `A`–`C` names and nothing else. The cap's own
    // reasoning is about the file's tail, which is what this keeps.
    let mut seen = std::collections::HashSet::with_capacity(out.len());
    out.retain(|n| seen.insert(n.clone()));
    out.truncate(MAX_PER_FILE);
    out.sort();
    out
}

/// A `cfg` gate that puts the item it applies to under test.
///
/// Matches the `test` **predicate**, not the substring: `#[cfg(feature =
/// "latest")]` and `#[cfg(feature = "attn-fastest")]` both contain "test" and
/// gate ordinary production code, and treating them as test gates discarded
/// every declaration below them. String literals are blanked before the scan
/// for exactly that reason, so only a bare `test` token counts.
///
/// `not` anywhere in the predicate disqualifies it. `#[cfg(not(test))]` is code
/// that exists *outside* tests — production code, and the one form where the
/// bare token means the opposite of what it says elsewhere. Rejecting the whole
/// family costs the rare `#[cfg(all(test, not(windows)))]`, which errs toward
/// recall exactly as this module's header asks.
fn is_test_gate(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("#[cfg(") else {
        return false;
    };
    let mut in_str = false;
    let cleaned: String = rest
        .chars()
        .map(|c| {
            if c == '"' {
                in_str = !in_str;
                ' '
            } else if in_str {
                ' '
            } else {
                c
            }
        })
        .collect();
    let mut tokens = cleaned.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
    let mut saw_test = false;
    for t in &mut tokens {
        if t == "not" {
            return false;
        }
        saw_test |= t == "test";
    }
    saw_test
}

/// An attribute marking the next declaration as a test case.
fn is_test_attribute(line: &str) -> bool {
    matches!(line, "#[test]" | "#[bench]" | "#[ignore]")
        || line.starts_with("#[test(")
        || line.ends_with("::test]")
        || line.ends_with("::test(async)]")
        || line.starts_with("#[rstest")
        || line.starts_with("#[test_case")
}

/// Identifier-ish tokens on a line, in order.
fn idents(line: &str) -> Vec<&str> {
    line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Language primitives, keywords, and structural words that are not domain
/// vocabulary in any project.
///
/// The rarity gate cannot remove these on its own, and the reason is worth
/// stating because it is counter-intuitive: a term is only filtered when it is
/// *widespread*, and these leak in through the weakest extraction paths — a
/// `#define` in one kernel header, a backticked word in one design doc, a key
/// in one YAML schema — so they land in a handful of directories and read as
/// highly distinctive. Measured on this repository before the list, the top
/// terms for `candle-kernels/src/simple/` were `usize`, `double2`,
/// `__float2bfloat16`, `half2` and `decltype`: every one a C++ primitive, none
/// of them anything a person would ask about.
const NOISE: &[&str] = &[
    // C / C++ / CUDA primitives and keywords.
    "int",
    "char",
    "bool",
    "void",
    "long",
    "short",
    "float",
    "double",
    "size_t",
    "ssize_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "unsigned",
    "signed",
    "const",
    "static",
    "inline",
    "extern",
    "auto",
    "decltype",
    "typename",
    "template",
    "typedef",
    "struct",
    "class",
    "enum",
    "union",
    "namespace",
    "public",
    "private",
    "protected",
    "virtual",
    "override",
    "constexpr",
    "noexcept",
    "nullptr",
    "true",
    "false",
    "half",
    "half2",
    "float2",
    "float4",
    "double2",
    "uchar",
    "ushort",
    "uint",
    "ulong",
    // Rust primitives and ubiquitous std types.
    "usize",
    "isize",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "f32",
    "f64",
    "str",
    "String",
    "Vec",
    "Option",
    "Some",
    "None",
    "Result",
    "Ok",
    "Err",
    "Box",
    "Self",
    "self",
    "super",
    "crate",
    "dyn",
    "impl",
    "where",
    "match",
    "let",
    "mut",
    "ref",
    "move",
    "fn",
    // Ubiquitous member names — present in nearly every type in every project.
    "new",
    "default",
    "clone",
    "fmt",
    "drop",
    "from",
    "into",
    "len",
    "get",
    "set",
    "add",
    "next",
    "iter",
    "hash",
    "eq",
    "cmp",
    "partial_cmp",
    "to_string",
    "as_str",
    "build",
    "run",
    "init",
    // Words that reach the index only through markdown backticks or config keys,
    // and carry no subject.
    "see",
    "note",
    "todo",
    "null",
    "true_",
    "yes",
    "no",
    "web",
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "type",
    "kind",
    "name",
    "value",
    "data",
    "item",
    "list",
    "map",
    "key",
    "title",
    "description",
    "required",
    "properties",
    "examples",
    "parameters",
    "category",
    "minimum",
    "maximum",
    "format",
    "content",
    "text",
    "input",
    "output",
    "error",
    "result",
    "config",
    "options",
    "params",
    "args",
    "test",
    "tests",
    "main",
    "mod",
    "lib",
    "src",
];

/// Whether `name` is structureless — a single all-lowercase or all-uppercase
/// word with no internal boundary.
///
/// Domain vocabulary is nearly always compound: `region_pool`, `ChunkGid`,
/// `Q5_K`, `apply_rotary_emb`. A bare word like `access` or `marlin` may still
/// be real, so this is a ranking signal rather than a rejection — see
/// [`super::idf::TermIndex::distinctive`].
pub fn is_structureless(name: &str) -> bool {
    if name.contains('_') || name.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    // A lower→upper transition is a camelCase / PascalCase boundary.
    !name
        .chars()
        .zip(name.chars().skip(1))
        .any(|(a, b)| a.is_ascii_lowercase() && b.is_ascii_uppercase())
}

/// Keep `name` if it looks like something a person would write in a question.
fn push(out: &mut Vec<String>, name: &str) {
    if name.len() < MIN_LEN || name.len() > MAX_LEN {
        return;
    }
    // A leading digit means this is a numeric literal the tokenizer split, not
    // an identifier.
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        return;
    }
    // Case-insensitive: `FALSE`, `False` and `false` are one word.
    let lower = name.to_ascii_lowercase();
    if NOISE.iter().any(|n| n.eq_ignore_ascii_case(&lower)) {
        return;
    }
    // A leading double underscore is a compiler or platform intrinsic
    // (`__float2bfloat16`, `__shfl_sync`), never project vocabulary.
    if name.starts_with("__") {
        return;
    }
    out.push(name.to_string());
}

/// The first identifier after any of `decl`, skipping tokens in `skip`.
///
/// The skip set is what lets a modifier chain resolve: `pub const fn foo` finds
/// `const`, steps over `fn`, and names `foo`.
fn after_decl(line: &str, decl: &[&str], skip: &[&str]) -> Option<String> {
    let toks = idents(line);
    let at = toks.iter().position(|t| decl.contains(t))?;
    toks[at + 1..]
        .iter()
        .find(|t| !skip.contains(*t) && !decl.contains(*t))
        .map(|t| t.to_string())
}

/// The identifier immediately preceding the first `(` — a function definition's
/// name in every C-descended syntax.
///
/// Two guards, and both are load-bearing:
///
/// * A line opening with a control keyword (`if (x) {`) puts the keyword before
///   the paren, not a name.
/// * A **call** and a **definition** differ in what precedes the paren. A call
///   statement is `launch_something(n);` — one identifier. A definition carries
///   at least a return type as well: `void bdp_recall(int n) {`. Requiring two
///   is what separates the names a file declares from the far larger set it
///   merely invokes, and without it a C file contributes its entire call graph.
fn before_paren(line: &str) -> Option<String> {
    const CONTROL: &[&str] = &[
        "if", "for", "while", "switch", "return", "else", "do", "catch", "sizeof", "case",
    ];
    let head = line.split('(').next()?;
    let toks = idents(head);
    if toks.len() < 2 {
        return None;
    }
    if CONTROL.contains(&toks[0]) {
        return None;
    }
    toks.last().map(|t| t.to_string())
}

const RUST_DECL: &[&str] = &[
    "fn",
    "struct",
    "enum",
    "trait",
    "const",
    "static",
    "type",
    "union",
    "macro_rules",
];

/// Rust tokens that may sit between a declaration keyword and the name.
const RUST_SKIP: &[&str] = &[
    "pub", "crate", "super", "in", "async", "unsafe", "extern", "mut", "dyn", "impl", "default",
    "self",
];

fn rust(line: &str, out: &mut Vec<String>) {
    if let Some(name) = after_decl(line, RUST_DECL, RUST_SKIP) {
        push(out, &name);
    }
    // `mod` is handled apart from the declaration set: a `use` line mentions
    // module names constantly, and `mod` inside one would name an import rather
    // than a declaration.
    if let Some(rest) = line.strip_prefix("mod ").or_else(|| {
        line.strip_prefix("pub mod ")
            .or_else(|| line.strip_prefix("pub(crate) mod "))
    }) {
        if let Some(name) = idents(rest).first() {
            push(out, name);
        }
    }
}

fn python(line: &str, out: &mut Vec<String>) {
    if let Some(name) = after_decl(line, &["def", "class"], &["async"]) {
        push(out, &name);
    }
}

fn script(line: &str, out: &mut Vec<String>) {
    const DECL: &[&str] = &["function", "class", "interface", "enum", "type"];
    const SKIP: &[&str] = &[
        "export", "default", "async", "declare", "abstract", "readonly",
    ];
    if let Some(name) = after_decl(line, DECL, SKIP) {
        push(out, &name);
    }
    // A bare `const` is a local in every function body; only an exported one
    // names something the folder offers outward.
    if line.starts_with("export ") {
        if let Some(name) = after_decl(line, &["const", "let", "var"], SKIP) {
            push(out, &name);
        }
    }
}

fn go(line: &str, out: &mut Vec<String>) {
    // A method carries its receiver in parentheses before the name
    // (`func (r *Repo) Scan()`), so the first identifier after `func` is the
    // receiver binding rather than the method.
    if let Some(rest) = line.strip_prefix("func (") {
        if let Some((_, after)) = rest.split_once(')') {
            if let Some(name) = idents(after).first() {
                push(out, name);
            }
            return;
        }
    }
    if let Some(name) = after_decl(line, &["func", "type"], &[]) {
        push(out, &name);
    }
}

fn c_family(line: &str, out: &mut Vec<String>) {
    if let Some(rest) = line.strip_prefix("#define ") {
        if let Some(name) = idents(rest).first() {
            push(out, name);
        }
        return;
    }
    const DECL: &[&str] = &["struct", "class", "enum", "union", "namespace"];
    if let Some(name) = after_decl(line, DECL, &["typedef", "public", "private", "protected"]) {
        push(out, &name);
    }
    // A definition, not a call: the line opens a body or ends a signature.
    // Without this the scan would name every function the file *calls*, which
    // is most of its identifiers and none of what it declares.
    if (line.ends_with('{') || line.ends_with(')') || line.ends_with(';')) && line.contains('(') {
        if let Some(name) = before_paren(line) {
            push(out, &name);
        }
    }
}

fn java(line: &str, out: &mut Vec<String>) {
    const DECL: &[&str] = &["class", "interface", "enum", "record"];
    const SKIP: &[&str] = &[
        "public",
        "private",
        "protected",
        "static",
        "final",
        "abstract",
    ];
    if let Some(name) = after_decl(line, DECL, SKIP) {
        push(out, &name);
    }
    if line.ends_with('{') && line.contains('(') {
        if let Some(name) = before_paren(line) {
            push(out, &name);
        }
    }
}

fn ruby(line: &str, out: &mut Vec<String>) {
    if let Some(name) = after_decl(line, &["def", "class", "module"], &["self"]) {
        push(out, &name);
    }
}

fn php(line: &str, out: &mut Vec<String>) {
    const SKIP: &[&str] = &[
        "public",
        "private",
        "protected",
        "static",
        "final",
        "abstract",
    ];
    if let Some(name) = after_decl(line, &["function", "class", "interface", "trait"], SKIP) {
        push(out, &name);
    }
}

fn bash(line: &str, out: &mut Vec<String>) {
    if let Some(name) = after_decl(line, &["function"], &[]) {
        push(out, &name);
        return;
    }
    // `name() {` — the POSIX form, which has no keyword to key on.
    if line.ends_with("{") && line.contains("()") {
        if let Some(name) = idents(line).first() {
            push(out, name);
        }
    }
}

/// Markdown contributes its **backticked spans**, not its prose.
///
/// A heading is a phrase in ordinary English, so it competes with the folder's
/// own summary and adds nothing the rarity gate can use. A backticked span in a
/// design doc is almost always a real identifier — which is exactly the kind of
/// term a conceptual probe needs, and often the only place a term is written
/// down in prose rather than as code.
fn markdown(line: &str, out: &mut Vec<String>) {
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        let span = &after[..close];
        // One identifier only. A backticked phrase (`cargo test --release`) is
        // a command line, not a name.
        let toks = idents(span);
        if toks.len() == 1 {
            push(out, toks[0]);
        }
        rest = &after[close + 1..];
    }
}

fn yaml(line: &str, out: &mut Vec<String>) {
    if let Some((key, _)) = line.split_once(':') {
        let key = key.trim().trim_start_matches("- ");
        if !key.is_empty()
            && key
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            push(out, &key.replace('-', "_"));
        }
    }
}

fn toml(line: &str, out: &mut Vec<String>) {
    if let Some(section) = line.strip_prefix('[') {
        let section = section.trim_end_matches(']').trim_end_matches(']');
        for tok in idents(section) {
            push(out, tok);
        }
        return;
    }
    if let Some((key, _)) = line.split_once('=') {
        let key = key.trim();
        if !key.is_empty()
            && key
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            push(out, &key.replace('-', "_"));
        }
    }
}

fn json(line: &str, out: &mut Vec<String>) {
    let Some(open) = line.find('"') else { return };
    let after = &line[open + 1..];
    let Some(close) = after.find('"') else { return };
    // A key, not a value: the quoted span is followed by a colon.
    if after[close + 1..].trim_start().starts_with(':') {
        let toks = idents(&after[..close]);
        if toks.len() == 1 {
            push(out, toks[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs(body: &str) -> Vec<String> {
        extract(body, Language::Rust)
    }

    #[test]
    fn rust_names_every_declaration_kind() {
        let out = rs("\
pub fn seal_block_to() {}
pub struct ChunkGid;
pub enum KvFormat {}
pub trait PagedKvArenas {}
pub const CHUNK_SIZE: usize = 32;
pub static SCAN_LIVE: u32 = 0;
pub type ArenaKey = u64;
");
        for want in [
            "seal_block_to",
            "ChunkGid",
            "KvFormat",
            "PagedKvArenas",
            "CHUNK_SIZE",
            "SCAN_LIVE",
            "ArenaKey",
        ] {
            assert!(
                out.contains(&want.to_string()),
                "{want} missing from {out:?}"
            );
        }
    }

    /// Visibility is NOT a filter: a private `fn scan_width` is exactly the kind
    /// of name a "how is the pool width decided" question needs, and the rarity
    /// gate — not this scan — is what removes the noise.
    #[test]
    fn rust_keeps_private_declarations() {
        assert!(rs("fn scan_width() {}").contains(&"scan_width".to_string()));
    }

    /// A modifier chain must not shadow the name.
    #[test]
    fn rust_steps_over_modifier_chains() {
        assert!(rs("pub const fn per_conversation_kv() {}")
            .contains(&"per_conversation_kv".to_string()));
        assert!(rs("pub(crate) async unsafe fn map_range() {}").contains(&"map_range".to_string()));
    }

    /// `use` lines name modules constantly. Treating `mod` as an ordinary
    /// declaration keyword would harvest every import path in the crate.
    #[test]
    fn rust_ignores_module_names_inside_use_lines() {
        let out = rs("use crate::kv_cache::chunked::backing;");
        assert!(!out.contains(&"backing".to_string()), "{out:?}");
        assert!(rs("pub mod region_pool;").contains(&"region_pool".to_string()));
    }

    #[test]
    fn c_family_names_definitions_but_not_calls() {
        let out = extract(
            "\
#define WARP_SIZE 32
struct ArenaTableEntry {
__global__ void bdp_recall_batched(int n) {
    launch_something(n);
}
",
            Language::Cpp,
        );
        assert!(out.contains(&"WARP_SIZE".to_string()), "{out:?}");
        assert!(out.contains(&"ArenaTableEntry".to_string()), "{out:?}");
        assert!(out.contains(&"bdp_recall_batched".to_string()), "{out:?}");
        assert!(
            !out.contains(&"launch_something".to_string()),
            "a call is not a declaration: {out:?}",
        );
    }

    /// A control-flow line ends in `{` and contains `(`, so without the control
    /// guard `if`/`for` would be harvested as function names in every C file.
    #[test]
    fn c_family_skips_control_flow() {
        let out = extract(
            "    if (x > 0) {\n    for (int i = 0; i < n; i++) {",
            Language::Cpp,
        );
        assert!(out.is_empty(), "{out:?}");
    }

    /// A Go method's receiver binding sits between `func` and the name.
    #[test]
    fn go_names_the_method_not_the_receiver() {
        let out = extract("func (r *Repository) ScanWorkspace() error {", Language::Go);
        assert!(out.contains(&"ScanWorkspace".to_string()), "{out:?}");
        assert!(!out.contains(&"Repository".to_string()), "{out:?}");
    }

    #[test]
    fn go_names_plain_functions_and_types() {
        let out = extract("func WalkTree() {}\ntype DirUnit struct {", Language::Go);
        assert!(out.contains(&"WalkTree".to_string()), "{out:?}");
        assert!(out.contains(&"DirUnit".to_string()), "{out:?}");
    }

    /// A local `const` is not part of the folder's surface; an exported one is.
    #[test]
    fn script_takes_exported_bindings_only() {
        let exported = extract("export const MAX_RETRIES = 3;", Language::TypeScript);
        assert!(
            exported.contains(&"MAX_RETRIES".to_string()),
            "{exported:?}"
        );
        let local = extract("    const tmpBuffer = alloc();", Language::TypeScript);
        assert!(!local.contains(&"tmpBuffer".to_string()), "{local:?}");
    }

    #[test]
    fn python_names_defs_and_classes() {
        let out = extract(
            "class Governor:\n    async def measure(self):",
            Language::Python,
        );
        assert!(out.contains(&"Governor".to_string()), "{out:?}");
        assert!(out.contains(&"measure".to_string()), "{out:?}");
    }

    /// A design doc is often the only place a term is written in prose. The
    /// backticked spans are the identifiers; the surrounding English is not.
    #[test]
    fn markdown_takes_backticked_identifiers_not_prose() {
        let out = extract(
            "The `weight_floor` boundary sits between `region_pool` and the tier.",
            Language::Markdown,
        );
        assert!(out.contains(&"weight_floor".to_string()), "{out:?}");
        assert!(out.contains(&"region_pool".to_string()), "{out:?}");
        assert!(!out.contains(&"boundary".to_string()), "{out:?}");
    }

    /// A backticked command line is not a name.
    #[test]
    fn markdown_ignores_multi_word_backticked_spans() {
        let out = extract("Run `cargo test --release` first.", Language::Markdown);
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn json_takes_keys_not_values() {
        let out = extract("  \"model_path\": \"qwen3\",", Language::Json);
        assert!(out.contains(&"model_path".to_string()), "{out:?}");
        assert!(!out.contains(&"qwen3".to_string()), "{out:?}");
    }

    /// A test name is maximally distinctive and completely useless: it occurs in
    /// one directory, so it tops the frequency ranking, and nobody has ever
    /// asked about one. This is not hypothetical — the first live run seeded
    /// `repo_scan/`'s conceptual probes from `an_empty_anchor_file_yields_nothing`.
    #[test]
    fn a_cfg_test_module_contributes_nothing() {
        let out = rs("\
pub fn seal_block_to() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_anchor_file_yields_nothing() {}

    fn helper_for_the_tests() {}
}
");
        assert_eq!(out, vec!["seal_block_to".to_string()], "{out:?}");
    }

    /// A `#[test]` outside a test module is suppressed too, attributes stacked.
    #[test]
    fn a_test_attribute_suppresses_its_declaration() {
        let out = rs("\
#[test]
fn a_probe_naming_its_own_path_is_refused() {}

#[tokio::test]
async fn another_case() {}

#[test]
#[ignore]
fn a_slow_gate_case() {}

pub fn real_api() {}
");
        assert_eq!(out, vec!["real_api".to_string()], "{out:?}");
    }

    /// The wider `cfg` forms used for test-only helpers gate the same way.
    #[test]
    fn a_test_helper_cfg_gate_also_suppresses() {
        let out = rs("\
pub fn live_api() {}

#[cfg(any(test, feature = \"test-helpers\"))]
pub fn only_for_tests() {}
");
        assert_eq!(out, vec!["live_api".to_string()], "{out:?}");
    }

    /// **A gate ends where its item ends.** A `#[cfg(test)]` on a single item is
    /// as ordinary as one on `mod tests`, and suppressing to end-of-file
    /// silently cost the rest of the file: live, `zend/src/code_read/mod.rs`
    /// gates a one-line `pub mod test_util;` on line 34 and contributed nothing
    /// from the 1,181 lines below it.
    #[test]
    fn a_gate_on_a_single_item_does_not_swallow_the_file() {
        let out = rs("\
pub mod carve;
#[cfg(test)]
pub mod test_util;
pub mod types;

pub fn resolve_header() {}
pub struct CarvedSpan;
");
        assert!(!out.contains(&"test_util".to_string()), "{out:?}");
        for want in ["carve", "types", "resolve_header", "CarvedSpan"] {
            assert!(out.contains(&want.to_string()), "{want} lost from {out:?}");
        }
    }

    /// The same for a `use`, which is the shape in `npcd/src/api.rs`.
    #[test]
    fn a_gated_use_does_not_swallow_the_file() {
        let out = rs("\
#[cfg(test)]
use axum::Router;

pub fn build_router() {}
");
        assert_eq!(out, vec!["build_router".to_string()], "{out:?}");
    }

    /// A gated block still suppresses its whole body — and stops at its closing
    /// brace, so declarations after it survive.
    #[test]
    fn a_gated_block_suppresses_its_body_and_no_more() {
        let out = rs("\
pub fn before() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_anchor_file_yields_nothing() {}

    fn helper_for_the_tests() {
        let closure = || { 0 };
    }
}

pub fn after_the_module() {}
");
        assert_eq!(
            out,
            vec!["after_the_module".to_string(), "before".to_string()],
            "{out:?}",
        );
    }

    /// **`test` is a predicate, not a substring.** `#[cfg(feature = "latest")]`
    /// gates ordinary code; reading it as a test gate discarded every
    /// declaration below it.
    #[test]
    fn a_feature_gate_whose_name_contains_test_is_not_a_test_gate() {
        for gate in [
            "#[cfg(feature = \"latest\")]",
            "#[cfg(feature = \"attn-fastest\")]",
            "#[cfg(all(unix, feature = \"fastest-path\"))]",
        ] {
            let out = rs(&format!(
                "{gate}\npub fn streaming_scan() {{}}\npub struct WaveTier;\n"
            ));
            assert!(
                out.contains(&"WaveTier".to_string()),
                "{gate} swallowed the file: {out:?}",
            );
        }
    }

    /// `#[cfg(not(test))]` is production code — the one form where the bare
    /// token means the opposite of what it means everywhere else.
    #[test]
    fn a_not_test_gate_is_production_code() {
        let out = rs("#[cfg(not(test))]\npub fn real_allocator() {}\n");
        assert_eq!(out, vec!["real_allocator".to_string()], "{out:?}");
    }

    /// The per-file cap keeps the head of the FILE, not the head of the
    /// alphabet — sorting before truncating gave a large file its `A`–`C`
    /// names and dropped everything else, the exact skew `idf` documents.
    #[test]
    fn the_per_file_cap_takes_the_declaration_tail_not_the_alphabetical_one() {
        let mut body = String::new();
        // Declared last, sorts first: it must be the one that is dropped.
        for i in 0..MAX_PER_FILE {
            body.push_str(&format!("pub fn zzz_decl_{i:04}() {{}}\n"));
        }
        body.push_str("pub fn aaa_declared_last() {}\n");
        let out = rs(&body);
        assert_eq!(out.len(), MAX_PER_FILE);
        assert!(
            !out.contains(&"aaa_declared_last".to_string()),
            "the cap kept a name declared past it because it sorts first",
        );
        assert!(
            out.contains(&"zzz_decl_0000".to_string()),
            "{:?}",
            &out[..3]
        );
        // Still sorted and still deterministic.
        let mut sorted = out.clone();
        sorted.sort();
        assert_eq!(out, sorted);
        assert_eq!(out, rs(&body));
    }

    /// Short names collide across the whole corpus and carry no signal.
    #[test]
    fn two_character_names_are_dropped() {
        let out = rs("pub fn go() {}\npub struct Ab;");
        assert!(out.is_empty(), "{out:?}");
    }

    /// The output feeds a content hash, so it must be stable and ordered.
    #[test]
    fn output_is_sorted_deduplicated_and_deterministic() {
        let body = "pub fn zeta() {}\npub fn alpha() {}\npub fn zeta() {}";
        let out = rs(body);
        assert_eq!(out, vec!["alpha".to_string(), "zeta".to_string()]);
        assert_eq!(out, rs(body));
    }

    /// Markup and prose declare nothing a question would name.
    #[test]
    fn markup_languages_contribute_nothing() {
        assert!(extract("<div class=\"x\">hello</div>", Language::Html).is_empty());
        assert!(extract(".btn { color: red; }", Language::Css).is_empty());
    }
}
