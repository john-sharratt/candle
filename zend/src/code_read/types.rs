//! Data types for the scope-aware code carver.

/// Kind of scope a chunk represents.  The carver tags each emitted
/// [`Scope`] with one of these so downstream telemetry / inspection
/// can distinguish a function body from a fixed-window fallback
/// chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    /// A function or method body.
    Function,
    /// A struct, class, enum, trait, interface, type alias, or
    /// equivalent named-type definition.
    TypeDefinition,
    /// One or more module-level constants grouped into one chunk.
    Constants,
    /// Anything else surfaced as a top-level item (Rust macro
    /// definition, Python module-level statement, etc).
    TopLevel,
    /// Header-based section split (Markdown `##`, structured-config
    /// top-level key, HTML landmark, CSS rule_set).
    HeaderSection,
    /// The leading comment block at the very top of a file — the module
    /// doc / license / file-overview comment section, split off as its
    /// own first turn so the opening summary describes the file as a
    /// whole rather than spending its budget on the first function.
    /// Always the first scope in a file and never merged forward (see
    /// `carve::refine`).
    FileHeader,
    /// Fixed-window fallback chunk — used when no language-specific
    /// parser fired.
    Fallback,
}

/// One carved scope in a file.  Line numbers are 1-indexed, inclusive
/// on both ends — same convention as the scope-header format
/// (`Lines: 47-93`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    /// Ordered list of nesting names from outermost to innermost,
    /// rendered with `>` as the join separator.  E.g.
    /// `["mod cache", "impl KvCache", "fn seal_chunk"]`.  Always
    /// non-empty for a carved scope.
    pub path: Vec<String>,
    pub kind: ChunkKind,
    pub start_line: u32,
    pub end_line: u32,
}

impl Scope {
    /// Joined nesting path — `mod cache > impl KvCache > fn
    /// seal_chunk`.  Names the scope in the carve/AST layer and in parser unit
    /// tests asserted against expected nesting shapes.
    #[allow(dead_code)]
    pub fn qualified_path(&self) -> String {
        self.path.join(" > ")
    }

    /// Inclusive line count for the scope (`end_line - start_line + 1`).
    /// Public utility used by the oversize-split tests; also surfaces
    /// in telemetry consumers that summarise carved scope sizes.
    #[allow(dead_code)]
    pub fn line_span(&self) -> u32 {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

/// Maximum lines per chunk before the carve splits at sub-blocks.
pub const MAX_SCOPE_LINES: u32 = 150;

/// Maximum characters a single carved turn should carry, enforced alongside
/// [`MAX_SCOPE_LINES`] so a run of dense (near-[`MAX_LINE_CHARS`]) lines can't
/// produce a turn far larger than a run of ordinary code at the same line count.
/// The two caps work together for CLEAN breaks: the refine pass stops merging a
/// scope forward at the last whole sub-scope (function / type / const run) that
/// fits within this budget — it clips a later section off into the next turn
/// rather than cutting one in half. A single sub-scope that exceeds this budget on
/// its own (a giant generated table, a minified blob) is the unavoidable case and
/// is split at line boundaries. Sized to ~2k tokens for code (≈ [`MAX_LINE_CHARS`]
/// × [`MIN_SCOPE_LINES`]); ordinary ~40-char lines reach [`MAX_SCOPE_LINES`] well
/// under this, so the char cap only binds on genuinely dense content.
pub const MAX_SCOPE_CHARS: u32 = 8_000;

/// Hard cap on characters per line fed to the carver. A single-line minified file
/// (or a generated one-line data blob) would otherwise be ONE line — so one scope
/// spanning the whole file, or an un-splittable line inside a scope — that blows
/// the per-turn token budget. Over-long lines are split at the latest safe point
/// (outside any string literal) at or before this width; a line with no safe break
/// is hard-clipped here. Sized so ~[`MIN_SCOPE_LINES`] lines at this width stays a
/// reasonable turn (50 × 160 ≈ 8 KB ≈ a couple thousand tokens).
pub const MAX_LINE_CHARS: usize = 160;

/// Soft target: once a split piece reaches this width, break at the next safe
/// point (outside a string, after a delimiter) rather than running to the hard cap
/// — keeps pieces averaging near this instead of always maxing out.
pub const SOFT_LINE_CHARS: usize = 100;

/// Minimum number of comment lines a file's leading comment block must span for
/// the carve to split it off as a standalone [`ChunkKind::FileHeader`] first turn
/// (see `carve::file_header_end`). A lone one-line comment (a stray `//`, a bare
/// shebang, a single copyright line) is not a "section" — splitting it would make a
/// useless one-line turn — so it stays attached to the following item; a real
/// multi-line module-doc / license header (≥ this many comment lines) becomes the
/// file-overview turn instead.
pub const MIN_FILE_HEADER_LINES: u32 = 2;

/// Minimum lines a carved scope should carry before it stops absorbing the
/// following scope. A single one-line `const`, a tiny helper `fn`, or a stray
/// blank/gap run makes a poor standalone chunk — it dilutes provenance across too
/// many near-empty turns. The refine pass (see `carve::refine`) merges a scope
/// forward into the next function / type / const run until it reaches this width
/// (never past [`MAX_SCOPE_LINES`]). Split points stay real functions and types;
/// small items are only ever absorbed, never split.
pub const MIN_SCOPE_LINES: u32 = 50;
