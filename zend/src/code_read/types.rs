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
    /// seal_chunk`.  Names the scope in the closing assistant segment of a
    /// code-read part turn (see `header::render_read_ack`) and in parser unit
    /// tests asserted against expected nesting shapes.
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
