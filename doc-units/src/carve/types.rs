//! What the carver emits, and the sizes it holds a unit to.

/// What kind of item a scope is — a function, a type, a paragraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    /// A function or method body.
    Function,
    /// A struct, class, enum, trait, interface, type alias, or equivalent
    /// named-type definition.
    TypeDefinition,
    /// One or more module-level constants.
    Constants,
    /// Any other item a parser surfaces (a Rust macro definition, a CSS rule).
    TopLevel,
    /// A top-level key of a structured-config file (TOML table, JSON key).
    HeaderSection,
    /// The leading comment block at the very top of a file — the module doc,
    /// licence or file overview — always its own first unit.
    FileHeader,
    /// A run of lines no item claims, or a paragraph of a file with no carver:
    /// one run of non-blank lines between blank lines.
    Paragraph,
    /// One line of a line-structured document (YAML).
    Line,
}

/// One carved scope. Line numbers are 1-indexed and inclusive on both ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    /// Nesting names from outermost to innermost, joined by ` > ` when shown:
    /// `["impl KvCache", "fn seal_chunk"]`. Empty for a unit that is named by
    /// its own first line (a paragraph, a YAML line, an unclaimed run).
    pub path: Vec<String>,
    pub kind: ChunkKind,
    pub start_line: u32,
    pub end_line: u32,
}

impl Scope {
    /// The joined nesting path — `mod cache > impl KvCache > fn seal_chunk`.
    pub fn qualified_path(&self) -> String {
        self.path.join(" > ")
    }

    /// Inclusive line count.
    pub fn line_span(&self) -> u32 {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

/// The most lines one unit holds. A longer item splits on its children's
/// boundaries first (`[part N]`), and at line boundaries only when it has
/// none — so a long function is several anchors, each small enough that
/// replacing it whole is not a rewrite of the file.
pub const MAX_UNIT_LINES: u32 = 80;

/// The most characters one unit holds, alongside [`MAX_UNIT_LINES`], so a run
/// of dense lines cannot make a unit far larger than the same count of
/// ordinary ones.
pub const MAX_UNIT_CHARS: u32 = 4_000;

/// The fewest comment lines a file's leading comment block spans before it is
/// split off as its own [`ChunkKind::FileHeader`] unit. A lone one-line comment
/// stays with the item below it.
pub const MIN_FILE_HEADER_LINES: u32 = 2;
