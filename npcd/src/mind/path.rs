//! A path inside the mind, and nothing outside it.
//!
//! Every read and write in this module names a file by a **relative** path that
//! arrived in a URL. That is the whole attack surface: `../../../etc/passwd`,
//! `C:\Windows\System32`, a NUL that truncates the name after validation, a
//! symlink pointing out of the tree. So the path is parsed into [`MindPath`]
//! once, at the edge, and nothing downstream takes a `&str`.
//!
//! # Two checks, not one
//!
//! Syntax is checked first — the segments must be plausible names — and then
//! the resolved path is checked to still be *under the root* after the
//! filesystem has had its say. Neither alone is enough:
//!
//! - Syntax alone misses symlinks. A segment can be a perfectly ordinary name
//!   that the filesystem resolves somewhere else entirely.
//! - Containment alone misses nothing on Unix, but on Windows it is reached
//!   through `canonicalize`, which fails for a path that does not exist yet —
//!   and creating a file is exactly the case where it does not.
//!
//! So both run, and the second is done against the deepest existing ancestor so
//! that creating a new file is checked as strictly as opening an old one.
//!
//! # Why not an allow-list of characters
//!
//! The obvious rule — `[a-z0-9_-]` — is what [`crate::registry::id`] uses for
//! document ids, and it is right there because an id is *chosen* by an author
//! naming a new file. This is different: these paths name files that already
//! exist, written over years by hand, and one of them is
//! `layers/world/relationships/bonds/protégés.md`. An allow-list would make
//! that file unreachable and unfixable through the console — it would be
//! invisible rather than protected. So the rule is a deny-list of the
//! constructs that are actually dangerous, and the containment check behind it
//! catches anything the deny-list did not think of.

use std::path::{Component, Path, PathBuf};

/// The deepest a path may go. `layers/world/relationships/bonds/protégés.md` is
/// five, so this is generous; it exists to bound the work, not to shape the
/// tree.
const MAX_DEPTH: usize = 16;

/// Longest single segment. Windows' own limit is 255 for a file name.
const MAX_SEGMENT: usize = 128;

/// Longest whole path. Well inside `MAX_PATH` once the mind root is prepended.
const MAX_TOTAL: usize = 1024;

/// Names Windows refuses to create, with or without an extension, in any case.
/// `con.md` is as unusable as `con`.
const RESERVED: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Why a path was refused.
///
/// Each variant names the construct rather than the file, so the message can be
/// shown to whoever typed it without telling them anything about the disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathError {
    /// A segment was `..`, or the path was absolute, or it named a drive.
    Traversal,
    /// An empty segment — `a//b`, or a leading or trailing separator.
    EmptySegment,
    /// A backslash, colon, NUL or control character inside a segment.
    BadChar(char),
    /// A segment ending in `.` or a space. Windows silently strips both, so
    /// `a.md ` and `a.md` would be two names for one file.
    TrailingDotOrSpace,
    /// `con`, `nul`, `com1`… — unusable on Windows whatever the extension.
    Reserved,
    TooDeep(usize),
    TooLong(usize),
    /// The extension is not one this editor writes. See [`MindPath::ext`].
    BadExtension,
    /// Resolved, and the result was not inside the mind after all — a symlink,
    /// a junction, or a case the syntax rules did not cover.
    Escapes,
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathError::Traversal => write!(f, "a path may not leave the mind directory"),
            PathError::EmptySegment => write!(f, "empty path segment"),
            PathError::BadChar(c) => write!(f, "{c:?} is not allowed in a path"),
            PathError::TrailingDotOrSpace => {
                write!(f, "a name may not end in a dot or a space")
            }
            PathError::Reserved => write!(f, "that name is reserved by the operating system"),
            PathError::TooDeep(n) => write!(f, "path is {n} deep, the limit is {MAX_DEPTH}"),
            PathError::TooLong(n) => write!(f, "path is {n} bytes, the limit is {MAX_TOTAL}"),
            PathError::BadExtension => write!(f, "only .md and .yaml files can be edited"),
            PathError::Escapes => write!(f, "that path resolves outside the mind directory"),
        }
    }
}

/// A relative path that has been checked, and can only be built by checking.
///
/// Held as segments rather than a string so that joining it to the root cannot
/// reintroduce a separator the parse rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MindPath {
    segments: Vec<String>,
}

impl MindPath {
    /// The mind root itself.
    pub fn root() -> Self {
        MindPath {
            segments: Vec::new(),
        }
    }

    /// Parse a `/`-separated relative path.
    ///
    /// The empty string is the root, which is what an unset `?path=` means.
    pub fn parse(raw: &str) -> Result<Self, PathError> {
        if raw.len() > MAX_TOTAL {
            return Err(PathError::TooLong(raw.len()));
        }
        let trimmed = raw.trim_matches('/');
        if trimmed.is_empty() {
            return Ok(MindPath::root());
        }
        // A backslash is a separator on Windows, so a segment containing one is
        // two segments wearing a disguise. Caught per-segment below, but the
        // whole-string check makes the intent unmistakable.
        let segments: Vec<&str> = trimmed.split('/').collect();
        if segments.len() > MAX_DEPTH {
            return Err(PathError::TooDeep(segments.len()));
        }
        let mut out = Vec::with_capacity(segments.len());
        for seg in segments {
            check_segment(seg)?;
            out.push(seg.to_owned());
        }
        Ok(MindPath { segments: out })
    }

    /// The segments, outermost first.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// The last segment — a file or directory name.
    pub fn name(&self) -> &str {
        self.segments.last().map(String::as_str).unwrap_or("")
    }

    /// This path with `name` appended. The name is checked like any segment, so
    /// a listing cannot build a path a parse would have refused.
    pub fn join(&self, name: &str) -> Result<MindPath, PathError> {
        check_segment(name)?;
        if self.segments.len() + 1 > MAX_DEPTH {
            return Err(PathError::TooDeep(self.segments.len() + 1));
        }
        let mut segments = self.segments.clone();
        segments.push(name.to_owned());
        Ok(MindPath { segments })
    }

    /// The top-level area — `layers`, `responses`, … — or `None` at the root.
    ///
    /// The world filter is expressed in terms of this, so it is named rather
    /// than indexed at each call site.
    pub fn area(&self) -> Option<&str> {
        self.segments.first().map(String::as_str)
    }

    /// The lowercased extension, if the last segment has one.
    pub fn ext(&self) -> Option<String> {
        let name = self.name();
        let (_, ext) = name.rsplit_once('.')?;
        (!ext.is_empty()).then(|| ext.to_ascii_lowercase())
    }

    /// Refuse anything this editor cannot write.
    ///
    /// The mind holds `.md` and `.yaml` and nothing else, and both are text
    /// this console can put in a textarea. Opening the set later is a line
    /// here; opening it accidentally — by letting any extension through — would
    /// mean a PUT could drop a `.exe` into a directory the daemon reads.
    pub fn check_editable(&self) -> Result<(), PathError> {
        match self.ext().as_deref() {
            Some("md") | Some("yaml") => Ok(()),
            _ => Err(PathError::BadExtension),
        }
    }

    /// The `/`-joined form, for the wire and for logs.
    pub fn as_str(&self) -> String {
        self.segments.join("/")
    }

    /// Resolve against the mind root, proving the result is inside it.
    ///
    /// The containment check runs against the deepest **existing** ancestor,
    /// because `canonicalize` fails on a path that is not there yet and
    /// creating a file is exactly that case. Checking the ancestor is enough:
    /// if every existing directory on the way down is inside the mind, a new
    /// leaf under them is too.
    pub fn resolve(&self, root: &Path) -> Result<PathBuf, PathError> {
        let root = root.canonicalize().map_err(|_| PathError::Escapes)?;
        let mut full = root.clone();
        for seg in &self.segments {
            full.push(seg);
        }

        // Walk up to something that exists, canonicalise that, and require it
        // to be the root or under it.
        let mut probe = full.as_path();
        loop {
            match probe.canonicalize() {
                Ok(real) => {
                    return if real == root || real.starts_with(&root) {
                        Ok(full)
                    } else {
                        Err(PathError::Escapes)
                    };
                }
                Err(_) => match probe.parent() {
                    // Ran out of ancestors without meeting the root, which
                    // means this was never under it.
                    None => return Err(PathError::Escapes),
                    Some(p) => probe = p,
                },
            }
        }
    }
}

/// One segment's syntax.
fn check_segment(seg: &str) -> Result<(), PathError> {
    if seg.is_empty() {
        return Err(PathError::EmptySegment);
    }
    if seg.len() > MAX_SEGMENT {
        return Err(PathError::TooLong(seg.len()));
    }
    if seg == "." || seg == ".." {
        return Err(PathError::Traversal);
    }
    if let Some(c) = seg
        .chars()
        .find(|c| matches!(c, '\\' | '/' | ':' | '\0') || c.is_control())
    {
        return Err(PathError::BadChar(c));
    }
    if seg.ends_with('.') || seg.ends_with(' ') {
        return Err(PathError::TrailingDotOrSpace);
    }
    // Reserved with or without an extension: `nul`, `nul.md`, `NUL.MD`.
    let stem = seg.split('.').next().unwrap_or(seg).to_ascii_lowercase();
    if RESERVED.contains(&stem.as_str()) {
        return Err(PathError::Reserved);
    }
    // Belt and braces: whatever the string looked like, the OS must read it as
    // one ordinary name. This is what catches a platform-specific spelling of
    // traversal that the checks above did not anticipate.
    let mut components = Path::new(seg).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(PathError::Traversal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ordinary_path_parses_and_keeps_its_segments() {
        let p = MindPath::parse("layers/world/ammo/bolt.md").expect("parses");
        assert_eq!(p.segments(), ["layers", "world", "ammo", "bolt.md"]);
        assert_eq!(p.area(), Some("layers"));
        assert_eq!(p.name(), "bolt.md");
        assert_eq!(p.ext().as_deref(), Some("md"));
        assert_eq!(p.as_str(), "layers/world/ammo/bolt.md");
    }

    /// The one file in the real mind with a non-ASCII name. An allow-list of
    /// `[a-z0-9_-]` would make it unreachable through the console — invisible
    /// rather than protected — so it is pinned here.
    #[test]
    fn a_unicode_name_is_a_valid_path() {
        let p = MindPath::parse("layers/world/relationships/bonds/protégés.md").expect("parses");
        assert_eq!(p.name(), "protégés.md");
        assert_eq!(p.ext().as_deref(), Some("md"));
    }

    #[test]
    fn the_empty_path_is_the_root() {
        for raw in ["", "/", "///"] {
            let p = MindPath::parse(raw).expect("parses");
            assert_eq!(p, MindPath::root(), "{raw:?}");
            assert_eq!(p.segments().len(), 0);
            assert_eq!(p.area(), None);
        }
    }

    /// Every spelling of "leave the directory" this is expected to meet.
    #[test]
    fn traversal_is_refused_however_it_is_spelled() {
        for raw in [
            "..",
            "../etc",
            "layers/../..",
            "layers/../../etc/passwd",
            "./../x",
            "layers/./../..",
        ] {
            assert!(
                MindPath::parse(raw).is_err(),
                "{raw:?} was accepted as a path"
            );
        }
    }

    /// A backslash is a separator on Windows, so a segment holding one is two
    /// segments in disguise — and `..\..\` is traversal that contains no `/`.
    #[test]
    fn a_backslash_is_not_an_ordinary_character() {
        for raw in [r"..\..\etc", r"layers\world", r"a\b.md"] {
            assert!(
                matches!(
                    MindPath::parse(raw),
                    Err(PathError::BadChar('\\')) | Err(PathError::Traversal)
                ),
                "{raw:?} was accepted"
            );
        }
    }

    /// A colon names a drive or an NTFS alternate data stream. Neither is a
    /// file in the mind.
    #[test]
    fn a_drive_or_stream_is_refused() {
        for raw in ["c:/windows", "c:", "layers/world:hidden", "x.md:stream"] {
            assert!(MindPath::parse(raw).is_err(), "{raw:?} was accepted");
        }
    }

    /// A NUL truncates the name in any C API downstream, so a path validated
    /// whole could be opened as a prefix of itself.
    #[test]
    fn a_nul_or_control_character_is_refused() {
        assert_eq!(
            MindPath::parse("layers/a\0b.md"),
            Err(PathError::BadChar('\0'))
        );
        assert_eq!(
            MindPath::parse("layers/a\nb.md"),
            Err(PathError::BadChar('\n'))
        );
    }

    /// Windows strips a trailing dot or space, so `a.md ` and `a.md` would name
    /// one file by two paths — and only one of them would have been checked.
    #[test]
    fn a_trailing_dot_or_space_is_refused() {
        for raw in ["layers/a.md ", "layers/a.md.", "layers/dir /x.md"] {
            assert_eq!(
                MindPath::parse(raw),
                Err(PathError::TrailingDotOrSpace),
                "{raw:?}"
            );
        }
    }

    #[test]
    fn device_names_are_refused_with_or_without_an_extension() {
        for raw in ["nul", "con.md", "layers/NUL.MD", "layers/com1.yaml", "aux"] {
            assert_eq!(MindPath::parse(raw), Err(PathError::Reserved), "{raw:?}");
        }
        // Only the exact stem: a real word that starts with one is fine.
        assert!(MindPath::parse("layers/console.md").is_ok());
        assert!(MindPath::parse("layers/nullify.md").is_ok());
    }

    #[test]
    fn an_empty_segment_is_refused() {
        assert_eq!(
            MindPath::parse("layers//world"),
            Err(PathError::EmptySegment)
        );
    }

    #[test]
    fn the_bounds_are_enforced() {
        let deep = (0..MAX_DEPTH + 1)
            .map(|_| "a")
            .collect::<Vec<_>>()
            .join("/");
        assert!(matches!(MindPath::parse(&deep), Err(PathError::TooDeep(_))));
        let long_seg = "a".repeat(MAX_SEGMENT + 1);
        assert!(matches!(
            MindPath::parse(&long_seg),
            Err(PathError::TooLong(_))
        ));
        let long_total = "a/".repeat(MAX_TOTAL);
        assert!(matches!(
            MindPath::parse(&long_total),
            Err(PathError::TooLong(_))
        ));
    }

    #[test]
    fn only_text_documents_are_editable() {
        for ok in ["a/b.md", "a/b.yaml", "a/B.YAML"] {
            let p = MindPath::parse(ok).expect("parses");
            assert!(p.check_editable().is_ok(), "{ok}");
        }
        for bad in ["a/b.exe", "a/b.png", "a/b", "a/b.md.exe"] {
            let p = MindPath::parse(bad).expect("parses");
            assert_eq!(p.check_editable(), Err(PathError::BadExtension), "{bad}");
        }
    }

    #[test]
    fn join_checks_the_name_it_is_given() {
        let base = MindPath::parse("layers").expect("parses");
        assert_eq!(base.join("world").unwrap().as_str(), "layers/world");
        for bad in ["..", "a/b", r"a\b", "nul", "x."] {
            assert!(base.join(bad).is_err(), "{bad} was joined");
        }
    }

    /// Built up a segment at a time, which is how `address` assembles one — the
    /// walk back down is [`Address::parent`]'s job, one level of abstraction up.
    #[test]
    fn a_path_is_built_by_joining_and_keeps_its_order() {
        let p = MindPath::parse("layers")
            .unwrap()
            .join("world")
            .unwrap()
            .join("ammo")
            .unwrap()
            .join("bolt.md")
            .unwrap();
        assert_eq!(p.as_str(), "layers/world/ammo/bolt.md");
        assert_eq!(p, MindPath::parse("layers/world/ammo/bolt.md").unwrap());
        assert_eq!(p.name(), "bolt.md");
        assert_eq!(p.area(), Some("layers"));
    }

    // ── resolve, against a real directory ────────────────────────────────────

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("npcd-mindpath-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn resolve_lands_inside_the_root() {
        let root = tmp("inside");
        std::fs::create_dir_all(root.join("layers/world")).unwrap();
        std::fs::write(root.join("layers/world/a.md"), "x").unwrap();

        let p = MindPath::parse("layers/world/a.md").unwrap();
        let full = p.resolve(&root).expect("resolves");
        assert!(full.ends_with("a.md"));
        assert_eq!(std::fs::read_to_string(&full).unwrap(), "x");
    }

    /// The case that makes the ancestor walk necessary: a file being created
    /// does not exist, so `canonicalize` on it fails, and refusing there would
    /// make every create impossible.
    #[test]
    fn resolve_allows_a_file_that_does_not_exist_yet() {
        let root = tmp("create");
        std::fs::create_dir_all(root.join("layers")).unwrap();
        let p = MindPath::parse("layers/brand-new.md").unwrap();
        let full = p.resolve(&root).expect("resolves");
        assert!(!full.exists());
        assert!(full.starts_with(root.canonicalize().unwrap()));
    }

    /// Whole directories that do not exist yet resolve too — creating
    /// `a/b/c.md` under a fresh `a/` is one write, not three checks.
    #[test]
    fn resolve_allows_a_whole_missing_branch() {
        let root = tmp("branch");
        let p = MindPath::parse("layers/brand/new/leaf.md").unwrap();
        assert!(p.resolve(&root).is_ok());
    }

    #[test]
    fn resolve_refuses_a_root_that_is_not_there() {
        let p = MindPath::parse("a.md").unwrap();
        assert_eq!(
            p.resolve(Path::new("/no/such/mind/anywhere")),
            Err(PathError::Escapes)
        );
    }
}
