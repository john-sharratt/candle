//! Diffs as patches: the changed lines, hunk by hunk, per file.
//!
//! One `--patch-with-raw -z` run gives both halves: the raw entries
//! (NUL-delimited, so paths are exact) and then the patch text, one section
//! per file in the same order. Sections are paired with entries by position
//! rather than by parsing paths out of `diff --git` headers, which are
//! ambiguous for paths containing spaces. A type change (file to symlink)
//! is two sections in git's output — a deletion and a creation — and is
//! paired as such.

use std::collections::HashMap;

use crate::error::GitError;
use crate::read::diff::{parse_raw, DiffEntry, DiffStatus};
use crate::types::{RepoPath, Rev};
use crate::Repo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    Context,
    Added,
    Removed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchLine {
    pub kind: LineKind,
    /// The line's bytes, without its prefix or newline.
    pub text: Vec<u8>,
    /// The file ends here without a trailing newline.
    pub no_newline_at_end: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    pub old_start: u32,
    pub old_lines: u32,
    pub new_start: u32,
    pub new_lines: u32,
    /// The text after the closing `@@` — usually the enclosing function.
    pub context: String,
    pub lines: Vec<PatchLine>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilePatch {
    pub entry: DiffEntry,
    /// Git judged the content binary; there are no hunks.
    pub binary: bool,
    pub hunks: Vec<Hunk>,
}

impl FilePatch {
    fn count(&self, kind: LineKind) -> Option<u32> {
        (!self.binary).then(|| {
            self.hunks
                .iter()
                .flat_map(|h| &h.lines)
                .filter(|l| l.kind == kind)
                .count() as u32
        })
    }

    /// Lines added, or `None` for a binary file — what `--numstat` reports.
    pub fn added(&self) -> Option<u32> {
        self.count(LineKind::Added)
    }

    /// Lines removed, or `None` for a binary file.
    pub fn removed(&self) -> Option<u32> {
        self.count(LineKind::Removed)
    }
}

/// Split `--patch-with-raw -z` output into its raw entries and patch text.
fn split_raw(out: &[u8]) -> Result<(Vec<DiffEntry>, &[u8]), GitError> {
    let bad = |what: &str| GitError::malformed("diff", what.to_string());
    let mut pos = 0;
    let field_end = |from: usize| -> Result<usize, GitError> {
        out[from..]
            .iter()
            .position(|&b| b == 0)
            .map(|i| from + i)
            .ok_or_else(|| bad("unterminated raw field"))
    };
    while pos < out.len() {
        match out[pos] {
            b':' => {
                let end = field_end(pos)?;
                let header = &out[pos..end];
                let status = header
                    .rsplit(|&b| b == b' ')
                    .next()
                    .and_then(|s| s.first().copied())
                    .ok_or_else(|| bad("raw header without a status"))?;
                let paths = if matches!(status, b'R' | b'C') { 2 } else { 1 };
                pos = end + 1;
                for _ in 0..paths {
                    pos = field_end(pos)? + 1;
                }
            }
            // The separator between the raw records and the patch.
            0 | b'\n' => {
                let entries = parse_raw(&out[..pos])?;
                return Ok((entries, &out[pos + 1..]));
            }
            _ => return Err(bad("expected a raw record or the patch separator")),
        }
    }
    Ok((parse_raw(out)?, &[]))
}

/// The patch text's sections, one per `diff --git` header.
fn sections(patch: &[u8]) -> Vec<&[u8]> {
    let mut starts: Vec<usize> = Vec::new();
    let mut line_start = 0;
    for (i, &b) in patch.iter().enumerate() {
        if i == line_start && patch[i..].starts_with(b"diff --git ") {
            starts.push(i);
        }
        if b == b'\n' {
            line_start = i + 1;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(n, &s)| &patch[s..starts.get(n + 1).copied().unwrap_or(patch.len())])
        .collect()
}

/// `@@ -a[,b] +c[,d] @@ context`.
fn parse_hunk_header(line: &[u8]) -> Result<Hunk, GitError> {
    let text = String::from_utf8_lossy(line);
    let bad = || GitError::malformed("diff", format!("hunk header {text:?}"));
    let rest = text.strip_prefix("@@ -").ok_or_else(bad)?;
    let (ranges, context) = rest.split_once(" @@").ok_or_else(bad)?;
    let (old, new) = ranges.split_once(" +").ok_or_else(bad)?;
    let range = |r: &str| -> Result<(u32, u32), GitError> {
        let (start, len) = r.split_once(',').unwrap_or((r, "1"));
        Ok((
            start.parse().map_err(|_| bad())?,
            len.parse().map_err(|_| bad())?,
        ))
    };
    let (old_start, old_lines) = range(old)?;
    let (new_start, new_lines) = range(new)?;
    Ok(Hunk {
        old_start,
        old_lines,
        new_start,
        new_lines,
        context: context.strip_prefix(' ').unwrap_or(context).to_string(),
        lines: Vec::new(),
    })
}

/// One section's binary flag and hunks. Each hunk takes exactly the lines
/// its header counts, so a content line can never be mistaken for a header.
fn parse_section(section: &[u8]) -> Result<(bool, Vec<Hunk>), GitError> {
    let mut lines: Vec<&[u8]> = section.split(|&b| b == b'\n').collect();
    if lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    let mut binary = false;
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        if line.starts_with(b"Binary files ") || line == b"GIT binary patch" {
            binary = true;
            continue;
        }
        if !line.starts_with(b"@@ ") {
            continue; // header lines: diff --git, index, ---, +++, modes
        }
        let mut hunk = parse_hunk_header(line)?;
        let (mut old, mut new) = (hunk.old_lines, hunk.new_lines);
        while old > 0 || new > 0 {
            let line = lines
                .get(i)
                .ok_or_else(|| GitError::malformed("diff", "hunk shorter than its header"))?;
            i += 1;
            let (kind, text) = match line.first() {
                Some(b' ') if old > 0 && new > 0 => {
                    old -= 1;
                    new -= 1;
                    (LineKind::Context, &line[1..])
                }
                Some(b'-') if old > 0 => {
                    old -= 1;
                    (LineKind::Removed, &line[1..])
                }
                Some(b'+') if new > 0 => {
                    new -= 1;
                    (LineKind::Added, &line[1..])
                }
                Some(b'\\') => {
                    mark_no_newline(&mut hunk)?;
                    continue;
                }
                _ => {
                    return Err(GitError::malformed(
                        "diff",
                        format!("unexpected hunk line {:?}", String::from_utf8_lossy(line)),
                    ))
                }
            };
            hunk.lines.push(PatchLine {
                kind,
                text: text.to_vec(),
                no_newline_at_end: false,
            });
        }
        // A marker after the hunk's last line belongs to it.
        while lines.get(i).is_some_and(|l| l.first() == Some(&b'\\')) {
            mark_no_newline(&mut hunk)?;
            i += 1;
        }
        hunks.push(hunk);
    }
    Ok((binary, hunks))
}

fn mark_no_newline(hunk: &mut Hunk) -> Result<(), GitError> {
    hunk.lines
        .last_mut()
        .map(|l| l.no_newline_at_end = true)
        .ok_or_else(|| GitError::malformed("diff", "no-newline marker before any line"))
}

/// Pair raw entries with their patch sections.
pub(crate) fn parse_patches(out: &[u8]) -> Result<Vec<FilePatch>, GitError> {
    let (entries, patch) = split_raw(out)?;
    let sections = sections(patch);
    let needed: usize = entries
        .iter()
        .map(|e| {
            if e.status == DiffStatus::TypeChanged {
                2
            } else {
                1
            }
        })
        .sum();
    if needed != sections.len() {
        return Err(GitError::malformed(
            "diff",
            format!("{} patch sections for {needed} expected", sections.len()),
        ));
    }
    let mut sections = sections.into_iter();
    entries
        .into_iter()
        .map(|entry| {
            let parts = if entry.status == DiffStatus::TypeChanged {
                2
            } else {
                1
            };
            let mut binary = false;
            let mut hunks = Vec::new();
            for _ in 0..parts {
                let (b, h) = parse_section(sections.next().expect("counted above"))?;
                binary |= b;
                hunks.extend(h);
            }
            Ok(FilePatch {
                entry,
                binary,
                hunks,
            })
        })
        .collect()
}

const PATCH_ARGS: &[&str] = &[
    "-z",
    "-M",
    "--patch-with-raw",
    "--no-abbrev",
    "--full-index",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
];

impl Repo {
    /// The patches from commit `from` to commit `to`, with `context` lines
    /// around each change, limited to `paths` when any are given.
    pub fn patches(
        &self,
        from: &Rev,
        to: &Rev,
        context: u32,
        paths: &[&RepoPath],
    ) -> Result<Vec<FilePatch>, GitError> {
        let mut inv = self
            .git("diff-tree")
            .arg("-r")
            .args(PATCH_ARGS)
            .arg(format!("-U{context}"))
            .arg("--end-of-options")
            .args([from.spec(), to.spec()]);
        if !paths.is_empty() {
            inv = inv.arg("--").args(paths.iter().map(|p| p.as_str()));
        }
        let out = inv
            .read_only()
            .about_rev(format!("{}..{}", from.spec(), to.spec()))
            .run_ok()?;
        parse_patches(&out)
    }

    /// The working tree's patches relative to commit `from`, untracked files
    /// excluded. Like [`Repo::diff_worktree`], never writes the index: the
    /// files that really changed are found first, and only they are
    /// diffed.
    pub fn patches_worktree(&self, from: &Rev, context: u32) -> Result<Vec<FilePatch>, GitError> {
        let changed = self.diff_worktree(from)?;
        if changed.is_empty() {
            return Ok(Vec::new());
        }
        let mut paths: Vec<&RepoPath> = Vec::new();
        let mut hashed: HashMap<&RepoPath, _> = HashMap::new();
        for e in &changed {
            for side in [&e.old, &e.new].into_iter().flatten() {
                if !paths.contains(&&side.path) {
                    paths.push(&side.path);
                }
            }
            if let Some(new) = &e.new {
                hashed.insert(&new.path, new.oid.clone());
            }
        }
        let out = self
            .git("diff-index")
            .args(PATCH_ARGS)
            .arg(format!("-U{context}"))
            .arg("--end-of-options")
            .arg(from.spec())
            .arg("--")
            .args(paths.iter().map(|p| p.as_str()))
            .read_only()
            .about_rev(from.spec())
            .run_ok()?;
        let mut patches = parse_patches(&out)?;
        for p in &mut patches {
            if let Some(new) = p.entry.new.as_mut() {
                if let Some(oid) = hashed.get(&new.path) {
                    new.oid = oid.clone();
                }
            }
        }
        Ok(patches)
    }

    /// The patch from `from` to `to` as bytes `git apply` accepts — full
    /// object ids and binary content included — for [`Repo::apply_patch`].
    pub fn patch_bytes(&self, from: &Rev, to: &Rev) -> Result<Vec<u8>, GitError> {
        self.git("diff-tree")
            .args([
                "-r",
                "-p",
                "-M",
                "--binary",
                "--full-index",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--end-of-options",
            ])
            .args([from.spec(), to.spec()])
            .read_only()
            .about_rev(format!("{}..{}", from.spec(), to.spec()))
            .run_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::changeset::ChangeSet;
    use crate::testing::TestRepo;
    use crate::types::{GitTime, Oid, Signature};

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    fn line(kind: LineKind, text: &str) -> PatchLine {
        PatchLine {
            kind,
            text: text.as_bytes().to_vec(),
            no_newline_at_end: false,
        }
    }

    #[test]
    fn a_patch_with_raw_parses_from_raw_bytes() {
        let z = "0".repeat(40);
        let raw = format!(
            ":100644 100644 {A} {B} M\0edit rs.rs\0\
             :100644 100644 {A} {B} M\0bin.dat\0\
             :000000 100644 {z} {B} A\0new.txt\0\
             :100644 100644 {A} {A} R100\0old.rs\0moved.rs\0\0"
        );
        // `concat!`, not `\` continuations, which would strip the leading
        // space that marks a context line.
        let patch = concat!(
            "diff --git a/edit rs.rs b/edit rs.rs\n",
            "index x..y 100644\n",
            "--- a/edit rs.rs\n",
            "+++ b/edit rs.rs\n",
            "@@ -1,3 +1,3 @@ fn main() {\n",
            " keep\n",
            "--- removed line that looks like a header\n",
            "++++ added line that looks like a header\n",
            " keep\n",
            "@@ -10 +10 @@\n",
            "-last\n",
            "\\ No newline at end of file\n",
            "+last\n",
            "diff --git a/bin.dat b/bin.dat\n",
            "index x..y 100644\n",
            "Binary files a/bin.dat and b/bin.dat differ\n",
            "diff --git a/new.txt b/new.txt\n",
            "new file mode 100644\n",
            "index x..y\n",
            "--- /dev/null\n",
            "+++ b/new.txt\n",
            "@@ -0,0 +1 @@\n",
            "+hello\n",
            "diff --git a/old.rs b/moved.rs\n",
            "similarity index 100%\n",
            "rename from old.rs\n",
            "rename to moved.rs\n",
        );
        let mut bytes = raw.into_bytes();
        bytes.extend_from_slice(patch.as_bytes());
        let p = parse_patches(&bytes).unwrap();
        assert_eq!(p.len(), 4);

        assert_eq!(p[0].entry.new.as_ref().unwrap().path.as_str(), "edit rs.rs");
        assert_eq!(p[0].hunks.len(), 2);
        let h = &p[0].hunks[0];
        assert_eq!(
            (h.old_start, h.old_lines, h.new_start, h.new_lines),
            (1, 3, 1, 3)
        );
        assert_eq!(h.context, "fn main() {");
        assert_eq!(
            h.lines,
            vec![
                line(LineKind::Context, "keep"),
                line(
                    LineKind::Removed,
                    "-- removed line that looks like a header"
                ),
                line(LineKind::Added, "+++ added line that looks like a header"),
                line(LineKind::Context, "keep"),
            ]
        );
        let h = &p[0].hunks[1];
        assert_eq!(
            (h.old_start, h.old_lines, h.new_start, h.new_lines),
            (10, 1, 10, 1)
        );
        assert!(h.lines[0].no_newline_at_end);
        assert!(!h.lines[1].no_newline_at_end);
        assert_eq!((p[0].added(), p[0].removed()), (Some(2), Some(2)));

        assert!(p[1].binary);
        assert_eq!((p[1].added(), p[1].removed()), (None, None));
        assert_eq!(p[2].hunks[0].lines, vec![line(LineKind::Added, "hello")]);
        assert_eq!(p[3].entry.status, DiffStatus::Renamed(100));
        assert!(p[3].hunks.is_empty());
    }

    #[test]
    fn a_hunk_shorter_than_its_header_is_malformed() {
        let raw = format!(":100644 100644 {A} {B} M\0f\0\0");
        let patch = "diff --git a/f b/f\n@@ -1,2 +1,2 @@\n-a\n+b\n";
        let mut bytes = raw.into_bytes();
        bytes.extend_from_slice(patch.as_bytes());
        assert!(parse_patches(&bytes).is_err());
    }

    #[test]
    fn a_section_count_mismatch_is_malformed() {
        let raw = format!(":100644 100644 {A} {B} M\0f\0:100644 100644 {A} {B} M\0g\0\0");
        let patch = "diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n";
        let mut bytes = raw.into_bytes();
        bytes.extend_from_slice(patch.as_bytes());
        assert!(parse_patches(&bytes).is_err());
    }

    /// Line counts agree with `git diff --numstat` for every file.
    #[test]
    fn counts_match_numstat_and_hunks_hold_the_exact_lines() {
        let t = TestRepo::init();
        let before: String = (1..=30).map(|i| format!("line {i}\n")).collect();
        t.write("a file.txt", before.as_bytes());
        t.write("bin.dat", &[0, 1, 2, 3, 0, 255]);
        t.write("gone.txt", b"gone\n");
        let base = t.commit_all("base");
        let after = before
            .replace("line 2\n", "line two\n")
            .replace("line 25\n", "")
            + "line 31\n";
        t.write("a file.txt", after.as_bytes());
        t.write("bin.dat", &[0, 9, 9, 9, 0, 255]);
        std::fs::remove_file(t.path.join("gone.txt")).unwrap();
        t.write("new.txt", b"one\ntwo\n");
        let next = t.commit_all("next");
        let repo = t.repo();

        let patches = repo
            .patches(&Rev::Oid(base.clone()), &Rev::Oid(next.clone()), 1, &[])
            .unwrap();
        let numstat = t.git(&["diff", "--numstat", base.as_str(), next.as_str()]);
        for line in numstat.lines() {
            let mut f = line.splitn(3, '\t');
            let (added, removed, path) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
            let p = patches
                .iter()
                .find(|p| {
                    p.entry
                        .new
                        .as_ref()
                        .or(p.entry.old.as_ref())
                        .unwrap()
                        .path
                        .as_str()
                        == path
                })
                .unwrap_or_else(|| panic!("{path} missing"));
            let fmt = |n: Option<u32>| n.map_or("-".to_string(), |n| n.to_string());
            assert_eq!(
                (fmt(p.added()), fmt(p.removed())),
                (added.into(), removed.into()),
                "{path}"
            );
        }

        let text = patches
            .iter()
            .find(|p| {
                p.entry
                    .new
                    .as_ref()
                    .is_some_and(|s| s.path.as_str() == "a file.txt")
            })
            .unwrap();
        assert_eq!(text.hunks.len(), 3, "{:?}", text.hunks);
        assert_eq!(
            text.hunks[0].lines,
            vec![
                line(LineKind::Context, "line 1"),
                line(LineKind::Removed, "line 2"),
                line(LineKind::Added, "line two"),
                line(LineKind::Context, "line 3"),
            ]
        );
        assert_eq!(text.hunks[0].old_start, 1);

        // Limited to one path.
        let only = RepoPath::parse("new.txt").unwrap();
        let limited = repo
            .patches(&Rev::Oid(base), &Rev::Oid(next), 3, &[&only])
            .unwrap();
        assert_eq!(limited.len(), 1);
        assert_eq!(
            limited[0].hunks[0].lines,
            vec![line(LineKind::Added, "one"), line(LineKind::Added, "two")]
        );
    }

    /// A file turned into a symlink is two sections in git's output, paired
    /// into one patch.
    #[test]
    fn a_type_change_pairs_both_sections() {
        let t = TestRepo::init();
        t.write("link", b"was a file\n");
        let base = t.commit_all("base");
        let repo = t.repo();
        let mut c = ChangeSet::new();
        c.symlink(RepoPath::parse("link").unwrap(), "target")
            .unwrap();
        let sig =
            Signature::new("A", "a@x", GitTime::parse_raw("1700000000 +0000").unwrap()).unwrap();
        let next = repo
            .commit_changes(&base, &c, "to link", &sig, &sig)
            .unwrap();
        let p = repo
            .patches(&Rev::Oid(base), &Rev::Oid(next), 3, &[])
            .unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].entry.status, DiffStatus::TypeChanged);
        assert_eq!((p[0].added(), p[0].removed()), (Some(1), Some(1)));
    }

    /// A worktree patch covers real edits only — a stat-only touch is not a
    /// change — carries the hashed new id, and leaves the index untouched.
    #[test]
    fn worktree_patches_cover_real_edits_only() {
        let t = TestRepo::init();
        t.write("same.txt", b"same\n");
        t.write("edit.txt", b"before\n");
        let base = t.commit_all("base");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        t.write("same.txt", b"same\n");
        t.write("edit.txt", b"after\n");
        let index = std::fs::read(t.path.join(".git/index")).unwrap();

        let p = t.repo().patches_worktree(&Rev::Oid(base), 3).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(
            p[0].hunks[0].lines,
            vec![
                line(LineKind::Removed, "before"),
                line(LineKind::Added, "after")
            ]
        );
        let expected = t.git(&["hash-object", "edit.txt"]);
        assert_eq!(
            p[0].entry
                .new
                .as_ref()
                .unwrap()
                .oid
                .as_ref()
                .map(Oid::as_str),
            Some(expected.trim())
        );
        assert_eq!(std::fs::read(t.path.join(".git/index")).unwrap(), index);
    }

    #[test]
    fn no_changes_is_no_patches() {
        let t = TestRepo::init();
        t.write("a", b"a\n");
        let c = t.commit_all("base");
        let repo = t.repo();
        assert!(repo
            .patches(&Rev::Oid(c.clone()), &Rev::Oid(c.clone()), 3, &[])
            .unwrap()
            .is_empty());
        assert!(repo.patches_worktree(&Rev::Oid(c), 3).unwrap().is_empty());
    }
}
