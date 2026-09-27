//! Who last changed each line of a file, from `blame --line-porcelain`.

use crate::error::GitError;
use crate::types::{GitTime, Oid, RepoPath, Rev, Signature};
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlameLine {
    /// The commit that last changed this line.
    pub commit: Oid,
    /// The line's number in that commit's version of the file (1-based).
    pub orig_line: u32,
    /// The line's number in the blamed revision (1-based).
    pub final_line: u32,
    /// The file's path in that commit — differs after a rename.
    pub orig_path: RepoPath,
    pub author: Signature,
    pub committer: Signature,
    pub summary: String,
    /// The line's bytes, without its newline.
    pub content: Vec<u8>,
}

/// An inclusive, 1-based line range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Default)]
struct Pending {
    name: Option<String>,
    mail: Option<String>,
    time: Option<String>,
    tz: Option<String>,
}

impl Pending {
    fn signature(&self, who: &str) -> Result<Signature, GitError> {
        let bad = || GitError::malformed("blame", format!("incomplete {who} fields"));
        let mail = self.mail.as_deref().ok_or_else(bad)?;
        let mail = mail
            .strip_prefix('<')
            .and_then(|m| m.strip_suffix('>'))
            .unwrap_or(mail);
        let raw = format!(
            "{} {}",
            self.time.as_deref().ok_or_else(bad)?,
            self.tz.as_deref().ok_or_else(bad)?
        );
        Ok(Signature::recorded(
            self.name.as_deref().ok_or_else(bad)?,
            mail,
            GitTime::parse_raw(&raw)?,
        ))
    }
}

/// A porcelain `filename` value. Git C-quotes it whenever it holds a `"`, a
/// `\` or a control character — `core.quotePath=false` only stops the
/// quoting of non-ASCII bytes — so `say "hi".txt` arrives as
/// `"say \"hi\".txt"`.
fn unquote_filename(value: &str) -> Result<String, GitError> {
    let Some(inner) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) else {
        return Ok(value.to_string());
    };
    let bad = || GitError::malformed("blame", format!("filename {value:?}"));
    let mut out = Vec::with_capacity(inner.len());
    let mut bytes = inner.bytes();
    while let Some(b) = bytes.next() {
        if b != b'\\' {
            out.push(b);
            continue;
        }
        out.push(match bytes.next().ok_or_else(bad)? {
            b'a' => 0x07,
            b'b' => 0x08,
            b't' => b'\t',
            b'n' => b'\n',
            b'v' => 0x0b,
            b'f' => 0x0c,
            b'r' => b'\r',
            b'"' => b'"',
            b'\\' => b'\\',
            d @ b'0'..=b'3' => {
                let mut n = u32::from(d - b'0');
                for _ in 0..2 {
                    match bytes.next() {
                        Some(o @ b'0'..=b'7') => n = n * 8 + u32::from(o - b'0'),
                        _ => return Err(bad()),
                    }
                }
                n as u8
            }
            _ => return Err(bad()),
        });
    }
    String::from_utf8(out).map_err(|_| bad())
}

/// Parse `--line-porcelain`: per line, a `<oid> <orig> <final> [<n>]`
/// header, `key value` lines, then the content after a tab.
pub(crate) fn parse_blame(out: &[u8]) -> Result<Vec<BlameLine>, GitError> {
    let bad = |what: String| GitError::malformed("blame", what);
    let mut lines = out.split(|&b| b == b'\n').peekable();
    let mut result = Vec::new();
    while let Some(header) = lines.next() {
        if header.is_empty() && lines.peek().is_none() {
            break;
        }
        let header = std::str::from_utf8(header).map_err(|e| bad(e.to_string()))?;
        let parts: Vec<&str> = header.split(' ').collect();
        if parts.len() < 3 {
            return Err(bad(format!("header {header:?}")));
        }
        let commit = Oid::parse(parts[0])?;
        let num = |s: &str| {
            s.parse::<u32>()
                .map_err(|_| bad(format!("header {header:?}")))
        };
        let (orig_line, final_line) = (num(parts[1])?, num(parts[2])?);
        let (mut author, mut committer) = (Pending::default(), Pending::default());
        let mut summary = String::new();
        let mut filename = None;
        let content = loop {
            let line = lines
                .next()
                .ok_or_else(|| bad("blame entry without content".into()))?;
            if let Some(content) = line.strip_prefix(b"\t") {
                break content.to_vec();
            }
            let line = String::from_utf8_lossy(line);
            let (key, value) = line.split_once(' ').unwrap_or((&line, ""));
            let value = value.to_string();
            match key {
                "author" => author.name = Some(value),
                "author-mail" => author.mail = Some(value),
                "author-time" => author.time = Some(value),
                "author-tz" => author.tz = Some(value),
                "committer" => committer.name = Some(value),
                "committer-mail" => committer.mail = Some(value),
                "committer-time" => committer.time = Some(value),
                "committer-tz" => committer.tz = Some(value),
                "summary" => summary = value,
                "filename" => filename = Some(value),
                _ => {} // previous, boundary
            }
        };
        result.push(BlameLine {
            commit,
            orig_line,
            final_line,
            orig_path: RepoPath::parse(&unquote_filename(
                &filename.ok_or_else(|| bad("blame entry without a filename".into()))?,
            )?)?,
            author: author.signature("author")?,
            committer: committer.signature("committer")?,
            summary,
            content,
        });
    }
    Ok(result)
}

impl Repo {
    /// Blame `path` at `rev`, following lines through renames and, within
    /// `lines` when given, only that range. `None` when `rev` holds no such
    /// file.
    pub fn blame(
        &self,
        rev: &Rev,
        path: &RepoPath,
        lines: Option<LineRange>,
    ) -> Result<Option<Vec<BlameLine>>, GitError> {
        let found = self.tree_entries(rev, &[path])?;
        if !found.iter().any(|e| e.mode.is_blob()) {
            return Ok(None);
        }
        // No text-conversion filters: they are programs named in the
        // repository's config, and blame runs them by default.
        let mut inv = self
            .git("blame")
            .args(["--line-porcelain", "--no-textconv"]);
        if let Some(r) = lines {
            inv = inv.arg(format!("-L{},{}", r.start, r.end));
        }
        let out = inv
            .arg(rev.spec())
            .arg("--")
            .arg(path.as_str())
            .read_only()
            .about_rev(rev.spec())
            .run_ok()?;
        parse_blame(&out).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::grep::GrepQuery;
    use crate::testing::TestRepo;

    #[test]
    fn line_porcelain_parses_from_raw_bytes() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let out = format!(
            "{a} 1 1 2\nauthor Ada\nauthor-mail <ada@x>\nauthor-time 1700000000\nauthor-tz +0100\n\
committer Bob\ncommitter-mail <bob@x>\ncommitter-time 1700000060\ncommitter-tz -0500\n\
summary first line\nboundary\nfilename old name.rs\n\tfn main() {{\n\
{a} 2 2\nauthor Ada\nauthor-mail <ada@x>\nauthor-time 1700000000\nauthor-tz +0100\n\
committer Bob\ncommitter-mail <bob@x>\ncommitter-time 1700000060\ncommitter-tz -0500\n\
summary first line\nfilename old name.rs\n\t\n"
        );
        let b = parse_blame(out.as_bytes()).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].commit.as_str(), a);
        assert_eq!((b[0].orig_line, b[0].final_line), (1, 1));
        assert_eq!(b[0].orig_path.as_str(), "old name.rs");
        assert_eq!(b[0].author.name(), "Ada");
        assert_eq!(b[0].author.email(), "ada@x");
        assert_eq!(b[0].author.when.offset_minutes, 60);
        assert_eq!(b[0].committer.when.offset_minutes, -300);
        assert_eq!(b[0].summary, "first line");
        assert_eq!(b[0].content, b"fn main() {");
        assert_eq!(b[1].content, b"", "an empty line is still a line");
    }

    /// **A quoted filename and an empty email both parse.** Git C-quotes a
    /// name holding `"` or `\` whatever `core.quotePath` says, and imported
    /// commits record `Name <>`.
    #[test]
    fn a_quoted_filename_and_an_empty_email_parse() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let out = format!(
            "{a} 1 1 1\nauthor svnuser\nauthor-mail <>\nauthor-time 1700000000\nauthor-tz +0000\n\
committer svnuser\ncommitter-mail <>\ncommitter-time 1700000000\ncommitter-tz +0000\n\
summary import\nfilename \"say \\\"hi\\\".txt\"\n\tx\n"
        );
        let b = parse_blame(out.as_bytes()).unwrap();
        assert_eq!(b[0].orig_path.as_str(), "say \"hi\".txt");
        assert_eq!(b[0].author.email(), "");
    }

    #[test]
    fn filename_unquoting_follows_git_c_style() {
        assert_eq!(unquote_filename("plain name.rs").unwrap(), "plain name.rs");
        assert_eq!(unquote_filename(r#""a\\b""#).unwrap(), "a\\b");
        assert_eq!(unquote_filename(r#""caf\303\251""#).unwrap(), "café");
        for bad in [r#""trailing\""#, r#""\q""#, r#""\30""#] {
            assert!(unquote_filename(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn each_line_names_the_commit_that_last_changed_it() {
        let t = TestRepo::init();
        t.write("f.txt", b"one\ntwo\nthree\n");
        let first = t.commit_all("first");
        t.write("f.txt", b"one\nTWO\nthree\nfour\n");
        let second = t.commit_all("second");
        let repo = t.repo();
        let path = RepoPath::parse("f.txt").unwrap();

        let b = repo.blame(&Rev::Head, &path, None).unwrap().unwrap();
        let got: Vec<(&Oid, u32, &[u8])> = b
            .iter()
            .map(|l| (&l.commit, l.final_line, l.content.as_slice()))
            .collect();
        assert_eq!(
            got,
            vec![
                (&first, 1, &b"one"[..]),
                (&second, 2, &b"TWO"[..]),
                (&first, 3, &b"three"[..]),
                (&second, 4, &b"four"[..]),
            ]
        );
        assert_eq!(b[1].summary, "second");
        assert_eq!(b[1].author.name(), "Setup");

        let ranged = repo
            .blame(&Rev::Head, &path, Some(LineRange { start: 2, end: 3 }))
            .unwrap()
            .unwrap();
        assert_eq!(ranged.len(), 2);
        assert_eq!(ranged[0].final_line, 2);

        // An older revision blames its own content.
        let old = repo
            .blame(&Rev::Oid(first.clone()), &path, None)
            .unwrap()
            .unwrap();
        assert!(old.iter().all(|l| l.commit == first));
        assert_eq!(old[1].content, b"two");
    }

    #[test]
    fn blame_follows_a_rename_and_reports_the_old_path() {
        let t = TestRepo::init();
        let body: String = (0..10).map(|i| format!("line {i}\n")).collect();
        t.write("old.rs", body.as_bytes());
        let created = t.commit_all("create");
        t.git(&["mv", "old.rs", "new.rs"]);
        t.commit_all("rename");
        let b = t
            .repo()
            .blame(&Rev::Head, &RepoPath::parse("new.rs").unwrap(), None)
            .unwrap()
            .unwrap();
        assert!(b.iter().all(|l| l.commit == created));
        assert_eq!(b[0].orig_path.as_str(), "old.rs");
    }

    /// **A text-conversion filter named in the repository's config never
    /// runs** under blame or grep. The control run proves the filter is live
    /// for plain git.
    #[test]
    fn textconv_filters_never_run() {
        let t = TestRepo::init();
        let marker = t.path.join("textconv.ran");
        let m = marker.to_string_lossy().replace('\\', "/");
        t.git(&[
            "config",
            "diff.conv.textconv",
            &format!("touch '{m}' && cat"),
        ]);
        t.write(".gitattributes", b"*.txt diff=conv\n");
        t.write("f.txt", b"hello\n");
        t.commit_all("base");
        let path = RepoPath::parse("f.txt").unwrap();

        t.git(&["blame", "--textconv", "--", "f.txt"]);
        assert!(marker.exists(), "the control run fires the filter");
        std::fs::remove_file(&marker).unwrap();

        let repo = t.repo();
        repo.blame(&Rev::Head, &path, None).unwrap().unwrap();
        repo.grep(&Rev::Head, &GrepQuery::regex("hello")).unwrap();
        assert!(!marker.exists(), "a textconv filter ran under the layer");
    }

    #[test]
    fn a_missing_file_or_a_folder_is_none() {
        let t = TestRepo::init();
        t.write("src/a.rs", b"a\n");
        t.commit_all("base");
        let repo = t.repo();
        for p in ["nope.rs", "src"] {
            assert_eq!(
                repo.blame(&Rev::Head, &RepoPath::parse(p).unwrap(), None)
                    .unwrap(),
                None,
                "{p}"
            );
        }
    }
}
