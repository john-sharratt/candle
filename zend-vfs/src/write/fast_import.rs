//! The `git fast-import` stream that writes one commit.
//!
//! The stream only ever names blobs by id — they are written first, through
//! the repository's filters (`write/blobs.rs`) — because fast-import applies
//! no filters to content it is given inline.

use crate::types::{FileMode, Oid, RefName, RepoPath, Signature};

/// One tree change in the commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FileOp {
    Modify {
        mode: FileMode,
        oid: Oid,
        path: RepoPath,
    },
    Delete {
        path: RepoPath,
    },
}

pub(crate) struct CommitStream<'a> {
    pub target: &'a RefName,
    pub author: &'a Signature,
    pub committer: &'a Signature,
    pub message: &'a str,
    pub parent: &'a Oid,
    pub ops: &'a [FileOp],
}

/// A path in fast-import's C-style quotes. Quoting is accepted everywhere
/// and makes spaces and a leading `"` unambiguous; a [`RepoPath`] holds no
/// backslash or control character, so `"` is the only byte to escape.
fn quote(path: &RepoPath) -> String {
    format!("\"{}\"", path.as_str().replace('"', "\\\""))
}

/// A commit message as git stores it: ending in exactly one newline.
pub(crate) fn normalize_message(message: &str) -> String {
    format!("{}\n", message.trim_end_matches('\n'))
}

impl CommitStream<'_> {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let message = normalize_message(self.message);
        let mut s = String::new();
        s.push_str(&format!("commit {}\n", self.target));
        s.push_str(&format!("author {}\n", self.author.to_header()));
        s.push_str(&format!("committer {}\n", self.committer.to_header()));
        s.push_str(&format!("data {}\n{message}\n", message.len()));
        s.push_str(&format!("from {}\n", self.parent));
        for op in self.ops {
            match op {
                FileOp::Modify { mode, oid, path } => {
                    s.push_str(&format!("M {} {oid} {}\n", mode.as_str(), quote(path)));
                }
                FileOp::Delete { path } => s.push_str(&format!("D {}\n", quote(path))),
            }
        }
        s.push_str("\ndone\n");
        s.into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GitTime;

    #[test]
    fn the_stream_is_exactly_these_bytes() {
        let target = RefName::parse("refs/zen/scratch/1").unwrap();
        let when = GitTime::parse_raw("1700000000 +0100").unwrap();
        let sig = Signature::new("Ada Lovelace", "ada@example.com", when).unwrap();
        let parent = Oid::parse("4b825dc642cb6eb9a060e54bf8d69288fbee4904").unwrap();
        let blob = Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap();
        let ops = [
            FileOp::Modify {
                mode: FileMode::Regular,
                oid: blob.clone(),
                path: RepoPath::parse("dir with space/a\"b.txt").unwrap(),
            },
            FileOp::Modify {
                mode: FileMode::Executable,
                oid: blob,
                path: RepoPath::parse("ünï.sh").unwrap(),
            },
            FileOp::Delete {
                path: RepoPath::parse("old.rs").unwrap(),
            },
        ];
        let stream = CommitStream {
            target: &target,
            author: &sig,
            committer: &sig,
            message: "Fix the tick\n\n\n",
            parent: &parent,
            ops: &ops,
        };
        let expected = "commit refs/zen/scratch/1\n\
author Ada Lovelace <ada@example.com> 1700000000 +0100\n\
committer Ada Lovelace <ada@example.com> 1700000000 +0100\n\
data 13\n\
Fix the tick\n\
\n\
from 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\
M 100644 ce013625030ba8dba906f756967f9e9ca394464a \"dir with space/a\\\"b.txt\"\n\
M 100755 ce013625030ba8dba906f756967f9e9ca394464a \"ünï.sh\"\n\
D \"old.rs\"\n\
\n\
done\n";
        assert_eq!(String::from_utf8(stream.to_bytes()).unwrap(), expected);
    }

    #[test]
    fn messages_end_in_exactly_one_newline() {
        assert_eq!(normalize_message("x"), "x\n");
        assert_eq!(normalize_message("x\n\n"), "x\n");
        assert_eq!(normalize_message("x\n\ny"), "x\n\ny\n");
    }
}
