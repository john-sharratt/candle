//! Three-way merges computed in the object store, and commits made from a
//! tree.
//!
//! The merge runs in a private index with plumbing every supported git has:
//!
//! 1. `read-tree -m -i --aggressive <base> <ours> <theirs>` settles every
//!    path only one side changed, both sides changed identically, or either
//!    side deleted unchanged.
//! 2. Each path left unmerged that both sides modified as a regular file is
//!    merged line by line with `merge-file`; any other shape of conflict —
//!    modify/delete, add/add, a mode or type clash, binary content — is a
//!    conflict.
//! 3. The merged files replace their unmerged entries, and `write-tree`
//!    stores the result.
//!
//! No rename detection runs: a file renamed on one side and edited on the
//! other is reported as a conflict rather than merged across the rename.

use std::collections::BTreeMap;

use crate::error::GitError;
use crate::library::objects;
use crate::runner::{utf8, Invocation};
use crate::types::{FileMode, Oid, RepoPath, Signature};
use crate::write::fast_import::normalize_message;
use crate::write::scratch::{PrivateIndex, ScratchFile};
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeOutcome {
    Clean {
        tree: Oid,
    },
    /// Nothing was written; these paths need a human.
    Conflicted {
        paths: Vec<RepoPath>,
    },
}

/// One side of an unmerged path: mode and blob.
type Side = (FileMode, Oid);

/// Parse `ls-files -u -z`: `<mode> <oid> <stage>\t<path>\0`, into each
/// path's base, ours and theirs (stages 1, 2 and 3).
pub(crate) fn parse_unmerged(
    out: &[u8],
) -> Result<BTreeMap<RepoPath, [Option<Side>; 3]>, GitError> {
    let text =
        std::str::from_utf8(out).map_err(|e| GitError::malformed("ls-files", e.to_string()))?;
    let mut paths: BTreeMap<RepoPath, [Option<Side>; 3]> = BTreeMap::new();
    for record in text.split('\0').filter(|r| !r.is_empty()) {
        let bad = || GitError::malformed("ls-files", record.to_string());
        let (meta, path) = record.split_once('\t').ok_or_else(bad)?;
        let parts: Vec<&str> = meta.split(' ').collect();
        let [mode, oid, stage] = parts[..] else {
            return Err(bad());
        };
        let stage: usize = stage.parse().map_err(|_| bad())?;
        if !(1..=3).contains(&stage) {
            return Err(bad());
        }
        paths.entry(RepoPath::parse(path)?).or_default()[stage - 1] =
            Some((FileMode::parse(mode)?, Oid::parse(oid)?));
    }
    Ok(paths)
}

/// A merge taken as far as it goes: every path it settled merged, and each
/// conflicting path left as `ours` holds it — absent where `ours` has none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialMerge {
    pub tree: Oid,
    /// The paths left as `ours` holds them.
    pub conflicts: Vec<RepoPath>,
}

impl Repo {
    /// Merge `theirs` into `ours` over `merge_base`, without a working tree
    /// or the user's index.
    pub fn merge_trees(
        &self,
        merge_base: &Oid,
        ours: &Oid,
        theirs: &Oid,
    ) -> Result<MergeOutcome, GitError> {
        let (tree, conflicts) = self.merge_in_index(merge_base, ours, theirs, false)?;
        Ok(match tree {
            Some(tree) => MergeOutcome::Clean { tree },
            None => MergeOutcome::Conflicted { paths: conflicts },
        })
    }

    /// Merge `theirs` into `ours` over `merge_base` as [`Self::merge_trees`]
    /// does, writing the tree even when paths conflict: each of those is left
    /// as `ours` holds it, for the conflict to be settled elsewhere.
    pub fn merge_trees_keeping_ours(
        &self,
        merge_base: &Oid,
        ours: &Oid,
        theirs: &Oid,
    ) -> Result<PartialMerge, GitError> {
        let (tree, conflicts) = self.merge_in_index(merge_base, ours, theirs, true)?;
        let tree = tree.ok_or_else(|| GitError::malformed("write-tree", "no tree was written"))?;
        Ok(PartialMerge { tree, conflicts })
    }

    /// The merge in a private index: the tree, unless paths conflict and
    /// `keep_ours` is false; and the conflicting paths.
    fn merge_in_index(
        &self,
        merge_base: &Oid,
        ours: &Oid,
        theirs: &Oid,
        keep_ours: bool,
    ) -> Result<(Option<Oid>, Vec<RepoPath>), GitError> {
        let _write = self.write_lock();
        let index = PrivateIndex::new(self.git_dir());
        let in_index = |inv: Invocation| inv.env("GIT_INDEX_FILE", &index.0);

        in_index(self.git("read-tree"))
            .args(["-m", "-i", "--aggressive", "--end-of-options"])
            .args([merge_base.as_str(), ours.as_str(), theirs.as_str()])
            .about_rev(format!("{merge_base} {ours} {theirs}"))
            .run_ok()?;
        let unmerged =
            parse_unmerged(&in_index(self.git("ls-files")).args(["-u", "-z"]).run_ok()?)?;

        // Each path settled at stage 0: merged, or — a conflict, kept as ours
        // — ours' side, or nothing where ours has none.
        let mut settled: Vec<(RepoPath, Option<Side>)> = Vec::new();
        let mut conflicts: Vec<RepoPath> = Vec::new();
        for (path, stages) in unmerged {
            let file = |m: &FileMode| matches!(m, FileMode::Regular | FileMode::Executable);
            let merged = match &stages {
                [Some((bm, bo)), Some((om, oo)), Some((tm, to))]
                    if file(bm) && file(om) && file(tm) =>
                {
                    // The mode one side changed and the other left alone is
                    // taken; both changing it differently is a conflict.
                    let mode = if om == tm || bm == tm {
                        Some(*om)
                    } else if bm == om {
                        Some(*tm)
                    } else {
                        None
                    };
                    match mode {
                        Some(mode) => self.merge_file(bo, oo, to)?.map(|oid| (mode, oid)),
                        None => None,
                    }
                }
                _ => None,
            };
            match merged {
                Some(side) => settled.push((path, Some(side))),
                None => {
                    settled.push((path.clone(), stages[1].clone()));
                    conflicts.push(path);
                }
            }
        }
        if !conflicts.is_empty() && !keep_ours {
            return Ok((None, conflicts));
        }

        if !settled.is_empty() {
            // A mode-0 line drops every stage of the path; the next line, if
            // any, adds the settled file at stage 0.
            let zero = "0".repeat(self.format().hex_len());
            let mut info = Vec::new();
            for (path, side) in &settled {
                info.extend_from_slice(format!("0 {zero}\t{path}\0").as_bytes());
                if let Some((mode, oid)) = side {
                    info.extend_from_slice(format!("{} {oid}\t{path}\0", mode.as_str()).as_bytes());
                }
            }
            in_index(self.git("update-index"))
                .args(["-z", "--index-info"])
                .stdin(info)
                .run_ok()?;
        }
        let tree = in_index(self.git("write-tree")).run_ok()?;
        let tree = Oid::parse(utf8("write-tree", tree)?.trim())?;
        Ok((Some(tree), conflicts))
    }

    /// Merge three versions of a file line by line: the merged blob, or
    /// `None` when the changes overlap or the content is binary.
    fn merge_file(&self, base: &Oid, ours: &Oid, theirs: &Oid) -> Result<Option<Oid>, GitError> {
        // A blob the library does not hold is one a partial clone has yet to
        // fetch, which `git` does on demand.
        let blob = |oid: &Oid| {
            if let Some(lib) = self.library() {
                if let Some(bytes) = objects::blob_bytes(&lib, oid)? {
                    return Ok(bytes);
                }
            }
            self.git("cat-file")
                .args(["blob", "--end-of-options"])
                .arg(oid.as_str())
                .run_ok()
        };
        let base_file = ScratchFile::new(self.git_dir(), "merge-base", &blob(base)?)?;
        let ours_file = ScratchFile::new(self.git_dir(), "merge-ours", &blob(ours)?)?;
        let theirs_file = ScratchFile::new(self.git_dir(), "merge-theirs", &blob(theirs)?)?;
        // Exit 0: clean. Positive: the number of conflicts. Negative: an
        // error, binary content among them. Only a clean merge is used.
        let out = self
            .git("merge-file")
            .args(["-p", "--end-of-options"])
            .arg(&ours_file.0)
            .arg(&base_file.0)
            .arg(&theirs_file.0)
            .run()?;
        if out.status != Some(0) {
            return Ok(None);
        }
        if let Some(lib) = self.library() {
            return Ok(Some(objects::hash_raw(&lib, &out.stdout)?));
        }
        let oid = self
            .git("hash-object")
            .args(["-w", "--stdin", "--no-filters"])
            .stdin(out.stdout)
            .run_ok()?;
        Ok(Some(Oid::parse(utf8("hash-object", oid)?.trim())?))
    }

    /// Write a commit of `tree` with `parents`, in order. Moves no ref.
    pub fn commit_tree(
        &self,
        tree: &Oid,
        parents: &[&Oid],
        message: &str,
        author: &Signature,
        committer: &Signature,
    ) -> Result<Oid, GitError> {
        let _write = self.write_lock();
        if let Some(lib) = self.library() {
            return objects::commit_tree(
                &lib,
                tree,
                parents,
                &normalize_message(message),
                author,
                committer,
            );
        }
        let date = |s: &Signature| format!("@{}", s.when.to_raw());
        let mut inv = self.git("commit-tree");
        for p in parents {
            inv = inv.arg("-p").arg(p.as_str());
        }
        let out = inv
            .args(["-F", "-", "--end-of-options"])
            .arg(tree.as_str())
            .env("GIT_AUTHOR_NAME", author.name())
            .env("GIT_AUTHOR_EMAIL", author.email())
            .env("GIT_AUTHOR_DATE", date(author))
            .env("GIT_COMMITTER_NAME", committer.name())
            .env("GIT_COMMITTER_EMAIL", committer.email())
            .env("GIT_COMMITTER_DATE", date(committer))
            .stdin(normalize_message(message).into_bytes())
            .run_ok()?;
        Oid::parse(utf8("commit-tree", out)?.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;
    use crate::types::{GitTime, Rev};

    #[test]
    fn unmerged_entries_parse_into_their_stages() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let b = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
        let out = format!(
            "100644 {a} 1\tf.rs\0100644 {b} 2\tf.rs\0100755 {a} 3\tf.rs\0100644 {b} 2\tgone in theirs.rs\0"
        );
        let u = parse_unmerged(out.as_bytes()).unwrap();
        let f = &u[&RepoPath::parse("f.rs").unwrap()];
        assert_eq!(f[0], Some((FileMode::Regular, Oid::parse(a).unwrap())));
        assert_eq!(f[1], Some((FileMode::Regular, Oid::parse(b).unwrap())));
        assert_eq!(f[2], Some((FileMode::Executable, Oid::parse(a).unwrap())));
        let g = &u[&RepoPath::parse("gone in theirs.rs").unwrap()];
        assert_eq!(
            (g[0].is_none(), g[1].is_some(), g[2].is_none()),
            (true, true, true)
        );
        assert!(parse_unmerged(format!("100644 {a} 0\tx\0").as_bytes()).is_err());
    }

    /// **Edits to different lines of one file merge line by line**, to the
    /// same tree `git merge` produces.
    #[test]
    fn edits_to_different_lines_of_one_file_merge_like_git_merge() {
        let t = TestRepo::init();
        let lines: Vec<String> = (1..=12).map(|i| format!("line {i}\n")).collect();
        t.write("f.txt", lines.concat().as_bytes());
        let base = t.commit_all("base");
        let mut ours_lines = lines.clone();
        ours_lines[1] = "OURS 2\n".into();
        t.write("f.txt", ours_lines.concat().as_bytes());
        let ours = t.commit_all("ours");
        t.git(&["checkout", "-q", "-b", "theirs", base.as_str()]);
        let mut theirs_lines = lines.clone();
        theirs_lines[10] = "THEIRS 11\n".into();
        t.write("f.txt", theirs_lines.concat().as_bytes());
        let theirs = t.commit_all("theirs");
        let repo = t.repo();

        let tree = match repo.merge_trees(&base, &ours, &theirs).unwrap() {
            MergeOutcome::Clean { tree } => tree,
            other => panic!("{other:?}"),
        };
        t.git(&["checkout", "-q", "-b", "oracle", ours.as_str()]);
        t.git(&["merge", "-q", "--no-edit", "theirs"]);
        assert_eq!(tree, t.oid("HEAD^{tree}"));
    }

    #[test]
    fn modify_delete_and_binary_clashes_are_conflicts() {
        let t = TestRepo::init();
        t.write("edited.txt", b"base\n");
        t.write("bin.dat", &[0, 1, 2, 0]);
        let base = t.commit_all("base");
        t.write("edited.txt", b"ours\n");
        t.write("bin.dat", &[0, 9, 9, 0]);
        let ours = t.commit_all("ours");
        t.git(&["checkout", "-q", "-b", "theirs", base.as_str()]);
        std::fs::remove_file(t.path.join("edited.txt")).unwrap();
        t.write("bin.dat", &[0, 7, 7, 0]);
        let theirs = t.commit_all("theirs");
        match t.repo().merge_trees(&base, &ours, &theirs).unwrap() {
            MergeOutcome::Conflicted { paths } => assert_eq!(
                paths,
                vec![
                    RepoPath::parse("bin.dat").unwrap(),
                    RepoPath::parse("edited.txt").unwrap()
                ]
            ),
            other => panic!("{other:?}"),
        }
        let leftovers = std::fs::read_dir(t.path.join(".git"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("zen-"))
            .count();
        assert_eq!(leftovers, 0, "scratch files are removed");
    }

    /// base → ours edits `a`, theirs edits `b` (clean); theirs2 edits `a`
    /// differently (conflict).
    #[test]
    fn disjoint_changes_merge_cleanly_and_overlapping_ones_conflict() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.write("b.txt", b"b\n");
        let base = t.commit_all("base");
        t.write("a.txt", b"ours\n");
        let ours = t.commit_all("ours");
        t.git(&["checkout", "-q", "-b", "theirs", base.as_str()]);
        t.write("b.txt", b"theirs\n");
        let theirs = t.commit_all("theirs");
        t.git(&["checkout", "-q", "-b", "theirs2", base.as_str()]);
        t.write("a.txt", b"conflicting\n");
        let theirs2 = t.commit_all("theirs2");
        let repo = t.repo();

        match repo.merge_trees(&base, &ours, &theirs).unwrap() {
            MergeOutcome::Clean { tree } => {
                let blobs = repo.blobs();
                let when = GitTime::parse_raw("1700000000 +0000").unwrap();
                let sig = Signature::new("Ada", "ada@example.com", when).unwrap();
                let merged = repo
                    .commit_tree(&tree, &[&ours, &theirs], "merge", &sig, &sig)
                    .unwrap();
                let at = |p: &str| {
                    blobs
                        .read_at(&Rev::Oid(merged.clone()), &RepoPath::parse(p).unwrap())
                        .unwrap()
                        .unwrap()
                };
                assert_eq!(at("a.txt"), b"ours\n");
                assert_eq!(at("b.txt"), b"theirs\n");
                let parents = t.git(&["rev-list", "--parents", "-n", "1", merged.as_str()]);
                assert_eq!(
                    parents.trim(),
                    format!("{merged} {ours} {theirs}"),
                    "both parents, in order"
                );
            }
            other => panic!("{other:?}"),
        }
        match repo.merge_trees(&base, &ours, &theirs2).unwrap() {
            MergeOutcome::Conflicted { paths } => {
                assert_eq!(paths, vec![RepoPath::parse("a.txt").unwrap()]);
            }
            other => panic!("{other:?}"),
        }
    }

    /// **A mode one side changed is kept when the other side edited the
    /// file**: the edit and the executable bit both land.
    #[test]
    fn a_mode_change_meets_an_edit_and_both_land() {
        let t = TestRepo::init();
        let lines: String = (1..=6).map(|i| format!("line {i}\n")).collect();
        t.write("run.sh", lines.as_bytes());
        let base = t.commit_all("base");
        t.write("run.sh", lines.replace("line 1\n", "ours\n").as_bytes());
        let ours = t.commit_all("ours");
        t.git(&["checkout", "-q", "-b", "theirs", base.as_str()]);
        t.git(&["update-index", "--chmod=+x", "run.sh"]);
        t.git(&["commit", "-q", "-m", "executable"]);
        let theirs = t.oid("HEAD");

        let tree = match t.repo().merge_trees(&base, &ours, &theirs).unwrap() {
            MergeOutcome::Clean { tree } => tree,
            other => panic!("{other:?}"),
        };
        let listed = t.git(&["ls-tree", tree.as_str(), "run.sh"]);
        assert!(listed.starts_with("100755 "), "{listed}");
        assert_eq!(
            t.git(&["show", &format!("{tree}:run.sh")]),
            lines.replace("line 1\n", "ours\n")
        );
    }

    /// **Kept as ours, a conflicting merge still writes its tree**: every
    /// settled path merged, each conflicting one as ours holds it — a file
    /// ours deleted stays deleted.
    #[test]
    fn a_merge_keeping_ours_writes_the_settled_tree() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.write("b.txt", b"b\n");
        t.write("gone.txt", b"g\n");
        let base = t.commit_all("base");
        t.write("a.txt", b"ours\n");
        std::fs::remove_file(t.path.join("gone.txt")).unwrap();
        let ours = t.commit_all("ours");
        t.git(&["checkout", "-q", "-b", "theirs", base.as_str()]);
        t.write("a.txt", b"theirs\n");
        t.write("b.txt", b"theirs b\n");
        t.write("gone.txt", b"theirs g\n");
        let theirs = t.commit_all("theirs");

        let merged = t
            .repo()
            .merge_trees_keeping_ours(&base, &ours, &theirs)
            .unwrap();
        assert_eq!(
            merged.conflicts,
            vec![
                RepoPath::parse("a.txt").unwrap(),
                RepoPath::parse("gone.txt").unwrap()
            ]
        );
        let listed = t.git(&["ls-tree", "--name-only", merged.tree.as_str()]);
        assert_eq!(listed, "a.txt\nb.txt\n");
        let show = |p: &str| t.git(&["show", &format!("{}:{p}", merged.tree)]);
        assert_eq!(show("a.txt"), "ours\n");
        assert_eq!(show("b.txt"), "theirs b\n");
    }
}
