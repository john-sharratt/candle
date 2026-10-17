//! Three versions of a file merged line by line in this process: libgit2's
//! `git_merge_file`, the same xdiff merge `git merge-file` runs, over buffers —
//! no repository, no scratch files.

use std::ffi::CString;
use std::os::raw::c_char;
use std::{mem, slice};

use libgit2_sys as raw;

use crate::error::GitError;

/// A merge's result: the text, and whether it merged with no overlap. A
/// merge of binary content never does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Merged {
    pub(crate) clean: bool,
    pub(crate) bytes: Vec<u8>,
}

/// The labels conflict markers name each side by.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Labels<'a> {
    pub(crate) ours: &'a str,
    pub(crate) base: &'a str,
    pub(crate) theirs: &'a str,
}

fn input(bytes: &[u8]) -> Result<raw::git_merge_file_input, GitError> {
    // SAFETY: an all-zero input is a valid one for the init to fill.
    let mut input: raw::git_merge_file_input = unsafe { mem::zeroed() };
    let rc = unsafe { raw::git_merge_file_input_init(&mut input, 1) };
    if rc < 0 {
        return Err(GitError::malformed(
            "merge-file",
            "libgit2 refused a merge input",
        ));
    }
    input.ptr = bytes.as_ptr() as *const c_char;
    input.size = bytes.len();
    Ok(input)
}

fn label(text: &str) -> Result<CString, GitError> {
    CString::new(text).map_err(|_| GitError::invalid("a merge label holding NUL"))
}

/// Merge `ours` and `theirs`, each changed from `base`, line by line.
pub(crate) fn merge_file(
    base: &[u8],
    ours: &[u8],
    theirs: &[u8],
    labels: Option<Labels<'_>>,
) -> Result<Merged, GitError> {
    raw::init();
    let (base_in, ours_in, theirs_in) = (input(base)?, input(ours)?, input(theirs)?);
    let names = match labels {
        Some(l) => Some((label(l.base)?, label(l.ours)?, label(l.theirs)?)),
        None => None,
    };
    // SAFETY: zeroed then initialised by libgit2; the label strings outlive
    // the call.
    let mut options: raw::git_merge_file_options = unsafe { mem::zeroed() };
    if unsafe { raw::git_merge_file_options_init(&mut options, 1) } < 0 {
        return Err(GitError::malformed(
            "merge-file",
            "libgit2 refused merge options",
        ));
    }
    if let Some((b, o, t)) = &names {
        options.ancestor_label = b.as_ptr();
        options.our_label = o.as_ptr();
        options.their_label = t.as_ptr();
    }
    // SAFETY: every input points at bytes alive for the call; the result is
    // copied out and freed once.
    unsafe {
        let mut result: raw::git_merge_file_result = mem::zeroed();
        let rc = raw::git_merge_file(&mut result, &base_in, &ours_in, &theirs_in, &options);
        if rc < 0 {
            let detail = git2::Error::last_error(rc).message().to_string();
            return Err(GitError::malformed("merge-file", detail));
        }
        let bytes = if result.ptr.is_null() {
            Vec::new()
        } else {
            slice::from_raw_parts(result.ptr as *const u8, result.len).to_vec()
        };
        let clean = result.automergeable != 0;
        raw::git_merge_file_result_free(&mut result);
        Ok(Merged { clean, bytes })
    }
}

/// How many conflicts `merged` holds: lines that open a conflict, git's
/// seven `<` and a space or the line's end.
pub(crate) fn conflicts_in(merged: &[u8]) -> usize {
    merged
        .split(|&b| b == b'\n')
        .filter(|line| {
            line.starts_with(b"<<<<<<<") && matches!(line.get(7), None | Some(b' ') | Some(b'\r'))
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Overlapping changes are kept between labelled markers**, byte for byte
    /// what `git merge-file -L yours -L base -L origin/main` writes.
    #[test]
    fn overlapping_changes_are_marked_as_git_marks_them() {
        let labels = Labels {
            ours: "yours",
            base: "base",
            theirs: "origin/main",
        };
        let merged = merge_file(
            b"a\nb\nc\n",
            b"a\nmine\nc\n",
            b"a\ntheirs\nc\n",
            Some(labels),
        )
        .unwrap();
        assert!(!merged.clean);
        assert_eq!(
            merged.bytes,
            b"a\n<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> origin/main\nc\n"
        );
        assert_eq!(conflicts_in(&merged.bytes), 1);
    }

    #[test]
    fn changes_to_different_lines_merge_cleanly() {
        let base: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        let ours = base.replace("line 2\n", "OURS\n");
        let theirs = base.replace("line 9\n", "THEIRS\n");
        let merged = merge_file(base.as_bytes(), ours.as_bytes(), theirs.as_bytes(), None).unwrap();
        assert!(merged.clean);
        assert_eq!(
            merged.bytes,
            base.replace("line 2\n", "OURS\n")
                .replace("line 9\n", "THEIRS\n")
                .into_bytes()
        );
        assert_eq!(conflicts_in(&merged.bytes), 0);
    }

    #[test]
    fn binary_content_does_not_merge() {
        let merged = merge_file(&[0, 1, 2, 0], &[0, 9, 9, 0], &[0, 7, 7, 0], None).unwrap();
        assert!(!merged.clean);
    }
}
