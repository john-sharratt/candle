//! A git patch's paths spelled so libgit2's parser reads them.
//!
//! Git writes a path holding a space as it stands — `rename from old name.rs`,
//! `diff --git a/old name.rs b/new name.rs` — and libgit2 reads an unquoted
//! path only up to its first space, so such a patch fails to parse with
//! "trailing data". Git also reads a path in C quotes, and so does libgit2:
//! every header path holding whitespace is rewritten quoted, taken from the
//! header lines that state it unambiguously — `rename`/`copy`, then `---` and
//! `+++` — and, for a section with neither, from the `diff --git` line's two
//! halves, which are then the same path.

/// `patch` with every header path that holds whitespace quoted.
pub(crate) fn quote_spaced_paths(patch: &[u8]) -> Vec<u8> {
    let lines: Vec<&[u8]> = patch.split_inclusive(|&b| b == b'\n').collect();
    let mut out = Vec::with_capacity(patch.len() + 64);
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let Some(rest) = line.strip_prefix(b"diff --git ") else {
            // Before the first file: nothing a parser reads as a path.
            out.extend_from_slice(line);
            i += 1;
            continue;
        };
        // This file's section: until the next `diff --git` line.
        let end = (i + 1..lines.len())
            .find(|&j| lines[j].starts_with(b"diff --git "))
            .unwrap_or(lines.len());
        let section = &lines[i + 1..end];
        let (old, new) = section_paths(section, trimmed(rest));
        match (old, new) {
            (Some(old), Some(new)) if needs_quotes(&old) || needs_quotes(&new) => {
                out.extend_from_slice(b"diff --git ");
                out.extend_from_slice(&quote(&[b"a/", old.as_slice()].concat()));
                out.push(b' ');
                out.extend_from_slice(&quote(&[b"b/", new.as_slice()].concat()));
                out.extend_from_slice(ending(line));
            }
            _ => out.extend_from_slice(line),
        }
        // Header lines up to the first hunk or binary patch; what follows is
        // content, where a removed line can begin `--- ` and is no header.
        let headers = section
            .iter()
            .position(|l| l.starts_with(b"@@") || l.starts_with(b"GIT binary patch"))
            .unwrap_or(section.len());
        for header in &section[..headers] {
            out.extend_from_slice(&rewrite_header(header));
        }
        for content in &section[headers..] {
            out.extend_from_slice(content);
        }
        i = end;
    }
    out
}

/// A line without its line ending.
fn trimmed(line: &[u8]) -> &[u8] {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// A line's ending: what [`trimmed`] took off.
fn ending(line: &[u8]) -> &[u8] {
    &line[trimmed(line).len()..]
}

/// The section's old and new paths, unprefixed and unquoted.
fn section_paths(section: &[&[u8]], diff_line: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let (mut old, mut new) = (None, None);
    for line in section {
        let line = trimmed(line);
        if line.starts_with(b"@@") || line.starts_with(b"GIT binary patch") {
            break;
        }
        // Each header that states a path: which side, and the prefix its path
        // carries.
        const STATED: [(&[u8], bool, &[u8]); 6] = [
            (b"rename from ", true, b""),
            (b"copy from ", true, b""),
            (b"rename to ", false, b""),
            (b"copy to ", false, b""),
            (b"--- ", true, b"a/"),
            (b"+++ ", false, b"b/"),
        ];
        for (prefix, is_old, strip) in STATED {
            let Some(path) = line.strip_prefix(prefix) else {
                continue;
            };
            let path = path.strip_suffix(b"\t").unwrap_or(path);
            if path.starts_with(b"\"") || path == b"/dev/null" {
                continue;
            }
            let slot = if is_old { &mut old } else { &mut new };
            if slot.is_none() {
                *slot = Some(path.strip_prefix(strip).unwrap_or(path).to_vec());
            }
        }
    }
    if old.is_none() && new.is_none() && !diff_line.contains(&b'"') {
        // `a/P b/P`: both halves the same path.
        let len = diff_line.len();
        if len > 5 && (len - 5).is_multiple_of(2) {
            let half = (len - 5) / 2;
            let (a, b) = (&diff_line[..half + 2], &diff_line[half + 3..]);
            if a.starts_with(b"a/")
                && b.starts_with(b"b/")
                && a[2..] == b[2..]
                && diff_line[half + 2] == b' '
            {
                return (Some(a[2..].to_vec()), Some(b[2..].to_vec()));
            }
        }
    }
    let (old, new) = match (old, new) {
        (Some(o), None) => (Some(o.clone()), Some(o)),
        (None, Some(n)) => (Some(n.clone()), Some(n)),
        pair => pair,
    };
    (old, new)
}

/// One header line with its path quoted when it holds whitespace.
fn rewrite_header(line: &[u8]) -> Vec<u8> {
    let body = trimmed(line);
    for prefix in [
        &b"rename from "[..],
        b"rename to ",
        b"copy from ",
        b"copy to ",
        b"--- ",
        b"+++ ",
    ] {
        if let Some(path) = body.strip_prefix(prefix) {
            let path = path.strip_suffix(b"\t").unwrap_or(path);
            if path.starts_with(b"\"") || !needs_quotes(path) {
                return line.to_vec();
            }
            let mut out = prefix.to_vec();
            out.extend_from_slice(&quote(path));
            out.extend_from_slice(ending(line));
            return out;
        }
    }
    line.to_vec()
}

fn needs_quotes(path: &[u8]) -> bool {
    path.iter()
        .any(|b| b.is_ascii_whitespace() || *b == b'"' || *b == b'\\')
}

/// `path` in git's C quotes.
fn quote(path: &[u8]) -> Vec<u8> {
    let mut out = vec![b'"'];
    for &b in path {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\t' => out.extend_from_slice(b"\\t"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            other => out.push(other),
        }
    }
    out.push(b'"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A rename of a path with spaces is quoted throughout**: the `diff
    /// --git` line from the rename lines, and the rename lines themselves.
    #[test]
    fn a_spaced_rename_is_quoted() {
        let patch = b"diff --git a/old name.rs b/new name.rs\n\
similarity index 100%\n\
rename from old name.rs\n\
rename to new name.rs\n";
        assert_eq!(
            String::from_utf8(quote_spaced_paths(patch)).unwrap(),
            "diff --git \"a/old name.rs\" \"b/new name.rs\"\n\
similarity index 100%\n\
rename from \"old name.rs\"\n\
rename to \"new name.rs\"\n"
        );
    }

    /// **An edit's paths come from its `---`/`+++` lines**, a trailing tab
    /// dropped; a binary change's from the `diff --git` line's equal halves.
    #[test]
    fn an_edit_and_a_binary_change_are_quoted() {
        let patch = b"diff --git a/my file.txt b/my file.txt\n\
index 1..2 100644\n\
--- a/my file.txt\t\n\
+++ b/my file.txt\t\n\
@@ -1 +1 @@\n\
-a\n\
+b\n\
diff --git a/big data.bin b/big data.bin\n\
index 3..4 100644\n\
GIT binary patch\n";
        assert_eq!(
            String::from_utf8(quote_spaced_paths(patch)).unwrap(),
            "diff --git \"a/my file.txt\" \"b/my file.txt\"\n\
index 1..2 100644\n\
--- \"a/my file.txt\"\n\
+++ \"b/my file.txt\"\n\
@@ -1 +1 @@\n\
-a\n\
+b\n\
diff --git \"a/big data.bin\" \"b/big data.bin\"\n\
index 3..4 100644\n\
GIT binary patch\n"
        );
    }

    /// **A patch with no spaced path is left byte for byte**, as is a hunk's
    /// content however it looks: a removed line `-- a/not a header` reads
    /// `--- a/not a header`, and is content, not a header.
    #[test]
    fn a_patch_with_no_spaced_path_is_unchanged() {
        let patch = b"diff --git a/f.txt b/f.txt\n\
--- a/f.txt\n\
+++ b/f.txt\n\
@@ -1 +1 @@\n\
--- a/not a header\n\
+++ b/neither\n";
        assert_eq!(quote_spaced_paths(patch), patch.to_vec());
    }
}
