//! Finding the file a program name on the `PATH` starts.
//!
//! Windows starts a program named without an extension only when it is an
//! `.exe`: `npm`, `npx`, `yarn` and many other toolchain entry points are
//! `.cmd` scripts beside it, and a command naming them would fail to start
//! though the same line works in any shell. So a bare name is looked up the
//! way the shell looks it up — each `PATH` folder in order, each `PATHEXT`
//! extension in order — and the file found is what is started. The standard
//! library runs a `.cmd` or `.bat` file named by its path through the command
//! interpreter with its arguments quoted for it.
//!
//! Elsewhere a name is started as it is: the system's own `PATH` search is the
//! shell's.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// The extensions tried when `PATHEXT` is not set — Windows' own default.
const DEFAULT_PATHEXT: &str = ".COM;.EXE;.BAT;.CMD";

/// What to start for `name`, a program looked up on the `PATH`.
pub(crate) fn program(name: &str) -> OsString {
    if !cfg!(windows) {
        return name.into();
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| DEFAULT_PATHEXT.to_string());
    match find(&path, &pathext, name) {
        Some(found) => found.into_os_string(),
        None => name.into(),
    }
}

/// The first `folder/name` + extension that is a file, over each folder in
/// `path` and each extension in `pathext`. A name that already ends in one of
/// those extensions, or names a folder, is left to the system. A dot that
/// ends in no program extension is part of the name: `python3.11` is looked
/// up as `python3.11.exe`.
fn find(path: &OsStr, pathext: &str, name: &str) -> Option<PathBuf> {
    let extensions: Vec<&str> = pathext.split(';').filter(|e| !e.is_empty()).collect();
    let has_one = Path::new(name).extension().is_some_and(|own| {
        extensions.iter().any(|ext| {
            ext.trim_start_matches('.')
                .eq_ignore_ascii_case(&own.to_string_lossy())
        })
    });
    if name.contains(['/', '\\']) || has_one {
        return None;
    }
    std::env::split_paths(path).find_map(|folder| {
        extensions.iter().find_map(|ext| {
            // `PATHEXT` is upper case; toolchains install their scripts in
            // lower case, and the name started is the one the log shows.
            let candidate = folder.join(format!("{name}{}", ext.to_ascii_lowercase()));
            candidate.is_file().then_some(candidate)
        })
    })
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn folder_with(files: &[&str]) -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        for file in files {
            std::fs::write(dir.path().join(file), b"").unwrap();
        }
        dir
    }

    fn joined(folders: &[&Path]) -> OsString {
        std::env::join_paths(folders).unwrap()
    }

    /// **`npm` is found as the `npm.cmd` beside it**, in `PATHEXT` order,
    /// and the first folder holding one wins.
    #[test]
    fn a_bare_name_is_found_by_extension_in_path_order() {
        let first = folder_with(&["npm.cmd"]);
        let second = folder_with(&["npm.exe"]);
        let path = joined(&[first.path(), second.path()]);
        assert_eq!(
            find(&path, ".COM;.EXE;.BAT;.CMD", "npm"),
            Some(first.path().join("npm.cmd"))
        );
        let both = folder_with(&["tool.cmd", "tool.exe"]);
        assert_eq!(
            find(&joined(&[both.path()]), ".EXE;.CMD", "tool"),
            Some(both.path().join("tool.exe")),
            "PATHEXT order decides within a folder"
        );
    }

    /// A name with no file, one already carrying an extension, and one with a
    /// separator are left to the system to start or refuse.
    #[test]
    fn what_is_not_found_is_left_as_named() {
        let dir = folder_with(&["npm.cmd"]);
        let path = joined(&[dir.path()]);
        assert_eq!(find(&path, DEFAULT_PATHEXT, "cargo"), None);
        assert_eq!(find(&path, DEFAULT_PATHEXT, "npm.cmd"), None);
        assert_eq!(find(&path, DEFAULT_PATHEXT, "bin/npm"), None);
        assert_eq!(find(&path, "", "npm"), None, "no extension to try");
    }

    /// **A dot that is not a program extension is part of the name**:
    /// `python3.11` is found as `python3.11.exe`.
    #[test]
    fn a_dotted_name_is_still_looked_up() {
        let dir = folder_with(&["python3.11.exe"]);
        assert_eq!(
            find(&joined(&[dir.path()]), DEFAULT_PATHEXT, "python3.11"),
            Some(dir.path().join("python3.11.exe"))
        );
    }

    /// A folder named like a program is not the program.
    #[test]
    fn a_folder_is_not_a_program() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("npm.cmd")).unwrap();
        assert_eq!(find(&joined(&[dir.path()]), ".CMD", "npm"), None);
    }
}
