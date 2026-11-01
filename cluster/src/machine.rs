//! The host a member or an agent runs on, as `CAPACITY` reports it (§5.3,
//! §11.2): its operating system, logical cores, the one-minute load average,
//! free memory, and which of a list of programs are found on its `PATH`.

use std::env::{consts, split_paths, var_os};
use std::ffi::OsString;
use std::path::Path;
use std::thread::available_parallelism;

use sysinfo::System;

/// One reading of the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Machine {
    pub os: String,
    pub cores: u16,
    /// The one-minute load average × 1000; zero where the platform keeps
    /// none.
    pub load_milli: u32,
    pub mem_free_mib: u32,
}

/// Read the host now.
pub fn measure() -> Machine {
    let mut sys = System::new();
    sys.refresh_memory();
    Machine {
        os: consts::OS.to_string(),
        cores: available_parallelism()
            .map_or(1, |n| n.get())
            .min(usize::from(u16::MAX)) as u16,
        load_milli: (System::load_average().one * 1000.0) as u32,
        mem_free_mib: (sys.available_memory() >> 20).min(u64::from(u32::MAX)) as u32,
    }
}

/// The endings an executable may carry on this platform.
fn endings() -> &'static [&'static str] {
    if cfg!(windows) {
        &["", ".exe", ".cmd", ".bat"]
    } else {
        &[""]
    }
}

/// Which of `programs` are found in the directories of `path`, in the order
/// given.
pub fn found_in(path: &OsString, programs: &[&str]) -> Vec<String> {
    let dirs: Vec<_> = split_paths(path).filter(|d| !is_store_stubs(d)).collect();
    programs
        .iter()
        .filter(|p| {
            dirs.iter().any(|d| {
                endings()
                    .iter()
                    .any(|e| is_file(&d.join(format!("{p}{e}"))))
            })
        })
        .map(|p| p.to_string())
        .collect()
}

/// Whether `dir` is Windows' folder of app-execution aliases. Its `python.exe`
/// is a stub that only offers the Microsoft Store: advertised as a toolchain, a
/// job went to a machine with no Python and exited 9009, and a worker tried it
/// again and again.
fn is_store_stubs(dir: &Path) -> bool {
    dir.file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case("WindowsApps"))
}

fn is_file(p: &Path) -> bool {
    p.metadata().is_ok_and(|m| m.is_file())
}

/// Which of `programs` this process finds on its `PATH`.
pub fn toolchains(programs: &[&str]) -> Vec<String> {
    var_os("PATH").map_or_else(Vec::new, |path| found_in(&path, programs))
}

#[cfg(test)]
mod tests {
    use std::env::join_paths;
    use std::fs;

    use super::*;

    /// A program is found in any directory of the path, with or without the
    /// platform's executable ending, and the order given is kept.
    #[test]
    fn programs_are_found_on_the_path() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        fs::write(a.path().join("cargo"), b"").unwrap();
        let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
        fs::write(b.path().join(npm), b"").unwrap();
        fs::create_dir(b.path().join("go")).unwrap();
        let path = join_paths([a.path(), b.path()]).unwrap();
        assert_eq!(
            found_in(&path, &["npm", "go", "cargo", "make"]),
            vec!["npm".to_string(), "cargo".to_string()]
        );
    }

    /// **A Store alias is not a toolchain**: the real program elsewhere on the
    /// path still is.
    #[test]
    fn the_store_aliases_are_not_toolchains() {
        let (root, real) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let stubs = root.path().join("Microsoft").join("WindowsApps");
        fs::create_dir_all(&stubs).unwrap();
        fs::write(stubs.join("python"), b"").unwrap();
        fs::write(stubs.join("node"), b"").unwrap();
        fs::write(real.path().join("node"), b"").unwrap();
        let path = join_paths([stubs.as_path(), real.path()]).unwrap();
        assert_eq!(
            found_in(&path, &["python", "node"]),
            vec!["node".to_string()]
        );
    }

    #[test]
    fn a_reading_names_the_platform_and_its_cores() {
        let m = measure();
        assert_eq!(m.os, consts::OS);
        assert!(m.cores >= 1);
    }
}
