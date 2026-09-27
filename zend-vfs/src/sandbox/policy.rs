//! The check a command passes before it runs on a repository's checkout.
//!
//! It judges the command line, not the program. A program that is allowed
//! runs with the daemon's own rights and can do anything they permit; what this
//! decides is which programs may be started at all, and that nothing on the
//! command line points the program outside the repository or into what the
//! file tools may never touch. Every rule:
//!
//! - **The program is named by the policy.** An allow-list, never a deny-list:
//!   a program nobody listed does not run. A program in the repository
//!   (`./gradlew`) is listed by that spelling, and must be a plain file inside
//!   the repository with no link on the way to it — checked on the checkout
//!   the command runs on, with the conversation's changes already laid down.
//! - **Never git directly.** Git as the program, or as a command in a shell's
//!   script, is refused with an error that sends the caller to the `git_*`
//!   tools ([`git_use`](super::git_use)) — checked first, listed or not, so
//!   that is the answer whatever else is wrong with the command.
//! - **No absolute path, and no path out of the repository.** Every argument
//!   is read as a path, and so is every value glued into one — after each `=`
//!   (`--out=../x`), after a short switch's letter (`-o../x`, `-I/etc`), and
//!   after a switch's first `:` (`/out:C:\x`). An absolute one (`/etc/x`,
//!   `C:\x`, `\\host\share`, `~/x`) or a relative one whose `..` climbs above
//!   the repository's folder is refused. On Windows a whole argument of a
//!   leading `/` followed by no further separator is a switch (`/C`,
//!   `/nologo`), not a path, and passes.
//! - **Nothing protected.** An argument with a `.git` or `secrets` component —
//!   the segments the file tools refuse ([`PROTECTED_SEGMENTS`]) — is refused
//!   here too, so a command cannot read what `file_read` would not.
//! - **No NUL**, which no program argument can carry.
//!
//! A shell, or any interpreter, on the allow-list runs whatever script it is
//! given: past the git check, its script is one argument checked as a path,
//! not parsed as a program. Listing one is allowing everything the daemon's
//! rights allow.

use std::collections::BTreeSet;
use std::path::Path;

use thiserror::Error;

use super::command::SandboxCommand;
use super::git_use::direct_git;
use crate::checkout::target;
use crate::vfs::PROTECTED_SEGMENTS;

/// Why a command was not run.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Refused {
    #[error("no program was named")]
    NoProgram,
    #[error("a command cannot carry a NUL character")]
    Nul,
    #[error("{program} is not a program this repository's sandbox runs")]
    NotAllowed { program: String },
    /// The command runs git directly: `command` is the part that does, as
    /// the caller wrote it.
    #[error(
        "`{command}` runs git directly, which commands here may not do; use the git tools \
         instead — git_status, git_log, git_show, git_grep and git_refs to read a \
         repository, git_commit, git_ref, git_fetch and git_push to change it"
    )]
    Git { command: String },
    #[error("{program}: {why}")]
    Program { program: String, why: String },
    #[error("{arg} is an absolute path; name files relative to the repository")]
    Absolute { arg: String },
    #[error("{arg} leads outside the repository")]
    Escapes { arg: String },
    #[error("{arg} names a protected folder (.git/ or secrets/), which commands may not touch")]
    Protected { arg: String },
}

/// Which programs a repository's sandbox may start.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandPolicy {
    programs: BTreeSet<String>,
}

impl CommandPolicy {
    /// A policy starting exactly `programs`, each spelled as a command names
    /// it: `cargo`, `./gradlew`.
    pub fn allowing(programs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            programs: programs.into_iter().map(Into::into).collect(),
        }
    }

    /// Whether `command` may run in the repository whose checkout is at
    /// `root`: [`Self::check_command`], then [`Self::check_on_checkout`].
    /// See the module for every rule.
    pub fn check(&self, root: &Path, command: &SandboxCommand) -> Result<(), Refused> {
        self.check_command(command)?;
        self.check_on_checkout(root, command)
    }

    /// Every rule that reads the command line alone — all of them but a
    /// repository program's existence. Needs no checkout, so a run makes it
    /// before touching one.
    pub fn check_command(&self, command: &SandboxCommand) -> Result<(), Refused> {
        let program = &command.program;
        if program.is_empty() {
            return Err(Refused::NoProgram);
        }
        if program.contains('\0') || command.args.iter().any(|a| a.contains('\0')) {
            return Err(Refused::Nul);
        }
        if let Some(command) = direct_git(command) {
            return Err(Refused::Git { command });
        }
        if !self.programs.contains(program) {
            return Err(Refused::NotAllowed {
                program: program.clone(),
            });
        }
        if program.contains('\\') {
            return Err(Refused::Program {
                program: program.clone(),
                why: "a program in the repository is named with `/`".into(),
            });
        }
        if command.is_repository_program() {
            check_path(program, program, true)?;
        }
        for arg in &command.args {
            check_path(arg, arg, true)?;
            for value in glued_values(arg) {
                check_path(arg, value, false)?;
            }
        }
        Ok(())
    }

    /// The rule that reads the checkout: a program in the repository is a
    /// plain file inside it, reached through no link — checked where the
    /// command will run, with the conversation's changes laid down.
    pub fn check_on_checkout(&self, root: &Path, command: &SandboxCommand) -> Result<(), Refused> {
        if !command.is_repository_program() {
            return Ok(());
        }
        let program = &command.program;
        let refuse = |why: String| Refused::Program {
            program: program.clone(),
            why,
        };
        let rel = program.strip_prefix("./").unwrap_or(program);
        let found = target::resolve(root, rel).map_err(|e| refuse(e.to_string()))?;
        if !found.abs.is_file() {
            return Err(refuse("there is no such file in the repository".into()));
        }
        Ok(())
    }
}

/// The paths an argument can carry past its own start: the text after every
/// `=` (`--out=../x`, `--a=b=../x`), after a short switch's letter
/// (`-o../x`, `-I/etc`), and after the first `:` of a switch (`-o:../x`,
/// `/out:C:\x`).
fn glued_values(arg: &str) -> Vec<&str> {
    let mut values: Vec<&str> = arg
        .match_indices('=')
        .map(|(at, _)| &arg[at + 1..])
        .collect();
    if arg.starts_with('-') && !arg.starts_with("--") {
        if let Some((at, _)) = arg.char_indices().nth(2) {
            values.push(&arg[at..]);
        }
    }
    if arg.starts_with(['-', '/']) {
        if let Some((_, value)) = arg.split_once(':') {
            values.push(value);
        }
    }
    values
}

/// `value`, read as a path, stays inside the repository and out of its
/// protected folders. `arg` is what a refusal names; `whole` is whether
/// `value` is the whole argument, which alone may be a Windows switch.
fn check_path(arg: &str, value: &str, whole: bool) -> Result<(), Refused> {
    if is_absolute(value, whole) {
        return Err(Refused::Absolute { arg: arg.into() });
    }
    let mut depth: usize = 0;
    for component in value.split(['/', '\\']) {
        if PROTECTED_SEGMENTS
            .iter()
            .any(|p| component.eq_ignore_ascii_case(p))
        {
            return Err(Refused::Protected { arg: arg.into() });
        }
        match component {
            "" | "." => {}
            ".." => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| Refused::Escapes { arg: arg.into() })?;
            }
            _ => depth += 1,
        }
    }
    Ok(())
}

fn is_absolute(value: &str, whole: bool) -> bool {
    let bytes = value.as_bytes();
    let drive = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    let rooted = match value.strip_prefix('/') {
        // A Windows switch is a whole argument of a `/` and a word; a rooted
        // path has a separator after its first component. A value glued into
        // an argument is never a switch.
        Some(rest) if cfg!(windows) && whole => rest.contains(['/', '\\']),
        Some(_) => true,
        None => false,
    };
    drive || rooted || value.starts_with('\\') || value.starts_with('~')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> CommandPolicy {
        CommandPolicy::allowing(["cargo", "./build.sh", "tools/gen", "git", "sh"])
    }

    fn check(root: &Path, program: &str, args: &[&str]) -> Result<(), Refused> {
        policy().check(
            root,
            &SandboxCommand::new(program).args(args.iter().copied()),
        )
    }

    fn root() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("build.sh"), b"#!/bin/sh\n").unwrap();
        std::fs::create_dir_all(dir.path().join("tools")).unwrap();
        std::fs::write(dir.path().join("tools/gen"), b"#!/bin/sh\n").unwrap();
        dir
    }

    /// **A listed program with arguments inside the repository runs**, and so
    /// does a listed program in the repository.
    #[test]
    fn a_listed_program_with_repository_arguments_passes() {
        let dir = root();
        let r = dir.path();
        assert_eq!(
            check(r, "cargo", &["test", "-p", "app", "--release"]),
            Ok(())
        );
        assert_eq!(
            check(r, "cargo", &["build", "--manifest-path=sub/Cargo.toml"]),
            Ok(())
        );
        assert_eq!(check(r, "cargo", &["run", "--", "a/../b.txt"]), Ok(()));
        assert_eq!(check(r, "./build.sh", &["out/x"]), Ok(()));
        assert_eq!(check(r, "tools/gen", &[]), Ok(()));
        assert_eq!(check(r, "sh", &["-c", "echo hi > out.txt"]), Ok(()));
    }

    /// **Only what the policy lists runs** — by exact spelling.
    #[test]
    fn an_unlisted_program_is_refused() {
        let dir = root();
        for program in ["python", "Cargo", "./cargo", "build.sh"] {
            assert_eq!(
                check(dir.path(), program, &[]),
                Err(Refused::NotAllowed {
                    program: program.into()
                }),
                "{program}"
            );
        }
        assert_eq!(
            CommandPolicy::default().check(dir.path(), &SandboxCommand::new("cargo")),
            Err(Refused::NotAllowed {
                program: "cargo".into()
            })
        );
        assert_eq!(check(dir.path(), "", &[]), Err(Refused::NoProgram));
    }

    /// **git run directly is refused, listed or not, with an answer that
    /// sends the caller to the git tools** — as the program, or as a command
    /// in a listed shell's script. The word `git` that runs no git passes.
    #[test]
    fn git_run_directly_is_refused_towards_the_git_tools() {
        let dir = root();
        let r = dir.path();
        assert_eq!(
            check(r, "git", &["status"]),
            Err(Refused::Git {
                command: "git status".into()
            })
        );
        assert_eq!(
            check(r, "sh", &["-c", "cargo build && git commit -am wip"]),
            Err(Refused::Git {
                command: "git commit -am wip".into()
            })
        );
        let why = check(r, "git", &["log"]).unwrap_err().to_string();
        assert!(why.starts_with("`git log` runs git directly"), "{why}");
        for tool in [
            "git_status",
            "git_log",
            "git_show",
            "git_commit",
            "git_push",
        ] {
            assert!(why.contains(tool), "{why}");
        }

        assert_eq!(check(r, "sh", &["-c", "grep -rn git src"]), Ok(()));
        assert_eq!(check(r, "cargo", &["add", "git2"]), Ok(()));
        assert_eq!(check(r, "sh", &["-c", "echo \"x; git push\""]), Ok(()));
    }

    /// **An argument that leaves the repository is refused** — absolute in
    /// any spelling, climbing out with `..`, or hidden behind `--name=`.
    #[test]
    fn a_path_out_of_the_repository_is_refused() {
        let dir = root();
        let r = dir.path();
        for arg in [
            "C:\\Windows\\win.ini",
            "c:/x",
            "\\\\host\\share",
            "\\rooted",
            "~/.ssh/id_rsa",
            "/etc/passwd",
            "--manifest-path=/elsewhere/Cargo.toml",
        ] {
            assert_eq!(
                check(r, "cargo", &[arg]),
                Err(Refused::Absolute { arg: arg.into() }),
                "{arg}"
            );
        }
        for arg in ["..", "../x", "a/../../x", "a\\..\\..\\x", "--out=../x"] {
            assert_eq!(
                check(r, "cargo", &[arg]),
                Err(Refused::Escapes { arg: arg.into() }),
                "{arg}"
            );
        }
    }

    /// On Windows a `/` switch is not a path; elsewhere every leading `/` is.
    #[test]
    fn a_windows_switch_is_not_a_path() {
        let dir = root();
        let switch = check(
            dir.path(),
            "cargo",
            &["/C", "/nologo", "/p:Configuration=Release"],
        );
        if cfg!(windows) {
            assert_eq!(switch, Ok(()));
        } else {
            assert!(matches!(switch, Err(Refused::Absolute { .. })));
        }
    }

    /// **The protected folders stay out of reach** of a command as of the
    /// file tools, in any case.
    #[test]
    fn a_protected_folder_is_refused() {
        let dir = root();
        for arg in [
            ".git/config",
            "sub/.GIT/hooks",
            "web/secrets/auth.yaml",
            "--f=Secrets/x",
        ] {
            assert_eq!(
                check(dir.path(), "cargo", &[arg]),
                Err(Refused::Protected { arg: arg.into() }),
                "{arg}"
            );
        }
        assert_eq!(check(dir.path(), "cargo", &["secretsauce.txt"]), Ok(()));
    }

    /// **A repository program must be a plain file inside the repository.**
    #[test]
    fn a_repository_program_must_be_a_file_there() {
        let dir = root();
        let listed = |program: &str| {
            CommandPolicy::allowing([program]).check(dir.path(), &SandboxCommand::new(program))
        };
        assert!(matches!(
            listed("./absent.sh"),
            Err(Refused::Program { .. })
        ));
        assert!(
            matches!(listed("tools"), Ok(())),
            "a bare name is looked up on the PATH"
        );
        assert!(
            matches!(listed("./tools"), Err(Refused::Program { .. })),
            "a folder"
        );
        assert!(matches!(
            listed("../outside.sh"),
            Err(Refused::Escapes { .. })
        ));
        assert!(matches!(
            listed(".git/hooks/x"),
            Err(Refused::Protected { .. })
        ));
        assert!(matches!(listed("tools\\gen"), Err(Refused::Program { .. })));
    }

    /// **A path glued to a switch is read as the path it is** — after every
    /// `=`, after a short switch's letter, after a `name:` — while an ordinary
    /// switch value passes.
    #[test]
    fn a_path_glued_to_a_switch_is_checked() {
        let dir = root();
        let r = dir.path();
        for arg in ["--a=b=../x", "-o../x", "-O..", "-x=../y"] {
            assert_eq!(
                check(r, "cargo", &[arg]),
                Err(Refused::Escapes { arg: arg.into() }),
                "{arg}"
            );
        }
        for arg in ["-I/etc", "-IC:\\x", "-Lc:/lib", "--out=\\\\host\\share"] {
            assert_eq!(
                check(r, "cargo", &[arg]),
                Err(Refused::Absolute { arg: arg.into() }),
                "{arg}"
            );
        }
        for arg in ["-o:../x", "/out:..\\x", "/out:C:\\x", "-f:.git/config"] {
            assert!(check(r, "cargo", &[arg]).is_err(), "{arg}");
        }
        for arg in [
            "-j4",
            "--jobs=4",
            "-Dfoo=bar",
            "-p",
            "-é",
            "-éx/y",
            "--",
            "-",
        ] {
            assert_eq!(check(r, "cargo", &[arg]), Ok(()), "{arg}");
        }
    }

    #[test]
    fn a_nul_is_refused() {
        let dir = root();
        assert_eq!(check(dir.path(), "cargo", &["a\0b"]), Err(Refused::Nul));
    }
}
