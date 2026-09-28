//! The programs `run_command` may start in a repository's sandbox.
//!
//! The sandbox's policy is an allow-list (`zend_vfs::sandbox::policy`): a
//! program nobody listed does not run. This deployment lists the build, test
//! and packaging toolchains a coding conversation runs — each by the name it
//! is started by, looked up on the `PATH` (on Windows through `PATHEXT`, so the
//! `.cmd` wrappers `npm` and `npx` install as start too).
//!
//! **What the list does and does not bound.** It decides which programs
//! start; it does not bound what they do. Several listed programs run
//! whatever they are handed — `python -c`, `node -e`, a `make` target, an
//! `npm` script, an `npx` package fetched on the spot — so a command can do
//! anything the daemon's own rights allow, git and the network included, and
//! the policy's argument checks (no path out of the repository, no `.git/`)
//! see only the command line, never what a script inside it does. The real
//! boundary is who may call `run_command` at all: it needs `Exec`, `DiskWrite`
//! and `Network`, which only the Comprehensive tools mode grants, and only an
//! admin may use Comprehensive (`crate::access`). The list keeps a model's
//! ordinary calls to the toolchains, and names them in a refusal so the next
//! call can be one.
//!
//! No shell is listed: a command is one program, never a script line.

use zend_vfs::CommandPolicy;

/// The programs a sandbox job may start, by the name a command gives them:
/// JavaScript and TypeScript, Rust, Python, Go, .NET, the JVM, then native
/// builds.
pub const PROGRAMS: &[&str] = &[
    "node", "npm", "npx", "yarn", "pnpm", "bun", "deno", "tsc", "cargo", "rustc", "rustfmt",
    "python", "python3", "py", "pip", "pytest", "go", "dotnet", "java", "javac", "mvn", "gradle",
    "make", "cmake", "ctest", "ninja",
];

/// The sandbox policy starting exactly [`PROGRAMS`].
pub fn policy() -> CommandPolicy {
    CommandPolicy::allowing(PROGRAMS.iter().copied())
}

#[cfg(test)]
mod tests {
    use zend_vfs::SandboxCommand;

    use super::*;

    /// **The toolchains run; a shell and git do not.**
    #[test]
    fn toolchains_run_and_shells_do_not() {
        let policy = policy();
        let check = |program: &str| policy.check_command(&SandboxCommand::new(program));
        for program in ["npm", "node", "cargo", "python", "pytest"] {
            assert_eq!(check(program), Ok(()), "{program}");
        }
        for program in ["cmd", "sh", "bash", "powershell", "pwsh", "curl", "git"] {
            assert!(check(program).is_err(), "{program} runs");
        }
    }
}
