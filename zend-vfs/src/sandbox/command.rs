//! The command a sandbox run executes.

use std::time::Duration;

/// How long a command runs before it is killed, unless it says otherwise.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// A program and its arguments, run in the repository's folder with no shell
/// in between: each argument reaches the program as written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxCommand {
    /// A program on the `PATH` (`cargo`), or a file in the repository named
    /// by a `/`-separated path relative to it (`./gradlew`, `scripts/check`).
    pub program: String,
    pub args: Vec<String>,
    /// Killed, with everything it started, when it runs longer than this.
    pub timeout: Duration,
}

impl SandboxCommand {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Whether the program is a file in the repository rather than a name
    /// looked up on the `PATH`.
    pub fn is_repository_program(&self) -> bool {
        self.program.contains('/')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_is_built_up_in_order() {
        let c = SandboxCommand::new("cargo")
            .arg("test")
            .args(["-p", "zend-vfs"])
            .timeout(Duration::from_secs(5));
        assert_eq!(c.program, "cargo");
        assert_eq!(c.args, ["test", "-p", "zend-vfs"]);
        assert_eq!(c.timeout, Duration::from_secs(5));
        assert_eq!(SandboxCommand::new("x").timeout, DEFAULT_TIMEOUT);
    }

    #[test]
    fn a_path_names_a_repository_program() {
        assert!(!SandboxCommand::new("cargo").is_repository_program());
        assert!(SandboxCommand::new("./gradlew").is_repository_program());
        assert!(SandboxCommand::new("scripts/check").is_repository_program());
    }
}
