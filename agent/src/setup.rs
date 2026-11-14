//! A repository's setup (§11.4): the commands an agent runs on a checkout when
//! it starts, before any job on that checkout — the tools the repository's jobs
//! need, which no machine's agent file has to name. The repository declares
//! them in [`SETUP_FILE`], on the branch the checkout has checked out:
//!
//! ```yaml
//! commands:
//!   # rustup installs the toolchain rust-toolchain.toml pins, if it is missing
//!   - program: cargo
//!     args: [--version]
//!   - program: cargo
//!     args: [install, cargo-nextest, --locked]
//!     timeout_s: 3600       # optional; SETUP_TIMEOUT_S when absent
//! ```
//!
//! Each is a program and its arguments, with no shell — exactly a run job's
//! `program` and `args` — and runs under the same command policy. A file that is
//! missing or empty declares no commands.

use serde::Deserialize;

/// Where a repository declares its setup, relative to its top level.
pub const SETUP_FILE: &str = ".agent/setup.yaml";

/// The longest a setup command runs when it names no `timeout_s`: an install
/// can build from source.
pub const SETUP_TIMEOUT_S: u64 = 3600;

/// One command a repository's setup runs.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetupCommand {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub timeout_s: Option<u64>,
}

impl SetupCommand {
    /// How long it may run.
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_s.unwrap_or(SETUP_TIMEOUT_S)
    }

    /// Its command line, as a job's is listed.
    pub fn line(&self) -> Vec<String> {
        std::iter::once(self.program.clone())
            .chain(self.args.iter().cloned())
            .collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetupFile {
    #[serde(default)]
    commands: Vec<SetupCommand>,
}

/// The commands `text`, a [`SETUP_FILE`], declares, in order.
pub fn parse_setup(text: &str) -> anyhow::Result<Vec<SetupCommand>> {
    let Some(file) = serde_yaml::from_str::<Option<SetupFile>>(text)? else {
        return Ok(Vec::new());
    };
    if let Some(i) = file
        .commands
        .iter()
        .position(|c| c.program.trim().is_empty())
    {
        anyhow::bail!("setup command {} names no program", i + 1);
    }
    Ok(file.commands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_lists_its_commands_in_order() {
        let commands = parse_setup(
            "commands:\n  - program: cargo\n    args: [--version]\n  - program: cargo\n    args: [install, cargo-nextest, --locked]\n    timeout_s: 1800\n",
        )
        .unwrap();
        assert_eq!(
            commands,
            [
                SetupCommand {
                    program: "cargo".into(),
                    args: vec!["--version".into()],
                    timeout_s: None,
                },
                SetupCommand {
                    program: "cargo".into(),
                    args: vec!["install".into(), "cargo-nextest".into(), "--locked".into()],
                    timeout_s: Some(1800),
                },
            ]
        );
        assert_eq!(commands[0].timeout_secs(), 3600);
        assert_eq!(commands[1].timeout_secs(), 1800);
        assert_eq!(
            commands[1].line(),
            ["cargo", "install", "cargo-nextest", "--locked"]
        );
    }

    #[test]
    fn an_empty_file_or_one_of_comments_declares_nothing() {
        assert_eq!(parse_setup("").unwrap(), []);
        assert_eq!(parse_setup("# nothing yet\n").unwrap(), []);
        assert_eq!(parse_setup("commands: []\n").unwrap(), []);
        assert_eq!(parse_setup("{}\n").unwrap(), []);
    }

    #[test]
    fn unknown_fields_and_a_missing_program_are_refused() {
        assert!(parse_setup("commands:\n  - program: cargo\n    shell: bash\n").is_err());
        assert!(parse_setup("steps: []\n").is_err());
        assert!(parse_setup("commands:\n  - args: [build]\n").is_err());
        let blank = parse_setup("commands:\n  - program: cargo\n  - program: ' '\n").unwrap_err();
        assert_eq!(blank.to_string(), "setup command 2 names no program");
    }
}
