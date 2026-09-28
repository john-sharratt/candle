//! `run_command` — run a program in a repository's sandbox and wait for it.

use std::path::Path;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{FileDelta, JobRequest, Rev};

use super::RunError;
use crate::exec;
use crate::sandboxes::{page, LogPage, SandboxesError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct RunRequest {
    /// The repository to run in. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The program to start, by name (`npm`, `node`, `cargo`, `python`). The program alone — no shell, so no `&&`, pipes or redirects; a script in the repository is an argument to its interpreter. Required.
    #[validate(length(min = 1))]
    pub program: String,
    /// Its arguments, each passed exactly as written, with no shell quoting or globbing: `["test"]` for `npm test`, `["test", "--", "--test-name-pattern", "lowStock"]`, `[]` for none. Required.
    pub args: Vec<String>,
    /// Seconds before the program and everything it started are killed. Optional; 600 by default, at most 1800.
    #[serde(default)]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 1800))]
    pub timeout_secs: Option<u32>,
}

/// A file the program changed, now among the conversation's changes.
#[derive(Serialize)]
pub struct Changed {
    pub path: String,
    /// `edited`, `written` (created or rewritten whole) or `deleted`.
    pub change: &'static str,
}

/// A file the program changed that the conversation's changes could not hold.
#[derive(Serialize)]
pub struct NotKept {
    pub path: String,
    pub why: String,
}

#[derive(Serialize)]
pub struct RunResponse {
    pub repo: String,
    /// This run's id — what `run_output` reads the rest of its output by.
    pub job: String,
    /// The program's exit code; `null` when it was killed.
    pub exit_code: Option<i32>,
    /// Whether it was killed for running past its timeout.
    pub timed_out: bool,
    /// What it changed, now among your uncommitted changes, in path order.
    pub changed: Vec<Changed>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub not_kept: Vec<NotKept>,
    /// The first page of what it printed, both streams in the order printed.
    pub output: LogPage,
    /// The program printed more than the log keeps (64 MiB); the rest was
    /// counted, not kept.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub output_cut: bool,
}

pub struct RunCommand;

impl Tool for RunCommand {
    const NAME: &'static str = "run_command";
    const DESCRIPTION: &'static str =
        "Run a program — a build, a test suite, a formatter, a script — in the named repository, \
         on your branch with your uncommitted changes laid down, and wait for it to finish. \
         `program` is the program alone (`npm`, `node`, `cargo`, `python`) and `args` its \
         arguments, each passed as written — `npm test` is program `npm` with args `[\"test\"]`, \
         and a script in the repository runs through its interpreter (`node` with \
         `[\"scripts/report.js\"]`). There is no shell, so no `&&`, pipes or redirects. \
         Files the program changes come back into your uncommitted changes. git cannot be run \
         here; use the git tools. Use for: running the tests after an edit, \
         building, formatting, running a script. Triggered by \"run the tests\", \"does it \
         build\", \"npm test\", \"cargo check\". Returns the exit code, the files it changed, \
         and the first page of its output; read further pages with run_output.";

    type Request = RunRequest;
    type Response = RunResponse;
    type Error = RunError;

    fn run(ctx: &ToolContext, req: RunRequest) -> Result<RunResponse, RunError> {
        let (program, args) = take_apart(req.program, req.args)?;
        let sandboxes = ctx.sandboxes()?.ok_or(RunError::NoSandboxes)?;
        let files = ctx.files.repo(&req.repo)?;
        sandboxes.check(&req.repo)?;
        let branch = match files.rev() {
            Some(Rev::Branch(branch)) => branch,
            _ => return Err(RunError::NoBranch(req.repo)),
        };
        let grant = ctx.grants().disk_write()?;
        let mut command = exec::sandbox_command(ctx.grants(), &program)?.args(args);
        if let Some(secs) = req.timeout_secs {
            command = command.timeout(Duration::from_secs(secs.into()));
        }
        let ran = sandboxes.run(
            grant,
            &req.repo,
            JobRequest {
                branch,
                files,
                command,
            },
        )?;
        let output = page::read(&ran.log, 0).map_err(SandboxesError::Log)?;
        let outcome = ran.outcome;
        Ok(RunResponse {
            repo: req.repo,
            job: ran.job.to_string(),
            exit_code: outcome.exit_code,
            timed_out: outcome.timed_out,
            changed: outcome
                .changed
                .into_iter()
                .map(|c| Changed {
                    change: change_of(&c.delta.delta),
                    path: c.path,
                })
                .collect(),
            not_kept: outcome
                .unrecorded
                .into_iter()
                .map(|u| NotKept {
                    path: u.path,
                    why: u.why,
                })
                .collect(),
            output,
            output_cut: outcome.output.truncated,
        })
    }
}

/// A `program` holding whitespace is a command line — a program and its
/// arguments in one string — which no program is named. Refused before
/// anything runs, with the call it should have been: measured live, a model
/// sent `"program": "npm test"`, was told only that no such program runs, and
/// gave up on running the tests at all. A program named with a space in its
/// own name cannot be run this way, and none is: the sandbox's policy lists
/// toolchains by their plain names.
///
/// **git is the exception**: a git command line is taken apart and passed on,
/// so the sandbox's policy refuses it towards the git tools — the refusal that
/// says what to do. Taken apart only to be refused again, it cost a round:
/// measured live, `"program": "git log -1"` was told to call again with
/// program `git`, which would only have been refused as git.
///
/// The arguments, split out or given, are checked for shell syntax before
/// anything else ([`refuse_shell_syntax`]), so a chained command line is told
/// about the chain rather than handed back a split that is refused next. A
/// command line with quotes in it is refused without a split: taken apart at
/// its spaces, `--grep "low stock"` would come back as two arguments carrying
/// the quotes.
fn take_apart(program: String, args: Vec<String>) -> Result<(String, Vec<String>), RunError> {
    let program = program.trim().to_string();
    let mut words = program.split_whitespace();
    let Some(first) = words.next().map(str::to_string) else {
        return Ok((program, args));
    };
    let rest: Vec<String> = words
        .map(str::to_string)
        .chain(args.iter().cloned())
        .collect();
    refuse_shell_syntax(&rest)?;
    if rest.len() == args.len() {
        return Ok((program, args));
    }
    if is_git(&first) {
        return Ok((first, rest));
    }
    if program.contains(['"', '\'']) {
        return Err(RunError::Shell(format!(
            "`{program}` is a command line with quotes in it, and a command here runs with no \
             shell to read them: put the program alone in `program` and each argument in \
             `args` as the program should receive it, without the quotes — one entry per \
             argument, spaces and all"
        )));
    }
    Err(RunError::CommandLine {
        given: program,
        program: first,
        args: rest,
    })
}

/// Arguments a shell would have read and no program means to receive. Not
/// `;`, which `find -exec` takes as its own argument.
const SHELL_OPERATORS: [&str; 7] = ["&&", "||", "|", "&", ">", ">>", "2>&1"];

/// What opens a redirect written against its target (`>out.txt`,
/// `2>/dev/null`, `&>log`, `<in`).
const REDIRECTS: [&str; 5] = [">", "<", "1>", "2>", "&>"];

/// Whether `arg` is a shell operator or a redirect.
fn is_shell_operator(arg: &str) -> bool {
    SHELL_OPERATORS.contains(&arg) || REDIRECTS.iter().any(|r| arg.starts_with(r))
}

/// An argument that is shell syntax is refused with the call it should have
/// been: with no shell between, `&&` and `2>&1` reach the program as
/// arguments, and a `\"` as a backslash and a quote. Measured live, all three:
/// `npm test 2>&1` failed on a file named `2>&1`; `npm pkg set … && cat
/// package.json` handed npm a `&&`; and `scripts.lint=\"node --check …\"`
/// would have set the script to its own quotes.
fn refuse_shell_syntax(args: &[String]) -> Result<(), RunError> {
    if let Some(op) = args.iter().find(|a| is_shell_operator(a)) {
        return Err(RunError::Shell(format!(
            "`{op}` is shell syntax, and a command here runs with no shell: it would reach the \
             program as an argument. Run one program per call, with only its own arguments; \
             both output streams come back together already"
        )));
    }
    // Where a value opens with one — the whole argument, or after `=`. Inside
    // code (`node -e 'console.log("a \"b\"")'`) an escaped quote is the code's.
    let shell_quoted = |a: &&String| a.starts_with("\\\"") || a.contains("=\\\"");
    if let Some(quoted) = args.iter().find(shell_quoted) {
        let meant = quoted.replace("\\\"", "");
        return Err(RunError::Shell(format!(
            "`{quoted}` carries `\\\"`, a shell's escaped quote, and a command here runs with \
             no shell: the program would get the backslash and the quote. Pass each argument \
             as the program should receive it — here {meant:?} — one argument, spaces and all"
        )));
    }
    Ok(())
}

/// Whether `program` names git, by name or by path, with or without `.exe`.
fn is_git(program: &str) -> bool {
    Path::new(program)
        .file_stem()
        .is_some_and(|stem| stem.eq_ignore_ascii_case("git"))
}

/// What a delta did to its file, as one word.
fn change_of(delta: &FileDelta) -> &'static str {
    match delta {
        FileDelta::Edit { .. } => "edited",
        FileDelta::Replace { .. } | FileDelta::ReplaceBinary { .. } => "written",
        FileDelta::Delete => "deleted",
    }
}

pub const RUN_COMMAND: RegisteredTool = RegisteredTool::new::<RunCommand>();

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    /// A program alone, trimmed, passes; a command line is taken apart into
    /// the call it should have been; git's is passed on to the git refusal.
    #[test]
    fn command_lines_are_taken_apart() {
        assert_eq!(
            take_apart("npm ".into(), strings(&["test"])).unwrap(),
            ("npm".to_string(), strings(&["test"]))
        );
        match take_apart("npm run lint".into(), strings(&["--", "-q"])) {
            Err(RunError::CommandLine {
                given,
                program,
                args,
            }) => {
                assert_eq!(given, "npm run lint");
                assert_eq!(program, "npm");
                assert_eq!(args, strings(&["run", "lint", "--", "-q"]));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            take_apart("git log -1".into(), vec![]).unwrap(),
            ("git".to_string(), strings(&["log", "-1"]))
        );
    }

    /// **A quoted command line is refused without a split**, and a chained
    /// one is told about the chain first.
    #[test]
    fn quoted_and_chained_command_lines_are_refused_as_such() {
        let quoted = take_apart(r#"npm test -- --grep "low stock""#.into(), vec![]);
        assert!(
            matches!(&quoted, Err(RunError::Shell(why)) if why.contains("with quotes in it")),
            "{quoted:?}"
        );
        let chained = take_apart("npm test && npm run lint".into(), vec![]);
        assert!(
            matches!(&chained, Err(RunError::Shell(why)) if why.starts_with("`&&` is shell syntax")),
            "{chained:?}"
        );
    }

    /// Every operator and every redirect written against its target is shell
    /// syntax; an argument that merely contains one is not.
    #[test]
    fn shell_operators_and_redirects_are_recognised() {
        for arg in [
            "&&",
            "||",
            "|",
            "&",
            ">",
            ">>",
            "2>&1",
            ">out.txt",
            "2>/dev/null",
            "2>nul",
            "&>log",
            "<in",
        ] {
            assert!(is_shell_operator(arg), "{arg}");
        }
        for arg in ["--grep=a|b", "a>b", "-x", "--", ";", "echo x > y"] {
            assert!(!is_shell_operator(arg), "{arg}");
        }
    }

    /// A shell-quoted value is refused with the argument as it should arrive;
    /// an escaped quote inside code is the code's own.
    #[test]
    fn shell_quoted_values_are_refused_and_code_is_not() {
        let quoted = refuse_shell_syntax(&strings(&["scripts.lint=\\\"node --check x.js\\\""]));
        assert!(
            matches!(&quoted, Err(RunError::Shell(why)) if why.contains(r#"here "scripts.lint=node --check x.js""#)),
            "{quoted:?}"
        );
        assert!(refuse_shell_syntax(&strings(&["\\\"x\\\""])).is_err());
        assert!(refuse_shell_syntax(&strings(&["-e", "console.log(\"a \\\"b\\\"\")"])).is_ok());
    }

    #[test]
    fn git_is_recognised_by_name_and_path() {
        for program in ["git", "GIT.EXE", "git.exe", "/usr/bin/git"] {
            assert!(is_git(program), "{program}");
        }
        for program in ["gitk", "legit", "npm"] {
            assert!(!is_git(program), "{program}");
        }
    }
}
