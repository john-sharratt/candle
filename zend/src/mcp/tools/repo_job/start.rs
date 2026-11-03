//! `repo_job_start {repo, branch, program, args, timeout_s?, stage?, node?}`:
//! run a program on a build agent against a branch as origin holds it
//! (`docs/build_deploy_tools.md` §2). It answers the moment the agent has
//! accepted the job — before origin is fetched — with the job's handle, the
//! machine and what to call next.
//!
//! The job goes over this member's own agent stream, placed as the gateway
//! places any job — sticky on the caller, spilling to an idle agent, low
//! priority last, a standby agent only when no other can — or on the agent
//! `node` names, by its node id or by its machine's name. The program is
//! checked against the agent's sandbox allow-list there; a shell line is
//! refused here, before anything is sent.

use agent::wire::RunJob;
use serde_json::{json, Value};
use zend_vfs::Snapshot;

use super::{host_in, phase, pinned, Handle, NO_AGENTS};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "repo_job_start";

/// The longest a job may run when the call names no `timeout_s`, and the most.
const DEFAULT_TIMEOUT_S: u64 = 1_800;
const MAX_TIMEOUT_S: u64 = 4 * 3_600;

/// The most arguments a job takes.
const MAX_ARGS: usize = 48;

/// Words a shell would read as joining or redirecting commands: an argument
/// that is one of these was meant for a shell, which a job does not have.
const SHELL_WORDS: &[&str] = &["&&", "||", "|", ";", ">", ">>", "<", "2>&1", "&"];

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Start a program — a build, a test run, clippy — on a build agent: a cluster machine \
            that checks out a repository and runs programs on it, never one serving traffic. It runs \
            against `branch` exactly as origin holds it, so push first; nothing of yours that is not pushed \
            is in it. Answers at once, before the agent fetches origin, with the job's handle \
            (`<node>:<id>`), the machine it runs on and its phase; the job then fetches, waits for the \
            checkout if another job holds it, runs, and stages what it built when `stage` asks. \
            Give a program and its arguments, one entry each — `cargo test -p web` is program `cargo`, args \
            `[\"test\", \"-p\", \"web\"]` — never a shell line: there is no shell, so `&&`, `|`, `;` and \
            redirects are refused; run one program per job. The agent runs only programs on its allow-list \
            (`cargo`); a program no agent runs is refused naming those that do. Run the narrowest thing \
            that answers the question — one crate's tests, `cargo check` or `clippy` for a compile — since a \
            workspace-wide test run takes many minutes. On a machine short of memory pass `-j 4` to cargo \
            and `--test-threads=4` after `--`. Then wait for it with `repo_job_status {job, wait_s}` — one \
            call that holds until the job ends, never a loop of calls — and read what it printed with \
            `repo_job_log {job}`, whose default summary gives the test results, errors and failures with \
            their line numbers. Stop it with `repo_job_cancel {job}`. Name `node` (a machine such as \
            `BlackWorld`, or a 16-hex-digit node) to pin an agent; otherwise the gateway places it, keeping \
            a caller's jobs on the agent whose build is warm. A standby agent (AlienWorld) takes a job only \
            when no other agent can, or when `node` names it. To build and try a service, use \
            `service_deploy`, which builds it itself. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The repository, by name: `candle`, `battle-cities`."},
                "branch": {"type": "string", "minLength": 1, "maxLength": 200, "description": "The branch on origin to run on: `main`, or one you pushed."},
                "program": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The program alone, e.g. `cargo` — no arguments, no shell."},
                "args": {"type": "array", "maxItems": MAX_ARGS, "items": {"type": "string", "maxLength": 400}, "description": "Its arguments, one entry each, passed exactly as written: `[\"clippy\", \"-p\", \"web\", \"--tests\", \"--\", \"-D\", \"warnings\"]`."},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_S, "description": "Seconds before it and everything it started are killed; 1800 by default."},
                "stage": {"type": "string", "minLength": 1, "maxLength": 200, "description": "A file it builds, relative to the checkout — `target/release/motherd.exe` — copied to that machine's stage folder once the job exits 0; `repo_job_status` says where."},
                "node": {"type": "string", "minLength": 1, "maxLength": 32, "description": "The agent to run on: its machine (`BlackWorld`) or its 16-hex-digit node. Left out, the gateway places it."}
            },
            "required": ["repo", "branch", "program", "args"],
            "additionalProperties": false
        }
    })
}

/// Why `program` with `args` is a shell line rather than a program and its
/// arguments, if it is one.
pub fn shell_line(program: &str, args: &[String]) -> Option<String> {
    if program.trim().contains(char::is_whitespace) {
        return Some(format!(
            "`program` is the program alone, and `{program}` is a command line: put the program in \
             `program` and each word after it in `args`"
        ));
    }
    args.iter()
        .find(|a| SHELL_WORDS.contains(&a.trim()))
        .map(|w| {
            format!(
                "`{w}` is shell syntax, and a job has no shell: it runs one program with its \
                 arguments. Start one job per program"
            )
        })
}

/// The job `args` asks for.
pub fn job_of(args: &Value) -> Result<RunJob, String> {
    let list = args
        .get("args")
        .and_then(Value::as_array)
        .ok_or("`args` is the program's arguments, a list of strings")?
        .iter()
        .map(|a| a.as_str().map(str::to_string))
        .collect::<Option<Vec<String>>>()
        .ok_or("each of `args` is a string")?;
    if list.len() > MAX_ARGS {
        return Err(format!("`args` holds at most {MAX_ARGS} arguments"));
    }
    let program = string(args, "program")?.to_string();
    if let Some(why) = shell_line(&program, &list) {
        return Err(why);
    }
    Ok(RunJob {
        repo: string(args, "repo")?.to_string(),
        branch: string(args, "branch")?.to_string(),
        overlay: Snapshot::default(),
        program,
        args: list,
        timeout_secs: args
            .get("timeout_s")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_S)
            .clamp(1, MAX_TIMEOUT_S),
        stage: args
            .get("stage")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
    })
}

/// What a caller does next with job `h`.
pub fn next_steps(h: &Handle) -> String {
    format!(
        "Next: `repo_job_status {{job: \"{h}\", wait_s: 180}}` waits for it to end and says how it \
         went; `repo_job_log {{job: \"{h}\"}}` summarises its log — test results, errors, \
         failures, with line numbers; `repo_job_cancel {{job: \"{h}\"}}` stops it."
    )
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let job = match job_of(args) {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let listed = match agents.listing().await {
        Ok(l) => l,
        Err(e) => return Answer::error(format!("the agents could not be listed: {e}")),
    };
    let to = match args.get("node").and_then(Value::as_str) {
        Some(pin) => match pinned(&listed, pin) {
            Ok(node) => Some(node),
            Err(e) => return Answer::error(e),
        },
        None => None,
    };
    let command = format!("{} {}", job.program, job.args.join(" "));
    let place = format!("{}@{}", job.repo, job.branch);
    let (at, view) = match agents.submit(&call.lineage, job, to).await {
        Ok(started) => started,
        Err(e) => return Answer::error(format!("the job could not start: {e}")),
    };
    let node = match agents.node_of(at) {
        Ok(node) => node,
        Err(e) => {
            return Answer::error(format!(
                "`{command}` started as job {}, on an agent whose node could not be read: {e}",
                view.id
            ))
        }
    };
    let h = Handle::new(node, &view.id);
    Answer::ok(format!(
        "Started job {h} on {}: `{command}` on {place} — {}.\n{}",
        host_in(&listed, node),
        phase(&view.phase, node),
        next_steps(&h)
    ))
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_is_a_program_and_its_arguments_on_a_branch() {
        let job = job_of(&json!({
            "repo": "candle", "branch": "fix-x", "program": "cargo",
            "args": ["test", "-p", "zend"], "timeout_s": 99_999,
            "stage": "target/release/motherd.exe"
        }))
        .unwrap();
        assert_eq!(job.stage.as_deref(), Some("target/release/motherd.exe"));
        assert_eq!(
            (job.repo.as_str(), job.branch.as_str(), job.program.as_str()),
            ("candle", "fix-x", "cargo")
        );
        assert_eq!(job.args, ["test", "-p", "zend"]);
        assert_eq!(job.timeout_secs, MAX_TIMEOUT_S);
        assert!(job_of(&json!({"repo": "r", "branch": "b", "program": "p", "args": [1]})).is_err());
        assert!(job_of(&json!({"repo": "r", "branch": "b", "program": "p"})).is_err());
    }

    /// **A shell line is refused before anything is sent**, saying how to
    /// write it instead: a command line in `program`, or a word that joins or
    /// redirects commands among the arguments. A `--` or a `-j 4` is not one.
    #[test]
    fn a_shell_line_is_refused_saying_why() {
        let line = shell_line("cargo test -p web", &[]).unwrap();
        assert!(line.contains("put the program in `program`"), "{line}");
        let joined = shell_line("cargo", &["check".into(), "&&".into(), "cargo".into()]).unwrap();
        assert!(joined.starts_with("`&&` is shell syntax"), "{joined}");
        assert_eq!(
            shell_line(
                "cargo",
                &["test".into(), "-j".into(), "4".into(), "--".into(), "--test-threads=4".into()]
            ),
            None
        );
        let args = json!({"repo": "candle", "branch": "main", "program": "cargo",
                          "args": ["test", "|", "tail"]});
        assert!(job_of(&args).unwrap_err().contains("`|` is shell syntax"));
    }

    #[test]
    fn the_next_steps_name_the_handle() {
        let next = next_steps(&Handle::new(0xab, "Xe"));
        assert!(next.contains(r#"`repo_job_status {job: "00000000000000ab:Xe", wait_s: 180}`"#));
        assert!(next.contains(r#"`repo_job_cancel {job: "00000000000000ab:Xe"}`"#));
    }
}
