//! A job's handle, `<kind>:<id>` (`docs/jobs.md` §1): the one string every
//! answer names a job by and every `job_*` tool takes back.

use std::fmt;

use crate::mcp::ops;
use crate::mcp::tools::repo_job::Handle as AgentJob;

/// A job, by kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobHandle {
    /// A task: a worker of Zen's, by its thread and its name below it.
    Task { thread: String, name: String },
    /// A build or test run on an agent.
    Build(AgentJob),
    /// A deploy, named by its build job.
    Deploy(AgentJob),
    /// A maintenance job on a member.
    Cluster { node: String, n: u64 },
}

/// The kinds, as handles and `job_list` name them.
pub const KINDS: [&str; 4] = ["task", "build", "deploy", "cluster"];

impl JobHandle {
    /// `text` as a handle, or why it is not one.
    pub fn parse(text: &str) -> Result<JobHandle, String> {
        let text = text.trim();
        let bad = |why: &str| {
            format!(
                "`{text}` is not a job: {why}. A job is `task:<thread>/<name>`, \
                 `build:<node>:<id>`, `deploy:<node>:<id>` or `cluster:<node>:<n>`, as the \
                 answer that started it wrote it — `job_list` lists them"
            )
        };
        let (kind, rest) = text.split_once(':').ok_or_else(|| bad("it names no kind"))?;
        match kind {
            "task" => {
                let (thread, name) = rest
                    .split_once('/')
                    .filter(|(t, n)| !t.is_empty() && !n.is_empty())
                    .ok_or_else(|| bad("a task is its thread and its name, `<thread>/<name>`"))?;
                Ok(JobHandle::Task {
                    thread: thread.to_string(),
                    name: name.to_string(),
                })
            }
            "build" | "deploy" => {
                let job = AgentJob::parse(rest).map_err(|_| bad("its id is `<node>:<id>`"))?;
                Ok(if kind == "build" {
                    JobHandle::Build(job)
                } else {
                    JobHandle::Deploy(job)
                })
            }
            "cluster" => {
                let (node, n) = rest
                    .split_once(':')
                    .ok_or_else(|| bad("a maintenance job is `<node>:<n>`"))?;
                let node = ops::node(node).map_err(|_| bad("its node is 16 hex digits"))?;
                let n = n
                    .parse::<u64>()
                    .map_err(|_| bad("its number is the job's id, digits"))?;
                Ok(JobHandle::Cluster { node, n })
            }
            other => Err(bad(&format!("`{other}` is not a kind of job"))),
        }
    }

    /// The kind, as [`KINDS`] names it.
    pub fn kind(&self) -> &'static str {
        match self {
            JobHandle::Task { .. } => "task",
            JobHandle::Build(_) => "build",
            JobHandle::Deploy(_) => "deploy",
            JobHandle::Cluster { .. } => "cluster",
        }
    }
}

impl fmt::Display for JobHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JobHandle::Task { thread, name } => write!(f, "task:{thread}/{name}"),
            JobHandle::Build(j) => write!(f, "build:{j}"),
            JobHandle::Deploy(j) => write!(f, "deploy:{j}"),
            JobHandle::Cluster { node, n } => write!(f, "cluster:{node}:{n}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Each kind reads back as itself**, written as every answer writes it.
    #[test]
    fn each_kind_reads_back_as_itself() {
        for text in [
            "task:review/verify",
            "task:review/verify/read",
            "build:76c3a0f4d39bd9db:XeV_I_QPzpg",
            "deploy:76c3a0f4d39bd9db:XeV_I_QPzpg",
            "cluster:a0ac88178cb2b57d:7",
        ] {
            let h = JobHandle::parse(text).unwrap();
            assert_eq!(h.to_string(), text);
        }
        assert!(JobHandle::parse(" BUILD:76c3a0f4d39bd9db:x ")
            .unwrap_err()
            .contains("`BUILD` is not a kind of job"));
        assert_eq!(
            JobHandle::parse("deploy:76c3a0f4d39bd9db:Xe").unwrap().kind(),
            "deploy"
        );
    }

    /// **A handle missing a part says which part**, and what a handle is.
    #[test]
    fn a_handle_missing_a_part_says_which() {
        for (text, says) in [
            ("XeV_I_QPzpg", "names no kind"),
            ("task:review", "its thread and its name"),
            ("build:76c3:Xe", "`<node>:<id>`"),
            ("cluster:a0ac88178cb2b57d:x", "digits"),
            ("cluster:a0ac:7", "16 hex digits"),
        ] {
            let err = JobHandle::parse(text).unwrap_err();
            assert!(err.contains(says), "{text}: {err}");
            assert!(err.contains("`job_list` lists them"), "{text}: {err}");
        }
    }
}
