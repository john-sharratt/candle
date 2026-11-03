//! How a task's latest turn went, by the numbers its turn's stream counted:
//! where the time went, its calls by tool, what failed and how, and when it
//! first changed a file and first committed — the close of a task's output
//! (`docs/jobs.md` §3), counted from the turn itself, never summarised by a
//! model.

use super::wait::handle;
use crate::mcp::activity::{duration, Activity};
use crate::mcp::threads::Worker;

/// `when`, an epoch time, as how far into the turn begun at `from` it fell.
fn into_turn(when: Option<u64>, from: u64) -> String {
    when.map_or_else(
        || "never".to_string(),
        |at| format!("{} in", duration(at.saturating_sub(from))),
    )
}

/// One line per tool, most used first: its calls, then its failures.
fn by_tool(activity: &Activity) -> Vec<String> {
    let mut tools: Vec<(&String, &u32)> = activity.tools.iter().collect();
    tools.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    tools
        .into_iter()
        .map(|(tool, calls)| {
            let failed: Vec<String> = activity
                .failed
                .iter()
                .filter_map(|(key, n)| {
                    let how = key.strip_prefix(tool.as_str())?.strip_prefix(' ')?;
                    Some(format!("{how} ×{n}"))
                })
                .collect();
            match failed.is_empty() {
                true => format!("  {tool} {calls}"),
                false => format!("  {tool} {calls}, failed: {}", failed.join(", ")),
            }
        })
        .collect()
}

/// The report on `worker`, read at `now`.
pub fn render(worker: &Worker, now: u64) -> String {
    let path = handle(worker);
    let state = format!("{:?}", worker.state).to_lowercase();
    let Some(activity) = &worker.activity else {
        return format!("{path}: {state}, and no turn has begun.");
    };
    let elapsed = worker
        .ended_ms
        .unwrap_or(now)
        .saturating_sub(worker.turn_ms);
    let calls: u32 = activity.tools.values().sum();
    let failed: u32 = activity.failed.values().sum();
    let mut lines = vec![
        format!(
            "{path}: {state}, its latest turn {}, {} rounds, {} thinking tokens.",
            duration(elapsed),
            activity.rounds,
            activity.thinking_total()
        ),
        format!(
            "Time: {} in tools, {} in the model.",
            duration(activity.tool_ms),
            duration(elapsed.saturating_sub(activity.tool_ms))
        ),
        format!(
            "First change: {}. First commit: {}.",
            into_turn(activity.first_change_ms, worker.turn_ms),
            into_turn(activity.first_commit_ms, worker.turn_ms)
        ),
        format!("{calls} calls, {failed} failed:"),
    ];
    lines.extend(by_tool(activity));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn worker() -> Worker {
        serde_json::from_value(json!({
            "thread": "icons", "name": "regen", "conv_id": "c", "prompt": "p",
            "state": "done", "role": "admin", "priority": "normal",
            "started_ms": 0, "turn_ms": 1_000, "beat_ms": 1_000, "ended_ms": 601_000,
            "activity": null, "no_entry": null,
        }))
        .unwrap()
    }

    /// **A run reads as one table**: time, milestones, and each tool's calls
    /// with how they failed, most used first.
    #[test]
    fn a_run_reads_as_one_table() {
        let mut w = worker();
        w.activity = Some(Activity {
            rounds: 12,
            thinking: 3_000,
            tool_ms: 240_000,
            first_change_ms: Some(301_000),
            tools: [("file_read".to_string(), 3), ("run_command".to_string(), 7)]
                .into_iter()
                .collect(),
            failed: [
                ("run_command exit 101".to_string(), 1),
                ("run_command no_agent".to_string(), 4),
            ]
            .into_iter()
            .collect(),
            ..Activity::default()
        });
        assert_eq!(
            render(&w, 900_000),
            "task:icons/regen: done, its latest turn 10m00s, 12 rounds, 3000 thinking tokens.\n\
             Time: 4m00s in tools, 6m00s in the model.\n\
             First change: 5m00s in. First commit: never.\n\
             10 calls, 5 failed:\n  \
             run_command 7, failed: exit 101 ×1, no_agent ×4\n  \
             file_read 3"
        );
    }

    /// One tool's name inside another's is not its failure.
    #[test]
    fn a_tools_failures_are_its_own() {
        let activity = Activity {
            tools: [("git_log".to_string(), 1), ("git".to_string(), 1)]
                .into_iter()
                .collect(),
            failed: [("git_log refused".to_string(), 1)].into_iter().collect(),
            ..Activity::default()
        };
        assert_eq!(
            by_tool(&activity),
            ["  git 1", "  git_log 1, failed: refused ×1"]
        );
    }

    #[test]
    fn a_worker_with_no_turn_says_so() {
        assert_eq!(
            render(&worker(), 2_000),
            "task:icons/regen: done, and no turn has begun."
        );
    }
}
