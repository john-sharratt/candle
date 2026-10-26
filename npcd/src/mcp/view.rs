//! How a character's mission reads in a tool's answer: its brief, each step
//! and how it went, its answer and report, and what waits behind it.

use crate::engine::bodies::{MissionStatus, MissionView};
use crate::engine::mission::StepOutcome;

/// One mission, a line per part.
pub fn mission(m: &MissionView) -> String {
    let mut out = format!(
        "{} — {}",
        if m.open { "Open" } else { "Finished" },
        m.prompt.trim()
    );
    if let Some(by) = m.origin.get("kind").and_then(|k| k.as_str()) {
        let detail = match by {
            "lodged" => m
                .origin
                .get("by")
                .and_then(|b| b.as_str())
                .map(|b| format!("given by {b}")),
            "random" => m
                .origin
                .get("routine")
                .and_then(|r| r.as_str())
                .map(|r| format!("a routine: {r}")),
            "generated" => Some("written by the command table".to_string()),
            _ => None,
        };
        if let Some(d) = detail {
            out.push_str(&format!(" ({d})"));
        }
    }
    for step in &m.todo {
        let mark = match (step.done, step.outcome) {
            (false, _) => "[ ]",
            (true, Some(StepOutcome::Thwarted)) => "[!]",
            (true, _) => "[x]",
        };
        let how = match step.outcome {
            Some(StepOutcome::Thwarted) if step.done => " — thwarted",
            _ => "",
        };
        out.push_str(&format!("\n  {mark} {}{how}", step.text.trim()));
    }
    if let Some(answer) = m.answer.as_deref().filter(|a| !a.trim().is_empty()) {
        out.push_str(&format!("\nAnswer: {}", answer.trim()));
    }
    if let Some(report) = &m.report {
        out.push_str(&format!(
            "\nReport: {} — {}",
            report.outcome,
            report.notes.trim()
        ));
    }
    out
}

/// What a character is carrying, what it last finished, and how many more
/// are queued for it.
pub fn status(npc: &str, s: &MissionStatus) -> String {
    let mut out = match (&s.mission, s.on_mission) {
        (Some(m), true) => format!("Character {npc} is on a mission.\n{}", mission(m)),
        _ => format!("Character {npc} is not on a mission."),
    };
    let last_shown = s.on_mission.then_some(()).and(s.mission.as_ref());
    if let Some(done) = &s.finished {
        if last_shown != Some(done) {
            out.push_str(&format!("\n\nLast finished:\n{}", mission(done)));
        }
    }
    if s.lodged > 0 {
        out.push_str(&format!(
            "\n\n{} more queued for it at the command table.",
            s.lodged
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::bodies::{ReportView, StepView};
    use serde_json::json;

    fn step(text: &str, done: bool, outcome: Option<StepOutcome>) -> StepView {
        StepView {
            text: text.into(),
            done,
            outcome,
            reports: false,
        }
    }

    fn view(open: bool) -> MissionView {
        MissionView {
            prompt: "Check the coolant valve.".into(),
            open,
            origin: json!({"kind": "lodged", "by": "John"}),
            work: json!(null),
            todo: vec![
                step("go to the plant room", true, Some(StepOutcome::Achieved)),
                step("scan the coolant valve", true, Some(StepOutcome::Thwarted)),
                step("report it", false, None),
            ],
            answer: None,
            observed: json!(null),
            report: None,
        }
    }

    #[test]
    fn a_mission_reads_its_brief_its_steps_and_how_each_went() {
        assert_eq!(
            mission(&view(true)),
            "Open — Check the coolant valve. (given by John)\n  \
             [x] go to the plant room\n  \
             [!] scan the coolant valve — thwarted\n  \
             [ ] report it"
        );
        let mut done = view(false);
        done.answer = Some("It is stuck shut.".into());
        done.report = Some(ReportView {
            outcome: "fail".into(),
            notes: "the scanner was down".into(),
        });
        assert!(mission(&done)
            .ends_with("\nAnswer: It is stuck shut.\nReport: fail — the scanner was down"));
    }

    #[test]
    fn a_status_says_what_is_carried_what_finished_and_what_waits() {
        let s = MissionStatus {
            on_mission: true,
            mission: Some(view(true)),
            finished: Some(view(false)),
            lodged: 2,
        };
        let text = status("1z", &s);
        assert!(text.starts_with("Character 1z is on a mission.\nOpen — "));
        assert!(text.contains("\n\nLast finished:\nFinished — "));
        assert!(text.ends_with("\n\n2 more queued for it at the command table."));
        let idle = MissionStatus {
            on_mission: false,
            mission: Some(view(false)),
            finished: Some(view(false)),
            lodged: 0,
        };
        assert_eq!(
            status("1z", &idle),
            format!(
                "Character 1z is not on a mission.\n\nLast finished:\n{}",
                mission(&view(false))
            )
        );
    }
}
