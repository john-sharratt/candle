//! The step a coding mission ends its work with: seeing its task through,
//! signed off by the engine when the task reports done — never on the
//! carrier's word (`docs/mission_templates.md` §7).

/// The step's text, as a template writes it.
pub const TASK_STEP: &str = "see the task through";

/// Whether `step` is the task step: its text, trimmed, any case, with any
/// words after it ("see the task through to its answer").
pub fn is_task_step(step: &str) -> bool {
    let lower = step.trim().to_ascii_lowercase();
    lower == TASK_STEP || lower.starts_with(&format!("{TASK_STEP} "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_task_step_is_known_by_its_words() {
        assert!(is_task_step("see the task through"));
        assert!(is_task_step("  See the task through to its answer "));
        assert!(!is_task_step("see the task"));
        assert!(!is_task_step("see the task throughout"));
        assert!(!is_task_step("go to the bench"));
    }
}
