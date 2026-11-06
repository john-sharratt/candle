//! What a task was asked to do, step by step, and whether its answer says of
//! each what became of it (`docs/zen_coding_challenge_fixes.md` Cause 3).
//!
//! A brief's steps are its own numbered or bulleted lines. Two tasks ended
//! with an answer that read as complete while steps they were given went
//! undone — one verified a change and ran nothing, one skipped the review and
//! the wider tests it was told to run — and their starter found out only by
//! reading the call listing. So a task given two or more steps answers with a
//! `Steps:` section, a line per step by its number saying `done`, `not done`
//! or `could not` and why; an answer that leaves a step out is sent back once
//! to account for it, as an answer citing what it never opened is
//! (`super::audit`).

/// The steps of `brief`: each of its lines that opens, unindented, with a
/// number (`1.`, `2)`) or a bullet (`-`, `*`, `•`), its marker taken off.
pub fn of(brief: &str) -> Vec<String> {
    brief
        .lines()
        .filter(|l| !l.starts_with(char::is_whitespace))
        .filter_map(|l| item(l).map(str::trim).filter(|t| !t.is_empty()))
        .map(str::to_string)
        .collect()
}

/// `line` without its list marker, if it has one.
fn item(line: &str) -> Option<&str> {
    for bullet in ["- ", "* ", "• "] {
        if let Some(rest) = line.strip_prefix(bullet) {
            return Some(rest);
        }
    }
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    let rest = &line[digits..];
    match digits > 0 {
        true => rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")),
        false => None,
    }
}

/// What a step's line may say became of it.
const OUTCOMES: [&str; 3] = ["not done", "could not", "done"];

/// The steps, by number from 1, that `answer`'s `Steps:` section says nothing
/// of: no line opening with the step's number and saying `done`, `not done`
/// or `could not`. A brief of fewer than two steps asks for no section.
pub fn unaccounted(steps: &[String], answer: &str) -> Vec<usize> {
    if steps.len() < 2 {
        return Vec::new();
    }
    let section: Vec<&str> = answer.lines().skip_while(|l| !heading(l)).skip(1).collect();
    (1..=steps.len())
        .filter(|n| {
            !section.iter().any(|l| {
                let l = l.trim().trim_start_matches(['-', '*', ' ']);
                let lower = l.to_ascii_lowercase();
                (l.starts_with(&format!("{n}.")) || l.starts_with(&format!("{n})")))
                    && OUTCOMES.iter().any(|o| lower.contains(o))
            })
        })
        .collect()
}

/// Whether `line` opens the `Steps:` section, however it is marked up.
fn heading(line: &str) -> bool {
    let bare = line.trim().trim_matches(['*', '#', '_', ' ']);
    bare.eq_ignore_ascii_case("steps:") || bare.eq_ignore_ascii_case("steps")
}

/// What a task whose answer left `missing` of `steps` out is told.
pub fn correction(steps: &[String], missing: &[usize]) -> String {
    let listed: String = missing
        .iter()
        .map(|&n| format!("\n{n}. {}", steps[n - 1]))
        .collect();
    format!(
        "Your answer does not say what became of {} of the {} steps you were given:{listed}\n\n\
         End your answer with a `Steps:` section: one line per step, by its number, saying \
         `done` — and the call, commit or job that shows it — or `not done` or `could not`, \
         and why. A step you did not do is said, not left out.",
        missing.len(),
        steps.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRIEF: &str = "Make file_edit show the lines it changed.\n\n\
                         1. Implement it in edit_changes.rs.\n\
                         2) Run the zend-tools tests.\n\
                         - Have a reviewer check it.\n  - nested, not a step\n";

    #[test]
    fn a_briefs_steps_are_its_numbered_and_bulleted_lines() {
        assert_eq!(
            of(BRIEF),
            [
                "Implement it in edit_changes.rs.",
                "Run the zend-tools tests.",
                "Have a reviewer check it."
            ]
        );
        assert!(of("Just answer the question.").is_empty());
    }

    /// **Every step needs a line saying what became of it**; one left out, or
    /// one with no outcome, is unaccounted for.
    #[test]
    fn a_step_the_answer_says_nothing_of_is_unaccounted() {
        let steps = of(BRIEF);
        let whole = "It shows them.\n\n**Steps:**\n1. done — 3dc39ea3c\n2. done — job \
                     build:x:y\n3. not done: no reviewer was free\n";
        assert!(unaccounted(&steps, whole).is_empty());
        let skipped = "It shows them.\n\nSteps:\n1. done\n2. done\n";
        assert_eq!(unaccounted(&steps, skipped), [3]);
        let unsaid = "Steps:\n1. done\n2. ran them\n3. could not, no reviewer\n";
        assert_eq!(unaccounted(&steps, unsaid), [2]);
        assert_eq!(unaccounted(&steps, "It shows them."), [1, 2, 3]);
        assert!(unaccounted(&of("One thing."), "Done.").is_empty());
    }

    #[test]
    fn the_correction_names_the_steps_left_out() {
        let steps = of(BRIEF);
        assert_eq!(
            correction(&steps, &[3]),
            "Your answer does not say what became of 1 of the 3 steps you were given:\n3. Have \
             a reviewer check it.\n\nEnd your answer with a `Steps:` section: one line per step, \
             by its number, saying `done` — and the call, commit or job that shows it — or `not \
             done` or `could not`, and why. A step you did not do is said, not left out."
        );
    }
}
