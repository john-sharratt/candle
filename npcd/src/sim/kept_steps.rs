//! A mission's steps rewritten without losing what is owed
//! (`docs/npc_conversations.md` §7.1): an edit through an agenda or a
//! conversation keeps every step already done and every critical step,
//! wherever the new list puts them, so nobody can drop an unsigned critical
//! step — or bring it back as an ordinary one — by rewording the list.

use crate::engine::mission::Todo;

/// `was`'s steps rewritten as `steps`: a step named again keeps its state; a
/// done or critical step not named again is kept, in its place before the
/// new ones; a new step is new.
pub fn kept_steps(was: &[Todo], steps: Vec<String>) -> Vec<Todo> {
    let named = |t: &Todo| {
        steps
            .iter()
            .any(|s| t.text.trim().eq_ignore_ascii_case(s.trim()))
    };
    let mut out: Vec<Todo> = was
        .iter()
        .filter(|t| (t.done || t.critical) && !named(t))
        .cloned()
        .collect();
    out.extend(steps.into_iter().map(|text| {
        was.iter()
            .find(|t| t.text.trim().eq_ignore_ascii_case(text.trim()))
            .cloned()
            .unwrap_or_else(|| Todo::new(text))
    }));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(todo: &[Todo]) -> Vec<(&str, bool, bool)> {
        todo.iter()
            .map(|t| (t.text.as_str(), t.done, t.critical))
            .collect()
    }

    #[test]
    fn an_edit_keeps_what_is_done_and_what_waits_on_a_signature() {
        let mut done = Todo::new("measure the lounge");
        done.done = true;
        let mut lease = Todo::new("sign the lease");
        lease.critical = true;
        let was = [done, lease, Todo::new("buy paint")];
        let edited = kept_steps(&was, vec!["paint the walls".into(), "Buy paint".into()]);
        assert_eq!(
            steps(&edited),
            [
                ("measure the lounge", true, false),
                ("sign the lease", false, true),
                ("paint the walls", false, false),
                ("buy paint", false, false),
            ]
        );
    }

    #[test]
    fn a_critical_step_named_again_stays_critical() {
        let mut lease = Todo::new("sign the lease");
        lease.critical = true;
        let edited = kept_steps(&[lease], vec!["Sign the lease".into()]);
        assert_eq!(steps(&edited), [("sign the lease", false, true)]);
    }
}
