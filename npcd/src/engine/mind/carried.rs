//! The sections of one collection a conversation holds, kept in line with what
//! the character should be carrying.
//!
//! A character's mission and its journal are both collections of sections
//! submitted to its own conversation. Submitted text is sealed and persisted
//! with the conversation but cannot be read back, so what was submitted is held
//! here, beside the sequence, and [`Carried::reconcile`] compares it with what
//! the character should carry now: a section is submitted when it is missing or
//! its text changed, resubmitted when it failed to seal, and removed when it is
//! no longer wanted. Text that has not changed costs nothing per turn.
//!
//! Streams an earlier run sealed for the conversation are invisible to a fresh
//! `Carried`, so the first reconcile of a rejoined conversation also removes the
//! names the caller says may be stale.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use candle_conversation::{SectionRef, SectionState, Sequence};

/// How often a wait for a seal looks at the section's state.
const SEAL_POLL: Duration = Duration::from_millis(20);

/// A section this conversation was given, and the text it was given.
struct Held {
    text: String,
    section: SectionRef,
}

/// One section to be carried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Want<'a> {
    pub name: &'a str,
    pub text: &'a str,
}

/// What a reconcile has to do, decided before any of it is done.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Plan {
    /// Indices into the wanted list to submit.
    submit: Vec<usize>,
    /// Names to remove.
    remove: Vec<String>,
}

/// One section already held: its name, its text, and whether it can still be
/// read (it has neither failed to seal nor been removed).
struct Standing<'a> {
    name: &'a str,
    text: &'a str,
    alive: bool,
}

/// What must change to carry `wanted`, given what is `held`.
///
/// `sweep` names are removed when `sweeping`, whether or not anything held
/// carries them, because a stream an earlier run left is not held.
fn plan(held: &[Standing<'_>], wanted: &[Want<'_>], sweep: &[&str], sweeping: bool) -> Plan {
    let mut plan = Plan::default();
    for (i, want) in wanted.iter().enumerate() {
        let current = held
            .iter()
            .any(|h| h.name == want.name && h.alive && h.text == want.text);
        if !current {
            plan.submit.push(i);
        }
    }
    for h in held {
        if !wanted.iter().any(|w| w.name == h.name) {
            plan.remove.push(h.name.to_string());
        }
    }
    if sweeping {
        for name in sweep {
            let carried = wanted.iter().any(|w| w.name == *name);
            if !carried && !plan.remove.iter().any(|r| r == name) {
                plan.remove.push((*name).to_string());
            }
        }
    }
    plan
}

/// The sections of one collection a conversation holds.
pub struct Carried {
    held: BTreeMap<String, Held>,
    /// Whether the streams an earlier run left for this conversation have been
    /// looked for. A rejoined conversation starts without having; a new one has
    /// none to find.
    swept: bool,
}

impl Carried {
    /// Nothing held. `swept` is true for a conversation just minted.
    pub fn new(swept: bool) -> Self {
        Self {
            held: BTreeMap::new(),
            swept,
        }
    }

    /// Bring the conversation's sections in line with `wanted`, and return the
    /// names that are sealed and selectable, in the order they were wanted.
    ///
    /// With `wait`, a submitted section is given that long to seal, because its
    /// text is meant to be in the prompt of the turn that follows; past it the
    /// section is left out of the answer and the stand-in member is read instead.
    pub async fn reconcile(
        &mut self,
        sequence: &mut Sequence,
        collection: &str,
        wanted: &[Want<'_>],
        sweep: &[&str],
        wait: Option<Duration>,
    ) -> Vec<String> {
        let standing: Vec<Standing<'_>> = self
            .held
            .iter()
            .map(|(name, h)| Standing {
                name,
                text: &h.text,
                alive: !matches!(
                    h.section.state(),
                    SectionState::Failed(_) | SectionState::Removed
                ),
            })
            .collect();
        let todo = plan(&standing, wanted, sweep, !self.swept);
        drop(standing);
        self.swept = true;

        for name in &todo.remove {
            self.held.remove(name);
            sequence.remove_section_named(name);
        }
        for i in todo.submit {
            let want = &wanted[i];
            self.held.remove(want.name);
            match sequence.submit_section(collection, want.name, want.text, 1.0) {
                Ok(section) => {
                    self.held.insert(
                        want.name.to_string(),
                        Held {
                            text: want.text.to_string(),
                            section,
                        },
                    );
                }
                Err(e) => tracing::warn!("section {} could not be submitted: {e}", want.name),
            }
        }

        let deadline = wait.map(|d| Instant::now() + d);
        let mut sealed = Vec::new();
        for want in wanted {
            let Some(held) = self.held.get(want.name) else {
                continue;
            };
            loop {
                match held.section.state() {
                    SectionState::Ready { .. } => {
                        sealed.push(want.name.to_string());
                        break;
                    }
                    SectionState::Failed(why) => {
                        tracing::warn!("section {} failed to seal: {why}", want.name);
                        break;
                    }
                    SectionState::Removed => break,
                    SectionState::Pending => match deadline {
                        Some(at) if Instant::now() < at => tokio::time::sleep(SEAL_POLL).await,
                        _ => break,
                    },
                }
            }
        }
        sealed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn want<'a>(name: &'a str, text: &'a str) -> Want<'a> {
        Want { name, text }
    }

    fn standing<'a>(name: &'a str, text: &'a str, alive: bool) -> Standing<'a> {
        Standing { name, text, alive }
    }

    #[test]
    fn a_section_that_is_already_held_unchanged_costs_nothing() {
        let held = [standing("journal/1", "a", true)];
        let p = plan(&held, &[want("journal/1", "a")], &[], false);
        assert_eq!(p, Plan::default());
    }

    #[test]
    fn a_missing_section_is_submitted() {
        let p = plan(&[], &[want("journal/1", "a")], &[], false);
        assert_eq!(p.submit, vec![0]);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_section_whose_text_changed_is_submitted_again() {
        let held = [standing("journal/open", "old", true)];
        let p = plan(&held, &[want("journal/open", "new")], &[], false);
        assert_eq!(p.submit, vec![0]);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_section_that_failed_to_seal_is_submitted_again() {
        let held = [standing("journal/1", "a", false)];
        let p = plan(&held, &[want("journal/1", "a")], &[], false);
        assert_eq!(p.submit, vec![0]);
    }

    #[test]
    fn a_held_section_no_longer_wanted_is_removed() {
        let held = [
            standing("journal/1", "a", true),
            standing("journal/2", "b", true),
        ];
        let p = plan(&held, &[want("journal/2", "b")], &[], false);
        assert!(p.submit.is_empty());
        assert_eq!(p.remove, vec!["journal/1".to_string()]);
    }

    #[test]
    fn the_first_reconcile_removes_what_an_earlier_run_may_have_left() {
        let p = plan(&[], &[], &["mission/standing"], true);
        assert_eq!(p.remove, vec!["mission/standing".to_string()]);
    }

    #[test]
    fn a_swept_conversation_does_not_remove_the_sweep_names_again() {
        let p = plan(&[], &[], &["mission/standing"], false);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_name_that_is_wanted_is_never_swept() {
        let p = plan(
            &[],
            &[want("mission/standing", "go")],
            &["mission/standing"],
            true,
        );
        assert_eq!(p.submit, vec![0]);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_name_that_is_held_and_swept_is_removed_once() {
        let held = [standing("journal/1", "a", true)];
        let p = plan(&held, &[], &["journal/1"], true);
        assert_eq!(p.remove, vec!["journal/1".to_string()]);
    }
}
