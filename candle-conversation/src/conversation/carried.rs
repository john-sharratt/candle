//! The sections of one collection a conversation holds, kept in line with what
//! it should be carrying.
//!
//! A collection of sections submitted to a conversation's own timeline — an npcd
//! character's mission and journal, a zend session's journal — is sealed and
//! persisted with the conversation, but submitted text cannot be read back, so
//! what was submitted is held here, beside the sequence, and
//! [`Carried::reconcile`] compares it with what the conversation should carry
//! now: a section is submitted when it is missing or its text changed,
//! resubmitted when it failed to seal, and removed when it is no longer wanted.
//! Text that has not changed costs nothing per turn.
//!
//! Streams an earlier run sealed for the conversation are invisible to a fresh
//! `Carried`, so the first reconcile of a rejoined conversation also removes the
//! names the caller says may be stale.
//!
//! # A chain
//!
//! A journal's entries are read one after another, so each is sealed as the
//! continuation of the ones before it ([`Sequence::submit_section`]'s
//! `after`): it attends to them and its recurrence runs on from them. Such a
//! collection is reconciled as a [`Linking::Chain`]: a section is current only
//! while what comes before it is too, so when an entry is resealed, or one
//! before it leaves, every section after it is resealed with it. And a turn
//! reads only the chain's sealed beginning — a section never shows without
//! the ones it was sealed after.

use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use super::{SectionRef, SectionState, Sequence};

/// How often a wait for a seal looks at the section's state.
const SEAL_POLL: Duration = Duration::from_millis(20);

/// A section this conversation was given, the text it was given, and the
/// sections it was sealed after.
struct Held {
    text: String,
    after: Vec<String>,
    section: SectionRef,
}

/// One section to be carried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Want<'a> {
    pub name: &'a str,
    pub text: &'a str,
}

/// How a collection's sections relate to one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linking {
    /// Each stands alone — read, or not, independently of the others.
    Apart,
    /// Each continues the ones before it, in the order they are wanted.
    Chain,
}

/// What a reconcile has to do, decided before any of it is done.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Plan {
    /// Indices into the wanted list to submit.
    submit: Vec<usize>,
    /// Names to remove.
    remove: Vec<String>,
}

/// One section already held: its name, its text, the sections it was sealed
/// after, whether it can still be read (it has neither failed to seal nor
/// been removed), and whether it is still waiting to seal.
struct Standing<'a> {
    name: &'a str,
    text: &'a str,
    after: &'a [String],
    alive: bool,
    pending: bool,
}

/// The names a chained section at `i` of `wanted` continues.
fn before(wanted: &[Want<'_>], i: usize) -> Vec<String> {
    wanted[..i].iter().map(|w| w.name.to_string()).collect()
}

/// What must change to carry `wanted`, given what is `held`.
///
/// In a [`Linking::Chain`] a section is current only when it was sealed after
/// exactly the sections now wanted before it, and the first that is not makes
/// every section after it not current either: they continue it.
///
/// `sweep` names are removed when `sweeping`, whether or not anything held
/// carries them, because a stream an earlier run left is not held.
///
/// **A chain still sealing is left to finish.** Replacing it while a link is
/// pending removes that link under the links sealed after it, and every one of
/// them fails with it. Under load a chain takes minutes to seal, the wants it
/// is reconciled to move faster than that, and a journal was resealed over and
/// over without converging: 519 seals and 454 failures in half an hour, ahead
/// of everything else the engine had to do. Once nothing is pending, the next
/// reconcile carries whatever is wanted then; meanwhile a turn reads the
/// chain's sealed beginning, as it does while any chain seals.
fn plan(
    held: &[Standing<'_>],
    wanted: &[Want<'_>],
    kept: Linking,
    sweep: &[&str],
    sweeping: bool,
) -> Plan {
    let mut plan = Plan::default();
    if kept == Linking::Chain && held.iter().any(|h| h.pending) {
        return plan;
    }
    let mut broken = false;
    for (i, want) in wanted.iter().enumerate() {
        let after = match kept {
            Linking::Apart => Vec::new(),
            Linking::Chain => before(wanted, i),
        };
        let current = !broken
            && held
                .iter()
                .any(|h| h.name == want.name && h.alive && h.text == want.text && h.after == after);
        if !current {
            plan.submit.push(i);
            broken = kept == Linking::Chain;
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

    /// Bring the conversation's sections in line with `wanted`, related as
    /// `kept` says, and return the names that are sealed and selectable, in the
    /// order they were wanted — for a chain, only its sealed beginning.
    ///
    /// With `wait`, a submitted section is given that long to seal, because its
    /// text is meant to be in the prompt of the turn that follows; past it the
    /// section is left out of the answer and the caller's stand-in is read
    /// instead. The crate runs on no async runtime of its own, so the wait
    /// sleeps through the caller's `sleep`.
    pub async fn reconcile<S, F>(
        &mut self,
        sequence: &mut Sequence,
        collection: &str,
        wanted: &[Want<'_>],
        kept: Linking,
        sweep: &[&str],
        wait: Option<Duration>,
        sleep: S,
    ) -> Vec<String>
    where
        S: Fn(Duration) -> F,
        F: Future<Output = ()>,
    {
        let standing: Vec<Standing<'_>> = self
            .held
            .iter()
            .map(|(name, h)| {
                let state = h.section.state();
                Standing {
                    name,
                    text: &h.text,
                    after: &h.after,
                    alive: !matches!(state, SectionState::Failed(_) | SectionState::Removed),
                    pending: matches!(state, SectionState::Pending),
                }
            })
            .collect();
        let todo = plan(&standing, wanted, kept, sweep, !self.swept);
        drop(standing);
        self.swept = true;

        for name in &todo.remove {
            self.held.remove(name);
            sequence.remove_section_named(name);
        }
        for i in todo.submit {
            let want = &wanted[i];
            self.held.remove(want.name);
            let after = match kept {
                Linking::Apart => Vec::new(),
                Linking::Chain => before(wanted, i),
            };
            // Submitted in order, so what a section continues is already held.
            let continues: Vec<SectionRef> = after
                .iter()
                .filter_map(|name| self.held.get(name).map(|h| h.section.clone()))
                .collect();
            if continues.len() != after.len() {
                tracing::warn!(
                    "section {} is not submitted: a section it continues could not be",
                    want.name
                );
                continue;
            }
            match sequence.submit_section(collection, want.name, want.text, 1.0, &continues) {
                Ok(section) => {
                    self.held.insert(
                        want.name.to_string(),
                        Held {
                            text: want.text.to_string(),
                            after,
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
            let ready = match self.held.get(want.name) {
                None => false,
                Some(held) => loop {
                    match held.section.state() {
                        SectionState::Ready { .. } => break true,
                        SectionState::Failed(why) => {
                            tracing::warn!("section {} failed to seal: {why}", want.name);
                            break false;
                        }
                        SectionState::Removed => break false,
                        SectionState::Pending => match deadline {
                            Some(at) if Instant::now() < at => sleep(SEAL_POLL).await,
                            _ => break false,
                        },
                    }
                },
            };
            if ready {
                sealed.push(want.name.to_string());
            } else if kept == Linking::Chain {
                // What follows was sealed after this; without it, it is read
                // out of the context it was computed in.
                break;
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

    const NONE: &[String] = &[];

    fn standing<'a>(name: &'a str, text: &'a str, alive: bool) -> Standing<'a> {
        Standing {
            name,
            text,
            after: NONE,
            alive,
            pending: false,
        }
    }

    fn chained<'a>(name: &'a str, text: &'a str, after: &'a [String]) -> Standing<'a> {
        Standing {
            name,
            text,
            after,
            alive: true,
            pending: false,
        }
    }

    /// **A chain with a link still sealing is not replaced under it.** Losing
    /// its first link would reseal every later one; while one of them is
    /// pending, nothing is submitted or removed, and the reconcile after it
    /// settles does the whole change.
    #[test]
    fn a_chain_still_sealing_is_left_to_finish() {
        let one = names(&["journal/1"]);
        let two = names(&["journal/1", "journal/2"]);
        let held = [
            chained("journal/1", "a", NONE),
            Standing {
                pending: true,
                ..chained("journal/2", "b", &one)
            },
            chained("journal/open", "x", &two),
        ];
        let wanted = [want("journal/2", "b"), want("journal/open", "y")];
        assert_eq!(
            plan(&held, &wanted, Linking::Chain, &[], false),
            Plan::default()
        );
        let settled = [
            chained("journal/1", "a", NONE),
            chained("journal/2", "b", &one),
            chained("journal/open", "x", &two),
        ];
        let p = plan(&settled, &wanted, Linking::Chain, &[], false);
        assert_eq!(p.submit, vec![0, 1]);
        assert_eq!(p.remove, vec!["journal/1".to_string()]);
    }

    /// Sections kept apart have nothing sealed after them, so a pending one is
    /// replaced as soon as its text changes.
    #[test]
    fn a_pending_section_kept_apart_is_replaced_at_once() {
        let held = [Standing {
            pending: true,
            ..standing("mission/standing", "go", true)
        }];
        let p = plan(
            &held,
            &[want("mission/standing", "stay")],
            Linking::Apart,
            &[],
            false,
        );
        assert_eq!(p.submit, vec![0]);
    }

    fn names(n: &[&str]) -> Vec<String> {
        n.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_section_that_is_already_held_unchanged_costs_nothing() {
        let held = [standing("journal/1", "a", true)];
        let p = plan(&held, &[want("journal/1", "a")], Linking::Apart, &[], false);
        assert_eq!(p, Plan::default());
    }

    #[test]
    fn a_missing_section_is_submitted() {
        let p = plan(&[], &[want("journal/1", "a")], Linking::Apart, &[], false);
        assert_eq!(p.submit, vec![0]);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_section_whose_text_changed_is_submitted_again() {
        let held = [standing("journal/open", "old", true)];
        let p = plan(
            &held,
            &[want("journal/open", "new")],
            Linking::Apart,
            &[],
            false,
        );
        assert_eq!(p.submit, vec![0]);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_section_that_failed_to_seal_is_submitted_again() {
        let held = [standing("journal/1", "a", false)];
        let p = plan(&held, &[want("journal/1", "a")], Linking::Apart, &[], false);
        assert_eq!(p.submit, vec![0]);
    }

    #[test]
    fn a_held_section_no_longer_wanted_is_removed() {
        let held = [
            standing("journal/1", "a", true),
            standing("journal/2", "b", true),
        ];
        let p = plan(&held, &[want("journal/2", "b")], Linking::Apart, &[], false);
        assert!(p.submit.is_empty());
        assert_eq!(p.remove, vec!["journal/1".to_string()]);
    }

    /// A new entry on the end of a chain reseals the entry and what follows
    /// it — the open items — and nothing before it.
    #[test]
    fn a_chain_grown_at_its_end_seals_only_the_new_link_and_what_follows() {
        let one = names(&["journal/1"]);
        let held = [
            chained("journal/1", "a", NONE),
            chained("journal/open", "x", &one),
        ];
        let wanted = [
            want("journal/1", "a"),
            want("journal/2", "b"),
            want("journal/open", "x"),
        ];
        let p = plan(&held, &wanted, Linking::Chain, &[], false);
        assert_eq!(p.submit, vec![1, 2], "the open items now follow entry 2");
        assert!(p.remove.is_empty());
    }

    /// The oldest entry ageing out changes what every later one continues,
    /// so all of them are sealed again, though not one of their texts moved.
    #[test]
    fn a_chain_that_loses_its_first_link_is_sealed_again_from_the_start() {
        let one = names(&["journal/1"]);
        let two = names(&["journal/1", "journal/2"]);
        let held = [
            chained("journal/1", "a", NONE),
            chained("journal/2", "b", &one),
            chained("journal/open", "x", &two),
        ];
        let wanted = [want("journal/2", "b"), want("journal/open", "x")];
        let p = plan(&held, &wanted, Linking::Chain, &[], false);
        assert_eq!(p.submit, vec![0, 1]);
        assert_eq!(p.remove, vec!["journal/1".to_string()]);
    }

    /// A link resealed in the middle takes every link after it along, even
    /// one whose text and predecessors are unchanged: what it continues is a
    /// new seal.
    #[test]
    fn a_link_resealed_mid_chain_takes_every_later_link_along() {
        let one = names(&["journal/1"]);
        let two = names(&["journal/1", "journal/2"]);
        let held = [
            chained("journal/1", "a", NONE),
            Standing {
                alive: false,
                ..chained("journal/2", "b", &one)
            },
            chained("journal/open", "x", &two),
        ];
        let wanted = [
            want("journal/1", "a"),
            want("journal/2", "b"),
            want("journal/open", "x"),
        ];
        let p = plan(&held, &wanted, Linking::Chain, &[], false);
        assert_eq!(p.submit, vec![1, 2]);
    }

    #[test]
    fn sections_kept_apart_never_reseal_for_one_another() {
        let held = [
            standing("journal/2", "b", true),
            standing("journal/open", "x", true),
        ];
        let wanted = [want("journal/2", "b"), want("journal/open", "x")];
        let p = plan(&held, &wanted, Linking::Apart, &[], false);
        assert!(p.submit.is_empty());
    }

    #[test]
    fn the_first_reconcile_removes_what_an_earlier_run_may_have_left() {
        let p = plan(&[], &[], Linking::Apart, &["mission/standing"], true);
        assert_eq!(p.remove, vec!["mission/standing".to_string()]);
    }

    #[test]
    fn a_swept_conversation_does_not_remove_the_sweep_names_again() {
        let p = plan(&[], &[], Linking::Apart, &["mission/standing"], false);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_name_that_is_wanted_is_never_swept() {
        let p = plan(
            &[],
            &[want("mission/standing", "go")],
            Linking::Apart,
            &["mission/standing"],
            true,
        );
        assert_eq!(p.submit, vec![0]);
        assert!(p.remove.is_empty());
    }

    #[test]
    fn a_name_that_is_held_and_swept_is_removed_once() {
        let held = [standing("journal/1", "a", true)];
        let p = plan(&held, &[], Linking::Apart, &["journal/1"], true);
        assert_eq!(p.remove, vec!["journal/1".to_string()]);
    }
}
