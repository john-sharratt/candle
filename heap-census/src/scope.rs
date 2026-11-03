//! Scopes: which part of the daemon an allocation was made in, named in the
//! code (`docs/memory_census.md` §10).
//!
//! A thread names its scope as it starts ([`set_thread`]) and a call narrows it
//! for as long as a guard lives ([`enter`]). [`Counting`](crate::Counting)
//! reads the calling thread's scope at each allocation of
//! [`LABELLED_FROM`](crate::counting::LABELLED_FROM) bytes or more and keeps it
//! in the block's header, so the free — on whichever thread — is taken off the
//! same scope. A scope names the code that *made* a block, not whatever holds
//! it after.
//!
//! The scope is one byte of the thread's census word (`thread_state`), read
//! with the shard in one TLS load.

use std::marker::PhantomData;

use super::thread_state::replace_scope;
#[cfg(test)]
use super::thread_state::shard_and_scope;

/// How many scopes there are. A power of two: a header byte is masked by it,
/// so no byte read back indexes past the counters.
pub const SCOPES: usize = 16;

/// The parts of the daemon a block can be made in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Scope {
    /// No scope named: a thread that never set one.
    Unscoped = 0,
    /// The async runtime's workers and blocking pool: HTTP, MCP, the apps'
    /// tasks.
    Async = 1,
    /// The conversation scheduler: waves, seals, projections.
    Scheduler = 2,
    /// The persistence pass: tier migration, purges, the store gauges.
    Persistence = 3,
    /// The log's owner: every append and every remote batch.
    LogOwner = 4,
    /// A batch other members wrote, decided and applied.
    RemoteApply = 5,
    /// The turns a remote batch completed, adopted.
    Adoption = 6,
    /// The summary tree's summariser.
    Summariser = 7,
    /// A submitted prompt section's seal.
    SectionSeal = 8,
    /// The repositories in the substrate, followed.
    GitRepos = 9,
    /// The journals: sync and moves.
    Journals = 10,
    /// A tenant's load.
    Loader = 11,
    /// The log's maintenance: compaction and segment unlinks.
    Maintenance = 12,
    /// The census itself: its readings and the store gauges.
    Census = 13,
    /// The cluster's sockets: replication sends and receives.
    Cluster = 14,
}

impl Scope {
    /// Every scope, in index order: at most [`SCOPES`].
    pub const ALL: &'static [Scope] = &[
        Scope::Unscoped,
        Scope::Async,
        Scope::Scheduler,
        Scope::Persistence,
        Scope::LogOwner,
        Scope::RemoteApply,
        Scope::Adoption,
        Scope::Summariser,
        Scope::SectionSeal,
        Scope::GitRepos,
        Scope::Journals,
        Scope::Loader,
        Scope::Maintenance,
        Scope::Census,
        Scope::Cluster,
    ];

    /// The scope's name, as the report and the log carry it.
    pub const fn name(self) -> &'static str {
        match self {
            Scope::Unscoped => "unscoped",
            Scope::Async => "async",
            Scope::Scheduler => "scheduler",
            Scope::Persistence => "persistence",
            Scope::LogOwner => "log_owner",
            Scope::RemoteApply => "remote_apply",
            Scope::Adoption => "adoption",
            Scope::Summariser => "summariser",
            Scope::SectionSeal => "section_seal",
            Scope::GitRepos => "git_repos",
            Scope::Journals => "journals",
            Scope::Loader => "loader",
            Scope::Maintenance => "maintenance",
            Scope::Census => "census",
            Scope::Cluster => "cluster",
        }
    }
}

/// The calling thread's scope, as its index. A thread past its thread-locals'
/// lifetime is unscoped.
#[cfg(test)]
pub(crate) fn current() -> u8 {
    shard_and_scope().1
}

/// Name the calling thread's scope for the rest of its life, or until a guard
/// narrows it. What a thread calls as it starts.
pub fn set_thread(scope: Scope) {
    replace_scope(scope as u8);
}

/// The calling thread's scope as it was before an [`enter`], put back when
/// this is dropped. Not `Send`: a guard held across an `.await` would name
/// the scope of whatever task the worker ran next.
#[must_use = "the scope ends when the guard is dropped"]
pub struct Scoped {
    previous: u8,
    _thread: PhantomData<*const ()>,
}

/// Narrow the calling thread's scope to `scope` until the guard is dropped.
pub fn enter(scope: Scope) -> Scoped {
    Scoped {
        previous: replace_scope(scope as u8),
        _thread: PhantomData,
    }
}

impl Drop for Scoped {
    fn drop(&mut self) {
        replace_scope(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    /// **Every scope sits at its own index**, so a header byte names it back.
    #[test]
    fn every_scope_sits_at_its_index() {
        for (i, s) in Scope::ALL.iter().enumerate() {
            assert_eq!(*s as usize, i, "{s:?}");
        }
        assert!(SCOPES.is_power_of_two());
        assert!(Scope::ALL.len() <= SCOPES);
    }

    /// **A guard narrows the scope and puts the outer one back**, nested.
    #[test]
    fn a_guard_restores_the_scope_it_narrowed() {
        thread::spawn(|| {
            assert_eq!(current(), Scope::Unscoped as u8);
            set_thread(Scope::LogOwner);
            {
                let _a = enter(Scope::RemoteApply);
                assert_eq!(current(), Scope::RemoteApply as u8);
                {
                    let _b = enter(Scope::Adoption);
                    assert_eq!(current(), Scope::Adoption as u8);
                }
                assert_eq!(current(), Scope::RemoteApply as u8);
            }
            assert_eq!(current(), Scope::LogOwner as u8);
        })
        .join()
        .unwrap();
    }

    /// **A scope is the thread's own**: another thread's is untouched.
    #[test]
    fn a_scope_is_per_thread() {
        let _g = enter(Scope::Journals);
        let other = thread::spawn(current).join().unwrap();
        assert_eq!(other, Scope::Unscoped as u8);
        assert_eq!(current(), Scope::Journals as u8);
    }
}
