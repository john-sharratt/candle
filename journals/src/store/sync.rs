//! Keeping the store in step with origin (§6.3), with the network part of it
//! run while the store is free.
//!
//! A sync is rounds of three steps. The store is lent for a moment to read
//! what a round sends ([`Store::transfer`]); the push and the fetch run on
//! that with nothing held ([`Transfer::run`]); and the store is lent again to
//! take in what came back ([`Store::settle`]). A write made while a round is
//! on the network commits at once on top of the head the round is pushing,
//! and the next round pushes it.

use std::sync::{Arc, Mutex};

use zend_vfs::{
    BranchName, FetchSpec, Lease, Oid, PushOutcome, PushSpec, RefOp, RefTransaction, Rejection,
    RemoteName, Repo,
};

use super::{journal_ref, Store, StoreError, Synced, PUSH_TRIES};
use crate::store::read::files_at;
use crate::tree::load::load;

/// Where a sync finds the store between its network steps.
pub trait Lend {
    /// `f` run on the store; `None` when there is no store to lend.
    fn lend<R>(&mut self, f: impl FnOnce(&mut Store) -> R) -> Option<R>;
}

/// A store its caller holds alone.
impl Lend for &mut Store {
    fn lend<R>(&mut self, f: impl FnOnce(&mut Store) -> R) -> Option<R> {
        Some(f(self))
    }
}

/// A store shared with writers: locked only for each lend, whatever a panic
/// elsewhere left in it — its state changes only once a commit has landed.
impl Lend for &Mutex<Option<Store>> {
    fn lend<R>(&mut self, f: impl FnOnce(&mut Store) -> R) -> Option<R> {
        self.lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
            .map(f)
    }
}

/// The push and fetch one round makes, as the store stood when it was read.
pub struct Transfer {
    repo: Arc<Repo>,
    remote: RemoteName,
    branch: BranchName,
    /// The head to push.
    head: Oid,
    /// What origin held when last fetched or pushed: the push's lease.
    origin_at: Oid,
}

/// What a round's network steps found.
#[derive(Debug, PartialEq, Eq)]
pub struct Transferred {
    /// The head origin took, when the push landed.
    landed: Option<Oid>,
    /// What origin's branch holds now.
    theirs: Oid,
}

/// Whether a sync goes on.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Round {
    Done,
    /// Origin moved first and the service's commits were replayed onto it:
    /// they go up in another round.
    Again,
}

impl Transfer {
    /// Push the head under the lease, then fetch: nothing of the store is
    /// held, so writes go on meanwhile.
    pub fn run(&self) -> Result<Transferred, StoreError> {
        let mut landed = None;
        if self.head != self.origin_at {
            let spec = PushSpec::branch(
                self.head.clone(),
                self.branch.clone(),
                Lease::Expect(self.origin_at.clone()),
            );
            let results = self.repo.push(&self.remote, &[spec])?;
            match results.first().map(|r| r.outcome.clone()) {
                Some(PushOutcome::Rejected(Rejection::Stale))
                | Some(PushOutcome::Rejected(Rejection::AtomicAborted)) => {}
                Some(PushOutcome::Rejected(other)) => {
                    return Err(StoreError::Repository(format!(
                        "origin refused the push: {other:?}"
                    )))
                }
                Some(_) => landed = Some(self.head.clone()),
                None => return Err(StoreError::Repository("the push reported nothing".into())),
            }
        }
        self.repo
            .fetch(&self.remote, &FetchSpec::Branch(self.branch.clone()))?;
        let theirs = self
            .repo
            .ref_target(&self.remote.tracking(&self.branch))?
            .ok_or_else(|| {
                StoreError::Repository(format!("{}/{} is gone", self.remote, self.branch))
            })?;
        Ok(Transferred { landed, theirs })
    }
}

impl Store {
    /// What the next round sends, and where.
    pub fn transfer(&self) -> Transfer {
        Transfer {
            repo: Arc::clone(&self.repo),
            remote: self.remote.clone(),
            branch: self.branch.clone(),
            head: self.head.clone(),
            origin_at: self.origin_at.clone(),
        }
    }

    /// Take in what `sent`'s round found, adding what it did to `done`.
    pub(super) fn settle(
        &mut self,
        sent: &Transfer,
        got: Transferred,
        done: &mut Synced,
    ) -> Result<Round, StoreError> {
        if sent.origin_at != self.origin_at {
            // Another sync took a round in between: this one's findings are
            // about a store that has moved on.
            return Ok(Round::Again);
        }
        if let Some(landed) = got.landed {
            done.pushed += self.count_between(&self.origin_at, &landed)?;
            self.origin_at = landed;
        }
        if got.theirs == self.origin_at {
            // Origin holds what it held: whatever was written during the
            // round goes up at the next sync.
            done.settled = !self.unpushed();
            return Ok(Round::Done);
        }
        done.fetched = true;
        if !self.unpushed() {
            // Nothing of ours is waiting: take origin as it is.
            let files = files_at(&self.repo, &got.theirs)?;
            let txn = RefTransaction::new().push(RefOp::Update {
                name: journal_ref(),
                new: got.theirs.clone(),
                old: self.head.clone(),
            });
            self.repo.update_refs(&txn)?;
            self.tree = Arc::new(load(&files));
            self.files = files;
            self.head = got.theirs.clone();
            self.origin_at = got.theirs;
            done.settled = true;
            return Ok(Round::Done);
        }
        let r = self.replay_onto(got.theirs)?;
        done.replayed += r.replayed;
        done.dropped.extend(r.dropped);
        done.renamed.extend(r.renamed);
        Ok(Round::Again)
    }
}

/// Push what origin lacks, fetch what it has, and replay the service's
/// commits onto it when it moved first, borrowing the store from `at` only
/// between the network steps. `None` when `at` has no store.
pub fn sync(mut at: impl Lend) -> Option<Result<Synced, StoreError>> {
    let mut done = Synced::default();
    for _ in 0..PUSH_TRIES {
        let sent = at.lend(|s| s.transfer())?;
        let got = match sent.run() {
            Ok(got) => got,
            Err(e) => return Some(Err(e)),
        };
        match at.lend(|s| s.settle(&sent, got, &mut done))? {
            Ok(Round::Done) => return Some(Ok(done)),
            Ok(Round::Again) => {}
            Err(e) => return Some(Err(e)),
        }
    }
    // What was replayed and dropped on the way is reported all the same: the
    // head no longer holds the dropped writes, and nothing later will say so
    // again.
    tracing::warn!(
        "journals: origin kept moving through {PUSH_TRIES} pushes; the next sync goes on"
    );
    Some(Ok(done))
}
