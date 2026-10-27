//! Submitting a draw and withdrawing it (`docs/guest_draw_orchestration.md`
//! §2): one seam, [`GuestDraw`], over the engine's guest queue, and a fake in
//! tests.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use candle_conversation::guest::{GuestError, GuestOutcome, GuestRequest, GuestSink};

use crate::api::Authored;
use crate::guest_routes::{cancel, submit_watched};

/// A guest job's outcome, to be awaited.
pub type Outcome = Pin<Box<dyn Future<Output = Result<GuestOutcome, GuestError>> + Send>>;

/// A submitted job: its number, to withdraw it by, and its outcome.
pub struct Submitted {
    pub seq: u64,
    pub outcome: Outcome,
}

/// The engine's guest queue, as the orchestration uses it.
pub trait GuestDraw {
    /// Queue `request`, reporting progress to `sink`.
    fn submit(&self, request: GuestRequest, sink: GuestSink) -> Result<Submitted, GuestError>;

    /// Withdraw job `seq` while it is still queued; `false` once a drain has
    /// taken it.
    fn cancel(&self, seq: u64) -> bool;
}

/// This member's engine, through [`submit_watched`] and [`cancel`].
#[derive(Clone)]
pub struct Engine(pub Arc<Authored>);

/// A shared handle on a guest draws as the guest does — what lets a withdrawal
/// outlive the call that refused it.
impl<T: GuestDraw> GuestDraw for Arc<T> {
    fn submit(&self, request: GuestRequest, sink: GuestSink) -> Result<Submitted, GuestError> {
        T::submit(self, request, sink)
    }

    fn cancel(&self, seq: u64) -> bool {
        T::cancel(self, seq)
    }
}

impl GuestDraw for Engine {
    fn submit(&self, request: GuestRequest, sink: GuestSink) -> Result<Submitted, GuestError> {
        let receipt = submit_watched(&self.0, request, sink)?;
        Ok(Submitted {
            seq: receipt.seq,
            outcome: Box::pin(receipt.wait_async()),
        })
    }

    fn cancel(&self, seq: u64) -> bool {
        cancel(&self.0, seq)
    }
}
