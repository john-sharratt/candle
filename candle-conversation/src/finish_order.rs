//! Several turns in flight at once, settled in the order they finish.
//!
//! A caller that submits a batch of turns and must do something with each —
//! read its answer, seal it, free its conversation — would otherwise wait on
//! them in the order it submitted them, and a turn that finished early would
//! sit settled behind a slower one before it was. Its conversation, and on a
//! hybrid model its recurrent store, stood idle that whole time: a calibration
//! unit's 24 questions each held ~288 MiB on BlackWorld until the slowest of
//! them was done. Settled as they finish, each is handled — and its slot
//! freed — the moment its turn ends.

use std::time::Duration;

use flume::{RecvError, Selector};

use crate::cancel::CancelToken;
use crate::error::ConversationError;
use crate::handle::{TurnHandle, TurnResponse};

/// How long a wait goes between looks at the cancel latch.
const POLL: Duration = Duration::from_millis(100);

/// A turn that has finished: where it stood in the batch, its handle (for
/// `finish_turn`), and its response or the error that ended it.
pub type Finished = (usize, TurnHandle, crate::Result<TurnResponse>);

/// See the module documentation.
pub struct FinishOrder {
    /// The turns not yet finished, each with its place in the batch.
    live: Vec<(usize, TurnHandle)>,
}

impl FinishOrder {
    /// `handles`, a batch numbered by position.
    pub fn new(handles: Vec<TurnHandle>) -> Self {
        FinishOrder {
            live: handles.into_iter().enumerate().collect(),
        }
    }

    /// Turns not yet finished.
    pub fn pending(&self) -> usize {
        self.live.len()
    }

    /// Block for whichever turn finishes next; `Ok(None)` once every one has.
    /// A turn whose scheduler went away finishes with
    /// [`ConversationError::SchedulerGone`]. A graceful shutdown latching
    /// `cancel` abandons the wait with [`ConversationError::IngestCancelled`],
    /// leaving the unfinished turns here for the caller to drop.
    pub fn next_cancellable(&mut self, cancel: &CancelToken) -> crate::Result<Option<Finished>> {
        loop {
            if self.live.is_empty() {
                return Ok(None);
            }
            let got = {
                let mut select = Selector::new();
                for (k, (_, handle)) in self.live.iter().enumerate() {
                    select = select.recv(handle.receiver(), move |event| (k, event));
                }
                select.wait_timeout(POLL)
            };
            let settled = match got {
                Ok((k, Ok(event))) => TurnHandle::settle(event).map(|result| (k, result)),
                Ok((k, Err(RecvError::Disconnected))) => {
                    Some((k, Err(ConversationError::SchedulerGone)))
                }
                Err(_) if cancel.is_cancelled() => return Err(ConversationError::IngestCancelled),
                Err(_) => None,
            };
            if let Some((k, result)) = settled {
                let (index, handle) = self.live.remove(k);
                return Ok(Some((index, handle, result)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::response as turn_response;
    use crate::handle::TurnEvent;

    fn batch(n: usize) -> (Vec<flume::Sender<TurnEvent>>, FinishOrder) {
        let (senders, handles): (Vec<_>, Vec<_>) = (0..n)
            .map(|_| {
                let (tx, rx) = flume::unbounded();
                (tx, TurnHandle::new(rx))
            })
            .unzip();
        (senders, FinishOrder::new(handles))
    }

    fn next_index(order: &mut FinishOrder) -> (usize, Result<String, String>) {
        let (index, _, result) = order
            .next_cancellable(&CancelToken::new())
            .expect("not cancelled")
            .expect("a turn left");
        (index, result.map(|r| r.text).map_err(|e| e.to_string()))
    }

    /// **Turns settle in the order they finish, not the order they were
    /// submitted**, each by its place in the batch; a token on the way is not
    /// a finish; a turn whose scheduler drops it finishes as gone; and an
    /// empty set says so.
    #[test]
    fn turns_settle_in_the_order_they_finish() {
        let (tx, mut order) = batch(3);
        tx[2].send(TurnEvent::Token(9)).unwrap();
        tx[2].send(TurnEvent::Done(turn_response("third"))).unwrap();
        assert_eq!(next_index(&mut order), (2, Ok("third".into())));

        tx[0].send(TurnEvent::Token(1)).unwrap();
        tx[0].send(TurnEvent::Done(turn_response("first"))).unwrap();
        assert_eq!(next_index(&mut order), (0, Ok("first".into())));
        assert_eq!(order.pending(), 1);

        drop(tx.into_iter().nth(1));
        let (index, gone) = next_index(&mut order);
        assert_eq!(index, 1);
        assert_eq!(gone, Err(ConversationError::SchedulerGone.to_string()));
        assert!(order
            .next_cancellable(&CancelToken::new())
            .unwrap()
            .is_none());
    }

    /// **A shutdown abandons the wait and leaves the turns** for the caller to
    /// drop; a turn's own error settles that turn alone.
    #[test]
    fn a_shutdown_abandons_the_wait() {
        let (tx, mut order) = batch(2);
        tx[1]
            .send(TurnEvent::Error(ConversationError::Channel("boom".into())))
            .unwrap();
        assert_eq!(
            next_index(&mut order),
            (
                1,
                Err(ConversationError::Channel("boom".into()).to_string())
            )
        );
        let cancel = CancelToken::new();
        cancel.request();
        assert!(matches!(
            order.next_cancellable(&cancel),
            Err(ConversationError::IngestCancelled)
        ));
        assert_eq!(order.pending(), 1);
        drop(tx);
    }
}
