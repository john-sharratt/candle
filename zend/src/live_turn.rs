//! A reply in flight, recorded so a client that reconnects can follow it.
//!
//! A turn runs to completion whether or not anyone is listening — a closed tab,
//! a phone that dropped the connection, a page reload. Without a record of it,
//! the client that comes back finds the conversation as the daemon last stored
//! it and sees nothing of the reply until the whole turn, every tool round of
//! it, has finished. So every item the turn streams is kept here, from the first,
//! for as long as the turn runs: [`LiveTurn::follow`] replays them all and then
//! follows the rest as they arrive, exactly as the original request saw them.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::Stream;
use tokio::sync::watch;

use crate::session::StreamItem;

/// One turn's streamed items so far.
pub struct LiveTurn {
    /// The message the reply answers.
    user: String,
    /// How many recovered history entries the conversation held when the reply
    /// started. Everything past them is the reply's own sealed rounds, which a
    /// follower receives from the stream instead.
    history_before: usize,
    state: Mutex<State>,
    /// Bumped on every push and on the finish, so a follower wakes for both.
    version: watch::Sender<u64>,
}

struct State {
    items: Vec<StreamItem>,
    done: bool,
}

impl LiveTurn {
    fn new(user: String, history_before: usize) -> Self {
        Self {
            user,
            history_before,
            state: Mutex::new(State {
                items: Vec::new(),
                done: false,
            }),
            version: watch::channel(0).0,
        }
    }

    /// The message the reply answers.
    pub fn user(&self) -> &str {
        &self.user
    }

    /// History entries that predate the reply — see the field.
    pub fn history_before(&self) -> usize {
        self.history_before
    }

    /// Record the next item the turn streamed.
    pub fn push(&self, item: StreamItem) {
        self.state.lock().unwrap().items.push(item);
        self.version.send_modify(|v| *v += 1);
    }

    /// The turn is over: followers drain what is left and end.
    pub fn finish(&self) {
        self.state.lock().unwrap().done = true;
        self.version.send_modify(|v| *v += 1);
    }

    /// Every item from the turn's first, then each new one as it is pushed,
    /// ending once the turn has finished and everything has been read.
    pub fn follow(
        self: Arc<Self>,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        // Subscribed before the first read, so a push between the read and the
        // wait is seen as a change rather than missed.
        let rx = self.version.subscribe();
        Box::pin(futures::stream::unfold(
            (self, rx, 0usize),
            |(turn, mut rx, cursor)| async move {
                loop {
                    {
                        let state = turn.state.lock().unwrap();
                        if let Some(item) = state.items.get(cursor) {
                            let item = item.clone();
                            drop(state);
                            return Some((Ok(item), (turn, rx, cursor + 1)));
                        }
                        if state.done {
                            return None;
                        }
                    }
                    if rx.changed().await.is_err() {
                        return None;
                    }
                }
            },
        ))
    }
}

/// The reply in flight for each conversation that has one.
#[derive(Default)]
pub struct LiveTurns {
    turns: Mutex<HashMap<String, Arc<LiveTurn>>>,
}

impl LiveTurns {
    /// Start recording a turn for `conv_id` answering `user`, replacing any
    /// record left there.
    pub fn begin(&self, conv_id: &str, user: String, history_before: usize) -> Arc<LiveTurn> {
        let turn = Arc::new(LiveTurn::new(user, history_before));
        self.turns
            .lock()
            .unwrap()
            .insert(conv_id.to_string(), Arc::clone(&turn));
        turn
    }

    /// The turn in flight for `conv_id`, if there is one.
    pub fn get(&self, conv_id: &str) -> Option<Arc<LiveTurn>> {
        self.turns.lock().unwrap().get(conv_id).cloned()
    }

    /// Finish `turn` and stop answering for it — unless a newer turn of the
    /// same conversation has already replaced it.
    pub fn end(&self, conv_id: &str, turn: &Arc<LiveTurn>) {
        turn.finish();
        let mut turns = self.turns.lock().unwrap();
        if turns.get(conv_id).is_some_and(|t| Arc::ptr_eq(t, turn)) {
            turns.remove(conv_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn text(items: Vec<anyhow::Result<StreamItem>>) -> Vec<String> {
        items
            .into_iter()
            .map(|r| match r.unwrap() {
                StreamItem::Token(t) => t,
                StreamItem::Status(s) => format!("status:{s}"),
                _ => "other".to_string(),
            })
            .collect()
    }

    /// A follower that arrives mid-turn gets everything from the first item,
    /// then what comes after, and ends with the turn.
    #[tokio::test]
    async fn a_late_follower_replays_then_follows_to_the_end() {
        let turns = LiveTurns::default();
        let turn = turns.begin("c", "hi".into(), 4);
        assert_eq!((turn.user(), turn.history_before()), ("hi", 4));
        turn.push(StreamItem::Status("thinking".into()));
        turn.push(StreamItem::Token("Hel".into()));

        let follower = tokio::spawn(turns.get("c").unwrap().follow().collect::<Vec<_>>());
        tokio::task::yield_now().await;
        turn.push(StreamItem::Token("lo".into()));
        turns.end("c", &turn);

        assert_eq!(
            text(follower.await.unwrap()),
            ["status:thinking", "Hel", "lo"]
        );
        assert!(
            turns.get("c").is_none(),
            "a finished turn is no longer live"
        );
    }

    /// A finished turn's own record does not remove the newer turn that
    /// replaced it.
    #[test]
    fn ending_an_old_turn_leaves_its_successor_live() {
        let turns = LiveTurns::default();
        let old = turns.begin("c", "first".into(), 0);
        let new = turns.begin("c", "second".into(), 2);
        turns.end("c", &old);
        assert!(turns.get("c").is_some_and(|t| Arc::ptr_eq(&t, &new)));
    }

    /// Following a turn that has already finished yields what it streamed and
    /// stops.
    #[tokio::test]
    async fn following_a_finished_turn_yields_its_items_and_stops() {
        let turn = Arc::new(LiveTurn::new(String::new(), 0));
        turn.push(StreamItem::Token("done".into()));
        turn.finish();
        assert_eq!(text(turn.follow().collect().await), ["done"]);
    }
}
