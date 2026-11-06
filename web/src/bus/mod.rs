//! The gateway's pub/sub bus (`docs/mcp_file_tools.md` §5): a publisher sends
//! a message on a topic and every subscriber to that topic gets it. Nothing
//! is kept — a subscriber that was not connected missed it, and resynchronises
//! from the source of truth when it (re)connects. A subscriber that falls
//! [`QUEUE`] messages behind is dropped, not buffered without bound.
//!
//! | Module | |
//! |---|---|
//! | [`topic`] | which topics a pattern admits |
//! | [`socket`] | `GET /gateway/events`, the WebSocket services subscribe on |
//! | [`http`] | `POST /gateway/events/publish` and `/request`, for members and services |
//!
//! A request ([`Bus::request`]) is a publish that waits for one answer on a
//! reply topic: how zend asks a repository's gate before a branch moves
//! (§5.4).

pub mod http;
pub mod socket;
pub mod topic;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::timeout;

/// How many messages a subscriber may fall behind before it is dropped.
pub const QUEUE: usize = 256;

/// One message as subscribers receive it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Message {
    pub topic: String,
    pub payload: Value,
    pub at_ms: u64,
}

struct Subscriber {
    id: u64,
    patterns: Vec<String>,
    tx: mpsc::Sender<Message>,
}

/// Why a request had no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unanswered {
    /// Nothing was subscribed to the topic, so nobody could answer.
    NoSubscriber,
    /// Somebody received it and did not answer in time.
    Timeout,
}

#[derive(Default)]
pub struct Bus {
    subscribers: Mutex<Vec<Subscriber>>,
    next: AtomicU64,
}

/// A subscription: its messages, and its id for adding topics or ending it.
pub struct Subscription {
    pub id: u64,
    pub messages: mpsc::Receiver<Message>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

impl Bus {
    pub fn new() -> Bus {
        Bus::default()
    }

    fn subscribers(&self) -> MutexGuard<'_, Vec<Subscriber>> {
        self.subscribers.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Subscribe to `patterns` (`topic::admits`).
    pub fn subscribe(&self, patterns: Vec<String>) -> Subscription {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, messages) = mpsc::channel(QUEUE);
        self.subscribers().push(Subscriber { id, patterns, tx });
        Subscription { id, messages }
    }

    /// Add `patterns` to subscription `id`; `false` when it has ended.
    pub fn extend(&self, id: u64, patterns: &[String]) -> bool {
        match self.subscribers().iter_mut().find(|s| s.id == id) {
            Some(s) => {
                s.patterns.extend(patterns.iter().cloned());
                true
            }
            None => false,
        }
    }

    /// End subscription `id`.
    pub fn unsubscribe(&self, id: u64) {
        self.subscribers().retain(|s| s.id != id);
    }

    /// Send `payload` on `topic` to every subscriber to it, answering how many
    /// it reached. A subscriber whose queue is full or whose receiver is gone
    /// is dropped.
    pub fn publish(&self, topic: &str, payload: Value) -> usize {
        let message = Message {
            topic: topic.to_string(),
            payload,
            at_ms: now_ms(),
        };
        let mut reached = 0;
        self.subscribers().retain(|s| {
            if !s.patterns.iter().any(|p| topic::admits(p, topic)) {
                return true;
            }
            match s.tx.try_send(message.clone()) {
                Ok(()) => {
                    reached += 1;
                    true
                }
                Err(_) => {
                    tracing::info!(
                        subscriber = s.id,
                        topic,
                        "bus: a subscriber fell behind or left, dropped"
                    );
                    false
                }
            }
        });
        reached
    }

    /// Publish `payload` on `topic` and wait at most `wait` for the first
    /// message on `reply`, whose payload is the answer. Subscribed to `reply`
    /// before publishing, so an answer cannot arrive unseen.
    pub async fn request(
        &self,
        topic: &str,
        payload: Value,
        reply: &str,
        wait: Duration,
    ) -> Result<Value, Unanswered> {
        // Only a subscriber naming the topic itself is one that answers it: a
        // wildcard watcher hears requests and answers none, and counting it
        // would turn a gateway with no answerer into a timeout, keeping the
        // asker from the gateway that has one.
        let answerable = self
            .subscribers()
            .iter()
            .any(|s| s.patterns.iter().any(|p| p == topic));
        if !answerable {
            return Err(Unanswered::NoSubscriber);
        }
        let mut sub = self.subscribe(vec![reply.to_string()]);
        let answer = if self.publish(topic, payload) == 0 {
            Err(Unanswered::NoSubscriber)
        } else {
            match timeout(wait, sub.messages.recv()).await {
                Ok(Some(m)) => Ok(m.payload),
                Ok(None) | Err(_) => Err(Unanswered::Timeout),
            }
        };
        self.unsubscribe(sub.id);
        answer
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn a_message_reaches_the_subscribers_to_its_topic_only() {
        let bus = Bus::new();
        let mut yen = bus.subscribe(vec!["repo/yenbot".into()]);
        let mut all = bus.subscribe(vec!["repo/*".into()]);
        let mut other = bus.subscribe(vec!["gate/yenbot".into()]);
        assert_eq!(bus.publish("repo/yenbot", json!({"commit": "4f1c"})), 2);
        assert_eq!(
            yen.messages.recv().await.unwrap().payload,
            json!({"commit": "4f1c"})
        );
        assert_eq!(all.messages.recv().await.unwrap().topic, "repo/yenbot");
        assert!(other.messages.try_recv().is_err());
        assert_eq!(bus.publish("repo/candle", json!({})), 1);
    }

    #[tokio::test]
    async fn a_subscription_grows_and_ends() {
        let bus = Bus::new();
        let mut sub = bus.subscribe(vec![]);
        assert_eq!(bus.publish("a", json!(1)), 0);
        assert!(bus.extend(sub.id, &["a".into()]));
        assert_eq!(bus.publish("a", json!(2)), 1);
        assert_eq!(sub.messages.recv().await.unwrap().payload, json!(2));
        bus.unsubscribe(sub.id);
        assert_eq!(bus.publish("a", json!(3)), 0);
        assert!(!bus.extend(sub.id, &["b".into()]));
    }

    /// **A slow subscriber is dropped, not buffered without bound.**
    #[tokio::test]
    async fn a_subscriber_that_falls_behind_is_dropped() {
        let bus = Bus::new();
        let _slow = bus.subscribe(vec!["t".into()]);
        for _ in 0..QUEUE {
            assert_eq!(bus.publish("t", json!(0)), 1);
        }
        assert_eq!(bus.publish("t", json!(0)), 0, "the full queue is dropped");
        assert_eq!(bus.publish("t", json!(0)), 0, "and stays gone");
    }

    #[tokio::test]
    async fn a_subscriber_that_left_is_dropped() {
        let bus = Bus::new();
        let sub = bus.subscribe(vec!["t".into()]);
        drop(sub);
        assert_eq!(bus.publish("t", json!(0)), 0);
    }

    /// **A request is answered by the first message on its reply topic.**
    #[tokio::test]
    async fn a_request_takes_the_first_answer() {
        let bus = Arc::new(Bus::new());
        let mut gate = bus.subscribe(vec!["gate/yenbot".into()]);
        let answerer = {
            let bus = bus.clone();
            tokio::spawn(async move {
                let ask = gate.messages.recv().await.unwrap();
                let id = ask.payload["id"].as_str().unwrap().to_string();
                bus.publish(&format!("gate/yenbot/result/{id}"), json!({"ok": true}));
                bus.publish(&format!("gate/yenbot/result/{id}"), json!({"ok": false}));
            })
        };
        let answer = bus
            .request(
                "gate/yenbot",
                json!({"id": "01JD"}),
                "gate/yenbot/result/01JD",
                Duration::from_secs(5),
            )
            .await;
        answerer.await.unwrap();
        assert_eq!(answer, Ok(json!({"ok": true})));
    }

    #[tokio::test]
    async fn a_request_nobody_hears_or_answers_is_unanswered() {
        let bus = Bus::new();
        assert_eq!(
            bus.request(
                "gate/x",
                json!({}),
                "gate/x/result/1",
                Duration::from_secs(5)
            )
            .await,
            Err(Unanswered::NoSubscriber)
        );
        // A wildcard watcher hears requests but answers none: still nobody
        // to ask, at once, not after the wait.
        let mut watcher = bus.subscribe(vec!["gate/*".into()]);
        assert_eq!(
            bus.request(
                "gate/x",
                json!({}),
                "gate/x/result/1",
                Duration::from_secs(5)
            )
            .await,
            Err(Unanswered::NoSubscriber)
        );
        assert!(watcher.messages.try_recv().is_err(), "and nothing was sent");
        let _deaf = bus.subscribe(vec!["gate/x".into()]);
        assert_eq!(
            bus.request(
                "gate/x",
                json!({}),
                "gate/x/result/1",
                Duration::from_millis(20)
            )
            .await,
            Err(Unanswered::Timeout)
        );
    }
}
