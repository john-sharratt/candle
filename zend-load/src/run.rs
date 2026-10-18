//! One scenario run: its clients at once, each holding its own conversation
//! through its turns one after another, every turn sampled.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::task::JoinSet;
use tokio::time::Instant;

use crate::client::{turn, Target};
use crate::ladder::Scenario;
use crate::sample::{Outcome, Sample};

/// What a run produced: its samples, its wall time, and the conversation
/// ids the cleanup deletes.
pub struct Ran {
    pub samples: Vec<Sample>,
    pub wall: Duration,
    pub conversations: Vec<String>,
}

/// Run `scenario` under the run name `run`, each turn given at most `limit`.
pub async fn run(target: Arc<Target>, scenario: &Scenario, run: &str, limit: Duration) -> Ran {
    let started = Instant::now();
    let mut clients = JoinSet::new();
    for client in 0..scenario.clients {
        let target = Arc::clone(&target);
        let scenario = scenario.clone();
        let run = run.to_string();
        clients.spawn(async move {
            let mut history = Vec::new();
            let mut samples = Vec::new();
            let mut conversation = String::new();
            for t in 0..scenario.turns {
                let (body, conv) =
                    scenario
                        .task
                        .body(&run, client, t, &history, scenario.max_tokens);
                conversation = conv;
                let (mut sample, content) = turn(&target, &scenario.name, &body, limit).await;
                sample.client = client;
                sample.turn = t;
                let answered = sample.outcome == Outcome::Answered;
                samples.push(sample);
                if !answered {
                    break;
                }
                let asked = body["messages"]
                    .as_array()
                    .and_then(|m| m.last())
                    .cloned()
                    .unwrap_or_default();
                history.push(asked);
                history.push(json!({"role": "assistant", "content": content}));
            }
            (samples, conversation)
        });
    }
    let mut samples = Vec::new();
    let mut conversations = Vec::new();
    while let Some(joined) = clients.join_next().await {
        if let Ok((s, conv)) = joined {
            samples.extend(s);
            if !conv.is_empty() {
                conversations.push(conv);
            }
        }
    }
    Ran {
        samples,
        wall: started.elapsed(),
        conversations,
    }
}
