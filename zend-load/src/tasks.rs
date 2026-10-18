//! What a client asks. **Decode** is the engine alone: a `passthrough` turn
//! with a short opening and a fixed decode, no projection and no tools.
//! **Read** is zend itself: a `zen-code` question about the candle repository
//! that it answers by reading code with its tools — the work a developer
//! actually brings it. **Write** is zend changing code: an edit to a
//! battle-cities crate, never candle, made in the conversation's overlay with
//! `file_edit` and shown as a diff — told never to commit or push, so nothing
//! reaches a branch or origin.

use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Decode,
    Read,
    Write,
}

/// The battle-cities crates a write task edits, one per client in turn.
const CRATES: [&str; 9] = [
    "math", "quadtree", "chat", "conf", "dmo", "llm", "lore", "recall", "sim",
];

/// What every write turn is told, so nothing it does leaves the overlay.
const WRITE_RULES: &str = "Work only in the battle-cities repository, only on the file named, \
     with file_edit. Do not commit, push, create branches, or touch any other file. \
     Finish by showing the diff of what you changed.";

/// Write follow-ups, asked after the first edit.
const WRITE_FOLLOW_UPS: [&str; 2] = [
    "Now add a unit test for that function in the same file.",
    "Review your change and fix anything that would not compile.",
];

/// Read questions: each needs the repository read to answer, and none needs
/// anything written.
const READS: [&str; 8] = [
    "Where does the gateway decide which cluster member a new conversation goes to, and what does it weigh?",
    "How does keep decide whether a trial build becomes the stable one? Name the file and the timeout.",
    "What does the ingest lease protect against, and how long can one reading hold it?",
    "Explain how a lagging cluster member is detected and how it recovers.",
    "Which function turns a non-streamed chat turn into its JSON response, and what does it do with a failure?",
    "How does a catch-up decide that a record it fetched short should be fetched again?",
    "What is SYNC_SLACK for, and where is it used?",
    "How is the CPU expert pool told which experts to compute for a layer?",
];

/// Follow-up turns, asked in the same conversation after the first answer.
const FOLLOW_UPS: [&str; 2] = [
    "Show the most important lines of that code and explain them.",
    "What would break if that logic were removed? Be specific.",
];

/// What a run's decode conversations open with: the run, then the client.
pub fn mark(run: &str, client: usize) -> String {
    format!("{} client {client}]", run_mark(run))
}

/// The part of [`mark`] every conversation of the run shares.
pub fn run_mark(run: &str) -> String {
    format!("[zend-load {run}")
}

impl Task {
    pub fn parse(name: &str) -> Option<Task> {
        match name {
            "decode" => Some(Task::Decode),
            "read" => Some(Task::Read),
            "write" => Some(Task::Write),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Task::Decode => "decode",
            Task::Read => "read",
            Task::Write => "write",
        }
    }

    pub fn max_tokens(self) -> usize {
        match self {
            Task::Decode => 256,
            Task::Read => 1024,
            Task::Write => 2048,
        }
    }

    /// The request body for `client`'s `turn` (from 0) in the run `run`, and
    /// the conversation id it names, which the cleanup deletes. A decode
    /// conversation is keyed by its opening, which carries the run and client
    /// so no two clients share one; each later turn sends the history so far.
    pub fn body(
        self,
        run: &str,
        client: usize,
        turn: usize,
        history: &[Value],
        max_tokens: usize,
    ) -> (Value, String) {
        match self {
            Task::Decode => {
                let mut messages = vec![json!({
                    "role": "system",
                    "content": "You are a storyteller.",
                })];
                messages.extend(history.iter().cloned());
                // The conversation's label is its first user message, so the
                // run's mark goes there for the cleanup to find.
                let ask = if turn == 0 {
                    format!(
                        "{} Tell a story about lighthouse keeper number {client}.",
                        mark(run, client)
                    )
                } else {
                    "Continue the story.".to_string()
                };
                messages.push(json!({"role": "user", "content": ask}));
                let body = json!({
                    "model": "passthrough",
                    "stream": true,
                    "max_tokens": max_tokens,
                    "messages": messages,
                });
                (body, String::new())
            }
            Task::Read => {
                let conv_id = format!("{run}-c{client}");
                let ask = if turn == 0 {
                    READS[client % READS.len()].to_string()
                } else {
                    FOLLOW_UPS[(turn - 1) % FOLLOW_UPS.len()].to_string()
                };
                let body = json!({
                    "model": "zen-code",
                    "conv_id": conv_id,
                    "stream": true,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": ask}],
                });
                (body, conv_id)
            }
            Task::Write => {
                let conv_id = format!("{run}-w{client}");
                let krate = CRATES[client % CRATES.len()];
                let ask = if turn == 0 {
                    format!(
                        "In battle-cities, open crates/{krate}/src/lib.rs and add a one-line \
                         doc comment above its first public function saying what it does. \
                         {WRITE_RULES}"
                    )
                } else {
                    format!(
                        "{} {WRITE_RULES}",
                        WRITE_FOLLOW_UPS[(turn - 1) % WRITE_FOLLOW_UPS.len()]
                    )
                };
                let body = json!({
                    "model": "zen-code",
                    "conv_id": conv_id,
                    "stream": true,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": ask}],
                });
                (body, conv_id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read turn names the client's own conversation and asks its
    /// question, then a follow-up; a decode turn opens with the run and
    /// client so each is its own conversation, and carries its history.
    #[test]
    fn bodies_name_their_conversation() {
        let (body, conv) = Task::Read.body("zl-1", 3, 0, &[], 1024);
        assert_eq!(conv, "zl-1-c3");
        assert_eq!(body["conv_id"], "zl-1-c3");
        assert_eq!(body["model"], "zen-code");
        assert_eq!(body["messages"][0]["content"], READS[3]);
        let (follow, _) = Task::Read.body("zl-1", 3, 1, &[], 1024);
        assert_eq!(follow["messages"][0]["content"], FOLLOW_UPS[0]);

        let (opening, none) = Task::Decode.body("zl-1", 5, 0, &[], 256);
        assert!(none.is_empty());
        assert_eq!(opening["model"], "passthrough");
        assert_eq!(
            opening["messages"][1]["content"],
            "[zend-load zl-1 client 5] Tell a story about lighthouse keeper number 5."
        );
        let said = vec![
            opening["messages"][1].clone(),
            json!({"role": "assistant", "content": "Once."}),
        ];
        let (decode, _) = Task::Decode.body("zl-1", 5, 1, &said, 256);
        let messages = decode["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["content"], "You are a storyteller.");
        assert_eq!(messages[3]["content"], "Continue the story.");
        assert!(mark("zl-1", 5).starts_with(&run_mark("zl-1")));
    }

    /// A write turn names its own conversation, edits a battle-cities crate —
    /// never candle — and carries the no-commit, no-push rules every turn.
    #[test]
    fn a_write_turn_stays_in_battle_cities_and_never_commits() {
        let (first, conv) = Task::Write.body("zl-2", 1, 0, &[], 2048);
        assert_eq!(conv, "zl-2-w1");
        let ask = first["messages"][0]["content"].as_str().unwrap();
        assert!(ask.contains("battle-cities, open crates/quadtree/src/lib.rs"));
        assert!(!ask.contains("candle"));
        assert!(ask.ends_with(WRITE_RULES));
        let (next, _) = Task::Write.body("zl-2", 1, 1, &[], 2048);
        let ask = next["messages"][0]["content"].as_str().unwrap();
        assert!(ask.starts_with(WRITE_FOLLOW_UPS[0]));
        assert!(ask.contains("Do not commit, push"));
        assert_eq!(Task::parse("write"), Some(Task::Write));
    }
}
