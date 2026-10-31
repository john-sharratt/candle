//! One thing the bridge did, as the trace keeps it and the log says it
//! (`docs/npcd_discord.md` §5.5): when, which way, what kind, the Discord and
//! world ids it touched, and what came of it. Never the text of a message,
//! never a token.

use serde::Serialize;

/// Which way something went.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// From Discord into the world.
    In,
    /// From the world to Discord.
    Out,
    /// The bot's gateway connection.
    Shard,
}

impl Direction {
    fn name(self) -> &'static str {
        match self {
            Direction::In => "in",
            Direction::Out => "out",
            Direction::Shard => "shard",
        }
    }
}

/// One event of the bridge's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Note {
    pub at_ms: u64,
    pub direction: Direction,
    /// `dm`, `channel`, `post`, `ready`, …
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guild: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<usize>,
    /// What came of it: `applied`, `echo`, `posted`, `failed 403/50013`, …
    pub outcome: String,
}

impl Note {
    /// A `kind` event going `direction`, at `at_ms`, that came to `outcome`.
    pub fn new(at_ms: u64, direction: Direction, kind: &str, outcome: &str) -> Self {
        Self {
            at_ms,
            direction,
            kind: kind.into(),
            message: None,
            guild: None,
            channel: None,
            author: None,
            thread: None,
            seq: None,
            parts: None,
            outcome: outcome.into(),
        }
    }

    pub fn message(mut self, id: u64) -> Self {
        self.message = Some(id);
        self
    }

    pub fn guild(mut self, id: u64) -> Self {
        self.guild = Some(id);
        self
    }

    pub fn channel(mut self, id: u64) -> Self {
        self.channel = Some(id);
        self
    }

    pub fn author(mut self, id: u64) -> Self {
        self.author = Some(id);
        self
    }

    pub fn thread(mut self, id: &str) -> Self {
        self.thread = Some(id.into());
        self
    }

    pub fn seq(mut self, seq: u64) -> Self {
        self.seq = Some(seq);
        self
    }

    pub fn parts(mut self, parts: usize) -> Self {
        self.parts = Some(parts);
        self
    }

    /// The event on one line, without its time: what the log says and the
    /// trace's text shows.
    pub fn line(&self) -> String {
        let mut line = format!("{} {}", self.direction.name(), self.kind);
        let ids = [
            ("message", self.message),
            ("guild", self.guild),
            ("channel", self.channel),
            ("author", self.author),
        ];
        for (name, id) in ids {
            if let Some(id) = id {
                line.push_str(&format!(" {name}={id}"));
            }
        }
        if let Some(thread) = &self.thread {
            line.push_str(&format!(" thread={thread}"));
        }
        if let Some(seq) = self.seq {
            line.push_str(&format!(" seq={seq}"));
        }
        if let Some(parts) = self.parts {
            line.push_str(&format!(" parts={parts}"));
        }
        format!("{line}: {}", self.outcome)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A note is these ids and its outcome, on one line and as JSON**,
    /// leaving out what it does not name.
    #[test]
    fn a_note_reads_on_one_line_and_as_json() {
        let applied = Note::new(1_000, Direction::In, "dm", "applied")
            .message(901)
            .channel(31)
            .author(77)
            .thread("th1")
            .seq(2);
        assert_eq!(
            applied.line(),
            "in dm message=901 channel=31 author=77 thread=th1 seq=2: applied"
        );
        assert_eq!(
            serde_json::to_value(&applied).unwrap(),
            json!({
                "at_ms": 1_000, "direction": "in", "kind": "dm", "message": 901,
                "channel": 31, "author": 77, "thread": "th1", "seq": 2, "outcome": "applied",
            })
        );
        let ready = Note::new(5, Direction::Shard, "ready", "online");
        assert_eq!(ready.line(), "shard ready: online");
        assert_eq!(
            serde_json::to_value(&ready).unwrap(),
            json!({ "at_ms": 5, "direction": "shard", "kind": "ready", "outcome": "online" })
        );
    }
}
