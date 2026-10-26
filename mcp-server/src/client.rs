//! Which client an MCP session is for, as its `clientInfo.name` says — which
//! bounds how long a call from it may wait.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Client {
    /// Claude Code: a call past two minutes moves to the background, and the
    /// server entry sets a long timeout, so a wait may hold an hour.
    ClaudeCode,
    /// claude.ai, Claude Desktop, or anyone else: a call is cut off at 240 s.
    Other,
}

impl Client {
    pub fn of(name: &str) -> Self {
        let name = name.to_ascii_lowercase();
        if name.contains("claude-code") || name.contains("claude code") {
            Self::ClaudeCode
        } else {
            Self::Other
        }
    }

    /// The longest a call from this client may wait, in seconds.
    pub fn wait_ceiling(self) -> u64 {
        match self {
            Self::ClaudeCode => 3_600,
            Self::Other => 200,
        }
    }

    /// The name a server that calls on to another names this client by, which
    /// [`Client::of`] reads back as the same client.
    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Other => "other",
        }
    }

    pub(crate) fn slug(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude_code",
            Self::Other => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_client_bounds_the_wait() {
        assert_eq!(Client::of("claude-code").wait_ceiling(), 3_600);
        assert_eq!(Client::of("Claude Code").wait_ceiling(), 3_600);
        assert_eq!(Client::of("claude-ai").wait_ceiling(), 200);
        assert_eq!(Client::of("anything").wait_ceiling(), 200);
    }

    #[test]
    fn a_client_passed_on_by_name_is_read_back_as_itself() {
        for c in [Client::ClaudeCode, Client::Other] {
            assert_eq!(Client::of(c.name()), c);
        }
    }
}
