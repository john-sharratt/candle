//! Which client an MCP session is for, as its `clientInfo.name` says — which
//! bounds how long a call from it may wait, and picks the guide a server
//! gives it (`docs/mcp_user_context.md` §2.6).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Client {
    /// Claude Code: reached through the claude.ai connector, whose calls are
    /// given up after about 300 s with nothing delivered, so a wait holds four
    /// minutes and is called again.
    ClaudeCode,
    /// xAI's Grok — in its apps and in the car — whose answers are spoken as
    /// often as read. Its calls wait as `Other`'s do.
    Grok,
    /// claude.ai, which names itself `claude-ai`: a call is cut off at 240 s.
    ClaudeAi,
    /// Claude Desktop, or anyone else: a call is cut off at 240 s.
    Other,
}

impl Client {
    /// The client an `initialize` names: by its `clientInfo.name`, or, when
    /// that names no client this knows, by its `User-Agent` — Grok calls
    /// itself `connectors-manager` and says it is Grok only in its agent,
    /// `grok-connectors-manager/0.1.0`.
    pub fn of_hello(name: &str, agent: &str) -> Self {
        match Self::of(name) {
            Self::Other => Self::of(agent),
            known => known,
        }
    }

    pub fn of(name: &str) -> Self {
        let name = name.to_ascii_lowercase();
        if name.contains("claude-code") || name.contains("claude code") {
            Self::ClaudeCode
        } else if name.contains("grok") || name.contains("xai") {
            Self::Grok
        } else if name.contains("claude-ai") || name.contains("claude.ai") {
            Self::ClaudeAi
        } else {
            Self::Other
        }
    }

    /// The longest a call from this client may wait, in seconds.
    pub fn wait_ceiling(self) -> u64 {
        match self {
            Self::ClaudeCode => 240,
            Self::Grok | Self::ClaudeAi | Self::Other => 200,
        }
    }

    /// The name a server that calls on to another names this client by, which
    /// [`Client::of`] reads back as the same client.
    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Grok => "grok",
            Self::ClaudeAi => "claude-ai",
            Self::Other => "other",
        }
    }

    pub(crate) fn slug(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude_code",
            Self::Grok => "grok",
            Self::ClaudeAi => "claude_ai",
            Self::Other => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_client_bounds_the_wait() {
        assert_eq!(Client::of("claude-code").wait_ceiling(), 240);
        assert_eq!(Client::of("Claude Code").wait_ceiling(), 240);
        assert_eq!(Client::of("claude-ai").wait_ceiling(), 200);
        assert_eq!(Client::of("grok").wait_ceiling(), 200);
        assert_eq!(Client::of("anything").wait_ceiling(), 200);
    }

    /// **Grok is known by either of its names**, in any case; nothing else is
    /// taken for it.
    #[test]
    fn grok_is_known_by_its_name_or_its_makers() {
        assert_eq!(Client::of("Grok"), Client::Grok);
        assert_eq!(Client::of("xai-mcp-client"), Client::Grok);
        assert_eq!(Client::of(""), Client::Other);
    }

    /// **claude.ai is known by its name**; Claude Desktop and the rest are
    /// `Other`.
    #[test]
    fn claude_ai_is_its_own_client() {
        assert_eq!(Client::of("claude-ai"), Client::ClaudeAi);
        assert_eq!(Client::of("claude.ai"), Client::ClaudeAi);
        assert_eq!(Client::of("claude-code"), Client::ClaudeCode);
        assert_eq!(Client::of("Claude Desktop"), Client::Other);
    }

    /// **Grok's own handshake is known by its agent**: it names itself
    /// `connectors-manager`, which says nothing, and Grok in its
    /// `User-Agent`. A name that does say is believed over the agent.
    #[test]
    fn a_handshake_is_known_by_its_name_or_else_its_agent() {
        assert_eq!(
            Client::of_hello("connectors-manager", "grok-connectors-manager/0.1.0"),
            Client::Grok
        );
        assert_eq!(
            Client::of_hello("claude-code", "grok-connectors-manager/0.1.0"),
            Client::ClaudeCode
        );
        assert_eq!(
            Client::of_hello("claude-ai", "python-httpx/0.27"),
            Client::ClaudeAi
        );
        assert_eq!(Client::of_hello("connectors-manager", ""), Client::Other);
    }

    #[test]
    fn a_client_passed_on_by_name_is_read_back_as_itself() {
        for c in [
            Client::ClaudeCode,
            Client::Grok,
            Client::ClaudeAi,
            Client::Other,
        ] {
            assert_eq!(Client::of(c.name()), c);
        }
    }
}
