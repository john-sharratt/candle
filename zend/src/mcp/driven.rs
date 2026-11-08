//! Whether anyone drives a task (`docs/zen_memory_and_guides.md` D5): a task
//! is `attended` when its coordinator waits on it, answers its questions and
//! checks its result, and `alone` when nobody will — Grok's, whose drive ends
//! seconds after it is spoken. The worker reads which in its system prompt, as
//! an option of the `driven` selector in `projection.yaml`, pre-sealed at
//! startup; `attended` emits nothing.

use mcp_server::Client;
use serde::{Deserialize, Serialize};

/// The `driven` selector's id in `projection.yaml`.
pub const DRIVEN_SELECTOR: &str = "driven";

/// Whether anyone drives a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Driven {
    #[default]
    Attended,
    Alone,
}

impl Driven {
    /// What a call names, or — naming none — what its client drives: Grok
    /// cannot stay with a task, every other client can.
    pub fn of(name: Option<&str>, client: Client) -> Result<Self, String> {
        match name {
            Some("attended") => Ok(Self::Attended),
            Some("alone") => Ok(Self::Alone),
            Some(other) => Err(format!("`driven` is `attended` or `alone`, not `{other}`")),
            None if client == Client::Grok => Ok(Self::Alone),
            None => Ok(Self::Attended),
        }
    }

    /// Its option of the `driven` selector.
    pub fn option(self) -> &'static str {
        match self {
            Self::Attended => "attended",
            Self::Alone => "alone",
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_yaml::Value;

    use super::*;

    /// **A call names how it is driven, or its client decides**: Grok's tasks
    /// run alone, every other client's attended; a mode nobody defined is
    /// refused.
    #[test]
    fn a_call_names_its_mode_or_its_client_decides() {
        assert_eq!(Driven::of(None, Client::Grok).unwrap(), Driven::Alone);
        assert_eq!(Driven::of(None, Client::ClaudeCode).unwrap(), Driven::Attended);
        assert_eq!(
            Driven::of(Some("attended"), Client::Grok).unwrap(),
            Driven::Attended
        );
        assert_eq!(
            Driven::of(Some("alone"), Client::ClaudeAi).unwrap(),
            Driven::Alone
        );
        assert_eq!(
            Driven::of(Some("solo"), Client::Grok).unwrap_err(),
            "`driven` is `attended` or `alone`, not `solo`"
        );
    }

    /// **`attended` emits nothing and `alone` one section**, so an attended
    /// task's prompt is unchanged to the byte.
    #[test]
    fn attended_emits_nothing_and_alone_one_section() {
        let schema: Value =
            serde_yaml::from_str(include_str!("../prompts/projection.yaml")).unwrap();
        let node = schema["system_prompt"]["items"]
            .as_sequence()
            .unwrap()
            .iter()
            .filter_map(|item| item["nodes"].as_sequence())
            .flatten()
            .find(|node| node["id"].as_str() == Some(DRIVEN_SELECTOR))
            .expect("projection.yaml declares the driven selector");
        assert_eq!(node["default"].as_str(), Some("attended"));
        let options: Vec<(&str, Option<&str>)> = node["options"]
            .as_sequence()
            .unwrap()
            .iter()
            .map(|o| (o["id"].as_str().unwrap(), o["content"].as_str()))
            .collect();
        assert_eq!(options.len(), 2);
        assert_eq!(options[0], ("attended", None));
        assert_eq!(options[1].0, "alone");
        assert!(options[1]
            .1
            .unwrap()
            .starts_with("Nobody will answer you on this task."));
    }
}
