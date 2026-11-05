//! What kind of work a worker does (`docs/claude_zend_gateway.md` §5.1).
//!
//! A task names its kind when it starts. The kind
//! selects an option of the `role` selector in `projection.yaml`: every option
//! is pre-sealed at startup, so a worker's first round already reads its
//! document and nothing is prefilled for it. A conversation that is no worker
//! takes the selector's default, `generic`, which emits nothing.
//!
//! [`WorkerKind::Chat`] is no task's kind: it is Zen's side of a session's chat
//! (`docs/zen_chat.md`), made by `zen_chat_send` alone, so `repo_task_start`
//! cannot ask for it.

use serde::{Deserialize, Serialize};

/// The `role` selector's id in `projection.yaml`.
pub const ROLE_SELECTOR: &str = "role";

/// The kind of work a task does: `coder` unless the call asks otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKind {
    #[default]
    Coder,
    Researcher,
    Reviewer,
    /// Zen answering Claude on a session's chat: its message is the text as
    /// sent, its answer is ready when its turn ends, and nothing of it is kept
    /// on a thread.
    Chat,
}

impl WorkerKind {
    /// Whether it is Zen's side of a chat rather than a task.
    pub fn is_chat(self) -> bool {
        self == Self::Chat
    }

    /// The kind a call names, `coder` when it names none.
    pub fn of(name: Option<&str>) -> Result<Self, String> {
        match name {
            None | Some("coder") => Ok(Self::Coder),
            Some("researcher") => Ok(Self::Researcher),
            Some("reviewer") => Ok(Self::Reviewer),
            Some(other) => Err(format!(
                "`role` is `coder`, `researcher` or `reviewer`, not `{other}`"
            )),
        }
    }

    /// Its option of the `role` selector.
    pub fn option(self) -> &'static str {
        match self {
            Self::Coder => "coder",
            Self::Researcher => "researcher",
            Self::Reviewer => "reviewer",
            Self::Chat => "chat",
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::models::Dialect;
    use candle_conversation::projection::{Builder, SystemPromptItem};
    use serde_yaml::Value;

    use super::*;

    const KINDS: [WorkerKind; 4] = [
        WorkerKind::Coder,
        WorkerKind::Researcher,
        WorkerKind::Reviewer,
        WorkerKind::Chat,
    ];

    /// The `role` selector's options as `projection.yaml` declares them.
    fn options() -> Vec<(String, Option<String>)> {
        let schema: Value =
            serde_yaml::from_str(include_str!("../prompts/projection.yaml")).unwrap();
        let node = schema["system_prompt"]["items"]
            .as_sequence()
            .unwrap()
            .iter()
            .filter_map(|item| item["nodes"].as_sequence())
            .flatten()
            .find(|node| node["id"].as_str() == Some(ROLE_SELECTOR))
            .expect("projection.yaml declares the role selector");
        assert_eq!(node["default"].as_str(), Some("generic"));
        node["options"]
            .as_sequence()
            .unwrap()
            .iter()
            .map(|o| {
                (
                    o["id"].as_str().unwrap().to_string(),
                    o["content"].as_str().map(str::to_string),
                )
            })
            .collect()
    }

    /// **Every role is one pre-sealed section**: the daemon's own parse gives
    /// each kind's option exactly one sealed variant, `generic` none, and the
    /// tree's default branch emits nothing, so what follows it seals unchanged.
    #[test]
    fn the_schema_seals_one_section_a_role_and_none_by_default() {
        let dialect = Dialect::chat_ml();
        let builder = Builder::from_yaml_with_vars_and_dialect(
            include_str!("../prompts/projection.yaml"),
            &[("workspace", "test")],
            Some(&dialect),
        )
        .expect("projection.yaml must parse");
        let tree = builder
            .schema()
            .system_prompt
            .items
            .iter()
            .find_map(|item| match item {
                SystemPromptItem::SectionTree(t)
                    if t.nodes.iter().any(|n| n.name == ROLE_SELECTOR) =>
                {
                    Some(t)
                }
                _ => None,
            })
            .expect("the role selector is a tree");
        assert_eq!(tree.nodes.len(), 1, "a tree of its own multiplies no dial");
        assert!(tree.default_present_ids.is_empty());
        let variants: Vec<(&str, usize)> = tree.nodes[0]
            .options
            .iter()
            .map(|o| (o.id.as_str(), o.variants.len()))
            .collect();
        assert_eq!(
            variants,
            [
                ("generic", 0),
                ("coder", 1),
                ("researcher", 1),
                ("reviewer", 1),
                ("chat", 1)
            ]
        );
    }

    /// **A task codes unless asked**, and a kind nobody defined is refused
    /// rather than leaving a worker on no document. A task can never be a
    /// chat: that kind is the chat's own.
    #[test]
    fn a_call_names_a_kind_or_gets_coder() {
        assert_eq!(WorkerKind::of(None).unwrap(), WorkerKind::Coder);
        assert_eq!(
            ["coder", "researcher", "reviewer"].map(|k| WorkerKind::of(Some(k)).unwrap()),
            KINDS[..3]
        );
        assert_eq!(
            WorkerKind::of(Some("tester")).unwrap_err(),
            "`role` is `coder`, `researcher` or `reviewer`, not `tester`"
        );
        assert_eq!(
            WorkerKind::of(Some("chat")).unwrap_err(),
            "`role` is `coder`, `researcher` or `reviewer`, not `chat`"
        );
        assert!(WorkerKind::Chat.is_chat() && !WorkerKind::Coder.is_chat());
    }

    /// **`generic` is today's prompt to the byte**: it emits nothing, so a
    /// conversation that is no worker seals no new section.
    #[test]
    fn generic_emits_nothing() {
        let options = options();
        assert_eq!(options[0], ("generic".to_string(), None));
    }

    /// Every kind names a real option, and only working rules: a worked
    /// exchange in a role would hijack the dialogues that borrow its turns.
    #[test]
    fn every_kind_has_an_option_of_its_own() {
        let options = options();
        assert_eq!(options.len(), KINDS.len() + 1);
        for kind in KINDS {
            let (_, content) = options
                .iter()
                .find(|(id, _)| id == kind.option())
                .unwrap_or_else(|| panic!("no option for {kind:?}"));
            let content = content.as_deref().unwrap();
            assert!(content.starts_with("You are working as a "), "{kind:?}");
            assert!(!content.contains("<|im_start|>"), "{kind:?}");
        }
    }
}
