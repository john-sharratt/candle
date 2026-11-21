//! Tests for folder naming, completion, tags, steering and the ingest prompt.
//!
//! The completion tests pin the protocol the crashed-partial sweep rests on: a
//! conversation is committed only once it carries a content key, and keying on
//! the directory tag instead would retire every good generation at the next boot.
//! The steering tests check that every tool the prefilled chain calls is pinned
//! into the catalog, and that a folder's calls are never left free. The prompt
//! test checks that the layer frames on the shared prompt at its default branch.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// **A folder is asked after as a folder**, one closing `/`, however its
    /// unit spells it.
    #[test]
    fn a_folder_is_named_with_one_closing_slash() {
        assert_eq!(as_folder("candle/mcp/src/guides"), "candle/mcp/src/guides/");
        assert_eq!(
            as_folder("candle/mcp/src/guides/"),
            "candle/mcp/src/guides/"
        );
        assert_eq!(as_folder(""), "/");
    }

    /// The completion protocol the crashed-partial sweep rests on: the directory
    /// tag is written at creation, the content key only on success, so `dir`
    /// without a key is the signature of an attempt that never finished.
    ///
    /// Pinned because the sweep tombstones on it. Keying on the wrong half would
    /// retire every GOOD generation on the next pass — the folder summaries are
    /// exactly the conversations that DO carry a key — and the blast radius is
    /// the whole layer, silently, one boot later.
    #[test]
    fn a_conversation_is_committed_only_once_it_carries_a_content_key() {
        let committed = |m: &BTreeMap<String, String>| ingest_committed(|k| m.contains_key(k));
        let mut meta = BTreeMap::new();
        assert!(!committed(&meta), "no metadata at all is not committed");

        meta.insert(DIR_KEY.to_string(), "candle-nn/src/".to_string());
        assert!(
            !committed(&meta),
            "the directory tag alone is a crashed partial — it is written at \
             conversation creation, before any work"
        );

        meta.insert(CONTENT_KEY.to_string(), "abc123".to_string());
        assert!(
            committed(&meta),
            "the content key is the commit, and is written only on success"
        );

        // The key is what counts, not the tag: a generation that somehow lost
        // its directory tag has still committed its content and must not be
        // swept. (`retire_crashed_partials` only visits `DIR_KEY` holders, so
        // this is belt-and-braces on the predicate itself.)
        let mut key_only = BTreeMap::new();
        key_only.insert(CONTENT_KEY.to_string(), "abc123".to_string());
        assert!(committed(&key_only));

        // A generation from before units were keyed by content carries only
        // the old hash, and is not committed.
        let mut old = BTreeMap::new();
        old.insert(DIR_KEY.to_string(), "candle-nn/src/".to_string());
        old.insert("content_sha256".to_string(), "abc123".to_string());
        assert!(!committed(&old));
    }

    pub(in super::super) fn unit(dir: &str) -> DirUnit {
        DirUnit {
            dir: dir.to_string(),
            listed: vec![format!("{dir}x.rs")],
            module_hint: None,
            content_key: "abc".to_string(),
        }
    }

    /// The gather-scope tags name the layer and the exact directory, so a
    /// tag-scoped gallery admits one folder's turns and no other's.
    #[test]
    fn tags_carry_the_layer_and_the_directory() {
        assert_eq!(
            dir_tags(&unit("zend/src/")),
            vec!["repo_map".to_string(), "zend/src/".to_string()],
        );
    }

    #[test]
    fn a_repository_root_tags_with_its_repository() {
        assert_eq!(
            dir_tags(&unit("candle/")),
            vec!["repo_map".to_string(), "candle/".to_string()],
        );
    }

    /// **A folder's calls are never left free.** Steering with no call grammar
    /// on `<tool_call>` is refused — the registry the repo map once ran on,
    /// the `<think>` trigger alone, is exactly that — and one that binds the
    /// grammar there passes, as does a vocabulary with no such token.
    #[test]
    fn a_pass_refuses_steering_that_leaves_its_calls_free() {
        use candle_conversation::stencil::{compile, NodeSpec, TestVocab, TreeSpec};
        let call = 7;
        let empty = TriggerRegistry::new();
        assert!(calls_are_held(&empty, Some(call)).is_err());
        assert!(calls_are_held(&empty, None).is_ok());
        let mut spec = TreeSpec::new("call");
        spec.root = spec.push(NodeSpec::End);
        let tree = Arc::new(compile(&spec, &TestVocab::new()).expect("an empty walk compiles"));
        let held = empty.with_trigger(call, tree);
        assert!(calls_are_held(&held, Some(call)).is_ok());
    }

    /// Every tool the prefilled chain calls must be pinned into the catalog, or
    /// the projection presents a `<tool_call>` for a tool it never defined.
    #[test]
    fn every_tool_the_chain_calls_is_pinned() {
        for tool in render::CHAIN_TOOLS {
            assert!(
                zend_tools::registry::find(tool).is_some(),
                "{tool} is called by the chain but not registered",
            );
        }
    }

    /// Parse the bundled projection.yaml the way the daemon does.
    pub(in super::super) fn bundled_builder() -> projection::Builder {
        let dialect = candle_conversation::models::Dialect::chat_ml();
        projection::Builder::from_yaml_with_vars_and_dialect(
            include_str!("../prompts/projection.yaml"),
            &[("workspace", "test"), ("model", "test-model")],
            Some(&dialect),
        )
        .expect("projection.yaml must parse")
    }

    /// The prompt every ingest layer frames on is the SHARED one at the schema's
    /// default branch — the same content priming injects as K/V, and the same
    /// sections a chat conversation injects. It must hold the dialogue persona
    /// (an ingest turn is borrowed into dialogue projections and carries the
    /// framing it was computed under), and no second persona or worked request.
    #[test]
    fn the_ingest_prompt_is_the_shared_prompt_at_its_default_branch() {
        let builder = bundled_builder();
        let prompt = shared_prompt_body(&builder);

        assert!(
            prompt.contains("You are Zen"),
            "the ingest must frame on the dialogue persona its turns are projected into",
        );
        assert!(
            !prompt.contains("code-summarization engine"),
            "no second persona — a turn sealed under one hijacks the dialogue it is \
             borrowed into",
        );
        assert!(
            !prompt.contains("Summarize the `worker/scheduling/` folder"),
            "no worked summarize examples — they teach the dialogue to summarize too",
        );
    }

    /// `history_stance` owns the lineage as reading and draws the line between
    /// reading and conversation — and names no face label: a reading's face
    /// commissions nothing, and telling the model a reading opens on "a label,
    /// not a request" taught it the seed was labels rather than what it read.
    #[test]
    fn history_stance_owns_the_lineage_as_reading() {
        let builder = bundled_builder();
        let stance = builder
            .schema()
            .system_prompt
            .items
            .iter()
            .find_map(|item| match item {
                projection::SystemPromptItem::Section(s) if s.name == "history_stance" => {
                    Some(s.content.clone())
                }
                _ => None,
            })
            .expect("projection.yaml declares history_stance");

        // The block is hard-wrapped for readability, so a quoted opener spans
        // whatever line break the authored text happens to fall on. Match the
        // wording, not the wrapping.
        let stance = stance.split_whitespace().collect::<Vec<_>>().join(" ");

        for label in ["file=", "folder="] {
            assert!(
                !stance.contains(label),
                "history_stance names no face label: {stance:?}",
            );
        }
        // The priming chain's links are inherited INTO `primary_conversation` and
        // sit as the conversation's earliest turns, so a section that spoke only of
        // content "placed ahead of the conversation" did not reach them. The split
        // has to be reading vs conversation.
        assert!(
            stance.contains("descends from"),
            "the section must own the lineage it descends from, not only placed \
             content: {stance:?}",
        );
        assert!(
            stance.contains("the turns that are not reading"),
            "the model needs the rule that separates its dialogue from its reading: \
             {stance:?}",
        );
    }

    /// `repo_map` finds a unit's conversations by `dir`, `code_read` by `path`.
    /// If they shared a key, each layer's plan would take the other's
    /// conversations for its own (a folder is never a file unit, and vice
    /// versa) and tombstone them.
    #[test]
    fn the_invalidation_key_is_distinct_from_code_reads() {
        assert_eq!(DIR_KEY, "dir");
        assert_ne!(DIR_KEY, "path");
    }
}
