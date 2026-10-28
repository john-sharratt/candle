//! The arms of a tool-call tree's name branches that spell given tool names:
//! what a turn given some of a catalog's tools holds its calls to
//! ([`TriggerRegistry::holding`](super::TriggerRegistry::holding)).
//!
//! The catalog's grammar is compiled once per tools mode, and every call it
//! lets a turn make — up to `MAX_TOOL_CALLS_PER_TURN`, each a level of the
//! loop with a name branch of its own — offers every tool in it. A turn given
//! fewer (a layer unit with its two tools, a reading pinned to `file_read`)
//! narrows each name branch to the arms its tools spell. They are found here
//! once, by the same reading named-tool steering does ([`arm_name`]); the tree
//! is shared and untouched, and what the turn carries is a short table of
//! branch nodes and those arms' token sequences — a few tokens per tool per
//! level.

use std::sync::Arc;

use super::named_arm::arm_name;
use super::tree::{NodeId, StencilTree};
use super::vocab::TokenId;

/// The arms each name branch of a tree is held to, by branch node.
#[derive(Debug, Clone, Default)]
pub struct HeldBranches {
    by_node: Vec<(NodeId, Arc<[Vec<TokenId>]>)>,
}

impl HeldBranches {
    /// The arms the branch at `node` is held to, when it is a held branch.
    pub fn arms_at(&self, node: NodeId) -> Option<&Arc<[Vec<TokenId>]>> {
        self.by_node
            .iter()
            .find(|(n, _)| *n == node)
            .map(|(_, arms)| arms)
    }

    /// How many branches are held.
    pub fn len(&self) -> usize {
        self.by_node.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_node.is_empty()
    }
}

/// Every name branch of `tree` held to the arms whose name is one of `names`,
/// read with `decode`.
///
/// A name branch is one whose arms spell exactly the names the tree's first
/// branch does — the catalog. Every level of a call loop has one; an enum's
/// values, an optional field's gate or the loop's go-again-or-finish choice
/// never spell the whole catalog. A branch is only decoded when it has as many
/// arms as the catalog, so the rest of the tree costs a comparison each.
/// Empty when the tree has no first branch or no arm is named.
pub fn held_branches(
    tree: &StencilTree,
    names: &[String],
    decode: impl Fn(&[TokenId]) -> Option<String>,
) -> HeldBranches {
    let Some(first) = tree.first_branch() else {
        return HeldBranches::default();
    };
    let named = |arm: &Vec<TokenId>| -> String {
        decode(arm)
            .map(|text| arm_name(text.as_bytes()).to_string())
            .unwrap_or_default()
    };
    let mut catalog: Vec<String> = first.arms().iter().map(named).collect();
    catalog.sort_unstable();
    let mut by_node = Vec::new();
    for i in 0..tree.len() as u32 {
        let node = NodeId(i);
        let Some(trie) = tree.branch(node).filter(|t| t.arm_count() == catalog.len()) else {
            continue;
        };
        let arms = trie.arms();
        let spelled: Vec<String> = arms.iter().map(named).collect();
        let mut sorted = spelled.clone();
        sorted.sort_unstable();
        if sorted != catalog {
            continue;
        }
        let held: Vec<Vec<TokenId>> = arms
            .into_iter()
            .zip(&spelled)
            .filter(|(_, name)| names.iter().any(|n| n == *name))
            .map(|(arm, _)| arm)
            .collect();
        if !held.is_empty() {
            by_node.push((node, Arc::from(held)));
        }
    }
    HeldBranches { by_node }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::tool_call::{
        compile_tool_call_loop, compile_tool_call_tree, parse_tools, ToolCallEnvelope,
    };
    use crate::stencil::vocab::{TestVocab, Vocab};

    pub(crate) const CATALOG: &str = r#"[
        {"name":"file_list","params":[{"name":"path","type":"string","required":true}]},
        {"name":"file_read","params":[{"name":"path","type":"string","required":true}]},
        {"name":"write","params":[{"name":"path","type":"string","required":true},
            {"name":"mode","type":"string","enum":["write","append","replace"],"required":true}]}]"#;

    fn single(v: &TestVocab) -> StencilTree {
        let spec =
            compile_tool_call_tree(&parse_tools(CATALOG).unwrap(), &ToolCallEnvelope::qwen3())
                .unwrap();
        compile(&spec, v).unwrap()
    }

    fn looped(v: &TestVocab, calls: usize) -> StencilTree {
        let spec = compile_tool_call_loop(
            &parse_tools(CATALOG).unwrap(),
            &ToolCallEnvelope::qwen3(),
            calls,
            "<|im_end|>",
        )
        .unwrap();
        compile(&spec, v).unwrap()
    }

    fn names(arms: &[Vec<TokenId>], v: &TestVocab) -> Vec<String> {
        arms.iter()
            .map(|a| arm_name(&v.decode(a)).to_string())
            .collect()
    }

    fn given(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// **The given names pick their arms and no others**, whatever order they
    /// were given in; a name the catalog does not have picks nothing.
    #[test]
    fn the_given_names_pick_their_arms() {
        let v = TestVocab::new();
        let t = single(&v);
        let decode = |tokens: &[TokenId]| String::from_utf8(v.decode(tokens)).ok();
        let held = held_branches(&t, &given(&["write", "file_read"]), decode);
        assert_eq!(held.len(), 1);
        let arms = held.arms_at(t.first_branch_node().unwrap()).unwrap();
        assert_eq!(names(arms, &v), ["file_read", "write"]);
        assert!(held_branches(&t, &given(&["web_search"]), decode).is_empty());
    }

    /// A name is the arm's whole identifier: `file` names neither `file_list`
    /// nor `file_read`.
    #[test]
    fn a_name_is_the_whole_identifier() {
        let v = TestVocab::new();
        let t = single(&v);
        let decode = |tokens: &[TokenId]| String::from_utf8(v.decode(tokens)).ok();
        assert!(held_branches(&t, &given(&["file"]), decode).is_empty());
    }

    /// **Every call of a loop is held**, one name branch per level — and an
    /// enum whose value is spelled like a tool (`write`) is not a name branch.
    #[test]
    fn every_level_of_a_call_loop_is_held_and_nothing_else() {
        let v = TestVocab::new();
        let t = looped(&v, 3);
        let decode = |tokens: &[TokenId]| String::from_utf8(v.decode(tokens)).ok();
        let held = held_branches(&t, &given(&["file_read"]), decode);
        assert_eq!(held.len(), 3);
        for i in 0..t.len() as u32 {
            if let Some(arms) = held.arms_at(NodeId(i)) {
                assert_eq!(names(arms, &v), ["file_read"]);
            }
        }
    }
}
