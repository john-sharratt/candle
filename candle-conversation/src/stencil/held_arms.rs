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

use std::collections::HashSet;
use std::sync::Arc;

use super::named_arm::arm_name;
use super::tree::{NodeId, StencilNode, StencilTree};
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
    held_branches_first(tree, names, None, decode)
}

/// [`held_branches`], with the first call held to `first` alone when given:
/// the turn's opening call is that tool, every call after it may be any of
/// `names`, and the turn may not end after the first — the choice that follows
/// it is held to the arm that opens another call. A turn whose first call must
/// say something before it acts — a note of what its thinking concluded
/// (`docs/zend_journal.md` §5.2) — is held this way, so a note is always
/// followed by the act it was written for.
pub fn held_branches_first(
    tree: &StencilTree,
    names: &[String],
    first_call: Option<&str>,
    decode: impl Fn(&[TokenId]) -> Option<String>,
) -> HeldBranches {
    let Some(first) = tree.first_branch() else {
        return HeldBranches::default();
    };
    let opening = tree.first_branch_node();
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
        let allowed = |name: &str| match first_call {
            Some(only) if Some(node) == opening => name == only,
            _ => names.iter().any(|n| n == name),
        };
        let held: Vec<Vec<TokenId>> = arms
            .into_iter()
            .zip(&spelled)
            .filter(|(_, name)| allowed(name))
            .map(|(arm, _)| arm)
            .collect();
        if !held.is_empty() {
            by_node.push((node, Arc::from(held)));
        }
    }
    if let (Some(only), Some(opening)) = (first_call, opening) {
        by_node.extend(continuation_after(tree, opening, only, &decode));
    }
    HeldBranches { by_node }
}

/// The go-again-or-finish choices a call reaches once the tool `first` is
/// written at the branch `opening`, each held to the arms that go again.
///
/// A choice is a branch a call's body leads to with an arm straight to the
/// tree's end — finishing the turn — beside one that does not, opening the
/// next call. Every way through the call's arguments meets one before anything
/// of the next call. There is one copy of it per way the call can close: a
/// value span that closed on a tail it wrote goes on through that tail's fork
/// (`FreeTextSpan::forks`), each fork lowered to its own copy of what follows,
/// so the walk follows every fork as well as `next`, and stops at each choice
/// it meets rather than walking on into the next call.
fn continuation_after(
    tree: &StencilTree,
    opening: NodeId,
    first: &str,
    decode: &impl Fn(&[TokenId]) -> Option<String>,
) -> Vec<(NodeId, Arc<[Vec<TokenId>]>)> {
    let Some(names) = tree.branch(opening) else {
        return Vec::new();
    };
    let Some(start) = names
        .arms()
        .into_iter()
        .find(|a| decode(a).is_some_and(|t| arm_name(t.as_bytes()) == first))
        .and_then(|arm| names.target_of(&arm))
    else {
        return Vec::new();
    };
    let ends = |node: NodeId| matches!(tree.node(node), StencilNode::End);
    let mut held = Vec::new();
    let mut todo = vec![start];
    let mut seen = HashSet::new();
    while let Some(node) = todo.pop() {
        if !seen.insert(node) {
            continue;
        }
        match tree.node(node) {
            StencilNode::Static { next, .. } => todo.push(*next),
            StencilNode::FreeText(span) => {
                todo.push(span.next);
                todo.extend(span.forks.iter().copied());
            }
            StencilNode::End => {}
            StencilNode::Branch { trie } => {
                let arms = trie.arms();
                let again: Vec<Vec<TokenId>> = arms
                    .iter()
                    .filter(|a| trie.target_of(a).is_some_and(|t| !ends(t)))
                    .cloned()
                    .collect();
                if !again.is_empty() && again.len() < arms.len() {
                    held.push((node, Arc::from(again)));
                    continue;
                }
                todo.extend(arms.iter().filter_map(|a| trie.target_of(a)));
            }
        }
    }
    held
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

    /// **A turn's first call can be held to one tool and the rest to the
    /// turn's**: the opening name branch offers only `first`, every later
    /// level the names given — and the choice after the first call is held to
    /// going on, so the turn cannot end on it.
    #[test]
    fn the_first_call_is_held_to_its_own_tool_and_goes_on() {
        let v = TestVocab::new();
        let t = looped(&v, 3);
        let decode = |tokens: &[TokenId]| String::from_utf8(v.decode(tokens)).ok();
        let held = held_branches_first(
            &t,
            &given(&["file_read", "write"]),
            Some("file_list"),
            decode,
        );
        assert_eq!(
            held.len(),
            4,
            "three name branches and the choice after the first"
        );
        let opening = t.first_branch_node().unwrap();
        assert_eq!(names(held.arms_at(opening).unwrap(), &v), ["file_list"]);
        let mut choices = 0;
        for i in 0..t.len() as u32 {
            let node = NodeId(i);
            if node == opening {
                continue;
            }
            let Some(arms) = held.arms_at(node) else {
                continue;
            };
            let trie = t.branch(node).unwrap();
            if trie.arm_count() == 2 {
                choices += 1;
                assert_eq!(arms.len(), 1, "held to going on alone");
                let next = trie.target_of(&arms[0]).unwrap();
                assert!(
                    !matches!(t.node(next), StencilNode::End),
                    "going on, not ending"
                );
                assert!(
                    String::from_utf8(v.decode(&arms[0]))
                        .unwrap()
                        .contains("<tool_call>"),
                    "the arm opens the next call"
                );
            } else {
                assert_eq!(names(arms, &v), ["file_read", "write"]);
            }
        }
        assert_eq!(choices, 1);
        assert_eq!(
            held_branches(&t, &given(&["file_read"]), decode).len(),
            3,
            "with no first call held, no choice is"
        );
    }

    /// **A function-block call goes on whichever way its last value closed**:
    /// each tail a value can close on forks to its own copy of the choice
    /// after the call, and every copy after the first call is held to going
    /// on — and none after the second, which the turn may end on.
    #[test]
    fn every_copy_of_the_choice_after_a_function_block_is_held() {
        let v = TestVocab::new();
        let spec = compile_tool_call_loop(
            &parse_tools(CATALOG).unwrap(),
            &ToolCallEnvelope::qwen35(),
            3,
            "<|im_end|>",
        )
        .unwrap();
        let t = compile(&spec, &v).unwrap();
        let decode = |tokens: &[TokenId]| String::from_utf8(v.decode(tokens)).ok();
        let is_choice = |node: NodeId| {
            t.branch(node).is_some_and(|trie| {
                let arms = trie.arms();
                let ending = arms
                    .iter()
                    .filter(|a| {
                        trie.target_of(a)
                            .is_some_and(|n| matches!(t.node(n), StencilNode::End))
                    })
                    .count();
                ending > 0 && ending < arms.len()
            })
        };
        let all: Vec<NodeId> = (0..t.len() as u32)
            .map(NodeId)
            .filter(|n| is_choice(*n))
            .collect();
        let held = held_branches_first(&t, &given(&["file_read"]), Some("file_list"), decode);
        let held_choices: Vec<NodeId> = all
            .iter()
            .copied()
            .filter(|n| held.arms_at(*n).is_some())
            .collect();
        assert!(held_choices.len() > 1, "one copy per way the call closes");
        assert!(held_choices.len() < all.len(), "the later calls' are free");
        // Everything a held choice's go-on arm leads to is the next call's:
        // no choice there is held.
        let mut later = HashSet::new();
        let mut todo: Vec<NodeId> = Vec::new();
        for &node in &held_choices {
            let trie = t.branch(node).unwrap();
            for arm in held.arms_at(node).unwrap().iter() {
                let next = trie.target_of(arm).unwrap();
                assert!(!matches!(t.node(next), StencilNode::End), "going on");
                todo.push(next);
            }
        }
        while let Some(node) = todo.pop() {
            if !later.insert(node) {
                continue;
            }
            match t.node(node) {
                StencilNode::Static { next, .. } => todo.push(*next),
                StencilNode::FreeText(span) => {
                    todo.push(span.next);
                    todo.extend(span.forks.iter().copied());
                }
                StencilNode::Branch { trie } => {
                    todo.extend(trie.arms().iter().filter_map(|a| trie.target_of(a)))
                }
                StencilNode::End => {}
            }
        }
        for node in &held_choices {
            assert!(!later.contains(node), "a later call's choice is held");
        }
        assert!(
            all.iter().any(|n| later.contains(n)),
            "the walk reached the later calls"
        );
    }
}
