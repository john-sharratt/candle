//! Tools a turn may call once: after one call names such a tool, no later call
//! of the same turn can (`docs/stencil_tree.md` §9.1).
//!
//! A turn's calls are one walk of the call loop, each level with a name branch
//! offering the turn's tools. Some are only ever right once a turn: a note of
//! what the round's thinking concluded is the case — the round's opening call
//! is held to one (`docs/zend_journal.md` §5.2), and the grammar then offered it
//! again at the next level, where a model wrote the same note a second time,
//! word for word, instead of the call its thinking had planned (BlackWorld,
//! 2026-11-09). Held out by the grammar, the second can not be written at all,
//! rather than being answered after the fact.
//!
//! The set is the turn's, the exclusion the walk's: which tools are spent is
//! known only as the walk names them, so the driver asks here as it enters each
//! name branch, with the tools the walk has named so far.

use std::collections::HashSet;
use std::sync::Arc;

use super::tree::{NodeId, StencilTree};
use super::vocab::TokenId;

/// The tools a turn may call at most once.
#[derive(Debug, Clone, Default)]
pub struct NeverTwice {
    tools: HashSet<Arc<str>>,
}

impl NeverTwice {
    pub fn new<S: AsRef<str>>(tools: impl IntoIterator<Item = S>) -> Self {
        NeverTwice {
            tools: tools.into_iter().map(|t| Arc::from(t.as_ref())).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The arms of the name branch at `node` a call may take after the walk
    /// has named `named`: `arms` without those that name a tool of this set
    /// already named. `None` when nothing is left out — `node` is not a name
    /// branch, no tool of the set has been named, or no arm names one — so the
    /// branch is walked as it would be without the set. Every arm left out
    /// also comes back `None`: a branch is never left with nothing to write.
    pub fn narrow(
        &self,
        tree: &StencilTree,
        node: NodeId,
        arms: &[Vec<TokenId>],
        named: &[Arc<str>],
    ) -> Option<Arc<[Vec<TokenId>]>> {
        if !tree.is_name_branch(node) {
            return None;
        }
        let spent: HashSet<&Arc<str>> = named.iter().filter(|t| self.tools.contains(*t)).collect();
        if spent.is_empty() {
            return None;
        }
        let kept: Vec<Vec<TokenId>> = arms
            .iter()
            .filter(|arm| {
                !tree
                    .tool_named(node, arm)
                    .is_some_and(|t| spent.contains(t))
            })
            .cloned()
            .collect();
        (!kept.is_empty() && kept.len() < arms.len()).then(|| Arc::from(kept))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};

    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::driver::{StencilDriver, StepMask};
    use crate::stencil::held_arms::held_branches_first;
    use crate::stencil::named_arm::arm_name;
    use crate::stencil::tool_call::{compile_tool_call_loop, parse_tools, ToolCallEnvelope};
    use crate::stencil::tree::StencilNode;
    use crate::stencil::trigger::TriggerRegistry;
    use crate::stencil::vocab::{TestVocab, Vocab};

    const CATALOG: &str = r#"[
        {"name":"file_list","params":[{"name":"path","type":"string","required":true}]},
        {"name":"file_read","params":[{"name":"path","type":"string","required":true}]},
        {"name":"journal_note","params":[{"name":"text","type":"string","required":true}]}]"#;

    const TRIGGER: TokenId = 1000;

    fn looped(v: &TestVocab) -> Arc<StencilTree> {
        let spec = compile_tool_call_loop(
            &parse_tools(CATALOG).unwrap(),
            &ToolCallEnvelope::qwen3(),
            3,
            "<|im_end|>",
        )
        .unwrap();
        Arc::new(compile(&spec, v).unwrap())
    }

    /// A driver for one turn, with `never_twice` given its tools.
    fn driver(tree: &Arc<StencilTree>, never_twice: &[&str]) -> StencilDriver {
        let mut registry = TriggerRegistry::new();
        registry.register(TRIGGER, Arc::clone(tree));
        if !never_twice.is_empty() {
            registry = registry.never_twice(TRIGGER, Arc::new(NeverTwice::new(never_twice)));
        }
        registry.driver_for(TRIGGER).unwrap()
    }

    /// Walk one whole call of `tool` and on into the next call's name branch:
    /// at a name branch the token of `tool`'s arm, at the choice after the
    /// call the arm that opens another, and every free value ` "a"`. Returns
    /// the driver waiting on the next call's name.
    fn call_then_next(mut d: StencilDriver, v: &TestVocab, tool: &str) -> StencilDriver {
        let mut value: VecDeque<TokenId> = VecDeque::new();
        let mut named = false;
        // The rest of the arm the walk is writing through a branch.
        let mut arm: VecDeque<TokenId> = VecDeque::new();
        loop {
            match d.step() {
                StepMask::Prefill(_) => {}
                StepMask::Free { .. } => {
                    if value.is_empty() {
                        value.extend(v.encode(" \"a\""));
                    }
                    let t = value.pop_front().unwrap();
                    d.accept(t, &v.token_bytes(t));
                }
                StepMask::Branch(set) => {
                    let (node, at_root) = branch_node(&d);
                    let tree = d.tree();
                    if at_root {
                        if tree.is_name_branch(node) && named {
                            return d;
                        }
                        let trie = tree.branch(node).unwrap();
                        let chosen = if tree.is_name_branch(node) {
                            named = true;
                            trie.arms()
                                .into_iter()
                                .find(|a| tree.tool_named(node, a).is_some_and(|t| &**t == tool))
                        } else {
                            // The choice after the call: the arm that goes on.
                            trie.arms().into_iter().find(|a| {
                                trie.target_of(a)
                                    .is_some_and(|n| !matches!(tree.node(n), StencilNode::End))
                            })
                        };
                        arm = chosen.unwrap().into();
                    }
                    let t = arm.pop_front().unwrap();
                    assert!(set.contains(t));
                    d.accept(t, &v.token_bytes(t));
                }
                StepMask::Done => panic!("the turn ended before a second call"),
            }
        }
    }

    fn branch_node(d: &StencilDriver) -> (NodeId, bool) {
        d.in_branch().unwrap()
    }

    /// Every tool the name branch `d` waits on can complete on.
    fn nameable(d: &StencilDriver, v: &TestVocab) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut stack = vec![(d.clone(), Vec::new())];
        while let Some((mut d, written)) = stack.pop() {
            match d.step() {
                StepMask::Branch(set) if d.tree().is_name_branch(branch_node(&d).0) => {
                    for &t in set.tokens() {
                        let mut next = d.clone();
                        let mut w = written.clone();
                        w.push(t);
                        next.accept(t, &v.token_bytes(t));
                        stack.push((next, w));
                    }
                }
                _ => {
                    out.insert(arm_name(&v.decode(&written)).to_string());
                }
            }
            assert!(stack.len() < 10_000, "the walk did not converge");
        }
        out
    }

    fn set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// **A tool called once can not be named again in the turn, and nothing
    /// else changes.** After a `journal_note` call the next call offers every
    /// tool but it; after a `file_list` call it offers all three; and without
    /// the set, a note after a note is as writable as anything.
    #[test]
    fn a_tool_called_once_is_not_offered_again() {
        let v = TestVocab::new();
        let t = looped(&v);
        let after =
            |tool: &str, never: &[&str]| nameable(&call_then_next(driver(&t, never), &v, tool), &v);
        assert_eq!(
            after("journal_note", &["journal_note"]),
            set(&["file_list", "file_read"])
        );
        assert_eq!(
            after("file_list", &["journal_note"]),
            set(&["file_list", "file_read", "journal_note"])
        );
        assert_eq!(
            after("journal_note", &[]),
            set(&["file_list", "file_read", "journal_note"])
        );
    }

    /// **A turn held to open with a note, as zend's rounds are, cannot follow
    /// it with another**: the opening call is held to `journal_note` and the
    /// later ones to the turn's tools, and the set narrows that hold.
    #[test]
    fn a_held_opening_note_is_not_followed_by_another() {
        let v = TestVocab::new();
        let t = looped(&v);
        let tools: Vec<String> = ["file_list", "file_read", "journal_note"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let held = held_branches_first(&t, &tools, Some("journal_note"), |x| {
            String::from_utf8(v.decode(x)).ok()
        });
        let mut registry = TriggerRegistry::new();
        registry.register(TRIGGER, Arc::clone(&t));
        let held = registry.holding(TRIGGER, Arc::new(held));
        let next = |r: &TriggerRegistry| {
            nameable(
                &call_then_next(r.driver_for(TRIGGER).unwrap(), &v, "journal_note"),
                &v,
            )
        };
        assert_eq!(
            next(&held),
            set(&["file_list", "file_read", "journal_note"]),
            "held alone, a second note is writable"
        );
        let once = held.never_twice(TRIGGER, Arc::new(NeverTwice::new(["journal_note"])));
        assert_eq!(next(&once), set(&["file_list", "file_read"]));
    }

    /// The narrowing leaves a branch that is not a name branch alone, and a
    /// set whose tools have not been named leaves every branch alone.
    #[test]
    fn nothing_is_narrowed_before_a_tool_of_the_set_is_named() {
        let v = TestVocab::new();
        let t = looped(&v);
        let once = NeverTwice::new(["journal_note"]);
        let node = t.first_branch_node().unwrap();
        let arms = t.branch(node).unwrap().arms();
        assert!(once.narrow(&t, node, &arms, &[]).is_none());
        assert!(once
            .narrow(&t, node, &arms, &[Arc::from("file_read")])
            .is_none());
        let kept = once
            .narrow(&t, node, &arms, &[Arc::from("journal_note")])
            .unwrap();
        assert_eq!(kept.len(), 2);
    }
}
