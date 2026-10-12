//! The token trie that backs a `Branch`.
//!
//! A branch's arms are token sequences (a tool name, an enum value).  They are
//! inserted into a trie; the live *frontier* at a trie node is the set of tokens
//! the sampler is masked to, and successive masked decodes walk the trie to a
//! leaf, whose `next` resumes the outer tree.
//!
//! Arms must be **token-level prefix-free** — no arm's token sequence may be a
//! full prefix of another's — so every accepting node is a pure leaf (no
//! outgoing edges) and a mask is never ambiguous between "stop" and "continue".
//! [`TokenTrie::build`] enforces this.
//!
//! # Spills: a token that finishes an arm and runs on
//!
//! An arm ends where the grammar stops writing, and the model's text begins.
//! A tokenizer does not stop there: Qwen writes a string's opening quote and
//! its first byte as one token — ` "-` in `["tree", "-p"]` — so with the arm's
//! last token (` "`) the only way on, the model was held to a boundary its
//! encoding never makes, and read `-p` as a token it had almost never seen
//! after a lone quote. Live, `cargo tree -p agent --depth 1` was written
//! `["tree", "depth"]`, call after call, until the array was `"p"` forty times.
//!
//! A **spill** ([`TokenTrie::spill_from`]) is such a token, offered beside the
//! arm's last token: taking it completes the arm, and the bytes past the arm
//! (`-`) are the first the successor reads. The compiler adds them only where
//! the successor is free text that takes those bytes as they are.

use super::error::BuildError;
use super::tree::NodeId;
use super::vocab::TokenId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrieNodeId(pub u32);

/// A spill in the trie that holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillId(pub u32);

/// A token completing an arm and running past it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spill {
    /// Where the completed arm resumes the outer tree.
    pub next: NodeId,
    /// The token's bytes past the arm: the successor's first.
    pub rest: Vec<u8>,
}

#[derive(Debug, Clone)]
struct TrieNode {
    /// (token, child), kept sorted by token for a binary-searchable frontier.
    edges: Vec<(TokenId, TrieNodeId)>,
    /// (token, spill), kept sorted by token, and never a token of `edges`.
    spills: Vec<(TokenId, SpillId)>,
    /// `Some(next)` when this node completes an arm.  An accepting node is
    /// always a leaf (`edges.is_empty()`), guaranteed by `build`.
    accept: Option<NodeId>,
}

#[derive(Debug, Clone)]
pub struct TokenTrie {
    nodes: Vec<TrieNode>,
    root: TrieNodeId,
    spills: Vec<Spill>,
}

/// The result of stepping a trie with a decoded token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Keep walking; the new trie position.
    Descend(TrieNodeId),
    /// An arm completed; resume the outer tree at this node.
    Accept(NodeId),
    /// An arm completed, and the token ran past it ([`TokenTrie::spill`]).
    Spill(SpillId),
}

impl TokenTrie {
    /// Build a trie from `(arm tokens, successor)` pairs.  Errors on an empty
    /// arm or on a token-level prefix collision between two arms.
    pub fn build(arms: &[(Vec<TokenId>, NodeId)]) -> Result<TokenTrie, BuildError> {
        let mut nodes = vec![TrieNode {
            edges: Vec::new(),
            spills: Vec::new(),
            accept: None,
        }];
        let root = TrieNodeId(0);

        for (tokens, next) in arms {
            if tokens.is_empty() {
                return Err(BuildError::EmptyArm {
                    arm: String::from("<empty>"),
                });
            }
            let mut cur = 0usize;
            for (depth, &tok) in tokens.iter().enumerate() {
                // An accepting node may not have children (would make a shorter
                // arm a prefix of this one).
                if nodes[cur].accept.is_some() {
                    return Err(BuildError::AmbiguousArms {
                        short: format!("{:?}", &tokens[..depth]),
                        long: format!("{tokens:?}"),
                    });
                }
                match nodes[cur].edges.iter().find(|(t, _)| *t == tok) {
                    Some((_, child)) => cur = child.0 as usize,
                    None => {
                        let child = nodes.len();
                        nodes.push(TrieNode {
                            edges: Vec::new(),
                            spills: Vec::new(),
                            accept: None,
                        });
                        nodes[cur].edges.push((tok, TrieNodeId(child as u32)));
                        nodes[cur].edges.sort_by_key(|(t, _)| *t);
                        cur = child;
                    }
                }
            }
            // `cur` is the arm's terminal node. It must be fresh (no edges, no
            // accept) — otherwise this arm is a prefix of, or equal to, another.
            if nodes[cur].accept.is_some() || !nodes[cur].edges.is_empty() {
                return Err(BuildError::AmbiguousArms {
                    short: format!("{tokens:?}"),
                    long: String::from("<another arm>"),
                });
            }
            nodes[cur].accept = Some(*next);
        }

        Ok(TokenTrie {
            nodes,
            root,
            spills: Vec::new(),
        })
    }

    /// Offer `spills` — each a token and its bytes past the arm — beside the
    /// last token of `arm`, an arm this trie was built with. A token that is
    /// already a way on from that position is left as it is. Returns `self`.
    pub fn spill_from(mut self, arm: &[TokenId], spills: Vec<(TokenId, Vec<u8>)>) -> Self {
        let Some((&last, head)) = arm.split_last() else {
            return self;
        };
        let mut at = self.root;
        for &t in head {
            match self.step(at, t) {
                Some(Step::Descend(p)) => at = p,
                _ => return self,
            }
        }
        let Some(Step::Accept(next)) = self.step(at, last) else {
            return self;
        };
        for (token, rest) in spills {
            let node = &self.nodes[at.0 as usize];
            let taken = node.edges.binary_search_by_key(&token, |(t, _)| *t).is_ok()
                || node
                    .spills
                    .binary_search_by_key(&token, |(t, _)| *t)
                    .is_ok();
            if taken || rest.is_empty() {
                continue;
            }
            let id = SpillId(self.spills.len() as u32);
            self.spills.push(Spill { next, rest });
            let node = &mut self.nodes[at.0 as usize];
            node.spills.push((token, id));
            node.spills.sort_by_key(|(t, _)| *t);
        }
        self
    }

    pub fn root(&self) -> TrieNodeId {
        self.root
    }

    /// The allowed token set at `pos` — the sorted edge and spill tokens.
    /// Never empty for a reachable non-leaf position.
    pub fn frontier(&self, pos: TrieNodeId) -> Vec<TokenId> {
        let node = &self.nodes[pos.0 as usize];
        let mut out: Vec<TokenId> = node.edges.iter().map(|(t, _)| *t).collect();
        out.extend(node.spills.iter().map(|(t, _)| *t));
        out.sort_unstable();
        out
    }

    /// Step `pos` with `token`.  Returns `None` if `token` is not a live edge
    /// (the caller violated the mask).  A token leading to a leaf yields
    /// `Accept`; one running past an arm, `Spill`; otherwise `Descend`.
    pub fn step(&self, pos: TrieNodeId, token: TokenId) -> Option<Step> {
        let node = &self.nodes[pos.0 as usize];
        let Ok(i) = node.edges.binary_search_by_key(&token, |(t, _)| *t) else {
            return node
                .spills
                .binary_search_by_key(&token, |(t, _)| *t)
                .ok()
                .map(|i| Step::Spill(node.spills[i].1));
        };
        let child = node.edges[i].1;
        let cn = &self.nodes[child.0 as usize];
        match cn.accept {
            Some(next) if cn.edges.is_empty() => Some(Step::Accept(next)),
            _ => Some(Step::Descend(child)),
        }
    }

    /// The spill `id` names.
    pub fn spill(&self, id: SpillId) -> &Spill {
        &self.spills[id.0 as usize]
    }

    /// How many spills the trie offers, at every position.
    pub fn spill_count(&self) -> usize {
        self.spills.len()
    }

    /// Every arm's token sequence, in token order — the choices this trie can
    /// complete, as `build` received them (up to order).
    pub fn arms(&self) -> Vec<Vec<TokenId>> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        self.collect_arms(self.root, &mut path, &mut out);
        out
    }

    fn collect_arms(&self, pos: TrieNodeId, path: &mut Vec<TokenId>, out: &mut Vec<Vec<TokenId>>) {
        let node = &self.nodes[pos.0 as usize];
        if node.accept.is_some() {
            out.push(path.clone());
        }
        for &(tok, child) in &node.edges {
            path.push(tok);
            self.collect_arms(child, path, out);
            path.pop();
        }
    }

    /// Number of distinct arms (accepting nodes) — used by the compiler to fold
    /// single-arm branches.
    pub fn arm_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.accept.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(n: u32) -> NodeId {
        NodeId(n)
    }

    #[test]
    fn builds_and_walks_distinct_arms() {
        // "ab" -> 10, "ac" -> 11  (share 'a', diverge at b/c)
        let trie = TokenTrie::build(&[
            (vec![b'a' as u32, b'b' as u32], nid(10)),
            (vec![b'a' as u32, b'c' as u32], nid(11)),
        ])
        .unwrap();
        assert_eq!(trie.arm_count(), 2);

        let root = trie.root();
        assert_eq!(trie.frontier(root), vec![b'a' as u32]);
        let after_a = match trie.step(root, b'a' as u32).unwrap() {
            Step::Descend(p) => p,
            _ => panic!("expected descend"),
        };
        let mut f = trie.frontier(after_a);
        f.sort();
        assert_eq!(f, vec![b'b' as u32, b'c' as u32]);
        assert_eq!(trie.step(after_a, b'c' as u32), Some(Step::Accept(nid(11))));
    }

    #[test]
    fn arms_lists_every_arm_in_token_order() {
        let trie = TokenTrie::build(&[
            (vec![b'a' as u32, b'c' as u32], nid(11)),
            (vec![b'a' as u32, b'b' as u32], nid(10)),
            (vec![b'z' as u32], nid(12)),
        ])
        .unwrap();
        assert_eq!(
            trie.arms(),
            vec![
                vec![b'a' as u32, b'b' as u32],
                vec![b'a' as u32, b'c' as u32],
                vec![b'z' as u32],
            ]
        );
    }

    #[test]
    fn out_of_mask_token_is_none() {
        let trie = TokenTrie::build(&[(vec![b'a' as u32], nid(1))]).unwrap();
        assert_eq!(trie.step(trie.root(), b'z' as u32), None);
    }

    #[test]
    fn prefix_collision_is_rejected() {
        // "a" is a prefix of "ab".
        let e = TokenTrie::build(&[
            (vec![b'a' as u32], nid(1)),
            (vec![b'a' as u32, b'b' as u32], nid(2)),
        ])
        .unwrap_err();
        assert!(matches!(e, BuildError::AmbiguousArms { .. }));
    }

    #[test]
    fn reverse_prefix_collision_is_rejected() {
        // Insert the longer one first, then its prefix.
        let e = TokenTrie::build(&[
            (vec![b'a' as u32, b'b' as u32], nid(2)),
            (vec![b'a' as u32], nid(1)),
        ])
        .unwrap_err();
        assert!(matches!(e, BuildError::AmbiguousArms { .. }));
    }

    /// **A spill is offered beside an arm's last token and completes the arm**,
    /// carrying its bytes past it; a token that is already a way on is not
    /// made a spill, and other positions are untouched.
    #[test]
    fn a_spill_completes_an_arm_and_carries_what_ran_past_it() {
        // `, "` (as `,` then ` "`) → 10; `]` → 11.
        let (comma, quote, close) = (b',' as TokenId, 300, b']' as TokenId);
        let trie = TokenTrie::build(&[(vec![comma, quote], nid(10)), (vec![close], nid(11))])
            .unwrap()
            .spill_from(
                &[comma, quote],
                vec![
                    (301, b"-".to_vec()),
                    (302, b"--".to_vec()),
                    (quote, b"x".to_vec()),
                ],
            );
        assert_eq!(trie.spill_count(), 2);
        assert_eq!(trie.frontier(trie.root()), vec![comma, close]);
        let Some(Step::Descend(after_comma)) = trie.step(trie.root(), comma) else {
            panic!("`,` walks into the arm");
        };
        assert_eq!(trie.frontier(after_comma), vec![quote, 301, 302]);
        assert_eq!(trie.step(after_comma, quote), Some(Step::Accept(nid(10))));
        let Some(Step::Spill(id)) = trie.step(after_comma, 302) else {
            panic!("` \"--` spills");
        };
        assert_eq!(
            trie.spill(id),
            &Spill {
                next: nid(10),
                rest: b"--".to_vec()
            }
        );
        assert_eq!(trie.step(trie.root(), 301), None);
        // The arms are what they were: a spill is not an arm.
        assert_eq!(trie.arm_count(), 2);
    }

    #[test]
    fn empty_arm_rejected() {
        let e = TokenTrie::build(&[(vec![], nid(1))]).unwrap_err();
        assert!(matches!(e, BuildError::EmptyArm { .. }));
    }

    #[test]
    fn single_token_arms() {
        let trie =
            TokenTrie::build(&[(vec![b't' as u32], nid(1)), (vec![b'f' as u32], nid(2))]).unwrap();
        assert_eq!(
            trie.step(trie.root(), b't' as u32),
            Some(Step::Accept(nid(1)))
        );
        assert_eq!(
            trie.step(trie.root(), b'f' as u32),
            Some(Step::Accept(nid(2)))
        );
    }
}
