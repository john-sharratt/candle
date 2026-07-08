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

use super::error::BuildError;
use super::tree::NodeId;
use super::vocab::TokenId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrieNodeId(pub u32);

#[derive(Debug, Clone)]
struct TrieNode {
    /// (token, child), kept sorted by token for a binary-searchable frontier.
    edges: Vec<(TokenId, TrieNodeId)>,
    /// `Some(next)` when this node completes an arm.  An accepting node is
    /// always a leaf (`edges.is_empty()`), guaranteed by `build`.
    accept: Option<NodeId>,
}

#[derive(Debug, Clone)]
pub struct TokenTrie {
    nodes: Vec<TrieNode>,
    root: TrieNodeId,
}

/// The result of stepping a trie with a decoded token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Keep walking; the new trie position.
    Descend(TrieNodeId),
    /// An arm completed; resume the outer tree at this node.
    Accept(NodeId),
}

impl TokenTrie {
    /// Build a trie from `(arm tokens, successor)` pairs.  Errors on an empty
    /// arm or on a token-level prefix collision between two arms.
    pub fn build(arms: &[(Vec<TokenId>, NodeId)]) -> Result<TokenTrie, BuildError> {
        let mut nodes = vec![TrieNode {
            edges: Vec::new(),
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

        Ok(TokenTrie { nodes, root })
    }

    pub fn root(&self) -> TrieNodeId {
        self.root
    }

    /// The allowed token set at `pos` — the sorted edge tokens.  Never empty for
    /// a reachable non-leaf position.
    pub fn frontier(&self, pos: TrieNodeId) -> Vec<TokenId> {
        self.nodes[pos.0 as usize]
            .edges
            .iter()
            .map(|(t, _)| *t)
            .collect()
    }

    /// Step `pos` with `token`.  Returns `None` if `token` is not a live edge
    /// (the caller violated the mask).  A token leading to a leaf yields
    /// `Accept`; otherwise `Descend`.
    pub fn step(&self, pos: TrieNodeId, token: TokenId) -> Option<Step> {
        let node = &self.nodes[pos.0 as usize];
        let child = node
            .edges
            .binary_search_by_key(&token, |(t, _)| *t)
            .ok()
            .map(|i| node.edges[i].1)?;
        let cn = &self.nodes[child.0 as usize];
        match cn.accept {
            Some(next) if cn.edges.is_empty() => Some(Step::Accept(next)),
            _ => Some(Step::Descend(child)),
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
