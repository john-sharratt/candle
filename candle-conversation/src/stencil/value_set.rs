//! A set of string values as a byte trie: the automaton a held set's
//! [`ValueRule`](super::value_rule::ValueRule) walks (`docs/stencil_tree.md`
//! §9.2.1).
//!
//! A trie is a deterministic automaton already — its nodes are the states, a
//! node's children its transitions, a value's last node accepting — so a set of
//! any size is held without compiling a pattern. Node 0 is the root, the empty
//! prefix. A value a rule could never write is left out: one holding `"` or `\`
//! (a rule admits no escape) or a byte below a space.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// One node: the bytes that continue from it, each to its child, in byte
/// order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Node {
    children: Vec<(u8, u32)>,
    accepting: bool,
}

/// A set of values as a byte trie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueSet {
    nodes: Vec<Node>,
    /// How many values the set holds.
    values: usize,
}

/// Whether a rule could ever write `value`.
fn writable(value: &str) -> bool {
    value.bytes().all(|b| b >= b' ' && b != b'"' && b != b'\\')
}

/// The values of `values` a rule could write, in order, each once.
fn kept<S: AsRef<str>>(values: &[S]) -> Vec<&str> {
    let mut kept: Vec<&str> = values
        .iter()
        .map(AsRef::as_ref)
        .filter(|v| writable(v))
        .collect();
    kept.sort_unstable();
    kept.dedup();
    kept
}

/// The digest of `kept`'s values, in order.
fn digest_of(kept: &[&str]) -> u64 {
    let mut h = DefaultHasher::new();
    for value in kept {
        value.hash(&mut h);
    }
    h.finish()
}

impl ValueSet {
    /// What the set of `values` would be known by — its digest and size —
    /// without building it: a caller holding sets by key builds one only when
    /// it holds none so known.
    pub fn key_of<S: AsRef<str>>(values: &[S]) -> (u64, usize) {
        let kept = kept(values);
        (digest_of(&kept), kept.len())
    }

    /// The set of `values`, those a rule could never write left out; the
    /// order and repeats of `values` do not matter.
    pub fn of<S: AsRef<str>>(values: &[S]) -> ValueSet {
        Self::built(&kept(values))
    }

    /// The set of exactly `values`, none left out: the strings that close a
    /// raw value (`</parameter>` and the tags after it), which hold the
    /// layout's own line breaks.
    pub fn literal(values: &[&str]) -> ValueSet {
        let mut kept = values.to_vec();
        kept.sort_unstable();
        kept.dedup();
        Self::built(&kept)
    }

    fn built(kept: &[&str]) -> ValueSet {
        let mut nodes = vec![Node::default()];
        for value in kept {
            let mut at = 0usize;
            for &byte in value.as_bytes() {
                at = match nodes[at].children.binary_search_by_key(&byte, |c| c.0) {
                    Ok(i) => nodes[at].children[i].1 as usize,
                    Err(i) => {
                        let child = nodes.len() as u32;
                        nodes.push(Node::default());
                        nodes[at].children.insert(i, (byte, child));
                        child as usize
                    }
                };
            }
            nodes[at].accepting = true;
        }
        ValueSet {
            nodes,
            values: kept.len(),
        }
    }

    /// How many values it holds.
    pub fn len(&self) -> usize {
        self.values
    }

    pub fn is_empty(&self) -> bool {
        self.values == 0
    }

    /// How many nodes its trie has.
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The node `byte` leads to from `node`, when one does.
    pub fn step(&self, node: u32, byte: u8) -> Option<u32> {
        let children = &self.nodes[node as usize].children;
        children
            .binary_search_by_key(&byte, |c| c.0)
            .ok()
            .map(|i| children[i].1)
    }

    /// Whether a value may end at `node`.
    pub fn accepting(&self, node: u32) -> bool {
        self.nodes[node as usize].accepting
    }

    /// Which nodes a walk can finish from in tokens of one byte each, given
    /// the bytes that have such a token (`single`) and whether the closing
    /// quote does (`close`): a node accepting with a quote to close it, or one
    /// with a child so reached that can finish.
    pub fn finishable(&self, single: &[bool; 256], close: bool) -> Vec<bool> {
        let mut done = vec![false; self.nodes.len()];
        // Every child is made after its parent, so walking the nodes from the
        // last decides each child before the node that leads to it.
        for at in (0..self.nodes.len()).rev() {
            let node = &self.nodes[at];
            done[at] = (node.accepting && close)
                || node
                    .children
                    .iter()
                    .any(|&(byte, child)| single[byte as usize] && done[child as usize]);
        }
        done
    }

    /// [`Self::finishable`] for a set whose every value ends the walk the
    /// moment it is written, as the strings closing a raw value do: a node
    /// finishes when a one-byte token leads on to a value's end, or to a node
    /// that finishes.
    pub fn finishable_on_arrival(&self, single: &[bool; 256]) -> Vec<bool> {
        let mut done = vec![false; self.nodes.len()];
        for at in (0..self.nodes.len()).rev() {
            done[at] = self.nodes[at].children.iter().any(|&(byte, child)| {
                single[byte as usize]
                    && (self.nodes[child as usize].accepting || done[child as usize])
            });
        }
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(set: &ValueSet, value: &str) -> Option<u32> {
        value
            .bytes()
            .try_fold(0u32, |node, byte| set.step(node, byte))
    }

    /// **Exactly the set's values reach an accepting node**; a prefix of one
    /// walks but does not accept, and anything else does not walk.
    #[test]
    fn a_set_accepts_its_values_and_nothing_else() {
        let set = ValueSet::of(&["src/lib.rs", "src/main.rs", "src", "README.md"]);
        assert_eq!(set.len(), 4);
        for value in ["src/lib.rs", "src/main.rs", "src", "README.md"] {
            assert!(set.accepting(walk(&set, value).unwrap()), "{value}");
        }
        assert!(!set.accepting(walk(&set, "src/").unwrap()));
        assert!(!set.accepting(walk(&set, "src/lib").unwrap()));
        assert_eq!(walk(&set, "src/turn.rs"), None);
        assert_eq!(walk(&set, "Cargo.toml"), None);
        assert!(!set.accepting(0), "the empty value is not in the set");
    }

    /// The empty value is a value like any other.
    #[test]
    fn the_empty_value_accepts_at_the_root() {
        let set = ValueSet::of(&["", "docs"]);
        assert!(set.accepting(0));
        assert_eq!(set.len(), 2);
    }

    /// **A value no rule could write is left out**, and the order and repeats
    /// of the values given change nothing — not even the key a set is known by.
    #[test]
    fn unwritable_values_are_left_out_and_order_does_not_matter() {
        let set = ValueSet::of(&["a\"b", "a\\b", "tab\there", "ok"]);
        assert_eq!(set.len(), 1);
        assert!(set.accepting(walk(&set, "ok").unwrap()));
        assert_eq!(ValueSet::key_of(&["a\"b", "ok"]), ValueSet::key_of(&["ok"]));
        let a = ValueSet::of(&["b", "a", "b"]);
        let b = ValueSet::of(&["a", "b"]);
        assert_eq!(a, b);
        assert_eq!(
            ValueSet::key_of(&["b", "a", "b"]),
            ValueSet::key_of(&["a", "b"])
        );
        assert_eq!(ValueSet::key_of(&["a", "b"]).1, a.len());
        assert_ne!(ValueSet::key_of(&["a", "b"]), ValueSet::key_of(&["a", "c"]));
    }

    /// **A node finishes when it accepts and can close, or one byte token
    /// leads on to a node that finishes**; a byte with no token of its own
    /// leads nowhere a walk could finish from.
    #[test]
    fn finishable_follows_the_single_byte_tokens() {
        let set = ValueSet::of(&["ab", "c"]);
        let mut single = [true; 256];
        let all = set.finishable(&single, true);
        assert!(all.iter().all(|f| *f));
        single[b'b' as usize] = false;
        let no_b = set.finishable(&single, true);
        let a = walk(&set, "a").unwrap() as usize;
        assert!(!no_b[a], "`a` can only go on by `b`");
        assert!(no_b[0], "`c` still finishes from the root");
        let no_close = set.finishable(&[true; 256], false);
        assert!(
            no_close.iter().all(|f| !*f),
            "nothing closes without a quote"
        );
    }

    /// **A literal set keeps every string**, line breaks included, which `of`
    /// leaves out.
    #[test]
    fn a_literal_set_keeps_its_line_breaks() {
        let closes = ValueSet::literal(&["</p>\n<p=", "</p>\r\n<p=", "</p>\n<p="]);
        assert_eq!(closes.len(), 2);
        assert!(closes.accepting(walk(&closes, "</p>\n<p=").unwrap()));
        assert!(closes.accepting(walk(&closes, "</p>\r\n<p=").unwrap()));
        assert_eq!(ValueSet::of(&["</p>\n<p="]).len(), 0);
    }

    /// **A string that ends the walk on arrival finishes its parent**, when a
    /// one-byte token writes its last byte; an end only a missing byte reaches
    /// finishes nothing before it.
    #[test]
    fn finishable_on_arrival_follows_the_single_byte_tokens() {
        let closes = ValueSet::literal(&["ab", "c"]);
        let mut single = [true; 256];
        let all = closes.finishable_on_arrival(&single);
        let a = walk(&closes, "a").unwrap() as usize;
        assert!(all[0] && all[a]);
        let ab = walk(&closes, "ab").unwrap() as usize;
        assert!(!all[ab], "an end has nothing left to finish");
        single[b'b' as usize] = false;
        let no_b = closes.finishable_on_arrival(&single);
        assert!(!no_b[a], "`a` can only end by `b`");
        assert!(no_b[0], "`c` still ends from the root");
    }
}
