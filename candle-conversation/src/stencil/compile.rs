//! The single compile backend: `TreeSpec` → `StencilTree`.
//!
//! One pass lowers the spec bottom-up, simultaneously **folding** single-arm
//! branches into static text, **fusing** adjacent static runs into one maximal
//! node, and **tokenizing in context** (so boundary merges match a real decode).
//! It then verifies the invariants the runtime relies on: no two adjacent
//! `Static` nodes, every `Branch` has ≥2 arms, every `FreeText` has a hard limit.

use super::error::BuildError;
use super::spec::{NodeSpec, SpecId, TreeSpec};
use super::tree::{FreeTextSpan, NodeId, StencilNode, StencilTree};
use super::trie::TokenTrie;
use super::vocab::{TokenId, Vocab};

/// Compile a string-space spec against a tokenizer.
pub fn compile(spec: &TreeSpec, vocab: &dyn Vocab) -> Result<StencilTree, BuildError> {
    validate(spec)?;
    let mut arena: Vec<StencilNode> = Vec::new();
    let root = lower(spec, vocab, spec.root, "", &mut arena)?;
    verify_invariants(&arena)?;
    // The bail set is tokenized standalone — it is emitted from an unknown point
    // (wherever the failsafe fires), so there is no stable left context.
    let bail = vocab.encode(&spec.bail);
    Ok(StencilTree::new(
        arena,
        root,
        vocab.eos(),
        vocab.fingerprint(),
        spec.label.clone(),
        bail,
    ))
}

// ── Validation (refs, cycles, structural) ───────────────────────────────────

fn validate(spec: &TreeSpec) -> Result<(), BuildError> {
    let n = spec.nodes.len();
    if spec.root.0 >= n {
        return Err(BuildError::BadRoot(spec.root.0));
    }
    for (i, node) in spec.nodes.iter().enumerate() {
        for s in successors(node) {
            if s.0 >= n {
                return Err(BuildError::BadRef { from: i, to: s.0 });
            }
        }
    }
    let mut color = vec![0u8; n]; // 0 white, 1 gray (on stack), 2 black
    let mut reaches_end = vec![false; n];
    dfs(spec, spec.root, &mut color, &mut reaches_end)?;
    if !reaches_end[spec.root.0] {
        return Err(BuildError::NoEnd(spec.root.0));
    }
    Ok(())
}

fn successors(node: &NodeSpec) -> Vec<SpecId> {
    match node {
        NodeSpec::Static { next, .. } => vec![*next],
        NodeSpec::FreeText { next, .. } => vec![*next],
        NodeSpec::Branch { arms } => arms.iter().map(|(_, s)| *s).collect(),
        NodeSpec::End => vec![],
    }
}

fn dfs(
    spec: &TreeSpec,
    n: SpecId,
    color: &mut [u8],
    reaches_end: &mut [bool],
) -> Result<bool, BuildError> {
    match color[n.0] {
        1 => return Err(BuildError::Cycle(n.0)),
        2 => return Ok(reaches_end[n.0]),
        _ => {}
    }
    color[n.0] = 1;
    let re = match spec.node(n) {
        NodeSpec::End => true,
        NodeSpec::Static { next, .. } | NodeSpec::FreeText { next, .. } => {
            dfs(spec, *next, color, reaches_end)?
        }
        NodeSpec::Branch { arms } => {
            if arms.is_empty() {
                return Err(BuildError::EmptyBranch(n.0));
            }
            let mut all = true;
            for (_, s) in arms {
                all &= dfs(spec, *s, color, reaches_end)?;
            }
            all
        }
    };
    reaches_end[n.0] = re;
    color[n.0] = 2;
    Ok(re)
}

// ── Lowering (fold + fuse + tokenize, bottom-up) ────────────────────────────

fn lower(
    spec: &TreeSpec,
    vocab: &dyn Vocab,
    cur: SpecId,
    prefix: &str,
    arena: &mut Vec<StencilNode>,
) -> Result<NodeId, BuildError> {
    // Gather a maximal run of statics and single-arm branches (fold + fuse).
    let mut run = String::new();
    let mut node = cur;
    loop {
        match spec.node(node) {
            NodeSpec::Static { text, next } => {
                run.push_str(text);
                node = *next;
            }
            NodeSpec::Branch { arms } if arms.len() == 1 => {
                run.push_str(&arms[0].0);
                node = arms[0].1;
            }
            _ => break,
        }
    }
    match spec.node(node) {
        // A multi-arm branch: heal the run→branch boundary.  A real BPE
        // tokenizer merges across grammar boundaries (e.g. `{` + `"key"` → `{"`),
        // so the branch point is placed at the token where the arms *diverge*,
        // not at the grammar boundary — any merged token is absorbed into the
        // arms.  (Single-arm branches were folded into `run` above.)
        NodeSpec::Branch { arms } => lower_branch(spec, vocab, prefix, &run, node.0, arms, arena),
        _ => {
            let term = lower_terminal(spec, vocab, node, arena)?;
            if run.is_empty() {
                Ok(term)
            } else {
                let tokens = tokenize_in_context(vocab, prefix, &run, node.0)?;
                Ok(push(arena, StencilNode::Static { tokens, next: term }))
            }
        }
    }
}

/// Lower a multi-arm branch reached after a static `run`, healing the
/// boundary: tokenize each arm in the full left context, take the longest
/// common token prefix as the static run to prefill, and start the branch at
/// the first diverging token of each arm.
fn lower_branch(
    spec: &TreeSpec,
    vocab: &dyn Vocab,
    prefix: &str,
    run: &str,
    node: usize,
    arms: &[(String, SpecId)],
    arena: &mut Vec<StencilNode>,
) -> Result<NodeId, BuildError> {
    let pre = vocab.encode(prefix);
    let context = format!("{prefix}{run}");
    let fulls: Vec<Vec<TokenId>> = arms
        .iter()
        .map(|(s, _)| vocab.encode(&format!("{context}{s}")))
        .collect();
    let common = longest_common_prefix(&fulls);
    // The healed static can only retract tokens belonging to `run`; it must not
    // reach into tokens the ancestors already committed (`pre`).
    if common < pre.len() {
        return Err(BuildError::BoundaryMerge {
            node,
            segment: arms.first().map(|(s, _)| s.clone()).unwrap_or_default(),
            pullback: pre.len() - common,
        });
    }
    let static_toks = fulls[0][pre.len()..common].to_vec();

    let mut trie_arms: Vec<(Vec<TokenId>, NodeId)> = Vec::with_capacity(arms.len());
    for ((arm_str, arm_next), full) in arms.iter().zip(&fulls) {
        let arm_toks = full[common..].to_vec();
        if arm_toks.is_empty() {
            return Err(BuildError::EmptyArm {
                arm: arm_str.clone(),
            });
        }
        // The arm's tokens are committed by the branch; its successor is
        // prefilled or decoded against that committed boundary, so lower it from
        // a FRESH boundary (empty prefix).  This stops the successor's first
        // static run from merging backward into the committed arm — e.g. the
        // name arm's closing `"` with the args_open `,` → `",`, or a value's
        // opening `"` with the preceding space → ` "` — which would be an
        // unrepresentable retract.  (Same rule as a free-text successor.)
        let next_id = lower(spec, vocab, *arm_next, "", arena)?;
        trie_arms.push((arm_toks, next_id));
    }
    let trie = TokenTrie::build(&trie_arms)?;
    let branch = push(arena, StencilNode::Branch { trie });
    if static_toks.is_empty() {
        Ok(branch)
    } else {
        Ok(push(
            arena,
            StencilNode::Static {
                tokens: static_toks,
                next: branch,
            },
        ))
    }
}

/// The number of leading tokens shared by every sequence.
fn longest_common_prefix(seqs: &[Vec<TokenId>]) -> usize {
    let Some(first) = seqs.first() else {
        return 0;
    };
    seqs[1..]
        .iter()
        .fold(first.len(), |n, s| n.min(common_prefix_len(first, s)))
}

fn lower_terminal(
    spec: &TreeSpec,
    vocab: &dyn Vocab,
    cur: SpecId,
    arena: &mut Vec<StencilNode>,
) -> Result<NodeId, BuildError> {
    match spec.node(cur) {
        NodeSpec::End => Ok(push(arena, StencilNode::End)),
        NodeSpec::FreeText {
            term,
            eos_ends,
            limits,
            close_token,
            suppress_close,
            next,
        } => {
            // A free-text value is unknown at compile time, so its successor is
            // tokenized from a FRESH boundary (empty prefix) rather than in the
            // value's context.  The runtime keeps that boundary clean:
            //
            // - Consumed-close (JSON string): the model's `"`+delimiter merge is
            //   healed — only the re-tokenized valid prefix is committed, the
            //   delimiter is dropped and re-emitted by the successor.
            // - Lookahead (number / JSON value): a delimiter that is its own
            //   token is pushed back into the successor; a delimiter merged with
            //   value bytes is healed like the string case.
            //
            // Lowering from a fresh boundary also avoids a compile-time merge
            // that would otherwise have to retract a committed byte of the value
            // context (the opening quote, or the value's last digit) — which is
            // unrepresentable.
            let next_id = lower(spec, vocab, *next, "", arena)?;
            Ok(push(
                arena,
                StencilNode::FreeText(FreeTextSpan {
                    term: *term,
                    eos_ends: *eos_ends,
                    limits: *limits,
                    // The close token is already a resolved id from the front-end;
                    // copy it verbatim (never tokenize it in context).
                    close_token: *close_token,
                    suppress_close: *suppress_close,
                    next: next_id,
                }),
            ))
        }
        NodeSpec::Static { .. } | NodeSpec::Branch { .. } => {
            unreachable!("statics and branches are consumed by lower()")
        }
    }
}

fn push(arena: &mut Vec<StencilNode>, node: StencilNode) -> NodeId {
    let id = NodeId(arena.len() as u32);
    arena.push(node);
    id
}

/// Tokenize `segment` as it appears after `prefix`.  Errors if the segment's
/// first token merges with the prefix's tail (a boundary the grammar must avoid).
fn tokenize_in_context(
    vocab: &dyn Vocab,
    prefix: &str,
    segment: &str,
    node: usize,
) -> Result<Vec<TokenId>, BuildError> {
    let pre = vocab.encode(prefix);
    let full = vocab.encode(&format!("{prefix}{segment}"));
    let k = common_prefix_len(&pre, &full);
    let pullback = pre.len() - k;
    if pullback > 0 {
        return Err(BuildError::BoundaryMerge {
            node,
            segment: segment.to_string(),
            pullback,
        });
    }
    Ok(full[k..].to_vec())
}

fn common_prefix_len(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

// ── Invariants ──────────────────────────────────────────────────────────────

fn verify_invariants(arena: &[StencilNode]) -> Result<(), BuildError> {
    for (i, node) in arena.iter().enumerate() {
        match node {
            StencilNode::Static { next, .. } => {
                if let StencilNode::Static { .. } = arena[next.0 as usize] {
                    // Should be impossible after fusion.
                    return Err(BuildError::BoundaryMerge {
                        node: i,
                        segment: String::from("<adjacent static>"),
                        pullback: 0,
                    });
                }
            }
            StencilNode::Branch { trie } => {
                debug_assert!(trie.arm_count() >= 2, "branch must have >=2 arms");
            }
            StencilNode::FreeText(span) => {
                if span.limits.forced_after == 0 {
                    return Err(BuildError::NoHardLimit(i));
                }
            }
            StencilNode::End => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::terminator::Terminator;
    use crate::stencil::tree::FreeTextLimits;
    use crate::stencil::vocab::TestVocab;

    // Build a tiny spec by hand: Static "ab" -> FreeText(string) -> Static "}" -> End.
    fn linear_spec() -> TreeSpec {
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let close = s.push(NodeSpec::Static {
            text: "}".into(),
            next: end,
        });
        let val = s.push(NodeSpec::FreeText {
            term: Terminator::JsonString,
            eos_ends: false,
            limits: FreeTextLimits::json_string(),
            close_token: None,
            suppress_close: false,
            next: close,
        });
        let open = s.push(NodeSpec::Static {
            text: "\"".into(),
            next: val,
        });
        s.root = open;
        s
    }

    #[test]
    fn compiles_linear() {
        let v = TestVocab::new();
        let tree = compile(&linear_spec(), &v).unwrap();
        // root: Static "\"" -> FreeText -> Static "}" -> End  (4 nodes)
        assert_eq!(tree.len(), 4);
        match tree.node(tree.root()) {
            StencilNode::Static { tokens, .. } => assert_eq!(tokens, &[b'"' as u32]),
            _ => panic!("root should be Static"),
        }
    }

    #[test]
    fn fuses_adjacent_statics() {
        // Static "a" -> Static "b" -> End  should fuse to one Static "ab".
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let b = s.push(NodeSpec::Static {
            text: "b".into(),
            next: end,
        });
        let a = s.push(NodeSpec::Static {
            text: "a".into(),
            next: b,
        });
        s.root = a;
        let tree = compile(&s, &TestVocab::new()).unwrap();
        assert_eq!(tree.len(), 2); // fused Static + End
        match tree.node(tree.root()) {
            StencilNode::Static { tokens, .. } => {
                assert_eq!(tokens, &[b'a' as u32, b'b' as u32])
            }
            _ => panic!(),
        }
    }

    #[test]
    fn folds_single_arm_branch() {
        // Branch with one arm "x" -> End  folds into Static "x".
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let br = s.push(NodeSpec::Branch {
            arms: vec![("x".into(), end)],
        });
        s.root = br;
        let tree = compile(&s, &TestVocab::new()).unwrap();
        match tree.node(tree.root()) {
            StencilNode::Static { tokens, .. } => assert_eq!(tokens, &[b'x' as u32]),
            _ => panic!("single-arm branch should fold to Static"),
        }
    }

    #[test]
    fn two_arm_branch_kept() {
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let br = s.push(NodeSpec::Branch {
            arms: vec![("ab".into(), end), ("cd".into(), end)],
        });
        s.root = br;
        let tree = compile(&s, &TestVocab::new()).unwrap();
        // Branch node + (End duplicated per arm because lower re-lowers).
        match tree.node(tree.root()) {
            StencilNode::Branch { trie } => assert_eq!(trie.arm_count(), 2),
            _ => panic!(),
        }
    }

    #[test]
    fn rejects_cycle() {
        let mut s = TreeSpec::new("t");
        // a -> a (cycle)
        let a = s.push(NodeSpec::Static {
            text: "x".into(),
            next: SpecId(0),
        });
        s.root = a;
        assert_eq!(
            compile(&s, &TestVocab::new()).unwrap_err(),
            BuildError::Cycle(0)
        );
    }

    #[test]
    fn rejects_bad_ref() {
        let mut s = TreeSpec::new("t");
        s.push(NodeSpec::Static {
            text: "x".into(),
            next: SpecId(99),
        });
        s.root = SpecId(0);
        assert!(matches!(
            compile(&s, &TestVocab::new()),
            Err(BuildError::BadRef { .. })
        ));
    }

    #[test]
    fn heals_branch_boundary_merge() {
        // Vocab where "a{" is one token: a Static "a" then a Branch whose arms
        // start "{".  The `a{` merge is HEALED — `a{` becomes the prefilled
        // static and the branch starts at the diverging token (x / y).  This is
        // exactly the real-BPE `{` + `"key"` → `{"` case.
        let v = TestVocab::new().with_special("a{", 300);
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let br = s.push(NodeSpec::Branch {
            arms: vec![("{x".into(), end), ("{y".into(), end)],
        });
        let a = s.push(NodeSpec::Static {
            text: "a".into(),
            next: br,
        });
        s.root = a;
        let tree = compile(&s, &v).unwrap();
        match tree.node(tree.root()) {
            // Static [300] (= "a{") then a 2-arm branch over the divergence.
            StencilNode::Static { tokens, next } => {
                assert_eq!(tokens, &[300]);
                match tree.node(*next) {
                    StencilNode::Branch { trie } => assert_eq!(trie.arm_count(), 2),
                    _ => panic!("expected a branch after the healed static"),
                }
            }
            _ => panic!("expected a healed Static -> Branch"),
        }
    }

    #[test]
    fn heals_merge_that_consumes_the_whole_run() {
        // The entire run merges into the arms: Static "{" then a branch whose
        // arms start `"`/`}`, with `{"` and `{}` both single tokens.  No static
        // is emitted; the `{` rides each branch token.
        let v = TestVocab::new()
            .with_special("{\"", 300)
            .with_special("{}", 301);
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let br = s.push(NodeSpec::Branch {
            arms: vec![("\"k".into(), end), ("}z".into(), end)],
        });
        let open = s.push(NodeSpec::Static {
            text: "{".into(),
            next: br,
        });
        s.root = open;
        let tree = compile(&s, &v).unwrap();
        // Root is the branch directly (no static): arms [300,'k'] and [301,'z'].
        match tree.node(tree.root()) {
            StencilNode::Branch { trie } => assert_eq!(trie.arm_count(), 2),
            _ => panic!("expected the branch at the root (run fully absorbed)"),
        }
    }

    #[test]
    fn in_context_merge_within_run_is_fine() {
        // "{\"" is one token; a single Static "{\"" tokenizes to [300], no error.
        let v = TestVocab::new().with_special("{\"", 300);
        let mut s = TreeSpec::new("t");
        let end = s.push(NodeSpec::End);
        let st = s.push(NodeSpec::Static {
            text: "{\"".into(),
            next: end,
        });
        s.root = st;
        let tree = compile(&s, &v).unwrap();
        match tree.node(tree.root()) {
            StencilNode::Static { tokens, .. } => assert_eq!(tokens, &[300]),
            _ => panic!(),
        }
    }
}
