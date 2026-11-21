//! Compiles a tool catalog into the tree that constrains a single call, or a
//! turn of several calls.
//!
//! A single-call tree ends the turn with its call. An action loop admits up to a
//! bound of calls and offers the turn terminator after each one, so a character
//! can stop after any act. The loop is unrolled level by level rather than
//! cycled, because the number of calls so far is the property the bound depends
//! on. Splicing one tree onto another makes the join a node edge, not a decoded
//! token that might never fire a trigger. A thinking turn is therefore a single
//! tree that begins inside its reasoning block and flows into its calls.

use super::*;

/// Compile a tool catalog into a [`TreeSpec`].  Errors on an empty catalog or a
/// name/enum collision the trie rejects.
pub fn compile_tool_call_tree(
    tools: &[ToolSpec],
    env: &ToolCallEnvelope,
) -> Result<TreeSpec, BuildError> {
    if tools.is_empty() {
        return Err(BuildError::ToolSchema("empty tool catalog".into()));
    }
    let mut b = ToolTreeBuilder::new(env);
    let end = b.spec.push(NodeSpec::End);

    // Each tool: name arm -> args_open static -> its argument object -> close.
    let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(tools.len());
    let mut shared = HashMap::new();
    for tool in tools {
        let arm_target = b.tool_arm_target(tool, end, &mut shared)?;
        // The arm carries the name's own terminator so prefix-related names
        // stay distinguishable in the trie — see `ToolCallEnvelope::name_close`.
        arms.push((format!("{}{}", tool.name, env.name_close), arm_target));
    }
    let name_branch = b.spec.push(NodeSpec::Branch { arms });
    b.spec.name_branches.insert(name_branch);
    let open = b.spec.push(NodeSpec::Static {
        text: env.open.clone(),
        next: name_branch,
    });
    b.spec.root = open;
    // Failsafe: if a token ever escapes the mask, close the JSON + the tool-call
    // block so the partial output is at least terminated for the extractor.
    b.spec.bail = env.close.clone();
    Ok(b.spec)
}

/// Compile a catalog into a tree that admits **up to `max_calls` calls in one
/// turn**, then ends.
///
/// [`compile_tool_call_tree`] is the assistant shape: one call *is* the whole
/// reply, so its close carries the turn's EOS and the turn is over. An **action
/// loop** is a different thing. A character turning as it speaks or moving as it
/// signals is doing two things in one moment, and a grammar that admits exactly
/// one act cannot express that — it would silently make every turn a single act
/// and nothing would report the amputation.
///
/// So after each call the grammar offers a choice, and the model makes it:
///
/// ```text
///   …}}</tool_call>  ─┬─ "<tool_call>{"name": "  → another act (levels remain)
///                     └─ close_turn              → done
/// ```
///
/// `max_calls` is a **bound, not a target**: every level offers the closing arm,
/// so a character that has said what it means stops at one. The bound exists
/// because the alternative is an unbounded loop the model can sit in — the same
/// runaway the reasoning block had, wearing a different hat.
///
/// The levels are unrolled rather than cycled: a cycle would make "how many so
/// far" a property the tree cannot see, and the count is the whole point.
///
/// `env.close` must **not** carry the turn terminator here — `close_turn` does,
/// on the finishing arm only. An `env.close` ending in EOS would end the turn on
/// the first call and the loop would be unreachable.
/// Copy `sub`'s nodes into `host`, redirecting its `End` to `tail`, and return
/// its root's id in `host`.
///
/// **One tree, not two joined by a trigger.** A trigger fires on a *decoded*
/// token, so a marker the grammar injects as static text does not necessarily
/// fire one — which would leave the model free at exactly the join the grammar
/// exists to close. Splicing makes the transition a node edge instead: the tree
/// walks from the end of one part into the start of the next with no moment in
/// between where control returns to the decoder.
pub(super) fn splice(host: &mut TreeSpec, sub: &TreeSpec, tail: SpecId) -> SpecId {
    let base = host.nodes.len();
    // `End` becomes `tail`; every other id shifts by the host's current length.
    let remap = |id: SpecId| -> SpecId {
        match sub.nodes.get(id.0) {
            Some(NodeSpec::End) => tail,
            _ => SpecId(id.0 + base),
        }
    };
    for node in &sub.nodes {
        let moved = match node {
            // Kept as a node so ids stay dense and `remap` above stays a pure
            // index shift; nothing ever reaches it, because every edge that
            // pointed at it now points at `tail`.
            NodeSpec::End => NodeSpec::End,
            NodeSpec::Static { text, next } => NodeSpec::Static {
                text: text.clone(),
                next: remap(*next),
            },
            NodeSpec::Branch { arms } => NodeSpec::Branch {
                arms: arms.iter().map(|(t, n)| (t.clone(), remap(*n))).collect(),
            },
            NodeSpec::FreeText {
                term,
                eos_ends,
                limits,
                close_token,
                suppress_close,
                next,
            } => NodeSpec::FreeText {
                term: *term,
                eos_ends: *eos_ends,
                limits: *limits,
                close_token: *close_token,
                suppress_close: *suppress_close,
                next: remap(*next),
            },
            NodeSpec::Value {
                pattern,
                opened,
                limits,
                next,
            } => NodeSpec::Value {
                pattern: pattern.clone(),
                opened: *opened,
                limits: *limits,
                next: remap(*next),
            },
            NodeSpec::Element {
                marker,
                tails,
                limits,
                forks,
                next,
            } => NodeSpec::Element {
                marker,
                tails,
                limits: *limits,
                forks: forks.iter().map(|f| remap(*f)).collect(),
                next: remap(*next),
            },
        };
        host.nodes.push(moved);
    }
    for (id, key) in &sub.keys {
        host.keys.insert(SpecId(id.0 + base), Arc::clone(key));
    }
    host.name_branches
        .extend(sub.name_branches.iter().map(|id| SpecId(id.0 + base)));
    for (id, chosen) in &sub.chosen {
        host.chosen.insert(SpecId(id.0 + base), chosen.clone());
    }
    for (tool, fixed) in &sub.fixed {
        host.fixed.insert(Arc::clone(tool), fixed.clone());
    }
    remap(sub.root)
}

pub fn compile_tool_call_loop(
    tools: &[ToolSpec],
    env: &ToolCallEnvelope,
    max_calls: usize,
    close_turn: &str,
) -> Result<TreeSpec, BuildError> {
    compile_action_loop(tools, env, max_calls, close_turn, None)
}

/// Compile a resource's invoke **body** — a JSON object of the schema's fields —
/// into a standalone sub-stencil (effector design §11).
///
/// This is front-end B applied to a body schema's parameters: the same
/// value-typing as a tool's argument object (string enums and booleans enforced
/// exactly; integer/number/array/object shaped as any structurally-valid JSON;
/// nested objects recursed), emitted as a bare `{ … }` object ending at [`End`]
/// so it can be simulated on its own **and** spliced into a turn's `invoke`
/// branch by [`compile_action_loop_with_body`]. Always JSON, whatever the outer
/// call style — an invoke body is a JSON object regardless of how the call that
/// carries it is written.
///
/// An empty schema (no fields) compiles to `{}` and nothing else.
///
/// [`End`]: crate::stencil::StencilNode::End
pub fn compile_invoke_body_tree(params: &[Param]) -> Result<TreeSpec, BuildError> {
    let env = ToolCallEnvelope::qwen3();
    let mut b = ToolTreeBuilder {
        spec: TreeSpec::new(INVOKE_BODY_TREE_LABEL),
        env: &env,
        array_depth: 0,
        body_override: None,
    };
    let end = b.spec.push(NodeSpec::End);
    let fields_entry = b.build_fields(params, "}", end)?;
    // The opening brace is always static — never a token in question.
    let root = b.spec.push(NodeSpec::Static {
        text: "{".to_string(),
        next: fields_entry,
    });
    b.spec.root = root;
    // Failsafe: close the object if a token ever escapes the mask, so a partial
    // body is at least a terminated JSON object for the parser.
    b.spec.bail = "}".to_string();
    Ok(b.spec)
}

/// The action loop, optionally preceded by a reasoning block.
///
/// **The whole turn in one grammar.** With `think` set the tree is: the block's
/// steered spans, its closing tag, then straight into the first call — no moment
/// between them where the decoder is free. Without it the tree starts at the
/// call.
///
/// This is what a mission-specific thinking dial needs. Some work is a thinking
/// problem — drafting a story into an eleven-year silence — and some is not, and
/// the same character must be able to do both. What must not vary is that the
/// turn ends in acts: a character that thinks and then writes prose has done
/// nothing, and the world cannot tell that from a character that chose to wait.
pub fn compile_action_loop(
    tools: &[ToolSpec],
    env: &ToolCallEnvelope,
    max_calls: usize,
    close_turn: &str,
    think: Option<&TreeSpec>,
) -> Result<TreeSpec, BuildError> {
    compile_action_loop_with_body(tools, env, max_calls, close_turn, think, None)
}

/// [`compile_action_loop`] with one tool's one parameter's value **spliced in**
/// from a pre-compiled body sub-stencil rather than free-typed (effector design
/// §11).
///
/// This is the fallback composition for the schema→stencil migration: rather
/// than swap a sub-tree onto the `invoke` branch at decode time (which the turn
/// driver does not expose), the typed body is spliced at compile time, giving a
/// per-`(resource, schema-fingerprint)` turn grammar while the frame it is built
/// from is unchanged. `body_override` is `(tool name, param name, body spec)`;
/// the body spec is a `{ … }` object from [`compile_invoke_body_tree`], and it
/// is spliced as the named parameter's value on every level's call to that tool.
/// `None` leaves every value free-typed, so a no-focus turn is byte-identical to
/// [`compile_action_loop`].
///
/// The override is ignored for [`CallStyle::FunctionBlock`], whose raw values
/// carry a JSON object as free text; only the JSON styles get the typed body.
pub fn compile_action_loop_with_body<'a>(
    tools: &[ToolSpec],
    env: &'a ToolCallEnvelope,
    max_calls: usize,
    close_turn: &str,
    think: Option<&TreeSpec>,
    body_override: Option<(&'a str, &'a str, &'a TreeSpec)>,
) -> Result<TreeSpec, BuildError> {
    if tools.is_empty() {
        return Err(BuildError::ToolSchema("empty tool catalog".into()));
    }
    if max_calls == 0 {
        return Err(BuildError::ToolSchema(
            "a turn that may make no calls is a turn that cannot act".into(),
        ));
    }
    let mut b = ToolTreeBuilder::new(env);
    let end = b.spec.push(NodeSpec::End);
    // The arm that finishes the turn: emit the terminator and stop.
    let finish = b.spec.push(NodeSpec::Static {
        text: close_turn.to_string(),
        next: end,
    });

    // **Built back to front.** Each level's continuation points at the level
    // above it, so the deepest is constructed first and the shallowest ends up
    // as the root. Forwards it would need a node id before the node exists.
    let mut after_call = finish;
    let mut root_open = None;
    for level in (0..max_calls).rev() {
        // One call: choose a name, fill its arguments, close the block.
        let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(tools.len());
        let mut shared = HashMap::new();
        for tool in tools {
            // The typed body applies only to the tool it is armed for; every
            // other tool builds its arguments free-typed as before.
            b.body_override =
                body_override.and_then(|(tn, pn, spec)| (tn == tool.name).then_some((pn, spec)));
            let arm_target = b.tool_arm_target(tool, after_call, &mut shared)?;
            // The name's own terminator, from the envelope — a hardcoded `"`
            // here spliced a JSON quote into a function block and produced act
            // names like `reflect"<parameter=inner_thoughts"`, rejected on
            // every turn. See `ToolCallEnvelope::name_close`.
            arms.push((format!("{}{}", tool.name, env.name_close), arm_target));
        }
        let name_branch = b.spec.push(NodeSpec::Branch { arms });
        b.spec.name_branches.insert(name_branch);
        let open = b.spec.push(NodeSpec::Static {
            text: env.open.clone(),
            next: name_branch,
        });

        // What the *previous* level's close leads to: go again, or finish. The
        // continuation arm carries what the template writes between two calls
        // and then the marker — exactly the text the model would have emitted
        // to start another call — so choosing it is choosing to act again.
        after_call = b.spec.push(NodeSpec::Branch {
            arms: vec![
                (format!("{}{}", env.between_calls, env.marker), open),
                (close_turn.to_string(), end),
            ],
        });
        if level == 0 {
            root_open = Some(open);
        }
    }

    let first_call = root_open.expect("max_calls > 0 builds at least one level");
    // With a reasoning block, the turn starts inside it and flows into the first
    // call as a node edge — see [`splice`] for why that is not a trigger.
    b.spec.root = match think {
        Some(prelude) => splice(&mut b.spec, prelude, first_call),
        None => first_call,
    };
    b.spec.bail = format!("{}{}", env.close, close_turn);
    Ok(b.spec)
}
