//! Wire the live `zend_tools` registry into one large stencil tree and simulate
//! the decode loop against it — positive (every tool produces valid JSON) and
//! negative (the stencil masks out every malformed attempt).
//!
//! The "model" here is a deterministic driver that follows a target call text:
//! at each masked/free decode it offers the next target byte and checks the
//! stencil's mask; prefilled static runs are emitted automatically.  This is the
//! decode loop, minus the forward pass.

use std::sync::Arc;

use candle_conversation::stencil::{
    compile, compile_tool_call_tree, AllowedSet, HfVocab, Param, ParamType, StencilAction,
    StencilNode, StencilSession, StencilTree, TestVocab, ToolCallEnvelope, ToolSpec, Vocab,
    WalkError,
};

// ── Building the tree from the live registry ────────────────────────────────

fn catalog() -> Vec<ToolSpec> {
    zend::tool_def::all()
        .iter()
        .map(|d| ToolSpec::from_json_schema(&d.name, &d.parameters))
        .collect()
}

fn build_tree() -> (Arc<StencilTree>, TestVocab) {
    let tools = catalog();
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3())
        .expect("the whole registry compiles to one tree");
    let vocab = TestVocab::new();
    let tree = compile(&spec, &vocab).expect("the spec tokenizes");
    (Arc::new(tree), vocab)
}

/// The full registry must also compile against a tokenizer that *merges* across
/// grammar boundaries — the realistic case a real BPE tokenizer (Qwen3) creates
/// at JSON structure (`{"`, `":`, `",`, `"}`, ` "`, digit+delimiter).  These are
/// exactly the boundary classes that previously broke compilation; if a new one
/// exists in some tool's shape, this fails on CPU naming the node instead of on
/// the GPU at daemon startup.
#[test]
fn full_catalog_compiles_with_structural_json_merges() {
    let tools = catalog();
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3())
        .expect("the whole registry compiles to one tree");
    let vocab = TestVocab::new()
        .with_special("{\"", 300)
        .with_special("\":", 301)
        .with_special("\",", 302)
        .with_special("\"}", 303)
        .with_special(" \"", 304)
        .with_special(", \"", 305)
        .with_special("\": \"", 306)
        .with_special("\": ", 307)
        .with_special("{}", 308)
        .with_special("0,", 309)
        .with_special("0}", 310)
        .with_special("\"]", 311)
        .with_special("[\"", 312);
    let tree =
        compile(&spec, &vocab).expect("full catalog must compile against structural JSON merges");
    assert!(tree.len() > 50);
}

/// Locate the cached Qwen3 `tokenizer.json` the daemon downloads (zend cache,
/// then the HF hub snapshot dirs).  Returns `None` if it was never fetched.
fn cached_qwen3_tokenizer() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(std::path::PathBuf::from)?;
    let zend = home.join(".cache/zend/models/tokenizer.json");
    if zend.exists() {
        return Some(zend);
    }
    // HF hub: ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/<commit>/
    let hub = home.join(".cache/huggingface/hub");
    for repo in ["models--Qwen--Qwen3-30B-A3B", "models--Qwen--Qwen3-8B"] {
        let snaps = hub.join(repo).join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snaps) {
            for e in entries.flatten() {
                let p = e.path().join("tokenizer.json");
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// The authoritative boundary check: compile the full registry against the
/// REAL Qwen3 BPE tokenizer (not the byte-level `TestVocab`, whose pure
/// longest-match doesn't model BPE merge ranking).  Ignored by default because
/// it needs the cached tokenizer; run with `--ignored` (or it runs at daemon
/// startup anyway).
#[test]
#[ignore = "requires the cached Qwen3 tokenizer.json"]
fn full_catalog_compiles_against_real_qwen3_tokenizer() {
    let Some(path) = cached_qwen3_tokenizer() else {
        panic!("Qwen3 tokenizer.json not cached — run the daemon once to fetch it");
    };
    let tok = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer.json");
    // eos/fingerprint are irrelevant to compilation (the tool grammar is not
    // eos-terminated), so any value works.
    let vocab = HfVocab::new(tok, &[0], 0);
    let tools = catalog();
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3())
        .expect("the whole registry compiles to one tree");
    let tree = compile(&spec, &vocab).expect("full catalog compiles against real Qwen3 BPE");
    assert!(tree.len() > 50);
    eprintln!("real-tokenizer tree: {} nodes", tree.len());
}

/// Walk `session` along `target` with a real vocabulary: prefills must match
/// it, a branch takes the longest allowed token that is a prefix of what is
/// left, and a free span takes one character per token. Returns how many
/// delimiters the session dropped on the way — the grammar then writes them.
fn follow_real(
    session: &mut StencilSession,
    vocab: &HfVocab,
    target: &str,
    out: &mut Vec<u32>,
) -> usize {
    use candle_conversation::stencil::Observe;
    let mut pos = 0usize;
    let mut dropped = 0usize;
    while pos < target.len() {
        match session.next_action() {
            StencilAction::Prefill(toks) => {
                let text = String::from_utf8(vocab.decode(&toks)).unwrap();
                assert!(
                    target[pos..].starts_with(&text),
                    "prefilled {text:?} where {:?} was wanted",
                    &target[pos..]
                );
                pos += text.len();
                out.extend(toks);
            }
            StencilAction::MaskedDecode(set) => {
                let (t, len) = set
                    .tokens()
                    .iter()
                    .map(|&t| (t, vocab.token_bytes(t)))
                    .filter(|(_, b)| target.as_bytes()[pos..].starts_with(b))
                    .map(|(t, b)| (t, b.len()))
                    .max_by_key(|&(_, len)| len)
                    .unwrap_or_else(|| panic!("no allowed token writes {:?}", &target[pos..]));
                let obs = session.observe(t, &vocab.token_bytes(t)).unwrap();
                assert!(
                    matches!(obs, Observe::Continue | Observe::ArmComplete),
                    "a branch token was refused: {obs:?}"
                );
                pos += len;
                out.push(t);
            }
            StencilAction::FreeDecode { .. } => {
                let c = target[pos..].chars().next().unwrap();
                let toks = vocab.encode(&c.to_string());
                assert_eq!(toks.len(), 1, "{c:?} is not one token");
                match session.observe(toks[0], c.to_string().as_bytes()).unwrap() {
                    Observe::DelimiterDropped => dropped += 1,
                    _ => {
                        pos += c.len_utf8();
                        out.push(toks[0]);
                    }
                }
            }
            StencilAction::Exit => panic!("the walk ended with {:?} unwritten", &target[pos..]),
        }
    }
    dropped
}

/// **The live `read_files` failure, against the real Qwen3 BPE.** Cline's
/// native `read_files` takes an array of `{path, start_line, end_line}`; left a
/// free value, the model closed it `"end_line": 3420]}}}` — no `}` for the
/// element — and the call was not JSON. The grammar now writes the array's
/// structure, and the `]`-led token where the element's `}` belongs is dropped
/// rather than committed, so the call that comes out parses.
///
/// Real tokens, because the merges this grammar sits between (`[{`, `}]`,
/// `, "`) are the tokenizer's, not the byte vocabulary's. Also prints how long
/// the catalog took to compile: passthrough compiles it per conversation.
#[test]
#[ignore = "requires the cached Qwen3 tokenizer.json"]
fn cline_read_files_with_a_misplaced_bracket_parses_with_real_tokens() {
    use candle_conversation::stencil::Observe;
    let Some(path) = cached_qwen3_tokenizer() else {
        panic!("Qwen3 tokenizer.json not cached — run the daemon once to fetch it");
    };
    let tok = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer.json");
    let vocab = HfVocab::new(tok, &[0], 0);
    let tools = [
        ToolSpec::from_json_schema(
            "read_files",
            &serde_json::json!({
                "type": "object",
                "properties": {"files": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": ["number", "null"]},
                        "end_line": {"type": ["number", "null"]}
                    },
                    "required": ["path"]
                }}},
                "required": ["files"]
            }),
        ),
        ToolSpec::from_json_schema(
            "run_commands",
            &serde_json::json!({
                "type": "object",
                "properties": {"commands": {"type": "array", "items": {"type": "string"}}},
                "required": ["commands"]
            }),
        ),
    ];
    let started = std::time::Instant::now();
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
    let tree = Arc::new(compile(&spec, &vocab).expect("compiles against real Qwen3 BPE"));
    eprintln!(
        "cline catalog: {} nodes, compiled in {:?}",
        tree.len(),
        started.elapsed()
    );

    let mut session = StencilSession::new(tree);
    let mut out = Vec::new();
    let head = "<tool_call>\n{\"name\": \"read_files\", \"arguments\": {\"files\": \
                [{\"path\": \"a.rs\"}, {\"path\": \"src/main.rs\", \"start_line\": 3380, \
                \"end_line\":";
    let head_drops = follow_real(&mut session, &vocab, head, &mut out);

    // The model's own tail: the number, then a bracket closing the wrong thing.
    let mut tail_dropped = false;
    for t in vocab.encode(" 3420]}}}") {
        assert!(
            matches!(session.next_action(), StencilAction::FreeDecode { .. }),
            "the value is still free when {:?} arrives",
            String::from_utf8_lossy(&vocab.token_bytes(t))
        );
        match session.observe(t, &vocab.token_bytes(t)).unwrap() {
            Observe::DelimiterDropped => {
                tail_dropped = true;
                break;
            }
            Observe::Bailed => panic!("bailed on {:?}", vocab.token_bytes(t)),
            _ => out.push(t),
        }
    }
    assert!(tail_dropped, "the misplaced bracket was not dropped");

    // What the grammar and a model choosing `]` write from there.
    let tail_drops = follow_real(&mut session, &vocab, "}]}}\n</tool_call>", &mut out);
    assert_eq!(session.next_action(), StencilAction::Exit);
    eprintln!("drops on the canonical path: head {head_drops}, tail {tail_drops}");

    let text = String::from_utf8(vocab.decode(&out)).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text))
        .unwrap_or_else(|e| panic!("not JSON: {text:?}: {e}"));
    assert_eq!(parsed["arguments"]["files"][1]["end_line"], 3420);
    assert_eq!(parsed["arguments"]["files"][0]["path"], "a.rs");
}

/// **A value written as a Python literal means the same JSON, in real tokens.**
/// Qwen3's tokenizer cuts ` True`, `':`, `None`, `'src` where it cuts them, and
/// the repairs are made one token at a time against what is already committed
/// — so the meaning has to survive the cuts a real vocabulary makes, not only
/// the byte vocabulary's.
///
/// Each model token is fed free and committed the way the decode loop commits
/// it: rewritten when the session hands back a rewrite, dropped when it drops.
#[test]
#[ignore = "requires the cached Qwen3 tokenizer.json"]
fn a_python_literal_value_means_the_same_json_with_real_tokens() {
    use candle_conversation::stencil::Observe;
    let Some(path) = cached_qwen3_tokenizer() else {
        panic!("Qwen3 tokenizer.json not cached — run the daemon once to fetch it");
    };
    let tok = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer.json");
    let vocab = HfVocab::new(tok, &[0], 0);
    let tools = [ToolSpec::from_json_schema(
        "pack",
        &serde_json::json!({
            "type": "object",
            "properties": {"values": {"type": "array", "items": true}},
            "required": ["values"]
        }),
    )];
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
    let tree = Arc::new(compile(&spec, &vocab).expect("compiles against real Qwen3 BPE"));

    let cases: &[(&str, serde_json::Value)] = &[
        (
            " [{'path': 'src/main.rs', 'recursive': True, 'start': None}, {path: 'b.rs', depth: .5}]",
            serde_json::json!([
                {"path": "src/main.rs", "recursive": true, "start": null},
                {"path": "b.rs", "depth": 0.5}
            ]),
        ),
        (
            " ['it\\'s', \"say \\'hi\\'\", 'tab\there', False, NULL, undefined, [1 2 3]]",
            serde_json::json!(["it's", "say 'hi'", "tab\there", false, null, null, [1, 2, 3]]),
        ),
    ];
    for (written, meant) in cases {
        let mut session = StencilSession::new(Arc::clone(&tree));
        let mut out = Vec::new();
        let head = "<tool_call>\n{\"name\": \"pack\", \"arguments\": {\"values\":";
        assert_eq!(follow_real(&mut session, &vocab, head, &mut out), 0);

        for t in vocab.encode(written) {
            match session.next_action() {
                StencilAction::FreeDecode { .. } => {}
                other => panic!("{written:?}: the value ended early, at {other:?}"),
            }
            let bytes = vocab.token_bytes(t);
            let observed = session.observe(t, &bytes).unwrap();
            match (session.take_rewrite(), observed) {
                (Some(rewrite), _) => {
                    out.extend(vocab.encode(&String::from_utf8(rewrite).unwrap()))
                }
                (None, Observe::TokenClosedDrop | Observe::DelimiterDropped) => {}
                (None, Observe::SpanClosed { leftover }) if leftover > 0 => out.extend(
                    vocab.encode(std::str::from_utf8(&bytes[..bytes.len() - leftover]).unwrap()),
                ),
                (None, _) => out.push(t),
            }
        }
        follow_real(&mut session, &vocab, "}}\n</tool_call>", &mut out);
        assert_eq!(session.next_action(), StencilAction::Exit);

        let text = String::from_utf8(vocab.decode(&out)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(json_body(&text))
            .unwrap_or_else(|e| panic!("not JSON: {text:?}: {e}"));
        assert_eq!(&parsed["arguments"]["values"], meant, "{text}");
    }
}

/// The stencil trigger is `token_to_id("<tool_call>")`. If that doesn't resolve
/// to the single special token (151657), the registry is empty and steering
/// NEVER engages — the model free-decodes every tool call. `<tool_call>` is an
/// `added_token` with `special=false`, exactly the case that can break
/// added-token lookup, so pin it down.
#[test]
#[ignore = "requires the cached Qwen3 tokenizer.json"]
fn tool_call_trigger_token_resolves() {
    let Some(path) = cached_qwen3_tokenizer() else {
        panic!("Qwen3 tokenizer.json not cached");
    };
    let tok = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer.json");
    let id = tok.token_to_id("<tool_call>");
    eprintln!("token_to_id(\"<tool_call>\") = {id:?}");
    eprintln!("decode([151657]) = {:?}", tok.decode(&[151657], false));
    assert_eq!(
        id,
        Some(151657),
        "the `<tool_call>` trigger must resolve, or the stencil registry is empty"
    );
}

/// A tool call is the whole assistant turn, so the steering tree's close ends
/// with the assistant EOS (`<|im_end|>`) — the decode loop detects that EOS in
/// the injected close run and seals the turn instead of free-decoding past the
/// call. The whole fix hinges on `<|im_end|>` tokenizing to the EOS as the final
/// token of the close run; pin that down.
#[test]
#[ignore = "requires the cached Qwen3 tokenizer.json"]
fn close_run_ends_with_eos() {
    let Some(path) = cached_qwen3_tokenizer() else {
        panic!("Qwen3 tokenizer.json not cached");
    };
    let tok = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer.json");
    let im_end = tok
        .token_to_id("<|im_end|>")
        .expect("<|im_end|> must resolve");
    let vocab = HfVocab::new(tok, &[im_end], 0);
    let close = vocab.encode("}}\n</tool_call><|im_end|>");
    eprintln!("close run = {close:?} (eos = {im_end})");
    assert_eq!(
        close.last().copied(),
        Some(im_end),
        "the close run must end with the EOS token so the turn seals after the call",
    );
}

// ── The decode-loop driver ──────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
enum DriveErr {
    /// The grammar masked out the target byte at `pos` (a rejected attempt).
    MaskRejected {
        pos: usize,
        byte: u8,
        allowed: Vec<u32>,
    },
    /// A prefilled static run did not match the target text — a tree/target
    /// formatting mismatch (a bug, not a model attempt).
    PrefillMismatch { pos: usize, got: String },
    /// The target ran out before the call completed.
    Truncated { pos: usize },
    /// The session itself errored (e.g. out-of-mask token observed).
    Walk(WalkError),
}

/// Drive the session to follow `target` (the full materialized call).  Returns
/// the emitted text on success, or the first place the grammar diverged.
fn drive(tree: Arc<StencilTree>, target: &str, vocab: &TestVocab) -> Result<String, DriveErr> {
    let bytes = target.as_bytes();
    let mut session = StencilSession::new(tree);
    let mut pos = 0usize;
    let mut out: Vec<u32> = Vec::new();
    let mut steps = 0usize;
    loop {
        steps += 1;
        assert!(steps < 1_000_000, "runaway driver");
        match session.next_action() {
            StencilAction::Prefill(toks) => {
                let pb = vocab.decode(&toks);
                if bytes.len() < pos + pb.len() || bytes[pos..pos + pb.len()] != pb[..] {
                    return Err(DriveErr::PrefillMismatch {
                        pos,
                        got: String::from_utf8_lossy(&pb).into_owned(),
                    });
                }
                pos += pb.len();
                out.extend(toks);
            }
            StencilAction::MaskedDecode(set) => {
                if pos >= bytes.len() {
                    return Err(DriveErr::Truncated { pos });
                }
                let tok = bytes[pos] as u32;
                if !set.contains(tok) {
                    return Err(DriveErr::MaskRejected {
                        pos,
                        byte: bytes[pos],
                        allowed: set.tokens().to_vec(),
                    });
                }
                out.push(tok);
                session
                    .observe(tok, &[bytes[pos]])
                    .map_err(DriveErr::Walk)?;
                pos += 1;
            }
            StencilAction::FreeDecode { .. } => {
                if pos >= bytes.len() {
                    return Err(DriveErr::Truncated { pos });
                }
                let tok = bytes[pos] as u32;
                out.push(tok);
                session
                    .observe(tok, &[bytes[pos]])
                    .map_err(DriveErr::Walk)?;
                pos += 1;
            }
            StencilAction::Exit => break,
        }
    }
    Ok(String::from_utf8_lossy(&vocab.decode(&out)).into_owned())
}

/// The mask offered at the very first decode (the name-branch frontier), after
/// the envelope prefill.
fn first_mask(tree: Arc<StencilTree>) -> AllowedSet {
    let mut session = StencilSession::new(tree);
    loop {
        match session.next_action() {
            StencilAction::Prefill(_) => {}
            StencilAction::MaskedDecode(set) => return set,
            other => panic!("expected a masked decode first, got {other:?}"),
        }
    }
}

// ── Generating a minimal valid call for every tool ──────────────────────────

fn minimal_value(p: &Param) -> String {
    if let Some(values) = &p.enum_values {
        return format!("\"{}\"", values[0]);
    }
    match p.ty {
        ParamType::String => "\"\"".into(),
        ParamType::Integer | ParamType::Number => "0".into(),
        ParamType::Boolean => "false".into(),
        ParamType::Array => "[]".into(),
        // An object with a schema is guided, so its required fields are forced
        // exactly as a call's are.
        ParamType::Object => match &p.properties {
            Some(fields) => format!("{{{}}}", required_fields(fields)),
            None => "{}".into(),
        },
    }
}

/// `"key": value` for every required field, in the tree's order.
fn required_fields(params: &[Param]) -> String {
    params
        .iter()
        .filter(|p| p.required)
        .map(|p| format!("\"{}\": {}", p.name, minimal_value(p)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A minimal valid call: name + every required field (in the tree's order) with
/// a minimal value, no optionals.  Matches the tree's exact formatting.
fn minimal_call(spec: &ToolSpec) -> String {
    let mut s = String::from("<tool_call>\n{\"name\": \"");
    s.push_str(&spec.name);
    s.push_str("\", \"arguments\": {");
    s.push_str(&required_fields(&spec.params));
    s.push_str("}}\n</tool_call>");
    s
}

fn json_body(text: &str) -> &str {
    text.trim_start_matches("<tool_call>\n")
        .trim_end_matches("\n</tool_call>")
}

// ── Tests: the tree exists and holds every tool ─────────────────────────────

/// Every tool contributes at least its own argument scaffold beside the
/// envelope and the name branch. Bounded by the catalog rather than a fixed
/// size: the compiler shares nodes reached along several paths, so the count
/// tracks the tools, not the number of optional-field combinations (it fell
/// from over 1,000 to 861 when sharing landed, with the same grammar).
#[test]
fn whole_registry_compiles_to_one_tree() {
    let (tree, _) = build_tree();
    let tools = catalog().len();
    assert!(
        tree.len() > tools + 2,
        "expected a node per tool ({tools}) beyond the envelope and name branch, got {}",
        tree.len()
    );
    // Root is the open envelope (a Static).
    assert!(matches!(tree.node(tree.root()), StencilNode::Static { .. }));
}

// ── Positive: every tool's minimal call drives to valid JSON ────────────────

#[test]
fn every_tool_minimal_call_is_valid() {
    let (tree, vocab) = build_tree();
    let tools = catalog();
    let mut checked = 0;
    for spec in &tools {
        let target = minimal_call(spec);
        let out = drive(Arc::clone(&tree), &target, &vocab).unwrap_or_else(|e| {
            panic!(
                "tool {:?}: drive failed: {e:?}\n  target={target:?}",
                spec.name
            )
        });
        assert_eq!(out, target, "tool {:?}: emitted text drifted", spec.name);
        // The output is valid JSON with the right name and EVERY required key.
        let parsed: serde_json::Value = serde_json::from_str(json_body(&out))
            .unwrap_or_else(|e| panic!("tool {:?}: not JSON: {out:?}: {e}", spec.name));
        assert_eq!(parsed["name"], spec.name.as_str());
        assert!(parsed["arguments"].is_object());
        for p in spec.params.iter().filter(|p| p.required) {
            assert!(
                parsed["arguments"].get(&p.name).is_some(),
                "tool {:?}: required field {:?} missing — the stencil failed to force it",
                spec.name,
                p.name
            );
        }
        checked += 1;
    }
    assert_eq!(checked, tools.len());
    assert!(checked >= 90, "expected ~93 tools, checked {checked}");
}

// ── Decode-loop mechanics: the name-branch mask ─────────────────────────────

#[test]
fn name_branch_mask_allows_real_names_only() {
    let (tree, _vocab) = build_tree();
    let mask = first_mask(Arc::clone(&tree));
    let tools = catalog();
    // Every real tool's first byte is allowed.
    for spec in &tools {
        let first = spec.name.as_bytes()[0] as u32;
        assert!(
            mask.contains(first),
            "first byte of {:?} not in the name mask",
            spec.name
        );
    }
    // A byte no tool name starts with is masked out. Tool names are
    // [a-z_]; an uppercase / digit / space is impossible.
    for bad in [b'Z' as u32, b'9' as u32, b' ' as u32, b'{' as u32] {
        assert!(
            !mask.contains(bad),
            "byte {bad} should be masked at the name"
        );
    }
}

// ── Negative: a non-existent tool name is rejected ──────────────────────────

#[test]
fn unknown_tool_name_is_masked() {
    let (tree, vocab) = build_tree();
    // A plausible-looking but non-existent name; the trie diverges from every
    // real name at some byte and rejects it.
    let target =
        "<tool_call>\n{\"name\": \"totally_made_up_tool\", \"arguments\": {}}\n</tool_call>";
    let err = drive(tree, target, &vocab).unwrap_err();
    assert!(
        matches!(err, DriveErr::MaskRejected { .. }),
        "expected a mask rejection for an unknown tool, got {err:?}"
    );
}

#[test]
fn name_with_bad_first_byte_is_masked_immediately() {
    let (tree, vocab) = build_tree();
    let target = "<tool_call>\n{\"name\": \"Zzz\", \"arguments\": {}}\n</tool_call>";
    match drive(tree, target, &vocab).unwrap_err() {
        // The very first name byte ('Z') is rejected.
        DriveErr::MaskRejected { byte, .. } => assert_eq!(byte, b'Z'),
        other => panic!("expected MaskRejected at 'Z', got {other:?}"),
    }
}

// ── Negative: a hallucinated parameter is rejected ──────────────────────────

#[test]
fn hallucinated_parameter_is_masked() {
    let (tree, vocab) = build_tree();
    // Pick a real tool that has at least one optional param so the args object
    // opens a gate branch; then try to emit a bogus key.
    let tools = catalog();
    let with_optional = tools
        .iter()
        .find(|t| t.params.iter().any(|p| !p.required) && t.params.iter().all(|p| p.required))
        .or_else(|| tools.iter().find(|t| t.params.iter().any(|p| !p.required)));
    let spec = with_optional.expect("some tool has an optional param");
    // Build: name + args_open + a fake key. The gate only allows the tool's real
    // optional keys (or the close), so the fake key diverges.
    let mut target = String::from("<tool_call>\n{\"name\": \"");
    target.push_str(&spec.name);
    target.push_str("\", \"arguments\": {");
    // Required fields must come first (forced); include them minimally so we
    // reach the optional gate, then inject the bogus key.
    let req: Vec<String> = spec
        .params
        .iter()
        .filter(|p| p.required)
        .map(|p| format!("\"{}\": {}", p.name, minimal_value(p)))
        .collect();
    target.push_str(&req.join(", "));
    if !req.is_empty() {
        target.push_str(", ");
    }
    target.push_str("\"__bogus__\": 1}}\n</tool_call>");
    let err = drive(tree, &target, &vocab).unwrap_err();
    assert!(
        matches!(err, DriveErr::MaskRejected { .. }),
        "tool {:?}: a hallucinated key should be masked, got {err:?}",
        spec.name
    );
}

// ── file_read's range is optional and ordered ───────────────────────────────

/// `file_read` may stop after any part of its range — the path alone reads the
/// whole file, and either bound alone reads to the file's edge — and names what
/// it gives in `path, start_line, end_line` order, the order its schema
/// declares them in.
#[test]
fn file_read_drives_any_part_of_its_range_in_declared_order() {
    let (tree, vocab) = build_tree();
    for (args, start, end) in [
        (r#""path": "a.rs""#, None, None),
        (r#""path": "a.rs", "start_line": 94"#, Some(94), None),
        (r#""path": "a.rs", "end_line": 30"#, None, Some(30)),
        (
            r#""path": "a.rs", "start_line": 598, "end_line": 630"#,
            Some(598),
            Some(630),
        ),
    ] {
        let target = format!(
            "<tool_call>\n{{\"name\": \"file_read\", \"arguments\": {{{args}}}}}\n</tool_call>"
        );
        let out = drive(Arc::clone(&tree), &target, &vocab)
            .unwrap_or_else(|e| panic!("{args} must drive, got {e:?}"));
        let parsed: serde_json::Value = serde_json::from_str(json_body(&out)).unwrap();
        let arguments = &parsed["arguments"];
        assert_eq!(arguments["path"], "a.rs");
        assert_eq!(arguments.get("start_line").and_then(|v| v.as_u64()), start);
        assert_eq!(arguments.get("end_line").and_then(|v| v.as_u64()), end);
    }
}

/// The bounds come start first. Optionals are offered in declared order, each
/// at most once, so a tree that offered `end_line` first would strand a call
/// written in reading order: once `start_line` is taken, the `end_line` gate
/// is behind it. The reverse order is what cannot be expressed.
#[test]
fn file_read_rejects_its_range_end_first() {
    let (tree, vocab) = build_tree();
    let target = "<tool_call>\n{\"name\": \"file_read\", \"arguments\": {\"path\": \"a.rs\", \
                  \"end_line\": 93, \"start_line\": 47}}\n</tool_call>";
    let err = drive(tree, target, &vocab).unwrap_err();
    assert!(
        matches!(
            err,
            DriveErr::PrefillMismatch { .. } | DriveErr::MaskRejected { .. }
        ),
        "got {err:?}"
    );
}

// ── Negative: a wrong boolean / wrong enum value is rejected ────────────────

#[test]
fn boolean_value_is_constrained() {
    // A synthetic tool with a required boolean, to drive an illegal value.
    let tools = vec![ToolSpec::from_json_schema(
        "toggle",
        &serde_json::json!({
            "type": "object",
            "properties": { "on": { "type": "boolean" } },
            "required": ["on"]
        }),
    )];
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
    let vocab = TestVocab::new();
    let tree = Arc::new(compile(&spec, &vocab).unwrap());

    // Valid: true / false drive cleanly.
    for good in ["true", "false"] {
        let target = format!(
            "<tool_call>\n{{\"name\": \"toggle\", \"arguments\": {{\"on\": {good}}}}}\n</tool_call>"
        );
        assert!(
            drive(Arc::clone(&tree), &target, &vocab).is_ok(),
            "{good} should drive"
        );
    }
    // Invalid: "maybe" diverges from both true and false at the first byte.
    let target =
        "<tool_call>\n{\"name\": \"toggle\", \"arguments\": {\"on\": maybe}}\n</tool_call>";
    let err = drive(tree, target, &vocab).unwrap_err();
    assert!(
        matches!(err, DriveErr::MaskRejected { byte: b'm', .. }),
        "got {err:?}"
    );
}

#[test]
fn enum_value_is_constrained() {
    let tools = vec![ToolSpec::from_json_schema(
        "set_level",
        &serde_json::json!({
            "type": "object",
            "properties": { "level": { "type": "string", "enum": ["low", "high"] } },
            "required": ["level"]
        }),
    )];
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
    let vocab = TestVocab::new();
    let tree = Arc::new(compile(&spec, &vocab).unwrap());

    let ok = "<tool_call>\n{\"name\": \"set_level\", \"arguments\": {\"level\": \"high\"}}\n</tool_call>";
    assert!(drive(Arc::clone(&tree), ok, &vocab).is_ok());

    // "medium" is not an allowed enum value.
    let bad = "<tool_call>\n{\"name\": \"set_level\", \"arguments\": {\"level\": \"medium\"}}\n</tool_call>";
    let err = drive(tree, bad, &vocab).unwrap_err();
    assert!(matches!(err, DriveErr::MaskRejected { .. }), "got {err:?}");
}

// ── Negative: a required field cannot be skipped ────────────────────────────

#[test]
fn required_field_cannot_be_closed_early() {
    // A tool with a required string param. Attempting to close the args object
    // immediately ( {} ) cannot even be expressed: the required key is a
    // prefilled static, so the close `}` lands where the static `"key": ` is.
    let tools = vec![ToolSpec::from_json_schema(
        "must",
        &serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
    )];
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
    let vocab = TestVocab::new();
    let tree = Arc::new(compile(&spec, &vocab).unwrap());

    // Empty arguments — the prefilled `"path": "` will not match `}`.
    let target = "<tool_call>\n{\"name\": \"must\", \"arguments\": {}}\n</tool_call>";
    let err = drive(Arc::clone(&tree), target, &vocab).unwrap_err();
    assert!(
        matches!(
            err,
            DriveErr::PrefillMismatch { .. } | DriveErr::MaskRejected { .. }
        ),
        "skipping a required field must be impossible, got {err:?}"
    );

    // The valid minimal call DOES include the required field.
    let ok = "<tool_call>\n{\"name\": \"must\", \"arguments\": {\"path\": \"\"}}\n</tool_call>";
    let out = drive(tree, ok, &vocab).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(json_body(&out)).unwrap();
    assert!(parsed["arguments"]["path"].is_string());
}

// ── Decode-loop mechanics: a full step-by-step action trace ─────────────────

#[test]
fn action_trace_alternates_prefill_and_decode() {
    let (tree, vocab) = build_tree();
    // Find a tool with exactly one required string param for a clean trace.
    let tools = catalog();
    let spec = tools
        .iter()
        .find(|t| {
            t.params.iter().filter(|p| p.required).count() == 1
                && t.params
                    .iter()
                    .all(|p| !p.required || matches!(p.ty, ParamType::String))
                && t.params.iter().all(|p| p.required)
        })
        .expect("some tool has exactly one required string param and no optionals");
    let target = minimal_call(spec);

    // Drive manually, recording the action kinds.
    let mut session = StencilSession::new(Arc::clone(&tree));
    let bytes = target.as_bytes();
    let mut pos = 0;
    let mut kinds = Vec::new();
    loop {
        match session.next_action() {
            StencilAction::Prefill(toks) => {
                kinds.push('P');
                pos += vocab.decode(&toks).len();
            }
            StencilAction::MaskedDecode(_) => {
                kinds.push('M');
                let b = bytes[pos];
                session.observe(b as u32, &[b]).unwrap();
                pos += 1;
            }
            StencilAction::FreeDecode { .. } => {
                kinds.push('F');
                let b = bytes[pos];
                session.observe(b as u32, &[b]).unwrap();
                pos += 1;
            }
            StencilAction::Exit => break,
        }
    }
    // Must begin with a prefill (the envelope) and contain masked decodes (the
    // name) and a free decode (the string value).
    assert_eq!(kinds.first(), Some(&'P'));
    assert!(kinds.contains(&'M'), "expected masked name decodes");
    assert!(kinds.contains(&'F'), "expected a free-text value decode");
}

// ── Failsafe: a token that escapes the mask bails and terminates the block ───

#[test]
fn escaped_token_bails_and_terminates() {
    use candle_conversation::stencil::Observe;
    let (tree, vocab) = build_tree();
    let mut session = StencilSession::new(Arc::clone(&tree));
    let mut out: Vec<u32> = Vec::new();

    // Drive the envelope prefill to the first masked decode (the name branch).
    let mask = loop {
        match session.next_action() {
            StencilAction::Prefill(toks) => out.extend(toks),
            StencilAction::MaskedDecode(set) => break set,
            other => panic!("expected a masked decode, got {other:?}"),
        }
    };

    // Feed a token the mask forbids — simulating a sampler that ignored the
    // mask.  The session must NOT error; it bails.
    let bad = b'Z' as u32;
    assert!(!mask.contains(bad), "'Z' should be masked at the name");
    out.push(bad);
    assert_eq!(session.observe(bad, b"Z").unwrap(), Observe::Bailed);

    // The remaining actions emit the bail tokens (the envelope close) then exit.
    loop {
        match session.next_action() {
            StencilAction::Prefill(toks) => out.extend(toks),
            StencilAction::Exit => break,
            other => panic!("after bail expected prefill/exit, got {other:?}"),
        }
    }

    let text = String::from_utf8_lossy(&vocab.decode(&out)).into_owned();
    assert!(
        text.contains('Z'),
        "the escaped token is still in the stream"
    );
    assert!(
        text.ends_with("</tool_call>"),
        "bail must terminate the tool-call block: {text:?}"
    );
}
