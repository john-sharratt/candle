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
    compile, compile_tool_call_loop, compile_tool_call_tree, AllowedSet, FreeTextLimits, HfVocab,
    NodeSpec, Observe, Param, ParamType, StencilAction, StencilNode, StencilSession, StencilTree,
    Terminator, TestVocab, ToolCallEnvelope, ToolSpec, TreeSpec, Vocab, WalkError,
    MAX_TOOL_CALLS_PER_TURN,
};

// ── Building the tree from the live registry ────────────────────────────────

/// The catalog the daemon compiles its stencil from — canonical names and
/// every alias.
fn catalog() -> Vec<ToolSpec> {
    zend::tools::tool_catalog().to_vec()
}

/// What a turn ends with once the loop decides to stop calling. The tests drive
/// whole turns, so every target carries it.
const TURN_CLOSE: &str = "<|im_end|>";

/// The marker that opens a call — and, the first time, fires the stencil.
const MARKER: &str = "<tool_call>";

/// The envelope `Engine::compile_tool_stencil` compiles with: the Qwen3 JSON
/// block with the marker taken off `open`, as `for_assistant_calls` does.
///
/// Not the raw [`ToolCallEnvelope::qwen3`]. The loop's contract is that `open`
/// excludes the marker — the model emits the first one itself to fire the
/// stencil, and the loop's continuation arm emits every later one before
/// `open`. Handed a marker-bearing `open`, every call after the first comes out
/// `<tool_call><tool_call>…`. Building the tests from the raw envelope is what
/// made the first two multi-call tests fail against a grammar that is correct.
fn production_envelope() -> ToolCallEnvelope {
    let base = ToolCallEnvelope::qwen3();
    ToolCallEnvelope {
        open: base
            .open
            .strip_prefix(&base.marker)
            .unwrap_or(&base.open)
            .to_string(),
        ..base
    }
}

/// The spec for a turn of up to `max_calls` calls over the live catalog.
fn loop_spec(max_calls: usize) -> TreeSpec {
    compile_tool_call_loop(&catalog(), &production_envelope(), max_calls, TURN_CLOSE)
        .expect("the whole registry compiles to one tree")
}

/// **The tree the daemon actually compiles.** `Engine::compile_tool_stencil`
/// builds a [`compile_tool_call_loop`] over the live catalog, so a test driving
/// the single-call tree would be exercising a grammar that no longer runs —
/// which is how a suite goes green against a shape production does not have.
fn build_tree() -> (Arc<StencilTree>, TestVocab) {
    let vocab = TestVocab::new();
    let tree = compile(&loop_spec(MAX_TOOL_CALLS_PER_TURN), &vocab).expect("the spec tokenizes");
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
    let spec = loop_spec(MAX_TOOL_CALLS_PER_TURN);
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
    let spec = loop_spec(MAX_TOOL_CALLS_PER_TURN);
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
    /// The tree finished with target text left over — the grammar ended the
    /// turn somewhere the target says it continued.
    Trailing { pos: usize },
    /// The session itself errored (e.g. out-of-mask token observed).
    Walk(WalkError),
}

/// Drive the session to follow `target` — the whole assistant turn as the model
/// writes it. Returns the emitted text on success, or the first place the
/// grammar diverged.
///
/// The target opens with [`MARKER`] because the turn does; the stencil does not
/// emit that one. In production the model decodes the first marker itself and
/// that token is what fires the stencil, so the tree begins just after it. The
/// driver takes the marker off the front for the walk and puts it back on the
/// result, so every target reads as the real turn text.
fn drive(tree: Arc<StencilTree>, target: &str, vocab: &TestVocab) -> Result<String, DriveErr> {
    let bytes = target.strip_prefix(MARKER).unwrap_or(target).as_bytes();
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
            StencilAction::Exit => {
                // Without this a target with anything after the turn's end —
                // prose, a fifth call — "drives" as long as its prefix does,
                // and every negative test built on it passes vacuously.
                if pos != bytes.len() {
                    return Err(DriveErr::Trailing { pos });
                }
                break;
            }
        }
    }
    Ok(format!(
        "{MARKER}{}",
        String::from_utf8_lossy(&vocab.decode(&out))
    ))
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
    s.push_str(TURN_CLOSE);
    s
}

/// The JSON object of a single-call turn — the envelope, the block close and the
/// turn terminator stripped.
fn json_body(text: &str) -> &str {
    text.trim_start_matches("<tool_call>\n")
        .trim_end_matches(TURN_CLOSE)
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

/// **An alias decodes as written.** `file_write` is `write` under another
/// name; the grammar compiled only canonical names used to heal it into
/// `file_list`, so a turn meant to create a file listed a directory instead.
#[test]
fn an_alias_name_drives_to_its_tools_arguments() {
    let (tree, vocab) = build_tree();
    let target = format!(
        "<tool_call>\n{{\"name\": \"file_write\", \"arguments\": {{\"path\": \"a.txt\", \
         \"content\": \"hi\"}}}}\n</tool_call>{TURN_CLOSE}"
    );
    let out = drive(tree, &target, &vocab).expect("an alias must not be masked");
    assert!(out.contains("\"file_write\""), "{out}");
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

// ── file_read's page is mandatory ───────────────────────────────────────────

/// `file_read` always names both `path` and `page` — the grammar forces both,
/// the same way it forces any other required field.
#[test]
fn file_read_always_drives_path_and_page() {
    let (tree, vocab) = build_tree();
    for page in [0, 3] {
        let target = format!(
            "<tool_call>\n{{\"name\": \"file_read\", \"arguments\": {{\"path\": \"a.rs\", \
             \"page\": {page}}}}}\n</tool_call>"
        );
        let out = drive(Arc::clone(&tree), &target, &vocab)
            .unwrap_or_else(|e| panic!("page {page} must drive, got {e:?}"));
        let parsed: serde_json::Value = serde_json::from_str(json_body(&out)).unwrap();
        let arguments = &parsed["arguments"];
        assert_eq!(arguments["path"], "a.rs");
        assert_eq!(arguments["page"], page);
    }
}

/// Omitting `page` cannot be expressed — it is forced exactly like `path`.
#[test]
fn file_read_rejects_a_call_missing_page() {
    let (tree, vocab) = build_tree();
    let target =
        "<tool_call>\n{\"name\": \"file_read\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>";
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

/// A production-shaped loop tree over a hand-written catalog, for tests that
/// need one precise parameter shape rather than the live registry.
fn synthetic_tree(tools: &[ToolSpec]) -> (Arc<StencilTree>, TestVocab) {
    let spec = compile_tool_call_loop(
        tools,
        &production_envelope(),
        MAX_TOOL_CALLS_PER_TURN,
        TURN_CLOSE,
    )
    .unwrap();
    let vocab = TestVocab::new();
    (Arc::new(compile(&spec, &vocab).unwrap()), vocab)
}

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
    let (tree, vocab) = synthetic_tree(&tools);

    // Valid: true / false drive cleanly.
    for good in ["true", "false"] {
        let target = format!(
            "<tool_call>\n{{\"name\": \"toggle\", \"arguments\": {{\"on\": {good}}}}}\n\
             </tool_call>{TURN_CLOSE}"
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
    let (tree, vocab) = synthetic_tree(&tools);

    let ok = "<tool_call>\n{\"name\": \"set_level\", \"arguments\": {\"level\": \"high\"}}\n\
              </tool_call><|im_end|>";
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
    let (tree, vocab) = synthetic_tree(&tools);

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
    let ok = "<tool_call>\n{\"name\": \"must\", \"arguments\": {\"path\": \"\"}}\n\
              </tool_call><|im_end|>";
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

    // Drive manually, recording the action kinds. The tree starts after the
    // marker the model emits itself — see `drive`.
    let mut session = StencilSession::new(Arc::clone(&tree));
    let bytes = target.strip_prefix(MARKER).unwrap_or(&target).as_bytes();
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
    // The bail closes the block AND the turn: a bail mid-call must not leave the
    // decoder free after `</tool_call>`, which in a loop tree is a point where
    // it would otherwise be choosing whether to call again.
    assert!(
        text.ends_with(&format!("</tool_call>{TURN_CLOSE}")),
        "bail must terminate the tool-call block and the turn: {text:?}"
    );
}

// ── Empty string arguments ──────────────────────────────────────────────────

/// **An empty string argument must drive.** `file_list`'s own exemplars teach
/// `{"path": ""}` — it is how the catalog says "list from the project root" —
/// so the grammar has to be able to express it.
///
/// A live turn produced this instead:
///
/// ```text
/// {"name": "file_list", "arguments": {"path": "}}
/// </tool_call>"}}
/// </tool_call>
/// ```
///
/// The model wrote `}}` where the value's closing quote belonged, and the value
/// span — a free-text run that ends at an unescaped `"` — swallowed the call's
/// own terminator as string content. This test establishes whether the grammar
/// can represent the empty value at all, which separates "the stencil cannot
/// express it" from "the model dropped a quote".
#[test]
fn an_empty_string_argument_drives() {
    let (tree, vocab) = build_tree();
    let target = "<tool_call>\n{\"name\": \"file_list\", \"arguments\": {\"path\": \"\"}}\n\
                  </tool_call><|im_end|>";
    let out = drive(Arc::clone(&tree), target, &vocab)
        .unwrap_or_else(|e| panic!("an empty path must drive, got {e:?}"));
    let parsed: serde_json::Value = serde_json::from_str(json_body(&out)).unwrap();
    assert_eq!(parsed["arguments"]["path"], "");
}

/// **Each call level costs one catalog's worth of grammar — linear, never
/// multiplicative.**
///
/// `compile_action_loop` builds every tool's argument sub-tree once per level,
/// so the *spec* grows linearly with `max_calls` by construction. The compiled
/// tree did not: lowering expanded the spec's shared joins into a tree, copying
/// level `k+1` once per path through level `k`, and with 95 tools the compile
/// for four levels never finished — this suite hung. Measured once the compiler
/// memoised shared successors: 902, 1,805, 2,707, 3,609 nodes for one to four
/// calls. The bound below allows a level to cost at most a little over one
/// single-call catalog, which the multiplicative regression overshoots at two
/// levels already.
#[test]
fn each_call_level_adds_one_catalog_of_grammar() {
    let vocab = TestVocab::new();
    let size = |max_calls| {
        compile(&loop_spec(max_calls), &vocab)
            .expect("the loop compiles")
            .len()
    };
    let one = size(1);
    for max_calls in 2..=MAX_TOOL_CALLS_PER_TURN {
        let n = size(max_calls);
        assert!(
            n <= one * max_calls + one / 10,
            "{max_calls} calls compiled to {n} nodes against {one} for one — levels are \
             being duplicated rather than shared"
        );
    }
}

// ── Several calls in one turn ───────────────────────────────────────────────

/// One `file_read` call on `path`, as the tree formats it.
fn read_call(path: &str) -> String {
    format!(
        "<tool_call>\n{{\"name\": \"file_read\", \"arguments\": {{\"path\": \"{path}\", \
         \"page\": 0}}}}\n</tool_call>"
    )
}

/// Several calls as the chat template lays them out: one block per call, a
/// newline between consecutive blocks.
fn read_calls<'a>(paths: impl IntoIterator<Item = &'a str>) -> String {
    paths
        .into_iter()
        .map(read_call)
        .collect::<Vec<_>>()
        .join("\n")
}

/// **Consecutive calls are separated the way the template separates them —
/// by a newline — and only that way.**
///
/// The checkpoint's chat template writes `</tool_call>\n<tool_call>` between
/// the calls of one message, and zend's own `openai_tools::canonical` renders a
/// multi-call reply identically. The loop first offered the marker glued
/// straight onto the previous close, so the one token a model trained on that
/// template writes to continue — the newline — was masked, and every turn ended
/// after one call: nine tool rounds, one call each, on a live codebase tour.
#[test]
fn consecutive_calls_are_separated_by_the_templates_newline() {
    let (tree, vocab) = build_tree();
    let a = read_call("a.rs");
    let b = read_call("b.rs");
    drive(Arc::clone(&tree), &format!("{a}\n{b}{TURN_CLOSE}"), &vocab)
        .expect("the template's `</tool_call>\\n<tool_call>` must drive");
    assert!(
        drive(Arc::clone(&tree), &format!("{a}{b}{TURN_CLOSE}"), &vocab).is_err(),
        "the marker glued to the previous close is not the template's layout"
    );
}

/// **Three reads in one turn drive, and every one of them is dispatched.**
///
/// This is the point of the loop: a model that already knows it wants three
/// files asks for them in one turn instead of paying a reasoning block, a
/// prefill and a belief scan for each. The grammar has to accept the run, and
/// the extractor downstream has to find every call in it — a turn whose second
/// and third calls were grammatical but never dispatched would look, from the
/// model's side, like tools that silently returned nothing.
#[test]
fn several_calls_in_one_turn_drive_and_are_all_extracted() {
    let (tree, vocab) = build_tree();
    let paths = ["src/main.rs", "src/lib.rs", "Cargo.toml"];
    let target = format!("{}{TURN_CLOSE}", read_calls(paths));
    let out = drive(Arc::clone(&tree), &target, &vocab)
        .unwrap_or_else(|e| panic!("three calls must drive, got {e:?}"));
    assert_eq!(out, target);

    let calls = zend::tools::extract_tool_calls(&out);
    let got: Vec<&str> = calls
        .iter()
        .map(|c| c.arguments["path"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(got, paths, "every call in the turn is extracted, in order");
}

/// **The ceiling is enforced by the grammar, not by a count downstream.** After
/// the last permitted call the only arm left is the turn terminator, so the
/// marker that would open one more call is off-grammar.
#[test]
fn a_turn_cannot_make_more_calls_than_the_limit() {
    let (tree, vocab) = build_tree();
    let names: Vec<String> = (0..MAX_TOOL_CALLS_PER_TURN)
        .map(|i| format!("f{i}.rs"))
        .collect();
    let at_limit = read_calls(names.iter().map(String::as_str));
    drive(
        Arc::clone(&tree),
        &format!("{at_limit}{TURN_CLOSE}"),
        &vocab,
    )
    .expect("exactly the limit drives");

    let over = format!("{at_limit}\n{}{TURN_CLOSE}", read_call("one_too_many.rs"));
    assert!(
        drive(Arc::clone(&tree), &over, &vocab).is_err(),
        "a call past MAX_TOOL_CALLS_PER_TURN must not drive"
    );
}

/// **Nothing but another call or the end of the turn may follow a call.** That
/// is what stops the model free-decoding an answer to a result it has not seen
/// — the single-call tree guaranteed it by baking the terminator into its
/// close, and the loop has to guarantee it at every level.
#[test]
fn a_call_cannot_be_followed_by_prose() {
    let (tree, vocab) = build_tree();
    let target = format!("{}The file says{TURN_CLOSE}", read_call("src/main.rs"));
    assert!(
        drive(Arc::clone(&tree), &target, &vocab).is_err(),
        "prose after a call must be masked"
    );
}

/// The live checkpoint's tokenizer — Qwen3.8-Flash-Next, the one the daemon
/// actually compiles its grammar against. The Qwen3 lookup above predates it and
/// finds an older vocabulary, so a boundary that tokenizes differently in 3.8
/// was invisible to every "real BPE" test.
fn cached_live_tokenizer() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(std::path::PathBuf::from)?;
    let snaps = home.join(".cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.exists())
}

// ── The live tokenizer's own encoding of a correct call ─────────────────────

/// Walk the grammar the way the daemon's decode loop does, feeding at every
/// decision point the token the **live tokenizer's natural encoding** of
/// `target` puts there — the token a model trained on that encoding writes.
///
/// The invariant is the one both of this build's live failures broke: **every
/// point where the model decides must fall on a boundary of the natural
/// encoding, and the natural token there must be one the grammar allows.** A
/// prefill that ends inside a natural token (` "` where the value is ` ""`)
/// hands the model a state its training never produced; a mask that forbids
/// the natural token (the `\n` between two calls) makes it pick something else.
/// A byte-level vocabulary has no multi-byte tokens and cannot see either,
/// which is how both passed the byte-level suite and failed live.
fn drive_natural(
    tree: Arc<StencilTree>,
    tok: &tokenizers::Tokenizer,
    target: &str,
) -> Result<(), String> {
    // The first marker is the model's: decoding it is what fires the stencil,
    // so the walk starts after it and the natural encoding is of the rest.
    let body = target
        .strip_prefix(MARKER)
        .ok_or("target must open with the marker")?;
    let bytes = body.as_bytes();
    let ids = tok.encode(body, false).map_err(|e| e.to_string())?;
    let pieces: Vec<(u32, Vec<u8>)> = ids
        .get_ids()
        .iter()
        .map(|&id| {
            (
                id,
                tok.decode(&[id], false).unwrap_or_default().into_bytes(),
            )
        })
        .collect();
    let mut starts = Vec::with_capacity(pieces.len());
    let mut at = 0usize;
    for (_, p) in &pieces {
        starts.push(at);
        at += p.len();
    }
    let token_at = |b: usize| starts.iter().position(|&s| s == b).map(|i| &pieces[i]);
    let before = |b: usize| String::from_utf8_lossy(&bytes[b.saturating_sub(24)..b]).into_owned();

    let mut session = StencilSession::new(tree);
    let mut b = 0usize;
    for _ in 0..100_000 {
        match session.next_action() {
            StencilAction::Prefill(toks) => {
                let text = tok.decode(&toks, false).map_err(|e| e.to_string())?;
                if !bytes[b..].starts_with(text.as_bytes()) {
                    return Err(format!(
                        "prefill {text:?} does not match the target after {:?}",
                        before(b)
                    ));
                }
                b += text.len();
            }
            // A masked decision fails only when the target's TEXT is unreachable:
            // no allowed token is a prefix of what comes next. Re-splitting the
            // same text into different tokens is how every masked branch works —
            // the model's `",` at the end of a tool name is only ever offered as
            // `"`, with the comma prefilled, and there is no choice in that. The
            // separator bug was different in kind: the allowed tokens spelled
            // other text, so the model had to choose something it did not mean.
            StencilAction::MaskedDecode(set) => {
                let (id, len) = set
                    .tokens()
                    .iter()
                    .filter_map(|&id| {
                        let t = tok.decode(&[id], false).ok()?;
                        (!t.is_empty() && bytes[b..].starts_with(t.as_bytes()))
                            .then_some((id, t.len()))
                    })
                    .max_by_key(|&(_, len)| len)
                    .ok_or_else(|| {
                        format!(
                            "no allowed token spells the target after {:?} — the model's \
                             continuation {:?} is masked out",
                            before(b),
                            String::from_utf8_lossy(&bytes[b..(b + 16).min(bytes.len())])
                        )
                    })?;
                session
                    .observe(id, &bytes[b..b + len])
                    .map_err(|e| format!("{e:?}"))?;
                b += len;
            }
            StencilAction::FreeDecode { .. } => {
                let (id, p) = token_at(b).ok_or_else(|| {
                    format!(
                        "a free decode starts INSIDE a natural token, after {:?} — \
                         the grammar prefilled part of a token the model writes whole",
                        before(b)
                    )
                })?;
                match session.observe(*id, p).map_err(|e| format!("{e:?}"))? {
                    Observe::TokenClosedDrop => {
                        return Err(format!(
                            "the natural token {:?} after {:?} was dropped as a repair",
                            String::from_utf8_lossy(p),
                            before(b)
                        ))
                    }
                    // The close fell inside this token: only the span's part is
                    // committed, and the successor writes the rest.
                    Observe::SpanClosed { leftover } if leftover > 0 && leftover < p.len() => {
                        b += p.len() - leftover
                    }
                    _ => b += p.len(),
                }
            }
            StencilAction::Exit => {
                return match b == bytes.len() {
                    true => Ok(()),
                    false => Err(format!(
                        "the turn ended with {:?} still to write",
                        &body[b..]
                    )),
                };
            }
        }
    }
    Err("runaway walk".into())
}

/// The daemon's grammar compiled against the live checkpoint's tokenizer.
fn live_tree() -> (Arc<StencilTree>, tokenizers::Tokenizer) {
    let path = cached_live_tokenizer().expect("Qwen3.8 tokenizer cached");
    let tok = tokenizers::Tokenizer::from_file(&path).unwrap();
    let im_end = tok.token_to_id(TURN_CLOSE).expect("<|im_end|> resolves");
    let vocab = HfVocab::new(tok.clone(), &[im_end], 0);
    let tree = compile(&loop_spec(MAX_TOOL_CALLS_PER_TURN), &vocab)
        .expect("the grammar compiles against the live tokenizer");
    (Arc::new(tree), tok)
}

/// **Every call a model would naturally write drives the live grammar.**
///
/// Each target is a whole turn exactly as the checkpoint's template lays it
/// out, walked with the live tokenizer's own encoding. The empty `path` and
/// the three-call turn are the two live failures; the rest pin the ordinary
/// shapes, so a grammar change that breaks one is caught here on the CPU
/// rather than in a conversation that ends silently.
#[test]
#[ignore = "requires the cached Qwen3.8 tokenizer.json"]
fn natural_calls_drive_the_live_grammar() {
    let (tree, tok) = live_tree();
    let calls = [
        r#"{"name": "file_list", "arguments": {"path": ""}}"#,
        r#"{"name": "file_list", "arguments": {}}"#,
        r#"{"name": "file_list", "arguments": {"path": "zend/src"}}"#,
        r#"{"name": "file_read", "arguments": {"path": "src/main.rs", "page": 0}}"#,
        r#"{"name": "file_grep", "arguments": {"pattern": "fn main", "prefix": "zend/"}}"#,
        r#"{"name": "calculator", "arguments": {"expression": "2 + 2"}}"#,
    ];
    let mut failures = Vec::new();
    for call in calls {
        let target = format!("{MARKER}\n{call}\n</tool_call>{TURN_CLOSE}");
        if let Err(e) = drive_natural(Arc::clone(&tree), &tok, &target) {
            failures.push(format!("{call}\n    {e}"));
        }
    }
    // Several calls in one turn, laid out as the template lays them out.
    let batched = ["src/main.rs", "src/lib.rs", "Cargo.toml"]
        .iter()
        .map(|p| {
            format!(
                "{MARKER}\n{{\"name\": \"file_read\", \"arguments\": {{\"path\": \"{p}\", \
                 \"page\": 0}}}}\n</tool_call>"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    if let Err(e) = drive_natural(Arc::clone(&tree), &tok, &format!("{batched}{TURN_CLOSE}")) {
        failures.push(format!("three file_reads in one turn\n    {e}"));
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// **The natural drive rejects both shapes that failed live.** A harness that
/// passes the fixed grammar proves nothing unless it fails the broken one, so
/// both old shapes are rebuilt here and must be refused — for the right reason.
#[test]
#[ignore = "requires the cached Qwen3.8 tokenizer.json"]
fn the_natural_drive_rejects_both_live_failures() {
    let path = cached_live_tokenizer().expect("Qwen3.8 tokenizer cached");
    let tok = tokenizers::Tokenizer::from_file(&path).unwrap();
    let im_end = tok.token_to_id(TURN_CLOSE).unwrap();
    let vocab = HfVocab::new(tok.clone(), &[im_end], 0);

    // 1. The marker glued to the previous close, as the loop first offered it.
    let glued = ToolCallEnvelope {
        between_calls: String::new(),
        ..production_envelope()
    };
    let spec =
        compile_tool_call_loop(&catalog(), &glued, MAX_TOOL_CALLS_PER_TURN, TURN_CLOSE).unwrap();
    let tree = Arc::new(compile(&spec, &vocab).unwrap());
    let two = ["a.rs", "b.rs"]
        .iter()
        .map(|p| {
            format!(
                "{MARKER}\n{{\"name\": \"file_read\", \"arguments\": {{\"path\": \"{p}\", \
                 \"page\": 0}}}}\n</tool_call>"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let err = drive_natural(tree, &tok, &format!("{two}{TURN_CLOSE}"))
        .expect_err("a glued separator must fail the natural drive");
    assert!(err.contains("masked out"), "wrong reason: {err}");

    // 2. A string value whose opening quote is prefilled, as every string was.
    let mut s = TreeSpec::new("prefilled-quote");
    let end = s.push(NodeSpec::End);
    let close = s.push(NodeSpec::Static {
        text: format!("}}}}\n</tool_call>{TURN_CLOSE}"),
        next: end,
    });
    let value = s.push(NodeSpec::FreeText {
        term: Terminator::JsonString,
        eos_ends: false,
        limits: FreeTextLimits::json_string(),
        close_token: None,
        suppress_close: false,
        next: close,
    });
    s.root = s.push(NodeSpec::Static {
        text: "\n{\"name\": \"file_list\", \"arguments\": {\"prefix\": \"".into(),
        next: value,
    });
    let tree = Arc::new(compile(&s, &vocab).unwrap());
    let target = format!(
        "{MARKER}\n{{\"name\": \"file_list\", \"arguments\": {{\"prefix\": \"\"}}}}\n\
         </tool_call>{TURN_CLOSE}"
    );
    let err = drive_natural(tree, &tok, &target)
        .expect_err("a prefilled opening quote must fail the natural drive");
    assert!(
        err.contains("INSIDE a natural token"),
        "wrong reason: {err}"
    );
}
