//! Tool orchestration for `zend`.
//!
//! Bridges the [`zend_tools`] registry to a live [`candle_conversation`]
//! conversation:
//!
//! 1. **Tool catalog** — at engine startup, generates one Hermes-format
//!    JSON line per registered tool and appends it as a system-prompt
//!    section via [`candle_conversation::projection::Builder::add_section`].
//!    The base conversation prefills each section, capturing per-section
//!    BDP sigs so the projection's `top_k` rule can pick the K most
//!    relevant tools at projection time.
//!
//! 2. **Chained tool turns** — after a turn's response is decoded,
//!    [`extract_tool_calls`] scans for `<tool_call>{json}</tool_call>`
//!    blocks *outside* the model's `<think>…</think>` reasoning (a call written
//!    mid-thought is deliberation, not an invocation, and is left inline).  Each
//!    call is dispatched via [`zend_tools::runner::run`];
//!    the JSON results are wrapped in `<tool_response>...</tool_response>`,
//!    concatenated, and submitted as the next turn's user message.  The
//!    loop continues until a turn produces no further calls or the
//!    iteration cap is hit.
//!
//! Tool calls and tool responses are filtered from the streamed token
//! output presented to the client — the user sees only the assistant's
//! final natural-language answer.

use std::collections::HashSet;
use std::sync::Arc;

use candle_conversation::models::Dialect;
use candle_conversation::projection::{
    Builder as ProjectionBuilder, GroupId, LayerId, Reserved, SectionId, SelectionRule,
};
use candle_conversation::think_strip::strip_think_blocks_keep_layout;
use serde::Deserialize;
use serde_json::Value;

use zend_tools::{registry, ToolContext};

/// The names of every tool that is **not** high-risk — the subset projected in
/// "Restricted" tools mode. Derived from the registry's `.risky()` policy (see
/// [`zend_tools::registry`]); "None" mode projects no tools, "Comprehensive"
/// projects all of them.
pub fn safe_tool_names() -> HashSet<String> {
    crate::tool_def::safe_names()
}

/// One Hermes tool-call block parsed from a model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// One executed tool's result, ready to be wrapped in `<tool_response>`.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The original call this result corresponds to.  Carried for
    /// diagnostic / test purposes — the orchestrator only ever
    /// serialises `response` into the `<tool_response>` body.
    #[allow(dead_code)]
    pub call: ToolCall,
    /// The JSON value produced by [`zend_tools::runner::run`].  On
    /// success this is the tool's typed `Response`; on failure it's a
    /// `{"error": "<code>", "detail": "..."}` shape.  The orchestrator
    /// passes both back to the model verbatim so it can react to
    /// errors.
    pub response: Value,
}

// ── Tool catalog injection ───────────────────────────────────────────────────

/// Install the registry's tool catalog as a `SectionCollection` inside
/// `dialogue`'s system prompt.
///
/// On entry the schema is expected to declare a YAML-defined collection
/// named `"tools"` (typically wedged between `tools_intro` and
/// `tools_outro` static sections).  This function:
///
/// 1. Looks up the existing `tools` collection's id.
/// 2. Calls [`ProjectionBuilder::add_section_to_collection`] for each
///    tool in the registry, registering one section per tool with the
///    Hermes-format JSON line as its content.
/// 3. Returns `(tool_name, section_id, json_line)` triples in registry
///    order.  Each section's BDP sigs come from the single prefill of
///    its `json_line` content during `insert_section_collection`.
///
/// Static framing sections (mode/frame/grounding/tools_intro/
/// tools_outro) are *outside* the collection — they always emit.  The
/// collection's own selection rule (declared in YAML, typically
/// `TopK { k: 3 }`) is what filters the tool list.
pub fn install_tool_catalog(
    builder: &mut ProjectionBuilder,
) -> anyhow::Result<Vec<(String, SectionId, String)>> {
    // A system prompt without a `tools` collection is a deliberately tool-free
    // projection (e.g. a conversational mind): install nothing rather than error.
    let Some(collection_id) = builder.id_for_system_collection("tools") else {
        tracing::info!("system prompt declares no 'tools' collection — tools disabled");
        return Ok(Vec::new());
    };
    let mut out: Vec<(String, SectionId, String)> = Vec::new();
    for def in crate::tool_def::all() {
        let json_line = def.json_line();
        let id = builder
            .add_section_to_collection(collection_id, def.name.clone(), &json_line, 100.0)
            .map_err(|e| anyhow::anyhow!("add_section_to_collection({}): {}", def.name, e))?;
        out.push((def.name.clone(), id, json_line));
    }
    // This function only lays down the per-tool sections. The tool-catalog
    // *overview* is sealed separately into the `ToolSummary` /
    // `ToolSummaryRestricted` reserved sections at session startup and associated
    // with this collection per mode in `build_mode_builder` (via
    // `set_collection_summary_section`), so projection emits the full name listing
    // ahead of the provenance-selected subset.
    Ok(out)
}

/// The selector id the calibration projection's `tools` collection reads
/// ([`SelectionRule::Named`]) to pin exactly one tool. The "Calibrating sections"
/// driver sets it per run via [`candle_conversation::TurnOptions::selection`].
pub const CALIB_TOOL_SELECTOR: &str = "tool";

/// Section-id partition for the calibration projection: high in the u32 space,
/// just below the per-kind reserved singleton band (`u32::MAX - 0..3`) and far
/// above the user schema's low 1..n ids and the tool ids allocated above them.
/// 4096 ids is ample headroom for the frame, its summary framing, the whole tool
/// catalog (one section each), and the outro.
const CALIB_SECTION_BASE: u32 = u32::MAX - 4096;

/// Build the hidden **calibration projection** used by the "Calibrating sections"
/// load phase.
///
/// A single `Reserved::Calibration` layer whose system prompt frames a `tools`
/// collection holding the **whole** catalog — one name-keyed section per tool —
/// governed by [`SelectionRule::Named`] on the [`CALIB_TOOL_SELECTOR`] selector.
/// A calibration run pins one tool by setting that selector to the tool's name
/// (via `TurnOptions::selection`), so the projection emits exactly that tool,
/// wrapped in the proven Hermes `<tools>…</tools>` envelope. The model then sees
/// a single available tool and free-decodes the full think→call trajectory (and
/// its reprojection wide-Q windows) — the same conditions under which the
/// `datetime` case calibrated cleanly, now reproduced for every tool.
///
/// The reserved layer keeps these conversations off every user projection (their
/// turns never enter dialogue retrieval), and the `tools` collection's distinct,
/// name-keyed member sections mean no two tools ever collide on one slot — the
/// failure mode of the earlier single-reserved-section approach.
///
/// The system-prompt envelope markers live in the section *content* (the
/// assembler does not wrap system sections): the frame opens with
/// `dialect.system_start` and the outro closes with `dialect.system_end`.
pub fn build_calibration_projection(
    dialect: &Dialect,
) -> anyhow::Result<(ProjectionBuilder, LayerId, GroupId)> {
    let layer = LayerId::reserved(Reserved::Calibration);
    let group = GroupId::reserved(Reserved::Calibration);

    // Frame: system-open marker + framing prose + the opening `<tools>` marker.
    // The collection's one selected tool line emits next; the outro closes it.
    let frame = format!(
        "{sys_start}You are a coding assistant. Use the available tool to satisfy the \
         user's request.\n\n# Tools\n\nYou may call the tool below. Return the call as a \
         JSON object with its name and arguments inside <tool_call></tool_call> XML tags.\n\n\
         <tools>\n",
        sys_start = dialect.system_start,
    );
    let mut builder =
        ProjectionBuilder::for_reserved_corpus(&frame, Reserved::Calibration, CALIB_SECTION_BASE);

    let collection = builder
        .add_collection(
            "tools",
            SelectionRule::Named {
                selector: CALIB_TOOL_SELECTOR.to_string(),
            },
            0.0,
        )
        .map_err(|e| anyhow::anyhow!("calibration add_collection: {e}"))?;
    for def in crate::tool_def::all() {
        builder
            .add_section_to_collection(collection, def.name.clone(), def.json_line(), 100.0)
            .map_err(|e| anyhow::anyhow!("calibration add tool {}: {e}", def.name))?;
    }

    // Outro: close the `<tools>` block, give the call instruction, and emit the
    // system-close marker so the whole system message is well-formed.
    let outro = format!(
        "</tools>\n\nFor a tool call, return:\n<tool_call>\n\
         {{\"name\": <tool-name>, \"arguments\": <args-json-object>}}\n</tool_call>{sys_end}",
        sys_end = dialect.system_end,
    );
    builder
        .add_section("tools_outro", outro, 50.0)
        .map_err(|e| anyhow::anyhow!("calibration add outro: {e}"))?;

    Ok((builder, layer, group))
}

// ── Tool-call extraction ─────────────────────────────────────────────────────

/// Top-level balanced `{...}` object spans in `text`, as `(start, end)`
/// half-open byte ranges. String literals are respected (braces inside JSON
/// strings don't perturb the depth count), and only outermost objects are
/// returned (a nested object is part of its parent's span). Operates on bytes;
/// the brace/quote/escape sentinels are ASCII, so multi-byte UTF-8 is safe.
fn balanced_object_spans(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    spans.push((start, i + 1));
                }
            }
            _ => {}
        }
    }
    spans
}

/// Scan a model response for tool-call blocks **outside** its reasoning.
///
/// `<think>…</think>` blocks are stripped before scanning: a tool call the model
/// writes while thinking is deliberation, not an invocation, so it is left inline
/// in the streamed text and never dispatched. Only calls in the post-think answer
/// are returned.
///
/// The canonical Hermes format is `<tool_call>{json}</tool_call>`, but
/// Qwen3-A3B (especially in `/no_think` mode) is unreliable about the tags:
/// sometimes it elides the opening `<tool_call>` and emits `{json}</tool_call>`,
/// and sometimes it emits a bare `{json}` with no tags at all. Verified by
/// inspection of raw token IDs in the sampling trace — the model genuinely
/// drops the tags; it's not a tokenizer/decoder bug. All three shapes are
/// accepted so we don't lose tool calls to that quirk.
///
/// Strategy:
/// 1. Strict `<tool_call>{json}</tool_call>` matches.
/// 2. Lenient `{json}</tool_call>` matches (elided opener) not already covered.
/// 3. Bare `{json}` objects (no tags) not already covered — but only when the
///    object names a real tool (or alias) via the registry, so prose JSON and
///    fabricated tool *responses* (`{"error": ...}`) aren't mistaken for calls.
/// Malformed blocks are silently skipped.
pub fn extract_tool_calls(response_text: &str) -> Vec<ToolCall> {
    use regex::Regex;
    use std::sync::OnceLock;
    // A `<tool_call>` emitted *inside* a `<think>…</think>` reasoning block is the
    // model thinking out loud, not an invocation to dispatch. Strip the reasoning
    // blocks first so only calls in the post-think answer are extracted — the JSON
    // still streams to the client inline, it is simply never executed.
    let response_text = strip_think_blocks_keep_layout(response_text);
    let response_text = response_text.as_str();
    // Strict, well-formed match: <tool_call>...{...}...</tool_call>
    static STRICT_RE: OnceLock<Regex> = OnceLock::new();
    let strict_re = STRICT_RE.get_or_init(|| {
        Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").expect("static regex")
    });
    // Lenient close-only match: ...{...}</tool_call> with NO opener
    // anywhere between the previous boundary and the JSON.
    //
    // The regex captures `(\{[^<]*?\})` to ensure the JSON body
    // doesn't contain any other `<` (which would suggest it crosses
    // another tag boundary).
    static LENIENT_RE: OnceLock<Regex> = OnceLock::new();
    let lenient_re = LENIENT_RE
        .get_or_init(|| Regex::new(r"(?s)(\{[^<]*?\})\s*</tool_call>").expect("static regex"));

    let mut out = Vec::new();
    let mut consumed_ends: Vec<usize> = Vec::new();
    // JSON-object byte spans already claimed by a tagged match, so the bare
    // pass (3) doesn't re-detect the same object.
    let mut consumed_spans: Vec<(usize, usize)> = Vec::new();

    // Pass 1: strict <tool_call>...</tool_call> matches.
    for cap in strict_re.captures_iter(response_text) {
        let full_end = cap.get(0).map(|m| m.end()).unwrap_or(0);
        let json = match cap.get(1) {
            Some(m) => m,
            None => continue,
        };
        if let Ok(raw) = serde_json::from_str::<RawCall>(json.as_str()) {
            if !raw.name.is_empty() {
                let arguments = raw.args();
                out.push(ToolCall {
                    name: raw.name,
                    arguments,
                });
            }
        }
        consumed_ends.push(full_end);
        consumed_spans.push((json.start(), json.end()));
    }

    // Pass 2: lenient `{json}</tool_call>` matches — only consider
    // matches whose `</tool_call>` end position isn't already covered
    // by a strict match.  This way, well-formed calls don't get
    // double-counted, but `<think></think>\n\n{...}</tool_call>`
    // (model elided the opener) still resolves to one tool call.
    for cap in lenient_re.captures_iter(response_text) {
        let full_end = cap.get(0).map(|m| m.end()).unwrap_or(0);
        if consumed_ends.contains(&full_end) {
            continue;
        }
        // If a `<tool_call>` opener appears between the JSON's start
        // and the previous match (or document start), the strict
        // regex would already have caught it; we wouldn't be here.
        // But guard against the JSON body itself containing `<` —
        // the `[^<]*?` in the regex already enforces that.
        let json = match cap.get(1) {
            Some(m) => m,
            None => continue,
        };
        if let Ok(raw) = serde_json::from_str::<RawCall>(json.as_str()) {
            if !raw.name.is_empty() {
                let arguments = raw.args();
                out.push(ToolCall {
                    name: raw.name,
                    arguments,
                });
            }
        }
        consumed_spans.push((json.start(), json.end()));
    }

    // Pass 3: bare `{json}` objects with no wrapper tags at all. Lower
    // precision than the tagged passes, so it's gated four ways: the object
    // must (a) not overlap a tagged match, (b) carry call values under either
    // `arguments` or `parameters` (Qwen3 uses the schema key for the call when it
    // degrades), (c) have NO top-level `description` — so a tool *definition* echo
    // (`{"name","description","parameters":<schema>}`) isn't taken for a call,
    // since it also carries `parameters` — and (d) name a real tool/alias in the
    // registry. This recovers the calls Qwen3 emits as raw JSON while ignoring
    // prose JSON, tool definitions, and fabricated tool responses.
    for (start, end) in balanced_object_spans(response_text) {
        let overlaps = consumed_spans.iter().any(|&(s, e)| start < e && s < end);
        if overlaps {
            continue;
        }
        if let Ok(raw) = serde_json::from_str::<RawCall>(&response_text[start..end]) {
            if !raw.name.is_empty()
                && raw.has_args()
                && !raw.looks_like_definition()
                && registry::find(&raw.name).is_some()
            {
                let arguments = raw.args();
                out.push(ToolCall {
                    name: raw.name,
                    arguments,
                });
            }
        }
    }
    out
}

#[derive(Deserialize)]
struct RawCall {
    /// `function` is accepted as an alias for `name`: Qwen3-30B-A3B
    /// occasionally emits `{"function":"<tool>", "arguments":{...}}` instead
    /// of `{"name":"<tool>", ...}`.  Tolerating it costs nothing and
    /// recovers an otherwise-lost tool call.
    #[serde(alias = "function")]
    name: String,
    #[serde(default)]
    arguments: Option<Value>,
    /// Qwen3-30B-A3B sometimes emits `"parameters"` — the tool *schema* key —
    /// in place of `"arguments"` for the call values (seen when it drops the
    /// `<tool_call>` wrapper too).  Accepted as a fallback so the args aren't lost.
    #[serde(default)]
    parameters: Option<Value>,
    /// Present only on a tool *definition* (`{"name","description","parameters"}`),
    /// never on a call.  Lets the bare pass reject a definition echo even though it
    /// also carries a top-level `"parameters"`.
    #[serde(default)]
    description: Option<Value>,
}

impl RawCall {
    /// The call arguments: `"arguments"` if present, else the `"parameters"`
    /// fallback, else null.
    fn args(&self) -> Value {
        self.arguments
            .clone()
            .or_else(|| self.parameters.clone())
            .unwrap_or(Value::Null)
    }
    /// Whether the object carries call values under either key.
    fn has_args(&self) -> bool {
        self.arguments.is_some() || self.parameters.is_some()
    }
    /// A tool *definition* echo carries a top-level `description`; a call never
    /// does (a param *named* `description` lives inside `parameters`, not here).
    fn looks_like_definition(&self) -> bool {
        self.description.is_some()
    }
}

// ── Tool dispatch ────────────────────────────────────────────────────────────

/// Run one parsed tool call against the registry.  Always returns a
/// JSON value the model can consume — successful tools return their
/// typed response, missing tools return
/// `{"error":"unknown_tool","detail":"..."}`.
pub fn run_tool(ctx: &ToolContext, call: &ToolCall) -> Value {
    match registry::find(&call.name) {
        Some(t) => (t.run)(ctx, &call.arguments),
        None => serde_json::json!({
            "error": "unknown_tool",
            "detail": format!("no tool named {:?}", call.name),
        }),
    }
}

/// Dispatch every parsed tool call sequentially and pair each call with
/// its result.  Sequential rather than concurrent because the same
/// session/notes/credential stores are mutated by some tools — running
/// concurrently would require per-tool locking discipline that is
/// currently the responsibility of the `zend-tools` state stores
/// (which are `Arc<RwLock<...>>` internally, but ordering across
/// distinct tools matters for stateful flows like
/// `ssh_session_open` → `ssh_session_exec`).
pub fn run_tool_calls(ctx: &ToolContext, calls: Vec<ToolCall>) -> Vec<ToolResult> {
    calls
        .into_iter()
        .map(|c| {
            let resp = run_tool(ctx, &c);
            ToolResult {
                call: c,
                response: resp,
            }
        })
        .collect()
}

/// Format executed tool results into a single chained-turn string,
/// suitable to pass as the user message of the next conversation turn.
///
/// Format: one `<tool_response>{json}</tool_response>` block per result,
/// separated by newlines.  This matches the Hermes spec — the model
/// reads each block and continues its prior reasoning.
pub fn format_tool_responses(results: &[ToolResult]) -> String {
    let mut out = String::new();
    for r in results {
        let body = serde_json::to_string(&r.response)
            .unwrap_or_else(|_| "{\"error\":\"internal_error\"}".to_string());
        out.push_str("<tool_response>");
        out.push_str(&body);
        out.push_str("</tool_response>\n");
    }
    out
}

// ── Public helper bundle ─────────────────────────────────────────────────────

/// Per-session tool execution context.  Holds the [`ToolContext`] and
/// the optional [`zend_tools::SubagentRunner`] (currently always `None`
/// — subagent loops aren't wired yet).  Cloned cheaply (Arc-shared
/// stores).
#[derive(Clone)]
pub struct ToolHost {
    pub ctx: Arc<ToolContext>,
}

impl ToolHost {
    pub fn new() -> Self {
        Self {
            ctx: Arc::new(ToolContext::default()),
        }
    }
}

impl Default for ToolHost {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_tool_calls_finds_well_formed_block() {
        let text = r#"Here is the result.
<tool_call>
{"name": "datetime", "arguments": {"timezone": "UTC"}}
</tool_call>
"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[0].arguments["timezone"], "UTC");
    }

    #[test]
    fn extract_tool_calls_finds_multiple_blocks() {
        let text = r#"
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>
some text
<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>
"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[1].name, "calculator");
    }

    #[test]
    fn extract_tool_calls_skips_malformed() {
        let text = r#"
<tool_call>not json</tool_call>
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>
<tool_call>{}</tool_call>
"#;
        let calls = extract_tool_calls(text);
        // Only the second block has a valid name.
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_handles_no_calls() {
        let text = "Just a normal response with no tools.";
        let calls = extract_tool_calls(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_tool_calls_tolerates_missing_opener() {
        // Qwen3-A3B in `/no_think` mode often emits the JSON without
        // the leading `<tool_call>` tag — the close is reliable but
        // the open is not.  Make sure we still recover the call.
        let text = "<think>\n\n</think>\n\n{\"name\": \"calculator\", \
                    \"arguments\": {\"expression\": \"1+1\"}}\n</tool_call>";
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "expected the opener-elided tool call to be recovered, got {calls:?}",
        );
        assert_eq!(calls[0].name, "calculator");
        assert_eq!(calls[0].arguments["expression"], "1+1");
    }

    #[test]
    fn extract_tool_calls_mixed_well_formed_and_opener_elided() {
        // First call has both tags, second only has the closer — both
        // should be recovered, in order.
        let text = r#"
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>
some text
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>
"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2, "got: {calls:?}");
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[1].name, "calculator");
    }

    #[test]
    fn extract_tool_calls_lenient_does_not_double_count_well_formed() {
        // A perfectly-formed `<tool_call>...</tool_call>` should
        // produce exactly one ToolCall — the lenient pass must not
        // re-match the same close-tag.
        let text = r#"<tool_call>{"name": "datetime", "arguments": {}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "well-formed block double-counted: {calls:?}"
        );
    }

    #[test]
    fn extract_tool_calls_lenient_ignores_orphan_close_without_json() {
        // A stray `</tool_call>` with no JSON object before it must
        // not synthesize a tool call from prior text.
        let text = "just some narrative</tool_call>";
        let calls = extract_tool_calls(text);
        assert!(calls.is_empty(), "got phantom call: {calls:?}");
    }

    #[test]
    fn extract_tool_calls_recovers_bare_json_no_tags() {
        // Qwen3 sometimes emits the call as raw JSON with no tags at all.
        let text = r#"{"name": "web_search", "arguments": {"query": "q", "max_results": 5}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "bare JSON call not recovered: {calls:?}");
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["max_results"], 5);
    }

    #[test]
    fn extract_tool_calls_bare_json_with_preamble_and_nested_args() {
        // Preamble prose before a bare, multi-line call with nested arguments.
        let text = "I'll run that for you.\n{\n  \"name\": \"calculator\",\n  \"arguments\": {\"expression\": \"(1+2)*3\"}\n}\n";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "got: {calls:?}");
        assert_eq!(calls[0].name, "calculator");
        assert_eq!(calls[0].arguments["expression"], "(1+2)*3");
    }

    #[test]
    fn extract_tool_calls_bare_json_resolves_alias() {
        // A bare call under a known alias is recovered (the registry gate
        // resolves it); the canonical name is settled later by `run_tool`.
        let text = r#"{"name": "file_create", "arguments": {"path": "a.txt", "content": "hi"}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "aliased bare call not recovered: {calls:?}");
        assert_eq!(calls[0].name, "file_create");
        assert!(registry::find(&calls[0].name).is_some());
    }

    #[test]
    fn extract_tool_calls_bare_json_ignores_non_tool_and_fabricated_response() {
        // Prose JSON that doesn't name a real tool, and a fabricated tool
        // *response*, must not be mistaken for calls.
        let unknown = r#"{"name": "definitely_not_a_tool", "arguments": {}}"#;
        assert!(extract_tool_calls(unknown).is_empty());
        let fabricated = r#"{"error": "No active HTTP sessions found."}"#;
        assert!(extract_tool_calls(fabricated).is_empty());
        // A tool *definition* echo (top-level `description` + a `parameters`
        // schema) is not a call — even though `parameters` is now accepted as the
        // args key, the `description` gate rejects it.
        let definition = r#"{"name": "ping_icmp", "description": "ping a host", "parameters": {}}"#;
        assert!(extract_tool_calls(definition).is_empty());
    }

    #[test]
    fn extract_tool_calls_recovers_bare_parameters_keyed_call() {
        // The exact Qwen3-A3B degradation from the substrate: no `<tool_call>`
        // wrapper AND the schema key `parameters` in place of `arguments`. Recover
        // it as a real call, carrying the values.
        let text = r#"{"name":"datetime","parameters":{"timezone":"Australia/Sydney"}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "bare parameters-keyed call not recovered: {calls:?}",
        );
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[0].arguments["timezone"], "Australia/Sydney");
    }

    #[test]
    fn extract_tool_calls_parameters_key_still_rejects_definition() {
        // A full definition echo carries `parameters` too, but its top-level
        // `description` must keep it from being mistaken for a call now that
        // `parameters` is an accepted args key.
        let def = r#"{"name":"datetime","description":"current time in a timezone","parameters":{"type":"object","properties":{"timezone":{"type":"string"}}}}"#;
        assert!(
            extract_tool_calls(def).is_empty(),
            "definition echo taken for a call",
        );
    }

    #[test]
    fn extract_tool_calls_tagged_call_keeps_parameters_values() {
        // Even inside a `<tool_call>` wrapper Qwen3 sometimes uses `parameters`;
        // the args must survive rather than dropping to null.
        let text =
            r#"<tool_call>{"name": "datetime", "parameters": {"timezone": "UTC"}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "got: {calls:?}");
        assert_eq!(calls[0].arguments["timezone"], "UTC");
    }

    #[test]
    fn extract_tool_calls_bare_pass_does_not_double_count_tagged() {
        // A fully-tagged call must yield exactly one ToolCall — the bare pass
        // must skip the object already claimed by the strict match.
        let text = r#"<tool_call>{"name": "datetime", "arguments": {}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "tagged call double-counted: {calls:?}");
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_ignores_call_inside_think_block() {
        // A tool call the model writes while reasoning is deliberation, not an
        // invocation — it must not be dispatched.
        let text = r#"<think>
Maybe I should call <tool_call>{"name": "datetime", "arguments": {}}</tool_call> to check.
</think>
The time doesn't matter here."#;
        let calls = extract_tool_calls(text);
        assert!(calls.is_empty(), "in-think call was dispatched: {calls:?}");
    }

    #[test]
    fn extract_tool_calls_dispatches_call_after_think_block() {
        // The model reasons, closes the block, then emits the real call.
        let text = r#"<think>
I'll check the time.
</think>
<tool_call>{"name": "datetime", "arguments": {"timezone": "UTC"}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "post-think call not dispatched: {calls:?}");
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_ignores_in_think_keeps_post_think() {
        // A call mid-thought is ignored; the one after `</think>` is kept.
        let text = r#"<think>
I could <tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call> but no.
</think>
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "expected only the post-think call: {calls:?}"
        );
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_ignores_bare_json_call_inside_think() {
        // The bare-JSON recovery pass must also respect the think boundary.
        let text =
            "<think>\n{\"name\": \"datetime\", \"arguments\": {}}\n</think>\nNo tool needed.";
        let calls = extract_tool_calls(text);
        assert!(
            calls.is_empty(),
            "bare in-think call was dispatched: {calls:?}"
        );
    }

    #[test]
    fn run_tool_unknown_returns_error_shape() {
        let ctx = ToolContext::default();
        let call = ToolCall {
            name: "tool_that_does_not_exist".to_string(),
            arguments: serde_json::json!({}),
        };
        let resp = run_tool(&ctx, &call);
        assert_eq!(resp["error"], "unknown_tool");
        assert!(resp["detail"]
            .as_str()
            .unwrap()
            .contains("tool_that_does_not_exist"));
    }

    #[test]
    fn run_tool_known_dispatches() {
        // datetime is a no-state tool that should always succeed with empty args.
        let ctx = ToolContext::default();
        let call = ToolCall {
            name: "datetime".to_string(),
            arguments: serde_json::json!({}),
        };
        let resp = run_tool(&ctx, &call);
        // Should be a successful response (not an error shape).
        assert!(
            resp.get("error").is_none(),
            "datetime call returned error: {resp:?}"
        );
    }

    #[test]
    fn format_tool_responses_wraps_each_result() {
        let results = vec![
            ToolResult {
                call: ToolCall {
                    name: "datetime".to_string(),
                    arguments: Value::Null,
                },
                response: serde_json::json!({"iso": "2026-05-09"}),
            },
            ToolResult {
                call: ToolCall {
                    name: "calculator".to_string(),
                    arguments: Value::Null,
                },
                response: serde_json::json!({"result": 4}),
            },
        ];
        let formatted = format_tool_responses(&results);
        assert!(formatted.contains("<tool_response>"));
        assert!(formatted.contains("</tool_response>"));
        assert!(formatted.contains("\"iso\":\"2026-05-09\""));
        assert!(formatted.contains("\"result\":4"));
    }

    #[test]
    fn tool_def_json_line_is_valid_hermes_format() {
        // The catalog deliberately emits a flat `{"name","description",
        // "parameters"}` shape (Qwen3-A3B echoes the canonical `"function"`
        // wrapper key back into calls; flattening also saves ~10 tokens/tool).
        let def = crate::tool_def::find("datetime").expect("datetime tool defined");
        let parsed: Value = serde_json::from_str(def.json_line().trim_end()).expect("must parse");
        assert_eq!(parsed["name"], "datetime");
        assert!(parsed["description"].is_string());
        assert!(parsed["parameters"].is_object());
    }
}
