//! Finds the tool calls in a model response, outside its reasoning, in each of
//! the shapes the model emits them.
//!
//! Qwen3 models sometimes drop the `<tool_call>` tags or emit a bare JSON
//! object, so calls are recognised in three passes: tagged, closer-only and
//! bare. A bare object counts only when it names a registered tool, carries
//! arguments and has no top-level description, so prose JSON and a tool
//! definition echoed back are not dispatched. A call written inside a reasoning
//! block is never returned. The function-block syntax of the Qwen3.5 and Qwen3.8
//! lineage is rewritten into JSON before the passes run.

use super::*;

/// One Hermes tool-call block parsed from a model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

// ── Tool-call extraction ─────────────────────────────────────────────────────

/// Top-level balanced `{...}` object spans in `text`, as `(start, end)`
/// half-open byte ranges. String literals are respected (braces inside JSON
/// strings don't perturb the depth count), and only outermost objects are
/// returned (a nested object is part of its parent's span). Operates on bytes;
/// the brace/quote/escape sentinels are ASCII, so multi-byte UTF-8 is safe.
pub(super) fn balanced_object_spans(text: &str) -> Vec<(usize, usize)> {
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
///
/// Malformed blocks are silently skipped.
///
/// # The function-block syntax is translated, not parsed a second time
///
/// Qwen3.5 / Qwen3.8 put no JSON in a call at all. Their [`CallStyle`] is
/// `FunctionBlock`, so the stencil *forces*
/// `<function=name>\n<parameter=k>\nraw</parameter>\n</function>` — inside the
/// ordinary `<tool_call>` markers. All three strategies above look for a `{`
/// straight after the marker, so all three found nothing: `calls.is_empty()`
/// read as "the model produced a final answer", the turn ended, no tool ran, and
/// nothing logged a failure — while the GUI drew a tool card, because the
/// markers themselves were exactly right. Every individual part worked as
/// written and a whole class of request was inert.
///
/// [`function_blocks_to_json`] rewrites that body into strategy 1's shape before
/// the passes run, so this file keeps the one syntax it has always understood
/// and the new one lives in exactly two places: the envelope that writes it and
/// the module that reads it.
///
/// [`CallStyle`]: candle_conversation::stencil::CallStyle
pub fn extract_tool_calls(response_text: &str) -> Vec<ToolCall> {
    calls_in_answer(&answer_text(response_text))
        .into_iter()
        .map(|(_, call)| call)
        .collect()
}

/// The part of a response that can make calls, in the one syntax the call
/// scanners read: `<think>…</think>` blocks blanked out, and function blocks
/// translated to JSON.
///
/// A `<tool_call>` emitted *inside* a reasoning block is the model thinking out
/// loud, not an invocation to dispatch, so the reasoning is stripped first — the
/// JSON still streams to the client inline, it is simply never executed.
///
/// **Translated after the strip**, so a call written mid-thought is already
/// gone — translating first would spend the work rewriting text about to be
/// discarded, and would put a dispatchable shape into deliberation.
///
/// Every scanner of a round's calls reads this same text, so the byte offsets
/// they report are comparable — see [`crate::tool_round`].
pub(crate) fn answer_text(response_text: &str) -> String {
    let stripped = strip_think_blocks_keep_layout(response_text);
    match function_blocks_to_json(&stripped, tool_catalog()) {
        Some(translated) => translated,
        None => stripped,
    }
}

/// Every call in `answer` (see [`answer_text`]), each with the byte offset it
/// starts at, in the order they appear.
///
/// Text order, not pass order: the results of a round are paired with the
/// calls that asked for them by position — the GUI puts the n-th result on the
/// n-th call card — so a call recovered by a later, looser pass must still take
/// its place among the rest.
pub(crate) fn calls_in_answer(answer: &str) -> Vec<(usize, ToolCall)> {
    use regex::Regex;
    use std::sync::OnceLock;
    let response_text = answer;
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
        let (full_start, full_end) = cap.get(0).map_or((0, 0), |m| (m.start(), m.end()));
        let json = match cap.get(1) {
            Some(m) => m,
            None => continue,
        };
        if let Ok(raw) = serde_json::from_str::<RawCall>(json.as_str()) {
            if !raw.name.is_empty() {
                let arguments = raw.args();
                out.push((
                    full_start,
                    ToolCall {
                        name: raw.name,
                        arguments,
                    },
                ));
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
                out.push((
                    json.start(),
                    ToolCall {
                        name: raw.name,
                        arguments,
                    },
                ));
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
                out.push((
                    start,
                    ToolCall {
                        name: raw.name,
                        arguments,
                    },
                ));
            }
        }
    }
    // Stable, so two calls reported at one offset keep their pass order.
    out.sort_by_key(|&(at, _)| at);
    out
}

/// Parse one call object's JSON. `Ok(None)` when it parses but names no tool.
pub(crate) fn parse_call(json: &str) -> Result<Option<ToolCall>, serde_json::Error> {
    let raw = serde_json::from_str::<RawCall>(json)?;
    Ok((!raw.name.is_empty()).then(|| ToolCall {
        arguments: raw.args(),
        name: raw.name,
    }))
}

#[derive(Deserialize)]
pub(super) struct RawCall {
    /// `function` and `tool` are accepted as aliases for `name`. Qwen3-30B-A3B
    /// occasionally emits `{"function":"<tool>", "arguments":{...}}` instead
    /// of `{"name":"<tool>", ...}`; Qwen3.6-35B-A3B, asked outright to emit a
    /// call, produces `{"tool": "datetime", "arguments": {}}` — right shape,
    /// right tool, wrong key, and without the alias the whole call is dropped.
    /// Tolerating them costs nothing (the bare pass still requires the name to
    /// resolve in the registry) and recovers an otherwise-lost tool call.
    #[serde(alias = "function", alias = "tool")]
    pub(super) name: String,
    #[serde(default)]
    pub(super) arguments: Option<Value>,
    /// Qwen3-30B-A3B sometimes emits `"parameters"` — the tool *schema* key —
    /// in place of `"arguments"` for the call values (seen when it drops the
    /// `<tool_call>` wrapper too).  Accepted as a fallback so the args aren't lost.
    #[serde(default)]
    pub(super) parameters: Option<Value>,
    /// Present only on a tool *definition* (`{"name","description","parameters"}`),
    /// never on a call.  Lets the bare pass reject a definition echo even though it
    /// also carries a top-level `"parameters"`.
    #[serde(default)]
    pub(super) description: Option<Value>,
}

impl RawCall {
    /// The call arguments: `"arguments"` if present, else the `"parameters"`
    /// fallback, else null.
    pub(super) fn args(&self) -> Value {
        self.arguments
            .clone()
            .or_else(|| self.parameters.clone())
            .unwrap_or(Value::Null)
    }
    /// Whether the object carries call values under either key.
    pub(super) fn has_args(&self) -> bool {
        self.arguments.is_some() || self.parameters.is_some()
    }
    /// A tool *definition* echo carries a top-level `description`; a call never
    /// does (a param *named* `description` lives inside `parameters`, not here).
    pub(super) fn looks_like_definition(&self) -> bool {
        self.description.is_some()
    }
}
