//! Runs a round of parsed tool calls and formats each result for the model's
//! next turn.
//!
//! A round runs its calls in order and pairs each with its result. A refusal is
//! itself a result the model can read, so the round keeps its shape. On a
//! resumed turn, a call that cannot be re-issued is answered with an error
//! instead of being run a second time. Results are wrapped in
//! `<tool_response>` blocks; when there is more than one, each block names the
//! call it answers, while a single result is left in the plain form the earlier
//! single-call prefills used.

use super::*;

/// One executed tool's result, ready to be wrapped in `<tool_response>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    /// The call this result answers. Its `name` is what labels the result's
    /// `<tool_response>` block when a turn made several calls — the text-side
    /// equivalent of OpenAI's `tool_call_id`. See [`format_tool_responses`].
    pub call: ToolCall,
    /// The JSON value produced by [`zend_tools::runner::run`].  On
    /// success this is the tool's typed `Response`; on failure it's a
    /// `{"error": "<code>", "detail": "..."}` shape.  The orchestrator
    /// passes both back to the model verbatim so it can react to
    /// errors.
    pub response: Value,
}

// ── Tool dispatch ────────────────────────────────────────────────────────────

/// `files` without the memory's repository (`crate::memory_paths`): what a
/// caller who is not an admin reads the workspace through.
pub fn memory_hidden(files: &RepoFiles) -> RepoFiles {
    files.without(memory_paths::memory_repo())
}

/// Run one parsed tool call against the registry.  Always returns a
/// JSON value the model can consume — successful tools return their
/// typed response; a refused call carries what the model needs to correct
/// it ([`crate::tool_guidance`]).
pub fn run_tool(ctx: &ToolContext, call: &ToolCall) -> Value {
    match registry::find(&call.name) {
        // An alias's name fixes some arguments; a call that contradicts it is
        // refused before it runs ([`alias_pins`]).
        Some(t) => match alias_pins::apply(&call.name, &call.arguments) {
            Ok(args) => tool_guidance::with_guidance(t.name, t.call(ctx, &args)),
            Err(refusal) => tool_guidance::with_guidance(t.name, refusal),
        },
        None => tool_guidance::unknown_tool(&call.name, ctx.grants()),
    }
}

/// Why a round of tool calls is being dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// The turn is running now, and every call it asks for is run.
    Live,
    /// The turn is being resumed after a restart lost the results of the calls
    /// its assistant half asked for. A call that cannot be re-issued
    /// ([`zend_tools::Tool::replay`]) is answered with an error instead of
    /// being run a second time.
    Resumed,
}

/// The answer a resumed turn gets for a call that must not run twice.
///
/// Shaped like every other tool failure — an `{"error", "detail"}` object — so
/// the model reads it the way it reads the rest, and can ask for the call
/// again itself if it still wants it.
pub(super) fn not_replayed(name: &str) -> Value {
    serde_json::json!({
        "error": "not_replayed_after_restart",
        "detail": format!(
            "zend restarted between this {name} call and its result. Running it again \
             could repeat an effect that already happened, so it was not run. Ask for \
             it again if you still need it."
        ),
    })
}

/// Dispatch every parsed tool call as a round ([`tool_round::run`]): in order,
/// pairing each call with its result.
///
/// On [`Dispatch::Resumed`] each call is put to its tool first: the ones that
/// leave what they left the first time are run, and the rest are answered with
/// [`not_replayed`]. The turn continues either way — a refusal is a result the
/// model can read, not a hole in the round.
pub fn run_tool_calls(
    ctx: &ToolContext,
    calls: Vec<ToolCall>,
    dispatch: Dispatch,
) -> Vec<ToolResult> {
    let steps = calls
        .into_iter()
        .map(|c| {
            if dispatch == Dispatch::Resumed && replay(&c.name, &c.arguments) == Replay::Unsafe {
                Step::Withheld(ToolResult {
                    response: not_replayed(&c.name),
                    call: c,
                })
            } else {
                Step::Run(c)
            }
        })
        .collect();
    tool_round::run(ctx, steps)
}

/// Format executed tool results into a single chained-turn string,
/// suitable to pass as the user message of the next conversation turn.
///
/// Format: one `<tool_response>{json}</tool_response>` block per result,
/// separated by newlines.  This matches the Hermes spec — the model
/// reads each block and continues its prior reasoning.
pub fn format_tool_responses(results: &[ToolResult]) -> TurnText {
    format_tool_blocks(results).0
}

/// [`format_tool_responses`], with each result's block as a byte range of the
/// text — `<tool_response>` through `</tool_response>`, the span a served
/// read's recurrent record is applied over
/// (`docs/archived/recurrent_reply_calibration.md` §7).
pub fn format_tool_blocks(results: &[ToolResult]) -> (TurnText, Vec<Range<usize>>) {
    let mut out = TurnText::default();
    let mut at = 0usize;
    let mut blocks = Vec::with_capacity(results.len());
    // A turn may make several calls, and the results come back as a run of
    // sibling blocks — so each one has to say which call it answers. This is
    // what OpenAI's `tool_call_id` is for; here the pairing rides in the text
    // the model is already reading, like the read header's line range and for
    // the same reason: a structured field beside a rendered string has to be
    // correlated with it, whereas a first line is read in passing.
    //
    // **Only when there is more than one.** A single-call round is the common
    // case for both a live turn and `code_reading`'s hidden per-file
    // conversation (which now runs this same function for its own real
    // `file_read` calls, not a synthetic prefill) — so the header appears
    // precisely when order alone stops being unambiguous, and it must, because
    // a failed call returns an error envelope rather than the shape its position
    // would imply.
    let label =
        |i: usize, r: &ToolResult| format!("[{}/{} {}]\n", i + 1, results.len(), r.call.name);
    for (i, r) in results.iter().enumerate() {
        let body = match &r.response {
            // A string result is already rendered for the model — placed in the
            // block verbatim rather than JSON-encoded. `file_read` returns a
            // numbered, fenced excerpt this way; encoding it would collapse the
            // source to one line of `\n` escapes.
            Value::String(rendered) => rendered.clone(),
            other => serde_json::to_string(other)
                .unwrap_or_else(|_| "{\"error\":\"internal_error\"}".to_string()),
        };
        // The wrapper is markup; what the tool returned is literal, so a file
        // that quotes a chat tag reaches the model as its text.
        //
        // The label is LITERAL, though it is this layer's framing rather than
        // the tool's output. [`tool_round_text`] rebuilds a stored round for
        // replay by taking everything between the tags as literal — it cannot
        // tell a label from a body — so a markup label would replay as a
        // different piece sequence from the one submitted. Literal costs nothing
        // here: `[n/total name]` holds a registry name and digits, never a
        // special-token string, so the two tokenize identically.
        let label = if results.len() > 1 {
            label(i, r)
        } else {
            String::new()
        };
        let block = "<tool_response>".len() + label.len() + body.len() + "</tool_response>".len();
        blocks.push(at..at + block);
        at += block + 1;
        out = out
            .then_markup("<tool_response>")
            .then_literal(label)
            .then_literal(body)
            .then_markup("</tool_response>\n");
    }
    (out, blocks)
}

/// A tool round's user half rebuilt from its stored text — the inverse of
/// [`format_tool_responses`].
///
/// The substrate keeps a turn's text, not its pieces, and the pieces decide the
/// tokens: a result's body is literal (a quoted chat tag stays text) while its
/// wrapper is markup. Re-submitting a recorded tool round therefore needs the
/// split back. Text that is not a run of `<tool_response>` blocks is ordinary
/// user text, which the chat submits as markup.
pub fn tool_round_text(text: &str) -> TurnText {
    match tool_round_blocks(text) {
        Some(blocks) => blocks.iter().fold(TurnText::default(), |out, b| {
            out.then_markup(&text[b.block.start..b.body.start])
                .then_literal(&text[b.body.clone()])
                .then_markup(&text[b.body.end..b.next])
        }),
        None => TurnText::markup(text),
    }
}

/// One `<tool_response>` block of a stored round, by bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundBlock {
    /// `<tool_response>` through `</tool_response>`.
    pub block: Range<usize>,
    /// What the tool returned, between the tags.
    pub body: Range<usize>,
    /// Where the next block starts: past this one's closing newline.
    pub next: usize,
}

/// The blocks of a stored tool round, or `None` for text that is not a run of
/// `<tool_response>` blocks.
pub fn tool_round_blocks(text: &str) -> Option<Vec<RoundBlock>> {
    const OPEN: &str = "<tool_response>";
    const CLOSE: &str = "</tool_response>";
    if !text.starts_with(OPEN) {
        return None;
    }
    let mut blocks = Vec::new();
    let mut at = 0usize;
    while at < text.len() {
        let rest = &text[at..];
        rest.strip_prefix(OPEN)?;
        let body = at + OPEN.len();
        let end = body + text[body..].find(CLOSE)?;
        let close_end = end + CLOSE.len();
        // Each block is closed by `</tool_response>\n`; the last may have lost
        // its newline to a trim.
        let next = if text[close_end..].starts_with('\n') {
            close_end + 1
        } else {
            close_end
        };
        blocks.push(RoundBlock {
            block: at..close_end,
            body: body..end,
            next,
        });
        at = next;
    }
    Some(blocks)
}
