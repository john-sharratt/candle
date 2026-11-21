//! The prompt, tool and turn grammar that make a file's hidden conversation read
//! the whole file and then answer.
//!
//! The opening prompt names the file's length and the exact number of calls. A
//! model left to work out how many calls cover a file can spend its whole decode
//! budget on that decision, so the arithmetic is settled in the prompt. Each
//! reply makes one `file_read` call, so each round places one page in the
//! projection. The first turn begins inside its call, and thinking is off in both
//! the prompt and the selection, so the reading starts with a call rather than a
//! choice.

use super::*;

/// The tools a file's reading may call: `file_read`, as the Restricted catalog
/// declares it, and nothing else.
pub fn reading_tools() -> Vec<ToolSpec> {
    catalog_for(ToolMode::Restricted)
        .into_iter()
        .filter(|spec| spec.name == FILE_READ_TOOL)
        .collect()
}

/// Real `file_read` rounds a reading of `pages` pages may run before it is made
/// to answer on what it has read: one a page, so a file of any length reads to
/// its end, and a reading that re-reads a page runs out of rounds rather than
/// running on.
pub(super) fn read_rounds(pages: usize) -> usize {
    pages.max(1)
}

/// The opening instruction for a file's hidden conversation. Names the file
/// and the one tool available, and REQUIRES reading it before answering —
/// left as "how much is enough", a model would sometimes skip the tool
/// entirely and guess a summary from the filename alone (measured: a
/// one-line `CHANGELOG.md` "summary" with no `file_read` call at all). "This
/// file, and only this file" also heads off a model wandering into whatever
/// else the workspace prompt might make it curious about.
///
/// **It names the file's LENGTH, and that is not a nicety.** Naming only the
/// per-call cap leaves the model to guess how many calls cover the file, and a
/// guess is a decision it can spend its whole decode budget failing to make: a
/// README read deliberated over whether the file was under the cap, whether a
/// read past EOF would error, and whether to risk it — until the token budget
/// cut it off mid-word, before any `file_read` call was emitted. It produced no
/// tool call, so no coupling and no second turn, and the chain stood in the
/// corpus as an assistant that deliberates and answers nothing. With the length
/// given, the arithmetic is settled before the model starts.
///
/// **It asks for ONE call per reply, and that is a placement decision.** The
/// projection places a reading by its tool rounds — one call and its result —
/// and a round of one page is placed or dropped at that page's cost, so a
/// question that reads two pages of a 48K-token file places two pages, not the
/// file (`docs/hit_driven_projection.md` §8.2). More, shorter turns are the
/// price. Naming every range in the prompt is what lets the model walk them in
/// order rather than discovering the file's length from the first read's
/// closing line.
///
/// **It names the ranges, not a count.** The ranges hold whole parts, so where
/// each ends depends on the file's content, not on its length; a model told
/// "lines 1-200" would ask for the 200 lines the read cannot hold whole, and
/// the next read would start where none ends. [`windows`] gives the ranges
/// reads of whole parts tile the file in, and the prompt hands them over.
///
/// A model losing track of its own earlier rounds several turns into a long
/// file was once worked around here, with prose telling it no further
/// question was coming — a symptom this prompt cannot fix, because the
/// symptom wasn't the prompt: this conversation's construction wasn't
/// forking `base_conv` the way a live turn does, and its layer's projection
/// window was sized for the old pre-carved scope excerpts, not a real
/// `file_read` response several times that size (see `session.rs`'s
/// `ingest_bases`, `resolver.rs`'s `score_belief_groups` target exemption,
/// and `code_reading`'s `window` in `projection.yaml`). With those fixed,
/// the model correctly recalls its own earlier rounds without being told to.
///
/// `key` is workspace-relative; the prompt names the repository and the path
/// inside it separately, the two arguments the call it asks for takes. `ranges`
/// are the `lines` ranges `file_read` reads the file in ([`windows`]).
pub(crate) fn opening_prompt(key: &str, lines: usize, ranges: &[(usize, usize)]) -> String {
    let (repo, path) = split(key);
    // Never zero calls: an empty file still reads as one range.
    let calls = ranges.len().max(1);
    let named = match ranges.is_empty() {
        true => "lines=1".to_string(),
        false => ranges
            .iter()
            .map(|&r| format!("lines={}", range_value(r)))
            .collect::<Vec<_>>()
            .join(", "),
    };
    // A reply holds one call — the reading's grammar ends the turn after it
    // (`READ_CALLS_PER_REPLY`) — so each range is its own reply, in order.
    format!(
        "Read the entire contents of `{path}` in the `{repo}` repository — and only \
         this file — using {FILE_READ_TOOL}. The file is {lines} lines long and \
         reads in {calls} range(s) of whole parts, at most {PAGE_LINES} lines each — \
         so it takes exactly {calls} {FILE_READ_TOOL} call(s), each passing one range \
         as `lines`, ONE call per reply, in this order: {named}. Each reply's result \
         comes back to you before your next reply. Once you've read the whole thing, \
         summarize what it contains: its purpose, its main structures or functions, \
         and how it fits into the codebase."
    )
}

/// The projection selection every turn of a file's conversation runs under:
/// tools on, forced to `file_read`, and thinking OFF.
///
/// Thinking is off in both halves, as `api::chat::dial_selection` sets a
/// thinking-off turn: the prompt carries the no-think switch and the effort
/// dial reads `off`, matching the `ThinkMode::Off` steering the turn's triggers
/// carry (`RefreshContext::think_triggers`). Set explicitly rather than left to
/// the schema's defaults, which are a dialogue turn's.
pub(super) fn file_selection() -> SelectionState {
    let mut selection = SelectionState::new();
    selection.set_optional(TOOLS_ENABLED_SELECTOR, OptionalState::Present);
    selection.select(FORCE_TOOL_SELECTOR, FILE_READ_TOOL);
    selection.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    selection.select("thinking_effort", "off");
    selection
}

/// Whether `msg` is a reading's own failure — it answered nothing, or read
/// some other path — rather than the engine's.
pub(super) fn is_reading_failure(msg: &str) -> bool {
    msg.contains(READ_NOTHING) || msg.contains(READ_ELSEWHERE)
}

/// The file a round's `file_read` found missing, as `repo:path`, when one did.
///
/// **A reading that read no file failed.** It reads the one file its prompt
/// names, so a `not_found` means a call named something else — and a reading
/// that went on from there searched for the file until its round cap and
/// answered about the search: committed as finished, it superseded the reading
/// that had read the file. On 2026-11-11 about 115 readings went that way,
/// `candle-conversation/src/turn_layout.rs` read as `src/turn_layout.rs` nine
/// times over. Failed, the attempt goes and the generation before it stays.
pub(super) fn read_not_found(results: &[ToolResult]) -> Option<String> {
    results
        .iter()
        .filter(|r| r.call.name == FILE_READ_TOOL)
        .find(|r| r.response.get("error").and_then(Value::as_str) == Some("not_found"))
        .map(|r| {
            let arg = |k: &str| {
                r.call
                    .arguments
                    .get(k)
                    .and_then(Value::as_str)
                    .unwrap_or("?")
            };
            format!("{}:{}", arg("repo"), arg("path"))
        })
}

/// A file reading's opening call ([`RefreshContext::read_opening`]): the call
/// loop over `file_read` as the Restricted catalog declares it, with the marker
/// to prefill. `None` when the tokenizer has no `<tool_call>` token or the
/// catalog offers no `file_read`, as no reading can call it then either.
pub fn read_opening(engine: &ConversationEngine) -> anyhow::Result<Option<OpeningCall>> {
    let tools = reading_tools();
    if tools.is_empty() {
        return Ok(None);
    }
    engine
        .opening_call(&tools, READ_CALLS_PER_REPLY)
        .map_err(|e| anyhow::anyhow!("file reading's opening call: {e}"))
}

/// The options a file's first turn adds to its turn's: begin inside
/// `opening`'s call loop, its marker prefilled, so the turn opens with a
/// `file_read` call rather than choosing whether to, its values held to `held`
/// — the reading's own file. Only the first turn: the turns after it carry the
/// reads' results, and the reading may answer once it holds the file.
pub(super) fn opening_options(
    options: TurnOptions,
    opening: &OpeningCall,
    held: Option<&Arc<HeldValues>>,
) -> TurnOptions {
    let grammar = TurnGrammar::of(Arc::clone(&opening.tree));
    TurnOptions {
        assistant_prefill: Some(opening.marker.clone()),
        turn_grammar: Some(match held {
            Some(held) => grammar.holding(Arc::clone(held)),
            None => grammar,
        }),
        ..options
    }
}

/// Drive `conv` as a REAL, hidden tool-using conversation asking it to read and
/// describe `path` — exactly the machinery a live chat turn uses
/// (`submit_turn_with_options` → real decode → real tool dispatch → repeat, see
/// `zend::session::run_inference_stream`), just run synchronously on this
/// worker thread and never shown to a user. [`opening_prompt`] requires the
/// model to read the whole file — one or more real `file_read` calls, ranged
/// if the file is long — before it answers; whatever it says once it stops
/// calling tools IS the file's summary — there is no separate summary-decode
/// step to run afterward.
///
/// The first turn begins inside `opening`'s call loop when given, its values
/// held to the second of the pair ([`opening_options`]): left to choose, a
/// reading has ended on its first token with no call and no answer.
pub(super) fn run_file_conversation(
    conv: &mut Sequence,
    path: &str,
    lines: usize,
    ranges: &[(usize, usize)],
    triggers: &Arc<TriggerRegistry>,
    opening: Option<(&OpeningCall, Option<&Arc<HeldValues>>)>,
    tool_ctx: &Arc<ToolContext>,
) -> anyhow::Result<String> {
    let tags = vec!["code".to_string(), path.to_string()];
    let selection = file_selection();

    let mut current_message: TurnText = TurnText::from(opening_prompt(path, lines, ranges));
    let mut closing = false;
    let rounds = read_rounds(ranges.len());
    for round in 0..=rounds {
        if crate::ingest_cancel::cancelled() {
            anyhow::bail!("shutdown cancelled mid-file");
        }
        if round == rounds {
            tracing::warn!(
                target: "zend::code_read::ingest",
                path,
                "hit the read-round cap — forcing an answer on what has been read so far",
            );
            closing = true;
        }
        // The closing turn is forced to answer in prose: ban the tool-call
        // opener so a model still trying to read more cannot open another
        // call, mirroring the live dialogue loop's own repeat-guard closing
        // round (`session::run_inference_stream`'s `closing` handling).
        let sampling = if closing {
            let mut s = conv.default_sampling();
            let open = s.tool_call_open_token_id;
            if open >= 0 {
                s.banned_tokens.push(open);
            }
            Some(s)
        } else {
            None
        };
        let options = TurnOptions {
            max_tokens: Some(FILE_TURN_MAX_TOKENS),
            sampling,
            selection: selection.clone(),
            triggers: Arc::clone(triggers),
            tags: tags.clone(),
            ..Default::default()
        };
        let options = match opening {
            Some((opening, held)) if round == 0 && !closing => {
                opening_options(options, opening, held)
            }
            _ => options,
        };
        let handle = conv
            .submit_turn_with_options(current_message, options)
            .map_err(|e| anyhow::anyhow!("submit_turn: {e}"))?;
        let resp = handle
            .wait_cancellable(crate::ingest_cancel::token())
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        let steps = tool_round::plan(&resp.answer);
        let call_idx = resp.seal.as_ref().and_then(|s| s.turn_index);
        conv.finish_turn(handle, &resp)
            .map_err(|e| anyhow::anyhow!("finish_turn: {e}"))?;

        if steps.is_empty() || closing {
            return Ok(resp.text);
        }

        let results = tool_round::run(tool_ctx, steps);
        if let Some(missing) = read_not_found(&results) {
            // What the round decoded, as written: whether its call was written
            // under the reading's holds is read off its text.
            let answer = resp.answer.text();
            let call = answer
                .find("<tool_call>")
                .map_or(&answer[..], |at| &answer[at..]);
            tracing::warn!(
                target: "zend::code_read::ingest",
                path,
                round,
                opened = round == 0 && opening.is_some(),
                held = opening.is_some_and(|(_, held)| held.is_some()),
                call = %call.chars().take(300).collect::<String>(),
                "a reading named a file other than its own",
            );
            anyhow::bail!("{READ_ELSEWHERE} {missing}: it read something other than `{path}`");
        }
        let response = format_tool_responses(&results);
        // A tool round that produces no response text must NOT spawn a
        // follow-up turn — see `session::run_inference_stream`'s identical
        // guard. The answer already decoded for this turn stands.
        if response.is_blank() {
            return Ok(resp.text);
        }
        // The round-trip is now certain — the tool returned real output and the
        // follow-up turn is guaranteed to be submitted below — so couple the
        // just-sealed call turn to it, by its OWN sealed index (never "the last
        // turn": the async summariser could append a summary turn in this
        // window).
        if let Some(idx) = call_idx {
            conv.couple_turn(idx)
                .map_err(|e| anyhow::anyhow!("couple_turn: {e}"))?;
        }
        current_message = response;
    }
    unreachable!("the closing round after the last page always returns")
}
