//! Reads a reflection's answer back out of the call the grammar forced, and
//! builds the grammar and exchange that a reflection runs under.
//!
//! The answer is taken from whichever call shape the dialect wrote. A raw newline
//! inside a JSON string is repaired first, so a paragraph-length brief is not
//! lost to a parse error. A refusal goes back in-band inside the tool-response
//! wrapper, which is the form the model reads as what came back. The grammar is
//! the engine's own tool-call tree, so a value closes on its terminator and an end
//! token inside the value is written by the tree instead. The think block is
//! closed up front, and a second block inside the answer is covered by the
//! sampling backstop.

use super::*;

/// Strip the scaffolding a reflection is not supposed to have emitted.
///
/// Two things, and both are guards rather than fixes — the prompt is where each
/// is actually addressed, and this is what catches the checkpoint doing it
/// anyway:
///
/// - **A leading `</think>`.** [`think_off`] injects the close, which is grammar
///   scaffolding rather than something the character said, and it arrives at the
///   head of the text.
/// - **A tool-call envelope.** A reflection has no acts and is told so twice,
///   but this checkpoint's prior toward calling something is strong enough to
///   beat both: asked what comes to it while standing still, it wrapped a
///   perfectly good paragraph in `<tool_call><function=reflect>`. The prose
///   inside is the answer, so it is taken out rather than thrown away — a
///   reflection that reached the character as raw markup would be worse than one
///   that lost its wrapper.
pub(crate) fn plain_prose(raw: &str) -> String {
    let mut s = raw.trim();
    if let Some(rest) = s.strip_prefix("</think>") {
        s = rest.trim();
    }
    // Everything inside the parameter bodies, in order, with the markup gone.
    // Deliberately not a parser: this is salvage, and a half-formed envelope is
    // exactly the case a strict one would drop on the floor.
    if s.contains("<tool_call>") {
        let mut out = String::new();
        let mut rest = s;
        while let Some(open) = rest.find('>') {
            let after = &rest[open + 1..];
            let end = after.find('<').unwrap_or(after.len());
            let body = after[..end].trim();
            if !body.is_empty() {
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                out.push_str(body);
            }
            rest = &after[end..];
            if rest.is_empty() {
                break;
            }
        }
        if !out.is_empty() {
            return out;
        }
    }
    s.to_string()
}

/// What the reflection asks for, as a vocabulary the frame can render.
///
/// **Deliberately not in [`crate::engine::tools::CATALOG`].** That catalog is
/// what a character may *do* — every entry reaches the world and is offered in
/// the acting grammar. These reach nothing: they are the shape an answer takes
/// inside a conversation the world is not reading. Putting them in the world
/// catalog would offer a character the chance to `dream` at somebody, and
/// `act::parse` would start accepting them on a live turn.
///
/// The cost of keeping them separate is that this module parses its own
/// answers — see [`Answered::parse`] — which is a dozen lines against a
/// vocabulary of two.
///
/// They are installed in the schema's `tools` collection all the same — see
/// `tools::install` — because that collection is something to read, not a
/// vocabulary: a member reaches a turn only when the turn names it, and only
/// this module names these two.
pub const ASKED: &[Tool] = &[REFLECTION, DREAM];

/// How many times one answer may be refused before its best form is taken.
///
/// **Uncapped refusal is the one way this gets genuinely expensive.** Each round
/// is a whole decode inside a blocking act, so a loop that argues until it wins
/// can spend a character's afternoon on one dream. Two rounds is enough to fix a
/// length — the third is a decode that is not going to converge.
pub(super) const MAX_REFUSALS: usize = 2;

/// What the character reports having thought — the line that crosses back.
///
/// Past tense on purpose: the character is not being asked to think on demand,
/// it is being asked what it *has* come to think. `think(…)` reads as an
/// instruction to perform; `reflection(said)` reads as a report of something
/// already there, which is the register `docs/reflection_and_dreams.md` §6
/// requires of the one line that reaches the character's own context.
pub(super) const REFLECTION: Tool = Tool {
    name: "reflection",
    at: &[],
    category: "Attention",
    plane: Plane::World,
    availability: Availability::Always,
    description: "What you have come to think — one sentence, first person, in your own voice, \
                  said as how it is for you rather than as a fact. Not a plan and not an \
                  explanation of itself.",
    params: &[Param {
        name: "said",
        ty: "string",
        required: true,
        description: "One sentence, framed as what you feel, what it seems like, or what you \
                      find yourself wanting — \"I feel like…\", \"it seems as if…\", \"I find \
                      myself…\". Never a statement of what you or anything is.",
    }],
    examples: &[],
};

/// The dream brief, as a call with two named parts.
pub(super) const DREAM: Tool = Tool {
    name: "dream",
    at: &[],
    category: "Dreaming",
    plane: Plane::World,
    availability: Availability::Always,
    description: "The dream you are asked for. `brief` is the dream itself, present tense, the \
                  dreamer addressed as \"you\". `assumption` names the one thing that is untrue \
                  in it, on its own line.",
    // **The dream first, the assumption after.** The grammar emits the fields in
    // this order, and with `assumption` first the checkpoint began the dream the
    // moment the first field opened: measured, an imagined dream of building in
    // "the spaces between worlds" went whole into `assumption` — seventeen
    // sentences — with a three-sentence summary of it left for `brief`, refused at
    // every round and never dreamt. Naming what a dream suspended is easy once
    // the dream is written; writing the dream inside a field meant for one line
    // is what it did instead.
    params: &[
        Param {
            name: "brief",
            ty: "string",
            required: true,
            description: "The dream, about two hundred words.",
        },
        Param {
            name: "assumption",
            ty: "string",
            required: true,
            description: "The single thing that has stopped being true in it. One short line.",
        },
    ],
    examples: &[],
};

/// Why an answer was refused, said in words the model can act on.
///
/// `None` when it passes. The message goes back inside a `<tool_response>`,
/// which is the one wrapper this checkpoint reads as *this is what came back* —
/// so a refusal is in-band and the model already knows what to do with it. The
/// acting frame trains exactly this: *"An act can be refused, and a refusal says
/// why. Do not call the same act again as though you had not been told."*
///
/// Only genuine failures come through here — an empty field, or a length outside
/// the window. Editorial guidance (be stranger, hold the room still) is a
/// separate prose turn, because a `<tool_response>` that says *failed* about a
/// call that succeeded teaches the model that success is arbitrary and blunts
/// the signal for the case where it is real.
pub(super) fn refusal(words: usize, min: usize, max: usize, what: &str) -> Option<String> {
    if words == 0 {
        return Some(format!("failed: the {what} came back empty. Write it."));
    }
    if words < min {
        return Some(format!(
            "failed: the {what} is about {words} words and needs to be about {min}. It is not \
             long enough to be a dream yet — write it again, fuller."
        ));
    }
    if words > max {
        return Some(format!(
            "failed: the {what} is about {words} words and must be under about {max}. Write it \
             again, shorter."
        ));
    }
    None
}

/// Roughly how many words a value is, for [`refusal`].
pub(super) fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

/// The grammar one answer is held to — the engine's own tool-call tree.
///
/// **Every format failure this conversation had was the cost of asking for prose
/// in a bespoke layout.** Labels the decode dropped, paragraphs it cut short, a
/// header it invented (`DREAM:`) when the opener gave it too short a runway —
/// all of it went away by answering in the shape the checkpoint is actually
/// tuned for, which is a named call with named arguments.
///
/// Two properties of `compile_tool_call_tree` matter here and neither was true
/// of the hand-rolled tree it replaces:
///
/// - **A value ends on its terminator, not on an EOS sample.** The parameter
///   span closes on whatever the dialect's call shape uses — a JSON string's
///   closing quote, or an element's `</parameter>` — so the model closes the
///   field by writing it, and an EOS inside the value is intercepted and the
///   close written by the tree. That is structurally immune to the truncation
///   that had been cutting briefs off mid-word at 29 tokens — every one of
///   those was EOS winning inside a span whose only other exit was a budget.
/// - **The turn opens inside the envelope**, so `<think>` is unrepresentable
///   without a separate think stencil. A turn carries one grammar and this one
///   already spends it.
pub(super) fn call_tree(
    engine: &Arc<ConversationEngine>,
    env: &ToolCallEnvelope,
    tool: &Tool,
) -> Option<Arc<StencilTree>> {
    let spec = ToolSpec {
        name: tool.name.to_string(),
        params: tool
            .params
            .iter()
            .map(|p| CallParam {
                name: p.name.to_string(),
                ty: ParamType::String,
                required: p.required,
                enum_values: None,
                items: None,
                min_items: 0,
                max_items: None,
                properties: None,
                nullable: false,
                minimum: None,
                maximum: None,
                max_tokens: None,
                requires: Vec::new(),
                shapes: Vec::new(),
                pick: Default::default(),
                pattern: None,
            })
            .collect(),
    };
    let tree = match compile_tool_call_tree(std::slice::from_ref(&spec), env) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("the `{}` answer grammar would not build: {e:#}", tool.name);
            return None;
        }
    };
    match engine.compile_stencil(&tree) {
        Ok(t) => Some(Arc::new(t)),
        Err(err) => {
            tracing::warn!(
                "the `{}` answer grammar would not compile: {err:#}",
                tool.name
            );
            None
        }
    }
}

/// Ask once, refuse in-band until the answer is the right length, take the best.
///
/// The shape of one exchange: a **prose** user turn asks, the assistant answers
/// in a **stencilled call**, and a **`<tool_response>`** either accepts it or
/// says why it failed and asks again. That is the direction the checkpoint is
/// trained in — models emit `<tool_call>` and read `<tool_response>` — and it is
/// why the caller's own turns are prose rather than calls: two calls in a row,
/// one in each direction, is a shape nothing was trained on and reads as the
/// character being asked to answer its own question.
///
/// Returns every turn's raw text in order, so the transcript keeps the refused
/// attempts as well as the one that stood.
pub(super) struct Exchange<'a> {
    pub(super) response_open: &'a str,
    pub(super) response_close: &'a str,
}

impl Exchange<'_> {
    /// Wrap a verdict the way the model reads verdicts.
    ///
    /// **The markers carry their own newlines.** `tool_response_open` is
    /// `"<tool_response>\n"` and the close is `"</tool_response>\n"`, so adding
    /// one after the open put a blank line between the tag and the verdict —
    /// subtly wrong in a wrapper whose whole value is that the model recognises
    /// it exactly.
    pub(super) fn respond(&self, body: &str) -> String {
        format!("{}{}\n{}", self.response_open, body, self.response_close)
    }
}

/// One answer, pulled back out of the call the grammar forced.
///
/// A dozen lines rather than [`crate::engine::act::parse`], because that parser
/// resolves a name against the world catalog and refuses anything not in it —
/// and these two are deliberately not in it. See [`ASKED`].
///
/// **Either call shape.** Which one the grammar forces is the dialect's to say —
/// a function element or a JSON block — so the answer is read from whichever it
/// wrote rather than assuming one: an element-only reader under JSON calls
/// found nothing in any answer, and refused every reflection for being empty.
pub(super) fn field(answer: &str, name: &str) -> Option<String> {
    let v = element(answer, name).or_else(|| json_argument(answer, name))?;
    let v = v.trim();
    (!v.is_empty()).then(|| v.to_string())
}

/// A `<parameter=name>` element's body — the function-block shape.
pub(super) fn element(answer: &str, name: &str) -> Option<String> {
    let open = format!("<parameter={name}>");
    let at = answer.find(&open)? + open.len();
    let rest = &answer[at..];
    let end = rest.find("</parameter>").unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// `arguments.name` of the first JSON object in the answer — the JSON-block
/// shape. Only the first value is read, so anything after the call is ignored
/// rather than failing the parse.
///
/// **Repaired before it is read.** A string span lets a raw newline through,
/// which is not valid JSON, and a brief is prose that runs to paragraphs — so
/// unrepaired, `serde` refused the whole call, the answer read as having no
/// assumption and no brief, and one run lost its dream on every pass. The
/// repair is the one the acting parser already applies, for the same reason —
/// see [`escape_control_in_strings`].
pub(super) fn json_argument(answer: &str, name: &str) -> Option<String> {
    let start = answer.find('{')?;
    let repaired = escape_control_in_strings(&answer[start..]);
    let call: serde_json::Value = serde_json::Deserializer::from_str(&repaired)
        .into_iter::<serde_json::Value>()
        .next()?
        .ok()?;
    call.get("arguments")?
        .get(name)?
        .as_str()
        .map(str::to_string)
}

/// A grammar that opens the turn with the reasoning block already closed, and
/// constrains nothing after that.
///
/// **This is half of the suppression and not the whole of it.** `ThinkMode::Off`
/// compiles to *inject `</think>`, then `End`* — so the turn begins past the
/// block, which is the convention the Qwen3.5 family itself uses. But `End` is
/// the end of the tree, and past it nothing is masked: a `<think>` the model
/// decodes **inside its answer** is outside the grammar entirely.
///
/// An earlier version of this comment claimed a block "cannot be written because
/// there is nowhere in the tree to write one". That is true of the block the turn
/// opens with and false of a second one, and the difference is what hid the bug:
/// a brief came back as a full `Thinking Process:` deliberation which ran long
/// enough to rehearse its own answer in `ASSUMPTION SUSPENDED:` / `BRIEF:` form,
/// and the brief parser harvested the rehearsal. The other half — a forced
/// segment close after one token — is set on the sampling config in
/// [`Reflect::run`], and it is a mechanism there rather than a backstop.
///
/// Measured before either existed: given `turn_grammar: None`, question two
/// closed its block immediately anyway — its prompt ends in a rigid output
/// format, which anchors it — while question one, which is an open question,
/// ran the block to five thousand words of the model deliberating about which
/// act to call, in the voice of a character it had invented. The prompt was
/// correct throughout; nothing was masking the token.
///
/// `None` when the checkpoint's tokenizer has no single `<think>` token, in
/// which case the turn free-decodes exactly as it did before.
pub(crate) fn think_off(
    engine: &Arc<ConversationEngine>,
    cfg: &SequenceConfig,
) -> Option<Arc<StencilTree>> {
    let tok = engine.tokenizer();
    let steer = ThinkSteerEnvelope {
        think_open: tok.token_to_id("<think>")?,
        think_close: tok.token_to_id("</think>")?,
        // A span ends on `</think>` or on the turn terminator, and a terminator
        // the tree does not know is one it cannot end on.
        eos: tok
            .token_to_id(cfg.dialect.assistant_end.trim_end())
            .unwrap_or(0),
        after_close: "",
    };
    // Every dial yields a tree, `Off` included — suppression is structural, so
    // there is no `None` to propagate here.
    let spec = compile_think_tree(ThinkMode::Off, &steer);
    match engine.compile_stencil(&spec) {
        Ok(t) => Some(Arc::new(t)),
        Err(e) => {
            tracing::warn!(
                "the think-suppression grammar would not compile: {e:#} — this reflection \
                 free-decodes, and may spend its whole budget reasoning"
            );
            None
        }
    }
}
