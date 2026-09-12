//! One character, stopped, thinking — and the dream brief that comes out of it.
//!
//! # Two questions and a review, in one throwaway conversation
//!
//! A reflection is a **brand-new conversation used once and never persisted**.
//! It is minted, marked transient so nothing it writes reaches cold storage, run
//! for its turns, and tombstoned. Nothing about it survives except two answers,
//! and only the first of those goes back to the character.
//!
//! The turns are: question one, question two, a retry if the brief came out
//! malformed, then the **repair pass** — a fixed review that rewrites the brief
//! whether or not anything was wrong with its shape. That last part reverses a
//! decision `docs/reflection_and_dreams.md` §5 originally took ("one retry and
//! no more"), because the retry answers a malformed brief and the measured
//! failure was a well-formed one that was not a dream.
//!
//! That is what makes the whole arrangement safe rather than a leak. A runaway
//! needs iteration on a persistent state, and there is none: reflection *N* is
//! not conditioned on reflection *N−1*, because reflection *N−1* no longer
//! exists. There is no accumulating surface for a phrasing to rhyme with. See
//! `docs/reflection_and_dreams.md`.
//!
//! # Why the two turns cannot share a rule about invention
//!
//! The first question is about the character's real situation and must stay
//! grounded. The second writes a dream brief and must invent. They run in the
//! same conversation under one system prompt, so the prompt rules on neither —
//! [`prompt::Stance::Reflecting`] is deliberately silent about it — and each
//! question carries its own.
//!
//! # The questions are content and live in the mind
//!
//! Neither turn is written here. Both are authored in the mind's
//! `projection.yaml` under `reflection:`, read once at load, and sent verbatim —
//! see [`crate::engine::schema::reflection`]. This module owns the *machinery*:
//! the transient timeline, the think stencil, the structural checks and the one
//! retry. It owns no wording, because a prompt that exists in two places is a
//! prompt whose two copies diverge, and the one being edited was not the one
//! being sent.
//!
//! # What comes back, and in what register
//!
//! The first answer is returned to the caller and lands in the character's own
//! context, where it competes in the next gather. So it must read as an
//! *inclination* rather than an instruction: "you are thinking of challenging
//! him" is weighed against everything else in the room, and "challenge him" is
//! the one shape that gets obeyed instead of weighed. That is not a style
//! preference — it is the difference between goal-layer content biasing a choice
//! and goal-layer content leaking as a command.
//!
//! # Serial, on purpose
//!
//! This blocks. The engine's standing rule is that a fast clock must never wait
//! on a slow one — but `reflect` is a pause act, so the character has already
//! chosen to stand still, and blocking it costs nothing it had not already
//! spent. The **dream** is the part that stays asynchronous; this is not.

use std::sync::{Arc, Mutex};

use candle_conversation::stencil::{
    compile_think_tree, compile_tool_call_tree, Param as CallParam, ParamType, StencilTree,
    ThinkMode, ThinkSteerEnvelope, ToolCallEnvelope, ToolSpec,
};
use candle_conversation::{ConversationEngine, SequenceConfig, TurnOptions};
use serde::Serialize;

use crate::engine::act::escape_control_in_strings;
use crate::engine::dreams;
use crate::engine::identity;
use crate::engine::mind::Projected;
use crate::engine::prompt::{self, Persona, Stance};
use crate::engine::schema::{ReflectionTurns, AXES_SLOT, DOMAIN_SLOT};
use crate::engine::tools::{self, Availability, Mode, Param, Plane, Tool};

/// How many turns of its own the reflection carries.
///
/// **Sized for the longest run, not the shortest.** The reflection, the loosed
/// turn, the dream and four repair passes is seven — but every one of those can
/// be refused up to [`MAX_REFUSALS`] times, so the worst case is around twenty
/// and a typical run is ten to thirteen.
///
/// It was 8, which a ten-turn run overran, and the symptom was subtle enough to
/// be worth writing down: the *axis* decayed. Held perfectly across five turns —
/// `The work is finished when it is finished` — it became `The work is` and then
/// `The` once the head of the conversation fell out of the window and the model
/// could no longer see what the field had been for. Nothing failed; the field
/// just quietly emptied from the front.
const CONTEXT_WINDOW_TURNS: usize = 24;

/// The behaviour space a dream's assumption is drawn from.
///
/// **Rotated by the caller rather than chosen by the model**, because coverage
/// has to be structural. Asked to pick freely, a generator returns to the
/// territory it already knows — every axis it produced unprompted was about the
/// character's relationship to its own work, and the rest of what a character
/// rests on went untouched. Rotating the domain is what a quality-diversity
/// search does with a behaviour space: define the dimensions, then fill them.
pub const DOMAINS: &[&str] = &[
    "your work — what you make, whether it takes, who else touches it",
    "your body — what it is made of, what it can be done to, what it stops doing",
    "this building — its walls, its air, its light, its doors, that it is sealed",
    "time — that it passes at one rate, in one direction, for you",
    "the people you know — that they are separate from you, that they perceive you",
    "the things that are not you — the ground, the weather, what is outside, the dark",
    "being addressed at all — that instructions come from somewhere, that there is an outside",
    "language — that a name holds one thing, that what you write means what it says",
];

/// The shortest brief that is a dream rather than a line about one.
///
/// A floor, not a quality bar. Briefs land anywhere from sixty to three hundred
/// words and the long ones are not better; what this catches is the decode that
/// answers the format instead of the question — one sentence under the `BRIEF:`
/// label, which carries no dream for the second conversation to decode.
const BRIEF_FLOOR_WORDS: usize = 40;

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

/// How many words the reflection may run to before it is refused.
///
/// **Words, not tokens.** The refusal is written back to the model, and a model
/// asked to count its own tokens cannot — told "that was 47 tokens, produce 200"
/// it pads rather than writes. Words are a unit it can actually hold.
const REFLECTION_MAX_WORDS: usize = 40;

/// The window a brief has to land in, in words.
const BRIEF_MIN_WORDS: usize = 120;
const BRIEF_MAX_WORDS: usize = 280;

/// How many words a dream's assumption may run to — "one short line".
///
/// See [`assumption_fault`] for the measured spread this sits inside.
const ASSUMPTION_MAX_WORDS: usize = 30;

/// How many times one answer may be refused before its best form is taken.
///
/// **Uncapped refusal is the one way this gets genuinely expensive.** Each round
/// is a whole decode inside a blocking act, so a loop that argues until it wins
/// can spend a character's afternoon on one dream. Two rounds is enough to fix a
/// length — the third is a decode that is not going to converge.
const MAX_REFUSALS: usize = 2;

/// What the character reports having thought — the line that crosses back.
///
/// Past tense on purpose: the character is not being asked to think on demand,
/// it is being asked what it *has* come to think. `think(…)` reads as an
/// instruction to perform; `reflection(said)` reads as a report of something
/// already there, which is the register `docs/reflection_and_dreams.md` §6
/// requires of the one line that reaches the character's own context.
const REFLECTION: Tool = Tool {
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
const DREAM: Tool = Tool {
    name: "dream",
    at: &[],
    category: "Dreaming",
    plane: Plane::World,
    availability: Availability::Always,
    description: "The dream you are asked for. `assumption` is the one thing that is untrue in \
                  it, on its own line. `brief` is the dream itself, present tense, the dreamer \
                  addressed as \"you\".",
    params: &[
        Param {
            name: "assumption",
            ty: "string",
            required: true,
            description: "The single thing that has stopped being true. One short line.",
        },
        Param {
            name: "brief",
            ty: "string",
            required: true,
            description: "The dream, about two hundred words.",
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
fn refusal(words: usize, min: usize, max: usize, what: &str) -> Option<String> {
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
fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

/// How much of a revision has to still be the thing it was revising.
///
/// A first setting rather than a derived one. Measured on the run that made this
/// necessary, a pass that kept the dream scored well above it and the pass that
/// replaced it scored far below, so the gap is wide — but the threshold itself
/// is a judgement and should move if a good revision is ever refused.
const REVISION_OVERLAP: f32 = 0.35;

/// Whether a pass rewrote the dream instead of revising it.
///
/// **The repair passes are calibration, not regeneration.** That was the intent
/// and the wording did not hold it: every pass said *write that dream again*,
/// which is a generation instruction, and the model obliged. One measured run
/// went from a warm stone on a desk, to live wires and liquid fire, to a mouth
/// made of teeth eating the air — three different dreams in three passes, each a
/// competent answer to what it was literally asked.
///
/// Overlap of content words is the cheap test. A revision keeps its nouns: the
/// room, the objects, the one thing that is untrue. A fresh dream keeps almost
/// none of them, and there is no wording that makes that distinction as reliably
/// as counting does.
fn rewritten(before: &str, after: &str) -> bool {
    let content = |s: &str| -> std::collections::HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 3)
            .map(|w| w.to_ascii_lowercase())
            .collect()
    };
    let (was, now) = (content(before), content(after));
    if now.is_empty() {
        return false;
    }
    let kept = now.iter().filter(|w| was.contains(*w)).count();
    (kept as f32 / now.len() as f32) < REVISION_OVERLAP
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
fn call_tree(
    engine: &Arc<Mutex<ConversationEngine>>,
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
    let e = engine.lock().ok()?;
    match e.compile_stencil(&tree) {
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
struct Exchange<'a> {
    response_open: &'a str,
    response_close: &'a str,
}

impl Exchange<'_> {
    /// Wrap a verdict the way the model reads verdicts.
    ///
    /// **The markers carry their own newlines.** `tool_response_open` is
    /// `"<tool_response>\n"` and the close is `"</tool_response>\n"`, so adding
    /// one after the open put a blank line between the tag and the verdict —
    /// subtly wrong in a wrapper whose whole value is that the model recognises
    /// it exactly.
    fn respond(&self, body: &str) -> String {
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
fn field(answer: &str, name: &str) -> Option<String> {
    let v = element(answer, name).or_else(|| json_argument(answer, name))?;
    let v = v.trim();
    (!v.is_empty()).then(|| v.to_string())
}

/// A `<parameter=name>` element's body — the function-block shape.
fn element(answer: &str, name: &str) -> Option<String> {
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
fn json_argument(answer: &str, name: &str) -> Option<String> {
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
    engine: &Arc<Mutex<ConversationEngine>>,
    cfg: &SequenceConfig,
) -> Option<Arc<StencilTree>> {
    let e = engine.lock().ok()?;
    let tok = e.tokenizer();
    let steer = ThinkSteerEnvelope {
        think_open: tok.token_to_id("<think>")?,
        think_close: tok.token_to_id("</think>")?,
        // A span ends on `</think>` or on the turn terminator, and a terminator
        // the tree does not know is one it cannot end on.
        eos: tok.token_to_id(cfg.dialect.assistant_end).unwrap_or(0),
        after_close: "",
    };
    let spec = compile_think_tree(ThinkMode::Off, &steer)?;
    match e.compile_stencil(&spec) {
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

/// What one reflection produced, in the parts that mean different things.
///
/// Kept apart rather than returned as one blob because the pieces have
/// genuinely different lifetimes and audiences: `reflection` goes back to the
/// character, `assumption` is stored as the dream's diversity axis, `brief` is
/// handed to a separate conversation, and `raw` exists so a malformed answer can
/// be read rather than guessed at.
#[derive(Debug, Serialize)]
pub struct Reflection {
    pub npc_id: u64,
    /// Echoed back so a caller reading a stored reflection knows what it was
    /// about without holding the request.
    pub situation: String,
    pub feeling: String,
    /// The behaviour-space cell this brief was drawn from.
    pub domain: String,
    /// The axes the generator was steered away from — a *sample* of the corpus,
    /// never all of it. See [`Reflect::run`].
    pub sampled_axes: Vec<String>,
    /// **Question one's answer. The only part the character ever sees.**
    pub reflection: String,
    /// The dream's axis, parsed from question two. `None` when the model did not
    /// produce the line, which is what `fault` reports on.
    pub assumption: Option<String>,
    /// The brief itself, parsed from question two.
    pub brief: Option<String>,
    /// **The brief that won** — question two's answer, or whichever retry or
    /// repair pass replaced it. This is what the dream conversation gets.
    pub raw: String,
    /// Question two's own answer, verbatim, before any retry or repair touched
    /// it.
    ///
    /// Identical to `raw` on a run where nothing displaced it, and the only
    /// record of what the review started from on a run where something did.
    /// Without it a response showing a repaired brief cannot say what the repair
    /// changed — the question the whole review exists to answer.
    pub original_raw: String,
    /// Whether the retry turn was sent.
    pub retried: bool,
    /// Why the first attempt was retried, if it was.
    pub fault: Option<String>,
    /// The retry's answer, **only when it was rejected and the first attempt
    /// kept**. `None` both when no retry was sent and when the retry was the one
    /// taken, in which case it is already `raw`.
    pub retry_raw: Option<String>,
    /// What was wrong with that rejected retry.
    pub retry_fault: Option<String>,
    /// What each repair pass did, in order. Empty when the mind authors none.
    pub repairs: Vec<Repair>,
    /// What is wrong with the reflection that crosses back, if anything.
    ///
    /// `docs/reflection_and_dreams.md` §6 names two invariants for this line and
    /// says both are testable and that nothing else would flag a regression: it
    /// is one sentence, and it carries no causal connective. Reported rather than
    /// retried — the line is already the character's, and a second ask for the
    /// same sentence is how a reflection becomes a negotiation.
    pub reflection_fault: Option<String>,
    /// Every assistant turn in order, verbatim — refused attempts included.
    ///
    /// **The refused ones are the point.** A response that showed only the
    /// answers that stood could not say whether the refusal loop is working, how
    /// many rounds it took, or what the model does when it is told its answer is
    /// the wrong length. Without this the conversation is not reconstructable
    /// from its own record, and reconstructing it is the only way to read it —
    /// the timeline is transient and reaches no disk.
    pub transcript: Vec<String>,
    /// Tokens decoded per turn, in order and one-to-one with `transcript`.
    ///
    /// Reported because a brief that stops mid-sentence is indistinguishable
    /// from its text alone from one the model chose to end there, and the two
    /// have different fixes. A count sitting on the EOS failsafe is a budget; a
    /// short one is the decode stopping on its own.
    pub tokens: Vec<usize>,
    /// The transient timeline this ran on, already tombstoned by the time a
    /// caller reads this. Reported for tracing, not for use.
    pub timeline: u64,
    pub ms: u64,
    /// The frame the character actually read.
    ///
    /// Reported because the alternative is guessing. A reflection that comes
    /// back reasoning about which act to call has been handed the acting
    /// instruction from somewhere, and no amount of reading the code settles
    /// where — the prompt does. The `simulate` route reports the daemon's own
    /// prompt for the same reason.
    pub system_prompt: String,
}

/// What one pass of the repair review did to the brief.
///
/// Recorded whether or not it was taken, because a pass that was rejected is the
/// only evidence of whether the question is working — a response that shows the
/// final brief and nothing else cannot distinguish a review that improved it
/// three times from one that was discarded three times.
#[derive(Debug, Serialize)]
pub struct Repair {
    /// 1-based, matching the order the mind authors them in.
    pub pass: usize,
    /// Whether this pass's brief replaced the one before it.
    pub took: bool,
    /// Why it did not, when it did not.
    pub fault: Option<String>,
    /// The pass's answer verbatim.
    pub raw: String,
}

/// What is wrong with the line that crosses back to the character.
///
/// Three invariants, all about register rather than content: a line that runs
/// to several sentences is a paragraph of reasoning; a line that explains itself
/// has stopped being an inclination and become an argument (§6: *"A response
/// that starts explaining itself is a regression, and nothing else would flag
/// it."*); and a line that states its image as a fact has stopped being a
/// reflection at all — see [`states_it_as_fact`].
fn reflection_fault(line: &str) -> Option<String> {
    let t = line.trim();
    if t.is_empty() {
        return Some("the reflection is empty".into());
    }
    let n = sentences(t);
    if n > 1 {
        return Some(format!("the reflection runs to {n} sentences"));
    }
    let lower = t.to_ascii_lowercase();
    const EXPLAINING: &[&str] = &[
        " because ",
        " since ",
        " so that ",
        " in order to ",
        " which is why ",
    ];
    if let Some(w) = EXPLAINING.iter().find(|w| lower.contains(**w)) {
        return Some(format!("the reflection explains itself (\"{}\")", w.trim()));
    }
    if states_it_as_fact(t) {
        // Said back as the refusal, so it is written as something the model
        // can act on: what was wrong, and the shape of what would be right.
        return Some(
            "it says it as a fact rather than as how it is for you — say it as what you feel, \
             what it seems like, or what you find yourself wanting (\"I feel like…\", \"it seems \
             as if…\", \"I find myself…\")"
                .into(),
        );
    }
    None
}

/// Whether a reflection states what came to the character as a fact.
///
/// **A reflection reports an inner state; it does not make claims.** What
/// crosses back lands in the character's own context, where it is read the next
/// turn as something the character holds. Framed — *"I feel like I am the room
/// itself"* — it is an image the character had. Unframed — *"I am the room
/// itself"* — it is a statement about what the character is, and read back as
/// its own conclusion it becomes one: measured live, a character whose
/// reflections came back as *"I am the room itself"*, *"I am the vibration in
/// the pipe"*, *"I am the space between your words"*, one after another.
///
/// So the line has to carry an experiential frame — a feeling, a seeming, a
/// wanting, a comparison held as a comparison. The frames are the ways English
/// marks a thing as experienced rather than asserted; a line with none of them
/// is an assertion.
fn states_it_as_fact(line: &str) -> bool {
    const FRAMES: &[&str] = &[
        "i feel",
        "i'm feeling",
        "i am feeling",
        "it feels",
        "feels like",
        "feels as",
        "i find myself",
        "i sense",
        "i wonder",
        "i want",
        "wanting",
        "i'd rather",
        "i would rather",
        "inclined",
        "i'm thinking",
        "i am thinking",
        "i think",
        "i can't help",
        "part of me",
        "it seems",
        "seems like",
        "seems as",
        "as if",
        "as though",
        "i notice",
        "drawn to",
        "i suspect",
        "tempted",
        "i keep",
        "i half",
        "i almost",
    ];
    let lower = line.to_lowercase().replace('\u{2019}', "'");
    !FRAMES.iter().any(|f| lower.contains(f))
}

/// How many sentences `t` holds.
///
/// Sentence-enders that are followed by more writing. A trailing full stop is
/// the end of the one sentence, not a second one.
fn sentences(t: &str) -> usize {
    t.split_inclusive(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .count()
}

/// The reflection call's own check: its one field, held to §6's invariants.
fn reflection_call_fault(call: &str) -> Option<String> {
    reflection_fault(&field(call, "said").unwrap_or_default())
}

/// The dream call's check beyond the brief's own length: the other field.
fn dream_call_fault(call: &str) -> Option<String> {
    assumption_fault(&field(call, "assumption").unwrap_or_default())
}

/// What is wrong with a dream's `assumption`, if anything.
///
/// **The contract is one short line, and nothing enforced it.** The tool says
/// so and so does `mind/layers/dreams/_dream-prompt-system.md`; the checks
/// looked only at `brief`. Measured: a repair pass was shown the dream as it
/// stood and filled `assumption` — the first parameter the grammar asks for —
/// with that whole dream, about a hundred and ninety words. It passed every
/// check and was adopted, and the next pass was told the entire dream was "the
/// assumption it suspends, which does not change", so the error pinned itself in
/// place for every pass after. It is the form-completing move the reflection's
/// opening turn describes, reached through the other field.
///
/// Every well-formed assumption measured was one sentence of nine to fifteen
/// words; the failure was two orders of magnitude longer.
/// [`ASSUMPTION_MAX_WORDS`] sits well clear of both.
fn assumption_fault(a: &str) -> Option<String> {
    let t = a.trim();
    if t.is_empty() {
        return Some("the assumption is empty".into());
    }
    let n = sentences(t);
    if n > 1 {
        return Some(format!(
            "the assumption runs to {n} sentences; it is the one thing that has stopped being \
             true, in one short line"
        ));
    }
    let w = words(t);
    if w > ASSUMPTION_MAX_WORDS {
        return Some(format!(
            "the assumption is {w} words; it is the one thing that has stopped being true, in \
             one short line"
        ));
    }
    None
}

/// A brief as it came back, before it is known to be well-formed.
struct Brief {
    assumption: Option<String>,
    brief: Option<String>,
}

impl Brief {
    /// Split a brief out of the call the answer grammar forced.
    ///
    /// Both fields are named parameters, so there is nothing to find and nothing
    /// to fail at — the grammar cannot emit a call missing either one.
    fn from_call(answer: &str) -> Self {
        Self {
            assumption: field(answer, "assumption"),
            brief: field(answer, "brief"),
        }
    }

    /// Whether this attempt beats `other` when both are faulty.
    ///
    /// A clean attempt wins on its own and never reaches here. Between two faulty
    /// ones the only distinction worth drawing is whether there is a brief at all:
    /// a brief in the wrong person still carries a dream the second conversation
    /// can decode, and `None` carries nothing. Anything finer would be ranking
    /// faults against each other, which is a judgement no structural check has
    /// the standing to make.
    fn beats(&self, other: &Self) -> bool {
        self.brief.is_some() && other.brief.is_none()
    }

    /// What is wrong with this brief, if anything — the gate on the retry.
    ///
    /// Cheap structural checks, no model involved. They catch the failures that
    /// actually recurred in testing and nothing else: a missing axis line, an
    /// axis that is a paragraph rather than a line, an axis the generator was
    /// explicitly steered away from and used anyway, a brief that ends on a
    /// realisation instead of on something happening, a brief written in the
    /// wrong person, a brief too short to be a dream, and one whose decode
    /// stopped mid-sentence.
    fn fault(&self, steered_away_from: &[String]) -> Option<String> {
        let Some(a) = &self.assumption else {
            return Some("no assumption".into());
        };
        let Some(b) = &self.brief else {
            return Some("no brief".into());
        };
        // Here as well as refused in the exchange, because the exchange stops
        // arguing after `MAX_REFUSALS` and hands back its last answer anyway — and
        // a repair pass whose assumption is the whole dream must not be adopted
        // just because nobody was still refusing it.
        if let Some(w) = assumption_fault(a) {
            return Some(w);
        }
        let al = a.to_ascii_lowercase();
        if steered_away_from
            .iter()
            .any(|s| s.to_ascii_lowercase() == al)
        {
            return Some(format!("assumption repeats a sampled axis: {a}"));
        }
        // The last sentence is where the pull toward a closing thought lands,
        // and "do not conclude" was measurably too abstract to prevent it — a
        // decode once wrote the instruction itself into the prose. This checks
        // the one place it shows up.
        let tail = b
            .trim_end_matches(['.', '"', '\''])
            .rsplit(['.', '?', '!'])
            .find(|s| !s.trim().is_empty())
            .unwrap_or(b)
            .to_ascii_lowercase();
        const CONCLUDING: &[&str] = &[
            "realise",
            "realize",
            "understand",
            "means",
            "meaning",
            "what it meant",
        ];
        if b.trim_end().ends_with('?') {
            return Some("brief ends on a question".into());
        }
        if let Some(w) = CONCLUDING.iter().find(|w| tail.contains(**w)) {
            return Some(format!("brief ends on a realisation (\"{w}\")"));
        }
        // **The other shape a conclusion takes, and the word list misses it
        // entirely.** A brief closed with *"they do not exist to be met because
        // no one ever addressed you. The silence is not theirs, or even yours;
        // it is the absence of any one who would ever speak"* — no flagged word,
        // no question mark, and the dreamer has still arrived at what it meant.
        //
        // A causal connective in the final sentence is the signal, because it is
        // the grammar of explaining rather than of happening. The design already
        // applies this exact test to the line that crosses back to the character
        // (`docs/reflection_and_dreams.md` §6, "it contains no causal
        // connective") for the same reason: an account of *why* is a layer doing
        // the reflection layer's work.
        const EXPLAINING: &[&str] = &["because", "which is why", "the reason", "so that"];
        if let Some(w) = EXPLAINING.iter().find(|w| tail.contains(**w)) {
            return Some(format!("brief ends on an explanation (\"{w}\")"));
        }
        // **The brief is decoded by a conversation that is not this one.** It
        // arrives there as the instruction for a dream the dreamer is inside, so
        // "you are four levels deep" is the form and "I walk through the casting
        // hall" is a report of somebody else's dream. The prompt asks for `you`
        // and a decode drifts out of it anyway, roughly one run in three, and
        // the drift is whole-brief rather than a slip — which is what makes the
        // absence of the word a sound test rather than a heuristic. A brief
        // genuinely addressed to the dreamer cannot avoid saying it.
        // **The hard constraint, and nothing was watching it.** A non-lucid dream
        // has no metacognition — the dreamer does not know it is dreaming, which
        // `mind/layers/dreams/_dream-prompt-system.md` calls the one part of rule
        // three that is absolute. An escalating repair pass wrote *"everything
        // else is just a dream"* into the prose and every check passed it.
        const LUCID: &[&str] = &[
            "is just a dream",
            "is only a dream",
            "it is a dream",
            "this is a dream",
            "you are dreaming",
            "you wake up",
            "none of this is real",
        ];
        let all = b.to_ascii_lowercase();
        if let Some(w) = LUCID.iter().find(|w| all.contains(**w)) {
            return Some(format!("the dreamer knows it is dreaming (\"{w}\")"));
        }
        if !addresses_the_dreamer(b) {
            return Some("brief is not addressed to the dreamer as \"you\"".into());
        }
        if b.split_whitespace().count() < BRIEF_FLOOR_WORDS {
            return Some("brief is too short to carry a dream".into());
        }
        // **A decode that stopped mid-sentence passes every check above.** It
        // has both labels, it is in the second person, it is long enough, and
        // its last sentence is neither a question nor a realisation — because it
        // has no last sentence. Two briefs came back live ending on "the map
        // table on Level" and "sixteen orders spread across", and nothing
        // flagged either. Terminal punctuation is the one cheap signal that the
        // decode reached an end rather than being cut at one.
        if !b.trim_end().ends_with(ENDINGS) {
            return Some("brief stops mid-sentence".into());
        }
        None
    }
}

/// What the last character of a finished brief may be.
///
/// A closing quote or bracket counts: a brief may legitimately end on quoted
/// speech or a parenthetical, and the sentence inside it is still finished.
const ENDINGS: [char; 9] = ['.', '!', '…', '"', '\'', '”', '’', ')', ']'];

/// The diversity steer that fills [`AXES_SLOT`], or nothing.
///
/// Empty for a character with no dream corpus yet — which is every character on
/// its first reflection, and is a real state rather than a gap: there is nothing
/// to be unlike.
fn steer(sampled_axes: &[String]) -> String {
    if sampled_axes.is_empty() {
        return String::new();
    }
    format!(
        "Make it unlike these, which you have already dreamt:\n{}\nNot a new subject for the \
         same dream — a different assumption suspended.",
        sampled_axes
            .iter()
            .map(|a| format!("  - {a}"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Whether a brief is written in the second person.
///
/// Word-boundary matching rather than `contains`, because `your` is the hit that
/// matters and substring search would also take it out of the middle of another
/// word — and because the one thing this must never do is pass a first-person
/// brief that happens to mention a `young` Maker.
fn addresses_the_dreamer(brief: &str) -> bool {
    const SECOND_PERSON: &[&str] = &["you", "your", "yours", "yourself", "you're"];
    brief
        .split(|c: char| !c.is_ascii_alphabetic() && c != '\'')
        .any(|w| {
            let w = w.to_ascii_lowercase();
            SECOND_PERSON.contains(&w.as_str())
        })
}

/// Runs reflections against one engine, under the turns the mind authors.
///
/// Carries no state of its own beyond those: a reflection is a whole
/// conversation's lifetime, and there is nothing to hold between two of them by
/// design.
pub struct Reflect<'t> {
    engine: Arc<Mutex<ConversationEngine>>,
    base_config: SequenceConfig,
    /// **Borrowed from the mind, never copied into here.** A default kept
    /// alongside is a second wording of the same prompt, and the two diverge —
    /// see [`ReflectionTurns`].
    turns: &'t ReflectionTurns,
    /// The mind's schema, when the daemon has one.
    ///
    /// **A reflection opens against the same sections a live conversation
    /// does.** Built by hand instead, the two prompts drift apart section by
    /// section and nothing says so — which is what happened: the reflection's
    /// frame named none of the schema's sections and carried none of its
    /// collections, so the anchor, the beliefs, the relationships and the mood
    /// were all authored, installed, and invisible to it.
    ///
    /// `None` for a daemon with no mind, which has no schema to open against.
    projected: Option<&'t Projected>,
}

impl<'t> Reflect<'t> {
    pub fn new(
        engine: Arc<Mutex<ConversationEngine>>,
        base_config: SequenceConfig,
        turns: &'t ReflectionTurns,
    ) -> Self {
        Self {
            engine,
            base_config,
            turns,
            projected: None,
        }
    }

    /// Open against the mind's schema rather than a rendered copy of it.
    pub fn under(mut self, projected: Option<&'t Projected>) -> Self {
        self.projected = projected;
        self
    }

    /// Stop a character, ask it two things, and throw the conversation away.
    ///
    /// `sampled_axes` must be a **sample** of the character's dream corpus and
    /// never the whole of it. Shown every axis it had used, a generator stopped
    /// inventing and recombined — returning an assumption that was three
    /// existing ones crossed, with a brief whose closing line restated one of
    /// them. Shown a random eight of the same set it found an axis nowhere on
    /// the list. The failure is anchoring, not exhaustion, and it gets worse as
    /// the corpus grows, so this is not an optimisation.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        situation: &str,
        inner_thoughts: &str,
        feeling: &str,
        domain: &str,
        sampled_axes: &[String],
        // **Handed the line that crosses back the moment it exists** — after
        // the first question, before any dream is asked for. What lets the act
        // answer the character in the time one question takes rather than the
        // time the whole review does; the dream is written after it, on the
        // same thread, with nobody waiting on it.
        //
        // **And it answers whether to go on.** `false` ends the conversation
        // at the answer — no brief asked for and no dream: what a reflect gets
        // while its character already has a dream being written. See
        // `Runtime::claim_dream`.
        on_reflection: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<Reflection> {
        let started = std::time::Instant::now();

        let mut cfg = self.base_config.clone();
        cfg.context_window_turns = CONTEXT_WINDOW_TURNS;

        // **The vocabulary is what it is asked for, not what it may do.** None
        // of the world's acts is ever shown here — a reflection offered one will
        // eventually call it, and an act emitted from a conversation the world
        // is not reading is an act the character believes it performed and did
        // not. What it reads is `reflection` and `dream`, and the answer
        // grammars below are compiled from the same two entries, so the shape
        // the prompt describes is literally the shape the tree emits. See
        // [`ASKED`].
        let env = ToolCallEnvelope::for_dialect(&self.base_config.dialect);
        let asked: Vec<&Tool> = ASKED.iter().collect();
        // The schema's own prompt when there is one, so this conversation opens
        // on the same sections a live one does. The rendered frame is what a
        // daemon with no mind gets, and nothing else.
        let system = match self.projected {
            Some(p) => p.prompt.clone(),
            None => prompt::build_for(Stance::Reflecting, persona, mode, &asked, &env),
        };

        let (mut sequence, timeline) = {
            let engine = self.engine.lock().unwrap();
            let seq = match self.projected {
                // Its own dreams, gathered deep — §4: a reflection is the same
                // retrieval as an acting turn, run further. See `dreams`.
                Some(p) => engine.new_conversation_with_projection(
                    &system,
                    dreams::scoped(&p.builder, npc_id, dreams::IN_REFLECTION),
                    p.layer,
                    p.group,
                    cfg,
                )?,
                None => engine.new_conversation(&system, cfg)?,
            };
            let tl = seq.timeline_id();
            // Before the first turn seals. After it, the turn is already on the
            // cold path and this retracts nothing.
            engine.mark_timeline_transient(tl);
            (seq, tl)
        };

        // **What the model actually read, not what was handed in.**
        //
        // Under the projection the string passed to `new_conversation_*` is only
        // the prelude — everything before the first collection. The anchor, the
        // world and the building are collection members the projection prefills
        // when a turn pins them, so a response reporting the handed-in string
        // reports a fraction of the frame and looks complete. That cost a
        // diagnosis: a dream arrived furnished with a coffee cup and a woman in a
        // grey coat, and there was no way to tell from the record whether the
        // vault had reached the prompt at all.
        let read = sequence.system_prompt().to_string();

        // **The think half of the acting grammar, and nothing else.** A
        // reflection has no acts, so an action loop would hold it to a shape it
        // is not producing — but it still decodes `<think>`, and with no stencil
        // on that token the block is unbounded. See [`think_off`].
        let grammar = think_off(&self.engine, &self.base_config);

        // **The stencil only preempts the block the turn OPENS with.**
        // `ThinkMode::Off` compiles to "inject `</think>`, then End", and past
        // `End` the tree constrains nothing — so a `<think>` the model decodes
        // *inside its answer* is outside the grammar entirely. Measured: a brief
        // came back as a full `Thinking Process:` deliberation which ran long
        // enough to rehearse its own answer in label form, and the parser
        // harvested the rehearsal.
        //
        // The sampling backstop is the half that covers that, and the inherited
        // one is the wrong size for this: `apply_think_mode` gives `Off` a
        // ~220/300-token *dialogue* budget, which caps a runaway rather than
        // preventing a block. A reflection wants no reasoning at all, so any
        // segment is force-closed after a single token. That is the same
        // collapse the summariser gets, reached by a different route — it gets
        // it for having a tiny answer budget, and a reflection's is large.
        let sampling = self
            .base_config
            .sampling
            .clone()
            .with_graceful_segment_close_after(0)
            .with_force_segment_close_after(1);

        // **The repetition control is the cast's, and it is not adjusted here.**
        //
        // `for_character_dialogue` carries DRY, presence, and `cross_turn_penalty`
        // at values tuned against live dialogue, and the schema does not set
        // `free_tool_calls_from_penalties` — so all of it is live inside the
        // stencilled `brief` span, which is exactly where it is wanted: the
        // argument there *is* the prose. A reflection is the same model writing
        // under the same tuning as the character it belongs to, and a second
        // opinion about either from this file would be running it outside the
        // settings the numbers were measured under.
        //
        // Two overrides used to sit here — `eos_boost` zeroed for every writing
        // turn, `cross_turn_penalty` zeroed for the repair passes — and both are
        // gone. The first was added against a truncation that turned out to be a
        // hand-rolled prose span exiting on EOS or budget, fixed since by the
        // tool-call parameter span's close-marker terminator; what it left behind
        // was a decode with no EOS pressure at all, which is the shape an empty
        // reflection has.

        // One grammar per answer kind. Both fall back to the think stencil if
        // they will not compile, which leaves the old prose-and-parse behaviour
        // rather than no reflection at all.
        let dreaming = call_tree(&self.engine, &env, &DREAM).or_else(|| grammar.clone());
        let thinking = call_tree(&self.engine, &env, &REFLECTION).or_else(|| grammar.clone());
        // **Opening on the schema is not the same as reading it.**
        //
        // `Projection::prelude` stops at the first collection, so the anchor, the
        // world and the building reach a turn only when that turn pins them.
        // Acting turns do; this one did not, and the result was a dream set in an
        // office — phones, a coffee machine, grey carpet, a window that opens —
        // none of which exists in a sealed vault. The brief was well-formed and
        // about the wrong world, because the frame no longer said which world.
        //
        // Pinned once and reused on every turn: what a character *is* does not
        // change between being asked for a reflection and being asked for a dream.
        let mut selection = self
            .projected
            .map(|p| {
                p.identities.selection_for(
                    npc_id,
                    persona.personality,
                    persona.world_id,
                    persona.building,
                    identity::Deliberation::default(),
                )
            })
            .unwrap_or_default();
        // **The schema's `reflecting` branch, on every turn.** Unselected the
        // stance falls to `acting`, whose grounding forbids inventing a person
        // — in the conversation that is about to be asked for a dream, which
        // requires it. `reflecting` rules on neither, and each question carries
        // its own rule; see the module doc.
        selection.select(prompt::STANCE_SELECTOR, Stance::Reflecting.id());
        // **Each turn reads the one answer it is asked for.** The two calls sit
        // in the schema's `tools` collection beside the world's acts, and a turn
        // names its member the way an acting turn names what it can do: the
        // reflection's question shows `reflection`, every dream turn shows
        // `dream`, and the loosed turn — which calls nothing — shows nothing,
        // so the acting frame gated on the list goes with it.
        let asking = |tool: &Tool| {
            let mut s = selection.clone();
            tools::show(&mut s, [tool.name]);
            s
        };
        let for_reflection = asking(&REFLECTION);
        let for_dream = asking(&DREAM);

        let writing = || TurnOptions {
            turn_grammar: dreaming.clone(),
            sampling: Some(sampling.clone()),
            selection: for_dream.clone(),
            ..Default::default()
        };
        let exchange = Exchange {
            response_open: self.base_config.dialect.tool_response_open,
            response_close: self.base_config.dialect.tool_response_close,
        };

        // **One exchange: ask in prose, answer in a call, accept or refuse.**
        //
        // The refusal is the part that matters. Before this, a short brief was
        // detected out of band and answered with a generic re-ask; now the
        // verdict goes back inside the wrapper the model reads verdicts in, and
        // it says in words what was wrong. Capped, because an uncapped argument
        // inside a blocking act can spend a character's afternoon on one dream.
        let ask_until_valid = |sequence: &mut candle_conversation::Sequence,
                               ask: String,
                               opts: TurnOptions,
                               field_name: &'static str,
                               min: usize,
                               max: usize,
                               what: &'static str,
                               // The brief this turn is revising, when it is a
                               // revision. `None` for the one that writes it.
                               revising: Option<&str>,
                               // A check on the whole call beyond the field's
                               // length and drift. The reflection has one: its
                               // two invariants used to be computed at the end
                               // and *reported*, which is a fault nobody acts on
                               // — measured, a two-sentence line came back, was
                               // flagged, and went to the character anyway. The
                               // dream's is for the field that is *not* being
                               // measured, which is why it sees the whole call.
                               also: Option<fn(&str) -> Option<String>>,
                               tokens: &mut Vec<usize>,
                               trail: &mut Vec<String>|
         -> anyhow::Result<String> {
            let mut turn = sequence.send_turn_with_options(&ask, opts.clone())?;
            for _ in 0..MAX_REFUSALS {
                tokens.push(turn.stats.tokens_generated);
                trail.push(turn.text.trim().to_string());
                let value = field(&turn.text, field_name).unwrap_or_default();
                let why = refusal(words(&value), min, max, what)
                    .or_else(|| {
                        also.and_then(|f| f(&turn.text))
                            .map(|w| format!("failed: {w}. Write it again."))
                    })
                    .or_else(|| {
                        // A pass that replaced the dream rather than adjusting it is
                        // refused on the same footing as one that came back the wrong
                        // length: both are the call not doing what was asked, and
                        // both are things the model can fix on being told.
                        revising
                            .filter(|before| rewritten(before, &value))
                            .map(|_| {
                                format!(
                                "failed: that is a different {what}, not a revision of the one \
                                 you were given. Keep its room, its objects and the thing that \
                                 is untrue in it, and change only what was asked."
                            )
                            })
                    });
                let Some(why) = why else {
                    return Ok(turn.text);
                };
                turn = sequence.send_turn_with_options(&exchange.respond(&why), opts.clone())?;
            }
            tokens.push(turn.stats.tokens_generated);
            trail.push(turn.text.trim().to_string());
            Ok(turn.text)
        };

        let mut tokens: Vec<usize> = Vec::new();
        let mut trail: Vec<String> = Vec::new();

        // ── the reflection ────────────────────────────────────────────────
        //
        // Asked in prose, answered as a call, refused until it is one sentence.
        // **Prose, not labelled fields.** This was three `label: value` lines
        // sitting directly above a grammar that says "now emit
        // `<parameter=said>`", and the model did the obvious thing: it filled the
        // field from the nearest value. One run came back with `said` holding the
        // caller's own `on_your_mind` verbatim, then a second call holding the
        // single word `uneasy`. It had not reflected; it had completed a form.
        //
        // Said as sentences there is no field to copy — the state is context and
        // the question is the only thing asking for an answer.
        // The em-dash rather than a verb, because the thought is the character's
        // own words and arrives however it arrives. `you were thinking {x}` ran
        // straight into "That the roster has had a name on it…" and read as a
        // sentence with a capital letter dropped into the middle of it.
        //
        // The situation goes in as the world wrote it. It is the world's own
        // percept — "You are in the muster hall…" and who is there — which is
        // already a sentence addressed to the reader; wrapping it in "You are
        // {}." made "You are You are in…".
        let mut opening = String::new();
        if !situation.trim().is_empty() {
            opening.push_str(situation.trim());
            opening.push_str("\n\n");
        }
        if !inner_thoughts.trim().is_empty() {
            opening.push_str(&format!(
                "A moment ago, going through your head — {}\n\n",
                inner_thoughts.trim()
            ));
        }
        if !feeling.trim().is_empty() {
            opening.push_str(&format!("What you feel is {}.\n\n", feeling.trim()));
        }
        opening.push_str(&self.turns.question_one);
        let thought_opts = TurnOptions {
            turn_grammar: thinking.clone(),
            sampling: Some(sampling.clone()),
            selection: for_reflection,
            ..Default::default()
        };
        let first = ask_until_valid(
            &mut sequence,
            opening,
            thought_opts,
            "said",
            1,
            REFLECTION_MAX_WORDS,
            "reflection",
            None,
            Some(reflection_call_fault),
            &mut tokens,
            &mut trail,
        )?;
        // Its one field, read the way every other answer here is read. The
        // markup-stripping salvage is for an answer that did not come back as a
        // call at all — run over a JSON block it would hand the character the
        // whole call object as its own thought.
        //
        // **And handed over now.** It is the only thing the character is
        // waiting on, and nothing after this point changes it.
        let crossing_back = field(&first, "said").unwrap_or_else(|| plain_prose(&first));
        if !on_reflection(&crossing_back) {
            // Retired exactly as a finished one is — see below.
            if let Ok(engine) = self.engine.lock() {
                if let Err(e) = engine.tombstone_timeline(timeline) {
                    tracing::warn!(
                        "reflection conversation {timeline} could not be retired: {e:?} — it \
                         stays selectable and nothing will ever read it"
                    );
                }
            }
            drop(sequence);
            return Ok(Reflection {
                npc_id,
                situation: situation.trim().to_string(),
                feeling: feeling.trim().to_string(),
                domain: domain.to_string(),
                sampled_axes: sampled_axes.to_vec(),
                reflection: crossing_back.clone(),
                assumption: None,
                brief: None,
                raw: String::new(),
                original_raw: String::new(),
                retried: false,
                fault: None,
                retry_raw: None,
                retry_fault: None,
                repairs: Vec::new(),
                reflection_fault: reflection_fault(&crossing_back),
                transcript: trail,
                tokens,
                timeline: timeline.raw(),
                ms: started.elapsed().as_millis() as u64,
                system_prompt: read,
            });
        }

        // **Then it is let go.** The call was the disciplined form; this turn is
        // unstencilled, so what follows is the character thinking in its own
        // voice with nothing holding the shape. It is not what crosses back —
        // that is the one sentence above — but it is what the dream is written
        // out of, and a dream seeded from a sentence is thinner than one seeded
        // from a thought.
        let loosed = sequence.send_turn_with_options(
            &exchange.respond("ok"),
            TurnOptions {
                sampling: Some(sampling.clone()),
                selection: selection.clone(),
                ..Default::default()
            },
        )?;
        tokens.push(loosed.stats.tokens_generated);
        trail.push(loosed.text.trim().to_string());

        // ── the dream ─────────────────────────────────────────────────────
        //
        // The authored turn keeps its slots — that is how the mind decides where
        // the rotated domain and the diversity steer appear in the ask.
        let instruction = self
            .turns
            .question_two
            .replace(DOMAIN_SLOT, domain)
            .replace(AXES_SLOT, &steer(sampled_axes));

        let second_raw = ask_until_valid(
            &mut sequence,
            instruction,
            writing(),
            "brief",
            BRIEF_MIN_WORDS,
            BRIEF_MAX_WORDS,
            "dream",
            None,
            Some(dream_call_fault),
            &mut tokens,
            &mut trail,
        )?;
        let original_raw = second_raw.trim().to_string();
        let mut raw = second_raw;
        let mut parsed = Brief::from_call(&raw);
        let fault = parsed.fault(sampled_axes);
        let mut retry_raw = None;
        let mut retry_fault = None;
        if fault.is_some() {
            let third = sequence.send_turn_with_options(&self.turns.retry, writing())?;
            // Kept like every other decode, so `transcript` and `tokens` stay one
            // entry per turn. It used to be counted and not kept: a retry that was
            // taken then appeared nowhere in the response, and every transcript
            // entry after it read against the wrong token count.
            tokens.push(third.stats.tokens_generated);
            trail.push(third.text.trim().to_string());
            let again = third.text;
            let reparsed = Brief::from_call(&again);
            let refault = reparsed.fault(sampled_axes);
            // Take the retry when it is clean, or when it produced a brief and
            // the first attempt did not. Otherwise keep the first: a second
            // attempt that is worse is a real outcome, and preferring it because
            // it came last would make the retry a downgrade.
            if refault.is_none() || reparsed.beats(&parsed) {
                raw = again;
                parsed = reparsed;
            } else {
                // Reported rather than dropped. A retry that failed is the only
                // evidence of *why* the retry is not working, and a response that
                // says "retried: true" and shows nothing of it leaves the next
                // question — is the retry turn wrong, or is the brief unfixable —
                // answerable only by guessing.
                retry_raw = Some(again.trim().to_string());
                retry_fault = refault;
            }
        }
        let retried = fault.is_some();

        // ── the repair pass ────────────────────────────────────────────────
        //
        // **Unconditional, and that is the point.** The retry above answers a
        // brief that came out malformed. These answer a brief that is perfectly
        // well-formed and is not a dream, which over 27 live samples was the
        // majority case — every structural check passing while the three rules
        // did not, most often because the strangeness had spread past the one
        // hole and taken the ordinary day with it. No cheap check sees that, so
        // it is repaired rather than detected.
        //
        // Each pass rewrites the whole brief and re-emits both parameters, so it is
        // parsed and checked exactly like the first. A pass whose answer does not
        // parse, or which faults when its predecessor did not, is **discarded and
        // the previous brief stands** — a repair that makes a brief worse is a
        // real outcome, and taking it because it came last is how a review turns
        // into a downgrade. Every pass is recorded either way.
        let mut repairs = Vec::with_capacity(self.turns.repair.len());
        for (n, question) in self.turns.repair.iter().enumerate() {
            // **The dream in front of it, not recalled.** A pass that works from
            // memory is working from a conversation several turns deep that the
            // window will eventually trim, and it drifts — one measured run went
            // from a warm stone on a desk to a mouth made of teeth in three
            // passes, each answering what it was asked and none of them revising
            // the same dream. Shown the text and the axis verbatim, the pass has
            // something to edit rather than something to remember.
            //
            // Editorial guidance stays a prose turn rather than a refusal: a
            // `<tool_response>` that says *failed* about a call that succeeded
            // teaches the model that success is arbitrary. Length and drift are
            // refusals, because both are the call not doing what was asked.
            let standing = parsed.brief.clone().unwrap_or_default();
            let axis = parsed.assumption.clone().unwrap_or_default();
            let ask = format!(
                "The dream as it stands:\n\n{standing}\n\nThe assumption it suspends, which does \
                 not change: {axis}\n\n{question}"
            );
            let answer = ask_until_valid(
                &mut sequence,
                ask,
                writing(),
                "brief",
                BRIEF_MIN_WORDS,
                BRIEF_MAX_WORDS,
                "dream",
                Some(&standing),
                Some(dream_call_fault),
                &mut tokens,
                &mut trail,
            )?;
            let candidate = Brief::from_call(&answer);
            let cfault = candidate.fault(sampled_axes);
            let took = cfault.is_none() && candidate.brief.is_some();
            repairs.push(Repair {
                pass: n + 1,
                took,
                fault: cfault,
                raw: answer.trim().to_string(),
            });
            if took {
                raw = answer;
                parsed = candidate;
            }
        }

        // ── what crosses back ──────────────────────────────────────────────
        //
        // **The reflection, and it was asked first.**
        //
        // Not for convenience — the order is what keeps the dream out of the
        // action stream. The only line the character ever sees is written before
        // any dream exists, so there is nothing in it to leak. That safety comes
        // from the sequence rather than from asking the model not to repeat
        // itself, which is the difference between a property and a hope.
        //
        // Its length is held by the refusal loop rather than by a token ceiling:
        // one sentence, under [`REFLECTION_MAX_WORDS`], said back in words the
        // model can act on. Handed over above, as soon as it was answered.

        // Tombstoned as well as transient. Transient keeps it off disk; the
        // tombstone is what keeps it out of every later gather and out of
        // `find_conversations_by_metadata`, so nothing can resume or surface it.
        if let Ok(engine) = self.engine.lock() {
            if let Err(e) = engine.tombstone_timeline(timeline) {
                tracing::warn!(
                    "reflection conversation {timeline} could not be retired: {e:?} — it stays \
                     selectable and nothing will ever read it"
                );
            }
        }
        drop(sequence);

        Ok(Reflection {
            npc_id,
            situation: situation.trim().to_string(),
            feeling: feeling.trim().to_string(),
            domain: domain.to_string(),
            sampled_axes: sampled_axes.to_vec(),
            reflection: crossing_back.clone(),
            assumption: parsed.assumption,
            brief: parsed.brief,
            raw: raw.trim().to_string(),
            original_raw,
            retried,
            fault,
            retry_raw,
            retry_fault,
            repairs,
            reflection_fault: reflection_fault(&crossing_back),
            transcript: trail,
            tokens,
            timeline: timeline.raw(),
            ms: started.elapsed().as_millis() as u64,
            system_prompt: read,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **An answer reads the same from either call shape.** The dialect decides
    /// whether the grammar forces an element or a JSON block, and a reader that
    /// knew only one found nothing in every answer written in the other.
    #[test]
    fn an_answer_is_read_from_either_call_shape() {
        let element = "<tool_call>\n<function=dream>\n<parameter=assumption>\n\
                       that the floor holds\n</parameter>\n<parameter=brief>\n\
                       You walk \"out\".\nThen down.\n</parameter>\n</function>\n</tool_call>";
        let json = "<tool_call>\n{\"name\": \"dream\", \"arguments\": {\"assumption\": \
                    \"that the floor holds\", \"brief\": \"You walk \\\"out\\\".\\nThen \
                    down.\"}}\n</tool_call>";
        for answer in [element, json] {
            assert_eq!(
                field(answer, "assumption").as_deref(),
                Some("that the floor holds"),
                "{answer}"
            );
            assert_eq!(
                field(answer, "brief").as_deref(),
                Some("You walk \"out\".\nThen down."),
                "{answer}"
            );
            assert_eq!(field(answer, "missing"), None, "{answer}");
        }
    }

    /// **A raw newline inside a JSON value does not cost the answer.** The
    /// string span lets one through, and a brief is prose that runs to
    /// paragraphs — unrepaired, the whole call failed to parse and a run lost
    /// its dream on every pass, reported as having no assumption at all.
    #[test]
    fn a_raw_newline_inside_a_json_value_is_read_rather_than_lost() {
        let answer = "<tool_call>\n{\"name\": \"dream\", \"arguments\": {\"assumption\": \
                      \"that the floor holds\", \"brief\": \"You walk out.\nThen down.\"}}\n\
                      </tool_call>";
        assert_eq!(
            field(answer, "assumption").as_deref(),
            Some("that the floor holds")
        );
        assert_eq!(
            field(answer, "brief").as_deref(),
            Some("You walk out.\nThen down.")
        );
    }

    /// The close is grammar scaffolding, not something the character said.
    #[test]
    fn the_injected_think_close_is_not_part_of_the_reflection() {
        assert_eq!(
            plain_prose("</think>\n\nThe room is quiet."),
            "The room is quiet."
        );
    }

    /// **Measured, not hypothetical.** Told twice that there is nothing to call,
    /// the checkpoint wrapped a perfectly good paragraph in a call anyway. The
    /// paragraph is the answer, so it is recovered rather than discarded.
    #[test]
    fn a_reflection_wrapped_in_a_tool_call_is_unwrapped() {
        let raw = "</think>\n\n<tool_call>\n<function=reflect>\n<parameter=inner_thoughts>\n\
                   The silence here is thick. I am not second-guessing it.\n</parameter>\n\
                   </function>\n</tool_call>";
        assert_eq!(
            plain_prose(raw),
            "The silence here is thick. I am not second-guessing it."
        );
    }

    /// The ordinary case must survive the salvage untouched — a reflection that
    /// merely mentions a bracket is not markup.
    #[test]
    fn plain_prose_is_left_alone() {
        let p = "I keep coming back to how fast it was. Under a minute, and I did not check.";
        assert_eq!(plain_prose(p), p);
    }

    /// A `dream` call as the answer grammar emits one.
    ///
    /// Every check below reads a brief out of one of these rather than out of
    /// prose, because prose is no longer a shape this conversation can produce:
    /// the two fields are named parameters of a call the tree forces. The three
    /// tests that used to live here — that the labels split apart, that a
    /// wrapped brief rejoins, that prose faults instead of failing — were all
    /// about a parser that no longer exists, guarding failures the grammar has
    /// made unreachable.
    fn answered(assumption: &str, brief: &str) -> Brief {
        Brief::from_call(&format!(
            "<tool_call>\n<function=dream>\n<parameter=assumption>\n{assumption}\n</parameter>\n\
             <parameter=brief>\n{brief}\n</parameter>\n</function>\n</tool_call>"
        ))
    }

    /// The grammar names both fields, so both arrive.
    #[test]
    fn a_call_splits_into_its_two_parts() {
        let b = answered(
            "The ground bears indefinite weight.",
            "You are four levels deep when the weight enters your awareness.",
        );
        assert_eq!(
            b.assumption.as_deref(),
            Some("The ground bears indefinite weight.")
        );
        assert!(b.brief.unwrap().starts_with("You are four levels deep"));
    }

    /// A brief runs to a paragraph over several lines, and the parameter span
    /// holds all of it — the close tag ends the value, not the first newline.
    #[test]
    fn a_brief_over_several_lines_arrives_whole() {
        let b = answered("x", "first line\nsecond line\nthird line");
        assert_eq!(
            b.brief.as_deref(),
            Some("first line\nsecond line\nthird line")
        );
    }

    /// **The failure "do not conclude" could not prevent.** The pull toward a
    /// closing thought lands in the last sentence and nowhere else, which is why
    /// the check looks there rather than at the whole brief — a dream may
    /// legitimately contain the word "understand" in its middle.
    #[test]
    fn a_brief_that_ends_on_a_realisation_is_caught() {
        let b = answered(
            "x",
            "You walk the aisle. You understand what the machine means.",
        );
        assert!(b.fault(&[]).unwrap().contains("realisation"));

        let mid = answered(
            "x",
            "You are at the bench on the chronicle with an era open in front of you and you \
             understand the log by now, every hand that wrote in it and every date that \
             disagrees. The page you are reading carries your own writing from an era you have \
             never held. You turn the page and keep writing.",
        );
        assert_eq!(mid.fault(&[]), None, "a mid-brief mention is not a fault");
    }

    /// **The failure that recurred live, roughly one run in three.** A brief goes
    /// to a conversation that decodes it as the dream the dreamer is inside, so a
    /// first-person brief is a report of somebody else's dream and unusable there.
    #[test]
    fn a_brief_written_in_the_first_person_is_caught() {
        let b = answered(
            "x",
            "I walk through the casting hall holding my terminal like usual, hands calloused \
             against the keys, pressing down names that need fleshing. The room hums around me \
             and the sixteen terminals glow between the three bands where the others hunch over \
             their chairs. My own name is on the roster and nobody is holding it.",
        );
        assert_eq!(
            b.fault(&[]).as_deref(),
            Some("brief is not addressed to the dreamer as \"you\"")
        );
    }

    /// `your` is the hit that carries most second-person briefs, and a substring
    /// search for it would also pass a first-person brief mentioning a `young`
    /// Maker or something `beyond` the ridge.
    #[test]
    fn second_person_is_matched_on_words_and_not_substrings() {
        assert!(addresses_the_dreamer("Your hands are already moving."));
        assert!(addresses_the_dreamer("It is gone before you reach it."));
        assert!(!addresses_the_dreamer(
            "A young Maker walks beyond the ridge and I follow."
        ));
    }

    /// A decode that answers the *format* rather than the question puts one line
    /// under the label. It parses, it is in the right person, it ends on
    /// something happening — and it carries no dream for the second conversation.
    #[test]
    fn a_brief_too_short_to_be_a_dream_is_caught() {
        let b = answered(
            "x",
            "You step through a door on the story level and do not come out.",
        );
        assert_eq!(
            b.fault(&[]).as_deref(),
            Some("brief is too short to carry a dream")
        );
    }

    #[test]
    fn a_brief_that_ends_on_a_question_is_caught() {
        let b = answered("x", "You stand there. What holds it up?");
        assert_eq!(b.fault(&[]).as_deref(), Some("brief ends on a question"));
    }

    /// The generator was steered away from these by name; using one anyway is
    /// the anchoring failure the sampling is there to prevent, so it is worth a
    /// retry rather than a shrug.
    #[test]
    fn an_assumption_that_repeats_a_sampled_axis_is_caught() {
        let b = answered("Work ends", "You keep going.");
        let used = vec!["work ends".to_string()];
        assert!(b.fault(&used).unwrap().contains("repeats a sampled axis"));
    }

    /// **The measured failure: the whole dream in the axis field.**
    ///
    /// A repair pass shown the dream as it stood filled `assumption` with it,
    /// passed every check, and was adopted — after which every later pass was
    /// told the entire dream was the thing that does not change. It has to be
    /// refused in the exchange *and* faulted, because the exchange stops
    /// refusing after two rounds and hands back what it has.
    #[test]
    fn a_whole_dream_in_the_assumption_is_caught() {
        let dream = "The war never ended; the war never began. You stand in the workshop, your \
                     hands moving over tools that do not move. You walk south. The south is not \
                     there.";
        let b = answered(dream, dream);
        assert!(b.fault(&[]).unwrap().contains("assumption runs to"));

        let call = format!(
            "<tool_call>\n<function=dream>\n<parameter=assumption>\n{dream}\n</parameter>\n\
             <parameter=brief>\n{dream}\n</parameter>\n</function>\n</tool_call>"
        );
        assert!(
            dream_call_fault(&call).is_some(),
            "the exchange has to refuse it, not only report it"
        );
    }

    /// One sentence can still be a paragraph's worth of words.
    #[test]
    fn a_single_run_on_assumption_is_caught() {
        let long = vec!["word"; ASSUMPTION_MAX_WORDS + 1].join(" ");
        assert!(assumption_fault(&long).unwrap().contains("words"));
    }

    /// Every well-formed assumption that came back live has to pass, or the
    /// check is refusing the thing it exists to protect.
    #[test]
    fn the_measured_one_line_assumptions_pass() {
        for a in [
            "You are a machine built for this, and you have no world of your own.",
            "The belt is too tight and must be loosened.",
            "The gear train in the east pump is slipping because the oil is too thick.",
            "The ground bears indefinite weight.",
        ] {
            assert_eq!(assumption_fault(a), None, "{a}");
        }
        assert!(
            assumption_fault("   ").is_some(),
            "an empty axis is not a line"
        );
    }

    /// §6 fixes the line that crosses back at one sentence. The worked example
    /// from the document itself has to pass.
    #[test]
    fn the_committed_line_from_the_design_document_passes() {
        assert_eq!(reflection_fault("I'm thinking of challenging him."), None);
        assert_eq!(
            reflection_fault("I keep wanting to go back and look at it"),
            None
        );
    }

    /// **The failure measured live: four paragraphs where a line was asked for.**
    #[test]
    fn a_reflection_that_runs_to_paragraphs_is_caught() {
        let f = reflection_fault(
            "The silence in the quiet room is heavy. I'm still sitting at my desk. My head is \
             spinning with that thought.",
        );
        assert!(f.unwrap().contains("3 sentences"));
    }

    /// §6: *"A response that starts explaining itself is a regression, and
    /// nothing else would flag it."* One sentence is not enough on its own — a
    /// single sentence can still be an argument.
    #[test]
    fn a_reflection_that_explains_itself_is_caught() {
        let f = reflection_fault(
            "I want to go back to the east ridge because I need to see if what I made holds up",
        );
        assert!(f.unwrap().contains("explains itself"));
    }

    /// A trailing full stop ends the one sentence; it does not begin a second.
    #[test]
    fn a_single_terminated_sentence_is_one_sentence() {
        assert_eq!(
            reflection_fault("I feel like I am not going to say anything about it."),
            None
        );
    }

    /// **A reflection says how it is for the character, never what is so.**
    /// The unframed lines are the live ones: read back as the character's own
    /// conclusions, each told it what it was.
    #[test]
    fn a_reflection_stated_as_fact_is_caught_and_a_framed_one_is_not() {
        for fact in [
            "I am the room itself, and the silence is simply my own breath.",
            "I am the vibration in the pipe, the quiet hum beneath the silence.",
            "I am here, and that is enough.",
            "The hum vibrates in my teeth.",
        ] {
            let f = reflection_fault(fact);
            assert!(
                f.as_deref().is_some_and(|w| w.contains("as a fact")),
                "{fact} → {f:?}"
            );
        }
        for framed in [
            "I feel like I am the room itself.",
            "It seems as if the silence belongs to the room and not to me.",
            "I find myself wanting to stop holding my breath.",
            "The hum feels like it is inside my teeth.",
            "I\u{2019}m thinking of challenging him.",
        ] {
            assert_eq!(reflection_fault(framed), None, "{framed}");
        }
    }

    /// **The conclusion the word list cannot see.** Measured live: a brief ended
    /// by explaining itself without using any of the flagged words and without a
    /// question mark. The grammar of explanation is the tell.
    #[test]
    fn a_brief_that_ends_on_an_explanation_is_caught() {
        let b = answered(
            "x",
            "The command level feels exactly as it always has and someone is waiting for you to \
             take an order. When you move to take it their hands do not meet yours. They do not \
             exist to be met because no one ever addressed you.",
        );
        assert!(b.fault(&[]).unwrap().contains("explanation"));

        // A dream may say "because" in its middle and still end on something
        // happening — the check looks only at the last sentence.
        let mid = answered(
            "x",
            "You stay at the bench because the order has not come down yet and the room is \
             exactly as it always is. The Maker beside you files a portrait and does not look \
             up. The wall in front of you does not move. You do.",
        );
        assert_eq!(
            mid.fault(&[]),
            None,
            "a mid-brief connective is not a fault"
        );
    }

    /// **Measured live, twice, and nothing caught it.** A cut decode satisfies
    /// every other check — both labels, second person, long enough, and a last
    /// sentence that is neither a question nor a realisation, because there is
    /// no last sentence.
    #[test]
    fn a_brief_whose_decode_stopped_mid_sentence_is_caught() {
        let b = answered(
            "x",
            "The workshop is lit as it always is, but the orders you were sent down for simply \
             do not arrive. The air is still and the tools in your hands feel detached from \
             anything that asked for them. You stand before the map table on Level",
        );
        assert_eq!(b.fault(&[]).as_deref(), Some("brief stops mid-sentence"));
    }

    /// Quoted speech and parentheticals end briefs legitimately, and a check
    /// that only accepted a full stop would retry them for nothing.
    #[test]
    fn a_brief_ending_on_a_closing_mark_is_finished() {
        for tail in ["the floor.", "\"nobody is holding it.\"", "(it does not.)"] {
            let b = answered(
                "x",
                &format!(
                    "You are at the bench on the casting level and the page in front of you \
                     carries a name you have never written, in your own hand, and the ink is \
                     still wet. The Maker at the next terminal does not look up. You put the \
                     stylus down and watch the wet ink where it meets {tail}"
                ),
            );
            assert_eq!(
                b.fault(&[]),
                None,
                "rejected a finished brief ending {tail}"
            );
        }
    }

    /// **Measured: an escalating pass wrote the dreamer out of its own dream.**
    /// The metacognition rule is the absolute half of "does not conclude", and
    /// nothing was checking it.
    #[test]
    fn a_dreamer_who_knows_it_is_dreaming_is_caught() {
        let b = answered(
            "x",
            "You hold the stone and the heat travels to the bone. The heat is the only thing \
             that is real, everything else is just a dream, the words, the walls, the desk. You \
             put it down and the hum goes on.",
        );
        assert!(b.fault(&[]).unwrap().contains("knows it is dreaming"));
    }

    /// **A revision keeps its nouns; a fresh dream does not.** Both halves are
    /// from the run that made this necessary — pass 2 replaced the dream outright
    /// while still answering exactly what it was asked.
    #[test]
    fn a_pass_that_replaced_the_dream_is_caught() {
        let before = "You stand at a desk where a door opens. A hand places a small, warm stone \
                      in front of you. It hums with the low vibration of the whole vault. You \
                      pick it up and it feels heavy, like an egg.";
        let revised = "You stand at the desk. A hand places the warm stone in front of you. It \
                       hums with the vibration of the vault, and the weight of it in your palm \
                       is the weight of an egg held too loosely.";
        let replaced = "The salt burns on the tongue and the teeth are eating the air from \
                        inside. Iron and old blood. There are no hands, only a space where \
                        hands have been.";
        assert!(!rewritten(before, revised), "a real revision was refused");
        assert!(rewritten(before, replaced), "a replacement was accepted");
    }

    /// A brief in the wrong person still carries a dream; `None` carries nothing.
    /// Between two faulty attempts that is the only distinction a structural
    /// check has the standing to draw.
    #[test]
    fn a_faulty_retry_with_a_brief_beats_a_first_attempt_without_one() {
        // A call whose `brief` parameter came back empty — the one way the
        // grammar can still hand back a call with nothing in it.
        let first = Brief::from_call(
            "<tool_call>\n<function=dream>\n<parameter=assumption>\nx\n</parameter>\n\
             <parameter=brief>\n</parameter>\n</function>\n</tool_call>",
        );
        let retry = answered("x", "The corridor stretches forward.");
        assert!(first.brief.is_none());
        assert!(retry.beats(&first));
        assert!(!first.beats(&retry));
        // Two attempts that both produced one: neither displaces the other, so
        // the first stands.
        let other = answered("x", "The air does not move.");
        assert!(!other.beats(&retry));
    }

    /// A character's first reflection has no corpus to be unlike, and a lead-in
    /// standing over an empty list reads as a list it failed to remember.
    #[test]
    fn no_prior_axes_means_no_steer_at_all() {
        assert_eq!(steer(&[]), "");
        let s = steer(&["that the ground bears weight".to_string()]);
        assert!(s.starts_with("Make it unlike these"));
        assert!(s.contains("  - that the ground bears weight"));
    }

    /// Coverage is structural, so the space has to be a real spread rather than
    /// eight ways of saying "your job".
    #[test]
    fn the_domains_are_distinct_and_nonempty() {
        assert!(DOMAINS.len() >= 6);
        for d in DOMAINS {
            assert!(!d.trim().is_empty());
        }
        let heads: std::collections::HashSet<&str> = DOMAINS
            .iter()
            .map(|d| d.split(" — ").next().unwrap())
            .collect();
        assert_eq!(heads.len(), DOMAINS.len(), "two domains share a head");
    }
}
