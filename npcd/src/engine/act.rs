//! Reading acts out of what a character said.
//!
//! # The wire format, and why it is this one
//!
//! A character acts by emitting a call. The format is one JSON object per line,
//! each naming a tool and its arguments:
//!
//! ```text
//! {"tool":"say","intent":"that he will not get the ledger"}
//! {"tool":"read","what":"the muster board"}
//! ```
//!
//! Not a function-calling API, and not XML. Three reasons, in order of how much
//! they cost when ignored:
//!
//! 1. **A partial line is detectable.** Generation can stop mid-token for a
//!    dozen reasons, and one-object-per-line means an interrupted emission
//!    loses exactly the last act rather than corrupting the parse of every act
//!    before it.
//! 2. **The model already writes it well.** ChatML-tuned models emit JSON
//!    objects reliably; the failure modes are known (trailing prose, a fenced
//!    block, a stray comma) and each is cheap to tolerate here.
//! 3. **It survives being wrong.** A line that is not JSON, or names a tool
//!    that does not exist, is *reported* rather than silently dropped — see
//!    [`Parsed::rejected`]. A parser that quietly discarded malformed acts would
//!    make a character that is failing to act indistinguishable from one
//!    choosing not to, and those need completely different fixes.
//!
//! # Prose around the calls is not an act
//!
//! A model will narrate. It will write "I turn to face him and say:" before the
//! call, or explain itself after. That text is kept — as [`Parsed::narration`],
//! so it is visible in Pulse and in the log — and it is *not* an act. The world
//! never sees it. This is the prompt's "if you write out what you are doing
//! instead of calling the tool, nothing happens", enforced rather than merely
//! requested.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::engine::tools;

/// One act a character took.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Act {
    pub tool: &'static str,
    /// The call's arguments, minus the `tool` key.
    pub args: Map<String, Value>,
}

impl Act {
    /// How this act reads in the Pulse feed and in the character's own window.
    ///
    /// Not the narrator's rendering — that is a separate concern and needs the
    /// character's voice. This is the act as an act: what was done, and with
    /// what intent, in a form a person reading the feed can scan.
    /// # The arguments are named when there is more than one
    ///
    /// A single value needs no label — `say — the redoubt burned twice` reads
    /// as what it is. Several bare values do not: `ask — Yaelis Vayne; where
    /// the data chips are` leaves a reader to infer which half is the person
    /// and which is the question, and for `act — steady them; Soren` or
    /// `post_notice — the muster board; the third era is written twice` the
    /// guess can go either way.
    ///
    /// So a multi-argument act carries `name: value` pairs, which the console
    /// splits to set the label and the value in different faces
    /// (`pulse.js::splitAct`). Single-argument acts stay bare, because a label
    /// there is noise standing in front of the only thing worth reading.
    pub fn summary(&self) -> String {
        let mut parts: Vec<(&str, String)> = Vec::new();
        // Ordered by the tool's own parameter list rather than by the map's
        // iteration order — `serde_json::Map` is a BTreeMap, so it would
        // otherwise render alphabetically and `manner` would precede `intent`.
        if let Some(t) = tools::by_name(self.tool) {
            for p in t.params {
                if let Some(v) = self.args.get(p.name) {
                    parts.push((p.name, render(v)));
                }
            }
        }
        match parts.len() {
            0 => self.tool.to_string(),
            1 => format!("{} — {}", self.tool, parts[0].1),
            _ => format!(
                "{} — {}",
                self.tool,
                parts
                    .iter()
                    .map(|(name, v)| format!("{name}: {v}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

fn render(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Why a line was not an act.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Rejected {
    /// Looked like a call and was not valid JSON.
    NotJson { line: String },
    /// Valid JSON, no `tool` key.
    NoTool { line: String },
    /// Named a tool that is not in the catalog. Carries the name so the log can
    /// say which — an invented tool is the single most useful signal that a
    /// character's prompt and its vocabulary have drifted apart.
    UnknownTool { tool: String },
    /// A required parameter was missing.
    MissingParam {
        tool: &'static str,
        param: &'static str,
    },
}

impl Rejected {
    /// What the character is told about a call that did not land.
    ///
    /// **Rejections have to come back.** A malformed call that is only logged
    /// is a character acting into silence: it does not know the act failed, so
    /// it has no reason to do anything differently, and it makes the same
    /// mistake every turn for as long as it runs. That is exactly what
    /// happened — `tell` without a `to`, over and over, each one a warning in a
    /// log nobody in the world can read.
    ///
    /// Second person, like every other thing a character reads, and it names
    /// what to do instead rather than only what was wrong.
    pub fn line(&self) -> String {
        match self {
            Rejected::NotJson { .. } => {
                "That did not come out as a call. One JSON object on a line, nothing else."
                    .to_string()
            }
            Rejected::NoTool { .. } => {
                "That call did not say which tool. Every call needs \"tool\".".to_string()
            }
            Rejected::UnknownTool { tool } => {
                format!("There is no \"{tool}\" you can do. Use one of the tools you were given.")
            }
            Rejected::MissingParam { tool, param } => {
                format!("Your \"{tool}\" needed \"{param}\" and did not have it. Nothing happened.")
            }
        }
    }
}

/// What one decode produced.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Parsed {
    pub acts: Vec<Act>,
    /// Text the model wrote around the calls. Kept and shown; never acted on.
    pub narration: String,
    /// Lines that tried to be acts and failed. Reported, never dropped.
    pub rejected: Vec<Rejected>,
}

impl Parsed {
    pub fn is_empty(&self) -> bool {
        self.acts.is_empty()
    }
}

/// Whether a line is plausibly a call rather than prose.
///
/// Deliberately loose: anything starting with `{` is *tried*, and failing the
/// try is reported. A tight test here would silently reclassify a malformed act
/// as narration, which is exactly the confusion this module exists to prevent.
fn looks_like_call(line: &str) -> bool {
    line.starts_with('{')
}

/// The envelope a `<tool_call>` block is wrapped in.
///
/// The model's own tool-call shape, and — far more importantly — **the shape the
/// stencil emits**. A constrained decode drives the grammar in
/// [`crate::engine::tools::specs`], which writes `<tool_call>` and then
/// `{"name": …, "arguments": {…}}` itself; a parser that only understood the
/// bare `{"tool": …}` line would reject every call the grammar produced, so the
/// stencil could never be armed. That is exactly what happened: the catalog
/// compiled, the registry armed, and nothing ever fired it because the prompt
/// taught a format the trigger token never appears in.
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// Escape raw control characters that appear **inside** a JSON string.
///
/// # The failure this exists for
///
/// The grammar's free-text span ends at the first unescaped `"`
/// (`stencil::Terminator::JsonString`) and excludes nothing else, so a model
/// filling a string argument may emit a literal newline. JSON forbids that —
/// a control character in a string must be escaped — so `serde_json` refuses
/// the object, and because [`parse`] walks *lines* (the `<tool_call>` envelope
/// spans them) the object arrives cut in two as well.
///
/// One character did this every turn for hours. The grammar had steered it
/// correctly — right tool, right addressee — and the first byte of its `about`
/// was a newline, so what came back was *"That did not come out as a call. One
/// JSON object on a line, nothing else"*, which it could not act on because it
/// had done exactly that. Each rejection then became another event in its
/// window: 947 seen against its neighbours' 506.
///
/// # Why repairing here is not papering over it
///
/// The model's **content** is right and only its encoding is wrong, and the
/// repair is exact rather than a guess: JSON defines the escape for every
/// control character, so a raw one has one correct reading and no other. This
/// recovers what was meant, byte for byte, and it is the same leniency any
/// parser applies to a producer it does not control.
///
/// It is not the whole answer. The span should not be able to emit an
/// unescapable byte in the first place, and that lives in the stencil's
/// free-text mask — a shared change with its own blast radius. This makes the
/// engine correct today and does not depend on that landing.
///
/// Outside a string a newline is structural — it is what separates one call
/// from the next — so only the inside is touched. Borrowed and untouched when
/// there is nothing to escape, which is every ordinary decode.
fn escape_control_in_strings(s: &str) -> std::borrow::Cow<'_, str> {
    if !needs_repair(s) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut in_string = false;
    let mut escaped = false;
    let mut out = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        if escaped {
            escaped = false;
            out.push(c);
            continue;
        }
        match c {
            '\\' if in_string => {
                escaped = true;
                out.push(c);
            }
            '"' => {
                in_string = !in_string;
                out.push(c);
            }
            // The whole point: a control character inside a string, written as
            // the escape JSON requires.
            c if in_string && (c as u32) < 0x20 => match c {
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                other => out.push_str(&format!("\\u{:04x}", other as u32)),
            },
            c => out.push(c),
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Whether any control character actually sits inside a string.
///
/// The scan above rewrites nothing when this is false, so an ordinary decode —
/// which is all of them — pays one pass and no allocation.
fn needs_repair(s: &str) -> bool {
    let (mut in_string, mut escaped) = (false, false);
    for c in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            c if in_string && (c as u32) < 0x20 => return true,
            _ => {}
        }
    }
    false
}

/// One call's JSON, from either shape a character may emit.
///
/// Both are one object naming a tool and its arguments; they differ in where the
/// name lives and whether the arguments are nested. Normalised here to the flat
/// form the rest of this module works in, so nothing downstream has to know
/// which the decode used.
fn flatten(obj: serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    let mut obj = obj;
    // `{"name": "say", "arguments": {"intent": "…"}}` — the stencil's shape.
    let Some(Value::String(name)) = obj.remove("name") else {
        return obj;
    };
    let mut flat = match obj.remove("arguments") {
        Some(Value::Object(args)) => args,
        // A call with a name and no arguments is a call with no arguments —
        // `wait` takes none — not a malformed one.
        _ => serde_json::Map::new(),
    };
    flat.insert("tool".to_string(), Value::String(name));
    flat
}

/// The function-block markers — see [`function_blocks_to_lines`].
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

/// Rewrite Qwen3.5's function-block calls into the flat one-object-per-line
/// form the rest of this module already understands.
///
/// # Why translate rather than parse twice
///
/// A call arrives in one of two syntaxes now — see
/// `candle_transformers::models::dialect::CallStyle` — and only the *surface*
/// differs. Both name an act and give it named arguments; everything after
/// that is identical, and all of it is the part with the teeth: unknown act,
/// missing required parameter, a value that is not a string. Parsing each
/// shape end to end would mean two copies of those checks, free to disagree,
/// and the disagreement would show up as one syntax quietly accepting a call
/// the other rejects.
///
/// So the block shape is normalised into the line shape and handed to the one
/// walk. What comes out the far end is the same `Parsed`, with the same
/// rejections, whichever syntax the model used.
///
/// # The values are raw, and stay raw
///
/// A `<parameter>` body is unescaped text that may hold quotes and newlines —
/// that is the whole point of the syntax. `serde_json` does the escaping when
/// the object is written, so a value survives the round trip intact rather
/// than being cut at its first quote.
///
/// `None` when the text holds no function block at all, so the ordinary path
/// costs one `find` and allocates nothing.
fn function_blocks_to_lines(s: &str) -> Option<String> {
    if !s.contains(FN_OPEN) {
        return None;
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find(FN_OPEN) {
        // Anything before the block is narration, and is kept — a character
        // that wrote a sentence before its call still wrote it.
        out.push_str(&rest[..at]);
        let after = &rest[at + FN_OPEN.len()..];
        // An unterminated block is left as text rather than guessed at: a
        // truncated decode is narration, not a call whose arguments we invent.
        let (Some(name_end), Some(body_end)) = (after.find('>'), after.find(FN_CLOSE)) else {
            out.push_str(rest);
            return Some(out);
        };
        if name_end > body_end {
            out.push_str(rest);
            return Some(out);
        }
        let name = after[..name_end].trim();
        let body = &after[name_end + 1..body_end];

        let mut obj = serde_json::Map::new();
        obj.insert("tool".to_string(), Value::String(name.to_string()));
        let mut scan = body;
        while let Some(p) = scan.find(PARAM_OPEN) {
            let tail = &scan[p + PARAM_OPEN.len()..];
            let Some(key_end) = tail.find('>') else {
                break;
            };
            // **`</function>` closes an open parameter.** The body is already
            // bounded by it, so a value with no `</parameter>` of its own runs
            // to the end of the body rather than being thrown away.
            //
            // Not tolerance for sloppiness — it is the shape the grammar
            // actually produces when a value span ends on an intercepted EOS.
            // The closing tag is *consumed by the span's terminator* rather
            // than injected by the tree, so a span that ends any other way
            // never writes one, and the parameter list is closed by
            // `</function>` instead. Dropping the value there cost the whole
            // act: a `reflect` whose last argument was cut short came back as
            // "needed `my_reflections` and did not have it", and the two
            // arguments the character had written were discarded with it.
            let val_end = tail.find(PARAM_CLOSE).unwrap_or(tail.len());
            if key_end > val_end {
                break;
            }
            let key = tail[..key_end].trim();
            // The element's own newlines are framing rather than content: the
            // template writes `<parameter=k>\n` before the value and `\n` after
            // it. Trimming exactly those keeps a value that deliberately ends
            // in a blank line, which `trim()` would eat.
            let value = tail[key_end + 1..val_end]
                .strip_prefix('\n')
                .unwrap_or(&tail[key_end + 1..val_end]);
            let value = value.strip_suffix('\n').unwrap_or(value);
            if !key.is_empty() {
                obj.insert(key.to_string(), Value::String(value.to_string()));
            }
            scan = &tail[(val_end + PARAM_CLOSE.len()).min(tail.len())..];
        }
        out.push('\n');
        out.push_str(&Value::Object(obj).to_string());
        out.push('\n');
        rest = &after[body_end + FN_CLOSE.len()..];
    }
    out.push_str(rest);
    Some(out)
}

/// Parse a decode into acts.
pub fn parse(output: &str) -> Parsed {
    let mut out = Parsed::default();
    let mut narration: Vec<&str> = Vec::new();

    // **Repair before anything splits on a newline.** A raw control character
    // inside a string argument is both invalid JSON and — because the walk
    // below is line-based — enough to cut one object into two unparseable
    // halves. Escaping it first puts the object back on one line, so everything
    // downstream sees the ordinary shape. See [`escape_control_in_strings`].
    // **Function blocks first, before anything assumes a line is JSON.** Their
    // values are raw text and may hold newlines, so a line-based repair run
    // over them would be working on fragments of a value. Translated here, the
    // rest of this function sees the one shape it has always seen. See
    // [`function_blocks_to_lines`].
    let output = match function_blocks_to_lines(output) {
        Some(translated) => translated,
        None => output.to_string(),
    };
    let output = escape_control_in_strings(&output);

    // The envelope is stripped before the line walk rather than inside it,
    // because a `<tool_call>` block spans lines: the markers and the object sit
    // on three lines of their own. Removing just the markers leaves the object
    // on its own line, which is what the walk below already understands.
    let output = &output.replace(CALL_OPEN, "\n").replace(CALL_CLOSE, "\n");

    for raw in output.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // A model asked for JSON will sometimes fence it. Strip the fence
        // markers rather than treating them as narration — they are neither.
        if line.starts_with("```") {
            continue;
        }
        if !looks_like_call(line) {
            narration.push(raw);
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Err(_) => out.rejected.push(Rejected::NotJson {
                line: line.to_string(),
            }),
            Ok(Value::Object(obj)) => {
                let mut obj = flatten(obj);
                let Some(name) = obj.remove("tool").and_then(|v| match v {
                    Value::String(s) => Some(s),
                    _ => None,
                }) else {
                    out.rejected.push(Rejected::NoTool {
                        line: line.to_string(),
                    });
                    continue;
                };
                let Some(tool) = tools::by_name(&name) else {
                    out.rejected.push(Rejected::UnknownTool { tool: name });
                    continue;
                };
                // A missing required parameter is a rejection, not a
                // best-effort act. `speak` with no intent is not a quieter
                // `speak`; it is a call the character did not finish making.
                if let Some(p) = tool
                    .params
                    .iter()
                    .find(|p| p.required && !obj.contains_key(p.name))
                {
                    out.rejected.push(Rejected::MissingParam {
                        tool: tool.name,
                        param: p.name,
                    });
                    continue;
                }
                out.acts.push(Act {
                    tool: tool.name,
                    args: obj,
                });
            }
            // Valid JSON that is not an object — a bare string or number on its
            // own line. Not a call, and not prose either.
            Ok(_) => out.rejected.push(Rejected::NoTool {
                line: line.to_string(),
            }),
        }
    }

    out.narration = narration.join("\n").trim().to_string();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── the function-block syntax ───────────────────────────────────────────

    /// **The shape Qwen3.5 actually emits**, end to end.
    ///
    /// Not a hypothetical: `Dialect::qwen35` is what the shipped checkpoint
    /// resolves to, so this is the only syntax a live decode produces. Until
    /// this existed the parser stripped `<tool_call>` and looked for JSON,
    /// found an element instead, and every act became narration.
    #[test]
    fn a_function_block_is_read_as_an_act() {
        let p = parse(
            "<tool_call>\n<function=ask>\n<parameter=to>\nYaelis Vayne\n</parameter>\n\
             <parameter=about>\nwhich of the two versions she has been working from\n\
             </parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(p.acts.len(), 1, "{:?} / {:?}", p.rejected, p.narration);
        assert_eq!(p.acts[0].tool, "ask");
        assert_eq!(p.acts[0].args["to"], "Yaelis Vayne");
        assert_eq!(
            p.acts[0].args["about"],
            "which of the two versions she has been working from"
        );
    }

    /// **Several calls in one turn**, which is the whole reason the family's
    /// template loops over `tool_calls`. `ACTS_PER_TURN` is 2, and a character
    /// turning as it speaks has to be expressible.
    #[test]
    fn several_function_blocks_in_one_turn_are_several_acts() {
        let p = parse(
            "<tool_call>\n<function=gesture>\n<parameter=intent>\nstop talking\n</parameter>\n\
             </function>\n</tool_call>\n\
             <tool_call>\n<function=move_to>\n<parameter=destination>\nthe green room\n\
             </parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(p.acts.len(), 2, "{:?}", p.rejected);
        assert_eq!(p.acts[0].tool, "gesture");
        assert_eq!(p.acts[1].tool, "move_to");
        assert_eq!(p.acts[1].args["destination"], "the green room");
    }

    /// **A raw value keeps its quotes and its newlines.**
    ///
    /// The gain the syntax exists for, and the thing the JSON shapes cannot do:
    /// `inner_thoughts` is prose, and prose has punctuation in it. The
    /// equivalent JSON call is the live failure recorded below — a newline in a
    /// value cut the object in half.
    #[test]
    fn a_function_block_value_survives_quotes_and_newlines() {
        let p = parse(
            "<tool_call>\n<function=reflect>\n<parameter=inner_thoughts>\n\
             it heard me. It said \"no\" and meant it.\nI am not going to ask twice.\n\
             </parameter>\n<parameter=feeling>\nwary\n</parameter>\n\
             <parameter=my_reflections>\nnothing has settled\n</parameter>\n\
             </function>\n</tool_call>",
        );
        assert_eq!(p.acts.len(), 1, "{:?}", p.rejected);
        assert_eq!(
            p.acts[0].args["inner_thoughts"],
            "it heard me. It said \"no\" and meant it.\nI am not going to ask twice."
        );
        assert_eq!(p.acts[0].args["feeling"], "wary");
    }

    /// The checks with teeth run on both syntaxes, because there is one walk.
    /// A block naming no real act, or missing a required argument, is rejected
    /// exactly as its JSON twin would be.
    #[test]
    fn a_function_block_faces_the_same_checks_as_a_json_call() {
        let unknown = parse(
            "<tool_call>\n<function=teleport>\n<parameter=to>\nthe moon\n</parameter>\n\
             </function>\n</tool_call>",
        );
        assert!(unknown.acts.is_empty());
        assert!(
            matches!(&unknown.rejected[..], [Rejected::UnknownTool { tool }] if tool == "teleport"),
            "{:?}",
            unknown.rejected
        );

        // `tell` needs both `to` and `intent`; one of them is not a quieter
        // `tell`, it is a call the character did not finish making.
        let short = parse(
            "<tool_call>\n<function=tell>\n<parameter=to>\nMaker-02\n</parameter>\n\
             </function>\n</tool_call>",
        );
        assert!(short.acts.is_empty());
        assert!(
            matches!(&short.rejected[..], [Rejected::MissingParam { .. }]),
            "{:?}",
            short.rejected
        );
    }

    /// **A value cut short by an intercepted EOS still lands.**
    ///
    /// The shape the grammar actually emits when the model stops mid-argument:
    /// the stencil swallows the EOS and injects `</function></tool_call>`, but
    /// the value span's own `</parameter>` is *consumed by its terminator*
    /// rather than injected by the tree — so a span that ended any other way
    /// never writes one, and the parameter list is closed by `</function>`.
    ///
    /// Dropping that argument cost the whole act. A live cast produced
    /// *"Your `reflect` needed `my_reflections` and did not have it"* on turn
    /// after turn, discarding two arguments the character had written in full
    /// along with the third.
    #[test]
    fn a_last_argument_closed_by_the_function_tag_is_still_read() {
        let p = parse(
            "<tool_call>\n<function=reflect>\n<parameter=inner_thoughts>\n\
             the box has given up a fold\n</parameter>\n<parameter=feeling>\nweary\n</parameter>\n\
             <parameter=my_reflections>\nnothing here is being kept</function>\n</tool_call>",
        );
        assert_eq!(p.acts.len(), 1, "{:?} / {:?}", p.rejected, p.narration);
        assert_eq!(p.acts[0].tool, "reflect");
        assert_eq!(p.acts[0].args["feeling"], "weary");
        assert_eq!(
            p.acts[0].args["my_reflections"], "nothing here is being kept",
            "the argument the function tag closed was dropped"
        );
    }

    /// A truncated block is narration, not a call with invented arguments. A
    /// decode that ran out of budget mid-element has not said anything the
    /// world should act on.
    #[test]
    fn an_unterminated_function_block_is_not_guessed_at() {
        let p = parse("<tool_call>\n<function=say>\n<parameter=intent>\nhalf a thoug");
        assert!(p.acts.is_empty(), "{:?}", p.acts);
        assert!(p.rejected.is_empty(), "{:?}", p.rejected);
    }

    /// Prose before a call is still prose, and the call still lands — the two
    /// are separated rather than one swallowing the other.
    #[test]
    fn narration_around_a_function_block_is_kept_apart_from_it() {
        let p = parse(
            "I should say something.\n\
             <tool_call>\n<function=say>\n<parameter=intent>\nthat I am here\n</parameter>\n\
             </function>\n</tool_call>",
        );
        assert_eq!(p.acts.len(), 1, "{:?}", p.rejected);
        assert!(
            p.narration.contains("I should say something"),
            "{:?}",
            p.narration
        );
    }

    /// **The JSON syntax still parses.** Other families are on it, and the
    /// translation must not have become the only path.
    #[test]
    fn the_json_syntax_is_untouched_by_the_block_translation() {
        let p = parse("{\"tool\":\"move_to\",\"destination\":\"the green room\"}");
        assert_eq!(p.acts.len(), 1, "{:?}", p.rejected);
        assert_eq!(p.acts[0].tool, "move_to");

        let wrapped = parse(
            "<tool_call>\n{\"name\": \"say\", \"arguments\": {\"intent\": \"that I am here\"}}\n\
             </tool_call>",
        );
        assert_eq!(wrapped.acts.len(), 1, "{:?}", wrapped.rejected);
        assert_eq!(wrapped.acts[0].args["intent"], "that I am here");
    }

    /// **A newline inside a string argument halves the call.**
    ///
    /// This is the live failure, reproduced. One character emitted this every
    /// turn for hours: the grammar steered it correctly — right tool, right
    /// addressee off the company list — and then the free-text span took a raw
    /// newline as its first byte. `parse` walks *lines*, so the object arrived
    /// cut in two, and what reached the character was a rejection telling it to
    /// put "one JSON object on a line" when it had done exactly that.
    ///
    /// It cannot recover from that advice, so it repeats, and every rejection is
    /// another event in its window: the failing character had seen 947 events
    /// against its neighbours' 506.
    ///
    /// A raw control character is not legal in a JSON string either — `serde_json`
    /// refuses it on one line as readily as across two — so this is malformed at
    /// the source and the span is what has to stop producing it.
    #[test]
    fn a_newline_inside_a_string_argument_still_lands() {
        let p = parse(
            "<tool_call>\n{\"name\": \"ask\", \"arguments\": {\"to\": \"Yaelis Vayne\", \
             \"about\": \"\nwhat the remaining duration is\"}}\n</tool_call>",
        );
        assert_eq!(p.rejected, Vec::new(), "{:?}", p.rejected);
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "ask");
        assert_eq!(p.acts[0].args["to"], "Yaelis Vayne");
        // The newline was content, and it is kept as content — the repair is an
        // encoding fix, not a rewrite of what the character said.
        assert_eq!(p.acts[0].args["about"], "\nwhat the remaining duration is");
    }

    /// Every control character JSON forbids raw, not only the newline that
    /// happened to be live. A tab and a `\0` are the same defect.
    #[test]
    fn every_raw_control_character_is_escaped_rather_than_rejected() {
        for (raw, want) in [
            ("\ttabbed", "\ttabbed"),
            ("\rcarriage", "\rcarriage"),
            ("\u{1}start of heading", "\u{1}start of heading"),
        ] {
            let p = parse(&format!(
                "{{\"name\": \"say\", \"arguments\": {{\"intent\": \"{raw}\"}}}}"
            ));
            assert_eq!(p.rejected, Vec::new(), "{raw:?}: {:?}", p.rejected);
            assert_eq!(p.acts.len(), 1, "{raw:?}");
            assert_eq!(p.acts[0].args["intent"], want, "{raw:?}");
        }
    }

    /// **Outside a string a newline is structural** — it is what separates one
    /// call from the next — so the repair must not touch it. Two calls on two
    /// lines have to stay two calls.
    #[test]
    fn a_newline_between_calls_is_left_alone() {
        let p = parse(
            "{\"name\": \"say\", \"arguments\": {\"intent\": \"one\"}}\n\
             {\"name\": \"say\", \"arguments\": {\"intent\": \"two\"}}",
        );
        assert_eq!(p.rejected, Vec::new(), "{:?}", p.rejected);
        assert_eq!(p.acts.len(), 2);
    }

    /// An escaped quote does not end the string, so the scan must track it —
    /// otherwise everything after `\"` is read as being outside a string and a
    /// later newline goes unrepaired.
    #[test]
    fn an_escaped_quote_does_not_end_the_string() {
        let p = parse(
            "{\"name\": \"say\", \"arguments\": {\"intent\": \"he said \\\"go\\\"\nand went\"}}",
        );
        assert_eq!(p.rejected, Vec::new(), "{:?}", p.rejected);
        assert_eq!(p.acts[0].args["intent"], "he said \"go\"\nand went");
    }

    /// The ordinary decode — every one of them — pays a scan and no allocation.
    #[test]
    fn a_clean_decode_is_not_rewritten() {
        let clean = "{\"name\": \"say\", \"arguments\": {\"intent\": \"nothing to repair\"}}";
        assert!(matches!(
            escape_control_in_strings(clean),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// **The shape the stencil emits parses.**
    ///
    /// The constrained decode writes this envelope itself — `<tool_call>`, then
    /// `{"name": …, "arguments": {…}}` — so a parser that only knew the bare
    /// line would reject every call the grammar produced. It did: the catalog
    /// compiled and the registry armed, and the trigger never fired because the
    /// prompt taught a format the marker does not appear in.
    #[test]
    fn the_stencils_own_envelope_parses() {
        let p = parse(
            "<tool_call>\n{\"name\": \"tell\", \"arguments\": {\"to\": \"Wyneth Vayne\", \
             \"intent\": \"that the gap is filled\"}}\n</tool_call>",
        );
        assert_eq!(p.rejected, Vec::new(), "{:?}", p.rejected);
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "tell");
        assert_eq!(p.acts[0].args["to"], "Wyneth Vayne");
        assert_eq!(p.acts[0].args["intent"], "that the gap is filled");
        // The envelope is not narration — it is framing the grammar wrote.
        assert!(p.narration.is_empty(), "{:?}", p.narration);
    }

    /// An empty `arguments` is a **call**, refused for what it is missing —
    /// never mistaken for malformed JSON.
    ///
    /// The distinction is the whole reason `rejected` carries a reason: the
    /// character is told "your `read` needed `what`", which it can act on,
    /// rather than "that did not come out as a call", which it cannot. No act
    /// in the catalog takes no arguments any more — `wait` was the last, and
    /// every act now makes the character name something — so this is the shape
    /// an empty envelope actually has.
    #[test]
    fn an_envelope_with_no_arguments_is_still_a_call() {
        let p = parse("<tool_call>\n{\"name\": \"read\", \"arguments\": {}}\n</tool_call>");
        assert!(p.acts.is_empty(), "{:?}", p.acts);
        assert_eq!(
            p.rejected,
            vec![Rejected::MissingParam {
                tool: "read",
                param: "what"
            }],
            "an empty call must be refused for its parameter, not for its shape"
        );
    }

    /// Both shapes in one decode, because a model mid-transition emits both and
    /// neither should be the one that silently fails.
    #[test]
    fn the_bare_line_and_the_envelope_both_still_work() {
        let p = parse(
            "{\"tool\":\"read\",\"what\":\"the muster board\"}\n\
             <tool_call>\n{\"name\": \"say\", \"arguments\": {\"intent\": \"that I heard it\"}}\n</tool_call>",
        );
        assert_eq!(p.rejected, Vec::new(), "{:?}", p.rejected);
        let tools: Vec<&str> = p.acts.iter().map(|a| a.tool).collect();
        assert_eq!(tools, vec!["read", "say"]);
    }

    /// A required parameter is still required in the envelope — the grammar
    /// makes it unreachable, and a decode that arrives without one anyway is
    /// still refused rather than half-performed.
    #[test]
    fn a_missing_required_argument_is_refused_in_either_shape() {
        for text in [
            r#"{"tool":"tell","intent":"that I am here"}"#,
            "<tool_call>\n{\"name\": \"tell\", \"arguments\": {\"intent\": \"that I am here\"}}\n</tool_call>",
        ] {
            let p = parse(text);
            assert!(p.acts.is_empty(), "{text}");
            assert!(
                matches!(
                    p.rejected.first(),
                    Some(Rejected::MissingParam { tool: "tell", param: "to" })
                ),
                "{text}: {:?}",
                p.rejected
            );
        }
    }

    #[test]
    fn a_single_call_parses() {
        let p = parse(r#"{"tool":"say","intent":"that I will not"}"#);
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "say");
        assert_eq!(p.acts[0].args["intent"], "that I will not");
        assert!(p.rejected.is_empty());
        assert!(p.narration.is_empty());
    }

    #[test]
    fn several_calls_keep_their_order() {
        let p = parse(
            "{\"tool\":\"read\",\"what\":\"the muster board\"}\n\
             {\"tool\":\"say\",\"intent\":\"that someone is coming\"}",
        );
        assert_eq!(
            p.acts.iter().map(|a| a.tool).collect::<Vec<_>>(),
            vec!["read", "say"]
        );
    }

    /// **Narration is not an act.** The model will write "I turn and say:"
    /// around its calls; the world must never see it, and it must still be
    /// visible to a person reading the feed.
    #[test]
    fn prose_around_calls_is_kept_and_not_acted_on() {
        let p = parse(
            "I have had enough of this.\n\
             {\"tool\":\"say\",\"intent\":\"that I am not handing over the ledger\",\"manner\":\"final\"}\n\
             He will not like that.",
        );
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "say");
        assert_eq!(
            p.narration,
            "I have had enough of this.\nHe will not like that."
        );
    }

    /// A model asked for JSON fences it about a third of the time. The fence is
    /// neither an act nor narration.
    #[test]
    fn a_fenced_block_is_unwrapped() {
        let p = parse("```json\n{\"tool\":\"read\",\"what\":\"the muster board\"}\n```");
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "read");
        assert!(
            p.narration.is_empty(),
            "the fence became narration: {:?}",
            p.narration
        );
    }

    /// **The failure this module exists for.** A character failing to act and a
    /// character choosing not to act look identical from the outside and need
    /// completely different fixes, so a malformed call is reported.
    #[test]
    fn a_malformed_call_is_rejected_rather_than_dropped() {
        let p = parse(r#"{"tool":"say","intent":}"#);
        assert!(p.acts.is_empty());
        assert_eq!(p.rejected.len(), 1);
        assert!(matches!(p.rejected[0], Rejected::NotJson { .. }));
    }

    /// An invented tool is the clearest signal that the prompt and the
    /// vocabulary have drifted apart, so the name is carried.
    #[test]
    fn an_invented_tool_is_named_in_the_rejection() {
        let p = parse(r#"{"tool":"teleport","to":"the keep"}"#);
        assert!(p.acts.is_empty());
        assert_eq!(
            p.rejected[0],
            Rejected::UnknownTool {
                tool: "teleport".into()
            }
        );
    }

    /// `speak` with no intent is not a quieter `speak` — it is a call the
    /// character did not finish making.
    #[test]
    fn a_missing_required_parameter_is_a_rejection_not_a_best_effort_act() {
        let p = parse(r#"{"tool":"say","manner":"flatly"}"#);
        assert!(p.acts.is_empty());
        assert_eq!(
            p.rejected[0],
            Rejected::MissingParam {
                tool: "say",
                param: "intent"
            }
        );
    }

    /// **Every rejection has something to say to the character.**
    ///
    /// A rejection that is only logged is a character acting into silence: it
    /// does not know the act failed, so it has no reason to do anything
    /// differently and makes the same malformed call every turn for as long as
    /// it runs. That happened — `tell` without a `to`, once every four seconds,
    /// each one a warning in a log nobody in the world can read.
    #[test]
    fn every_rejection_reads_as_something_the_character_can_act_on() {
        let all = [
            Rejected::NotJson { line: "{\"".into() },
            Rejected::NoTool {
                line: "{\"intent\":\"x\"}".into(),
            },
            Rejected::UnknownTool {
                tool: "speak".into(),
            },
            Rejected::MissingParam {
                tool: "tell",
                param: "to",
            },
        ];
        for r in all {
            let line = r.line();
            assert!(!line.is_empty(), "{r:?} says nothing");
            assert!(line.ends_with('.'), "{r:?}: {line}");
            // Written to the character, so it must not carry the machinery's
            // own vocabulary into the one place the model reads.
            for leak in ["Rejected", "MissingParam", "NotJson", "{\"", "Err("] {
                assert!(!line.contains(leak), "{r:?} leaked {leak:?}: {line}");
            }
        }

        // And it names the thing that was wrong, so the next attempt can differ.
        let missing = Rejected::MissingParam {
            tool: "tell",
            param: "to",
        }
        .line();
        assert!(
            missing.contains("tell") && missing.contains("to"),
            "{missing}"
        );
        let unknown = Rejected::UnknownTool {
            tool: "teleport".into(),
        }
        .line();
        assert!(unknown.contains("teleport"), "{unknown}");
    }

    /// An optional parameter's absence is fine — that is what optional means.
    #[test]
    fn an_absent_optional_parameter_is_not_a_rejection() {
        let p = parse(r#"{"tool":"say","intent":"hello"}"#);
        assert_eq!(p.acts.len(), 1);
        assert!(p.rejected.is_empty());
    }

    #[test]
    fn an_object_without_a_tool_key_is_rejected() {
        let p = parse(r#"{"intent":"something"}"#);
        assert!(p.acts.is_empty());
        assert_eq!(p.rejected.len(), 1);
        assert!(matches!(p.rejected[0], Rejected::NoTool { .. }));
    }

    /// Only a line starting with `{` is *tried* as a call. A bare string or
    /// number is prose that happens to be valid JSON — a model writing `42` on
    /// its own line has narrated, not acted, and reporting it as a malformed
    /// call would fill the rejection list with things nobody meant as acts.
    #[test]
    fn json_that_is_not_an_object_is_narration() {
        for line in [r#""just a string""#, "42", "true"] {
            let p = parse(line);
            assert!(p.acts.is_empty(), "{line} produced an act");
            assert!(
                p.rejected.is_empty(),
                "{line} was reported as a failed call"
            );
            assert_eq!(p.narration, line);
        }
    }

    /// One bad line must not cost the good ones. Generation stopping mid-token
    /// is common, and losing the whole batch to a truncated last line would
    /// make every interrupted decode a total loss.
    #[test]
    fn a_truncated_last_line_costs_only_itself() {
        let p = parse(
            "{\"tool\":\"read\",\"what\":\"the ridge survey\"}\n\
             {\"tool\":\"spea",
        );
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "read");
        assert_eq!(p.rejected.len(), 1);
    }

    /// The summary renders in the tool's declared parameter order.
    /// `serde_json::Map` is a BTreeMap, so without that it would come out
    /// alphabetically and `manner` would precede `intent`.
    #[test]
    fn a_summary_follows_the_tools_parameter_order_not_the_maps() {
        // `tell`, because it is the one that declares `to` before `manner` —
        // the order under test is the tool's, not the map's.
        let p = parse(r#"{"tool":"tell","manner":"flatly","to":"Hess","intent":"that I refuse"}"#);
        let s = p.acts[0].summary();
        // Named, because three bare values leave a reader guessing which is
        // which — see [`Act::summary`].
        assert_eq!(s, "tell — to: Hess; intent: that I refuse; manner: flatly");
        let intent = s.find("that I refuse").unwrap();
        let manner = s.find("flatly").unwrap();
        assert!(
            intent < manner,
            "alphabetical order leaked into the summary: {s}"
        );
    }

    /// **One argument stays bare.** A label in front of the only thing worth
    /// reading is noise, and most acts a character takes have exactly one.
    #[test]
    fn a_single_argument_needs_no_label_to_be_understood() {
        let p = parse(r#"{"tool":"say","intent":"the redoubt burned twice"}"#);
        assert_eq!(p.acts[0].summary(), "say — the redoubt burned twice");
    }

    /// Built directly rather than parsed: no act in the catalog takes no
    /// arguments any more, so this property has no witness the parser would
    /// accept — but it is still the property, and a future act with only
    /// optional parameters would land on it.
    #[test]
    fn an_argumentless_act_summarises_as_its_name() {
        let a = Act {
            tool: "read",
            args: serde_json::Map::new(),
        };
        assert_eq!(a.summary(), "read");
    }

    /// A decode with nothing in it is not an error — a character can be handed
    /// a moment and have nothing to say about it.
    #[test]
    fn an_empty_decode_is_empty_not_broken() {
        let p = parse("   \n\n  ");
        assert!(p.is_empty());
        assert!(p.rejected.is_empty());
        assert!(p.narration.is_empty());
    }

    /// Every tool in the catalog must round-trip through its own example — the
    /// examples are what calibration prefills, so a format the parser rejects
    /// would be taught to the model as correct.
    #[test]
    fn every_catalog_example_parses_as_an_act() {
        for t in tools::CATALOG.iter() {
            for e in t.examples {
                let mut obj: Map<String, Value> = serde_json::from_str(e.call).unwrap();
                obj.insert("tool".into(), Value::String(t.name.into()));
                let line = serde_json::to_string(&Value::Object(obj)).unwrap();
                let p = parse(&line);
                assert_eq!(
                    p.acts.len(),
                    1,
                    "{}'s example does not parse as an act: {:?}",
                    t.name,
                    p.rejected
                );
                assert_eq!(p.acts[0].tool, t.name);
            }
        }
    }
}
