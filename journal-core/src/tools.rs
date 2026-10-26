//! The journal's calls, the grammar they are held to, and how an answer is read
//! back.
//!
//! A draft is answered with one call: `journal_write`, the entry, or — when the
//! journal already holds an entry to weigh the stretch against —
//! `journal_skip`, the reason there is nothing to add. Asking both in one turn
//! means the writer decides with the stretch in front of it, and a stretch
//! worth writing costs one answer, not a question and then an answer.
//!
//! **Neither call is ever in a catalog of things that can be done.** They reach
//! nothing. They are the shapes an answer takes when the writer is asked for its
//! entry — the question describes them and the grammar forces one — and a
//! caller that put them in its live tool catalog would start accepting them on
//! an ordinary turn.
//!
//! # The grammar
//!
//! [`write_spec`] is built per draft, because what its enums may hold depends on
//! the stretch being written up — the turns a claim may cite, and the items that
//! are open to restate or settle. A claim cannot cite a turn that is not there,
//! and an item that is not open cannot be settled, because neither is a thing the
//! grammar offers. It is the flat `{name, params}` JSON a
//! `candle_conversation::stencil::ToolSpec` deserializes from, so this crate
//! needs no engine.
//!
//! # Why `cite` is one value and not a list
//!
//! A grammar guides an object inside an array, and one array level. An array
//! inside an array element decodes free. A claim is an element of `claims`, so
//! its citations are `cite` and an optional `also_cite` — two guided enums —
//! rather than a list the model would write unconstrained.
//!
//! # Reading an answer back
//!
//! [`parse`] reads whichever call shape the dialect wrote. The JSON block is
//! grammar-guided all the way down; the function block writes raw values, so an
//! array or object there is JSON text inside an element and the same checks run
//! on it — [`crate::verify`] is the net either way.

use serde_json::{json, Map, Value};

use crate::cut::close_cut;
use crate::entry::{Item, Typed};
use crate::journal::MAX_OPEN;
use crate::json_repair::escape_control_in_strings;
use crate::verify::{Draft, DraftClaim, DraftOpen, Relates, MAX_CLAIMS, MAX_INTEND};

/// The entry itself.
pub const WRITE: &str = "journal_write";
/// Nothing to add, and why.
pub const SKIP: &str = "journal_skip";

/// What one answer to a draft said.
#[derive(Debug, PartialEq)]
pub enum Answer {
    /// An entry, still to be checked.
    Write(Draft),
    /// The stretch adds nothing to the journal, for this reason — empty when
    /// none was given.
    Skip(String),
}

pub(crate) fn text(name: &str, required: bool) -> Value {
    json!({ "name": name, "type": "string", "required": required })
}

fn choice(name: &str, required: bool, values: &[String]) -> Value {
    json!({ "name": name, "type": "string", "required": required, "enum": values })
}

/// A list of `item`, between `min_items` and `max_items` long.
///
/// **The most is the grammar's to hold, not only the check's.** Held only by
/// [`crate::verify`], a writer that went on past it — a worker whose research
/// turned up fifteen findings — had its whole entry refused, every valid claim
/// with it, and with one round to write in, kept nothing at all. Offered no
/// element past the most, it closes the list and keeps the entry.
fn list_of(name: &str, required: bool, min_items: usize, max_items: usize, item: Value) -> Value {
    json!({
        "name": name,
        "type": "array",
        "required": required,
        "min_items": min_items,
        "max_items": max_items,
        "items": item,
    })
}

fn object(name: &str, required: bool, properties: Vec<Value>) -> Value {
    json!({ "name": name, "type": "object", "required": required, "properties": properties })
}

/// The fewest claims an entry about `citable` turns is held to: one for every
/// two of them, at least one and at most [`CLAIM_FLOOR_MAX`].
///
/// **The grammar asks for them because the question alone does not.** Decoded
/// greedily, a writer offered "another claim, or close the list" closes it:
/// live drafts over a request, a file read and an answer kept one claim — the
/// request — however the question asked for one per step. A floor of half the
/// citable turns is what a stretch of steps establishes at the least, since
/// each step is a call and its result.
pub fn claim_floor(citable: usize) -> usize {
    citable.div_ceil(2).clamp(1, CLAIM_FLOOR_MAX)
}

/// The most claims the grammar insists on, however long the stretch.
pub const CLAIM_FLOOR_MAX: usize = 6;

/// What a claim may carry beside its sentence and the turns it cites.
#[derive(Clone, Copy, Debug)]
pub struct Claims<'a> {
    /// A typed fact the world is asked about ([`crate::style::Style::facts`]):
    /// offered only where the world can read what claims are about. Left out,
    /// the call has no slot for one, so none can be written.
    pub facts: bool,
    /// The entries of other journals a claim may rest on, by address
    /// ([`crate::verify::World::entries`]): `per` offers exactly these, and
    /// is left out when there are none.
    pub entries: &'a [String],
}

/// The `journal_write` call, for the stretch whose turns are `citable`, with
/// `open` items standing, its claims shaped by `claims`. It holds at least
/// [`claim_floor`] claims.
///
/// `None` when nothing in the stretch can be cited: there is no entry to write,
/// and a `cite` with no value to choose would be a grammar that cannot be
/// satisfied.
pub fn write_spec(citable: &[u64], open: &[Item], claims: Claims<'_>) -> Option<Value> {
    if citable.is_empty() {
        return None;
    }
    let ids: Vec<String> = citable.iter().map(u64::to_string).collect();
    let mut relations = vec!["new".to_string()];
    for item in open {
        relations.push(format!("restates {}", item.id));
        relations.push(format!("resolves {}", item.id));
    }

    let mut properties = vec![
        text("text", true),
        choice("cite", true, &ids),
        choice("also_cite", false, &ids),
    ];
    if !claims.entries.is_empty() {
        properties.push(choice("per", false, claims.entries));
    }
    if claims.facts {
        properties.push(object(
            "typed",
            false,
            vec![
                text("subject", true),
                text("attribute", true),
                text("value", true),
            ],
        ));
    }
    let claim = object("", true, properties);
    let item = object(
        "",
        true,
        vec![text("text", true), choice("relates", true, &relations)],
    );
    Some(json!({
        "name": WRITE,
        "params": [
            list_of("claims", true, claim_floor(citable.len()), MAX_CLAIMS, claim),
            text("summary", true),
            list_of("intend", false, 0, MAX_INTEND, text("", true)),
            list_of("open", false, 0, MAX_OPEN, item),
        ],
    }))
}

/// The `journal_skip` call: one sentence on why the stretch adds nothing.
pub fn skip_spec() -> Value {
    json!({ "name": SKIP, "params": [text("reason", true)] })
}

/// The calls a draft's answer may be, in the flat `{name, params}` form: the
/// write, its claims shaped by `claims`, and the skip when `may_skip`. `None`
/// when nothing can be cited, as for [`write_spec`].
pub fn draft_specs(
    citable: &[u64],
    open: &[Item],
    may_skip: bool,
    claims: Claims<'_>,
) -> Option<Vec<Value>> {
    let mut specs = vec![write_spec(citable, open, claims)?];
    if may_skip {
        specs.push(skip_spec());
    }
    Some(specs)
}

/// The call's name and arguments, from whichever shape was written.
pub fn arguments(answer: &str) -> Option<(String, Map<String, Value>)> {
    json_call(answer).or_else(|| element_call(answer))
}

/// `{"name": …, "arguments": {…}}` — the first JSON object in the answer.
fn json_call(answer: &str) -> Option<(String, Map<String, Value>)> {
    let start = answer.find('{')?;
    let repaired = escape_control_in_strings(&answer[start..]);
    let call: Value = serde_json::Deserializer::from_str(&repaired)
        .into_iter::<Value>()
        .next()?
        .ok()?;
    let name = call.get("name")?.as_str()?.to_string();
    let args = match call.get("arguments") {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    };
    Some((name, args))
}

/// `<function=name>` with a `<parameter=key>` element per argument. A value that
/// is JSON is read as JSON; anything else is the text it is.
fn element_call(answer: &str) -> Option<(String, Map<String, Value>)> {
    const FUNCTION: &str = "<function=";
    const PARAMETER: &str = "<parameter=";
    const PARAMETER_CLOSE: &str = "</parameter>";
    let at = answer.find(FUNCTION)? + FUNCTION.len();
    let name_end = answer[at..].find('>')?;
    let name = answer[at..at + name_end].trim().to_string();
    let mut rest = &answer[at + name_end..];
    let mut args = Map::new();
    while let Some(open) = rest.find(PARAMETER) {
        let after = &rest[open + PARAMETER.len()..];
        let key_end = after.find('>')?;
        let key = after[..key_end].trim().to_string();
        let body = &after[key_end + 1..];
        let end = body.find(PARAMETER_CLOSE).unwrap_or(body.len());
        let raw = body[..end].trim();
        let value =
            serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        args.insert(key, value);
        rest = &body[(end + PARAMETER_CLOSE.len()).min(body.len())..];
    }
    Some((name, args))
}

pub(crate) fn text_of(v: &Value) -> Option<String> {
    let s = v.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// A turn number, written as a string or as a number.
fn turn_of(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.trim().trim_start_matches('#').parse().ok(),
        _ => None,
    }
}

fn relates_of(v: &Value) -> Result<Relates, String> {
    let said = v.as_str().unwrap_or_default().trim().to_lowercase();
    if said == "new" {
        return Ok(Relates::New);
    }
    let number = |verb: &str| -> Option<u64> {
        said.strip_prefix(verb)?
            .trim()
            .trim_start_matches('#')
            .parse()
            .ok()
    };
    if let Some(id) = number("restates") {
        return Ok(Relates::Restates(id));
    }
    if let Some(id) = number("resolves") {
        return Ok(Relates::Resolves(id));
    }
    Err("`relates` is new, restates N or resolves N, where N is the number of an open item.".into())
}

fn claim_of(n: usize, v: &Value) -> Result<DraftClaim, String> {
    let at = n + 1;
    let text = v
        .get("text")
        .and_then(text_of)
        .ok_or_else(|| format!("Claim {at} has no `text`."))?;
    let mut cite = Vec::new();
    for key in ["cite", "also_cite"] {
        if let Some(raw) = v.get(key).filter(|r| !r.is_null()) {
            let turn = turn_of(raw)
                .ok_or_else(|| format!("Claim {at}: `{key}` is the number of a turn."))?;
            if !cite.contains(&turn) {
                cite.push(turn);
            }
        }
    }
    if cite.is_empty() {
        return Err(format!("Claim {at} cites no turn."));
    }
    let typed = match v.get("typed").filter(|t| !t.is_null()) {
        None => None,
        Some(t) => {
            let part = |k: &str| t.get(k).and_then(text_of);
            match (part("subject"), part("attribute"), part("value")) {
                (Some(subject), Some(attribute), Some(value)) => Some(Typed {
                    subject,
                    attribute,
                    value,
                }),
                _ => {
                    return Err(format!(
                        "Claim {at}: `typed` needs a `subject`, an `attribute` and a `value`."
                    ))
                }
            }
        }
    };
    let per = v.get("per").and_then(text_of);
    Ok(DraftClaim {
        text,
        cite,
        typed,
        per,
    })
}

fn item_of(n: usize, o: &Value) -> Result<DraftOpen, String> {
    let text = o
        .get("text")
        .and_then(text_of)
        .ok_or_else(|| format!("Open item {} has no `text`.", n + 1))?;
    let relates = relates_of(o.get("relates").unwrap_or(&Value::Null))?;
    Ok(DraftOpen { text, relates })
}

/// A list written as JSON text inside an element, read, and whether the answer
/// was cut in the middle of it — closed first, then ([`close_cut`]).
fn listed(v: &Value) -> Option<(Vec<Value>, bool)> {
    let Value::String(s) = v else {
        return None;
    };
    let s = s.trim();
    if !s.starts_with('[') {
        return None;
    }
    match serde_json::from_str(s) {
        Ok(Value::Array(list)) => Some((list, false)),
        Ok(_) => None,
        Err(_) => match serde_json::from_str(&close_cut(s)?.text) {
            Ok(Value::Array(list)) => Some((list, true)),
            _ => None,
        },
    }
}

/// Each of `list` read by `read`. In an answer that was `cut`, a last element
/// the cut left unreadable is dropped rather than refusing the rest.
fn each<T>(
    list: &[Value],
    cut: bool,
    read: impl Fn(usize, &Value) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let mut out = Vec::with_capacity(list.len());
    for (n, v) in list.iter().enumerate() {
        match read(n, v) {
            Ok(t) => out.push(t),
            Err(_) if cut && n + 1 == list.len() => {}
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// The draft `args` say. `cut_in` names the argument the answer was cut off
/// in, whose last element may be dropped.
fn draft_of(args: &Map<String, Value>, cut_in: Option<&str>) -> Result<Draft, String> {
    let claims = match args.get("claims") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(list)) => each(list, cut_in == Some("claims"), claim_of)?,
        Some(v) => match listed(v) {
            Some((list, cut)) => each(&list, cut, claim_of)?,
            None => return Err("`claims` is a list.".into()),
        },
    };
    let intend = match args.get("intend") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(list)) => list.iter().filter_map(text_of).collect(),
        Some(_) => return Err("`intend` is a list of short lines.".into()),
    };
    let open = match args.get("open") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(list)) => each(list, cut_in == Some("open"), item_of)?,
        Some(v) => match listed(v) {
            Some((list, cut)) => each(&list, cut, item_of)?,
            None => return Err("`open` is a list.".into()),
        },
    };
    let summary = args.get("summary").and_then(text_of).unwrap_or_default();
    Ok(Draft {
        summary,
        claims,
        intend,
        open,
    })
}

/// Read one answer, which may be a `journal_skip` only when `may_skip`. The
/// error is worded for the model, which is shown it with its refused attempt
/// and answers again.
///
/// An answer its token budget cut off is read as far as it got: closed
/// ([`close_cut`]), its last string ending in `...`, and a last claim or open
/// item the cut left incomplete dropped.
pub fn parse(answer: &str, may_skip: bool) -> Result<Answer, String> {
    let (name, args, cut_in) = match arguments(answer) {
        Some((name, args)) => (name, args, None),
        None => {
            let closed = answer
                .find('{')
                .and_then(|start| close_cut(&answer[start..]))
                .ok_or_else(|| "That was not a journal_write call.".to_string())?;
            let (name, args) = json_call(&closed.text).ok_or_else(|| {
                "That was a journal_write call cut off where it could not be closed.".to_string()
            })?;
            let cut_in = match closed.open.as_slice() {
                [outer, field, ..] if outer == "arguments" => Some(field.clone()),
                _ => None,
            };
            (name, args, cut_in)
        }
    };
    match name.as_str() {
        WRITE => draft_of(&args, cut_in.as_deref()).map(Answer::Write),
        SKIP if may_skip => Ok(Answer::Skip(
            args.get("reason").and_then(text_of).unwrap_or_default(),
        )),
        _ => Err(format!(
            "There is no call named `{name}` here; write with journal_write."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::testing::item;

    /// Claims that may carry a typed fact, with no journal above to rest on.
    const FACTS: Claims<'static> = Claims {
        facts: true,
        entries: &[],
    };

    /// **A world that cannot read what claims are about offers no fact
    /// slot**, and the journals above offer exactly their entries to rest on.
    #[test]
    fn a_claim_carries_a_fact_only_when_offered_and_rests_only_on_entries_above() {
        let names = |claims: Claims<'_>| -> Vec<String> {
            let spec = write_spec(&[1], &[], claims).unwrap();
            param(&spec, "claims")["items"]["properties"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["name"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(names(FACTS), ["text", "cite", "also_cite", "typed"]);
        let above = ["T:review#4".to_string(), "P:candle#2".to_string()];
        let workspace = Claims {
            facts: false,
            entries: &above,
        };
        assert_eq!(names(workspace), ["text", "cite", "also_cite", "per"]);
        let spec = write_spec(&[1], &[], workspace).unwrap();
        let per = property(&param(&spec, "claims")["items"], "per");
        assert_eq!(per["enum"], json!(["T:review#4", "P:candle#2"]));
        assert_eq!(per["required"], false);
        let bare = Claims {
            facts: false,
            entries: &[],
        };
        assert_eq!(names(bare), ["text", "cite", "also_cite"]);
    }

    #[test]
    fn a_claim_resting_on_an_entry_above_is_read_with_it() {
        let said = r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "1", "per": "T:review#4"}]}}"#;
        assert_eq!(
            parse(said).unwrap().claims[0].per.as_deref(),
            Some("T:review#4")
        );
    }

    /// An answer read as an entry, where a skip is not on offer.
    fn parse(said: &str) -> Result<Draft, String> {
        match super::parse(said, false)? {
            Answer::Write(draft) => Ok(draft),
            Answer::Skip(why) => panic!("a skip that was not on offer was read: {why}"),
        }
    }

    const SKIP_SAID: &str =
        r#"{"name": "journal_skip", "arguments": {"reason": "Already in entry 2."}}"#;

    #[test]
    fn a_skip_on_offer_is_read_with_its_reason() {
        assert_eq!(
            super::parse(SKIP_SAID, true),
            Ok(Answer::Skip("Already in entry 2.".into()))
        );
        let bare = r#"{"name": "journal_skip", "arguments": {}}"#;
        assert_eq!(super::parse(bare, true), Ok(Answer::Skip(String::new())));
    }

    #[test]
    fn a_skip_not_on_offer_is_refused_by_name() {
        assert_eq!(
            super::parse(SKIP_SAID, false),
            Err("There is no call named `journal_skip` here; write with journal_write.".into())
        );
    }

    #[test]
    fn a_write_is_read_whether_or_not_a_skip_is_on_offer() {
        let said =
            r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "4"}]}}"#;
        for may_skip in [false, true] {
            assert!(
                matches!(super::parse(said, may_skip), Ok(Answer::Write(_))),
                "{may_skip}"
            );
        }
    }

    #[test]
    fn the_skip_call_asks_one_required_reason() {
        assert_eq!(
            skip_spec(),
            json!({
                "name": "journal_skip",
                "params": [{ "name": "reason", "type": "string", "required": true }],
            })
        );
    }

    #[test]
    fn a_draft_offers_the_write_and_the_skip_only_when_asked_to() {
        let write = write_spec(&[1], &[], FACTS).unwrap();
        assert_eq!(
            draft_specs(&[1], &[], false, FACTS),
            Some(vec![write.clone()])
        );
        assert_eq!(
            draft_specs(&[1], &[], true, FACTS),
            Some(vec![write, skip_spec()])
        );
        assert_eq!(draft_specs(&[], &[], true, FACTS), None);
    }

    fn param<'a>(spec: &'a Value, name: &str) -> &'a Value {
        spec["params"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == name)
            .unwrap()
    }

    fn property<'a>(object: &'a Value, name: &str) -> &'a Value {
        object["properties"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == name)
            .unwrap()
    }

    #[test]
    fn the_write_call_takes_claims_then_a_summary_then_intend_then_open() {
        let spec = write_spec(&[1, 2], &[], FACTS).unwrap();
        assert_eq!(spec["name"], WRITE);
        let names: Vec<(&str, bool)> = spec["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p["name"].as_str().unwrap(),
                    p["required"].as_bool().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            names,
            [
                ("claims", true),
                ("summary", true),
                ("intend", false),
                ("open", false)
            ]
        );
    }

    #[test]
    fn an_entry_cannot_be_written_without_a_claim() {
        let spec = write_spec(&[1, 2], &[], FACTS).unwrap();
        let min: Vec<u64> = spec["params"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["min_items"].as_u64())
            .collect();
        assert_eq!(min, [1, 0, 0], "claims, intend, open");
    }

    /// **An entry cannot be written past what the check keeps**: the grammar
    /// closes each list at the check's own most, so an entry is never refused
    /// for holding too many.
    #[test]
    fn an_entry_cannot_be_written_past_what_the_check_keeps() {
        let spec = write_spec(&[1, 2], &[], FACTS).unwrap();
        let max: Vec<u64> = spec["params"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["max_items"].as_u64())
            .collect();
        assert_eq!(
            max,
            [MAX_CLAIMS as u64, MAX_INTEND as u64, MAX_OPEN as u64],
            "claims, intend, open"
        );
    }

    #[test]
    fn the_claim_floor_is_half_the_citable_turns_from_one_to_six() {
        let floors: Vec<usize> = [0, 1, 2, 3, 4, 6, 7, 12, 13, 64]
            .into_iter()
            .map(claim_floor)
            .collect();
        assert_eq!(floors, [1, 1, 1, 2, 2, 3, 4, 6, 6, 6]);
        let spec = write_spec(&[49, 50, 51, 52], &[], FACTS).unwrap();
        assert_eq!(param(&spec, "claims")["min_items"], 2);
    }

    #[test]
    fn a_stretch_with_nothing_to_cite_has_no_writing_grammar() {
        assert!(write_spec(&[], &[item(1, "the door")], FACTS).is_none());
    }

    #[test]
    fn cite_offers_exactly_the_turns_of_the_stretch() {
        let spec = write_spec(&[7, 9, 12], &[], FACTS).unwrap();
        let element = &param(&spec, "claims")["items"];
        for key in ["cite", "also_cite"] {
            assert_eq!(
                property(element, key)["enum"],
                json!(["7", "9", "12"]),
                "{key}"
            );
        }
        assert_eq!(property(element, "cite")["required"], true);
        assert_eq!(property(element, "also_cite")["required"], false);
        assert_eq!(property(element, "typed")["required"], false);
    }

    #[test]
    fn relates_offers_new_and_restating_or_settling_each_open_item_only() {
        let spec = write_spec(&[1], &[item(3, "the valve"), item(5, "the roster")], FACTS).unwrap();
        let relates = property(&param(&spec, "open")["items"], "relates");
        assert_eq!(
            relates["enum"],
            json!([
                "new",
                "restates 3",
                "resolves 3",
                "restates 5",
                "resolves 5"
            ])
        );
    }

    #[test]
    fn with_nothing_open_the_only_relation_is_new() {
        let spec = write_spec(&[1], &[], FACTS).unwrap();
        let relates = property(&param(&spec, "open")["items"], "relates");
        assert_eq!(relates["enum"], json!(["new"]));
    }

    #[test]
    fn a_full_write_reads_into_a_draft() {
        let said = r#"{"name": "journal_write", "arguments": {
            "claims": [
              {"text": "Maker-04 said the valve is shut.", "cite": "12"},
              {"text": "The gauge read 4.", "cite": "13", "also_cite": "14",
               "typed": {"subject": "gauge", "attribute": "reading", "value": "4"}}
            ],
            "intend": ["ask Maker-04 again"],
            "open": [{"text": "who shut the valve", "relates": "restates 3"},
                     {"text": "the roster", "relates": "resolves 5"},
                     {"text": "the lift", "relates": "new"}]
        }}"#;
        let draft = parse(said).expect("reads as a write");
        assert_eq!(draft.claims.len(), 2);
        assert_eq!(draft.claims[0].cite, vec![12]);
        assert_eq!(draft.claims[0].typed, None);
        assert_eq!(draft.claims[1].cite, vec![13, 14]);
        assert_eq!(
            draft.claims[1].typed,
            Some(Typed {
                subject: "gauge".into(),
                attribute: "reading".into(),
                value: "4".into(),
            })
        );
        assert_eq!(draft.intend, vec!["ask Maker-04 again".to_string()]);
        let relations: Vec<Relates> = draft.open.iter().map(|o| o.relates).collect();
        assert_eq!(
            relations,
            vec![Relates::Restates(3), Relates::Resolves(5), Relates::New]
        );
    }

    #[test]
    fn a_write_in_the_function_shape_carries_json_in_its_elements() {
        let said = "<tool_call>\n<function=journal_write>\n\
            <parameter=claims>\n[{\"text\": \"The lift stopped.\", \"cite\": \"4\"}]</parameter>\n\
            <parameter=intend>\n[\"take the stairs\"]</parameter>\n\
            </function>\n</tool_call>";
        let draft = parse(said).expect("reads as a write");
        assert_eq!(draft.claims[0].text, "The lift stopped.");
        assert_eq!(draft.claims[0].cite, vec![4]);
        assert_eq!(draft.intend, vec!["take the stairs".to_string()]);
    }

    #[test]
    fn a_turn_may_be_written_as_a_number() {
        let said =
            r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": 4}]}}"#;
        assert_eq!(parse(said).unwrap().claims[0].cite, vec![4]);
    }

    #[test]
    fn a_repeated_turn_is_cited_once() {
        let said = r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "4", "also_cite": "4"}]}}"#;
        assert_eq!(parse(said).unwrap().claims[0].cite, vec![4]);
    }

    #[test]
    fn a_raw_newline_inside_a_claim_is_repaired_not_refused() {
        let said = "{\"name\": \"journal_write\", \"arguments\": {\"claims\": [{\"text\": \"line one\nline two\", \"cite\": \"4\"}]}}";
        assert_eq!(parse(said).unwrap().claims[0].text, "line one\nline two");
    }

    #[test]
    fn a_write_with_only_open_items_is_a_draft() {
        let said = r#"{"name": "journal_write", "arguments": {"open": [{"text": "the lift", "relates": "new"}]}}"#;
        let draft = parse(said).unwrap();
        assert!(draft.claims.is_empty());
        assert_eq!(draft.open.len(), 1);
    }

    #[test]
    fn each_malformed_write_is_refused_in_words_that_name_the_claim() {
        let cases = [
            (
                r#"{"name": "journal_write", "arguments": {"claims": [{"cite": "1"}]}}"#,
                "Claim 1 has no `text`.",
            ),
            (
                r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "1"}, {"text": "b"}]}}"#,
                "Claim 2 cites no turn.",
            ),
            (
                r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "x"}]}}"#,
                "Claim 1: `cite` is the number of a turn.",
            ),
            (
                r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "1", "typed": {"subject": "gauge"}}]}}"#,
                "Claim 1: `typed` needs a `subject`, an `attribute` and a `value`.",
            ),
            (
                r#"{"name": "journal_write", "arguments": {"claims": "none"}}"#,
                "`claims` is a list.",
            ),
            (
                r#"{"name": "journal_write", "arguments": {"open": [{"text": "a", "relates": "forgets 2"}]}}"#,
                "`relates` is new, restates N or resolves N, where N is the number of an open item.",
            ),
            (
                r#"{"name": "journal_write", "arguments": {"open": [{"relates": "new"}]}}"#,
                "Open item 1 has no `text`.",
            ),
        ];
        for (said, why) in cases {
            assert_eq!(parse(said), Err(why.to_string()), "{said}");
        }
    }

    #[test]
    fn a_call_that_is_not_ours_is_named_and_refused() {
        assert_eq!(
            parse("{\"name\": \"tell\", \"arguments\": {\"to\": \"Maker-04\"}}"),
            Err("There is no call named `tell` here; write with journal_write.".to_string())
        );
    }

    #[test]
    fn prose_is_not_a_call() {
        assert_eq!(
            parse("I think nothing much happened."),
            Err("That was not a journal_write call.".to_string())
        );
    }

    /// **A cut-off write keeps what it got to**: the string it was writing
    /// ends in `...`, and a last claim the cut left without a citation is
    /// dropped rather than refusing the claims before it.
    #[test]
    fn a_write_cut_off_by_its_budget_is_read_as_far_as_it_got() {
        let cut = r#"<tool_call>
{"name": "journal_write", "arguments": {"claims": [
  {"text": "The scan's parity test fails on the 3090.", "cite": "4"},
  {"text": "The fix moves the sign table into shar"#;
        let draft = parse(cut).expect("read as far as it got");
        assert_eq!(draft.claims.len(), 1);
        assert_eq!(
            draft.claims[0].text,
            "The scan's parity test fails on the 3090."
        );

        let mid_summary = r#"{"name": "journal_write", "arguments": {"claims": [
  {"text": "a", "cite": "4"}], "summary": "Found why the scan fa"#;
        assert_eq!(
            parse(mid_summary).unwrap().summary,
            "Found why the scan fa..."
        );

        let mid_text_of_a_cited_claim = r#"{"name": "journal_write", "arguments": {"claims": [
  {"cite": "4", "text": "The scan fa"#;
        assert_eq!(
            parse(mid_text_of_a_cited_claim).unwrap().claims[0].text,
            "The scan fa..."
        );
    }

    #[test]
    fn a_whole_write_with_a_bad_last_claim_is_still_refused() {
        let said = r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "1"}, {"text": "b"}]}}"#;
        assert_eq!(parse(said), Err("Claim 2 cites no turn.".to_string()));
    }

    /// A cut elsewhere excuses nothing in a list the answer had closed.
    #[test]
    fn a_cut_in_the_summary_does_not_excuse_a_bad_claim() {
        let said = r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "1"}, {"text": "b"}], "summary": "Fou"#;
        assert_eq!(parse(said), Err("Claim 2 cites no turn.".to_string()));
    }

    #[test]
    fn a_function_shape_list_cut_off_is_closed_too() {
        let said = "<tool_call>\n<function=journal_write>\n\
            <parameter=claims>\n[{\"text\": \"The lift stopped.\", \"cite\": \"4\"}, {\"text\": \"The sta";
        let draft = parse(said).unwrap();
        assert_eq!(draft.claims.len(), 1);
        assert_eq!(draft.claims[0].text, "The lift stopped.");
    }

    #[test]
    fn restating_is_read_whatever_the_case_or_hash() {
        for said in ["Restates 3", "restates #3", "  restates   3 "] {
            assert_eq!(
                relates_of(&Value::String(said.into())),
                Ok(Relates::Restates(3)),
                "{said}"
            );
        }
    }
}
