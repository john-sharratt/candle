//! The journal's one call, the grammar it is held to, and how an answer is read
//! back.
//!
//! **Deliberately not in [`crate::engine::tools::CATALOG`].** That catalog is
//! what a character may *do*; `journal_write` reaches nothing. It is the shape an
//! answer takes when the character is asked to write its entry — the question
//! describes it and the grammar forces it — and in the catalog `act::parse` would
//! start accepting it on a live turn.
//!
//! # The grammar
//!
//! [`write_spec`] is built per draft, because what its enums may hold depends on
//! the stretch being written up — the turns a claim may cite, and the items that
//! are open to restate or settle. A claim cannot cite a turn that is not there,
//! and an item that is not open cannot be settled, because neither is a thing the
//! grammar offers.
//!
//! # Why `cite` is one value and not a list
//!
//! The grammar guides an object inside an array, and one array level. An array
//! inside an array element decodes free. A claim is an element of `claims`, so
//! its citations are `cite` and an optional `also_cite` — two guided enums —
//! rather than a list the model would write unconstrained.
//!
//! # Reading an answer back
//!
//! [`parse`] reads whichever call shape the dialect wrote. The JSON block is
//! grammar-guided all the way down; the function block writes raw values, so an
//! array or object there is JSON text inside an element and the same checks run
//! on it — [`verify`](crate::engine::journal::verify) is the net either way.

use candle_conversation::stencil::{Param as CallParam, ParamType, ToolSpec};
use serde_json::{Map, Value};

use crate::engine::act::escape_control_in_strings;
use crate::engine::journal::entry::{Item, Typed};
use crate::engine::journal::verify::{Draft, DraftClaim, DraftOpen, Relates};

/// The entry itself.
pub const WRITE: &str = "journal_write";

fn field(name: &str, ty: ParamType, required: bool) -> CallParam {
    CallParam {
        name: name.to_string(),
        ty,
        required,
        enum_values: None,
        items: None,
        min_items: 0,
        properties: None,
        nullable: false,
        minimum: None,
        requires: Vec::new(),
        shapes: Vec::new(),
    }
}

fn choice(name: &str, required: bool, values: Vec<String>) -> CallParam {
    CallParam {
        enum_values: Some(values),
        ..field(name, ParamType::String, required)
    }
}

fn list_of(name: &str, required: bool, min_items: usize, item: CallParam) -> CallParam {
    CallParam {
        items: Some(Box::new(item)),
        min_items,
        ..field(name, ParamType::Array, required)
    }
}

fn object(name: &str, required: bool, properties: Vec<CallParam>) -> CallParam {
    CallParam {
        properties: Some(properties),
        ..field(name, ParamType::Object, required)
    }
}

/// The `journal_write` call, for the stretch whose turns are `citable` and with
/// `open` items standing.
///
/// `None` when nothing in the stretch can be cited: there is no entry to write,
/// and a `cite` with no value to choose would be a grammar that cannot be
/// satisfied.
pub fn write_spec(citable: &[u64], open: &[Item]) -> Option<ToolSpec> {
    if citable.is_empty() {
        return None;
    }
    let ids: Vec<String> = citable.iter().map(u64::to_string).collect();
    let mut relations = vec!["new".to_string()];
    for item in open {
        relations.push(format!("restates {}", item.id));
        relations.push(format!("resolves {}", item.id));
    }

    let claim = object(
        "",
        true,
        vec![
            field("text", ParamType::String, true),
            choice("cite", true, ids.clone()),
            choice("also_cite", false, ids),
            object(
                "typed",
                false,
                vec![
                    field("subject", ParamType::String, true),
                    field("attribute", ParamType::String, true),
                    field("value", ParamType::String, true),
                ],
            ),
        ],
    );
    let item = object(
        "",
        true,
        vec![
            field("text", ParamType::String, true),
            choice("relates", true, relations),
        ],
    );
    Some(ToolSpec {
        name: WRITE.to_string(),
        params: vec![
            list_of("claims", true, 1, claim),
            list_of("intend", false, 0, field("", ParamType::String, true)),
            list_of("open", false, 0, item),
        ],
    })
}

/// The call's name and arguments, from whichever shape was written.
fn arguments(answer: &str) -> Option<(String, Map<String, Value>)> {
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

fn text_of(v: &Value) -> Option<String> {
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
    Ok(DraftClaim { text, cite, typed })
}

fn items_of(v: &Value) -> Result<Vec<DraftOpen>, String> {
    let Some(list) = v.as_array() else {
        return Err("`open` is a list.".into());
    };
    list.iter()
        .enumerate()
        .map(|(n, o)| {
            let text = o
                .get("text")
                .and_then(text_of)
                .ok_or_else(|| format!("Open item {} has no `text`.", n + 1))?;
            let relates = relates_of(o.get("relates").unwrap_or(&Value::Null))?;
            Ok(DraftOpen { text, relates })
        })
        .collect()
}

fn draft_of(args: &Map<String, Value>) -> Result<Draft, String> {
    let claims = match args.get("claims") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(list)) => list
            .iter()
            .enumerate()
            .map(|(n, c)| claim_of(n, c))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("`claims` is a list.".into()),
    };
    let intend = match args.get("intend") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(list)) => list.iter().filter_map(text_of).collect(),
        Some(_) => return Err("`intend` is a list of short lines.".into()),
    };
    let open = match args.get("open") {
        None | Some(Value::Null) => Vec::new(),
        Some(v) => items_of(v)?,
    };
    Ok(Draft {
        claims,
        intend,
        open,
    })
}

/// Read one answer. The error is worded for the model, which is shown it with
/// its refused attempt and writes again.
pub fn parse(answer: &str) -> Result<Draft, String> {
    let (name, args) =
        arguments(answer).ok_or_else(|| "That was not a journal_write call.".to_string())?;
    if name != WRITE {
        return Err(format!(
            "There is no call named `{name}` here; write with journal_write."
        ));
    }
    draft_of(&args)
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::{compile_tool_call_tree, ToolCallEnvelope};

    use super::*;
    use crate::engine::act::{self, Rejected};
    use crate::engine::tools::CATALOG;

    fn item(id: u64, text: &str) -> Item {
        Item {
            id,
            text: text.into(),
        }
    }

    #[test]
    fn the_journal_call_is_not_one_the_world_accepts() {
        assert!(
            CATALOG.iter().all(|c| c.name != WRITE),
            "`{WRITE}` must not be in the world catalog: `act::parse` would accept it live"
        );
    }

    #[test]
    fn a_live_turn_that_makes_the_journal_call_acts_on_nothing() {
        let parsed = act::parse(&format!(r#"{{"tool": "{WRITE}"}}"#));
        assert!(parsed.acts.is_empty(), "`{WRITE}` became an act");
        assert!(
            matches!(&parsed.rejected[..], [Rejected::UnknownTool { tool }] if tool == WRITE),
            "`{WRITE}` was not refused as unknown: {:?}",
            parsed.rejected
        );
    }

    #[test]
    fn the_write_call_takes_claims_then_intend_then_open() {
        let write = write_spec(&[1, 2], &[]).unwrap();
        let names: Vec<(&str, bool)> = write
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.required))
            .collect();
        assert_eq!(
            names,
            [("claims", true), ("intend", false), ("open", false)]
        );
    }

    #[test]
    fn an_entry_cannot_be_written_without_a_claim() {
        let write = write_spec(&[1, 2], &[]).unwrap();
        let min: Vec<usize> = write.params.iter().map(|p| p.min_items).collect();
        assert_eq!(min, [1, 0, 0]);
    }

    #[test]
    fn a_stretch_with_nothing_to_cite_has_no_writing_grammar() {
        assert!(write_spec(&[], &[item(1, "the door")]).is_none());
    }

    #[test]
    fn cite_offers_exactly_the_turns_of_the_stretch() {
        let write = write_spec(&[7, 9, 12], &[]).unwrap();
        let claims = write.params.iter().find(|p| p.name == "claims").unwrap();
        let element = claims.items.as_ref().unwrap();
        let props = element.properties.as_ref().unwrap();
        let ids = ["7", "9", "12"].map(String::from);
        for key in ["cite", "also_cite"] {
            let p = props.iter().find(|p| p.name == key).unwrap();
            assert_eq!(p.enum_values.as_deref(), Some(&ids[..]), "{key}");
        }
        assert!(props.iter().find(|p| p.name == "cite").unwrap().required);
        assert!(
            !props
                .iter()
                .find(|p| p.name == "also_cite")
                .unwrap()
                .required
        );
        assert!(!props.iter().find(|p| p.name == "typed").unwrap().required);
    }

    #[test]
    fn relates_offers_new_and_restating_or_settling_each_open_item_only() {
        let open = [item(3, "the valve"), item(5, "the roster")];
        let write = write_spec(&[1], &open).unwrap();
        let list = write.params.iter().find(|p| p.name == "open").unwrap();
        let element = list.items.as_ref().unwrap();
        let relates = element
            .properties
            .as_ref()
            .unwrap()
            .iter()
            .find(|p| p.name == "relates")
            .unwrap();
        assert_eq!(
            relates.enum_values.as_deref().unwrap(),
            [
                "new",
                "restates 3",
                "resolves 3",
                "restates 5",
                "resolves 5"
            ]
        );
    }

    #[test]
    fn with_nothing_open_the_only_relation_is_new() {
        let write = write_spec(&[1], &[]).unwrap();
        let list = write.params.iter().find(|p| p.name == "open").unwrap();
        let relates = list
            .items
            .as_ref()
            .unwrap()
            .properties
            .as_ref()
            .unwrap()
            .iter()
            .find(|p| p.name == "relates")
            .unwrap();
        assert_eq!(relates.enum_values.as_deref().unwrap(), ["new"]);
    }

    #[test]
    fn the_writing_grammar_compiles_under_both_call_shapes() {
        let citable: Vec<u64> = (1..=16).collect();
        let open: Vec<Item> = (1..=6).map(|i| item(i, "something")).collect();
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            let spec = write_spec(&citable, &open).unwrap();
            compile_tool_call_tree(&[spec], &env).expect("the writing grammar compiles");
        }
    }

    /// The writing grammar is built once per draft, with the largest span and the
    /// most open items a draft can have. It has to stay a size that is cheap to
    /// compile — an array level is unrolled 64 times, so what one element costs
    /// is multiplied.
    #[test]
    fn the_writing_grammar_stays_small_at_its_largest() {
        let citable: Vec<u64> = (1..=64).collect();
        let open: Vec<Item> = (1..=6).map(|i| item(i, "something")).collect();
        let spec = write_spec(&citable, &open).unwrap();
        let tree = compile_tool_call_tree(&[spec], &ToolCallEnvelope::qwen3()).unwrap();
        let nodes = tree.nodes.len();
        assert!(
            nodes < 10_000,
            "{nodes} nodes: an array element that costs this much, times 64 levels, is a compile \
             that is not worth a journal entry"
        );
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
        let Ok(draft) = parse(said) else {
            panic!("did not read as a write");
        };
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
        let Ok(draft) = parse(said) else {
            panic!("did not read as a write");
        };
        assert_eq!(draft.claims[0].text, "The lift stopped.");
        assert_eq!(draft.claims[0].cite, vec![4]);
        assert_eq!(draft.intend, vec!["take the stairs".to_string()]);
    }

    #[test]
    fn a_turn_may_be_written_as_a_number() {
        let said =
            r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": 4}]}}"#;
        let Ok(draft) = parse(said) else {
            panic!("did not read as a write");
        };
        assert_eq!(draft.claims[0].cite, vec![4]);
    }

    #[test]
    fn a_repeated_turn_is_cited_once() {
        let said = r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a", "cite": "4", "also_cite": "4"}]}}"#;
        let Ok(draft) = parse(said) else {
            panic!("did not read as a write");
        };
        assert_eq!(draft.claims[0].cite, vec![4]);
    }

    #[test]
    fn a_raw_newline_inside_a_claim_is_repaired_not_refused() {
        let said = "{\"name\": \"journal_write\", \"arguments\": {\"claims\": [{\"text\": \"line one\nline two\", \"cite\": \"4\"}]}}";
        let Ok(draft) = parse(said) else {
            panic!("did not read as a write");
        };
        assert_eq!(draft.claims[0].text, "line one\nline two");
    }

    #[test]
    fn a_write_with_only_open_items_is_a_draft() {
        let said = r#"{"name": "journal_write", "arguments": {"open": [{"text": "the lift", "relates": "new"}]}}"#;
        let Ok(draft) = parse(said) else {
            panic!("did not read as a write");
        };
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
