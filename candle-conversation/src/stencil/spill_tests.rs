//! A string element the model opens in its own tokens (the `trie` module's
//! spills): `["tree", "-p", "--depth", "1"]` written with the tokens a
//! tokenizer merges a quote and a dash into, through the whole grammar.

use super::session::{Observe, StencilAction};
use super::sim::{simulate, Oracle, SimRun};
use super::tests::{bytes_of, tree_of};
use super::tool_call::ToolSpec;
use super::vocab::{TestVocab, TokenId};

/// ` "` — an element's opening quote, as the separator `, "` ends.
const QUOTE: TokenId = 300;
/// ` "-` and ` "--` — the quote merged with a flag's dashes.
const QUOTE_DASH: TokenId = 301;
const QUOTE_DASHES: TokenId = 302;
/// ` ""` — a whole empty string, which closes the string it would open.
const EMPTY: TokenId = 303;
/// `"-` — the quote merged with a dash, with no space before it.
const BARE_QUOTE_DASH: TokenId = 304;

fn vocab() -> TestVocab {
    TestVocab::new()
        .with_special(" \"", QUOTE)
        .with_special(" \"-", QUOTE_DASH)
        .with_special(" \"--", QUOTE_DASHES)
        .with_special(" \"\"", EMPTY)
        .with_special("\"-", BARE_QUOTE_DASH)
}

/// One tool, `run`, whose `args` is an array of strings, at least
/// `min_items` long.
fn run_tool(min_items: usize) -> Vec<ToolSpec> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "args": {"type": "array", "items": {"type": "string"}, "minItems": min_items}
        },
        "required": ["args"]
    });
    vec![ToolSpec::from_json_schema("run", &schema)]
}

fn run(min_items: usize, script: Vec<TokenId>) -> SimRun {
    let v = vocab();
    simulate(
        tree_of(&run_tool(min_items), &v),
        &v,
        Oracle::Scripted(script),
        4000,
    )
    .unwrap()
}

fn masks(run: &SimRun) -> Vec<Vec<TokenId>> {
    run.actions
        .iter()
        .filter_map(|a| match a {
            StencilAction::MaskedDecode(set) => Some(set.tokens().to_vec()),
            _ => None,
        })
        .collect()
}

const ARGS: &str = "<tool_call>\n{\"name\": \"run\", \"arguments\": \
                    {\"args\": [\"tree\", \"-p\", \"--depth\", \"1\"]}}\n</tool_call>";

/// **A flag after the first element is written in the tokens that merge its
/// dashes onto the quote**, and the call reads exactly as meant. Held to a
/// bare ` "`, the model wrote `["tree", "depth"]` for `-p agent --depth 1`.
#[test]
fn an_element_opens_on_the_token_that_merges_its_quote_and_dashes() {
    let script = [
        bytes_of("\"tree\""),
        bytes_of(","),
        vec![QUOTE_DASH],
        bytes_of("p\""),
        bytes_of(","),
        vec![QUOTE_DASHES],
        bytes_of("depth\""),
        bytes_of(","),
        vec![QUOTE],
        bytes_of("1\""),
        bytes_of("]"),
    ]
    .concat();
    let run = run(0, script);
    assert!(
        !run.observes.contains(&Observe::Bailed),
        "{:?}",
        run.observes
    );
    assert_eq!(run.text(&vocab()), ARGS);
    assert!(run.tokens.contains(&QUOTE_DASH) && run.tokens.contains(&QUOTE_DASHES));
    assert!(run.dropped.is_empty(), "{:?}", run.dropped);
}

/// After `,` the separator offers its own ` "` and the tokens that run on into
/// the string — and never ` ""`, whose second quote would close a string the
/// grammar has not opened as one.
#[test]
fn the_separator_offers_what_runs_on_and_not_what_closes() {
    let script = [
        bytes_of("\"tree\""),
        bytes_of(","),
        vec![QUOTE],
        bytes_of("x\""),
        bytes_of("]"),
    ]
    .concat();
    let run = run(0, script);
    let after_comma = masks(&run)
        .into_iter()
        .find(|m| m.contains(&QUOTE))
        .expect("the separator's quote is a masked choice");
    assert_eq!(after_comma, vec![QUOTE, QUOTE_DASH, QUOTE_DASHES]);
}

/// **An array that must hold an element opens its first string the same
/// way.** Its `"` is the grammar's whole first element — no `]` beside it — so
/// it was prefilled, and the model could not write `"-`; the run's last token
/// is held back as a choice beside the tokens that spill past it.
#[test]
fn a_required_first_element_opens_on_the_merged_token_too() {
    let script = [
        vec![BARE_QUOTE_DASH],
        bytes_of("p\""),
        bytes_of(","),
        vec![QUOTE],
        bytes_of("x\""),
        bytes_of("]"),
    ]
    .concat();
    let run = run(1, script);
    assert!(
        !run.observes.contains(&Observe::Bailed),
        "{:?}",
        run.observes
    );
    assert_eq!(masks(&run)[0], vec![b'"' as TokenId, BARE_QUOTE_DASH]);
    assert_eq!(
        run.text(&vocab()),
        "<tool_call>\n{\"name\": \"run\", \"arguments\": \
         {\"args\": [\"-p\", \"x\"]}}\n</tool_call>"
    );
}

/// Taking the bare quote at that choice writes the element as before.
#[test]
fn a_required_first_element_still_takes_the_bare_quote() {
    let script = [bytes_of("\"tree\""), bytes_of("]")].concat();
    let run = run(1, script);
    assert_eq!(
        run.text(&vocab()),
        "<tool_call>\n{\"name\": \"run\", \"arguments\": {\"args\": [\"tree\"]}}\n</tool_call>"
    );
}
