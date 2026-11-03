//! `timeline_read {id, view, turn?, from?, offset?, limit?, probe?}`: one
//! conversation in the substrate (`docs/claude_zend_gateway.md` §8.5), read as
//! `view` asks — the views a diagnosis of a run needs, each the in-process read
//! of a route the daemon serves. Admins only.
//!
//! | `view` | Module | What |
//! |---|---|---|
//! | `turns` | here | the head and a page of its turns, each cut to its first line |
//! | `turn` | here | one turn in full, a page of characters at a time |
//! | `calls` | [`calls`] | each turn's tool calls, repeats flagged |
//! | `projections` | [`projections`] | one turn's projection points: the tools offered, scored and shown |
//! | `prompt` | [`prompt`] | the prompt as it would be decoded now, piece by piece |
//! | `selection` | [`selection`] | the dialogue turns the latest projection kept |
//! | `probe` | [`probe`] | how a text ranks the tools on a fresh probe |

use host::substrate_views::TimelineDetail;
use serde_json::{json, Value};
use tokio::runtime::Handle;
use tokio::task::spawn_blocking;

use super::{Answer, Ctx, LOADING};
use crate::mcp::names::Path;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::threads;
use crate::mcp::workers::first_line;

mod calls;
mod probe;
mod projections;
mod prompt;
mod selection;

pub const NAME: &str = "timeline_read";

const DEFAULT_LIMIT: usize = 40;
const MAX_LIMIT: usize = 400;
/// How much of each half of a turn a page shows.
const HALF: usize = 160;
/// How much of a metadata value the head shows.
const META_VALUE: usize = 80;
/// How many characters of one turn a `turn` read answers at a time.
const TURN_PAGE: usize = 16_000;

/// The views, as `view` names them.
pub const VIEWS: [&str; 7] = [
    "turns",
    "turn",
    "calls",
    "projections",
    "prompt",
    "selection",
    "probe",
];

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read one stored conversation — a timeline in the substrate, the store everything \
            is kept in — by the id `layer_read` lists, a worker's own by its path \
            (`T:<thread>/<name>`), or any conversation by its id as the web UI's `#conv=` link names it \
            (a uuid, or `mcp:<owner>:<thread>/<name>`, escaped or not), as `view` asks. `turns`: its \
            label, layer, tokens, metadata (the dials it ran with) and lineage, then its turns a page \
            at a time (`offset`, `limit`), each with its index, kind and the first line of each half. \
            `turn`: turn `turn` in full — the user half (a tool round's results arrive here) and the \
            assistant half (its thinking and the calls it made) — a page of characters from `from`. \
            `calls`: each turn's tool calls, one line each, with a call identical to one already made \
            flagged as a repeat — for a run that went round in circles. `projections`: turn `turn`'s \
            projection points — when each was taken, the tools whose signatures it showed with their \
            scores against the bar in force, the strongest it left out, and its tokens by bucket — for \
            why a tool's signature was or was not in the prompt. `prompt`: the prompt as it would be \
            decoded now, piece by piece with tokens and scores (it scans the gallery, and needs the \
            conversation open on the member that answers). `selection`: the dialogue turns and \
            summaries its latest projection kept, and why. `probe`: how the text `probe` ranks the \
            tools on a fresh probe, against the 800 a signature needs (it runs a prefill). Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1, "maxLength": 200, "description": "The timeline id, a decimal number as `layer_read` lists it; a worker's path, `T:<thread>/<name>`; or a conversation id, as the web UI's `#conv=` names it."},
                "view": {"type": "string", "enum": VIEWS, "description": "What to read: `turns`, `turn`, `calls`, `projections`, `prompt`, `selection` or `probe`."},
                "turn": {"type": "integer", "minimum": 0, "description": "`turn` and `projections`: the turn's index, as `turns` lists it."},
                "from": {"type": "integer", "minimum": 0, "description": "`turn`: the character to read from (0 by default)."},
                "offset": {"type": "integer", "minimum": 0, "description": "`turns` and `calls`: the first turn (0 by default)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "`turns` and `calls`: how many turns (40 by default)."},
                "probe": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "`probe`: the text to rank the tools by — a task's prompt, a user's question."}
            },
            "required": ["id", "view"],
            "additionalProperties": false
        }
    })
}

/// A timeline's head and a page of its turns.
pub fn render(t: &TimelineDetail, offset: usize, limit: usize) -> String {
    let mut out = head(t);
    out.push_str(&format!(
        "\nturns {}–{}:",
        offset.min(t.turns.len()),
        (offset + limit).min(t.turns.len())
    ));
    for turn in t.turns.iter().skip(offset).take(limit) {
        out.push_str(&format!(
            "\n#{} {} {} tokens\n  user: {}\n  assistant: {}",
            turn.index,
            turn.kind,
            turn.tokens,
            first_line(&turn.user, HALF),
            first_line(&turn.assistant, HALF)
        ));
    }
    out
}

/// What a timeline is: its id, label, conversation, layer, size, lineage and
/// metadata.
pub fn head(t: &TimelineDetail) -> String {
    let mut out = format!(
        "timeline {} — {} ({}), layer {} group {}, {} tokens, {} turns{}",
        t.timeline,
        if t.label.is_empty() {
            "untitled"
        } else {
            &t.label
        },
        if t.conv_id.is_empty() {
            "no conv_id"
        } else {
            &t.conv_id
        },
        t.layer,
        t.group,
        t.total_tokens,
        t.turn_count,
        if t.archived { ", archived" } else { "" }
    );
    if t.inherited_chain.len() > 1 {
        out.push_str(&format!("\nlineage: {}", t.inherited_chain.join(" → ")));
    }
    if !t.custom.is_empty() {
        // The values too: the dials a worker ran with (`dial_effort`,
        // `dial_think`, …) are what a review of its run asks after.
        let pairs: Vec<String> = t
            .custom
            .iter()
            .map(|(k, v)| format!("{k}={}", first_line(v, META_VALUE)))
            .collect();
        out.push_str(&format!("\nmetadata: {}", pairs.join(", ")));
    }
    out
}

/// One turn in full — both halves, the tool calls an assistant half makes and
/// the results a user half carries included — a page of [`TURN_PAGE`]
/// characters from character `from`, and where to read on.
pub fn render_turn(t: &TimelineDetail, index: u32, from: usize) -> Result<String, String> {
    let Some(turn) = t.turns.iter().find(|turn| turn.index == index) else {
        return Err(no_turn(t, index));
    };
    let whole = format!("--- user\n{}\n--- assistant\n{}", turn.user, turn.assistant);
    let total = whole.chars().count();
    let page: String = whole.chars().skip(from).take(TURN_PAGE).collect();
    let to = (from + TURN_PAGE).min(total);
    let mut out = format!(
        "timeline {} turn #{} {}, {} tokens — characters {}–{to} of {total}",
        t.timeline,
        turn.index,
        turn.kind,
        turn.tokens,
        from.min(total)
    );
    if to < total {
        out.push_str(&format!(" (call again with `from: {to}` to read on)"));
    }
    out.push('\n');
    out.push_str(&page);
    Ok(out)
}

/// What a read of a turn `t` does not have answers.
fn no_turn(t: &TimelineDetail, index: u32) -> String {
    format!(
        "timeline {} has no turn {index}; it has {} turns",
        t.timeline, t.turn_count
    )
}

const ID_IS: &str =
    "`id` is a timeline id, a decimal number as `layer_read` lists it; a worker's path, \
     `T:<thread>/<name>`; or a conversation id, as the web UI's `#conv=` names it — a uuid, \
     or `mcp:<owner>:<thread>/<name>`";

/// The conversation an `id` names: a timeline by number, a worker of the
/// caller's by its path, or any conversation by its id — what the web UI's
/// `#conv=` carries, so a conversation someone is looking at can be read.
#[derive(Debug, PartialEq, Eq)]
enum Which {
    Timeline(u64),
    Worker { thread: String, name: String },
    Conversation(String),
}

/// `id` as the web UI's link spells it, with its `%3A` and `%2F` undone.
fn unescaped(id: &str) -> String {
    id.replace("%3A", ":")
        .replace("%3a", ":")
        .replace("%2F", "/")
        .replace("%2f", "/")
}

/// Whether `id` is a conversation id: a uuid, or an MCP conversation's.
fn is_conversation(id: &str) -> bool {
    let uuid = id.len() == 36
        && id.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit(),
        });
    uuid || (id.starts_with("mcp:") && id.len() > 4)
}

impl Which {
    fn of(id: &str) -> Result<Self, String> {
        let id = unescaped(id.trim());
        if let Ok(raw) = id.parse::<u64>() {
            return Ok(Self::Timeline(raw));
        }
        if is_conversation(&id) {
            return Ok(Self::Conversation(id));
        }
        match Path::parse(&id) {
            Ok(Path::Worker { thread, worker }) => Ok(Self::Worker {
                thread,
                name: worker,
            }),
            _ => Err(ID_IS.into()),
        }
    }
}

/// The timeline `which` names, for `call`'s caller.
fn found(call: &OpsCall, which: &Which) -> Result<TimelineDetail, String> {
    let t = match which {
        Which::Timeline(raw) => call.session.substrate_timeline(*raw),
        Which::Worker { thread, name } => {
            let owner = call
                .caller
                .as_ref()
                .map(|c| c.key())
                .ok_or("a worker's path is read as the person it works for, and this call names none")?;
            call.session
                .conversation_detail(&threads::conv_id(&owner, thread, name))
        }
        Which::Conversation(conv) => call.session.conversation_detail(conv),
    };
    t.ok_or_else(|| match which {
        Which::Timeline(raw) => format!("there is no timeline {raw}"),
        Which::Worker { thread, name } => {
            format!("`T:{thread}/{name}` has no conversation of yours")
        }
        Which::Conversation(conv) => format!("there is no conversation {conv}"),
    })
}

/// The integer argument `key`, if given.
fn uint(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(Value::as_u64)
}

/// The turn `view` reads, which it needs.
fn turn_of(args: &Value, view: &str) -> Result<u32, String> {
    uint(args, "turn")
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| format!("`view: {view}` reads one turn: give `turn`, its index as `turns` lists it"))
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    if !call.admin() {
        return Answer::error(format!("`{NAME}` is an Admin's"));
    }
    let Some(id) = args.get("id").and_then(|v| {
        v.as_str()
            .map(str::to_string)
            .or_else(|| v.as_u64().map(|n| n.to_string()))
    }) else {
        return Answer::error(ID_IS);
    };
    let view = args.get("view").and_then(Value::as_str).unwrap_or("turns");
    if !VIEWS.contains(&view) {
        return Answer::error(format!(
            "`{view}` is not a view: {}",
            VIEWS.join(", ")
        ));
    }
    let which = match Which::of(&id) {
        Ok(w) => w,
        Err(e) => return Answer::error(e),
    };
    if !call.session.is_loaded() {
        return Answer::error(LOADING);
    }
    let t = match found(call, &which) {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    let offset = uint(args, "offset").unwrap_or(0) as usize;
    let limit = (uint(args, "limit").unwrap_or(DEFAULT_LIMIT as u64) as usize).clamp(1, MAX_LIMIT);
    let answer = match view {
        "turn" => turn_of(args, view).and_then(|index| {
            render_turn(&t, index, uint(args, "from").unwrap_or(0) as usize)
        }),
        "calls" => Ok(calls::render(&t, offset, limit)),
        "projections" => match turn_of(args, view) {
            Ok(index) => {
                let points = call.session.conversation_projections(&t.conv_id);
                projections::render(&t, index, &points)
            }
            Err(e) => Err(e),
        },
        "prompt" => prompt::read(call, &t).await,
        "selection" => selection::read(call, &t),
        "probe" => match args.get("probe").and_then(Value::as_str).map(str::trim) {
            Some(text) if !text.is_empty() => probe::read(call, text).await,
            _ => Err("`view: probe` ranks the tools by a text: give `probe`".into()),
        },
        _ => Ok(render(&t, offset, limit)),
    };
    match answer {
        Ok(text) => Answer::ok(text),
        Err(e) => Answer::error(e),
    }
}

/// An MCP call: its substrate reads wait on the disk, so the whole read runs on
/// the blocking pool, holding up none of the calls the runtime is running.
pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (call, args) = (OpsCall::of(ctx), args.clone());
    let runtime = Handle::current();
    spawn_blocking(move || runtime.block_on(run(&call, &args)))
        .await
        .unwrap_or_else(|e| Answer::error(format!("the read did not finish: {e}")))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use host::substrate_views::TurnView;

    /// **An id is a timeline's number or a worker's path**; anything else is
    /// refused saying what it may be.
    #[test]
    fn an_id_is_a_timeline_number_or_a_workers_path() {
        assert_eq!(
            Which::of("7303474180578274787"),
            Ok(Which::Timeline(7303474180578274787))
        );
        assert_eq!(
            Which::of(" T:night-test/verify-branch.2 "),
            Ok(Which::Worker {
                thread: "night-test".into(),
                name: "verify-branch.2".into()
            })
        );
        assert_eq!(Which::of("T:night-test"), Err(ID_IS.to_string()));
        assert_eq!(Which::of("seven"), Err(ID_IS.to_string()));
    }

    /// **A conversation reads by the id the web UI's link carries**: a uuid,
    /// or an MCP conversation's id, escaped as the link escapes it or not.
    #[test]
    fn a_conversation_reads_by_the_id_its_link_carries() {
        assert_eq!(
            Which::of("1a7f30b0-2e8d-4ba4-ab2e-13cc882fc015"),
            Ok(Which::Conversation(
                "1a7f30b0-2e8d-4ba4-ab2e-13cc882fc015".into()
            ))
        );
        assert_eq!(
            Which::of("mcp%3A154797f34c5a6ca6%3Azen-ops%2Fread-stencil-checks"),
            Ok(Which::Conversation(
                "mcp:154797f34c5a6ca6:zen-ops/read-stencil-checks".into()
            ))
        );
        assert_eq!(Which::of("1a7f30b0-2e8d"), Err(ID_IS.to_string()));
    }

    #[test]
    fn a_timeline_reads_as_its_head_then_its_turns() {
        let t = timeline();
        assert_eq!(
            render(&t, 0, 10),
            "timeline 77 — Verify (mcp:o:review/verify), layer dialogue group chat, 1200 tokens, \
             1 turns\nlineage: 1 → 77\nmetadata: decode_band=normal\nturns 0–1:\n#0 normal 40 \
             tokens\n  user: check the expiry\n  assistant: It is inclusive."
        );
    }

    /// **A view that reads one turn asks for it** by name.
    #[test]
    fn a_view_of_one_turn_asks_for_the_turn() {
        assert!(turn_of(&json!({}), "projections")
            .unwrap_err()
            .contains("give `turn`"));
        assert_eq!(turn_of(&json!({"turn": 3}), "turn"), Ok(3));
    }

    pub(super) fn timeline() -> TimelineDetail {
        TimelineDetail {
            timeline: "77".into(),
            conv_id: "mcp:o:review/verify".into(),
            label: "Verify".into(),
            archived: false,
            layer: "dialogue".into(),
            group: "chat".into(),
            total_tokens: 1200,
            custom: BTreeMap::from([("decode_band".into(), "normal".into())]),
            inherited_chain: vec!["1".into(), "77".into()],
            peaks: vec![],
            couplings: vec![],
            turn_count: 1,
            turns: vec![turn_view(0, "check the expiry\nmore", "It is inclusive.")],
        }
    }

    pub(super) fn turn_view(index: u32, user: &str, assistant: &str) -> TurnView {
        TurnView {
            index,
            kind: "normal",
            height: 0,
            tokens: 40,
            no_think: false,
            children: vec![],
            coupled: false,
            peak: true,
            user: user.into(),
            assistant: assistant.into(),
            layout: None,
        }
    }

    /// **One turn reads in full, a page at a time**: both halves whole, and
    /// where to read on when they run past a page.
    #[test]
    fn one_turn_reads_in_full_a_page_at_a_time() {
        let mut t = timeline();
        assert_eq!(
            render_turn(&t, 0, 0).unwrap(),
            "timeline 77 turn #0 normal, 40 tokens — characters 0–61 of 61\n--- user\ncheck the \
             expiry\nmore\n--- assistant\nIt is inclusive."
        );
        assert_eq!(
            render_turn(&t, 3, 0).unwrap_err(),
            "timeline 77 has no turn 3; it has 1 turns"
        );
        t.turns[0].user = "x".repeat(TURN_PAGE + 5);
        let first = render_turn(&t, 0, 0).unwrap();
        assert!(first.contains(&format!(
            "characters 0–{TURN_PAGE} of {} (call again with `from: {TURN_PAGE}` to read on)",
            TURN_PAGE + 5 + 40
        )));
        let rest = render_turn(&t, 0, TURN_PAGE).unwrap();
        assert!(
            rest.ends_with("xxx\n--- assistant\nIt is inclusive."),
            "{rest}"
        );
    }
}
