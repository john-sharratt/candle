//! What a character reaches beyond its world, performed where the world runs
//! (`docs/zen_moon_mind.md` §16): awaited, because each act reaches something
//! real — a forecast, the web, the image model.
//!
//! | Act | Runs |
//! |---|---|
//! | the world outside, the reckoning, the guides | zend's own tool, here, on the blocking pool, under [`grants`] |
//! | `http_session_*`, `http_request` | the same, in the character's own sessions |
//! | `journal_read`, `journal_grep` | the same, over the character's own journal ([`Shelf`]) |
//! | `notes_*` | the world's notebook ([`notes`]) |
//! | `image_generate`, `image_edit` | npcd's own `/v1/npc/mcp`, as Zen, through the gateway |

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use journal_core::journal::Holds;
use serde_json::Value;
use tokio::task::spawn_blocking;
use zend_tools::state::{Secrets, SessionRegistry};
use zend_tools::tools::journal::JournalShelf;
use zend_tools::{run, Capability, Grants, ToolContext};

use super::zen_task::call_as_zen;
use crate::api::now_ms;
use crate::engine::act::Act;
use crate::engine::beyond::answer::{answer_of, cut};
use crate::engine::beyond::call::{call_of, Call, ToolCall};
use crate::engine::beyond::{notes, reached};
use crate::engine::journal::shelf::{lines_of, Own, Shelf};
use crate::engine::journal::text;
use crate::engine::runtime::{Recorded, Runtime, LANDED, REFUSED};
use crate::tenant::NAME as NPCD_APP;

/// npcd's own MCP server, as a member reaches it.
const NPC_MCP: &str = "/v1/npc/mcp";

/// How long a picture may take to draw: what zend allows npcd's draw.
const DRAW_WITHIN: Duration = Duration::from_secs(240);

/// The id a tool run here is called under, in its logs.
const CALL_ID: &str = "npcd-beyond";

/// What zend's tools run under here: the network, and nothing else — no
/// program, no disk, no stored credential.
pub fn grants() -> Grants {
    Grants::NONE.with(Capability::Network)
}

/// What zend's tools run with here: one context for the house, and each
/// character's own HTTP sessions, so what one opens no other lists or uses.
#[derive(Default)]
pub struct Tools {
    /// Built on the blocking pool at the first call, since its HTTP client is
    /// a blocking one, which the async runtime may neither build nor drop.
    house: Arc<OnceLock<ToolContext>>,
    sessions: Mutex<HashMap<u64, Arc<SessionRegistry>>>,
}

impl Tools {
    /// Character `npc`'s sessions, opened at its first call.
    fn sessions_of(&self, npc: u64) -> Arc<SessionRegistry> {
        let mut all = self.sessions.lock().unwrap();
        Arc::clone(all.entry(npc).or_default())
    }

    /// Close every session `npc` holds: it is no longer here.
    pub fn forget(&self, npc: u64) {
        self.sessions.lock().unwrap().remove(&npc);
    }
}

fn recorded(act: &Act, landed: bool, answer: String) -> Recorded {
    let first = answer.lines().next().unwrap_or_default().to_string();
    Recorded {
        feed: match landed {
            true => format!("{} {LANDED} {first}", act.summary()),
            false => format!("{} {REFUSED} {first}", act.summary()),
        },
        answer,
        landed,
        departing: false,
    }
}

/// Where a call reaches, for its record: a fetch's or a request's address, a
/// search's words, a draw's prompt — or nothing beyond the act's name.
fn reaching(call: &Call) -> String {
    let of = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).map(str::to_string);
    match call {
        Call::Here(ToolCall { tool, args }) if *tool == "http_request" => format!(
            "{} {}",
            of(args, "method").unwrap_or_default().to_uppercase(),
            of(args, "path").unwrap_or_default()
        ),
        Call::Here(ToolCall { args, .. }) => of(args, "url")
            .or_else(|| of(args, "query"))
            .or_else(|| of(args, "location"))
            .or_else(|| of(args, "host"))
            .or_else(|| of(args, "journal"))
            .unwrap_or_default(),
        Call::Npcd(ToolCall { args, .. }) => of(args, "image")
            .or_else(|| of(args, "prompt"))
            .unwrap_or_default(),
        Call::Notebook(_) => String::new(),
    }
}

/// The house's context, built once, on this blocking thread.
fn house<'a>(house: &'a OnceLock<ToolContext>, secrets: &Arc<Secrets>) -> &'a ToolContext {
    house.get_or_init(|| {
        ToolContext::new()
            .with_secrets(Arc::clone(secrets))
            .granting(grants())
    })
}

/// The house's context as one character's calls run under it: its own
/// sessions, and its own journal when it keeps one.
fn character(
    house: &ToolContext,
    sessions: Arc<SessionRegistry>,
    journal: Option<Arc<dyn JournalShelf>>,
) -> ToolContext {
    let mut ctx = house.clone();
    ctx.sessions = sessions;
    match journal {
        Some(shelf) => ctx.with_journals(shelf),
        None => ctx,
    }
}

impl Runtime {
    /// The daemon's secrets, for zend's tools — the search key among them.
    pub fn set_tool_secrets(&self, secrets: Arc<Secrets>) {
        *self.tool_secrets.write().unwrap() = secrets;
    }

    /// Character `npc`'s own journal, as its `journal_read` and
    /// `journal_grep` reach it: `None` before the substrate is open.
    fn shelf_of(&self, npc: u64) -> Option<Arc<dyn JournalShelf>> {
        let shared = self.substrate()?;
        let lines = self
            .scheduler
            .journal_of(npc, |j| lines_of(j.journal()))
            .unwrap_or_default();
        Some(Arc::new(Shelf::new(Own::new(lines, text::of(shared, npc)))))
    }

    /// Run zend's tool for `npc` on the blocking pool, and read its answer.
    async fn run_here(&self, npc: u64, call: ToolCall) -> Result<String, String> {
        let house_of = Arc::clone(&self.tools.house);
        let secrets = Arc::clone(&self.tool_secrets.read().unwrap());
        let sessions = self.tools.sessions_of(npc);
        let journal = self.shelf_of(npc);
        let result = spawn_blocking(move || {
            let ctx = character(house(&house_of, &secrets), sessions, journal);
            run(call.tool, CALL_ID, &call.args, &ctx)
        })
        .await
        .map_err(|e| format!("The tool stopped before it answered: {e}"))?;
        answer_of(&result)
    }

    /// An act beyond the world made for `npc`: the call it stands for, where
    /// that call runs, if its personality reaches it.
    pub(crate) async fn beyond_act(&self, npc: u64, act: &Act) -> Recorded {
        if self.scheduler.is_suspended(npc) {
            return recorded(
                act,
                false,
                "You are held still: nothing you do reaches anyone.".into(),
            );
        }
        let Some((hosted, _)) = self.body_of(npc) else {
            return recorded(act, false, "You have no body here to do that with.".into());
        };
        if !reached(&self.reaches_of(npc), act.tool) {
            return recorded(
                act,
                false,
                format!(
                    "`{}` is not yours to reach: nothing you are reaches it.",
                    act.tool
                ),
            );
        }
        let call = match call_of(act.tool, &act.args) {
            Ok(call) => call,
            Err(why) => return recorded(act, false, why),
        };
        tracing::info!(
            npc,
            act = act.tool,
            to = %reaching(&call),
            "reached beyond the world"
        );
        let answer = match call {
            Call::Here(call) => self.run_here(npc, call).await,
            Call::Notebook(asked) => {
                let world = hosted.id().to_string();
                match self.substrate() {
                    Some(shared) => notes::run(&shared, &world, &asked, now_ms()),
                    None => Err("The notebook is not open yet. Try again in a moment.".into()),
                }
            }
            Call::Npcd(ToolCall { tool, args }) => match self.standing.member().cloned() {
                Some(member) => {
                    let session = format!("beyond-{npc}");
                    call_as_zen(
                        &member,
                        NPCD_APP,
                        NPC_MCP,
                        DRAW_WITHIN,
                        tool,
                        &args,
                        &session,
                    )
                    .await
                }
                None => Err(
                    "The image model cannot be reached from here: this house is on no \
                             cluster."
                        .into(),
                ),
            },
        };
        match answer {
            Ok(text) => recorded(act, true, cut(&text)),
            Err(why) => recorded(act, false, cut(&why)),
        }
    }
}

#[cfg(test)]
mod tests {
    use journal_core::transcript::{Filed, Part};
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    fn here(tool: &str, v: Value) -> Call {
        call_of(tool, v.as_object().unwrap()).unwrap()
    }

    fn run_on(ctx: &ToolContext, call: Call) -> Value {
        let Call::Here(ToolCall { tool, args }) = call else {
            panic!("{call:?} does not run here")
        };
        run(tool, CALL_ID, &args, ctx)
    }

    /// **zend's tools run here under the network alone**: no program, no
    /// disk, no stored credential.
    #[test]
    fn the_tools_run_under_the_network_alone() {
        let g = grants();
        assert!(g.require(Capability::Network).is_ok());
        for cap in [
            Capability::Exec,
            Capability::DiskWrite,
            Capability::Secrets,
            Capability::Sandbox,
        ] {
            assert!(g.require(cap).is_err(), "{cap:?}");
        }
    }

    /// **The reckoning runs here and answers in the tool's own JSON**; a
    /// guide is its text, and an unknown one says which there are.
    #[test]
    fn reckoning_and_the_guides_run_here() {
        let cell = OnceLock::new();
        let ctx = house(&cell, &Arc::new(Secrets::empty()));
        let sum = run_on(
            ctx,
            here("calculator", json!({"expression": "1250 * 1.1 * 12"})),
        );
        assert_eq!(sum["result"], json!(16500.0), "{sum}");
        let converted = run_on(
            ctx,
            here(
                "unit_convert",
                json!({"value": "100", "from": "celsius", "to": "fahrenheit"}),
            ),
        );
        assert_eq!(converted["result"], json!(212.0), "{converted}");
        let list = answer_of(&run_on(ctx, here("guide_list", json!({})))).unwrap();
        assert!(list.contains("\n- deploy — "), "{list}");
        let unknown = answer_of(&run_on(ctx, here("guide_read", json!({"guide": "nope"}))));
        assert!(unknown
            .unwrap_err()
            .starts_with("unknown_guide: there is no guide `nope`"));
    }

    /// **A request to the house's own addresses is refused before it is
    /// sent**, and **a character's sessions are its own**: what one opens no
    /// other lists, and forgetting it closes them.
    #[test]
    fn a_characters_sessions_are_its_own_and_reach_no_house_address() {
        let cell = OnceLock::new();
        let base = house(&cell, &Arc::new(Secrets::empty()));
        let tools = Tools::default();
        let zen = character(base, tools.sessions_of(7), None);
        let wick = character(base, tools.sessions_of(8), None);
        let opened = run_on(&zen, here("http_session_open", json!({})));
        let id = opened["session_id"]
            .as_str()
            .unwrap_or_else(|| panic!("no session: {opened}"))
            .to_string();
        assert_eq!(zen.sessions.list_http().len(), 1);
        assert!(wick.sessions.list_http().is_empty());
        for url in [
            "http://127.0.0.1:8081/v1/status",
            "http://192.168.1.3:8092/",
        ] {
            let answered = run_on(
                &zen,
                here(
                    "http_request",
                    json!({"session_id": id, "path": url, "method": "GET"}),
                ),
            );
            assert_eq!(answered["error"], "url_blocked", "{url}: {answered}");
        }
        let fetched = run_on(
            &zen,
            here("web_fetch", json!({"url": "http://localhost:8081/"})),
        );
        assert_eq!(fetched["error"], "url_blocked", "{fetched}");
        tools.forget(7);
        assert!(tools.sessions_of(7).list_http().is_empty(), "forgotten");
    }

    /// **`journal_read` and `journal_grep` reach the character's own
    /// journal**: its entries list, and the turns behind them read.
    #[test]
    fn the_journal_tools_read_the_characters_own() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let turns = text::of(shared, 7);
        turns
            .file(&Filed {
                entry: 2,
                parts: vec![Part {
                    id: 1,
                    at_ms: 0,
                    text: "John asked whether rain was coming.".into(),
                }],
            })
            .unwrap();
        let shelf: Arc<dyn JournalShelf> = Arc::new(Shelf::new(Own::new(
            vec!["#2 John asked about the rain.".into()],
            turns,
        )));
        let cell = OnceLock::new();
        let ctx = character(
            house(&cell, &Arc::new(Secrets::empty())),
            Arc::default(),
            Some(shelf),
        );
        let listed = answer_of(&run_on(
            &ctx,
            here("journal_read", json!({"journal": "self"})),
        ));
        assert!(
            listed
                .as_deref()
                .unwrap()
                .contains("#2 John asked about the rain."),
            "{listed:?}"
        );
        let found = answer_of(&run_on(
            &ctx,
            here(
                "journal_grep",
                json!({"journal": "self", "pattern": "RAIN"}),
            ),
        ));
        assert!(
            found.as_deref().unwrap().contains("rain was coming"),
            "{found:?}"
        );
    }

    /// **What a call reaches is named in its record.**
    #[test]
    fn a_call_names_where_it_reaches() {
        assert_eq!(
            reaching(&here("web_fetch", json!({"url": "https://example.com/"}))),
            "https://example.com/"
        );
        assert_eq!(
            reaching(&here(
                "http_request",
                json!({"session_id": "s1", "path": "https://api.example.com/x", "method": "post"})
            )),
            "POST https://api.example.com/x"
        );
        assert_eq!(
            reaching(&here("weather", json!({"location": "Sydney"}))),
            "Sydney"
        );
        assert_eq!(
            reaching(&here(
                "journal_grep",
                json!({"journal": "self", "pattern": "x"})
            )),
            "self"
        );
        assert_eq!(
            reaching(&here("calculator", json!({"expression": "1"}))),
            ""
        );
    }
}
