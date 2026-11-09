//! ZenJ's tools on the `/mcp` surface (§13), served by `zenj` itself at
//! `/v1/zenj/mcp` and listed by the aggregator as the `dj` backend. Each
//! answers from memory — the live view, the event log, or a one-line reply
//! from the brain's loop — inside the 30 ms rule.
//!
//! | Module | Tool |
//! |---|---|
//! | [`now`] | `zenj_now` — the decks, who has them, the next track and why |
//! | [`queue`] | `zenj_queue` — the next and the requests waiting; pin one next |
//! | [`request`] | `zenj_request` — ask for a track |
//! | [`hands`] | `zenj_hands` — Human, Assist or ZenJ |
//! | [`rekordbox`] | `zenj_rekordbox` — open or close rekordbox |
//! | [`lights`] | `lights_scene`, `lights_house` — a scene held, the house lights |
//! | [`moment`] | `zenj_moment` — book a track, a dedication or a sample for a time |
//! | [`sfx`] | `zenj_sfx` — fire a sample now or on the next beat or bar |
//! | [`events`] | `zenj_events` — what happened since a cursor |
//! | [`said`] | `zenj_said` — a line of the room's conversation, for Zen's next choice |
//! | [`midnight`] | `zenj_midnight` — the track whose drop lands on midnight |
//! | [`energy`] | `zenj_energy` — more or less energy for a while |

pub mod energy;
pub mod events;
pub mod hands;
pub mod lights;
pub mod midnight;
pub mod moment;
pub mod now;
pub mod queue;
pub mod rekordbox;
pub mod request;
pub mod said;
pub mod sfx;

use std::net::IpAddr;
use std::sync::{Arc, LazyLock};

use arc_swap::ArcSwap;
use axum::http::HeaderMap;
use mcp_server::listing;
use mcp_server::protocol::text_result;
use mcp_server::{Call, Identity, NoCaller, SealedStore, Server, Store};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};
use web::auth::forwarded::identify;
use web::auth::session::Identity as Person;

use crate::brain::run::Ask;
use crate::events::Events;
use crate::live::Live;
use crate::rekordbox::control::Rekordbox;

const DESCRIPTION: &str = "ZenJ: Zen as DJ for the party at John's — what is playing and next, \
requests, who has the decks, moments and samples, rekordbox itself, and what the room says.";

const INSTRUCTIONS: &str = "zenj_now says what is playing, who has the decks and the next track \
with its reason. Requests go to zenj_request; a line someone in the room said to you goes to \
zenj_said, so the next choice hears it. zenj_hands hands the decks between a person and ZenJ: \
moving toward ZenJ needs a person's yes in this chat first. zenj_rekordbox opens or closes \
rekordbox. lights_scene holds a lighting scene and lights_house turns the house lights on. \
zenj_events follows what happens.";

static IDENTITY: LazyLock<Identity> = LazyLock::new(|| Identity {
    name: "zenj".into(),
    title: "ZenJ".into(),
    description: DESCRIPTION.into(),
    instructions: INSTRUCTIONS.into(),
    icon: None,
    version: env!("CARGO_PKG_VERSION").into(),
    // Its client is the aggregator, which tells its own clients.
    list_changed: false,
});

/// The version of what `tools/list` answers: every caller is listed the same
/// tools, so one version, once.
static TOOLS_VERSION: LazyLock<String> = LazyLock::new(|| listing::version(&definitions()));

/// What a guest — signed in, not one of the night's people — is told.
const NOT_LISTED: &str = "ZenJ is for the people whose party it is; ask John to add you";

/// A tool's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub text: String,
    pub is_error: bool,
}

impl Answer {
    pub fn ok(text: impl Into<String>) -> Answer {
        Answer {
            text: text.into(),
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Answer {
        Answer {
            text: text.into(),
            is_error: true,
        }
    }
}

/// The DJ's server: what the tools read, and where they send what they ask.
pub struct Dj {
    pub live: Arc<ArcSwap<Live>>,
    pub events: Arc<Events>,
    pub asks: mpsc::Sender<Ask>,
    pub rekordbox: Arc<Rekordbox>,
    /// The lights thread — `None` with no bulbs mapped.
    pub lights: Option<lights::LightsHandle>,
    /// Who may call, by email, and the name each is known by.
    pub people: Vec<(String, String)>,
    pub store: Arc<SealedStore>,
}

impl Dj {
    /// The name a caller is known by: their person's, else their own.
    pub fn name_of(&self, p: &Person) -> String {
        let email = p.email.to_ascii_lowercase();
        self.people
            .iter()
            .find(|(e, _)| *e == email)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| p.name.clone())
    }

    /// Send `make`'s ask to the brain and wait for its one line.
    pub async fn ask(&self, make: impl FnOnce(oneshot::Sender<String>) -> Ask) -> Answer {
        let (tx, rx) = oneshot::channel();
        if self.asks.send(make(tx)).await.is_err() {
            return Answer::error("ZenJ's brain is not running");
        }
        match rx.await {
            Ok(text) => Answer::ok(text),
            Err(_) => Answer::error("ZenJ's brain did not answer"),
        }
    }
}

/// One tool.
pub trait Tool {
    const NAME: &'static str;
    fn definition() -> Value;
}

pub fn definitions() -> Value {
    json!([
        now::Now::definition(),
        queue::Queue::definition(),
        request::RequestTool::definition(),
        hands::HandsTool::definition(),
        rekordbox::RekordboxTool::definition(),
        moment::MomentTool::definition(),
        sfx::Sfx::definition(),
        events::EventsTool::definition(),
        said::Said::definition(),
        midnight::Midnight::definition(),
        energy::Energy::definition(),
        lights::SceneTool::definition(),
        lights::HouseTool::definition(),
    ])
}

pub async fn call(dj: &Dj, who: &Person, name: &str, args: &Value) -> Answer {
    let by = dj.name_of(who);
    match name {
        now::Now::NAME => now::call(dj),
        queue::Queue::NAME => queue::call(dj, args).await,
        request::RequestTool::NAME => request::call(dj, &by, args).await,
        hands::HandsTool::NAME => hands::call(dj, args).await,
        rekordbox::RekordboxTool::NAME => rekordbox::call(dj, args).await,
        moment::MomentTool::NAME => moment::call(dj, args).await,
        sfx::Sfx::NAME => sfx::call(dj, args).await,
        events::EventsTool::NAME => events::call(dj, args),
        said::Said::NAME => said::call(dj, &by, args).await,
        midnight::Midnight::NAME => midnight::call(dj, args).await,
        energy::Energy::NAME => energy::call(dj, args).await,
        lights::SceneTool::NAME => lights::scene(dj, args),
        lights::HouseTool::NAME => lights::house(dj, args),
        other => Answer::error(format!("there is no tool `{other}`")),
    }
}

/// A required string argument.
pub fn text<'a>(args: &'a Value, key: &str) -> Result<&'a str, Answer> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Answer::error(format!("`{key}` is required")))
}

impl Server for Dj {
    type Caller = Person;
    type Answer = Answer;

    fn identity(&self) -> &Identity {
        &IDENTITY
    }

    fn caller(&self, headers: &HeaderMap, _peer: Option<IpAddr>) -> Result<Person, NoCaller> {
        let Ok(p) = identify(headers) else {
            return Err(NoCaller::SignedOut);
        };
        let email = p.email.to_ascii_lowercase();
        if self.people.iter().any(|(e, _)| *e == email) {
            Ok(p)
        } else {
            Err(NoCaller::NotAllowed(NOT_LISTED.into()))
        }
    }

    fn owner(&self, caller: &Person) -> String {
        let digest = Sha256::digest(format!("{}\0{}", caller.provider, caller.sub).as_bytes());
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }

    fn store(&self) -> Result<Arc<dyn Store>, String> {
        Ok(self.store.clone())
    }

    async fn tools(_this: &Arc<Self>, _call: &Call<Self>) -> Result<Value, String> {
        Ok(definitions())
    }

    async fn call(this: &Arc<Self>, call: &Call<Self>, name: &str, args: &Value) -> Answer {
        self::call(this, &call.caller, name, args).await
    }

    fn tools_version(&self) -> Option<&str> {
        Some(&TOOLS_VERSION)
    }

    /// zenj's client is the aggregator, which tells its own clients.
    fn list_changed(&self, _call: &Call<Self>) -> bool {
        false
    }

    fn result(&self, answer: &Answer) -> Value {
        text_result(&answer.text, answer.is_error)
    }

    async fn delivered(_this: &Arc<Self>, _call: &Call<Self>, _answer: &Answer) {}
}

#[cfg(test)]
pub mod fixture {
    use std::sync::atomic::AtomicBool;

    use super::*;

    /// A server over a live view, an event log, and the receiving end of
    /// its asks.
    pub fn dj() -> (Dj, mpsc::Receiver<Ask>) {
        let (tx, rx) = mpsc::channel(8);
        let live = Arc::new(ArcSwap::from_pointee(Live::default()));
        let events = Arc::new(Events::new());
        (
            Dj {
                rekordbox: Arc::new(Rekordbox::new(
                    "Z:/no/such/rekordbox.exe".into(),
                    "zenj-no-such-process.exe",
                    live.clone(),
                    events.clone(),
                    Arc::new(AtomicBool::new(false)),
                )),
                live,
                events,
                asks: tx,
                lights: None,
                people: vec![("john@example.com".into(), "john".into())],
                store: Arc::new(SealedStore),
            },
            rx,
        )
    }

    pub fn person(email: &str, name: &str) -> Person {
        Person {
            provider: "google".into(),
            sub: "1".into(),
            email: email.into(),
            name: name.into(),
            picture: String::new(),
            exp: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    #[test]
    fn every_tool_has_its_name_a_description_and_an_object_schema() {
        let names: Vec<String> = definitions()
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                assert!(d["description"].as_str().unwrap().len() > 40, "{d}");
                assert_eq!(d["inputSchema"]["type"], "object", "{d}");
                d["name"].as_str().unwrap().to_string()
            })
            .collect();
        assert_eq!(
            names,
            [
                "zenj_now",
                "zenj_queue",
                "zenj_request",
                "zenj_hands",
                "zenj_rekordbox",
                "zenj_moment",
                "zenj_sfx",
                "zenj_events",
                "zenj_said",
                "zenj_midnight",
                "zenj_energy",
                "lights_scene",
                "lights_house"
            ]
        );
    }

    #[test]
    fn a_caller_is_known_by_their_persons_name() {
        let (dj, _) = dj();
        assert_eq!(dj.name_of(&person("John@Example.com", "J S")), "john");
        assert_eq!(dj.name_of(&person("x@y.z", "Guest")), "Guest");
    }
}
