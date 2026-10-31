//! Discord's gateway protocol, as much of it as a bot that hears its DMs
//! speaks (`docs.discord.com/developers/events/gateway`): the payloads it
//! sends, the one it reads, and what a close code means for the shard.

use std::env::consts::OS;

use serde::Deserialize;
use serde_json::{json, Value};

/// The gateway's address and the protocol it is spoken in.
pub const PRODUCTION: &str = "wss://gateway.discord.gg";

/// What every connection asks for: version 10, JSON, uncompressed.
pub const QUERY: &str = "/?v=10&encoding=json";

/// The `DIRECT_MESSAGES` intent: a bot's own DMs, their content included.
pub const DIRECT_MESSAGES: u64 = 1 << 12;

/// The gateway's opcodes.
pub mod op {
    pub const DISPATCH: u8 = 0;
    pub const HEARTBEAT: u8 = 1;
    pub const IDENTIFY: u8 = 2;
    pub const RESUME: u8 = 6;
    pub const RECONNECT: u8 = 7;
    pub const INVALID_SESSION: u8 = 9;
    pub const HELLO: u8 = 10;
    pub const HEARTBEAT_ACK: u8 = 11;
}

/// The close code a shard leaves with when it means to resume: anything but
/// `1000` and `1001`, which end the session.
pub const CLOSE_TO_RESUME: u16 = 4000;

/// One payload, as the gateway sends it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Payload {
    pub op: u8,
    #[serde(default)]
    pub d: Value,
    #[serde(default)]
    pub s: Option<u64>,
    #[serde(default)]
    pub t: Option<String>,
}

/// The URL a shard connects to, from a gateway's base.
pub fn url(base: &str) -> String {
    format!("{}{QUERY}", base.trim_end_matches('/'))
}

/// The presence a shard comes online with: online, its status line saying
/// where its character is ("in Life").
pub fn presence(activity: &str) -> Value {
    json!({
        "since": null,
        "activities": [{ "name": "Custom Status", "type": 4, "state": activity }],
        "status": "online",
        "afk": false,
    })
}

/// `IDENTIFY`: a new session for `token`, hearing `intents`, coming online
/// with `activity`.
pub fn identify(token: &str, intents: u64, activity: &str) -> Value {
    json!({
        "op": op::IDENTIFY,
        "d": {
            "token": token,
            "intents": intents,
            "properties": { "os": OS, "browser": "npcd", "device": "npcd" },
            "presence": presence(activity),
        },
    })
}

/// `RESUME`: session `session_id` taken up again after the last dispatch it
/// saw, `seq`.
pub fn resume(token: &str, session_id: &str, seq: Option<u64>) -> Value {
    json!({
        "op": op::RESUME,
        "d": { "token": token, "session_id": session_id, "seq": seq },
    })
}

/// `HEARTBEAT`, naming the last dispatch seen.
pub fn heartbeat(seq: Option<u64>) -> Value {
    json!({ "op": op::HEARTBEAT, "d": seq })
}

/// What a close code means for a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    /// `4004`: Discord does not know the token. The shard stops until a new
    /// one is connected.
    Refused,
    /// The shard asked for something Discord will never give — an intent it
    /// is not allowed, a version it does not speak — and stops.
    Fatal,
    /// The session is gone: start a new one.
    Reidentify,
    /// The connection went; the session goes on.
    Resume,
}

/// What close `code` means — `None` for a connection that ended without one.
pub fn closed(code: Option<u16>) -> Closed {
    match code {
        Some(4004) => Closed::Refused,
        Some(4010..=4014) => Closed::Fatal,
        Some(4007 | 4009) => Closed::Reidentify,
        _ => Closed::Resume,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **`IDENTIFY` is the payload Discord documents**, with only the DM
    /// intent, and a status line naming the world.
    #[test]
    fn identify_asks_for_dms_and_names_the_world() {
        assert_eq!(
            identify("tok", DIRECT_MESSAGES, "in Life"),
            json!({
                "op": 2,
                "d": {
                    "token": "tok",
                    "intents": 4096,
                    "properties": { "os": OS, "browser": "npcd", "device": "npcd" },
                    "presence": {
                        "since": null,
                        "activities": [{ "name": "Custom Status", "type": 4, "state": "in Life" }],
                        "status": "online",
                        "afk": false,
                    },
                },
            })
        );
    }

    #[test]
    fn resume_and_heartbeat_name_the_last_dispatch() {
        assert_eq!(
            resume("tok", "sess-1", Some(4)),
            json!({ "op": 6, "d": { "token": "tok", "session_id": "sess-1", "seq": 4 } })
        );
        assert_eq!(heartbeat(None), json!({ "op": 1, "d": null }));
        assert_eq!(heartbeat(Some(9)), json!({ "op": 1, "d": 9 }));
        assert_eq!(
            url("wss://gateway.discord.gg/"),
            "wss://gateway.discord.gg/?v=10&encoding=json"
        );
    }

    /// **4004 is a refused token; the 4010s stop; 4007 and 4009 start
    /// afresh; anything else resumes.**
    #[test]
    fn a_close_code_says_what_the_shard_does() {
        assert_eq!(closed(Some(4004)), Closed::Refused);
        for code in [4010, 4011, 4012, 4013, 4014] {
            assert_eq!(closed(Some(code)), Closed::Fatal, "{code}");
        }
        assert_eq!(closed(Some(4007)), Closed::Reidentify);
        assert_eq!(closed(Some(4009)), Closed::Reidentify);
        for code in [Some(1000), Some(1006), Some(4000), Some(4008), None] {
            assert_eq!(closed(code), Closed::Resume, "{code:?}");
        }
    }

    #[test]
    fn a_payload_reads_with_its_optional_fields_absent() {
        let p: Payload = serde_json::from_str(r#"{"op":11}"#).unwrap();
        assert_eq!(
            p,
            Payload {
                op: op::HEARTBEAT_ACK,
                d: Value::Null,
                s: None,
                t: None
            }
        );
        let p: Payload =
            serde_json::from_str(r#"{"op":0,"s":3,"t":"READY","d":{"v":10}}"#).unwrap();
        assert_eq!((p.op, p.s, p.t.as_deref()), (0, Some(3), Some("READY")));
    }
}
