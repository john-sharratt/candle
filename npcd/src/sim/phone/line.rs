//! One thing said on a thread (`docs/npc_conversations.md` §11.1): the words
//! that were said, who said them by id, and what a conversation needs to know
//! about them — the intent that decided them, the critical step they asked an
//! authorizer for, the account a person sent them from.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Where a line said on a thread came from, outside the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Origin {
    /// A Discord message: its id, and the channel it was written in.
    Discord { message: u64, channel: u64 },
}

/// One thing said on a thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who said it: a member id — a body id, or a person's account handle.
    pub from: String,
    /// The words, as they were said. What every reader is shown and what
    /// Discord posts: a character's line is decoded before it is sent, so no
    /// intent ever reaches a thread.
    pub text: String,
    /// What the character meant when it decided to say it, kept for
    /// diagnostics. Absent for a person's line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// Its name on the thread: strictly increasing, never reused, never
    /// restarted, and kept through trimming.
    pub seq: u64,
    /// When it was said, in real milliseconds.
    pub at_ms: u64,
    /// The Discord message it mirrors, so the mirror never posts it back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
    /// The member it was said to by name, in a group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    /// The critical step this line asked an authorizer to sign off: a yes
    /// that answers it signs the step, and a yes to anything else signs
    /// nothing (§7.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests: Option<String>,
    /// The account a person sent it from, as the gateway verified it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
}

/// What a line is before the thread names it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Said {
    pub text: String,
    pub intent: Option<String>,
    pub origin: Option<Origin>,
    pub to: Option<String>,
    pub requests: Option<String>,
    pub account: Option<String>,
}

impl Said {
    /// Words and nothing else: a line said inside the world.
    pub fn words(text: &str) -> Said {
        Said {
            text: text.to_string(),
            ..Said::default()
        }
    }
}

/// Now, in real milliseconds: when a line is said.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A line round-trips exactly as its record holds it**, its optional
    /// parts left out when absent.
    #[test]
    fn a_line_round_trips_through_its_record() {
        let plain = Message {
            from: "npc-7".into(),
            text: "It's at the door.".into(),
            intent: None,
            seq: 3,
            at_ms: 1_800_000_000_000,
            origin: None,
            to: None,
            requests: None,
            account: None,
        };
        let wire = serde_json::to_string(&plain).unwrap();
        assert_eq!(
            wire,
            "{\"from\":\"npc-7\",\"text\":\"It's at the door.\",\"seq\":3,\
             \"at_ms\":1800000000000}"
        );
        assert_eq!(serde_json::from_str::<Message>(&wire).unwrap(), plain);
        let full = Message {
            intent: Some("that the parcel is at the door".into()),
            origin: Some(Origin::Discord {
                message: 11,
                channel: 12,
            }),
            to: Some("npc-7".into()),
            requests: Some("Sign the lease".into()),
            account: Some("u_8812".into()),
            ..plain
        };
        let wire = serde_json::to_string(&full).unwrap();
        assert_eq!(
            wire,
            "{\"from\":\"npc-7\",\"text\":\"It's at the door.\",\
             \"intent\":\"that the parcel is at the door\",\"seq\":3,\
             \"at_ms\":1800000000000,\"origin\":{\"kind\":\"discord\",\"message\":11,\
             \"channel\":12},\"to\":\"npc-7\",\"requests\":\"Sign the lease\",\
             \"account\":\"u_8812\"}"
        );
        assert_eq!(serde_json::from_str::<Message>(&wire).unwrap(), full);
    }
}
