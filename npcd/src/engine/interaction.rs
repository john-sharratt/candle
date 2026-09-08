//! A person and a character, in one another's company for a while.
//!
//! # What an interaction is, and what it is not
//!
//! It is a **session**: somebody outside the world is present to a character, in
//! a mode, for as long as neither of them walks away. Opening one does not fork
//! anything and does not start a second loop — the character goes on being the
//! character, ticking on its own schedule, in the one world it stands in.
//!
//! That is deliberate, and it is the difference between this and a chat window.
//! A forked substrate would give an operator a private copy of a mind to talk
//! at: nothing said in it would have happened, the character would not remember
//! it afterwards, and the rest of the world could not see that it had taken
//! place. What is said here goes through the same door everything else does —
//! [`crate::engine::pulse::inject`] delivers it, the character perceives it on
//! its next turn, and it answers with the acts it always had.
//!
//! So the session holds **who is present, in what mode, and since when**. The
//! conversation itself lives where every other thing that happened lives.
//!
//! # Why the mode matters
//!
//! [`Mode`] decides what a character can *do* while you are there, through
//! [`crate::engine::tools::specs_within`]: somebody on a voice call cannot be
//! handed an object and cannot be shown a picture, and a character standing in
//! the room with you can be. The mode is therefore not decoration on a
//! transcript — it is the reach each of you has, and the grammar is built from
//! it.
//!
//! # Idle, and why it is a fact rather than a timer
//!
//! A session ends when nobody has said anything for long enough. There is no
//! task doing the reaping: `idle_remaining` is computed from the last thing
//! said, so a session nobody has looked at for an hour is already over by the
//! time anybody asks, and a daemon that was asleep does not wake up owing
//! anybody a sweep of expired conversations.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::Serialize;
use serde_json::{json, Value};

use crate::engine::tools::Mode;

/// How long a session survives with nothing said on it.
///
/// **Physical company is the short one, and that is not an oversight.** Standing
/// in a room with somebody who has said nothing for five minutes is a thing that
/// has ended; a message thread left for a day has not. The mode is the whole
/// difference between the two, so it is what decides.
fn idle_timeout_secs(mode: Mode) -> u64 {
    match mode {
        Mode::Physical => 300,
        Mode::VoiceCall | Mode::VideoCall => 900,
        Mode::InstantMessage => 86_400,
    }
}

/// Who the character is with.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Interlocutor {
    /// `operator` for a person at the console. Named rather than assumed
    /// because a character may one day be on the other end.
    pub kind: String,
    pub id: String,
    pub display: String,
}

/// One live session.
#[derive(Clone, Debug)]
pub struct Interaction {
    pub id: String,
    pub npc_id: u64,
    pub mode: Mode,
    pub interlocutor: Interlocutor,
    pub opened_world_ms: u64,
    /// When something was last said, either way. What idle is measured from.
    pub last_ms: u64,
    pub act_count: usize,
    pub narration_count: usize,
    /// Ended deliberately, rather than by going quiet.
    pub closed: bool,
}

impl Interaction {
    /// Seconds left before going quiet ends it. Zero once it has.
    pub fn idle_remaining_secs(&self, now_ms: u64) -> u64 {
        let quiet = now_ms.saturating_sub(self.last_ms) / 1000;
        idle_timeout_secs(self.mode).saturating_sub(quiet)
    }

    /// Whether it is still going: not ended by hand, and not gone quiet.
    pub fn is_live(&self, now_ms: u64) -> bool {
        !self.closed && self.idle_remaining_secs(now_ms) > 0
    }

    /// The shape the console reads.
    pub fn wire(&self, now_ms: u64) -> Value {
        json!({
            "interaction_id": self.id,
            // A string, for the reason `TickRecord::npc_id` is one: a real
            // character id is past the 2^53 where a JavaScript number stops
            // being exact, and an id that arrives rounded matches nothing else
            // on the page.
            "npc_id": self.npc_id.to_string(),
            "mode": self.mode.as_wire(),
            "interlocutor": self.interlocutor,
            "state": if self.is_live(now_ms) { "live" } else { "ended" },
            "idle_timeout_secs": idle_timeout_secs(self.mode),
            "idle_remaining_secs": self.idle_remaining_secs(now_ms),
            "act_count": self.act_count,
            "narration_count": self.narration_count,
            "opened_world_ms": self.opened_world_ms,
        })
    }
}

/// Every session this daemon is holding.
///
/// Bounded by nothing but the mode timeouts, which is sound because an expired
/// session is dropped the next time the map is touched — see [`Interactions::sweep`].
#[derive(Default)]
pub struct Interactions {
    live: Mutex<BTreeMap<String, Interaction>>,
    /// Monotonic, so two sessions opened in the same millisecond are two.
    next: Mutex<u64>,
}

impl Interactions {
    pub fn new() -> Interactions {
        Interactions::default()
    }

    fn mint(&self, now_ms: u64) -> String {
        let mut n = self.next.lock().expect("interaction ids");
        *n += 1;
        format!("{now_ms}{:04}", *n)
    }

    /// Drop what has gone quiet. Called on every read, so nothing has to reap.
    fn sweep(&self, live: &mut BTreeMap<String, Interaction>, now_ms: u64) {
        live.retain(|_, ix| ix.is_live(now_ms));
    }

    /// Open one. **A character has at most one session per person and mode** —
    /// asking again continues the one that is already open rather than making a
    /// second, because two live sessions with the same person in the same mode
    /// is not a thing that can be true of a conversation.
    pub fn open(
        &self,
        npc_id: u64,
        mode: Mode,
        interlocutor: Interlocutor,
        now_ms: u64,
    ) -> Interaction {
        let mut live = self.live.lock().expect("interactions");
        self.sweep(&mut live, now_ms);
        if let Some(found) = live.values_mut().find(|ix| {
            ix.npc_id == npc_id && ix.mode == mode && ix.interlocutor.id == interlocutor.id
        }) {
            found.last_ms = now_ms;
            return found.clone();
        }
        let ix = Interaction {
            id: self.mint(now_ms),
            npc_id,
            mode,
            interlocutor,
            opened_world_ms: now_ms,
            last_ms: now_ms,
            act_count: 0,
            narration_count: 0,
            closed: false,
        };
        live.insert(ix.id.clone(), ix.clone());
        ix
    }

    pub fn get(&self, id: &str, now_ms: u64) -> Option<Interaction> {
        let mut live = self.live.lock().expect("interactions");
        self.sweep(&mut live, now_ms);
        live.get(id).cloned()
    }

    /// Every live session a character is in.
    pub fn for_npc(&self, npc_id: u64, now_ms: u64) -> Vec<Interaction> {
        let mut live = self.live.lock().expect("interactions");
        self.sweep(&mut live, now_ms);
        live.values().filter(|ix| ix.npc_id == npc_id).cloned().collect()
    }

    /// Note that something was said, so the session does not go quiet under
    /// somebody who is still talking.
    pub fn touched(&self, id: &str, now_ms: u64) -> bool {
        let mut live = self.live.lock().expect("interactions");
        match live.get_mut(id) {
            Some(ix) if !ix.closed => {
                ix.last_ms = now_ms;
                ix.act_count += 1;
                true
            }
            _ => false,
        }
    }

    /// End one deliberately. `false` if there was none.
    pub fn end(&self, id: &str) -> bool {
        let mut live = self.live.lock().expect("interactions");
        live.remove(id).is_some()
    }

    /// How many are open, for the character record.
    pub fn count_for(&self, npc_id: u64, now_ms: u64) -> usize {
        self.for_npc(npc_id, now_ms).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn me() -> Interlocutor {
        Interlocutor {
            kind: "operator".into(),
            id: "u_8812".into(),
            display: "Wren".into(),
        }
    }

    const T0: u64 = 1_700_000_000_000;

    #[test]
    fn opening_one_puts_it_on_the_list() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), T0);
        assert_eq!(ix.for_npc(7, T0).len(), 1);
        assert_eq!(ix.get(&one.id, T0).map(|i| i.npc_id), Some(7));
        assert!(ix.for_npc(8, T0).is_empty());
    }

    /// **Asking twice continues the one that is open.** Two live sessions with
    /// the same person in the same mode is not a thing that can be true of a
    /// conversation, and a console that opened a second would leave the first
    /// running with nobody reading it.
    #[test]
    fn opening_the_same_conversation_twice_continues_it() {
        let ix = Interactions::new();
        let a = ix.open(7, Mode::Physical, me(), T0);
        let b = ix.open(7, Mode::Physical, me(), T0 + 5_000);
        assert_eq!(a.id, b.id);
        assert_eq!(ix.for_npc(7, T0 + 5_000).len(), 1);
    }

    /// A different mode is a different conversation: standing in the room with
    /// somebody and messaging them are not the same company.
    #[test]
    fn a_different_mode_is_a_different_conversation() {
        let ix = Interactions::new();
        let a = ix.open(7, Mode::Physical, me(), T0);
        let b = ix.open(7, Mode::InstantMessage, me(), T0);
        assert_ne!(a.id, b.id);
        assert_eq!(ix.for_npc(7, T0).len(), 2);
    }

    /// **Going quiet ends it, and nothing has to notice.** Computed from the
    /// last thing said rather than reaped by a timer, so a daemon that was
    /// asleep does not wake up owing anybody a sweep.
    #[test]
    fn a_session_nobody_has_spoken_on_is_over_without_being_reaped() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), T0);
        assert!(one.is_live(T0 + 299_000));
        assert!(!one.is_live(T0 + 301_000));
        // And it is gone from the listing rather than lingering as ended.
        assert!(ix.for_npc(7, T0 + 301_000).is_empty());
        assert!(ix.get(&one.id, T0 + 301_000).is_none());
    }

    /// Standing in a room is the short one; a message thread left for a day is
    /// not a conversation that ended.
    #[test]
    fn how_long_quiet_is_allowed_depends_on_the_mode() {
        let ix = Interactions::new();
        let room = ix.open(7, Mode::Physical, me(), T0);
        let thread = ix.open(8, Mode::InstantMessage, me(), T0);
        assert!(!room.is_live(T0 + 600_000));
        assert!(thread.is_live(T0 + 600_000));
    }

    #[test]
    fn saying_something_keeps_it_from_going_quiet() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), T0);
        assert!(ix.touched(&one.id, T0 + 200_000));
        // 200s in, the clock restarted — so at 400s it is still live, where an
        // untouched one would have ended at 300.
        assert_eq!(ix.for_npc(7, T0 + 400_000).len(), 1);
        assert_eq!(ix.get(&one.id, T0 + 400_000).unwrap().act_count, 1);
    }

    #[test]
    fn ending_one_takes_it_off_the_list() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), T0);
        assert!(ix.end(&one.id));
        assert!(!ix.end(&one.id), "ended twice");
        assert!(ix.for_npc(7, T0).is_empty());
        assert!(!ix.touched(&one.id, T0), "an ended session took a line");
    }

    /// The console reads ids as strings, because a real character id is past
    /// the 2^53 where a JavaScript number stops being exact.
    #[test]
    fn the_wire_shape_carries_the_character_id_as_a_string() {
        let ix = Interactions::new();
        let one = ix.open(6_817_662_845_163_923_144, Mode::VoiceCall, me(), T0);
        let w = one.wire(T0);
        assert_eq!(w["npc_id"], "6817662845163923144");
        assert_eq!(w["mode"], "voice_call");
        assert_eq!(w["state"], "live");
        assert_eq!(w["idle_timeout_secs"], 900);
        assert_eq!(w["interlocutor"]["display"], "Wren");
    }

    #[test]
    fn two_opened_in_one_millisecond_are_two() {
        let ix = Interactions::new();
        let a = ix.open(7, Mode::Physical, me(), T0);
        let b = ix.open(8, Mode::Physical, me(), T0);
        assert_ne!(a.id, b.id);
    }
}
