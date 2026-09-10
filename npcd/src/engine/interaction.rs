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
    /// The world's own instant it began at, for the console's label.
    pub opened_world_ms: u64,
    /// When something was last said, either way, **on the wall clock**.
    ///
    /// **Not the world's clock, and the difference is not academic.** Going
    /// quiet is a fact about a person at a console — five real minutes of
    /// saying nothing — while the world's clock is the characters' and can be
    /// paused, jumped or run at a different rate. Measuring one against the
    /// other made every session read as expired the moment the two diverged,
    /// which took the visitor's body out of the room while they were still
    /// typing.
    pub last_ms: u64,
    pub act_count: usize,
    pub narration_count: usize,
    /// Ended deliberately, rather than by going quiet.
    pub closed: bool,
    /// The world this person is standing in, for a physical session. `None`
    /// for the modes that reach somebody who is nowhere near.
    pub world: Option<String>,
    /// The body this person has in that world.
    ///
    /// **Derived from the account rather than minted per session**, so opening
    /// the same conversation twice does not leave two of somebody standing in
    /// one room — and so a character that was told a name yesterday is told the
    /// same one today.
    pub body: String,
}

/// The body a person has when they walk into the world.
///
/// Prefixed, because a body id shares a namespace with the characters' and a
/// person is not one: nothing should ever bind a mind to this, and the prefix
/// is what makes that visible at a glance in a log or a world dump.
pub fn body_of(handle: &str) -> String {
    format!("visitor:{handle}")
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
    /// `now_ms` is the wall clock, which is what going quiet is measured
    /// against; `world_ms` is the world's own instant, which is only a label.
    pub fn open(
        &self,
        npc_id: u64,
        mode: Mode,
        interlocutor: Interlocutor,
        world: Option<String>,
        now_ms: u64,
        world_ms: u64,
    ) -> Interaction {
        let mut live = self.live.lock().expect("interactions");
        self.sweep(&mut live, now_ms);
        if let Some(found) = live.values_mut().find(|ix| {
            ix.npc_id == npc_id && ix.mode == mode && ix.interlocutor.id == interlocutor.id
        }) {
            found.last_ms = now_ms;
            return found.clone();
        }
        let body = body_of(&interlocutor.id);
        let ix = Interaction {
            id: self.mint(now_ms),
            npc_id,
            mode,
            interlocutor,
            opened_world_ms: world_ms,
            last_ms: now_ms,
            act_count: 0,
            narration_count: 0,
            closed: false,
            world,
            body,
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
        live.values()
            .filter(|ix| ix.npc_id == npc_id)
            .cloned()
            .collect()
    }

    /// Note that somebody is **there**, without anything having been said.
    ///
    /// A console holding the stream open is a person in the room. Going quiet is
    /// meant to catch the one who walked off, and measuring it from the last
    /// *line* instead of the last sign of life threw out anybody who sat and
    /// listened for five minutes — which is most of what being in a room with
    /// somebody consists of. Distinct from [`Interactions::touched`] because it
    /// is not an act and must not be counted as one.
    pub fn attended(&self, id: &str, now_ms: u64) -> bool {
        let mut live = self.live.lock().expect("interactions");
        match live.get_mut(id) {
            Some(ix) if !ix.closed => {
                ix.last_ms = now_ms;
                true
            }
            _ => false,
        }
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

    /// Every person standing in a world right now, and who they came to see.
    ///
    /// Only the physical sessions: a message thread reaches somebody who is
    /// nowhere near, and putting its author into the room would make "not being
    /// there" impossible to express.
    pub fn visiting(&self, world: &str, now_ms: u64) -> Vec<(String, u64)> {
        let mut live = self.live.lock().expect("interactions");
        self.sweep(&mut live, now_ms);
        live.values()
            .filter(|ix| ix.mode == Mode::Physical && ix.world.as_deref() == Some(world))
            .map(|ix| (ix.body.clone(), ix.npc_id))
            .collect()
    }

    /// Everything that has gone quiet since anybody last looked, so their
    /// bodies can be taken out of the world.
    ///
    /// **Expiry has to be observable, not merely computed.** `is_live` makes an
    /// abandoned session read as gone, which is enough for a console; it is not
    /// enough for a body standing in a room, because nothing about a number
    /// going to zero removes an actor. This is what the sweep drains.
    pub fn take_expired(&self, now_ms: u64) -> Vec<Interaction> {
        let mut live = self.live.lock().expect("interactions");
        let gone: Vec<Interaction> = live
            .values()
            .filter(|ix| !ix.is_live(now_ms))
            .cloned()
            .collect();
        live.retain(|_, ix| ix.is_live(now_ms));
        gone
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

    /// The world a visitor is standing in. `None` for the modes that reach
    /// somebody who is nowhere near.
    fn here() -> Option<String> {
        Some("creators-vault".to_string())
    }

    /// The wall clock, which is what going quiet is measured against.
    const T0: u64 = 1_700_000_000_000;

    /// The world's own instant, deliberately nowhere near [`T0`] — the two are
    /// different clocks, and a test that used one number for both would pass
    /// with them swapped.
    const WORLD_T0: u64 = 42_000;

    #[test]
    fn opening_one_puts_it_on_the_list() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
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
        let a = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        let b = ix.open(7, Mode::Physical, me(), here(), T0 + 5_000, WORLD_T0);
        assert_eq!(a.id, b.id);
        assert_eq!(ix.for_npc(7, T0 + 5_000).len(), 1);
    }

    /// A different mode is a different conversation: standing in the room with
    /// somebody and messaging them are not the same company.
    #[test]
    fn a_different_mode_is_a_different_conversation() {
        let ix = Interactions::new();
        let a = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        let b = ix.open(7, Mode::InstantMessage, me(), here(), T0, WORLD_T0);
        assert_ne!(a.id, b.id);
        assert_eq!(ix.for_npc(7, T0).len(), 2);
    }

    /// **Going quiet ends it, and nothing has to notice.** Computed from the
    /// last thing said rather than reaped by a timer, so a daemon that was
    /// asleep does not wake up owing anybody a sweep.
    #[test]
    fn a_session_nobody_has_spoken_on_is_over_without_being_reaped() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
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
        let room = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        let thread = ix.open(8, Mode::InstantMessage, me(), here(), T0, WORLD_T0);
        assert!(!room.is_live(T0 + 600_000));
        assert!(thread.is_live(T0 + 600_000));
    }

    #[test]
    fn saying_something_keeps_it_from_going_quiet() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        assert!(ix.touched(&one.id, T0 + 200_000));
        // 200s in, the clock restarted — so at 400s it is still live, where an
        // untouched one would have ended at 300.
        assert_eq!(ix.for_npc(7, T0 + 400_000).len(), 1);
        assert_eq!(ix.get(&one.id, T0 + 400_000).unwrap().act_count, 1);
    }

    /// **Listening is being there.** Idle is meant to catch the person who
    /// walked off, and it was measured from the last thing *said* — so somebody
    /// who sat and listened for five minutes had their body taken out of the
    /// room from under them while they were still watching.
    #[test]
    fn a_console_holding_the_stream_open_is_somebody_in_the_room() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        assert!(ix.attended(&one.id, T0 + 200_000));
        assert_eq!(ix.for_npc(7, T0 + 400_000).len(), 1);
        // Being there is not an act. `act_count` is what the console shows as
        // the character's own doing, and a poll tick is nobody's.
        assert_eq!(ix.get(&one.id, T0 + 400_000).unwrap().act_count, 0);
    }

    #[test]
    fn attending_a_session_that_is_over_says_so() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        assert!(ix.end(&one.id));
        assert!(!ix.attended(&one.id, T0), "an ended session took a poll");
    }

    #[test]
    fn ending_one_takes_it_off_the_list() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
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
        let one = ix.open(
            6_817_662_845_163_923_144,
            Mode::InstantMessage,
            me(),
            here(),
            T0,
            WORLD_T0,
        );
        let w = one.wire(T0);
        assert_eq!(w["npc_id"], "6817662845163923144");
        assert_eq!(w["mode"], "instant_message");
        assert_eq!(w["state"], "live");
        assert_eq!(w["idle_timeout_secs"], 86_400);
        assert_eq!(w["interlocutor"]["display"], "Wren");
    }

    // ── visiting ────────────────────────────────────────────────────────────

    /// A body derived from the account, not minted per session — or opening the
    /// same conversation twice leaves two of somebody standing in one room.
    #[test]
    fn one_person_has_one_body_however_often_they_open_it() {
        let ix = Interactions::new();
        let a = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        let b = ix.open(7, Mode::Physical, me(), here(), T0 + 5_000, WORLD_T0);
        let c = ix.open(9, Mode::Physical, me(), here(), T0, WORLD_T0);
        assert_eq!(a.body, b.body);
        assert_eq!(a.body, c.body, "one person, two conversations, two bodies");
        assert_eq!(a.body, body_of("u_8812"));
    }

    /// **Only the physical sessions put somebody in a room.** A message thread
    /// reaches somebody who is nowhere near, and standing its author in the
    /// room would make "not being there" impossible to express.
    #[test]
    fn only_a_physical_session_is_standing_in_the_world() {
        let ix = Interactions::new();
        ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        ix.open(8, Mode::InstantMessage, me(), None, T0, WORLD_T0);

        let standing = ix.visiting("creators-vault", T0);
        assert_eq!(standing.len(), 1, "{standing:?}");
        assert_eq!(standing[0].1, 7);
    }

    #[test]
    fn a_visitor_is_only_in_the_world_it_is_standing_in() {
        let ix = Interactions::new();
        ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        assert!(ix.visiting("battle-cities", T0).is_empty());
    }

    /// **A session going quiet has to actually hand its body back.** `is_live`
    /// makes it read as gone and does nothing whatever to an actor standing in
    /// a room, so a visitor who closed the tab would stay in the vault for ever
    /// and the character would go on being told it had company.
    #[test]
    fn what_has_gone_quiet_is_handed_back_once() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);

        assert!(
            ix.take_expired(T0 + 10_000).is_empty(),
            "taken while still live"
        );
        let gone = ix.take_expired(T0 + 301_000);
        assert_eq!(gone.len(), 1);
        assert_eq!(gone[0].id, one.id);
        assert_eq!(gone[0].body, body_of("u_8812"));

        // Drained, so a second sweep does not try to remove the same body again.
        assert!(ix.take_expired(T0 + 301_000).is_empty());
        assert!(ix.visiting("creators-vault", T0 + 301_000).is_empty());
    }

    #[test]
    fn ending_by_hand_takes_it_off_before_the_sweep_sees_it() {
        let ix = Interactions::new();
        let one = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        assert!(ix.end(&one.id));
        assert!(ix.take_expired(T0 + 301_000).is_empty());
        assert!(ix.visiting("creators-vault", T0).is_empty());
    }

    #[test]
    fn two_opened_in_one_millisecond_are_two() {
        let ix = Interactions::new();
        let a = ix.open(7, Mode::Physical, me(), here(), T0, WORLD_T0);
        let b = ix.open(8, Mode::Physical, me(), here(), T0, WORLD_T0);
        assert_ne!(a.id, b.id);
    }
}
