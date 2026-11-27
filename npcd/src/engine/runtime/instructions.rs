//! The standing instructions a character is given when nothing else asks of it.
//!
//! Having nothing to do is itself an instruction, so an idle character is set on
//! its work rather than left still. Each instruction names the next thing once,
//! with no place and no verb of motion, because a restated destination becomes a
//! route the whole cast follows identically. A named object would be one every
//! character is then told to have, so what to take up is left to what the
//! character can see. When company is present the instruction names who is there,
//! since an addressee the character cannot name cannot be reached.

use super::*;

// A bench works on the mind itself — see `Runtime::new`, which points the
// worlds at it. **The root is wide and the reach is narrow**, rather than the
// other way round: `sim::bench::EDITABLE_AREAS` decides which parts of the mind
// answer, which keeps the dangerous set (`projection.yaml`, `mind.yaml`,
// `schema/`) named in one place beside the guard that enforces it instead of
// implied by a directory nobody would think to check.

/// What a character with nothing assigned is set on.
///
/// **Not a placeholder.** A person with no task does not stand still, and an
/// NPC that does reads as scenery — so having nothing to do is itself a
/// standing instruction, and this is it. When missions exist they replace this
/// one; they do not fill a hole it was leaving.
///
/// Worded as a disposition rather than a script. *Go to the green room* would
/// be a standing order every character in the world followed identically, which
/// is the recommendation-not-affordance failure at its worst: it fires on every
/// idle turn, for every character, and never appears in a log as anything but a
/// heartbeat.
/// **Points at the work, not at the map.** The first version sent a character
/// off to see a room it had not been in, and it did exactly that — three of
/// them toured a seventy-eight room building for hours and came back with
/// nothing to say, because a room is not a subject. What a character is for is
/// in front of it, and the instruction says so without naming a thing.
///
/// **It names no particular object.** Every example it once carried — a ledger,
/// a filed story, a date on the wall — belonged to one world, and it was read by
/// the characters of all of them: one with no such thing in reach invented one,
/// described it to a colleague, and the colleague asked after it in turn. A
/// named thing in a standing instruction is a thing the whole cast is told to
/// have, so what to take up is left to what the character can actually see.
/// **One instruction, and no verb of motion in it.**
///
/// This has now been wrong in three ways. It began as *go and see a room you
/// have not been in*, and characters toured a seventy-eight room building for
/// hours and arrived with nothing to say, because a room is not a subject. It
/// was then rewritten to point at the work — and still ended with *then go
/// where people are*, which is a movement instruction sitting in the most
/// recent position in the window, where attention weights hardest. Every act in
/// the harness became `move_to`, nineteen for nineteen.
///
/// A standing task restated every ninety seconds does not need to describe a
/// plan. It needs to name the **next** thing, once. The going will happen on
/// its own when the character has something worth carrying.
pub const NO_MISSION: &str = "Nothing has been asked of you, and you are on your own. That is \
                              not nothing to do: the work is in front of you. Take up one thing \
                              you can actually see or read here, and carry it far enough to have \
                              a view about it you could defend. If nothing here has anything for \
                              you, say so.";

/// The idle wake for a character that takes no orders
/// ([`crate::sim::missions::Missions::set_unordered`]): [`NO_MISSION`]'s place,
/// with nothing set for it to do — not even "look around". It is woken all the
/// same, because with nothing else arriving a character gone quiet stays quiet.
pub const YOUR_OWN_TIME: &str = "Nobody gives you orders, and nothing is waiting on you. Your \
                                 time is your own, to spend however you choose.";

/// What a character reads at the head of a turn while `request`, which
/// `party` made of it on the phone, is still to be done and told — with
/// `table`, for a character that gives orders, what the order table holds of
/// its posting ([`orders_line`]).
///
/// **One who gives orders is told to post first.** The close once read "do it
/// yourself" for everyone, beneath a table line saying a Drone's work is
/// posted. With only the table to answer through, Zen posted every time; with
/// the cluster's terminals in reach as well, it took "yourself" at its word and
/// spent thirty turns trying to run the research it had meant to give a Drone,
/// from acts that could not reach the repository.
pub fn still_asked(party: &str, request: &str, table: Option<&str>) -> String {
    let (table, act) = match table {
        Some(t) => (
            format!("\n{t}"),
            "Do it now: post it, if it is a Drone's to carry, or else take the act that answers \
             it yourself. Then tell",
        ),
        None => (
            String::new(),
            "Do it yourself, now, with the act that answers it, then tell",
        ),
    };
    format!(
        "Still yours to do, and nothing comes before it: what {party} asked of you on the \
         phone.\n{}{table}\n{act} {party} what came of it — what you read, or that you could \
         not. Telling them you are about to is not an answer.",
        request.trim()
    )
}

/// The order table as it stands for one who posts on it: each of `out` — a
/// brief and who carries it, `None` while nobody has taken it — or that
/// nothing of theirs is there.
///
/// **The table, not the memory of it.** Zen told the Creator, from a
/// conversation that can only talk, "I have posted the order for a Drone"; the
/// conversation's journal kept that as done, and the turns after read the
/// journal, believed it, and told him the Drone had not reported yet — of an
/// order that was never posted. What is on the table is read off it every turn
/// the request stands, so a claim cannot stand in for the act.
///
/// `post` is the table's `order_post` address, named so the way to post is
/// the act itself: told only to "`order_post` it", Zen put the work on its own
/// agenda instead, an act it could name.
pub fn orders_line(out: &[(String, Option<String>)], post: Option<&str>) -> String {
    if out.is_empty() {
        let how = match post {
            Some(url) => format!("post it with `invoke` {url}"),
            None => "post it at the order table".to_string(),
        };
        return format!(
            "Nothing you have posted is on the order table: no Drone has been given any of it. \
             If it is theirs to do, {how}; until it is posted, nobody is doing it."
        );
    }
    let each: Vec<String> = out
        .iter()
        .map(|(brief, with)| {
            let brief = brief.trim().trim_end_matches('.');
            match with {
                Some(name) => format!("\"{brief}\" — {name} has it"),
                None => format!("\"{brief}\" — waiting to be taken"),
            }
        })
        .collect();
    format!("On the order table, of yours: {}.", each.join("; "))
}

/// What a character is told when its mission is called off: the errand is
/// withdrawn, nothing of it is still owed, and what it said about it earlier is
/// no longer a commitment.
pub const MISSION_WITHDRAWN: &str = "The errand you were on has been withdrawn. It is no longer \
                                     yours to finish and there is nothing to report on it. Put it \
                                     down, and do not carry its subject into what you do next \
                                     unless somebody raises it again.";

/// How many words of the mission's document `body`'s working copy holds,
/// uncommitted, when `doc` is the step that writes it and that is up to the
/// mission's floor — see [`doc_compass`].
pub(super) fn written_up_to_floor(
    s: &Sim,
    body: &str,
    mission: &Mission,
    doc: &DocStep,
) -> Option<usize> {
    let DocStep::Write(p) = doc else {
        return None;
    };
    let work = mission.work.as_ref()?;
    if work.min_words == 0 || !p.eq_ignore_ascii_case(&work.writes) {
        return None;
    }
    let words = s
        .bench
        .opened(body)?
        .changed_text(&work.writes)?
        .split_whitespace()
        .count();
    (words >= work.min_words).then_some(words)
}

/// What a character is told as the table gives it a mission
/// ([`Runtime::assign_work`]), followed by the mission as it reads it.
///
/// **A mission is given, never fetched** (`docs/zen_everywhere.md` §3). Called
/// to the table to take one up, Makers stood at it with work waiting, scanned
/// it, reflected, rode the lift and came back — forty minutes without one
/// collected — and a Drone with nothing to carry was woken every thirty
/// seconds to be told to go and wait. The table hands the work over where the
/// character is, and a character with none is at rest.
pub const MISSION_GIVEN: &str = "The command table has given you a mission, and it is what you \
                                 are working on now. Carry it out, and when it is done go to \
                                 the table and report it there.";

/// What a character **at the command table with its mission's work done** —
/// everything but the report signed off — is told. See
/// [`Runtime::at_table_report`].
pub const REPORT_AT_THE_TABLE: &str =
    "You are at the command table and every step of your mission is done but the \
     report. Report it now, here: `invoke` the table's `report_done` with exactly \
     what you found — or its `report_stuck` with what stopped you. Do not take up \
     another until it is reported.";

/// [`REPORT_AT_THE_TABLE`] for a **review**, whose report is a verdict.
///
/// **A review has nothing to find; it has a judgement to give.** Told to report
/// "exactly what you found", a reviewer that had read the draft and corrected a
/// title stood at the table saying there was "nothing substantive to report" —
/// it was looking for a discovery, and a review's answer is pass or reject.
pub const VERDICT_AT_THE_TABLE: &str =
    "You are at the command table and you have read the draft you were sent to \
     review. Give your verdict now, here. If it can stand — as you found it, or \
     with what you mended — `invoke` the table's `report_done` saying what you \
     checked and what, if anything, you changed; \"I checked its dates against \
     the era and its voice against the life, and corrected its title\" is a whole \
     report. If it cannot stand, `invoke` the table's `report_rejected` saying \
     why. Nothing more is needed from you than that judgement.";

/// The same instruction, for a character that is not alone.
///
/// **Company changes what there is to do, so it changes the instruction.** This
/// is not advice smuggled into a percept — it is the standing task, and having
/// somebody in front of you is a different standing task from having nobody.
///
/// It exists because of what the first one did on its own. Told to go somewhere
/// new every quiet turn, two characters explored a seventy-eight room building
/// beautifully and never once held a conversation: each moved every four
/// seconds, so being in a room together lasted exactly one tick and neither had
/// a reason to stay for the second.
/// **It names them.** `{who}` is filled with who is actually standing there.
///
/// The unnamed version — "Somebody else is here" — was the thing that made a
/// room full of people unusable. `tell` and `ask` both take a name and refuse
/// one they cannot find, and the *only* place a character is told a name is the
/// situation percept, which is deliberately suppressed while nothing moves
/// (see `npc_map::delta`). So a character standing still learns who its company
/// is exactly once, on arrival, and that line then ages out of the verbatim
/// window while the person is still in front of it.
///
/// What it has after that is this instruction, restated every quiet tick in the
/// most recent position in the window — telling it to talk to somebody it can
/// no longer name. It addressed `you` and was refused; it turned to face "the
/// person standing in the anteroom"; it spoke to the room and nobody was
/// obliged to answer. Every one of those is a character reaching for an
/// addressee it has been told it has and not been told the name of.
/// **Asks for something specific, because the generic version got generic
/// answers.** "Say what you have been looking at" produced characters agreeing
/// with each other about silence and gaps for a hundred turns — three of them
/// converging on one abstraction because none had named a thing. A conversation
/// about work needs a *piece* of work in it, and it has to be one the character
/// has: something it actually read or saw, which is why no example is given —
/// a given example is the thing every character then reports having read.
pub const IN_COMPANY: &str = "Nothing has been asked of you, and you are not alone. {who} \
                              here with you. Say something about the work, now — speak to a \
                              person here, `message` another character — and name a particular \
                              thing you have actually seen or read here, or ask them what they \
                              have. Ask them something they have to answer, or answer what they \
                              asked you. Address them exactly as written.";
