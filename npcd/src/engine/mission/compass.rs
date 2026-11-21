//! The mission compass: the next thing a character is told to do, and where the
//! people and machines it names stand.
//!
//! The compass gives the one next step and where whoever it names is standing, so
//! a character is sent to the room they are in rather than the one they were last
//! seen in. An act a step needs is named by its address when the body stands at the
//! desk that holds it, and by the way to that desk otherwise. A judgement, which
//! nothing in the world signs off, is said to be settled in the body's own mind.
//! A publish that has not landed points at the merge table, since the order table
//! refuses a report until the work has landed.

use super::*;

/// What a character is told when the engine signs a step off for it: `lead` (what
/// it has just done, "You are in the plant room"), that the step is done, and the
/// one thing to do next — the next step, or the report when that is all that is
/// left.
///
/// `here` is the machines standing where the body is. When the next step reads
/// one of them, the line says how in the grammar's own terms — `scan`, naming
/// no place — because "scan the breaker panel" read as naming the panel as the
/// place to look at, and a character standing beside it scanned another room.
pub fn progress_line(lead: &str, next: Option<&Todo>, here: &[String]) -> String {
    let here: Vec<String> = here.iter().map(|n| plain(n)).collect();
    match next {
        Some(step)
            if !step.reports
                && reading_of(&step.text).is_some_and(|of| here.iter().any(|m| names(&of, m))) =>
        {
            format!(
                "{lead}: that step of your mission is done. Next: {}. It is here, in this room: \
                 `scan` naming no place, and it lists it with the state it is in.",
                step.text.trim().trim_end_matches('.')
            )
        }
        Some(step) if !step.reports => format!(
            "{lead}: that step of your mission is done. Next: {}.",
            step.text.trim().trim_end_matches('.')
        ),
        _ => format!(
            "{lead}: that was the last step of your mission. Now go back to the table where \
             work is handed out, `scan` it, and `invoke` its `report_done` with exactly what \
             you found."
        ),
    }
}

/// The mission compass: the one next thing to do, and where whoever it names is
/// right now. `people` is everybody else in the world, each with the room they
/// are in, or `None` when they are in the body's own room.
///
/// **Where somebody is is the engine's to say.** A character sent to find
/// someone went to the room it had last seen them in, found it empty, and gave
/// the mission up as stuck — while the person stood in the next room, and then
/// walked into its own. It cannot track the cast; the world can.
///
/// Each person's `Some` is the way to them as a phrase — "the plant room:
/// `move_to` it", or the lift ride when they are on another level. `way` is the
/// way to the room or machine the step names, when it names one.
pub fn compass_line(next: &Todo, people: &[(String, Option<String>)], way: Option<&str>) -> String {
    let step = next.text.trim().trim_end_matches('.');
    let lower = step.to_lowercase();
    let mut line = format!("Your mission, next: {step}.");
    if let Some(way) = way {
        line.push(' ');
        line.push_str(way);
    }
    let speak = SPEAKING.iter().any(|v| lower.starts_with(v));
    for (who, room) in people.iter().filter(|(who, _)| mentions(&lower, who)) {
        match room {
            None if speak => line.push_str(&format!(
                " {who} is here with you: say it to them now — `message` them."
            )),
            None => line.push_str(&format!(" {who} is here with you.")),
            // Or by phone: somebody busy on their own errand keeps moving, and
            // a character chasing them room to room gave the mission up as
            // stuck. A `message` to them is a word to them like any other.
            Some(way) => line.push_str(&format!(
                " {who} is in {way} — or `message` them, which reaches them wherever they are."
            )),
        }
    }
    line
}

/// How an order-giver posts work, for the compass of a step that posts it: the
/// act at its address (`order_post`, `invoke`d at `at` when the table's address
/// is known), the standing jobs it may post by name in the catalogue's order,
/// and that a notice on a board is not work — the step's own word, "post",
/// otherwise reads as `post_notice`.
pub fn post_work_line(routines: &[Routine], at: Option<&str>) -> String {
    let act = match at {
        Some(url) => format!("`invoke` `{url}`"),
        None => "`order_post` at the order table".to_string(),
    };
    let what = match routines.is_empty() {
        true => ", with a `brief` and its `steps`".to_string(),
        false => {
            let ids: Vec<String> = routines.iter().map(|r| format!("`{}`", r.id)).collect();
            format!(
                " — a standing `job` by name when one fits what the families need: {}; or a \
                 `brief` and its `steps` for what none of them covers",
                ids.join(", ")
            )
        }
    };
    format!(
        "Work is posted with {act}{what}. A notice on a board is not work: nobody takes it up. \
         {TABLE_LIMIT}"
    )
}

/// The addresses a document step is done through, at the desk the body stands
/// at — each `None` when no desk here offers it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeskVerbs {
    pub read: Option<String>,
    /// Where the piece is written whole — `compose`.
    pub compose: Option<String>,
    pub edit: Option<String>,
    pub commit: Option<String>,
}

/// The mission compass for a step that sets the year: the act at a time machine
/// standing here, else the way to one (`way`).
///
/// **The year is set at a machine, and the compass says which act.** Told only
/// "set your time to 2837 at a time machine", four Makers walked between the
/// lift and their rooms for an hour, told each other the lift was broken and
/// went looking for a level nothing had sent them to.
pub fn time_compass(year: u32, machine_here: bool, way: Option<&str>) -> String {
    match (machine_here, way) {
        (true, _) => format!(
            "Your mission, next: set your time to {year}. A time machine is here: `time_travel` \
             naming the year {year}."
        ),
        (false, Some(way)) => format!(
            "Your mission, next: set your time to {year}. The time machines are in {way}; there, \
             `time_travel` naming the year {year}."
        ),
        (false, None) => format!(
            "Your mission, next: set your time to {year}, at a time machine on the time level: \
             `move_to` a room on it that has one, and the lift takes you there."
        ),
    }
}

/// The mission compass for a step about a document: the exact act that does it
/// when the body stands at a desk, else the way to one (`way`).
///
/// **The act, not the intention.** Told only "next: read layers/eras/…", a
/// Keeper who had reached the writing room reflected and dreamt for ten minutes
/// and never touched the desk: reading a document is an `invoke` of the desk's
/// `file_read` naming it, and nothing in what it read said so.
///
/// `written` is how many words of the document to write the working copy holds,
/// uncommitted, when that is already up to its floor.
///
/// **A written piece is committed, not written again.** Told "sit down and
/// write it whole" straight after it had composed five hundred words, a Maker
/// composed the same story five times over and never committed it.
pub fn doc_compass(
    doc: &DocStep,
    at: &DeskVerbs,
    way: Option<&str>,
    written: Option<usize>,
) -> String {
    match (doc, at, written) {
        (
            DocStep::Write(p),
            DeskVerbs {
                read: Some(read),
                commit: Some(commit),
                ..
            },
            Some(words),
        ) => format!(
            "Your mission, next: commit {p}. It is written — {words} words in your working copy. \
             Read it back with `invoke` {read} with `path` \"{p}\"; if it stands, `invoke` {commit} \
             with `why`, one line saying what it is. Put one passage right first if one is wrong, \
             but do not write it again."
        ),
        (doc, at, _) => doc_compass_unwritten(doc, at, way),
    }
}

/// The compass for a document whose text the brief gives whole — a mission
/// template: written as given with `file_write`, then committed. `write` and
/// `commit` are the desk's addresses when the body stands at one, `way` the
/// way to one when it does not, and `refused` what the table said when it last
/// would not take the document — led with, so the next write puts that right.
pub fn given_compass(
    p: &str,
    write: Option<&str>,
    commit: Option<&str>,
    way: Option<&str>,
    refused: Option<&str>,
) -> String {
    let write_it = given_write(p, write, commit, way);
    match refused {
        None => write_it,
        Some(refusal) => format!(
            "When you last reported {p}, the table would not take it: {}\nPut that right in \
             the text and write it again. {write_it}",
            bounded_refusal(refusal)
        ),
    }
}

/// [`given_compass`]'s own line: the acts that write and commit it.
pub(super) fn given_write(
    p: &str,
    write: Option<&str>,
    commit: Option<&str>,
    way: Option<&str>,
) -> String {
    match (write, commit) {
        (Some(write), Some(commit)) => format!(
            "Your mission, next: write {p} exactly as your brief gives it, whole: `invoke` \
             {write} with `path` \"{p}\" and that text; then `invoke` {commit} with `why`, one \
             line saying what it is."
        ),
        _ => {
            let at = way.map_or_else(String::new, |w| format!(" The nearest desk is in {w}."));
            format!(
                "Your mission, next: write {p} exactly as your brief gives it, whole, with \
                 `file_write` at a desk, then commit it with `bench_commit`.{at}"
            )
        }
    }
}

/// The compass for a document the table refused at its last report:
/// `refusal` is what it said, and `at` and `way` the desk as
/// [`doc_compass`] takes them.
///
/// **Both ways out, and neither forbidden.** Told by the table "mend it with
/// `file_edit`, or write it again whole with `compose`", and by the compass on
/// every turn after "commit it … but do not write it again", three Makers whose
/// edits had not cleared the brief's words out of their drafts had no way left
/// they were allowed to take, and their operations failed stuck. The refusal
/// stays in front of them until a report is taken, and so do both ways.
///
/// **What it carries of the refusal is bounded** ([`COMPASS_REFUSAL_CHARS`]).
/// It is in front of the Maker on every turn until a report is taken: one
/// refusal of 412,000 characters, carried so, grew a turn to ~190k tokens.
pub fn refused_compass(p: &str, refusal: &str, at: &DeskVerbs, way: Option<&str>) -> String {
    let desk = doc_compass_unwritten(&DocStep::Write(p.to_string()), at, way);
    format!(
        "When you last reported {p}, the table would not take it: {}\nIf you have put that \
         right since, commit it and report again. If not, mend those passages with \
         `file_edit` — or write it whole again: {desk}",
        bounded_refusal(refusal)
    )
}

/// A refusal cut to [`COMPASS_REFUSAL_CHARS`], saying so when it is cut.
pub(super) fn bounded_refusal(refusal: &str) -> String {
    match refusal.char_indices().nth(COMPASS_REFUSAL_CHARS) {
        None => refusal.to_string(),
        Some((cut, _)) => format!(
            "{} … (the table said more; report it again to hear all of it)",
            &refusal[..cut]
        ),
    }
}

/// The most of a refusal the compass carries, in characters: a gate's refusal
/// names five faults of a kind and counts the rest, well inside it.
pub const COMPASS_REFUSAL_CHARS: usize = 4_000;

pub(super) fn doc_compass_unwritten(doc: &DocStep, at: &DeskVerbs, way: Option<&str>) -> String {
    match (doc, at) {
        (
            DocStep::Read(p),
            DeskVerbs {
                read: Some(url), ..
            },
        ) => format!(
            "Your mission, next: read {p}. `invoke` {url} with `path` \"{p}\" — it comes back a \
             page at a time."
        ),
        // **Sit down to it.** The piece is written whole with `compose`, in one
        // sitting with the brief and the sources open — see `engine::compose`;
        // a draft put together line by line between other acts carried the room
        // it was written in. An edit is for a passage the checks name.
        (
            DocStep::Write(p),
            DeskVerbs {
                compose: Some(compose),
                edit,
                commit: Some(commit),
                ..
            },
        ) => {
            let mend = match edit {
                Some(edit) => format!(
                    " To put one passage right afterwards, `invoke` {edit} with the paragraph's \
                     anchor from your read as `at`, `do` replace, the words to change as `old` \
                     and what they become as `text`."
                ),
                None => String::new(),
            };
            format!(
                "Your mission, next: write {p}. Sit down and write it whole: `invoke` {compose} — \
                 you write it through with your brief and what you read open in front of you, \
                 and it goes into your working copy.{mend} Then `invoke` {commit} with `why`, one \
                 line saying what it is. Until it is committed nobody else can see it, and the \
                 mission is not done."
            )
        }
        (doc, _) => {
            let (verb, p, desk) = match doc {
                DocStep::Read(p) => ("read", p, "Documents are read at a desk"),
                DocStep::Write(p) => ("write", p, "A piece is written at a writing desk"),
            };
            match way {
                Some(way) => {
                    format!("Your mission, next: {verb} {p}. {desk}: the nearest is in {way}.")
                }
                None => format!(
                    "Your mission, next: {verb} {p}. {desk}; at one, `scan` shows its address."
                ),
            }
        }
    }
}

/// The mission compass when the work is done and only the report is left.
/// `table` is the way to the table — its room, and the lift ride when it is on
/// another level; `at_table` whether the body is at it;
/// `seen` what the engine observed toward the mission, which is what the report
/// is to say.
///
/// **The report is about this mission.** Characters filed accounts of the
/// mission before last, or of the scan they made at the table; handed what it
/// saw on this one, in the line that tells it to report, it reports that.
///
/// **A review's report is its verdict** (`verdict`): told to report "exactly
/// what you found", a reviewer that had read the draft and corrected its title
/// stood at the table saying there was nothing substantive to report.
pub fn compass_report_line(
    table: Option<&str>,
    at_table: bool,
    seen: &[String],
    verdict: bool,
) -> String {
    let mut line = report_way(table, at_table, verdict);
    if !seen.is_empty() {
        line.push_str(&format!(" What you saw on it: {}.", seen.join("; ")));
    }
    line
}

/// The mission compass on a publish step whose work has not landed yet:
/// `doc` is what is to land, `status` and `publish` the merge table's own
/// addresses when the body stands at one, and `way` the way to the nearest
/// when it does not.
///
/// **The landing, not the report.** Told on every turn that the work was done
/// and to report it, a Maker carrying a publish stood at the order table
/// calling `report_done` and `report_stuck` for half an hour, each refused
/// with "it has not landed", while the merge table stood in the same room.
pub fn publish_compass(
    doc: &str,
    status: Option<&str>,
    publish: Option<&str>,
    way: Option<&str>,
) -> String {
    match (status, publish, way) {
        (Some(status), Some(publish), _) => format!(
            "Your mission, next: land {doc} in the record. The merge table is here: `invoke` \
             {status} to see what it changes and whether it merges cleanly, then `invoke` \
             {publish} to land it. Report at the table once it has landed."
        ),
        (_, _, Some(way)) => format!(
            "Your mission, next: land {doc} in the record, at the merge table in {way}. There, \
             `merge_status` shows whether it merges cleanly and `merge_publish` lands it. \
             Report at the table once it has landed."
        ),
        _ => format!(
            "Your mission, next: land {doc} in the record, at the merge table: `merge_status` \
             shows whether it merges cleanly and `merge_publish` lands it. Report at the table \
             once it has landed."
        ),
    }
}

pub(super) fn report_way(table: Option<&str>, at_table: bool, verdict: bool) -> String {
    let what = match verdict {
        true => {
            "its `report_done` if the draft can stand, saying what you checked and what you \
                 changed — or its `report_rejected` if it cannot, saying why. That verdict is the \
                 whole report"
        }
        false => "its `report_done` with exactly what you found",
    };
    match (at_table, table) {
        (true, _) => format!(
            "Your mission's work is done and you are at the table: report it now — `invoke` \
             {what}."
        ),
        (false, Some(way)) => format!(
            "Your mission's work is done. Go back to the table, in {way}, and report it: \
             `invoke` {what}."
        ),
        (false, None) => format!(
            "Your mission's work is done. Go back to the table where work is handed out and \
             report it: `invoke` {what}."
        ),
    }
}

/// How a step that is about saying something to somebody begins.
/// How a step the body settles in its own mind begins — nothing in the world
/// signs it off, and it is carried to the table in the report.
pub(super) const JUDGING: &[&str] = &[
    "settle ",
    "decide ",
    "note ",
    "judge ",
    "work out ",
    "think ",
    "conclude ",
];

/// Whether a step is one the body settles in its own mind ([`JUDGING`]).
pub fn is_judgement(step: &str) -> bool {
    let lower = step.trim().to_lowercase();
    JUDGING.iter().any(|v| lower.starts_with(v))
}

/// What the compass adds to a step the body settles in its own mind.
///
/// **Nothing in the world signs it off.** Sent to "settle how the Sharratt
/// family is today", a Drone walked between their room and the lift looking
/// for the act that would, and said so to itself.
pub const JUDGEMENT_LINE: &str = "Nothing in the world marks that done: settle it from what you \
     have seen, and say it in your report — `report_done` at the table carries it.";

/// [`JUDGEMENT_LINE`] for a judgement with a step the world can see still
/// after it ([`Mission::step_to_do`]): settled, the body goes on to that one.
pub fn judgement_then(later: &str, way: Option<&str>) -> String {
    let later = later.trim().trim_end_matches('.');
    let at = way.map_or_else(String::new, |w| format!(" It is in {w}."));
    format!(
        "Nothing in the world marks that done: settle it from what you have seen, and say it in \
         your report. Then go on to what is still to do: {later}.{at}"
    )
}
