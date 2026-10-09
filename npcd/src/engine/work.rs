//! Performing the station and bench acts.
//!
//! [`super::enact`] performs what a body does with things it carries and ground
//! it stands on. This performs what it does through a *station* — the record's
//! own acts, and the working loop over them.
//!
//! # Nearly all of it is one store's state machine
//!
//! Writing a chronicle entry, drafting into a silence, drawing a face and
//! writing a place are the same handful of moves over [`crate::sim::record`]:
//! take a thing, write into it, offer it, file it. The differences that look
//! large in the fiction — an era against a portrait — are differences in *which
//! station can reach it*, and that is the map's business rather than this
//! file's.
//!
//! So the dispatch below is short and the interesting rules live in the store,
//! where one bug is one bug rather than six.
//!
//! # A refusal is second person and says what would have worked
//!
//! The grammar already makes most bad calls unreachable — `read`'s argument is
//! bound to what the station actually holds. What is left refuses for reasons
//! the grammar cannot see: somebody else is holding it, it is already filed, a
//! thing let go without a reason cannot be argued with later. Each of those is
//! written for a character to read and act on.

use serde_json::{Map, Value};

use crate::engine::act::Act;
use crate::engine::body::Outcome;
use crate::engine::chronology;
use crate::engine::mission::{
    progress_line, time_step, Aim, DocStep, Mission, Outcome as Verdict, Stage,
};
use crate::engine::mission_gen::{gates, leakage, rejection};
use crate::engine::passage;
use crate::sim::record::{slug_of, Condition, Item, Kind, State};
use crate::sim::Sim;
use crate::world::Hosted;
use npc_map::route;
use npc_map::schema::Where;

/// The acts this module performs.
pub fn is_mine(tool: &str) -> bool {
    crate::engine::station::STATION_ACTS
        .iter()
        .chain(crate::engine::bench::BENCH_ACTS)
        .chain(crate::engine::mission_acts::MISSION_ACTS)
        .any(|t| t.name == tool)
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// An argument taken exactly as it was written.
///
/// **The contents of a document are not trimmed.** [`text`] trims, which is
/// right for a name or an intent and wrong here twice over: a trailing newline
/// is part of the file, and the leading whitespace of a line is part of what an
/// edit has to match. Trimming either writes something other than what was
/// asked for, and does it silently — the act reports success and the document
/// is subtly not what the character said.
fn verbatim(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?;
    (!s.trim().is_empty()).then(|| s.to_string())
}

/// The one required argument naming what an act is about, whatever it is called.
///
/// Station acts name their subject differently because the sentence differs —
/// `of` a portrait, `to` an era, `for` a gap, `from` a change. Which word it is
/// carries meaning for the model and none at all for the store, so the store is
/// asked once here rather than in forty match arms.
fn subject(args: &Map<String, Value>) -> Option<String> {
    // `why` and `called` are here because two acts carry no subject of their
    // own — a commit acts on whatever you have open, and a new place is named
    // rather than found — and reaching the dispatch at all is what lets them
    // say so themselves rather than being turned away as subjectless.
    for key in [
        "what", "of", "in", "to", "for", "from", "on", "about", "between", "under", "path", "why",
        "called",
    ] {
        if let Some(v) = text(args, key) {
            return Some(v);
        }
    }
    None
}

/// Perform a mission act.
///
/// Take one up at the desk, record progress on it wherever the work happened, or
/// report how it went. The rich model is [`crate::engine::mission`]; whose it is
/// is [`crate::sim::missions`]; this moves a mission between those states and
/// hands the character back a line it reads.
fn mission(hosted: &Hosted, body: &str, act: &Act) -> Outcome {
    let a = &act.args;
    match act.tool {
        "collect_mission" => {
            // **One mission at a time.** `collect_mission` is offered whenever a
            // body stands at the desk, including when it has come back to report
            // — so without this a character could draw a fresh mission over an
            // open one, discarding the answer it built and never filing it to
            // `done`. Report it first; then the desk has something new to give.
            if let Some((carried, next)) = hosted.sim(|s| {
                s.missions.active(body).map(|m| {
                    (
                        m.standing_text(),
                        m.next_step().filter(|t| !t.reports).map(|t| t.text.clone()),
                    )
                })
            }) {
                // **Say the one thing to do, not the whole brief again.** A
                // character that came back to the table mid-mission and was handed
                // its brief whole went back out to the first step it had already
                // done; one with everything done needs telling that the report is
                // what is left, here, now.
                let what_now = match next {
                    Some(step) => format!(
                        "It is not finished: the next thing to do on it is to {}. Go and do \
                         that, then come back and `invoke` this table's `report_done` with what \
                         you found — or `report_stuck` if it cannot be done.",
                        step.trim().trim_end_matches('.')
                    ),
                    None => "Every step of it is done, and you are at the table: report it now — \
                             `invoke` this table's `report_done` with exactly what you found."
                        .to_string(),
                };
                return Outcome::Refused(format!(
                    "You are already carrying a mission, and the table gives out one at a time. \
                     {what_now}\n{carried}"
                ));
            }
            Outcome::Did(format!("You take it up.\n{}", take_up(hosted, body)))
        }
        "report_done" => {
            let Some(account) = text(a, "account") else {
                return Outcome::Refused(
                    "You meant to report it done, but did not say what you found.".into(),
                );
            };
            // **An account says what was found.** A character filed its own
            // name as the account of a reading, six times over; a few words are
            // not an answer anybody can use. Turned back with what the engine
            // saw on the mission, so the next try has the facts in front of it.
            if account.split_whitespace().count() < MIN_ACCOUNT_WORDS {
                let seen = hosted
                    .sim(|s| s.missions.active(body).map(|m| m.observed.clone()))
                    .unwrap_or_default();
                let hint = match seen.is_empty() {
                    true => String::new(),
                    false => format!(" What you saw on it: {}.", seen.join("; ")),
                };
                return Outcome::Refused(format!(
                    "\"{account}\" is not an account of what you found. Say it in a sentence — \
                     what you found, made or concluded.{hint}"
                ));
            }
            // **A step the engine can see, not yet done and still doable, holds
            // the report.** A character sent to two rooms messaged the channel
            // and reported the mission done from the table; the journeys were
            // never made. The way to the step is the refusal.
            if let Some(way) = still_doable(hosted, body) {
                let step = hosted
                    .sim(|s| s.missions.active(body)?.next_step().map(|t| t.text.clone()))
                    .unwrap_or_default();
                return Outcome::Refused(format!(
                    "You have not done this yet: {step}. {way} Then come back and report it. If \
                     it cannot be done, `invoke` the table's `report_stuck` and say why."
                ));
            }
            // **Work on the record is done when the record says so.** A mission
            // that writes a document is reported done once that document is
            // committed — by this character, while it carried the mission (see
            // [`Mission::written_up`]). Its own word that the writing is
            // finished is the label `docs/asynchronous_mind_hierarchy.md` §7.3
            // says must never stand uncorroborated, and a step ticked on its
            // word is that label: a story was reported done with its write
            // step struck "thwarted" and no document anywhere.
            if let Some(path) = hosted.sim(|s| {
                let m = s.missions.active(body)?;
                (!m.written_up()).then(|| m.work.as_ref().map(|w| w.writes.clone()))?
            }) {
                return Outcome::Refused(format!(
                    "You have not written it yet: {path} is not committed. Go to a desk where \
                     documents are written, `compose` it there — you write it whole with your \
                     brief and what you read in front of you — then `bench_commit` it with one \
                     line saying what it is, and come back and report it. If it cannot be \
                     written, `invoke` this table's `report_stuck` and say why."
                ));
            }
            // **A reading nobody made cannot be reported.** A mission to read a
            // machine was filed as a pass by a character whose own account said
            // it never read it. Reading is a `scan` in the machine's room, which
            // the engine sees; until it has happened, done is not the report.
            if let Some(step) = hosted.sim(|s| {
                let machines: Vec<String> = s.devices.iter().map(|d| d.name.clone()).collect();
                s.missions
                    .active(body)?
                    .unread_step(&machines)
                    .map(str::to_string)
            }) {
                return Outcome::Refused(format!(
                    "You have not done this yet: {step}. Go to the room it is in and `scan` there \
                     — what you scan shows each machine with the state it is in. Then come back \
                     and report it. If you cannot get to it or read it, `invoke` the table's \
                     `report_stuck` and say why."
                ));
            }
            // **The quality gate, while it can still be mended.** An
            // operation's document is held to the engine's checks before its
            // draft or its review may be reported done — see
            // `mission_gen::gates`.
            if let Some(refusal) = report_block(hosted, body) {
                return Outcome::Refused(refusal);
            }
            hosted.with_sim(|s| {
                // The account is both the report's notes (how it went) and the
                // answer (what was found) — for a mission done, the two are the
                // same line, so it is filed under both.
                match s
                    .missions
                    .report(body, Verdict::Pass, &account, Some(account.clone()))
                {
                    Some(m) => {
                        tracing::info!(npc = body, prompt = %m.mission_text(), %account, "mission reported DONE at the command table");
                        Outcome::Did(format!(
                            "Reported done, and your answer filed: {}{}",
                            m.mission_text(),
                            back_to_now(&m)
                        ))
                    }
                    None => Outcome::Refused("You are not carrying a mission to report on.".into()),
                }
            })
        }
        "report_stuck" => {
            let Some(why) = text(a, "why") else {
                return Outcome::Refused(
                    "You meant to report it stuck, but did not say why.".into(),
                );
            };
            // **Stuck is for a step that cannot be done, and the world can see
            // when one can.** Characters gave missions up as stuck a minute in —
            // the person "nowhere to be found" one room away, the machine "not
            // responding" in a room they never reached. When the next step's
            // room, machine or person is within reach, the desk says where and
            // how instead. Twice, then it takes the report: a character that has
            // been told the way twice and still cannot is reporting a real block.
            if let Some(way) = still_doable(hosted, body) {
                let refused = hosted.with_sim(|s| s.missions.refuse_stuck(body));
                if refused <= STUCK_REFUSALS {
                    return Outcome::Refused(format!(
                        "It is not stuck yet — {way} Do that, then come back to the table and \
                         `invoke` its `report_done` with what you found."
                    ));
                }
            }
            // **Nothing is stuck when nothing is left.** A Maker whose story was
            // written, committed and every step signed off reported it stuck —
            // "the mission is done, but I cannot report it" — and the stuck
            // report failed the operation and threw the finished draft away.
            // With every step but the report done and its document committed,
            // the world says the work stands; the report that matches is done.
            if let Some(done) = hosted.sim(|s| finished(s, body)) {
                return Outcome::Refused(done);
            }
            hosted.with_sim(
                |s| match s.missions.report(body, Verdict::Fail, &why, None) {
                    Some(m) => {
                        tracing::info!(npc = body, prompt = %m.mission_text(), %why, "mission reported STUCK at the command table");
                        Outcome::Did(format!(
                            "Reported as not done, with your reasons: {}{}",
                            m.mission_text(),
                            back_to_now(&m)
                        ))
                    }
                    None => Outcome::Refused("You are not carrying a mission to report on.".into()),
                },
            )
        }
        "report_rejected" => {
            let Some(why) = text(a, "why") else {
                return Outcome::Refused(
                    "You meant to reject the draft, but did not say why it cannot stand.".into(),
                );
            };
            if why.split_whitespace().count() < MIN_ACCOUNT_WORDS {
                return Outcome::Refused(format!(
                    "\"{why}\" does not say why the draft cannot stand. Say what is wrong with it \
                     that cannot be mended in place."
                ));
            }
            // A review must have read what it rejects.
            if let Some(way) = still_doable(hosted, body) {
                let step = hosted
                    .sim(|s| s.missions.active(body)?.next_step().map(|t| t.text.clone()))
                    .unwrap_or_default();
                return Outcome::Refused(format!(
                    "You have not done this yet: {step}. {way} Read it, and mend what the table \
                     found, before you judge it — reject only what your mending could not save."
                ));
            }
            let unsupported = hosted.sim(|s| {
                let m = s.missions.active(body)?;
                let (_, stage) = m.operation()?;
                rejection::unsupported(s.bench.mind_root()?, m.work.as_ref()?, stage, &why)
            });
            if let Some(refusal) = unsupported {
                tracing::info!(npc = body, %why, "operation rejection refused: no evidence");
                return Outcome::Refused(refusal);
            }
            hosted.with_sim(|s| {
                let Some(m) = s.missions.reject(body, &why) else {
                    return Outcome::Refused(
                        "You are not reviewing anybody's draft; there is nothing to reject.".into(),
                    );
                };
                // **A rejected draft leaves the record.** It is moved aside, not
                // deleted, so an operator can still read what was refused; a
                // rejected correction's document goes back to what it said —
                // see `Sim::set_aside_failed`. Settled here at once, so the
                // reviewer's world shows it; the generator's loop settles every
                // other failure the same way, and doing so again is harmless.
                let op = m
                    .operation()
                    .and_then(|(id, _)| s.missions.operations().get(id))
                    .cloned();
                let moved = op.as_ref().and_then(|o| s.set_aside_failed(o));
                let gone = match op.as_ref().is_some_and(|o| o.leaves_on_failure()) {
                    true => "the draft leaves the record",
                    false => "the document goes back to what it said before",
                };
                tracing::info!(npc = body, prompt = %m.mission_text(), %why, moved = ?moved, "operation REJECTED on review");
                Outcome::Did(format!(
                    "Rejected, with your reasons; {gone}. {}{}",
                    m.mission_text(),
                    back_to_now(&m)
                ))
            })
        }
        other => Outcome::Refused(format!("`{other}` is not a mission act.")),
    }
}

/// Take up the next mission for `body` at the table — lodged for it, waiting
/// in the pool, or a routine from the bank — and answer with its brief as the
/// character reads it.
pub fn take_up(hosted: &Hosted, body: &str) -> String {
    // The character's own name, so a routine that would send it to visit
    // "the makers here" is not built around visiting itself.
    let me = hosted
        .read(|w| w.actor(body).map(|actor| actor.name.clone()))
        .unwrap_or_default();
    hosted.with_both(|w, s| {
        let room_name = |place: &str| {
            let (area, node) = place.split_once('/')?;
            w.node(&Where::new(area, node)).map(|n| n.name.clone())
        };
        let reach: Vec<String> = w
            .actor(body)
            .map(|actor| {
                std::iter::once(actor.at.clone())
                    .chain(route::reachable_from(w.map(), &actor.at))
                    .map(|at| format!("{}/{}", at.area, at.node))
                    .collect()
            })
            .unwrap_or_default();
        let material = s.mission_material(&me, &room_name, &reach);
        let mission = s.missions.collect(body, &material.facts());
        tracing::info!(npc = body, prompt = %mission.mission_text(), "mission taken up at the command table");
        mission.standing_text()
    })
}

/// What stands between the body and reporting its operation's document done:
/// the gate's faults, once the engine has tidied what needs no judgement (see
/// [`gates::tidy`]). `None` when nothing does.
pub fn report_block(hosted: &Hosted, body: &str) -> Option<String> {
    let tidy = hosted.sim(|s| {
        let m = s.missions.active(body)?;
        m.operation()?;
        Some((
            s.bench.mind_root()?.to_path_buf(),
            m.work.as_ref()?.writes.clone(),
        ))
    });
    if let Some((root, path)) = &tidy {
        if gates::tidy_on_disk(root, path) {
            tracing::info!(npc = body, %path, "operation document tidied");
        }
    }
    let vocabulary = hosted.read(leakage::vocabulary);
    let refusal = hosted.sim(|s| gate_refusal(s, body, &vocabulary))?;
    // A document that does not stand is to be written again — said in the
    // mission's own steps, so the compass sends the Maker back to its desk
    // instead of on to a report that will be refused. One gone from the record
    // has nothing to write again.
    if tidy.is_some_and(|(root, path)| root.join(path).is_file()) {
        hosted.with_sim(|s| s.missions.reopen_write(body));
    }
    Some(refusal)
}

/// Why the body's operation document may not be reported done yet: the
/// engine's quality gate finds faults in it as it stands on the record — the
/// page's own checks, and the writers' room in it (see `leakage`, whose
/// `vocabulary` is the world the Makers stand in). `None` when it passes, or
/// when the mission is not an operation's.
fn gate_refusal(s: &Sim, body: &str, vocabulary: &[String]) -> Option<String> {
    let m = s.missions.active(body)?;
    let (_, stage) = m.operation()?;
    let path = &m.work.as_ref()?.writes;
    let root = s.bench.mind_root()?;
    let text = match std::fs::read_to_string(root.join(path)) {
        Ok(t) => t,
        Err(_) if stage == Stage::Draft => return None,
        Err(_) => {
            return Some(format!(
                "{path} is not on the record any more, so there is nothing to pass. `invoke` this \
                 table's `report_rejected` and say so."
            ))
        }
    };
    let mut faults = gates::check(path, &text, gates::life_voice(root, path));
    if gates::Form::of(path) != gates::Form::Other {
        let leaked = leakage::leaks(&text, vocabulary, &leakage::lore(root));
        if !leaked.is_empty() {
            faults.push(leakage::fault(&leaked));
        }
    }
    if faults.is_empty() {
        return None;
    }
    let mend = match stage {
        Stage::Draft => {
            "Mend it with `file_edit` (or write it again whole with `compose`) and \
                         `bench_commit`, then come back and report it."
        }
        Stage::Review | Stage::Canon => {
            "Mend it yourself with `file_edit` and `bench_commit` before you pass it — \
                          or, if it cannot be mended, `invoke` this table's `report_rejected` and \
                          say why."
        }
    };
    Some(format!(
        "{path} is not ready to stand in the record:\n- {}\n{mend}",
        faults.join("\n- ")
    ))
}

/// Why the body may not write `path`: it carries a mission whose work is a
/// different document. `None` when it may.
///
/// **A mission's Maker writes the mission's document and nothing else.** A
/// Keeper on a mission to write a year of a Zenling's life rewrote the closing
/// line of the era it had been sent to read — a flourish of its own in canon, a
/// link lost — and another left a second copy of its event under a name it
/// made up, which the life then read as a second event. What a mission is sent
/// to read it reads; what it writes is the one document its brief names.
///
/// **Exactly the path, capitals and all.** The disk does not tell `X.Md` from
/// `X.md`, but the mind does: a life document written as `….Md` is never read
/// as an episode, and two were, before the comparison stopped ignoring case.
fn outside_the_mission(s: &Sim, body: &str, path: &str) -> Option<String> {
    let work = s.missions.active(body)?.work.as_ref()?;
    let norm = |p: &str| p.trim().trim_start_matches('/').to_string();
    (norm(path) != norm(&work.writes)).then(|| {
        format!(
            "Your mission writes {} and nothing else. {path} stands as it is — read it, but leave \
             it. If something in it is wrong, say so in your report.",
            work.writes
        )
    })
}

/// Why an edit to the body's mission piece is refused: it takes most of the
/// piece, which is writing it again by hand — see [`passage`]. `None` for any
/// other document, a mission with no prose floor, or a passage.
fn rewritten_by_hand(s: &Sim, body: &str, what: &str, old: &str) -> Option<String> {
    let work = s.missions.active(body)?.work.as_ref()?;
    if work.min_words == 0 || !what.eq_ignore_ascii_case(&work.writes) {
        return None;
    }
    let current = s.bench.read(body, &work.writes).ok()?;
    passage::rewrites_the_piece(&work.writes, &current, old)
}

/// Why the body's open mission document is not ready to commit — it is in the
/// working set and shorter than the mission's floor — or `None` when it is, or
/// when the commit does not touch it.
fn short_of_the_floor(s: &Sim, body: &str) -> Option<String> {
    let mission = s.missions.active(body)?;
    let work = mission.work.as_ref()?;
    if work.min_words == 0 {
        return None;
    }
    let touched = s
        .bench
        .opened(body)?
        .changed()
        .iter()
        .any(|p| p.eq_ignore_ascii_case(&work.writes));
    if !touched {
        return None;
    }
    let n = s
        .bench
        .read(body, &work.writes)
        .ok()?
        .split_whitespace()
        .count();
    (n < work.min_words).then(|| {
        short_by(
            &work.writes,
            n,
            work.min_words,
            what_happens(&mission.prompt),
        )
    })
}

/// The label a brief gives the substance of the piece it asks for.
const WHAT_HAPPENS: &str = "What happens:";

/// The most of a brief's "What happens" a refusal quotes, in characters.
const QUOTED_HAPPENS: usize = 900;

/// What the brief says happens in the piece, from its [`WHAT_HAPPENS`]
/// paragraph — `None` for a brief that has none.
pub(crate) fn what_happens(brief: &str) -> Option<String> {
    let at = brief.find(WHAT_HAPPENS)? + WHAT_HAPPENS.len();
    let paragraph = brief[at..].split("\n\n").next()?.trim();
    let quoted: String = paragraph.chars().take(QUOTED_HAPPENS).collect();
    (!quoted.is_empty()).then_some(quoted)
}

/// The refusal for a piece under its floor: how far short it is, that the way
/// on is to add to it, and what the piece is to hold.
///
/// "This is a sketch of it" alone was read as a verdict on the writing: Makers
/// resubmitted the same text word for word, or edited it shorter, and called
/// the refusal a complaint that nothing was new. The gap is a number, and it is
/// given as one.
///
/// **And the scene to add from.** Told it was a hundred and twenty-three words
/// short, a Maker wrote the same hundred and twenty-seven words again and again
/// and said there was nothing more to say — its draft was about the room it
/// stood in and the colleagues passing through, not the scene its brief asked
/// for. The brief's own account of what happens is put back in front of it.
fn short_by(writes: &str, n: usize, min: usize, happens: Option<String>) -> String {
    let mut s = format!(
        "{writes} is {n} of the {min} words asked for — about {} more. It is not finished: keep \
         what is there and add to it — more of the scene, what is done and said in it — with \
         `file_edit`, or write it whole again with `compose`, then commit.",
        min - n
    );
    if let Some(happens) = happens {
        s.push_str(&format!(
            " What happens in it, as your brief puts it: {happens} Write that, moment by moment."
        ));
    }
    s
}

/// Why a stuck report contradicts what the engine sees — every step of the
/// body's mission but the report is done, and any document it writes is
/// committed — or `None` when something is genuinely left.
fn finished(s: &Sim, body: &str) -> Option<String> {
    let m = s.missions.active(body)?;
    let left = m.next_step().is_some_and(|t| !t.reports);
    if left || !m.written_up() {
        return None;
    }
    let made = match &m.work {
        Some(w) if !w.edit_optional => format!(" {} is written and committed.", w.writes),
        _ => String::new(),
    };
    let verdict = match m.operation() {
        Some((_, Stage::Draft)) | None => "`invoke` the table's `report_done` with what you made \
                                            or found"
            .to_string(),
        Some(_) => "give your verdict at the table: `report_done` if it stands, \
                    `report_rejected` with why if it cannot be mended"
            .to_string(),
    };
    Some(format!(
        "Nothing here is stuck: every step of your mission is done.{made} What is left is the \
         report — {verdict}."
    ))
}

/// The fewest words `report_done` takes as an account of what was found.
const MIN_ACCOUNT_WORDS: usize = 4;

/// How many times `report_stuck` is turned away while the next step is within
/// reach before the report is taken anyway.
const STUCK_REFUSALS: u32 = 2;

/// How the next step of the body's mission can be done from here, when the
/// engine can see that it can: its room is within reach, its machine stands in
/// a room within reach, or its person is in one. `None` when there is no such
/// step, or what it names is nowhere the body can get to.
fn still_doable(hosted: &Hosted, body: &str) -> Option<String> {
    hosted.with_both(|w, s| {
        let step = s
            .missions
            .active(body)?
            .next_step()
            .filter(|t| !t.reports)?
            .text
            .clone();
        if let Some(year) = time_step(&step) {
            let here = w.actor(body)?.at.clone();
            let at_machine = w
                .map()
                .parts_at(w.node(&here)?)
                .any(|(part, _)| part.id == "time-machine");
            return Some(match at_machine {
                true => format!("A time machine is here: `time_travel` naming the year {year}."),
                false => format!(
                    "You are not working in {year} yet. The time machines are on the time level: \
                     `move_to` the lift, `lift_use` naming the time level, `move_to` the first \
                     time room, then `time_travel` naming the year {year}."
                ),
            });
        }
        let here = w.actor(body)?.at.clone();
        let reach: Vec<Where> = std::iter::once(here.clone())
            .chain(route::reachable_from(w.map(), &here))
            .collect();
        let room_of = |at: &Where| w.node(at).map(|n| n.name.clone());
        let level_of = |at: &Where| w.map().get(&at.area).map(|a| a.name.clone());
        let place = |at: &Where| format!("{}/{}", at.area, at.node);
        // **A document is read and written at a desk, and the desks can be
        // reached.** Makers gave drafts up as stuck "between the lift and the
        // command room", the chronicle level "not accessible" — one from the
        // landing with the car on its way — and each report failed its
        // operation and threw the work away.
        if DocStep::of(&step).is_some() {
            let desks: Vec<&Where> = reach
                .iter()
                .filter(|at| s.part_offers(&place(at), "file_read"))
                .collect();
            let desk = desks
                .iter()
                .find(|at| at.area == here.area)
                .or_else(|| desks.first())?;
            let room = room_of(desk)?;
            return Some(match desk.area == here.area {
                true => format!("A desk is within reach in {room}: `move_to` it, and work there."),
                false => format!(
                    "The desks are on {level}, and the lift goes there: `move_to` the lift, \
                     `lift_use` naming {level} — it calls the car and waits for it — then \
                     `move_to` {room}.",
                    level = level_of(desk)?
                ),
            });
        }
        let aim = Aim::of(&step)?;
        if let Some(at) = reach.iter().find(|at| {
            room_of(at).is_some_and(|name| aim.is_room(&name, &level_of(at).unwrap_or_default()))
        }) {
            let room = room_of(at)?;
            return Some(match at.area == here.area {
                true => format!("{room} is within reach: `move_to` it."),
                false => format!(
                    "{room} is on {}: `move_to` the lift, `lift_use` naming {}, then `move_to` \
                     {room}.",
                    level_of(at)?,
                    level_of(at)?
                ),
            });
        }
        for d in s.devices.iter().filter(|d| aim.is_machine(&d.name)) {
            if let Some(at) = reach.iter().find(|at| place(at) == d.at) {
                let room = room_of(at)?;
                return Some(format!(
                    "{} stands in {room}, within reach: `move_to` {room}, and you read its \
                     state as you arrive.",
                    d.name
                ));
            }
        }
        for a in w
            .actors()
            .filter(|a| a.id != body && aim.is_person(&a.name))
        {
            if reach.contains(&a.at) {
                let room = room_of(&a.at)?;
                return Some(match a.at == here {
                    true => format!("{} is here with you: `ask` or `tell` them.", a.name),
                    false => format!(
                        "{} is in {room}, within reach: `move_to` {room} and `ask` or `tell` \
                         them — or `message` them, which reaches them wherever they are.",
                        a.name
                    ),
                });
            }
        }
        None
    })
}

/// Perform one station or bench act.
pub fn perform(hosted: &Hosted, body: &str, act: &Act) -> Outcome {
    let a = &act.args;
    // **The craft libraries are addressed by two halves**, which library and
    // which piece, so no single argument is the subject. Dispatched before the
    // subject is looked for rather than being turned away as subjectless — the
    // same reason `why` and `called` are in [`subject`]'s list.
    if matches!(act.tool, "library_read" | "library_write") {
        return library(hosted, body, act.tool, a);
    }
    // Names a year, which is no subject.
    if act.tool == "time_travel" {
        return time_travel(hosted, body, a);
    }
    // The mission acts before the subject is looked for: `collect_mission`
    // names nothing, and the reports carry their subject under `account` / `why`,
    // which the shared [`subject`] list does not scan.
    if crate::engine::mission_acts::is_mine(act.tool) {
        return mission(hosted, body, act);
    }
    let Some(what) = subject(a)
        .or_else(|| Some(String::new()))
        .filter(|s| !s.is_empty())
    else {
        // Only the argument-free bench acts get here legitimately.
        return bench_no_subject(hosted, body, act.tool);
    };

    match act.tool {
        // ── writing a document into the record ──────────────────────────────
        //
        // Everything here is an era-shaped document: a record item with a path
        // [`crate::sim::record::Record::settle_path`] mints, so the write goes
        // through the bench's working set and lands on the disk at the commit. A
        // place's entry is one of them — its `Place` kind now settles into
        // `layers/world/locations`. The character verbs and a place's local
        // history are *not* here: they reach the mind folder a different way and
        // are handled below.
        "chronicle_add_entry" | "story_draft" | "place_write_entry" | "chronicle_rewrite_page" => {
            // **The subject is the naming argument, never `what`.** Every act
            // here carries its text in `what` and the thing it writes into in a
            // preposition — `to` an era, `for` a gap, `in` a page, `of` a
            // character. Falling back to `what` looked harmless and meant
            // `story_draft` wrote the draft into a record item named after the
            // draft's own prose, and then reported success.
            let Some(target) = text(a, "to")
                .or_else(|| text(a, "of"))
                .or_else(|| text(a, "in"))
                .or_else(|| text(a, "for"))
            else {
                return Outcome::Refused("You meant to write, but not into what.".into());
            };
            let Some(body_text) = text(a, "what") else {
                return Outcome::Refused("You meant to write something, but not what.".into());
            };
            write_into(hosted, body, &target, &body_text)
        }
        // ── the character sheet, and the memory beside it ───────────────────
        //
        // **Not era-shaped documents.** Who a character is and what they want
        // are fields on their personality sheet, edited in place the way a
        // portrait's art direction is (`portrait_draw`, below) so the comments a
        // person wrote around them survive; what they remember is appended to
        // their own memory layer. None is a record item with a path, so none
        // goes through `write_into` — each reaches the mind folder straight
        // through the bench, and a world with no mind folder keeps the in-RAM
        // draft the shipped handler always did.
        "character_write_identity" => author_sheet(hosted, body, a, &["anchor"]),
        "character_write_wants" => author_sheet(hosted, body, a, &["wants"]),
        "character_write_memories" => {
            let (Some(who), Some(t)) = (text(a, "of"), text(a, "what")) else {
                return Outcome::Refused(
                    "You meant to write a memory, but not whose, or not what.".into(),
                );
            };
            hosted.with_sim(|s| {
                // No mind folder — the mind-less test daemon — keeps the draft
                // in the record, exactly as it did before any of this reached
                // disk.
                if !s.bench.has_root() {
                    return match s.record.write(&who, body, &t) {
                        Ok(n) => Outcome::Did(format!("You write into {n}: {t}")),
                        Err(why) => Outcome::Refused(why),
                    };
                }
                let path = format!("layers/memory/{}/memories.md", slug_of(&who));
                match s.bench.append(body, &who, &path, &t) {
                    Ok(p) => Outcome::Did(format!(
                        "Something that happened to {who} is written down ({p}). Nobody else sees \
                         it until you commit."
                    )),
                    Err(why) => Outcome::Refused(why),
                }
            })
        }
        "place_write_local_history" => {
            let (Some(place), Some(t)) = (text(a, "of"), text(a, "what")) else {
                return Outcome::Refused(
                    "You meant to write a place's history, but not whose, or not what.".into(),
                );
            };
            hosted.with_sim(|s| {
                // A place's local history is a second document, in geography —
                // its entry (its own `path`) is where you stand, this is what
                // happened here. No mind folder keeps it in the record's draft.
                let path = match s.bench.has_root() {
                    true => s.record.history_path(&place),
                    false => None,
                };
                let Some(path) = path else {
                    return match s.record.write(&place, body, &t) {
                        Ok(n) => Outcome::Did(format!("You write into {n}: {t}")),
                        Err(why) => Outcome::Refused(why),
                    };
                };
                if let Err(why) = s.record.claim_for_write(&place, body) {
                    return Outcome::Refused(why);
                }
                match s.bench.append(body, &place, &path, &t) {
                    Ok(p) => Outcome::Did(format!(
                        "The local history of {place} is written down ({p}). Nobody else sees it \
                         until you commit."
                    )),
                    Err(why) => Outcome::Refused(why),
                }
            })
        }
        // ── likenesses ──────────────────────────────────────────────────────
        //
        // **The words, not the picture.** A likeness is drawn from a prompt
        // authored beside the personality, and the picture is regenerated from
        // it — so what a Maker at an easel changes is the art direction, and
        // the plate follows. Editing the PNG is not something a character does
        // with a pen.
        "portrait_draw" | "portrait_prompt_edit" => {
            let Some(carrying) = text(a, "carrying") else {
                return Outcome::Refused(
                    "You meant to draw somebody, but did not say what the face has to carry."
                        .into(),
                );
            };
            let path = personality_path(&what);
            hosted.with_sim(|s| {
                match s
                    .bench
                    .write_field(body, &what, &path, PORTRAIT_PROMPT, &carrying)
                {
                    Ok(p) => Outcome::Did(format!(
                        "{what} is drawn from these words now ({p}): {carrying}"
                    )),
                    Err(why) => Outcome::Refused(why),
                }
            })
        }
        "portrait_prompt_read" => {
            let path = personality_path(&what);
            hosted.sim(|s| match s.bench.read_field(body, &path, PORTRAIT_PROMPT) {
                Ok(prompt) => Outcome::Did(format!("{what} is drawn from: {prompt}")),
                Err(why) => Outcome::Refused(why),
            })
        }

        // ── moving something along ──────────────────────────────────────────
        "story_file" | "portrait_file_plate" => set_state(hosted, body, &what, State::Filed),
        "chronicle_retire_entry" => set_state(hosted, body, &what, State::Retired),

        // ── settling, which takes two ───────────────────────────────────────
        "chronicle_settle_boundary"
        | "portrait_settle_likeness"
        | "character_settle_relation"
        | "place_settle_route"
        | "map_settle_border" => settle(hosted, body, a, &what),

        // ── the map ─────────────────────────────────────────────────────────
        "map_add_place" => {
            let Some(called) = text(a, "called") else {
                return Outcome::Refused("A new place needs a name.".into());
            };
            let Some(wheres) = text(a, "where") else {
                return Outcome::Refused(
                    "A place added without saying what it sits between contradicts its neighbours."
                        .into(),
                );
            };
            hosted.with_sim(|s| {
                if s.record.by_name(&called).is_some() {
                    return Outcome::Refused(format!(
                        "There is already somewhere called {called}."
                    ));
                }
                let mut i = Item::new(
                    called.to_lowercase().replace(' ', "_"),
                    &called,
                    Kind::Place,
                );
                i.body = wheres.clone();
                i.state = State::Draft;
                i.holder = Some(body.to_string());
                s.record.put(i);
                Outcome::Did(format!("{called} is on the map, {wheres}."))
            })
        }
        "map_remove_place" => {
            hosted.with_sim(|s| match s.record.let_go(&what, "taken out of the world") {
                Ok(n) => Outcome::Did(format!(
                "{n} is off the map. Everything written about it still stands and now has to be \
                 reckoned with."
            )),
                Err(why) => Outcome::Refused(why),
            })
        }

        // ── custody, appraisal, description, condition ──────────────────────
        "record_accession" => {
            let Some(from) = text(a, "from") else {
                return Outcome::Refused(
                    "Nothing is taken in without saying where it came from — no later work \
                     supplies an origin nobody wrote down."
                        .into(),
                );
            };
            hosted.with_sim(|s| {
                if s.record.by_name(&what).is_none() {
                    let mut i = Item::new(
                        what.to_lowercase().replace(' ', "_"),
                        &what,
                        Kind::Accession,
                    );
                    i.provenance = Some(from.clone());
                    i.holder = Some(body.to_string());
                    i.state = State::Held;
                    s.record.put(i);
                    return Outcome::Did(format!("{what} is taken in, from {from}."));
                }
                match s.record.set_provenance(&what, &from) {
                    Ok(n) => Outcome::Did(format!("{n} is taken in, from {from}.")),
                    Err(why) => Outcome::Refused(why),
                }
            })
        }
        "record_write_provenance" => {
            let Some(from) = text(a, "from") else {
                return Outcome::Refused("You meant to write down an origin, but not what.".into());
            };
            hosted.with_sim(|s| match s.record.set_provenance(&what, &from) {
                Ok(n) => Outcome::Did(format!("Where {n} came from is now written down: {from}")),
                Err(why) => Outcome::Refused(why),
            })
        }
        "record_hand_on" => {
            let Some(to) = text(a, "to") else {
                return Outcome::Refused("You meant to hand something on, but not to whom.".into());
            };
            hosted.with_sim(|s| match s.record.give_back(&what, body) {
                Ok(n) => Outcome::Did(format!(
                    "You hand {n} to {to}, with what you did to it and what you did not."
                )),
                Err(why) => Outcome::Refused(why),
            })
        }
        "record_appraise" => {
            let Some(verdict) = text(a, "verdict") else {
                return Outcome::Refused("An appraisal without a judgement is a look.".into());
            };
            hosted.with_sim(|s| {
                s.ledger.record_verdict(&what, body, &verdict, None);
                Outcome::Did(format!("You weigh {what}: {verdict}"))
            })
        }
        "record_write_reason" => {
            let Some(why) = text(a, "why") else {
                return Outcome::Refused(
                    "A reason nobody can read is a reason nobody can disagree with.".into(),
                );
            };
            hosted.with_sim(|s| {
                s.ledger.record_verdict(&what, body, "let go", Some(&why));
                Outcome::Did(format!("Why {what} was let go is written down: {why}"))
            })
        }
        "record_let_go" => {
            let Some(because) = text(a, "because") else {
                return Outcome::Refused(
                    "Nothing is let go without a reason — whoever comes after will want to know, \
                     and the only honest answer is one written at the time."
                        .into(),
                );
            };
            hosted.with_sim(|s| match s.record.let_go(&what, &because) {
                Ok(n) => Outcome::Did(format!("{n} is let go: {because}")),
                Err(why) => Outcome::Refused(why),
            })
        }
        "record_describe" => with_second(hosted, a, "how", &what, |s, w, how| {
            s.record
                .describe(w, how)
                .map(|n| format!("The way in to {n}: {how}"))
        }),
        "record_arrange" => with_second(hosted, a, "under", &what, |s, w, under| {
            s.record
                .describe(w, under)
                .map(|n| format!("{n} now sits under {under}, where somebody would look for it."))
        }),
        "record_cross_reference" => with_second(hosted, a, "to", &what, |s, w, to| {
            s.record
                .cross_reference(w, to)
                .map(|t| format!("{w} now points at {t}."))
        }),
        "record_leave_note" => with_second(hosted, a, "what", &what, |s, w, note| {
            s.record
                .describe(w, note)
                .map(|n| format!("A note on {n} for whoever picks it up: {note}"))
        }),
        // **Tidying leaves the tidied thing changed.** This used to check the
        // item existed and return prose, which reads as work and is not: the
        // index it claimed to correct was exactly as wrong afterwards.
        "record_tidy_index" => hosted.with_sim(|s| {
            let Some(i) = s.record.by_name(&what) else {
                return Outcome::Refused(format!("There is nothing called {what} here."));
            };
            let (name, kind) = (i.name.clone(), i.kind);
            let entry = format!("{name} — {}", kind.describes());
            match s.record.describe(&name, &entry) {
                Ok(n) => Outcome::Did(format!(
                    "You bring {n} back to what it actually indexes: {entry}. An index that lies \
                     stops people looking."
                )),
                Err(why) => Outcome::Refused(why),
            }
        }),
        "record_mend" => with_second(hosted, a, "how", &what, |s, w, how| {
            s.record
                .set_condition(w, Condition::Mended)
                .map(|n| format!("You mend {n}: {how}"))
        }),
        "record_mark_repair" => {
            hosted.with_sim(|s| match s.record.set_condition(&what, Condition::Mended) {
                Ok(n) => Outcome::Did(format!(
                    "The repair to {n} is left visible. A mend passed off as an original is worse \
                 than the damage."
                )),
                Err(why) => Outcome::Refused(why),
            })
        }

        // ── the record facing outward ───────────────────────────────────────
        "enquiry_take_question" => hosted.with_sim(|s| match s.record.take(&what, body) {
            Ok(n) => Outcome::Did(format!("You take {n}. It is yours until it is answered.")),
            Err(why) => Outcome::Refused(why),
        }),
        "enquiry_answer_from_record" => with_second(hosted, a, "answer", &what, |s, w, ans| {
            s.record
                .write(w, "", ans)
                .map(|n| format!("You answer {n}: {ans}"))
        }),
        "enquiry_name_the_gap" => with_second(hosted, a, "missing", &what, |s, w, missing| {
            s.record
                .describe(w, missing)
                .map(|n| format!("{n} cannot be answered. What is missing: {missing}"))
        }),
        "enquiry_raise_work" => {
            let Some(work) = text(a, "work") else {
                return Outcome::Refused("You meant to raise work, but did not say what.".into());
            };
            hosted.with_sim(|s| {
                s.ledger.set_order(&work, body, None);
                Outcome::Did(format!("A gap turned into work anybody can take: {work}"))
            })
        }

        // ── orders, dispatch, the cast ──────────────────────────────────────
        "orders_set" => hosted.with_sim(|s| {
            let on_behalf = text(a, "for");
            s.ledger.set_order(&what, body, on_behalf.as_deref());
            Outcome::Did(match on_behalf {
                Some(w) => format!("On the board, in {w}'s name: {what}"),
                None => format!("On the board: {what}"),
            })
        }),
        "orders_hand_to" => {
            let Some(to) = text(a, "to") else {
                return Outcome::Refused(
                    "You meant to hand an order to somebody, but not who.".into(),
                );
            };
            hosted.with_sim(|s| match s.ledger.hand_to(&what, &to) {
                Ok(()) => Outcome::Did(format!("{to} has it: {what}")),
                Err(why) => Outcome::Refused(why),
            })
        }
        "orders_report_done" => hosted.with_sim(|s| match s.ledger.finish(body, &what) {
            true => Outcome::Did(format!("Reported done: {what}")),
            false => Outcome::Refused(format!("You are not holding an order to {what}.")),
        }),
        "dispatch_post_wake" => with_second(hosted, a, "breaks", &what, |s, w, breaks| {
            s.ledger.set_order(breaks, "the wake", Some(w));
            Ok(format!("Posted, so they hear it from you: {breaks}"))
        }),
        "cast_report_disagreement" => with_second(hosted, a, "what", &what, |s, w, disag| {
            s.ledger.set_order(disag, "the watch", Some(w));
            Ok(format!("Reported: {disag}. It is not yours to correct."))
        }),
        "creator_present" => hosted.with_sim(|s| match s.record.by_name(&what) {
            Some(i) if i.state == State::Filed => Outcome::Did(format!(
                "You present {}. What you would still change about it is the part worth hearing.",
                i.name
            )),
            Some(i) => Outcome::Refused(format!(
                "{} is not filed yet. Present it when it is finished, not while it is still yours.",
                i.name
            )),
            None => Outcome::Refused(format!(
                "There is nothing called {what} to present. The chair presents a finished piece \
                 of work by its name; a report is not presented here — it is handed in at the \
                 table where work is handed out, with its `report_done`."
            )),
        }),

        // ── the plant and the stores ────────────────────────────────────────
        "plant_note_drift" => with_second(hosted, a, "drift", &what, |s, w, drift| {
            s.ledger
                .set_order(&format!("look at {w}: {drift}"), "the panel", None);
            Ok(format!("Noted, so somebody looks: {w} is {drift}"))
        }),
        "plant_raise_fault" => with_second(hosted, a, "why", &what, |s, w, why| {
            s.ledger
                .set_order(&format!("the fault in {w}"), "the panel", None);
            Ok(format!(
                "Raised, and you may be wrong in public: {w} — {why}"
            ))
        }),
        // **Putting back something you are not holding is not putting it
        // back.** This reported success on every failure — a thing nobody has
        // heard of, a thing already racked — so a character could rack the same
        // imaginary object every turn and be told it worked each time.
        "stores_put_back" => hosted.with_sim(|s| match s.record.give_back(&what, body) {
            Ok(n) => Outcome::Did(format!("{n} is racked where it belongs.")),
            Err(why) => Outcome::Refused(why),
        }),
        "stores_take_out" => hosted.with_sim(|s| match s.record.take(&what, body) {
            Ok(n) => Outcome::Did(format!("{n} is out, and out until somebody racks it.")),
            Err(why) => Outcome::Refused(why),
        }),

        // ── the shape of a piece ────────────────────────────────────────────
        // ── the shape of a piece, and each one leaves a finding ──────────────
        //
        // **A judgement nobody can read is a judgement nobody can disagree
        // with.** All three of these used to return prose and change nothing:
        // the character did the work, said so, and the next character to pick
        // the piece up found no sign of it — so the work was done again, and
        // again, with nothing accumulating. Each now lands a verdict on the
        // ledger, which is what `record_appraise` already does and what makes a
        // reading survive the turn that produced it.
        "structure_lay_out_scenes" => hosted.with_sim(|s| {
            let Some(i) = s.record.by_name(&what) else {
                return Outcome::Refused(format!("There is nothing called {what} to lay out."));
            };
            let name = i.name.clone();
            s.ledger
                .record_verdict(&name, body, "laid out as its scenes", None);
            Outcome::Did(format!(
                "{name} is pinned up as its scenes. The one where nobody wants anything shows \
                 itself from here and could not from inside the prose."
            ))
        }),
        "structure_test_the_want" => with_second(hosted, a, "scene", &what, |s, w, scene| {
            if s.record.by_name(w).is_none() {
                return Err(format!("There is nothing called {w} here."));
            }
            s.ledger.record_verdict(
                w,
                body,
                &format!("the want in {scene} was tested"),
                Some("a want you could not photograph being satisfied is a mood"),
            );
            Ok(format!(
                "You test what is wanted in {scene}, in {w}: could you photograph the moment it \
                 is satisfied? If not it is a mood, and it has to be replaced."
            ))
        }),
        "structure_find_the_slack" => hosted.with_sim(|s| {
            let Some(i) = s.record.by_name(&what) else {
                return Outcome::Refused(format!("There is nothing called {what} here."));
            };
            let name = i.name.clone();
            s.ledger.record_verdict(
                &name,
                body,
                "the slack was found",
                Some("where it stops being a chain of consequences and becomes a list of events"),
            );
            Outcome::Did(format!(
                "You find where {name} stops being a chain of consequences and becomes a list of \
                 events."
            ))
        }),
        "gather_call" => with_second(hosted, a, "who", &what, |s, about, who| {
            s.ledger.set_order(
                &format!("come to the reading about {about}"),
                "the table",
                Some(who),
            );
            Ok(format!(
                "Called: everybody whose work touches {about} — {who}"
            ))
        }),

        // ── the bench ───────────────────────────────────────────────────────
        _ => bench(hosted, body, act.tool, a, &what),
    }
}

/// Reading and changing a piece of the craft.
///
/// Kept out of the main dispatch because these are the only acts addressed by
/// two arguments rather than a subject and a preposition — and because what
/// they change is not a document about the world but *how everybody in it
/// reads*, which is worth having in one place somebody can find.
/// What a Maker handing in `mission` is told of its time: back in the world's
/// present when the mission was worked in a year of the past — the year is the
/// mission's, and leaves with it.
fn back_to_now(mission: &Mission) -> String {
    match mission.year {
        Some(year) => format!(
            "\nYou are back in the world's present; {year} and what came before it are behind \
             you again."
        ),
        None => String::new(),
    }
}

/// `time_travel`: work in the year `args` names, within the world's history,
/// for as long as the mission the body carries — see
/// [`crate::engine::mission::Mission::travelled`].
fn time_travel(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    // A year arrives as text or as a number — the device's body takes either,
    // and a Maker that sent `{"year": 2937}` was told to name it as a number.
    let year = match args.get("year") {
        Some(Value::Number(n)) => n.as_u64().and_then(|y| u32::try_from(y).ok()),
        _ => text(args, "year").and_then(|y| {
            y.chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<u32>()
                .ok()
        }),
    };
    // **A refusal says what to do, in the mission's own terms.** "Name the year
    // as a number: 2937" sent a Maker whose mission asked for 2950 to ask its
    // colleagues what had happened in 2950 — it read a malformed call as a gap
    // in what it knew, and nobody it asked could see the call either.
    let Some(year) = year else {
        let asked = hosted
            .sim(|s| {
                s.missions
                    .active(body)?
                    .todo
                    .iter()
                    .filter(|t| !t.done)
                    .find_map(|t| time_step(&t.text))
            })
            .map(|y| {
                format!(
                    " Your mission asks you to work in {y}: `time_travel` with the year \"{y}\"."
                )
            })
            .unwrap_or_default();
        return Outcome::Refused(format!(
            "Your call named no year — the machine needs the year written in it.{asked}"
        ));
    };
    let eras = hosted
        .sim(|s| s.bench.mind_root().map(chronology::eras))
        .unwrap_or_default();
    if let (Some(first), Some(now)) = (eras.first(), eras.last()) {
        if year < first.opens {
            return Outcome::Refused(format!(
                "{year} is before the world's history opens, in {} ({}).",
                first.opens, first.title
            ));
        }
        if year > now.opens {
            return Outcome::Refused(format!(
                "{year} is after the present ({}); there is nothing there yet.",
                now.opens
            ));
        }
    }
    let era = chronology::standing_in(&eras, year)
        .last()
        .map(|e| format!(", in {}", e.title))
        .unwrap_or_default();
    hosted.with_sim(|s| match s.missions.travelled(body, year) {
        None => Outcome::Refused(
            "You carry no work to stand in a year for. Take up a mission at the command table \
             first."
                .into(),
        ),
        Some(ticked) => {
            tracing::info!(npc = body, year, "time machine set");
            let mut said = format!(
                "You work in {year} now{era}. Nothing after it comes back to you until you report \
                 the work you carry."
            );
            if ticked {
                let next = s.missions.active(body).and_then(|m| m.next_step());
                said.push('\n');
                said.push_str(&progress_line("The time is set", next, &[]));
            }
            Outcome::Did(said)
        }
    })
}

fn library(hosted: &Hosted, body: &str, tool: &str, args: &Map<String, Value>) -> Outcome {
    let path = match library_path(args) {
        Ok(p) => p,
        Err(why) => return Outcome::Refused(why),
    };
    if tool == "library_read" {
        return hosted.sim(|s| match s.bench.read_field(body, &path, &["template"]) {
            Ok(t) => Outcome::Did(format!("{path}:\n{t}")),
            Err(why) => Outcome::Refused(why),
        });
    }
    let (Some(field), Some(t)) = (text(args, "field"), verbatim(args, "text")) else {
        return Outcome::Refused(
            "Changing a piece of the craft is a field and what it says. Say both.".into(),
        );
    };
    if !LIBRARY_FIELDS.contains(&field.as_str()) {
        return Outcome::Refused(format!(
            "There is no {field} to change. What can be changed is {}.",
            LIBRARY_FIELDS.join(" or ")
        ));
    }
    hosted.with_sim(
        |s| match s.bench.write_field(body, &path, &path, &[&field], &t) {
            Ok(p) => Outcome::Did(format!(
                "The {field} of {p} says what you wrote. It is how everybody here reads, once you \
             commit it."
            )),
            Err(why) => Outcome::Refused(why),
        },
    )
}

/// Where a likeness's art direction lives inside a personality.
const PORTRAIT_PROMPT: &[&str] = &[crate::personality_portrait::FIELD, "prompt"];

/// The fields of a craft piece a Maker may change.
///
/// `id` and `category` are deliberately absent: an id is how every other
/// document refers to this one and a category is how the engine selects it, so
/// changing either from inside the world silently unhooks the piece from
/// everything that points at it.
const LIBRARY_FIELDS: &[&str] = &["description", "template"];

/// The document a personality is.
fn personality_path(who: &str) -> String {
    format!("personalities/{}.yaml", slug_of(who))
}

/// Write one field of a character's founding sheet, in place.
///
/// The `portrait_draw` pattern for the two identity fields — `anchor` (who they
/// are) and `wants` (what they are after). The sheet is a YAML document and the
/// field is spliced into it through [`crate::sim::bench::Benches::write_field`],
/// so the reasoning a person wrote around it in comments survives where a
/// round-trip would lose it. A world with no mind folder has no sheet to edit
/// and keeps the in-RAM draft the shipped handler always kept.
fn author_sheet(hosted: &Hosted, body: &str, args: &Map<String, Value>, at: &[&str]) -> Outcome {
    let (Some(who), Some(t)) = (text(args, "of"), text(args, "what")) else {
        return Outcome::Refused(
            "You meant to write who somebody is, but not whose, or not what.".into(),
        );
    };
    hosted.with_sim(|s| {
        if !s.bench.has_root() {
            return match s.record.write(&who, body, &t) {
                Ok(n) => Outcome::Did(format!("You write into {n}: {t}")),
                Err(why) => Outcome::Refused(why),
            };
        }
        let path = personality_path(&who);
        match s.bench.write_field(body, &who, &path, at, &t) {
            Ok(p) => Outcome::Did(format!(
                "{who} is written into {p}: {t}. Nobody else sees it until you commit."
            )),
            Err(why) => Outcome::Refused(why),
        }
    })
}

/// The document a piece of the craft is, from the two halves that address it.
fn library_path(args: &Map<String, Value>) -> Result<String, String> {
    let (Some(kind), Some(id)) = (text(args, "kind"), text(args, "id")) else {
        return Err("A piece of the craft is named by its library and its id. Say both.".into());
    };
    let dir = match kind.trim().to_lowercase().as_str() {
        "mood" | "moods" => "moods",
        "response" | "responses" => "responses",
        other => {
            return Err(format!(
                "There is no {other} library. There is `mood` and there is `response`."
            ))
        }
    };
    Ok(format!("{dir}/{}.yaml", slug_of(&id)))
}

/// Write into something, taking it if nobody has it.
///
/// **A thing that is a document is written as one.** An era, a story — anything
/// [`crate::sim::record::Record::index_canon`] found on the disk — carries a
/// path, and the write goes through the bench's working set: unseen until the
/// commit, checked against somebody else's commit when it lands, and part of
/// the mind afterwards. Everything else is a judgement rather than a document
/// and stays in the record, which is where a judgement belongs.
fn write_into(hosted: &Hosted, body: &str, what: &str, text: &str) -> Outcome {
    hosted.with_sim(|s| {
        // An era or a story that has no document yet gets one here — the first
        // entry written into an era is what creates it. A world with nowhere to
        // keep documents settles nothing and falls through to the record.
        let path = match s.bench.has_root() {
            true => s.record.settle_path(what),
            false => s.record.path_of(what),
        };
        let Some(path) = path else {
            return match s.record.write(what, body, text) {
                Ok(n) => Outcome::Did(format!("You write into {n}: {text}")),
                Err(why) => Outcome::Refused(why),
            };
        };
        // The record's custody rules still decide *whether* it may be written —
        // somebody else holding it refuses here exactly as it always did — and
        // only then does the text go to the document.
        if let Err(why) = s.record.claim_for_write(what, body) {
            return Outcome::Refused(why);
        }
        match s.bench.append(body, what, &path, text) {
            Ok(p) => Outcome::Did(format!(
                "You write into {what} ({p}). Nobody else sees it until you commit."
            )),
            Err(why) => Outcome::Refused(why),
        }
    })
}

fn set_state(hosted: &Hosted, body: &str, what: &str, to: State) -> Outcome {
    hosted.with_sim(|s| match s.record.set_state(what, body, to) {
        Ok(n) => Outcome::Did(match to {
            State::Filed => format!("{n} is filed. It is part of the record and no longer yours."),
            State::Retired => format!("{n} is out of the record."),
            _ => format!("{n} is {to:?}."),
        }),
        Err(why) => Outcome::Refused(why),
    })
}

/// An act needing a second argument, applied to the store.
fn with_second(
    hosted: &Hosted,
    args: &Map<String, Value>,
    key: &str,
    what: &str,
    f: impl FnOnce(&mut crate::sim::Sim, &str, &str) -> Result<String, String>,
) -> Outcome {
    let Some(second) = text(args, key) else {
        return Outcome::Refused(format!("You meant to, but did not say `{key}`."));
    };
    hosted.with_sim(|s| match f(s, what, &second) {
        Ok(line) => Outcome::Did(line),
        Err(why) => Outcome::Refused(why),
    })
}

/// The settling acts, which all need somebody else and all land in both records.
fn settle(hosted: &Hosted, body: &str, args: &Map<String, Value>, mine: &str) -> Outcome {
    let theirs = text(args, "and")
        .or_else(|| text(args, "with"))
        .or_else(|| text(args, "to"));
    let Some(theirs) = theirs else {
        return Outcome::Refused(
            "Settling takes two. Name the other side of it — you cannot decide it alone.".into(),
        );
    };
    hosted.with_sim(|s| {
        if s.record.by_name(mine).is_none() {
            return Outcome::Refused(format!("There is nothing called {mine} here."));
        }
        match s.record.cross_reference(mine, &theirs) {
            Ok(t) => {
                let _ = s.record.cross_reference(&t, mine);
                s.ledger
                    .set_order(&format!("settle {mine} against {t}"), body, None);
                Outcome::Did(format!(
                    "Settled between {mine} and {t}, and it lands in both at once."
                ))
            }
            Err(why) => Outcome::Refused(why),
        }
    })
}

/// The bench acts that name nothing — they act on what you have open.
///
/// **Two stores, one act.** A body at a bench holds a record item — the thing
/// in the world it has taken on — *and* a working set of documents it has
/// changed and not committed. Setting work aside means both, and so does
/// throwing it away, which is why each of these touches the pair rather than
/// picking one. Either half alone is a legitimate state: a Maker can hold an
/// era without having edited a file yet, and can be editing a scratch document
/// the record has never heard of.
fn bench_no_subject(hosted: &Hosted, body: &str, tool: &str) -> Outcome {
    let held = hosted.sim(|s| s.record.held_by(body));
    match tool {
        "bench_diff" => {
            let changed = hosted.sim(|s| s.bench.diff(body));
            match (changed.is_empty(), held.first()) {
                (false, _) => Outcome::Did(format!(
                    "Changed by you and not yet committed: {}.",
                    changed.join(", ")
                )),
                (true, Some(w)) => Outcome::Did(format!(
                    "{w} is open and you have not changed anything in it."
                )),
                (true, None) => Outcome::Did("You have nothing open.".into()),
            }
        }
        "bench_stash" => {
            let set_aside = hosted.with_sim(|s| s.bench.stash(body));
            match held.first() {
                Some(w) => {
                    let w = w.clone();
                    hosted.with_sim(|s| {
                        let _ = s.record.give_back(&w, body);
                    });
                    Outcome::Did(format!("{w} is set aside. It waits for you."))
                }
                None if set_aside => {
                    Outcome::Did("Your changes are set aside. They wait for you.".into())
                }
                None => Outcome::Refused("You have nothing open to set aside.".into()),
            }
        }
        "bench_stash_pop" => hosted.with_sim(|s| match s.bench.pop(body) {
            Ok(about) => Outcome::Did(format!("You pick {about} back up, where you left it.")),
            Err(why) => Outcome::Refused(why),
        }),
        "bench_restore" => {
            let threw = hosted.with_sim(|s| s.bench.discard(body));
            match held.first() {
                Some(w) => {
                    let w = w.clone();
                    hosted.with_sim(|s| {
                        let _ = s.record.give_back(&w, body);
                    });
                    Outcome::Did(format!(
                        "Your changes to {w} are gone. It is back to what it was."
                    ))
                }
                None if threw => Outcome::Did(
                    "Your changes are gone. The documents are back to what they were.".into(),
                ),
                None => Outcome::Refused("You have nothing open to throw away.".into()),
            }
        }
        "bench_stage" => {
            let offered = hosted.with_sim(|s| s.bench.set_offered(body, true));
            match held.first() {
                Some(w) => set_state(hosted, body, &w.clone(), State::Offered),
                None if offered => Outcome::Did(
                    "Your changes are up to be looked at. They are not part of what stands yet."
                        .into(),
                ),
                None => Outcome::Refused("You have nothing open to offer.".into()),
            }
        }
        "bench_unstage" => {
            let withdrawn = hosted.with_sim(|s| s.bench.set_offered(body, false));
            match held.first() {
                Some(w) => set_state(hosted, body, &w.clone(), State::Draft),
                None if withdrawn => {
                    Outcome::Did("You take your changes back. They are yours again.".into())
                }
                None => Outcome::Refused("You have nothing offered to take back.".into()),
            }
        }
        "bench_status" => {
            let (changed, offered) = hosted.sim(|s| {
                (
                    s.bench.diff(body),
                    s.bench.opened(body).is_some_and(|w| w.offered),
                )
            });
            let mut lines: Vec<String> = Vec::new();
            if !held.is_empty() {
                lines.push(format!("Open: {}", held.join(", ")));
            }
            if !changed.is_empty() {
                lines.push(format!(
                    "{}: {}",
                    match offered {
                        true => "Offered",
                        false => "Changed and not committed",
                    },
                    changed.join(", ")
                ));
            }
            Outcome::Did(match lines.is_empty() {
                true => "Nothing open, nothing offered.".into(),
                false => format!("{}.", lines.join(". ")),
            })
        }
        _ => Outcome::Refused(format!("`{tool}` needs to know what it is about.")),
    }
}

/// The bench acts that name what they are about.
fn bench(
    hosted: &Hosted,
    body: &str,
    tool: &str,
    args: &Map<String, Value>,
    what: &str,
) -> Outcome {
    if matches!(tool, "file_write" | "file_edit" | "file_delete") {
        if let Some(why) = hosted.sim(|s| outside_the_mission(s, body, what)) {
            return Outcome::Refused(why);
        }
    }
    match tool {
        // A working set can be opened on something the record has never heard
        // of — making a new document is exactly that — so the record is taken
        // only when the name is one it knows.
        "bench_branch" => hosted.with_sim(|s| {
            let opened = match s.bench.open_on(body, what) {
                Ok(fresh) => fresh,
                Err(why) => return Outcome::Refused(why),
            };
            if s.record.by_name(what).is_some() {
                if let Err(why) = s.record.take(what, body) {
                    // Undo only a set this call actually opened; one that was
                    // already there holds work that is not ours to throw away.
                    if opened {
                        s.bench.discard(body);
                    }
                    return Outcome::Refused(why);
                }
            }
            Outcome::Did(format!(
                "{what} is open and yours. Nobody else can commit over you while it is."
            ))
        }),
        // **The one act somebody else's work can refuse**, and the refusal is
        // the whole point: it names the other party, and two Makers who have to
        // settle something have a reason to be in one room that no idle nudge
        // could ever manufacture.
        "bench_commit" => {
            let Some(why_line) = text(args, "why") else {
                return Outcome::Refused(
                    "A commit needs one line saying what it is, for whoever reads the history in \
                     a year."
                        .into(),
                );
            };
            hosted.with_sim(|s| {
                // **A mission's document is committed whole or not at all.** A
                // story asked for at six hundred words was committed at three
                // hundred, a summary of the scene in place of the scene; the
                // floor (`Work::min_words`) turns that back while it can still
                // be finished.
                if let Some(short) = short_of_the_floor(s, body) {
                    return Outcome::Refused(short);
                }
                // The documents go first: it is the half somebody else's work
                // can refuse, and a refusal has to leave everything — the
                // working set and the record item — exactly as it was.
                let written = match s.bench.opened(body).is_some() {
                    true => match s.bench.commit(body) {
                        Ok(w) => w,
                        Err(collision) => {
                            return Outcome::Refused(format!(
                                "{collision} That is somebody to talk to, not something to try \
                                 again."
                            ))
                        }
                    },
                    false => Vec::new(),
                };
                // Committing a document a mission writes is that step done, and
                // what comes next is said with the commit.
                let progress = s.missions.committed(body, &written).then(|| {
                    let next = s.missions.active(body).and_then(|m| m.next_step());
                    progress_line("It is committed", next, &[])
                });
                let with_progress = |line: String| match &progress {
                    Some(p) => format!("{line}\n{p}"),
                    None => line,
                };
                let held = s.record.held_by(body);
                let Some(target) = held.first().cloned() else {
                    return match written.is_empty() {
                        true => Outcome::Refused("You have nothing open to merge.".into()),
                        false => Outcome::Did(with_progress(format!(
                            "{} is part of what stands: {why_line}",
                            written.join(", ")
                        ))),
                    };
                };
                // **A filed document that is changed and committed stays
                // filed.** Before documents were real, filing was a one-way
                // door — a thing became part of the record and the only move
                // left was to retire it. A document is not like that: it is
                // opened, changed and committed over and over, and what the
                // commit moves is the file rather than its standing. So the
                // record is *released* instead, which is what lets the next
                // Maker open it.
                let standing = s.record.by_name(&target).map(|i| i.state);
                let landed = match standing {
                    Some(State::Filed) => s.record.give_back(&target, body),
                    _ => s.record.set_state(&target, body, State::Filed),
                };
                match landed {
                    Ok(n) => Outcome::Did(with_progress(match written.is_empty() {
                        true => format!("{n} is merged into what stands: {why_line}"),
                        false => format!(
                            "{n} is merged into what stands, and {} with it: {why_line}",
                            written.join(", ")
                        ),
                    })),
                    Err(collision) => Outcome::Refused(format!(
                        "{collision} That is somebody to talk to, not something to try again."
                    )),
                }
            })
        }
        // A document that has been committed answers for itself, and the answer
        // is not the holder's to write: it is who the commit was made by.
        "bench_blame" | "bench_log" if hosted.sim(|s| s.bench.last_hand(what).is_some()) => hosted
            .sim(|s| {
                let who = s.bench.last_hand(what).unwrap_or_default().to_string();
                Outcome::Did(format!("{what} was last committed by {who}."))
            }),
        "bench_blame" => hosted.sim(|s| match s.record.by_name(what) {
            Some(i) => Outcome::Did(match (&i.holder, &i.provenance) {
                (Some(h), _) => format!("{} is held by {h}.", i.name),
                (None, Some(p)) => format!("{} came from {p}. Nobody holds it now.", i.name),
                (None, None) => format!(
                    "{} has no origin written anywhere, and that is where the chain goes quiet.",
                    i.name
                ),
            }),
            None => Outcome::Refused(format!("There is nothing called {what} to trace.")),
        }),
        "bench_log" => hosted.sim(|s| match s.record.by_name(what) {
            Some(i) => Outcome::Did(format!(
                "{}: {:?}, {:?}{}.",
                i.name,
                i.state,
                i.condition,
                match &i.let_go_because {
                    Some(r) => format!(", let go because {r}"),
                    None => String::new(),
                }
            )),
            None => Outcome::Refused(format!("There is nothing called {what} here.")),
        }),
        // ── the documents ───────────────────────────────────────────────────
        //
        // A read shows the body its *own* version — the uncommitted one when it
        // has made changes, and what stands otherwise. That asymmetry is the
        // whole working set: it is what "your work is yours alone until you
        // offer it" means when somebody actually tries to look.
        // `start_line` is a string because the grammar cannot bound an integer,
        // so the act parses it. Anything unparseable reads as the start rather
        // than refusing: a character that wrote "the top" meant line one, and
        // turning that into a refusal spends its turn on arithmetic.
        "file_read" => {
            let from = text(args, "start_line")
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(1);
            hosted.with_sim(|s| match s.bench.excerpt(body, what, from) {
                Ok(mut rendered) => {
                    // Reading a document a mission names is that step done.
                    if s.missions.read_doc(body, what) {
                        let next = s.missions.active(body).and_then(|m| m.next_step());
                        rendered.push('\n');
                        rendered.push_str(&progress_line("You have read it", next, &[]));
                    }
                    Outcome::Did(rendered)
                }
                Err(why) => Outcome::Refused(why),
            })
        }
        "file_list" => hosted.sim(|s| match s.bench.list(body, what) {
            Ok(names) if names.is_empty() => Outcome::Did(format!("Nothing under {what}.")),
            Ok(names) => Outcome::Did(format!("Under {what}: {}.", names.join(", "))),
            Err(why) => Outcome::Refused(why),
        }),
        "file_write" => {
            let Some(t) = verbatim(args, "content") else {
                return Outcome::Refused("You meant to write a document, but not what.".into());
            };
            hosted.with_sim(|s| match s.bench.write(body, what, what, &t) {
                Ok(p) => Outcome::Did(format!(
                    "{p} says what you wrote. Nobody else sees it until you commit."
                )),
                Err(why) => Outcome::Refused(why),
            })
        }
        "file_edit" => {
            let (Some(old), Some(new)) = (verbatim(args, "old_str"), verbatim(args, "new_str"))
            else {
                return Outcome::Refused(
                    "An edit is what you are replacing and what stands there instead. Say both."
                        .into(),
                );
            };
            hosted.with_sim(|s| {
                if let Some(why) = rewritten_by_hand(s, body, what, &old) {
                    return Outcome::Refused(why);
                }
                match s.bench.edit(body, what, what, &old, &new) {
                    Ok(p) => {
                        Outcome::Did(format!("{p} is changed, and the rest of it is untouched."))
                    }
                    Err(why) => Outcome::Refused(why),
                }
            })
        }
        // The record's guard comes first: a *filed* thing is retired with a
        // reason, and letting `file_delete` take one would lose the reason
        // whoever comes after will want.
        "file_delete" => hosted.with_sim(|s| {
            if let Some(i) = s.record.by_name(what) {
                if i.state == State::Filed {
                    return Outcome::Refused(format!(
                        "{} is part of the record. Taking it out is `record_let_go`, which keeps \
                         the reason.",
                        i.name
                    ));
                }
            }
            match s.bench.remove(body, what, what) {
                Ok(p) => Outcome::Did(format!(
                    "{p} goes when you commit. It was never part of the record."
                )),
                Err(why) => Outcome::Refused(why),
            }
        }),
        other => Outcome::Refused(format!("Nothing here knows how to {other}.")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use npc_map::world::Where;
    use serde_json::json;

    fn act(tool: &'static str, args: Value) -> Act {
        Act {
            tool,
            args: args.as_object().unwrap().clone(),
        }
    }

    /// **A piece under its floor is told how far short it is**, and to add to
    /// what is there — not that it is a sketch, which Makers read as a verdict
    /// on the writing and answered by resubmitting it, or cutting it shorter.
    #[test]
    fn a_short_piece_is_told_the_gap_and_to_add_to_it() {
        let why = short_by("layers/life/zen/2491 The Silence.md", 218, 250, None);
        assert!(
            why.starts_with(
                "layers/life/zen/2491 The Silence.md is 218 of the 250 words asked for — about \
                 32 more."
            ),
            "{why}"
        );
        assert!(why.contains("keep what is there and add to it"), "{why}");
        assert!(!why.contains("sketch"), "{why}");
    }

    /// **And it is told what the piece is to hold.** A Maker short of its
    /// length rewrote the same words, sure there was nothing more to say; the
    /// brief's own account of what happens is the scene to add from.
    #[test]
    fn a_short_piece_is_told_what_happens_in_it() {
        let brief = "Write Zen's life where the record has nothing: the year 2491.\n\n\
                     What happens: Zen walks the archive at the end of a maintenance cycle and \
                     finds one log entry nobody wrote.\n\nIt must agree with: The Awakening.";
        let happens = what_happens(brief);
        assert_eq!(
            happens.as_deref(),
            Some(
                "Zen walks the archive at the end of a maintenance cycle and finds one log \
                 entry nobody wrote."
            )
        );
        let why = short_by("layers/life/zen/2491 The Silence.md", 127, 250, happens);
        assert!(
            why.ends_with(
                "What happens in it, as your brief puts it: Zen walks the archive at the end of \
                 a maintenance cycle and finds one log entry nobody wrote. Write that, moment by \
                 moment."
            ),
            "{why}"
        );
        assert_eq!(what_happens("A brief with no such paragraph."), None);
    }

    fn vault() -> Hosted {
        let h = Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the vault must load");
        h.with(|w| {
            w.enter(
                "m1",
                "Perrin Vastwood",
                Where::new("vault-chronicle", "early-range"),
            )
            .unwrap();
            w.enter(
                "m2",
                "Orion Vance",
                Where::new("vault-chronicle", "early-range"),
            )
            .unwrap();
        });
        h
    }

    /// The vault, with somewhere to keep documents and two documents in it.
    ///
    /// A world without a root is a world with nothing to edit, so every test
    /// below that touches a document needs this rather than [`vault`].
    fn vault_with_documents(name: &str) -> (Hosted, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("npcd-work-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("layers/eras")).unwrap();
        std::fs::write(
            root.join("layers/eras/third.md"),
            "the third era\nburned in the spring\n",
        )
        .unwrap();
        std::fs::write(root.join("layers/eras/fourth.md"), "the fourth era\n").unwrap();

        // A mood and a personality, both carrying the comments that are the
        // reason the splice exists.
        std::fs::create_dir_all(root.join("moods")).unwrap();
        std::fs::write(
            root.join("moods/undone.yaml"),
            "id: undone\ncategory: mood\ndescription: Opened by what just happened.\n\n\
             # The felt register — its KV is loaded; a spike selects it.\ntemplate: |\n  \
             Something has been opened.\n",
        )
        .unwrap();
        std::fs::create_dir_all(root.join("personalities")).unwrap();
        std::fs::write(
            root.join("personalities/ash-the-drifter.yaml"),
            "# ash-the-drifter — identity definition.\n#\n# Biography is NOT here.\n\n\
             id: ash-the-drifter\nname: Ash\n\nportrait:\n  \
             image: portraits/ash-the-drifter.png\n  prompt: >-\n    a lean sun-darkened man\n",
        )
        .unwrap();

        let h = vault();
        h.set_bench_root(&root);
        (h, root)
    }

    /// **A mission that writes the record is reported done only once its
    /// document is committed by the one reporting** — whatever its steps say.
    /// A story was reported done with its write step struck "thwarted" on the
    /// character's word and no document anywhere. Committing it also signs the
    /// write step off and says what is next.
    #[test]
    fn a_mission_that_writes_the_record_is_done_when_its_document_is_committed() {
        use crate::engine::mission::{Mission, Origin, StepOutcome, Todo, Work};
        let (h, root) = vault_with_documents("gate");
        let path = "layers/stories/the-water-schedule.md";
        let mission = || {
            Mission::new(
                "Tell the story of the water schedule.",
                vec![
                    Todo::new(format!("write {path} and commit it")),
                    Todo::report("go back to the table and report it"),
                ],
                Origin::Lodged { by: "u_op".into() },
            )
            .with_work(Work {
                writes: path.into(),
                reads: vec![],
                min_words: 6,
                edit_optional: false,
                anew: false,
            })
        };
        h.with_sim(|s| s.missions.assign("m1", mission()));
        let report = || {
            perform(
                &h,
                "m1",
                &act(
                    "report_done",
                    json!({"account": "I wrote the story of the water schedule."}),
                ),
            )
        };

        // Struck on its word, with nothing on the record: still refused.
        h.with_sim(|s| {
            s.missions.check_off(
                "m1",
                &format!("write {path} and commit it"),
                StepOutcome::Thwarted,
            )
        });
        match report() {
            Outcome::Refused(why) => assert!(why.contains("is not committed"), "{why}"),
            other => panic!("reported done with nothing written: {other:?}"),
        }

        // Another document is not this mission's to write.
        match perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path": "layers/eras/third.md", "content": "new\n"}),
            ),
        ) {
            Outcome::Refused(why) => assert!(
                why.starts_with(&format!("Your mission writes {path} and nothing else.")),
                "{why}"
            ),
            other => panic!("a mission wrote outside its document: {other:?}"),
        }
        // Its own document under other capitals is another document.
        let shouted = path.replace(".md", ".Md");
        assert!(matches!(
            perform(
                &h,
                "m1",
                &act("file_write", json!({"path": shouted, "content": "x\n"})),
            ),
            Outcome::Refused(_)
        ));

        // Committed short of the floor: refused while it can still be
        // finished.
        h.with_sim(|s| s.missions.assign("m1", mission()));
        assert!(perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path": path, "content": "# The Water Schedule\n"})
            ),
        )
        .happened());
        match perform(&h, "m1", &act("bench_commit", json!({"why": "a start"}))) {
            Outcome::Refused(why) => assert!(why.contains("is 4 of the 6 words"), "{why}"),
            other => panic!("a sketch was committed: {other:?}"),
        }
        assert!(!root.join(path).is_file(), "nothing reached the disk");

        // Written whole and committed: the commit says the work is done, and
        // the report is taken.
        assert!(perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path": path, "content": "# The Water Schedule\n\nThe clerks finished it.\n"})
            ),
        )
        .happened());
        let committed = perform(
            &h,
            "m1",
            &act(
                "bench_commit",
                json!({"why": "the story of the water schedule"}),
            ),
        );
        assert!(
            committed
                .line()
                .unwrap()
                .contains("last step of your mission"),
            "{committed:?}"
        );
        assert!(root.join(path).is_file());
        // **Finished work cannot be reported stuck.** It used to be taken, and
        // a stuck report failed the operation and threw the draft away.
        match perform(
            &h,
            "m1",
            &act(
                "report_stuck",
                json!({"why": "the mission is done, but I cannot report it"}),
            ),
        ) {
            Outcome::Refused(why) => {
                assert!(why.starts_with("Nothing here is stuck"), "{why}");
                assert!(
                    why.contains(&format!("{path} is written and committed")),
                    "{why}"
                );
                assert!(why.contains("`report_done`"), "{why}");
            }
            other => panic!("finished work was reported stuck: {other:?}"),
        }
        assert!(
            report().happened(),
            "a committed document is a mission done"
        );
    }

    /// **An operation's draft is held to the quality gate at its report; its
    /// review goes to somebody else, who may reject it, and a rejected draft
    /// leaves the record.**
    #[test]
    fn an_operation_is_gated_reviewed_by_another_and_rejected_out_of_the_record() {
        use crate::engine::mission::bank::Facts;
        use crate::engine::mission::{Mission, Origin, Stage, Todo, Work};
        use crate::engine::mission_gen::corpus::Corpus;
        use crate::engine::mission_gen::reading::{review_mission, Verdict};
        use crate::sim::operations::Phase;
        let (h, root) = vault_with_documents("operation");
        let path = "layers/stories/the-water-schedule.md";
        let draft = Mission::new(
            "Tell the story of the water schedule.",
            vec![
                Todo::new(format!("write {path} and commit it")),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Generated {
                generator: "untold".into(),
                target: "era:layers/eras/third.md".into(),
                operation: 0,
                stage: Stage::Draft,
            },
        )
        .with_work(Work {
            writes: path.into(),
            reads: vec![],
            min_words: 6,
            edit_optional: false,
            anew: false,
        });
        let id = h.with_sim(|s| {
            let id = s.missions.launch(draft, 1, "the water schedule", None);
            s.missions.collect("m1", &Facts::default());
            id
        });
        // Four hundred words, no phrase said twice, and no heading.
        let prose: String = (0..50)
            .map(|i| format!("Clerk{i} tallied cistern{i} beside wall{i} before dawn{i} broke."))
            .collect::<Vec<_>>()
            .chunks(5)
            .map(|c| c.join(" "))
            .collect::<Vec<_>>()
            .join("\n\n");
        let write = |text: &str| {
            assert!(perform(
                &h,
                "m1",
                &act("file_write", json!({"path": path, "content": text}))
            )
            .happened());
            assert!(
                perform(&h, "m1", &act("bench_commit", json!({"why": "the story"}))).happened()
            );
        };
        let report = |who: &str| {
            perform(
                &h,
                who,
                &act(
                    "report_done",
                    json!({"account": "I wrote the story of the water schedule."}),
                ),
            )
        };
        // A sentence said twice is the writer's to mend.
        write(&format!(
            "{prose}\n\nThe clerks went home before the count was done. The clerks went home \
             before the count was done."
        ));
        match report("m1") {
            Outcome::Refused(why) => {
                assert!(why.contains("is not ready to stand in the record"), "{why}");
                assert!(why.contains("is said twice"), "{why}");
                assert!(
                    !why.contains("heading"),
                    "the missing heading is the engine's to add: {why}"
                );
            }
            other => panic!("an ungated draft was reported: {other:?}"),
        }
        // Refused, the draft is work to do again: the mission's next step is
        // its writing, not the report, so nothing tells the writer it is done.
        let next = h.sim(|s| {
            s.missions
                .active("m1")
                .and_then(|m| m.next_step())
                .map(|t| t.text.clone())
        });
        assert!(
            next.as_deref().is_some_and(|t| t.starts_with("write ")),
            "{next:?}"
        );
        // Mended, and still without a heading: the engine gives it its title.
        write(&prose);
        assert!(report("m1").happened(), "the mended draft passes the gate");
        let stood = std::fs::read_to_string(root.join(path)).unwrap();
        assert!(stood.starts_with("# The Water Schedule\n\n"), "{stood}");

        // The review is set; the writer cannot draw it, the other Maker does.
        h.with_sim(|s| {
            let op = s.missions.operations().get(id).unwrap().clone();
            let corpus = Corpus::read(&root, "creators-vault");
            let review = review_mission(
                &op,
                "The table's verdict: sound.",
                Some(Verdict::Sound),
                &corpus,
                None,
            );
            s.missions
                .offer_review(id, review, "The table's verdict: sound.", true);
            let writer_draws = s.missions.collect("m1", &Facts::default()).operation();
            assert_eq!(writer_draws, None, "not its own draft");
            s.missions.cancel("m1");
            let reviewer_draws = s.missions.collect("m2", &Facts::default()).operation();
            assert_eq!(reviewer_draws, Some((id, Stage::Review)));
        });
        assert!(perform(&h, "m2", &act("file_read", json!({"path": path}))).happened());
        assert!(perform(
            &h,
            "m2",
            &act("file_read", json!({"path": "layers/eras/third.md"}))
        )
        .happened());
        assert!(matches!(
            perform(&h, "m2", &act("report_rejected", json!({"why": "bad"}))),
            Outcome::Refused(_)
        ));
        // A reason that points at nothing in the draft is refused.
        match perform(
            &h,
            "m2",
            &act(
                "report_rejected",
                json!({"why": "It counts the cisterns fifty times and nothing happens in it."}),
            ),
        ) {
            Outcome::Refused(why) => assert!(why.contains("does not quote"), "{why}"),
            other => panic!("an unquoted rejection was taken: {other:?}"),
        }
        assert!(perform(
            &h,
            "m2",
            &act(
                "report_rejected",
                json!({"why": "\"Clerk7 tallied cistern7 beside wall7\" — and so on fifty times; \
                                nothing happens in it."})
            ),
        )
        .happened());
        assert!(
            !root.join(path).exists(),
            "the rejected draft left the record"
        );
        assert!(root
            .join("rejected/operation-iron-lantern/the-water-schedule.md")
            .is_file());
        let phase = h.sim(|s| s.missions.operations().get(id).unwrap().phase);
        assert_eq!(phase, Phase::Failed);
    }

    /// **A write is in memory and nowhere else until the commit.** The one
    /// behaviour the whole bench is built around, asserted against the disk
    /// rather than against a report.
    #[test]
    fn a_written_document_reaches_the_disk_only_at_the_commit() {
        let (h, root) = vault_with_documents("commit");
        let wrote = perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/fifth.md","content":"the fifth era\n"}),
            ),
        );
        assert!(wrote.happened(), "{wrote:?}");
        assert!(
            !root.join("layers/eras/fifth.md").exists(),
            "the disk moved early"
        );

        let out = perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"named the fifth"})),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/fifth.md")).unwrap(),
            "the fifth era\n"
        );
    }

    /// A body reads its own uncommitted work; nobody else can see it at all.
    #[test]
    fn a_read_shows_your_own_changes_and_nobody_elses() {
        let (h, _) = vault_with_documents("read");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"rewritten\n"}),
            ),
        );
        let mine = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(mine.line().unwrap().contains("rewritten"), "{mine:?}");

        let theirs = perform(
            &h,
            "m2",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(
            theirs.line().unwrap().contains("burned in the spring"),
            "a working set leaked: {theirs:?}"
        );
    }

    /// **The subject of a file act is its `path`, never its text.** `story_draft`
    /// once wrote a draft into a record item named after the draft's own prose
    /// because a generic subject read the text first. The same shape is here —
    /// `file_write` carries a path and a body of text — so it is pinned.
    #[test]
    fn the_document_written_is_the_one_the_path_names() {
        let (h, root) = vault_with_documents("subject");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/sixth.md","content":"a night at the gate"}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert!(root.join("layers/eras/sixth.md").exists());
        assert!(
            !root
                .join("layers/eras")
                .join("a night at the gate")
                .exists(),
            "wrote into the prose"
        );
    }

    /// **A document is written exactly as it was given.** A trailing newline is
    /// part of the file, and the generic argument reader trims — which would
    /// make every document the bench writes quietly differ from what the
    /// character said, and report success doing it.
    #[test]
    fn a_document_is_written_exactly_as_it_was_given() {
        let (h, root) = vault_with_documents("verbatim");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/fifth.md","content":"a line\n\nand another\n"}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/fifth.md")).unwrap(),
            "a line\n\nand another\n"
        );
    }

    /// The same reason from the other side: an edit has to be able to match
    /// leading whitespace, or it cannot touch a `.yaml` document at all — the
    /// indentation *is* the structure there.
    #[test]
    fn an_edit_matches_the_indentation_it_was_given() {
        let (h, root) = vault_with_documents("indent");
        std::fs::write(root.join("layers/eras/third.md"), "a:\n  b: 1\n  c: 1\n").unwrap();
        let out = perform(
            &h,
            "m1",
            &act(
                "file_edit",
                json!({"path":"layers/eras/third.md","old_str":"  b: 1","new_str":"  b: 2"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/third.md")).unwrap(),
            "a:\n  b: 2\n  c: 1\n"
        );
    }

    /// A read is capped and says so, so a long document cannot swallow a
    /// character's context window in one act.
    #[test]
    fn a_read_is_capped_and_carries_its_own_continuation() {
        let (h, root) = vault_with_documents("cap");
        let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
        std::fs::write(root.join("layers/eras/long.md"), &body).unwrap();

        let first = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/long.md"})),
        );
        let shown = first.line().unwrap();
        assert!(shown.contains("(lines 1-200 of 900)"), "{shown}");
        assert!(
            !shown.contains("line 201"),
            "the cap did not hold at the act"
        );

        let next = perform(
            &h,
            "m1",
            &act(
                "file_read",
                json!({"path":"layers/eras/long.md","start_line":"201"}),
            ),
        );
        assert!(
            next.line().unwrap().contains("(lines 201-400 of 900)"),
            "{next:?}"
        );
    }

    /// `start_line` is a string because the grammar cannot bound an integer, so
    /// the act parses it — and something unparseable means the start rather
    /// than a refusal that spends the turn on arithmetic.
    #[test]
    fn an_unparseable_start_line_reads_from_the_top() {
        let (h, _) = vault_with_documents("start-junk");
        for junk in ["the top", "", "-4", "one"] {
            let out = perform(
                &h,
                "m1",
                &act(
                    "file_read",
                    json!({"path":"layers/eras/third.md","start_line":junk}),
                ),
            );
            assert!(out.happened(), "{junk:?}: {out:?}");
            assert!(out.line().unwrap().contains("the third era"), "{junk:?}");
        }
    }

    /// **Every file act takes its subject from `path`.** The subject is picked
    /// by scanning a list of argument names in order, and the new parameters
    /// sit next to entries in that list — `start_line` beside `for`, `content`
    /// beside `what`. If one of them ever wins, the act operates on the prose
    /// instead of the document, reports success, and nobody finds out until the
    /// record is wrong. That is exactly what `story_draft` did.
    #[test]
    fn the_subject_of_every_file_act_is_its_path() {
        for args in [
            json!({"path":"layers/eras/third.md"}),
            json!({"path":"layers/eras/third.md","start_line":"12"}),
            json!({"path":"layers/eras/third.md","content":"what the document says"}),
            json!({"path":"layers/eras/third.md","old_str":"a","new_str":"b"}),
        ] {
            let map = args.as_object().unwrap().clone();
            assert_eq!(
                subject(&map).as_deref(),
                Some("layers/eras/third.md"),
                "the subject was taken from the wrong argument in {args}"
            );
        }
    }

    #[test]
    fn an_edit_leaves_everything_it_did_not_name_alone() {
        let (h, root) = vault_with_documents("edit");
        let out = perform(
            &h,
            "m1",
            &act(
                "file_edit",
                json!({"path":"layers/eras/third.md","old_str":"spring","new_str":"autumn"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(
            &h,
            "m1",
            &act(
                "bench_commit",
                json!({"why":"dated it against its neighbours"}),
            ),
        );
        assert_eq!(
            std::fs::read_to_string(root.join("layers/eras/third.md")).unwrap(),
            "the third era\nburned in the autumn\n"
        );
    }

    #[test]
    fn an_ambiguous_edit_is_refused_rather_than_applied_to_the_wrong_one() {
        let (h, root) = vault_with_documents("ambiguous");
        std::fs::write(root.join("layers/eras/third.md"), "a fire\nand a fire\n").unwrap();
        let out = perform(
            &h,
            "m1",
            &act(
                "file_edit",
                json!({"path":"layers/eras/third.md","old_str":"a fire","new_str":"a flood"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains('2'), "{out:?}");
    }

    #[test]
    fn an_edit_missing_half_of_itself_says_which_half() {
        let (h, _) = vault_with_documents("half");
        let out = perform(
            &h,
            "m1",
            &act(
                "file_edit",
                json!({"path":"layers/eras/third.md","old_str":"x"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("both"), "{out:?}");
    }

    #[test]
    fn a_listing_shows_the_disk_and_your_own_new_documents() {
        let (h, _) = vault_with_documents("list");
        let before = perform(&h, "m1", &act("file_list", json!({"path":"layers/eras"})));
        assert!(before.line().unwrap().contains("third.md"), "{before:?}");

        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/fifth.md","content":"…"}),
            ),
        );
        let after = perform(&h, "m1", &act("file_list", json!({"path":"layers/eras"})));
        assert!(after.line().unwrap().contains("fifth.md"), "{after:?}");
        // …and not to anybody else, because it is not committed.
        let theirs = perform(&h, "m2", &act("file_list", json!({"path":"layers/eras"})));
        assert!(!theirs.line().unwrap().contains("fifth.md"), "{theirs:?}");
    }

    #[test]
    fn a_deleted_document_goes_from_the_disk_at_the_commit() {
        let (h, root) = vault_with_documents("delete");
        let out = perform(
            &h,
            "m1",
            &act("file_delete", json!({"path":"layers/eras/fourth.md"})),
        );
        assert!(out.happened(), "{out:?}");
        assert!(
            root.join("layers/eras/fourth.md").exists(),
            "gone before the commit"
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"never canon"})));
        assert!(!root.join("layers/eras/fourth.md").exists());
    }

    /// The one act somebody else's work can refuse — now over a real document,
    /// and the refusal still names the other party.
    #[test]
    fn two_makers_on_one_document_collide_at_the_commit() {
        let (h, _) = vault_with_documents("collide");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"mine\n"}),
            ),
        );
        perform(
            &h,
            "m2",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"mine too\n"}),
            ),
        );

        assert!(perform(&h, "m1", &act("bench_commit", json!({"why":"first"}))).happened());
        let out = perform(&h, "m2", &act("bench_commit", json!({"why":"second"})));
        assert!(!out.happened());
        assert!(
            out.line().unwrap().contains("m1"),
            "no other party named: {out:?}"
        );
        assert!(
            out.line().unwrap().contains("talk to"),
            "a collision read as a retryable error: {out:?}"
        );
        // The refused body still has all of its work.
        let still = perform(&h, "m2", &act("bench_diff", json!({})));
        assert!(
            still.line().unwrap().contains("layers/eras/third.md"),
            "{still:?}"
        );
    }

    #[test]
    fn setting_work_aside_and_picking_it_up_again_survives_the_round_trip() {
        let (h, _) = vault_with_documents("stash");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"half\n"}),
            ),
        );
        assert!(perform(&h, "m1", &act("bench_stash", json!({}))).happened());
        let gone = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(
            gone.line().unwrap().contains("burned in the spring"),
            "{gone:?}"
        );

        assert!(perform(&h, "m1", &act("bench_stash_pop", json!({}))).happened());
        let back = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(back.line().unwrap().contains("half"), "{back:?}");
    }

    #[test]
    fn throwing_the_work_away_puts_the_document_back() {
        let (h, _) = vault_with_documents("restore");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"wrong\n"}),
            ),
        );
        assert!(perform(&h, "m1", &act("bench_restore", json!({}))).happened());
        let back = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(
            back.line().unwrap().contains("burned in the spring"),
            "{back:?}"
        );
    }

    #[test]
    fn status_and_diff_name_the_documents_that_are_changed() {
        let (h, _) = vault_with_documents("status");
        let quiet = perform(&h, "m1", &act("bench_status", json!({})));
        assert!(quiet.line().unwrap().contains("Nothing open"), "{quiet:?}");

        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"changed\n"}),
            ),
        );
        let diff = perform(&h, "m1", &act("bench_diff", json!({})));
        assert!(
            diff.line()
                .unwrap()
                .contains("layers/eras/third.md (changed)"),
            "{diff:?}"
        );

        perform(&h, "m1", &act("bench_stage", json!({})));
        let staged = perform(&h, "m1", &act("bench_status", json!({})));
        assert!(staged.line().unwrap().contains("Offered"), "{staged:?}");
    }

    /// The paths are written by a language model, so the guard has to hold at
    /// the act, not only in the store beneath it.
    #[test]
    fn a_path_that_leaves_the_world_is_refused_at_the_act() {
        let (h, root) = vault_with_documents("escape");
        for bad in [
            "../stolen.md",
            "layers/eras/../../stolen.md",
            "c:/windows/x.md",
            "layers/eras/x.exe",
        ] {
            let out = perform(
                &h,
                "m1",
                &act("file_write", json!({"path":bad,"content":"owned"})),
            );
            assert!(!out.happened(), "{bad} was written");
            assert!(!perform(&h, "m1", &act("file_read", json!({"path":bad}))).happened());
        }
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert!(!root.parent().unwrap().join("stolen.md").exists());
    }

    /// A world with nothing to edit refuses rather than inventing somewhere.
    #[test]
    fn a_world_with_no_documents_refuses_every_file_act() {
        let h = vault();
        for a in [
            act("file_read", json!({"path":"layers/eras/third.md"})),
            act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"x"}),
            ),
            act("file_list", json!({"path":"layers/eras"})),
        ] {
            let out = perform(&h, "m1", &a);
            assert!(!out.happened(), "{:?} happened with no documents", a.tool);
        }
    }

    /// Committing a document is not the same as being the one who holds it, and
    /// the history says who actually did it.
    #[test]
    fn blame_on_a_document_names_whoever_committed_it() {
        let (h, _) = vault_with_documents("blame");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"mine\n"}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        let out = perform(
            &h,
            "m2",
            &act("bench_blame", json!({"what":"layers/eras/third.md"})),
        );
        assert!(out.line().unwrap().contains("m1"), "{out:?}");
    }

    // ── the join: stations write documents ──────────────────────────────────
    //
    // The whole point of `index_canon` and `settle_path`. Before these, every
    // act at a chronicle terminal reported success against an in-memory fixture
    // and changed nothing that survived a restart.

    /// **An entry written at a chronicle terminal reaches the disk.** The one
    /// test that says the stations are real.
    #[test]
    fn an_entry_written_into_an_era_lands_in_the_era_document() {
        let (h, root) = vault_with_documents("era-entry");
        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"}))
        )
        .happened());
        let out = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"The redoubt fell in the spring."}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(
            !root.join("layers/eras/the-third-era.md").exists(),
            "the disk moved before the commit"
        );

        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"dated the fall"}))
        )
        .happened());
        let written = std::fs::read_to_string(root.join("layers/eras/the-third-era.md")).unwrap();
        assert!(
            written.contains("The redoubt fell in the spring."),
            "{written}"
        );
    }

    /// A second entry is added to the first, not written over it.
    #[test]
    fn a_second_entry_is_added_rather_than_replacing_the_first() {
        let (h, root) = vault_with_documents("era-append");
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"})),
        );
        perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"First."}),
            ),
        );
        perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"Second."}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"two entries"})));

        let written = std::fs::read_to_string(root.join("layers/eras/the-third-era.md")).unwrap();
        assert!(written.contains("First."), "{written}");
        assert!(
            written.contains("Second."),
            "the second entry replaced the first: {written}"
        );
        assert!(
            written.find("First.") < written.find("Second."),
            "the record came back out of order: {written}"
        );
    }

    /// A document already on the disk *is* the thing the world names, rather
    /// than a twin standing beside it.
    #[test]
    fn an_era_document_on_disk_becomes_the_thing_the_world_already_names() {
        let (h, root) = vault_with_documents("era-index");
        std::fs::write(
            root.join("layers/eras/burning.md"),
            "# the fourth era\n\nQuiet, and then not.\n",
        )
        .unwrap();
        h.set_bench_root(&root);

        h.sim(|s| {
            let names: Vec<&str> = s.record.iter().map(|i| i.name.as_str()).collect();
            assert_eq!(
                names.iter().filter(|n| **n == "the fourth era").count(),
                1,
                "the document made a twin: {names:?}"
            );
            assert_eq!(
                s.record.path_of("the fourth era").as_deref(),
                Some("layers/eras/burning.md"),
                "the era did not take its document"
            );
        });
    }

    /// A story is a document too, and lands in its own layer.
    #[test]
    fn a_draft_written_into_a_silence_lands_in_the_stories_layer() {
        let (h, root) = vault_with_documents("story-draft");
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third silence"})),
        );
        let out = perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"A night at the gate."}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"filled the longest silence"})),
        );

        let p = root.join("layers/stories/the-third-silence.md");
        assert!(p.exists(), "the draft did not reach the stories layer");
        assert!(std::fs::read_to_string(p)
            .unwrap()
            .contains("A night at the gate."));
    }

    /// **A filed document is not written over casually.** It refuses until the
    /// writer has opened it, which is the bench loop said by the store.
    #[test]
    fn a_filed_era_refuses_a_write_until_it_is_opened() {
        let (h, _root) = vault_with_documents("era-filed");
        let closed = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the fourth era","what":"…"}),
            ),
        );
        assert!(!closed.happened());
        assert!(
            closed.line().unwrap().contains("Open it first"),
            "{closed:?}"
        );

        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the fourth era"}))
        )
        .happened());
        let open = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the fourth era","what":"And then not."}),
            ),
        );
        assert!(open.happened(), "{open:?}");
    }

    /// Two Makers on one era collide at the commit, over a real file.
    #[test]
    fn two_makers_writing_one_era_collide_at_the_commit() {
        let (h, _) = vault_with_documents("era-collide");
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"})),
        );
        perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"Mine."}),
            ),
        );
        // m1 is holding it, so m2 is refused by custody before it reaches the
        // document at all — the record's own rule, still doing its job.
        let blocked = perform(
            &h,
            "m2",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"Mine too."}),
            ),
        );
        assert!(!blocked.happened());
        assert!(blocked.line().unwrap().contains("m1"), "{blocked:?}");
    }

    /// A thing that is a judgement rather than a document still lives in the
    /// record — the join must not sweep everything onto the disk.
    #[test]
    fn a_judgement_stays_in_the_record_and_gets_no_document() {
        let (h, root) = vault_with_documents("judgement");
        let out = perform(
            &h,
            "m1",
            &act(
                "record_appraise",
                json!({"what":"the third era","verdict":"sound"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| assert!(s.record.path_of("the western intake").is_none()));
        assert!(
            !root
                .join("layers/eras")
                .join("the-western-intake.md")
                .exists(),
            "an accession was given a document"
        );
    }

    // ── character and place authoring reach the disk ─────────────────────────
    //
    // D.6 #1: `character_write_*` and `place_write_*` used to fall to an in-RAM
    // `Record.body` that a restart threw away. They write the mind folder now,
    // through the bench working copy, and survive a reload.

    /// The vault, with a mind folder holding a courier's sheet — everything the
    /// character and place authoring tests write into or read back.
    fn vault_authoring(name: &str) -> (Hosted, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("npcd-author-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("personalities")).unwrap();
        std::fs::write(
            root.join("personalities/the-courier.yaml"),
            "# the courier — identity definition.\nid: the-courier\nname: The Courier\n",
        )
        .unwrap();
        // The record already names the courier and the eastern flats (the vault
        // seed); the world/ layers are made on the first commit into them.
        let h = vault();
        h.set_bench_root(&root);
        (h, root)
    }

    /// **Who a character is reaches the sheet on disk, and not before the
    /// commit.** The `anchor` field, written the way a portrait's words are.
    #[test]
    fn character_identity_is_written_to_the_sheet_and_survives_a_commit() {
        let (h, root) = vault_authoring("identity");
        let out = perform(
            &h,
            "m1",
            &act(
                "character_write_identity",
                json!({"of":"the courier","what":"steady so long that being doubted is the thing they cannot take"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(
            !std::fs::read_to_string(root.join("personalities/the-courier.yaml"))
                .unwrap()
                .contains("cannot take"),
            "the sheet moved before the commit"
        );

        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"cast the courier"}))
        )
        .happened());
        let sheet = std::fs::read_to_string(root.join("personalities/the-courier.yaml")).unwrap();
        let doc: Value = serde_yaml::from_str(&sheet).unwrap();
        assert_eq!(
            doc["anchor"],
            json!("steady so long that being doubted is the thing they cannot take")
        );
        // The splice kept the sheet's own header and its other fields.
        assert!(
            sheet.contains("# the courier — identity definition."),
            "the header was lost: {sheet}"
        );
        assert_eq!(
            doc["name"],
            json!("The Courier"),
            "an untouched field moved"
        );
    }

    /// **What a character wants becomes a new top-level field on the sheet**,
    /// beside the anchor, and the anchor written first is still there.
    #[test]
    fn character_wants_becomes_a_new_field_beside_the_anchor() {
        let (h, root) = vault_authoring("wants");
        perform(
            &h,
            "m1",
            &act(
                "character_write_identity",
                json!({"of":"the courier","what":"reliable to a fault"}),
            ),
        );
        perform(
            &h,
            "m1",
            &act(
                "character_write_wants",
                json!({"of":"the courier","what":"to be trusted again by the one house that stopped"}),
            ),
        );
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"gave the courier a reason"})),
        );

        let sheet = std::fs::read_to_string(root.join("personalities/the-courier.yaml")).unwrap();
        let doc: Value = serde_yaml::from_str(&sheet).unwrap();
        assert_eq!(doc["anchor"], json!("reliable to a fault"));
        assert_eq!(
            doc["wants"],
            json!("to be trusted again by the one house that stopped")
        );
    }

    /// **A memory lands as a document in the character's own memory layer**, not
    /// on the sheet — and not before the commit.
    #[test]
    fn a_memory_lands_in_the_characters_memory_layer() {
        let (h, root) = vault_authoring("memory");
        let out = perform(
            &h,
            "m1",
            &act(
                "character_write_memories",
                json!({"of":"the courier","what":"the afternoon they waited four hours at a gate that was never going to open"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        let p = root.join("layers/memory/the-courier/memories.md");
        assert!(!p.exists(), "the memory reached the disk before the commit");

        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"remembered the gate"})),
        );
        let written = std::fs::read_to_string(&p).unwrap();
        assert!(
            written.contains("waited four hours at a gate that was never going to open"),
            "{written}"
        );
    }

    /// **A second memory is added to the first**, the way an era's entries are.
    #[test]
    fn a_second_memory_is_added_rather_than_replacing_the_first() {
        let (h, root) = vault_authoring("memory-append");
        perform(
            &h,
            "m1",
            &act(
                "character_write_memories",
                json!({"of":"the courier","what":"First."}),
            ),
        );
        perform(
            &h,
            "m1",
            &act(
                "character_write_memories",
                json!({"of":"the courier","what":"Second."}),
            ),
        );
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"two afternoons"})),
        );
        let written =
            std::fs::read_to_string(root.join("layers/memory/the-courier/memories.md")).unwrap();
        assert!(written.contains("First."), "{written}");
        assert!(
            written.find("First.") < written.find("Second."),
            "the memories came back out of order: {written}"
        );
    }

    /// **A place entry lands in the locations layer and survives a reload.** A
    /// fresh, empty record reads the committed disk and the place is part of the
    /// record again — the durability this whole change exists for.
    #[test]
    fn a_place_entry_lands_in_the_locations_layer_and_survives_a_reload() {
        let (h, root) = vault_authoring("place-entry");
        let out = perform(
            &h,
            "m1",
            &act(
                "place_write_entry",
                json!({"of":"the eastern flats","what":"ground fused smooth, no cover anywhere on it, and a wind that does not stop"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        let p = root.join("layers/world/locations/the-eastern-flats.md");
        assert!(!p.exists(), "the disk moved before the commit");

        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"surveyed the flats"}))
        )
        .happened());
        assert!(
            std::fs::read_to_string(&p)
                .unwrap()
                .contains("a wind that does not stop"),
            "the entry did not reach the locations layer"
        );

        // A fresh process, an empty record: it reads the committed disk and the
        // place is filed, at its path, again.
        let mut fresh = crate::sim::record::Record::new();
        fresh.index_canon(&root);
        assert_eq!(
            fresh.path_of("the eastern flats").as_deref(),
            Some("layers/world/locations/the-eastern-flats.md")
        );
        assert_eq!(
            fresh.by_name("the eastern flats").unwrap().state,
            State::Filed
        );
        assert_eq!(
            fresh.by_name("the eastern flats").unwrap().kind,
            Kind::Place
        );
    }

    /// **A place's local history lands in geography, not on its entry**, and
    /// survives a reload the same way.
    #[test]
    fn a_places_local_history_lands_in_geography_and_survives_a_reload() {
        let (h, root) = vault_authoring("place-history");
        let out = perform(
            &h,
            "m1",
            &act(
                "place_write_local_history",
                json!({"of":"the eastern flats","what":"why the road stops here — what came across it, and the year it stopped being worth rebuilding"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"gave the flats a reason"}))
        )
        .happened());
        let p = root.join("layers/world/geography/the-eastern-flats.md");
        assert!(
            std::fs::read_to_string(&p)
                .unwrap()
                .contains("the year it stopped being worth rebuilding"),
            "the history did not reach the geography layer"
        );

        // A place known only by its history is part of the record again, but its
        // history is NOT its entry path: `Item.path` names the entry, which is
        // still unwritten here, and the history is found name-derived instead.
        let mut fresh = crate::sim::record::Record::new();
        fresh.index_canon(&root);
        assert!(
            fresh.path_of("the eastern flats").is_none(),
            "the geography history was adopted as the entry path"
        );
        assert_eq!(
            fresh.history_path("the eastern flats").as_deref(),
            Some("layers/world/geography/the-eastern-flats.md")
        );
        assert_eq!(
            fresh.by_name("the eastern flats").unwrap().state,
            State::Filed
        );
    }

    /// **A place with BOTH documents writes its entries into the entry file, not
    /// its history.** After both are on disk and the record has re-indexed them,
    /// `Item.path` names the locations entry — so `place_write_entry` appends
    /// there, and the geography history it also has is left untouched. This is
    /// the defect the geography-pass fix closes: an entry that leaked into the
    /// history because indexing re-pointed `Item.path` at geography.
    #[test]
    fn a_place_with_both_documents_writes_entries_to_the_entry_not_the_history() {
        let (h, root) = vault_authoring("place-both");
        std::fs::create_dir_all(root.join("layers/world/locations")).unwrap();
        std::fs::create_dir_all(root.join("layers/world/geography")).unwrap();
        std::fs::write(
            root.join("layers/world/locations/the-eastern-flats.md"),
            "# the eastern flats\n\nGround fused smooth.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("layers/world/geography/the-eastern-flats.md"),
            "# the eastern flats\n\nWhy the road stops here.\n",
        )
        .unwrap();
        // Re-index off the disk so the record adopts both documents.
        h.set_bench_root(&root);
        h.sim(|s| {
            assert_eq!(
                s.record.path_of("the eastern flats").as_deref(),
                Some("layers/world/locations/the-eastern-flats.md"),
                "the entry path was overwritten by the geography history"
            );
        });

        let out = perform(
            &h,
            "m1",
            &act(
                "place_write_entry",
                json!({"of":"the eastern flats","what":"a new survey line about the wind"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"surveyed the flats again"}))
        )
        .happened());

        let entry =
            std::fs::read_to_string(root.join("layers/world/locations/the-eastern-flats.md"))
                .unwrap();
        assert!(
            entry.contains("a new survey line about the wind"),
            "the entry prose did not reach the entry file: {entry}"
        );
        let history =
            std::fs::read_to_string(root.join("layers/world/geography/the-eastern-flats.md"))
                .unwrap();
        assert!(
            !history.contains("a new survey line about the wind"),
            "the entry prose leaked into the history file: {history}"
        );
    }

    /// **The mind-less daemon still works.** With no mind folder, every one of
    /// the five verbs keeps its in-RAM draft rather than panicking — the shipped
    /// behaviour, preserved for the world (Battle Cities) that has no disk.
    #[test]
    fn character_and_place_authoring_without_a_mind_folder_stays_in_ram() {
        let h = vault();
        for a in [
            act(
                "character_write_identity",
                json!({"of":"the courier","what":"steady"}),
            ),
            act(
                "character_write_wants",
                json!({"of":"the courier","what":"to be trusted"}),
            ),
            act(
                "character_write_memories",
                json!({"of":"the courier","what":"an afternoon at a gate"}),
            ),
            act(
                "place_write_entry",
                json!({"of":"the eastern flats","what":"smooth ground and a wind"}),
            ),
            act(
                "place_write_local_history",
                json!({"of":"the eastern flats","what":"why the road stops"}),
            ),
        ] {
            let out = perform(&h, "m1", &a);
            assert!(
                out.happened(),
                "{} did not survive the no-root path: {out:?}",
                a.tool
            );
        }
        h.sim(|s| {
            assert!(
                s.record
                    .by_name("the courier")
                    .unwrap()
                    .body
                    .contains("steady"),
                "the courier's draft was lost"
            );
            assert!(
                s.record
                    .by_name("the eastern flats")
                    .unwrap()
                    .body
                    .contains("smooth ground"),
                "the flats' draft was lost"
            );
        });
    }

    // ── the craft libraries ─────────────────────────────────────────────────

    #[test]
    fn a_mood_can_be_read_and_changed_and_lands_on_the_disk() {
        let (h, root) = vault_with_documents("mood");
        let read = perform(
            &h,
            "m1",
            &act("library_read", json!({"kind":"mood","id":"undone"})),
        );
        assert!(
            read.line().unwrap().contains("Something has been opened"),
            "{read:?}"
        );

        let out = perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({
                    "kind":"mood","id":"undone","field":"description",
                    "text":"So thoroughly opened that the whole interior has rearranged."
                }),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(
            &h,
            "m1",
            &act(
                "bench_commit",
                json!({"why":"it read as two things at once"}),
            ),
        );

        let on_disk = std::fs::read_to_string(root.join("moods/undone.yaml")).unwrap();
        assert!(
            on_disk.contains("whole interior has rearranged"),
            "{on_disk}"
        );
    }

    /// **The comments survive.** This is the entire reason the write goes
    /// through the splice: a `serde_yaml` round trip still produces a document
    /// that loads perfectly, with the half a person wrote silently gone.
    #[test]
    fn changing_a_field_keeps_everything_written_around_it() {
        let (h, root) = vault_with_documents("mood-comments");
        perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({
                    "kind":"mood","id":"undone","field":"description","text":"Rearranged."
                }),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));

        let on_disk = std::fs::read_to_string(root.join("moods/undone.yaml")).unwrap();
        assert!(
            on_disk.contains("# The felt register — its KV is loaded"),
            "the comment was lost: {on_disk}"
        );
        assert!(
            on_disk.contains("category: mood"),
            "an untouched field moved: {on_disk}"
        );
        assert!(
            on_disk.contains("Something has been opened"),
            "the template was rewritten"
        );
    }

    #[test]
    fn a_library_that_does_not_exist_is_refused_by_name() {
        let (h, _) = vault_with_documents("mood-bad-kind");
        let out = perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({"kind":"weather","id":"x","field":"template","text":"y"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("mood"), "{out:?}");
    }

    /// `id` and `category` are how everything else finds this piece, so they
    /// are not a Maker's to change from inside the world.
    #[test]
    fn only_the_fields_that_are_a_makers_to_change_are_accepted() {
        let (h, _) = vault_with_documents("mood-bad-field");
        for field in ["id", "category", "anything"] {
            let out = perform(
                &h,
                "m1",
                &act(
                    "library_write",
                    json!({
                        "kind":"mood","id":"undone","field":field,"text":"x"
                    }),
                ),
            );
            assert!(!out.happened(), "{field} was accepted");
        }
        let ok = perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({
                    "kind":"mood","id":"undone","field":"template","text":"Steady."
                }),
            ),
        );
        assert!(ok.happened(), "{ok:?}");
    }

    // ── likenesses ──────────────────────────────────────────────────────────

    #[test]
    fn the_words_a_likeness_is_drawn_from_can_be_read_and_changed() {
        let (h, root) = vault_with_documents("portrait");
        let read = perform(
            &h,
            "m1",
            &act("portrait_prompt_read", json!({"of":"ash-the-drifter"})),
        );
        assert!(
            read.line().unwrap().contains("lean sun-darkened man"),
            "{read:?}"
        );

        let out = perform(
            &h,
            "m1",
            &act(
                "portrait_prompt_edit",
                json!({
                    "of":"ash-the-drifter",
                    "carrying":"the same man after the winter, quieter and harder to read"
                }),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"the winter changed him"})),
        );

        let on_disk =
            std::fs::read_to_string(root.join("personalities/ash-the-drifter.yaml")).unwrap();
        assert!(on_disk.contains("after the winter"), "{on_disk}");
        // The picture it names is untouched — a redraw follows the words.
        assert!(
            on_disk.contains("image: portraits/ash-the-drifter.png"),
            "{on_disk}"
        );
        assert!(
            on_disk.contains("# Biography is NOT here."),
            "the header was lost"
        );
    }

    #[test]
    fn drawing_somebody_the_world_has_never_heard_of_is_refused() {
        let (h, _) = vault_with_documents("portrait-absent");
        let out = perform(
            &h,
            "m1",
            &act("portrait_prompt_read", json!({"of":"nobody at all"})),
        );
        assert!(!out.happened(), "{out:?}");
    }

    /// **An act that reports work leaves a trace of it.**
    ///
    /// These four returned prose and changed nothing: the character did the
    /// reading, said so, and the next character to pick the piece up found no
    /// sign of it — so the work was done again, and again, with nothing
    /// accumulating. They read as working from every angle except the only one
    /// that counts.
    #[test]
    fn the_craft_readings_leave_a_finding_behind_them() {
        let h = vault();
        let before = h.sim(|s| s.ledger.verdicts_on("the third era").len());

        assert!(perform(
            &h,
            "m1",
            &act("structure_lay_out_scenes", json!({"what":"the third era"}))
        )
        .happened());
        assert!(perform(
            &h,
            "m1",
            &act(
                "structure_test_the_want",
                json!({"what":"the third era","scene":"the gate"})
            ),
        )
        .happened());
        assert!(perform(
            &h,
            "m1",
            &act("structure_find_the_slack", json!({"what":"the third era"}))
        )
        .happened());

        let after: Vec<String> = h.sim(|s| {
            s.ledger
                .verdicts_on("the third era")
                .iter()
                .map(|v| v.judgement.clone())
                .collect()
        });
        assert_eq!(
            after.len(),
            before + 3,
            "a reading was reported and left nothing: {after:?}"
        );
        assert!(after.iter().any(|j| j.contains("scenes")), "{after:?}");
        assert!(after.iter().any(|j| j.contains("want")), "{after:?}");
        assert!(after.iter().any(|j| j.contains("slack")), "{after:?}");
    }

    /// Tidying an index leaves the index changed, rather than saying it did.
    #[test]
    fn tidying_an_index_writes_the_way_in() {
        let h = vault();
        h.sim(|s| {
            assert!(s
                .record
                .by_name("the third era")
                .unwrap()
                .description
                .is_none())
        });
        let out = perform(
            &h,
            "m1",
            &act("record_tidy_index", json!({"what":"the third era"})),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let entry = s
                .record
                .by_name("the third era")
                .unwrap()
                .description
                .clone()
                .expect("the index entry was not written");
            assert!(entry.contains("named span"), "{entry}");
        });
    }

    /// A reading of something the world does not have is refused, rather than
    /// leaving a finding about nothing.
    #[test]
    fn a_reading_of_nothing_is_refused() {
        let h = vault();
        for tool in [
            "structure_lay_out_scenes",
            "structure_find_the_slack",
            "record_tidy_index",
        ] {
            let out = perform(&h, "m1", &act(tool, json!({"what":"a thing nobody has"})));
            assert!(!out.happened(), "{tool} reported work on nothing");
        }
        let out = perform(
            &h,
            "m1",
            &act(
                "structure_test_the_want",
                json!({"what":"a thing nobody has","scene":"x"}),
            ),
        );
        assert!(!out.happened(), "{out:?}");
    }

    /// A mission taken up at the desk, worked with its progress recorded, and
    /// reported — the whole loop through `perform`. `perform` does not gate on
    /// availability (the grammar does), so this drives the acts directly.
    #[test]
    fn a_mission_is_collected_worked_and_reported() {
        let h = vault();
        // Nothing to report before collecting.
        assert!(matches!(
            perform(&h, "m1", &act("report_done", json!({"account":"nothing"}))),
            Outcome::Refused(_)
        ));

        // Collecting draws a mission (a bank routine here — the vault has no
        // records indexed) and makes it the body's open one.
        assert!(perform(&h, "m1", &act("collect_mission", json!({}))).happened());
        assert!(h.sim(|s| s.missions.is_on_mission("m1")));

        // A second collect while already carrying one is refused — one at a
        // time, so a mission in progress is never discarded by drawing another.
        // The refusal says what the open mission is, so the character is not
        // left to guess what it is already carrying.
        let brief = h.sim(|s| s.missions.active("m1").unwrap().mission_text());
        match perform(&h, "m1", &act("collect_mission", json!({}))) {
            Outcome::Refused(why) => {
                assert!(why.contains("already carrying a mission"), "{why}");
                assert!(why.contains(&brief), "{why}");
                assert!(why.contains("report_done"), "{why}");
            }
            other => panic!("a second collect must be refused, got {other:?}"),
        }

        // The bank's first routine goes to a room and reads a machine there, and
        // a journey not made cannot be reported done: the desk says what is left
        // and the way to it, and that `report_stuck` is the honest close if it
        // cannot be made.
        match perform(
            &h,
            "m1",
            &act("report_done", json!({"account":"it all looks fine to me"})),
        ) {
            Outcome::Refused(why) => {
                assert!(
                    why.starts_with("You have not done this yet: go to "),
                    "{why}"
                );
                assert!(why.contains("`move_to`"), "{why}");
                assert!(why.contains("report_stuck"), "{why}");
            }
            other => panic!("a journey not made cannot be reported done, got {other:?}"),
        }
        // Going there and reading it signs both steps off.
        let room = h.sim(|s| {
            let step = &s.missions.active("m1").unwrap().todo[0].text;
            step.trim_start_matches("go to ").to_string()
        });
        let machines: Vec<String> = h.sim(|s| s.devices.iter().map(|d| d.name.clone()).collect());
        assert!(h.with_sim(|s| s.missions.arrived_in("m1", &room, "the command level")));
        assert!(h.with_sim(|s| s.missions.read_off("m1", &machines)));
        h.with_sim(|s| {
            s.missions
                .observe("m1", "in the plant room, the coolant valve: open")
        });

        // A name is not an account: turned back with what was seen.
        match perform(
            &h,
            "m1",
            &act("report_done", json!({"account":"Paxon Vael"})),
        ) {
            Outcome::Refused(why) => {
                assert!(why.contains("is not an account of what you found"), "{why}");
                assert!(why.contains("the coolant valve: open"), "{why}");
            }
            other => panic!("a name is not an account, got {other:?}"),
        }

        // Reporting done closes it, frees the character, and files the answer so
        // an operator can still read it.
        assert!(perform(
            &h,
            "m1",
            &act(
                "report_done",
                json!({"account":"the ledger is two years out"})
            )
        )
        .happened());
        assert!(
            !h.sim(|s| s.missions.is_on_mission("m1")),
            "reporting frees the character for the next mission"
        );
        assert_eq!(
            h.sim(|s| s.missions.done("m1").unwrap().answer.clone()),
            Some("the ledger is two years out".to_string())
        );
    }

    /// **A writing step is not stuck while a desk can be reached**: the report
    /// is turned away with the way to one.
    #[test]
    fn a_writing_step_is_not_stuck_while_a_desk_can_be_reached() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let (h, _root) = vault_with_documents("stuck-at-desk");
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Tell the story of the water schedule.",
                    vec![
                        Todo::new("write layers/stories/the-water-schedule.md and commit it"),
                        Todo::report("go back to the table and report it"),
                    ],
                    Origin::Random {
                        routine: "x".into(),
                    },
                ),
            )
        });
        match perform(
            &h,
            "m1",
            &act(
                "report_stuck",
                json!({"why":"I keep arriving at the lift and cannot reach the desk"}),
            ),
        ) {
            Outcome::Refused(why) => {
                assert!(why.starts_with("It is not stuck yet — "), "{why}");
                assert!(why.contains("desk"), "{why}");
                assert!(why.contains("`move_to`"), "{why}");
            }
            other => panic!("a reachable desk is not stuck, got {other:?}"),
        }
    }

    /// **Stuck is turned away while the next step is within reach**, with where
    /// and how — twice. The third time the report is taken: a character told
    /// the way twice that still cannot is reporting a real block.
    #[test]
    fn stuck_is_refused_while_the_next_step_is_within_reach() {
        let h = vault();
        assert!(perform(&h, "m1", &act("collect_mission", json!({}))).happened());
        let stuck = || {
            perform(
                &h,
                "m1",
                &act("report_stuck", json!({"why":"it will not answer"})),
            )
        };
        for _ in 0..2 {
            match stuck() {
                Outcome::Refused(why) => {
                    assert!(why.starts_with("It is not stuck yet — "), "{why}");
                    assert!(why.contains("`move_to`"), "{why}");
                }
                other => panic!("a reachable step is not stuck, got {other:?}"),
            }
        }
        assert!(stuck().happened(), "the third time, the report is taken");
        assert!(!h.sim(|s| s.missions.is_on_mission("m1")));
    }

    #[test]
    fn every_station_and_bench_act_is_dispatched() {
        for t in crate::engine::station::STATION_ACTS
            .iter()
            .chain(crate::engine::bench::BENCH_ACTS)
            .chain(crate::engine::mission_acts::MISSION_ACTS)
        {
            assert!(
                is_mine(t.name),
                "`{}` is declared and not dispatched",
                t.name
            );
        }
    }

    #[test]
    fn writing_takes_the_thing_and_appends_to_it() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"a night at the gate"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let i = s.record.by_name("the third silence").expect("a gap");
            assert!(i.body.contains("night at the gate"), "{}", i.body);
            assert_eq!(i.holder.as_deref(), Some("m1"));
        });
    }

    /// The subject is the preposition, never `what`.
    ///
    /// `story_draft` carries the gap in `for` and the prose in `what`. A
    /// generic "first argument that looks like a subject" read `what` first and
    /// wrote the draft into a thing named after the draft's own text — then
    /// reported success, which is the failure that does not announce itself.
    #[test]
    fn the_thing_written_into_is_the_one_the_preposition_names() {
        let h = vault();
        perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"a night at the gate"}),
            ),
        );
        h.sim(|s| {
            assert!(
                s.record.by_name("a night at the gate").is_none(),
                "wrote into the prose"
            );
            assert!(!s
                .record
                .by_name("the third silence")
                .unwrap()
                .body
                .is_empty());
        });
    }

    #[test]
    fn a_second_maker_cannot_write_into_work_somebody_is_holding() {
        let h = vault();
        perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"mine"}),
            ),
        );
        let out = perform(
            &h,
            "m2",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"mine too"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("m1"), "{out:?}");
    }

    /// A filed thing is changed by making a change and warning whoever it
    /// breaks, not by writing over it where it stands.
    #[test]
    fn a_filed_thing_is_not_written_over() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"…"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("filed"), "{out:?}");
    }

    #[test]
    fn a_commit_that_collides_names_the_other_party_rather_than_failing_blankly() {
        let h = vault();
        // m1 opens it; m2 tries to take it and is told who has it.
        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"}))
        )
        .happened());
        let out = perform(
            &h,
            "m2",
            &act("bench_branch", json!({"what":"the third era"})),
        );
        assert!(!out.happened());
        assert!(
            out.line().unwrap().contains("m1"),
            "the holder was not named: {out:?}"
        );
    }

    #[test]
    fn the_working_loop_runs_branch_to_commit() {
        let h = vault();
        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third silence"}))
        )
        .happened());
        assert!(perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"a night at the gate"})
            )
        )
        .happened());
        assert!(perform(&h, "m1", &act("bench_stage", json!({}))).happened());
        let out = perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"filled the longest silence"})),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let i = s.record.by_name("the third silence").unwrap();
            assert_eq!(i.state, State::Filed);
            assert!(i.holder.is_none(), "filing did not release it");
        });
    }

    #[test]
    fn letting_something_go_without_a_reason_is_refused() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act("record_let_go", json!({"what":"the third era"})),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("reason"), "{out:?}");
    }

    #[test]
    fn settling_needs_the_other_side_named() {
        let h = vault();
        let alone = perform(
            &h,
            "m1",
            &act(
                "chronicle_settle_boundary",
                json!({"between":"the third era"}),
            ),
        );
        assert!(!alone.happened());
        assert!(alone.line().unwrap().contains("two"), "{alone:?}");

        let both = perform(
            &h,
            "m1",
            &act(
                "chronicle_settle_boundary",
                json!({"between":"the third era","and":"the fourth era"}),
            ),
        );
        assert!(both.happened(), "{both:?}");
    }

    #[test]
    fn blame_says_where_the_chain_goes_quiet() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act("bench_blame", json!({"what":"the third era"})),
        );
        assert!(out.line().unwrap().contains("quiet"), "{out:?}");

        perform(
            &h,
            "m1",
            &act(
                "record_write_provenance",
                json!({"of":"the third era","from":"the western intake"}),
            ),
        );
        let then = perform(
            &h,
            "m1",
            &act("bench_blame", json!({"what":"the third era"})),
        );
        assert!(then.line().unwrap().contains("western intake"), "{then:?}");
    }

    #[test]
    fn a_filed_document_is_not_deleted_as_though_it_were_scratch() {
        let h = vault();
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third silence"})),
        )
        .happened();
        perform(
            &h,
            "m1",
            &act("story_draft", json!({"for":"the third silence","what":"x"})),
        );
        perform(
            &h,
            "m1",
            &act("story_file", json!({"what":"the third silence"})),
        );
        let out = perform(
            &h,
            "m1",
            &act("file_delete", json!({"path":"the third silence"})),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("record_let_go"), "{out:?}");
    }
}
