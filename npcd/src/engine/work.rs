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
use crate::sim::record::{Condition, Item, Kind, State};
use crate::world::Hosted;

/// The acts this module performs.
pub fn is_mine(tool: &str) -> bool {
    crate::engine::station::STATION_ACTS
        .iter()
        .chain(crate::engine::bench::BENCH_ACTS)
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
        "what", "of", "in", "to", "for", "from", "on", "about", "between", "under", "path",
        "why", "called",
    ] {
        if let Some(v) = text(args, key) {
            return Some(v);
        }
    }
    None
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
    let Some(what) = subject(a).or_else(|| Some(String::new())).filter(|s| !s.is_empty()) else {
        // Only the argument-free bench acts get here legitimately.
        return bench_no_subject(hosted, body, act.tool);
    };

    match act.tool {
        // ── writing into the record ─────────────────────────────────────────
        "chronicle_add_entry" | "story_draft" | "character_write_identity"
        | "character_write_wants" | "character_write_memories" | "place_write_entry"
        | "place_write_local_history" | "chronicle_rewrite_page" => {
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
                match s.bench.write_field(body, &what, &path, PORTRAIT_PROMPT, &carrying) {
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
        "chronicle_settle_boundary" | "portrait_settle_likeness" | "character_settle_relation"
        | "place_settle_route" | "map_settle_border" => settle(hosted, body, a, &what),

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
                    return Outcome::Refused(format!("There is already somewhere called {called}."));
                }
                let mut i = Item::new(called.to_lowercase().replace(' ', "_"), &called, Kind::Place);
                i.body = wheres.clone();
                i.state = State::Draft;
                i.holder = Some(body.to_string());
                s.record.put(i);
                Outcome::Did(format!("{called} is on the map, {wheres}."))
            })
        }
        "map_remove_place" => hosted.with_sim(|s| match s.record.let_go(&what, "taken out of the world") {
            Ok(n) => Outcome::Did(format!(
                "{n} is off the map. Everything written about it still stands and now has to be \
                 reckoned with."
            )),
            Err(why) => Outcome::Refused(why),
        }),

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
                    let mut i = Item::new(what.to_lowercase().replace(' ', "_"), &what, Kind::Accession);
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
            s.record.describe(w, how).map(|n| format!("The way in to {n}: {how}"))
        }),
        "record_arrange" => with_second(hosted, a, "under", &what, |s, w, under| {
            s.record
                .describe(w, under)
                .map(|n| format!("{n} now sits under {under}, where somebody would look for it."))
        }),
        "record_cross_reference" => with_second(hosted, a, "to", &what, |s, w, to| {
            s.record.cross_reference(w, to).map(|t| format!("{w} now points at {t}."))
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
        "record_mark_repair" => hosted.with_sim(|s| match s.record.set_condition(&what, Condition::Mended) {
            Ok(n) => Outcome::Did(format!(
                "The repair to {n} is left visible. A mend passed off as an original is worse \
                 than the damage."
            )),
            Err(why) => Outcome::Refused(why),
        }),

        // ── the record facing outward ───────────────────────────────────────
        "enquiry_take_question" => hosted.with_sim(|s| match s.record.take(&what, body) {
            Ok(n) => Outcome::Did(format!("You take {n}. It is yours until it is answered.")),
            Err(why) => Outcome::Refused(why),
        }),
        "enquiry_answer_from_record" => with_second(hosted, a, "answer", &what, |s, w, ans| {
            s.record.write(w, "", ans).map(|n| format!("You answer {n}: {ans}"))
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
                return Outcome::Refused("You meant to hand an order to somebody, but not who.".into());
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
            None => Outcome::Refused(format!("There is nothing called {what} to present.")),
        }),

        // ── the plant and the stores ────────────────────────────────────────
        "plant_note_drift" => with_second(hosted, a, "drift", &what, |s, w, drift| {
            s.ledger.set_order(&format!("look at {w}: {drift}"), "the panel", None);
            Ok(format!("Noted, so somebody looks: {w} is {drift}"))
        }),
        "plant_raise_fault" => with_second(hosted, a, "why", &what, |s, w, why| {
            s.ledger.set_order(&format!("the fault in {w}"), "the panel", None);
            Ok(format!("Raised, and you may be wrong in public: {w} — {why}"))
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
            s.ledger.record_verdict(&name, body, "laid out as its scenes", None);
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
            s.ledger.set_order(&format!("come to the reading about {about}"), "the table", Some(who));
            Ok(format!("Called: everybody whose work touches {about} — {who}"))
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
    hosted.with_sim(|s| match s.bench.write_field(body, &path, &path, &[&field], &t) {
        Ok(p) => Outcome::Did(format!(
            "The {field} of {p} says what you wrote. It is how everybody here reads, once you \
             commit it."
        )),
        Err(why) => Outcome::Refused(why),
    })
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
    format!("personalities/{}.yaml", slug(who))
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
    Ok(format!("{dir}/{}.yaml", slug(&id)))
}

/// A name as a filename: what a character calls a thing, as the disk spells it.
fn slug(name: &str) -> String {
    let s: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| match c.is_ascii_alphanumeric() || c == '_' {
            true => c,
            false => '-',
        })
        .collect();
    s.trim_matches('-').replace("--", "-")
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
                (true, Some(w)) => {
                    Outcome::Did(format!("{w} is open and you have not changed anything in it."))
                }
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
            Ok(about) => Outcome::Did(format!(
                "You pick {about} back up, where you left it."
            )),
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
                    Outcome::Did(format!("Your changes to {w} are gone. It is back to what it was."))
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
            let (changed, offered) =
                hosted.sim(|s| (s.bench.diff(body), s.bench.opened(body).is_some_and(|w| w.offered)));
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
                let held = s.record.held_by(body);
                let Some(target) = held.first().cloned() else {
                    return match written.is_empty() {
                        true => Outcome::Refused("You have nothing open to merge.".into()),
                        false => Outcome::Did(format!(
                            "{} is part of what stands: {why_line}",
                            written.join(", ")
                        )),
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
                    Ok(n) => Outcome::Did(match written.is_empty() {
                        true => format!("{n} is merged into what stands: {why_line}"),
                        false => format!(
                            "{n} is merged into what stands, and {} with it: {why_line}",
                            written.join(", ")
                        ),
                    }),
                    Err(collision) => Outcome::Refused(format!(
                        "{collision} That is somebody to talk to, not something to try again."
                    )),
                }
            })
        }
        // A document that has been committed answers for itself, and the answer
        // is not the holder's to write: it is who the commit was made by.
        "bench_blame" | "bench_log" if hosted.sim(|s| s.bench.last_hand(what).is_some()) => {
            hosted.sim(|s| {
                let who = s.bench.last_hand(what).unwrap_or_default().to_string();
                Outcome::Did(format!("{what} was last committed by {who}."))
            })
        }
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
            hosted.sim(|s| match s.bench.excerpt(body, what, from) {
                Ok(rendered) => Outcome::Did(rendered),
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
            hosted.with_sim(|s| match s.bench.edit(body, what, what, &old, &new) {
                Ok(p) => Outcome::Did(format!("{p} is changed, and the rest of it is untouched.")),
                Err(why) => Outcome::Refused(why),
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
        Act { tool, args: args.as_object().unwrap().clone() }
    }

    fn vault() -> Hosted {
        let h = Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the vault must load");
        h.with(|w| {
            w.enter("m1", "Perrin Vastwood", Where::new("vault-chronicle", "early-range"))
                .unwrap();
            w.enter("m2", "Orion Vance", Where::new("vault-chronicle", "early-range"))
                .unwrap();
        });
        h
    }

    /// The vault, with somewhere to keep documents and two documents in it.
    ///
    /// A world without a root is a world with nothing to edit, so every test
    /// below that touches a document needs this rather than [`vault`].
    fn vault_with_documents(name: &str) -> (Hosted, std::path::PathBuf) {
        let root = std::env::temp_dir()
            .join(format!("npcd-work-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("layers/eras")).unwrap();
        std::fs::write(root.join("layers/eras/third.md"), "the third era\nburned in the spring\n")
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

    /// **A write is in memory and nowhere else until the commit.** The one
    /// behaviour the whole bench is built around, asserted against the disk
    /// rather than against a report.
    #[test]
    fn a_written_document_reaches_the_disk_only_at_the_commit() {
        let (h, root) = vault_with_documents("commit");
        let wrote = perform(
            &h,
            "m1",
            &act("file_write", json!({"path":"layers/eras/fifth.md","content":"the fifth era\n"})),
        );
        assert!(wrote.happened(), "{wrote:?}");
        assert!(!root.join("layers/eras/fifth.md").exists(), "the disk moved early");

        let out = perform(&h, "m1", &act("bench_commit", json!({"why":"named the fifth"})));
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
            &act("file_write", json!({"path":"layers/eras/third.md","content":"rewritten\n"})),
        );
        let mine = perform(&h, "m1", &act("file_read", json!({"path":"layers/eras/third.md"})));
        assert!(mine.line().unwrap().contains("rewritten"), "{mine:?}");

        let theirs = perform(&h, "m2", &act("file_read", json!({"path":"layers/eras/third.md"})));
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
            &act("file_write", json!({"path":"layers/eras/sixth.md","content":"a night at the gate"})),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert!(root.join("layers/eras/sixth.md").exists());
        assert!(
            !root.join("layers/eras").join("a night at the gate").exists(),
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
            &act("file_write", json!({"path":"layers/eras/fifth.md","content":"a line\n\nand another\n"})),
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
            &act("file_edit", json!({"path":"layers/eras/third.md","old_str":"  b: 1","new_str":"  b: 2"})),
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

        let first = perform(&h, "m1", &act("file_read", json!({"path":"layers/eras/long.md"})));
        let shown = first.line().unwrap();
        assert!(shown.contains("(lines 1-200 of 900)"), "{shown}");
        assert!(!shown.contains("line 201"), "the cap did not hold at the act");

        let next = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/long.md","start_line":"201"})),
        );
        assert!(next.line().unwrap().contains("(lines 201-400 of 900)"), "{next:?}");
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
                &act("file_read", json!({"path":"layers/eras/third.md","start_line":junk})),
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
            &act("file_edit", json!({"path":"layers/eras/third.md","old_str":"spring","new_str":"autumn"})),
        );
        assert!(out.happened(), "{out:?}");
        perform(&h, "m1", &act("bench_commit", json!({"why":"dated it against its neighbours"})));
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
            &act("file_edit", json!({"path":"layers/eras/third.md","old_str":"a fire","new_str":"a flood"})),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains('2'), "{out:?}");
    }

    #[test]
    fn an_edit_missing_half_of_itself_says_which_half() {
        let (h, _) = vault_with_documents("half");
        let out = perform(&h, "m1", &act("file_edit", json!({"path":"layers/eras/third.md","old_str":"x"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("both"), "{out:?}");
    }

    #[test]
    fn a_listing_shows_the_disk_and_your_own_new_documents() {
        let (h, _) = vault_with_documents("list");
        let before = perform(&h, "m1", &act("file_list", json!({"path":"layers/eras"})));
        assert!(before.line().unwrap().contains("third.md"), "{before:?}");

        perform(&h, "m1", &act("file_write", json!({"path":"layers/eras/fifth.md","content":"…"})));
        let after = perform(&h, "m1", &act("file_list", json!({"path":"layers/eras"})));
        assert!(after.line().unwrap().contains("fifth.md"), "{after:?}");
        // …and not to anybody else, because it is not committed.
        let theirs = perform(&h, "m2", &act("file_list", json!({"path":"layers/eras"})));
        assert!(!theirs.line().unwrap().contains("fifth.md"), "{theirs:?}");
    }

    #[test]
    fn a_deleted_document_goes_from_the_disk_at_the_commit() {
        let (h, root) = vault_with_documents("delete");
        let out = perform(&h, "m1", &act("file_delete", json!({"path":"layers/eras/fourth.md"})));
        assert!(out.happened(), "{out:?}");
        assert!(root.join("layers/eras/fourth.md").exists(), "gone before the commit");
        perform(&h, "m1", &act("bench_commit", json!({"why":"never canon"})));
        assert!(!root.join("layers/eras/fourth.md").exists());
    }

    /// The one act somebody else's work can refuse — now over a real document,
    /// and the refusal still names the other party.
    #[test]
    fn two_makers_on_one_document_collide_at_the_commit() {
        let (h, _) = vault_with_documents("collide");
        perform(&h, "m1", &act("file_write", json!({"path":"layers/eras/third.md","content":"mine\n"})));
        perform(&h, "m2", &act("file_write", json!({"path":"layers/eras/third.md","content":"mine too\n"})));

        assert!(perform(&h, "m1", &act("bench_commit", json!({"why":"first"}))).happened());
        let out = perform(&h, "m2", &act("bench_commit", json!({"why":"second"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("m1"), "no other party named: {out:?}");
        assert!(
            out.line().unwrap().contains("talk to"),
            "a collision read as a retryable error: {out:?}"
        );
        // The refused body still has all of its work.
        let still = perform(&h, "m2", &act("bench_diff", json!({})));
        assert!(still.line().unwrap().contains("layers/eras/third.md"), "{still:?}");
    }

    #[test]
    fn setting_work_aside_and_picking_it_up_again_survives_the_round_trip() {
        let (h, _) = vault_with_documents("stash");
        perform(&h, "m1", &act("file_write", json!({"path":"layers/eras/third.md","content":"half\n"})));
        assert!(perform(&h, "m1", &act("bench_stash", json!({}))).happened());
        let gone = perform(&h, "m1", &act("file_read", json!({"path":"layers/eras/third.md"})));
        assert!(gone.line().unwrap().contains("burned in the spring"), "{gone:?}");

        assert!(perform(&h, "m1", &act("bench_stash_pop", json!({}))).happened());
        let back = perform(&h, "m1", &act("file_read", json!({"path":"layers/eras/third.md"})));
        assert!(back.line().unwrap().contains("half"), "{back:?}");
    }

    #[test]
    fn throwing_the_work_away_puts_the_document_back() {
        let (h, _) = vault_with_documents("restore");
        perform(&h, "m1", &act("file_write", json!({"path":"layers/eras/third.md","content":"wrong\n"})));
        assert!(perform(&h, "m1", &act("bench_restore", json!({}))).happened());
        let back = perform(&h, "m1", &act("file_read", json!({"path":"layers/eras/third.md"})));
        assert!(back.line().unwrap().contains("burned in the spring"), "{back:?}");
    }

    #[test]
    fn status_and_diff_name_the_documents_that_are_changed() {
        let (h, _) = vault_with_documents("status");
        let quiet = perform(&h, "m1", &act("bench_status", json!({})));
        assert!(quiet.line().unwrap().contains("Nothing open"), "{quiet:?}");

        perform(&h, "m1", &act("file_write", json!({"path":"layers/eras/third.md","content":"changed\n"})));
        let diff = perform(&h, "m1", &act("bench_diff", json!({})));
        assert!(diff.line().unwrap().contains("layers/eras/third.md (changed)"), "{diff:?}");

        perform(&h, "m1", &act("bench_stage", json!({})));
        let staged = perform(&h, "m1", &act("bench_status", json!({})));
        assert!(staged.line().unwrap().contains("Offered"), "{staged:?}");
    }

    /// The paths are written by a language model, so the guard has to hold at
    /// the act, not only in the store beneath it.
    #[test]
    fn a_path_that_leaves_the_world_is_refused_at_the_act() {
        let (h, root) = vault_with_documents("escape");
        for bad in ["../stolen.md", "layers/eras/../../stolen.md", "c:/windows/x.md", "layers/eras/x.exe"] {
            let out = perform(&h, "m1", &act("file_write", json!({"path":bad,"content":"owned"})));
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
            act("file_write", json!({"path":"layers/eras/third.md","content":"x"})),
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
        perform(&h, "m1", &act("file_write", json!({"path":"layers/eras/third.md","content":"mine\n"})));
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        let out = perform(&h, "m2", &act("bench_blame", json!({"what":"layers/eras/third.md"})));
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
        assert!(perform(&h, "m1", &act("bench_branch", json!({"what":"the third era"}))).happened());
        let out = perform(
            &h,
            "m1",
            &act("chronicle_add_entry", json!({"to":"the third era","what":"The redoubt fell in the spring."})),
        );
        assert!(out.happened(), "{out:?}");
        assert!(
            !root.join("layers/eras/the-third-era.md").exists(),
            "the disk moved before the commit"
        );

        assert!(perform(&h, "m1", &act("bench_commit", json!({"why":"dated the fall"}))).happened());
        let written = std::fs::read_to_string(root.join("layers/eras/the-third-era.md")).unwrap();
        assert!(written.contains("The redoubt fell in the spring."), "{written}");
    }

    /// A second entry is added to the first, not written over it.
    #[test]
    fn a_second_entry_is_added_rather_than_replacing_the_first() {
        let (h, root) = vault_with_documents("era-append");
        perform(&h, "m1", &act("bench_branch", json!({"what":"the third era"})));
        perform(&h, "m1", &act("chronicle_add_entry", json!({"to":"the third era","what":"First."})));
        perform(&h, "m1", &act("chronicle_add_entry", json!({"to":"the third era","what":"Second."})));
        perform(&h, "m1", &act("bench_commit", json!({"why":"two entries"})));

        let written = std::fs::read_to_string(root.join("layers/eras/the-third-era.md")).unwrap();
        assert!(written.contains("First."), "{written}");
        assert!(written.contains("Second."), "the second entry replaced the first: {written}");
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
        perform(&h, "m1", &act("bench_branch", json!({"what":"the third silence"})));
        let out = perform(
            &h,
            "m1",
            &act("story_draft", json!({"for":"the third silence","what":"A night at the gate."})),
        );
        assert!(out.happened(), "{out:?}");
        perform(&h, "m1", &act("bench_commit", json!({"why":"filled the longest silence"})));

        let p = root.join("layers/stories/the-third-silence.md");
        assert!(p.exists(), "the draft did not reach the stories layer");
        assert!(std::fs::read_to_string(p).unwrap().contains("A night at the gate."));
    }

    /// **A filed document is not written over casually.** It refuses until the
    /// writer has opened it, which is the bench loop said by the store.
    #[test]
    fn a_filed_era_refuses_a_write_until_it_is_opened() {
        let (h, _root) = vault_with_documents("era-filed");
        let closed = perform(
            &h,
            "m1",
            &act("chronicle_add_entry", json!({"to":"the fourth era","what":"…"})),
        );
        assert!(!closed.happened());
        assert!(closed.line().unwrap().contains("Open it first"), "{closed:?}");

        assert!(perform(&h, "m1", &act("bench_branch", json!({"what":"the fourth era"}))).happened());
        let open = perform(
            &h,
            "m1",
            &act("chronicle_add_entry", json!({"to":"the fourth era","what":"And then not."})),
        );
        assert!(open.happened(), "{open:?}");
    }

    /// Two Makers on one era collide at the commit, over a real file.
    #[test]
    fn two_makers_writing_one_era_collide_at_the_commit() {
        let (h, _) = vault_with_documents("era-collide");
        perform(&h, "m1", &act("bench_branch", json!({"what":"the third era"})));
        perform(&h, "m1", &act("chronicle_add_entry", json!({"to":"the third era","what":"Mine."})));
        // m1 is holding it, so m2 is refused by custody before it reaches the
        // document at all — the record's own rule, still doing its job.
        let blocked = perform(
            &h,
            "m2",
            &act("chronicle_add_entry", json!({"to":"the third era","what":"Mine too."})),
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
            &act("record_appraise", json!({"what":"the third era","verdict":"sound"})),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| assert!(s.record.path_of("the western intake").is_none()));
        assert!(
            !root.join("layers/eras").join("the-western-intake.md").exists(),
            "an accession was given a document"
        );
    }

    // ── the craft libraries ─────────────────────────────────────────────────

    #[test]
    fn a_mood_can_be_read_and_changed_and_lands_on_the_disk() {
        let (h, root) = vault_with_documents("mood");
        let read = perform(&h, "m1", &act("library_read", json!({"kind":"mood","id":"undone"})));
        assert!(read.line().unwrap().contains("Something has been opened"), "{read:?}");

        let out = perform(
            &h,
            "m1",
            &act("library_write", json!({
                "kind":"mood","id":"undone","field":"description",
                "text":"So thoroughly opened that the whole interior has rearranged."
            })),
        );
        assert!(out.happened(), "{out:?}");
        perform(&h, "m1", &act("bench_commit", json!({"why":"it read as two things at once"})));

        let on_disk = std::fs::read_to_string(root.join("moods/undone.yaml")).unwrap();
        assert!(on_disk.contains("whole interior has rearranged"), "{on_disk}");
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
            &act("library_write", json!({
                "kind":"mood","id":"undone","field":"description","text":"Rearranged."
            })),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));

        let on_disk = std::fs::read_to_string(root.join("moods/undone.yaml")).unwrap();
        assert!(
            on_disk.contains("# The felt register — its KV is loaded"),
            "the comment was lost: {on_disk}"
        );
        assert!(on_disk.contains("category: mood"), "an untouched field moved: {on_disk}");
        assert!(on_disk.contains("Something has been opened"), "the template was rewritten");
    }

    #[test]
    fn a_library_that_does_not_exist_is_refused_by_name() {
        let (h, _) = vault_with_documents("mood-bad-kind");
        let out = perform(
            &h,
            "m1",
            &act("library_write", json!({"kind":"weather","id":"x","field":"template","text":"y"})),
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
                &act("library_write", json!({
                    "kind":"mood","id":"undone","field":field,"text":"x"
                })),
            );
            assert!(!out.happened(), "{field} was accepted");
        }
        let ok = perform(
            &h,
            "m1",
            &act("library_write", json!({
                "kind":"mood","id":"undone","field":"template","text":"Steady."
            })),
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
        assert!(read.line().unwrap().contains("lean sun-darkened man"), "{read:?}");

        let out = perform(
            &h,
            "m1",
            &act("portrait_prompt_edit", json!({
                "of":"ash-the-drifter",
                "carrying":"the same man after the winter, quieter and harder to read"
            })),
        );
        assert!(out.happened(), "{out:?}");
        perform(&h, "m1", &act("bench_commit", json!({"why":"the winter changed him"})));

        let on_disk =
            std::fs::read_to_string(root.join("personalities/ash-the-drifter.yaml")).unwrap();
        assert!(on_disk.contains("after the winter"), "{on_disk}");
        // The picture it names is untouched — a redraw follows the words.
        assert!(on_disk.contains("image: portraits/ash-the-drifter.png"), "{on_disk}");
        assert!(on_disk.contains("# Biography is NOT here."), "the header was lost");
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

        assert!(perform(&h, "m1", &act("structure_lay_out_scenes", json!({"what":"the third era"}))).happened());
        assert!(perform(
            &h,
            "m1",
            &act("structure_test_the_want", json!({"what":"the third era","scene":"the gate"})),
        )
        .happened());
        assert!(perform(&h, "m1", &act("structure_find_the_slack", json!({"what":"the third era"}))).happened());

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
        h.sim(|s| assert!(s.record.by_name("the third era").unwrap().description.is_none()));
        let out = perform(&h, "m1", &act("record_tidy_index", json!({"what":"the third era"})));
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
        for tool in ["structure_lay_out_scenes", "structure_find_the_slack", "record_tidy_index"] {
            let out = perform(&h, "m1", &act(tool, json!({"what":"a thing nobody has"})));
            assert!(!out.happened(), "{tool} reported work on nothing");
        }
        let out = perform(
            &h,
            "m1",
            &act("structure_test_the_want", json!({"what":"a thing nobody has","scene":"x"})),
        );
        assert!(!out.happened(), "{out:?}");
    }

    #[test]
    fn every_station_and_bench_act_is_dispatched() {
        for t in crate::engine::station::STATION_ACTS
            .iter()
            .chain(crate::engine::bench::BENCH_ACTS)
        {
            assert!(is_mine(t.name), "`{}` is declared and not dispatched", t.name);
        }
    }

    #[test]
    fn writing_takes_the_thing_and_appends_to_it() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act("story_draft", json!({"for":"the third silence","what":"a night at the gate"})),
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
            &act("story_draft", json!({"for":"the third silence","what":"a night at the gate"})),
        );
        h.sim(|s| {
            assert!(s.record.by_name("a night at the gate").is_none(), "wrote into the prose");
            assert!(!s.record.by_name("the third silence").unwrap().body.is_empty());
        });
    }

    #[test]
    fn a_second_maker_cannot_write_into_work_somebody_is_holding() {
        let h = vault();
        perform(&h, "m1", &act("story_draft", json!({"for":"the third silence","what":"mine"})));
        let out = perform(
            &h,
            "m2",
            &act("story_draft", json!({"for":"the third silence","what":"mine too"})),
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
            &act("chronicle_add_entry", json!({"to":"the third era","what":"…"})),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("filed"), "{out:?}");
    }

    #[test]
    fn a_commit_that_collides_names_the_other_party_rather_than_failing_blankly() {
        let h = vault();
        // m1 opens it; m2 tries to take it and is told who has it.
        assert!(perform(&h, "m1", &act("bench_branch", json!({"what":"the third era"}))).happened());
        let out = perform(&h, "m2", &act("bench_branch", json!({"what":"the third era"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("m1"), "the holder was not named: {out:?}");
    }

    #[test]
    fn the_working_loop_runs_branch_to_commit() {
        let h = vault();
        assert!(perform(&h, "m1", &act("bench_branch", json!({"what":"the third silence"}))).happened());
        assert!(perform(&h, "m1", &act("story_draft", json!({"for":"the third silence","what":"a night at the gate"}))).happened());
        assert!(perform(&h, "m1", &act("bench_stage", json!({}))).happened());
        let out = perform(&h, "m1", &act("bench_commit", json!({"why":"filled the longest silence"})));
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
        let out = perform(&h, "m1", &act("record_let_go", json!({"what":"the third era"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("reason"), "{out:?}");
    }

    #[test]
    fn settling_needs_the_other_side_named() {
        let h = vault();
        let alone = perform(&h, "m1", &act("chronicle_settle_boundary", json!({"between":"the third era"})));
        assert!(!alone.happened());
        assert!(alone.line().unwrap().contains("two"), "{alone:?}");

        let both = perform(
            &h,
            "m1",
            &act("chronicle_settle_boundary", json!({"between":"the third era","and":"the fourth era"})),
        );
        assert!(both.happened(), "{both:?}");
    }

    #[test]
    fn blame_says_where_the_chain_goes_quiet() {
        let h = vault();
        let out = perform(&h, "m1", &act("bench_blame", json!({"what":"the third era"})));
        assert!(out.line().unwrap().contains("quiet"), "{out:?}");

        perform(
            &h,
            "m1",
            &act("record_write_provenance", json!({"of":"the third era","from":"the western intake"})),
        );
        let then = perform(&h, "m1", &act("bench_blame", json!({"what":"the third era"})));
        assert!(then.line().unwrap().contains("western intake"), "{then:?}");
    }

    #[test]
    fn a_filed_document_is_not_deleted_as_though_it_were_scratch() {
        let h = vault();
        perform(&h, "m1", &act("bench_branch", json!({"what":"the third silence"}))).happened();
        perform(&h, "m1", &act("story_draft", json!({"for":"the third silence","what":"x"})));
        perform(&h, "m1", &act("story_file", json!({"what":"the third silence"})));
        let out = perform(&h, "m1", &act("file_delete", json!({"path":"the third silence"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("record_let_go"), "{out:?}");
    }
}
