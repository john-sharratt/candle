//! Routes each station act to what it does: writing a document into the record,
//! changing a character's sheet, raising faults, custody, and the rest.
//!
//! The subject is the act's naming argument, never its text, so a draft cannot be
//! written into a record item named after its own prose. Character sheets and
//! memories reach the mind folder directly, and a world without one keeps them in
//! its draft. A fault is raised only in a machine the vault has. Putting back
//! something the body is not holding is refused rather than reported as done, so a
//! character cannot repeat an imaginary act and be told each time that it worked.

use super::*;

/// The acts this module performs.
pub fn is_mine(tool: &str) -> bool {
    crate::engine::station::STATION_ACTS
        .iter()
        .chain(crate::engine::bench::BENCH_ACTS)
        .chain(crate::engine::mission_acts::MISSION_ACTS)
        .any(|t| t.name == tool)
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
    // The merge table's acts name nothing: each is about the operation whose
    // publish step the body carries.
    if matches!(act.tool, MERGE_STATUS | MERGE_PUBLISH | MERGE_SYNC) {
        return archive::perform(hosted, body, act.tool);
    }
    // The mission acts before the subject is looked for: `order_results`
    // names nothing, and the reports carry their subject under `found` / `why`,
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
            let shelf = hosted.shelf_of(body);
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
                match s.bench.append(body, &who, &path, &t, &*shelf) {
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
            let shelf = hosted.shelf_of(body);
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
                match s.bench.append(body, &place, &path, &t, &*shelf) {
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
            let shelf = hosted.shelf_of(body);
            hosted.with_sim(|s| {
                match s
                    .bench
                    .write_field(body, &what, &path, PORTRAIT_PROMPT, &carrying, &*shelf)
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
            let shelf = hosted.shelf_of(body);
            hosted.sim(
                |s| match s.bench.read_field(body, &path, PORTRAIT_PROMPT, &*shelf) {
                    Ok(prompt) => Outcome::Did(format!("{what} is drawn from: {prompt}")),
                    Err(why) => Outcome::Refused(why),
                },
            )
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
            // Held by body id, as every reader of the board matches it.
            let Some((id, to)) = hosted.read(|w| {
                w.actors()
                    .find(|a| a.name.eq_ignore_ascii_case(&to))
                    .map(|a| (a.id.clone(), a.name.clone()))
            }) else {
                return Outcome::Refused(format!("There is nobody here called {to}."));
            };
            if hosted.sim(|s| s.missions.is_unordered(&id)) {
                return Outcome::Refused(format!(
                    "{to} takes no orders: work for {to} is not handed over, it is asked for."
                ));
            }
            match hosted.with_sim(|s| s.ledger.hand_to(&what, &id)) {
                Ok(()) => Outcome::Did(format!("{to} has it: {what}")),
                Err(why) => Outcome::Refused(order_refusal(hosted, &what, why)),
            }
        }
        "orders_report_done" => hosted.with_sim(|s| match s.ledger.finish(body, &what) {
            true => Outcome::Did(format!("Reported done: {what}")),
            false => {
                let held = s.orders_held_by(body);
                let holds = match held.is_empty() {
                    true => "You hold no order from the orders board.".to_string(),
                    false => format!("The orders you hold: {}.", held.join("; ")),
                };
                Outcome::Refused(format!(
                    "You are not holding an order to {what}. {holds} A mission is not an order: \
                     one from the table is reported with `report_done`."
                ))
            }
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
        // **A fault is in a machine the vault has.** Taking `what` as written,
        // the board filled with orders like "the fault in The breaker panel is
        // closed" and "the fault in the coolant valve is tight" — a Maker's
        // reading of a machine posted as work for everyone, under a name no
        // machine had.
        "plant_note_drift" => with_second(hosted, a, "drift", &what, |s, w, drift| {
            let w = machine_named(s, w)?;
            s.ledger
                .set_order(&format!("look at {w}: {drift}"), "the panel", None);
            Ok(format!("Noted, so somebody looks: {w} is {drift}"))
        }),
        "plant_raise_fault" => with_second(hosted, a, "why", &what, |s, w, why| {
            let w = machine_named(s, w)?;
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
