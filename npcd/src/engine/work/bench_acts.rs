//! Performs the bench acts: writing into and settling documents, and committing or
//! setting aside a body's working set.
//!
//! A thing that is a document is written through the body's working set, where it
//! stays unseen by others until the commit. A judgement stays in the record and
//! gets no document. Committing a mission's document is all or nothing, and a
//! filed document that is changed stays filed, so the record is released rather
//! than retired. A commit that another body holds is refused, and the refusal
//! names that body.

use super::*;

/// Write into something, taking it if nobody has it.
///
/// **A thing that is a document is written as one.** An era, a story — anything
/// [`crate::sim::record::Record::index_canon`] found on the disk — carries a
/// path, and the write goes through the bench's working set: unseen until the
/// commit, checked against somebody else's commit when it lands, and part of
/// the mind afterwards. Everything else is a judgement rather than a document
/// and stays in the record, which is where a judgement belongs.
pub(super) fn write_into(hosted: &Hosted, body: &str, what: &str, text: &str) -> Outcome {
    let shelf = hosted.shelf_of(body);
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
        match s.bench.append(body, what, &path, text, &*shelf) {
            Ok(p) => Outcome::Did(format!(
                "You write into {what} ({p}). Nobody else sees it until you commit."
            )),
            Err(why) => Outcome::Refused(why),
        }
    })
}

pub(super) fn set_state(hosted: &Hosted, body: &str, what: &str, to: State) -> Outcome {
    hosted.with_sim(|s| match s.record.set_state(what, body, to) {
        Ok(n) => Outcome::Did(match to {
            State::Filed => format!("{n} is filed. It is part of the record and no longer yours."),
            State::Retired => format!("{n} is out of the record."),
            _ => format!("{n} is {to:?}."),
        }),
        Err(why) => Outcome::Refused(why),
    })
}

/// The machine `said` names, by its own name — or why there is none, with
/// each name there is once.
///
/// **A name stands once in the list however many rooms have one.** Seven plant
/// rooms each hold a breaker panel, and a Maker that asked for the air handler
/// was told "the breaker panel" seven times over before the list moved on.
pub(super) fn machine_named(s: &Sim, said: &str) -> Result<String, String> {
    if let Some(d) = s.devices.by_name(said) {
        return Ok(d.name.clone());
    }
    let mut names: Vec<String> = Vec::new();
    for d in s.devices.iter() {
        if !names.contains(&d.name) {
            names.push(d.name.clone());
        }
    }
    Err(format!(
        "There is no machine called {said}. `what` is the machine, by its name: {}.",
        list(&names)
    ))
}

/// An act needing a second argument, applied to the store.
pub(super) fn with_second(
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
pub(super) fn settle(
    hosted: &Hosted,
    body: &str,
    args: &Map<String, Value>,
    mine: &str,
) -> Outcome {
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
pub(super) fn bench_no_subject(hosted: &Hosted, body: &str, tool: &str) -> Outcome {
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
pub(super) fn bench(
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
            let shelf = hosted.shelf_of(body);
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
                // A mission that writes a document commits that document alone
                // (the bench's `commit_only`); anything else open stays open.
                let only = s
                    .missions
                    .active(body)
                    .and_then(|m| m.work.as_ref())
                    .map(|w| w.writes.clone());
                // Changes to where no bench may reach can never land; they
                // are dropped, and said, rather than refusing every commit.
                let dropped = s.bench.drop_unreachable(body);
                let written = match s.bench.opened(body).is_some() {
                    true => {
                        let committed = match &only {
                            Some(doc) => s.bench.commit_only(body, doc, &*shelf),
                            None => s.bench.commit(body, &*shelf),
                        };
                        match committed {
                            Ok(w) => w,
                            Err(collision) => {
                                return Outcome::Refused(format!(
                                    "{collision} That is somebody to talk to, not something to \
                                     try again."
                                ))
                            }
                        }
                    }
                    false => Vec::new(),
                };
                // What stays open after a mission's commit, said with it, so
                // nothing left behind is mistaken for committed.
                let left = s.bench.diff(body);
                // Committing a document a mission writes is that step done, and
                // what comes next is said with the commit.
                let progress = s.missions.committed(body, &written).then(|| {
                    let next = s.missions.active(body).and_then(|m| m.next_step());
                    progress_line("It is committed", next, &[])
                });
                let with_progress = |line: String| {
                    let line = match left.is_empty() {
                        true => line,
                        false => format!(
                            "{line} Still open, and not part of this commit: {}.",
                            left.join(", ")
                        ),
                    };
                    let line = match dropped.is_empty() {
                        true => line,
                        false => format!(
                            "{line} Left out and thrown away, because no bench works there: {}.",
                            dropped.join(", ")
                        ),
                    };
                    match &progress {
                        Some(p) => format!("{line}\n{p}"),
                        None => line,
                    }
                };
                let held = s.record.held_by(body);
                let Some(target) = held.first().cloned() else {
                    return match written.is_empty() {
                        true if !dropped.is_empty() => Outcome::Did(format!(
                            "Nothing was committed. Thrown away, because no bench works there: {}.",
                            dropped.join(", ")
                        )),
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
        // `page` is a string because the grammar cannot bound an integer, so
        // the act parses it. Anything unparseable reads as the first page
        // rather than refusing: a character that wrote "the top" meant page
        // one, and turning that into a refusal spends its turn on arithmetic.
        "file_read" => {
            let page = text(args, "page")
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(1);
            let shelf = hosted.shelf_of(body);
            hosted.with_sim(|s| match s.bench.excerpt(body, what, page, &*shelf) {
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
        "file_list" => {
            let shelf = hosted.shelf_of(body);
            hosted.sim(|s| match s.bench.list(body, what, &*shelf) {
                Ok(names) if names.is_empty() => Outcome::Did(format!("Nothing under {what}.")),
                Ok(names) => Outcome::Did(format!("Under {what}: {}.", names.join(", "))),
                Err(why) => Outcome::Refused(why),
            })
        }
        "file_write" => {
            let Some(t) = verbatim(args, "content") else {
                return Outcome::Refused("You meant to write a document, but not what.".into());
            };
            let shelf = hosted.shelf_of(body);
            hosted.with_sim(|s| match s.bench.write(body, what, what, &t, &*shelf) {
                Ok(p) => Outcome::Did(format!(
                    "{p} says what you wrote. Nobody else sees it until you commit."
                )),
                Err(why) => Outcome::Refused(why),
            })
        }
        "file_edit" => {
            let edit = match edit_of(args) {
                Ok(edit) => edit,
                Err(why) => return Outcome::Refused(why),
            };
            let shelf = hosted.shelf_of(body);
            hosted.with_sim(|s| {
                if let Some(why) = rewritten_by_hand(s, body, what, &edit, &*shelf) {
                    return Outcome::Refused(why);
                }
                match s.bench.edit(body, what, what, &edit, &*shelf) {
                    Ok(reply) => {
                        let note = s
                            .bench
                            .read(body, what, &*shelf)
                            .ok()
                            .and_then(|text| gates::length_note(what, &text));
                        Outcome::Did(match note {
                            Some(note) => format!("{reply}\n{note}"),
                            None => reply,
                        })
                    }
                    Err(why) => Outcome::Refused(why),
                }
            })
        }
        // The record's guard comes first: a *filed* thing is retired with a
        // reason, and letting `file_delete` take one would lose the reason
        // whoever comes after will want.
        "file_delete" => {
            let shelf = hosted.shelf_of(body);
            hosted.with_sim(|s| {
                if let Some(why) = the_mission_piece(s, body, what) {
                    return Outcome::Refused(why);
                }
                if let Some(i) = s.record.by_name(what) {
                    if i.state == State::Filed {
                        return Outcome::Refused(format!(
                            "{} is part of the record. Taking it out is `record_let_go`, which \
                             keeps the reason.",
                            i.name
                        ));
                    }
                }
                match s.bench.remove(body, what, what, &*shelf) {
                    Ok(p) => Outcome::Did(format!(
                        "{p} goes when you commit. It was never part of the record."
                    )),
                    Err(why) => Outcome::Refused(why),
                }
            })
        }
        other => Outcome::Refused(format!("Nothing here knows how to {other}.")),
    }
}
