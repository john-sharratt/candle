//! Works out what stands between a body and reporting its mission done: the
//! document's gate faults, a publish step not yet landed, a short piece, or steps
//! still undone.
//!
//! A document that fails its gate is sent back to be written again, with its
//! faults named, rather than on to a report that will be refused. A publish step is
//! held until its operation has landed on the record. A mission writes only its own
//! document, and a short piece is told how far short it is and what it is to hold,
//! because a bare verdict was resubmitted unchanged.

use super::*;

/// What stands between the body and reporting its operation's document done:
/// the gate's faults, once the engine has tidied what needs no judgement (see
/// [`gates::tidy`]). `None` when nothing does.
pub fn report_block(hosted: &Hosted, body: &str) -> Option<String> {
    // A publish step not yet landed is held at the report, but nothing in the
    // document is wrong: there is nothing to write again.
    if let Some(why) = hosted.sim(|s| unpublished(s, body)) {
        return Some(why);
    }
    let shelf = hosted.shelf_of(body);
    let path = hosted.sim(|s| {
        let m = s.missions.active(body)?;
        m.operation()?;
        Some(m.work.as_ref()?.writes.clone())
    });
    if let Some(path) = &path {
        if gates::tidy_on(&*shelf, path) {
            tracing::info!(npc = body, %path, "operation document tidied");
        }
    }
    let vocabulary = hosted.read(leakage::vocabulary);
    let lore = hosted.sim(|s| s.bench.files()).map(|f| leakage::lore(&f));
    // Kept on the mission until a report is taken, so the compass can say it
    // on every turn between — see `mission::refused_compass`.
    let Some(refusal) = hosted.sim(|s| gate_refusal(s, body, &*shelf, &vocabulary, &lore)) else {
        hosted.with_sim(|s| s.missions.set_gate_refusal(body, None));
        return None;
    };
    hosted.with_sim(|s| s.missions.set_gate_refusal(body, Some(refusal.clone())));
    // A document that does not stand is to be written again — said in the
    // mission's own steps, so the compass sends the Maker back to its desk
    // instead of on to a report that will be refused. One gone from the store
    // has nothing to write again.
    if path.is_some_and(|path| shelf.read(&path).is_some()) {
        hosted.with_sim(|s| s.missions.reopen_write(body));
    }
    Some(refusal)
}

/// Why the body's publish step may not be reported yet: its operation has not
/// landed on the record (`published`). `None` when it has, or the step does
/// not check it.
pub(crate) fn unpublished(s: &Sim, body: &str) -> Option<String> {
    let m = s.missions.active(body)?;
    let (id, _) = m.operation()?;
    let work = m.work.as_ref()?;
    if !work.checks.iter().any(|c| c == PUBLISHED) {
        return None;
    }
    s.missions
        .operations()
        .get(id)?
        .published
        .is_none()
        .then(|| {
            // **Where the tools are, not only what they are called.** Told
            // the names alone, four Makers stood beside the merge table and
            // called `report_done` again and again, and asked each other how
            // its tools were invoked: a station's acts are addresses, and
            // `here` is where a room's addresses are read.
            format!(
                "{} has not landed in the record yet. At the merge table, `merge_status` shows \
             whether it merges cleanly, and `merge_publish` lands it — `invoke` them at the \
             table's own address, which `http://local/here` lists in its room. Report once \
             it has.",
                work.writes
            )
        })
}

/// Why the body's operation document may not be reported done yet: it fails
/// a check its workflow step names — the page's own (`gates::CHECKS`), the
/// writers' room in it (`leakage`, whose `vocabulary` is the world the Makers
/// stand in), its brief's words copied into it (`copied`), or, on a step that
/// must change it, being as the step found it (`changed`). The document is
/// read from `shelf`, the operation's store; `lore` is the published record's
/// eras and world. `None` when it passes, or when the mission is not an
/// operation's.
pub(super) fn gate_refusal(
    s: &Sim,
    body: &str,
    shelf: &dyn Shelf,
    vocabulary: &[String],
    lore: &Option<Arc<String>>,
) -> Option<String> {
    // A world with no documents has nothing to hold to a gate.
    if !s.bench.has_root() {
        return None;
    }
    let m = s.missions.active(body)?;
    let (id, _) = m.operation()?;
    let work = m.work.as_ref()?;
    let checks = |name: &str| work.checks.iter().any(|c| c == name);
    let path = &work.writes;
    let op = s.missions.operations().get(id);
    let text = match shelf.read(path) {
        Some(t) => t,
        // A document written anew is not there until it is written.
        None if work.anew => return None,
        None => {
            return Some(format!(
                "{path} is not on the record any more, so there is nothing to pass. `invoke` this \
                 table's `report_rejected` and say so."
            ))
        }
    };
    let mut faults: Vec<String> = gates::check(path, &text, gates::life_voice(shelf, path))
        .into_iter()
        .filter(|f| checks(f.check))
        .map(|f| f.text)
        .collect();
    if checks(gates::LEAKAGE) && gates::Form::of(path) != gates::Form::Other {
        let leaked = leakage::leaks(&text, vocabulary, lore.as_deref().map_or("", |l| l));
        if !leaked.is_empty() {
            faults.push(leakage::fault(&leaked));
        }
    }
    let found = op.and_then(|o| o.found.as_deref());
    if checks(CHANGED) && !work.edit_optional && found == Some(text.as_str()) {
        faults.push(
            "it is word for word what you were given: what was found in it is still to put right"
                .to_string(),
        );
    }
    let brief = op.and_then(|o| o.fields.get("happens"));
    if let Some(fault) = brief
        .filter(|_| checks(BRIEF_COPIED))
        .and_then(|b| copied::copied(&text, b))
    {
        faults.push(fault);
    }
    // A mission template that would not parse must never publish, or it breaks
    // the world's loader for every routine. The write step carries this check.
    if checks(ROUTINE_PARSES) {
        if let Err(why) = Routine::parse(path, &text) {
            faults.push(format!("it is not a valid mission template: {why}"));
        }
    }
    if faults.is_empty() {
        return None;
    }
    let mend = match m.may_reject() {
        false => {
            "Mend it with `file_edit` (or write it again whole with `compose`) and \
             `bench_commit`, then come back and report it."
        }
        true => {
            "Mend it yourself with `file_edit` and `bench_commit` before you pass it — or, if it \
             cannot be mended, `invoke` this table's `report_rejected` and say why."
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
/// Why `body` may not take `path` out: it is the document its mission writes.
///
/// **A mission's own piece is mended, never taken out.** Odile, sent to mend a
/// draft, deleted it; every edit after that was refused ("You have taken it
/// out"), the working-set verbs that could undo it are not an operation's, and
/// the review could neither be finished nor reported. `None` for any other
/// document, or a body whose mission writes none.
pub(super) fn the_mission_piece(s: &Sim, body: &str, path: &str) -> Option<String> {
    let work = s.missions.active(body)?.work.as_ref()?;
    let norm = |p: &str| p.trim().trim_start_matches('/').to_string();
    (norm(path) == norm(&work.writes)).then(|| {
        format!(
            "{} is the piece your mission is for: mend it in place — `file_edit` a passage, or \
             write it whole again — rather than take it out.",
            work.writes
        )
    })
}

pub(super) fn outside_the_mission(s: &Sim, body: &str, path: &str) -> Option<String> {
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

/// Why the body's open mission document is not ready to commit — it is in the
/// working set and shorter than the mission's floor — or `None` when it is, or
/// when the commit does not touch it.
pub(super) fn short_of_the_floor(s: &Sim, body: &str) -> Option<String> {
    let mission = s.missions.active(body)?;
    let work = mission.work.as_ref()?;
    if work.min_words == 0 {
        return None;
    }
    let n = s
        .bench
        .opened(body)?
        .changed_text(&work.writes)?
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
pub(super) const WHAT_HAPPENS: &str = "What happens:";

/// The most of a brief's "What happens" a refusal quotes, in characters.
pub(super) const QUOTED_HAPPENS: usize = 900;

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
pub(super) fn short_by(writes: &str, n: usize, min: usize, happens: Option<String>) -> String {
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
pub(crate) fn finished(s: &Sim, body: &str) -> Option<String> {
    let m = s.missions.active(body)?;
    let left = m.next_step().is_some_and(|t| !t.reports);
    if left || !m.written_up() {
        return None;
    }
    // **What is left is the landing, when it has not landed.** Sent from the
    // stuck desk to `report_done` and from `report_done` back to the stuck
    // desk, a Maker on a publish step went between the two refusals and never
    // to the merge table: the one way on is said instead.
    if let Some(why) = unpublished(s, body) {
        return Some(format!(
            "Nothing here is stuck: every step of your mission is done but the landing. {why}"
        ));
    }
    let made = match &m.work {
        Some(w) if !w.edit_optional => format!(" {} is written and committed.", w.writes),
        _ => String::new(),
    };
    let verdict = match m.may_reject() {
        false => "`invoke` the table's `report_done` with what you made or found",
        true => {
            "give your verdict at the table: `report_done` if it stands, `report_rejected` with \
             why if it cannot be mended"
        }
    };
    Some(format!(
        "Nothing here is stuck: every step of your mission is done.{made} What is left is the \
         report — {verdict}."
    ))
}
