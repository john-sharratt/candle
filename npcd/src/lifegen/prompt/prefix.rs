//! Builds the shared prefix that each lifegen phase is primed with, so every fork
//! of a phase attends the same text.
//!
//! The prefix holds the character sheet, the arc, the cast and the rules of the
//! phase. It holds nothing that varies per fork, since such content would give
//! every fork a prompt written for one of them. A span of dormancy is stated with
//! its duration and as a gap to write around, so the arc does not bridge it. The
//! outline is not in the prefix. A fork's own beat travels in its turn, and it is
//! given only the years either side of its own.

use super::*;

/// The seed, rendered as the standing facts every phase writes against.
pub(super) fn who(c: &Checked) -> String {
    let s = &c.seed;
    let mut out = format!(
        "THE CHARACTER\n\
         Name: {}\n\
         Born: {}\n\
         Place: {}\n\
         Became: {}\n\
         The authored life runs to {}, when they are {}.\n",
        s.display,
        s.born,
        s.place,
        s.role,
        s.through,
        c.age()
    );
    // **The spans, including the ones with nothing in them.**
    //
    // A dormant stretch has to reach the writer or the arc will quietly bridge it — a mind
    // archived for three centuries gets an account of those centuries, which is the one thing
    // that cannot have happened. Stated as a span with a duration and an explicit "they
    // remember none of it", so the gap is a fact to write around rather than an absence to
    // fill.
    // Each span states its subject and its handling on separate labelled lines. Run together on
    // one line — `{instruction} {what}` — the two were a pair of finished sentences about the
    // character, in a prompt whose output is sentences about the character, and a model
    // reproduced them as narration rather than writing from them. "How to write it:" cannot be
    // mistaken for a line of a diary; the sentence it used to sit beside could.
    if c.eras.len() > 1 || c.eras.iter().any(|e| !e.kind.is_written()) {
        out.push_str("\nHOW THIS EXISTENCE DIVIDES\nNotes for you, not text to reuse.\n");
        for e in &c.eras {
            let span = e.to.year - e.from.year;
            out.push_str(&format!(
                "- {} to {} ({} year{})",
                e.from,
                e.to,
                span.max(1),
                if span == 1 { "" } else { "s" },
            ));
            if !e.what.is_empty() {
                out.push_str(&format!(" — {}", e.what));
            }
            out.push_str(&format!("\n  How to write it: {}\n", e.kind.instruction()));
        }
        if c.eras.iter().any(|e| !e.kind.is_written()) {
            out.push_str(
                "\nThe dormant spans are gaps in what they can remember. Do not narrate them, \
                 do not have the character reflect on passing through them, and do not let the \
                 arc flow smoothly across one — waking after a span like that is a discontinuity \
                 and should read as one.\n",
            );
        }
    }
    if !s.facts.is_empty() {
        out.push_str("\nTHINGS THAT MUST BE TRUE\n");
        for f in &s.facts {
            out.push_str(&format!("- {f}\n"));
        }
    }
    if !s.world.is_empty() {
        out.push_str(
            "\nTHE WORLD THIS LIFE HAPPENED IN\n\
             These are fixed. Other people in this world remember them the same way, \
             so do not move them, rename them, or invent others like them.\n",
        );
        for w in &s.world {
            out.push_str(&format!("- {}: {}\n", w.date, w.what));
        }
    }
    if !s.cast.is_empty() {
        out.push_str("\nPEOPLE WHO ALREADY EXIST\n");
        for p in &s.cast {
            out.push_str(&format!("- {} ({}): {}\n", p.entity_id, p.display, p.what));
        }
    }
    out
}

/// The story phase's prefix. One decode, and the only one that may invent.
pub fn story_prefix(c: &Checked, voice: &str) -> String {
    format!(
        "{VOICE}\n\n{}{}\n\nYOUR TASK\n\
         Write the shape of this whole life, then name its people, then divide it into \
         years. Everything below this phase expands what you decide here and invents \
         nothing of its own, so anything you leave out will not appear later.\n\n\
         {}\n",
        who(c),
        in_their_voice(voice),
        c.seed.cadence.instruction()
    )
}

/// Everything below the story shares this: the arc, the cast and the outline.
///
/// One function rather than three near-copies, because the three phases must
/// agree about what a name refers to, and the surest way to guarantee that is
/// for them to be reading the same text.
/// What every phase below the story shares.
///
/// # The character sheet is part of it, and used not to be
///
/// This carried the story, the cast and the outline — everything the story
/// phase *produced* — and none of what it was produced *from*. So a month knew
/// the arc of the life and not who was living it: no birth, no place, no trade,
/// no spans, none of the author's fixed facts, and no world. The lower a phase
/// sat, the less it knew about the person, which is exactly backwards. A day is
/// the most concrete thing in the ladder and had the least to be concrete with.
///
/// [`who`] is now the first thing in the shared prefix, so the anchor reaches
/// every rung. It costs prefix tokens on a prompt that is primed once per phase
/// and shared by every node in it — the cheapest place in the whole system to
/// put something every node needs.
pub(super) fn common(plan: &Plan, voice: &str) -> String {
    let mut out = String::new();
    out.push_str(VOICE);
    out.push_str("\n\n");
    // Rebuilt from the plan's own seed rather than passed in: the seed is
    // already stored on the plan, and re-deriving it here means the lower
    // phases cannot drift from what the story phase was told.
    match crate::lifegen::seed::check(&plan.seed) {
        Ok(c) => out.push_str(&who(&c)),
        // A plan that fails its own seed check is one written before a rule
        // existed. The phases below still have the story to work from, and a
        // hard failure here would make the plan ungeneratable rather than
        // merely thinner.
        Err(_) => out.push_str(&format!(
            "THE CHARACTER\nName: {}\nBorn: {}\nPlace: {}\nBecame: {}\n",
            plan.seed.display, plan.seed.born, plan.seed.place, plan.seed.role
        )),
    }
    out.push_str("\n\nTHE LIFE\n");
    out.push_str(plan.story.content.text.trim());
    out.push_str("\n\nTHE PEOPLE IN IT\n");
    if plan.story.cast.is_empty() {
        out.push_str("(nobody named yet)\n");
    }
    for p in &plan.story.cast {
        out.push_str(&format!("- {} ({}): {}\n", p.entity_id, p.display, p.what));
    }
    out.push_str(
        "\nRefer to these people by their id when a consequence names one. Do not \
         introduce anyone who is not on this list.\n",
    );
    // **The outline is not in the shared prefix, and used to be.**
    //
    // Every one of its lines was here, which was affordable while an outline was
    // eight beats and is not now that it is one per year lived: a 367-year life
    // put 4,900 tokens of year list in front of every fork of every phase, out of
    // a 6,144-token context. Kaelor's years phase refused all forty-one nodes on
    // exactly that — "a 4,897-token prompt plus 2,400 tokens of answer passes the
    // 6,144-token context" — and the longer lives had no room to begin with.
    //
    // It was redundant as well as ruinous. A year's own beat travels in its turn,
    // which is where per-fork information belongs and where this file's opening
    // paragraph already says it goes; the shared prefix carried three hundred and
    // sixty-six other years so that each fork could ignore them. What a fork
    // genuinely needs of its neighbours is a window, and a window is per-fork —
    // see [`outline_window`].
    out.push_str(&format!(
        "\nTHE SHAPE OF IT\n{} year(s) are outlined, {}.\n",
        plan.story.outline.len(),
        match (plan.story.outline.first(), plan.story.outline.last()) {
            (Some(a), Some(b)) => format!("{} to {}", a.year, b.year),
            _ => "none of them yet".to_string(),
        }
    ));
    // Last in the prefix, so it is the closest thing to the task the model
    // reads — the same placement reasoning the task's own restatements use.
    out.push_str(&in_their_voice(voice));
    out
}

/// The years phase's prefix.
pub fn years_prefix(plan: &Plan, voice: &str) -> String {
    format!(
        "{}\n\
         YOUR TASK\n\
         You are writing out ONE year of this life in full. The year has already been \
         decided above — you are expanding it, not choosing it. Do not add events the \
         outline does not give this year, and do not reach into another year.\n",
        common(plan, voice)
    )
}

/// The months phase's prefix.
pub fn months_prefix(plan: &Plan, voice: &str) -> String {
    format!(
        "{}\n\
         YOUR TASK\n\
         You are writing out ONE month of this life in full, and then saying which of \
         its days became memories.\n\n\
         The year this month belongs to has already been written, and you will be given \
         it. Expand the part of it that falls in your month. **Do not invent events the \
         year does not contain** — other months of the same year are being written at \
         the same time and cannot see what you write, so anything you add here will \
         contradict them.\n\n\
         Every month is written. A quiet month is written as a quiet month, in full, \
         and not skipped.\n",
        common(plan, voice)
    )
}

/// The days phase's prefix.
pub fn days_prefix(plan: &Plan, voice: &str) -> String {
    format!(
        "{}\n\
         YOUR TASK\n\
         You are writing ONE day of this life — a day that became a memory.\n\n\
         You will be given the month it falls in and, sometimes, what this day is \
         required to have produced: a conviction the character came away with, a person \
         they came to trust or stopped trusting, something they resolved to do. Those \
         are not yours to choose. Write the day so that it **earns** them — so a reader \
         finishes it and finds the conclusion obvious. Do not state the conclusion \
         outright and do not write it as a lesson; show the day that produced it.\n",
        common(plan, voice)
    )
}
