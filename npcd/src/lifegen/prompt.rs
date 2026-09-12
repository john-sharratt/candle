//! Each phase's shared prefix, its per-fork instruction, and reading the
//! answers back.
//!
//! # What goes in the prefix, and what cannot
//!
//! The prefix **is** the system prompt, and a fork shares it. So the split is
//! not a convention to remember — it is enforced by where the text is put:
//!
//! - **Prefix** — the seed, the arc, the cast, the year outline. Identical for
//!   every fork of a phase, primed once, attended by all of them.
//! - **Turn** — which year, which month, which day; the parent text being
//!   expanded; the consequences this day must earn. Per fork.
//!
//! Putting "expand month 3" in the prefix is not a mistake that costs
//! performance. It is not expressible.
//!
//! # Assign above, expand below
//!
//! Every instruction past the story says *expand what you were given* and says
//! it explicitly, because the alternative breaks the fan-out rather than merely
//! reading differently. Siblings never see each other: two months told to
//! "continue the story" both introduce a stranger at the gate, and neither can
//! know the other did.
//!
//! # The register is the character's own second person
//!
//! `layers/life/` documents are written as *"You came back to the yard and the
//! granary was gone"* — the same voice the semantic layers use, because they
//! are read back as the character's own memory. A document in third person is
//! a biography of somebody the character has never met.
//!
//! # Why the structured tails are lines, not JSON
//!
//! Each phase returns prose plus a short list — the cast, the year outline, the
//! days worth remembering. Asking for JSON around a paragraph of prose gets
//! prose with broken JSON in it. A `- a | b | c` line is what a model produces
//! reliably when it has just finished writing prose, it is readable in the
//! document if a parse ever fails, and [`parse_story`] and [`parse_month`] are
//! forgiving about spacing while staying strict about structure.

use serde_json::Value;

use super::consequence::Consequence;
use super::plan::{CastMember, Plan, YearBeat};
use super::seed::Checked;
use crate::engine::authoring::by_name;

/// The character's own idiom, drawn from the personality they are a life for.
///
/// # Why a life needs the personality's voice and not just its facts
///
/// These documents are **injected into a character's prefix** — they are what
/// that character remembers, read back to them as their own memory. A memory
/// written in a neutral literary register is a memory in somebody else's words,
/// and stuffing a few thousand tokens of somebody else's words into a prefix
/// teaches the model to answer in them.
///
/// So the seed is not enough. A seed says a character was a shot-firer in a
/// quarry; the personality says he counts down out loud, does the sums where
/// people can check them, and never hurries. The first is biography and the
/// second is how the diary has to sound.
///
/// The anchor is taken whole because it is the same text the engine already
/// makes resident for that character, so a life written against it and a
/// character speaking from it are working from one description rather than two
/// that can drift. `personality.voice`, where an author wrote one, is appended
/// because it says explicitly what the anchor only demonstrates.
pub fn voice_of(personality: &Value) -> String {
    let mut out = String::new();
    if let Some(a) = personality.get("anchor").and_then(|v| v.as_str()) {
        out.push_str(a.trim());
    }
    if let Some(v) = personality
        .get("personality")
        .and_then(|p| p.get("voice"))
        .and_then(|v| v.as_str())
    {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("How they speak: ");
        out.push_str(v.trim());
    }
    out
}

/// The diary instruction, wrapped around whatever [`voice_of`] found.
///
/// Empty when the personality carries neither an anchor nor a voice, so a life
/// for a character with no personality behind it still generates — plainly,
/// rather than against an empty heading that reads as a missing section.
fn in_their_voice(voice: &str) -> String {
    if voice.trim().is_empty() {
        return String::new();
    }
    format!(
        "\n\nWHOSE VOICE THIS IS IN\n{voice}\n\n\
         Write it the way THIS person would write it in a diary — their vocabulary, their \
         rhythm, the things they notice and the things they would not bother to say. A \
         soldier who counts and a duelist who does not explain herself do not describe the \
         same afternoon the same way.\n\n\
         This is the character remembering, not somebody describing the character. Never \
         explain them from outside, never name their traits, and never write a sentence they \
         would not have written."
    )
}

/// How the daemon addresses the writer in every phase.
/// The voice every phase writes in.
///
/// # The example used to be a scene, and models wrote the scene
///
/// This demonstrated second person with a sentence: *"You came back to the yard
/// and the granary was gone."* It is a good sentence, which is the problem — a
/// yard and a burnt granary are an image, and an image in a prompt is content.
/// Two different models, a 9 B and a 14 B, both reproduced it word for word: a
/// Vietnamese duelist's life story opened on that yard, and so did a Zenling
/// combat drone's, whose entire existence is eleven kilometres of desert and
/// which has no yard, no granary and no capacity to miss one.
///
/// The illustration is now grammatical rather than pictorial — a pronoun
/// contrast with no scene attached, so there is nothing in it worth stealing.
/// Anything demonstrating style to a model has to be inert as content, because
/// a model cannot tell the two apart from position alone.
const VOICE: &str = "\
You are writing one character's diary — their own account of their own life, \
which a game engine will read back to them as memory. It is not a biography and \
there is no narrator: the only person here is the one whose life it is.

Write in the second person, the way a diary written to oneself reads: \
\"You waited\" — never \"She waited\", and never the character's name as the \
subject. Plain past tense. No headings inside the prose, no bullet lists, no \
commentary about the writing.

Write what they would actually set down. A diary is uneven — it lingers on what \
mattered to THEM and passes over what did not, states without justifying, and \
leaves out what they would consider obvious. It does not summarise a person from \
outside or name their qualities; someone writing about their own year does not \
call themselves patient.

Write only this character's life. Nothing from any example, instruction or \
other person's record above belongs in the prose.";

/// The seed, rendered as the standing facts every phase writes against.
fn who(c: &Checked) -> String {
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

/// The story phase's instruction.
pub fn story_turn(c: &Checked) -> String {
    // **Only the years that are written.** Asking for "every year from first to last" was
    // right for a person and wrong for a mind with a gap in it: Keeper's span runs 2461 to
    // 3087, but 295 of those years are hibernation and have no document to outline. The old
    // phrasing demanded 627 lines for 332 years of life, and the 295 it invented would have
    // been an account of being unconscious.
    let years = c.years();
    let (first, last) = (
        years.first().copied().unwrap_or(0),
        years.last().copied().unwrap_or(0),
    );
    let gaps: Vec<&super::seed::CheckedEra> =
        c.eras.iter().filter(|e| !e.kind.is_written()).collect();
    let listed = if gaps.is_empty() {
        format!("every year from {first} to {last} inclusive, none skipped")
    } else {
        format!(
            "every year this character was actually there for — {} year(s) in total, \
             {first} to {last} but skipping {}",
            years.len(),
            gaps.iter()
                .map(|g| format!("{} to {}", g.from.year, g.to.year))
                .collect::<Vec<_>>()
                .join(" and ")
        )
    };
    // **The turn names its subject.** The identity is in the system prompt, and a thinking
    // model that spends its reasoning on the *structure* of the answer will then write the
    // structure and invent the person: asked for Keeper's life it produced a woman called
    // Elara Vance on Mars, with only the birth year and a garbled "Kaelen" leaking through
    // from the prefix it had stopped attending to. Re-anchoring costs one line.
    let s = &c.seed;
    format!(
        "Write the life of {}, {} — the character described above. Everything you write must \
         be about them: their place, the facts that must be true of them, the people named in \
         their cast, and the world events they lived through. Do not invent a different \
         person, a different world, or names that are not in the cast.\n\n\
         Begin with the arc of {}'s life in four to eight paragraphs of prose: what formed \
         them, what they came to want, what it cost them, and who they are by \
         {last}. Prose only — no headings, no lists, and nothing about what you are \
         going to write.\n\n\
         Then this section, in exactly this shape:\n\n",
        s.display, s.role, s.display,
    ) + &format!(
        // **The person is stated last, because the last instruction read wins.**
        //
        // The fourth time this file has learned that, and the first at this rung.
        // Everything above is necessarily written *about* the character — "the
        // life of Rook", "what formed them" — so a turn that is third person
        // throughout gets a third-person answer, whatever the system prompt said
        // several thousand tokens earlier. The story came back as "She was
        // ground, position, and the reading of a room": a biography of somebody
        // the character has never met, which is the one thing the register
        // exists to prevent.
        //
        // **Shown as the sections themselves, not described as a pair of steps.**
        //
        // This used to read "Then, on a line of its own, `## Cast` — …", twice.
        // A model answered it as a script: it narrated the instructions back —
        // "First, the arc of Rook's life in four to eight paragraphs of prose:",
        // then "Then, on a line of its own:" — and quoted the headings and every
        // row as inline code, because that is how they were shown to it. A
        // template a model can copy has to *be* the output, or the description
        // of the output becomes the output.
        // **The outline is no longer asked for here.** It used to be a third
        // section of this answer, and it lost: prose, cast and outline competed
        // for one budget, the outline was last, and across twenty-seven
        // characters lives of three hundred and sixty-seven years produced four
        // to eighteen year lines. Raising the budget changed nothing, because
        // the model was not truncated — it was done. [`outline_turn`] asks for
        // the years a run at a time instead, which is a request a model
        // completes.
        //
        // The span is still stated, because the arc has to know its own shape
        // even though it is not enumerating it here.
        "## Cast\n\
         One line per person who matters to this life:\n\
         - entity-id | Display Name | who they are to this character\n\
         The id is lowercase with hyphens and is how every later part of the life will \
         refer to them, so choose it once and use nothing else.\n\n\
         The life you are writing covers {listed}. Do not list the years — the arc is what \
         is wanted here, and the years are asked for separately.\n\n\
         The prose is in the SECOND person — \"You waited\", never the character's name and \
         never \"she\" or \"he\" as the subject. It is their own diary, not an account of them.\n"
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
fn common(plan: &Plan, voice: &str) -> String {
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

/// One fork's instruction for a year.
/// The outline either side of `year`, as the years around it.
///
/// A fork needs to know what it is between — what was underway before this year
/// and what the life is walking towards — and it needs that far more than it
/// needs the other three hundred and sixty. The whole outline used to be in the
/// shared prefix for this; a window is the same information at a fiftieth of the
/// tokens, and it is per-fork, which is what the turn is for.
fn outline_window(plan: &Plan, year: i32) -> String {
    let outline = &plan.story.outline;
    let Some(at) = outline.iter().position(|b| b.year == year) else {
        return String::new();
    };
    let from = at.saturating_sub(OUTLINE_WINDOW);
    let to = (at + OUTLINE_WINDOW + 1).min(outline.len());

    let mut out = String::from("\n\nTHE YEARS AROUND IT\n");
    for b in &outline[from..to] {
        let here = if b.year == year { " <- this one" } else { "" };
        out.push_str(&format!(
            "- {} | {} | {}{here}\n",
            b.year, b.title, b.premise
        ));
    }
    out
}

/// How many years either side of a fork's own the turn shows it.
const OUTLINE_WINDOW: usize = 3;

pub fn year_turn(plan: &Plan, year: i32) -> String {
    let beat = plan.story.beat(year);
    let around = outline_window(plan, year);
    // Which kind of time this year was. A year inside a development span is not a year in the
    // world, and a writer told only the date will write it as one.
    let era = match plan.era_for(year) {
        Some(e) if e.kind != crate::lifegen::seed::EraKind::Lived || !e.what.is_empty() => {
            format!(
                "\n\nThis year falls in: {}{}{}",
                e.kind.instruction(),
                if e.what.is_empty() { "" } else { " " },
                e.what
            )
        }
        _ => String::new(),
    };
    format!(
        "Write the year {year}.{}{around}{era}\n\n\
         Begin with a single line `# ` and a title for the year, then the prose: six to \
         ten paragraphs covering the whole year as the character remembers it, in the \
         second person.\n\n\
         **A year is a span, not a scene.** Move through it — what was underway when it \
         began, what changed across it, what recurred, where it had got to by the end. Do \
         not write a single day or one continuous episode; a day written here takes the \
         place of the year, and the days below this phase are what days are for.",
        match beat {
            Some(b) => format!(
                "\n\nThe outline gives this year as:\n{} | {}",
                b.title, b.premise
            ),
            None => String::new(),
        }
    )
}

/// One fork's instruction for a month. Carries its year's text, which is the
/// per-fork half and never enters the shared prefix.
pub fn month_turn(plan: &Plan, year: i32, month: u32) -> String {
    let text = plan
        .year(year)
        .map(|y| y.content.text.trim())
        .unwrap_or_default();
    // **Shown as the section itself, for the reason `story_turn` records.** A
    // model handed "Then, on a line of its own, `## Days` — …" answers with the
    // sentence narrated back and the heading quoted as inline code, and a quoted
    // heading is not a heading: the day list is then read as part of the month's
    // prose and no day is ever written from it.
    format!(
        "THE YEAR {year}, AS ALREADY WRITTEN\n{text}\n\n\
         Write {} of {year}.\n\n\
         Begin with a single line starting `# ` and a title for the month, then three to \
         six paragraphs of prose. Cover the whole month, moving across its weeks — a \
         single episode belongs to one of the days below, not here.\n\n\
         Then this section, in exactly this shape:\n\n\
         ## Days\n\
         One line per day of this month that became a memory, if any did:\n\
         - DAY | Title | what happened, in a sentence\n\
         DAY is the day of the month as a number. Most months have none or one. A month \
         where nothing was memorable still gets its prose; it simply has no days under \
         the heading.\n\n\
         The prose is in the SECOND person — \"You waited\", never the character's name \
         and never \"she\" or \"he\" as the subject.",
        month_name(month)
    )
}

/// One fork's instruction for a day, carrying its month and its required
/// consequences.
pub fn day_turn(plan: &Plan, year: i32, month: u32, day: u32) -> String {
    let m = plan.month(year, month);
    let text = m.map(|m| m.content.text.trim()).unwrap_or_default();
    let d = m.and_then(|m| m.days.iter().find(|d| d.day == day));
    let premise = d
        .map(|d| d.content.title.as_str())
        .filter(|t| !t.is_empty())
        .unwrap_or_default();

    // **The year as well as the month.**
    //
    // A day used to be handed only its month, which left the most concrete rung
    // of the ladder with the narrowest view: a day in September knew what
    // September held and not what the year was doing around it — what had
    // started in the spring, what was coming, which of the character's spans it
    // fell inside. The month is the immediate context and the year is the shape
    // the day is a moment in, and a writer given the second writes the first
    // differently.
    //
    // Both, in order, widest first, so the model reads the year and then narrows
    // to the month it is inside — and above both, in the shared prefix, the
    // character sheet `common` now carries.
    let year_text = plan
        .year(year)
        .map(|y| y.content.text.trim())
        .unwrap_or_default();

    let mut out = String::new();
    if !year_text.is_empty() {
        out.push_str(&format!(
            "THE YEAR {year}, AS ALREADY WRITTEN\n{year_text}\n\n"
        ));
    }
    out.push_str(&format!(
        "{} {year}, AS ALREADY WRITTEN\n{text}\n\n\
         Write the {} of {} {year}",
        month_name(month),
        ordinal(day),
        month_name(month)
    ));
    if !premise.is_empty() {
        out.push_str(&format!(" — the day the month calls \"{premise}\""));
    }
    out.push_str(".\n\n");

    if let Some(cs) = d
        .map(|d| d.consequences.as_slice())
        .filter(|c| !c.is_empty())
    {
        out.push_str("THIS DAY MUST PRODUCE\n");
        for c in cs {
            out.push_str(&format!("- {}\n", describe(c)));
        }
        out.push_str(
            "\nWrite the day so each of those follows from what happened. Do not \
             announce them; earn them.\n\n",
        );
    }
    out.push_str(
        "Begin with a single line `# ` and a title for the day, then three to six \
         paragraphs of prose.",
    );
    out
}

/// One consequence as an instruction to a writer.
///
/// The dials are rendered as words for the same reason
/// [`crate::engine::persona`] renders them as words: a model handed
/// `trust: -0.7` has to invent a scale, and two prompts that invent different
/// ones describe the same number differently.
fn describe(c: &Consequence) -> String {
    let s = |k: &str| c.args.get(k).and_then(|v| v.as_str()).unwrap_or("").trim();
    let n = |k: &str| c.args.get(k).and_then(|v| v.as_f64());
    let who = if s("display").is_empty() {
        s("entity_id")
    } else {
        s("display")
    };
    match c.tool.as_str() {
        "form_belief" => {
            let sure = match n("confidence").unwrap_or(1.0) {
                x if x >= 0.9 => "and they are certain of it",
                x if x >= 0.7 => "and they are fairly sure of it",
                x if x >= 0.4 => "though they are not certain",
                _ => "though they only half believe it",
            };
            format!("A conviction: \"{}\" — {sure}.", s("statement"))
        }
        "form_relationship" => {
            let mut parts = vec![format!("{who} enters this character's life")];
            if let Some(t) = n("trust") {
                parts.push(
                    match t {
                        t if t >= 0.6 => "they come to trust them",
                        t if t >= 0.2 => "they come to take their word",
                        t if t > -0.2 => "trust is not settled either way",
                        t if t > -0.6 => "they do not quite trust them",
                        _ => "they do not trust them at all",
                    }
                    .to_string(),
                );
            }
            if let Some(a) = n("affect") {
                parts.push(
                    match a {
                        a if a >= 0.6 => "they are fond of them",
                        a if a >= 0.2 => "they like them",
                        a if a > -0.2 => "they feel little either way",
                        a if a > -0.6 => "they grate on them",
                        _ => "they cannot stand them",
                    }
                    .to_string(),
                );
            }
            let notes = s("notes");
            if !notes.is_empty() {
                parts.push(notes.to_string());
            }
            format!("{}.", parts.join("; "))
        }
        "revise_relationship" => {
            let mut parts = Vec::new();
            if let Some(t) = n("trust") {
                parts.push(if t < -0.2 {
                    format!("this day costs them their trust in {who}")
                } else if t > 0.2 {
                    format!("this day earns {who} their trust")
                } else {
                    format!("their trust in {who} is unsettled by this day")
                });
            }
            if let Some(a) = n("affect") {
                parts.push(if a < -0.2 {
                    format!("they like {who} less afterwards")
                } else if a > 0.2 {
                    format!("they warm to {who}")
                } else {
                    format!("how they feel about {who} shifts")
                });
            }
            let notes = s("notes");
            if !notes.is_empty() {
                parts.push(notes.to_string());
            }
            if parts.is_empty() {
                parts.push(format!("where they stand with {who} changes"));
            }
            format!("{}.", parts.join("; "))
        }
        "leave_intent" => {
            let until = s("until");
            let mut out = format!("They come away set on this: {}", s("intent"));
            if !until.is_empty() {
                out.push_str(&format!(", until {until}"));
            }
            out.push('.');
            out
        }
        // A tool added to the catalog without a rendering here still reaches
        // the writer, named and with its arguments, rather than silently not
        // being asked for.
        other => match by_name(other) {
            Some(t) => format!("{}: {}", t.description, render_args(c)),
            None => format!("{other}: {}", render_args(c)),
        },
    }
}

fn render_args(c: &Consequence) -> String {
    c.args
        .iter()
        .map(|(k, v)| {
            format!(
                "{k}={}",
                v.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn month_name(month: u32) -> &'static str {
    match month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => "that month",
    }
}

fn ordinal(day: u32) -> String {
    let suffix = match (day % 10, day % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{day}{suffix}")
}

/// Drop a leading reasoning block.
///
/// **A thinking model's reasoning is not the document.** Qwen3.5 opens with `<think>…</think>`
/// and the analysis inside it is about the *task* — "Total = 2792 - 2461 + 1 = 332. Correct." —
/// which is exactly the sort of thing that must never reach a character's memory. Stored
/// verbatim it would be ingested, gathered, and eventually read back to the character as
/// something they remember thinking.
///
/// Tolerant of the block being absent, unterminated, or preceded by whitespace, because all
/// three occur and none of them is a reason to lose the prose.
fn strip_reasoning(raw: &str) -> &str {
    let t = raw.trim_start();
    let Some(rest) = t.strip_prefix("<think>") else {
        return raw;
    };
    match rest.find("</think>") {
        Some(end) => rest[end + "</think>".len()..].trim_start(),
        // Opened and never closed: the whole decode was reasoning and there is no document in
        // it. Better an empty answer the caller reports than a page of the model's notes.
        None => "",
    }
}

/// Drop the code-fence lines a model wraps a long answer in.
///
/// **A fence is markup, not diary prose.** Asked for a page of prose followed by
/// two `##` sections, a model reasonably decides the whole thing is a document
/// and wraps it in ```` ``` ````. Nothing downstream minds — the headings still
/// parse, the outline still lands — except that the fence markers themselves sit
/// in the prose, and the prose is what gets written into
/// `layers/memory/<who>/life-story.md` and read back to that character as their
/// own memory. A life that opens on a row of backticks is a life with markup in
/// it.
///
/// Every line that is *only* a fence goes, wherever it sits, rather than just a
/// matched opening and closing pair: an answer that closes its fence and then
/// opens an empty one has occurred, and a line of backticks is never a sentence
/// somebody wrote about their life, so there is nothing to be careful of.
fn unfence(raw: &str) -> String {
    raw.lines()
        .filter(|l| !l.trim_start().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every heading the ladder asks any rung for.
///
/// Named as a set so a *bare* label — a line that is just `Cast`, with no `##`
/// in front of it — can be recognised as the heading it plainly is. A line of
/// diary prose is never one of these words and nothing else.
const SECTIONS: [&str; 3] = ["Cast", "Years", "Days"];

/// The heading a line names, if it names one.
///
/// # Three spellings, one heading
///
/// Asked for `## Cast`, models have returned all of these, and each one used to
/// lose the entire structured tail — read as prose, pasted onto the end of the
/// character's own life story, with no cast and no outline for the rungs below:
///
/// - `## Cast` — what was asked for.
/// - `` `## Cast` `` — the shape was shown to it as inline code, so it answered
///   in inline code. Fixed at the source too, but the decoration is not
///   information and there is no reason to be strict about it.
/// - `Cast` — no marker at all, the heading as a plain label. One character's
///   decode carried a complete cast and a six-year outline this way and parsed
///   to nothing.
fn heading_name(line: &str) -> Option<String> {
    let t = line.trim().trim_matches('`').trim();
    if let Some(rest) = t.strip_prefix("##") {
        return Some(rest.trim().to_string());
    }
    SECTIONS
        .iter()
        .find(|s| t.eq_ignore_ascii_case(s))
        .map(|s| s.to_string())
}

/// Split a decode into the text before a heading and the lines under it.
///
/// Case-insensitive on the heading and tolerant of leading whitespace, because
/// a model that writes `##Cast` or indents the block has still answered the
/// question. Everything after the *next* heading belongs to that one.
fn section<'a>(raw: &'a str, heading: &str) -> (String, Vec<&'a str>) {
    let mut before = String::new();
    let mut lines = Vec::new();
    let mut inside = false;
    // **The prose ends at the first heading, whichever heading that is.** The
    // test used to be "have we collected any of our own lines yet", which is only
    // the same question when the wanted heading comes first. A decode that writes
    // `## Outline` before `## Cast` — which the story phase asks for in that order
    // — had the whole outline block read as prose, so the arc a year expands
    // arrived with a table of year titles pasted onto the end of it.
    let mut heading_seen = false;
    for line in raw.lines() {
        // A heading is a heading however the model dressed it — see
        // [`heading_name`], which is where the three spellings are recorded.
        let t = line.trim().trim_matches('`').trim();
        if let Some(name) = heading_name(line) {
            inside = name.eq_ignore_ascii_case(heading);
            heading_seen = true;
            continue;
        }
        if inside {
            if !t.is_empty() {
                lines.push(t);
            }
        } else if !heading_seen {
            before.push_str(line);
            before.push('\n');
        }
        // Otherwise: under some other heading. Neither prose nor ours.
    }
    (before.trim().to_string(), lines)
}

/// One `- a | b | c` row, split on pipes with the leading bullet removed.
fn row(line: &str) -> Vec<String> {
    line.trim_start_matches(['-', '*', '•'])
        .trim()
        .split('|')
        .map(|s| s.trim().to_string())
        .collect()
}

/// An entity id as the rest of the life will spell it.
///
/// Normalised here rather than trusted, because it is the join key between a
/// relationship and an NPC, and a stray capital or space in one document is a
/// second person who never appears again.
pub fn slug(raw: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in raw.trim().chars() {
        if c.is_ascii_alphanumeric() {
            if dash && !out.is_empty() {
                out.push('-');
            }
            dash = false;
            out.extend(c.to_lowercase());
        } else {
            dash = true;
        }
    }
    out
}

/// What the story phase produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoryOut {
    pub prose: String,
    pub cast: Vec<CastMember>,
    pub outline: Vec<YearBeat>,
}

/// Read the story decode back.
///
/// Forgiving about a missing field, strict about the shape of the ones present:
/// a cast line without an id is dropped rather than becoming a person called
/// `""`, because that person would be referred to by every later phase.
pub fn parse_story(raw: &str) -> StoryOut {
    let raw = unfence(strip_reasoning(raw));
    let raw = raw.as_str();
    let (prose, cast_lines) = section(raw, "Cast");
    let (_, year_lines) = section(raw, "Years");

    let mut cast = Vec::new();
    for l in cast_lines {
        let parts = row(l);
        let id = slug(parts.first().map(String::as_str).unwrap_or(""));
        if id.is_empty() {
            continue;
        }
        let display = parts
            .get(1)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| id.clone());
        if cast.iter().any(|c: &CastMember| c.entity_id == id) {
            continue;
        }
        cast.push(CastMember {
            entity_id: id,
            display,
            what: parts.get(2).cloned().unwrap_or_default(),
            npc_id: None,
            from_seed: false,
        });
    }

    let mut outline = Vec::new();
    for l in year_lines {
        let parts = row(l);
        let Some(year) = parts.first().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        if outline.iter().any(|b: &YearBeat| b.year == year) {
            continue;
        }
        outline.push(YearBeat {
            year,
            title: parts.get(1).cloned().unwrap_or_default(),
            premise: parts.get(2).cloned().unwrap_or_default(),
        });
    }
    outline.sort_by_key(|b| b.year);

    StoryOut {
        prose,
        cast,
        outline,
    }
}

/// Year beats out of one outline turn.
///
/// # Why the rows are read with no heading required
///
/// [`parse_story`] takes its year lines from under a `Years` heading, because
/// the story answer has three parts and the heading is what separates them. An
/// outline turn asks for one thing — a run of year lines — so a heading is
/// optional decoration, and demanding one would throw away an answer that is
/// entirely correct for want of a label nobody needs. Rows under a heading are
/// still read, for the model that supplies one anyway.
pub fn parse_outline(raw: &str) -> Vec<YearBeat> {
    let raw = unfence(strip_reasoning(raw));
    let (before, under) = section(&raw, "Years");

    let mut out = Vec::new();
    for l in before.lines().chain(under) {
        let parts = row(l);
        let Some(year) = parts.first().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        if out.iter().any(|b: &YearBeat| b.year == year) {
            continue;
        }
        out.push(YearBeat {
            year,
            title: parts.get(1).cloned().unwrap_or_default(),
            premise: parts.get(2).cloned().unwrap_or_default(),
        });
    }
    out.sort_by_key(|b| b.year);
    out
}

/// One turn of the outline, asking for a run of years and no more.
///
/// # Why the outline is built a run at a time
///
/// The story rung used to ask for the whole outline in its one answer, and the
/// answer is where it died: measured across twenty-seven characters, lives of
/// three hundred and sixty-seven years came back with four to eighteen year
/// lines, and two came back with none. Raising the token budget changed nothing,
/// because the model was not running out of room — it was finishing. A request
/// for three hundred lines is not a request a model refuses so much as one it
/// quietly satisfies at a scale of its own choosing.
///
/// A request for twenty is one it completes. So the outline is a sequence of
/// turns, each asking for the next run and each shown where the life had got to,
/// which is also what makes it *evolve* rather than being enumerated: a year
/// written after the twenty before it can follow from them.
///
/// `recent` is a tail, not the whole outline. The context is 6,144 tokens and a
/// finished outline is more than twice that, so a turn that carried every beat
/// so far would stop fitting about a third of the way through the life it was
/// writing.
pub fn outline_turn(display: &str, prose: &str, recent: &[YearBeat], want: &[i32]) -> String {
    let mut out = String::new();

    if !prose.trim().is_empty() {
        out.push_str("THE LIFE, AS ALREADY WRITTEN\n");
        out.push_str(prose.trim());
        out.push_str("\n\n");
    }

    if recent.is_empty() {
        out.push_str("The outline of this life has not been started.\n\n");
    } else {
        out.push_str("THE YEARS ALREADY OUTLINED, MOST RECENT LAST\n");
        for b in recent {
            out.push_str(&format!("- {} | {} | {}\n", b.year, b.title, b.premise));
        }
        out.push_str("\nCarry that forward. What you write next follows from it.\n\n");
    }

    let (first, last) = (
        want.first().copied().unwrap_or_default(),
        want.last().copied().unwrap_or_default(),
    );
    out.push_str(&format!(
        "Write the next {} year(s) of {display}'s life, {first} to {last}, one line each and \
         in this shape:\n\n\
         - YEAR | Title | what this year is for, in a sentence or two\n\n\
         Exactly these years, in order: {}\n\n\
         Write nothing else — no prose, no heading, no commentary. Each year is a step the \
         life takes, not a restatement of who they are: a year that could be any year of \
         this life is a year that has not been written.",
        want.len(),
        want.iter()
            .map(|y| y.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    ));
    out
}

/// A `# Title` first line and the prose under it.
///
/// A decode that forgot the title is not a failure: the prose is the valuable
/// half, and a document titled from its first few words is better than one that
/// did not get written.
pub fn parse_titled(raw: &str) -> (String, String) {
    let raw = unfence(strip_reasoning(raw));
    let raw = raw.trim();
    let mut lines = raw.lines();
    if let Some(first) = lines.next() {
        if let Some(title) = first.trim().strip_prefix('#') {
            let title = title.trim_start_matches('#').trim();
            if !title.is_empty() {
                return (
                    title.to_string(),
                    lines.collect::<Vec<_>>().join("\n").trim().to_string(),
                );
            }
        }
    }
    let fallback: String = raw
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(['.', ',', ';', ':'])
        .to_string();
    (fallback, raw.to_string())
}

/// What a month decode produced: its own document, and the days it marked.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MonthOut {
    pub title: String,
    pub prose: String,
    /// `(day, title)`, in date order, each a real day of that month.
    pub days: Vec<(u32, String)>,
}

/// Read a month decode back, discarding days the calendar does not have.
///
/// A day outside the month is dropped rather than clamped. Clamping would put
/// an episode on a day the model did not choose and nobody asked for, and the
/// document would look deliberate.
pub fn parse_month(raw: &str, year: i32, month: u32) -> MonthOut {
    let raw = unfence(strip_reasoning(raw));
    let raw = raw.as_str();
    let (head, day_lines) = section(raw, "Days");
    let (title, prose) = parse_titled(&head);
    let last = super::calendar::days_in_month(year, month);

    let mut days: Vec<(u32, String)> = Vec::new();
    for l in day_lines {
        let parts = row(l);
        let Some(d) = parts.first().and_then(|s| {
            s.trim_matches(|c: char| !c.is_ascii_digit())
                .parse::<u32>()
                .ok()
        }) else {
            continue;
        };
        if d == 0 || d > last || days.iter().any(|(x, _)| *x == d) {
            continue;
        }
        let title = parts
            .get(1)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("{} {}", month_name(month), d));
        days.push((d, title));
    }
    days.sort_by_key(|(d, _)| *d);

    MonthOut { title, prose, days }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifegen::plan::{Content, Month, NodeId, Year};
    use crate::lifegen::seed::{check, Cadence, EraKind, Seed, WorldEvent};
    use serde_json::json;

    fn seed() -> Seed {
        Seed {
            who: "cindy-tan".into(),
            display: "Cindy Tan".into(),
            born: "1998-09-14".into(),
            through: "2001-03-02".into(),
            place: "Nanyang".into(),
            role: "a records clerk".into(),
            cadence: Cadence::Punctuated,
            facts: vec!["She never learned to swim.".into()],
            world: vec![WorldEvent {
                date: "2001-04-02".into(),
                what: "The east granary burned.".into(),
            }],
            cast: Vec::new(),
            eras: Vec::new(),
        }
    }

    fn plan() -> Plan {
        let mut p = Plan::new(&check(&seed()).unwrap());
        p.story.content = Content {
            title: "An Arc".into(),
            text: "You were born in the rain.".into(),
            edited: false,
            stale: false,
        };
        p.story.cast.push(CastMember {
            entity_id: "prof-lim".into(),
            display: "Professor Lim".into(),
            what: "taught you to question".into(),
            npc_id: Some(3),
            from_seed: false,
        });
        p.story.outline.push(YearBeat {
            year: 1999,
            title: "The Quiet Year".into(),
            premise: "Nothing arrives.".into(),
        });
        p
    }

    fn cons(tool: &str, args: serde_json::Value) -> Consequence {
        Consequence {
            tool: tool.into(),
            args: match args {
                serde_json::Value::Object(m) => m,
                _ => unreachable!(),
            },
        }
    }

    /// **The character's own idiom reaches every phase.**
    ///
    /// These documents are injected into that character's prefix, so a life
    /// written in a neutral literary register is a few thousand tokens of
    /// somebody else's voice teaching the model how to sound.
    #[test]
    fn the_characters_voice_reaches_every_phase() {
        let person = json!({
            "anchor": "You count the rounds and you say the number out loud.",
            "personality": { "voice": "Flat, arithmetical, never hurried." },
        });
        let v = voice_of(&person);
        assert!(v.contains("say the number out loud"), "no anchor: {v}");
        assert!(v.contains("Flat, arithmetical"), "no voice trait: {v}");

        let c = check(&seed()).unwrap();
        let p = plan();
        for prefix in [
            story_prefix(&c, &v),
            years_prefix(&p, &v),
            months_prefix(&p, &v),
            days_prefix(&p, &v),
        ] {
            assert!(prefix.contains("say the number out loud"), "voice missing");
            assert!(prefix.contains("WHOSE VOICE THIS IS IN"), "no heading");
            assert!(prefix.contains("diary"), "not asked for as a diary");
        }
    }

    /// A personality with neither an anchor nor a voice still generates — in the
    /// plain diary register, rather than against an empty heading that reads as
    /// a section somebody forgot to fill in.
    #[test]
    fn a_character_with_no_authored_voice_gets_no_empty_heading() {
        assert_eq!(voice_of(&json!({})), "");
        let p = plan();
        assert!(!years_prefix(&p, "").contains("WHOSE VOICE THIS IS IN"));
        // The diary framing is in VOICE itself, so it survives a missing anchor.
        assert!(years_prefix(&p, "").contains("diary"));
    }

    /// **Each rung has to say how much time it covers.**
    ///
    /// A year and a day are the same instruction otherwise, separated only by a
    /// number the model has no reason to read as a duration — and a year of
    /// Yen's life came back as one morning in a courtyard. The rung that stated
    /// its span ("One day, closely") was the rung that got its span right.
    #[test]
    fn a_year_is_told_it_is_a_span_and_a_month_is_told_its_weeks() {
        let p = plan();
        let y = year_turn(&p, 1999);
        assert!(
            y.contains("whole year"),
            "the year does not claim the year:\n{y}"
        );
        assert!(
            y.to_lowercase().contains("not a scene"),
            "the year is not warned off writing a scene:\n{y}"
        );
        assert!(
            y.contains("second person"),
            "the year does not restate person:\n{y}"
        );

        let m = month_turn(&p, 1999, 9);
        assert!(
            m.contains("whole month"),
            "the month does not claim the month:\n{m}"
        );
        // Case-insensitive: the rungs emphasise the register differently — the
        // month rung shouts it, because it is the last line of that turn and the
        // last instruction read is the one that wins. What matters is that the
        // turn restates it at all, not how loudly.
        assert!(
            m.to_lowercase().contains("second person"),
            "the month does not restate person:\n{m}"
        );
    }

    /// **The voice's example must not be usable as content.**
    ///
    /// It was a sentence about a yard and a burnt granary, and two different
    /// models reproduced it verbatim as the opening line of lives that had
    /// neither — including a combat drone's. A demonstration of style sits in
    /// the same prompt as the material, and a model cannot tell them apart by
    /// position, so the demonstration has to be inert.
    #[test]
    fn the_voice_demonstrates_grammar_without_giving_away_a_scene() {
        // `VOICE` itself, not a rendered prefix: an author's own seed may
        // legitimately name a granary, and that is content the life is supposed
        // to be about. What must not carry an image is the instruction.
        let v = VOICE.to_lowercase();
        for stolen in ["granary", "yard"] {
            assert!(
                !v.contains(stolen),
                "the voice still hands the model a `{stolen}` to write about",
            );
        }
        // The grammatical point still has to be made, and made by contrast.
        assert!(VOICE.contains("second person"), "no voice instruction");
        assert!(VOICE.contains("You waited"), "no example of the right form");
        assert!(VOICE.contains("She waited"), "no example of the wrong form");
    }

    /// **Every rung knows who it is writing about.**
    ///
    /// The shared prefix used to carry only what the story phase produced — the
    /// arc, the cast, the outline — and none of what it was produced from. So a
    /// month knew the shape of the life and not the person living it, and a day,
    /// the most concrete rung of the four, had the least to be concrete with.
    #[test]
    fn every_phase_below_the_story_carries_the_character_sheet() {
        let p = plan();
        for prefix in [
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            assert!(prefix.contains("THE CHARACTER"), "no character sheet");
            assert!(prefix.contains("Cindy Tan"), "no name");
            assert!(prefix.contains("Nanyang"), "no place");
            assert!(prefix.contains("a records clerk"), "no trade");
            // The author's fixed facts are the closest thing the seed has to an
            // anchor, and they are what a day has to stay true to.
            assert!(prefix.contains("never learned to swim"), "no facts");
            // And the world, so a day cannot contradict what other lives in it
            // remember.
            assert!(prefix.contains("east granary"), "no world events");
        }
    }

    /// **A day is written against its year as well as its month.**
    ///
    /// The month is the immediate context; the year is the shape the day is a
    /// moment inside. A day handed only its month writes a scene with no idea
    /// what the year was doing around it.
    #[test]
    fn a_day_carries_both_the_year_and_the_month_above_it() {
        let mut p = plan();
        // `Plan::new` already laid the years out from the seed's span, so this
        // fills the one that is there rather than pushing a second 1999 that
        // `Plan::year` would never reach.
        let y: &mut Year = p
            .years
            .iter_mut()
            .find(|y| y.year == 1999)
            .expect("the seed spans 1999");
        y.content = Content {
            title: "The Quiet Year".into(),
            text: "THE-YEAR-PROSE".into(),
            edited: false,
            stale: false,
        };
        // Same again for the month: the seed's grain may already have laid
        // September out, and a pushed duplicate is one `Plan::month` never
        // returns.
        let content = Content {
            title: "September".into(),
            text: "THE-MONTH-PROSE".into(),
            edited: false,
            stale: false,
        };
        match y.months.iter_mut().find(|m| m.month == 9) {
            Some(m) => m.content = content,
            None => y.months.push(Month {
                month: 9,
                content,
                days: Vec::new(),
            }),
        }
        let turn = day_turn(&p, 1999, 9, 14);
        assert!(
            turn.contains("THE-YEAR-PROSE"),
            "the year is missing:\n{turn}"
        );
        assert!(
            turn.contains("THE-MONTH-PROSE"),
            "the month is missing:\n{turn}"
        );
        // Widest first, so the model reads the year and narrows into the month.
        let (y, m) = (
            turn.find("THE-YEAR-PROSE").unwrap(),
            turn.find("THE-MONTH-PROSE").unwrap(),
        );
        assert!(y < m, "the year should precede the month it contains");
    }

    /// **The load-bearing property of the whole fan-out.** Anything that varies
    /// per fork must be absent from the shared prefix, or every fork attends a
    /// prompt built for one of them.
    #[test]
    fn no_phase_prefix_names_a_specific_year_month_or_day() {
        let p = plan();
        for prefix in [
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            // The outline legitimately lists years; what must not appear is an
            // instruction naming one particular unit to expand.
            for needle in ["January", "the 14th", "Write the year 1999"] {
                assert!(
                    !prefix.contains(needle),
                    "a per-fork detail reached the shared prefix: {needle}"
                );
            }
        }
    }

    /// The prefix carries the cast, and says the lower phases may not add to it
    /// — the rule that stops one professor becoming four slugs.
    #[test]
    fn every_expansion_prefix_carries_the_cast_and_forbids_inventing_people() {
        let p = plan();
        for prefix in [
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            assert!(prefix.contains("prof-lim"));
            assert!(prefix.contains("Do not introduce anyone who is not on this list"));
        }
    }

    /// Every instruction below the story says *expand*, because a child that
    /// invents breaks the parallelism rather than merely reading differently.
    #[test]
    fn the_month_prefix_says_why_inventing_would_break_things() {
        let p = plan();
        let m = months_prefix(&p, "");
        assert!(m.contains("Do not invent events the year does not contain"));
        assert!(m.contains("cannot see what you write"));
        assert!(m.contains("Every month is written"));
    }

    /// The register is not optional: a third-person document is a biography of
    /// somebody the character has never met.
    #[test]
    fn every_prefix_asks_for_the_characters_own_second_person() {
        let c = check(&seed()).unwrap();
        let p = plan();
        for prefix in [
            story_prefix(&c, ""),
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            assert!(prefix.contains("second person"), "{prefix}");
            // The third-person prohibition, made by contrast rather than by
            // naming the grammar — see `VOICE` for why the example is a pronoun
            // and not a scene.
            assert!(prefix.contains("never \"She waited\""));
            assert!(prefix.contains("never the character's name as the"));
        }
    }

    #[test]
    fn the_story_prefix_carries_the_seed_and_the_cadence() {
        let c = check(&seed()).unwrap();
        let s = story_prefix(&c, "");
        assert!(s.contains("Cindy Tan") && s.contains("Nanyang") && s.contains("1998-09-14"));
        assert!(s.contains("She never learned to swim."));
        assert!(s.contains("The east granary burned."));
        assert!(s.contains(Cadence::Punctuated.instruction()));
        // The world events are marked as shared, so a life cannot move one.
        assert!(s.contains("Other people in this world remember them the same way"));
    }

    #[test]
    fn the_story_turn_names_the_whole_span_of_years() {
        let c = check(&seed()).unwrap();
        let t = story_turn(&c);
        assert!(t.contains("from 1998 to 2001 inclusive, none skipped"));
        assert!(t.contains("## Cast"));
        // The outline is `outline_turn`'s, a run of years at a time. Asked for
        // here it came back four to eighteen lines long for lives of three
        // hundred and sixty-seven years, whatever the token budget.
        assert!(
            !t.contains("## Years"),
            "the story turn is asking for the outline again: {t}"
        );
    }

    /// **A gap is named as a gap, and no line is asked for inside it.** Asking for "every
    /// year from first to last" would have Keeper outline 295 years of hibernation — an
    /// account of being unconscious, which is the one thing that cannot be written.
    #[test]
    fn the_story_turn_asks_for_no_outline_inside_a_dormant_span() {
        use crate::lifegen::seed::{Era, EraKind, Grain};
        let mut s = seed();
        s.born = "2461-01-01".into();
        s.through = "3087-01-01".into();
        s.eras = vec![
            Era {
                from: "2461-01-01".into(),
                to: "2792-12-31".into(),
                kind: EraKind::Lived,
                grain: Grain::Years,
                what: "The long watch.".into(),
            },
            Era {
                from: "2793-01-01".into(),
                to: "3087-01-01".into(),
                kind: EraKind::Dormant,
                grain: Grain::Years,
                what: "Hibernation.".into(),
            },
        ];
        let c = check(&s).unwrap();
        let t = story_turn(&c);
        assert!(t.contains("332 year(s) in total"), "{t}");
        assert!(t.contains("skipping 2793 to 3087"), "{t}");
        assert!(!t.contains("none skipped"));

        // The outline itself can no longer stray into the gap by construction
        // rather than by instruction: `build_outline` asks only for the years
        // the plan actually has, and a dormant span has none.
        let want: Vec<i32> = c.years();
        assert!(!want.is_empty());
        assert!(
            !want.iter().any(|y| (2793..=3087).contains(y)),
            "a dormant span produced years to outline"
        );

        // And the prefix tells the writer the same thing, so the arc does not flow across it.
        let p = story_prefix(&c, "");
        assert!(p.contains("HOW THIS EXISTENCE DIVIDES"));
        assert!(p.contains("do not narrate this span"));
        assert!(p.contains("should read as one"));
    }

    /// A fenced answer is a document, not a diary with backticks in it.
    ///
    /// Asked for prose plus two `##` sections, a model wrapped the whole answer
    /// in a code fence and then opened a second, empty one after it. The
    /// headings still parsed and the outline still landed — the only casualty
    /// was the prose, which is the part written into the character's memory and
    /// read back to them.
    #[test]
    fn a_fenced_answer_keeps_its_prose_and_loses_its_fences() {
        let out = parse_story(
            "```\nYou read ground before you read people.\n```\n\n```\n\n\
             ## Years\n- 2751 | Certified | You were counted into the Twelve Hundred.\n",
        );

        assert_eq!(out.prose, "You read ground before you read people.");
        assert!(!out.prose.contains("```"), "markup in a character's memory");
        assert_eq!(
            out.outline.len(),
            1,
            "the outline still parses: {:?}",
            out.outline
        );
        assert_eq!(out.outline[0].year, 2751);
    }

    /// **The shared prefix does not grow with the length of the life.**
    ///
    /// It carried every line of the outline, which was affordable at eight beats
    /// and ruinous at one per year lived: a 367-year life put 4,900 tokens of
    /// year list in front of every fork, and a character's years phase refused
    /// all forty-one of its nodes because the prompt no longer left room for an
    /// answer. A fork's own beat is in its turn; its neighbours are a window.
    #[test]
    fn the_shared_prefix_does_not_carry_the_whole_outline() {
        let mut s = seed();
        s.born = "2719-01-01".into();
        s.through = "3087-01-01".into();
        let mut p = Plan::new(&check(&s).unwrap());

        let short = years_prefix(&p, "").len();
        p.story.outline = (2719..3087)
            .map(|year| YearBeat {
                year,
                title: format!("Year {year}"),
                premise: "You held the ground that cost them most to take.".into(),
            })
            .collect();
        let long = years_prefix(&p, "").len();

        assert!(
            long.abs_diff(short) < 200,
            "the prefix moved {} chars for 368 years of outline",
            long.abs_diff(short)
        );
    }

    /// What a fork actually needs of its neighbours: the years either side, and
    /// which one is its own.
    #[test]
    fn a_year_turn_carries_a_window_of_the_outline_and_not_all_of_it() {
        let mut s = seed();
        s.born = "2719-01-01".into();
        s.through = "3087-01-01".into();
        let mut p = Plan::new(&check(&s).unwrap());
        p.story.outline = (2719..3087)
            .map(|year| YearBeat {
                year,
                title: format!("Year {year}"),
                premise: "Held.".into(),
            })
            .collect();

        let t = year_turn(&p, 2800);
        assert!(t.contains("Year 2800"), "its own beat is missing: {t}");
        assert!(t.contains("<- this one"), "its own beat is unmarked: {t}");
        assert!(t.contains("Year 2797"), "the years before are missing");
        assert!(t.contains("Year 2803"), "the years after are missing");
        assert!(!t.contains("Year 2750"), "the whole outline came along");
        assert!(!t.contains("Year 2850"), "the whole outline came along");
    }

    /// An outline turn asks for one thing, so its rows need no heading over
    /// them. Requiring one would discard an answer that is entirely correct for
    /// want of a label nobody needs.
    #[test]
    fn an_outline_turn_parses_bare_rows() {
        let out = parse_outline(
            "- 2751 | Certified | You were counted into the Twelve Hundred.\n\
             - 2752 | The First Position | You held ground that cost them to take.\n",
        );

        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].year, 2751);
        assert_eq!(out[0].title, "Certified");
        assert_eq!(out[1].year, 2752);
    }

    /// A model that supplies a heading anyway is not punished for it.
    #[test]
    fn an_outline_turn_parses_rows_under_a_heading() {
        let out = parse_outline("## Years\n- 2751 | Certified | You were counted in.\n");
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].year, 2751);
    }

    /// The turn carries the years it wants, the arc they follow, and the tail of
    /// what has already been written — which is what lets a year follow from the
    /// ones before it rather than being invented beside them.
    #[test]
    fn an_outline_turn_names_its_run_and_what_came_before() {
        let recent = vec![YearBeat {
            year: 2750,
            title: "The Floor".into(),
            premise: "You were tested.".into(),
        }];
        let t = outline_turn("Rook", "You read ground.", &recent, &[2751, 2752]);

        assert!(t.contains("You read ground."), "the arc is missing: {t}");
        assert!(t.contains("2750 | The Floor"), "the tail is missing: {t}");
        assert!(t.contains("Carry that forward"));
        assert!(t.contains("2751, 2752"), "the run is not named: {t}");
        assert!(t.contains("Write the next 2 year(s)"), "{t}");
    }

    /// The first run has no tail, and has to say so rather than opening on an
    /// empty heading that reads as a section the model failed to receive.
    #[test]
    fn the_first_outline_turn_says_the_outline_is_unstarted() {
        let t = outline_turn("Rook", "You read ground.", &[], &[2751]);
        assert!(t.contains("has not been started"), "{t}");
        assert!(!t.contains("Carry that forward"));
    }

    /// A bare label is a heading too. One character's decode carried a full cast
    /// and a six-year outline written as plain `Cast` and `Years` lines, and
    /// parsed to nothing at all: the whole tail was read as prose and pasted
    /// onto the end of that character's own life story.
    #[test]
    fn an_unmarked_heading_is_still_a_heading() {
        let out = parse_story(
            "You laid fire and watched it take.\n\n\
             Cast\n- hart | Hart | brought you to the Volunteer Call\n\n\
             Years\n- 2722 | Workshop Years | You begin as a new beam.\n",
        );

        assert_eq!(out.prose, "You laid fire and watched it take.");
        assert_eq!(out.cast.len(), 1, "cast: {:?}", out.cast);
        assert_eq!(out.cast[0].entity_id, "hart");
        assert_eq!(out.outline.len(), 1, "outline: {:?}", out.outline);
        assert_eq!(out.outline[0].year, 2722);
    }

    /// Only an exact section label counts. A sentence that happens to contain
    /// the word is prose, and losing a paragraph to a false heading would be a
    /// worse failure than the one this catches.
    #[test]
    fn a_word_inside_a_sentence_is_not_a_heading() {
        assert_eq!(heading_name("The cast of your life was small."), None);
        assert_eq!(heading_name("Years passed."), None);
        assert_eq!(heading_name("  Cast  "), Some("Cast".to_string()));
        assert_eq!(heading_name("`## Years`"), Some("Years".to_string()));
    }

    /// A model shown its output shape as inline code answers in inline code.
    /// The quoting is decoration, so the structured tail still has to parse —
    /// otherwise the whole outline is read as prose and pasted onto the life.
    #[test]
    fn a_backtick_quoted_heading_is_still_a_heading() {
        let out = parse_story(
            "You read ground before you read people.\n\n\
             `## Cast`\n`- vasko | Vasko | certified you`\n\n\
             `## Years`\n`- 2751 | Certified | You were counted in.`\n",
        );

        assert_eq!(out.prose, "You read ground before you read people.");
        assert_eq!(out.cast.len(), 1, "cast: {:?}", out.cast);
        assert_eq!(out.cast[0].entity_id, "vasko");
        assert_eq!(out.outline.len(), 1, "outline: {:?}", out.outline);
        assert_eq!(out.outline[0].year, 2751);
    }

    /// The turn shows the two sections as themselves rather than describing them
    /// as steps. Described, a model narrates the description back — "Then, on a
    /// line of its own:" arrived at the top of a life story, with every heading
    /// quoted as code beneath it.
    #[test]
    fn the_story_turn_shows_the_sections_rather_than_narrating_them() {
        let mut s = seed();
        s.display = "Rook".into();
        let t = story_turn(&check(&s).unwrap());

        assert!(
            t.contains("\n## Cast\n"),
            "the heading is not shown as one: {t}"
        );
        assert!(
            !t.contains('`'),
            "a backtick in the turn is a backtick in the answer: {t}"
        );
        assert!(
            !t.contains("on a line of its own"),
            "the turn still describes its own steps, which get narrated back: {t}"
        );
    }

    /// The month rung shows its section the same way the story rung does, and
    /// for the same reason: a described heading comes back quoted, and a quoted
    /// heading is not a heading, so no day is ever written from the list.
    #[test]
    fn the_month_turn_shows_its_section_rather_than_narrating_it() {
        let mut p = Plan::new(&check(&seed()).unwrap());
        if let Some(y) = p.years.first_mut() {
            y.content.text = "You held the line.".into();
        }
        let year = p.years.first().map(|y| y.year).unwrap_or_default();
        let t = month_turn(&p, year, 3);

        assert!(
            t.contains("\n## Days\n"),
            "the heading is not shown as one: {t}"
        );
        assert!(
            !t.contains("on a line of its own"),
            "the turn describes its own steps, which get narrated back: {t}"
        );
        assert!(
            t.contains("SECOND person"),
            "the register is not restated: {t}"
        );
    }

    /// The turn is written *about* the character throughout — it has to be — so
    /// the register has to be restated in it, and last, or the answer comes back
    /// as a biography. It did: "She was ground, position, and the reading of a
    /// room."
    #[test]
    fn the_story_turn_ends_by_naming_the_second_person() {
        let t = story_turn(&check(&seed()).unwrap());

        assert!(t.contains("SECOND person"), "{t}");
        let tail = t.trim_end();
        let last = tail.lines().last().unwrap_or_default();
        assert!(
            last.contains("diary") || last.contains("SECOND person"),
            "the register is not the last thing read, so it loses: {last}"
        );
    }

    /// Only a line that is *entirely* a fence goes. A sentence that happens to
    /// contain backticks is prose and stays whole.
    #[test]
    fn unfencing_leaves_prose_alone() {
        assert_eq!(
            unfence("You waited.\nIt did not stop."),
            "You waited.\nIt did not stop."
        );
        assert_eq!(unfence("```rust\nYou waited.\n```"), "You waited.");
    }

    /// An era note may not be a sentence the diary could have written.
    ///
    /// A model handed `"They were being developed inside a simulation for this
    /// — conscious, but in a world with rules somebody else set…"` moved it to
    /// the second person and used it as narration, three times in one story.
    /// The tell is grammatical, so the test is too: an era note opens on a
    /// fragment or an imperative, never on a pronoun with a past-tense verb
    /// behind it, and it carries an instruction to the writer.
    #[test]
    fn an_era_note_reads_as_an_instruction_and_not_as_prose() {
        for kind in [EraKind::Lived, EraKind::Developed, EraKind::Dormant] {
            let note = kind.instruction();

            let opener = note.split_whitespace().next().unwrap().to_lowercase();
            assert!(
                !["they", "he", "she", "you", "it"].contains(&opener.as_str()),
                "{kind:?} note opens on a pronoun, so it reads as narration: {note}"
            );

            // Every note tells the writer what to do with the span.
            let lower = note.to_lowercase();
            assert!(
                lower.contains("write") || lower.contains("do not narrate"),
                "{kind:?} note gives the writer no instruction: {note}"
            );
        }
    }

    /// A fork's turn carries the parent text, which is exactly what must NOT be
    /// in the prefix.
    #[test]
    fn a_month_turn_carries_its_own_years_text_and_names_the_month() {
        let mut p = plan();
        p.content_mut(NodeId::Year { year: 1999 })
            .unwrap()
            .generated("The Quiet Year".into(), "You learned to wait.".into());
        let t = month_turn(&p, 1999, 1);
        assert!(t.contains("You learned to wait."));
        assert!(t.contains("Write January of 1999"));
        assert!(t.contains("## Days"));
    }

    #[test]
    fn a_year_turn_carries_the_beat_the_story_assigned_it() {
        let t = year_turn(&plan(), 1999);
        assert!(t.contains("Write the year 1999"));
        assert!(t.contains("The Quiet Year | Nothing arrives."));
    }

    /// **The inversion.** The consequences are an input to the day, rendered as
    /// direction to a writer rather than as dials nobody has a scale for.
    #[test]
    fn a_day_turn_asks_the_prose_to_earn_its_consequences() {
        let mut p = plan();
        p.content_mut(NodeId::Month {
            year: 1999,
            month: 1,
        })
        .unwrap()
        .generated("January".into(), "The month was cold.".into());
        let d = p.ensure_day(1999, 1, 14).unwrap();
        d.content.title = "The Argument".into();
        d.consequences = vec![
            cons(
                "form_belief",
                json!({"statement": "Hess burned the east granary", "confidence": 0.95}),
            ),
            cons(
                "revise_relationship",
                json!({"entity_id": "hess", "trust": -0.8}),
            ),
        ];
        let t = day_turn(&p, 1999, 1, 14);

        assert!(t.contains("The month was cold."), "carries its month");
        assert!(t.contains("Write the 14th of January 1999"));
        assert!(t.contains("the day the month calls \"The Argument\""));
        assert!(t.contains("THIS DAY MUST PRODUCE"));
        assert!(t.contains("\"Hess burned the east granary\" — and they are certain of it."));
        assert!(t.contains("this day costs them their trust in hess"));
        assert!(t.contains("Do not announce them; earn them."));
        // No raw dials reach the writer — a model handed -0.8 has to invent a
        // scale, and two prompts would invent different ones.
        assert!(!t.contains("-0.8") && !t.contains("0.95"));
    }

    /// A day with nothing required still gets written; the section is simply
    /// absent rather than empty.
    #[test]
    fn a_day_with_no_required_consequences_omits_the_section() {
        let mut p = plan();
        p.ensure_day(1999, 1, 3).unwrap();
        let t = day_turn(&p, 1999, 1, 3);
        assert!(!t.contains("THIS DAY MUST PRODUCE"));
        assert!(t.contains("Write the 3rd of January 1999"));
    }

    #[test]
    fn ordinals_read_the_way_a_person_says_them() {
        let cases = [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (21, "21st"),
            (22, "22nd"),
            (31, "31st"),
        ];
        for (d, want) in cases {
            assert_eq!(ordinal(d), want);
        }
    }

    /// **A thinking model's reasoning is not the document.** Stored verbatim it would be
    /// ingested and eventually read back to the character as something they remember thinking —
    /// and the real one contained "Total = 2792 - 2461 + 1 = 332. Correct."
    #[test]
    fn a_reasoning_block_never_reaches_the_document() {
        let raw = "<think>\nTotal = 2792 - 2461 + 1 = 332. Correct.\n</think>\n\n\
                   You were built to remember.\n\n## Cast\n- kaelor | Kaelor | the commander\n";
        let out = parse_story(raw);
        assert_eq!(out.prose, "You were built to remember.");
        assert!(!out.prose.contains("think"));
        assert_eq!(out.cast.len(), 1);

        // The other two parsers strip it too — a year or a month decode reasons just as much.
        let (t, p) = parse_titled("<think>weighing it up</think>\n# A Year\n\nIt rained.");
        assert_eq!((t.as_str(), p.as_str()), ("A Year", "It rained."));
        assert_eq!(
            parse_month("<think>x</think>\n# May\n\nCold.", 1998, 5).title,
            "May"
        );

        // Absent, and unterminated — the second is a decode that was *all* reasoning, and an
        // empty answer the caller reports beats a page of the model's notes.
        assert_eq!(parse_titled("# Plain\n\nProse.").0, "Plain");
        assert_eq!(parse_story("<think>never closed").prose, "");
    }

    /// **The turn names its subject.** Without it a thinking model spends its reasoning on the
    /// shape of the answer and then invents the person — it produced a woman called Elara
    /// Vance on Mars when asked for Keeper.
    #[test]
    fn the_story_turn_names_who_it_is_about() {
        let c = check(&seed()).unwrap();
        let t = story_turn(&c);
        assert!(t.contains("Write the life of Cindy Tan, a records clerk"));
        assert!(t.contains("the character described above"));
        assert!(t.contains("Do not invent a different person"));
        assert!(t.contains("the arc of Cindy Tan's life"));
    }

    #[test]
    fn a_story_decode_reads_back_into_prose_cast_and_outline() {
        let raw = "\
You were born in the rain.

It did not stop for a week.

## Cast
- prof-lim | Professor Lim | taught you to question assumptions
- hess | Hess | the man who burned the granary

## Years
- 1998 | Born | You arrive.
- 1999 | The Quiet Year | Nothing arrives.
";
        let out = parse_story(raw);
        assert_eq!(
            out.prose,
            "You were born in the rain.\n\nIt did not stop for a week."
        );
        assert_eq!(out.cast.len(), 2);
        assert_eq!(out.cast[0].entity_id, "prof-lim");
        assert_eq!(out.cast[0].display, "Professor Lim");
        assert_eq!(out.cast[1].what, "the man who burned the granary");
        assert_eq!(
            out.outline.iter().map(|b| b.year).collect::<Vec<_>>(),
            vec![1998, 1999]
        );
        assert_eq!(out.outline[1].title, "The Quiet Year");
    }

    /// **The id is the join key between a relationship and an NPC**, so it is
    /// normalised rather than trusted — a stray capital is a second person who
    /// never appears again.
    #[test]
    fn cast_ids_are_normalised_and_duplicates_dropped() {
        let out = parse_story(
            "p\n## Cast\n- Prof Lim | Professor Lim | x\n- prof-lim | Again | y\n- | Nameless | z\n",
        );
        assert_eq!(out.cast.len(), 1);
        assert_eq!(out.cast[0].entity_id, "prof-lim");
        assert_eq!(out.cast[0].display, "Professor Lim");
    }

    #[test]
    fn slugs_are_lowercase_hyphenated_and_free_of_punctuation() {
        assert_eq!(slug("Professor Lim"), "professor-lim");
        assert_eq!(slug("  O'Brien, Jr. "), "o-brien-jr");
        assert_eq!(slug("已知"), "");
        assert_eq!(slug("hess"), "hess");
    }

    /// A heading a model indented or ran together is still an answer.
    #[test]
    fn section_headings_are_read_leniently() {
        let out = parse_story("p\n  ##cast\n- a | A | x\n  ## YEARS \n- 1998 | T | P\n");
        assert_eq!(out.cast.len(), 1);
        assert_eq!(out.outline.len(), 1);
    }

    /// **The prose ends at the first heading, whichever heading that is.**
    ///
    /// The prose was collected until the wanted section started producing lines,
    /// which is only the same rule when the wanted heading comes first. A decode
    /// that puts `## Years` before `## Cast` had its whole year table read as
    /// prose — and the arc is what every year of the life is then expanded from,
    /// so twelve forks each opened with a list of year titles pasted onto the end
    /// of the story they were meant to be continuing.
    #[test]
    fn a_section_written_out_of_order_does_not_become_prose() {
        let out = parse_story(
            "She grew up in the delta.\n\n\
             ## Years\n- 1998 | Arrival | She arrives.\n\n\
             ## Cast\n- lim | Professor Lim | her tutor\n",
        );
        assert_eq!(out.prose, "She grew up in the delta.");
        assert_eq!(out.cast.len(), 1);
        assert_eq!(out.outline.len(), 1);
    }

    /// The same rule for a month: the day list is a section, not part of the
    /// month's prose, whichever order the model writes them in.
    #[test]
    fn a_months_prose_stops_at_the_first_heading() {
        let out = parse_month(
            "# A Quiet January\n\nNothing much happened.\n\n\
             ## Notes\nignore me\n\n\
             ## Days\n- 3 | the letter\n",
            1999,
            1,
        );
        assert!(
            !out.prose.contains("ignore me"),
            "another section's body became the month's prose: {:?}",
            out.prose
        );
        assert_eq!(out.days.len(), 1);
    }

    #[test]
    fn a_titled_document_splits_into_its_title_and_prose() {
        let (t, p) = parse_titled("# First Week at Nanyang\n\nYou arrived late.\n");
        assert_eq!(t, "First Week at Nanyang");
        assert_eq!(p, "You arrived late.");
    }

    /// A decode that forgot the title is not a failure — the prose is the
    /// valuable half, and a document titled from its opening beats one that did
    /// not get written.
    #[test]
    fn a_document_with_no_title_line_is_titled_from_its_opening() {
        let (t, p) = parse_titled("You came back to the yard and it was gone.");
        assert_eq!(t, "You came back to the yard");
        assert_eq!(p, "You came back to the yard and it was gone.");
    }

    #[test]
    fn a_month_decode_reads_back_with_its_notable_days() {
        let raw = "\
# First Term

You arrived late and stayed late.

## Days
- 14 | First Week at Nanyang | You met Lim.
- 30 | The Argument | It ended badly.
";
        let out = parse_month(raw, 1998, 9);
        assert_eq!(out.title, "First Term");
        assert_eq!(out.prose, "You arrived late and stayed late.");
        assert_eq!(
            out.days,
            vec![
                (14, "First Week at Nanyang".to_string()),
                (30, "The Argument".to_string())
            ]
        );
    }

    /// **A day outside the month is dropped, not clamped.** Clamping would put
    /// an episode on a day nobody chose, and the document would look
    /// deliberate.
    #[test]
    fn a_day_the_month_does_not_have_is_dropped() {
        let out = parse_month(
            "# F\n\nP.\n\n## Days\n- 31 | Nope | x\n- 0 | Nope | x\n- 12 | Yes | x\n",
            1999,
            2,
        );
        assert_eq!(out.days, vec![(12, "Yes".to_string())]);
        // February 2000 is a leap year, so the 29th is real.
        let leap = parse_month("# F\n\nP.\n\n## Days\n- 29 | Yes | x\n", 2000, 2);
        assert_eq!(leap.days.len(), 1);
    }

    #[test]
    fn a_month_with_no_notable_days_still_has_its_prose() {
        let out = parse_month("# A Quiet Month\n\nNothing came.\n", 1998, 9);
        assert_eq!(out.title, "A Quiet Month");
        assert_eq!(out.prose, "Nothing came.");
        assert!(out.days.is_empty());
    }

    #[test]
    fn duplicate_and_unordered_days_are_deduplicated_and_sorted() {
        let out = parse_month(
            "# F\n\nP.\n\n## Days\n- 30 | Late | x\n- 14 | Early | x\n- 14 | Again | x\n",
            1998,
            9,
        );
        assert_eq!(out.days, vec![(14, "Early".into()), (30, "Late".into())]);
    }

    /// A catalog tool with no bespoke rendering still reaches the writer named
    /// and with its arguments, rather than silently not being asked for.
    #[test]
    fn every_catalog_tool_renders_as_an_instruction() {
        for t in crate::engine::authoring::CATALOG {
            let parsed =
                crate::engine::authoring::parse(&format!("<tool_call>{}</tool_call>", t.example));
            let c = Consequence {
                tool: parsed.calls[0].tool.to_string(),
                args: parsed.calls[0].args.clone(),
            };
            let d = describe(&c);
            assert!(!d.trim().is_empty(), "{} renders as nothing", t.name);
            assert!(!d.contains("0."), "{} leaked a raw dial: {d}", t.name);
        }
    }
}
