//! The day, month, year and outline instructions in the lifegen phases, built
//! from the arc and the year-by-year plan.
//!
//! The turn is where per-fork content goes, which keeps it out of the shared
//! prefix. It names the subject, the span, the context and the register, and it
//! shows its output sections as they are, since a described heading comes back
//! quoted. The outline is built as a sequence of turns, each shown the recent
//! tail of the life, so that a year can follow from the years before it.

use super::*;

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
    let gaps: Vec<&super::super::seed::CheckedEra> =
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

/// One fork's instruction for a year.
/// The outline either side of `year`, as the years around it.
///
/// A fork needs to know what it is between — what was underway before this year
/// and what the life is walking towards — and it needs that far more than it
/// needs the other three hundred and sixty. The whole outline used to be in the
/// shared prefix for this; a window is the same information at a fiftieth of the
/// tokens, and it is per-fork, which is what the turn is for.
pub(super) fn outline_window(plan: &Plan, year: i32) -> String {
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
pub(super) const OUTLINE_WINDOW: usize = 3;

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
