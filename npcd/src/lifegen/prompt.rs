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

use super::consequence::Consequence;
use super::plan::{CastMember, Plan, YearBeat};
use super::seed::Checked;
use crate::engine::authoring::by_name;

/// How the daemon addresses the writer in every phase.
const VOICE: &str = "\
You are writing one character's remembered life, for a game engine that will \
read it back as that character's own memory.

Write in the second person, as the character's memory of their own life — \
\"You came back to the yard and the granary was gone.\" Never third person, \
never their name as a subject. Plain past tense. No headings inside the prose, \
no bullet lists, no commentary about the writing.";

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
    if c.eras.len() > 1 || c.eras.iter().any(|e| !e.kind.is_written()) {
        out.push_str("\nHOW THIS EXISTENCE DIVIDES\n");
        for e in &c.eras {
            let span = e.to.year - e.from.year;
            out.push_str(&format!(
                "- {} to {} ({} year{}): {}{}{}\n",
                e.from,
                e.to,
                span.max(1),
                if span == 1 { "" } else { "s" },
                e.kind.instruction(),
                if e.what.is_empty() { "" } else { " " },
                e.what,
            ));
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
pub fn story_prefix(c: &Checked) -> String {
    format!(
        "{VOICE}\n\n{}\n\nYOUR TASK\n\
         Write the shape of this whole life, then name its people, then divide it into \
         years. Everything below this phase expands what you decide here and invents \
         nothing of its own, so anything you leave out will not appear later.\n\n\
         {}\n",
        who(c),
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
         Write three things, in this order.\n\n\
         First, the arc of {}'s life in four to eight paragraphs of prose: what formed \
         them, what they came to want, what it cost them, and who they are by \
         {last}.\n\n",
        s.display, s.role, s.display,
    ) + &format!(
        "Then, on a line of its own, `## Cast` — every person who matters to this life, \
         one per line as:\n\
         `- entity-id | Display Name | who they are to this character`\n\
         The id is lowercase with hyphens and is how every later part of the life will \
         refer to them, so choose it once and use nothing else.\n\n\
         Then, on a line of its own, `## Years` — one line for {listed}:\n\
         `- YEAR | Title | what this year is for, in a sentence or two`\n\
         Give no line at all for a year they were not there for. A skipped span is not a \
         quiet stretch of their life; it is a stretch that is not in their life.\n"
    )
}

/// Everything below the story shares this: the arc, the cast and the outline.
///
/// One function rather than three near-copies, because the three phases must
/// agree about what a name refers to, and the surest way to guarantee that is
/// for them to be reading the same text.
fn common(plan: &Plan) -> String {
    let mut out = String::new();
    out.push_str(VOICE);
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
    out.push_str("\nTHE YEARS\n");
    for b in &plan.story.outline {
        out.push_str(&format!("- {} | {} | {}\n", b.year, b.title, b.premise));
    }
    out
}

/// The years phase's prefix.
pub fn years_prefix(plan: &Plan) -> String {
    format!(
        "{}\n\
         YOUR TASK\n\
         You are writing out ONE year of this life in full. The year has already been \
         decided above — you are expanding it, not choosing it. Do not add events the \
         outline does not give this year, and do not reach into another year.\n",
        common(plan)
    )
}

/// The months phase's prefix.
pub fn months_prefix(plan: &Plan) -> String {
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
        common(plan)
    )
}

/// The days phase's prefix.
pub fn days_prefix(plan: &Plan) -> String {
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
        common(plan)
    )
}

/// One fork's instruction for a year.
pub fn year_turn(plan: &Plan, year: i32) -> String {
    let beat = plan.story.beat(year);
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
        "Write the year {year}.{}{era}\n\n\
         Begin with a single line `# ` and a title for the year, then the prose: six to \
         ten paragraphs covering the whole year as the character remembers it.",
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
    format!(
        "THE YEAR {year}, AS ALREADY WRITTEN\n{text}\n\n\
         Write {} of {year}.\n\n\
         Begin with a single line `# ` and a title for the month, then three to six \
         paragraphs of prose.\n\n\
         Then, on a line of its own, `## Days` — the days of this month that became \
         memories, if any did:\n\
         `- DAY | Title | what happened, in a sentence`\n\
         DAY is the day of the month as a number. Most months have none or one. A month \
         where nothing was memorable still gets its prose; it simply has no days under \
         the heading.",
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

    let mut out = format!(
        "{} {year}, AS ALREADY WRITTEN\n{text}\n\n\
         Write the {} of {} {year}",
        month_name(month),
        ordinal(day),
        month_name(month)
    );
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

/// Split a decode into the text before a `## Heading` and the lines under it.
///
/// Case-insensitive on the heading and tolerant of leading whitespace, because
/// a model that writes `##Cast` or indents the block has still answered the
/// question. Everything after the *next* `##` heading belongs to that one.
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
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("##") {
            let name = rest.trim();
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
    let raw = strip_reasoning(raw);
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

/// A `# Title` first line and the prose under it.
///
/// A decode that forgot the title is not a failure: the prose is the valuable
/// half, and a document titled from its first few words is better than one that
/// did not get written.
pub fn parse_titled(raw: &str) -> (String, String) {
    let raw = strip_reasoning(raw).trim();
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
    let raw = strip_reasoning(raw);
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
    use crate::lifegen::plan::{Content, NodeId};
    use crate::lifegen::seed::{check, Cadence, Seed, WorldEvent};
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

    /// **The load-bearing property of the whole fan-out.** Anything that varies
    /// per fork must be absent from the shared prefix, or every fork attends a
    /// prompt built for one of them.
    #[test]
    fn no_phase_prefix_names_a_specific_year_month_or_day() {
        let p = plan();
        for prefix in [years_prefix(&p), months_prefix(&p), days_prefix(&p)] {
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
        for prefix in [years_prefix(&p), months_prefix(&p), days_prefix(&p)] {
            assert!(prefix.contains("prof-lim"));
            assert!(prefix.contains("Do not introduce anyone who is not on this list"));
        }
    }

    /// Every instruction below the story says *expand*, because a child that
    /// invents breaks the parallelism rather than merely reading differently.
    #[test]
    fn the_month_prefix_says_why_inventing_would_break_things() {
        let p = plan();
        let m = months_prefix(&p);
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
            story_prefix(&c),
            years_prefix(&p),
            months_prefix(&p),
            days_prefix(&p),
        ] {
            assert!(prefix.contains("second person"), "{prefix}");
            assert!(prefix.contains("Never third person"));
        }
    }

    #[test]
    fn the_story_prefix_carries_the_seed_and_the_cadence() {
        let c = check(&seed()).unwrap();
        let s = story_prefix(&c);
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
        assert!(t.contains("## Cast") && t.contains("## Years"));
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
        assert!(t.contains("not in their life"));
        assert!(!t.contains("none skipped"));

        // And the prefix tells the writer the same thing, so the arc does not flow across it.
        let p = story_prefix(&c);
        assert!(p.contains("HOW THIS EXISTENCE DIVIDES"));
        assert!(p.contains("They were archived for this"));
        assert!(p.contains("should read as one"));
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
