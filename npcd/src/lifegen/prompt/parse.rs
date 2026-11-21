//! Reads the structured answers of each lifegen phase back into typed values,
//! forgiving about spacing and strict about shape.
//!
//! A cast line without an id is dropped rather than made into a person with an
//! empty name, because every later phase would refer to that person. Outline rows
//! are read with or without a heading, since an outline turn asks for one thing,
//! and a correct answer should not be lost for want of a label. A month's days
//! that the month does not have are dropped rather than clamped, so an episode
//! never lands on a day nobody chose.

use super::*;

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
    let last = super::super::calendar::days_in_month(year, month);

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
