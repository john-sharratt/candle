//! Renders consequences and tool calls as instructions to a writer, so that the
//! writer is never handed raw numbers.
//!
//! Dials are rendered as words, since a writer handed a value such as a trust of
//! minus 0.7 has to invent a scale, and two prompts would invent different ones. A
//! tool with no bespoke rendering still reaches the writer, named and with its
//! arguments, rather than being left out silently. Month names and ordinals are
//! spelled the way a person would say them.

use super::*;

/// One consequence as an instruction to a writer.
///
/// The dials are rendered as words for the same reason
/// [`crate::engine::persona`] renders them as words: a model handed
/// `trust: -0.7` has to invent a scale, and two prompts that invent different
/// ones describe the same number differently.
pub(super) fn describe(c: &Consequence) -> String {
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

pub(super) fn render_args(c: &Consequence) -> String {
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

pub(super) fn ordinal(day: u32) -> String {
    let suffix = match (day % 10, day % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{day}{suffix}")
}
