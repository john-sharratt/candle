//! `zenj_now`: what is playing on each deck and where in it, who has the
//! decks, the next track and why, and anything wrong.

use std::fmt::Write;

use serde_json::{json, Value};

use super::lights::line;
use super::{Answer, Dj, Tool};
use crate::lights::View;
use crate::live::Live;

pub struct Now;

impl Tool for Now {
    const NAME: &'static str = "zenj_now";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "What is playing at the party now: each deck's track, how far in it is and which is on air; who has the decks (human, assist or zenj); the next track and why it was chosen; requests waiting; and any fault.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": true},
        })
    }
}

/// The view, said.
pub fn say(l: &Live) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Hands: {}{}",
        l.holder.name(),
        if l.automix { " (Automix on)" } else { "" }
    );
    for d in &l.decks {
        if d.title.is_empty() {
            let _ = writeln!(s, "Deck {}: empty", d.number);
            continue;
        }
        let air = if d.on_air { "ON AIR " } else { "" };
        let left = d
            .bars_left
            .map(|b| format!(", {b:.0} bars left"))
            .unwrap_or_default();
        let _ = writeln!(
            s,
            "Deck {}: {air}{} — {} ({:.1} BPM{}{left})",
            d.number,
            d.artist,
            d.title,
            d.bpm,
            if d.playing { "" } else { ", stopped" }
        );
    }
    match &l.next {
        Some(n) => {
            let by = n
                .requested_by
                .as_deref()
                .map(|b| format!(" — {b}'s request"))
                .unwrap_or_default();
            let _ = writeln!(s, "Next: {} — {}{by}. Why: {}", n.artist, n.title, n.why);
            if let Some(style) = &n.style {
                let _ = writeln!(s, "Armed: {style}");
            }
        }
        None => {
            let _ = writeln!(s, "Next: not chosen yet");
        }
    }
    if !l.requests.is_empty() {
        let _ = writeln!(s, "Requests waiting: {}", l.requests.join("; "));
    }
    let _ = writeln!(
        s,
        "Target energy {:.2}, the room {:.2}; {} to midnight",
        l.target_energy,
        l.landing,
        to_midnight(l.to_midnight_s)
    );
    if let Some(b) = &l.blind {
        let _ = writeln!(s, "The deck reader is blind: {b}");
    }
    if let Some(f) = &l.fault {
        let _ = writeln!(s, "Fault: {f}");
    }
    s.trim_end().to_string()
}

fn to_midnight(s: i64) -> String {
    if s < 0 {
        return "past".into();
    }
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

pub fn call(dj: &Dj) -> Answer {
    let lights = match &dj.lights {
        Some(l) => line(&l.view.load()),
        None => line(&View::default()),
    };
    Answer::ok(format!("{}\n{lights}", say(&dj.live.load())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hands::Holder;
    use crate::live::{DeckView, NextView};

    #[test]
    fn the_view_is_said_plainly() {
        let l = Live {
            holder: Holder::Zenj,
            decks: vec![
                DeckView {
                    number: 1,
                    title: "One More Time".into(),
                    artist: "Daft Punk".into(),
                    bpm: 122.0,
                    beat: Some(100.0),
                    bars_left: Some(40.0),
                    playing: true,
                    on_air: true,
                },
                DeckView {
                    number: 2,
                    ..DeckView::default()
                },
            ],
            next: Some(NextView {
                title: "Dancing Queen".into(),
                artist: "ABBA".into(),
                why: "Yen's mum is here".into(),
                requested_by: Some("yen".into()),
                style: Some("a → b (blend)".into()),
            }),
            target_energy: 0.72,
            landing: 0.6,
            to_midnight_s: 4_830,
            ..Live::default()
        };
        assert_eq!(
            say(&l),
            "Hands: zenj\n\
             Deck 1: ON AIR Daft Punk — One More Time (122.0 BPM, 40 bars left)\n\
             Deck 2: empty\n\
             Next: ABBA — Dancing Queen — yen's request. Why: Yen's mum is here\n\
             Armed: a → b (blend)\n\
             Target energy 0.72, the room 0.60; 1:20:30 to midnight"
        );
    }
}
