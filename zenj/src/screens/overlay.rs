//! The layer ZenJ owns on every screen (§8): now playing and next, a
//! request coming up, Zen's one-line reason, the mode — and the midnight
//! countdown, which takes the whole screen from 23:59:00 and says the new
//! year for a minute after.

use chrono::{DateTime, Local};

use crate::hands::Mode;
use crate::live::Live;

/// When the countdown takes over, in seconds before midnight.
pub const COUNTDOWN_S: i64 = 60;
/// How long the new year is said after.
pub const AFTER_S: i64 = 60;

/// What a screen shows over its scene.
#[derive(Debug, Clone, PartialEq)]
pub enum Overlay {
    /// Lines at the foot of the screen.
    Corner(Vec<String>),
    /// One large line in the middle: the countdown.
    Big(String),
}

/// The overlay for `l` at `now`, given midnight.
pub fn overlay(l: &Live, now: DateTime<Local>, midnight: DateTime<Local>) -> Overlay {
    let to = (midnight - now).num_milliseconds();
    if to > 0 && to <= COUNTDOWN_S * 1000 {
        // Whole seconds, rounded up: "1" until the instant it is midnight.
        let s = (to + 999) / 1000;
        return Overlay::Big(s.to_string());
    }
    if to <= 0 && -to < AFTER_S * 1000 {
        return Overlay::Big("Happy New Year!".into());
    }
    let mut lines = Vec::new();
    if let Some(d) = l.decks.iter().find(|d| d.on_air && !d.title.is_empty()) {
        lines.push(format!("♪ {} — {}", d.artist, d.title));
    }
    if let Some(n) = &l.next {
        let line = match &n.requested_by {
            Some(by) => format!("{by}'s request is up next: {} — {}", n.artist, n.title),
            None => format!("Next: {} — {}", n.artist, n.title),
        };
        lines.push(line);
    }
    if let Some(b) = l.banner.as_ref().filter(|b| !b.is_empty()) {
        lines.push(format!("Zen: {b}"));
    }
    let mode = match l.mode {
        Mode::Human => None,
        Mode::Assist => Some("ZenJ is assisting"),
        Mode::Zenj => Some("ZenJ is mixing"),
    };
    if let Some(m) = mode {
        lines.push(m.to_string());
    }
    Overlay::Corner(lines)
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone};

    use super::*;
    use crate::live::{DeckView, NextView};

    fn midnight() -> DateTime<Local> {
        Local.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn the_corner_says_now_next_why_and_who_mixes() {
        let l = Live {
            mode: Mode::Zenj,
            decks: vec![DeckView {
                number: 1,
                title: "Waterloo".into(),
                artist: "ABBA".into(),
                on_air: true,
                ..DeckView::default()
            }],
            next: Some(NextView {
                title: "Dancing Queen".into(),
                artist: "ABBA".into(),
                requested_by: Some("Yen".into()),
                ..NextView::default()
            }),
            banner: Some("one for Yen's mum".into()),
            ..Live::default()
        };
        assert_eq!(
            overlay(&l, midnight() - Duration::hours(1), midnight()),
            Overlay::Corner(vec![
                "♪ ABBA — Waterloo".into(),
                "Yen's request is up next: ABBA — Dancing Queen".into(),
                "Zen: one for Yen's mum".into(),
                "ZenJ is mixing".into(),
            ])
        );
    }

    /// **From 23:59:00 the count takes the screen, to the second; then the
    /// new year, for a minute.**
    #[test]
    fn the_countdown_takes_over_the_last_minute() {
        let l = Live::default();
        let at = |ms: i64| overlay(&l, midnight() - Duration::milliseconds(ms), midnight());
        assert_eq!(at(61_000), Overlay::Corner(vec![]));
        assert_eq!(at(60_000), Overlay::Big("60".into()));
        assert_eq!(at(9_500), Overlay::Big("10".into()));
        assert_eq!(at(1), Overlay::Big("1".into()));
        assert_eq!(at(0), Overlay::Big("Happy New Year!".into()));
        assert_eq!(at(-59_999), Overlay::Big("Happy New Year!".into()));
        assert_eq!(at(-60_000), Overlay::Corner(vec![]));
    }
}
