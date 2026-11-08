//! Closed-loop loading (§6.2): the next track sits at the top of *NYE Next*;
//! to load it the browser is moved to that playlist's first row and "load to
//! deck *n*" pressed, then the deck reader is read until the deck holds the
//! intended track. Not there in time, it is tried again — twice at most —
//! and then the load has failed, and Automix takes that transition.

use std::time::{Duration, Instant};

use crate::control::Button;
use crate::deck::Snapshot;
use crate::music::TrackId;

/// Presses of browse-up that reach the first row from anywhere in a short
/// playlist.
pub const TO_TOP: usize = 24;
/// How long a load may take to show on the deck.
pub const SHOWS_WITHIN: Duration = Duration::from_secs(4);
/// Tries after the first.
pub const RETRIES: u32 = 2;

/// Where a load is.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// Press these, then watch the deck.
    Press(Vec<Button>),
    Waiting,
    Loaded,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Load {
    pub deck: u8,
    pub track: TrackId,
    tries: u32,
    pressed_at: Option<Instant>,
}

impl Load {
    pub fn new(deck: u8, track: TrackId) -> Load {
        Load {
            deck,
            track,
            tries: 0,
            pressed_at: None,
        }
    }

    fn presses(&self) -> Vec<Button> {
        let mut p = vec![Button::BrowseUp; TO_TOP];
        p.push(Button::Load(self.deck));
        p
    }

    /// The next step at `now`, given what the reader shows.
    pub fn step(&mut self, now: Instant, s: &Snapshot) -> Step {
        let holds = s
            .deck(self.deck)
            .and_then(|d| d.track.as_ref())
            .is_some_and(|t| *t == self.track);
        if holds {
            return Step::Loaded;
        }
        match self.pressed_at {
            None => {
                self.pressed_at = Some(now);
                Step::Press(self.presses())
            }
            Some(at) if now.duration_since(at) < SHOWS_WITHIN => Step::Waiting,
            Some(_) if self.tries < RETRIES => {
                self.tries += 1;
                self.pressed_at = Some(now);
                Step::Press(self.presses())
            }
            Some(_) => {
                let held = s
                    .deck(self.deck)
                    .and_then(|d| d.track.as_ref())
                    .map_or("nothing".to_string(), |t| t.to_string());
                Step::Failed(format!(
                    "deck {} holds {held}, not {}, after {} tries",
                    self.deck,
                    self.track,
                    RETRIES + 1
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::Deck;

    fn holding(track: Option<&str>) -> Snapshot {
        Snapshot {
            decks: vec![Deck {
                number: 2,
                track: track.map(|t| TrackId(t.into())),
                bpm: 120.0,
                position_ms: 0.0,
                beat: Some(0.0),
                playing: false,
            }],
            master: None,
            blind: None,
        }
    }

    /// **A load presses, watches, retries twice, and gives up naming what
    /// the deck holds**; a deck that shows the track is loaded at once.
    #[test]
    fn a_load_is_confirmed_or_given_up() {
        let t0 = Instant::now();
        let mut l = Load::new(2, TrackId("want".into()));
        let s = holding(Some("other"));
        let Step::Press(p) = l.step(t0, &s) else {
            panic!()
        };
        assert_eq!(p.len(), TO_TOP + 1);
        assert_eq!(p.last(), Some(&Button::Load(2)));
        assert_eq!(l.step(t0 + Duration::from_secs(1), &s), Step::Waiting);
        for i in 1..=RETRIES as u64 {
            assert!(matches!(
                l.step(
                    t0 + SHOWS_WITHIN * (i as u32) + Duration::from_millis(1),
                    &s
                ),
                Step::Press(_)
            ));
        }
        assert_eq!(
            l.step(t0 + SHOWS_WITHIN * 4, &s),
            Step::Failed("deck 2 holds other, not want, after 3 tries".into())
        );
        assert_eq!(l.step(t0, &holding(Some("want"))), Step::Loaded);
    }
}
