//! The night as the brain keeps it: what has played and whose it was, what
//! is next and why, the requests waiting and where each was placed, the
//! moments booked, the conversation's last lines, and the midnight track.
//! Everything here is decided from what it is given — the clock, a deck
//! snapshot, Zen's answer — so a night replays in a test.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, Duration, Local};
use serde::Serialize;

use super::budget::Budget;
use super::choose::{Offer, Scene};
use super::curve::Curve;
use super::midnight::out_beat_for;
use super::shortlist::{shortlist, Moment, Scored, SHORTLIST};
use super::transition::{plan, Pair};
use crate::deck::Snapshot;
use crate::library::Library;
use crate::music::{PhraseKind, Profile, TrackId};
use crate::plan::Plan;

/// The beat the midnight track's drop is on: its first chorus, else beat 64.
fn drop_beat(t: &Profile) -> u32 {
    t.phrase_from(PhraseKind::Chorus, 0).map_or(64, |p| p.start)
}

/// When `t` must start for its drop to land on `midnight`.
fn starts_for(t: &Profile, midnight: DateTime<Local>) -> DateTime<Local> {
    midnight - Duration::milliseconds((drop_beat(t) as f64 * t.grid.ms_per_beat()) as i64)
}

/// How many of the conversation's lines Zen is shown.
pub const SAID_KEPT: usize = 8;

/// A request, placed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Request {
    pub track: TrackId,
    pub by: String,
    pub note: String,
    /// `0..=1`.
    pub urgency: f32,
    /// "Play it now": mixed in at the next phrase, past the shortlist.
    pub now: bool,
}

/// What is booked for a time.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Booked {
    Track(TrackId),
    Dedication(String),
    Sample(u8),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Booking {
    pub at: DateTime<Local>,
    pub what: Booked,
    pub done: bool,
}

/// The next track, chosen.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Next {
    pub track: TrackId,
    pub why: String,
    /// Who asked for it, when it is a request.
    pub requested_by: Option<String>,
}

/// What changed when a snapshot was seen.
#[derive(Debug, Clone, PartialEq)]
pub enum Seen {
    /// A new track is on air, on deck `deck`.
    OnAir { deck: u8, track: TrackId },
}

pub struct Night {
    pub library: Library,
    pub curve: Curve,
    pub budget: Budget,
    pub midnight: DateTime<Local>,
    pub midnight_track: Option<TrackId>,
    pub played: Vec<(TrackId, DateTime<Local>)>,
    played_set: HashSet<TrackId>,
    requested_ids: HashSet<TrackId>,
    pub on_air: Option<(u8, TrackId)>,
    pub next: Option<Next>,
    pub requests: Vec<Request>,
    pub bookings: Vec<Booking>,
    pub said: VecDeque<String>,
}

impl Night {
    pub fn new(library: Library, curve: Curve, budget: Budget, midnight: DateTime<Local>) -> Night {
        Night {
            library,
            curve,
            budget,
            midnight,
            midnight_track: None,
            played: Vec::new(),
            played_set: HashSet::new(),
            requested_ids: HashSet::new(),
            on_air: None,
            next: None,
            requests: Vec::new(),
            bookings: Vec::new(),
            said: VecDeque::new(),
        }
    }

    pub fn track(&self, id: &TrackId) -> Option<&Profile> {
        self.library.get(id)
    }

    /// What `s` changes: a new track on air is played, counted to whoever's
    /// it is, and the next cleared if it was the next.
    pub fn see(&mut self, s: &Snapshot, now: DateTime<Local>) -> Option<Seen> {
        let d = s.on_air()?;
        let track = d.track.clone()?;
        if self.on_air.as_ref().is_some_and(|(_, t)| *t == track) {
            return None;
        }
        self.on_air = Some((d.number, track.clone()));
        if self.played_set.insert(track.clone()) {
            self.played.push((track.clone(), now));
            let requested = self.requested_ids.contains(&track);
            if let Some(p) = self.library.get(&track).cloned() {
                self.budget.count(&p, requested);
            }
        }
        self.requests.retain(|r| r.track != track);
        if self.next.as_ref().is_some_and(|n| n.track == track) {
            self.next = None;
        }
        Some(Seen::OnAir {
            deck: d.number,
            track,
        })
    }

    /// The playing track, when the reader names one the library holds.
    pub fn playing(&self) -> Option<&Profile> {
        self.on_air.as_ref().and_then(|(_, t)| self.library.get(t))
    }

    /// When the playing track ends, from where its deck is.
    pub fn ends_at(&self, s: &Snapshot, now: DateTime<Local>) -> Option<DateTime<Local>> {
        let (deck, _) = self.on_air.as_ref()?;
        let d = s.deck(*deck)?;
        let t = self.playing()?;
        let rate = (d.bpm / t.grid.bpm).max(0.01) as f64;
        let left = (t.duration_ms as f64 - d.position_ms).max(0.0) / rate;
        Some(now + Duration::milliseconds(left as i64))
    }

    fn recent_artists(&self) -> Vec<String> {
        self.played
            .iter()
            .rev()
            .filter_map(|(id, _)| self.library.get(id))
            .flat_map(|p| p.artists.clone())
            .collect()
    }

    /// The shortlist for the track after the playing one, which plays at
    /// `when`.
    pub fn shortlist(&self, when: DateTime<Local>, landing: f32) -> Vec<Scored> {
        let candidates = self.library.playable();
        let requests: HashMap<TrackId, f32> = self
            .requests
            .iter()
            .map(|r| (r.track.clone(), r.urgency))
            .collect();
        let artists = self.recent_artists();
        let m = Moment {
            after: self.playing(),
            target: self.curve.target(when),
            played: &self.played_set,
            recent_artists: &artists,
            requests: &requests,
            budget: &self.budget,
            landing,
        };
        shortlist(&candidates, &m, SHORTLIST)
    }

    /// The scene and offers Zen is shown for `list`.
    pub fn offers<'a>(&'a self, list: &'a [Scored]) -> Vec<Offer<'a>> {
        list.iter()
            .filter_map(|s| {
                let track = self.library.get(&s.id)?;
                let request = self
                    .requests
                    .iter()
                    .find(|r| r.track == s.id)
                    .map(|r| (r.by.as_str(), r.note.as_str()));
                Some(Offer {
                    track,
                    scored: s,
                    request,
                })
            })
            .collect()
    }

    /// What is owed, for Zen to read.
    pub fn owed(&self) -> Vec<(String, f32)> {
        self.budget
            .people()
            .map(|p| (p.clone(), self.budget.owed(p)))
            .filter(|(_, o)| *o > 0.05)
            .collect()
    }

    pub fn scene<'a>(
        &'a self,
        clock: &'a str,
        now: DateTime<Local>,
        target: f32,
        said: &'a [String],
        owed: &'a [(String, f32)],
    ) -> Scene<'a> {
        Scene {
            clock,
            to_midnight_min: (self.midnight - now).num_minutes(),
            playing: self.playing(),
            target,
            said,
            owed,
        }
    }

    /// Choose `track` as next, for `why`.
    pub fn choose(&mut self, track: TrackId, why: String) {
        let requested_by = self
            .requests
            .iter()
            .find(|r| r.track == track)
            .map(|r| r.by.clone());
        self.next = Some(Next {
            track,
            why,
            requested_by,
        });
    }

    /// Place a request; a "now" one is next at once, past the shortlist.
    pub fn request(&mut self, r: Request) {
        self.requested_ids.insert(r.track.clone());
        if r.now {
            let why = format!("{}'s request, now", r.by);
            self.requests.retain(|x| x.track != r.track);
            self.requests.push(r.clone());
            self.choose(r.track, why);
            return;
        }
        self.requests.retain(|x| x.track != r.track);
        self.requests.push(r);
    }

    /// The forced next, ahead of any choice: the midnight track when its
    /// run-in is due, a track booked for now, or a "now" request.
    pub fn forced(&self, now: DateTime<Local>, ends: Option<DateTime<Local>>) -> Option<(TrackId, String)> {
        if let (Some(m), Some(ends)) = (&self.midnight_track, ends) {
            if !self.played_set.contains(m) {
                if let Some(t) = self.library.get(m) {
                    let starts = starts_for(t, self.midnight);
                    // The track before it must hand over at `starts`: when the
                    // playing one would end later than that, it is next.
                    if ends + Duration::minutes(1) >= starts && now < self.midnight {
                        return Some((m.clone(), "the midnight track".into()));
                    }
                }
            }
        }
        self.bookings
            .iter()
            .filter(|b| !b.done && b.at <= now)
            .find_map(|b| match &b.what {
                Booked::Track(t) => Some((t.clone(), "booked for now".into())),
                _ => None,
            })
    }

    /// The plan from the playing track to the next, on `next_deck`, at the
    /// playing deck's `beat` now. For the midnight track, the start is moved
    /// so its drop lands on midnight.
    pub fn plan_next(&self, next_deck: u8, beat: f64, bpm: f32, now: DateTime<Local>) -> Option<Plan> {
        let (deck, _) = self.on_air.as_ref()?;
        let from = self.playing()?;
        let next = self.next.as_ref()?;
        let to = self.library.get(&next.track)?;
        let mut p = plan(&Pair {
            from,
            from_deck: *deck,
            to,
            to_deck: next_deck,
            now: beat,
        });
        if self.midnight_track.as_ref() == Some(&next.track) {
            if let Some(at) = out_beat_for(starts_for(to, self.midnight), now, beat, bpm) {
                let shift = at - p.start;
                for s in &mut p.steps {
                    s.at += shift;
                }
                p.start += shift;
                p.end += shift;
                p.why = format!("{}; timed so the drop lands on midnight", p.why);
            }
        }
        Some(p)
    }

    /// The bookings due at `now`, marked done.
    pub fn due(&mut self, now: DateTime<Local>) -> Vec<Booked> {
        let mut out = Vec::new();
        for b in self.bookings.iter_mut().filter(|b| !b.done && b.at <= now) {
            if !matches!(b.what, Booked::Track(_)) {
                out.push(b.what.clone());
            }
            b.done = true;
        }
        out
    }

    /// A line the room said.
    pub fn heard(&mut self, who: &str, text: &str) {
        self.said.push_back(format!("{who}: {text}"));
        while self.said.len() > SAID_KEPT {
            self.said.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{NaiveDate, TimeZone};

    use super::*;
    use crate::brain::curve::PointCfg;
    use crate::deck::Deck;
    use crate::music::track::fixture::club;

    fn at(h: u32, m: u32) -> DateTime<Local> {
        let d = if h < 12 {
            NaiveDate::from_ymd_opt(2027, 1, 1).unwrap()
        } else {
            NaiveDate::from_ymd_opt(2026, 12, 31).unwrap()
        };
        Local.from_local_datetime(&d.and_hms_opt(h, m, 0).unwrap()).earliest().unwrap()
    }

    fn analysed(id: &str, bpm: f32, key: &str, energy: f32) -> Profile {
        let mut p = club(id, bpm, key, energy);
        p.anlz = Some(format!("/PIONEER/USBANLZ/{id}/ANLZ0000.DAT"));
        p.artists = vec![format!("{id} artist")];
        p
    }

    fn night() -> Night {
        let curve = Curve::of(
            NaiveDate::from_ymd_opt(2026, 12, 31).unwrap(),
            &[PointCfg { at: "20:00".into(), energy: 0.5 }, PointCfg { at: "00:00".into(), energy: 0.9 }],
        )
        .unwrap();
        let lib = Library::new(vec![
            analysed("a", 128.0, "8A", 0.6),
            analysed("b", 127.0, "9A", 0.7),
            analysed("c", 128.0, "8A", 0.5),
            analysed("m", 128.0, "8A", 1.0),
        ]);
        Night::new(
            lib,
            curve,
            Budget::new([("john".to_string(), 0.5), ("yen".to_string(), 0.5)].into(), 0.2),
            at(0, 0),
        )
    }

    fn on(track: &str, position_ms: f64) -> Snapshot {
        Snapshot {
            decks: vec![Deck {
                number: 1,
                track: Some(TrackId(track.into())),
                bpm: 128.0,
                position_ms,
                beat: Some(position_ms / 468.75),
                playing: true,
            }],
            master: Some(1),
            blind: None,
        }
    }

    /// **A new track on air is played once, and clears the next if it was
    /// the next**; the same track seen again changes nothing.
    #[test]
    fn a_track_on_air_is_played_once() {
        let mut n = night();
        n.choose(TrackId("a".into()), "x".into());
        assert_eq!(
            n.see(&on("a", 0.0), at(22, 0)),
            Some(Seen::OnAir { deck: 1, track: TrackId("a".into()) })
        );
        assert_eq!(n.next, None);
        assert_eq!(n.see(&on("a", 5000.0), at(22, 1)), None);
        assert_eq!(n.played.len(), 1);
        let ends = n.ends_at(&on("a", 60_000.0), at(22, 1)).unwrap();
        assert_eq!((ends - at(22, 1)).num_milliseconds(), 60_000, "120 s long, 60 s in");
    }

    /// **The shortlist leaves out what has played; a request leads it, and
    /// a "now" request is next at once.**
    #[test]
    fn requests_lead_and_now_is_now() {
        let mut n = night();
        n.see(&on("a", 0.0), at(22, 0));
        let ids = |l: Vec<Scored>| l.into_iter().map(|s| s.id.0).collect::<Vec<_>>();
        let l = ids(n.shortlist(at(22, 2), 0.5));
        assert!(!l.contains(&"a".to_string()));
        n.request(Request { track: TrackId("c".into()), by: "yen".into(), note: "for mum".into(), urgency: 1.0, now: false });
        assert_eq!(ids(n.shortlist(at(22, 2), 0.5))[0], "c");
        let list = n.shortlist(at(22, 2), 0.5);
        let offers = n.offers(&list);
        assert_eq!(offers[0].request, Some(("yen", "for mum")));
        n.request(Request { track: TrackId("b".into()), by: "john".into(), note: String::new(), urgency: 1.0, now: true });
        assert_eq!(n.next.as_ref().unwrap().track, TrackId("b".into()));
        assert_eq!(n.next.as_ref().unwrap().requested_by.as_deref(), Some("john"));
    }

    /// **The midnight track is forced when the playing one would run past
    /// its start, and its plan is moved so its drop lands on midnight.**
    #[test]
    fn the_midnight_track_lands_its_drop_on_midnight() {
        let mut n = night();
        n.midnight_track = Some(TrackId("m".into()));
        n.see(&on("a", 0.0), at(23, 50));
        // m's drop is beat 64 at 128 BPM: it starts 30 s before midnight.
        let early = at(23, 50);
        assert_eq!(n.forced(early, Some(early + Duration::minutes(2))), None, "ten minutes out");
        let near = at(23, 58);
        assert_eq!(
            n.forced(near, Some(near + Duration::seconds(90))).map(|f| f.0),
            Some(TrackId("m".into()))
        );
        n.choose(TrackId("m".into()), "the midnight track".into());
        // At 23:58:00 on beat 0, the start must be 90 s on: beat 192.
        let p = n.plan_next(2, 0.0, 128.0, near).unwrap();
        assert_eq!(p.start, 192.0);
        assert!(p.why.ends_with("timed so the drop lands on midnight"));
    }

    #[test]
    fn bookings_fire_once_when_due() {
        let mut n = night();
        n.bookings.push(Booking { at: at(22, 30), what: Booked::Sample(3), done: false });
        n.bookings.push(Booking { at: at(22, 30), what: Booked::Track(TrackId("c".into())), done: false });
        assert!(n.due(at(22, 29)).is_empty());
        assert_eq!(n.forced(at(22, 31), None).map(|f| f.0), Some(TrackId("c".into())));
        assert_eq!(n.due(at(22, 31)), vec![Booked::Sample(3)]);
        assert!(n.due(at(22, 32)).is_empty());
        for i in 0..10 {
            n.heard("yen", &i.to_string());
        }
        assert_eq!(n.said.len(), SAID_KEPT);
        assert_eq!(n.said.front().map(String::as_str), Some("yen: 2"));
    }
}
