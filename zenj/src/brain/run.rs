//! The brain's loop (§2): per track, and per second when the room asks. It
//! sees what the reader sees, chooses the next track — the shortlist put to
//! Zen, or the shortlist's best when Zen is late — keeps *NYE Next* stocked
//! with it, and in Assist and ZenJ has the conductor load it and, in ZenJ,
//! arms the transition. It never touches MIDI: everything it does to the
//! decks is a command to the conductor.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use chrono::{DateTime, Local, TimeDelta};
use tokio::sync::{mpsc, oneshot};

use super::choose::{prompt, read, Choice, SYSTEM};
use super::night::{Booked, Booking, Night, Request, Seen};
use super::shortlist::Scored;
use crate::conductor::{Command, Quant, State as ConductorState};
use crate::deck::Snapshot;
use crate::events::{Events, Kind};
use crate::hands::Holder;
use crate::library::crates::NEXT;
use crate::live::{DeckView, Live, NextView};
use crate::music::{Grid, Profile, TrackId};
use crate::tidal::Tidal;
use crate::zen::Zen;

/// How often the brain looks.
pub const PERIOD: Duration = Duration::from_millis(500);

/// What the tools, the phones and the conversation ask of the brain. Each
/// that answers says what came of it in one line.
pub enum Ask {
    Request {
        query: String,
        by: String,
        note: String,
        urgency: f32,
        now: bool,
        reply: oneshot::Sender<String>,
    },
    /// Give the decks to `holder`.
    Hands {
        holder: Holder,
        reply: oneshot::Sender<String>,
    },
    /// Make a track the next, past the shortlist.
    Pin {
        query: String,
        reply: oneshot::Sender<String>,
    },
    Book {
        at: DateTime<Local>,
        what: BookWhat,
        reply: oneshot::Sender<String>,
    },
    Sample {
        slot: u8,
        when: Quant,
    },
    Said {
        who: String,
        text: String,
    },
    Midnight {
        query: String,
        reply: oneshot::Sender<String>,
    },
    Energy {
        by: f32,
        minutes: i64,
        reply: oneshot::Sender<String>,
    },
}

/// A booking as asked, before its track is found.
pub enum BookWhat {
    Track(String),
    Dedication(String),
    Sample(u8),
}

/// Everything the brain's loop holds.
pub struct Brain {
    pub night: Night,
    pub snapshot: Arc<ArcSwap<Snapshot>>,
    pub conductor: Arc<ArcSwap<ConductorState>>,
    pub commands: Sender<Command>,
    pub events: Arc<Events>,
    pub live: Arc<ArcSwap<Live>>,
    pub zen: Option<Arc<Zen>>,
    pub tidal: Option<Arc<Tidal>>,
    /// The room's landing, `0..=1`, as `f32` bits (`room::Room::landing`).
    pub landing: Arc<AtomicU32>,
    /// Whether rekordbox runs, as the deck reader last found it.
    pub rekordbox: Arc<AtomicBool>,
}

/// A choice in flight, and what it was asked from.
struct Asking {
    list: Vec<Scored>,
    answer: oneshot::Receiver<Result<String, String>>,
}

fn line(p: &Profile) -> String {
    format!("{} — {}", p.artist_line(), p.title)
}

impl Brain {
    /// The library track `query` names: an id, an ISRC, a TIDAL or Spotify
    /// id, or words of its artist and title.
    fn find(&self, query: &str) -> Option<TrackId> {
        let q = query.trim();
        let lib = &self.night.library;
        if let Some(t) = lib.get(&TrackId(q.to_string())) {
            return Some(t.id.clone());
        }
        lib.tracks
            .iter()
            .find(|t| {
                t.isrc.as_deref() == Some(q)
                    || t.tidal_id.as_deref() == Some(q)
                    || t.spotify_id.as_deref() == Some(q)
            })
            .or_else(|| lib.search(q).into_iter().next())
            .map(|t| t.id.clone())
    }

    fn title(&self, id: &TrackId) -> String {
        self.night
            .track(id)
            .map(line)
            .unwrap_or_else(|| id.to_string())
    }

    fn handle(&mut self, ask: Ask, now: DateTime<Local>) {
        match ask {
            Ask::Request {
                query,
                by,
                note,
                urgency,
                now: right_now,
                reply,
            } => {
                let said = match self.find(&query) {
                    Some(track) => {
                        let title = self.title(&track);
                        self.night.request(Request {
                            track: track.clone(),
                            by: by.clone(),
                            note,
                            urgency: urgency.clamp(0.0, 1.0),
                            now: right_now,
                        });
                        let place = if right_now {
                            "next, at the coming phrase"
                        } else {
                            "placed where it fits, usually within two tracks"
                        };
                        let text = format!("{by}'s request: {title} — {place}");
                        self.events.push_at(now, Kind::Request, text.clone());
                        text
                    }
                    None => {
                        let closest: Vec<String> = self
                            .night
                            .library
                            .search(&query)
                            .iter()
                            .take(3)
                            .map(|t| line(t))
                            .collect();
                        let text = if closest.is_empty() {
                            format!("{by} asked for `{query}`: not in tonight's library")
                        } else {
                            format!(
                                "{by} asked for `{query}`: not in tonight's library — closest: {}",
                                closest.join("; ")
                            )
                        };
                        self.events.push_at(now, Kind::Request, text.clone());
                        text
                    }
                };
                let _ = reply.send(said);
            }
            Ask::Hands { holder, reply } => {
                let _ = self.commands.send(Command::Hand(holder));
                self.events.push_at(
                    now,
                    Kind::Hands,
                    format!("the decks to {} asked for", holder.name()),
                );
                let _ = reply.send(format!("the decks are now {}'s", holder.name()));
            }
            Ask::Pin { query, reply } => {
                let said = match self.find(&query) {
                    Some(t) => {
                        let title = self.title(&t);
                        self.night.choose(t, "asked for next".into());
                        self.events
                            .push_at(now, Kind::Next, format!("next: {title} (pinned)"));
                        format!("next: {title}")
                    }
                    None => format!("`{query}` is not in tonight's library"),
                };
                let _ = reply.send(said);
            }
            Ask::Book { at, what, reply } => {
                let booked = match what {
                    BookWhat::Track(q) => self.find(&q).map(Booked::Track).ok_or(q),
                    BookWhat::Dedication(text) => Ok(Booked::Dedication(text)),
                    BookWhat::Sample(slot) => Ok(Booked::Sample(slot)),
                };
                let said = match booked {
                    Ok(what) => {
                        self.night.bookings.push(Booking {
                            at,
                            what,
                            done: false,
                        });
                        format!("booked for {}", at.format("%H:%M"))
                    }
                    Err(q) => format!("`{q}` is not in tonight's library"),
                };
                let _ = reply.send(said);
            }
            Ask::Sample { slot, when } => {
                let _ = self.commands.send(Command::Sample { slot, when });
            }
            Ask::Said { who, text } => {
                self.night.heard(&who, &text);
                self.events
                    .push_at(now, Kind::Said, format!("{who}: {text}"));
            }
            Ask::Midnight { query, reply } => {
                let said = match self.find(&query) {
                    Some(t) => {
                        let title = self.title(&t);
                        self.night.midnight_track = Some(t);
                        format!("the midnight track is {title}; its drop lands on 00:00:00")
                    }
                    None => format!("`{query}` is not in tonight's library"),
                };
                let _ = reply.send(said);
            }
            Ask::Energy { by, minutes, reply } => {
                self.night.curve = self
                    .night
                    .curve
                    .nudged(now, TimeDelta::minutes(minutes), by);
                let _ = reply.send(format!(
                    "energy {} by {:.2} for {minutes} minutes: target now {:.2}",
                    if by >= 0.0 { "up" } else { "down" },
                    by.abs(),
                    self.night.curve.target(now)
                ));
            }
        }
    }

    /// Start asking Zen to choose from `list`.
    fn ask_zen(&self, list: Vec<Scored>, now: DateTime<Local>) -> Option<Asking> {
        let zen = self.zen.clone()?;
        let said: Vec<String> = self.night.said.iter().cloned().collect();
        let owed = self.night.owed();
        let clock = now.format("%H:%M").to_string();
        let target = self.night.curve.target(now);
        let offers = self.night.offers(&list);
        let p = prompt(
            &self.night.scene(&clock, now, target, &said, &owed),
            &offers,
        );
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let _ = tx.send(zen.ask(SYSTEM, &p).await);
        });
        Some(Asking { list, answer: rx })
    }

    /// Settle a choice from Zen's `answer` to `list`.
    fn settle(&mut self, list: &[Scored], answer: Result<String, String>, now: DateTime<Local>) {
        let Some(top) = list.first() else { return };
        let choice = match &answer {
            Ok(text) => {
                let offers = self.night.offers(list);
                read(text, &offers)
            }
            Err(_) => None,
        };
        let (track, why) = match choice {
            Some(Choice::Pick { id, why }) => (id, why),
            Some(Choice::Search { query, why }) => match self.find(&query) {
                Some(id) => (id, why),
                None => (
                    top.id.clone(),
                    format!("{query} is not in tonight's library; the best fit instead"),
                ),
            },
            None => {
                let why = match answer {
                    Err(e) => format!("the best fit (Zen: {e})"),
                    Ok(_) => "the best fit".into(),
                };
                (top.id.clone(), why)
            }
        };
        let title = self.title(&track);
        self.events
            .push_at(now, Kind::Next, format!("next: {title} — {why}"));
        self.night.choose(track, why);
    }

    /// Put `track` alone on *NYE Next*, for the loader to find at its top.
    fn stock(&self, track: &TrackId) {
        let (Some(tidal), Some(list)) = (
            self.tidal.clone(),
            self.night.library.playlists.get(NEXT).cloned(),
        ) else {
            return;
        };
        let Some(id) = self.night.track(track).and_then(|t| t.tidal_id.clone()) else {
            return;
        };
        let events = self.events.clone();
        tokio::spawn(async move {
            let result = async {
                let items = tidal.items(&list).await?;
                if !items.is_empty() {
                    tidal.remove(&list, &items).await?;
                }
                tidal.add(&list, &[id], None).await
            }
            .await;
            if let Err(e) = result {
                events.push(Kind::Fault, format!("NYE Next could not be stocked: {e}"));
            }
        });
    }

    fn publish(
        &self,
        s: &Snapshot,
        c: &ConductorState,
        now: DateTime<Local>,
        banner: &Option<String>,
    ) {
        let decks = s
            .decks
            .iter()
            .map(|d| {
                let t = d.track.as_ref().and_then(|t| self.night.track(t));
                DeckView {
                    number: d.number,
                    title: t.map(|t| t.title.clone()).unwrap_or_default(),
                    artist: t.map(|t| t.artist_line()).unwrap_or_default(),
                    bpm: d.bpm,
                    beat: d.beat,
                    bars_left: t
                        .zip(d.beat)
                        .map(|(t, b)| (t.beats() as f64 - b).max(0.0) / 4.0),
                    playing: d.playing,
                    on_air: s.on_air().is_some_and(|o| o.number == d.number),
                }
            })
            .collect();
        let next = self.night.next.as_ref().map(|n| {
            let t = self.night.track(&n.track);
            NextView {
                title: t.map(|t| t.title.clone()).unwrap_or_default(),
                artist: t.map(|t| t.artist_line()).unwrap_or_default(),
                why: n.why.clone(),
                requested_by: n.requested_by.clone(),
                style: c.armed.clone(),
            }
        });
        let requests = self
            .night
            .requests
            .iter()
            .map(|r| format!("{}: {}", r.by, self.title(&r.track)))
            .collect();
        self.live.store(Arc::new(Live {
            holder: c.holder,
            decks,
            next,
            requests,
            automix: c.automix,
            fault: c.fault.clone(),
            blind: s.blind.clone(),
            target_energy: self.night.curve.target(now),
            landing: f32::from_bits(self.landing.load(Ordering::Relaxed)),
            to_midnight_s: (self.night.midnight - now).num_seconds(),
            banner: banner.clone(),
            live: s.decks.iter().any(|d| d.playing),
            rekordbox: self.rekordbox.load(Ordering::Relaxed),
        }));
    }

    /// Run until the asks' sender is gone.
    pub async fn run(mut self, mut asks: mpsc::Receiver<Ask>) {
        let mut asking: Option<Asking> = None;
        let mut stocked: Option<TrackId> = None;
        let mut armed_for: Option<TrackId> = None;
        let mut banner: Option<String> = None;
        let mut tick = tokio::time::interval(PERIOD);
        loop {
            tokio::select! {
                ask = asks.recv() => match ask {
                    Some(a) => self.handle(a, Local::now()),
                    None => return,
                },
                _ = tick.tick() => {}
            }
            let now = Local::now();
            let snap = self.snapshot.load_full();
            let cond = self.conductor.load_full();

            if let Some(Seen::OnAir { deck, track }) = self.night.see(&snap, now) {
                self.events.push_at(
                    now,
                    Kind::Track,
                    format!("on air, deck {deck}: {}", self.title(&track)),
                );
                let tracks: Vec<(u8, Grid, u64)> = snap
                    .decks
                    .iter()
                    .filter_map(|d| {
                        let t = self.night.track(d.track.as_ref()?)?;
                        Some((d.number, t.grid, t.duration_ms))
                    })
                    .collect();
                let _ = self.commands.send(Command::Tracks(tracks));
                armed_for = None;
            }

            // The next: forced, or chosen.
            let ends = self.night.ends_at(&snap, now);
            if let Some((track, why)) = self.night.forced(now, ends) {
                if self.night.next.as_ref().map(|n| &n.track) != Some(&track) {
                    self.events.push_at(
                        now,
                        Kind::Next,
                        format!("next: {} — {why}", self.title(&track)),
                    );
                    self.night.choose(track, why);
                    asking = None;
                }
            }
            for b in self.night.due(now) {
                match b {
                    Booked::Sample(slot) => {
                        let _ = self.commands.send(Command::Sample {
                            slot,
                            when: Quant::Bar,
                        });
                    }
                    Booked::Dedication(text) => {
                        self.events.push_at(now, Kind::Moment, text.clone());
                        banner = Some(text);
                    }
                    Booked::Track(_) => {}
                }
            }
            if let Some(a) = asking.as_mut() {
                if let Ok(answer) = a.answer.try_recv() {
                    let list = std::mem::take(&mut a.list);
                    asking = None;
                    self.settle(&list, answer, now);
                }
            }
            if self.night.next.is_none() && asking.is_none() && self.night.playing().is_some() {
                let when = ends.unwrap_or(now);
                let landing = f32::from_bits(self.landing.load(Ordering::Relaxed));
                let list = self.night.shortlist(when, landing);
                if !list.is_empty() {
                    asking = match self.ask_zen(list.clone(), now) {
                        Some(a) => Some(a),
                        None => {
                            self.settle(&list, Err("not reachable".into()), now);
                            None
                        }
                    };
                }
            }
            if let Some(n) = &self.night.next {
                banner = Some(n.why.clone())
                    .filter(|w| !w.is_empty())
                    .or(banner.take());
            }

            // Stocked, loaded, armed.
            if let Some(next) = self.night.next.clone() {
                if stocked.as_ref() != Some(&next.track) {
                    self.stock(&next.track);
                    stocked = Some(next.track.clone());
                }
                let on_air = self.night.on_air.as_ref().map(|(d, _)| *d);
                let idle = match on_air {
                    Some(1) => 2,
                    _ => 1,
                };
                let loaded = cond
                    .loaded
                    .iter()
                    .any(|(d, t)| *d == idle && *t == next.track);
                if matches!(cond.holder, Holder::Assist | Holder::Zenj)
                    && !loaded
                    && cond.loading.is_none()
                {
                    let _ = self.commands.send(Command::Load {
                        deck: idle,
                        track: next.track.clone(),
                    });
                }
                if cond.holder == Holder::Zenj
                    && loaded
                    && cond.armed.is_none()
                    && armed_for.as_ref() != Some(&next.track)
                {
                    let air = on_air.and_then(|d| snap.deck(d));
                    if let Some((beat, bpm)) = air.and_then(|d| d.beat.map(|b| (b, d.bpm))) {
                        if let Some(p) = self.night.plan_next(idle, beat, bpm, now) {
                            self.events.push_at(
                                now,
                                Kind::Transition,
                                format!("{} at beat {:.0}: {}", p.style.name(), p.start, p.why),
                            );
                            let _ = self.commands.send(Command::Arm(p));
                            armed_for = Some(next.track.clone());
                        }
                    }
                }
            }
            self.publish(&snap, &cond, now, &banner);
        }
    }
}
