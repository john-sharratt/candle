//! Text left somewhere in the world for whoever comes by.
//!
//! # The thing the world was missing
//!
//! A room could contain machines, items, orders and people, and not one of those
//! is *something to read*. `read` was therefore bound to the machines standing
//! here — every device in the room, whether or not it had anything to say — and
//! reading one reported the state it was switched to:
//!
//! ```text
//! You read the accession desk. It stands at `reading`.
//! ```
//!
//! Which is the question restated as an answer. A `Device` carries no text at
//! all; a station's subject lives on the body holding it, not on the station. So
//! there was nothing for the act to return and never had been, and forty-seven
//! of fifty acts in a live feed were that line.
//!
//! # Why a store of its own
//!
//! Neither existing store can answer *what is readable here*. [`crate::sim::record`]
//! holds the archive — documents, appraisals, judgements — and an `Item` has no
//! place in the world; [`crate::sim::ledger`] holds orders, which are world-wide
//! and are a job to take rather than a text to read. A posting is the third
//! thing: **words, at a place, that accumulate**.
//!
//! # Stateful, and the state is per reader
//!
//! A posting is not a string. It is a run of lines that grows, each knowing who
//! wrote it, and a cursor per reader saying how far down they have got.
//!
//! **That cursor is what makes the act terminate.** The treadmill above was not
//! caused by the prose being poor — it was caused by the act being repeatable
//! with an unchanged answer, so a character brought straight back by
//! `arm_followup` read the same thing again, and again. With a cursor, a board
//! you have already read has nothing for you; `readable_at` stops offering it;
//! the ordinary empty-set rule takes `read` out of the grammar. The character
//! cannot loop because it cannot *say* the act, which is the discipline every
//! other closed set in this engine already follows — a refusal it can re-emit
//! would have been the same bug in a politer voice.
//!
//! Somebody adding a line makes it unread for everybody again, which is what a
//! notice board is for.
//!
//! It is the same mechanism as [`crate::sim::phone::Thread`]'s delivery cursor
//! and `npc_map::Attention`'s, for the same reason: without a per-reader mark,
//! a shared thing either re-delivers itself forever or is read once by whoever
//! got there first.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// How many lines one posting keeps.
///
/// A board is a rolling record and not an archive — the same rule
/// [`crate::sim::phone::KEEP_DIRECT`] holds, and for the same reason: what
/// reaches a character is read in full into a single turn, and nobody can act on
/// four hundred notices. What falls off the top is gone; a thing that mattered
/// longer than that belonged in the record, which is a different store on
/// purpose.
pub const KEEP_LINES: usize = 32;

/// One thing written on a posting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Line {
    /// Who wrote it, by the name the world writes down. The world itself when
    /// nobody in particular did — a printed sign has an author in the fiction
    /// and none in the simulation.
    pub by: String,
    /// What it says.
    pub text: String,
}

/// Something at a place with words on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Posting {
    pub id: String,
    /// What a character calls it: "the muster board". This is the name `read`
    /// binds to, so it is also the name the grammar offers.
    pub name: String,
    /// `area/node` — the room it is in, matching `npc_map`'s `Where`.
    pub at: String,
    /// What is on it, oldest first.
    pub lines: Vec<Line>,
    /// How far down each reader has got, by body id.
    ///
    /// **Body id rather than display name.** A read cursor is bookkeeping that
    /// nobody addresses, unlike a thread's membership, so it keys on the thing
    /// that cannot be renamed out from under it.
    read_to: BTreeMap<String, usize>,
}

impl Posting {
    /// What this reader has not seen.
    pub fn unread_for(&self, who: &str) -> &[Line] {
        let from = self.read_to.get(who).copied().unwrap_or(0);
        self.lines.get(from..).unwrap_or_default()
    }

    pub fn has_unread_for(&self, who: &str) -> bool {
        !self.unread_for(who).is_empty()
    }

    /// Mark everything on it as read by somebody.
    pub fn mark_read(&mut self, who: &str) {
        self.read_to.insert(who.to_string(), self.lines.len());
    }

    /// Add a line, dropping the oldest once it has grown past [`KEEP_LINES`].
    ///
    /// **Every reader's cursor moves when the front is dropped.** Trimming
    /// without shifting them leaves valid numbers pointing at the wrong lines,
    /// so a reader who was up to date is silently handed somebody else's
    /// backlog — the same trap [`crate::sim::phone::Thread::say`] documents, and
    /// the reason both are tested for it rather than reasoned about.
    pub fn say(&mut self, line: Line) {
        self.lines.push(line);
        let Some(drop) = self.lines.len().checked_sub(KEEP_LINES).filter(|d| *d > 0) else {
            return;
        };
        self.lines.drain(..drop);
        for at in self.read_to.values_mut() {
            *at = at.saturating_sub(drop);
        }
    }
}

/// Everything posted anywhere in one world.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Postings {
    postings: BTreeMap<String, Posting>,
}

/// The id a posting is filed under. One per name per place, so posting to "the
/// muster board" twice appends rather than making a second board beside it.
fn key(at: &str, name: &str) -> String {
    format!("{at}#{}", name.trim().to_lowercase())
}

impl Postings {
    pub fn new() -> Postings {
        Postings::default()
    }

    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }

    pub fn len(&self) -> usize {
        self.postings.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Posting> {
        self.postings.values()
    }

    /// Put an empty surface at a place, if there is not one there already.
    ///
    /// What the map's fixtures become at load — a board exists before anybody
    /// writes on it, and has to, or the first person to write would be creating
    /// the furniture rather than using it. Empty is offered to no reader, so a
    /// world full of blank boards costs nothing until somebody writes.
    ///
    /// Never clears one that is standing: this runs on every load of a world
    /// whose postings may have been restored around it.
    pub fn stand_up(&mut self, at: &str, name: &str) {
        self.postings
            .entry(key(at, name))
            .or_insert_with(|| Posting {
                id: key(at, name),
                name: name.trim().to_string(),
                at: at.to_string(),
                lines: Vec::new(),
                read_to: BTreeMap::new(),
            });
    }

    /// Everything here that can be written on, whether or not anything is on it
    /// yet — what `post_notice.on` binds to.
    ///
    /// Distinct from [`Postings::unread_names_at`] and the difference is the
    /// whole point: you write on a blank board, and you do not read one.
    pub fn postable_names_at(&self, place: &str) -> Vec<String> {
        self.at(place).into_iter().map(|p| p.name.clone()).collect()
    }

    /// Write something at a place, creating the posting if it is not there yet.
    ///
    /// Create-or-append, because "put a notice on the board" and "put the first
    /// notice on the board" are the same act to whoever is doing it, and making
    /// the caller check first is how two boards of one name end up in one room.
    pub fn post(&mut self, at: &str, name: &str, by: &str, text: &str) -> &Posting {
        let id = key(at, name);
        let p = self.postings.entry(id.clone()).or_insert_with(|| Posting {
            id,
            name: name.trim().to_string(),
            at: at.to_string(),
            lines: Vec::new(),
            read_to: BTreeMap::new(),
        });
        p.say(Line {
            by: by.to_string(),
            text: text.trim().to_string(),
        });
        p
    }

    /// Everything standing at a place.
    pub fn at(&self, place: &str) -> Vec<&Posting> {
        self.postings.values().filter(|p| p.at == place).collect()
    }

    pub fn by_name_at(&self, place: &str, name: &str) -> Option<&Posting> {
        let want = name.trim().to_lowercase();
        self.postings
            .values()
            .find(|p| p.at == place && p.name.to_lowercase() == want)
    }

    pub fn by_name_at_mut(&mut self, place: &str, name: &str) -> Option<&mut Posting> {
        let want = name.trim().to_lowercase();
        self.postings
            .values_mut()
            .find(|p| p.at == place && p.name.to_lowercase() == want)
    }

    /// What is here that this reader has not read — by the name it would use.
    ///
    /// **The set `read.what` binds to.** Empty when there is nothing new here,
    /// which takes the act out of the grammar rather than leaving a character
    /// able to ask again and be told the same thing.
    pub fn unread_names_at(&self, place: &str, who: &str) -> Vec<String> {
        self.at(place)
            .into_iter()
            .filter(|p| p.has_unread_for(who))
            .map(|p| p.name.clone())
            .collect()
    }

    /// Read one, and mark how far this reader got.
    ///
    /// `None` when there is no such posting here. An empty `Vec` when there is
    /// one and it holds nothing new — a different answer, and the caller says so
    /// differently.
    pub fn read(&mut self, place: &str, name: &str, who: &str) -> Option<Vec<Line>> {
        let want = name.trim().to_lowercase();
        let p = self
            .postings
            .values_mut()
            .find(|p| p.at == place && p.name.to_lowercase() == want)?;
        let fresh = p.unread_for(who).to_vec();
        p.mark_read(who);
        Some(fresh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HERE: &str = "vault-command/receiving";

    fn board() -> Postings {
        let mut p = Postings::new();
        p.post(
            HERE,
            "the muster board",
            "Maker-01",
            "the lift on five is out",
        );
        p
    }

    #[test]
    fn posting_twice_to_one_name_appends_rather_than_making_a_second() {
        let mut p = board();
        p.post(
            HERE,
            "the muster board",
            "Maker-02",
            "and the stairs are wet",
        );
        assert_eq!(p.len(), 1, "a second board appeared in one room");
        assert_eq!(
            p.by_name_at(HERE, "the muster board").unwrap().lines.len(),
            2
        );
        // Named the way it is written, whatever case the caller used.
        assert!(p.by_name_at(HERE, "The Muster Board").is_some());
    }

    /// **The property that stops the treadmill.**
    ///
    /// `read` is in `body::ANSWERS`, so a character is brought straight back to
    /// use what it just learnt. If reading again returns the same thing, it
    /// reads again — which is exactly what forty-seven of fifty live acts were.
    /// Nothing new means nothing offered, so the act leaves the grammar and the
    /// loop cannot be expressed.
    #[test]
    fn a_board_you_have_read_has_nothing_further_to_offer_you() {
        let mut p = board();
        assert_eq!(p.unread_names_at(HERE, "m1"), vec!["the muster board"]);

        let first = p.read(HERE, "the muster board", "m1").expect("it is here");
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].text, "the lift on five is out");

        assert!(
            p.unread_names_at(HERE, "m1").is_empty(),
            "it was offered again with nothing new on it"
        );
        assert_eq!(
            p.read(HERE, "the muster board", "m1").unwrap().len(),
            0,
            "reading it twice handed the same line over twice"
        );
    }

    /// And somebody writing on it brings it back for everybody — which is what
    /// a board is for.
    #[test]
    fn a_new_line_makes_it_worth_reading_again() {
        let mut p = board();
        p.read(HERE, "the muster board", "m1");
        p.read(HERE, "the muster board", "m2");
        assert!(p.unread_names_at(HERE, "m1").is_empty());

        p.post(HERE, "the muster board", "Maker-03", "the lift is back");
        for who in ["m1", "m2"] {
            assert_eq!(
                p.unread_names_at(HERE, who),
                vec!["the muster board"],
                "{who} was not told there was something new"
            );
        }
        // And only the new line, not the whole board again.
        let fresh = p.read(HERE, "the muster board", "m1").unwrap();
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].text, "the lift is back");
    }

    /// Each reader has their own answer. One character reading a board must not
    /// mark it read for the room.
    #[test]
    fn one_reader_catching_up_does_not_catch_everybody_up() {
        let mut p = board();
        p.read(HERE, "the muster board", "m1");
        assert!(p.unread_names_at(HERE, "m1").is_empty());
        assert_eq!(p.unread_names_at(HERE, "m2"), vec!["the muster board"]);
    }

    /// A reader who has never seen it gets the whole thing, not nothing.
    #[test]
    fn somebody_who_has_never_looked_gets_what_is_on_it() {
        let mut p = board();
        p.post(HERE, "the muster board", "Maker-02", "second");
        let fresh = p.read(HERE, "the muster board", "newcomer").unwrap();
        assert_eq!(fresh.len(), 2);
    }

    /// A board is a rolling record. What falls off the top is gone, and every
    /// reader's cursor has to move with it — a valid number pointing at the
    /// wrong line is the failure nothing reports.
    #[test]
    fn trimming_moves_every_readers_cursor_with_it() {
        let mut p = board();
        p.read(HERE, "the muster board", "m1");
        for i in 0..(KEEP_LINES + 10) {
            p.post(HERE, "the muster board", "Maker-02", &format!("line {i}"));
        }
        let b = p.by_name_at(HERE, "the muster board").expect("here");
        assert_eq!(b.lines.len(), KEEP_LINES);
        assert!(
            b.unread_for("m1").len() <= KEEP_LINES,
            "a cursor outlived the lines it pointed at"
        );
        assert!(
            b.lines.iter().all(|l| l.text != "the lift on five is out"),
            "the oldest line survived the trim"
        );
    }

    /// A posting is somewhere. Reading is asked of the room a body is standing
    /// in, so a board two floors up is not on offer.
    #[test]
    fn a_posting_belongs_to_its_room_and_is_not_readable_from_another() {
        let mut p = board();
        p.post(
            "vault-command/watch",
            "the watch list",
            "Keeper",
            "who is up",
        );
        assert_eq!(p.unread_names_at(HERE, "m1"), vec!["the muster board"]);
        assert_eq!(
            p.unread_names_at("vault-command/watch", "m1"),
            vec!["the watch list"]
        );
        assert!(p.by_name_at(HERE, "the watch list").is_none());
    }

    /// Reading something that is not here is a different answer from reading
    /// something here that has nothing new, and the caller renders them apart.
    #[test]
    fn nothing_here_by_that_name_is_not_the_same_as_nothing_new_on_it() {
        let mut p = board();
        assert!(p.read(HERE, "the dispatch board", "m1").is_none());
        p.read(HERE, "the muster board", "m1");
        assert_eq!(
            p.read(HERE, "the muster board", "m1").map(|f| f.len()),
            Some(0)
        );
    }

    #[test]
    fn who_wrote_it_is_kept() {
        let p = board();
        let b = p.by_name_at(HERE, "the muster board").unwrap();
        assert_eq!(b.lines[0].by, "Maker-01");
    }
}
