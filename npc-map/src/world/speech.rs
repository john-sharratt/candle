//! Lets bodies speak, gesture and act in a room, and records who each act could
//! be addressed to.
//!
//! Speech is heard by everyone in the room whatever it is aimed at, so direction
//! is part of the utterance rather than a filter on its audience. A decision
//! records the company its body was offered when it began, so a line can still
//! reach someone who stepped out while it was being chosen, and only that
//! person. The world refuses an addressee who is not in the room, and a shout
//! reaches every room that can see into this one.

use super::*;

/// Who was in a body's room at the moment it began deciding what to do.
#[derive(Debug, Clone)]
pub(super) struct Deliberation {
    /// Where the body stood.
    pub(super) at: Where,
    /// Everybody else standing there, by id.
    pub(super) company: BTreeSet<String>,
}

impl World {
    /// Fix who `id` can address, as of now: everybody standing in its room.
    ///
    /// **A body decides over seconds, and the room does not wait.** What it may
    /// say and to whom is read once, before it starts, and the person it means
    /// to answer can walk out while it works. Without a record of who was there,
    /// the only thing left to check the finished act against is the room as it
    /// has become, and that refuses a line the body was correctly offered. This
    /// is the record: [`World::just_left`] and [`World::call_after`] read it, so
    /// an addressee drawn from the company the body decided with is still
    /// reachable after they step out.
    ///
    /// Does not move the clock — it observes the world, it does not happen in
    /// it. A second call replaces the first, and [`World::end_decision`] ends it.
    pub fn begin_decision(&mut self, id: &str) {
        let Some(actor) = self.actors.get(id) else {
            return;
        };
        let at = actor.at.clone();
        let company = self
            .actors_at(&at)
            .into_iter()
            .filter(|a| a.id != id)
            .map(|a| a.id.clone())
            .collect();
        self.deciding
            .insert(id.into(), Deliberation { at, company });
    }

    /// The decision `id` was making is over: nothing it does from here on is
    /// answered from the company it decided with.
    pub fn end_decision(&mut self, id: &str) {
        self.deciding.remove(id);
    }

    /// The decision `id` is making, as a world elsewhere would take it in: the
    /// place it decided from and the ids of the company it decided with.
    ///
    /// What a body acting in a world it does not run carries with its act, so
    /// the world that runs it answers the act from the company the body was
    /// offered — [`World::restore_decision`] puts it back.
    pub fn deliberation(&self, id: &str) -> Option<(Where, Vec<String>)> {
        self.deciding
            .get(id)
            .map(|d| (d.at.clone(), d.company.iter().cloned().collect()))
    }

    /// Take in a decision `id` began in another copy of this world: from `at`,
    /// with `company`. Replaces any decision it was making here, as a second
    /// [`World::begin_decision`] would.
    pub fn restore_decision(&mut self, id: &str, at: Where, company: Vec<String>) {
        if self.actors.contains_key(id) {
            self.deciding.insert(
                id.into(),
                Deliberation {
                    at,
                    company: company.into_iter().collect(),
                },
            );
        }
    }

    /// Where `id` stood when it began deciding, while it is deciding.
    ///
    /// The room the grammar was built from, and so the room an act chosen from it
    /// is about — whatever leg a moving body has covered since. `None` once
    /// [`World::end_decision`] has run, and for a body that is not deciding.
    pub fn decided_at(&self, id: &str) -> Option<&Where> {
        self.deciding.get(id).map(|d| &d.at)
    }

    /// Somebody `speaker` could call after: named `name`, in the company it
    /// began deciding with ([`World::begin_decision`]), and no longer in the room
    /// it stands in now.
    ///
    /// The name must be the whole of theirs, in any case. Somebody still with
    /// the speaker is not here — that is the ordinary case, answered by an
    /// ordinary aimed line — and a name nobody in the company answered to
    /// resolves to nobody.
    pub fn just_left(&self, speaker: &str, name: &str) -> Option<String> {
        let want = name.trim();
        let now_at = &self.actor(speaker)?.at;
        self.deciding
            .get(speaker)?
            .company
            .iter()
            .filter_map(|id| self.actor(id))
            .filter(|a| &a.at != now_at)
            .find(|a| a.name.eq_ignore_ascii_case(want))
            .map(|a| a.id.clone())
    }

    /// Say something to the room. Nobody outside it hears.
    pub fn say(&mut self, id: &str, words: impl Into<String>) -> Done {
        self.utter(id, None, words, Voice::Said)
    }

    /// Call out, to anybody within earshot: the room, and every room that can
    /// see into it. Aimed at nobody — a shout is for whoever hears it.
    pub fn shout(&mut self, id: &str, words: impl Into<String>) -> Done {
        self.utter(id, None, words, Voice::Shouted)
    }

    /// Say something to one person too quietly for anybody else to make out.
    ///
    /// The rest of the room still sees it happen — two heads together is a
    /// thing people notice — and hears none of it. The listener must be here.
    pub fn whisper(&mut self, id: &str, to: &str, words: impl Into<String>) -> Done {
        self.utter(id, Some(to), words, Voice::Whispered)
    }

    /// Say something to somebody in particular.
    ///
    /// It still lands in the room. **Direction is part of the utterance, not a
    /// filter on who receives it** — you aim your voice at a person and
    /// everybody standing there hears you do it, and knows it was not for them.
    /// That asymmetry is the whole social value of a shared room: overhearing
    /// *the creator told Maker-01 to get out* is a different fact from being
    /// told it, and both are true of the same event.
    ///
    /// The listener must be in the room. Talking *about* somebody who is not
    /// there is [`World::say`] with their name in the words.
    pub fn tell(&mut self, id: &str, to: &str, words: impl Into<String>) -> Done {
        self.utter(id, Some(to), words, Voice::Said)
    }

    /// Do something in the room without speaking. Everybody standing there sees
    /// it; nobody outside does — the same reach as [`World::say`], because it is
    /// the same room.
    ///
    /// # `what` is a predicate, verb and all
    ///
    /// It is rendered as `"{who} {what}"` — `witness::verb_phrase` supplies a
    /// verb for every *other* happening (`said …`, `came in`, `took …`) and
    /// takes this one as given, because only the caller knows whether the deed
    /// was a gesture, a hand laid on somebody, or a body going still.
    ///
    /// So `"gestures towards the table"`, not `"towards the table"`. A fragment
    /// reaches the other characters as a sentence with the verb missing —
    /// measured live, nine gestures arrived as *"Yaelis Vayne towards the table,
    /// indicating the standing orders"*.
    ///
    /// Do **not** name the target in `what`: pass it as `to` and the aiming is
    /// added once, as `", at X"`. Both is how one act came to read
    /// `"to Wren: steadying her, at Wren"`.
    pub fn show(&mut self, id: &str, to: Option<&str>, what: impl Into<String>) -> Done {
        let (place, to) = self.aim(id, to)?;
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::Did {
                to,
                what: what.into(),
            },
        });
        Ok(())
    }

    /// Raise your voice after somebody who has just left the room.
    ///
    /// The one aimed line that carries to somebody out of the room. [`World::tell`] and its
    /// kin refuse an addressee who is not here, because two people talking need
    /// to be in the same room — but somebody mid-sentence when the other steps
    /// out has one parting line to call after them, and losing it turned a moving
    /// cast's exchanges into questions nobody ever answered.
    ///
    /// Delivered under a rule the world enforces itself rather than trusts from
    /// the caller: `to` must have been in the company `id` began deciding with
    /// ([`World::begin_decision`]) — somebody who was there when the line was
    /// chosen — and must still be in the world. How far they have got does not
    /// matter: the line is addressed to them, and [`crate::witness`] carries it
    /// to the one it is aimed at and to nobody else, at any distance.
    ///
    /// It logs where the speaker decided, which is where the line was meant
    /// for, so a speaker who has also walked out has not moved its words.
    pub fn call_after(
        &mut self,
        id: &str,
        to: &str,
        words: impl Into<String>,
        voice: Voice,
    ) -> Done {
        if !self.actors.contains_key(id) {
            return Err(Refused::NoSuchActor(id.into()));
        }
        if id == to {
            return Err(Refused::SpeakingToYourself);
        }
        let place = match self.deciding.get(id) {
            Some(d) if d.company.contains(to) && self.actors.contains_key(to) => d.at.clone(),
            _ => return Err(Refused::NotHere { who: to.into() }),
        };
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::Said {
                to: Some(to.into()),
                words: words.into(),
                voice,
            },
        });
        Ok(())
    }

    /// The building does something in a room. Nobody did it.
    ///
    /// Takes a place rather than an actor, which is the only entry point here
    /// that does — see [`Happening::Stirred`]. A room that is not in the map is
    /// refused rather than logged, because an event nobody can ever be standing
    /// in is one that will sit in the window until it is trimmed.
    pub fn stir(&mut self, place: &Where, text: impl Into<String>, weight: Weight) -> Done {
        if self.node(place).is_none() {
            return Err(Refused::NoSuchPlace(place.clone()));
        }
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            // No actor. Every reader compares this against its own id to decide
            // whether an event is its own doing, and no body is called this, so
            // a stirring is nobody's doing to everybody.
            actor: String::new(),
            place: place.clone(),
            what: Happening::Stirred {
                text: text.into(),
                weight,
            },
        });
        Ok(())
    }

    /// Resolve who an utterance or gesture is aimed at, and where it happens.
    ///
    /// Shared by speech and gesture because the rule is about the *room*, not
    /// about the channel: you can only aim at somebody standing with you, and
    /// aiming at yourself is not a thing you can do.
    pub(super) fn aim(
        &self,
        id: &str,
        to: Option<&str>,
    ) -> Result<(Where, Option<String>), Refused> {
        let actor = self.actors.get(id).ok_or(Refused::NoSuchActor(id.into()))?;
        let place = actor.at.clone();
        if let Some(to) = to {
            if to == id {
                return Err(Refused::SpeakingToYourself);
            }
            if !self.actors.get(to).is_some_and(|a| a.at == place) {
                return Err(Refused::NotHere { who: to.into() });
            }
        }
        Ok((place, to.map(String::from)))
    }

    pub(super) fn utter(
        &mut self,
        id: &str,
        to: Option<&str>,
        words: impl Into<String>,
        voice: Voice,
    ) -> Done {
        let (place, to) = self.aim(id, to)?;
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::Said {
                to,
                words: words.into(),
                voice,
            },
        });
        Ok(())
    }
}
