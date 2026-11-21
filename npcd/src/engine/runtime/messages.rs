//! Carrying messages from people outside a world to its characters and channels.
//!
//! A person's message to a character, or to a world's standing channel, lands on
//! the thread the characters use and reaches them through the ordinary world
//! moment. The character's reply is then the character speaking from inside the
//! world, which the world can see and answer. A broadcast that put words into a
//! mind from outside would be seen by nothing in the world, so it is not used
//! here. A person is enrolled on the channel before their words go onto it, and a
//! character with no handset is said to be unable to answer rather than left
//! looking like a working console.

use super::*;

impl Runtime {
    /// Post one line to the standing chat channel of every world this daemon
    /// runs, as the command table — the channel half of a call to the table.
    pub fn say_to_all_channels(&self, from: &Who, text: &str) {
        for id in self.hosted.ids() {
            if self.standing.writes(&id) {
                let _ = self.say_on_channel(&id, from, text);
            }
        }
    }

    /// Say something to a character on its handset, as a person outside the
    /// world.
    ///
    /// **The same thread the characters use, not a side channel.** A person
    /// messaging a character is one more party on a conversation: it lands in
    /// [`crate::sim::phone`], the character is told about it by the ordinary
    /// sweep, and it answers with the ordinary `message` act. A private pipe
    /// between a console and a mind would be a different thing wearing the same
    /// word — the reply would not be the character speaking from inside the
    /// world, and nothing else in the world could ever see that it had happened.
    ///
    /// Returns what the character is called and whether it can answer. `None`
    /// when it has no body, because there is then nothing to reach it on.
    ///
    /// `origin` is where the message was written when not in the console —
    /// a Discord DM — and goes onto the thread with it, so the mirror knows
    /// it for one it must not post back.
    pub fn message_npc(
        &self,
        npc_id: u64,
        from: &Who,
        text: &str,
        origin: Option<Origin>,
    ) -> Option<MessageSent> {
        let (hosted, body) = self.body_of(npc_id)?;
        let them = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()))?;
        let sent = hosted.with_sim(|sim| {
            // Everybody a handset can reach comes from the cast rather than the
            // room, so a person messaging in from outside has to be made known
            // to the world's phones or the character is offered nobody to
            // answer. See [`crate::sim::Sim::enrol`].
            sim.enrol(from);
            let id = sim.threads.reach(from, &Who::new(&body, &them));
            // A person is a member by their account, and says so on the line.
            let said = Said {
                text: text.to_string(),
                origin,
                account: Some(from.id.clone()),
                ..Said::default()
            };
            let _ = sim.threads.send_on(&id, &from.id, said);
            MessageSent {
                with: them.clone(),
                waiting_for_them: sim.threads.get(&id).map_or(0, |t| t.unread_for(&body)),
                // A character with no handset cannot answer at all — the
                // messaging acts are gated on carrying one — and saying so is
                // better than a console that looks like it is working.
                can_reply: sim.has_phone(&body),
            }
        });
        Some(sent)
    }

    /// The conversation between a person and a character, oldest first: who
    /// said each line, by name, and its words.
    pub fn messages_with(
        &self,
        npc_id: u64,
        from: &str,
    ) -> Option<(String, Vec<(String, String)>)> {
        let (hosted, body) = self.body_of(npc_id)?;
        let them = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()))?;
        let said = hosted.sim(|sim| {
            sim.threads
                .direct_between(from, &body)
                .map(|t| {
                    t.messages
                        .iter()
                        .map(|m| {
                            let by = t.member(&m.from).map_or(m.from.clone(), |w| w.name.clone());
                            (by, m.text.clone())
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        });
        Some((them, said))
    }

    /// Say something on a world's standing channel, as a person outside it.
    ///
    /// **The same channel the cast is on, not an operator broadcast.** There is
    /// already an act that reaches every character regardless of where they are
    /// standing — `pulse::broadcast` — and it is a different thing on purpose:
    /// it puts words into a mind from outside the fiction, and nothing in the
    /// world can see that it happened or answer it. This is a person speaking
    /// on a conversation, so it lands in [`crate::sim::phone`], arrives through
    /// the ordinary sweep, and can be answered with the ordinary `message` act
    /// by any character that has something to say back.
    ///
    /// Returns who is on the channel to hear it. `None` when the world is not
    /// hosted.
    pub fn say_on_channel(&self, world_id: &str, from: &Who, text: &str) -> Option<Vec<String>> {
        self.say_on_channel_from(world_id, from, text, None)
    }

    /// [`Self::say_on_channel`], for words that came from `origin` outside
    /// the world — the world's Discord server's `#the-channel`, whose bridge
    /// never posts them back — or from inside it, with `None`. Answers who
    /// heard it, by name.
    pub fn say_on_channel_from(
        &self,
        world_id: &str,
        from: &Who,
        text: &str,
        origin: Option<Origin>,
    ) -> Option<Vec<String>> {
        let hosted = self.hosted.get(world_id)?;
        Some(hosted.with_sim(|sim| {
            // Enrolled first, so somebody speaking for the first time is on the
            // channel before their own words go onto it — otherwise `send`
            // refuses, having correctly found that they are on no such thread.
            sim.enrol(from);
            let said = Said {
                text: text.to_string(),
                origin,
                account: Some(from.id.clone()),
                ..Said::default()
            };
            match sim.threads.send(&from.id, CHANNEL, said) {
                Ok((_, reached)) => reached
                    .iter()
                    .map(|r| sim.threads.name_of(r).unwrap_or(r).to_string())
                    .collect(),
                Err(_) => Vec::new(),
            }
        }))
    }

    /// What has been said on a world's standing channel since `who` came
    /// onto it, oldest first: who said each line, by name, and its words.
    pub fn channel(&self, world_id: &str, who: &str) -> Option<Vec<(String, String)>> {
        let hosted = self.hosted.get(world_id)?;
        Some(hosted.sim(|sim| {
            sim.threads
                .by_kind_for(who, ThreadKind::Group, CHANNEL)
                .map(|t| {
                    t.since_joined(who)
                        .iter()
                        .map(|m| {
                            let by = t.member(&m.from).map_or(m.from.clone(), |w| w.name.clone());
                            (by, m.text.clone())
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }))
    }

    /// Leave words on something in a world, as a person outside it.
    ///
    /// **The world's half of `post_notice`.** A character can write on a board
    /// it is standing at; this is how whoever runs the world puts something on
    /// one without a body to do it — a notice that was there before the cast
    /// arrived, an instruction from outside, a sign somebody hung years ago.
    ///
    /// It stands the surface up if the map never named one, so a world can grow
    /// a board where it needs one rather than needing its map edited and
    /// reloaded.
    ///
    /// Returns how many lines the surface now holds. `None` when the world is
    /// not hosted.
    pub fn post_in_world(
        &self,
        world_id: &str,
        at: &str,
        on: &str,
        by: &str,
        text: &str,
    ) -> Option<usize> {
        let hosted = self.hosted.get(world_id)?;
        // A place that is not in the map would be a board nobody can ever stand
        // at to read — writable, invisible, and impossible to diagnose from the
        // outside. Refused here rather than written and lost.
        let real =
            hosted.read(|w| npc_map::world::Where::parse(at).is_some_and(|p| w.node(&p).is_some()));
        if !real {
            return None;
        }
        Some(hosted.with_sim(|s| {
            s.post(at, on, by, text);
            s.postings
                .by_name_at(at, on)
                .map(|p| p.lines.len())
                .unwrap_or(0)
        }))
    }
}
