//! The world's state that is not its shape.
//!
//! `npc_map` says where everybody is and what they are holding. This says
//! everything else a tool needs in order to do something: what a body carries,
//! what machines stand in a room, what can be dug out of the ground, what is
//! shooting, what the tower can afford, and the small durable things — promises,
//! orders, verdicts — that let a mission outlive a conversation.
//!
//! # No spatial simulation
//!
//! There is no geometry here, no distance, no line of sight, no facing. A room
//! is a string and two bodies in one are simply in it. That is deliberate: the
//! map already carries the only spatial fact the catalogue asks about — *is it
//! here* — and everything finer would be a model nothing reads, which is the
//! same defect as a parameter formula code cannot branch on.
//!
//! # A world instantiates only what it is
//!
//! The vault has terminals, orders and verdicts and no hostiles, no seams and no
//! tower. Battle Cities has all of it. Nothing declares that: **an absent facet
//! is an empty live set**, and the catalogue already drops a tool whose required
//! argument has nothing to choose from.
//!
//! So `engage` is not "disabled in the vault" — it is unreachable there, because
//! nothing in the vault can be engaged. One rule, no configuration, and no way
//! for a capability list to disagree with what the world actually holds.
//!
//! # Determinism
//!
//! Every collection here is ordered, every seed is explicit, and nothing reads a
//! clock or a random source. Two runs of the same seed produce byte-identical
//! state, which is what makes the tool tests able to assert a live set rather
//! than assert that one is plausible.

pub mod bench;
pub mod device;
pub mod field;
pub mod item;
pub mod ledger;
pub mod phone;
pub mod posting;
pub mod record;
pub mod seed;
pub mod tower;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use bench::Benches;
use device::Devices;
use field::{Field, Resource};
use item::Pack;
use ledger::Ledger;
use tower::Tower;

/// Everything about a world that is not its map.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sim {
    packs: BTreeMap<String, Pack>,
    /// The documents open at a bench, uncommitted — see [`bench`].
    pub bench: Benches,
    pub devices: Devices,
    pub field: Field,
    pub ledger: Ledger,
    /// What the world is made of, as the Makers hold it — see [`record`].
    pub record: record::Record,
    /// Every conversation carried on a handset — see [`phone`].
    pub threads: phone::Threads,
    /// Words left at a place for whoever comes by — see [`posting`].
    ///
    /// `serde(default)` because saved worlds predate it, and a world that loads
    /// with no postings is a world where nobody has written anything yet, which
    /// is the correct reading of an absent field rather than a migration.
    #[serde(default)]
    pub postings: posting::Postings,
    /// Everybody a phone could reach, by the name the world writes down.
    ///
    /// Not derived from who is standing where: the point of a handset is that
    /// it reaches somebody who is nowhere near, so the set has to come from the
    /// cast rather than from the room.
    roster: Vec<String>,
    /// Absent in a world with no tower — the vault has none, and Keeper's whole
    /// vocabulary goes with it.
    pub tower: Option<Tower>,
    /// `area/node` → the tools the parts standing there declare.
    ///
    /// Read straight off the map at seed time. What makes an act reachable is
    /// the part that offers it, and a part offers a tool whether or not it also
    /// has states to be put into.
    part_tools: BTreeMap<String, Vec<String>>,
    /// Where a recall lands, if this world has anywhere to muster from.
    home: Option<String>,
}

impl Sim {
    pub fn new() -> Sim {
        Sim::default()
    }

    // ---- what a body carries ----

    /// A body's pack, empty rather than absent for one that has never held
    /// anything. "Carrying nothing" and "not a body" are the same to every
    /// caller, and an `Option` here would make every call site say so twice.
    pub fn pack(&self, body: &str) -> &Pack {
        static EMPTY: std::sync::OnceLock<Pack> = std::sync::OnceLock::new();
        self.packs
            .get(body)
            .unwrap_or_else(|| EMPTY.get_or_init(Pack::new))
    }

    pub fn pack_mut(&mut self, body: &str) -> &mut Pack {
        self.packs.entry(body.to_string()).or_default()
    }

    /// Move something between bodies, all or nothing.
    ///
    /// `Err` carries what went wrong in the second person, because it is going
    /// to be read by the character that tried it.
    pub fn hand_over(
        &mut self,
        from: &str,
        to: &str,
        name: &str,
        count: u32,
    ) -> Result<item::Item, String> {
        let Some(held) = self.pack(from).by_name(name).cloned() else {
            return Err(format!("You are not carrying {name}."));
        };
        if count == 0 {
            return Err("Handing over none of something is not handing it over.".into());
        }
        let Some(moved) = self.pack_mut(from).take(&held.id, count) else {
            return Err(format!(
                "You have {} {}, not {count}.",
                held.count, held.name
            ));
        };
        self.pack_mut(to).add(moved.clone());
        Ok(moved)
    }

    /// Whether anybody in this world carries anything. What keeps `give`,
    /// `equip` and `use` out of a world with no items in it at all.
    pub fn has_packs(&self) -> bool {
        self.packs.values().any(|p| !p.is_empty())
    }

    // ---- live sets ----
    //
    // Each of these answers "what may this argument be, for this body, standing
    // here". An empty answer drops the parameter when it is optional and the
    // whole tool when it is required, which is how a world without a facet ends
    // up without the tools that need it.

    /// `give.what`, `use.what` and the rest of the pack-bound arguments.
    pub fn carried(&self, body: &str) -> Vec<String> {
        self.pack(body).names()
    }

    pub fn equippable(&self, body: &str) -> Vec<String> {
        self.pack(body).equippable()
    }

    pub fn usable(&self, body: &str) -> Vec<String> {
        self.pack(body).usable()
    }

    /// `operate.what`.
    pub fn operable(&self, place: &str) -> Vec<String> {
        self.devices.operable_at(place)
    }

    /// `operate.mode` — the dependent one, which needs to know what was chosen.
    pub fn modes_of(&self, place: &str, what: &str) -> Vec<String> {
        self.devices.modes_of(place, what)
    }

    /// Every mode of every machine here, deduplicated and ordered.
    ///
    /// What the grammar's `mode` branch is built from, because that branch is
    /// built once per situation and not once per device. It is a *narrowing* —
    /// a body on the rampart may write "locked" and be refused by the turret
    /// rather than by the tree — and the honest half of a dependent binding the
    /// trie could express and the current front end does not.
    ///
    /// # A machine is not offered the state it is already in
    ///
    /// [`crate::sim::device::Device::set`] accepts any mode the machine admits,
    /// including the one it currently holds — so setting the accession desk to
    /// `reading` while it is already `reading` succeeds, changes nothing, and
    /// reports *"You set the accession desk to reading."* A character has no
    /// way to see it was already there, so nothing discourages a fourth attempt:
    /// measured live, three in a row.
    ///
    /// That is the same defect as `read` above and as the channel `invite`
    /// (see [`Self::invitable_for`]) — an act the grammar offers that cannot
    /// accomplish anything — and it leaves the same way. A mode is offered only
    /// while some machine here both admits it *and* is not already in it.
    ///
    /// Still a narrowing rather than a guarantee, for the reason above: the
    /// branch is per situation, not per device, so a room holding a desk in
    /// `reading` and a door in `open` still offers both words and the act
    /// checks the pairing. What it can no longer do is offer a state nothing
    /// here could move into.
    pub fn modes_here(&self, place: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for d in self.devices.at(place) {
            for m in &d.modes {
                if m.eq_ignore_ascii_case(&d.mode) {
                    continue;
                }
                if !out.contains(m) {
                    out.push(m.clone());
                }
            }
        }
        out
    }

    /// `read.what` — what there is here with something on it this body has not
    /// seen.
    ///
    /// # Machines are not readable, and offering them was the whole bug
    ///
    /// This returned every device in the room. A `Device` holds no text — its
    /// only content is the mode it is switched to — so `read` answered with
    /// that, and *"You read the accession desk. It stands at `reading`."* was
    /// forty-seven of fifty acts in a live feed. The act reported success,
    /// returned nothing a character could use, and `body::ANSWERS` brought the
    /// character straight back to use it, so it read the same thing again.
    ///
    /// Nothing about that was fixable in the prose. A machine is *operated*,
    /// and what a station is holding belongs to the body holding it; neither is
    /// a text. So machines are gone from here, and what is left is the store
    /// that actually holds words at a place — see [`posting`].
    ///
    /// # Bound to the reader, so the act can run out
    ///
    /// The set is what this body has **not read**, not what is here. A board
    /// somebody has already read offers them nothing, so the empty-set rule
    /// takes `read` out of the grammar rather than leaving an act that can be
    /// re-emitted for the same answer for ever. That is the same discipline
    /// every other closed set here follows, and it is what makes the loop
    /// unrepresentable instead of merely discouraged.
    pub fn readable_at(&self, place: &str, who: &str) -> Vec<String> {
        let mut out = self.postings.unread_names_at(place, who);
        // The orders keep their own cursor, for the same reason a board does:
        // they read the same every time, so offering them unconditionally is
        // the same act-that-returns-nothing loop in a different costume. See
        // [`ledger::Ledger::unheld_unseen_by`].
        if self.ledger.unheld_unseen_by(who).is_some() && !out.iter().any(|n| n.contains("board")) {
            out.push("the standing orders".to_string());
        }
        out
    }

    /// `post_notice.on` — the surfaces here words can be left on.
    ///
    /// Everything standing here regardless of what is on it, unlike
    /// [`Sim::readable_at`]: a blank board is the one you most want to write on
    /// and the one there is nothing to read on.
    pub fn postable_at(&self, place: &str) -> Vec<String> {
        self.postings.postable_names_at(place)
    }

    /// Leave words on a surface, standing it up if the map never named one.
    pub fn post(&mut self, at: &str, name: &str, by: &str, text: &str) {
        self.postings.post(at, name, by, text);
    }

    /// `claim.what` — what a body standing here can take and hold.
    ///
    /// Two unlike things, deliberately in one set: the machines that are worked
    /// exclusively, and the orders lying unheld on a board. A character does not
    /// distinguish them — both are *taking something on* — and splitting them
    /// would be two tools for one act.
    pub fn claimable_at(&self, place: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .devices
            .at(place)
            .into_iter()
            .filter(|d| d.claimable && d.working)
            .map(|d| d.name.clone())
            .collect();
        out.extend(self.ledger.unheld());
        out
    }

    /// `gather.what`.
    pub fn extractable(&self, place: &str) -> Vec<String> {
        self.field.extractable_at(place)
    }

    /// `engage.target`.
    pub fn hostiles(&self, place: &str) -> Vec<String> {
        self.field.hostiles_at(place)
    }

    /// Whether an act is reachable from a place because a part there offers it.
    ///
    /// **The map is what decides, through the tools its parts declare.** Not
    /// the device list — a part becomes a device only if it has *modes*, and a
    /// bridge console has none: it is worked by being spoken to, not by being
    /// put into a state. Reading availability off the devices therefore silently
    /// lost every station whose whole job is one act, which is most of them.
    ///
    /// A world that grows a second kind of fabricator, or moves the console,
    /// needs no change here: it declares the tool on the part and the act
    /// arrives with it.
    pub fn part_offers(&self, place: &str, tool: &str) -> bool {
        self.part_tools
            .get(place)
            .is_some_and(|ts| ts.iter().any(|t| t == tool))
    }

    /// Whether this body is carrying a handset.
    ///
    /// The one gate on every messaging act. Nothing else needs to know: an
    /// empty thread list is what removes them, and this is why it is empty.
    pub fn has_phone(&self, body: &str) -> bool {
        self.pack(body).has(phone::PHONE)
    }

    /// `message.to`, `invite.to`, `sign_off.to` — the conversations this
    /// character has, as it names them. Empty without a handset.
    pub fn threads_for(&self, me: &str, body: &str) -> Vec<String> {
        match self.has_phone(body) {
            true => self.threads.names_for(me),
            false => Vec::new(),
        }
    }

    /// `sign_off.to` — the conversations this character can actually leave.
    ///
    /// Its own threads, less the world's standing channels. Leaving a
    /// conversation is a real decision and belongs to the character; leaving
    /// the open channel is not one, because there is no way back to it and
    /// nothing in the world would report that somebody had gone. See
    /// [`crate::engine::tools::Choices::Leavable`].
    pub fn leavable_for(&self, me: &str, body: &str) -> Vec<String> {
        if !self.has_phone(body) {
            return Vec::new();
        }
        self.threads
            .of(me)
            .into_iter()
            .filter(|t| !t.permanent)
            .map(|t| t.as_named_to(me))
            .collect()
    }

    /// `invite.to` — the conversations somebody could actually be brought into.
    ///
    /// Its own threads, less the ones every invitable person is **already on**.
    ///
    /// # Why this is not [`Self::threads_for`]
    ///
    /// [`phone::CHANNEL`] holds the entire cast, permanently, by construction —
    /// so there is nobody alive who could be invited to it, and `invite` against
    /// it fails the same way every time: `"X is already on it."` Offered the
    /// whole thread list, a character picks the one thread it shares with
    /// everybody, is refused, and picks it again.
    ///
    /// Measured over 48 turns of three characters: 17 invites, every one of
    /// them to the channel, 16 refused. It was the second-most-called act in
    /// the cast and it could never once have worked.
    ///
    /// A refusal a character cannot avoid is a defect in the branch, not a
    /// lesson — so the thread leaves the set. Empty for a character whose only
    /// thread is the channel, which takes `invite` out of the grammar entirely:
    /// correct, because there is then nowhere to invite anybody.
    pub fn invitable_for(&self, me: &str, body: &str) -> Vec<String> {
        if !self.has_phone(body) {
            return Vec::new();
        }
        let who = self.contacts_for(me, body);
        self.threads
            .of(me)
            .into_iter()
            .filter(|t| who.iter().any(|w| !t.has(w)))
            .map(|t| t.as_named_to(me))
            .collect()
    }

    /// `invite.who` — who could be brought into **any** conversation this
    /// character could bring somebody into.
    ///
    /// # Why this is not [`Self::contacts_for`]
    ///
    /// `invite` takes two arguments and they are not independent: a thread, and
    /// somebody to put on it. A flat grammar states each arm on its own, so
    /// narrowing them separately says "these threads have room" and "these
    /// people are reachable" — and lets the character pick a thread and a
    /// person already on it. Measured after the thread set was narrowed: five
    /// more `"X is already on it."`, from the pairing rather than from either
    /// half.
    ///
    /// The conservative intersection is what a flat grammar can state
    /// truthfully: offer only people who are on **none** of the threads on
    /// offer, and every pair the character can name is a pair that works. It
    /// gives up some legal combinations — somebody on one of two threads could
    /// have been invited to the other — and that is the right trade here, where
    /// a refusal a character cannot foresee is worth more than a call it never
    /// gets to make.
    pub fn invitees_for(&self, me: &str, body: &str) -> Vec<String> {
        let threads = self.invitable_for(me, body);
        if threads.is_empty() {
            return Vec::new();
        }
        self.contacts_for(me, body)
            .into_iter()
            .filter(|who| {
                self.threads
                    .of(me)
                    .into_iter()
                    .filter(|t| threads.contains(&t.as_named_to(me)))
                    .all(|t| !t.has(who))
            })
            .collect()
    }

    /// `reach_out.to`, `open_group.with` — who could be reached and has no
    /// conversation of their own with this character yet.
    ///
    /// **Everybody the world writes down, less the people already reachable a
    /// shorter way.** Offering somebody you are already talking to under
    /// `reach_out` is offering a choice whose right answer is the other act.
    ///
    /// # "Already talking to" means a *direct* thread, not any thread
    ///
    /// This asked whether the two shared any conversation at all, which was the
    /// same question while the only way to share one was to have started it.
    /// [`phone::CHANNEL`] made every character share a thread with every other
    /// character on arrival — so the answer became "everybody", this returned
    /// empty for the whole cast, and the ordinary empty-set rule took
    /// `reach_out`, `invite` and `open_group` out of the grammar completely.
    /// The cast would have gained a channel and lost the ability to say
    /// anything to one person in private.
    ///
    /// A direct thread is the right test because it is exactly what
    /// [`phone::Threads::reach`] would find: being on a channel with fifty
    /// people is not having a conversation with any of them.
    pub fn contacts_for(&self, me: &str, body: &str) -> Vec<String> {
        if !self.has_phone(body) {
            return Vec::new();
        }
        self.roster
            .iter()
            .filter(|n| n.as_str() != me)
            .filter(|n| self.threads.direct_between(me, n).is_none())
            .cloned()
            .collect()
    }

    /// Everybody a phone could reach, by the name the world writes down.
    pub fn set_roster(&mut self, names: Vec<String>) {
        self.roster = names;
    }

    /// Make somebody known to this world's phones: on the roster, and on the
    /// standing channel.
    ///
    /// **One place, because the two must not drift.** Being on the roster is
    /// what lets a character `reach_out` to you; being on the channel is what
    /// lets it hear you at all. Somebody with one and not the other is a
    /// half-present person — reachable but silent, or audible and impossible to
    /// answer — and both halves were being written by hand at every site that
    /// introduced somebody.
    ///
    /// Idempotent, and called on every arrival and every message from outside
    /// the world, because neither of those is a one-off event.
    pub fn enrol(&mut self, name: &str) {
        if !self.roster.iter().any(|n| n == name) {
            self.roster.push(name.to_string());
            self.roster.sort();
        }
        self.threads.join_channel(phone::CHANNEL, name);
    }

    /// Who is on the roster now.
    pub fn contacts_roster(&self) -> Vec<String> {
        self.roster.clone()
    }

    /// How many messages are waiting for somebody across every thread.
    pub fn messages_waiting(&self, me: &str) -> usize {
        self.threads.waiting_for(me)
    }

    /// Every act the parts standing here carry, in a stable order.
    ///
    /// What gates `Availability::AtPart`. Straight off the map, so a room that
    /// grows a station grows its acts, and walking out takes them away.
    pub fn station_tools(&self, place: &str) -> Vec<String> {
        self.part_tools.get(place).cloned().unwrap_or_default()
    }

    /// Whether anything here can make something.
    pub fn makes_here(&self, place: &str) -> bool {
        self.tower.is_some() && self.part_offers(place, "produce")
    }

    /// Whether the tower can be spoken to as a whole from here.
    pub fn commands_here(&self, place: &str) -> bool {
        self.tower.is_some() && self.part_offers(place, "command_tower")
    }

    /// Where a body may be recalled to, which is nowhere in a world with no
    /// muster point. `recall` binds its destination to this, so the act is
    /// absent rather than refused in a building nobody musters from.
    pub fn homes(&self) -> Vec<String> {
        self.home.iter().cloned().collect()
    }

    /// `produce.what` — only what the stockpile covers.
    pub fn makeable(&self) -> Vec<String> {
        self.tower.as_ref().map(Tower::makeable).unwrap_or_default()
    }

    /// `produce.queue`.
    pub fn free_queues(&self) -> Vec<String> {
        self.tower
            .as_ref()
            .map(Tower::free_queues)
            .unwrap_or_default()
    }

    /// `command_tower.action` — what the tower can afford right now.
    pub fn tower_actions(&self) -> Vec<String> {
        self.tower.as_ref().map(Tower::actions).unwrap_or_default()
    }

    /// `remind.which` — what this person owes you.
    pub fn owed_to(&self, me: &str, by: &str) -> Vec<String> {
        self.ledger.owed_to(me, by)
    }

    /// `claim.what` at an order table, and `release`'s subject.
    pub fn unheld_orders(&self) -> Vec<String> {
        self.ledger.unheld()
    }

    /// Whether this world has anywhere to be recalled to.
    pub fn has_home(&self) -> bool {
        self.home.is_some()
    }

    /// Record what the map says, at seed time.
    pub fn set_part_tools(&mut self, at: impl Into<String>, tools: Vec<String>) {
        self.part_tools.insert(at.into(), tools);
    }

    /// Name the muster point a recall lands at.
    pub fn set_home(&mut self, at: impl Into<String>) {
        self.home = Some(at.into());
    }

    /// Point this world's benches at the documents they work on.
    ///
    /// A world told nothing has no documents, and every bench act refuses
    /// rather than inventing somewhere to write — the same rule the tower and
    /// the field run on.
    pub fn set_bench_root(&mut self, root: impl Into<std::path::PathBuf>) {
        let root = root.into();
        // The record learns what is actually there at the same moment, so there
        // is no window in which a bench can reach a document the stations have
        // never heard of.
        self.record.index_canon(&root);
        self.bench.set_root(root);
    }

    /// Put a resource into the tower's stockpile, which is where a gathered
    /// load ends up once it is carried home.
    pub fn bank(&mut self, r: Resource, amount: u64) -> bool {
        match &mut self.tower {
            Some(t) => {
                t.put(r, amount);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use item::{Item, Kind};

    fn two_bodies() -> Sim {
        let mut s = Sim::new();
        s.pack_mut("c1")
            .add(Item::new("stimpak", "stimpak", Kind::Consumable, 2));
        s.pack_mut("c1")
            .add(Item::new("mono_sword", "mono sword", Kind::Weapon, 1));
        s.pack_mut("c2")
            .add(Item::new("bolt", "bolt rounds", Kind::Ammunition, 30));
        s
    }

    #[test]
    fn a_body_that_has_never_held_anything_carries_nothing_rather_than_being_absent() {
        let s = Sim::new();
        assert!(s.pack("nobody").is_empty());
        assert!(s.carried("nobody").is_empty());
    }

    /// A cast whose phones are on the channel and nothing else.
    fn on_the_channel(names: &[&str]) -> Sim {
        let mut s = Sim::new();
        for n in names {
            s.pack_mut(n)
                .add(Item::new(phone::PHONE, "handset", Kind::Gear, 1));
            s.enrol(n);
        }
        s
    }

    /// **A machine is not offered the state it is already in.**
    ///
    /// `Device::set` accepts the current mode, so the act succeeded and changed
    /// nothing — three identical `operate` calls in a row, live, with the
    /// character told "You set the accession desk to reading" each time.
    #[test]
    fn the_state_a_machine_is_already_in_is_not_offered() {
        let mut s = Sim::new();
        s.devices.install(device::Device::new(
            "desk",
            "the accession desk",
            device::Kind::Terminal,
            "receiving",
            &["reading", "idle"],
        ));
        // It starts in the first of its modes.
        let modes = s.modes_here("receiving");
        assert!(
            !modes.iter().any(|m| m == "reading"),
            "the desk was offered the state it is already in: {modes:?}"
        );
        assert!(
            modes.iter().any(|m| m == "idle"),
            "the state it could move to was dropped: {modes:?}"
        );
    }

    /// And moving it makes the state it left available again — the narrowing
    /// tracks the world rather than blacklisting a word.
    #[test]
    fn the_state_a_machine_has_left_becomes_offerable_again() {
        let mut s = Sim::new();
        s.devices.install(device::Device::new(
            "desk",
            "the accession desk",
            device::Kind::Terminal,
            "receiving",
            &["reading", "idle"],
        ));
        s.devices
            .by_name_at_mut("receiving", "the accession desk")
            .unwrap()
            .set("idle");
        let modes = s.modes_here("receiving");
        assert!(modes.iter().any(|m| m == "reading"), "{modes:?}");
        assert!(!modes.iter().any(|m| m == "idle"), "{modes:?}");
    }

    /// **The channel is never somewhere anybody can be invited.**
    ///
    /// Everybody is on it by construction, so `invite` against it is refused
    /// every time — measured live at 16 refusals out of 17 invites, the
    /// second-most-called act in the cast. It leaves the branch rather than
    /// being refused, so the character cannot make the mistake at all.
    #[test]
    fn the_channel_is_not_offered_as_somewhere_to_invite_anybody() {
        let s = on_the_channel(&["Wren", "Wailen Wylde", "Marek"]);
        assert!(
            s.threads_for("Wren", "Wren")
                .iter()
                .any(|t| t == "the channel"),
            "the channel should still be a thread Wren can message"
        );
        assert!(
            s.invitable_for("Wren", "Wren").is_empty(),
            "the channel was offered as an invite target: {:?}",
            s.invitable_for("Wren", "Wren")
        );
    }

    /// And with nowhere to bring anybody, `invite` leaves the grammar by the
    /// ordinary empty-set rule rather than by a special case — the set steers,
    /// so an empty one drops the required parameter and the act with it.
    #[test]
    fn a_character_with_only_the_channel_is_not_offered_invite_at_all() {
        use crate::engine::tools::Choices;
        let s = on_the_channel(&["Wren", "Marek"]);
        assert!(s.invitable_for("Wren", "Wren").is_empty());
        assert!(
            Choices::Invitable.steers(),
            "an empty set must drop the act"
        );
    }

    /// **Every pair the grammar admits is a pair that works.**
    ///
    /// The two arms of an `invite` are not independent, and narrowing them
    /// separately is not enough: the thread set said "these have room" and the
    /// person set said "these are reachable", and a character picked a thread
    /// and somebody already on it. Five more refusals, from the pairing rather
    /// than from either half.
    #[test]
    fn nobody_offered_for_an_invite_is_already_on_a_thread_being_offered() {
        let mut s = on_the_channel(&["Wren", "Wailen Wylde", "Marek"]);
        // Wren has a private thread with Wailen. Marek is on neither it nor any
        // other private thread, so he is the only sound answer.
        s.threads.reach("Wren", "Wailen Wylde");

        let threads = s.invitable_for("Wren", "Wren");
        let who = s.invitees_for("Wren", "Wren");
        assert!(!threads.is_empty(), "there was nowhere to invite anybody");
        assert!(who.iter().any(|w| w == "Marek"), "{who:?}");
        assert!(
            !who.iter().any(|w| w == "Wailen Wylde"),
            "somebody already on the offered thread was offered: {who:?}"
        );

        // The property, stated directly: no offered person is on any offered
        // thread, so no combination the character can name is refusable.
        for t in s.threads.of("Wren") {
            if !threads.contains(&t.as_named_to("Wren")) {
                continue;
            }
            for w in &who {
                assert!(!t.has(w), "{w} is already on {}", t.as_named_to("Wren"));
            }
        }
    }

    /// With nobody left to bring in, `invite` leaves the grammar rather than
    /// offering a thread and no one to put on it.
    #[test]
    fn an_invite_with_nobody_to_bring_offers_nobody() {
        let mut s = on_the_channel(&["Wren", "Wailen Wylde"]);
        s.threads.reach("Wren", "Wailen Wylde");
        assert!(
            s.invitees_for("Wren", "Wren").is_empty(),
            "the only other person is already on the only thread"
        );
    }

    /// A private thread still takes people — the fix removes the impossible
    /// target, not the act.
    #[test]
    fn a_private_thread_is_still_somewhere_to_bring_somebody() {
        let mut s = on_the_channel(&["Wren", "Wailen Wylde", "Marek"]);
        s.threads.reach("Wren", "Wailen Wylde");
        let invitable = s.invitable_for("Wren", "Wren");
        assert!(
            invitable.iter().any(|t| t == "Wailen Wylde"),
            "the private thread should accept Marek: {invitable:?}"
        );
        assert!(
            !invitable.iter().any(|t| t == "the channel"),
            "the channel is still not invitable: {invitable:?}"
        );
    }

    #[test]
    fn handing_over_moves_it_and_leaves_the_giver_short() {
        let mut s = two_bodies();
        let moved = s.hand_over("c2", "c1", "bolt rounds", 10).unwrap();
        assert_eq!(moved.count, 10);
        assert_eq!(s.pack("c2").get("bolt").unwrap().count, 20);
        assert_eq!(s.pack("c1").get("bolt").unwrap().count, 10);
    }

    #[test]
    fn handing_over_more_than_is_held_moves_nothing_and_says_how_many() {
        let mut s = two_bodies();
        let err = s.hand_over("c2", "c1", "bolt rounds", 99).unwrap_err();
        assert!(err.contains("30"), "the real count was not named: {err}");
        assert_eq!(s.pack("c2").get("bolt").unwrap().count, 30);
        assert!(
            !s.pack("c1").has("bolt"),
            "a refused hand-over still arrived"
        );
    }

    #[test]
    fn handing_over_what_you_do_not_have_is_refused_in_the_second_person() {
        let mut s = two_bodies();
        let err = s.hand_over("c1", "c2", "a plasma rifle", 1).unwrap_err();
        assert!(err.starts_with("You are not carrying"), "{err}");
    }

    #[test]
    fn the_pack_bound_sets_split_by_what_the_verb_can_take() {
        let s = two_bodies();
        assert_eq!(s.carried("c1"), vec!["mono sword", "stimpak"]);
        assert_eq!(s.equippable("c1"), vec!["mono sword".to_string()]);
        assert_eq!(s.usable("c1"), vec!["stimpak".to_string()]);
        // Ammunition is neither equipped nor used — the simulator spends it.
        assert!(s.equippable("c2").is_empty());
        assert!(s.usable("c2").is_empty());
    }

    #[test]
    fn a_world_with_no_tower_answers_every_tower_question_with_nothing() {
        // This is what keeps Keeper's vocabulary out of the vault, with no
        // capability flag to get out of step with the world.
        let mut s = Sim::new();
        assert!(!s.has_home());
        assert!(s.makeable().is_empty());
        assert!(s.free_queues().is_empty());
        assert!(s.tower_actions().is_empty());
        assert!(
            !s.bank(Resource::Ore, 10),
            "banked into a world with no tower"
        );
    }

    #[test]
    fn a_world_with_no_field_offers_nothing_to_gather_or_engage() {
        let s = two_bodies();
        assert!(s.extractable("anywhere").is_empty());
        assert!(s.hostiles("anywhere").is_empty());
        assert!(s.field.is_empty());
    }

    #[test]
    fn state_is_the_same_on_two_builds_of_the_same_seed() {
        // The property the tool tests rest on: assert a live set, not a shape.
        assert_eq!(two_bodies(), two_bodies());
        assert_eq!(
            serde_json::to_string(&two_bodies()).unwrap(),
            serde_json::to_string(&two_bodies()).unwrap()
        );
    }
}
