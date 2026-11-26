//! Holds what is true where a character stands, in the terms the grammar and its
//! cache use.
//!
//! `Within` gathers what the world knows about one body at one place: who is
//! present, which acts its surroundings and possessions make possible, its
//! cooldowns and the conversation it is in. It is small and copied per turn because
//! it is the key the compiled grammar is cached under, so two turns with the same
//! answers share one tree. Each set comes from the world, and an empty set is what
//! keeps an act out of a place that has no business offering it.

use super::*;

/// What is true where a character stands, in the grammar's own terms.
///
/// Small and copied per turn on purpose: it is the *key* the compiled grammar
/// is cached under, so it has to be cheap to compare and cheap to hash.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Within {
    /// Who is here, by the name the world writes down. Empty when alone.
    pub company: Vec<String>,
    /// The subset of [`Self::company`] that are people — `visitor:{handle}`
    /// bodies — by the same names. What room speech needs
    /// ([`Availability::PersonNearby`]).
    pub people: Vec<String>,
    /// Whether a shout from here reaches a person's body, by perception's own
    /// rule (`npc_map::witness::carries`).
    pub heard_by_person: bool,
    /// Whether this body is in field play: it holds a field stance, or a
    /// hostile is within its reach.
    pub in_field: bool,
    /// Everywhere this character may walk to, never including where it stands.
    /// Empty for a character with nowhere to go, which takes `move_to` out of
    /// the grammar entirely.
    pub places: Vec<String>,
    /// Every room this body may turn its attention to — [`Choices::Rooms`].
    /// Empty for a body that walks, which takes `focus` out of the grammar.
    pub rooms: Vec<String>,
    /// The world's standing jobs, by name — [`Choices::Jobs`].
    pub jobs: Vec<String>,
    /// The parameters each standing job that takes any declares, by job: what
    /// choosing it as `order_post`'s `job` brings as fields of its own.
    pub job_params: Vec<(String, Vec<String>)>,
    /// The house links in this room, by name — [`Choices::Links`]. Empty in a
    /// room with no network to survey.
    pub links: Vec<String>,
    /// The parts here linked to a Sonos player, by name — [`Choices::Sonos`].
    pub sonos: Vec<String>,
    /// The parts here linked to an AV receiver, by name — [`Choices::Receiver`].
    pub receiver: Vec<String>,
    /// The parts here linked to a Cast device, by name — [`Choices::Cast`].
    pub cast: Vec<String>,
    /// The parts here linked to a WiZ bulb, by name — [`Choices::Lights`].
    pub lights: Vec<String>,
    /// The TVs here the world pairs with and has no key for yet, by name —
    /// [`Choices::Pairable`]. What `pair_device` begins; empty once a TV is
    /// paired, which takes the act away.
    pub pairable: Vec<String>,
    /// The Google TVs here not yet paired, by name — [`Choices::Pairing`]. What
    /// `pair_code` finishes; a webOS TV never appears, since it takes no code.
    pub pairing: Vec<String>,
    /// The terminals a mind with no body may connect to, by name, never the
    /// one it is connected to — [`Choices::Terminals`]. Empty for a body.
    pub terminals: Vec<String>,
    /// The checks of the machine a mind is connected to, by name —
    /// [`Choices::Checks`].
    pub checks: Vec<String>,
    /// The devices on the house network, by the names they go by, for a mind
    /// connected to it — [`Choices::Devices`].
    pub devices: Vec<String>,
    /// The house's modes, where this character may set them —
    /// [`Choices::HouseModes`].
    pub house_modes: Vec<String>,
    /// The acts the carried mission grants here (`docs/zen_moon_mind.md` §14):
    /// what [`Availability::Granted`] reads. Empty unless the body carries a
    /// mission granting them, stands in its room, and the room is a mirror.
    pub grants: Vec<String>,
    /// The devices the last survey here found that no part stands for yet, by
    /// label — [`Choices::Findings`].
    pub findings: Vec<String>,
    /// The parts here that stand for no device yet, by name —
    /// [`Choices::Unlinked`].
    pub unlinked: Vec<String>,
    /// The devices here onboarded from a survey, by name — what may be
    /// offboarded: [`Choices::Onboarded`].
    pub onboarded: Vec<String>,
    /// The rooms off this one that may be undiscovered — in the mirror, and
    /// nobody in them — by name: [`Choices::Doors`].
    pub doors: Vec<String>,
    /// Acts this body has taken too recently to take again — see
    /// [`crate::engine::cooldown`].
    ///
    /// Struck from the grammar rather than refused afterwards, the same as
    /// everything else here: a character cannot get stuck emitting an act it
    /// has no way to say. Nearly always empty.
    pub cooling: Vec<String>,
    /// The only acts a conversation's turn may take (`docs/npc_conversations.md`
    /// §6.2), set by [`crate::engine::talk::within`]; `None` for the main loop,
    /// which the room alone bounds. Emptying the world's sets is not enough to
    /// keep a thread's turn off the room: `sleep`, `read`, `invite` and the
    /// rest need none of them.
    pub allow: Option<Vec<String>>,
    /// The registers `<mind>/moods/` has a mood written for — see
    /// [`Choices::Feelings`]. The same list for every character in the daemon,
    /// because the library is ingested untagged and shared.
    pub feelings: Vec<String>,

    /// Everything in the pack, by the name the world wrote for it.
    pub carried: Vec<String>,
    /// The subset of [`Self::carried`] that can be readied and is not already.
    pub equippable: Vec<String>,
    /// The subset of [`Self::carried`] that can be spent or worked.
    pub usable: Vec<String>,
    /// The machines standing here.
    pub operable: Vec<String>,
    /// The union of every mode of every machine here.
    ///
    /// **A superset of what any one machine admits**, and deliberately so. The
    /// grammar's arm for `mode` is built once per situation rather than once per
    /// device, so it offers every state something here could be in; the act
    /// itself then checks the mode against the device actually named. A door
    /// told to *free fire* is refused by [`crate::sim::device::Device::set`]
    /// rather than by the tree.
    ///
    /// The alternative — a genuinely dependent sub-branch per device — is what
    /// the trie could express and what a later front end should build. This is
    /// the honest half-measure until it does, and it is a narrowing rather than
    /// a guarantee.
    pub device_modes: Vec<String>,
    /// What can be dug, stripped or harvested here.
    pub extractable: Vec<String>,
    /// What is hostile and still standing here.
    pub hostiles: Vec<String>,
    /// What the fabricators can make, given the stockpile.
    pub makeable: Vec<String>,
    /// Which production queues are free.
    pub queues: Vec<String>,
    /// What the tower can afford to do.
    pub tower_actions: Vec<String>,
    /// What the people here owe this character.
    pub owed: Vec<String>,
    /// What can be taken and held from here.
    pub claimable: Vec<String>,
    /// What this body is holding now — the orders it has taken and the station
    /// it has claimed. What [`Availability::Holding`] reads.
    pub held: Vec<String>,
    /// What there is here to read.
    pub readable: Vec<String>,
    /// The surfaces here that can be written on, whether or not anything is on
    /// them yet — see [`Choices::Postable`].
    pub postable: Vec<String>,
    /// This character's own filed work — see [`Choices::Presentable`].
    pub presentable: Vec<String>,
    /// The stances available, which is nothing at all when nothing is hostile.
    pub postures: Vec<String>,
    /// The conversations on this character's phone, as it names them.
    pub threads: Vec<String>,
    /// The subset of [`Self::threads`] it has with one other person — see
    /// [`Choices::Private`].
    pub private: Vec<String>,
    /// The subset of [`Self::threads`] that are channels and groups — see
    /// [`Choices::Channels`].
    pub channels: Vec<String>,
    /// The subset of [`Self::threads`] where somebody else spoke last — see
    /// [`Choices::Awaiting`].
    pub awaiting: Vec<String>,
    /// The titles of this character's agenda — [`Choices::Agenda`].
    pub agenda: Vec<String>,
    /// The priorities an agenda takes, for a character that keeps one —
    /// [`Choices::Priorities`].
    pub priorities: Vec<String>,
    /// Whether this is a character that takes orders, whose mission a
    /// conversation's party may change: its mission acts are the ordered
    /// ones ([`agenda::ordered`]), and [`Self::agenda`] holds the brief of
    /// the mission it carries.
    pub orders: bool,
    /// The subset of [`Self::threads`] it is able to leave — see
    /// [`Choices::Leavable`].
    pub leavable: Vec<String>,
    /// The subset of [`Self::threads`] somebody could still be brought into —
    /// see [`Choices::Invitable`].
    pub invitable: Vec<String>,
    /// Who could be brought into any of [`Self::invitable`] — see
    /// [`Choices::Invitees`].
    pub invitees: Vec<String>,
    /// Who it could start one with and has not.
    pub contacts: Vec<String>,
    /// The acts the parts standing here carry, straight off the map.
    pub station: Vec<String>,
    /// The parts standing here, by id — what decides which shared material
    /// this turn draws on (see [`crate::engine::bearings`]).
    pub parts: Vec<String>,
    /// The year of the world's history this body works in, set at a time
    /// machine for the mission it carries; `None` is the present. Nothing after
    /// it reaches the turn's recall (see [`crate::engine::bearings`]).
    pub year: Option<u32>,
    /// The years the body's mission still asks it to set at a time machine.
    /// What [`Choices::Years`] binds `time_travel` to.
    pub years: Vec<String>,
    /// **Every address an `invoke` may act on** — each reachable resource's
    /// verb-paths (`<resource-url>/<verb>`), with the act each one runs. Bound to
    /// `invoke`'s `url` so the grammar forces a whole verb-path a resource
    /// actually affords: a character cannot `invoke` a bare resource (a `405`) or
    /// a verb that is not there. The act is what types the body that goes with
    /// it — see [`Choices::InvokeUrl`] and [`invoke_body`].
    pub invokable: Vec<Invokable>,
    /// Whether this body is standing somewhere that is not where it musters
    /// from — what [`Availability::AwayFromHome`] reads.
    ///
    /// `false` for a body with no world and for one standing at home, both of
    /// which are the same answer to the only question it is asked: is there a
    /// journey home to make.
    pub away_from_home: bool,
    /// Whether this body is standing on a lift landing — where the lift acts can
    /// be reached. See [`Availability::AtLift`] / [`Availability::AtLanding`].
    pub at_lift: bool,
    /// Whether the car is open at this landing right now, so a body here can step
    /// in and ride. `false` while the car is away or its doors are shut.
    pub lift_here: bool,
    /// The floors the lift can take this body to, by the level's name — never the
    /// one it is standing on. What [`Choices::Floors`] binds `lift_use` to.
    pub floors: Vec<String>,
    /// This character's own name, as the world writes it.
    ///
    /// Needed because some live sets are about the *relationship* between this
    /// character and the others here — what they owe it — and those are kept
    /// under names on both sides.
    pub me: String,
}

impl Within {
    /// A character with no world — the probe, and anything asking what the
    /// catalog looks like in the abstract.
    pub fn nowhere() -> Self {
        Self::default()
    }

    /// In the company of people, somewhere with the ordinary two ways out, and
    /// nobody waiting on you.
    pub fn among(company: &[&str]) -> Self {
        Self {
            people: company.iter().map(|n| n.to_string()).collect(),
            // A person standing here hears a shout from here.
            heard_by_person: !company.is_empty(),
            ..Self::among_characters(company)
        }
    }

    /// [`Self::among`], with characters for company and no person here — who
    /// are reached by `message`, not by room speech.
    pub fn among_characters(company: &[&str]) -> Self {
        Self {
            company: company.iter().map(|n| n.to_string()).collect(),
            places: vec!["the green room".into(), "the chronicle".into()],
            ..Self::default()
        }
    }

    /// Everything a world's non-spatial state knows about this body, standing
    /// here.
    ///
    /// The one place [`crate::sim`] reaches the grammar. Each field is a
    /// question the world already has an answer to, and an empty answer is what
    /// keeps a tool out of a world that has no business offering it — `engage`
    /// is not disabled in the vault, it is unreachable there, because nothing in
    /// the vault can be engaged.
    pub fn from_sim(mut self, sim: &crate::sim::Sim, body: &str, place: &str) -> Self {
        self.carried = sim.carried(body);
        self.equippable = sim.equippable(body);
        self.usable = sim.usable(body);
        self.operable = sim.operable(place);
        self.device_modes = sim.modes_here(place);
        self.extractable = sim.extractable(place);
        self.hostiles = sim.hostiles(place);
        // **Field play**: a stance held, or something hostile within reach. A
        // shout is for the field as much as for a person within earshot.
        self.in_field = sim.field.stance(body).is_some() || !self.hostiles.is_empty();
        // **A stance needs a fight.** Nothing hostile here means no posture to
        // take, which takes `engage` out of the grammar by the ordinary
        // empty-set rule rather than by a rule of its own.
        self.postures = if self.hostiles.is_empty() {
            Vec::new()
        } else {
            crate::sim::field::POSTURES
                .iter()
                .map(|p| (*p).to_string())
                .collect()
        };
        // **The tower is spoken to from its own stations, not from anywhere in
        // the world.** A body on the open ground can no more start a fabricator
        // than it can read a terminal two floors up: the acts arrive with the
        // machine, which is what every other part-bound act already does.
        let makes = sim.makes_here(place);
        self.makeable = if makes { sim.makeable() } else { Vec::new() };
        self.queues = if makes { sim.free_queues() } else { Vec::new() };
        self.tower_actions = if sim.commands_here(place) {
            sim.tower_actions()
        } else {
            Vec::new()
        };
        // By the names the world writes down, on both sides — the set a
        // character reads and the set `remind` is bound to have to be the same
        // set, and one of them cannot be body ids.
        let me = self.me.clone();
        self.owed = self
            .company
            .iter()
            .flat_map(|who| sim.owed_to(&me, who))
            .collect();
        self.claimable = sim.claimable_at(place, body);
        // The orders only: a claimed station lives in the world, not the sim, so
        // the caller that has the world adds it ahead of this call.
        self.held.extend(sim.orders_held_by(body));
        // By body id, not by display name: a read cursor is bookkeeping nobody
        // addresses, so it keys on the thing a rename cannot move.
        self.readable = sim.readable_at(place, body);
        self.postable = sim.postable_at(place);
        self.presentable = sim.presentable(body);
        // What the parts here carry, less what the table does not offer this
        // body — the rule `invoke`'s addresses already keep. Without it a Maker
        // carrying an operation was still handed the bench's working-set verbs
        // as acts of their own, and stashed and logged round and round.
        self.year = sim.missions.year_of(body);
        self.jobs = sim
            .missions
            .routines()
            .iter()
            .map(|r| r.id.clone())
            .collect();
        self.job_params = sim
            .missions
            .routines()
            .iter()
            .filter(|r| !r.params.is_empty())
            .map(|r| (r.id.clone(), r.params.clone()))
            .collect();
        self.years = sim
            .missions
            .active(body)
            .map(|m| {
                m.todo
                    .iter()
                    .filter(|t| !t.done)
                    .filter_map(|t| mission::time_step(&t.text))
                    .map(|y| y.to_string())
                    .collect()
            })
            .unwrap_or_default();
        let carrying = mission_acts::carrying(sim, body);
        self.station = sim.station_tools(place);
        self.station.retain(|t| mission_acts::offered(t, carrying));
        // A world with no muster point is one nobody can be called back to, so
        // there is no journey home from anywhere in it.
        self.away_from_home = sim.homes().iter().any(|home| home != place);
        // **A mind that takes no orders has no body** (`docs/zen_everywhere.md`
        // §4): nothing of the place its seat is in is its to work — no
        // machine, board, claim or journey — but the order table's own acts.
        if carrying.unordered {
            self.makeable.clear();
            self.queues.clear();
            self.tower_actions.clear();
            self.claimable.clear();
            self.readable.clear();
            self.postable.clear();
            self.station.retain(|t| mission_acts::is_mine(t));
            self.away_from_home = false;
        }
        // The phone. Empty without a handset in the pack, which is what takes
        // every messaging act out of the grammar for somebody who has not got
        // one — or has had it taken off them.
        // Every phone set is read by the body's id, which is who it is on its
        // threads, and named the way the character calls each one.
        self.threads = sim.threads_for(body);
        // What waits unread is read by the name the character calls it — the
        // name its notice gave — and not only off whatever board is here.
        self.readable.extend(sim.unread_on_phone_for(body));
        self.private = sim.threads_of_kind_for(body, ThreadKind::Direct);
        self.channels = sim.threads_of_kind_for(body, ThreadKind::Group);
        self.awaiting = sim.awaiting_for(body);
        // An agenda is kept by a character that takes no orders.
        if sim.missions.is_unordered(body) {
            self.agenda = sim.missions.agenda_titles(body);
            self.priorities = Priority::ALL
                .iter()
                .map(|p| p.as_str().to_string())
                .collect();
        }
        self.leavable = sim.leavable_for(body);
        self.invitable = sim.invitable_for(body);
        self.invitees = sim.invitees_for(body);
        self.contacts = sim.contacts_for(body);
        self
    }

    pub fn alone(&self) -> bool {
        self.company.is_empty()
    }
}
