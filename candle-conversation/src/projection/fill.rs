//! The slot fill (`docs/hit_driven_projection.md` §7.2 step 4, §8.5).
//!
//! One fill places everything a dialogue's slot holds after its system prompt,
//! in two tiers, the first taking its room from the second:
//!
//! 1. **Floors**, in their own order: the current exchange, the dialogue's
//!    first request, then the recent exchanges newest first. When they pass
//!    the room they give way from the bottom — a recent exchange sheds its tool
//!    rounds oldest first and then leaves whole, oldest exchange first; then the
//!    current exchange's older tool rounds go. What is left past the room
//!    carries the slot past the trained window.
//! 2. **Everything else by score**, highest first, ties oldest first: each
//!    round admitted while its kind is under its cap, bringing its piece's floor
//!    — a reading's request and answer, an older exchange's pair — with it, and
//!    only if all of them fit.
//!
//! Each kind's **cap** is its share of the room (`working_set.shares`), and the
//! shares sum to the room; each other layer's window caps that layer's
//! exchanges. A cap bounds how much of a kind the scored tier takes and never
//! reserves room, and floors count toward their kind's cap without ever being
//! skipped for it. There is no per-piece cap: one reading the indexer reads
//! above everything else may take the whole of its kind's.
//!
//! **A third of a working-set kind's cap may enter per boundary** (§7.2 step
//! 5). With an [`Allowance`], the working-set rounds the slot already holds
//! keep their seats first, and a round that is not held enters only while its
//! kind's allowance has room — rounded up to whole turns, a new piece's floor
//! counted with it. A held round leaves only as an entry needs its room,
//! lowest score first; a piece whose last tool round leaves goes whole, floor
//! and all. With none — the first boundary after a new user message — the
//! working set is chosen whole, by score.

use std::collections::{HashMap, HashSet};

/// What a placed unit is, for the caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A round of a file's reading — under `readings`.
    Reading,
    /// A round of a folder summary — under `folders`.
    Folder,
    /// A round of a guide — under `guides`.
    Guide,
    /// A round of a journal note — under `journals`.
    Journal,
    /// A round of the dialogue or of its inherited lineage, or an exchange of
    /// another group on the dialogue's layer — under `dialogue`.
    Dialogue,
    /// An exchange of a group on another layer of the projection — under that
    /// layer's `window`, `Caps::layers[i]`.
    Layer(usize),
}

/// One unit the fill places or leaves: a round, or a floor pair placed as one.
/// `id` is the caller's name for it; `piece` the piece it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unit {
    pub id: usize,
    pub piece: usize,
    pub kind: Kind,
    pub tokens: usize,
    pub score: f32,
}

/// A floor exchange: units that hold their seats together, and the tool
/// rounds it sheds first when the floors pass the room, oldest first.
#[derive(Debug, Clone, Default)]
pub struct FloorExchange {
    pub kept: Vec<Unit>,
    pub rounds: Vec<Unit>,
}

/// How much of each kind the scored tier may take, in tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caps {
    pub readings: usize,
    pub folders: usize,
    pub guides: usize,
    pub journals: usize,
    pub dialogue: usize,
    /// Each other layer's `window`, indexed by [`Kind::Layer`]. A layer with
    /// no entry takes nothing.
    pub layers: Vec<usize>,
}

impl Kind {
    /// Whether this is a working-set kind — what the allowance bounds.
    fn is_working(self) -> bool {
        matches!(
            self,
            Kind::Reading | Kind::Folder | Kind::Guide | Kind::Journal
        )
    }
}

/// How many tokens of each working-set kind may enter the slot at this
/// boundary: a third of its cap.
pub type Allowance = HashMap<Kind, usize>;

impl Caps {
    /// A third of each working-set kind's cap, rounded up — what may enter
    /// at one boundary (§7.2 step 5).
    pub fn allowance(&self) -> Allowance {
        [Kind::Reading, Kind::Folder, Kind::Guide, Kind::Journal]
            .into_iter()
            .map(|k| (k, self.of(k).div_ceil(3)))
            .collect()
    }

    /// The cap `kind` counts toward.
    fn of(&self, kind: Kind) -> usize {
        match kind {
            Kind::Reading => self.readings,
            Kind::Folder => self.folders,
            Kind::Guide => self.guides,
            Kind::Journal => self.journals,
            Kind::Dialogue => self.dialogue,
            Kind::Layer(i) => self.layers.get(i).copied().unwrap_or(0),
        }
    }
}

/// Everything one fill decides over.
#[derive(Debug, Clone, Default)]
pub struct FillInput {
    /// The room: the trained window less the reserve, the system prompt as
    /// emitted and what the turn has already appended.
    pub room: usize,
    /// The current exchange: `kept` is its request, its newest tool round and
    /// the reply; `rounds` its older tool rounds.
    pub current: FloorExchange,
    pub first_request: Option<Unit>,
    /// The recent exchanges, newest first.
    pub recent: Vec<FloorExchange>,
    /// Every other unit, by score.
    pub scored: Vec<Unit>,
    /// Each piece's floor, placed with its first scored unit.
    pub piece_floors: HashMap<usize, Vec<Unit>>,
    /// The units whose rounds the slot already holds.
    pub held: HashSet<usize>,
    /// What may enter per working-set kind at this boundary; `None` on the
    /// first boundary after a new user message, which chooses whole.
    pub allowance: Option<Allowance>,
}

/// What the fill placed: the units by id, and how far past the room the
/// floors that could not give way carried it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fill {
    pub placed: Vec<usize>,
    pub over_room: usize,
}

/// Tokens placed per cap, as the fill goes.
#[derive(Default, Clone)]
struct Used {
    total: usize,
    by_kind: HashMap<Kind, usize>,
}

impl Used {
    fn add(&mut self, u: &Unit) {
        self.total += u.tokens;
        *self.by_kind.entry(u.kind).or_default() += u.tokens;
    }

    fn kind(&self, k: Kind) -> usize {
        self.by_kind.get(&k).copied().unwrap_or(0)
    }

    /// Whether `units` fit under the cap of every kind they add to.
    fn under_caps(&self, units: &[Unit], caps: &Caps) -> bool {
        let mut add: HashMap<Kind, usize> = HashMap::new();
        for u in units {
            *add.entry(u.kind).or_default() += u.tokens;
        }
        add.iter().all(|(&k, &t)| self.kind(k) + t <= caps.of(k))
    }
}

/// Fill the slot (module docs).
pub fn fill(input: &FillInput, caps: &Caps) -> Fill {
    let mut used = Used::default();
    let mut placed: Vec<usize> = Vec::new();
    fn place(u: &Unit, used: &mut Used, placed: &mut Vec<usize>) {
        used.add(u);
        placed.push(u.id);
    }

    // 1. Floors, giving way from the bottom when they pass the room.
    let mut recent: Vec<FloorExchange> = input.recent.clone();
    let mut current_rounds: Vec<Unit> = input.current.rounds.clone();
    let floor_tokens = |recent: &[FloorExchange], current_rounds: &[Unit]| -> usize {
        let ex = |e: &FloorExchange| {
            e.kept
                .iter()
                .chain(&e.rounds)
                .map(|u| u.tokens)
                .sum::<usize>()
        };
        input.current.kept.iter().map(|u| u.tokens).sum::<usize>()
            + current_rounds.iter().map(|u| u.tokens).sum::<usize>()
            + input.first_request.map_or(0, |u| u.tokens)
            + recent.iter().map(ex).sum::<usize>()
    };
    let fits = |n: usize| used.total + n <= input.room;
    // The recent exchanges give way oldest first: each sheds its tool rounds
    // oldest first, then leaves whole.
    while !fits(floor_tokens(&recent, &current_rounds)) {
        let Some(oldest) = recent.last_mut() else {
            break;
        };
        if oldest.rounds.is_empty() {
            recent.pop();
        } else {
            oldest.rounds.remove(0);
        }
    }
    // Then the current exchange's older tool rounds, oldest first.
    while !fits(floor_tokens(&recent, &current_rounds)) && !current_rounds.is_empty() {
        current_rounds.remove(0);
    }
    for u in input.current.kept.iter().chain(&current_rounds) {
        place(u, &mut used, &mut placed);
    }
    if let Some(u) = &input.first_request {
        place(u, &mut used, &mut placed);
    }
    for e in &recent {
        for u in e.kept.iter().chain(&e.rounds) {
            place(u, &mut used, &mut placed);
        }
    }
    let over_room = used.total.saturating_sub(input.room);

    // 2. Everything else by score, highest first, ties oldest first.
    let mut order: Vec<&Unit> = input.scored.iter().collect();
    order.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    // A unit is placed once, whether it came in by its own score or with a
    // round of its piece; a piece's floor is brought by whichever of its
    // units enters first.
    let in_slot: HashSet<usize> = placed.iter().copied().collect();
    let mut slot = Slot {
        used,
        placed,
        in_slot,
    };
    match &input.allowance {
        None => {
            for u in order {
                slot.try_place(u, input, caps);
            }
        }
        Some(allowance) => {
            let mut left = allowance.clone();
            // 2a. The working-set rounds the slot holds keep their seats.
            for u in order
                .iter()
                .filter(|u| u.kind.is_working() && input.held.contains(&u.id))
            {
                slot.try_place(u, input, caps);
            }
            // 2b. The rest, an entry only while its kind's allowance has room,
            // displacing held rounds of its kind it outscores when it needs
            // their room.
            for u in &order {
                if slot.in_slot.contains(&u.id) {
                    continue;
                }
                let with = slot.with_floor(u, input);
                let entering: usize = with
                    .iter()
                    .filter(|w| w.kind.is_working() && !input.held.contains(&w.id))
                    .map(|w| w.tokens)
                    .sum();
                if entering > 0 && left.get(&u.kind).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let placed = slot.fits(&with, input, caps)
                    || (u.kind.is_working() && slot.displace_for(u, &with, input, caps));
                if placed {
                    slot.place_all(&with);
                    if let Some(l) = left.get_mut(&u.kind) {
                        *l = l.saturating_sub(entering);
                    }
                }
            }
        }
    }
    Fill {
        placed: slot.placed,
        over_room,
    }
}

/// The scored tier's slot as it fills.
struct Slot {
    used: Used,
    placed: Vec<usize>,
    in_slot: HashSet<usize>,
}

impl Slot {
    /// `u` with the units of its piece's floor not yet in the slot.
    fn with_floor(&self, u: &Unit, input: &FillInput) -> Vec<Unit> {
        let mut with: Vec<Unit> = Vec::new();
        if let Some(f) = input.piece_floors.get(&u.piece) {
            with.extend(
                f.iter()
                    .filter(|f| f.id != u.id && !self.in_slot.contains(&f.id))
                    .copied(),
            );
        }
        with.push(*u);
        with
    }

    /// Whether `with` fits the room and every cap it adds to.
    fn fits(&self, with: &[Unit], input: &FillInput, caps: &Caps) -> bool {
        let cost: usize = with.iter().map(|w| w.tokens).sum();
        self.used.total + cost <= input.room && self.used.under_caps(with, caps)
    }

    fn place_all(&mut self, with: &[Unit]) {
        for w in with {
            self.in_slot.insert(w.id);
            self.used.add(w);
            self.placed.push(w.id);
        }
    }

    /// Place `u` with its floor when they fit; `false` when they do not.
    fn try_place(&mut self, u: &Unit, input: &FillInput, caps: &Caps) -> bool {
        if self.in_slot.contains(&u.id) {
            return true;
        }
        let with = self.with_floor(u, input);
        if !self.fits(&with, input, caps) {
            return false;
        }
        self.place_all(&with);
        true
    }

    fn remove(&mut self, u: &Unit) {
        if self.in_slot.remove(&u.id) {
            self.used.total -= u.tokens;
            *self.used.by_kind.entry(u.kind).or_default() -= u.tokens;
            self.placed.retain(|id| *id != u.id);
        }
    }

    /// Make room for `with`, which `u` brings, by taking out held rounds of
    /// `u`'s kind scored below it, lowest first — a piece whose last tool
    /// round goes leaving whole, its floor with it. Commits and answers `true`
    /// when `with` then fits; changes nothing and answers `false` otherwise.
    fn displace_for(&mut self, u: &Unit, with: &[Unit], input: &FillInput, caps: &Caps) -> bool {
        let mut victims: Vec<&Unit> = input
            .scored
            .iter()
            .filter(|v| {
                v.kind == u.kind
                    && v.piece != u.piece
                    && v.score < u.score
                    && input.held.contains(&v.id)
                    && self.in_slot.contains(&v.id)
            })
            .collect();
        victims.sort_by(|a, b| a.score.total_cmp(&b.score).then(b.id.cmp(&a.id)));
        let mut trial = Slot {
            used: self.used.clone(),
            placed: self.placed.clone(),
            in_slot: self.in_slot.clone(),
        };
        for v in victims {
            if trial.fits(with, input, caps) {
                break;
            }
            trial.remove(v);
            let floor: Vec<Unit> = input
                .piece_floors
                .get(&v.piece)
                .cloned()
                .unwrap_or_default();
            let floor_ids: HashSet<usize> = floor.iter().map(|f| f.id).collect();
            let rounds_left = input.scored.iter().any(|r| {
                r.piece == v.piece && !floor_ids.contains(&r.id) && trial.in_slot.contains(&r.id)
            });
            if !rounds_left {
                for f in &floor {
                    trial.remove(f);
                }
            }
        }
        if !trial.fits(with, input, caps) {
            return false;
        }
        *self = trial;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(id: usize, piece: usize, kind: Kind, tokens: usize, score: f32) -> Unit {
        Unit {
            id,
            piece,
            kind,
            tokens,
            score,
        }
    }

    fn caps() -> Caps {
        Caps {
            readings: 1_000,
            folders: 200,
            guides: 300,
            journals: 300,
            dialogue: 1_000,
            layers: vec![100],
        }
    }

    /// A third of each working-set kind's cap, rounded up.
    #[test]
    fn the_allowance_is_a_third_of_each_working_cap() {
        let a = caps().allowance();
        assert_eq!(a[&Kind::Reading], 334);
        assert_eq!(a[&Kind::Folder], 67);
        assert_eq!(a[&Kind::Guide], 100);
        assert_eq!(a[&Kind::Journal], 100);
        assert!(!a.contains_key(&Kind::Dialogue));
    }

    fn readings(scored: Vec<Unit>, held: &[usize], allowance: Option<Allowance>) -> FillInput {
        FillInput {
            room: 10_000,
            scored,
            held: held.iter().copied().collect(),
            allowance,
            ..Default::default()
        }
    }

    /// **Held rounds keep their seats, and entries stop at the allowance**:
    /// with 100 tokens a boundary, the held round scored 0.1 stays though four
    /// stronger rounds wait, and only the first of them enters.
    #[test]
    fn held_rounds_stay_and_entries_stop_at_the_allowance() {
        let r = |id, score| u(id, id, Kind::Reading, 100, score);
        let scored = vec![r(1, 0.1), r(2, 0.9), r(3, 0.8), r(4, 0.7)];
        let allowance = Allowance::from([(Kind::Reading, 100)]);
        let f = fill(&readings(scored.clone(), &[1], Some(allowance)), &caps());
        assert_eq!(sorted(f.placed), vec![1, 2]);
        // On the first boundary after a question, everything by score.
        let f = fill(&readings(scored, &[1], None), &caps());
        assert_eq!(sorted(f.placed), vec![1, 2, 3, 4]);
    }

    /// **An entry is rounded up to whole turns**: 50 tokens of allowance
    /// still lets one 100-token round in, and nothing after it.
    #[test]
    fn an_entry_rounds_up_to_a_whole_turn() {
        let r = |id, score| u(id, id, Kind::Reading, 100, score);
        let allowance = Allowance::from([(Kind::Reading, 50)]);
        let f = fill(
            &readings(vec![r(1, 0.9), r(2, 0.8)], &[], Some(allowance)),
            &caps(),
        );
        assert_eq!(f.placed, vec![1]);
    }

    /// **A held round leaves only as an entry needs its room, lowest first,
    /// and a piece whose last tool round leaves goes whole**: the readings
    /// cap of 250 is full of piece 7's floor (50) and its two held rounds
    /// (100 each). An entry scored 0.9 needs 100: the held round scored 0.2
    /// goes; piece 7 keeps its other round and its floor. A second entry at
    /// 0.8 takes the last round, and the floor goes with it.
    #[test]
    fn an_entry_displaces_the_weakest_held_round() {
        let capped = Caps {
            readings: 250,
            ..caps()
        };
        let floor = u(70, 7, Kind::Reading, 50, 0.0);
        let mut floors = HashMap::new();
        floors.insert(7, vec![floor]);
        let input = |scored: Vec<Unit>| FillInput {
            room: 10_000,
            scored,
            piece_floors: floors.clone(),
            held: HashSet::from([70, 71, 72]),
            allowance: Some(Allowance::from([(Kind::Reading, 1_000)])),
            ..Default::default()
        };
        let held = [
            u(71, 7, Kind::Reading, 100, 0.2),
            u(72, 7, Kind::Reading, 100, 0.5),
        ];
        let one = input(vec![held[0], held[1], u(1, 1, Kind::Reading, 100, 0.9)]);
        assert_eq!(sorted(fill(&one, &capped).placed), vec![1, 70, 72]);
        let two = input(vec![
            held[0],
            held[1],
            u(1, 1, Kind::Reading, 100, 0.9),
            u(2, 2, Kind::Reading, 100, 0.8),
        ]);
        assert_eq!(sorted(fill(&two, &capped).placed), vec![1, 2]);
    }

    /// Guides and journal notes each fill their own cap: a full guides cap
    /// leaves the journals' untouched.
    #[test]
    fn guides_and_journals_are_capped_apart() {
        let input = FillInput {
            room: 10_000,
            scored: vec![
                u(1, 1, Kind::Guide, 300, 0.9),
                u(2, 2, Kind::Guide, 10, 0.8),
                u(3, 3, Kind::Journal, 300, 0.7),
            ],
            ..Default::default()
        };
        assert_eq!(sorted(fill(&input, &caps()).placed), vec![1, 3]);
    }

    /// No cap per piece: one reading's rounds may take the whole readings cap.
    #[test]
    fn one_reading_may_take_the_whole_readings_cap() {
        let input = FillInput {
            room: 10_000,
            scored: (0..10).map(|i| u(i, 7, Kind::Reading, 100, 0.9)).collect(),
            ..Default::default()
        };
        assert_eq!(fill(&input, &caps()).placed.len(), 10);
    }

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    /// A kind at its cap is skipped and its room goes to the next by score.
    #[test]
    fn a_capped_kind_gives_its_room_to_the_next() {
        let input = FillInput {
            room: 10_000,
            scored: vec![
                u(1, 1, Kind::Folder, 150, 0.9),
                u(2, 2, Kind::Folder, 100, 0.8),
                u(3, 3, Kind::Dialogue, 100, 0.7),
            ],
            ..Default::default()
        };
        assert_eq!(sorted(fill(&input, &caps()).placed), vec![1, 3]);
    }

    /// Floors count toward their kind's cap but are never skipped for it.
    #[test]
    fn floors_count_toward_caps_and_are_never_skipped() {
        let input = FillInput {
            room: 10_000,
            current: FloorExchange {
                kept: vec![u(1, 9, Kind::Dialogue, 1_200, 0.0)],
                rounds: vec![],
            },
            scored: vec![u(2, 3, Kind::Dialogue, 10, 0.9)],
            ..Default::default()
        };
        let f = fill(&input, &caps());
        assert_eq!(f.placed, vec![1], "the window is full of floor");
    }

    /// Floors past the room give way in order: the oldest recent exchange's
    /// rounds, then that exchange, then the next, then the current exchange's
    /// older rounds — and what is left carries the slot past the room.
    #[test]
    fn floors_give_way_from_the_bottom() {
        let d = |id, t| u(id, 0, Kind::Dialogue, t, 0.0);
        let input = |room| FillInput {
            room,
            current: FloorExchange {
                kept: vec![d(1, 100)],
                rounds: vec![d(2, 50), d(3, 50)],
            },
            first_request: Some(d(4, 20)),
            recent: vec![
                FloorExchange {
                    kept: vec![d(5, 30)],
                    rounds: vec![d(6, 40)],
                },
                FloorExchange {
                    kept: vec![d(7, 30)],
                    rounds: vec![d(8, 40), d(9, 40)],
                },
            ],
            ..Default::default()
        };
        let all = 100 + 100 + 20 + 70 + 110;
        assert_eq!(
            sorted(fill(&input(all), &caps()).placed),
            (1..=9).collect::<Vec<_>>()
        );
        // 40 short: the oldest exchange's oldest round goes.
        assert_eq!(
            sorted(fill(&input(all - 40), &caps()).placed),
            vec![1, 2, 3, 4, 5, 6, 7, 9]
        );
        // 110 short: the oldest exchange goes whole.
        assert_eq!(
            sorted(fill(&input(all - 110), &caps()).placed),
            vec![1, 2, 3, 4, 5, 6]
        );
        // 180 short: both recent exchanges go.
        assert_eq!(
            sorted(fill(&input(all - 180), &caps()).placed),
            vec![1, 2, 3, 4]
        );
        // 210 short: the current exchange's older rounds go too, oldest first.
        assert_eq!(
            sorted(fill(&input(all - 210), &caps()).placed),
            vec![1, 3, 4]
        );
        // Nothing left to give: the rest carries the slot past the room.
        let f = fill(&input(50), &caps());
        assert_eq!(sorted(f.placed.clone()), vec![1, 4]);
        assert_eq!(f.over_room, 70);
    }

    /// A round of a piece not yet placed brings its piece's floor with it, and
    /// only if all of them fit; once the floor is in, the piece's next round
    /// comes alone.
    #[test]
    fn a_round_brings_its_pieces_floor() {
        let mut floors = HashMap::new();
        floors.insert(7, vec![u(70, 7, Kind::Reading, 50, 0.0)]);
        let input = FillInput {
            room: 160,
            scored: vec![
                u(71, 7, Kind::Reading, 60, 0.9),
                u(72, 7, Kind::Reading, 60, 0.8),
                u(80, 8, Kind::Dialogue, 40, 0.5),
            ],
            piece_floors: floors,
            ..Default::default()
        };
        assert_eq!(fill(&input, &caps()).placed, vec![70, 71, 80]);
        // Room for the round but not its floor: neither goes in.
        let tight = FillInput {
            room: 100,
            ..input.clone()
        };
        assert_eq!(fill(&tight, &caps()).placed, vec![80]);
    }

    /// A thin working set leaves its room to the dialogue: the caps sum past
    /// the room, and the dialogue takes what the readings do not.
    #[test]
    fn a_thin_working_set_leaves_its_room_to_the_dialogue() {
        let input = FillInput {
            room: 1_200,
            scored: vec![
                u(1, 1, Kind::Reading, 100, 0.9),
                u(2, 2, Kind::Dialogue, 600, 0.8),
                u(3, 3, Kind::Dialogue, 350, 0.7),
            ],
            ..Default::default()
        };
        assert_eq!(fill(&input, &caps()).placed, vec![1, 2, 3]);
    }

    /// An older exchange's pair is a unit of its own and the floor of its
    /// tool rounds: placed by its own score, it is not placed again when a
    /// round of it follows; a round that ranks first brings it.
    #[test]
    fn a_pair_is_placed_once_whichever_comes_first() {
        let pair = u(10, 1, Kind::Dialogue, 30, 0.9);
        let mut floors = HashMap::new();
        floors.insert(1, vec![pair]);
        let input = FillInput {
            room: 1_000,
            scored: vec![pair, u(11, 1, Kind::Dialogue, 20, 0.5)],
            piece_floors: floors.clone(),
            ..Default::default()
        };
        assert_eq!(fill(&input, &caps()).placed, vec![10, 11]);
        let round_first = FillInput {
            room: 1_000,
            scored: vec![
                u(10, 1, Kind::Dialogue, 30, 0.1),
                u(11, 1, Kind::Dialogue, 20, 0.5),
            ],
            piece_floors: floors,
            ..Default::default()
        };
        assert_eq!(fill(&round_first, &caps()).placed, vec![10, 11]);
    }

    /// Another layer's exchanges compete by score under that layer's window,
    /// and a layer with no window entry takes nothing.
    #[test]
    fn another_layer_is_capped_by_its_own_window() {
        let input = FillInput {
            room: 10_000,
            scored: vec![
                u(1, 1, Kind::Layer(0), 60, 0.9),
                u(2, 2, Kind::Layer(0), 60, 0.8),
                u(3, 3, Kind::Layer(0), 40, 0.7),
                u(4, 4, Kind::Layer(1), 10, 0.6),
                u(5, 5, Kind::Dialogue, 100, 0.5),
            ],
            ..Default::default()
        };
        assert_eq!(fill(&input, &caps()).placed, vec![1, 3, 5]);
    }

    /// Ties go to the older unit — the lower id.
    #[test]
    fn ties_break_oldest_first() {
        let input = FillInput {
            room: 10,
            scored: vec![
                u(5, 5, Kind::Dialogue, 10, 0.5),
                u(4, 4, Kind::Dialogue, 10, 0.5),
            ],
            ..Default::default()
        };
        assert_eq!(fill(&input, &caps()).placed, vec![4]);
    }
}
