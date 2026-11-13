//! The slot fill (`docs/hit_driven_projection.md` §7.2 step 4, §8.5).
//!
//! One fill places everything a dialogue's slot holds after its system prompt,
//! in three tiers, each taking its room from the tiers below it:
//!
//! 1. **Locks**, whole: the reads the user asked for, whatever they cost.
//! 2. **Floors**, in their own order: the current exchange, the dialogue's
//!    first request, then the recent exchanges newest first. When they pass
//!    the room they give way from the bottom — a recent exchange sheds its tool
//!    rounds oldest first and then leaves whole, oldest exchange first; then the
//!    current exchange's older tool rounds go. What is left past the room
//!    carries the slot past the trained window, as locks do.
//! 3. **Everything else by score**, highest first, ties oldest first: each
//!    round admitted while its kind is under its cap, bringing its piece's floor
//!    — a reading's request and answer, an older exchange's pair — with it, and
//!    only if all of them fit.
//!
//! The working set's shares, the dialogue's window and each other layer's
//! window are **caps**: they bound
//! how much of a kind the scored tier takes and never reserve room, and floors
//! count toward their kind's cap without ever being skipped for it.

use std::collections::{HashMap, HashSet};

/// What a placed unit is, for the caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A round of a file's reading — under `budget` and its piece's `max_file`.
    Reading,
    /// A round of a folder summary — under `budget` and `folder`.
    Folder,
    /// A round of a guide or a journal note — under `budget` and `guidance`.
    Guidance,
    /// A round of the dialogue or of its inherited lineage, or an exchange of
    /// another group on the dialogue's layer — under `window`.
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

/// How much of each kind the scored tier may take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caps {
    pub budget: usize,
    pub folder: usize,
    pub guidance: usize,
    pub max_file: usize,
    pub window: usize,
    /// Each other layer's `window`, indexed by [`Kind::Layer`]. A layer with
    /// no entry takes nothing.
    pub layers: Vec<usize>,
}

/// Everything one fill decides over.
#[derive(Debug, Clone, Default)]
pub struct FillInput {
    /// The room: the trained window less the reserve, the system prompt as
    /// emitted and what the turn has already appended.
    pub room: usize,
    pub locks: Vec<Unit>,
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
}

/// What the fill placed: the units by id, and how far past the room the
/// locks and the floors that could not give way carried it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fill {
    pub placed: Vec<usize>,
    pub over_room: usize,
}

/// Tokens placed per cap, as the fill goes.
#[derive(Default)]
struct Used {
    total: usize,
    by_kind: HashMap<Kind, usize>,
    by_piece: HashMap<usize, usize>,
}

impl Used {
    fn add(&mut self, u: &Unit) {
        self.total += u.tokens;
        *self.by_kind.entry(u.kind).or_default() += u.tokens;
        *self.by_piece.entry(u.piece).or_default() += u.tokens;
    }

    fn kind(&self, k: Kind) -> usize {
        self.by_kind.get(&k).copied().unwrap_or(0)
    }

    /// Whether `units` fit under every cap they count toward.
    fn under_caps(&self, units: &[Unit], caps: &Caps) -> bool {
        let mut add: HashMap<Kind, usize> = HashMap::new();
        let mut by_piece: HashMap<usize, usize> = HashMap::new();
        for u in units {
            *add.entry(u.kind).or_default() += u.tokens;
            if u.kind == Kind::Reading {
                *by_piece.entry(u.piece).or_default() += u.tokens;
            }
        }
        let after = |k: Kind| self.kind(k) + add.get(&k).copied().unwrap_or(0);
        let working = after(Kind::Reading) + after(Kind::Folder) + after(Kind::Guidance);
        let working_added = add.get(&Kind::Reading).copied().unwrap_or(0)
            + add.get(&Kind::Folder).copied().unwrap_or(0)
            + add.get(&Kind::Guidance).copied().unwrap_or(0);
        (working_added == 0 || working <= caps.budget)
            && (!add.contains_key(&Kind::Folder) || after(Kind::Folder) <= caps.folder)
            && (!add.contains_key(&Kind::Guidance) || after(Kind::Guidance) <= caps.guidance)
            && (!add.contains_key(&Kind::Dialogue) || after(Kind::Dialogue) <= caps.window)
            && add.keys().all(|k| match *k {
                Kind::Layer(i) => after(*k) <= caps.layers.get(i).copied().unwrap_or(0),
                _ => true,
            })
            && by_piece
                .iter()
                .all(|(p, t)| self.by_piece.get(p).copied().unwrap_or(0) + t <= caps.max_file)
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

    // 1. Locks, whole, whatever they cost.
    for u in &input.locks {
        place(u, &mut used, &mut placed);
    }

    // 2. Floors, giving way from the bottom when they pass the room.
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

    // 3. Everything else by score, highest first, ties oldest first.
    let mut order: Vec<&Unit> = input.scored.iter().collect();
    order.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    // A unit is placed once, whether it came in by its own score or with a
    // round of its piece; a piece's floor is brought by whichever of its
    // units enters first.
    let mut in_slot: HashSet<usize> = placed.iter().copied().collect();
    for u in order {
        if in_slot.contains(&u.id) {
            continue;
        }
        let mut with: Vec<Unit> = Vec::new();
        if let Some(f) = input.piece_floors.get(&u.piece) {
            with.extend(
                f.iter()
                    .filter(|f| f.id != u.id && !in_slot.contains(&f.id))
                    .copied(),
            );
        }
        with.push(*u);
        let cost: usize = with.iter().map(|w| w.tokens).sum();
        if used.total + cost > input.room || !used.under_caps(&with, caps) {
            continue;
        }
        for w in &with {
            in_slot.insert(w.id);
            place(w, &mut used, &mut placed);
        }
    }
    Fill { placed, over_room }
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
            budget: 1_000,
            folder: 200,
            guidance: 300,
            max_file: 500,
            window: 1_000,
            layers: vec![100],
        }
    }

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    /// Locks past the room are placed anyway and carry the slot past it;
    /// nothing scored gets in behind them.
    #[test]
    fn locks_past_the_slot_are_placed() {
        let input = FillInput {
            room: 100,
            locks: vec![u(1, 1, Kind::Reading, 150, 0.0)],
            scored: vec![u(2, 2, Kind::Dialogue, 10, 9.0)],
            ..Default::default()
        };
        let f = fill(&input, &caps());
        assert_eq!(f.placed, vec![1]);
        assert_eq!(f.over_room, 50);
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
