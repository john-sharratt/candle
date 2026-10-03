//! One *placed* part, and the stable id that names it.
//!
//! A [`Part`] is a catalogue entry — *chronicle terminal*, *seat*, *order
//! table* — and the same entry is placed into many rooms by reference. What an
//! effector route addresses is not the catalogue entry but *this* placement of
//! it: the third chronicle terminal in the early range, not chronicle terminals
//! in general. That is a [`PartInstance`].
//!
//! # The id is a pure function of the map
//!
//! The world is never persisted — bodies re-enter a freshly-seeded world every
//! time the daemon starts — so an instance id cannot be a row number handed out
//! at runtime. It has to be computable from the map alone, the same on every
//! run, so that a URL a character followed yesterday still resolves today and a
//! test can name an instance without standing anywhere.
//!
//! So the id is the catalogue part and a world-wide ordinal:
//!
//! ```text
//!   <part-id> ~ <ordinal>
//! ```
//!
//! `part-id` is the catalogue id, and `ordinal` is 0-based **in base-36**
//! ([`base36`]), counting the placements of *that* catalogue part across the
//! **whole world** in map order (areas by id, nodes in file order, placements in
//! file order). So the six character terminals in the casting floor's first band
//! are `character-terminal~0` .. `character-terminal~5`, a lone map table is
//! `map-table~0`, and the thirty-seventh seat in a hall is `seat~11`, not
//! `seat~37` — short, and free of the area and node names that made the old
//! `<area>~<node>~<part>~<ordinal>` id (`vault-command~command-room~order-table~0`)
//! so long to read and so many tokens to carry in a grammar enum.
//!
//! The ordinal is base-36 because these ids are the handle every stateless
//! world-API call carries — they are spent in the model's context over and over
//! as the world grows — so the encoding is chosen to be the fewest tokens that
//! stays unique and unambiguous: base-36 keeps a filling room's ordinals to one
//! character up to `z` (35) and two to `zz` (1295), where decimal would run to
//! three and four digits. It is not a higher radix over exotic characters (a
//! "base-1024"), because those fall back to per-byte tokens and cost *more*, not
//! fewer — the win is a short string over the tokeniser's cheap `0-9a-z`.
//!
//! The ordinal is world-wide, not per-node, which is what keeps it unique without
//! the place in the string: two rooms that each hold the same part get
//! consecutive ranges, never a collision. It stays a pure function of the map
//! because the walk is deterministic — [`MapSet`] precomputes each node's
//! starting offset once at load ([`MapSet::part_offset`]), so
//! [`instances_at`](MapSet::instances_at) adds a node-local count to it in O(node)
//! and the same URL resolves the same on every run. The tilde is URL-unreserved
//! (RFC 3986 §2.3) and appears in no kebab-case id, so the whole id rides in a
//! path segment without escaping, and [`MapSet::resolve_instance`] splits it back
//! into part and ordinal.

use std::collections::BTreeMap;

use crate::load::MapSet;
use crate::part::{Part, PartKind};
use crate::schema::Where;

/// The character joining an instance id's part and ordinal.
///
/// Tilde because it is URL-unreserved and appears in no kebab-case part id, so
/// the join is reversible: [`MapSet::resolve_instance`] splits on the **last**
/// tilde into a part id and an ordinal.
pub const SEP: char = '~';

/// Write a world-wide ordinal in base-36 (`0-9a-z`), lowercase.
///
/// The inverse of [`u32::from_str_radix(_, 36)`], which [`MapSet::
/// resolve_instance`] reads it back with. Lowercase because the ids sit in
/// URLs and the model's context beside kebab-case part names, and a single case
/// tokenises and reads more cleanly than a mixed one; `from_str_radix` accepts
/// either case on the way back, so a hand-typed uppercase id still resolves.
/// `0` maps to `"0"`, never the empty string.
pub fn base36(mut n: u32) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".to_string();
    }
    let mut buf = Vec::new();
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    buf.reverse();
    // Every byte is an ASCII digit from `DIGITS`, so this is always valid UTF-8.
    String::from_utf8(buf).expect("base-36 digits are ASCII")
}

/// One placement of a catalogue [`Part`] into a node, with the identity that
/// tells it from every other placement in the world.
///
/// Borrows its part from the map it was enumerated out of, so it is cheap to
/// make and never outlives the map. Everything a route handler needs to answer
/// for the thing — its globally-unique [`id`](PartInstance::id), the catalogue
/// [`part_id`](PartInstance::part_id) it is an instance of, the
/// [`place`](PartInstance::place) it stands in, and the part's own name and
/// kind — comes off it without a second lookup.
#[derive(Clone, Debug)]
pub struct PartInstance<'a> {
    place: Where,
    part: &'a Part,
    ordinal: u32,
}

impl<'a> PartInstance<'a> {
    /// Assemble an instance from the place it stands in, the catalogue part it
    /// is one of, and its ordinal within the node. Called by the map's
    /// enumeration; a caller reaches instances through [`MapSet::instances_at`].
    pub(crate) fn new(place: Where, part: &'a Part, ordinal: u32) -> PartInstance<'a> {
        PartInstance {
            place,
            part,
            ordinal,
        }
    }

    /// The globally-unique id: `<part-id>~<ordinal>`, the ordinal in base-36.
    ///
    /// Short — the part and its world-wide ordinal, no area or node name — and
    /// stable across restarts because the ordinal is a pure function of the map.
    /// The ordinal is written in base-36 ([`base36`]) rather than decimal: these
    /// ids are the handle every stateless world-API call carries, so they are
    /// spent in the model's context again and again, and base-36 keeps the suffix
    /// to one character up to `z` (35) and two to `zz` (1295) where decimal would
    /// run to three or four digits as a room fills. This is the handle a URL
    /// carries and [`MapSet::resolve_instance`] reads back.
    pub fn id(&self) -> String {
        format!("{}{SEP}{}", self.part.id, base36(self.ordinal))
    }

    /// The catalogue id this is an instance of — `chronicle-terminal`, `seat`.
    pub fn part_id(&self) -> &str {
        &self.part.id
    }

    /// Where it stands, as a map coordinate.
    pub fn place(&self) -> &Where {
        &self.place
    }

    /// Its 0-based **world-wide** ordinal among placements of the same catalogue
    /// part — `0` for the first anywhere, and `0` for a part placed only once in
    /// the whole world. This is the number the id carries.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// What the part is called, singular and bare — the catalogue [`Part::name`].
    pub fn name(&self) -> &str {
        &self.part.name
    }

    /// What sort of thing it is — station, fixture, seat.
    pub fn kind(&self) -> PartKind {
        self.part.kind
    }

    /// The catalogue part behind it, for a caller that needs its modes, binds,
    /// or long description.
    pub fn part(&self) -> &'a Part {
        self.part
    }
}

impl MapSet {
    /// **The placed instances standing in a node, each with its own id.**
    ///
    /// One entry per placement — a part placed with `count: 6` yields six
    /// instances, ordinals `0..6` — so two same-kind parts in one room are two
    /// addressable things rather than one. This is the per-instance counterpart
    /// of [`MapSet::part_ids_at`], which de-duplicates by catalogue id and is
    /// what a caller that only wants *which kinds of thing are here* reads
    /// instead.
    ///
    /// Ordinals count within `(node, part-id)`: a node placing a part in two
    /// separate placements numbers them `0, 1, …` across both, in file order,
    /// so the id of a given placement does not shift when an unrelated part is
    /// added beside it.
    pub fn instances_at(&self, at: &Where) -> Vec<PartInstance<'_>> {
        let Some(node) = self.node_at(at) else {
            return Vec::new();
        };
        let mut out: Vec<PartInstance<'_>> = Vec::new();
        // The node-local count per part, added to the node's precomputed global
        // offset ([`MapSet::part_offset`]) to give each instance its world-wide
        // ordinal — so the walk stays O(node) rather than counting the whole
        // world on every call.
        let mut local: BTreeMap<&str, u32> = BTreeMap::new();
        for placement in &node.parts {
            let Some(part) = self.part(placement.part()) else {
                continue;
            };
            let base = self.part_offset(at, &part.id);
            for _ in 0..placement.count() {
                let n = local.entry(part.id.as_str()).or_default();
                out.push(PartInstance::new(at.clone(), part, base + *n));
                *n += 1;
            }
        }
        out
    }

    /// **Turn an instance id back into the thing it names.**
    ///
    /// The inverse of [`PartInstance::id`], for a route handler holding an id off
    /// a URL. Splits `<part-id>~<ordinal>` on the last tilde, reads the ordinal
    /// as base-36 ([`base36`]), then finds the one node whose ordinal range for
    /// that part contains it — the ranges are contiguous and non-overlapping
    /// because [`part_offset`](MapSet::part_offset) hands each node the running
    /// count of everything before it. So the walk stops at the first node that
    /// could hold it and never guesses.
    ///
    /// `None` when the id is malformed (no ordinal, or not a base-36 one), names
    /// a part the catalogue does not have, or names an ordinal past the last
    /// placement of that part anywhere — so a resolved instance is always one
    /// that really stands in the world.
    pub fn resolve_instance(&self, id: &str) -> Option<PartInstance<'_>> {
        let (part_id, ordinal) = id.rsplit_once(SEP)?;
        // The ordinal is base-36 ([`base36`]); `from_str_radix` is its inverse
        // and accepts either case, so a hand-typed uppercase suffix still lands.
        let ordinal = u32::from_str_radix(ordinal, 36).ok()?;
        let part = self.part(part_id)?;
        for area in self.areas() {
            for node in &area.nodes {
                let count: u32 = node
                    .parts
                    .iter()
                    .filter(|p| p.part() == part_id)
                    .map(|p| p.count())
                    .sum();
                if count == 0 {
                    continue;
                }
                let at = Where::new(area.id.clone(), node.id.clone());
                let base = self.part_offset(&at, part_id);
                if ordinal >= base && ordinal < base + count {
                    return Some(PartInstance::new(at, part, ordinal));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::load::MapSet;
    use crate::schema::Where;

    /// The shipped vault, which the id tests read so an ordinal is asserted
    /// against a room the map really lays out.
    fn vault() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps")).expect("the vault loads")
    }

    #[test]
    fn six_terminals_in_a_room_are_six_distinct_instance_ids() {
        let set = vault();
        // Band one places `character-terminal` with count 6.
        let at = Where::new("vault-casting", "band-one");
        let instances: Vec<_> = set
            .instances_at(&at)
            .into_iter()
            .filter(|i| i.part_id() == "character-terminal")
            .collect();
        assert_eq!(instances.len(), 6, "band one holds six character terminals");

        let ids: Vec<String> = instances.iter().map(|i| i.id()).collect();
        // Each is `character-terminal~<n>` — the part and a world ordinal, no area
        // or node in the string.
        for id in &ids {
            assert!(
                id.starts_with("character-terminal~"),
                "not the short id form: {id}"
            );
        }
        // Every one of them is its own thing — no two ids collapse.
        let distinct: std::collections::BTreeSet<&String> = ids.iter().collect();
        assert_eq!(distinct.len(), 6, "two placements minted the same id");
        // Six in one node take a consecutive run of world ordinals.
        let ords: Vec<u32> = instances.iter().map(|i| i.ordinal()).collect();
        for pair in ords.windows(2) {
            assert_eq!(pair[1], pair[0] + 1, "ordinals in one node are consecutive");
        }
    }

    #[test]
    fn every_instance_round_trips_through_its_id_to_the_same_place_and_part() {
        let set = vault();
        let at = Where::new("vault-casting", "band-one");
        let instances = set.instances_at(&at);
        assert!(
            instances.iter().any(|i| i.part_id() == "light-ring"),
            "band one should hold a part besides its terminals"
        );
        for inst in instances {
            let id = inst.id();
            let back = set
                .resolve_instance(&id)
                .unwrap_or_else(|| panic!("`{id}` did not resolve"));
            assert_eq!(back.id(), id, "the id did not round-trip");
            assert_eq!(back.place(), &at);
            assert_eq!(back.part_id(), inst.part_id());
            assert_eq!(back.ordinal(), inst.ordinal());
        }
    }

    #[test]
    fn a_part_placed_once_still_gets_a_stable_zero_ordinal_id() {
        let set = vault();
        // The map room holds exactly one map table.
        let at = Where::new("vault-cartography", "map-room");
        let instances = set.instances_at(&at);
        let table: Vec<&str> = instances
            .iter()
            .filter(|i| i.part_id() == "map-table")
            .map(|i| i.part_id())
            .collect();
        assert_eq!(table.len(), 1, "one map table stands in the map room");

        let one = instances
            .iter()
            .find(|i| i.part_id() == "map-table")
            .expect("the map table");
        // One map table in the world, so it is `map-table~0` — the part and its
        // ordinal, nothing else.
        assert_eq!(one.id(), "map-table~0");
        assert_eq!(one.ordinal(), 0);
        // A singleton resolves the same way a numbered instance does.
        let back = set
            .resolve_instance(&one.id())
            .expect("the singleton resolves");
        assert_eq!(back.place(), &at);
        assert_eq!(back.part_id(), "map-table");
    }

    #[test]
    fn a_malformed_or_absent_id_resolves_to_nothing() {
        let set = vault();
        // No ordinal at all.
        assert!(set.resolve_instance("map-table").is_none());
        // An ordinal with a character outside base-36's `0-9a-z` — note `last`
        // *is* valid base-36, so a genuinely malformed ordinal needs a symbol.
        assert!(set.resolve_instance("map-table~!!").is_none());
        // An ordinal past the last placement of that part anywhere (base-36, so
        // `zz` is 1295 — well past the one map table).
        assert!(set.resolve_instance("map-table~zz").is_none());
        // A part the catalogue does not hold.
        assert!(set.resolve_instance("no-such-part~0").is_none());
        // The old long form is no longer an id — its head is not a catalogue part.
        assert!(set
            .resolve_instance("vault-casting~band-one~character-terminal~0")
            .is_none());
    }

    /// **The ordinal is base-36, and it round-trips.** `base36` writes `0-9a-z`
    /// lowest-significant-last, `from_str_radix(_, 36)` reads it back, and the
    /// boundaries — the single-digit ceiling `z`, the first two-digit `10`, and
    /// a mixed value — map exactly as the id scheme promises. Uppercase is read
    /// leniently on the way back so a hand-typed id still lands.
    #[test]
    fn the_ordinal_is_base36_and_round_trips() {
        use crate::instance::base36;
        for (n, s) in [
            (0u32, "0"),
            (9, "9"),
            (10, "a"),
            (35, "z"),
            (36, "10"),
            (1295, "zz"),
        ] {
            assert_eq!(base36(n), s, "{n} did not encode to `{s}`");
            assert_eq!(
                u32::from_str_radix(s, 36).unwrap(),
                n,
                "`{s}` did not decode to {n}"
            );
        }
        // Read back leniently: a hand-typed uppercase suffix resolves the same.
        assert_eq!(u32::from_str_radix("ZZ", 36).unwrap(), 1295);
    }

    #[test]
    fn instances_at_a_place_with_nothing_placed_is_empty() {
        let set = vault();
        // A corridor stands nothing in it.
        let at = Where::new("vault-casting", "ring-north");
        assert!(set.instances_at(&at).is_empty());
    }
}
