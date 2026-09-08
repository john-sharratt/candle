//! What a body carries, and what carrying it lets it do.
//!
//! # An inventory is a live set before it is a possession
//!
//! The reason this exists is not bookkeeping. `give`, `equip`, `use` and
//! `gather` all bind an argument to *what you have*, and a bound argument is
//! only as good as the set behind it — so the inventory is what makes those
//! four tools unable to name a thing that is not there. A character with an
//! empty pack is not offered `use` at all, which is the same discipline `tell`
//! keeps when nobody is in the room.
//!
//! # Counts, not instances
//!
//! Twelve rounds of ammunition are one entry with a count, not twelve entries.
//! The world never needs to tell one round from another, and an instance model
//! would make `give { what, count }` a set operation over identical things for
//! no gain anybody can observe.
//!
//! Equipment is the exception that proves it: a weapon is carried at count one
//! and is either readied or not, so it needs no instance identity either.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// What sort of thing an item is, which decides what can be done with it.
///
/// **The kind gates the verb, so the grammar can.** `equip` offers only what is
/// wearable or wieldable; `use` offers only what is consumable or operable.
/// Without the split a character is invited to equip a stimpak, and a model
/// handed the option takes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Wielded. One at a time in each hand's worth of loadout.
    Weapon,
    /// Worn. Protects, and may carry systems of its own.
    Armour,
    /// Feeds a weapon. Spent by the simulator, not by an act.
    Ammunition,
    /// Spent in one use, on yourself or somebody else.
    Consumable,
    /// Used repeatedly without being spent — a scanner, a cutting torch.
    Gear,
    /// Raw material. Gathered, carried, handed in, never equipped or used.
    Resource,
    /// Something written. Carried between rooms, read, handed on.
    Document,
}

impl Kind {
    /// Whether `equip` should offer it.
    pub fn equippable(&self) -> bool {
        matches!(self, Kind::Weapon | Kind::Armour | Kind::Gear)
    }

    /// Whether `use` should offer it.
    ///
    /// Gear is usable *and* equippable — a scanner is worn and operated — which
    /// is why these two are not complements of one another.
    pub fn usable(&self) -> bool {
        matches!(self, Kind::Consumable | Kind::Gear)
    }
}

/// One kind of thing, and how many of it a body has.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    /// The identifier the world uses. Stable, lower case, underscore-joined.
    pub id: String,
    /// What a character calls it, bare and singular: "plasma rifle".
    pub name: String,
    pub kind: Kind,
    pub count: u32,
    /// Readied rather than merely carried. Only ever true for [`Kind::equippable`].
    pub equipped: bool,
}

impl Item {
    pub fn new(id: impl Into<String>, name: impl Into<String>, kind: Kind, count: u32) -> Item {
        Item {
            id: id.into(),
            name: name.into(),
            kind,
            count,
            equipped: false,
        }
    }
}

/// Everything one body is carrying.
///
/// Ordered by id, so what a character is offered is the same on two runs. A
/// hash map here would make the grammar's argument order depend on the
/// allocator, and a test that asserts a live set would pass or fail by luck.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pack {
    items: BTreeMap<String, Item>,
}

impl Pack {
    pub fn new() -> Pack {
        Pack::default()
    }

    /// Put something in, merging with what is already there.
    ///
    /// Merging rather than replacing is what makes `gather` repeatable: walking
    /// back to the same seam twice leaves one entry of twice the size, which is
    /// what a character would say it has.
    pub fn add(&mut self, item: Item) {
        match self.items.get_mut(&item.id) {
            Some(had) => had.count = had.count.saturating_add(item.count),
            None => {
                self.items.insert(item.id.clone(), item);
            }
        }
    }

    /// Take some out. `None` when there are not that many — never a partial
    /// take, because a character that asked to hand over ten and handed over
    /// three has told the other party something false.
    pub fn take(&mut self, id: &str, count: u32) -> Option<Item> {
        let held = self.items.get_mut(id)?;
        if held.count < count {
            return None;
        }
        let mut out = held.clone();
        out.count = count;
        out.equipped = false;
        held.count -= count;
        if held.count == 0 {
            self.items.remove(id);
        }
        Some(out)
    }

    pub fn get(&self, id: &str) -> Option<&Item> {
        self.items.get(id)
    }

    pub fn has(&self, id: &str) -> bool {
        self.items.contains_key(id)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Item> {
        self.items.values()
    }

    /// Ready something already carried. `false` when it is not held, or is not
    /// the sort of thing that is readied.
    pub fn equip(&mut self, id: &str) -> bool {
        match self.items.get_mut(id) {
            Some(item) if item.kind.equippable() => {
                item.equipped = true;
                true
            }
            _ => false,
        }
    }

    /// Put something back to merely carried.
    pub fn unequip(&mut self, id: &str) -> bool {
        match self.items.get_mut(id) {
            Some(item) if item.equipped => {
                item.equipped = false;
                true
            }
            _ => false,
        }
    }

    /// Spend one of something. `false` when none is held.
    pub fn spend(&mut self, id: &str) -> bool {
        self.take(id, 1).is_some()
    }

    /// The names of what `equip` should offer — carried, equippable, not
    /// already readied.
    pub fn equippable(&self) -> Vec<String> {
        self.items
            .values()
            .filter(|i| i.kind.equippable() && !i.equipped)
            .map(|i| i.name.clone())
            .collect()
    }

    /// The names of what `use` should offer.
    pub fn usable(&self) -> Vec<String> {
        self.items
            .values()
            .filter(|i| i.kind.usable())
            .map(|i| i.name.clone())
            .collect()
    }

    /// Every name held, which is what `give` offers.
    pub fn names(&self) -> Vec<String> {
        self.items.values().map(|i| i.name.clone()).collect()
    }

    /// Find what is held by the name a character would use for it.
    ///
    /// Case-insensitive, because the grammar hands back the name the world
    /// wrote and a model that has seen it once may not reproduce its case.
    pub fn by_name(&self, name: &str) -> Option<&Item> {
        let want = name.trim().to_lowercase();
        self.items
            .values()
            .find(|i| i.name.to_lowercase() == want || i.id == want)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rounds(n: u32) -> Item {
        Item::new("bolt", "bolt rounds", Kind::Ammunition, n)
    }

    #[test]
    fn adding_the_same_thing_twice_makes_one_entry_of_both() {
        let mut p = Pack::new();
        p.add(rounds(12));
        p.add(rounds(8));
        assert_eq!(p.len(), 1);
        assert_eq!(p.get("bolt").unwrap().count, 20);
    }

    #[test]
    fn taking_more_than_is_held_takes_nothing_at_all() {
        // A partial take would let a character promise ten and deliver three.
        let mut p = Pack::new();
        p.add(rounds(4));
        assert!(p.take("bolt", 5).is_none());
        assert_eq!(p.get("bolt").unwrap().count, 4, "a refused take still took");
    }

    #[test]
    fn taking_the_last_of_something_removes_the_entry() {
        let mut p = Pack::new();
        p.add(rounds(3));
        assert_eq!(p.take("bolt", 3).unwrap().count, 3);
        assert!(!p.has("bolt"));
        assert!(p.is_empty());
    }

    #[test]
    fn what_is_taken_arrives_unequipped() {
        // Handing over a readied weapon hands over a weapon, not a readiness.
        let mut p = Pack::new();
        p.add(Item::new("mono_sword", "mono sword", Kind::Weapon, 1));
        p.equip("mono_sword");
        let given = p.take("mono_sword", 1).unwrap();
        assert!(!given.equipped);
    }

    #[test]
    fn only_the_right_kind_of_thing_is_equipped_or_used() {
        let mut p = Pack::new();
        p.add(Item::new("stimpak", "stimpak", Kind::Consumable, 1));
        p.add(Item::new("mono_sword", "mono sword", Kind::Weapon, 1));
        p.add(Item::new("scanner", "advanced scanner", Kind::Gear, 1));

        assert!(!p.equip("stimpak"), "a stimpak was readied");
        assert!(p.equip("mono_sword"));

        // Gear is both: worn, and operated. A consumable is only ever used, so
        // the two sets overlap rather than partition — and readying something
        // takes it out of `equippable` without taking it out of `usable`.
        assert_eq!(p.equippable(), vec!["advanced scanner"], "the sword is readied");
        assert!(p.equip("scanner"));
        assert_eq!(p.usable(), vec!["advanced scanner", "stimpak"]);
        assert!(p.equippable().is_empty(), "everything wearable is now worn");
    }

    #[test]
    fn an_equipped_thing_is_not_offered_for_equipping_again() {
        let mut p = Pack::new();
        p.add(Item::new("mono_sword", "mono sword", Kind::Weapon, 1));
        assert_eq!(p.equippable(), vec!["mono sword".to_string()]);
        p.equip("mono_sword");
        assert!(p.equippable().is_empty(), "readied twice");
    }

    #[test]
    fn a_thing_is_found_by_the_name_the_world_wrote_whatever_the_case() {
        let mut p = Pack::new();
        p.add(Item::new("mono_sword", "mono sword", Kind::Weapon, 1));
        assert!(p.by_name("Mono Sword").is_some());
        assert!(p.by_name("  mono sword ").is_some());
        assert!(p.by_name("mono_sword").is_some(), "the id should resolve too");
        assert!(p.by_name("plasma rifle").is_none());
    }

    #[test]
    fn what_is_offered_is_in_the_same_order_every_time() {
        // The live set feeds a grammar; two runs must compile the same tree.
        let mut p = Pack::new();
        for id in ["zeta", "alpha", "mu"] {
            p.add(Item::new(id, id, Kind::Gear, 1));
        }
        assert_eq!(p.names(), vec!["alpha", "mu", "zeta"]);
    }

    #[test]
    fn spending_removes_exactly_one() {
        let mut p = Pack::new();
        p.add(Item::new("stimpak", "stimpak", Kind::Consumable, 2));
        assert!(p.spend("stimpak"));
        assert_eq!(p.get("stimpak").unwrap().count, 1);
        assert!(p.spend("stimpak"));
        assert!(!p.spend("stimpak"), "spent what was not there");
    }
}
