//! The slots one gap's claim may write (`docs/kv_quantize_in_the_gap.md` §3.2,
//! §3.8 rule 3).
//!
//! **A lease lives for one gap.** It is allocated for that gap's demand
//! (`demand`) after the gap's compaction, the lowest-placed arena first, and handed to the
//! device as a stack of slot addresses, one run per class. Each slot is held
//! with the address it had when it was allocated, so building the stack is a
//! copy. At the catch-up the slots installed blocks used are taken out of it by
//! the indices the summary names, and every other slot goes back to its pool:
//! nothing is leased between gaps.
//!
//! **The host is the only owner.** A lease owns its slots — taken out of the
//! pool before the gap that may write them is launched, so nothing else hands
//! them out. The device is told addresses and owns none of them.

use candle::Result;

use super::demand::PerClass;

/// The lease as the device reads it: every slot's address, class by class,
/// with each class's first entry and length.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StackImage {
    pub addrs: Vec<u64>,
    pub off: PerClass<u32>,
    pub len: PerClass<u32>,
}

/// The slots held for one gap's claim, per class, in stack order, each with its
/// device address. `T` is the slot's owning handle — a `ChunkGid` in
/// production, which returns the slot to its pool when dropped.
#[derive(Debug)]
pub(crate) struct Lease<T> {
    slots: PerClass<Vec<(T, u64)>>,
}

impl<T> Default for Lease<T> {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Vec::new()),
        }
    }
}

impl<T> Lease<T> {
    /// The slots each class holds.
    pub(crate) fn held(&self) -> PerClass<usize> {
        std::array::from_fn(|c| self.slots[c].len())
    }

    /// Whether the lease holds no slot.
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.iter().all(Vec::is_empty)
    }

    /// Add `slots`, each with its address, to class `class`'s run, after the
    /// ones it holds.
    pub(crate) fn add(&mut self, class: usize, slots: impl IntoIterator<Item = (T, u64)>) {
        self.slots[class].extend(slots);
    }

    /// The stack image the claim reads: the held addresses, copied.
    pub(crate) fn stack_image(&self) -> Result<StackImage> {
        let mut image = StackImage {
            addrs: Vec::with_capacity(self.slots.iter().map(Vec::len).sum()),
            ..StackImage::default()
        };
        for (c, run) in self.slots.iter().enumerate() {
            image.off[c] = u32::try_from(image.addrs.len())
                .map_err(|_| candle::Error::Msg("gap lease: stack past u32".into()))?;
            image.len[c] = u32::try_from(run.len())
                .map_err(|_| candle::Error::Msg("gap lease: class run past u32".into()))?;
            image.addrs.extend(run.iter().map(|&(_, addr)| addr));
        }
        Ok(image)
    }

    /// Take class `class`'s slots at `indices` — positions in the run as the
    /// stack image laid it out — out of the lease, in the order named. The
    /// slots left keep their positions' order. Refuses an index past the run
    /// or one named twice, taking nothing: either is a summary that does not
    /// match the lease it was claimed against.
    pub(crate) fn take(&mut self, class: usize, indices: &[u32]) -> Result<Vec<T>> {
        let run = &mut self.slots[class];
        let mut named = vec![false; run.len()];
        for &i in indices {
            let i = i as usize;
            if i >= run.len() || std::mem::replace(&mut named[i], true) {
                candle::bail!(
                    "gap lease: class {class} index {i} is past its {} slots or named twice",
                    run.len()
                );
            }
        }
        let mut taken: Vec<Option<T>> = std::iter::repeat_with(|| None).take(run.len()).collect();
        let mut kept = Vec::with_capacity(run.len() - indices.len());
        for (i, slot) in run.drain(..).enumerate() {
            if named[i] {
                taken[i] = Some(slot.0);
            } else {
                kept.push(slot);
            }
        }
        *run = kept;
        Ok(indices
            .iter()
            .map(|&i| taken[i as usize].take().expect("named exactly once"))
            .collect())
    }

    /// Empty the lease: every slot still held, class by class, for the caller
    /// to return to its pool. What the catch-up does once every installed
    /// block has taken its slots.
    pub(crate) fn release(&mut self) -> PerClass<Vec<T>> {
        std::array::from_fn(|c| {
            std::mem::take(&mut self.slots[c])
                .into_iter()
                .map(|(slot, _)| slot)
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::size_class::SizeClass;
    use super::*;

    const N: usize = SizeClass::COUNT;

    /// Slots as plain numbers, each at an address derived from it, so the
    /// lease is checked without a device.
    fn lease(runs: &[(usize, &[u64])]) -> Lease<u64> {
        let mut l = Lease::default();
        for &(c, slots) in runs {
            l.add(c, slots.iter().map(|&s| (s, s * 0x100)));
        }
        l
    }

    #[test]
    fn the_stack_lays_each_class_out_as_one_run_in_ladder_order() {
        let l = lease(&[(2, &[2, 3]), (0, &[1]), (2, &[4])]);
        let img = l.stack_image().unwrap();
        assert_eq!(img.addrs, vec![0x100, 0x200, 0x300, 0x400]);
        assert_eq!((img.off[0], img.len[0]), (0, 1));
        assert_eq!((img.off[1], img.len[1]), (1, 0));
        assert_eq!((img.off[2], img.len[2]), (1, 3));
        assert_eq!(img.off[3], 4);
        assert_eq!(l.held()[2], 3);
    }

    #[test]
    fn an_empty_lease_lays_out_an_empty_stack() {
        let l: Lease<u64> = Lease::default();
        assert!(l.is_empty());
        let img = l.stack_image().unwrap();
        assert!(img.addrs.is_empty());
        assert_eq!(img.off, [0; N]);
        assert_eq!(img.len, [0; N]);
    }

    #[test]
    fn a_take_returns_slots_in_the_order_named_and_keeps_the_rest_in_order() {
        let mut l = lease(&[(1, &[10, 11, 12, 13, 14])]);
        assert_eq!(l.take(1, &[3, 0]).unwrap(), vec![13, 10]);
        assert_eq!(l.stack_image().unwrap().addrs, vec![0xB00, 0xC00, 0xE00]);
    }

    #[test]
    fn a_take_naming_a_slot_twice_or_past_the_run_takes_nothing() {
        let mut l = lease(&[(0, &[1, 2, 3])]);
        assert!(l.take(0, &[1, 1]).is_err());
        assert!(l.take(0, &[3]).is_err());
        assert_eq!(l.held()[0], 3);
    }

    #[test]
    fn taking_nothing_leaves_the_lease_as_it_was() {
        let mut l = lease(&[(2, &[7, 8, 9])]);
        assert!(l.take(2, &[]).unwrap().is_empty());
        assert_eq!(l.held()[2], 3);
    }

    #[test]
    fn taking_every_slot_in_reverse_empties_the_run_in_that_order() {
        let mut l = lease(&[(0, &[1, 2, 3, 4])]);
        assert_eq!(l.take(0, &[3, 2, 1, 0]).unwrap(), vec![4, 3, 2, 1]);
        assert_eq!(l.held()[0], 0);
        assert!(l.take(0, &[0]).is_err(), "an empty run has nothing to take");
    }

    #[test]
    fn a_take_from_one_class_leaves_the_others_and_their_offsets() {
        let mut l = lease(&[(0, &[1, 2]), (1, &[3, 4, 5]), (2, &[6])]);
        l.take(1, &[1]).unwrap();
        let img = l.stack_image().unwrap();
        assert_eq!(img.addrs, vec![0x100, 0x200, 0x300, 0x500, 0x600]);
        assert_eq!((img.off[1], img.len[1]), (2, 2));
        assert_eq!((img.off[2], img.len[2]), (4, 1));
    }

    /// The catch-up's shape: takes for installed blocks, then everything else
    /// back. Every slot leaves exactly once, and the lease ends empty.
    #[test]
    fn after_the_takes_the_release_returns_every_other_slot_and_empties_it() {
        let mut l = lease(&[(0, &[1, 2, 3]), (3, &[4, 5])]);
        let used0 = l.take(0, &[2, 0]).unwrap();
        let used3 = l.take(3, &[1]).unwrap();
        let back = l.release();
        assert!(l.is_empty());
        assert_eq!(l.held(), [0; N]);
        assert_eq!((used0, used3), (vec![3, 1], vec![5]));
        assert_eq!(back[0], vec![2]);
        assert_eq!(back[3], vec![4]);
        let mut every: Vec<u64> = [vec![3, 1, 5], back.concat()].concat();
        every.sort_unstable();
        assert_eq!(every, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn releasing_an_untouched_lease_returns_it_whole_in_stack_order() {
        let mut l = lease(&[(1, &[9, 8, 7])]);
        assert_eq!(l.release()[1], vec![9, 8, 7]);
        assert!(l.is_empty());
        assert!(
            l.release().iter().all(Vec::is_empty),
            "a second release is empty"
        );
    }

    #[test]
    fn nothing_can_be_taken_from_a_released_lease() {
        let mut l = lease(&[(0, &[1, 2])]);
        drop(l.release());
        assert!(l.take(0, &[0]).is_err());
        assert!(l.take(0, &[]).unwrap().is_empty());
    }

    #[test]
    fn slots_added_after_a_take_follow_the_ones_left() {
        let mut l = lease(&[(2, &[1, 2, 3])]);
        l.take(2, &[0]).unwrap();
        l.add(2, [(9, 0x900)]);
        let img = l.stack_image().unwrap();
        assert_eq!(img.addrs, vec![0x200, 0x300, 0x900]);
        // Index 2 is now the added slot.
        assert_eq!(l.take(2, &[2]).unwrap(), vec![9]);
    }

    #[test]
    fn a_lease_refilled_after_its_release_starts_a_fresh_stack() {
        let mut l = lease(&[(0, &[1]), (4, &[2, 3])]);
        drop(l.release());
        l.add(4, [(7, 0x700)]);
        let img = l.stack_image().unwrap();
        assert_eq!(img.addrs, vec![0x700]);
        assert_eq!((img.off[4], img.len[4]), (0, 1));
        assert_eq!(img.len[0], 0);
    }

    /// Owning handles drop back to their pool when released: a handle that
    /// counts its drops shows each slot dropped once, after the release.
    #[test]
    fn released_handles_drop_once_each() {
        use std::cell::Cell;
        use std::rc::Rc;
        struct Slot(Rc<Cell<usize>>);
        impl Drop for Slot {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let dropped = Rc::new(Cell::new(0));
        let mut l = Lease::default();
        l.add(0, (0..4).map(|a| (Slot(dropped.clone()), a)));
        let kept = l.take(0, &[1]).unwrap();
        assert_eq!(dropped.get(), 0);
        drop(l.release());
        assert_eq!(dropped.get(), 3, "the three unused slots went back");
        drop(kept);
        assert_eq!(dropped.get(), 4);
    }
}
