//! The applied set on the wire (§7.5), as `CATCHUP.since` and `HAVE` carry it:
//!
//! ```text
//! u16 n_origins
//! repeat: u64 origin | varint top | varint n_gaps | n_gaps × (varint start_delta, varint len)
//! ```
//!
//! `top` is the origin's highest covered `oseq`; the gaps are the uncovered
//! ranges below it, ascending. `start_delta` is a gap's start less the end of
//! the gap before it (zero before the first), and `len` its length; both are at
//! least one. Origins ascend and none repeats, so one set has one spelling. A
//! member that is caught up sends about ten bytes per origin.

use applied_set::{AppliedSet, OriginApplied};

use crate::wire::{Reader, WireError, Writer};

pub fn write_applied(w: &mut Writer, set: &AppliedSet) {
    let origins: Vec<(u64, &OriginApplied)> = set.origins().filter(|(_, o)| o.top() > 0).collect();
    w.u16(origins.len() as u16);
    for (origin, applied) in origins {
        w.u64(origin).varint(applied.top());
        let gaps = applied.gaps();
        w.varint(gaps.len() as u64);
        let mut cursor = 0;
        for (start, end) in gaps {
            w.varint(start - cursor).varint(end - start + 1);
            cursor = end;
        }
    }
}

pub fn read_applied(r: &mut Reader) -> Result<AppliedSet, WireError> {
    let what = r.what();
    let invalid = |value: u64| WireError::Invalid { what, value };
    let mut set = AppliedSet::default();
    let mut last_origin = None;
    for _ in 0..r.u16()? {
        let origin = r.u64()?;
        if origin == 0 || last_origin.is_some_and(|last| origin <= last) {
            return Err(invalid(origin));
        }
        last_origin = Some(origin);
        let top = r.varint()?;
        if top == 0 {
            return Err(invalid(top));
        }
        let n = r.varint()?;
        if n > r.remaining() as u64 {
            return Err(WireError::Truncated {
                what,
                need: (n - r.remaining() as u64) as usize,
            });
        }
        let mut gaps = Vec::with_capacity(n as usize);
        let mut cursor = 0u64;
        for _ in 0..n {
            let delta = r.varint()?;
            let len = r.varint()?;
            if delta == 0 || len == 0 {
                return Err(invalid(0));
            }
            let start = cursor.checked_add(delta).ok_or_else(|| invalid(delta))?;
            let end = start.checked_add(len - 1).ok_or_else(|| invalid(len))?;
            // A gap must leave the covered range between it and the one
            // before it non-empty, and end below the top.
            if (cursor > 0 && start <= cursor + 1) || end >= top {
                return Err(invalid(start));
            }
            gaps.push((start, end));
            cursor = end;
        }
        set.set_origin(origin, OriginApplied::from_top_and_gaps(top, gaps));
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(set: &AppliedSet) -> Vec<u8> {
        let mut w = Writer::new();
        write_applied(&mut w, set);
        w.into_bytes()
    }

    fn unpack(bytes: &[u8]) -> Result<AppliedSet, WireError> {
        let mut r = Reader::new(bytes, "HAVE");
        let set = read_applied(&mut r)?;
        r.finish()?;
        Ok(set)
    }

    /// A caught-up origin is its id, its top and no gaps: eleven bytes here.
    #[test]
    fn a_caught_up_origin_packs_to_exact_bytes() {
        let mut set = AppliedSet::default();
        set.insert_range(0x0102_0304_0506_0708, 1, 300);
        assert_eq!(
            pack(&set),
            vec![
                1, 0, // one origin
                8, 7, 6, 5, 4, 3, 2, 1, // origin
                0xac, 0x02, // top 300
                0,    // no gaps
            ]
        );
        assert_eq!(unpack(&pack(&set)).unwrap(), set);
    }

    /// An event that overtook lower ones leaves gaps, packed exactly; the
    /// lower ones are still wanted after a round trip.
    #[test]
    fn gaps_pack_as_deltas_and_survive_the_round_trip() {
        let mut set = AppliedSet::default();
        set.insert_range(9, 1, 3); // through 3
        set.insert_range(9, 6, 7); // gap 4..=5
        set.insert(9, 10); // gap 8..=9
        set.insert(2, 5); // gap 1..=4
        assert_eq!(
            pack(&set),
            vec![
                2, 0, //
                // origin 2: top 5, gap 1..=4
                2, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 4, //
                // origin 9: top 10, gaps 4..=5 and 8..=9
                9, 0, 0, 0, 0, 0, 0, 0, 10, 2, 4, 2, 3, 2, //
            ]
        );
        let back = unpack(&pack(&set)).unwrap();
        assert_eq!(back, set);
        assert!(!back.contains(9, 4) && !back.contains(9, 9) && back.contains(9, 10));
        assert!(!back.contains(2, 1) && back.contains(2, 5));
    }

    #[test]
    fn an_empty_set_is_two_bytes() {
        assert_eq!(pack(&AppliedSet::default()), vec![0, 0]);
        assert_eq!(unpack(&[0, 0]).unwrap(), AppliedSet::default());
    }

    /// One set has one spelling: unsorted origins, empty or touching gaps and a
    /// gap reaching the top are refused.
    #[test]
    fn a_second_spelling_is_refused() {
        let origin = |o: u8| [o, 0, 0, 0, 0, 0, 0, 0];
        let mut unsorted = vec![2, 0];
        unsorted.extend(origin(9));
        unsorted.extend([1, 0]);
        unsorted.extend(origin(2));
        unsorted.extend([1, 0]);
        assert!(unpack(&unsorted).is_err());

        let mut touching = vec![1, 0];
        touching.extend(origin(3));
        touching.extend([10, 2, 2, 2, 1, 2]); // gaps 2..=3 and 4..=5 touch
        assert!(unpack(&touching).is_err());

        let mut at_top = vec![1, 0];
        at_top.extend(origin(3));
        at_top.extend([5, 1, 4, 2]); // gap 4..=5 reaches top 5
        assert!(unpack(&at_top).is_err());

        let mut zero_top = vec![1, 0];
        zero_top.extend(origin(3));
        zero_top.extend([0, 0]);
        assert!(unpack(&zero_top).is_err());
    }
}
