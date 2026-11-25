// `assert!(cfg!(…))` is the point: the probe must read on Windows and Linux
// and may return `None` elsewhere, so the platform constant is what is asserted.
#![allow(clippy::assertions_on_constants)]

use super::*;

/// Memory the liveness check holds touched: several times this test binary's
/// own footprint (~75 MB), so the committed figure crosses it only while it is
/// held.
const HELD: usize = 512 << 20;

/// **This process is found, and its counters are live**: its committed bytes
/// are under [`HELD`] before the test touches that much and at least that
/// after, so the figure moved with the allocation; and the fault count never
/// goes back. The other tests of the binary allocate and free beside this one,
/// so the figures are bounded against what this test holds, never compared
/// with each other. Committed bytes, because the OS may trim a resident figure
/// at any moment.
#[test]
fn own_paging_is_read_and_its_fault_count_never_falls() {
    let Some(first) = own_paging() else {
        assert!(cfg!(not(any(windows, target_os = "linux"))));
        return;
    };
    assert_eq!(first.pid, std::process::id());
    assert!(first.working_set > 0, "{first:?}");
    assert!(
        first.private_bytes > 0 && first.private_bytes < HELD as u64,
        "{first:?}"
    );
    let held = std::hint::black_box(vec![1u8; HELD]);
    let second = own_paging().unwrap();
    assert!(
        second.hard_faults >= first.hard_faults,
        "{first:?} then {second:?}"
    );
    assert!(
        second.private_bytes >= HELD as u64,
        "{second:?} with {HELD} bytes touched"
    );
    drop(held);
}

/// **A process is found by its image name**, case and `.exe` aside.
#[test]
fn a_process_is_found_by_name() {
    let Some(me) = own_paging() else {
        return;
    };
    let found = paging_of_named(&me.name.to_ascii_uppercase());
    assert!(
        found.iter().any(|p| p.pid == me.pid),
        "{me:?} not in {found:?}"
    );
    assert_eq!(bare_name("MotherD.EXE"), "motherd");
    assert_eq!(bare_name("motherd"), "motherd");
}

#[cfg(windows)]
mod windows_records {
    use super::super::windows::{
        parse, HARD_FAULTS, NAME_LEN, NAME_PTR, NEXT, PID, PRIVATE, RECORD, WS, WS_PRIVATE,
    };

    /// Two records and a name area, laid out as the kernel lays them out.
    fn buffer() -> Vec<u64> {
        let mut words = vec![0u64; 512];
        let base = words.as_ptr() as u64;
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), words.len() * 8) };
        let put32 =
            |b: &mut [u8], at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let put64 =
            |b: &mut [u8], at: usize, v: u64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
        // The name, UTF-16, after both records.
        let name_at = 3 * RECORD;
        for (i, u) in "motherd.exe".encode_utf16().enumerate() {
            bytes[name_at + 2 * i..name_at + 2 * i + 2].copy_from_slice(&u.to_le_bytes());
        }
        // Record 0: the idle process, no name; next record at 256.
        put32(bytes, NEXT, 256);
        put64(bytes, PID, 0);
        // Record 1, at 256: motherd.
        let r = 256;
        put32(bytes, r + NEXT, 0);
        put64(bytes, r + WS_PRIVATE, 18 << 30);
        put32(bytes, r + HARD_FAULTS, 123_456);
        put32(bytes, r + NAME_LEN, 22);
        put64(bytes, r + NAME_PTR, base + name_at as u64);
        put64(bytes, r + PID, 2060);
        put64(bytes, r + WS, 29 << 30);
        put64(bytes, r + PRIVATE, 80 << 30);
        words
    }

    /// **The record fields are read at their offsets**, the name through its
    /// pointer into the same buffer, and the walk ends at a zero link.
    #[test]
    fn the_records_are_read_at_their_offsets() {
        let words = buffer();
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 8) };
        let all = parse(bytes);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].pid, 0);
        assert_eq!(all[0].name, "");
        let m = &all[1];
        assert_eq!(m.pid, 2060);
        assert_eq!(m.name, "motherd.exe");
        assert_eq!(m.hard_faults, 123_456);
        assert_eq!(m.private_working_set, 18 << 30);
        assert_eq!(m.working_set, 29 << 30);
        assert_eq!(m.private_bytes, 80 << 30);
    }

    /// A link past the buffer ends the walk rather than reading past it.
    #[test]
    fn a_link_past_the_buffer_ends_the_walk() {
        let mut words = buffer();
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), words.len() * 8) };
        bytes[256 + NEXT..256 + NEXT + 4].copy_from_slice(&100_000u32.to_le_bytes());
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 8) };
        assert_eq!(parse(bytes).len(), 2);
    }
}

#[cfg(target_os = "linux")]
mod linux_stat {
    use super::super::linux::parse_stat;

    /// **`majflt` is the tenth field after the name**, and a name holding
    /// spaces and parentheses does not move it.
    #[test]
    fn majflt_is_read_after_the_last_parenthesis() {
        let stat = "2060 (mother d) (x)) S 1 2060 2060 0 -1 4194560 912 0 77 0 5 3 0 0 20 0";
        assert_eq!(parse_stat(stat), Some(("mother d) (x)".to_string(), 77)));
    }
}
