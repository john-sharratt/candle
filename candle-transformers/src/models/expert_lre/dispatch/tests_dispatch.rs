//! Tests of the read-ahead layout and of the waiter that polls a routing summary
//! word.
//!
//! The waiter's idle work must run once every fixed number of polls while the word
//! is outstanding, and an error from that idle work must end the wait. The layout
//! test pins each row's three offsets and its rounded image size.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A row's read-ahead layout is its three projection offsets and its image
    /// size rounded up to the 16-byte copy unit.
    #[test]
    fn the_row_layout_is_the_offsets_and_the_rounded_image() {
        let t = LiveTable::host_only(2, 4, [0x40, 0x1040, 0x2040]);
        assert_eq!(
            row_layout(&t, &[0x3000, 0x2f01]),
            vec![0x40, 0x1040, 0x2040, 0x3000, 0x40, 0x1040, 0x2040, 0x2f10]
        );
    }

    /// Store `word` where bucketize stores a slot's sequence word.
    fn write_word(ring: &SummaryRing, slot: usize, word: u32) {
        // SAFETY: the word is inside the mapped ring, after the slot's counts.
        unsafe { write_volatile(ring.host_slot_ptr(slot).add(ring.n_experts), word) };
    }

    /// The waiter's idle work runs every `WAIT_IDLE_EVERY` polls while the word
    /// is outstanding — here it is what finally writes the word — and not once
    /// when the word is already there.
    #[test]
    fn a_wait_runs_its_idle_work_until_the_word_arrives() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = SummaryRing::new(8).unwrap();
        let mut calls = 0u32;
        ring.wait(3, 77, || {
            calls += 1;
            if calls == 5 {
                write_word(&ring, 3, 77);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 5);
        assert!(ring.ready(3, 77));

        let mut late = 0u32;
        ring.wait(3, 77, || {
            late += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(late, 0, "a word already written needs no idle work");
    }

    /// An error from the idle work ends the wait with that error.
    #[test]
    fn an_idle_error_ends_the_wait() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = SummaryRing::new(8).unwrap();
        let e = ring
            .wait(1, 9, || candle::bail!("copier stopped"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("copier stopped"), "{e}");
    }
}
