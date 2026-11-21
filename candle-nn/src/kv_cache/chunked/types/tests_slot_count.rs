//! Tests for the agreement check between the decode header's entry count and
//! the device buffer's.
//!
//! A header promising more entries than the buffer holds would walk off the slot
//! and read unmapped memory, a fault that surfaces later at an unrelated call.
//! The check refuses that direction. It also refuses a buffer longer than the
//! host list, because the divergence has one cause and should be caught where it
//! arises. Equal counts pass without error.

use super::*;

#[cfg(test)]
pub(super) mod slot_count_tests {
    use super::*;

    /// The agreeing case is the whole hot path — it must not cost an error.
    #[test]
    fn equal_counts_pass() {
        for n in [0usize, 1, 7, 4096] {
            assert!(SequenceState::slot_counts_agree(n, n).is_ok(), "n={n}");
        }
    }

    /// **The direction that reaches a kernel.** A header promising more entries
    /// than the buffer holds walks off the end of the slot and past its slab,
    /// reading unmapped memory as slice records and dereferencing the pointers
    /// inside them. That is a device-side out-of-range access on every warp at
    /// once — it poisons the CUDA context and kills the process, surfacing at
    /// whatever unrelated call happens to synchronise next, which is as far from
    /// the omission as a diagnosis can land.
    #[test]
    fn a_host_list_longer_than_the_buffer_is_refused() {
        let err = SequenceState::slot_counts_agree(9, 8)
            .expect_err("9 host chunks against an 8-entry buffer must not reach a kernel");
        let msg = err.to_string();
        assert!(msg.contains('9') && msg.contains('8'), "both counts: {msg}");
        assert!(
            msg.contains("clear"),
            "the message must name the omission that causes this — a `chunks` \
             mutation that skipped `gpu_chunks.clear()`: {msg}"
        );
    }

    /// Stale in the safe direction, and still refused: the kernel would only
    /// under-read, but the divergence has exactly one cause, and letting it pass
    /// here would leave that cause to be found the next time it lands the other
    /// way round.
    #[test]
    fn a_buffer_longer_than_the_host_list_is_also_refused() {
        assert!(SequenceState::slot_counts_agree(8, 9).is_err());
    }
}
