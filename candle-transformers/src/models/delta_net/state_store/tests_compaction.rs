//! Tests that a compaction moves every state it plans to whole, including after a
//! step whose later segment failed — the sequence WhiteWorld's NaN followed.
//!
//! On WhiteWorld (2026-11-22 08:49) a creep window failed on a given-up expert
//! inside a step, the step was taken back, a recurrent compaction ran, and three
//! stores then entered their next wave non-finite from layer 20 up — the upper
//! layers' slots, the ones a compaction moves — after every wave in between had
//! left them clean. These tests run that sequence on the store alone.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    use crate::models::gpu_test_lock::gpu_serial;

    use super::super::tests_layers::tests::{dims, filled_store_on, kinds, long_span, short_span};

    /// Stores enough to fill more than a region of these slots (a region holds
    /// ~6,500 of the test geometry's 2.5 KiB states, three a store), so the two
    /// claimed after them stand in an arena the pass can empty.
    #[cfg(feature = "cuda")]
    const FILLERS: usize = 2_500;

    /// Every layer's live state and conv tail, read back.
    #[cfg(feature = "cuda")]
    fn live(store: &RecurrentStateStore) -> Result<Vec<Vec<f32>>> {
        store
            .slots
            .iter()
            .flat_map(|s| [&s.live.state.s, &s.live.state.conv_tail])
            .map(|t| t.flatten_all()?.to_vec1())
            .collect()
    }

    /// Every element of every buffer one more than in `before`.
    #[cfg(feature = "cuda")]
    fn bumped(before: &[Vec<f32>]) -> Vec<Vec<f32>> {
        before
            .iter()
            .map(|b| b.iter().map(|v| v + 1.0).collect())
            .collect()
    }

    /// **A failed step, then a compaction, leaves both its stores as they entered,
    /// and the retry lands on them.** A decoding store inside a step commits one
    /// segment's short spans, then a later segment fails with a creep's store
    /// beside it, whose long spans had written its scratch; both waves roll back
    /// and the step is undone. Stores claimed below them are dropped, so the
    /// compaction moves their upper slots down. Both read back exactly the state
    /// they entered with, and the creep's retried window lands one long span on
    /// each layer.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_compaction_after_a_failed_step_moves_every_state_whole() -> Result<()> {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0)?;
        let d = dims();
        // Below them: stores whose slots become the holes the pass moves into.
        let fillers: Vec<RecurrentStateStore> = (0..FILLERS)
            .map(|_| RecurrentStateStore::new(&kinds(), &d, &dev))
            .collect::<Result<_>>()?;
        let mut decode = filled_store_on(&dev);
        let mut creep = filled_store_on(&dev);
        creep.ensure_scratch()?;
        let (decode_in, creep_in) = (live(&decode)?, live(&creep)?);

        // The step: one segment commits the decode's short spans (pending inside
        // the step), the next fails with the creep's long spans beside it.
        decode.begin_step()?;
        decode.begin_wave()?;
        short_span(&mut decode, 0, 1.0, 1.0);
        short_span(&mut decode, 1, 1.0, 1.0);
        commit_stores(&mut [(&mut decode, true)])?;
        decode.begin_wave()?;
        creep.begin_wave()?;
        short_span(&mut decode, 3, 1.0, 1.0);
        for layer in [0, 1, 3] {
            long_span(&mut creep, layer);
        }
        decode.rollback_wave()?;
        creep.rollback_wave()?;
        decode.undo_step()?;
        assert!(!decode.has_pending(), "the undo dropped the step's rows");

        drop(fillers);
        let pass = compact_stores([&mut decode, &mut creep], &d, &dev, 256)?;
        assert!(pass.planned > 0, "the dropped stores' slots draw moves");
        assert_eq!(
            pass.moved, pass.planned,
            "every planned source is held here"
        );

        assert_eq!(live(&decode)?, decode_in, "the decode moved as it entered");
        assert_eq!(live(&creep)?, creep_in, "the creep moved as it entered");

        // The creep's retry: the window again, committed this time.
        creep.begin_wave()?;
        for layer in [0, 1, 3] {
            long_span(&mut creep, layer);
        }
        commit_stores(&mut [(&mut creep, true)])?;
        assert_eq!(live(&creep)?, bumped(&creep_in), "the retry landed once");
        Ok(())
    }

    /// **A compaction between two committed windows moves the state the first
    /// one installed.** The creep's long spans exchange its scratch into the
    /// live state at each commit, so after a window the live half is the slot the
    /// scratch was; the pass that follows must carry that state, not the one the
    /// window replaced.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_compaction_between_windows_moves_the_installed_state() -> Result<()> {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0)?;
        let d = dims();
        let fillers: Vec<RecurrentStateStore> = (0..FILLERS)
            .map(|_| RecurrentStateStore::new(&kinds(), &d, &dev))
            .collect::<Result<_>>()?;
        let mut creep = filled_store_on(&dev);
        creep.ensure_scratch()?;
        let entered = live(&creep)?;

        creep.begin_wave()?;
        for layer in [0, 1, 3] {
            long_span(&mut creep, layer);
        }
        commit_stores(&mut [(&mut creep, true)])?;
        let installed = live(&creep)?;
        assert_eq!(
            installed,
            bumped(&entered),
            "the window installed its state"
        );

        drop(fillers);
        let pass = compact_stores([&mut creep], &d, &dev, 256)?;
        assert!(pass.planned > 0, "the dropped stores' slots draw moves");
        assert_eq!(live(&creep)?, installed, "the installed state moved whole");

        creep.ensure_scratch()?;
        creep.begin_wave()?;
        for layer in [0, 1, 3] {
            long_span(&mut creep, layer);
        }
        commit_stores(&mut [(&mut creep, true)])?;
        assert_eq!(
            live(&creep)?,
            bumped(&installed),
            "the next window read the moved state"
        );
        Ok(())
    }
}
