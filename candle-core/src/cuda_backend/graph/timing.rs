//! Device time of the segments a trial compares (`regime`).
//!
//! A segment on trial is bracketed by two events on the compute stream: one
//! recorded as it begins — before its launches are issued, eagerly or into a
//! capture — and one once they are all on the stream. The time between them
//! is what the device spent from the moment it could start the segment to the
//! moment it finished it, idle waits on a late host included: exactly what a
//! graph launch is meant to remove, measured the same way for both modes.
//!
//! **Never a wait.** A pair is read only once its end event has fired
//! (`cuEventQuery`), at the start of a later wave; one still in flight stays
//! queued. Events are reused, so a trial creates a handful and no more.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::{result, sys, CudaEvent, CudaStream};

use super::regime::{Key, Mode};
use crate::cuda_backend::WrapErr;
use crate::Result;

/// A bracketed segment whose end has been recorded.
struct Timed {
    key: Key,
    mode: Mode,
    start: CudaEvent,
    end: CudaEvent,
}

/// See the module documentation.
#[derive(Default)]
pub(super) struct SegmentTimer {
    free: Vec<CudaEvent>,
    pending: VecDeque<Timed>,
}

impl SegmentTimer {
    fn event(&mut self, on: &Arc<CudaStream>) -> Result<CudaEvent> {
        match self.free.pop() {
            Some(e) => Ok(e),
            // Timing enabled, named: `new_event(None)` is cudarc's
            // `CU_EVENT_DISABLE_TIMING`, whose pairs `elapsed` refuses — every
            // trial on `.5` (2026-10-17) judged with no sample at all.
            None => on
                .context()
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .w(),
        }
    }

    /// Record a segment's start on `on`.
    pub(super) fn start(&mut self, on: &Arc<CudaStream>) -> Result<CudaEvent> {
        let ev = self.event(on)?;
        ev.record(on).w()?;
        Ok(ev)
    }

    /// Record the end of the segment `start` began, as a run of `key` in
    /// `mode`.
    pub(super) fn end(
        &mut self,
        on: &Arc<CudaStream>,
        start: CudaEvent,
        key: Key,
        mode: Mode,
    ) -> Result<()> {
        let end = self.event(on)?;
        end.record(on).w()?;
        self.pending.push_back(Timed {
            key,
            mode,
            start,
            end,
        });
        Ok(())
    }

    /// A started segment that turned out not to be timed: its event is reused.
    pub(super) fn discard(&mut self, start: CudaEvent) {
        self.free.push(start);
    }

    /// Hand `took` every pair that has finished, oldest first, stopping at the
    /// first still in flight. A pair the driver will not time is an error, not
    /// a sample quietly dropped: dropped, the trial judges on nothing.
    pub(super) fn drain(&mut self, mut took: impl FnMut(Key, Mode, Duration)) -> Result<()> {
        while self.pending.front().is_some_and(|t| t.end.is_complete()) {
            let t = self.pending.pop_front().expect("checked above");
            // SAFETY: both events are this timer's, of one context, and both
            // have completed — the end was queried, and the start precedes it
            // on the same stream — so the read does not wait.
            let ms = unsafe { result::event::elapsed(t.start.cu_event(), t.end.cu_event()) };
            self.free.push(t.start);
            self.free.push(t.end);
            took(
                t.key,
                t.mode,
                Duration::from_secs_f64(f64::from(ms.w()?.max(0.0)) / 1e3),
            );
        }
        Ok(())
    }
}
