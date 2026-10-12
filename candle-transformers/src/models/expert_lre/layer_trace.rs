//! A forward's walk through the routed layers, as the telemetry page draws it:
//! per layer, how many experts it routed and where each one stood when
//! bucketize classified it, and how long the device took to reach it from the
//! layer before.
//!
//! Built on the pipeline thread from what it reads anyway. A row's routing
//! summary lands in mapped memory the moment that row's bucketize runs, and the
//! thread is already polling for it — so the gap between two rows' arrivals is
//! the device's time between the two launches, measured with no event and no
//! fence. The summary's tier bits say the rest.
//!
//! The newest finished pass of each kind is kept — one decode, one prefill — so
//! a page polling every few seconds sees both shapes rather than whichever kind
//! happened to finish last. The pipeline thread publishes with `try_lock` and
//! drops a trace whose slot a reader holds: the next forward's replaces it, and
//! the thread that feeds the GPU never waits on a page.

use std::mem::take;
use std::sync::Mutex;
use std::time::Instant;

/// One routed layer of one forward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerVisit {
    /// The MoE row.
    pub row: usize,
    /// Routed experts already resident in VRAM.
    pub hot: u32,
    /// Routed experts in the pinned host pool, read over the link by the
    /// layer's workers.
    pub warm: u32,
    /// Routed experts in no pinned slot, which waited on a pack read.
    pub cold: u32,
    /// Seconds from the previous row's routing summary to this row's: the
    /// previous row's expert GEMMs, then this row's attention and router.
    /// `None` for the first row of a pass, which has no earlier summary to be
    /// timed against.
    pub secs: Option<f64>,
    /// The summary was already written when the thread began waiting for it,
    /// so it sat unread and [`Self::secs`] is the thread's pace rather than the
    /// device's.
    pub late: bool,
}

impl LayerVisit {
    /// The experts the layer routed.
    pub fn routed(&self) -> u32 {
        self.hot + self.warm + self.cold
    }
}

/// One finished forward.
#[derive(Clone, Debug)]
pub struct PassTrace {
    /// A prompt-prefill launch (`RoutedLayer::prefill_width`) on any row.
    pub prefill: bool,
    /// When its last row's summary arrived.
    pub at: Instant,
    /// Every routed row, in order.
    pub layers: Vec<LayerVisit>,
}

/// The pass being walked, on the pipeline thread.
#[derive(Default)]
pub(crate) struct TraceBuilder {
    pass: Option<u64>,
    prefill: bool,
    last: Option<(Instant, usize)>,
    layers: Vec<LayerVisit>,
}

/// One row's routing, as the pipeline thread read it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowRouting {
    pub pass: u64,
    pub row: usize,
    pub arrived: Instant,
    pub prefill: bool,
    /// Every expert the row routed, whatever its tier.
    pub routed: usize,
    pub pinned: usize,
    pub cold: usize,
    /// See [`LayerVisit::late`].
    pub late: bool,
}

impl TraceBuilder {
    /// Add one row, returning a pass that is finished: this one when the row is
    /// the model's last (`rows` is its routed row count), or the one before it
    /// when this row begins a new pass. A pass ends either way — a decode's
    /// walk stops short of the drafter's row, which the draft step then routes
    /// as a pass of its own.
    ///
    /// Only a pass that begins at row 0 is a forward through the layers; one
    /// that begins further in — the draft step's walk over the drafter's row
    /// alone — is dropped.
    pub(crate) fn visit(&mut self, r: RowRouting, rows: usize) -> Option<PassTrace> {
        let mut ended = None;
        if self.pass != Some(r.pass) {
            ended = self.finish();
            self.pass = Some(r.pass);
            self.prefill = false;
            self.last = None;
        }
        let secs = match self.last {
            Some((at, prev)) if r.row == prev + 1 => {
                Some(r.arrived.saturating_duration_since(at).as_secs_f64())
            }
            _ => None,
        };
        self.last = Some((r.arrived, r.row));
        self.prefill |= r.prefill;
        // Pinned and cold are exclusive — a cold expert is one in no pinned
        // slot — and what is neither was resident.
        let warm = r.pinned as u32;
        let cold = r.cold as u32;
        let hot = (r.routed.saturating_sub(r.pinned + r.cold)) as u32;
        self.layers.push(LayerVisit {
            row: r.row,
            hot,
            warm,
            cold,
            secs,
            late: r.late,
        });
        if r.row + 1 < rows {
            return ended;
        }
        self.finish().or(ended)
    }

    /// The pass walked so far, when it is a forward through the layers.
    fn finish(&mut self) -> Option<PassTrace> {
        self.pass = None;
        let layers = take(&mut self.layers);
        let (at, _) = self.last?;
        (layers.first()?.row == 0).then_some(PassTrace {
            prefill: self.prefill,
            at,
            layers,
        })
    }
}

/// The newest decode pass and the newest prefill pass.
static LATEST: Mutex<[Option<PassTrace>; 2]> = Mutex::new([None, None]);

/// Keep `trace` as its kind's newest, unless a reader holds the slot.
pub(crate) fn publish(trace: PassTrace) {
    if let Ok(mut slots) = LATEST.try_lock() {
        let kind = usize::from(trace.prefill);
        slots[kind] = Some(trace);
    }
}

/// The newest pass of each kind that has finished, decode first.
pub fn latest() -> Vec<PassTrace> {
    let slots = LATEST.lock().unwrap_or_else(|e| e.into_inner());
    slots.iter().flatten().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn row(
        pass: u64,
        row: usize,
        at: Instant,
        routed: usize,
        pinned: usize,
        cold: usize,
    ) -> RowRouting {
        RowRouting {
            pass,
            row,
            arrived: at,
            prefill: false,
            routed,
            pinned,
            cold,
            late: false,
        }
    }

    /// A pass is returned on its last row, each row split by tier and timed
    /// from the row before it; the first row has nothing to be timed against.
    #[test]
    fn a_pass_is_returned_on_its_last_row_with_each_row_timed_from_the_last() {
        let t0 = Instant::now();
        let mut b = TraceBuilder::default();
        assert!(b.visit(row(7, 0, t0, 8, 2, 1), 3).is_none());
        assert!(b
            .visit(row(7, 1, t0 + Duration::from_millis(4), 8, 0, 0), 3)
            .is_none());
        let p = b
            .visit(row(7, 2, t0 + Duration::from_millis(10), 6, 3, 3), 3)
            .expect("the last row finishes the pass");
        assert!(!p.prefill);
        assert_eq!(
            p.layers,
            vec![
                LayerVisit {
                    row: 0,
                    hot: 5,
                    warm: 2,
                    cold: 1,
                    secs: None,
                    late: false
                },
                LayerVisit {
                    row: 1,
                    hot: 8,
                    warm: 0,
                    cold: 0,
                    secs: Some(0.004),
                    late: false
                },
                LayerVisit {
                    row: 2,
                    hot: 0,
                    warm: 3,
                    cold: 3,
                    secs: Some(0.006),
                    late: false
                },
            ]
        );
        assert_eq!(p.layers[2].routed(), 6);
    }

    /// A decode's walk stops short of the drafter's row (the last), and the
    /// draft step routes that row as a pass of its own: the walk is returned
    /// when the draft pass begins, and the draft pass — which starts past row
    /// 0 — is never returned.
    #[test]
    fn a_walk_that_stops_short_ends_at_the_next_pass_and_a_draft_pass_is_dropped() {
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(3);
        let mut b = TraceBuilder::default();
        assert!(b.visit(row(1, 0, t0, 4, 0, 0), 3).is_none());
        assert!(b.visit(row(1, 1, t1, 4, 0, 0), 3).is_none());
        let walk = b
            .visit(row(2, 2, t1 + Duration::from_millis(1), 1, 0, 0), 3)
            .expect("the draft pass beginning ends the walk");
        assert_eq!(
            walk.layers.iter().map(|l| l.row).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(walk.at, t1, "the walk ended at its own last row");
        // The draft pass ended on the last row, but it is no walk.
        assert!(b.visit(row(3, 0, t1, 4, 0, 0), 3).is_none());
    }

    /// A row whose predecessor was not seen is not timed against an older one.
    #[test]
    fn a_gap_is_not_timed() {
        let t0 = Instant::now();
        let mut b = TraceBuilder::default();
        b.visit(row(2, 0, t0, 4, 0, 0), 3);
        let p = b
            .visit(row(2, 2, t0 + Duration::from_millis(5), 4, 0, 0), 3)
            .expect("row 2 is the last row");
        assert_eq!(p.layers.len(), 2);
        assert_eq!(p.layers[1].secs, None);
    }

    /// Any prefill-width row makes the pass a prefill.
    #[test]
    fn a_prefill_width_row_makes_the_pass_a_prefill() {
        let t0 = Instant::now();
        let mut b = TraceBuilder::default();
        let mut r = row(3, 0, t0, 2, 0, 0);
        r.prefill = true;
        let p = b.visit(r, 1).expect("one row is the whole pass");
        assert!(p.prefill);
    }
}
