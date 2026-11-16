//! The gallery arena, reached from another thread.
//!
//! A [`RemoteArena`] is a [`BeliefArena`] whose launches run on the scheduler's
//! wave thread — the arena's one scan thread — and nothing else does: a belief
//! scan holding one runs on its own thread, assembling its galleries from the
//! substrate and normalizing what comes back there, and sends the wave thread
//! only the launches ([`SchedulerRequest::ArenaScan`]). The seal scan is the
//! one that does: every sealed turn's, on the sealing conversation's thread.

use flume::Sender;

use super::SchedulerRequest;
use crate::provenance::gallery_arena::{
    scan_split, BeliefArena, HeldScan, PagedSegment, ScanAnswer, ScanReply, TokenStats, TokenVotes,
};
use crate::provenance::WideQSig;

/// See the module documentation.
pub(crate) struct RemoteArena {
    tx: Sender<SchedulerRequest>,
    n_groups: usize,
}

impl RemoteArena {
    /// The scheduler's arena, reached over `tx`; `None` when it has none or is
    /// gone — a scan then scores on the host, as one with no arena does.
    pub(crate) fn connect(tx: &Sender<SchedulerRequest>) -> Option<Self> {
        let (response_tx, response_rx) = flume::bounded(1);
        tx.send(SchedulerRequest::ArenaGroups { response_tx })
            .ok()?;
        let n_groups = response_rx.recv().ok()??;
        Some(Self {
            tx: tx.clone(),
            n_groups,
        })
    }

    /// Run `scan` on the wave thread and wait for it, folding the arena's time
    /// into this thread's split.
    fn ask(&self, scan: HeldScan) -> candle::Result<ScanAnswer> {
        let (response_tx, response_rx) = flume::bounded(1);
        self.tx
            .send(SchedulerRequest::ArenaScan { scan, response_tx })
            .map_err(|_| candle::Error::Msg("arena scan: the scheduler is gone".into()))?;
        let ScanReply { answer, split } = response_rx
            .recv()
            .map_err(|_| candle::Error::Msg("arena scan: the scheduler is gone".into()))?;
        scan_split::merge(split);
        Ok(answer)
    }

    fn scores(&self, scan: HeldScan) -> candle::Result<Vec<Vec<f32>>> {
        match self.ask(scan)? {
            ScanAnswer::Scores(r) => r.map_err(candle::Error::Msg),
            _ => Err(candle::Error::Msg(
                "arena scan: a scores launch answered with something else".into(),
            )),
        }
    }
}

impl BeliefArena for RemoteArena {
    fn n_groups(&self) -> usize {
        self.n_groups
    }

    fn scan_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> candle::Result<Vec<Vec<f32>>> {
        self.scores(HeldScan::weighted(segments, probes, group_weights))
    }

    fn scan_segment_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        segment_weights: &[&[f32]],
    ) -> candle::Result<Vec<Vec<f32>>> {
        self.scores(HeldScan::segment_weighted(
            segments,
            probes,
            segment_weights,
        ))
    }

    fn scan_votes(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
    ) -> candle::Result<TokenVotes> {
        match self.ask(HeldScan::votes(segments, probes))? {
            ScanAnswer::Votes(r) => r.map_err(candle::Error::Msg),
            _ => Err(candle::Error::Msg(
                "arena scan: a votes launch answered with something else".into(),
            )),
        }
    }

    fn scan_stats(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
    ) -> candle::Result<TokenStats> {
        match self.ask(HeldScan::stats(segments, probes))? {
            ScanAnswer::Stats(r) => r.map_err(candle::Error::Msg),
            _ => Err(candle::Error::Msg(
                "arena scan: a statistics launch answered with something else".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::provenance::gallery_arena::scan_split::ArenaScanSplit;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// A stand-in wave thread: answers the fold-group count with `groups`, and
    /// each launch with the scores `answer` makes of its probes, counting one
    /// scan and `kernel_us` of kernel.
    fn wave(
        groups: Option<usize>,
        kernel_us: u64,
        answer: fn(&[&[WideQSig]]) -> Result<Vec<Vec<f32>>, String>,
    ) -> Sender<SchedulerRequest> {
        let (tx, rx) = flume::unbounded::<SchedulerRequest>();
        thread::spawn(move || {
            for request in rx.iter() {
                match request {
                    SchedulerRequest::ArenaGroups { response_tx } => {
                        let _ = response_tx.send(groups);
                    }
                    SchedulerRequest::ArenaScan { scan, response_tx } => {
                        let _ = response_tx.send(ScanReply {
                            answer: ScanAnswer::Scores(answer(&scan.probes())),
                            split: ArenaScanSplit {
                                scans: 1,
                                kernel_us,
                                ..ArenaScanSplit::default()
                            },
                        });
                    }
                    _ => {}
                }
            }
        });
        tx
    }

    /// **A launch runs on the wave thread and its answer comes back**, with
    /// the probes it was sent, and the arena's time joins this thread's split.
    #[test]
    fn a_launch_runs_there_and_its_time_is_counted_here() {
        let tx = wave(Some(3), 40, |probes| {
            Ok(probes
                .iter()
                .map(|p| vec![p.len() as f32, p[0].words[0] as f32])
                .collect())
        });
        let arena = RemoteArena::connect(&tx).expect("the wave thread has an arena");
        assert_eq!(arena.n_groups(), 3);
        scan_split::take();
        let tail = vec![sig(5), sig(6)];
        let question = vec![sig(9)];
        let got = arena
            .scan_segment_weighted(&[], &[&tail, &question], &[])
            .expect("the launch answers");
        assert_eq!(got, vec![vec![2.0, 5.0], vec![1.0, 9.0]]);
        let got = arena.scan_weighted(&[], &[&question], &[1.0]).unwrap();
        assert_eq!(got, vec![vec![1.0, 9.0]]);
        let split = scan_split::take();
        assert_eq!((split.scans, split.kernel_us), (2, 80));
    }

    /// **A launch's failure is the caller's error**, in the launch's words.
    #[test]
    fn a_failed_launch_is_an_error_here() {
        let tx = wave(Some(3), 0, |_| Err("paged scan: no backend".to_string()));
        let arena = RemoteArena::connect(&tx).unwrap();
        let err = arena.scan_weighted(&[], &[&[sig(1)]], &[1.0]).unwrap_err();
        assert_eq!(err.to_string(), "paged scan: no backend");
    }

    /// **No arena, or no scheduler, is no handle**: the scan scores on the
    /// host.
    #[test]
    fn no_arena_or_no_scheduler_is_no_handle() {
        assert!(RemoteArena::connect(&wave(None, 0, |_| Ok(Vec::new()))).is_none());
        let (tx, rx) = flume::unbounded::<SchedulerRequest>();
        drop(rx);
        assert!(RemoteArena::connect(&tx).is_none());
    }
}
