//! The gallery on the GPU arena: every covered `code_reading` turn resident in
//! a [`GalleryArena`], each file one case of one competition — the shape the
//! live scan gives a file-scored group — and each probe's raw votes read back
//! for the host to weigh, gate and fold under every setting a sweep tries.

use std::array::from_fn;

use candle::Device;
use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::provenance::gallery_arena::{PagedSegment, PagedWindow};
use candle_conversation::provenance::window_score::Vote;
use candle_conversation::provenance::{GalleryArena, WideQSig, FOLD_GROUPS};

use crate::corpus::File;

/// The locked fold: words per token and fold groups.
const WPT: usize = 24;

/// Every covered turn of every file, resident on the device.
pub struct GpuGallery<'a> {
    arena: GalleryArena,
    /// `(file, turn, its signatures, the file's timeline)`.
    turns: Vec<(usize, usize, &'a Vec<WideQSig>, u64)>,
    n_files: usize,
}

impl<'a> GpuGallery<'a> {
    pub fn new(files: &'a [File]) -> anyhow::Result<GpuGallery<'a>> {
        let device = Device::new_cuda(0)?;
        let arena = GalleryArena::new(&device, WPT, FOLD_GROUPS)?;
        let turns = files
            .iter()
            .enumerate()
            .flat_map(|(fi, f)| {
                f.turns
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.covered())
                    .map(move |(ti, t)| (fi, ti, &t.sigs, f.timeline))
            })
            .collect();
        Ok(GpuGallery {
            arena,
            turns,
            n_files: files.len(),
        })
    }

    /// Each probe's per-token, per-fold-group vote — the file it went to and
    /// its `z × margin` weight — against every turn `masked` does not name, as
    /// `(file, turn)`. A vote does not carry the gallery token it was won at.
    pub fn votes(
        &self,
        probes: &[&[WideQSig]],
        masked: &dyn Fn(usize, usize) -> bool,
    ) -> anyhow::Result<Vec<Vec<[Option<Vote>; FOLD_GROUPS]>>> {
        let windows: Vec<PagedWindow> = self
            .turns
            .iter()
            .filter(|(f, t, _, _)| !masked(*f, *t))
            .map(|&(f, t, sigs, timeline)| PagedWindow {
                sid: turn_stream_id(timeline, t as u32),
                fingerprint: 1,
                turn: sigs.into(),
                start: 0,
                end: sigs.len(),
                case: f,
            })
            .collect();
        let segment = PagedSegment {
            windows,
            n_cases: self.n_files,
        };
        let raw = self.arena.scan_votes(&[segment], probes)?;
        Ok((0..probes.len())
            .map(|p| {
                let first = raw.first_token(p);
                (first..first + raw.per_probe[p])
                    .map(|t| {
                        from_fn(|g| {
                            raw.at(t, g, 0).map(|(file, weight)| Vote {
                                token: usize::MAX,
                                file,
                                weight,
                            })
                        })
                    })
                    .collect()
            })
            .collect())
    }
}
