//! What a group's exclusion learns over, as one substrate version holds it:
//! each competition's gallery streams, and every calibration question the
//! sample is drawn from (`docs/incremental_chunk_exclusion.md` §2, §4.1).

use std::collections::BTreeMap;

use crate::persistence::streams::StreamId;
use crate::projection::belief_files::assemble_file_scans;
use crate::projection::ids::{GroupId, TimelineId};
use crate::provenance::chunk_exclusion::anomaly::{K, K_CALIBRATION};
use crate::provenance::chunk_exclusion::stats::FileId;
use crate::provenance::SigWindow;
use crate::substrate::Substrate;

/// The two galleries a file-scored group is scored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Competition {
    /// The files' own turns.
    Content,
    /// The files' calibration questions, the sample's own left out.
    Calibration,
}

impl Competition {
    pub(crate) const BOTH: [Competition; 2] = [Competition::Content, Competition::Calibration];

    pub(crate) fn index(self) -> usize {
        match self {
            Competition::Content => 0,
            Competition::Calibration => 1,
        }
    }

    /// The robust deviations a chunk of this competition is cut at.
    pub(crate) fn k(self) -> f32 {
        match self {
            Competition::Content => K,
            Competition::Calibration => K_CALIBRATION,
        }
    }
}

/// One gallery stream: a turn, the file whose case it votes for, the token
/// ranges of it the competition scans, its signature, and the stamp of the
/// record holding it.
#[derive(Clone, Debug)]
pub(crate) struct GalleryStream {
    pub sid: StreamId,
    pub file: FileId,
    pub sig: SigWindow,
    pub windows: Vec<(usize, usize)>,
    pub hlc: u64,
}

impl GalleryStream {
    /// Tokens its windows span.
    pub(crate) fn span(&self) -> usize {
        self.windows.iter().map(|&(s, e)| e.saturating_sub(s)).sum()
    }
}

/// One calibration question: its stream, the file it asks about, its
/// signature, and the stamp of the record holding it.
#[derive(Clone, Debug)]
pub(crate) struct Question {
    pub sid: StreamId,
    pub owner: FileId,
    pub sig: SigWindow,
    pub hlc: u64,
}

/// See the module documentation. The calibration gallery holds every
/// calibration question here; the learner leaves the sample's out.
#[derive(Clone, Debug, Default)]
pub(crate) struct GalleryView {
    pub content: Vec<GalleryStream>,
    pub calibration: Vec<GalleryStream>,
    pub questions: Vec<Question>,
}

impl GalleryView {
    /// `group`'s galleries and questions as `sub` holds them, each file's
    /// gallery read from its timeline where one is held
    /// ([`Substrate::file_gallery`]).
    pub(crate) fn of(sub: &Substrate, group: GroupId) -> Self {
        let timelines: Vec<TimelineId> = sub.active_timelines_for_group(group).collect();
        let (files, _) = assemble_file_scans(sub, timelines);
        let mut view = GalleryView::default();
        for f in &files {
            let file = f.timeline.raw();
            let mut by_arc: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
            for &(k, s, e, _) in &f.windows {
                by_arc.entry(k).or_default().push((s, e));
            }
            for (k, windows) in by_arc {
                let sid = f.sid(k);
                let Some(hlc) = sub.wide_sig_hlc(sid) else {
                    continue;
                };
                view.content.push(GalleryStream {
                    sid,
                    file,
                    sig: f.arc(k).clone(),
                    windows,
                    hlc,
                });
            }
            for &(k, s, e) in &f.cal_windows {
                let sid = f.sid(k);
                let Some(hlc) = sub.wide_sig_hlc(sid) else {
                    continue;
                };
                let sig = f.arc(k).clone();
                view.calibration.push(GalleryStream {
                    sid,
                    file,
                    sig: sig.clone(),
                    windows: vec![(s, e)],
                    hlc,
                });
                view.questions.push(Question {
                    sid,
                    owner: file,
                    sig,
                    hlc,
                });
            }
        }
        view
    }

    /// A competition's gallery streams.
    pub(crate) fn streams(&self, comp: Competition) -> &[GalleryStream] {
        match comp {
            Competition::Content => &self.content,
            Competition::Calibration => &self.calibration,
        }
    }

    /// The signature width the galleries hold — what a question's tokens must
    /// match to be scanned; `None` with no gallery.
    pub(crate) fn width(&self) -> Option<usize> {
        self.content
            .iter()
            .chain(&self.calibration)
            .map(|s| s.sig.width())
            .next()
    }
}
