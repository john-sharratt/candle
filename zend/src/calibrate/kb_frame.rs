//! The two knowledge bases a file's questions are asked behind
//! (`docs/provenance_system_collection.md` §7 step 3, §12 D11, D12).
//!
//! A file's questions take them in turn, by their order in the draft: the
//! first, third, fifth… behind the **empty** knowledge base — `files_open`
//! with nothing under it, the frame a query has before it has found the file —
//! and the second, fourth, sixth… behind the **target**, the file alone in
//! the knowledge base, the frame a query has once the file is in front of it.
//! Each frame's questions are sealed as a calibration of their own, behind the
//! frame they were asked in, and say which ([`CALIBRATION_FRAME_KEY`]).

use candle_conversation::files::{FileKey, FilesPin};

/// Conversation metadata on a calibration: the frame its questions were asked
/// and sealed in — `empty` or `target`.
pub const CALIBRATION_FRAME_KEY: &str = "calibration_frame";

/// A knowledge base a calibration is asked behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KbFrame {
    /// `files_open` with no file under it.
    Empty,
    /// The target file alone.
    Target,
}

impl KbFrame {
    /// Both frames, in the order a file's questions alternate through them.
    pub const ALL: [KbFrame; 2] = [KbFrame::Empty, KbFrame::Target];

    /// The frame the `i`-th question of a draft (from 0) is asked behind: the
    /// first, third, fifth… behind the empty knowledge base.
    pub fn of_question(i: usize) -> KbFrame {
        if i.is_multiple_of(2) {
            KbFrame::Empty
        } else {
            KbFrame::Target
        }
    }

    /// What the frame's knowledge base holds, for the file `file`.
    pub fn pin(self, file: FileKey) -> FilesPin {
        match self {
            KbFrame::Empty => FilesPin::Empty,
            KbFrame::Target => FilesPin::Only(vec![file]),
        }
    }

    /// The frame's name, as [`CALIBRATION_FRAME_KEY`] records it.
    pub fn as_str(self) -> &'static str {
        match self {
            KbFrame::Empty => "empty",
            KbFrame::Target => "target",
        }
    }

    /// The frame [`Self::as_str`] names.
    pub fn parse(name: &str) -> Option<KbFrame> {
        KbFrame::ALL.into_iter().find(|f| f.as_str() == name)
    }

    /// The indices of `n` questions this frame asks, in draft order.
    pub fn questions_of(self, n: usize) -> Vec<usize> {
        (0..n)
            .filter(|&i| KbFrame::of_question(i) == self)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Questions alternate empty then target in draft order.**
    #[test]
    fn questions_alternate_empty_then_target_in_draft_order() {
        let frames: Vec<KbFrame> = (0..5).map(KbFrame::of_question).collect();
        assert_eq!(
            frames,
            [
                KbFrame::Empty,
                KbFrame::Target,
                KbFrame::Empty,
                KbFrame::Target,
                KbFrame::Empty
            ]
        );
        assert_eq!(KbFrame::Empty.questions_of(5), [0, 2, 4]);
        assert_eq!(KbFrame::Target.questions_of(5), [1, 3]);
        assert!(KbFrame::Target.questions_of(1).is_empty());
    }

    /// **An empty frame holds no file, a target frame its own.**
    #[test]
    fn an_empty_frame_is_the_heading_alone_and_a_target_its_file() {
        let f = FileKey(42);
        assert_eq!(KbFrame::Empty.pin(f), FilesPin::Empty);
        assert_eq!(KbFrame::Target.pin(f), FilesPin::Only(vec![f]));
        for frame in KbFrame::ALL {
            assert_eq!(KbFrame::parse(frame.as_str()), Some(frame));
        }
        assert_eq!(KbFrame::parse("all"), None);
    }
}
