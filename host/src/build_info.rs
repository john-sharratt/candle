//! What a daemon's build says of itself (`docs/replicated_substrate.md` §3.9):
//! the substrate format it opens, every record type it decodes, and the model
//! it is prescribed — what a member checks an artifact against before it
//! installs it. And the two things every binary does before it parses a flag:
//! answer `--build-info`, or run as an upgrade's watchdog.

use candle_conversation::persistence::log_file::FILE_FORMAT_VERSION;
use candle_conversation::persistence::survival::WRITTEN_RECORD_TYPES;
use cluster::upgrade::os::run_if_asked;
use cluster::version::info::{answer_if_asked, Opens};

use crate::model::MODEL;

/// What this build opens.
pub fn opens() -> Opens {
    let mut record_tags: Vec<u8> = WRITTEN_RECORD_TYPES.iter().map(|rt| rt.tag()).collect();
    record_tags.sort_unstable();
    Opens {
        log_format: Some(FILE_FORMAT_VERSION),
        record_tags,
        model: MODEL.override_key(),
    }
}

/// Before anything else: `--build-info` answers and exits, and
/// `--upgrade-watchdog <plan>` watches an install and exits. `kind` is the
/// binary's: `zend`, `npcd`, `tokerad`.
pub fn first(kind: &str) {
    answer_if_asked(kind, opens);
    run_if_asked();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build opens the log format it writes, decodes every record type it
    /// writes, in tag order, and runs the prescribed model.
    #[test]
    fn a_build_opens_what_it_writes() {
        let o = opens();
        assert_eq!(o.log_format, Some(FILE_FORMAT_VERSION));
        assert_eq!(o.record_tags.len(), WRITTEN_RECORD_TYPES.len());
        assert!(o.record_tags.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(o.model.as_deref(), Some("Qwen38_FlashNext"));
    }
}
