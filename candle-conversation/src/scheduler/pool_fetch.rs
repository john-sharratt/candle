//! A pool refresh's index pages, fetched together.
//!
//! A refresh reads the index page of every turn it walks before it places any
//! (`pool_refresh`). A page held in memory comes back at once; one only the log
//! has is read off the disk, and the walk used to read them one after another
//! — on BlackWorld a reprojection admitting a dozen conversations read 50–170
//! pages cold, 40–180 ms of disk waits end to end (2026-11-21). The reads are
//! independent, so they are made from a few threads at once; what the walk does
//! with each page, and in what order, is unchanged.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use crate::error::ConversationError;
use crate::index_pages::StoredPages;
use crate::projection::{Conversation, TimelineId, TurnIndex};

/// The most threads one refresh reads with.
const READERS: usize = 8;

/// Below this many pages the reads are made on the calling thread: starting
/// threads costs more than a held page's read.
const ALONE_BELOW: usize = 4;

/// What one turn's index page read gave.
pub(super) type Fetched = Result<Option<Arc<StoredPages>>, ConversationError>;

/// Every page of `wanted`, by its turn.
pub(super) fn fetch_pages(
    conversation: &Conversation,
    wanted: &[(TimelineId, TurnIndex)],
) -> HashMap<(TimelineId, TurnIndex), Fetched> {
    let read = |&(tl, i): &(TimelineId, TurnIndex)| ((tl, i), conversation.turn_index_page(tl, i));
    if wanted.len() < ALONE_BELOW {
        return wanted.iter().map(read).collect();
    }
    let shares = shares(wanted.len(), READERS);
    thread::scope(|s| {
        let mut rest = wanted;
        let readers: Vec<_> = shares
            .into_iter()
            .map(|n| {
                let (mine, after) = rest.split_at(n);
                rest = after;
                s.spawn(move || mine.iter().map(read).collect::<Vec<_>>())
            })
            .collect();
        readers
            .into_iter()
            .flat_map(|r| r.join().expect("an index page reader panicked"))
            .collect()
    })
}

/// `n` reads split over at most `readers` threads, as evenly as they go, none
/// of them empty.
fn shares(n: usize, readers: usize) -> Vec<usize> {
    let k = readers.min(n).max(1);
    (0..k).map(|i| n / k + usize::from(i < n % k)).collect()
}

#[cfg(test)]
mod tests {
    use super::shares;

    /// Every read goes to exactly one reader, the readers' shares differ by at
    /// most one, and no reader is started for nothing.
    #[test]
    fn reads_split_evenly_over_the_readers() {
        assert_eq!(shares(17, 8), vec![3, 2, 2, 2, 2, 2, 2, 2]);
        assert_eq!(shares(16, 8), vec![2; 8]);
        assert_eq!(shares(5, 8), vec![1; 5]);
        assert_eq!(shares(1, 8), vec![1]);
        assert_eq!(shares(0, 8), vec![0]);
    }
}
