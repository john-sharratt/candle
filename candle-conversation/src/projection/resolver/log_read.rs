//! Reads a turn's index page and other stored records from the redo log, without
//! blocking on compaction.
//!
//! A durable record is read without the log's owner, and a record still staged in
//! RAM is read by the owner. Compaction relocates a record and repoints the index
//! before it unlinks the old segment, so a failed read is retried only if the index
//! has moved. Windows reports a file pending deletion as a permission error rather
//! than as not found, so the index is consulted either way. A held page is returned
//! from memory, so a reprojection does not read each page off the log again.

use super::*;

impl Conversation {
    /// The payload of the record at `loc`, wherever compaction has moved it
    /// since: `moved` answers where the index names the record now, given the
    /// location a read failed at.
    ///
    /// **A durable record is read without the log's owner** ([`SealedReader`]);
    /// one past the active segment's last commit, which may still be staged in
    /// RAM, is read by the owner.
    ///
    /// A stale location is the one thing to handle, on either path: compaction
    /// relocates the record, repoints the index, and only then unlinks the old
    /// segment. So after **any** failed read the location is read again, and
    /// the read is retried only if the index moved: what the error says varies
    /// by platform (a file pending deletion on Windows fails to open with
    /// `PermissionDenied`, not `NotFound`), whether the record moved does not.
    /// A record that is unreadable where the index still points is reported on
    /// its first failure. Bounded, because each retry follows a completed
    /// relocation; the bound only protects against a store that keeps
    /// compacting under the read.
    pub(super) fn read_located(
        &self,
        mut loc: RecordLoc,
        moved: impl Fn(RecordLoc) -> Option<RecordLoc>,
    ) -> PersistenceResult<Vec<u8>> {
        for _ in 0..SEALED_READ_ATTEMPTS {
            let read_span = profile::span("record:sealed_read");
            let read = self.sealed.read_record_payload(&loc);
            read_span.end();
            let Some(read) = read else {
                break;
            };
            match read {
                Ok(bytes) => return Ok(bytes),
                Err(_) => match moved(loc) {
                    Some(now) => loc = now,
                    None => break,
                },
            }
        }
        let mut attempt = 0;
        loop {
            let read_span = profile::span("record:owner_read");
            let read = self.persistence.call(|p| p.read_record_payload(&loc));
            read_span.end();
            match read {
                Ok(bytes) => return Ok(bytes),
                Err(e) => {
                    attempt += 1;
                    match moved(loc) {
                        Some(now) if attempt < SEALED_READ_ATTEMPTS => loc = now,
                        _ => return Err(e),
                    }
                }
            }
        }
    }

    /// A turn's QSA index page — held, or read from the log ([`IndexPage`]).
    /// `None` when the turn has none; an error when one is indexed and cannot
    /// be read, since a borrowed turn's K/V without its index rows is a slot
    /// whose tokens no cache accounts for.
    pub fn turn_index_page(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Result<Option<Arc<StoredPages>>, ConversationError> {
        self.read_index_page(turn_stream_id(timeline.raw(), index.0))
    }

    /// [`Self::turn_index_page`] for any stream — a prompt section's by its
    /// content-addressed id.
    ///
    /// **Read the way a snapshot is** ([`Self::read_located`]): without the
    /// log's owner where the record is durable, and read again at the index's
    /// new location whenever a read fails because compaction moved it. The
    /// substrate goes on naming only where the page is; the bytes
    /// are the caller's for as long as it keeps the `Arc`, and while any caller
    /// does, a read of the same record finds them without the log
    /// ([`HeldIndexPages`]).
    pub fn read_index_page(
        &self,
        stream_id: StreamId,
    ) -> Result<Option<Arc<StoredPages>>, ConversationError> {
        let (loc, id) = match self.read().index_page_record_of(stream_id) {
            None => return Ok(None),
            Some((IndexPage::Resident(pages), _)) => return Ok(Some(pages)),
            Some((IndexPage::Stored(loc), id)) => (loc, id),
        };
        if let Some(page) = self.held_index_pages.get(stream_id, id) {
            return Ok(Some(page));
        }
        let page = Arc::new(StoredPages::new(
            self.read_stored_index_page(stream_id, loc)?,
        ));
        // Held under the record it was read for only if the substrate still
        // names that record: the read follows the page wherever it moved, and a
        // newer page that took the stream meanwhile is not the record `id` names.
        let still = self
            .read()
            .index_page_record_of(stream_id)
            .is_some_and(|(_, now)| now == id);
        if still {
            self.held_index_pages.hold(stream_id, id, &page);
        }
        Ok(Some(page))
    }

    /// [`Self::turn_index_page`] without the log: the page while the substrate
    /// still holds its bytes — sealed and not yet written — or while a caller
    /// holds the bytes a read produced ([`HeldIndexPages`]). `None` for a page
    /// only the log has, which this never reads.
    pub fn turn_index_page_in_memory(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Arc<StoredPages>> {
        let stream_id = turn_stream_id(timeline.raw(), index.0);
        match self.read().index_page_record_of(stream_id)? {
            (IndexPage::Resident(pages), _) => Some(pages),
            (IndexPage::Stored(_), id) => self.held_index_pages.get(stream_id, id),
        }
    }

    /// The bytes of the index page `stream_id` keeps in the log, read first at
    /// `loc` and again wherever compaction moved it.
    pub(super) fn read_stored_index_page(
        &self,
        stream_id: StreamId,
        loc: RecordLoc,
    ) -> Result<Vec<u8>, ConversationError> {
        self.read_located(loc, |stale| {
            self.read()
                .index_page_of(stream_id)
                .and_then(|p| p.stored())
                .filter(|now| *now != stale)
        })
        .map_err(|e| {
            ConversationError::Channel(format!(
                "QSA index page of stream {} is indexed but unreadable: {e}",
                stream_id.0
            ))
        })
    }
}
