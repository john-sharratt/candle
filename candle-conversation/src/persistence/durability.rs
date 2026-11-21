//! Makes appended records durable: staging, write-through, group commit and segment sealing, with the commit as the one durability point.
//!
//! Records are staged and written through once the buffer reaches the write-through bound, but nothing is durable until a commit syncs the active segment. A read by location flushes staged records first, so a record appended a moment ago is found. Rotation happens at a commit boundary, or on the append path once the active segment and its staged bytes reach the target, so one large batch cannot overshoot the target by its whole length. A released writer commits what it staged, refuses every later write, and gives up the directory's lock.

use super::*;

impl SubstratePersistence {
    /// The logical end of what is written to the active segment — synced only
    /// once committed.
    pub fn write_offset(&self) -> u64 {
        self.segments.write_offset()
    }

    /// The active (append target) segment's id — the segment fresh appends
    /// land in.
    pub fn active_segment(&self) -> SegmentId {
        self.segments.active_id()
    }

    /// Commit what is staged, then stop writing and give up the directory's
    /// [`WriterLock`], so the next writer can open it while this handle is
    /// still alive. From here on the handle behaves as a read-only one: every
    /// write refuses, and reads go on. Idempotent; a read-only handle has
    /// nothing to release.
    ///
    /// An engine's `shutdown` ends here. The engine can outlive it — another
    /// thread still holding it — and its successor opens the same workspace in
    /// the meantime; nothing the shut-down engine does may reach the log the
    /// successor writes.
    pub fn release_writer(&mut self) -> Result<()> {
        if self.is_read_only() {
            return Ok(());
        }
        self.commit()?;
        self.segments.close_for_writing();
        self.writer_lock = None;
        Ok(())
    }

    /// Write what is staged to its segment, so a read by location finds it —
    /// a record appended a moment ago is in the staging buffer, not the file.
    /// Not a commit: nothing is made durable.
    pub(super) fn flush_staged(&mut self) -> Result<()> {
        if self.segments.pending_len() > 0 {
            self.segments.flush()?;
        }
        Ok(())
    }

    /// Flush and `fsync` the active segment — the group-commit durability
    /// point. Seals + rotates the active if it has reached the size target.
    /// A no-op on a read-only handle, which never has anything staged.
    pub fn commit(&mut self) -> Result<()> {
        if self.is_read_only() {
            return Ok(());
        }
        self.write_replica_if_changed()?;
        let committed = self.segments.commit();
        self.watch_space(committed)?;
        self.maybe_rotate_active()
    }

    /// Bytes staged but not yet flushed to the active segment. Returns 0 when
    /// there is nothing staged — which, once a large append has flushed part
    /// way, is not the same as nothing to commit ([`Self::commit_if_pending`]).
    pub fn pending_bytes(&self) -> usize {
        self.segments.pending_len()
    }

    /// Write the staged records through to the active segment once they reach
    /// [`STAGE_FLUSH_BYTES`], leaving the sync to the next commit.
    pub(super) fn write_through_if_over_stage(&mut self) -> Result<()> {
        if self.segments.pending_len() >= STAGE_FLUSH_BYTES {
            let flushed = self.segments.flush();
            self.watch_space(flushed)?;
        }
        Ok(())
    }

    /// Group-commit if (and only if) there is something to make durable —
    /// records staged, or written through by a large append and not yet
    /// synced. Returns `Ok(true)` when a flush+fsync actually happened,
    /// `Ok(false)` for the no-op idle path. Cheap to call on a tight timer.
    pub fn commit_if_pending(&mut self) -> Result<bool> {
        if self.is_read_only() || (!self.segments.needs_commit() && !self.replica_dirty) {
            return Ok(false);
        }
        self.write_replica_if_changed()?;
        let committed = self.segments.commit();
        self.watch_space(committed)?;
        self.maybe_rotate_active()?;
        Ok(true)
    }

    /// Seal the active segment and mint a fresh one when it has reached the
    /// size target (§12). Runs only at a commit boundary — the active is
    /// durable and has no staged records — so a segment is never split
    /// mid-commit. Completes the sealed segment's `HeaderIndex` chain first
    /// (fast recovery), then resets the chain state for the fresh active.
    pub(super) fn maybe_rotate_active(&mut self) -> Result<()> {
        if !self.segments.should_rotate() {
            return Ok(());
        }
        self.seal_active()
    }

    /// Seal + rotate the active segment when its durable **plus staged** bytes
    /// have reached the size target — a byte-based check on the append path, not
    /// only at external commit boundaries. A long append batch (migration
    /// relocation, large-file ingest) stages many records between commits, so
    /// `should_rotate` alone — which fires only after a group-commit — lets the
    /// active overshoot the target by the whole batch (observed: ~8 GB segments
    /// against the 4 GB target). Checking `write_offset + pending_len` on every
    /// data append bounds each segment to ~target + one record and, as a side
    /// benefit, caps the in-memory staging buffer at ~target. [`Self::seal_active`]
    /// flushes the index chain and commits the staged records durable first, so
    /// the sealed segment is well-formed and the batch continues in a fresh
    /// active. Cheap on the common path: two field reads; the seal fires at most
    /// once per segment fill. Never called for `HeaderIndex` appends (the seal
    /// path emits those), so it can't recurse.
    pub(super) fn rotate_if_over_target(&mut self) -> Result<()> {
        let projected = self.segments.write_offset() + self.segments.pending_len() as u64;
        if projected >= self.segments.target_bytes() {
            self.seal_active()?;
        }
        Ok(())
    }

    /// Seal the active segment and mint a fresh one, unconditionally. Completes
    /// the sealing segment's `HeaderIndex` chain (so its records recover via the
    /// chain fast path, not a full walk), rotates, and resets the chain state
    /// for the fresh active. The caller must have committed first; this flushes
    /// the index but assumes no un-flushed data records.
    ///
    /// A no-op on a read-only handle: the segment set stays exactly as the
    /// process that owns it left it.
    pub fn seal_active(&mut self) -> Result<()> {
        if self.is_read_only() {
            return Ok(());
        }
        self.flush_header_index()?;
        // A write-through may have left bytes on the segment that no commit
        // has synced yet; a sealed segment's handle is dropped, so sync now.
        if self.segments.needs_commit() {
            self.segments.commit()?;
        }
        self.segments.seal_and_rotate()?;
        // The fresh active starts its own `HeaderIndex` chain — the sealed
        // segment keeps the chain the flush above completed.
        self.last_index = None;
        self.pending_index.clear();
        Ok(())
    }
}
