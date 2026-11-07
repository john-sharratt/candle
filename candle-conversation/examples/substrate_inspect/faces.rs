//! `retire-faces`: the maintenance job that undoes the first face backfill
//! (`docs/zend_working_set.md` §4.9), run against a stopped daemon's substrate.
//!
//! That backfill sealed every ingest unit's face with a whole recurrent
//! snapshot, 65–112 MB for a turn of a few dozen tokens, and ~3,400 of them
//! filled WhiteWorld's substrate drive. This tombstones every face — the face
//! and every record of it, its snapshot included, become dead — then runs the
//! log's own maintenance until no op is left, so the dead segments are
//! compacted and dropped. A unit whose face is gone holds its own opening
//! until the next face pass seals it a face again, with no snapshot.
//!
//! It writes: a tombstone per face, and whatever maintenance relocates. So the
//! daemon must be stopped, and the volume needs room for a segment's live
//! records — maintenance appends them to the active segment before it unlinks
//! the source.

use std::path::Path;

use anyhow::{Context, Result};
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::TimelineId;
use candle_conversation::substrate::Substrate;

/// Why each face is tombstoned, recorded in its tombstone.
const REASON: &str = "a face that kept a recurrent snapshot";

/// Tombstone every face in the substrate under `workspace` (the directory
/// holding `substrate/`), then run maintenance until it has nothing to do.
pub fn retire_faces(workspace: &Path) -> Result<()> {
    let mut substrate = Substrate::new();
    let mut persistence =
        SubstratePersistence::open_in_with_substrate(workspace, &mut substrate)
            .with_context(|| format!("opening the substrate under {}", workspace.display()))?;
    let faces: Vec<TimelineId> = substrate
        .timelines_with_metadata_key(Substrate::FACE_OF_KEY)
        .into_iter()
        .map(|(timeline, _)| timeline)
        .collect();
    println!("faces: {}", faces.len());
    for face in &faces {
        persistence
            .write_tombstone(face.raw(), Some(REASON))
            .with_context(|| format!("tombstoning face {}", face.raw()))?;
        substrate.tombstone_timeline(*face);
    }
    persistence.commit().context("committing the tombstones")?;
    println!("tombstoned: {}", faces.len());
    let mut ops = 0usize;
    while let Some(op) = persistence
        .run_maintenance(&mut substrate)
        .context("maintenance")?
    {
        ops += 1;
        println!("maintenance {ops}: {op:?}");
    }
    persistence.commit().context("committing maintenance")?;
    println!("maintenance ops: {ops}");
    Ok(())
}
