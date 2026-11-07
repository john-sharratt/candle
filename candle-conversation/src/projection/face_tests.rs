//! A unit's face lives and dies with it (`Substrate::face_of`): tombstoning
//! the unit tombstones the face, in memory and on disk.

use tempfile::TempDir;

use super::{Conversation, GroupId, LayerId, TimelineId};
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

fn open(dir: &TempDir) -> Conversation {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    Conversation::from_parts(substrate, sp)
}

/// A unit and a face, each naming the other.
fn unit_and_face(conv: &Conversation) -> (TimelineId, TimelineId) {
    let unit = conv.mint_timeline(LayerId::new(1), GroupId::new(1));
    let face = conv.mint_timeline(LayerId::new(2), GroupId::new(2));
    conv.set_conversation_metadata(unit, Substrate::FACE_KEY, &face.raw().to_string())
        .unwrap();
    conv.set_conversation_metadata(face, Substrate::FACE_OF_KEY, &unit.raw().to_string())
        .unwrap();
    (unit, face)
}

/// **Tombstoning a unit tombstones its face**, and a reload still holds both
/// dead.
#[test]
fn a_tombstoned_unit_takes_its_face_with_it() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let (unit, face) = unit_and_face(&conv);
    conv.tombstone_timeline(unit).unwrap();
    assert!(conv.read().is_tombstoned(unit));
    assert!(conv.read().is_tombstoned(face));
    conv.wait_written(conv.write_ticket());
    drop(conv);

    let again = open(&dir);
    assert!(again.read().is_tombstoned(unit));
    assert!(again.read().is_tombstoned(face));
}

/// **Tombstoning a face leaves its unit**, whose own opening then stands.
#[test]
fn a_tombstoned_face_leaves_its_unit() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let (unit, face) = unit_and_face(&conv);
    conv.tombstone_timeline(face).unwrap();
    assert!(conv.read().is_tombstoned(face));
    assert!(!conv.read().is_tombstoned(unit));
}
