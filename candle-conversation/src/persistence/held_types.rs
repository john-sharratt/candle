//! The record types a substrate holds (`docs/replicated_substrate.md` §3.9): a
//! build that replaces this one must decode every one of them, or the walker
//! would skip them and the next compaction would delete them.

use super::SubstratePersistence;

impl SubstratePersistence {
    /// Every record type the substrate has held since it opened — the open
    /// walk's, and each appended since — by tag, ascending.
    pub fn held_record_tags(&self) -> Vec<u8> {
        (0..32u8)
            .filter(|t| self.held_types & (1 << t) != 0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::super::record::RecordType;
    use super::SubstratePersistence;
    use crate::substrate::Substrate;

    fn open(dir: &TempDir) -> SubstratePersistence {
        SubstratePersistence::open_in_with_substrate(dir.path(), &mut Substrate::new()).unwrap()
    }

    /// A type appended is held at once, and still held — by the census — once
    /// the substrate is opened again.
    #[test]
    fn a_type_written_is_held_and_survives_a_reopen() {
        let dir = TempDir::new().unwrap();
        let mut sp = open(&dir);
        let debug_id = RecordType::DebugId.tag();
        assert!(!sp.held_record_tags().contains(&debug_id));
        let id = sp.mint();
        let payload = br#"{"timeline_id":4096,"debug_id":"x"}"#;
        sp.append_carried(RecordType::DebugId, 0, 0, 0, 0, 0, payload, id)
            .unwrap();
        assert!(sp.held_record_tags().contains(&debug_id));
        sp.commit().unwrap();
        drop(sp);
        assert!(open(&dir).held_record_tags().contains(&debug_id));
    }
}
