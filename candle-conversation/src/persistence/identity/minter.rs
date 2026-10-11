//! The one place a substrate's record identities are minted.
//!
//! The persistence layer mints when it appends a record it originates, and the
//! runtime write paths mint when they change the substrate in memory, before
//! the record that carries the change reaches the writer thread. Both draw on
//! one [`Origination`], so every identity this node issues is unique, its
//! `oseq`s never repeat, and each in-memory value holds, from the moment it is
//! set, the stamp its record will carry on disk.
//!
//! The minter also knows the **generation** of each section this node is
//! writing (`docs/replicated_substrate.md` §6.5): every record of a section
//! carries, in its `epoch`, the `oseq` of the declaration that started the
//! section's current generation, so a tombstone can end one generation and
//! leave the next standing whatever order their records arrive in.
//!
//! Minting takes a lock of its own and nothing else, so it is safe to call
//! with the substrate lock or the persistence lock held.

use std::sync::{Arc, Mutex, MutexGuard};

use super::super::record::RecordType;
use super::origination::{now_ms, Origination};
use super::record_identity::RecordIdentity;
use super::section_generations::SectionGenerations;

/// What a minter guards: the identity state and the live section generations.
#[derive(Clone, Debug)]
struct Minting {
    origination: Origination,
    sections: SectionGenerations,
}

/// A shared handle on a substrate's [`Origination`].
#[derive(Clone, Debug)]
pub struct Minter(Arc<Mutex<Minting>>);

/// Whether a record of type `rt` belongs to the generation of the stream it is
/// written under: a stream's own records, other than the declaration that
/// starts the generation.
fn generational(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::Chunk
            | RecordType::Tokens
            | RecordType::Commit
            | RecordType::TurnIndexPage
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
    )
}

impl Minter {
    pub fn new(origination: Origination) -> Minter {
        Minter::with_sections(origination, SectionGenerations::default())
    }

    /// A minter that goes on writing the section generations `sections`.
    pub fn with_sections(origination: Origination, sections: SectionGenerations) -> Minter {
        Minter(Arc::new(Mutex::new(Minting {
            origination,
            sections,
        })))
    }

    fn lock(&self) -> MutexGuard<'_, Minting> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The identity of the next record this node originates, under claim or
    /// lease `epoch` (zero for none), stamped with the wall clock now.
    pub fn mint(&self, epoch: u64) -> RecordIdentity {
        self.lock().origination.mint(now_ms(), epoch)
    }

    /// The identity of the next record of type `rt` this node writes under
    /// stream `stream_id`: a record of a section this node is writing carries
    /// the section's generation in its `epoch`, every other record none.
    pub fn mint_for(&self, rt: RecordType, stream_id: u64) -> RecordIdentity {
        let mut m = self.lock();
        let epoch = if generational(rt) {
            m.sections.generation(stream_id).unwrap_or(0)
        } else {
            0
        };
        m.origination.mint(now_ms(), epoch)
    }

    /// This node started a generation of section `stream_id` with the
    /// declaration it wrote as `oseq`.
    pub fn start_section(&self, stream_id: u64, oseq: u64) {
        self.lock().sections.start(stream_id, oseq);
    }

    /// The generation of section `stream_id` that this node's declaration
    /// `oseq` started has ended.
    pub fn end_section(&self, stream_id: u64, oseq: u64) {
        self.lock().sections.end(stream_id, oseq);
    }

    /// The generation of section `stream_id` this node is writing, if any.
    pub fn section_generation(&self, stream_id: u64) -> Option<u64> {
        self.lock().sections.generation(stream_id)
    }

    /// The member is cut off: what this node mints from here on is detached
    /// ([`Origination::detach`]).
    pub fn detach(&self, from: Option<u64>) {
        self.lock().origination.detach(from);
    }

    /// The member rejoined under generation `gen` ([`Origination::attach`]).
    pub fn attach(&self, gen: u64) {
        self.lock().origination.attach(gen);
    }

    /// Take a record this substrate holds into account (see
    /// [`Origination::observe`]).
    pub fn observe(&self, id: &RecordIdentity) {
        self.lock().origination.observe(id);
    }

    /// The writer's identity state as it stands.
    pub fn origination(&self) -> Origination {
        self.lock().origination.clone()
    }

    /// This node.
    pub fn node(&self) -> u64 {
        self.lock().origination.node()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two handles on one minter draw on one sequence and one clock.
    #[test]
    fn clones_share_one_sequence() {
        let a = Minter::new(Origination::new(9));
        let b = a.clone();
        let first = a.mint(0);
        let second = b.mint(0);
        assert_eq!((first.oseq, second.oseq), (1, 2));
        assert!(second.stamp() > first.stamp());
        assert_eq!(a.origination().next_oseq(), 3);
    }

    /// A record observed through one handle bounds what the other mints.
    #[test]
    fn an_observation_bounds_every_handle() {
        let a = Minter::new(Origination::new(9));
        let b = a.clone();
        let far = RecordIdentity {
            origin: 4,
            oseq: 1,
            gen: 2,
            epoch: 0,
            hlc: u64::MAX >> 1,
        };
        a.observe(&far);
        let next = b.mint(0);
        assert_eq!(next.gen, 2);
        assert!(next.hlc > far.hlc);
    }

    /// A section's own records carry its generation; its declaration, other
    /// streams' records and other record types carry none; an ended generation
    /// stops being stamped.
    #[test]
    fn a_sections_records_carry_its_generation() {
        let m = Minter::new(Origination::new(9));
        let decl = m.mint_for(RecordType::StreamDecl, 40);
        assert_eq!(decl.epoch, 0);
        m.start_section(40, decl.oseq);
        assert_eq!(m.mint_for(RecordType::Chunk, 40).epoch, decl.oseq);
        assert_eq!(m.mint_for(RecordType::Commit, 40).epoch, decl.oseq);
        assert_eq!(m.mint_for(RecordType::StreamDecl, 40).epoch, 0);
        assert_eq!(m.mint_for(RecordType::Chunk, 41).epoch, 0);
        assert_eq!(m.mint_for(RecordType::Label, 40).epoch, 0);
        m.end_section(40, decl.oseq);
        assert_eq!(m.mint_for(RecordType::Chunk, 40).epoch, 0);
    }
}
