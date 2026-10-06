//! The identity of an injected piece's positional page.
//!
//! A model that carries positional state (Qwen3.8-Flash-Next's QSA index) gets
//! that state for an injected piece from the page the piece sealed, and places
//! it once however many slots inject the same piece — so it needs the page's
//! identity, and the page's bytes are the only thing that identifies it.
//!
//! The digest is taken by whoever holds the page, once, when the page enters
//! memory: a projection hands the same pages over on every rebuild, and a
//! turn's page runs to a megabyte and more, so hashing it per hand-over was a
//! third of what re-injecting a turn cost.

/// A page's identity: the BLAKE3 digest of its bytes. The page's rows are a pure
/// function of those bytes — rows, widths, layer order — and where a page sits
/// is recorded by the cache that holds it, not by the page, so pages with equal
/// bytes are interchangeable in every slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PieceKey([u8; 32]);

impl PieceKey {
    pub fn of(page: &[u8]) -> Self {
        Self(*blake3::hash(page).as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::PieceKey;

    #[test]
    fn a_key_is_the_digest_of_the_bytes() {
        assert_eq!(
            PieceKey::of(b"a sealed piece"),
            PieceKey::of(b"a sealed piece")
        );
        assert_ne!(
            PieceKey::of(b"a sealed piece"),
            PieceKey::of(b"another piece")
        );
        assert_eq!(PieceKey::of(b""), PieceKey(*blake3::hash(b"").as_bytes()));
    }
}
