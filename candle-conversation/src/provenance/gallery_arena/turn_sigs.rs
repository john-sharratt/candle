//! A turn's signatures as the arena receives them: already decoded, or a stored
//! window it decodes only when the turn has to be uploaded.

use super::super::{SigWindow, WideQSig};

/// The whole window of one turn, handed to the arena for residency.
///
/// A resident turn whose fingerprint still matches is a hit and reads nothing
/// from here but its length, so a scan over an [`Encoded`](Self::Encoded)
/// gallery that is already on the device decodes nothing at all.
#[derive(Clone, Copy, Debug)]
pub enum TurnSigs<'a> {
    /// Signatures the caller already holds decoded.
    Decoded(&'a [WideQSig]),
    /// A stored window, decoded for the length of an upload.
    Encoded(&'a SigWindow),
}

impl TurnSigs<'_> {
    /// Tokens in the turn.
    pub fn len(&self) -> usize {
        match self {
            Self::Decoded(s) => s.len(),
            Self::Encoded(w) => w.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether token `t` has `wpt` words — the width the arena's pages hold.
    /// A stored window's width is uniform; decoded signatures are checked one
    /// token at a time.
    pub fn token_has_width(&self, t: usize, wpt: usize) -> bool {
        match self {
            Self::Decoded(s) => s[t].words.len() == wpt,
            Self::Encoded(w) => w.width() == wpt,
        }
    }

    /// The width of the turn's first token, `None` for an empty turn.
    pub fn first_width(&self) -> Option<usize> {
        match self {
            Self::Decoded(s) => s.first().map(|s| s.words.len()),
            Self::Encoded(w) => (!w.is_empty()).then(|| w.width()),
        }
    }

    /// Run `f` over the turn's decoded signatures. A stored window is decoded
    /// for the call and dropped after it.
    pub fn with_decoded<R>(&self, f: impl FnOnce(&[WideQSig]) -> R) -> R {
        match self {
            Self::Decoded(s) => f(s),
            Self::Encoded(w) => f(&w.decode()),
        }
    }

    /// The turn's words, token-major, `width` to a token — a token of another
    /// width cut or zero-padded to it, so token `t` is always at `t × width`.
    pub fn words(&self, width: usize) -> Vec<u64> {
        match self {
            Self::Encoded(w) if w.width() == width => w.words(),
            _ => self.with_decoded(|sigs| {
                let mut out = vec![0u64; sigs.len() * width];
                for (t, s) in sigs.iter().enumerate() {
                    let n = s.words.len().min(width);
                    out[t * width..t * width + n].copy_from_slice(&s.words[..n]);
                }
                out
            }),
        }
    }
}

impl<'a> From<&'a [WideQSig]> for TurnSigs<'a> {
    fn from(sigs: &'a [WideQSig]) -> Self {
        Self::Decoded(sigs)
    }
}

impl<'a> From<&'a Vec<WideQSig>> for TurnSigs<'a> {
    fn from(sigs: &'a Vec<WideQSig>) -> Self {
        Self::Decoded(sigs)
    }
}

impl<'a> From<&'a SigWindow> for TurnSigs<'a> {
    fn from(window: &'a SigWindow) -> Self {
        Self::Encoded(window)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::provenance::encode_wide_sigs;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// Both forms of one turn report the same length and width and hand the
    /// same signatures to an upload.
    #[test]
    fn decoded_and_encoded_forms_of_a_turn_agree() {
        let sigs = vec![sig(1), sig(2), sig(3)];
        let window = SigWindow::new(Arc::from(encode_wide_sigs(&sigs))).unwrap();
        let decoded = TurnSigs::from(&sigs);
        let encoded = TurnSigs::from(&window);
        for form in [decoded, encoded] {
            assert_eq!(form.len(), 3);
            assert_eq!(form.first_width(), Some(24));
            assert!(form.token_has_width(2, 24));
            assert!(!form.token_has_width(2, 12));
            assert_eq!(form.with_decoded(<[WideQSig]>::to_vec), sigs);
            let words = form.words(24);
            assert_eq!(words.len(), 72);
            assert_eq!((words[0], words[24], words[71]), (1, 2, 3));
        }
    }

    /// Words at another width are cut or zero-padded, one stride a token.
    #[test]
    fn words_hold_every_token_at_the_asked_width() {
        let mut sigs = vec![sig(5), sig(6)];
        sigs[1].words.truncate(2);
        let words = TurnSigs::from(&sigs).words(4);
        assert_eq!(words, vec![5, 5, 5, 5, 6, 6, 0, 0]);
    }

    /// A decoded turn is checked per token: one short token is refused alone.
    #[test]
    fn a_decoded_turn_checks_each_tokens_width() {
        let mut sigs = vec![sig(1), sig(2)];
        sigs[1].words.truncate(12);
        let form = TurnSigs::from(&sigs);
        assert!(form.token_has_width(0, 24));
        assert!(!form.token_has_width(1, 24));
    }
}
