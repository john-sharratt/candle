//! Sealing a character's bot token.
//!
//! A bot token is the bot: whoever holds it speaks as the character on
//! Discord. It is kept only sealed, with ChaCha20-Poly1305 under a key every
//! member derives from the cluster's group key
//! (`host::cluster::Join::derive_key`, for [`PURPOSE`]), so any member can
//! open it when it takes a world over and nothing that reads the record
//! without the key learns the token. A daemon with no cluster is its own only
//! member and mints a key at start, as [`crate::image_links`] does, so a token
//! it sealed cannot be opened after it restarts and is connected again.
//!
//! A sealed token is `base64(nonce(12) || ciphertext || tag(16))`, the nonce
//! fresh from the OS for every seal. The record's key, `discord/bot/<npc_id>`,
//! is the associated data, so a sealed token copied onto another character's
//! record does not open there.

use std::fmt::{self, Debug, Formatter};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use rand::rngs::OsRng;
use rand::TryRngCore;

use crate::discord::store;

/// What the sealing key is derived for, from the group key.
pub const PURPOSE: &str = "npcd discord tokens";

/// The nonce's length, in bytes: ChaCha20-Poly1305's 96 bits.
const NONCE: usize = 12;

/// The tag's length, in bytes.
const TAG: usize = 16;

/// Seals bot tokens, and opens them.
#[derive(Clone)]
pub struct Sealer {
    key: [u8; 32],
}

impl Debug for Sealer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Sealer(…)")
    }
}

/// Why a sealed token did not open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsealable {
    /// It is not base64.
    NotBase64,
    /// It is shorter than a nonce and a tag.
    TooShort,
    /// It was not sealed under this key for this record, or was changed since.
    Forged,
    /// It opened, and what it held is not text.
    NotText,
}

impl Sealer {
    /// Tokens sealed under `key`.
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Tokens sealed under a key of this process's own, from the OS — a
    /// daemon with no cluster to share one with.
    pub fn minted() -> Self {
        let mut key = [0u8; 32];
        OsRng
            .try_fill_bytes(&mut key)
            .expect("the OS must provide entropy");
        Self { key }
    }

    /// `token`, sealed for character `npc_id`'s record under a fresh nonce.
    pub fn seal(&self, npc_id: u64, token: &str) -> String {
        let mut nonce = [0u8; NONCE];
        OsRng
            .try_fill_bytes(&mut nonce)
            .expect("the OS must provide entropy");
        self.seal_with(nonce, store::key(npc_id).as_bytes(), token.as_bytes())
    }

    /// The token sealed for character `npc_id`'s record in `sealed`.
    pub fn open(&self, npc_id: u64, sealed: &str) -> Result<String, Unsealable> {
        let bytes = self.open_with(store::key(npc_id).as_bytes(), sealed)?;
        String::from_utf8(bytes).map_err(|_| Unsealable::NotText)
    }

    /// `plaintext` sealed under `nonce` with `aad` as associated data, as
    /// `base64(nonce || ciphertext || tag)`. [`Self::seal`] with the nonce
    /// named, so the bytes it writes can be checked exactly.
    pub fn seal_with(&self, nonce: [u8; NONCE], aad: &[u8], plaintext: &[u8]) -> String {
        let sealed = self
            .cipher()
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .expect("ChaCha20-Poly1305 seals any message shorter than 256 GiB");
        let mut out = Vec::with_capacity(NONCE + sealed.len());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&sealed);
        STANDARD.encode(out)
    }

    /// The plaintext of `sealed`, checked against `aad`.
    fn open_with(&self, aad: &[u8], sealed: &str) -> Result<Vec<u8>, Unsealable> {
        let bytes = STANDARD.decode(sealed).map_err(|_| Unsealable::NotBase64)?;
        if bytes.len() < NONCE + TAG {
            return Err(Unsealable::TooShort);
        }
        let (nonce, body) = bytes.split_at(NONCE);
        self.cipher()
            .decrypt(Nonce::from_slice(nonce), Payload { msg: body, aad })
            .map_err(|_| Unsealable::Forged)
    }

    fn cipher(&self) -> ChaCha20Poly1305 {
        ChaCha20Poly1305::new(Key::from_slice(&self.key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 8439 §2.8.2's key: `80 81 … 9f`.
    fn rfc_key() -> [u8; 32] {
        std::array::from_fn(|i| 0x80 + i as u8)
    }

    /// RFC 8439 §2.8.2's nonce: the fixed part `07 00 00 00`, then the IV.
    const RFC_NONCE: [u8; 12] = [
        0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    ];

    const RFC_AAD: [u8; 12] = [
        0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    ];

    const RFC_PLAINTEXT: &str = "Ladies and Gentlemen of the class of '99: If I could offer you \
                                 only one tip for the future, sunscreen would be it.";

    /// RFC 8439 §2.8.2's ciphertext, then its tag.
    const RFC_SEALED: [u8; 130] = [
        0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb, 0x7b, 0x86, 0xaf, 0xbc, 0x53, 0xef, 0x7e,
        0xc2, 0xa4, 0xad, 0xed, 0x51, 0x29, 0x6e, 0x08, 0xfe, 0xa9, 0xe2, 0xb5, 0xa7, 0x36, 0xee,
        0x62, 0xd6, 0x3d, 0xbe, 0xa4, 0x5e, 0x8c, 0xa9, 0x67, 0x12, 0x82, 0xfa, 0xfb, 0x69, 0xda,
        0x92, 0x72, 0x8b, 0x1a, 0x71, 0xde, 0x0a, 0x9e, 0x06, 0x0b, 0x29, 0x05, 0xd6, 0xa5, 0xb6,
        0x7e, 0xcd, 0x3b, 0x36, 0x92, 0xdd, 0xbd, 0x7f, 0x2d, 0x77, 0x8b, 0x8c, 0x98, 0x03, 0xae,
        0xe3, 0x28, 0x09, 0x1b, 0x58, 0xfa, 0xb3, 0x24, 0xe4, 0xfa, 0xd6, 0x75, 0x94, 0x55, 0x85,
        0x80, 0x8b, 0x48, 0x31, 0xd7, 0xbc, 0x3f, 0xf4, 0xde, 0xf0, 0x8e, 0x4b, 0x7a, 0x9d, 0xe5,
        0x76, 0xd2, 0x65, 0x86, 0xce, 0xc6, 0x4b, 0x61, 0x16, // the tag
        0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06,
        0x91,
    ];

    /// **A seal is RFC 8439's AEAD, byte for byte**, laid out as the nonce,
    /// then the ciphertext and tag.
    #[test]
    fn a_seal_is_the_rfc_8439_test_vector_after_its_nonce() {
        let sealed =
            Sealer::new(rfc_key()).seal_with(RFC_NONCE, &RFC_AAD, RFC_PLAINTEXT.as_bytes());
        let mut expected = RFC_NONCE.to_vec();
        expected.extend_from_slice(&RFC_SEALED);
        assert_eq!(STANDARD.decode(&sealed).unwrap(), expected);
    }

    /// **The text is standard, padded base64** of those bytes: the nonce's
    /// twelve bytes are exactly sixteen characters, the ciphertext begins on
    /// the next, and the last byte of the tag (`0x91`) pads the end.
    #[test]
    fn a_seal_is_written_as_padded_standard_base64() {
        let sealed =
            Sealer::new(rfc_key()).seal_with(RFC_NONCE, &RFC_AAD, RFC_PLAINTEXT.as_bytes());
        assert_eq!(&sealed[..16], "BwAAAEBBQkNERUZH");
        assert_eq!(&sealed[16..20], "0xqN");
        assert_eq!(&sealed[sealed.len() - 4..], "kQ==");
        assert_eq!(sealed.len(), 192, "142 bytes is 192 base64 characters");
    }

    #[test]
    fn a_sealed_token_opens_for_its_own_character() {
        let s = Sealer::new([7; 32]);
        let sealed = s.seal(42, "MTIz.abc.def");
        assert_eq!(s.open(42, &sealed), Ok("MTIz.abc.def".to_string()));
        // A fresh nonce each time, so the same token seals differently.
        assert_ne!(s.seal(42, "MTIz.abc.def"), sealed);
        assert!(!sealed.contains("MTIz"), "{sealed}");
    }

    /// **A changed byte anywhere, another key, or another character's record
    /// does not open.**
    #[test]
    fn a_tampered_or_misplaced_or_foreign_seal_does_not_open() {
        let s = Sealer::new([7; 32]);
        let sealed = s.seal(42, "token");
        let bytes = STANDARD.decode(&sealed).unwrap();
        for at in [0, NONCE, bytes.len() - 1] {
            let mut changed = bytes.clone();
            changed[at] ^= 1;
            assert_eq!(
                s.open(42, &STANDARD.encode(&changed)),
                Err(Unsealable::Forged),
                "byte {at}"
            );
        }
        assert_eq!(
            Sealer::new([8; 32]).open(42, &sealed),
            Err(Unsealable::Forged)
        );
        assert_eq!(s.open(43, &sealed), Err(Unsealable::Forged));
        assert_eq!(s.open(42, "not base64!"), Err(Unsealable::NotBase64));
        assert_eq!(
            s.open(42, &STANDARD.encode([0u8; 27])),
            Err(Unsealable::TooShort)
        );
    }

    #[test]
    fn a_minted_key_is_the_processes_own() {
        let a = Sealer::minted();
        let sealed = a.seal(1, "token");
        assert_eq!(a.open(1, &sealed), Ok("token".to_string()));
        assert_eq!(Sealer::minted().open(1, &sealed), Err(Unsealable::Forged));
    }
}
