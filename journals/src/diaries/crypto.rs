//! A diary's file, sealed and opened (`docs/private_mode.md` §5.3).
//!
//! - **The passphrase** is stretched with the repository's salt by Argon2id
//!   ([`KdfParams::PRODUCTION`]) into a 32-byte master key, from which
//!   HKDF-SHA256 derives the **kek**. Both are zeroized when dropped.
//! - **Each diary** has its own random 32-byte **DEK**, wrapped under the kek.
//!   Whoever's kek unwraps it holds the diary: a trial unwrap of every file is
//!   how a passphrase finds its diaries, and nothing else says whose any is.
//! - **The file**, padded to a power of two from 4 KiB so its size says little:
//!
//! ```text
//! version (1) || wrap nonce (24) || wrapped DEK (32 + 16)
//!             || nonce (24) || ciphertext (padded plaintext + 16)
//! padded plaintext = length (4, BE) || plaintext || zeros
//! ```
//!
//! Both seals are XChaCha20-Poly1305 with the version and the diary's id as
//! associated data, so a file copied under another diary's name opens
//! nowhere.

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use rand::rngs::OsRng;
use rand::TryRngCore;
use sha2::Sha256;
use zeroize::Zeroizing;

use super::layout::DiaryId;

/// The layout's version, the file's first byte.
pub const VERSION: u8 = 1;

const NONCE: usize = 24;
const TAG: usize = 16;
const KEY: usize = 32;
/// Where the wrapped DEK starts, and where the ciphertext's nonce does.
const WRAP_AT: usize = 1;
const SEALED_AT: usize = WRAP_AT + NONCE + KEY + TAG;
/// Everything in a file but the padded plaintext.
const OVERHEAD: usize = SEALED_AT + NONCE + TAG;
/// The smallest file.
pub const MIN_FILE: usize = 4096;

/// What HKDF derives the kek for.
const KEK_INFO: &[u8] = b"zen diary kek";

/// Argon2id's cost: memory in KiB, passes, lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdfParams {
    pub m_kib: u32,
    pub t: u32,
    pub p: u32,
}

impl KdfParams {
    /// What every passphrase is stretched with: 64 MiB, three passes, one
    /// lane — about a second on BlackWorld, and memory-hard.
    pub const PRODUCTION: KdfParams = KdfParams {
        m_kib: 64 * 1024,
        t: 3,
        p: 1,
    };
}

/// Why a diary could not be sealed or opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// The passphrase could not be stretched.
    Kdf(String),
    /// The file is not one of ours, or was changed: it does not open.
    Unopenable,
}

/// The key a passphrase's diaries are wrapped under.
pub struct Kek(Zeroizing<[u8; KEY]>);

/// One diary's own key.
pub struct Dek(Zeroizing<[u8; KEY]>);

fn random<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    OsRng
        .try_fill_bytes(&mut b)
        .expect("the OS must provide entropy");
    b
}

impl Kek {
    /// `passphrase` stretched with `salt` under `params`.
    pub fn derive(passphrase: &str, salt: &[u8], params: KdfParams) -> Result<Kek, CryptoError> {
        let argon = Argon2::new(
            Algorithm::Argon2id,
            Version::V0x13,
            Params::new(params.m_kib, params.t, params.p, Some(KEY))
                .map_err(|e| CryptoError::Kdf(e.to_string()))?,
        );
        let mut master = Zeroizing::new([0u8; KEY]);
        argon
            .hash_password_into(passphrase.as_bytes(), salt, &mut master[..])
            .map_err(|e| CryptoError::Kdf(e.to_string()))?;
        let mut kek = Zeroizing::new([0u8; KEY]);
        Hkdf::<Sha256>::new(None, &master[..])
            .expand(KEK_INFO, &mut kek[..])
            .expect("32 bytes is a length HKDF-SHA256 expands to");
        Ok(Kek(kek))
    }

    /// Whether `other` is this same key, compared in constant time.
    pub fn same(&self, other: &Kek) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    }
}

impl Dek {
    /// A new diary's key, from the OS.
    pub fn random() -> Dek {
        Dek(Zeroizing::new(random()))
    }
}

/// The associated data both seals are bound to.
fn aad(id: &DiaryId) -> [u8; 17] {
    let mut a = [0u8; 17];
    a[0] = VERSION;
    a[1..].copy_from_slice(&id.0);
    a
}

/// How long a file holding `plain` bytes is: a power of two, at least
/// [`MIN_FILE`].
pub fn file_len(plain: usize) -> usize {
    (plain + 4 + OVERHEAD).next_power_of_two().max(MIN_FILE)
}

fn wrap(kek: &Kek, dek: &Dek, id: &DiaryId) -> [u8; NONCE + KEY + TAG] {
    let nonce: [u8; NONCE] = random();
    let wrapped = XChaCha20Poly1305::new(Key::from_slice(&kek.0[..]))
        .encrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: &dek.0[..],
                aad: &aad(id),
            },
        )
        .expect("a key wraps");
    let mut out = [0u8; NONCE + KEY + TAG];
    out[..NONCE].copy_from_slice(&nonce);
    out[NONCE..].copy_from_slice(&wrapped);
    out
}

/// `plain`, diary `id`'s, sealed under `dek` wrapped under `kek`: the whole
/// file, under fresh nonces.
pub fn seal(kek: &Kek, dek: &Dek, id: &DiaryId, plain: &[u8]) -> Vec<u8> {
    let len = file_len(plain.len());
    let mut padded = Zeroizing::new(vec![0u8; len - OVERHEAD]);
    padded[..4].copy_from_slice(&(plain.len() as u32).to_be_bytes());
    padded[4..4 + plain.len()].copy_from_slice(plain);
    let nonce: [u8; NONCE] = random();
    let sealed = XChaCha20Poly1305::new(Key::from_slice(&dek.0[..]))
        .encrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: &padded[..],
                aad: &aad(id),
            },
        )
        .expect("a diary seals");
    let mut file = Vec::with_capacity(len);
    file.push(VERSION);
    file.extend_from_slice(&wrap(kek, dek, id));
    file.extend_from_slice(&nonce);
    file.extend_from_slice(&sealed);
    debug_assert_eq!(file.len(), len);
    file
}

/// Diary `id`'s key, when `kek` unwraps it from `file`: the trial every
/// passphrase makes of every diary.
pub fn unwrap(kek: &Kek, id: &DiaryId, file: &[u8]) -> Option<Dek> {
    if file.len() < OVERHEAD || file[0] != VERSION {
        return None;
    }
    let (nonce, wrapped) = file[WRAP_AT..SEALED_AT].split_at(NONCE);
    let dek = XChaCha20Poly1305::new(Key::from_slice(&kek.0[..]))
        .decrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: wrapped,
                aad: &aad(id),
            },
        )
        .ok()?;
    let mut key = Zeroizing::new([0u8; KEY]);
    key.copy_from_slice(&Zeroizing::new(dek));
    Some(Dek(key))
}

/// Diary `id`'s plaintext, from `file`, under its `dek`.
pub fn open(dek: &Dek, id: &DiaryId, file: &[u8]) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
    if file.len() < OVERHEAD || file[0] != VERSION {
        return Err(CryptoError::Unopenable);
    }
    let (nonce, sealed) = file[SEALED_AT..].split_at(NONCE);
    let padded = Zeroizing::new(
        XChaCha20Poly1305::new(Key::from_slice(&dek.0[..]))
            .decrypt(
                XNonce::from_slice(nonce),
                Payload {
                    msg: sealed,
                    aad: &aad(id),
                },
            )
            .map_err(|_| CryptoError::Unopenable)?,
    );
    let len = u32::from_be_bytes(padded[..4].try_into().expect("four bytes")) as usize;
    if 4 + len > padded.len() {
        return Err(CryptoError::Unopenable);
    }
    Ok(Zeroizing::new(padded[4..4 + len].to_vec()))
}

/// `file`, diary `id`'s, with its `dek` wrapped under `kek` instead: what a
/// new passphrase holds it by. The ciphertext is kept as it is.
pub fn rewrap(kek: &Kek, dek: &Dek, id: &DiaryId, file: &[u8]) -> Vec<u8> {
    let mut out = file.to_vec();
    out[WRAP_AT..SEALED_AT].copy_from_slice(&wrap(kek, dek, id));
    out
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// Cheap enough for tests; the production cost has its own test.
    const FAST: KdfParams = KdfParams {
        m_kib: 64,
        t: 1,
        p: 1,
    };

    fn kek(pass: &str) -> Kek {
        Kek::derive(pass, &[5; 16], FAST).unwrap()
    }

    #[test]
    fn a_passphrase_derives_one_kek_and_another_another() {
        assert!(kek("synthetic pass one").same(&kek("synthetic pass one")));
        assert!(!kek("synthetic pass one").same(&kek("synthetic pass two")));
        let other_salt = Kek::derive("synthetic pass one", &[6; 16], FAST).unwrap();
        assert!(!kek("synthetic pass one").same(&other_salt));
    }

    /// **A file is the layout's bytes**: its version, its sizes, and padded
    /// to a power of two from 4 KiB.
    #[test]
    fn a_file_is_laid_out_and_padded() {
        let id = DiaryId([1; 16]);
        let file = seal(&kek("p1"), &Dek::random(), &id, b"hello");
        assert_eq!(file.len(), 4096);
        assert_eq!(file[0], VERSION);
        assert_eq!(file_len(0), 4096);
        assert_eq!(file_len(4096 - 4 - OVERHEAD), 4096);
        assert_eq!(file_len(4096 - 4 - OVERHEAD + 1), 8192);
        assert_eq!(file_len(5 * 1024 * 1024), 8 * 1024 * 1024);
        assert_eq!(OVERHEAD, 1 + 24 + 48 + 24 + 16);
    }

    #[test]
    fn a_diary_opens_with_its_passphrase_alone() {
        let id = DiaryId([2; 16]);
        let dek = Dek::random();
        let file = seal(&kek("synthetic right"), &dek, &id, b"synthetic marker text");
        let got = unwrap(&kek("synthetic right"), &id, &file).expect("its kek unwraps it");
        assert_eq!(&open(&got, &id, &file).unwrap()[..], b"synthetic marker text");
        assert!(unwrap(&kek("synthetic wrong"), &id, &file).is_none());
        assert!(unwrap(&kek("synthetic right"), &DiaryId([3; 16]), &file).is_none(), "another id");
        let mut changed = file.clone();
        changed[200] ^= 1;
        assert_eq!(open(&got, &id, &changed).unwrap_err(), CryptoError::Unopenable);
        assert!(!file.windows(7).any(|w| w == b"synthet"), "no plaintext in the file");
    }

    /// **Each seal is fresh**: the same text sealed twice differs everywhere
    /// but its version.
    #[test]
    fn each_seal_is_under_fresh_nonces() {
        let id = DiaryId([4; 16]);
        let (k, d) = (kek("p"), Dek::random());
        let a = seal(&k, &d, &id, b"same");
        let b = seal(&k, &d, &id, b"same");
        assert_eq!(a.len(), b.len());
        assert_ne!(a[1..25], b[1..25]);
        assert_ne!(a[SEALED_AT..], b[SEALED_AT..]);
    }

    /// **A rewrapped diary opens under the new passphrase, not the old**, its
    /// ciphertext unchanged.
    #[test]
    fn a_rewrapped_diary_opens_under_the_new_passphrase_only() {
        let id = DiaryId([7; 16]);
        let dek = Dek::random();
        let file = seal(&kek("synthetic old"), &dek, &id, b"kept");
        let moved = rewrap(&kek("synthetic new"), &dek, &id, &file);
        assert_eq!(moved[SEALED_AT..], file[SEALED_AT..]);
        assert!(unwrap(&kek("synthetic old"), &id, &moved).is_none());
        let got = unwrap(&kek("synthetic new"), &id, &moved).unwrap();
        assert_eq!(&open(&got, &id, &moved).unwrap()[..], b"kept");
    }

    /// **The production cost stretches a passphrase**, and is what diaries
    /// use; its time is printed for tuning on each machine.
    #[test]
    fn the_production_cost_stretches_a_passphrase() {
        let started = Instant::now();
        let k = Kek::derive("synthetic pass", &[9; 16], KdfParams::PRODUCTION).unwrap();
        eprintln!("argon2id at production cost: {:?}", started.elapsed());
        assert!(!k.same(&kek("synthetic pass")), "the cost is part of the key");
    }
}
