//! rekordbox's `master.db` is SQLCipher 4 with its defaults: 4096-byte
//! pages, each AES-256-CBC under a key PBKDF2-HMAC-SHA512 derives from the
//! passphrase and the file's 16-byte salt (256,000 rounds), and each page
//! carrying 80 reserved bytes — its IV and an HMAC-SHA512 over its
//! ciphertext, IV and page number. Pages are checked and decrypted here into
//! a plain SQLite file, read-only, which the workspace's SQLite opens.

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes256;
use hmac::{Hmac, Mac};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha512;

pub const PAGE: usize = 4096;
const SALT: usize = 16;
const IV: usize = 16;
const MAC: usize = 64;
/// IV and HMAC, rounded up to the cipher's block: 80.
pub const RESERVE: usize = 80;
/// SQLCipher 4's key derivation rounds.
pub const ROUNDS: u32 = 256_000;
/// The rounds of the HMAC key's derivation from the cipher key.
const MAC_ROUNDS: u32 = 2;
/// What a plain SQLite file starts with, where SQLCipher keeps its salt.
const HEADER: &[u8; 16] = b"SQLite format 3\0";

/// rekordbox 6 and 7's passphrase for `master.db`, as every reader of the
/// library uses it.
pub const REKORDBOX_KEY: &str = "402fd482c38817c35ffa8ffb8c7d93143b749e7d315df7a81732a1ff43608497";

/// The cipher key and the HMAC key for `salt`.
pub fn keys(pass: &[u8], salt: &[u8], rounds: u32) -> ([u8; 32], [u8; 32]) {
    let mut key = [0u8; 32];
    pbkdf2_hmac::<Sha512>(pass, salt, rounds, &mut key);
    let mac_salt: Vec<u8> = salt.iter().map(|b| b ^ 0x3A).collect();
    let mut mac_key = [0u8; 32];
    pbkdf2_hmac::<Sha512>(&key, &mac_salt, MAC_ROUNDS, &mut mac_key);
    (key, mac_key)
}

fn page_mac(mac_key: &[u8; 32], body: &[u8], iv: &[u8], pgno: u32) -> [u8; MAC] {
    let mut m = <Hmac<Sha512> as Mac>::new_from_slice(mac_key).expect("any key length");
    m.update(body);
    m.update(iv);
    m.update(&pgno.to_le_bytes());
    let mut out = [0u8; MAC];
    out.copy_from_slice(&m.finalize().into_bytes());
    out
}

/// The region of page `index` (0-based) that is enciphered: after the salt
/// on the first page, up to the reserved bytes.
fn body_range(index: usize) -> std::ops::Range<usize> {
    let start = if index == 0 { SALT } else { 0 };
    start..PAGE - RESERVE
}

/// `file`, a SQLCipher 4 database under `pass`, as a plain SQLite file.
pub fn decrypt(file: &[u8], pass: &[u8], rounds: u32) -> Result<Vec<u8>, String> {
    if file.len() < PAGE || file.len() % PAGE != 0 {
        return Err(format!("{} bytes is not a whole number of pages", file.len()));
    }
    let (key, mac_key) = keys(pass, &file[..SALT], rounds);
    let cipher = Aes256::new(GenericArray::from_slice(&key));
    let mut out = Vec::with_capacity(file.len());
    for (i, page) in file.chunks_exact(PAGE).enumerate() {
        let body = &page[body_range(i)];
        let reserve = &page[PAGE - RESERVE..];
        let (iv, mac) = (&reserve[..IV], &reserve[IV..IV + MAC]);
        if page_mac(&mac_key, body, iv, i as u32 + 1)[..] != *mac {
            return Err(format!(
                "page {} fails its HMAC: the wrong key, or a file changed while it was read",
                i + 1
            ));
        }
        if i == 0 {
            out.extend_from_slice(HEADER);
        }
        let mut prev: [u8; 16] = iv.try_into().expect("an IV is a block");
        for block in body.chunks_exact(16) {
            let mut b = GenericArray::clone_from_slice(block);
            cipher.decrypt_block(&mut b);
            for (x, p) in b.iter_mut().zip(prev) {
                *x ^= p;
            }
            out.extend_from_slice(&b);
            prev = block.try_into().expect("a block");
        }
        out.extend_from_slice(reserve);
    }
    Ok(out)
}

/// `plain`, a SQLite file with 80 reserved bytes a page, enciphered as
/// SQLCipher 4 would with `salt` and the IVs `iv_of` gives each page.
pub fn encrypt(
    plain: &[u8],
    pass: &[u8],
    rounds: u32,
    salt: &[u8; 16],
    iv_of: impl Fn(usize) -> [u8; 16],
) -> Vec<u8> {
    let (key, mac_key) = keys(pass, salt, rounds);
    let cipher = Aes256::new(GenericArray::from_slice(&key));
    let mut out = Vec::with_capacity(plain.len());
    for (i, page) in plain.chunks_exact(PAGE).enumerate() {
        if i == 0 {
            out.extend_from_slice(salt);
        }
        let iv = iv_of(i);
        let mut prev = iv;
        let mut body = Vec::with_capacity(PAGE);
        for block in page[body_range(i)].chunks_exact(16) {
            let mut b = GenericArray::clone_from_slice(block);
            for (x, p) in b.iter_mut().zip(prev) {
                *x ^= p;
            }
            cipher.encrypt_block(&mut b);
            prev.copy_from_slice(&b);
            body.extend_from_slice(&b);
        }
        let mac = page_mac(&mac_key, &body, &iv, i as u32 + 1);
        out.extend_from_slice(&body);
        out.extend_from_slice(&iv);
        out.extend_from_slice(&mac);
        out.extend_from_slice(&[0u8; RESERVE - IV - MAC]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ROUNDS: u32 = 4;

    /// Two pages of a plain file: the header, then a pattern.
    fn plain() -> Vec<u8> {
        let mut p: Vec<u8> = (0..2 * PAGE).map(|i| (i * 7 % 251) as u8).collect();
        p[..16].copy_from_slice(HEADER);
        for page in p.chunks_exact_mut(PAGE) {
            page[PAGE - RESERVE..].fill(0);
        }
        p
    }

    fn iv(i: usize) -> [u8; 16] {
        [i as u8 + 1; 16]
    }

    /// **A file round-trips**: enciphered with its salt, the plain file
    /// comes back byte for byte — the header where the salt was.
    #[test]
    fn a_file_round_trips_through_the_cipher() {
        let salt = [0x5A; 16];
        let enc = encrypt(&plain(), b"secret", TEST_ROUNDS, &salt, iv);
        assert_eq!(&enc[..16], &salt);
        assert_ne!(&enc[16..PAGE - RESERVE], &plain()[16..PAGE - RESERVE]);
        let dec = decrypt(&enc, b"secret", TEST_ROUNDS).unwrap();
        let mut want = plain();
        // The reserve comes back as it was stored: the IV and the HMAC.
        for (i, page) in want.chunks_exact_mut(PAGE).enumerate() {
            page[PAGE - RESERVE..].copy_from_slice(&enc[(i + 1) * PAGE - RESERVE..(i + 1) * PAGE]);
        }
        assert_eq!(dec, want);
        assert_eq!(&dec[..16], HEADER);
    }

    /// **The wrong key, or one changed byte, fails the page's HMAC** rather
    /// than decrypting to garbage.
    #[test]
    fn a_wrong_key_or_a_changed_byte_is_refused() {
        let salt = [0x11; 16];
        let mut enc = encrypt(&plain(), b"secret", TEST_ROUNDS, &salt, iv);
        assert_eq!(
            decrypt(&enc, b"wrong", TEST_ROUNDS).unwrap_err(),
            "page 1 fails its HMAC: the wrong key, or a file changed while it was read"
        );
        enc[PAGE + 100] ^= 1;
        assert!(decrypt(&enc, b"secret", TEST_ROUNDS).unwrap_err().starts_with("page 2 "));
        assert!(decrypt(&enc[..100], b"secret", TEST_ROUNDS).is_err());
    }

    /// The derivation is PBKDF2-HMAC-SHA512 as published: RFC-style vector
    /// for one round of "password"/"salt".
    #[test]
    fn the_key_derivation_is_pbkdf2_sha512() {
        let mut k = [0u8; 32];
        pbkdf2_hmac::<Sha512>(b"password", b"salt", 1, &mut k);
        assert_eq!(
            k[..8],
            [0x86, 0x7f, 0x70, 0xcf, 0x1a, 0xde, 0x02, 0xcf]
        );
    }
}
