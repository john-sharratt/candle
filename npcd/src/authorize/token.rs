//! An authorization: a prompt cleared for a purpose, signed so any member can
//! check it (`docs/guest_draw_orchestration.md` §1).
//!
//! It signs, under a key every member derives from the group key, the purpose,
//! the SHA-256 of the exact prompt, and the moment it stops counting — the
//! bytes `"{purpose}\n{prompt_sha256}\n{exp_ms}"`. Neither the purpose nor a
//! hex digest holds a newline, so no other triple signs the same bytes. A
//! daemon with no cluster mints its own key at start, as image links do
//! (`crate::image_links`).

use std::fmt::{self, Debug, Formatter};
use std::time::Duration;

use candle_conversation::persistence::blob::{hex, parse_hex};
use hmac::{Hmac, Mac};
use rand::rngs::OsRng;
use rand::TryRngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// What the signing key is derived for, from the group key.
pub const KEY_PURPOSE: &str = "npcd prompt authorization";

/// The purpose a draw's prompt is authorized for.
pub const IMAGE: &str = "image";

/// How long an authorization counts: long enough to finish a draw that waited
/// for room, short enough not to be a standing permission.
pub const LIFETIME: Duration = Duration::from_secs(10 * 60);

/// A prompt cleared for a purpose until `exp_ms`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Authorization {
    pub purpose: String,
    /// The SHA-256 of the exact prompt, in hex.
    pub prompt_sha256: String,
    /// Unix milliseconds; it no longer counts from this moment on.
    pub exp_ms: u64,
    /// HMAC-SHA256 over the three above, in hex.
    pub sig: String,
}

/// Why an authorization does not clear a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invalid {
    /// Not this key's signature over its own fields.
    Signature,
    /// Signed for another purpose.
    Purpose,
    /// Signed for another prompt.
    Prompt,
    /// Signed, and its time is up.
    Expired,
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Signature => "the authorization's signature is not this cluster's",
            Self::Purpose => "the authorization is for another purpose",
            Self::Prompt => "the authorization is for another prompt",
            Self::Expired => "the authorization has expired",
        })
    }
}

/// The SHA-256 of `prompt`, in hex: what an authorization names it by.
pub fn prompt_sha256(prompt: &str) -> String {
    let digest: [u8; 32] = Sha256::digest(prompt.as_bytes()).into();
    hex(&digest)
}

/// Signs authorizations, and checks them.
#[derive(Clone)]
pub struct Signer {
    key: [u8; 32],
}

impl Debug for Signer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Signer(…)")
    }
}

impl Signer {
    /// Authorizations signed with `key` — the group key's derivation for
    /// [`KEY_PURPOSE`].
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Authorizations signed with a key of this process's own: a daemon with
    /// no cluster to share one with.
    pub fn minted() -> Self {
        let mut key = [0u8; 32];
        OsRng
            .try_fill_bytes(&mut key)
            .expect("the OS must provide entropy");
        Self { key }
    }

    /// `prompt` cleared for `purpose` from `now_ms`, for [`LIFETIME`].
    pub fn sign(&self, purpose: &str, prompt: &str, now_ms: u64) -> Authorization {
        let prompt_sha256 = prompt_sha256(prompt);
        let exp_ms = now_ms + LIFETIME.as_millis() as u64;
        let sig: [u8; 32] = self
            .mac(purpose, &prompt_sha256, exp_ms)
            .finalize()
            .into_bytes()
            .into();
        Authorization {
            purpose: purpose.to_string(),
            prompt_sha256,
            exp_ms,
            sig: hex(&sig),
        }
    }

    /// Whether `auth` clears `prompt` for `purpose` at `now_ms`: the signature
    /// first and in constant time, so a forgery learns nothing from which
    /// refusal it gets.
    pub fn verify(
        &self,
        auth: &Authorization,
        purpose: &str,
        prompt: &str,
        now_ms: u64,
    ) -> Result<(), Invalid> {
        let sig: [u8; 32] = parse_hex(&auth.sig).ok_or(Invalid::Signature)?;
        self.mac(&auth.purpose, &auth.prompt_sha256, auth.exp_ms)
            .verify_slice(&sig)
            .map_err(|_| Invalid::Signature)?;
        if auth.purpose != purpose {
            return Err(Invalid::Purpose);
        }
        if auth.prompt_sha256 != prompt_sha256(prompt) {
            return Err(Invalid::Prompt);
        }
        if now_ms >= auth.exp_ms {
            return Err(Invalid::Expired);
        }
        Ok(())
    }

    fn mac(&self, purpose: &str, prompt_sha256: &str, exp_ms: u64) -> HmacSha256 {
        let mut m = HmacSha256::new_from_slice(&self.key).expect("HMAC takes a key of any length");
        m.update(format!("{purpose}\n{prompt_sha256}\n{exp_ms}").as_bytes());
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    fn signer() -> Signer {
        Signer::new([7; 32])
    }

    /// **The token is these exact bytes**: the prompt's digest, the expiry ten
    /// minutes on, and the HMAC over `purpose\ndigest\nexp`.
    #[test]
    fn an_authorization_signs_its_purpose_prompt_and_expiry() {
        let a = signer().sign(IMAGE, "a lighthouse", NOW);
        assert_eq!(a.purpose, "image");
        let digest: [u8; 32] = Sha256::digest(b"a lighthouse").into();
        assert_eq!(
            a.prompt_sha256,
            hex(&digest),
            "the digest of the exact prompt"
        );
        assert_eq!(a.exp_ms, NOW + 600_000);
        let mut m = HmacSha256::new_from_slice(&[7; 32]).unwrap();
        m.update(format!("image\n{}\n{}", a.prompt_sha256, a.exp_ms).as_bytes());
        let want: [u8; 32] = m.finalize().into_bytes().into();
        assert_eq!(a.sig, hex(&want));
    }

    /// **It clears exactly its prompt, for its purpose, until it expires, under
    /// its key** — and nothing else.
    #[test]
    fn it_clears_only_what_it_was_signed_for() {
        let s = signer();
        let a = s.sign(IMAGE, "a lighthouse", NOW);
        assert_eq!(s.verify(&a, IMAGE, "a lighthouse", NOW + 1), Ok(()));
        assert_eq!(
            s.verify(&a, IMAGE, "a lighthouse ", NOW),
            Err(Invalid::Prompt),
            "the exact prompt, untrimmed"
        );
        assert_eq!(
            s.verify(&a, "portrait", "a lighthouse", NOW),
            Err(Invalid::Purpose)
        );
        assert_eq!(
            s.verify(&a, IMAGE, "a lighthouse", a.exp_ms),
            Err(Invalid::Expired),
            "the expiry itself no longer counts"
        );
        assert_eq!(
            Signer::new([8; 32]).verify(&a, IMAGE, "a lighthouse", NOW),
            Err(Invalid::Signature)
        );
        let mut forged = a.clone();
        forged.exp_ms += 1;
        assert_eq!(
            s.verify(&forged, IMAGE, "a lighthouse", NOW),
            Err(Invalid::Signature),
            "a lengthened expiry breaks the signature"
        );
        forged.sig = "zz".into();
        assert_eq!(
            s.verify(&forged, IMAGE, "a lighthouse", NOW),
            Err(Invalid::Signature)
        );
    }

    /// The token travels as JSON with exactly its four fields.
    #[test]
    fn it_reads_back_as_written() {
        let a = signer().sign(IMAGE, "x", NOW);
        let json = serde_json::to_value(&a).unwrap();
        let mut keys: Vec<&String> = json.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["exp_ms", "prompt_sha256", "purpose", "sig"]);
        assert_eq!(serde_json::from_value::<Authorization>(json).unwrap(), a);
    }
}
