//! A job's id: a random 64-bit number, as URL-safe base64.
//!
//! Eleven characters from `A–Z a–z 0–9 - _` and no padding, so an id is its
//! own log file's name on every platform and goes into a URL as it is.

use std::fmt;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;

/// One job's id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(String);

impl JobId {
    /// A new id, drawn at random.
    pub fn random() -> Self {
        Self::from_number(rand::random::<u64>())
    }

    fn from_number(n: u64) -> Self {
        Self(URL_SAFE_NO_PAD.encode(n.to_be_bytes()))
    }

    /// `text` as an id: exactly the form [`Self::random`] makes — eight bytes,
    /// URL-safe base64, no padding. `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        let bytes = URL_SAFE_NO_PAD.decode(text).ok()?;
        (bytes.len() == 8 && URL_SAFE_NO_PAD.encode(&bytes) == text).then(|| Self(text.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **An id is the number's eight bytes, big-endian, as URL-safe base64
    /// without padding** — byte for byte.
    #[test]
    fn an_id_is_its_numbers_bytes_in_url_safe_base64() {
        assert_eq!(JobId::from_number(0).as_str(), "AAAAAAAAAAA");
        assert_eq!(
            JobId::from_number(0x0102_0304_0506_0708).as_str(),
            "AQIDBAUGBwg"
        );
        assert_eq!(JobId::from_number(u64::MAX).as_str(), "__________8");
    }

    #[test]
    fn random_ids_are_well_formed_and_differ() {
        let a = JobId::random();
        let b = JobId::random();
        assert_ne!(a, b);
        for id in [&a, &b] {
            assert_eq!(id.as_str().len(), 11);
            assert_eq!(JobId::parse(id.as_str()).as_ref(), Some(id));
        }
    }

    /// Only the exact form parses: no padding, no standard-alphabet `+` or
    /// `/`, no other length, nothing a path could be made of.
    #[test]
    fn only_the_exact_form_parses() {
        for bad in [
            "",
            "AAAAAAAAAAA=",
            "AAAAAAAAAA",
            "AAAAAAAAAAAA",
            "//////////8",
            "++++++++++8",
            "../../etc/x",
            "__________9",
        ] {
            assert_eq!(JobId::parse(bad), None, "{bad:?}");
        }
        assert!(JobId::parse("__________8").is_some());
    }
}
