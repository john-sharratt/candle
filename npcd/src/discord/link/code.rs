//! The codes a person links their Discord account with
//! (`docs/npcd_discord.md` §3.2).
//!
//! A code is [`LENGTH`] characters of Crockford's base32, good for
//! [`VALID_MS`], used once, one standing per account — asking again replaces
//! it. Each is the custom object `discord/link/<code>` → `{ account,
//! expires_ms }`, so whichever member's bot the code is sent to can redeem
//! it. It names no account to whoever reads it, and grants only the one link.
//!
//! **Long enough that guessing one is hopeless.** Anybody can DM a bot, and a
//! live code links the sender's Discord account to whoever asked for it. Six
//! digits were a million guesses; twelve base32 characters are 2^60. The
//! alphabet has no I, L, O or U, so nothing in it is misread, and a code is
//! shown in fours with dashes ([`shown`]) and taken back with or without them,
//! in either case ([`normal`]).

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Where codes live among the substrate's custom objects.
const PREFIX: &str = "discord/link/";

/// How long a code is good for: ten minutes.
pub const VALID_MS: u64 = 10 * 60 * 1000;

/// Crockford's base32: the digits and the capitals, less I, L, O and U.
const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// How many characters a code has: 60 bits.
pub const LENGTH: usize = 12;

/// The record key of `code`, in its [`normal`] form.
pub fn key(code: &str) -> String {
    format!("{PREFIX}{code}")
}

/// `text` as a code, if it is one: its dashes and spaces dropped and its
/// letters made capitals — `"abcd-efgh-jkmn"` is `ABCDEFGHJKMN`.
pub fn normal(text: &str) -> Option<String> {
    let code: String = text
        .chars()
        .filter(|c| *c != '-' && !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    (code.len() == LENGTH && code.bytes().all(|b| ALPHABET.contains(&b))).then_some(code)
}

/// A [`normal`] code as a person reads it: in fours, with dashes.
pub fn shown(code: &str) -> String {
    code.as_bytes()
        .chunks(4)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect::<Vec<_>>()
        .join("-")
}

/// A standing code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Code {
    /// The account it links.
    pub account: String,
    /// When it stops being good, in real milliseconds.
    pub expires_ms: u64,
}

/// Whether `text` is shaped like a code — see [`normal`].
pub fn is_code(text: &str) -> bool {
    normal(text).is_some()
}

/// The code standing for `account` at `now_ms`, if one does: the code as it
/// is [`shown`], and when it expires.
pub fn standing(shared: &SharedSubstrate, account: &str, now_ms: u64) -> Option<(String, u64)> {
    shared
        .custom_objects_under(PREFIX)
        .into_iter()
        .filter_map(|(o, _)| {
            let code: Code = serde_json::from_value(o.blob).ok()?;
            (code.account == account && code.expires_ms > now_ms)
                .then(|| (shown(&o.key[PREFIX.len()..]), code.expires_ms))
        })
        .next()
}

/// A fresh code for `account` at `now_ms`, replacing any it had: the code as
/// it is [`shown`], and when it expires.
pub fn mint(shared: &SharedSubstrate, account: &str, now_ms: u64) -> anyhow::Result<(String, u64)> {
    let held: Vec<(String, Code)> = shared
        .custom_objects_under(PREFIX)
        .into_iter()
        .filter_map(|(o, _)| {
            Some((
                o.key[PREFIX.len()..].to_string(),
                serde_json::from_value(o.blob).ok()?,
            ))
        })
        .collect();
    for (code, c) in &held {
        // An account's old code, and any code past its time: neither is good
        // for anything, and leaving them would only crowd the space of codes.
        if c.account == account || c.expires_ms <= now_ms {
            cease(shared, code)?;
        }
    }
    let live = |code: &str| {
        held.iter()
            .any(|(c, held)| c == code && held.account != account && held.expires_ms > now_ms)
    };
    let mut rng = rand::rng();
    let code = loop {
        let code: String = (0..LENGTH)
            .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())] as char)
            .collect();
        if !live(&code) {
            break code;
        }
    };
    let expires_ms = now_ms + VALID_MS;
    let body = serde_json::to_value(Code {
        account: account.to_string(),
        expires_ms,
    })
    .expect("a code serialises");
    shared
        .put_custom_object(CustomObjectPayload::new(key(&code), body))
        .map_err(|e| anyhow::anyhow!("a link code was not written: {e:?}"))?;
    Ok((shown(&code), expires_ms))
}

/// Redeem `code` — as typed: see [`normal`] — at `now_ms`: the account it
/// links, if it is good, and it is good no more.
pub fn redeem(shared: &SharedSubstrate, code: &str, now_ms: u64) -> anyhow::Result<Option<String>> {
    let Some(code) = normal(code) else {
        return Ok(None);
    };
    let Some(held) = shared
        .custom_object(&key(&code))
        .and_then(|o| serde_json::from_value::<Code>(o.blob).ok())
    else {
        return Ok(None);
    };
    cease(shared, &code)?;
    Ok((held.expires_ms > now_ms).then_some(held.account))
}

/// Tombstone `code`.
fn cease(shared: &SharedSubstrate, code: &str) -> anyhow::Result<()> {
    shared
        .put_custom_object(CustomObjectPayload::ceased(key(code)))
        .map_err(|e| anyhow::anyhow!("a link code was not removed: {e:?}"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    const NOW: u64 = 1_800_000_000_000;

    /// **A code is twelve base32 characters, shown in fours, the record
    /// `discord/link/<code>` naming its account, good for ten minutes.**
    #[test]
    fn a_code_is_twelve_characters_good_for_ten_minutes() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let (code, expires_ms) = mint(&shared, "0a1b", NOW).unwrap();
        assert!(is_code(&code), "{code}");
        assert_eq!(code.len(), 14, "{code}");
        assert_eq!(code.matches('-').count(), 2, "{code}");
        let normal_code = normal(&code).unwrap();
        assert_eq!(shown(&normal_code), code);
        assert_eq!(expires_ms, NOW + 600_000);
        assert_eq!(
            shared.custom_object(&key(&normal_code)).unwrap().blob,
            json!({ "account": "0a1b", "expires_ms": NOW + 600_000 })
        );
        assert_eq!(
            standing(&shared, "0a1b", NOW),
            Some((code.clone(), expires_ms))
        );
        assert_eq!(standing(&shared, "0a1b", NOW + 600_000), None);
        assert_eq!(standing(&shared, "ffee", NOW), None);
    }

    /// **Used once**: the first redeem names the account, the second nothing.
    #[test]
    fn a_code_is_used_once() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let (code, _) = mint(&shared, "0a1b", NOW).unwrap();
        assert_eq!(
            redeem(&shared, &code, NOW + 1).unwrap().as_deref(),
            Some("0a1b")
        );
        assert_eq!(redeem(&shared, &code, NOW + 2).unwrap(), None);
        assert_eq!(shared.custom_object(&key(&normal(&code).unwrap())), None);
    }

    /// **A code is taken back however it is typed** — without its dashes, in
    /// small letters.
    #[test]
    fn a_code_typed_without_dashes_or_capitals_still_links() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let (code, _) = mint(&shared, "0a1b", NOW).unwrap();
        let typed = code.replace('-', "").to_lowercase();
        assert_eq!(
            redeem(&shared, &typed, NOW + 1).unwrap().as_deref(),
            Some("0a1b")
        );
    }

    /// **Past its ten minutes it links nothing**, and is gone.
    #[test]
    fn an_expired_code_links_nothing() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let (code, expires_ms) = mint(&shared, "0a1b", NOW).unwrap();
        assert_eq!(redeem(&shared, &code, expires_ms).unwrap(), None);
        assert_eq!(shared.custom_object(&key(&normal(&code).unwrap())), None);
    }

    /// **One standing code per account**: asking again replaces it.
    #[test]
    fn asking_again_replaces_the_standing_code() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let (first, _) = mint(&shared, "0a1b", NOW).unwrap();
        let (other, _) = mint(&shared, "ffee", NOW).unwrap();
        let (second, _) = mint(&shared, "0a1b", NOW + 5).unwrap();
        assert_eq!(
            standing(&shared, "0a1b", NOW + 5).map(|s| s.0),
            Some(second.clone())
        );
        if first != second {
            assert_eq!(redeem(&shared, &first, NOW + 6).unwrap(), None);
        }
        assert_eq!(
            redeem(&shared, &other, NOW + 6).unwrap().as_deref(),
            Some("ffee")
        );
        assert_eq!(
            redeem(&shared, &second, NOW + 6).unwrap().as_deref(),
            Some("0a1b")
        );
    }

    /// **Only twelve characters of the alphabet are a code**, dashes and spaces
    /// aside and either case; six digits, a misreadable letter, or one too few
    /// or too many are an ordinary message.
    #[test]
    fn only_twelve_base32_characters_are_a_code() {
        for (text, want) in [
            ("ABCD-EFGH-JKMN", Some("ABCDEFGHJKMN")),
            ("abcd-efgh-jkmn", Some("ABCDEFGHJKMN")),
            ("ABCDEFGHJKMN", Some("ABCDEFGHJKMN")),
            ("0123 4567 89XZ", Some("0123456789XZ")),
            ("123456", None),
            ("ABCD-EFGH-JKM", None),
            ("ABCD-EFGH-JKMNP", None),
            ("ABCD-EFGH-JKMI", None),
            ("ABCD-EFGH-JKML", None),
            ("ABCD-EFGH-JKMO", None),
            ("ABCD-EFGH-JKMU", None),
            ("ＡＢＣＤ-ＥＦＧＨ-ＪＫＭＮ", None),
        ] {
            assert_eq!(normal(text).as_deref(), want, "{text:?}");
            assert_eq!(is_code(text), want.is_some(), "{text:?}");
        }
        assert_eq!(shown("ABCDEFGHJKMN"), "ABCD-EFGH-JKMN");
    }
}
