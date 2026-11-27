//! The tokens that would close a required string value blank
//! ([`FreeTextLimits::nonblank`](super::tree::FreeTextLimits::nonblank)).
//!
//! A required value may not be written empty, so while nothing but whitespace
//! is written in it the session hands the sampler the tokens that would end it
//! there, to be left out of that step's draw. Which those are depends on where
//! the value stands ([`Blank`]):
//!
//! - **Unopened** — a value whose opening quote is the model's to write
//!   ([`Terminator::JsonStringValue`](super::terminator::Terminator)): any
//!   token that opens and closes it at once, ` ""`, `"",` or `""}`. Qwen spells
//!   an empty string as that one token, which is why it was the one written.
//! - **Opened** — past the quote with only whitespace after it: any token that
//!   closes it, `"`, `",` or ` "}`.
//!
//! Both are a scan of the vocabulary once per tokenizer — a few hundred tokens
//! each — kept for the process's life by the vocabulary's fingerprint, so a
//! step that needs one pays only for its copy into the step's bans.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwap;

use super::terminator::Blank;
use super::vocab::{TokenId, Vocab};

/// The tokens that close a blank value, by where it stands.
#[derive(Debug, PartialEq, Eq)]
pub struct BlankBans {
    unopened: Arc<[TokenId]>,
    opened: Arc<[TokenId]>,
}

/// The bans made so far, by vocabulary fingerprint.
type BanMap = HashMap<u64, Arc<BlankBans>>;

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// `bytes` past their leading whitespace.
fn after_ws(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|&b| !is_ws(b)).unwrap_or(bytes.len());
    &bytes[start..]
}

/// Whether a token of `bytes`, written where a value's opening quote belongs,
/// opens it and closes it at once: whitespace, a quote, whitespace, a quote.
fn writes_empty(bytes: &[u8]) -> bool {
    match after_ws(bytes) {
        [b'"', rest @ ..] => after_ws(rest).first() == Some(&b'"'),
        _ => false,
    }
}

/// Whether a token of `bytes`, written inside a value holding only
/// whitespace, closes it: whitespace, then a quote.
fn closes_blank(bytes: &[u8]) -> bool {
    after_ws(bytes).first() == Some(&b'"')
}

impl BlankBans {
    /// The bans of `vocab`, made the first time it is asked for.
    pub fn of(vocab: &dyn Vocab) -> Arc<BlankBans> {
        static BANS: OnceLock<ArcSwap<BanMap>> = OnceLock::new();
        let held = BANS.get_or_init(ArcSwap::default);
        let key = vocab.fingerprint();
        if let Some(bans) = held.load().get(&key) {
            return Arc::clone(bans);
        }
        let bans = Arc::new(Self::over(vocab.extensions(b"")));
        held.rcu(|map| {
            let mut map = BanMap::clone(map);
            map.insert(key, Arc::clone(&bans));
            map
        });
        bans
    }

    /// The bans over `tokens`, each with its bytes.
    fn over(tokens: Vec<(TokenId, Vec<u8>)>) -> BlankBans {
        let mut unopened: Vec<TokenId> = Vec::new();
        let mut opened: Vec<TokenId> = Vec::new();
        for (token, bytes) in tokens {
            if writes_empty(&bytes) {
                unopened.push(token);
            }
            if closes_blank(&bytes) {
                opened.push(token);
            }
        }
        unopened.sort_unstable();
        opened.sort_unstable();
        BlankBans {
            unopened: Arc::from(unopened),
            opened: Arc::from(opened),
        }
    }

    /// The tokens that would close a value standing at `blank`.
    pub fn at(&self, blank: Blank) -> &Arc<[TokenId]> {
        match blank {
            Blank::Unopened => &self.unopened,
            Blank::Opened => &self.opened,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bans(tokens: &[(TokenId, &str)]) -> BlankBans {
        BlankBans::over(
            tokens
                .iter()
                .map(|(t, s)| (*t, s.as_bytes().to_vec()))
                .collect(),
        )
    }

    /// **Unopened, a token that opens and closes the value is banned**, with
    /// whatever follows it; one that opens it and writes content is not, nor
    /// one that only opens it.
    #[test]
    fn unopened_bans_the_tokens_that_write_it_empty() {
        let b = bans(&[
            (1, " \"\""),
            (2, "\"\","),
            (3, "\"\"}"),
            (4, " \" \""),
            (5, " \""),
            (6, "\""),
            (7, " \"a"),
            (8, "}}"),
            (9, " "),
        ]);
        assert_eq!(&b.at(Blank::Unopened)[..], [1, 2, 3, 4]);
    }

    /// **Opened, a token that closes the value is banned**, leading space and
    /// what follows the quote with it; content, and an escaped quote, are not.
    #[test]
    fn opened_bans_the_tokens_that_close_it() {
        let b = bans(&[
            (1, "\""),
            (2, "\","),
            (3, " \"}"),
            (4, "\"\""),
            (5, "a\""),
            (6, "\\\""),
            (7, " "),
            (8, "src"),
        ]);
        assert_eq!(&b.at(Blank::Opened)[..], [1, 2, 3, 4]);
    }
}
