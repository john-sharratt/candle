//! A signed-in account's tokens, kept in ZenJ's data folder — one file per
//! account (`tokens/spotify-john.json`) — and refreshed before they expire.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What a token endpoint answers.
#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    pub expires_in: u64,
    #[serde(default)]
    pub scope: Option<String>,
}

/// A kept token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    pub access: String,
    pub refresh: String,
    /// Unix seconds.
    pub expires_at: u64,
    #[serde(default)]
    pub scope: String,
}

/// Refresh this long before a token expires.
pub const MARGIN_S: u64 = 120;

impl Token {
    /// The kept token `answer` makes at `now`, keeping `previous`'s refresh
    /// token when the answer carries none (a refresh need not rotate it).
    pub fn from_answer(answer: Answer, now: u64, previous: Option<&Token>) -> Result<Token, String> {
        let refresh = answer
            .refresh_token
            .or_else(|| previous.map(|p| p.refresh.clone()))
            .ok_or("the sign-in returned no refresh token")?;
        Ok(Token {
            access: answer.access_token,
            refresh,
            expires_at: now + answer.expires_in,
            scope: answer.scope.unwrap_or_default(),
        })
    }

    /// Whether it should be refreshed at `now`.
    pub fn stale(&self, now: u64) -> bool {
        now + MARGIN_S >= self.expires_at
    }
}

/// Where the account `name`'s token is kept under `data`.
pub fn path(data: &Path, name: &str) -> PathBuf {
    data.join("tokens").join(format!("{name}.json"))
}

pub fn load(data: &Path, name: &str) -> Result<Token, String> {
    let p = path(data, name);
    let text = fs::read_to_string(&p)
        .map_err(|e| format!("{}: {e} — sign in with `zenj login {name}`", p.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
}

/// Keep `token`, replacing the file whole.
pub fn save(data: &Path, name: &str, token: &Token) -> Result<(), String> {
    let p = path(data, name);
    if let Some(dir) = p.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let tmp = p.with_extension("tmp");
    let text = serde_json::to_string_pretty(token).expect("a token serializes");
    fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))
}

/// Unix seconds now.
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(refresh: Option<&str>) -> Answer {
        Answer {
            access_token: "acc".into(),
            refresh_token: refresh.map(str::to_string),
            expires_in: 3600,
            scope: Some("playlist-read-private".into()),
        }
    }

    #[test]
    fn a_refresh_keeps_the_refresh_token_it_was_not_given() {
        let first = Token::from_answer(answer(Some("r1")), 1000, None).unwrap();
        assert_eq!(first.expires_at, 4600);
        let again = Token::from_answer(answer(None), 2000, Some(&first)).unwrap();
        assert_eq!(again.refresh, "r1");
        assert!(Token::from_answer(answer(None), 0, None).is_err());
        assert!(!first.stale(4600 - MARGIN_S - 1));
        assert!(first.stale(4600 - MARGIN_S));
    }

    #[test]
    fn a_token_is_kept_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let t = Token::from_answer(answer(Some("r")), 5, None).unwrap();
        save(dir.path(), "tidal", &t).unwrap();
        assert_eq!(load(dir.path(), "tidal").unwrap(), t);
        assert!(load(dir.path(), "spotify-yen")
            .unwrap_err()
            .ends_with("sign in with `zenj login spotify-yen`"));
    }
}
