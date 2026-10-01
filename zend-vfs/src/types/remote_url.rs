//! Remote URLs.

use std::fmt;

use crate::error::GitError;

/// Transports that do not fetch from a repository at all: `ext::` runs an
/// arbitrary command and `fd::` talks over inherited file descriptors. A URL
/// naming either is refused, and every invocation also disables them in
/// config (`protocol.ext.allow=never`).
const REFUSED_TRANSPORTS: &[&str] = &["ext::", "fd::"];

/// Where a remote lives: an SSH, HTTPS or `file://` URL, an scp-like
/// `user@host:path`, or a local path.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RemoteUrl(String);

impl RemoteUrl {
    pub fn parse(url: &str) -> Result<Self, GitError> {
        let refuse = |why: &str| Err(GitError::invalid(format!("remote url {url:?} {why}")));
        if url.trim().is_empty() {
            return refuse("is empty");
        }
        if url.starts_with('-') {
            return refuse("starts with `-`");
        }
        if url.chars().any(|c| c.is_control()) {
            return refuse("contains a control character");
        }
        let lower = url.to_ascii_lowercase();
        if REFUSED_TRANSPORTS.iter().any(|t| lower.starts_with(t)) {
            return refuse("uses a transport that runs commands rather than fetching");
        }
        Ok(Self(url.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RemoteUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for RemoteUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RemoteUrl({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_urls_parse() {
        for ok in [
            "git@github.com:john-sharratt/candle.git",
            "ssh://git@github.com/x/y.git",
            "https://github.com/x/y.git",
            "file:///C:/Users/x/origin",
            "C:\\Users\\x\\origin",
            "/srv/git/origin.git",
        ] {
            assert_eq!(RemoteUrl::parse(ok).unwrap().as_str(), ok);
        }
    }

    #[test]
    fn command_transports_flags_and_control_characters_are_refused() {
        for bad in [
            "",
            "-uevil",
            "ext::sh -c touch% /tmp/pwned",
            "EXT::sh",
            "fd::3",
            "https://x/y\n",
        ] {
            assert!(RemoteUrl::parse(bad).is_err(), "{bad:?} should be refused");
        }
    }
}
