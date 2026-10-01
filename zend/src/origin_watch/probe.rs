//! Asking origin for its branch tips: protocol v2 `ls-refs` over HTTPS where
//! the origin has an HTTPS form, `git ls-remote` over the remote's own
//! transport where it does not (`docs/zend_branch_ingest.md` §4.1, §4.3).
//!
//! The protocol's bytes are `zend_vfs::probe`'s; this carries them.

use std::fmt;
use std::time::Duration;

use reqwest::{Client, Response, StatusCode};
use zend_vfs::probe::{self, BranchTip, ProbeUrl};
use zend_vfs::{ObjectFormat, RemoteName, Repo};

/// The one host whose probes carry the daemon's `github_token`.
const GITHUB: &str = "github.com";

/// The user a token is presented as over git's HTTP transport.
const TOKEN_USER: &str = "x-access-token";

/// How long one probe may take.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Why a probe got no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeFailure {
    /// The host will not answer this probe — no HTTPS form it serves, no
    /// credentials it accepts, no v2 `ls-refs`. The watcher probes with
    /// `git ls-remote` instead, for good.
    Refused(String),
    /// Anything that may pass: the network, a rate limit, a server error.
    Transient(String),
}

/// How a repository's origin is probed. Its `Debug` never shows the token.
#[derive(Clone, PartialEq, Eq)]
pub enum Transport {
    /// `ls-refs` over HTTPS, with the token presented to its host, if any.
    Https {
        url: ProbeUrl,
        token: Option<String>,
    },
    /// `git ls-remote origin`; `local` when origin is a path on this machine,
    /// which nothing throttles.
    Git { local: bool },
}

impl Transport {
    /// How `origin_url` is probed: over HTTPS when it has an HTTPS form, the
    /// token going only to GitHub; with git otherwise.
    pub fn for_origin(origin_url: &str, github_token: Option<&str>) -> Self {
        match probe::https_url(origin_url) {
            Some(url) => {
                let token = (url.host == GITHUB)
                    .then(|| github_token.map(str::to_string))
                    .flatten();
                Transport::Https { url, token }
            }
            None => Transport::Git {
                local: is_local(origin_url),
            },
        }
    }
}

impl fmt::Debug for Transport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Transport::Https { url, token } => f
                .debug_struct("Https")
                .field("url", &url.url)
                .field("token", &token.as_ref().map(|_| "***"))
                .finish(),
            Transport::Git { local } => f.debug_struct("Git").field("local", local).finish(),
        }
    }
}

/// Whether `origin_url` names a repository on this machine — a path or a
/// `file://` URL — rather than a host.
pub fn is_local(origin_url: &str) -> bool {
    let url = origin_url.trim();
    if url.to_ascii_lowercase().starts_with("file://") {
        return true;
    }
    if url.contains("://") {
        return false;
    }
    // Not scp-like: `host:path` has a host of two characters or more with no
    // separator in it; a drive letter is one.
    match url.split_once(':') {
        Some((host, _)) => host.len() < 2 || host.contains('/') || host.contains('\\'),
        None => true,
    }
}

/// The advertisement check, made once when a repository's HTTPS probing
/// starts: the server must speak v2 and answer `ls-refs`.
pub async fn check_https(
    client: &Client,
    url: &ProbeUrl,
    token: Option<&str>,
) -> Result<(), ProbeFailure> {
    let mut req = client
        .get(url.advertisement())
        .header("Git-Protocol", "version=2");
    if let Some(token) = token {
        req = req.basic_auth(TOKEN_USER, Some(token));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| ProbeFailure::Transient(format!("advertisement: {e}")))?;
    let body = answer(resp).await?;
    match probe::advertises_ls_refs(&body) {
        Ok(true) => Ok(()),
        Ok(false) => Err(ProbeFailure::Refused(
            "the server does not offer protocol v2 ls-refs".into(),
        )),
        Err(e) => Err(ProbeFailure::Refused(format!("advertisement: {e}"))),
    }
}

/// Origin's branch tips over HTTPS.
pub async fn https_tips(
    client: &Client,
    url: &ProbeUrl,
    token: Option<&str>,
    format: ObjectFormat,
) -> Result<Vec<BranchTip>, ProbeFailure> {
    let mut req = client
        .post(url.upload_pack())
        .header("Git-Protocol", "version=2")
        .header("Content-Type", "application/x-git-upload-pack-request")
        .header("Accept", "application/x-git-upload-pack-result")
        .body(probe::request(format));
    if let Some(token) = token {
        req = req.basic_auth(TOKEN_USER, Some(token));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| ProbeFailure::Transient(format!("ls-refs: {e}")))?;
    let body = answer(resp).await?;
    probe::parse_response(&body).map_err(|e| match e {
        probe::ProbeError::Refused(why) => ProbeFailure::Refused(why),
        probe::ProbeError::Malformed(why) => ProbeFailure::Transient(why),
    })
}

/// A response's body, or why it is not an answer: a status saying the host
/// will not answer this repository is a refusal, any other failure passes.
async fn answer(resp: Response) -> Result<Vec<u8>, ProbeFailure> {
    let status = resp.status();
    if !status.is_success() {
        return Err(classify(status));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| ProbeFailure::Transient(format!("reading the answer: {e}")))
}

/// What an unsuccessful status means for the watcher.
fn classify(status: StatusCode) -> ProbeFailure {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::NOT_FOUND => {
            ProbeFailure::Refused(format!("origin answered {status}"))
        }
        _ => ProbeFailure::Transient(format!("origin answered {status}")),
    }
}

/// Origin's branch tips over the remote's own transport — blocking; call it
/// off the async runtime.
pub fn git_tips(repo: &Repo, origin: &RemoteName) -> Result<Vec<BranchTip>, ProbeFailure> {
    let refs = repo
        .ls_remote(origin)
        .map_err(|e| ProbeFailure::Transient(format!("ls-remote: {e}")))?;
    let mut tips: Vec<BranchTip> = refs
        .refs
        .into_iter()
        .filter_map(|(name, tip)| {
            Some(BranchTip {
                name: name.branch()?,
                tip,
            })
        })
        .collect();
    tips.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    Ok(tips)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_origins_are_probed_over_https_with_the_token() {
        assert_eq!(
            Transport::for_origin("git@github.com:john-sharratt/candle.git", Some("ghp_x")),
            Transport::Https {
                url: probe::https_url("https://github.com/john-sharratt/candle.git").unwrap(),
                token: Some("ghp_x".into()),
            }
        );
    }

    /// The token is GitHub's, and goes nowhere else.
    #[test]
    fn another_host_is_probed_without_the_token() {
        let Transport::Https { url, token } =
            Transport::for_origin("git@gitlab.com:group/y.git", Some("ghp_x"))
        else {
            panic!("an scp-like origin has an HTTPS form");
        };
        assert_eq!(url.url, "https://gitlab.com/group/y.git");
        assert_eq!(token, None);
    }

    /// The transport is logged when a watcher starts; the token must never be
    /// in what is logged.
    #[test]
    fn the_token_never_reaches_a_log() {
        let t = Transport::for_origin("git@github.com:x/y.git", Some("ghp_secretvalue"));
        let shown = format!("{t:?}");
        assert!(!shown.contains("ghp_secretvalue"), "{shown}");
        assert_eq!(
            shown,
            "Https { url: \"https://github.com/x/y.git\", token: Some(\"***\") }"
        );
    }

    #[test]
    fn an_origin_without_an_https_form_is_probed_with_git() {
        for (origin, local) in [
            ("C:\\Users\\x\\origin", true),
            ("/srv/git/origin.git", true),
            ("file:///C:/Users/x/origin", true),
            ("http://example.com/y.git", false),
        ] {
            assert_eq!(
                Transport::for_origin(origin, Some("ghp_x")),
                Transport::Git { local },
                "{origin}"
            );
        }
    }

    #[test]
    fn a_status_refusing_the_repository_is_a_refusal_and_the_rest_pass() {
        for status in [
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
        ] {
            assert!(
                matches!(classify(status), ProbeFailure::Refused(_)),
                "{status}"
            );
        }
        for status in [
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
        ] {
            assert!(
                matches!(classify(status), ProbeFailure::Transient(_)),
                "{status}"
            );
        }
    }
}
