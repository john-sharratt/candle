//! A failed git process's exit status and stderr, as a [`GitError`].
//!
//! Every invocation runs in the `C` locale, so git's messages are the English
//! ones matched here. Anything not recognised is `Unclassified` with the full
//! stderr — never folded into a nearby variant.

use std::path::Path;

use crate::error::GitError;
use crate::redact::redact_urls;
use crate::runner::Context;
use crate::types::RefName;

const NOT_A_REPOSITORY: &[&str] = &["not a git repository"];

const AUTH_FAILED: &[&str] = &[
    "permission denied (publickey",
    "authentication failed",
    "could not read username",
    "could not read password",
    "host key verification failed",
];

const UNREACHABLE: &[&str] = &[
    "could not resolve host",
    "could not resolve hostname",
    "connection refused",
    "connection timed out",
    "network is unreachable",
    "no route to host",
    "unable to access",
    "does not appear to be a git repository",
    "could not read from remote repository",
];

const UNKNOWN_REVISION: &[&str] = &[
    "unknown revision",
    "bad revision",
    "needed a single revision",
    "ambiguous argument",
    "not a valid object name",
    "invalid object name",
    "not a valid commit name",
    // A fetch of one branch the remote does not have.
    "couldn't find remote ref",
];

fn mentions(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// The ref named in git's `cannot lock ref '<name>': <reason>`, and the reason.
fn locked_ref(stderr: &str) -> Option<(RefName, String)> {
    let start = stderr.find("cannot lock ref '")? + "cannot lock ref '".len();
    let rest = &stderr[start..];
    let end = rest.find('\'')?;
    let name = RefName::parse(&rest[..end]).ok()?;
    let reason = rest[end + 1..]
        .trim_start_matches(':')
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    Some((name, reason))
}

pub(crate) fn classify(
    dir: &Path,
    context: &Context,
    args: Vec<String>,
    status: Option<i32>,
    stderr: String,
) -> GitError {
    // Git echoes remote URLs in its errors; a token in one must not travel
    // on in the error.
    let stderr = redact_urls(&stderr);
    let lower = stderr.to_ascii_lowercase();
    if mentions(&lower, NOT_A_REPOSITORY) {
        return GitError::NotARepository {
            dir: dir.to_path_buf(),
        };
    }
    if let Some((name, reason)) = locked_ref(&stderr) {
        let reason_lower = reason.to_ascii_lowercase();
        if reason_lower.contains("unable to create") && reason_lower.contains("file exists") {
            return GitError::RefLocked { name };
        }
        return GitError::StaleRef {
            name,
            detail: reason,
        };
    }
    if let Some(remote) = &context.remote {
        if mentions(&lower, AUTH_FAILED) {
            return GitError::AuthFailed {
                remote: remote.clone(),
                detail: stderr.trim().to_string(),
            };
        }
        if mentions(&lower, UNREACHABLE) {
            return GitError::RemoteUnreachable {
                remote: remote.clone(),
                detail: stderr.trim().to_string(),
            };
        }
    }
    if let Some(rev) = &context.rev {
        if mentions(&lower, UNKNOWN_REVISION) {
            return GitError::UnknownRevision { rev: rev.clone() };
        }
    }
    GitError::Unclassified {
        args,
        status,
        stderr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RemoteName;

    fn origin() -> Context {
        Context {
            remote: Some(RemoteName::parse("origin").unwrap()),
            rev: None,
        }
    }

    fn run(context: &Context, stderr: &str) -> GitError {
        classify(
            Path::new("repo"),
            context,
            vec!["x".into()],
            Some(128),
            stderr.into(),
        )
    }

    /// SSH authentication failures, as OpenSSH and git print them. These
    /// cannot be produced offline, so they are pinned as text.
    #[test]
    fn ssh_authentication_failures_are_auth_failed() {
        for stderr in [
            "git@github.com: Permission denied (publickey).\r\nfatal: Could not read from remote repository.\n",
            "Host key verification failed.\nfatal: Could not read from remote repository.\n",
            "fatal: could not read Username for 'https://github.com': terminal prompts disabled\n",
            "remote: Invalid username or password.\nfatal: Authentication failed for 'https://github.com/x/y.git/'\n",
        ] {
            assert!(
                matches!(run(&origin(), stderr), GitError::AuthFailed { .. }),
                "{stderr:?}"
            );
        }
    }

    #[test]
    fn network_failures_are_unreachable() {
        for stderr in [
            "ssh: Could not resolve hostname github.invalid: Name or service not known\nfatal: Could not read from remote repository.\n",
            "ssh: connect to host github.com port 22: Connection refused\n",
            "fatal: unable to access 'https://github.invalid/x.git/': Could not resolve host: github.invalid\n",
        ] {
            assert!(
                matches!(run(&origin(), stderr), GitError::RemoteUnreachable { .. }),
                "{stderr:?}"
            );
        }
    }

    /// Without a remote in context, network wording is not guessed at.
    #[test]
    fn remote_wording_without_a_remote_is_unclassified() {
        let e = run(&Context::default(), "Permission denied (publickey).\n");
        assert!(matches!(e, GitError::Unclassified { .. }), "{e}");
    }

    #[test]
    fn a_lock_file_is_ref_locked_and_a_wrong_value_is_stale() {
        let locked = "fatal: cannot lock ref 'refs/heads/b': Unable to create '/r/.git/refs/heads/b.lock': File exists.\n";
        assert!(matches!(
            run(&Context::default(), locked),
            GitError::RefLocked { .. }
        ));
        let stale = "fatal: cannot lock ref 'refs/heads/b': is at 1111111111111111111111111111111111111111 but expected 2222222222222222222222222222222222222222\n";
        match run(&Context::default(), stale) {
            GitError::StaleRef { name, detail } => {
                assert_eq!(name.as_str(), "refs/heads/b");
                assert!(detail.starts_with("is at "), "{detail}");
            }
            other => panic!("{other}"),
        }
    }

    /// A token in a URL git echoed never reaches the error.
    #[test]
    fn credentials_in_git_output_are_redacted() {
        let stderr = "fatal: unable to access 'https://me:ghp_secret@github.com/x.git/': 403\n";
        for e in [run(&origin(), stderr), run(&Context::default(), stderr)] {
            let text = e.to_string();
            assert!(!text.contains("ghp_secret"), "{text}");
            assert!(text.contains("https://***@github.com"), "{text}");
        }
    }

    #[test]
    fn unrecognised_stderr_keeps_everything() {
        match run(&Context::default(), "fatal: something new\n") {
            GitError::Unclassified { status, stderr, .. } => {
                assert_eq!(status, Some(128));
                assert_eq!(stderr, "fatal: something new\n");
            }
            other => panic!("{other}"),
        }
    }
}
