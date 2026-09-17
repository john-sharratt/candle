//! Deployment-level secrets for the tools that call third-party APIs.
//!
//! # Why this is not the credential store
//!
//! [`CredentialStore`](crate::state::CredentialStore) holds what a *conversation*
//! saved: a bearer token handed to `credential_save`, an SSH key a session was
//! opened with. Those belong to the chat that created them and go away with it.
//! A Tavily key is the opposite kind of thing — one per deployment, owned by
//! whoever runs the daemon, and needed by `web_search` on the first call of a
//! brand new conversation. Routing it through the credential store would mean
//! asking the user to paste it into every chat before search worked.
//!
//! # Where it lives
//!
//! [`RELATIVE_PATH`](ToolSecrets::RELATIVE_PATH) — `secrets/tools.yaml` under the
//! daemon's working directory, beside the gateway's own `web/secrets/`.
//!
//! ```yaml
//! # secrets/tools.yaml
//! tavily_api_key: tvly-...
//! ```
//!
//! # Two protections, because either alone is worthless
//!
//! **`.gitignore` keeps it out of commits.** That is all it does.
//!
//! **The VFS refuses it, which keeps it out of the model.** This is the half
//! that is easy to forget, and the reason it matters is specific: the `file_*`
//! tools mount the daemon's working directory as the lower layer of the overlay,
//! and [`VfsStore::read`](crate::state::VfsStore::read) resolves a path straight
//! to disk. A gitignored file is invisible to `git` and perfectly readable by
//! `file_read` — the listing walk honours ignore rules, the read path never did.
//! So [`VfsStore`](crate::state::VfsStore) refuses every path with a `secrets`
//! segment outright, and that refusal is what makes an in-repository secret
//! safe to keep. Without it this file would be one tool call from the
//! transcript; without the `.gitignore` entry it would be one `git add -A` from
//! the public history.
//!
//! An absent document is not an error: a deployment that sets no key runs with
//! every secret unset, and a tool that needs one says so when it is called.
//! A *malformed* one is an operator error and is reported as such — including
//! an unknown field, so that `tavily_key:` fails loudly at load instead of
//! silently leaving search unconfigured.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SecretsError {
    #[error("{path} could not be read: {source}")]
    Unreadable {
        path: String,
        source: std::io::Error,
    },
    #[error("{path} is not valid secrets YAML: {source}")]
    Malformed {
        path: String,
        source: serde_yaml::Error,
    },
}

/// The parsed secrets document.
///
/// Fields are private and reached through accessors so that a value can be
/// normalised on the way out — see [`ToolSecrets::tavily_api_key`].
#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ToolSecrets {
    /// Tavily API key for `web_search`. Obtained from tavily.com.
    #[serde(default)]
    tavily_api_key: Option<String>,
}

impl ToolSecrets {
    /// The document's path, relative to the daemon's working directory.
    ///
    /// The leading segment is `secrets`, which is exactly what
    /// [`VfsStore`](crate::state::VfsStore) refuses to serve — the two are the
    /// same decision and must move together.
    pub const RELATIVE_PATH: &'static str = "secrets/tools.yaml";

    /// A document with nothing set — what an absent file parses to, and what
    /// every test and non-daemon caller gets.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The Tavily key, or `None` when the document does not usefully set one.
    ///
    /// Blank counts as unset. An operator who empties the value to turn search
    /// off means the same thing as one who deletes the line, and without this an
    /// empty string would be sent to Tavily as a real request and come back as
    /// an opaque `HTTP 401` rather than as "no key configured".
    pub fn tavily_api_key(&self) -> Option<&str> {
        self.tavily_api_key
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }

    /// Where the document sits under `workspace`.
    pub fn path_in(workspace: &Path) -> PathBuf {
        workspace.join(Self::RELATIVE_PATH)
    }

    /// Parse a document. `path` names the file in any error.
    pub fn from_yaml(text: &str, path: &str) -> Result<Self, SecretsError> {
        // A file holding only comments or whitespace is YAML `null`, which does
        // not deserialize into a struct — and "I wrote the file but have not
        // filled it in yet" is plainly an empty set of secrets, not an error.
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_yaml::from_str(text).map_err(|source| SecretsError::Malformed {
            path: path.to_string(),
            source,
        })
    }

    /// Read the document at `path`.
    ///
    /// An absent file is an empty set of secrets, because not configuring web
    /// search is an ordinary way to run the daemon. Every other I/O failure is
    /// reported: a file that exists and cannot be read is a machine that needs
    /// attention, and silently behaving as though it held nothing would hide it.
    pub fn load(path: &Path) -> Result<Self, SecretsError> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => {
                return Err(SecretsError::Unreadable {
                    path: path.display().to_string(),
                    source,
                })
            }
        };
        Self::from_yaml(&text, &path.display().to_string())
    }

    /// Read the document belonging to `workspace`.
    pub fn load_from_workspace(workspace: &Path) -> Result<Self, SecretsError> {
        Self::load(&Self::path_in(workspace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ordinary configured case, asserted against the literal key text.
    #[test]
    fn a_document_supplies_the_key() {
        let s = ToolSecrets::from_yaml("tavily_api_key: tvly-dev-abc123\n", "t.yaml").unwrap();
        assert_eq!(s.tavily_api_key(), Some("tvly-dev-abc123"));
    }

    /// Surrounding whitespace is not part of the key. A value pasted with a
    /// trailing space would otherwise be sent to Tavily verbatim and rejected.
    #[test]
    fn the_key_is_trimmed() {
        let s = ToolSecrets::from_yaml("tavily_api_key: \"  tvly-x  \"\n", "t.yaml").unwrap();
        assert_eq!(s.tavily_api_key(), Some("tvly-x"));
    }

    /// Blank means unset, so the tool reports "no key" rather than sending one.
    #[test]
    fn a_blank_value_is_unset() {
        for doc in ["tavily_api_key: \"\"\n", "tavily_api_key: \"   \"\n"] {
            let s = ToolSecrets::from_yaml(doc, "t.yaml").unwrap();
            assert_eq!(s.tavily_api_key(), None, "{doc:?} should read as unset");
        }
    }

    /// A document that is only comments is an empty set, not a parse failure.
    #[test]
    fn an_empty_document_is_an_empty_set() {
        for doc in ["", "   \n", "# nothing set yet\n"] {
            let s = ToolSecrets::from_yaml(doc, "t.yaml").unwrap();
            assert_eq!(s, ToolSecrets::empty(), "{doc:?} should parse as empty");
            assert_eq!(s.tavily_api_key(), None);
        }
    }

    /// **A typo must not read as "unconfigured".** Without `deny_unknown_fields`
    /// a misspelled key deserializes to a default `ToolSecrets`, and the
    /// operator sees "no tavily_api_key" while looking straight at a file that
    /// appears to set one.
    #[test]
    fn an_unknown_field_is_an_error() {
        let e = ToolSecrets::from_yaml("tavily_key: tvly-x\n", "t.yaml").unwrap_err();
        let msg = e.to_string();
        assert!(
            msg.contains("t.yaml"),
            "the error should name the file: {msg}"
        );
        assert!(
            msg.contains("tavily_key"),
            "the error should name the unknown field: {msg}"
        );
    }

    /// Malformed YAML names the file it came from.
    #[test]
    fn malformed_yaml_names_the_file() {
        let e = ToolSecrets::from_yaml("tavily_api_key: [unclosed\n", "secrets.yaml").unwrap_err();
        assert!(matches!(e, SecretsError::Malformed { .. }));
        assert!(e.to_string().contains("secrets.yaml"));
    }

    /// An absent file is an empty set — the ordinary case for a deployment that
    /// has not configured search — and reaches no error path.
    #[test]
    fn an_absent_file_is_an_empty_set() {
        let dir = tempfile::tempdir().unwrap();
        let s = ToolSecrets::load_from_workspace(dir.path()).unwrap();
        assert_eq!(s, ToolSecrets::empty());
        assert_eq!(s.tavily_api_key(), None);
    }

    /// A file that is present is read from the workspace-relative location.
    #[test]
    fn a_present_file_is_read_from_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let path = ToolSecrets::path_in(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "tavily_api_key: tvly-from-disk\n").unwrap();
        let s = ToolSecrets::load_from_workspace(dir.path()).unwrap();
        assert_eq!(s.tavily_api_key(), Some("tvly-from-disk"));
    }

    /// A directory where a file was expected is reported, not silently treated
    /// as unconfigured: something is wrong with the machine and the operator
    /// needs to hear about it.
    #[test]
    fn an_unreadable_path_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let e = ToolSecrets::load(dir.path()).unwrap_err();
        assert!(matches!(e, SecretsError::Unreadable { .. }), "{e}");
    }

    /// **The document's path starts with the segment the VFS refuses.** The
    /// whole reason an in-repository secret is safe is that `file_read` will not
    /// serve it, and that refusal keys off the leading `secrets` segment. If
    /// this constant ever moves out from under it, the file becomes readable by
    /// the model and nothing else in the build would notice.
    #[test]
    fn the_document_sits_under_the_protected_segment() {
        assert_eq!(
            ToolSecrets::RELATIVE_PATH
                .split('/')
                .next()
                .expect("a relative path has a first segment"),
            crate::state::vfs::PROTECTED_SEGMENT,
        );
    }
}
