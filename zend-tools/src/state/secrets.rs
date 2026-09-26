//! The daemon's secrets: the API keys and tokens its tools and its git layer
//! present to third-party services.
//!
//! # Why this is not the credential store
//!
//! [`CredentialStore`](crate::state::CredentialStore) holds what a *conversation*
//! saved: a bearer token handed to `credential_save`, an SSH key a session was
//! opened with. Those belong to the chat that created them and go away with it.
//! A Tavily key or a GitHub token is the opposite kind of thing — one per person
//! running the daemon, and needed on the first call of a brand new conversation.
//! Routing it through the credential store would mean pasting it into every chat.
//!
//! # Where it lives
//!
//! One YAML document per user, by default at
//! [`DEFAULT_RELATIVE_PATH`](Secrets::DEFAULT_RELATIVE_PATH) under the home
//! folder — `~/.zend/secrets.yaml` — beside the other per-user credentials a
//! developer machine holds (`~/.ssh`, `~/.gitconfig`). The daemon's
//! `--secrets <path>` names another file instead.
//!
//! ```yaml
//! # ~/.zend/secrets.yaml
//! github_token: ghp_...
//! tavily_api_key: tvly-...
//! ```
//!
//! The document sits outside every workspace, so no repository the `file_*`
//! tools mount can reach it, and the code sandbox reaches nothing but those
//! mounts. The VFS's refusal of any `secrets` path segment
//! ([`PROTECTED_SEGMENT`](crate::state::vfs::PROTECTED_SEGMENT)) stays as a
//! second guard for a secrets folder someone keeps inside a repository.
//!
//! A document that is not private to the daemon's user is refused, the rule
//! OpenSSH applies to a private key: a key the whole machine can read is
//! already leaked, and loading it quietly would hide that. What private
//! means on each platform is `secrets_exposure`'s concern.
//!
//! An absent document is not an error: a deployment that sets no key runs with
//! every secret unset, and a tool that needs one says so when it is called,
//! naming the file to edit. A *malformed* one is an operator error and is
//! reported as such — including an unknown field, so that `tavily_key:` fails
//! loudly at load instead of silently leaving search unconfigured.

use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use crate::state::secrets_exposure;

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
    #[error("{path} is not private to the daemon's user: it {reason}")]
    Exposed { path: String, reason: String },
}

/// The parsed secrets document.
///
/// Fields are private and reached through accessors so that a value can be
/// normalised on the way out — see [`Secrets::tavily_api_key`].
#[derive(Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Secrets {
    /// GitHub personal access token, for pushing branches and opening pull
    /// requests on a repository's origin. A classic token (`ghp_`) needs the
    /// `repo` scope; a fine-grained one (`github_pat_`) needs contents and pull
    /// request access on each repository.
    #[serde(default)]
    github_token: Option<String>,
    /// Tavily API key for `web_search`. Obtained from tavily.com.
    #[serde(default)]
    tavily_api_key: Option<String>,
    /// The file this document was read from, or would have been had it
    /// existed — the file an operator edits to set a missing secret. Not part
    /// of the document itself.
    #[serde(skip)]
    source: Option<PathBuf>,
}

/// Redacts every value — a derived `Debug` would print them verbatim into any
/// log line or panic message that formats this struct.
impl fmt::Debug for Secrets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redacted = |v: &Option<String>| v.as_ref().map(|_| "<redacted>");
        f.debug_struct("Secrets")
            .field("github_token", &redacted(&self.github_token))
            .field("tavily_api_key", &redacted(&self.tavily_api_key))
            .field("source", &self.source)
            .finish()
    }
}

/// A value, or `None` when it is absent or blank.
///
/// Blank counts as unset. An operator who empties a value to turn a feature
/// off means the same thing as one who deletes the line, and without this an
/// empty string would be sent to the service as a real credential and come
/// back as an opaque `HTTP 401` rather than as "not configured". Surrounding
/// whitespace is not part of a key: a value pasted with a trailing space would
/// otherwise be rejected by the service.
fn set(value: &Option<String>) -> Option<&str> {
    value.as_deref().map(str::trim).filter(|s| !s.is_empty())
}

impl Secrets {
    /// The default document's path, relative to the user's home folder.
    pub const DEFAULT_RELATIVE_PATH: &'static str = ".zend/secrets.yaml";

    /// A document with nothing set and no file behind it — what every test and
    /// non-daemon caller gets.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The GitHub token, or `None` when the document does not usefully set one.
    pub fn github_token(&self) -> Option<&str> {
        set(&self.github_token)
    }

    /// The Tavily key, or `None` when the document does not usefully set one.
    pub fn tavily_api_key(&self) -> Option<&str> {
        set(&self.tavily_api_key)
    }

    /// The file this document came from, or `None` for one the daemon did not
    /// read from disk ([`Self::empty`]).
    pub fn source(&self) -> Option<&Path> {
        self.source.as_deref()
    }

    /// Where the default document sits under `home`.
    ///
    /// Joined a component at a time, so the path is spelled with the
    /// platform's own separator throughout — joining the `/`-separated
    /// [`Self::DEFAULT_RELATIVE_PATH`] whole would log `C:\Users\x\.zend/secrets.yaml`.
    pub fn default_path_in(home: &Path) -> PathBuf {
        Self::DEFAULT_RELATIVE_PATH
            .split('/')
            .fold(home.to_path_buf(), |path, part| path.join(part))
    }

    /// The default document for the user running the daemon, or `None` when
    /// the platform reports no home folder.
    pub fn default_path() -> Option<PathBuf> {
        std::env::home_dir().map(|home| Self::default_path_in(&home))
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
    /// An absent file is an empty set of secrets that still records `path` as
    /// its source, because not configuring a service is an ordinary way to run
    /// the daemon and the refusal should name the file to create. Every other
    /// I/O failure is reported: a file that exists and cannot be read is a
    /// machine that needs attention, and behaving as though it held nothing
    /// would hide it.
    pub fn load(path: &Path) -> Result<Self, SecretsError> {
        let shown = path.display().to_string();
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                return Ok(Self {
                    source: Some(path.to_path_buf()),
                    ..Self::default()
                })
            }
            Err(source) => {
                return Err(SecretsError::Unreadable {
                    path: shown,
                    source,
                })
            }
        };
        secrets_exposure::check(path).map_err(|reason| SecretsError::Exposed {
            path: shown.clone(),
            reason,
        })?;
        let mut secrets = Self::from_yaml(&text, &shown)?;
        secrets.source = Some(path.to_path_buf());
        Ok(secrets)
    }
}

#[cfg(test)]
mod tests {
    use std::path::MAIN_SEPARATOR;

    use super::*;

    /// Write `text` as an owner-only secrets file in a fresh temp dir.
    fn file_with(text: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = Secrets::default_path_in(dir.path());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, text).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        }
        (dir, path)
    }

    /// The ordinary configured case, asserted against the literal key text.
    #[test]
    fn a_document_supplies_both_keys() {
        let s = Secrets::from_yaml(
            "github_token: ghp_abc123\ntavily_api_key: tvly-dev-abc123\n",
            "t.yaml",
        )
        .unwrap();
        assert_eq!(s.github_token(), Some("ghp_abc123"));
        assert_eq!(s.tavily_api_key(), Some("tvly-dev-abc123"));
    }

    /// Each key is independent: setting one leaves the other unset.
    #[test]
    fn one_key_set_leaves_the_other_unset() {
        let s = Secrets::from_yaml("github_token: ghp_only\n", "t.yaml").unwrap();
        assert_eq!(s.github_token(), Some("ghp_only"));
        assert_eq!(s.tavily_api_key(), None);
    }

    /// Surrounding whitespace is not part of a key.
    #[test]
    fn keys_are_trimmed() {
        let s = Secrets::from_yaml(
            "github_token: \" ghp_x \"\ntavily_api_key: \"  tvly-x  \"\n",
            "t.yaml",
        )
        .unwrap();
        assert_eq!(s.github_token(), Some("ghp_x"));
        assert_eq!(s.tavily_api_key(), Some("tvly-x"));
    }

    /// Blank means unset, so a tool reports "no key" rather than sending one.
    #[test]
    fn a_blank_value_is_unset() {
        for doc in [
            "tavily_api_key: \"\"\ngithub_token: \"\"\n",
            "tavily_api_key: \"   \"\ngithub_token: \"  \"\n",
        ] {
            let s = Secrets::from_yaml(doc, "t.yaml").unwrap();
            assert_eq!(s.tavily_api_key(), None, "{doc:?} should read as unset");
            assert_eq!(s.github_token(), None, "{doc:?} should read as unset");
        }
    }

    /// A document that is only comments is an empty set, not a parse failure.
    #[test]
    fn an_empty_document_is_an_empty_set() {
        for doc in ["", "   \n", "# nothing set yet\n"] {
            let s = Secrets::from_yaml(doc, "t.yaml").unwrap();
            assert_eq!(s, Secrets::empty(), "{doc:?} should parse as empty");
        }
    }

    /// **A typo must not read as "unconfigured".** Without `deny_unknown_fields`
    /// a misspelled key deserializes to an empty document, and the operator
    /// sees "no tavily_api_key" while looking at a file that appears to set one.
    #[test]
    fn an_unknown_field_is_an_error() {
        let e = Secrets::from_yaml("tavily_key: tvly-x\n", "t.yaml").unwrap_err();
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

    /// `source` is the loader's record, not a key the document may set.
    #[test]
    fn the_source_cannot_be_set_from_the_document() {
        let e = Secrets::from_yaml("source: /tmp/elsewhere.yaml\n", "t.yaml").unwrap_err();
        assert!(matches!(e, SecretsError::Malformed { .. }), "{e}");
    }

    /// Malformed YAML names the file it came from.
    #[test]
    fn malformed_yaml_names_the_file() {
        let e = Secrets::from_yaml("tavily_api_key: [unclosed\n", "secrets.yaml").unwrap_err();
        assert!(matches!(e, SecretsError::Malformed { .. }));
        assert!(e.to_string().contains("secrets.yaml"));
    }

    /// The default document is `.zend/secrets.yaml` under the home folder,
    /// spelled with the platform's separator throughout.
    #[test]
    fn the_default_document_is_under_the_home_folder() {
        let home = Path::new("home-dir");
        let path = Secrets::default_path_in(home);
        assert_eq!(path, home.join(".zend").join("secrets.yaml"));
        let sep = MAIN_SEPARATOR;
        assert_eq!(
            path.to_str().unwrap(),
            format!("home-dir{sep}.zend{sep}secrets.yaml")
        );
    }

    /// An absent file is an empty set that still names the file to create.
    #[test]
    fn an_absent_file_is_an_empty_set_that_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.yaml");
        let s = Secrets::load(&path).unwrap();
        assert_eq!(s.tavily_api_key(), None);
        assert_eq!(s.github_token(), None);
        assert_eq!(s.source(), Some(path.as_path()));
    }

    /// A present file is read, and records where it came from.
    #[test]
    fn a_present_file_is_read_and_records_its_source() {
        let (_dir, path) = file_with("tavily_api_key: tvly-from-disk\ngithub_token: ghp_disk\n");
        let s = Secrets::load(&path).unwrap();
        assert_eq!(s.tavily_api_key(), Some("tvly-from-disk"));
        assert_eq!(s.github_token(), Some("ghp_disk"));
        assert_eq!(s.source(), Some(path.as_path()));
    }

    /// A directory where a file was expected is reported, not silently treated
    /// as unconfigured.
    #[test]
    fn an_unreadable_path_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let e = Secrets::load(dir.path()).unwrap_err();
        assert!(matches!(e, SecretsError::Unreadable { .. }), "{e}");
    }

    /// **A document other users can read is refused**, whatever it holds.
    #[cfg(unix)]
    #[test]
    fn a_group_or_world_readable_document_is_refused() {
        use std::os::unix::fs::PermissionsExt;

        for mode in [0o644, 0o640, 0o604, 0o660] {
            let (_dir, path) = file_with("tavily_api_key: tvly-x\n");
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
            let e = Secrets::load(&path).unwrap_err();
            assert!(
                matches!(e, SecretsError::Exposed { .. }),
                "mode {mode:o}: {e}"
            );
            assert!(!e.to_string().contains("tvly-x"), "{e}");
        }
    }

    /// `Debug` must never print a raw value — a log line or panic message that
    /// formats a `Secrets` would otherwise put a live credential in the logs.
    #[test]
    fn debug_output_never_contains_a_key() {
        let s = Secrets::from_yaml(
            "tavily_api_key: tvly-dev-abc123\ngithub_token: ghp_secretvalue\n",
            "t.yaml",
        )
        .unwrap();
        let debug = format!("{s:?}");
        for key in ["tvly-dev-abc123", "ghp_secretvalue"] {
            assert!(!debug.contains(key), "{debug:?} leaked {key}");
        }
        assert!(
            debug.contains("redacted"),
            "{debug:?} should say a field is set"
        );
    }
}
