//! Which secrets file the daemon reads, and reading it once at launch.
//!
//! The file is `~/.zend/secrets.yaml` for the user running the daemon
//! ([`Secrets::default_path`]), or the one `--secrets <path>` names for an
//! operator who keeps it elsewhere. What the document holds and how it is
//! protected is [`zend_tools::state::secrets`]'s concern; this module decides
//! which file, and how a problem with it affects the launch.
//!
//! - **A named file that does not exist fails the launch.** The operator asked
//!   for that file, so its absence is a typo in the command, and running on with
//!   every secret unset would hide the typo until a tool needed a key.
//! - **An absent default is ordinary.** A machine that configures no service
//!   runs with every secret unset, and a tool that needs one names the file.
//! - **A malformed or exposed file is a WARN, not a failed launch.** Refusing
//!   to boot the model over a stray character in a key file would be out of
//!   proportion; every secret reads as unset and the log says why.
//!
//! Only the *presence* of each key is logged, never its value — a log line is
//! the one place a secret reliably escapes a process.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::bail;
use zend_tools::state::Secrets;

/// Read the daemon's secrets: from `explicit` when `--secrets` named a file,
/// otherwise from the default under the home folder.
pub fn load(explicit: Option<&Path>) -> anyhow::Result<Arc<Secrets>> {
    Ok(Arc::new(load_from(explicit, Secrets::default_path())?))
}

/// [`load`] with the default path supplied, so the choice is testable without
/// the real home folder.
fn load_from(explicit: Option<&Path>, default: Option<PathBuf>) -> anyhow::Result<Secrets> {
    let path = match explicit {
        Some(path) => {
            if !path.is_file() {
                bail!(
                    "--secrets {}: no such file (omit the flag to use ~/{})",
                    path.display(),
                    Secrets::DEFAULT_RELATIVE_PATH
                );
            }
            path.to_path_buf()
        }
        None => match default {
            Some(path) => path,
            None => {
                tracing::warn!(
                    "no home folder to find ~/{} in, and no --secrets given; every secret \
                     reads as unset",
                    Secrets::DEFAULT_RELATIVE_PATH
                );
                return Ok(Secrets::empty());
            }
        },
    };
    match Secrets::load(&path) {
        Ok(secrets) => {
            tracing::info!(
                path = %path.display(),
                github = secrets.github_token().is_some(),
                tavily = secrets.tavily_api_key().is_some(),
                "secrets loaded"
            );
            Ok(secrets)
        }
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "secrets could not be read; every secret reads as unset"
            );
            Ok(Secrets::empty())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dir/name` holding `text`, owner-only on Unix so the loader accepts it.
    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        path
    }

    /// **`--secrets` wins over the default**, and the default is not read.
    #[test]
    fn a_named_file_is_read_instead_of_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let default = write(
            dir.path(),
            "home/.zend/secrets.yaml",
            "tavily_api_key: tvly-home\n",
        );
        let named = write(dir.path(), "elsewhere.yaml", "tavily_api_key: tvly-named\n");
        let s = load_from(Some(&named), Some(default)).unwrap();
        assert_eq!(s.tavily_api_key(), Some("tvly-named"));
        assert_eq!(s.source(), Some(named.as_path()));
    }

    /// With no flag, the default under the home folder is read.
    #[test]
    fn without_a_flag_the_default_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let default = write(
            dir.path(),
            "home/.zend/secrets.yaml",
            "github_token: ghp_home\n",
        );
        let s = load_from(None, Some(default.clone())).unwrap();
        assert_eq!(s.github_token(), Some("ghp_home"));
        assert_eq!(s.source(), Some(default.as_path()));
    }

    /// **A named file that does not exist fails the launch**, naming it.
    #[test]
    fn a_named_file_that_does_not_exist_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("typo.yaml");
        let e = load_from(Some(&missing), None).unwrap_err().to_string();
        assert!(e.contains("typo.yaml"), "{e}");
        assert!(e.contains("no such file"), "{e}");
    }

    /// An absent default is an ordinary unconfigured launch that still names
    /// the file to create.
    #[test]
    fn an_absent_default_is_unset_and_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let default = dir.path().join("home/.zend/secrets.yaml");
        let s = load_from(None, Some(default.clone())).unwrap();
        assert_eq!(s.tavily_api_key(), None);
        assert_eq!(s.source(), Some(default.as_path()));
    }

    /// No home folder and no flag: every secret unset, launch continues.
    #[test]
    fn no_home_folder_is_unset() {
        let s = load_from(None, None).unwrap();
        assert_eq!(s.tavily_api_key(), None);
        assert_eq!(s.source(), None);
    }

    /// A malformed file does not fail the launch; every secret reads as unset.
    #[test]
    fn a_malformed_file_reads_as_unset() {
        let dir = tempfile::tempdir().unwrap();
        let named = write(dir.path(), "bad.yaml", "tavily_key: tvly-typo\n");
        let s = load_from(Some(&named), None).unwrap();
        assert_eq!(s.tavily_api_key(), None);
    }
}
