//! The identity this repository's commits are made under.
//!
//! Read from git's own configuration rather than passed in by the caller, so
//! a commit the layer writes carries the same name and email a commit the
//! developer makes by hand would. A repository with no configured identity is
//! an error and not a default: git itself refuses to commit without one, and
//! inventing a placeholder would put a fictitious author into permanent
//! history.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::GitError;
use crate::library::config;
use crate::runner::utf8;
use crate::types::{GitTime, Signature};
use crate::Repo;

impl Repo {
    /// The configured `user.name` and `user.email`, either `None` when unset,
    /// from one `config` process rather than one each.
    ///
    /// `--get-regexp` prints every entry in the order git reads its files, so
    /// the last one for a key is the value `--get` would have answered: the one
    /// that wins. No `--end-of-options`: older releases' `config` rejects it,
    /// and the pattern is a fixed literal from this file.
    fn configured_identity(&self) -> Result<(Option<String>, Option<String>), GitError> {
        if let Some(lib) = self.library() {
            return config::identity(&lib);
        }
        let out = self
            .git("config")
            .args(["-z", "--get-regexp", r"^user\.(name|email)$"])
            .read_only()
            .run_accepting(&[0, 1])?;
        let text = utf8("config", out.stdout)?;
        let (mut name, mut email) = (None, None);
        for entry in text.split('\0').filter(|e| !e.is_empty()) {
            let (key, value) = entry.split_once('\n').unwrap_or((entry, ""));
            match key {
                "user.name" => name = Some(value.trim().to_string()),
                "user.email" => email = Some(value.trim().to_string()),
                _ => {}
            }
        }
        Ok((name, email))
    }

    /// The repository's configured `user.name` and `user.email`, stamped with
    /// the current time in UTC.
    ///
    /// The offset is `+0000` rather than the machine's local zone: the daemon
    /// may be running anywhere, and a commit timestamped in the server's zone
    /// tells the reader nothing true about when the author was working.
    pub fn identity(&self) -> Result<Signature, GitError> {
        let missing = |key: &str| {
            GitError::invalid(format!(
                "this repository has no {key} configured, so there is no identity to \
                 commit under; set it with `git config {key} …`"
            ))
        };
        let (name, email) = self.configured_identity()?;
        let name = name
            .filter(|v| !v.is_empty())
            .ok_or_else(|| missing("user.name"))?;
        let email = email
            .filter(|v| !v.is_empty())
            .ok_or_else(|| missing("user.email"))?;
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Signature::new(
            &name,
            &email,
            GitTime {
                seconds,
                offset_minutes: 0,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::TestRepo;

    /// The configured identity is what comes back, verbatim.
    #[test]
    fn the_configured_name_and_email_are_read() {
        let t = TestRepo::init();
        let sig = t.repo().identity().unwrap();
        assert_eq!(sig.name(), "Setup");
        assert_eq!(sig.email(), "setup@example.com");
        assert_eq!(sig.when.offset_minutes, 0);
        assert!(sig.when.seconds > 1_700_000_000, "stamped with now");
    }

    /// **A repository with no usable identity is an error, never a
    /// placeholder**, and the error names the key that is missing so the
    /// caller can say what to set.
    ///
    /// The absence is staged as an empty repository-level value rather than
    /// by unsetting one. `config --get` reads the whole hierarchy, so
    /// unsetting the repository's value would fall through to whatever
    /// global identity the machine running the suite happens to have — the
    /// test would pass or fail on the developer's own git config. An empty
    /// local value shadows the global one, which is the same condition from
    /// this code's point of view and is the same thing git itself treats as
    /// usable while producing a commit with an empty author.
    #[test]
    fn a_repository_without_an_identity_refuses_rather_than_inventing_one() {
        let t = TestRepo::init();
        t.git(&["config", "user.email", ""]);
        let err = t.repo().identity().unwrap_err().to_string();
        assert!(err.contains("user.email"), "{err}");
        assert!(
            err.contains("git config"),
            "the error says how to fix it: {err}"
        );

        t.git(&["config", "user.name", ""]);
        let err = t.repo().identity().unwrap_err().to_string();
        assert!(err.contains("user.name"), "{err}");
    }
}
