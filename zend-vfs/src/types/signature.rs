//! Commit identities and timestamps.

use crate::error::GitError;

/// A point in time as git records it: seconds since the epoch and the
/// author's UTC offset in minutes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GitTime {
    pub seconds: i64,
    pub offset_minutes: i32,
}

impl GitTime {
    /// Git's raw form, `1700000000 +0130`.
    pub fn to_raw(self) -> String {
        let sign = if self.offset_minutes < 0 { '-' } else { '+' };
        let abs = self.offset_minutes.unsigned_abs();
        format!("{} {sign}{:02}{:02}", self.seconds, abs / 60, abs % 60)
    }

    /// Parse git's raw form.
    pub fn parse_raw(raw: &str) -> Result<Self, GitError> {
        let bad = || GitError::invalid(format!("{raw:?} is not a raw git date"));
        let (secs, tz) = raw.split_once(' ').ok_or_else(bad)?;
        let seconds: i64 = secs.parse().map_err(|_| bad())?;
        let (sign, digits) = match tz.as_bytes().first() {
            Some(b'+') => (1, &tz[1..]),
            Some(b'-') => (-1, &tz[1..]),
            _ => return Err(bad()),
        };
        if digits.len() != 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }
        let hours: i32 = digits[..2].parse().map_err(|_| bad())?;
        let minutes: i32 = digits[2..].parse().map_err(|_| bad())?;
        Ok(Self {
            seconds,
            offset_minutes: sign * (hours * 60 + minutes),
        })
    }
}

/// Who made a commit, and when. Every commit the layer writes names its
/// author and committer explicitly; nothing falls back to the user's
/// configured identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    name: String,
    email: String,
    pub when: GitTime,
}

impl Signature {
    /// A signature as an existing commit records it, taken without judging
    /// it. Imported histories carry `Name <>` and worse; a read must still
    /// show them, so the rules of [`Self::new`] apply only to what the layer
    /// writes.
    pub(crate) fn recorded(name: &str, email: &str, when: GitTime) -> Self {
        Self {
            name: name.to_string(),
            email: email.to_string(),
            when,
        }
    }

    /// Refuses a name or email git could not record intact: `<`, `>` and
    /// control characters would break the header line.
    pub fn new(name: &str, email: &str, when: GitTime) -> Result<Self, GitError> {
        for (what, value) in [("name", name), ("email", email)] {
            if value.trim().is_empty()
                || value
                    .chars()
                    .any(|c| c == '<' || c == '>' || c.is_control())
            {
                return Err(GitError::invalid(format!(
                    "signature {what} {value:?} is empty or contains `<`, `>` or a control character"
                )));
            }
        }
        Ok(Self {
            name: name.to_string(),
            email: email.to_string(),
            when,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn email(&self) -> &str {
        &self.email
    }

    /// The identity as a commit header carries it: `Name <email> 1700000000 +0000`.
    pub fn to_header(&self) -> String {
        format!("{} <{}> {}", self.name, self.email, self.when.to_raw())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_times_round_trip_with_every_offset_sign() {
        for (raw, offset) in [
            ("1700000000 +0000", 0),
            ("1700000000 +0130", 90),
            ("1700000000 -0800", -480),
        ] {
            let t = GitTime::parse_raw(raw).unwrap();
            assert_eq!(t.offset_minutes, offset);
            assert_eq!(t.to_raw(), raw);
        }
    }

    #[test]
    fn malformed_raw_times_are_refused() {
        for bad in [
            "1700000000",
            "x +0000",
            "1700000000 0000",
            "1700000000 +000",
        ] {
            assert!(GitTime::parse_raw(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_signature_prints_as_a_commit_header() {
        let when = GitTime::parse_raw("1700000000 +0100").unwrap();
        let sig = Signature::new("Ada Lovelace", "ada@example.com", when).unwrap();
        assert_eq!(
            sig.to_header(),
            "Ada Lovelace <ada@example.com> 1700000000 +0100"
        );
    }

    #[test]
    fn a_signature_that_would_break_the_header_is_refused() {
        let when = GitTime {
            seconds: 0,
            offset_minutes: 0,
        };
        for (name, email) in [
            ("", "a@b"),
            ("A", ""),
            ("A <x>", "a@b"),
            ("A", "a@b>"),
            ("A\nB", "a@b"),
        ] {
            assert!(
                Signature::new(name, email, when).is_err(),
                "{name:?} {email:?}"
            );
        }
    }
}
