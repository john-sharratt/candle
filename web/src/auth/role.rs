//! Who may do what.
//!
//! Three levels, and the whole estate makes its decisions with them:
//!
//! | Role | Who | May |
//! |---|---|---|
//! | [`Role::Unauthenticated`] | the gateway named nobody | read what is public |
//! | [`Role::User`] | signed in | everything above, plus their own characters |
//! | [`Role::Admin`] | named in the config | everything above, plus edit authored content |
//!
//! # Why a config file and not a database
//!
//! Being an admin is a property of the deployment, not of a user record. It is
//! decided by whoever can edit the config and restart the process, which is the
//! same person who can edit the files admin-ness grants power over — so putting
//! it in a database would add a way to grant it that is *weaker* than the thing
//! it protects. There is deliberately no API to promote anybody.
//!
//! # Why a principal is `sub` or `email`, spelled out
//!
//! [`Identity::sub`] is the account key everywhere else in this estate,
//! precisely because an email can be reassigned by a provider and a display
//! name is not unique. Keying admin on `sub` is therefore the durable answer —
//! and `108000000000000000000` in a config file is not a thing a human
//! maintains.
//!
//! So both are allowed and neither is guessed: an entry says which it is.
//!
//! ```yaml
//! roles:
//!   admins:
//!     - sub: "108000000000000000000"    # durable; survives an email change
//!     - email: someone@example.com      # readable; inherits the provider's
//!                                       # reassignment risk
//! ```
//!
//! An entry that matched "whichever field it looks like" would be a rule nobody
//! could hold in their head, and the failure would be silent in the direction
//! that grants access.

use serde::{Deserialize, Serialize};

use super::session::Identity;

/// What a caller is allowed to do, ordered least to most.
///
/// `Ord` is derived from the declaration order, which is what makes
/// `role >= Role::Admin` the whole of an access check. Adding a level between
/// two of these changes every comparison in the estate at once — deliberately,
/// because the alternative is a set of unrelated booleans that drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The gateway did not name this caller. Not an error — most of the console
    /// is readable without signing in.
    Unauthenticated,
    /// Signed in. Owns characters, owns a profile, owns nothing else.
    User,
    /// Named in the config's `roles.admins`. May change authored content on
    /// disk: worlds, personalities, and anything else the daemon writes into a
    /// mind.
    Admin,
}

impl Role {
    /// The wire and log spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Unauthenticated => "unauthenticated",
            Role::User => "user",
            Role::Admin => "admin",
        }
    }

    /// Whether this role clears a bar. Reads as the sentence the call site
    /// means, where `>=` on an enum reads as a puzzle.
    pub fn at_least(self, min: Role) -> bool {
        self >= min
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The two fields an admin entry may name, before it is checked.
///
/// This exists only because `serde_yaml` renders an externally-tagged enum as
/// a YAML **tag** (`- !sub "123"`) rather than a map, and a config a human
/// maintains should read as `- sub: "123"`. Deserialising through a struct and
/// validating gets the natural spelling and a better error than a tag mismatch.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrincipalFields {
    #[serde(default)]
    sub: Option<String>,
    #[serde(default)]
    email: Option<String>,
}

/// One way of naming a person in the config.
///
/// Exactly one field, checked at parse time: an entry naming both would have to
/// mean *either* or *both*, and a rule nobody can predict is worse in a file
/// that grants administrative access.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "PrincipalFields")]
pub enum Principal {
    /// The provider's subject id — the durable one.
    Sub(String),
    /// The provider-verified email address. Matched case-insensitively, because
    /// no provider treats `A@x.com` and `a@x.com` as different people.
    Email(String),
}

impl TryFrom<PrincipalFields> for Principal {
    type Error = String;

    fn try_from(f: PrincipalFields) -> std::result::Result<Self, String> {
        match (f.sub, f.email) {
            (Some(s), None) => Ok(Principal::Sub(s)),
            (None, Some(e)) => Ok(Principal::Email(e)),
            (Some(_), Some(_)) => Err("an admin entry names `sub` or `email`, never both".into()),
            (None, None) => Err("an admin entry must name `sub` or `email`".into()),
        }
    }
}

impl Principal {
    /// Whether this entry names that identity.
    ///
    /// An empty value never matches anything. The gateway omits a header whose
    /// value will not fit, so an identity can legitimately arrive with no email
    /// — and an `email: ""` entry in the config matching every such caller
    /// would hand out admin for a truncated header.
    fn matches(&self, id: &Identity) -> bool {
        match self {
            Principal::Sub(want) => !want.is_empty() && want == &id.sub,
            Principal::Email(want) => {
                let want = want.trim();
                !want.is_empty() && want.eq_ignore_ascii_case(id.email.trim())
            }
        }
    }
}

/// The estate's role table, as configured.
///
/// Empty by default, which means there are no admins and nothing on disk can be
/// edited through the API. That is the right default for a deployment that
/// forgot to configure this: read-only is a visible failure, and an implicit
/// admin is not.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Roles {
    #[serde(default)]
    pub admins: Vec<Principal>,
}

impl Roles {
    /// The role of a caller the gateway did or did not name.
    pub fn of(&self, id: Option<&Identity>) -> Role {
        let Some(id) = id else {
            return Role::Unauthenticated;
        };
        // A blank subject is not an identity at all — `identify` refuses one,
        // and this refuses to promote one in case another caller ever does not.
        if id.sub.is_empty() {
            return Role::Unauthenticated;
        }
        if self.admins.iter().any(|p| p.matches(id)) {
            Role::Admin
        } else {
            Role::User
        }
    }

    /// Whether anybody at all can edit authored content here. Worth logging at
    /// startup: a deployment with no admins is one where every save will be
    /// refused, and finding that out from a 403 is worse than from a line in
    /// the log.
    pub fn is_empty(&self) -> bool {
        self.admins.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(sub: &str, email: &str) -> Identity {
        Identity {
            provider: "google".into(),
            sub: sub.into(),
            email: email.into(),
            name: "Wren".into(),
            picture: String::new(),
            exp: u64::MAX,
        }
    }

    fn admins(yaml: &str) -> Roles {
        serde_yaml::from_str(yaml).expect("parses")
    }

    #[test]
    fn the_levels_are_ordered_so_a_comparison_is_the_whole_check() {
        assert!(Role::Admin > Role::User);
        assert!(Role::User > Role::Unauthenticated);
        assert!(Role::Admin.at_least(Role::User));
        assert!(!Role::User.at_least(Role::Admin));
        assert!(Role::Unauthenticated.at_least(Role::Unauthenticated));
    }

    #[test]
    fn nobody_is_an_admin_without_being_named() {
        let r = Roles::default();
        assert!(r.is_empty());
        assert_eq!(r.of(None), Role::Unauthenticated);
        assert_eq!(r.of(Some(&id("g1", "a@x.com"))), Role::User);
    }

    #[test]
    fn a_subject_entry_matches_only_that_subject() {
        let r = admins("admins:\n  - sub: '108000000000000000000'\n");
        assert_eq!(r.of(Some(&id("108000000000000000000", ""))), Role::Admin);
        assert_eq!(r.of(Some(&id("111930197703828817753", ""))), Role::User);
        // The subject is opaque and case-sensitive; a provider that emits
        // base64 can have two different people differing only in case.
        let r = admins("admins:\n  - sub: 'AbC'\n");
        assert_eq!(r.of(Some(&id("abc", ""))), Role::User);
    }

    #[test]
    fn an_email_entry_matches_case_insensitively_and_ignores_spacing() {
        let r = admins("admins:\n  - email: Someone@Example.COM\n");
        assert_eq!(r.of(Some(&id("g1", "someone@example.com"))), Role::Admin);
        assert_eq!(
            r.of(Some(&id("g1", "  SOMEONE@example.com  "))),
            Role::Admin
        );
        assert_eq!(
            r.of(Some(&id("g1", "someone.else@example.com"))),
            Role::User
        );
    }

    /// The gateway drops a header it cannot fit, so an identity with no email
    /// is a real thing that arrives. It must never match an entry.
    #[test]
    fn an_empty_value_on_either_side_never_grants_anything() {
        for cfg in ["admins:\n  - email: ''\n", "admins:\n  - sub: ''\n"] {
            let r = admins(cfg);
            assert_eq!(r.of(Some(&id("g1", "a@x.com"))), Role::User, "{cfg}");
            assert_eq!(r.of(Some(&id("", ""))), Role::Unauthenticated, "{cfg}");
        }
        let r = admins("admins:\n  - email: someone@example.com\n");
        assert_eq!(r.of(Some(&id("g1", ""))), Role::User);
    }

    /// A caller with no identity is never anything but unauthenticated, however
    /// the table is configured.
    #[test]
    fn an_anonymous_caller_cannot_be_promoted() {
        let r = admins("admins:\n  - sub: g1\n  - email: a@x.com\n");
        assert_eq!(r.of(None), Role::Unauthenticated);
        // Nor by presenting a blank subject with an admin's email.
        assert_eq!(r.of(Some(&id("", "a@x.com"))), Role::Unauthenticated);
    }

    #[test]
    fn either_form_grants_it_and_the_config_says_which_is_which() {
        let r = admins("admins:\n  - sub: g1\n  - email: b@x.com\n");
        assert_eq!(r.of(Some(&id("g1", "nothing@x.com"))), Role::Admin);
        assert_eq!(r.of(Some(&id("g2", "b@x.com"))), Role::Admin);
        assert_eq!(r.of(Some(&id("g2", "c@x.com"))), Role::User);
    }

    /// A typo in the key is a refused config, not a silently ignored entry that
    /// leaves somebody without the access they think they configured.
    #[test]
    fn an_unknown_principal_field_is_a_configuration_error() {
        for bad in [
            "admins:\n  - subject: g1\n",
            "admins:\n  - mail: a@x.com\n",
            "admins:\n  - sub: g1\n    email: a@x.com\n",
            "admin:\n  - sub: g1\n",
        ] {
            assert!(
                serde_yaml::from_str::<Roles>(bad).is_err(),
                "accepted `{bad}`"
            );
        }
    }

    #[test]
    fn the_wire_spelling_is_the_users_vocabulary() {
        assert_eq!(Role::Unauthenticated.to_string(), "unauthenticated");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Admin.to_string(), "admin");
        assert_eq!(serde_json::to_string(&Role::Admin).unwrap(), r#""admin""#);
    }
}
