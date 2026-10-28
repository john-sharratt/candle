//! Who may do what.
//!
//! Five levels, and the whole estate makes its decisions with them:
//!
//! | Role | Who | May |
//! |---|---|---|
//! | [`Role::Unauthenticated`] | the gateway named nobody | read what is public |
//! | [`Role::Guest`] | signed in, not in the table | the same, and their own account (`docs/guest_access.md`) |
//! | [`Role::User`] | in the table as `user` | everything above, plus their own characters, zend and `/mcp` |
//! | [`Role::Admin`] | in the table as `admin` | everything above, plus edit authored content |
//! | [`Role::Creator`] | in the table as `creator` | everything above, and is known inside the fiction |
//!
//! # The table is an allowlist
//!
//! Signing in proves who someone is; it grants nothing. Every capability from
//! [`Role::User`] up belongs to the people the table names, each with the role
//! written beside them, and anyone else who signs in is a [`Role::Guest`]. So a
//! route whose minimum is `User` refuses a stranger with a Google account
//! without knowing guests exist.
//!
//! # The Creator is the one level that is also a *character*
//!
//! Every other role here answers only "may this request proceed". The Creator
//! answers that and one more thing: worlds built by this estate are worlds whose
//! inhabitants were made, and the person who made them is somebody they can be
//! addressed by. So the role is read in two places that no other role reaches —
//! the name a person carries on a thread inside the world, and their membership
//! of the world's open channel.
//!
//! It sits above [`Role::Admin`] rather than beside it because everything an
//! admin may do to authored content, the person who authored it may also do.
//! Placing it above is what keeps every existing `at_least(Role::Admin)` check
//! correct without being rewritten — which is the property the ordering exists
//! for.
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
//! people:
//!   - sub: "108000000000000000000"    # durable; survives an email change
//!     role: admin
//!   - email: someone@example.com      # readable; inherits the provider's
//!     role: user                      # reassignment risk
//! ```
//!
//! An entry that matched "whichever field it looks like" would be a rule nobody
//! could hold in their head, and the failure would be silent in the direction
//! that grants access.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    /// Signed in, and not in the table: who they are is known, and that is
    /// all. Owns their account and nothing that runs the model.
    Guest,
    /// In the table as `user`. Owns characters, a profile, zend conversations
    /// and `/mcp` access.
    User,
    /// In the table as `admin`. May change authored content on disk: worlds,
    /// personalities, and anything else the daemon writes into a mind.
    Admin,
    /// In the table as `creator`. Everything an admin may do, and the one role
    /// the fiction itself knows about — see the module header.
    ///
    /// **Deliberately the last variant.** `Ord` comes from the declaration
    /// order, so being last is what makes a creator clear every bar an admin
    /// clears. A variant inserted before `Admin` instead would silently demote
    /// the person who owns the estate.
    Creator,
}

impl Role {
    /// The wire and log spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Unauthenticated => "unauthenticated",
            Role::Guest => "guest",
            Role::User => "user",
            Role::Admin => "admin",
            Role::Creator => "creator",
        }
    }

    /// Whether this caller is known to the worlds themselves.
    ///
    /// Read where a person's *name inside the fiction* is decided, rather than
    /// where access is decided — the one question `at_least` cannot answer,
    /// because it is not about a bar being cleared. See the module header.
    pub fn is_creator(self) -> bool {
        self == Role::Creator
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

/// The fields one entry of the table may name, before it is checked.
///
/// This exists only because `serde_yaml` renders an externally-tagged enum as
/// a YAML **tag** (`- !sub "123"`) rather than a map, and a config a human
/// maintains should read as `- sub: "123"`. Deserialising through a struct and
/// validating gets the natural spelling and a better error than a tag mismatch.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PersonFields {
    #[serde(default)]
    sub: Option<String>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    email_sha256: Option<String>,
    role: Role,
}

/// The SHA-256 of `email` as the table names it — trimmed and lowercased,
/// since no provider treats `A@x.com` and `a@x.com` as different people — in
/// lowercase hex: what an `email_sha256` entry holds.
pub fn email_sha256(email: &str) -> String {
    Sha256::digest(email.trim().to_ascii_lowercase().as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One way of naming a person in the table.
///
/// Exactly one field, checked at parse time: an entry naming both would have to
/// mean *either* or *both*, and a rule nobody can predict is worse in a file
/// that grants administrative access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Principal {
    /// The provider's subject id — the durable one.
    Sub(String),
    /// The provider-verified email address. Matched case-insensitively, because
    /// no provider treats `A@x.com` and `a@x.com` as different people.
    Email(String),
    /// The SHA-256 of the email address ([`email_sha256`]), in lowercase hex:
    /// how a table committed somewhere public names a person without
    /// publishing their address. It hides the address from a reader, not from
    /// someone who already suspects it and hashes their guess.
    EmailSha256(String),
}

/// One person the table names, and the role written beside them.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "PersonFields")]
pub struct Person {
    pub who: Principal,
    pub role: Role,
}

impl TryFrom<PersonFields> for Person {
    type Error = String;

    fn try_from(f: PersonFields) -> std::result::Result<Self, String> {
        let who = match (f.sub, f.email, f.email_sha256) {
            (Some(s), None, None) => Principal::Sub(s),
            (None, Some(e), None) => Principal::Email(e),
            (None, None, Some(h)) => {
                let h = h.trim().to_ascii_lowercase();
                if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(format!("`email_sha256: {h}` is not 64 hex digits"));
                }
                Principal::EmailSha256(h)
            }
            (None, None, None) => {
                return Err("an entry must name `sub`, `email` or `email_sha256`".into())
            }
            _ => return Err("an entry names one of `sub`, `email` or `email_sha256`".into()),
        };
        // `guest` and `unauthenticated` are what the table's absence means;
        // written into it they would name someone in order to grant nothing.
        if !f.role.at_least(Role::User) {
            return Err(format!(
                "an entry's role is `user`, `admin` or `creator`, not `{}`",
                f.role
            ));
        }
        Ok(Person { who, role: f.role })
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
            // An identity with no email has nothing to hash: the hash of the
            // empty string must not name it.
            Principal::EmailSha256(want) => {
                !id.email.trim().is_empty() && *want == email_sha256(&id.email)
            }
        }
    }

    /// Whether this and `other` name the same person as written: the same
    /// subject, or the same email — however it is cased and spaced, and
    /// whether written out or hashed.
    fn same_as(&self, other: &Principal) -> bool {
        match (self, other) {
            (Principal::Sub(a), Principal::Sub(b)) => a == b,
            (Principal::Sub(_), _) | (_, Principal::Sub(_)) => false,
            (a, b) => a.email_hash() == b.email_hash(),
        }
    }

    /// The email this names, hashed as [`email_sha256`] hashes it; `None` for
    /// a subject.
    fn email_hash(&self) -> Option<String> {
        match self {
            Principal::Sub(_) => None,
            Principal::Email(e) => Some(email_sha256(e)),
            Principal::EmailSha256(h) => Some(h.clone()),
        }
    }
}

/// The table's fields, before the people in it are checked against each other.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RolesFields {
    #[serde(default)]
    people: Vec<Person>,
}

/// The estate's role table, as configured: the allowlist, each person with a
/// role.
///
/// Empty by default, which means nobody is more than a guest and nothing on
/// disk can be edited through the API. That is the right default for a
/// deployment that forgot to configure this: read-only is a visible failure,
/// and an implicit admin is not.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(try_from = "RolesFields")]
pub struct Roles {
    people: Vec<Person>,
}

impl TryFrom<RolesFields> for Roles {
    type Error = String;

    fn try_from(f: RolesFields) -> std::result::Result<Self, String> {
        Roles::new(f.people)
    }
}

impl Roles {
    /// The table naming `people`. A person named twice is refused rather than
    /// resolved by order: two entries for one person with two roles is a
    /// question the file has not answered, and either answer is somebody's
    /// surprise.
    pub fn new(people: Vec<Person>) -> std::result::Result<Self, String> {
        for (i, p) in people.iter().enumerate() {
            if people[..i].iter().any(|q| q.who.same_as(&p.who)) {
                return Err(format!("{:?} is named twice", p.who));
            }
        }
        Ok(Roles { people })
    }

    /// The role of a caller the gateway did or did not name: the role the
    /// table gives them, or [`Role::Guest`] when it names them nowhere.
    ///
    /// **The highest of every entry that names them.** One person can be
    /// named twice in forms the load cannot compare — once by `sub`, once by
    /// `email` — and the first entry winning would let a line written above
    /// the owner's demote them in silence.
    pub fn of(&self, id: Option<&Identity>) -> Role {
        let Some(id) = id else {
            return Role::Unauthenticated;
        };
        // A blank subject is not an identity at all — `identify` refuses one,
        // and this refuses to promote one in case another caller ever does not.
        if id.sub.is_empty() {
            return Role::Unauthenticated;
        }
        self.people
            .iter()
            .filter(|p| p.who.matches(id))
            .map(|p| p.role)
            .max()
            .unwrap_or(Role::Guest)
    }

    /// Whether nobody here can edit authored content — nobody at `admin` or
    /// above. Worth logging at startup: a deployment like that refuses every
    /// save, and finding that out from a 403 is worse than from a line in the
    /// log.
    pub fn names_no_admin(&self) -> bool {
        !self.people.iter().any(|p| p.role.at_least(Role::Admin))
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

    fn table(yaml: &str) -> Roles {
        serde_yaml::from_str(yaml).expect("parses")
    }

    /// A table naming one person, by `field` (`sub` or `email`), as `role`.
    fn one(field: &str, value: &str, role: &str) -> Roles {
        table(&format!(
            "people:\n  - {field}: '{value}'\n    role: {role}\n"
        ))
    }

    #[test]
    fn the_levels_are_ordered_so_a_comparison_is_the_whole_check() {
        assert!(Role::Creator > Role::Admin);
        assert!(Role::Admin > Role::User);
        assert!(Role::User > Role::Guest);
        assert!(Role::Guest > Role::Unauthenticated);
        assert!(
            !Role::Guest.at_least(Role::User),
            "a guest clears no user bar"
        );
        assert!(Role::Admin.at_least(Role::User));
        assert!(!Role::User.at_least(Role::Admin));
        assert!(Role::Unauthenticated.at_least(Role::Unauthenticated));
    }

    /// **A creator clears every bar an admin clears.**
    ///
    /// The whole reason the level was added on top rather than beside: every
    /// `at_least(Role::Admin)` already written across the estate has to keep
    /// admitting the person who owns it, without one of them being revisited.
    #[test]
    fn a_creator_clears_every_bar_an_admin_does() {
        for bar in [Role::Unauthenticated, Role::Guest, Role::User, Role::Admin] {
            assert!(
                Role::Creator.at_least(bar),
                "a creator was refused a bar an admin clears: {bar}"
            );
        }
        assert!(!Role::Admin.at_least(Role::Creator));
    }

    /// **Each person gets the role written beside them; anyone else signed in
    /// is a guest.**
    #[test]
    fn the_table_gives_each_person_their_role_and_everyone_else_guest() {
        let r = table(
            "people:\n\
             \x20 - email: owner@example.com\n\
             \x20   role: creator\n\
             \x20 - email: ops@example.com\n\
             \x20   role: admin\n\
             \x20 - email: friend@example.com\n\
             \x20   role: user\n",
        );
        assert_eq!(r.of(Some(&id("g1", "owner@example.com"))), Role::Creator);
        assert_eq!(r.of(Some(&id("g2", "ops@example.com"))), Role::Admin);
        assert_eq!(r.of(Some(&id("g3", "friend@example.com"))), Role::User);
        assert_eq!(r.of(Some(&id("g4", "stranger@example.com"))), Role::Guest);
        assert_eq!(r.of(None), Role::Unauthenticated);
    }

    /// **A person named twice is refused**, however the two entries spell
    /// them: two roles for one person is a question the file has not
    /// answered.
    #[test]
    fn a_person_named_twice_is_refused() {
        for twice in [
            "people:\n  - email: me@example.com\n    role: admin\n  - email: me@example.com\n    role: creator\n",
            "people:\n  - email: me@example.com\n    role: user\n  - email: ' ME@example.com'\n    role: user\n",
            "people:\n  - sub: g1\n    role: user\n  - sub: g1\n    role: admin\n",
        ] {
            let err = serde_yaml::from_str::<Roles>(twice).unwrap_err().to_string();
            assert!(err.contains("named twice"), "{twice}: {err}");
        }
    }

    /// **An email's hash is SHA-256 of the address trimmed and lowercased**, in
    /// lowercase hex — digests from `sha256sum`, so a table written with any
    /// tool names the person this one matches.
    #[test]
    fn an_email_hash_is_sha256_of_the_trimmed_lowercased_address() {
        let want = "478abec7430569163161dfea8513b8ce89d05f559456a26e945c66e1fe55a29d";
        assert_eq!(email_sha256("a@x.com"), want);
        assert_eq!(email_sha256("  A@X.com "), want);
    }

    /// **A hashed entry names the person whose address hashes to it**, and
    /// nobody else — not an identity with no email, whose empty address must
    /// not match the hash of the empty string.
    #[test]
    fn a_hashed_entry_names_its_person_and_nobody_else() {
        let r = table(
            "people:\n  - email_sha256: 8C2A47D3BDB8D3096A6479F53EAC3B724291DB5F1C31611100F675BE5537329D\n    role: creator\n",
        );
        assert_eq!(r.of(Some(&id("g1", "Me@Example.com"))), Role::Creator);
        assert_eq!(r.of(Some(&id("g2", "other@example.com"))), Role::Guest);
        let empty = table(&format!(
            "people:\n  - email_sha256: {}\n    role: admin\n",
            email_sha256("")
        ));
        assert_eq!(empty.of(Some(&id("g3", ""))), Role::Guest);
    }

    /// **One person named once by `email` and once by `email_sha256` is named
    /// twice**, and a hash that is not one is refused.
    #[test]
    fn a_hash_and_its_address_are_the_same_person() {
        let twice = format!(
            "people:\n  - email: me@example.com\n    role: user\n  - email_sha256: {}\n    role: admin\n",
            email_sha256("me@example.com")
        );
        let err = serde_yaml::from_str::<Roles>(&twice)
            .unwrap_err()
            .to_string();
        assert!(err.contains("named twice"), "{err}");
        for bad in [
            "people:\n  - email_sha256: abc\n    role: user\n",
            "people:\n  - email_sha256: zz\n    role: user\n",
            "people:\n  - email: a@x.com\n    email_sha256: 478abec7430569163161dfea8513b8ce89d05f559456a26e945c66e1fe55a29d\n    role: user\n",
        ] {
            assert!(serde_yaml::from_str::<Roles>(bad).is_err(), "{bad}");
        }
    }

    /// **One person named by `sub` and by `email` gets the higher role**,
    /// whichever line comes first: the load cannot tell the two are one
    /// person, and the owner must not be demoted by a line above theirs.
    #[test]
    fn a_person_named_by_sub_and_email_gets_the_higher_role() {
        for yaml in [
            "people:\n  - sub: g1\n    role: user\n  - email: me@x.com\n    role: creator\n",
            "people:\n  - email: me@x.com\n    role: creator\n  - sub: g1\n    role: user\n",
        ] {
            assert_eq!(table(yaml).of(Some(&id("g1", "me@x.com"))), Role::Creator);
        }
    }

    /// **A table names a role a person can be granted**: `guest` and
    /// `unauthenticated` are what being absent from it means.
    #[test]
    fn an_entry_names_user_admin_or_creator() {
        for role in ["guest", "unauthenticated", "owner"] {
            let yaml = format!("people:\n  - email: a@x.com\n    role: {role}\n");
            assert!(serde_yaml::from_str::<Roles>(&yaml).is_err(), "{role}");
        }
        let no_role = "people:\n  - email: a@x.com\n";
        assert!(serde_yaml::from_str::<Roles>(no_role).is_err());
    }

    /// A deployment whose only principal is a creator is not read-only, and the
    /// startup warning must not say it is; one naming only users is.
    #[test]
    fn the_startup_warning_reads_admins_and_creators() {
        assert!(!one("email", "me@example.com", "creator").names_no_admin());
        assert!(!one("email", "me@example.com", "admin").names_no_admin());
        assert!(one("email", "me@example.com", "user").names_no_admin());
        assert!(Roles::default().names_no_admin());
    }

    /// **Every level above a bar clears it, for every bar.**
    ///
    /// The property an access check is allowed to assume, and the one an exact
    /// match quietly breaks. `matches!(role, Role::Admin)` was correct for as
    /// long as `Admin` was the top of the ladder, and became a silent demotion
    /// the moment something went above it: the estate's owner signed in and was
    /// refused a control an admin had, with no error anywhere. This is what
    /// makes that a property of the type rather than a thing to remember at
    /// each call site.
    #[test]
    fn a_higher_level_clears_every_bar_a_lower_one_does() {
        let ladder = [
            Role::Unauthenticated,
            Role::Guest,
            Role::User,
            Role::Admin,
            Role::Creator,
        ];
        for (i, higher) in ladder.iter().enumerate() {
            for lower in &ladder[..=i] {
                assert!(
                    higher.at_least(*lower),
                    "{higher} does not clear {lower}'s bar"
                );
            }
            for above in &ladder[i + 1..] {
                assert!(!higher.at_least(*above), "{higher} cleared {above}'s bar");
            }
        }
    }

    /// The wire spelling is what a refusal body and the log line carry, so each
    /// level needs its own and none may collide.
    #[test]
    fn every_level_has_its_own_spelling() {
        let all = [
            Role::Unauthenticated,
            Role::Guest,
            Role::User,
            Role::Admin,
            Role::Creator,
        ];
        for (i, role) in all.iter().enumerate() {
            assert!(!role.as_str().is_empty());
            for other in &all[i + 1..] {
                assert_ne!(role.as_str(), other.as_str(), "two levels spell the same");
            }
        }
        assert!(Role::Creator.is_creator());
        assert!(!Role::Admin.is_creator());
    }

    #[test]
    fn nobody_is_more_than_a_guest_without_being_named() {
        let r = Roles::default();
        assert!(r.names_no_admin());
        assert_eq!(r.of(None), Role::Unauthenticated);
        assert_eq!(r.of(Some(&id("g1", "a@x.com"))), Role::Guest);
    }

    #[test]
    fn a_subject_entry_matches_only_that_subject() {
        let r = one("sub", "108000000000000000000", "admin");
        assert_eq!(r.of(Some(&id("108000000000000000000", ""))), Role::Admin);
        assert_eq!(r.of(Some(&id("111930197703828817753", ""))), Role::Guest);
        // The subject is opaque and case-sensitive; a provider that emits
        // base64 can have two different people differing only in case.
        let r = one("sub", "AbC", "admin");
        assert_eq!(r.of(Some(&id("abc", ""))), Role::Guest);
    }

    #[test]
    fn an_email_entry_matches_case_insensitively_and_ignores_spacing() {
        let r = one("email", "Someone@Example.COM", "admin");
        assert_eq!(r.of(Some(&id("g1", "someone@example.com"))), Role::Admin);
        assert_eq!(
            r.of(Some(&id("g1", "  SOMEONE@example.com  "))),
            Role::Admin
        );
        assert_eq!(
            r.of(Some(&id("g1", "someone.else@example.com"))),
            Role::Guest
        );
    }

    /// The gateway drops a header it cannot fit, so an identity with no email
    /// is a real thing that arrives. It must never match an entry.
    #[test]
    fn an_empty_value_on_either_side_never_grants_anything() {
        for r in [one("email", "", "admin"), one("sub", "", "admin")] {
            assert_eq!(r.of(Some(&id("g1", "a@x.com"))), Role::Guest);
            assert_eq!(r.of(Some(&id("", ""))), Role::Unauthenticated);
        }
        let r = one("email", "someone@example.com", "admin");
        assert_eq!(r.of(Some(&id("g1", ""))), Role::Guest);
    }

    /// A caller with no identity is never anything but unauthenticated, however
    /// the table is configured.
    #[test]
    fn an_anonymous_caller_cannot_be_promoted() {
        let r =
            table("people:\n  - sub: g1\n    role: admin\n  - email: a@x.com\n    role: admin\n");
        assert_eq!(r.of(None), Role::Unauthenticated);
        // Nor by presenting a blank subject with an admin's email.
        assert_eq!(r.of(Some(&id("", "a@x.com"))), Role::Unauthenticated);
    }

    #[test]
    fn either_form_grants_it_and_the_config_says_which_is_which() {
        let r =
            table("people:\n  - sub: g1\n    role: admin\n  - email: b@x.com\n    role: admin\n");
        assert_eq!(r.of(Some(&id("g1", "nothing@x.com"))), Role::Admin);
        assert_eq!(r.of(Some(&id("g2", "b@x.com"))), Role::Admin);
        assert_eq!(r.of(Some(&id("g2", "c@x.com"))), Role::Guest);
    }

    /// A typo in the key is a refused config, not a silently ignored entry that
    /// leaves somebody without the access they think they configured — and the
    /// lists the table replaced are typos now too.
    #[test]
    fn an_unknown_principal_field_is_a_configuration_error() {
        for bad in [
            "people:\n  - subject: g1\n    role: user\n",
            "people:\n  - mail: a@x.com\n    role: user\n",
            "people:\n  - sub: g1\n    email: a@x.com\n    role: user\n",
            "people:\n  - role: user\n",
            "person:\n  - sub: g1\n    role: user\n",
            "admins:\n  - sub: g1\n",
            "creators:\n  - email: a@x.com\n",
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
        assert_eq!(Role::Guest.to_string(), "guest");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Admin.to_string(), "admin");
        assert_eq!(serde_json::to_string(&Role::Admin).unwrap(), r#""admin""#);
    }
}
