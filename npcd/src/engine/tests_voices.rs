//! Checks how a person is named inside a world, including the creator's mark and
//! the names of everyone else.
//!
//! The creator's mark is part of the name itself, because a character reads its
//! name and answers to it, so the mark must survive whatever form the name takes.
//! Ordinary players and admins carry only their own name, since a mark that
//! everybody had would say nothing. A creator with no display name still carries
//! the mark on the handle, so the creator is never a stranger to their own cast.

use super::*;

#[cfg(test)]
pub(super) mod speaking_as_tests {
    use super::{speaking_as, CREATOR_MARK};
    use web::auth::session::Identity;
    use web::auth::Roles;

    fn id(email: &str, name: &str) -> Identity {
        Identity {
            provider: "google".into(),
            sub: "g1".into(),
            email: email.into(),
            name: name.into(),
            picture: String::new(),
            exp: 0,
        }
    }

    fn roles(yaml: &str) -> Roles {
        serde_yaml::from_str(yaml).expect("parses")
    }

    /// **The point of the whole flag.** A character reads this string and
    /// answers to it, so the mark has to be in the name itself.
    #[test]
    fn the_creator_is_named_as_such_inside_the_world() {
        let r = roles("people:\n  - email: me@example.com\n    role: creator\n");
        let me = speaking_as(&id("me@example.com", "Johnathan Sharratt"), "u_1", &r);
        assert_eq!(me, format!("Johnathan Sharratt {CREATOR_MARK}"));
        assert!(me.contains("(The Creator)"));
    }

    /// And nobody else is. An ordinary signed-in player carries their own name
    /// and nothing more — a mark everybody had would say nothing.
    #[test]
    fn an_ordinary_person_is_just_their_name() {
        let r = roles("people:\n  - email: me@example.com\n    role: creator\n");
        assert_eq!(
            speaking_as(&id("someone@example.com", "Wren"), "u_2", &r),
            "Wren"
        );
        // Nor is an admin, who is trusted with the files and is still not the
        // person the fiction was made by.
        let r = roles("people:\n  - email: boss@example.com\n    role: admin\n");
        assert_eq!(
            speaking_as(&id("boss@example.com", "Boss"), "u_3", &r),
            "Boss"
        );
    }

    /// The handle is the fallback when the provider gave no display name, and
    /// the mark still lands — a creator whose gateway dropped the name header
    /// must not silently become an anonymous stranger to its own cast.
    #[test]
    fn a_creator_with_no_display_name_still_carries_the_mark() {
        let r = roles("people:\n  - email: me@example.com\n    role: creator\n");
        let me = speaking_as(&id("me@example.com", ""), "u_1a2b3c4d", &r);
        assert_eq!(me, format!("u_1a2b3c4d {CREATOR_MARK}"));
    }
}
