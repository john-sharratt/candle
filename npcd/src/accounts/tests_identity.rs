//! Tests of sign-in: what it refreshes, what it keeps, and when it writes at all.
//!
//! The provider's email and display name refresh on every sign-in, while the
//! author's own name and profile are never overwritten by one. A sign-in that
//! changes nothing writes nothing, and a field the gateway did not forward keeps
//! its stored value. An account is reached only through the issuer and subject
//! that created it, and it records the identity it was made for.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_store::tests::{ident, open, patch, tmp, who};

    /// The provider is the authority on email and display name.
    #[test]
    fn provider_fields_refresh_on_each_sign_in() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "old@example.com", "Old Name"), 1)
            .unwrap();
        let me = a
            .upsert(&ident("g1", "new@example.com", "New Name"), 2)
            .unwrap();
        assert_eq!(me["email"], "new@example.com");
        assert_eq!(me["display"], "New Name");
    }

    /// And the author is the authority on theirs — a sign-in must not revert a
    /// name somebody chose.
    #[test]
    fn author_owned_fields_are_not_overwritten_by_signing_in_again() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "wren@example.com", "Wren"), 1)
            .unwrap();
        a.put_profile(
            &who("g1"),
            &patch(json!({"description": "Ex-surveyor."})),
            2,
        )
        .unwrap();

        let me = a
            .upsert(&ident("g1", "wren@example.com", "Wren"), 2)
            .unwrap();
        assert_eq!(me["profile"]["description"], "Ex-surveyor.");
        assert_eq!(me["unique_name"], "wren");
    }

    /// A `/v1/me` on every page load must not write a record.
    #[test]
    fn an_unchanged_sign_in_writes_no_record() {
        let dir = tmp();
        let mut a = open(&dir);
        let id = ident("g1", "wren@example.com", "Wren");
        a.upsert(&id, 1).unwrap();

        let first = a.shared.custom_objects_version();
        a.upsert(&id, 2).unwrap();
        assert_eq!(
            a.shared.custom_objects_version(),
            first,
            "an unchanged upsert wrote a record"
        );
    }

    /// **A sign-in that changes nothing is read, not written**: `current`
    /// answers nothing for a new identity, the record once it is upserted,
    /// nothing again when the provider sends a new name, and the record when a
    /// field arrives empty — the stored value kept, as `upsert` keeps it.
    #[test]
    fn current_answers_only_when_signing_in_would_change_nothing() {
        let dir = tmp();
        let mut a = open(&dir);
        let id = ident("g1", "wren@example.com", "Wren");
        assert_eq!(a.current(&id, 1), None, "never seen");
        let me = a.upsert(&id, 1).unwrap();
        assert_eq!(a.current(&id, 2), Some(me));
        assert_eq!(
            a.current(&ident("g1", "wren@example.com", "Wren S"), 3),
            None,
            "a new name is a change"
        );
        assert!(
            a.current(&ident("g1", "", "Wren"), 4).is_some(),
            "an email not sent keeps the stored one"
        );
    }

    /// Every provider's subject shape has to become a usable file name.
    #[test]
    fn any_provider_subject_becomes_a_valid_id() {
        for sub in [
            "107839274652839471028",                // Google: digits
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890", // UUID
            "AbCdEf_-123==",                        // base64ish, uppercase, padding
            "user@example.com",                     // an email as a subject
            "日本語",
        ] {
            let key = file_id("google", sub);
            assert_eq!(
                crate::registry::id::check(&key),
                Ok(()),
                "`{sub}` → `{key}`"
            );
        }
    }

    /// A field the gateway could not forward must not blank the stored one.
    ///
    /// `identify` defaults a missing header to `""`, and the gateway skips any
    /// field whose value will not fit in one — so an overlong display name
    /// would have wiped the account's, permanently, on a sign-in that otherwise
    /// worked. Not being told something is not being told it is empty.
    #[test]
    fn a_field_the_gateway_did_not_send_keeps_its_stored_value() {
        let dir = tmp();
        let mut a = open(&dir);
        let full = ident("g1", "wren@example.com", "Wren S");
        a.upsert(&full, 1_000).unwrap();

        // The same person, signing in through a gateway that could not forward
        // the descriptive fields.
        let bare = ident("g1", "", "");
        let me = a.upsert(&bare, 2_000).unwrap();
        assert_eq!(me["email"], "wren@example.com", "the email was blanked");
        assert_eq!(me["display"], "Wren S", "the display name was blanked");

        // And it is durable, not just what this call returned.
        drop(a);
        let mut again = open(&dir);
        assert_eq!(again.get(&full).unwrap()["display"], "Wren S");

        // A real change still lands.
        let renamed = ident("g1", "wren@example.com", "Wren Sharratt");
        let me = again.upsert(&renamed, 3_000).unwrap();
        assert_eq!(me["display"], "Wren Sharratt");
    }

    /// **The account key is issuer AND subject.**
    ///
    /// A subject is unique per issuer, not globally. Keyed on the subject
    /// alone, two providers emitting the same string would land on one account
    /// — silently, and unseparably, because by then the records would already
    /// be merged.
    #[test]
    fn the_same_subject_from_two_providers_is_two_accounts() {
        let a = file_id("google", "1234");
        let b = file_id("github", "1234");
        assert_ne!(a, b, "two issuers collided on one account");

        // And the separator does its job: without it, `("goog", "le1234")` and
        // `("google", "1234")` would hash the same bytes and be one account.
        assert_ne!(file_id("goog", "le1234"), file_id("google", "1234"));
        assert_ne!(file_id("a", "bc"), file_id("ab", "c"));
    }

    /// An account is reached only by the pair that created it. Keying on one
    /// half is what the previous design did, so this pins the other.
    #[test]
    fn an_account_is_not_reachable_from_another_provider() {
        let dir = tmp();
        let mut a = open(&dir);
        let mut google = ident("1234", "wren@example.com", "Wren");
        google.provider = "google".into();
        a.upsert(&google, 1_000).unwrap();

        let mut elsewhere = google.clone();
        elsewhere.provider = "github".into();
        assert!(
            a.get(&elsewhere).is_none(),
            "the same subject from another issuer reached this account"
        );
        assert!(a.get(&google).is_some());
    }

    /// **The identity an account was made for comes back from its record**,
    /// so the same person arriving another way is the same person.
    #[test]
    fn an_account_gives_back_the_identity_it_was_made_for() {
        let dir = tmp();
        let mut a = open(&dir);
        let mut id = ident("g1", "wren@example.com", "Wren S");
        id.picture = "https://example.com/w.png".into();
        a.upsert(&id, 1).unwrap();
        assert_eq!(
            a.identity(&key_of(&id)),
            Some(Identity {
                exp: 0,
                ..id.clone()
            })
        );
        assert_eq!(user_id_of(&id), public_id(&key_of(&id)));
        assert_eq!(a.identity("0123456789abcdef0123456789abcdef"), None);
    }
}
