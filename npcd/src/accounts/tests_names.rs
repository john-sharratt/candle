//! Tests that suggested names and chosen names pass the same gate, and that a linked
//! Discord account survives sign-in.
//!
//! Every suggestion must be accepted by the setter that guards it. A chosen name is
//! trimmed before it is stored and survives a restart. A linked Discord account is
//! a field of the account, which a sign-in keeps and only an explicit unlink
//! removes.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_store::tests::{ident, open, tmp, who};

    /// The store must never suggest a name its own setter would refuse — the
    /// author would open the page, press Save without typing, and be told the
    /// name they were given is invalid.
    #[test]
    fn every_suggested_name_passes_the_gate_that_guards_it() {
        for (email, name) in [
            ("wren@example.com", "Wren S"),
            ("a@example.com", "A"),
            ("@example.com", "Solitary"),
            ("....@example.com", "Punctuation Only"),
            ("averyveryverylonglocalpartindeed@example.com", "Long"),
        ] {
            let suggested = default_unique_name(&ident("g", email, name));
            assert!(
                check_unique_name(&suggested).is_ok(),
                "`{email}` → `{suggested}`, which the setter refuses"
            );
        }
    }

    #[test]
    fn a_chosen_name_persists_across_a_restart() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "wren@example.com", "Wren S"), 1)
            .unwrap();
        a.put_unique_name(&who("g1"), "  ridge-walker  ").unwrap();

        drop(a);
        let again = open(&dir);
        // Trimmed on the way in, so the record holds the address and not the
        // author's stray whitespace.
        assert_eq!(
            again.get(&who("g1")).unwrap()["unique_name"],
            "ridge-walker"
        );
    }

    /// **A linked Discord account is the account's `discord` field**, in
    /// exactly this shape; a sign-in does not unlink it, and unlinking
    /// removes the field.
    #[test]
    fn a_linked_discord_account_is_a_field_a_sign_in_keeps() {
        let dir = tmp();
        let mut a = open(&dir);
        let id = ident("g1", "wren@example.com", "Wren S");
        a.upsert(&id, 1).unwrap();
        let key = key_of(&id);
        let user = DiscordUser {
            id: "1300000000000000013".into(),
            username: "wren".into(),
            linked_ms: 1_800_000_000_000,
        };
        assert!(a.set_discord(&key, Some(&user)).unwrap());
        assert_eq!(
            a.shared
                .custom_object(&format!("account/{key}"))
                .unwrap()
                .blob["discord"],
            json!({
                "id": "1300000000000000013",
                "username": "wren",
                "linked_ms": 1_800_000_000_000u64,
            })
        );
        a.upsert(&id, 2).unwrap();
        assert_eq!(a.discord(&key), Some(user.clone()));

        let version = a.shared.custom_objects_version();
        assert!(a.set_discord(&key, Some(&user)).unwrap());
        assert_eq!(
            a.shared.custom_objects_version(),
            version,
            "an unchanged link wrote"
        );

        assert!(a.set_discord(&key, None).unwrap());
        assert_eq!(a.discord(&key), None);
        assert!(a.get(&id).unwrap().get("discord").is_none());
        assert!(!a
            .set_discord("0123456789abcdef0123456789abcdef", None)
            .unwrap());
    }
}
