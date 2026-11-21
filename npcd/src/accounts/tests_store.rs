//! Tests of the account store against a real substrate directory: records,
//! restarts and migration.
//!
//! Each test gets its own counted directory, because timestamps on Windows are too
//! coarse to keep concurrently running tests apart. The tests check that an account
//! survives a restart, that it is stored as a custom object record, that the
//! profile history stays out of the page-load payload, and that an older account
//! file is taken in once and corrected at the next sign-in. The provider's subject
//! never reaches the wire.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::npcs::Npcs;

    /// A directory of this test's own.
    ///
    /// Counted rather than timestamped. `SystemTime::now()` is only as fine as
    /// the platform clock — around 15ms on Windows — so tests starting together
    /// were handed the *same* nanosecond and therefore the same directory, and
    /// then wrote over each other's accounts. It failed as an account that had
    /// just been saved coming back missing, which reads as a bug in the store.
    pub(in super::super) fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-accounts-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // A previous run of the same PID could have left one behind.
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The accounts of the substrate in `dir`, opened as the daemon opens them
    /// — a second call over the same `dir` is a restart.
    pub(in super::super) fn open(dir: &std::path::Path) -> Accounts {
        let shared = Npcs::load(dir).unwrap().substrate();
        Accounts::open(shared, &dir.join("accounts")).unwrap()
    }

    /// `json!` builds a `Value`; the store takes a `Map`, because a patch that
    /// is not an object is not a patch. This is the seam the API does for real.
    pub(in super::super) fn patch(v: Value) -> Map<String, Value> {
        v.as_object().expect("a test patch is an object").clone()
    }

    pub(in super::super) fn ident(sub: &str, email: &str, name: &str) -> Identity {
        Identity {
            provider: "google".into(),
            sub: sub.into(),
            email: email.into(),
            name: name.into(),
            picture: String::new(),
            exp: 9_999_999_999,
        }
    }

    /// A subject, as the identity the account methods now take.
    ///
    /// They take a whole `Identity` rather than a `sub` because the account key
    /// is issuer *and* subject: two arguments that must agree are two arguments
    /// that can disagree, and the identity is the thing that already holds a
    /// matched pair.
    pub(in super::super) fn who(sub: &str) -> Identity {
        ident(sub, "a@b.c", "A B")
    }

    #[test]
    fn a_first_sight_creates_an_account_that_survives_a_restart() {
        let dir = tmp();
        let mut a = open(&dir);
        let me = a
            .upsert(&ident("google-1", "wren@example.com", "Wren S"), 1_000)
            .unwrap();
        assert_eq!(me["email"], "wren@example.com");
        assert_eq!(me["unique_name"], "wren");

        drop(a);
        let again = open(&dir);
        assert_eq!(again.len(), 1);
        assert_eq!(again.get(&who("google-1")).unwrap()["display"], "Wren S");
    }

    /// **An account is a record**, not a file: the substrate holds it under
    /// `account/<id>`.
    #[test]
    fn an_account_is_a_custom_object_record() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "wren@example.com", "Wren"), 1)
            .unwrap();
        let key = format!("account/{}", file_id("google", "g1"));
        let held = a.shared.custom_object(&key).unwrap();
        assert_eq!(held.blob["email"], "wren@example.com");
        assert_eq!(held.blob["sub"], "g1", "the subject stays in the record");
        assert!(!dir.join("accounts").exists(), "no file is written");
    }

    /// The history is the larger half of the record and is asked for rarely.
    #[test]
    fn the_history_does_not_ride_along_on_every_page_load() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let me = a
            .put_profile(&who("g1"), &patch(json!({"description": "a"})), 2)
            .unwrap()
            .unwrap();
        assert!(me.get("profile_history").is_none());
        assert!(a.get(&who("g1")).unwrap().get("profile_history").is_none());
    }

    /// **An account file kept before accounts were records is taken in once**,
    /// and its folder goes; a record written under an older profile shape
    /// corrects itself the next time its owner signs in — no field survives
    /// that the profile does not have, and a `gender` outside the allowed set
    /// is cleared rather than carried.
    #[test]
    fn an_older_file_is_taken_in_and_corrected_on_the_next_sign_in() {
        let dir = tmp();
        let key = file_id("google", "g1");
        std::fs::create_dir_all(dir.join("accounts")).unwrap();
        std::fs::write(
            dir.join("accounts").join(format!("{key}.yaml")),
            serde_yaml::to_string(&json!({
                "sub": "g1",
                "provider": "google",
                "created_ms": 1,
                "unique_name": "wren",
                "profile": {
                    "description": "Ex-surveyor.",
                    "gender": "—",
                    "pronouns": "they/them",
                    "history": "",
                    "turn_index": 0,
                    "revision": 0
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut a = open(&dir);
        assert!(!dir.join("accounts").exists(), "the folder is taken in");
        assert_eq!(a.len(), 1);
        let me = a
            .upsert(&ident("g1", "wren@example.com", "Wren"), 2)
            .unwrap();

        assert!(me["profile"].get("pronouns").is_none(), "{me}");
        assert_eq!(me["profile"]["gender"], "", "an invalid gender was kept");
        // What the profile does have is untouched.
        assert_eq!(me["profile"]["description"], "Ex-surveyor.");

        // Durable, not re-derived on every read.
        drop(a);
        let again = open(&dir);
        let stored = &again.get(&who("g1")).unwrap()["profile"];
        assert!(stored.get("pronouns").is_none(), "{stored}");
        assert_eq!(stored["gender"], "");
    }

    #[test]
    fn the_provider_subject_never_reaches_the_wire() {
        let dir = tmp();
        let mut a = open(&dir);
        let me = a
            .upsert(&ident("secret-subject-id", "w@e.com", "W"), 1)
            .unwrap();
        let text = serde_json::to_string(&me).unwrap();
        assert!(!text.contains("secret-subject-id"), "{text}");
        assert!(text.contains("user_id"));
    }
}
