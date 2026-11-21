//! Tests that profile edits are revisions rather than overwrites, and that a write
//! cannot reach the identity fields.
//!
//! A save tombstones the previous revision rather than erasing it, so text an NPC
//! has cited stays readable. History is bounded to the live revision plus the cap,
//! keeping the newest. A preview never splits a character. A patch that is not an
//! object, a wrong-typed field, or an unknown key cannot file a revision into
//! history or change a field the profile does not have.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_store::tests::{ident, open, patch, tmp, who};

    /// A profile edit must not be a way to become somebody else.
    #[test]
    fn a_profile_edit_cannot_reach_identity_fields() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "wren@example.com", "Wren"), 1)
            .unwrap();
        let me = a
            .put_profile(
                &who("g1"),
                &patch(json!({"description": "x", "email": "attacker@evil.com", "sub": "someone-else"})),
                2,
            )
            .unwrap()
            .unwrap();

        assert_eq!(me["email"], "wren@example.com", "email was reachable");
        assert!(me.get("sub").is_none(), "the subject leaked to the wire");
        // The stray keys landed in the profile, which is inert, not on the record.
        assert_eq!(me["profile"]["description"], "x");
    }

    #[test]
    fn the_revision_is_the_stores_to_bump() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let one = a
            .put_profile(&who("g1"), &patch(json!({"description": "a"})), 2)
            .unwrap()
            .unwrap();
        assert_eq!(one["profile"]["revision"], 1);
        // A caller cannot set it.
        let two = a
            .put_profile(
                &who("g1"),
                &patch(json!({"description": "b", "revision": 99})),
                3,
            )
            .unwrap()
            .unwrap();
        assert_eq!(two["profile"]["revision"], 2);
    }

    /// The claim the console makes on the Save button, as a test: a revision is
    /// superseded, never erased. An NPC that cited the old text must still be
    /// able to find it.
    #[test]
    fn a_revision_is_tombstoned_rather_than_overwritten() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        a.put_profile(&who("g1"), &patch(json!({"description": "Surveyor."})), 100)
            .unwrap();
        a.put_profile(
            &who("g1"),
            &patch(json!({"description": "Ex-surveyor."})),
            200,
        )
        .unwrap();

        // The index is summaries — enough to choose one, not to read it.
        let h = a.profile_history(&who("g1")).unwrap();
        assert_eq!(h.len(), 3, "live revision plus its two predecessors");

        // Newest first, and exactly one is live.
        assert_eq!(h[0]["preview"], "Ex-surveyor.");
        assert_eq!(h[0]["revision"], 2);
        assert_eq!(h[0]["live"], true);
        assert_eq!(h[0]["tombstoned_ms"], Value::Null);

        assert_eq!(h[1]["preview"], "Surveyor.");
        assert_eq!(h[1]["revision"], 1);
        assert_eq!(h[1]["live"], false);
        assert_eq!(h[1]["tombstoned_ms"], 200);

        // The empty profile the account was born with.
        assert_eq!(h[2]["preview"], "");
        assert_eq!(h[2]["revision"], 0);
        assert_eq!(h[2]["tombstoned_ms"], 100);

        // The index carries no prose — that is the point of it being an index.
        assert!(h[1].get("description").is_none(), "{:?}", h[1]);
        assert!(h[1].get("history").is_none(), "{:?}", h[1]);
        // And the full text is one fetch away.
        let full = a.profile_revision(&who("g1"), 1).unwrap();
        assert_eq!(full["description"], "Surveyor.");

        // And it survives a restart, because the record carries it.
        drop(a);
        let again = open(&dir);
        assert_eq!(again.profile_history(&who("g1")).unwrap().len(), 3);
    }

    /// History is bounded, because the whole file is resident.
    ///
    /// Every save appends a full copy of the outgoing profile — paragraphs of
    /// it — and every member holds every account's record.
    /// Unbounded, an author who edits often carries their entire writing
    /// history forever to support an undo that reaches back a few steps.
    #[test]
    fn the_kept_history_is_bounded_and_keeps_the_newest() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();

        let n = KEEP_REVISIONS + 25;
        for i in 0..n {
            a.put_profile(
                &who("g1"),
                &patch(json!({ "description": format!("v{i}") })),
                i as u64,
            )
            .unwrap();
        }

        // The live one plus the cap, and no more.
        let h = a.profile_history(&who("g1")).unwrap();
        assert_eq!(h.len(), KEEP_REVISIONS + 1);

        // What survives is the newest. The most recent edit is live; the oldest
        // kept is far enough back to be a real undo and no further.
        assert_eq!(h[0]["preview"], format!("v{}", n - 1));
        assert_eq!(h[0]["live"], true);
        assert_eq!(h[h.len() - 1]["revision"], (n - KEEP_REVISIONS) as u64);

        // The dropped ones are gone from lookup too, not merely hidden.
        assert!(a.profile_revision(&who("g1"), 0).is_none());
        assert!(a.profile_revision(&who("g1"), n as u64).is_some());
    }

    /// A preview is one line, and cutting it must not split a character.
    #[test]
    fn a_preview_is_short_and_never_splits_a_character() {
        assert_eq!(preview(""), "");
        assert_eq!(preview("short"), "short");
        // Newlines and runs of spaces collapse — this is one line in a chooser.
        assert_eq!(preview("two\n\nlines   here"), "two lines here");

        // Multi-byte throughout: `&s[..90]` would land mid-codepoint and panic.
        let wide = "日".repeat(300);
        let cut = preview(&wide);
        assert!(cut.ends_with('…'));
        assert_eq!(cut.chars().count(), 91, "90 characters plus the ellipsis");

        // A boundary case either side of the limit.
        for n in [89, 90, 91] {
            let s = "é".repeat(n);
            let p = preview(&s);
            assert!(
                p.chars().count() <= 91,
                "{n} produced {}",
                p.chars().count()
            );
        }
    }

    /// Everything a caller may write is checked before any of it is merged.
    ///
    /// The failure this guards is silent rather than loud: `normalise_profile`
    /// blanks a wrong-typed field on its way to disk, which is right for
    /// repairing a record and catastrophic for a live write — a `200 OK` that
    /// erased the author's prose. `null` and `3` are refused as the wrong
    /// *kind* of thing, separately from a string that is not an allowed gender,
    /// because those are two different mistakes to fix.
    #[test]
    fn a_wrong_typed_field_is_refused_rather_than_blanked() {
        for bad in [
            json!({"description": 42}),
            json!({"history": null}),
            json!({"description": ["a"]}),
            json!({"gender": null}),
            json!({"gender": 3}),
            json!({"gender": {"v": "Male"}}),
        ] {
            let m = patch(bad.clone());
            assert!(
                matches!(check_patch(&m), Err(PatchError::NotText(_))),
                "{bad} was accepted as text"
            );
        }

        // A string that is simply not one of the values is its own answer.
        assert_eq!(
            check_patch(&patch(json!({"gender": "Other"}))),
            Err(PatchError::BadGender)
        );

        for good in [
            json!({}),
            json!({"description": ""}),
            json!({"gender": ""}),
            json!({"gender": "Female", "history": "x"}),
            // Unknown keys are the store's to discard, not a reason to refuse.
            json!({"nickname": 7}),
        ] {
            assert_eq!(check_patch(&patch(good.clone())), Ok(()), "{good} refused");
        }
    }

    /// A patch that is not an object cannot reach the store at all, so it
    /// cannot file a revision into history without advancing the counter.
    ///
    /// It used to be able to: the merge was guarded by `patch.as_object()` but
    /// the tombstone and the write below it were not, so `PUT []` retired the
    /// live profile, left `revision` where it was, and grew the file on every
    /// repeat until two turns claimed the same number. The signature is the fix
    /// — `Map` rather than `Value` — and this is the assertion that the shape
    /// really is unrepresentable.
    #[test]
    fn a_body_that_is_not_an_object_is_not_a_patch() {
        for not_a_patch in [json!([]), json!("x"), json!(3), json!(null)] {
            assert!(
                not_a_patch.as_object().is_none(),
                "{not_a_patch} would still reach the store"
            );
        }

        // And a legitimate empty patch still advances exactly one revision.
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let me = a
            .put_profile(&who("g1"), &patch(json!({})), 2)
            .unwrap()
            .unwrap();
        assert_eq!(me["profile"]["revision"], 1);
        assert_eq!(a.profile_history(&who("g1")).unwrap().len(), 2);
    }

    /// A caller cannot add one back, either.
    #[test]
    fn a_patch_cannot_introduce_a_field_the_profile_does_not_have() {
        let dir = tmp();
        let mut a = open(&dir);
        a.upsert(&ident("g1", "w@e.com", "W"), 1).unwrap();
        let me = a
            .put_profile(
                &who("g1"),
                &patch(json!({"description": "x", "pronouns": "they/them", "nickname": "boss"})),
                2,
            )
            .unwrap()
            .unwrap();

        assert_eq!(me["profile"]["description"], "x");
        assert!(me["profile"].get("pronouns").is_none(), "{me}");
        assert!(me["profile"].get("nickname").is_none(), "{me}");
    }
}
