//! Tests that a sealed turn keeps its own boundaries, suppression flag and framing.
//!
//! A sealed turn's no-think flag is derived from its own segments and survives the
//! persistence round trip, so no later turn can move it. The baked boundaries must
//! reproduce exactly the token stream the projection spine used to emit, compared
//! for equality, since one dropped or doubled closing marker would shift every
//! boundary after it. A compression turn must open with exactly one role marker,
//! as a dialogue turn does.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Regression: a code_read scope turn must open EXACTLY like a normal turn —
    /// the leading `user_start` is ethereal glue (`kv: None`, materialized by the
    /// projection spine), NOT a baked `UserStart` span, and there is no separate
    /// `NoThink` glue segment: the scope folds its `/no_think` into the user body
    /// and seals with `no_think() == false`, so the assembler emits exactly one
    /// `user_start` and no duplicate soft-switch. Guards against the
    /// doubled-soft-switch / mislabeled-opener bug where `user_content_start > 0` +
    /// `no_think = true` made every code_read turn reconstruct as
    /// `[user_start][/no_think][/no_think][user]…`.
    #[test]
    fn code_read_scope_opener_matches_a_normal_turn() {
        // Scope params after the fix: user_content_start = 0, no_think = false, no
        // baked soft-switch.
        let layout = TurnLayout::from_flat_grid(
            0,
            5,
            8,
            24,
            2,
            3,
            "Summarize `x` (lines 1-5) in no more than two sentences.".into(),
            Some("<tool_call>…</tool_call>…<tool_response>…".into()),
            false,
        );
        // Leading glue is an ETHEREAL user_start — the spine materializes it, it is
        // not baked from this turn's grid.
        assert!(
            matches!(
                layout.segments.first(),
                Some(TurnSegment::Glue {
                    marker: GlueKind::UserStart,
                    kv: None,
                }),
            ),
            "expected an ethereal leading UserStart, got {:?}",
            layout.segments.first(),
        );
        // No NoThink glue segment at all — the scope carries no soft-switch.
        assert!(
            !layout.segments.iter().any(|s| matches!(
                s,
                TurnSegment::Glue {
                    marker: GlueKind::NoThink,
                    ..
                }
            )),
            "a scope turn must not carry a NoThink glue segment",
        );
        // The turn's no_think flag is false, so the assembler emits no `/no_think`.
        assert!(!layout.no_think());
        assert_eq!(layout.user_content_start(), 0);
        assert_eq!(layout.assistant_content_start(), 8);
    }

    /// **T3.6 — a sealed turn's suppression flag is a property of its own bytes,
    /// and nothing later can move it.**
    ///
    /// This is the condition the bake was granted on. §10 decision 9: *"assume
    /// fixed once sealed, and add the test. If the test fails, fall back to
    /// leaving `NoThink` ethereal."* Baking the switch into the turn's grid
    /// freezes a decision that used to be re-made on every projection, so the
    /// assumption underneath it has to be checked rather than read.
    ///
    /// Three things together make it hold, and each is asserted:
    ///
    /// 1. the flag is **derived from the segments**, not stored beside them, so
    ///    there is no second copy to drift;
    /// 2. it **survives the persistence round-trip**, which is the only way a
    ///    sealed layout ever comes back; and
    /// 3. two turns sealed under different dial settings keep their **own**
    ///    answers — the leak the live-glue path had to guard against, where a
    ///    past suppressed turn put a stale switch on a later thinking turn,
    ///    cannot happen when each turn's grid holds its own.
    #[test]
    fn a_sealed_turns_no_think_flag_is_fixed_by_its_own_grid() {
        // `/no_think` is a 3-token rider after the 1-token `user_start`, so the
        // suppressed turn's body starts at 4 and the thinking one's at 1.
        let suppressed = TurnLayout::from_flat_grid_with_tail(
            4,
            9,
            11,
            16,
            1,
            2,
            1,
            "hi".into(),
            Some("there".into()),
            true,
        );
        let thinking = TurnLayout::from_flat_grid_with_tail(
            1,
            6,
            8,
            13,
            1,
            2,
            1,
            "hi".into(),
            Some("there".into()),
            false,
        );

        assert!(suppressed.no_think(), "the baked switch must be visible");
        assert!(!thinking.no_think());

        // (1) Derived, not stored: the answer is exactly "is there a NoThink
        // segment", so removing it changes the answer and nothing else holds a
        // stale copy that could disagree.
        let mut stripped = suppressed.clone();
        stripped.segments.retain(|s| {
            !matches!(
                s,
                TurnSegment::Glue {
                    marker: GlueKind::NoThink,
                    ..
                }
            )
        });
        assert!(
            !stripped.no_think(),
            "the flag survived removal of the segment it is supposed to be read \
             from — there is a second copy, and the two can drift"
        );

        // (2) Survives the round-trip the substrate actually performs.
        for layout in [&suppressed, &thinking] {
            let json = serde_json::to_string(layout).expect("serialize");
            let back: TurnLayout = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(
                back.no_think(),
                layout.no_think(),
                "a reloaded turn disagreed with the one that was sealed"
            );
            assert_eq!(back.segments, layout.segments);
        }

        // (3) Independent: neither turn's answer depends on the other's, in
        // either order.
        let both = [&suppressed, &thinking];
        assert!(both[0].no_think() && !both[1].no_think());
        let reversed = [&thinking, &suppressed];
        assert!(!reversed[0].no_think() && reversed[1].no_think());
    }

    /// **Stream equivalence — the load-bearing test for boundary ownership.**
    ///
    /// The projection spine used to emit `user_start` before every sealed turn
    /// and `assistant_end` after it, so a run of turns came out as
    ///
    /// ```text
    ///   US body₀ AE  US body₁ AE  US body₂ AE
    ///   └island┘     └─island─┘    └─island─┘
    /// ```
    ///
    /// with each `AE ++ US` pair forming one gap-filled island between turns.
    /// Now each turn's own grid carries `US … AE`, and the turns abut directly.
    ///
    /// The token stream must be **identical** — equality, not tolerance, because
    /// §4.7a proves the split is exact rather than approximate. The failure this
    /// catches is a dropped or doubled `<|im_end|>`, which reads perfectly and
    /// shifts every boundary after it.
    #[test]
    fn baked_boundaries_reproduce_the_spine_emitted_stream_exactly() {
        // Dialect markers, as token ids.
        const US: &[u32] = &[100, 101];
        const AE: &[u32] = &[200];
        const UE: &[u32] = &[150];
        const AS: &[u32] = &[160, 161];

        // Three turns' bodies.
        let bodies: [(&[u32], &[u32]); 3] = [
            (&[1, 2, 3], &[10, 11]),
            (&[4, 5], &[12, 13, 14]),
            (&[6], &[15]),
        ];

        // ── The reference: what the spine used to emit. ──────────────────────
        let mut reference: Vec<u32> = Vec::new();
        for (user, answer) in bodies {
            reference.extend_from_slice(US);
            reference.extend_from_slice(user);
            reference.extend_from_slice(UE);
            reference.extend_from_slice(AS);
            reference.extend_from_slice(answer);
            reference.extend_from_slice(AE);
        }

        // ── The baked arrangement: each turn's grid carries its own markers,
        //    and `realize()` walks only what the turn actually owns. ──────────
        let mut baked: Vec<u32> = Vec::new();
        for (user, answer) in bodies {
            let mut grid: Vec<u32> = Vec::new();
            grid.extend_from_slice(US);
            grid.extend_from_slice(user);
            grid.extend_from_slice(UE);
            grid.extend_from_slice(AS);
            grid.extend_from_slice(answer);
            grid.extend_from_slice(AE);

            let us_len = US.len() as u32;
            let user_end = us_len + user.len() as u32;
            let asst_start = user_end + UE.len() as u32 + AS.len() as u32;
            let total = grid.len() as u32;
            let layout = TurnLayout::from_flat_grid_with_tail(
                us_len,
                user_end,
                asst_start,
                total,
                UE.len() as u32,
                AS.len() as u32,
                AE.len() as u32,
                "u".into(),
                Some("a".into()),
                false,
            );
            assert_eq!(
                layout.validate_tiling(total),
                Ok(()),
                "a baked turn must tile its own grid exactly"
            );
            // Every real segment, in order — which is what the inject path walks.
            for (_, toks) in layout.realize(&grid) {
                baked.extend_from_slice(toks);
            }
        }

        assert_eq!(
            baked, reference,
            "the baked-boundary stream diverged from the spine-emitted one — a \
             dropped or doubled boundary marker reads fine and shifts every \
             token position after it"
        );
    }

    /// The same equivalence with the `/no_think` switch present: it rides the
    /// opener, inside the turn that carries it.
    ///
    /// Each turn holding its own switch is *stronger* than the spine re-deciding
    /// it per projection, not weaker — the leak the live path guarded against
    /// (a past suppressed turn putting a stale switch on a later thinking-on
    /// turn) cannot happen when the switch lives in the suppressed turn's grid.
    #[test]
    fn a_suppressed_turns_switch_is_baked_into_its_own_opener() {
        const US: &[u32] = &[100];
        const NT: &[u32] = &[42];
        const AE: &[u32] = &[200];

        // head = user_start ++ no_think, so the opener span covers both.
        let head_len = (US.len() + NT.len()) as u32;
        let mut grid: Vec<u32> = Vec::new();
        grid.extend_from_slice(US);
        grid.extend_from_slice(NT);
        grid.extend_from_slice(&[1, 2]); // user body
        grid.extend_from_slice(&[150]); // user_end
        grid.extend_from_slice(&[160]); // assistant_start
        grid.extend_from_slice(&[10]); // answer
        grid.extend_from_slice(AE);

        let layout = TurnLayout::from_flat_grid_with_tail(
            head_len,
            head_len + 2,
            head_len + 4,
            grid.len() as u32,
            1,
            1,
            AE.len() as u32,
            "u".into(),
            Some("a".into()),
            true,
        );
        assert_eq!(layout.validate_tiling(grid.len() as u32), Ok(()));

        // The opener span covers `user_start ++ no_think` together.
        match layout.segments.first() {
            Some(TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(span),
            }) => {
                assert_eq!(*span, KvSpan::new(0, head_len));
                assert_eq!(&grid[span.range()], &[100, 42]);
            }
            other => panic!("expected a real leading UserStart, got {other:?}"),
        }
        // Realising reproduces the grid, switch included.
        let rebuilt: Vec<u32> = layout
            .realize(&grid)
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, grid);
    }

    /// **A compression turn is framed exactly like a dialogue turn.**
    ///
    /// It builds its grid directly rather than going through the turn-submit
    /// funnel, and it used to carry no head at all — correct while the assembler
    /// re-emitted one around every sealed turn, and silently wrong the moment
    /// turns started owning their boundaries. A summary with no opener runs
    /// straight on from the previous turn's closer with no role marker to say
    /// whose words it is.
    ///
    /// Exactly **one** opener: not zero (the assembler stopped and the builder
    /// was not updated) and not two (both fired).
    #[test]
    fn a_compression_turn_is_framed_like_a_dialogue_turn() {
        const US: &[u32] = &[100, 101];
        const AE: &[u32] = &[200];

        // [user_start][question(3)][user_end(1)][assistant_start(1)][answer(2)][im_end]
        let mut grid: Vec<u32> = Vec::new();
        grid.extend_from_slice(US);
        let user_content_start = grid.len() as u32;
        grid.extend_from_slice(&[1, 2, 3]);
        let user_end_at = grid.len() as u32;
        grid.extend_from_slice(&[150]);
        grid.extend_from_slice(&[160]);
        let asst_start_at = grid.len() as u32;
        grid.extend_from_slice(&[10, 11]);
        grid.extend_from_slice(AE);

        let layout = TurnLayout::from_flat_grid_with_tail(
            user_content_start,
            user_end_at,
            asst_start_at,
            grid.len() as u32,
            1,
            1,
            AE.len() as u32,
            "q".into(),
            Some("a".into()),
            false,
        );
        assert_eq!(layout.validate_tiling(grid.len() as u32), Ok(()));

        let openers = layout
            .segments
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    TurnSegment::Glue {
                        marker: GlueKind::UserStart,
                        kv: Some(_)
                    }
                )
            })
            .count();
        assert_eq!(
            openers, 1,
            "a compression turn must have exactly one opener"
        );

        // Real at both ends, and it reproduces its own grid.
        assert!(matches!(
            layout.segments.last(),
            Some(TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(_)
            })
        ));
        let rebuilt: Vec<u32> = layout
            .realize(&grid)
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, grid);
    }
}
