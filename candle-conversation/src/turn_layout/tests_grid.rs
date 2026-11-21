//! Tests of how a turn's segments tile its token grid and realise back to it.
//!
//! They check that a flat grid tiles and reproduces its tokens, that ethereal
//! segments add no positions, and that a reserved tail or leading boundary becomes
//! a real span the turn owns. A zero reservation must be byte-identical to the
//! ethereal form, and an over-large one must clamp rather than consume the answer,
//! so tiling holds for every case the constructors accept.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A normal user→assistant turn's grid tiled by its segments, validated to
    /// reproduce the exact token ids.
    #[test]
    fn flat_grid_tiles_and_realizes() {
        // grid: [user(3)][im_end(2)][assistant_start(3)][response(4)] = 12 tokens
        let grid: Vec<u32> = vec![
            10, 11, 12, /*im_end*/ 90, 91, /*a_start*/ 80, 81, 82, /*resp*/ 1, 2, 3,
            4,
        ];
        let layout = TurnLayout::from_flat_grid(
            0,  // user_content_start
            3,  // user_content_end
            8,  // assistant_start (3 + 2 + 3)
            12, // total
            2,  // im_end_len
            3,  // assistant_start_len
            "hi".into(),
            Some("ok".into()),
            false,
        );
        // Tiling holds against the real grid length.
        assert_eq!(layout.validate_tiling(grid.len() as u32), Ok(()));
        assert_eq!(layout.kv_len(), 12);
        // Real segments slice the grid back exactly, in order.
        let realized = layout.realize(&grid);
        let kinds: Vec<&str> = realized
            .iter()
            .map(|(s, _)| match s {
                TurnSegment::User { .. } => "user",
                TurnSegment::Assistant { .. } => "assistant",
                TurnSegment::Glue {
                    marker: GlueKind::ImEnd,
                    ..
                } => "im_end",
                TurnSegment::Glue {
                    marker: GlueKind::AssistantStart,
                    ..
                } => "a_start",
                _ => "?",
            })
            .collect();
        assert_eq!(kinds, vec!["user", "im_end", "a_start", "assistant"]);
        assert_eq!(realized[0].1, &[10, 11, 12]); // user body
        assert_eq!(realized[1].1, &[90, 91]); // im_end marker
        assert_eq!(realized[2].1, &[80, 81, 82]); // assistant_start marker
        assert_eq!(realized[3].1, &[1, 2, 3, 4]); // response
                                                  // Concatenating the real runs reproduces the whole grid.
        let rebuilt: Vec<u32> = realized
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, grid);
    }

    /// Ethereal segments (dropped-thinking, `/no_think`) contribute no K/V and do
    /// not advance the offset cursor.
    #[test]
    fn ethereal_segments_add_no_kv() {
        let layout = TurnLayout::new(vec![
            TurnSegment::Glue {
                marker: GlueKind::NoThink,
                kv: None,
            },
            TurnSegment::User {
                text: "q".into(),
                kv: KvSpan::new(0, 5),
            },
            // reasoning recorded but its K/V was dropped — adds zero positions
            TurnSegment::Thinking {
                text: "reasoned…".into(),
                kv: None,
            },
            TurnSegment::Assistant {
                text: Some("a".into()),
                kv: KvSpan::new(5, 7),
            },
        ]);
        assert_eq!(layout.kv_len(), 12);
        assert_eq!(layout.validate_tiling(12), Ok(()));
    }

    /// **The trailing twin.** A reserved tail makes the closing `<|im_end|>` a
    /// real span the turn owns, and the assistant body stops short of it.
    ///
    /// The pair of this and `leading_boundary_offsets_user_body` is what makes
    /// both-ends ownership representable: a turn that owned only its opener
    /// would drop every closer, because the assembler's inter-turn island is
    /// exactly `assistant_end ++ user_start` and splitting it at that seam is
    /// only lossless when both halves find an owner.
    #[test]
    fn a_reserved_tail_bakes_the_closing_im_end() {
        // grid: [us(4)][user(5)][im_end(2)][a_start(3)][answer(6)][im_end(2)]
        //        0..4   4..9     9..11      11..14      14..20     20..22
        let layout = TurnLayout::from_flat_grid_with_tail(
            4,
            9,
            14,
            22,
            2,
            3,
            2, // the reserved trailing marker
            "u".into(),
            Some("a".into()),
            false,
        );
        assert_eq!(layout.validate_tiling(22), Ok(()));

        // The answer stops short of the reserved tail.
        assert_eq!(layout.assistant_span(), KvSpan::new(14, 6));

        // Both ends are real, and the turn owns them.
        match layout.segments.first() {
            Some(TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(span),
            }) => assert_eq!(*span, KvSpan::new(0, 4)),
            other => panic!("expected a real leading UserStart, got {other:?}"),
        }
        match layout.segments.last() {
            Some(TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(span),
            }) => assert_eq!(*span, KvSpan::new(20, 2)),
            other => panic!("expected a real trailing ImEnd, got {other:?}"),
        }
    }

    /// A zero reservation reproduces the ethereal tail **exactly** — which is
    /// what every caller passes today, so the mechanism can exist before it is
    /// exercised without moving a single sealed turn.
    #[test]
    fn a_zero_tail_reservation_is_byte_identical_to_the_ethereal_form() {
        let args = || {
            (
                0u32,
                3u32,
                8u32,
                12u32,
                2u32,
                3u32,
                "hi".to_string(),
                Some("ok".to_string()),
                false,
            )
        };
        let (a, b, c, d, e, f, g, h, i) = args();
        let ethereal = TurnLayout::from_flat_grid(a, b, c, d, e, f, g, h, i);
        let (a, b, c, d, e, f, g, h, i) = args();
        let explicit = TurnLayout::from_flat_grid_with_tail(a, b, c, d, e, f, 0, g, h, i);
        assert_eq!(
            ethereal, explicit,
            "a zero tail must not change the layout in any way"
        );
        assert!(matches!(
            ethereal.segments.last(),
            Some(TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: None
            })
        ));
    }

    /// An over-large reservation cannot eat the assistant body — it clamps to
    /// what the grid actually holds past the assistant start.
    #[test]
    fn an_over_large_tail_reservation_clamps() {
        let layout = TurnLayout::from_flat_grid_with_tail(
            0,
            3,
            8,
            12,
            2,
            3,
            999,
            "u".into(),
            Some("a".into()),
            false,
        );
        assert_eq!(layout.validate_tiling(12), Ok(()));
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 0));
    }

    /// A leading boundary offset (non-zero `user_content_start`) is a real
    /// `UserStart` span — the user body is offset past it.
    #[test]
    fn leading_boundary_offsets_user_body() {
        let layout =
            TurnLayout::from_flat_grid(4, 9, 14, 18, 2, 3, "u".into(), Some("a".into()), false);
        assert_eq!(layout.validate_tiling(18), Ok(()));
        match &layout.segments[0] {
            TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(span),
            } => assert_eq!(*span, KvSpan::new(0, 4)),
            other => panic!("expected leading UserStart glue, got {other:?}"),
        }
    }

    /// Semantic accessors read the right segments back out of a flat-grid
    /// layout (and survive a thinking split).
    #[test]
    fn accessors_read_segments() {
        let layout =
            TurnLayout::from_flat_grid(0, 3, 8, 12, 2, 3, "hi".into(), Some("ok".into()), false);
        assert_eq!(layout.user_text(), "hi");
        assert_eq!(layout.assistant_text().as_deref(), Some("ok"));
        assert_eq!(layout.thinking_text(), None);
        assert!(!layout.no_think());
        assert_eq!(layout.user_span(), KvSpan::new(0, 3));
        assert_eq!(layout.user_content_start(), 0);
        assert_eq!(layout.user_content_end(), 3);
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 4));
        // No thinking split → assistant content starts at the assistant span.
        assert_eq!(layout.assistant_content_start(), 8);

        // /no_think turn + a real thinking split moves the assistant content
        // start back to the thinking span's offset.
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            20,
            2,
            3,
            "u".into(),
            Some("<think>r</think>a".into()),
            true,
        )
        .with_thinking_split("<think>r</think>".into(), 5, false);
        assert!(layout.no_think());
        assert_eq!(layout.thinking_text(), Some("<think>r</think>"));
        assert_eq!(layout.assistant_content_start(), 8); // thinking span offset
        assert_eq!(layout.assistant_span(), KvSpan::new(13, 7));
    }

    /// A code_read tool-exchange turn: the single trailing `Assistant` span is
    /// replaced by the real `Assistant → ImEnd → UserStart → User → ImEnd →
    /// AssistantStart → Assistant` run, and the result still tiles the grid.
    #[test]
    fn assistant_split_tiles_tool_exchange() {
        // Base: user[0,3) im_end[3,5) a_start[5,8) assistant[8,24).
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            24,
            2,
            3,
            "excerpt".into(),
            Some("call…resp…ack".into()),
            true,
        );
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 16));
        // Sub-segments tiling [8,24): tc[8,10) im_end[10,12) us[12,15) tr[15,18)
        // im_end[18,20) as[20,23) ack[23,24).
        let subs = vec![
            TurnSegment::Assistant {
                text: Some("<tool_call>".into()),
                kv: KvSpan::new(8, 2),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(10, 2)),
            },
            TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(KvSpan::new(12, 3)),
            },
            TurnSegment::User {
                text: "<tool_response>".into(),
                kv: KvSpan::new(15, 3),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(18, 2)),
            },
            TurnSegment::Glue {
                marker: GlueKind::AssistantStart,
                kv: Some(KvSpan::new(20, 3)),
            },
            TurnSegment::Assistant {
                text: Some("Read …".into()),
                kv: KvSpan::new(23, 1),
            },
        ];
        let layout = layout.with_assistant_split(subs);
        assert_eq!(layout.validate_tiling(24), Ok(()));
        // Two user segments (header + tool response) and two assistant segments
        // (tool call + confirmation).
        let users = layout
            .segments
            .iter()
            .filter(|s| matches!(s, TurnSegment::User { .. }))
            .count();
        let assts = layout
            .segments
            .iter()
            .filter(|s| matches!(s, TurnSegment::Assistant { .. }))
            .count();
        assert_eq!((users, assts), (2, 2));
        // Assistant content still begins at the first assistant body (tool call).
        assert_eq!(layout.assistant_content_start(), 8);
    }

    /// **The user body's span excludes the baked opener.** The message body is
    /// `user_content_start..user_content_end`, starting *past* the baked
    /// marker, so a reader of the body alone never takes the role marker with
    /// it. This pins that the span is the user body alone.
    #[test]
    fn the_user_body_excludes_the_baked_opener() {
        let layout = TurnLayout::from_flat_grid_with_tail(
            4, // the opener occupies [0, 4)
            9,
            14,
            22,
            2,
            3,
            2,
            "u".into(),
            Some("a".into()),
            false,
        );
        let span = layout.user_span();
        assert_eq!(
            span,
            KvSpan::new(4, 5),
            "the user body must start past the opener, or a reader of the body \
             takes the role marker with it"
        );
        assert_eq!(layout.user_content_start(), 4);
        assert_eq!(layout.user_content_end(), 9);
    }

    /// A gap between real segments is rejected.
    #[test]
    fn tiling_rejects_gaps() {
        let layout = TurnLayout::new(vec![
            TurnSegment::User {
                text: "q".into(),
                kv: KvSpan::new(0, 3),
            },
            // starts at 5, not 3 — a 2-token hole
            TurnSegment::Assistant {
                text: None,
                kv: KvSpan::new(5, 4),
            },
        ]);
        assert_eq!(
            layout.validate_tiling(9),
            Err(TilingError::OffsetMismatch {
                segment: 1,
                expected: 3,
                found: 5,
            })
        );
    }
}
