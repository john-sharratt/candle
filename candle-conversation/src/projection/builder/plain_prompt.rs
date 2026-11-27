//! Builds the one-layer, one-group schema that a plain system prompt needs.
//!
//! A conversation given only a prompt string gets a synthetic dialogue layer with
//! a single frame section, so it needs no YAML. The reserved forms take their ids
//! from the top of the id range, so an engine-internal conversation can share a
//! substrate with a YAML schema without its turns entering that projection. A
//! closed reserved group admits only its own turns, because an untagged group
//! would let one job read another job's question.

use super::*;

impl Builder {
    // ── Synthetic construction ────────────────────────────────────────────────

    /// Build a minimal single-layer schema around a pre-rendered system-prompt
    /// string, for callers that provide a plain `&str` rather than a YAML
    /// schema.
    ///
    /// The resulting builder has one `dialogue` layer (32 768-token window),
    /// one `primary_conversation` group with `AlwaysVisible` selection (all
    /// turns pass through — budget trimming is left to the existing sliding-
    /// window config downstream), and the system-prompt text as a single
    /// `frame` section.
    ///
    /// The layer and group ids are fixed at 1. The frame is sealed under
    /// `frame`, which comes from
    /// [`ConversationEngine::plain_prompt_section`](crate::ConversationEngine::plain_prompt_section)
    /// for this same text: a section that already exists is reused by id
    /// without its text being compared, so a fixed id would hand this
    /// conversation whichever prompt had sealed there first.
    ///
    /// Its one layer is a dialogue, so it decodes at `High`.
    pub fn for_plain_prompt(system_prompt_text: &str, frame: SectionId) -> Self {
        Self::synthetic_single_section(
            system_prompt_text,
            LayerId::new(1),
            GroupId::new(1),
            frame,
            DecodePriority::High,
        )
    }

    /// Same as [`Self::for_plain_prompt`] but takes the layer and group ids
    /// from the [`Reserved`] kind's slot at the top of the u32 range. Use this
    /// for engine-internal conversations (the daemon's titler) that must
    /// coexist in the same substrate as a YAML schema without their turns
    /// entering its projection. The frame is `frame`, for the same reason as
    /// there.
    ///
    /// The caller names the band it decodes at: an engine-internal conversation
    /// may be a person's (a client's passthrough dialogue) or background to one
    /// (the titler), and only the caller knows which.
    ///
    /// [`Reserved`]: super::Reserved
    pub fn for_plain_prompt_reserved(
        system_prompt_text: &str,
        kind: Reserved,
        frame: SectionId,
        decode_priority: DecodePriority,
    ) -> Self {
        Self::synthetic_single_section(
            system_prompt_text,
            LayerId::reserved(kind),
            GroupId::reserved(kind),
            frame,
            decode_priority,
        )
    }

    /// [`Self::for_plain_prompt_reserved`] for a conversation opened for one
    /// job, its group **closed**: it reads its own system prompt and turns, and
    /// none of the other conversations written to the same reserved group.
    ///
    /// Every job of one [`Reserved`] kind writes to that kind's one group, and a
    /// group with no tags admits every turn in it — one job would read another's
    /// question. Tagged with something no turn carries, it admits none of them.
    /// The reader's own turns are always its own ([`Self::set_group_tags`]).
    ///
    /// [`Reserved`]: super::Reserved
    pub fn for_closed_reserved(
        system_prompt_text: &str,
        kind: Reserved,
        frame: SectionId,
        decode_priority: DecodePriority,
    ) -> Self {
        let mut builder =
            Self::for_plain_prompt_reserved(system_prompt_text, kind, frame, decode_priority);
        builder
            .set_group_tags(PLAIN_PROMPT_GROUP, vec![CLOSED_GROUP_TAG.to_string()])
            .expect("a plain-prompt schema has its one group");
        builder
    }

    /// Like [`Self::for_plain_prompt_reserved`], but bases the synthetic schema's
    /// **section** ids at `section_base` instead of the single reserved-slot id.
    /// The layer and group still occupy the reserved slot — the conversation's
    /// turns never enter a user projection — but the frame plus any sections later
    /// added via [`Self::add_collection`] / [`Self::add_section_to_collection`]
    /// allocate upward from `section_base`, occupying a dedicated id partition.
    ///
    /// This is for engine-internal conversations that hold a whole *corpus* of
    /// sections (more than the one-per-kind the [`Reserved`] band at the top of
    /// u32 allows). The partition keeps every section id clear of the user
    /// schema's low `1..n` ids, the tool ids allocated above them, and the
    /// reserved band — however much the user schema grows — since the substrate
    /// keys section state globally by id. The caller names the band it decodes
    /// at, as for [`Self::for_plain_prompt_reserved`].
    ///
    /// [`Reserved`]: super::Reserved
    pub fn for_reserved_corpus(
        system_prompt_text: &str,
        kind: Reserved,
        section_base: u32,
        decode_priority: DecodePriority,
    ) -> Self {
        Self::synthetic_single_section(
            system_prompt_text,
            LayerId::reserved(kind),
            GroupId::reserved(kind),
            SectionId::new(section_base),
            decode_priority,
        )
    }

    /// Shared body of [`Self::for_plain_prompt`] /
    /// [`Self::for_plain_prompt_reserved`] — a single-layer / single-group
    /// schema with one `"frame"` section in the system prompt, using
    /// caller-supplied ids, decoding at `decode_priority`.
    pub(super) fn synthetic_single_section(
        system_prompt_text: &str,
        layer_id: LayerId,
        group_id: GroupId,
        section_id: SectionId,
        decode_priority: DecodePriority,
    ) -> Self {
        // Id for the summary's answer framing section (never emitted; sealed
        // lazily only if this conversation is summarised), the one beside the
        // frame — every frame partition leaves that id free for it. There is no
        // question-half section: a summary's user half is derived, never
        // decoded, so it has no prompt to frame.
        let summary_a_id = SectionId::new(section_id.raw() + 1);
        let schema = Schema {
            // Off, like every schema that does not ask; a caller whose tool
            // arguments are quotations asks with
            // [`Builder::free_tool_calls_from_penalties`].
            free_tool_calls_from_penalties: false,
            hits: HitsConfig::default(),
            focus: false,
            system_prompt: SystemPromptSchema {
                items: vec![SystemPromptItem::Section(SectionSchema {
                    id: section_id,
                    name: "frame".to_string(),
                    content: system_prompt_text.to_string(),
                    priority: 50.0,
                    depends_on: None,
                    depends_on_absent: None,
                    depends_on_configured: None,
                    depends_on_unconfigured: None,
                    is_template: false,
                    min_tokens: 0,
                    template_tokens: None,
                })],
                pins: PinSet::default(),
            },
            layers: vec![LayerSchema {
                id: layer_id,
                name: "dialogue".to_string(),
                description: String::new(),
                score_threshold: 0.0,
                window: 32_768,
                // A programmatic schema has no ingest layer to read from, so
                // there is nothing for a working set to carry.
                working_set: None,
                budget: Budget {
                    priority: 100.0,
                    min_percent: None,
                    max_percent: None,
                    adaptive: None,
                },
                dials: LayerDials::default(),
                // Template-less fallback summary so a plain-prompt conversation
                // is still summarisable (the production path supplies its own
                // tailored summary via the YAML template). Generic faithful
                // compression — not a product prompt. Only `turns` is provided;
                // SummaryOfSummaries nodes reuse it.
                summary: LayerSummary {
                    turns: TurnSummary {
                        scope: Scope::Union,
                        assistant: CompressionPrompt {
                            system_prompt: SectionSchema {
                                id: summary_a_id,
                                name: "__summary_turns_assistant__".to_string(),
                                content: "You are a faithful compressor. Compress the assistant's \
                                          reply below into a much shorter version that preserves \
                                          the specific facts, numbers, decisions, and advice, in \
                                          the original voice. No commentary, headings, or lists."
                                    .to_string(),
                                priority: 50.0,
                                depends_on: None,
                                depends_on_absent: None,
                                depends_on_configured: None,
                                depends_on_unconfigured: None,
                                is_template: false,
                                min_tokens: 0,
                                template_tokens: None,
                            },
                            user_prompt:
                                "Compress the assistant reply above into a faithful, much \
                                          shorter version that preserves the specific facts, \
                                          numbers, decisions, and advice, in the original voice."
                                    .to_string(),
                        },
                        max_tokens: 384,
                        content: Content::Decode,
                    },
                    summaries: None,
                },
                groups: vec![GroupSchema {
                    id: group_id,
                    name: PLAIN_PROMPT_GROUP.to_string(),
                    selection: SelectionRule::AlwaysVisible,
                    score_threshold: None,
                    policy_band_declared: false,
                    policy: SelectionPolicy::default_policy(),
                    budget: Budget {
                        priority: 100.0,
                        min_percent: None,
                        max_percent: None,
                        adaptive: None,
                    },
                    budget_adaptive: None,
                    locality: None,
                    anchor: None,
                    max_swaps: DEFAULT_MAX_SWAPS,
                    draws: None,
                    recency: None,
                }],
                policy: SelectionPolicy::default_policy(),
                gather_scope: GatherScope::default(),
                gathered: true,
                // The only layer, so the bottom of its own stack.
                rank: 0,
                decode_priority,
                on_corrupt_turn: CorruptTurnPolicy::DropTurn,
                ingest_unit: None,
                in_tool_rounds: true,
                ingest: None,
                calibrate: None,
            }],
        };
        validate(&schema).expect("synthetic schema must always be valid");

        // Populate name maps so `id_for_layer` / `id_for_group` /
        // `id_for_section_in` work the same as for YAML-derived builders.
        let mut name_maps = NameMaps::default();
        name_maps
            .layer_names
            .insert("dialogue".to_string(), layer_id);
        name_maps
            .group_names
            .insert(PLAIN_PROMPT_GROUP.to_string(), group_id);
        name_maps
            .section_names
            .insert((LayerId::system_prompt(), "frame".to_string()), section_id);

        // A synthetic single-section schema, not parsed from YAML.
        Self {
            schema,
            name_maps,
            source_yaml: None,
            script: None,
        }
    }
}
