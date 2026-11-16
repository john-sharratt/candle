//! The hidden projection every calibration case is written under.

use candle_conversation::projection::{
    Builder, DecodePriority, GroupId, LayerId, Reserved, SelectionRule,
};

use super::exemplar::Calibrated;

/// The selector the calibration projection's `tools` collection reads
/// ([`SelectionRule::Named`]) to pin exactly one tool. The phase sets it per
/// case through the turn's selection.
pub const TOOL_SELECTOR: &str = "tool";

/// The gather-scope tag every calibration turn carries: a live `tools`
/// collection whose policy names it reads these turns as its belief gallery.
pub const TOOL_SCOPE_TAG: &str = "tool";

/// Section-id partition for the calibration projection: high in the u32 space,
/// just below the per-kind reserved singleton band (`u32::MAX - 0..3`) and far
/// above the user schema's low 1..n ids and the tool ids allocated above them.
/// 4096 ids is ample headroom for the frame, its summary framing, the whole tool
/// catalog (one section each), and the outro.
const SECTION_BASE: u32 = u32::MAX - 4096;

/// The system prompt a case's one tool is shown inside: what comes before the
/// tool's line, and what comes after it.
///
/// The daemon's own, because it is what its model reads around a tool when it
/// chooses one live — the instruction to call, and the envelope a call is
/// written in. The system-prompt envelope markers live in the text itself (the
/// assembler does not wrap system sections): `open` begins with the dialect's
/// `system_start` and `close` ends with its `system_end`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub open: String,
    pub close: String,
}

/// Build the calibration projection: a single [`Reserved::Calibration`] layer
/// whose system prompt frames a `tools` collection holding the **whole**
/// catalog — one name-keyed section per tool — governed by
/// [`SelectionRule::Named`] on [`TOOL_SELECTOR`]. A case pins one tool by
/// setting that selector to the tool's name, so the projection emits exactly
/// that tool between the frame's two halves.
///
/// The reserved layer keeps these conversations off every user projection
/// (their turns never enter dialogue retrieval), and the collection's distinct,
/// name-keyed member sections mean no two tools ever collide on one slot.
///
/// Background to any dialogue that is up while it runs: [`DecodePriority::Low`],
/// the band that steps aside for a live turn.
pub fn build<T: Calibrated>(
    frame: &Frame,
    defs: &[T],
) -> anyhow::Result<(Builder, LayerId, GroupId)> {
    let layer = LayerId::reserved(Reserved::Calibration);
    let group = GroupId::reserved(Reserved::Calibration);
    let mut builder = Builder::for_reserved_corpus(
        &frame.open,
        Reserved::Calibration,
        SECTION_BASE,
        DecodePriority::Low,
    );
    let collection = builder
        .add_collection(
            "tools",
            SelectionRule::Named {
                selector: TOOL_SELECTOR.to_string(),
            },
            0.0,
        )
        .map_err(|e| anyhow::anyhow!("calibration add_collection: {e}"))?;
    for def in defs {
        builder
            .add_section_to_collection(collection, def.name().to_string(), def.line(), 100.0)
            .map_err(|e| anyhow::anyhow!("calibration add tool {}: {e}", def.name()))?;
    }
    builder
        .add_section("tools_outro", frame.close.clone(), 50.0)
        .map_err(|e| anyhow::anyhow!("calibration add outro: {e}"))?;
    Ok((builder, layer, group))
}
