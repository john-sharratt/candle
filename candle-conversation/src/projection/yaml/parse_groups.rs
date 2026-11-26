//! Turns a group's YAML blocks into typed selection, budget, locality, anchor and
//! draw settings, checking each rail as it goes.
//!
//! Each parser checks the values it reads before the engine uses them. A priority
//! or percentage out of range is an error. An adaptive budget's ceiling must not
//! sit below the static ceiling it overrides. A draw may name only a group declared
//! earlier in its layer. A collection cannot use a working-set selection, because
//! that selection carries whole conversations.

use super::*;

/// A group's `selection.recency`, for a working-set group with a positive
/// half-life and a metadata key to date it by.
pub(super) fn parse_recency(
    group: &str,
    ys: &YamlSelection,
    selection: &SelectionRule,
) -> Result<Option<Recency>, ConstructionError> {
    let Some(yr) = &ys.recency else {
        return Ok(None);
    };
    let invalid = |reason: &str| ConstructionError::InvalidPolicy {
        name: group.to_string(),
        reason: format!("selection.recency: {reason}"),
    };
    if !matches!(selection, SelectionRule::WorkingSet { .. }) {
        return Err(invalid("only a working_set selection ages its units"));
    }
    if !(yr.half_life_days.is_finite() && yr.half_life_days > 0.0) {
        return Err(invalid("half_life_days must be a positive number of days"));
    }
    if yr.dated_by.trim().is_empty() {
        return Err(invalid(
            "dated_by must name the metadata that holds a unit's path",
        ));
    }
    Ok(Some(Recency {
        half_life_days: yr.half_life_days,
        dated_by: yr.dated_by.clone(),
    }))
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Resolve a group's `draws:` block against the groups of its layer declared
/// before it. A drawing group and the group it draws from are both working-set
/// groups — the units move between their shares, not into a belief selection —
/// and it names at least one path.
pub(super) fn parse_draws(
    group: &str,
    yd: Option<&YamlDraws>,
    selection: &SelectionRule,
    earlier: &[GroupSchema],
) -> Result<Option<Draws>, ConstructionError> {
    let Some(yd) = yd else {
        return Ok(None);
    };
    let invalid = |reason: String| ConstructionError::InvalidPolicy {
        name: group.to_string(),
        reason: format!("draws.{reason}"),
    };
    if !matches!(selection, SelectionRule::WorkingSet { .. }) {
        return Err(invalid(
            "a drawing group needs a working_set selection".into(),
        ));
    }
    let from = earlier.iter().find(|g| g.name == yd.from).ok_or_else(|| {
        invalid(format!(
            "from: {:?} is no earlier group of this layer",
            yd.from
        ))
    })?;
    if !from.is_working_set() || from.draws.is_some() {
        return Err(invalid(format!(
            "from: {:?} must be a working_set group that draws nothing itself",
            yd.from
        )));
    }
    if yd.paths.is_empty() || yd.paths.iter().any(String::is_empty) {
        return Err(invalid(
            "paths must name at least one non-empty prefix".into(),
        ));
    }
    if yd
        .except
        .iter()
        .any(|name| name.is_empty() || name.contains('/'))
    {
        return Err(invalid(
            "except must name files, not paths: no empty name and no '/'".into(),
        ));
    }
    Ok(Some(Draws {
        from: from.id,
        paths: yd.paths.clone(),
        except: yd.except.clone(),
    }))
}

/// Resolve a layer's `working_set:` block, refusing values the arithmetic
/// cannot use: a leak outside `(0, 1]`, a negative floor, shares that do not
/// make the room. The slot is the engine's trained window, which the schema
/// does not know: the reserve is measured against it at each fill, which runs
/// on the rung above when it passes it (`WorkingSetConfig::slot_window`).
pub(super) fn parse_working_set(
    layer: &str,
    yw: Option<&YamlWorkingSet>,
) -> Result<Option<WorkingSetConfig>, ConstructionError> {
    let Some(yw) = yw else {
        return Ok(None);
    };
    let invalid = |reason: &str| ConstructionError::InvalidPolicy {
        name: layer.to_string(),
        reason: format!("working_set.{reason}"),
    };
    if !(yw.beta > 0.0 && yw.beta <= 1.0) {
        return Err(invalid("beta must be in (0, 1]"));
    }
    if yw.min_momentum < 0.0 {
        return Err(invalid("min_momentum must be >= 0"));
    }
    let specs: Vec<TypeShareSpec> = yw
        .type_shares
        .iter()
        .map(|t| (t.name.clone(), t.files.clone(), t.max_share))
        .collect();
    let type_shares = TypeShares::new(&specs).map_err(|e| invalid(&format!("type_shares: {e}")))?;
    let s = &yw.shares;
    let fill_shares = FillShares::new(s.readings, s.folders, s.guides, s.journals, s.dialogue)
        .map_err(|e| invalid(&e))?;
    Ok(Some(WorkingSetConfig {
        beta: yw.beta,
        min_momentum: yw.min_momentum,
        reserve_tokens: yw.reserve_tokens,
        type_shares,
        fill_shares,
    }))
}

/// Parse a group's Concept C `locality:` block, validating its rails.
pub(super) fn parse_locality(
    name: &str,
    yl: Option<&YamlLocality>,
) -> Result<Option<LocalityConfig>, ConstructionError> {
    let Some(yl) = yl else { return Ok(None) };
    let bad = |reason: &str| ConstructionError::InvalidPolicy {
        name: name.to_string(),
        reason: reason.to_string(),
    };
    if !(0.0..=1.0).contains(&yl.decay) || yl.decay == 0.0 {
        return Err(bad("locality.decay must be in (0, 1]"));
    }
    if yl.seed_threshold < 0.0 {
        return Err(bad("locality.seed_threshold must be >= 0"));
    }
    if yl.extend_per < 0.0 {
        return Err(bad("locality.extend_per must be >= 0"));
    }
    Ok(Some(LocalityConfig {
        seed_threshold: yl.seed_threshold,
        decay: yl.decay,
        base_radius: yl.base_radius,
        extend_per: yl.extend_per,
        extra_radius_max: yl.extra_radius_max,
    }))
}

/// Parse a group's Concept D `anchor:` block.
pub(super) fn parse_anchor(
    name: &str,
    ya: Option<&YamlAnchor>,
) -> Result<Option<AnchorConfig>, ConstructionError> {
    let Some(ya) = ya else { return Ok(None) };
    match ya.member.as_str() {
        "first" => Ok(Some(AnchorConfig {
            member: AnchorMember::First,
        })),
        other => Err(ConstructionError::InvalidPolicy {
            name: name.to_string(),
            reason: format!("unknown anchor member {other:?} (first)"),
        }),
    }
}

/// Parse a group's Concept B `budget_adaptive:` block.
pub(super) fn parse_member_budget_adaptive(
    name: &str,
    yb: Option<&YamlMemberBudgetAdaptive>,
) -> Result<Option<MemberBudgetAdaptive>, ConstructionError> {
    let Some(yb) = yb else { return Ok(None) };
    if yb.per_extra <= 0.0 {
        return Err(ConstructionError::InvalidPolicy {
            name: name.to_string(),
            reason: "budget_adaptive.per_extra must be > 0".to_string(),
        });
    }
    Ok(Some(MemberBudgetAdaptive {
        per_extra: yb.per_extra,
        absolute_max: yb.absolute_max,
    }))
}

pub(super) fn validate_priority(name: &str, v: f32) -> Result<(), ConstructionError> {
    if v <= 0.0 {
        return Err(ConstructionError::InvalidPriority {
            name: name.to_string(),
            value: v,
        });
    }
    Ok(())
}

pub(super) fn validate_percent(name: &str, v: f32) -> Result<(), ConstructionError> {
    if !(0.0..=100.0).contains(&v) {
        return Err(ConstructionError::InvalidPercentage {
            name: name.to_string(),
            value: v,
        });
    }
    Ok(())
}

pub(super) fn parse_budget(name: &str, yb: &YamlBudget) -> Result<Budget, ConstructionError> {
    let priority = yb.priority.unwrap_or(50.0);
    validate_priority(name, priority)?;
    if let Some(v) = yb.min_percent {
        validate_percent(name, v)?;
    }
    if let Some(v) = yb.max_percent {
        validate_percent(name, v)?;
    }
    let adaptive = match &yb.adaptive {
        None => None,
        Some(ya) => {
            if ya.gain < 0.0 {
                return Err(ConstructionError::InvalidPolicy {
                    name: name.to_string(),
                    reason: "budget.adaptive.gain must be >= 0".to_string(),
                });
            }
            if let Some(v) = ya.max_percent {
                validate_percent(name, v)?;
                // The adaptive ceiling is the outer rail: it must not sit
                // below the static ceiling it overrides.
                if let Some(stat) = yb.max_percent {
                    if v < stat {
                        return Err(ConstructionError::InvalidPolicy {
                            name: name.to_string(),
                            reason: "budget.adaptive.max_percent must be >= max_percent"
                                .to_string(),
                        });
                    }
                }
            }
            Some(BudgetAdaptive {
                gain: ya.gain,
                max_percent: ya.max_percent,
            })
        }
    };
    Ok(Budget {
        priority,
        min_percent: yb.min_percent,
        max_percent: yb.max_percent,
        adaptive,
    })
}

/// A collection's selection rule: any a turn group may declare but
/// `working_set`, which carries whole conversations and a collection has none.
pub(super) fn parse_collection_selection(
    name: &str,
    ys: &YamlSelection,
) -> Result<SelectionRule, ConstructionError> {
    match parse_selection(name, ys)? {
        SelectionRule::WorkingSet { .. } => Err(ConstructionError::InvalidPolicy {
            name: name.to_string(),
            reason: "a collection cannot select by working_set".to_string(),
        }),
        rule => Ok(rule),
    }
}

pub(super) fn parse_selection(
    name: &str,
    ys: &YamlSelection,
) -> Result<SelectionRule, ConstructionError> {
    match ys.kind.as_str() {
        "" | "always_visible" => Ok(SelectionRule::AlwaysVisible),
        "top_k" => {
            let k = ys.k.ok_or_else(|| ConstructionError::InvalidTopK {
                name: name.to_string(),
            })?;
            if k == 0 {
                return Err(ConstructionError::InvalidTopK {
                    name: name.to_string(),
                });
            }
            Ok(SelectionRule::TopK { k })
        }
        "single" => Ok(SelectionRule::Single),
        "named" => {
            let selector = ys
                .selector
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ConstructionError::EmptyNamedSelector {
                    name: name.to_string(),
                })?;
            Ok(SelectionRule::Named { selector })
        }
        "offered" => {
            let selector = ys
                .selector
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ConstructionError::EmptyNamedSelector {
                    name: name.to_string(),
                })?;
            let k =
                ys.k.filter(|k| *k > 0)
                    .ok_or_else(|| ConstructionError::InvalidTopK {
                        name: name.to_string(),
                    })?;
            Ok(SelectionRule::Offered { selector, k })
        }
        "conversation" => {
            let recent = ys.recent.unwrap_or(0);
            let historical_top_k = ys.historical_top_k.unwrap_or(0);
            if recent == 0 && historical_top_k == 0 {
                return Err(ConstructionError::InvalidConversationK {
                    name: name.to_string(),
                });
            }
            let lineage_base = ys.lineage_base.unwrap_or(LINEAGE_BASE);
            if !lineage_base.is_finite() || lineage_base < 0.0 {
                return Err(ConstructionError::InvalidPolicy {
                    name: name.to_string(),
                    reason: format!("lineage_base must be a finite score >= 0, got {lineage_base}"),
                });
            }
            Ok(SelectionRule::Sequence {
                recent,
                historical_top_k,
                lineage_base,
            })
        }
        "working_set" => {
            let share = match ys.share.as_deref() {
                Some("readings") => WorkingSetShare::Readings,
                Some("folders") => WorkingSetShare::Folders,
                Some("guides") => WorkingSetShare::Guides,
                Some("journals") => WorkingSetShare::Journals,
                other => {
                    let reason = format!(
                        "a working_set selection needs share: readings | folders | guides | \
                         journals, got {other:?}"
                    );
                    return Err(ConstructionError::InvalidPolicy {
                        name: name.to_string(),
                        reason,
                    });
                }
            };
            Ok(SelectionRule::WorkingSet { share })
        }
        other => Err(ConstructionError::UnknownSelectionKind(other.to_string())),
    }
}

/// Convert the YAML `default:` block into a [`SelectionDefault`], dropping an
/// empty tag (treated as absent).
pub(super) fn parse_default(yd: Option<&YamlSelectionDefault>) -> Option<SelectionDefault> {
    yd.map(|d| SelectionDefault {
        tag: d.tag.trim().to_string(),
    })
    .filter(|d| !d.tag.is_empty())
}
