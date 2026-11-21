//! Parses `policy:` blocks into typed policies, each layered on an inherited base
//! and checked for consistency.
//!
//! A named preset sets the base, individual fields override it, and an absent
//! block inherits unchanged. The fusion mode, the case unit, the gallery keep, the
//! votes per token and the power are validated together. The non-additive fusion
//! modes score each fold group by its own one-vote scan, and the validation
//! refuses settings that would not do that.

use super::*;

/// A `policy:` block: an optional preset base plus per-field overrides and an
/// optional gather-scope tag filter. Any field left unset inherits from the
/// enclosing node (layer → collection/group) or the schema default.
#[derive(Deserialize)]
pub(super) struct YamlPolicy {
    #[serde(default)]
    pub(super) preset: Option<String>,
    #[serde(default)]
    pub(super) beta: Option<f32>,
    #[serde(default)]
    pub(super) min_score: Option<f32>,
    #[serde(default)]
    pub(super) evict_score: Option<f32>,
    #[serde(default)]
    pub(super) early_window_tokens: Option<usize>,
    #[serde(default)]
    pub(super) early_min_score: Option<f32>,
    #[serde(default)]
    pub(super) early_evict_score: Option<f32>,
    #[serde(default)]
    pub(super) budget: Option<YamlPolicyBudget>,
    #[serde(default)]
    pub(super) tags: Option<Vec<String>>,
    /// Per-fold-group vote weights (`[g0, g1, g2]`); omitted ⇒ inherit / uniform.
    #[serde(default)]
    pub(super) layer_weights: Option<Vec<f32>>,
    /// `layer_weights` for the calibration competition's votes; omitted ⇒
    /// inherit / uniform.
    #[serde(default)]
    pub(super) calibration_layer_weights: Option<Vec<f32>>,
    /// Concept G fusion mode: `additive` | `content_gated` | `consensus_min` |
    /// `consensus_geo`, with `gate_group` for `content_gated`. Omitted ⇒ inherit.
    #[serde(default)]
    pub(super) fusion: Option<YamlFusion>,
    /// Concept F: pin the turn's question window as a second scanned probe.
    #[serde(default)]
    pub(super) question_pin: Option<bool>,
    /// Concept B: mass constants.
    #[serde(default)]
    pub(super) mass: Option<YamlMass>,
    /// What a turn group's scan scores as one case: `exchange` | `file`.
    /// Omitted ⇒ inherit.
    #[serde(default)]
    pub(super) cases: Option<String>,
    /// The weight of the calibration competition's vote against a file's
    /// own. Omitted ⇒ inherit.
    #[serde(default)]
    pub(super) calibration_weight: Option<f32>,
    /// Which of each scanned turn's tokens the GPU gallery holds: `all`, or
    /// the first `n` of each code within the turn. Omitted ⇒ inherit.
    #[serde(default)]
    pub(super) gallery_keep: Option<YamlKeep>,
    /// `gallery_keep` for the calibration competition's questions.
    #[serde(default)]
    pub(super) calibration_gallery_keep: Option<YamlKeep>,
    /// Cases a probe token votes for in the group's own competitions,
    /// `1..=8`. Omitted ⇒ inherit.
    #[serde(default)]
    pub(super) vote_top_k: Option<usize>,
    /// The power each kept vote is raised to before it is summed. Omitted ⇒
    /// inherit.
    #[serde(default)]
    pub(super) vote_power: Option<f32>,
    /// How a file-scored group's scores meet the band: `background` |
    /// `kind_z`. Omitted ⇒ inherit.
    #[serde(default)]
    pub(super) normalize: Option<String>,
    /// Leave the target's lineage out of the group's candidates. Omitted ⇒
    /// inherit.
    #[serde(default)]
    pub(super) exclude_lineage: Option<bool>,
}

/// `all`, or a count of tokens to keep per code.
#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum YamlKeep {
    PerCode(u8),
    Named(String),
}

/// `keep` as the policy `name`'s `field` names it.
pub(super) fn parse_keep(
    name: &str,
    field: &str,
    keep: &YamlKeep,
) -> Result<Keep, ConstructionError> {
    match keep {
        YamlKeep::PerCode(n) if *n > 0 => Ok(Keep::FirstPerCode(*n)),
        YamlKeep::Named(s) if s == "all" => Ok(Keep::All),
        _ => Err(ConstructionError::InvalidPolicy {
            name: name.to_string(),
            reason: format!("{field} must be `all` or a count of tokens per code above 0"),
        }),
    }
}

#[derive(Deserialize)]
pub(super) struct YamlFusion {
    pub(super) mode: String,
    #[serde(default)]
    pub(super) gate_group: usize,
}

#[derive(Deserialize)]
pub(super) struct YamlMass {
    #[serde(default)]
    pub(super) concentration_top_k: Option<usize>,
    #[serde(default)]
    pub(super) concentration_rho: Option<f32>,
}

#[derive(Deserialize)]
pub(super) struct YamlPolicyBudget {
    #[serde(default)]
    pub(super) min: Option<usize>,
    #[serde(default)]
    pub(super) max: Option<usize>,
}

/// Resolve a `policy:` block against an inherited base. A named `preset:` sets
/// the base config; individual fields then override it; `tags` replaces the
/// inherited scope. An absent block yields the inherited policy verbatim.
pub(super) fn parse_policy(
    name: &str,
    yp: Option<&YamlPolicy>,
    inherited: &SelectionPolicy,
) -> Result<SelectionPolicy, ConstructionError> {
    let Some(yp) = yp else {
        return Ok(inherited.clone());
    };
    let mut config = match &yp.preset {
        Some(p) => PolicyPreset::from_name(p)
            .ok_or_else(|| ConstructionError::UnknownPolicyPreset {
                name: name.to_string(),
                preset: p.clone(),
            })?
            .config(),
        None => inherited.config,
    };
    if let Some(v) = yp.beta {
        config.beta = v;
    }
    if let Some(v) = yp.min_score {
        config.min_score = v;
    }
    if let Some(v) = yp.evict_score {
        config.evict_score = v;
    }
    if let Some(v) = yp.early_window_tokens {
        config.early_window_tokens = v;
    }
    if let Some(v) = yp.early_min_score {
        config.early_min_score = v;
    }
    if let Some(v) = yp.early_evict_score {
        config.early_evict_score = v;
    }
    if let Some(b) = &yp.budget {
        if let Some(m) = b.min {
            config.budget_min = m;
        }
        if let Some(m) = b.max {
            config.budget_max = m;
        }
    }
    validate_policy(name, &config)?;
    let tags = match &yp.tags {
        Some(t) => t.clone(),
        None => inherited.tags.clone(),
    };
    let layer_weights = match &yp.layer_weights {
        Some(w) => w.clone(),
        None => inherited.layer_weights.clone(),
    };
    let calibration_layer_weights = match &yp.calibration_layer_weights {
        Some(w) => w.clone(),
        None => inherited.calibration_layer_weights.clone(),
    };
    let mut scan = inherited.scan;
    if let Some(f) = &yp.fusion {
        scan.fusion = match f.mode.as_str() {
            "additive" => FusionMode::Additive,
            "content_gated" => FusionMode::ContentGated {
                gate_group: f.gate_group,
            },
            "consensus_min" => FusionMode::ConsensusMin,
            "consensus_geo" => FusionMode::ConsensusGeo,
            other => {
                return Err(ConstructionError::InvalidPolicy {
                    name: name.to_string(),
                    reason: format!(
                        "unknown fusion mode {other:?} (additive | content_gated | \
                         consensus_min | consensus_geo)"
                    ),
                })
            }
        };
    }
    if let Some(q) = yp.question_pin {
        scan.question_pin = q;
    }
    if let Some(m) = &yp.mass {
        if let Some(k) = m.concentration_top_k {
            scan.mass_top_k = k;
        }
        if let Some(rho) = m.concentration_rho {
            scan.mass_rho = rho;
        }
    }
    if let Some(c) = &yp.cases {
        scan.cases = match c.as_str() {
            "exchange" => ScanCases::Exchange,
            "file" => ScanCases::File,
            other => {
                return Err(ConstructionError::InvalidPolicy {
                    name: name.to_string(),
                    reason: format!("unknown cases {other:?} (exchange | file)"),
                })
            }
        };
    }
    if let Some(w) = yp.calibration_weight {
        scan.calibration_weight = w;
    }
    if let Some(k) = &yp.gallery_keep {
        scan.gallery_keep = parse_keep(name, "gallery_keep", k)?;
    }
    if let Some(k) = &yp.calibration_gallery_keep {
        scan.calibration_gallery_keep = parse_keep(name, "calibration_gallery_keep", k)?;
    }
    if let Some(k) = yp.vote_top_k {
        scan.law.top_k = k;
    }
    if let Some(p) = yp.vote_power {
        scan.law.power = p;
    }
    if let Some(n) = &yp.normalize {
        scan.normalize = match n.as_str() {
            "background" => ScanNorm::Background,
            "kind_z" => ScanNorm::KindZ,
            other => {
                return Err(ConstructionError::InvalidPolicy {
                    name: name.to_string(),
                    reason: format!("unknown normalize {other:?} (background | kind_z)"),
                })
            }
        };
    }
    if let Some(x) = yp.exclude_lineage {
        scan.exclude_lineage = x;
    }
    validate_scan(name, &scan)?;
    if calibration_layer_weights
        .iter()
        .any(|w| w.is_nan() || *w < 0.0)
    {
        return Err(ConstructionError::InvalidPolicy {
            name: name.to_string(),
            reason: "calibration_layer_weights must each be >= 0".to_string(),
        });
    }
    Ok(SelectionPolicy {
        config,
        tags,
        layer_weights,
        calibration_layer_weights,
        scan,
    })
}

pub(super) fn validate_scan(name: &str, s: &ScanPolicy) -> Result<(), ConstructionError> {
    let bad = |reason: &str| ConstructionError::InvalidPolicy {
        name: name.to_string(),
        reason: reason.to_string(),
    };
    if s.mass_top_k == 0 {
        return Err(bad("mass.concentration_top_k must be > 0"));
    }
    if s.mass_rho < 0.0 {
        return Err(bad("mass.concentration_rho must be >= 0"));
    }
    if s.calibration_weight.is_nan() || s.calibration_weight < 0.0 {
        return Err(bad("calibration_weight must be >= 0"));
    }
    if !s.law.is_valid() {
        return Err(bad(&format!(
            "vote_top_k must be in 1..={} and vote_power a positive number",
            VoteLaw::MAX_TOP_K
        )));
    }
    // The non-additive modes score each fold group by its own one-vote scan.
    if s.law.top_k > 1 && s.fusion != FusionMode::Additive {
        return Err(bad("vote_top_k above 1 needs fusion: additive"));
    }
    // A file's kind is read once per file, and an exchange-scored file is
    // several cases.
    if s.normalize == ScanNorm::KindZ && s.cases != ScanCases::File {
        return Err(bad("normalize: kind_z needs cases: file"));
    }
    Ok(())
}

pub(super) fn validate_policy(name: &str, c: &PolicyConfig) -> Result<(), ConstructionError> {
    let bad = |reason: &str| ConstructionError::InvalidPolicy {
        name: name.to_string(),
        reason: reason.to_string(),
    };
    if !(0.0..=1.0).contains(&c.beta) {
        return Err(bad("beta must be in [0, 1]"));
    }
    if c.min_score < 0.0 || c.evict_score < 0.0 {
        return Err(bad("min_score and evict_score must be >= 0"));
    }
    if c.evict_score > c.min_score {
        return Err(bad("evict_score must be <= min_score (hysteresis band)"));
    }
    if c.early_min_score < 0.0 || c.early_evict_score < 0.0 {
        return Err(bad("early_min_score and early_evict_score must be >= 0"));
    }
    if c.early_evict_score > c.early_min_score {
        return Err(bad(
            "early_evict_score must be <= early_min_score (hysteresis band)",
        ));
    }
    if c.early_min_score > c.min_score {
        return Err(bad(
            "early_min_score must be <= min_score (the window only lowers the bar)",
        ));
    }
    if c.budget_max == 0 {
        return Err(bad("budget.max must be > 0"));
    }
    if c.budget_min > c.budget_max {
        return Err(bad("budget.min must be <= budget.max"));
    }
    Ok(())
}
