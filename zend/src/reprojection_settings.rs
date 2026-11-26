//! How zend's conversations reproject in the middle of a decode: the probe's
//! reach and whether a think block reprojects (`reproject_probe_tokens:` and
//! `reproject_in_thinking:` in `motherd.yaml`'s `zend:` section, or their
//! flags).

use candle_conversation::SequenceConfig;

/// `--reproject-probe-tokens`'s default: a reprojection's probe takes the last
/// 128 decoded tokens.
pub const REPROJECT_PROBE_TOKENS: usize = 128;

/// The reprojection settings every conversation zend opens decodes under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReprojectionSettings {
    /// How many decoded tokens, back from the decode position, a
    /// reprojection's probe takes (`SequenceConfig::reproject_max_probe_tokens`).
    pub probe_tokens: usize,
    /// Whether a boundary inside the think block reprojects
    /// (`SequenceConfig::reproject_in_thinking`).
    pub in_thinking: bool,
}

impl Default for ReprojectionSettings {
    fn default() -> Self {
        Self {
            probe_tokens: REPROJECT_PROBE_TOKENS,
            in_thinking: true,
        }
    }
}

impl ReprojectionSettings {
    /// Set both on `config`.
    pub fn apply(&self, config: &mut SequenceConfig) {
        config.reproject_max_probe_tokens = self.probe_tokens.max(1);
        config.reproject_in_thinking = self.in_thinking;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MODEL;

    #[test]
    fn the_settings_land_on_the_conversation_config() {
        let mut config = MODEL.conversation_config();
        ReprojectionSettings {
            probe_tokens: 96,
            in_thinking: false,
        }
        .apply(&mut config);
        assert_eq!(config.reproject_max_probe_tokens, 96);
        assert!(!config.reproject_in_thinking);
    }

    #[test]
    fn a_zero_probe_takes_one_token() {
        let mut config = MODEL.conversation_config();
        ReprojectionSettings {
            probe_tokens: 0,
            in_thinking: true,
        }
        .apply(&mut config);
        assert_eq!(config.reproject_max_probe_tokens, 1);
        assert!(config.reproject_in_thinking);
    }
}
