//! Fixed personality presets shared by conversation hosts.
//!
//! Presets are data-only YAML documents. They are loaded by the common
//! conversation layer so `npcd`, `zend`, and other hosts can select the same
//! personality definition without copying it into application crates.

use std::io::Read;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct PersonalityPreset {
    pub trigger_prompts: Vec<String>,
    pub personality_steers: Vec<PersonalitySteer>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PersonalitySteer {
    pub text: String,
    pub weight: f32,
}

impl PersonalityPreset {
    pub fn from_reader(reader: impl Read) -> anyhow::Result<Self> {
        Ok(serde_yaml::from_reader(reader)?)
    }

    pub fn from_path(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_reader(file)
    }
}
