//! What a daemon's build says of itself (`docs/replicated_substrate.md` §3.9):
//! the substrate format it opens, every record type it decodes, and the model
//! each of its apps runs — what a member checks an artifact against before it
//! installs it. And the one thing every binary does before it parses a flag:
//! answer `--build-info`.

use std::collections::BTreeMap;

use candle_conversation::models::Model;
use candle_conversation::persistence::log_file::FILE_FORMAT_VERSION;
use candle_conversation::persistence::survival::WRITTEN_RECORD_TYPES;
use cluster::version::info::{answer_if_asked, Opens};

/// Each app a binary can run, by name, and the model that app's tenant names.
pub type AppModels = &'static [(&'static str, Model)];

/// What a build running `apps` opens.
pub fn opens(apps: &[(&str, Model)]) -> Opens {
    let mut record_tags: Vec<u8> = WRITTEN_RECORD_TYPES.iter().map(|rt| rt.tag()).collect();
    record_tags.sort_unstable();
    Opens {
        log_format: Some(FILE_FORMAT_VERSION),
        record_tags,
        models: models_of(apps),
    }
}

/// Each app and the name its model is known by: the preset's name, or a
/// custom model's file.
fn models_of(apps: &[(&str, Model)]) -> BTreeMap<String, String> {
    apps.iter()
        .map(|(app, model)| {
            let name = model
                .override_key()
                .unwrap_or_else(|| model.clone().spec().model_filename);
            (app.to_string(), name)
        })
        .collect()
}

/// Before anything else: `--build-info` answers and exits. `kind` is the
/// binary's: `zend`, `npcd`, `motherd`; `apps` every app it can run.
pub fn first(kind: &str, apps: AppModels) {
    answer_if_asked(kind, || opens(apps));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build opens the log format it writes, decodes every record type it
    /// writes, in tag order, and names each app's model by its preset.
    #[test]
    fn a_build_opens_what_it_writes() {
        let o = opens(&[
            ("zend", Model::Qwen38_FlashNext),
            ("npcd", Model::Qwen36_35B_A3B_AntiLoop_StyleTune),
        ]);
        assert_eq!(o.log_format, Some(FILE_FORMAT_VERSION));
        assert_eq!(o.record_tags.len(), WRITTEN_RECORD_TYPES.len());
        assert!(o.record_tags.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(
            o.models,
            BTreeMap::from([
                (
                    "npcd".to_string(),
                    "Qwen36_35B_A3B_AntiLoop_StyleTune".to_string()
                ),
                ("zend".to_string(), "Qwen38_FlashNext".to_string()),
            ])
        );
    }
}
