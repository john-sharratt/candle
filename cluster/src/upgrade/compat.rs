//! Whether a build can open a member's substrate (§3.9): the same log format,
//! every record type the substrate holds, and, for each app the member runs,
//! the model that app is locked to. A build that fails any of them is never
//! installed there: the member is stranded, with the reason.

use std::collections::BTreeMap;

use crate::version::info::Opens;

/// What a member's substrate needs of a build: its log's format, the record
/// types it holds now, and the model each of its apps runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Needs {
    pub log_format: Option<u32>,
    /// Every record type the substrate holds, by tag.
    pub census: Vec<u8>,
    /// Each app the member runs, by name, and its model's preset.
    pub models: BTreeMap<String, String>,
}

/// `Ok` when a build that opens `opens` can take over a substrate that
/// `needs`; otherwise why not, as the status page shows it.
pub fn check(opens: &Opens, needs: &Needs) -> Result<(), String> {
    if opens.log_format != needs.log_format {
        return Err(format!(
            "log format {} ≠ the substrate's {}: the build could not open it",
            spell(opens.log_format),
            spell(needs.log_format)
        ));
    }
    let lacking: Vec<String> = needs
        .census
        .iter()
        .filter(|t| !opens.record_tags.contains(t))
        .map(u8::to_string)
        .collect();
    if !lacking.is_empty() {
        return Err(format!(
            "the substrate holds record types {} the build does not decode: they would be \
             invisible to it, then compacted away",
            lacking.join(", ")
        ));
    }
    // Only the apps the member runs: an app the build opens and the member
    // leaves out holds no substrate here, and which apps a member runs is its
    // config's to say, not the build's.
    for (app, runs) in &needs.models {
        match opens.models.get(app) {
            Some(built) if built == runs => {}
            Some(built) => {
                return Err(format!(
                    "the build runs {app} on {built}, the app runs {runs}: a new model is moved \
                     to by hand"
                ))
            }
            None => {
                return Err(format!(
                    "the build does not run {app}, which this member does: its substrate would \
                     have no daemon to open it"
                ))
            }
        }
    }
    Ok(())
}

fn spell(v: Option<u32>) -> String {
    v.map_or("none".into(), |v| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(a, m)| (a.to_string(), m.to_string()))
            .collect()
    }

    fn opens() -> Opens {
        Opens {
            log_format: Some(3),
            record_tags: vec![1, 2, 3, 31],
            models: models(&[("npcd", "Qwen38_FlashNext"), ("zend", "Qwen38_FlashNext")]),
        }
    }

    fn needs() -> Needs {
        Needs {
            log_format: Some(3),
            census: vec![1, 31],
            models: models(&[("npcd", "Qwen38_FlashNext"), ("zend", "Qwen38_FlashNext")]),
        }
    }

    #[test]
    fn a_build_that_opens_everything_fits() {
        assert_eq!(check(&opens(), &needs()), Ok(()));
        // An agent: nothing to open, nothing needed.
        assert_eq!(check(&Opens::default(), &Needs::default()), Ok(()));
    }

    /// Each check strands the member with its reason.
    #[test]
    fn each_mismatch_says_why() {
        let other_format = Opens {
            log_format: Some(4),
            ..opens()
        };
        assert_eq!(
            check(&other_format, &needs()).unwrap_err(),
            "log format 4 ≠ the substrate's 3: the build could not open it"
        );
        let older = Opens {
            record_tags: vec![1, 2],
            ..opens()
        };
        assert_eq!(
            check(&older, &needs()).unwrap_err(),
            "the substrate holds record types 31 the build does not decode: they would be \
             invisible to it, then compacted away"
        );
        let other_model = Opens {
            models: models(&[
                ("npcd", "Qwen36_35B_A3B_AntiLoop_StyleTune"),
                ("zend", "Qwen38_FlashNext"),
            ]),
            ..opens()
        };
        assert_eq!(
            check(&other_model, &needs()).unwrap_err(),
            "the build runs npcd on Qwen36_35B_A3B_AntiLoop_StyleTune, the app runs \
             Qwen38_FlashNext: a new model is moved to by hand"
        );
        let without_an_app = Opens {
            models: models(&[("zend", "Qwen38_FlashNext")]),
            ..opens()
        };
        assert_eq!(
            check(&without_an_app, &needs()).unwrap_err(),
            "the build does not run npcd, which this member does: its substrate would have no \
             daemon to open it"
        );
    }

    /// An app the build opens and the member leaves out is not the member's
    /// to hold to a model: BlackWorld, running zend alone, takes a build that
    /// moved npcd to another model.
    #[test]
    fn an_app_the_member_leaves_out_is_not_held_to_its_model() {
        let moved = Opens {
            models: models(&[
                ("npcd", "Qwen36_35B_A3B_AntiLoop_StyleTune"),
                ("zend", "Qwen38_FlashNext"),
            ]),
            ..opens()
        };
        let zend_only = Needs {
            models: models(&[("zend", "Qwen38_FlashNext")]),
            ..needs()
        };
        assert_eq!(check(&moved, &zend_only), Ok(()));
    }

    /// Each app is held to its own model: a build that moves one app and
    /// leaves the other is checked app by app, not as one model for both.
    #[test]
    fn two_apps_on_two_models_fit_when_each_matches() {
        let two = models(&[
            ("npcd", "Qwen36_35B_A3B_AntiLoop_StyleTune"),
            ("zend", "Qwen38_FlashNext"),
        ]);
        let opens = Opens {
            models: two.clone(),
            ..opens()
        };
        let needs = Needs {
            models: two,
            ..needs()
        };
        assert_eq!(check(&opens, &needs), Ok(()));
    }
}
