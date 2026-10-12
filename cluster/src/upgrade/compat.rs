//! Whether a build can open a member's substrate (§3.9): the same log format,
//! every record type the substrate holds, and the model the app is locked to.
//! A build that fails any of them is never installed there: the member is
//! stranded, with the reason.

use crate::version::info::Opens;

/// What a member's substrate needs of a build: its log's format, the record
/// types it holds now, and the model its build runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Needs {
    pub log_format: Option<u32>,
    /// Every record type the substrate holds, by tag.
    pub census: Vec<u8>,
    pub model: Option<String>,
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
    if opens.model != needs.model {
        return Err(format!(
            "the build runs {}, the app runs {}: a new model is moved to by hand",
            opens.model.as_deref().unwrap_or("no model"),
            needs.model.as_deref().unwrap_or("no model")
        ));
    }
    Ok(())
}

fn spell(v: Option<u32>) -> String {
    v.map_or("none".into(), |v| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opens() -> Opens {
        Opens {
            log_format: Some(3),
            record_tags: vec![1, 2, 3, 31],
            model: Some("Qwen38_FlashNext_Q4KO".into()),
        }
    }

    fn needs() -> Needs {
        Needs {
            log_format: Some(3),
            census: vec![1, 31],
            model: Some("Qwen38_FlashNext_Q4KO".into()),
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
            model: Some("Qwen36_35B".into()),
            ..opens()
        };
        assert_eq!(
            check(&other_model, &needs()).unwrap_err(),
            "the build runs Qwen36_35B, the app runs Qwen38_FlashNext_Q4KO: a new model is \
             moved to by hand"
        );
    }
}
