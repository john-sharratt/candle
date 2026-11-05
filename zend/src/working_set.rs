//! The dialogue's working set, as the daemon drives it
//! (`docs/zend_working_set.md`): what it is seeded with ([`seeds`]), which
//! ingested readings stand for a file ([`corpus`]), and whether a reading
//! covers its whole file ([`coverage`]).
//!
//! The state and the projection rule live in `candle_conversation`; this is
//! the part that knows about tools, repositories and turns.

pub mod corpus;
pub mod coverage;
pub mod seeds;

use anyhow::bail;
use candle_conversation::projection::{LayerId, Schema};
use candle_conversation::working_set::WorkingSetConfig;
use zend_tools::registry;

/// The working set the dialogue layer declares, with every `release_on` name
/// checked against the tool registry.
///
/// An unknown name is a load error, not a warning: a release rule that names
/// no tool never fires, so a misspelt write would leave every lock standing
/// across the very change it exists to catch.
pub fn dialogue_config(
    schema: &Schema,
    dialogue: LayerId,
) -> anyhow::Result<Option<WorkingSetConfig>> {
    let Some(config) = schema
        .layers
        .iter()
        .find(|l| l.id == dialogue)
        .and_then(|l| l.working_set.clone())
    else {
        return Ok(None);
    };
    let unknown: Vec<&str> = config
        .release_on
        .iter()
        .map(String::as_str)
        .filter(|name| registry::find(name).is_none())
        .collect();
    if !unknown.is_empty() {
        bail!("working_set.release_on names no registered tool: {unknown:?}");
    }
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::models::Dialect;
    use candle_conversation::projection::Builder;

    /// Every name in the bundled schema's `release_on` is a registered tool —
    /// the check `dialogue_config` makes at load.
    #[test]
    fn the_bundled_release_on_names_real_tools() {
        let dialect = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(
            include_str!("prompts/projection.yaml"),
            &[("workspace", "test")],
            Some(&dialect),
        )
        .expect("projection.yaml must parse");
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let c = dialogue_config(b.schema(), dialogue)
            .expect("every release_on name is registered")
            .expect("the dialogue layer declares a working set");
        for name in [
            "write",
            "file_edit",
            "file_delete",
            "git_commit",
            "git_merge",
            "git_ref",
            "git_switch",
            "git_reset",
            "run_command",
            "code_run",
            "code_session_exec",
        ] {
            assert!(c.releases(name), "{name} releases the working set");
        }
    }
}
