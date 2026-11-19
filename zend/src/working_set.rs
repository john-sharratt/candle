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

use candle_conversation::projection::{LayerId, Schema};
use candle_conversation::working_set::WorkingSetConfig;

/// The working set the dialogue layer declares, if it declares one.
pub fn dialogue_config(schema: &Schema, dialogue: LayerId) -> Option<WorkingSetConfig> {
    schema
        .layers
        .iter()
        .find(|l| l.id == dialogue)
        .and_then(|l| l.working_set.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::models::Dialect;
    use candle_conversation::projection::Builder;

    /// The bundled schema's dialogue layer declares a working set.
    #[test]
    fn the_bundled_dialogue_layer_declares_a_working_set() {
        let dialect = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(
            include_str!("prompts/projection.yaml"),
            &[("workspace", "test"), ("model", "test-model")],
            Some(&dialect),
        )
        .expect("projection.yaml must parse");
        let dialogue = b.id_for_layer("dialogue").unwrap();
        assert!(dialogue_config(b.schema(), dialogue).is_some());
    }
}
