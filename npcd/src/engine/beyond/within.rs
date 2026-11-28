//! The acts a character's personality reaches, as its turn's grammar offers
//! them: wherever it stands ([`Availability::Reached`]).
//!
//! [`Availability::Reached`]: crate::engine::tools::Availability::Reached

use super::Group;
use crate::engine::tools::Within;

impl Within {
    /// This situation with every act of `groups` reached.
    pub fn reaching(mut self, groups: &[Group]) -> Self {
        self.reaches = groups
            .iter()
            .flat_map(|g| g.acts())
            .map(|a| (*a).to_string())
            .collect();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::beyond::is_mine;
    use crate::engine::tools::{specs_within, Mode};

    /// **A reach offers its acts, and the grammar then compiles them**; with
    /// none, nothing of them is offered.
    #[test]
    fn a_reach_offers_its_acts_and_the_grammar_compiles_them() {
        let within = Within::nowhere().reaching(&[Group::Reckoning]);
        assert_eq!(within.reaches, ["calculator", "unit_convert", "random"]);
        let offered = |w: &Within| -> Vec<String> {
            let mut names: Vec<String> = specs_within(Mode::Physical, w)
                .into_iter()
                .map(|s| s.name)
                .filter(|n| is_mine(n))
                .collect();
            names.sort();
            names
        };
        assert_eq!(offered(&within), ["calculator", "random", "unit_convert"]);
        assert!(offered(&Within::nowhere()).is_empty());
    }
}
