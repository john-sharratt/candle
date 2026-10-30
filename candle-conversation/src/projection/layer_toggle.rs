//! Runtime, in-memory projection-layer kill switch — a DIAGNOSTIC toggle that
//! excludes a layer from projection assembly without touching the schema or
//! persisting anything. Flipping a layer off makes [`super::project::run`] skip
//! it on the next (re)projection; a restart clears every toggle.
//!
//! Distinct from the boot-time `DaemonConfig::disabled_layers`, which is an
//! OPERATOR decision covering the whole process: that flag clears
//! [`super::schema::LayerSchema::gathered`], so the layer is not ingested, not
//! normalization-warmed, not scored and not assembled — out of service until the
//! daemon is relaunched without it. This toggle is a DIAGNOSTIC: flipped live,
//! forgotten on restart, and aimed at an *already-populated, otherwise healthy*
//! layer so you can A/B whether that layer is what's breaking coherence and then
//! flip it straight back.
//!
//! Keyed by layer NAME — the stable external key, since `LayerId` is only stable
//! within one schema build — and held on the substrate
//! ([`crate::substrate::Substrate::toggle_layer`]), so it affects every
//! conversation on that substrate and no conversation on another. Assembly reads
//! it through [`crate::substrate::ContentResolver::layer_disabled`].

use crate::projection::Conversation;

impl Conversation {
    /// Flip the diagnostic toggle on layer `name` for this substrate; returns
    /// the new state (`true` = now excluded from projection assembly).
    pub fn toggle_layer(&self, name: &str) -> bool {
        self.update(|s| s.toggle_layer(name))
    }

    /// Whether layer `name` is toggled out on this substrate.
    pub fn is_layer_disabled(&self, name: &str) -> bool {
        self.read().is_layer_disabled(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_flips_and_reports() {
        let conv = Conversation::ephemeral();
        let n = "dialogue";
        assert!(!conv.is_layer_disabled(n));
        assert!(conv.toggle_layer(n), "first toggle disables");
        assert!(conv.is_layer_disabled(n));
        assert!(conv.read().disabled_layers().contains(n));
        assert!(!conv.toggle_layer(n), "second toggle re-enables");
        assert!(!conv.is_layer_disabled(n));
    }

    /// Two substrates' toggles are independent, even for a layer both name the
    /// same: a toggle in one tenant must not reach the other.
    #[test]
    fn a_toggle_on_one_substrate_does_not_reach_another() {
        let zend = Conversation::ephemeral();
        let npcd = Conversation::ephemeral();
        assert!(zend.toggle_layer("memory"));
        assert!(zend.is_layer_disabled("memory"));
        assert!(!npcd.is_layer_disabled("memory"));
        assert!(!npcd.read().any_layer_disabled());
    }

    /// The fast path follows the set: while any layer is toggled off every
    /// query consults it, and it clears only with the last entry.
    #[test]
    fn the_fast_path_tracks_the_set_not_one_name() {
        let conv = Conversation::ephemeral();
        assert!(conv.toggle_layer("a"));
        assert!(conv.toggle_layer("b"));
        assert!(!conv.toggle_layer("a"));
        assert!(conv.read().any_layer_disabled());
        assert!(conv.is_layer_disabled("b"));
        assert!(!conv.toggle_layer("b"));
        assert!(!conv.read().any_layer_disabled());
    }
}
