//! Runtime, in-memory projection-layer kill switch — a DIAGNOSTIC toggle that
//! excludes a layer from projection assembly without touching the schema or
//! persisting anything. Flipping a layer off makes [`super::project::run`] skip
//! it on the next (re)projection; a restart clears every toggle.
//!
//! Distinct from the boot-time `DaemonConfig::disabled_layers`, which only
//! suppresses a layer's startup INGEST — there the layer stays fully projected.
//! This one excludes an *already-populated* layer from the assembled context so
//! you can A/B whether that layer is what's breaking coherence.
//!
//! Keyed by layer NAME — the stable external key, since `LayerId` is only stable
//! within one schema build. Process-global on purpose: the toggle affects the
//! live projection for every conversation, which is exactly the "does turning
//! this layer off restore coherence?" experiment it exists for.

use std::collections::HashSet;
use std::sync::{LazyLock, RwLock};

static DISABLED: LazyLock<RwLock<HashSet<String>>> = LazyLock::new(|| RwLock::new(HashSet::new()));

/// True while `name` is toggled OFF (excluded from projection assembly).
pub fn is_layer_disabled(name: &str) -> bool {
    DISABLED.read().map(|s| s.contains(name)).unwrap_or(false)
}

/// Flip `name`'s state; returns the NEW disabled flag (`true` = now excluded).
pub fn toggle_layer(name: &str) -> bool {
    let mut set = DISABLED.write().unwrap_or_else(|e| e.into_inner());
    if set.remove(name) {
        false
    } else {
        set.insert(name.to_string());
        true
    }
}

/// Snapshot of every currently-disabled layer name — for surfacing UI state.
pub fn disabled_layers() -> HashSet<String> {
    DISABLED.read().map(|s| s.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_flips_and_reports() {
        // Use a name unlikely to collide with any real layer under test.
        let n = "__layer_toggle_unit_test__";
        assert!(!is_layer_disabled(n));
        assert!(toggle_layer(n), "first toggle disables");
        assert!(is_layer_disabled(n));
        assert!(disabled_layers().contains(n));
        assert!(!toggle_layer(n), "second toggle re-enables");
        assert!(!is_layer_disabled(n));
    }
}
