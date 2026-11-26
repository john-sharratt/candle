//! One engine slot per model the tenants name (`docs/parasite_model.md` §4).
//!
//! Tenants that name the same model share one slot, so one engine: the
//! highest-ranked of them builds it with every one's requests merged, and the
//! rest attach their substrates. Tenants on different models take different
//! slots, so different engines. A daemon the process leaves out still asks of
//! the engine of its model, through that model's slot — and asks nothing of a
//! model no running tenant names, since no engine is built for it.

use std::collections::HashMap;
use std::sync::Arc;

use candle_conversation::engine_slot::CoreRequests;
use candle_conversation::models::Model;
use candle_conversation::{Capabilities, EngineSlot, EngineSource, TenantRank};

/// A tenant this process runs: its rank, its name, its model.
pub struct Seated {
    pub rank: TenantRank,
    pub name: &'static str,
    pub model: Model,
}

/// A daemon this process leaves out, and what it asks of its model's engine.
pub struct Standing {
    pub rank: TenantRank,
    pub name: &'static str,
    pub model: Model,
    pub asks: CoreRequests,
}

/// The slot each tenant takes its engine from.
pub struct Slots {
    by_tenant: HashMap<&'static str, Arc<EngineSlot>>,
}

/// One model's tenants and the asks of the daemons standing on it, keyed by
/// its checkpoint.
type Group = (
    (String, String),
    Vec<&'static str>,
    Vec<(TenantRank, CoreRequests)>,
);

/// What makes two models one: the checkpoint each resolves to.
fn checkpoint(model: &Model) -> (String, String) {
    let spec = model.clone().spec();
    (spec.model_repo, spec.model_filename)
}

impl Slots {
    /// A slot per distinct model of `tenants`, each seating the standing
    /// daemons of its model. A standing daemon whose model no tenant runs is
    /// left out, and said so.
    pub fn plan(tenants: &[Seated], standing: Vec<Standing>) -> Self {
        let mut groups: Vec<Group> = Vec::new();
        for t in tenants {
            let key = checkpoint(&t.model);
            match groups.iter_mut().find(|(k, _, _)| *k == key) {
                Some((_, names, _)) => names.push(t.name),
                None => groups.push((key, vec![t.name], Vec::new())),
            }
        }
        for s in standing {
            let key = checkpoint(&s.model);
            match groups.iter_mut().find(|(k, _, _)| *k == key) {
                Some((_, _, asks)) => {
                    tracing::info!(
                        daemon = s.name,
                        model = %key.1,
                        "not run here; its asks of its model's engine still count"
                    );
                    asks.push((s.rank, s.asks));
                }
                None => tracing::info!(
                    daemon = s.name,
                    model = %key.1,
                    "not run here, and no tenant here runs its model: it asks nothing of an \
                     engine"
                ),
            }
        }
        let mut by_tenant = HashMap::new();
        for (_, names, asks) in groups {
            let slot = EngineSlot::with_standing(&names, asks);
            for name in names {
                by_tenant.insert(name, Arc::clone(&slot));
            }
        }
        Self { by_tenant }
    }

    /// The source tenant `name`, ranked `rank`, takes its engine from.
    ///
    /// Panics for a name [`Self::plan`] was not given: every tenant the host
    /// starts was planned.
    pub fn source(
        &self,
        rank: TenantRank,
        name: &'static str,
        requires: Capabilities,
    ) -> EngineSource {
        self.slot(name).source(rank, name, requires)
    }

    /// Tenant `name` will never arrive at its slot.
    pub fn withdraw(&self, name: &'static str) {
        self.slot(name).withdraw(name);
    }

    fn slot(&self, name: &'static str) -> &Arc<EngineSlot> {
        self.by_tenant
            .get(name)
            .unwrap_or_else(|| panic!("{name} was not planned a slot"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::guest::GuestRegistry;
    use candle_conversation::models::StrataTokens;

    fn seated(rank: usize, name: &'static str, model: Model) -> Seated {
        Seated {
            rank: TenantRank::nth(rank).unwrap(),
            name,
            model,
        }
    }

    fn asks() -> CoreRequests {
        CoreRequests {
            prefill_pass_tokens: None,
            qsa_selection_budget: None,
            qsa_strata: StrataTokens::DEFAULT,
            hit_layers: None,
            max_response_tokens: 1024,
            max_concurrent: 1,
            thinking: None,
            guests: GuestRegistry::new(),
        }
    }

    /// Two tenants naming one model share one slot, so one engine.
    #[test]
    fn one_model_is_one_slot() {
        let slots = Slots::plan(
            &[
                seated(0, "zend", Model::Qwen38_FlashNext),
                seated(1, "npcd", Model::Qwen38_FlashNext),
            ],
            Vec::new(),
        );
        assert!(Arc::ptr_eq(slots.slot("zend"), slots.slot("npcd")));
    }

    /// Two tenants on two models take two slots, so two engines.
    #[test]
    fn two_models_are_two_slots() {
        let slots = Slots::plan(
            &[
                seated(0, "zend", Model::Qwen38_FlashNext),
                seated(1, "npcd", Model::Qwen36_35B_A3B_AntiLoop_StyleTune),
            ],
            Vec::new(),
        );
        assert!(!Arc::ptr_eq(slots.slot("zend"), slots.slot("npcd")));
    }

    /// A slot expects only the tenants of its own model, so a tenant on
    /// another model never holds its build waiting.
    #[test]
    fn a_slot_expects_its_own_models_tenants() {
        let slots = Slots::plan(
            &[
                seated(0, "zend", Model::Qwen38_FlashNext),
                seated(1, "npcd", Model::Qwen36_35B_A3B_AntiLoop_StyleTune),
            ],
            vec![Standing {
                rank: TenantRank::nth(2).unwrap(),
                name: "left-out",
                model: Model::Qwen38_FlashNext,
                asks: asks(),
            }],
        );
        assert_eq!(
            format!("{:?}", slots.source(TenantRank::FIRST, "zend", caps())),
            r#"EngineSource { tenant: "zend", rank: TenantRank(0), expected: 1 }"#
        );
        assert_eq!(
            format!(
                "{:?}",
                slots.source(TenantRank::nth(1).unwrap(), "npcd", caps())
            ),
            r#"EngineSource { tenant: "npcd", rank: TenantRank(1), expected: 1 }"#
        );
    }

    /// A standing daemon whose model nobody here runs seats nowhere.
    #[test]
    fn a_standing_daemon_of_an_unrun_model_asks_nothing() {
        let slots = Slots::plan(
            &[seated(0, "zend", Model::Qwen38_FlashNext)],
            vec![Standing {
                rank: TenantRank::nth(1).unwrap(),
                name: "npcd",
                model: Model::Qwen36_35B_A3B_AntiLoop_StyleTune,
                asks: asks(),
            }],
        );
        assert_eq!(slots.by_tenant.len(), 1);
    }

    fn caps() -> Capabilities {
        Capabilities {
            tokens: &[],
            dialect: Model::Qwen38_FlashNext.spec().chat_format,
        }
    }
}
