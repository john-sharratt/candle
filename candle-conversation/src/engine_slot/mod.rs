//! One engine for every tenant in a process.
//!
//! A host running several tenants — zend and npcd in `tokerad` — loads one
//! model and runs one scheduler, and gives each tenant its own substrate on
//! them (`docs/shared_model_host.md` §3.2). Each tenant's loader reaches this
//! slot where it would otherwise build an engine of its own:
//!
//! 1. it deposits what it asks of the shared engine ([`CoreRequests`]);
//! 2. the highest-ranked tenant waits until every tenant has arrived or
//!    withdrawn, merges the requests and builds the engine on its own thread,
//!    with its own progress reporting and its own substrate;
//! 3. every other tenant then binds its own substrate and attaches it to that
//!    engine, on its own thread ([`crate::EngineAttacher`]).
//!
//! Builders never cross threads: each tenant's builder stays with the tenant
//! and only the settings travel. A slot of one tenant builds at once, so a
//! daemon run alone behaves exactly as it would with no slot at all.
//!
//! A tenant that fails before it arrives withdraws — dropping its unused
//! [`EngineTicket`] does it — so the others are not left waiting for it.

mod capabilities;
mod core_requests;
mod rendezvous;

pub use capabilities::Capabilities;
pub use core_requests::CoreRequests;

use std::cell::RefCell;
use std::sync::Arc;

use candle::Device;

use crate::engine::{ConversationEngine, EngineAttacher};
use crate::error::ConversationError;
use crate::models::ModelBuilder;
use crate::tenant_rank::TenantRank;
use rendezvous::Rendezvous;

/// Where every tenant of one process gets its engine.
pub struct EngineSlot {
    rendezvous: Rendezvous<EngineAttacher>,
}

impl EngineSlot {
    /// A slot for `expected` tenants.
    pub fn new(expected: usize) -> Arc<Self> {
        Arc::new(Self {
            rendezvous: Rendezvous::new(expected),
        })
    }

    /// The source a tenant ranked `rank` and named `name` takes its engine
    /// from, checked against `requires`.
    pub fn source(
        self: &Arc<Self>,
        rank: TenantRank,
        name: &'static str,
        requires: Capabilities,
    ) -> EngineSource {
        EngineSource {
            slot: Arc::clone(self),
            rank,
            name,
            requires,
        }
    }
}

/// One tenant's way to its engine. Cloned freely; each load takes an
/// [`EngineTicket`] from it.
#[derive(Clone)]
pub struct EngineSource {
    slot: Arc<EngineSlot>,
    rank: TenantRank,
    name: &'static str,
    requires: Capabilities,
}

impl std::fmt::Debug for EngineSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineSource")
            .field("tenant", &self.name)
            .field("rank", &self.rank)
            .field("expected", &self.slot.rendezvous.expected())
            .finish()
    }
}

impl EngineSource {
    /// A source of its own, for a tenant run alone — a daemon's own binary,
    /// or a test.
    pub fn solo(name: &'static str, requires: Capabilities) -> Self {
        EngineSlot::new(1).source(TenantRank::FIRST, name, requires)
    }

    /// The rank this tenant runs at.
    pub fn rank(&self) -> TenantRank {
        self.rank
    }

    /// The tenant's name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// What this tenant needs of the model.
    pub fn requires(&self) -> &Capabilities {
        &self.requires
    }

    /// A ticket for one load. Use it once, or drop it to withdraw.
    pub fn ticket(&self) -> EngineTicket {
        EngineTicket {
            source: self.clone(),
            used: false,
        }
    }
}

/// One tenant's claim on the slot for one load: spent by
/// [`Self::engine_with_progress`], or withdrawn when dropped unspent.
pub struct EngineTicket {
    source: EngineSource,
    used: bool,
}

impl EngineTicket {
    /// The engine for this tenant, through `builder`: built with it when this
    /// tenant ranks highest of those that arrive — the merged
    /// [`CoreRequests`] of every tenant applied — otherwise this tenant's own
    /// substrate, bound and attached to the engine that one built. Blocks until
    /// the engine exists. Refused, naming the tenant, when the model lacks a
    /// capability the tenant needs.
    pub fn engine_with_progress(
        mut self,
        builder: &mut ModelBuilder,
        device: &Device,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> crate::Result<ConversationEngine> {
        self.used = true;
        let source = self.source.clone();
        builder.set_tenant_rank(source.rank);
        let requests = builder.take_core_requests();
        // The two closures both need the builder, and only one of them runs.
        let builder_cell = RefCell::new(builder);
        let engine = source.slot.rendezvous.arrive(
            source.rank,
            source.name,
            requests,
            |merged| {
                let mut b = builder_cell.borrow_mut();
                b.apply_core_requests(merged);
                let engine = b.engine_with_progress(device, progress)?;
                let attacher = engine.attacher();
                Ok((engine, attacher))
            },
            |attacher| builder_cell.borrow_mut().attach_to(&attacher, device),
        )?;
        let builder = builder_cell.into_inner();
        source
            .requires
            .check(engine.tokenizer(), builder.spec().chat_format)
            .map_err(|missing| {
                ConversationError::Other(format!(
                    "{}: the model {} lacks what this tenant needs: {missing}",
                    source.name,
                    engine.model_key().describe()
                ))
            })?;
        Ok(engine)
    }
}

impl Drop for EngineTicket {
    fn drop(&mut self) {
        if self.used {
            return;
        }
        self.source.slot.rendezvous.withdraw();
        tracing::warn!(
            tenant = self.source.name,
            "withdrew from the shared engine before arriving; the others build without it"
        );
    }
}
