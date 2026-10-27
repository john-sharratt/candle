//! The content check behind an authorization (`docs/guest_draw_orchestration.md`
//! §1): one seam, [`ContentCheck`], whose implementation is today's four
//! questions on the resident model ([`crate::compliance`]), and a fake in
//! tests.

use std::future::Future;
use std::sync::Arc;

use super::cache::Cache;
use super::token::{Authorization, Signer};
use crate::api::{now_ms, Authored};
use crate::compliance::{self, Denial};

/// Whether a prompt may be drawn.
pub trait ContentCheck {
    fn check(&self, prompt: &str) -> impl Future<Output = Result<(), Denial>> + Send;
}

/// The check on this member's own resident model. Made per call, never kept
/// on the state it reads.
pub struct Compliance(pub Arc<Authored>);

impl ContentCheck for Compliance {
    fn check(&self, prompt: &str) -> impl Future<Output = Result<(), Denial>> + Send {
        compliance::check(&self.0, prompt)
    }
}

/// `prompt` authorized for `purpose` on this member: the one in `cache` while
/// it counts, otherwise checked here, signed, and kept.
pub async fn local<C: ContentCheck>(
    check: &C,
    signer: &Signer,
    cache: &Cache,
    purpose: &str,
    prompt: &str,
) -> Result<Authorization, Denial> {
    if let Some(a) = cache.get(purpose, prompt, now_ms()) {
        return Ok(a);
    }
    check.check(prompt).await?;
    let a = signer.sign(purpose, prompt, now_ms());
    cache.put(a.clone(), now_ms());
    Ok(a)
}
