//! Where an order's HTTP-01 answers are published: on this gateway alone, or
//! on it and on every peer, each of which acknowledges before the CA is asked
//! to validate — so wherever the CA's request lands, that gateway has the
//! answer (§13.4).

use std::future::Future;

use super::challenges::Challenges;

pub trait Publish: Sync {
    /// Answer `token` with `answer`; returns once every gateway the CA's
    /// request may reach answers it.
    fn publish(&self, token: &str, answer: &str) -> impl Future<Output = ()> + Send;
    /// Stop answering `token`, everywhere it was published.
    fn withdraw(&self, token: &str);
}

/// This gateway alone.
impl Publish for Challenges {
    async fn publish(&self, token: &str, answer: &str) {
        self.set(token, answer);
    }

    fn withdraw(&self, token: &str) {
        self.clear(token);
    }
}
