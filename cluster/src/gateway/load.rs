//! A request a member is answering for this gateway, held from the pick until
//! the answer is finished (§5.3). The count it keeps is what `Best` divides a
//! member's power by, so a burst of new conversations spreads over the members
//! as it arrives instead of all reading the same `CAPACITY`.

use std::sync::Arc;

use super::hub::Hub;
use super::registry::Pick;

/// One request in flight to a member; dropping it ends the request.
pub struct Load {
    hub: Arc<Hub>,
    app: String,
    node: u64,
    session: u64,
    /// A draw, which the member's draw count holds too.
    draw: bool,
}

impl Load {
    pub(super) fn new(hub: Arc<Hub>, app: &str, node: u64, session: u64, draw: bool) -> Self {
        Self {
            hub,
            app: app.to_string(),
            node,
            session,
            draw,
        }
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.hub
            .state()
            .registry
            .end(&self.app, self.node, self.session, self.draw);
    }
}

/// Where a request goes, and the load it puts there while it is answered.
pub struct Picked {
    pub pick: Pick,
    /// Held for a request that is real work for the member it reaches — a
    /// `Best`, sticky or owner route — and `None` for the rest.
    pub load: Option<Load>,
}
