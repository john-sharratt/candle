//! A `zend` member as the gateway's tests admit it: node `node` on host
//! `m<node>`, its own card, build 10, on session `session`.

use super::registry::MemberView;
use crate::frame::BuildId;

pub fn member(node: u64, session: u64) -> MemberView {
    MemberView {
        node,
        host: format!("m{node}"),
        card: node,
        app: "zend".into(),
        build: BuildId {
            count: 10,
            hash: "h10".into(),
            local: String::new(),
        },
        weight: 100,
        session,
        repos: Vec::new(),
        capacity: None,
        routes: None,
        high_water: 0,
        applied: Vec::new(),
        lagging: false,
        syncing: false,
        silent: false,
        in_flight: 0,
        drawing: 0,
        handed: 0,
        closing: false,
    }
}
