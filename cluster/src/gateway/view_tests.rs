//! The hub's views of its members that a request reads: how it is routed,
//! decided under one lock; a route table shared rather than copied; and
//! members' labels read under one lock.

use std::sync::Arc;

use super::hub::Hub;
use super::member_fixture::member;
use super::registry::MemberView;
use super::routing::{Routing, Standing};
use crate::dispatch::{Dispatch, Route, RouteTable};
use crate::frame::{Capacity, MemberState};
use crate::handshake::GroupKey;

fn table() -> RouteTable {
    RouteTable::new(vec![
        Route::new(&["GET"], "/v1/status", Dispatch::Any),
        Route::new(&["POST"], "/v1/chat/completions", Dispatch::Best),
    ])
    .unwrap()
}

/// A hub with node 1 of `zend` admitted, ready, and declaring [`table`].
fn hub() -> Arc<Hub> {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    {
        let mut s = hub.state();
        s.registry.join(ready(member(1, 7)));
        s.registry.routes("zend", 1, table());
    }
    hub
}

/// **A request's table is the member's, shared**: two lookups hand back the
/// same allocation, and it reads as the table the member declared.
#[test]
fn a_table_is_shared_not_copied() {
    let hub = hub();
    let a = hub
        .table_for("zend", "POST", "/v1/chat/completions")
        .unwrap();
    let b = hub.table_for("zend", "GET", "/v1/status").unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(*a, table());
    assert!(Arc::ptr_eq(&a, &hub.table("zend").unwrap()));
}

fn ready(mut m: MemberView) -> MemberView {
    m.capacity = Some(Capacity {
        state: MemberState::Ready,
        ..Capacity::default()
    });
    m
}

/// **Routing is decided in one look**: an app with no member admitted, one
/// whose members are all loading (the lowest session stands by), one whose
/// ready member has declared no table yet, and a ready app's shared table.
#[test]
fn routing_names_each_case() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    assert_eq!(
        hub.routing("zend", "GET", "/v1/status"),
        Routing::Unadmitted
    );
    assert_eq!(hub.standing("zend"), Standing::Unadmitted);

    hub.state().registry.join(member(2, 9));
    hub.state().registry.join(member(3, 4));
    assert_eq!(
        hub.routing("zend", "GET", "/v1/status"),
        Routing::StandingBy(Some(3))
    );
    assert_eq!(hub.standing("zend"), Standing::By(Some(3)));

    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    hub.state().registry.join(ready(member(1, 7)));
    assert_eq!(
        hub.routing("zend", "GET", "/v1/status"),
        Routing::StandingBy(None)
    );

    let hub = self::hub();
    let Routing::Table(table) = hub.routing("zend", "GET", "/v1/status") else {
        panic!("a ready app is routed by its table");
    };
    assert!(Arc::ptr_eq(&table, &hub.table("zend").unwrap()));
}

/// **Labels come back in the nodes' order**, `("", 0)` for one not
/// admitted; a member's presence and host are read without copying it.
#[test]
fn member_labels_answer_each_node_in_order() {
    let hub = hub();
    assert_eq!(
        hub.member_labels("zend", &[9, 1]),
        [(String::new(), 0), ("m1".to_string(), 10)]
    );
    assert!(hub.has_member("zend", 1));
    assert!(!hub.has_member("zend", 9));
    assert_eq!(hub.member_host("zend", 1).as_deref(), Some("m1"));
    assert_eq!(hub.member_host("zend", 9), None);
}
