//! A member's dispatch table (§5.2): each route's methods, path pattern,
//! dispatch, mirror mode and reach, as `ROUTES` carries it and the gateway
//! matches it.
//!
//! Path patterns are axum's — `:param` for one segment, `*rest` for the rest —
//! and are matched with `matchit`, the matcher axum itself uses. One path may
//! be registered several times with different methods (a read open to all, a
//! write for admins), each with its own dispatch; a route with no methods
//! answers every method.
//!
//! On the wire: `list<route>`, a route being `list<str> methods · str path ·
//! dispatch · mirror · reach`; a dispatch `u8` tag — 0 `Best`, 1 `Sticky
//! list<key>`, 2 `Owner key`, 3 `Aggregate u8 merge` (0 envelope, 1 first
//! found, 2 lines), 4 `Any`, 5 `Ui`; a key `u8` tag — 0 path, 1 header, 2 body,
//! 3 derived — then `str` its name; a mirror `u8` — 0 none, 1 shape, 2 `Exact
//! list<str>`; a reach `u8` — 0 any, 1 draw (protocol 6).

use std::collections::HashMap;
use std::fmt;

use matchit::Router as Matcher;

use super::{Derived, Dispatch, KeySource, Merge, Mirror, Reach};
use crate::wire::{Reader, WireError, Writer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    /// Upper-case method names; empty for every method.
    pub methods: Vec<String>,
    pub path: String,
    pub dispatch: Dispatch,
    pub mirror: Mirror,
    pub reach: Reach,
}

impl Route {
    pub fn new(methods: &[&str], path: &str, dispatch: Dispatch) -> Route {
        Route {
            methods: methods.iter().map(|m| m.to_string()).collect(),
            path: path.to_string(),
            dispatch,
            mirror: Mirror::None,
            reach: Reach::Any,
        }
    }

    /// Whether the gateway must buffer the request body to route it: to read
    /// its key, or to size its draw.
    pub fn reads_body(&self) -> bool {
        self.dispatch.reads_body() || self.reach != Reach::Any
    }

    fn answers(&self, method: &str) -> bool {
        self.methods.is_empty() || self.methods.iter().any(|m| m == method)
    }
}

/// The table, with its matcher built.
#[derive(Clone)]
pub struct RouteTable {
    routes: Vec<Route>,
    matcher: Matcher<Vec<usize>>,
}

impl fmt::Debug for RouteTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RouteTable")
            .field("routes", &self.routes)
            .finish()
    }
}

impl PartialEq for RouteTable {
    fn eq(&self, other: &Self) -> bool {
        self.routes == other.routes
    }
}

/// A route matched for a request, and the path parameters it bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matched<'a> {
    pub route: &'a Route,
    pub params: Vec<(String, String)>,
}

impl RouteTable {
    pub fn new(routes: Vec<Route>) -> Result<RouteTable, String> {
        let mut by_path: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut order: Vec<&str> = Vec::new();
        for (i, r) in routes.iter().enumerate() {
            let entry = by_path.entry(r.path.as_str()).or_insert_with(|| {
                order.push(r.path.as_str());
                Vec::new()
            });
            entry.push(i);
        }
        let mut matcher = Matcher::new();
        for path in order {
            matcher
                .insert(path, by_path[path].clone())
                .map_err(|e| format!("route {path}: {e}"))?;
        }
        Ok(RouteTable { routes, matcher })
    }

    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// The route answering `method` on `path`, if any.
    pub fn lookup(&self, method: &str, path: &str) -> Option<Matched<'_>> {
        let m = self.matcher.at(path).ok()?;
        let route = m
            .value
            .iter()
            .map(|i| &self.routes[*i])
            .find(|r| r.answers(method))?;
        Some(Matched {
            route,
            params: m
                .params
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        })
    }

    /// The dispatch for `method` on `path`: a path no route declares is a UI
    /// file.
    pub fn dispatch(&self, method: &str, path: &str) -> Dispatch {
        self.lookup(method, path)
            .map_or(Dispatch::Ui, |m| m.route.dispatch.clone())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.list(&self.routes, |w, r| {
            w.list(&r.methods, |w, m| {
                w.str(m);
            })
            .str(&r.path);
            write_dispatch(w, &r.dispatch);
            write_mirror(w, &r.mirror);
            write_reach(w, r.reach);
        });
        w.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<RouteTable, WireError> {
        let mut r = Reader::new(bytes, "ROUTES table");
        let routes = r.list(|r| {
            Ok(Route {
                methods: r.list(|r| r.str())?,
                path: r.str()?,
                dispatch: read_dispatch(r)?,
                mirror: read_mirror(r)?,
                reach: read_reach(r)?,
            })
        })?;
        r.finish()?;
        RouteTable::new(routes).map_err(|_| WireError::Invalid {
            what: "ROUTES table",
            value: 0,
        })
    }
}

fn write_key(w: &mut Writer, k: &KeySource) {
    match k {
        KeySource::Path(n) => w.u8(0).str(n),
        KeySource::Header(n) => w.u8(1).str(n),
        KeySource::Body(n) => w.u8(2).str(n),
        KeySource::Derived(d) => w.u8(3).str(d.name()),
    };
}

fn read_key(r: &mut Reader) -> Result<KeySource, WireError> {
    let tag = r.u8()?;
    let name = r.str()?;
    Ok(match tag {
        0 => KeySource::Path(name),
        1 => KeySource::Header(name),
        2 => KeySource::Body(name),
        3 => KeySource::Derived(Derived::from_name(&name).ok_or(WireError::Invalid {
            what: r.what(),
            value: 3,
        })?),
        t => {
            return Err(WireError::Invalid {
                what: r.what(),
                value: t as u64,
            })
        }
    })
}

fn write_dispatch(w: &mut Writer, d: &Dispatch) {
    match d {
        Dispatch::Best => {
            w.u8(0);
        }
        Dispatch::Sticky(keys) => {
            w.u8(1).list(keys, write_key);
        }
        Dispatch::Owner(key) => {
            w.u8(2);
            write_key(w, key);
        }
        Dispatch::Aggregate(merge) => {
            let m = match merge {
                Merge::Envelope => 0,
                Merge::FirstFound => 1,
                Merge::Lines => 2,
                Merge::Union => 3,
            };
            w.u8(3).u8(m);
        }
        Dispatch::Any => {
            w.u8(4);
        }
        Dispatch::Ui => {
            w.u8(5);
        }
    }
}

fn read_dispatch(r: &mut Reader) -> Result<Dispatch, WireError> {
    let invalid = |r: &Reader, v: u8| WireError::Invalid {
        what: r.what(),
        value: v as u64,
    };
    Ok(match r.u8()? {
        0 => Dispatch::Best,
        1 => Dispatch::Sticky(r.list(read_key)?),
        2 => Dispatch::Owner(read_key(r)?),
        3 => Dispatch::Aggregate(match r.u8()? {
            0 => Merge::Envelope,
            1 => Merge::FirstFound,
            2 => Merge::Lines,
            3 => Merge::Union,
            v => return Err(invalid(r, v)),
        }),
        4 => Dispatch::Any,
        5 => Dispatch::Ui,
        v => return Err(invalid(r, v)),
    })
}

fn write_mirror(w: &mut Writer, m: &Mirror) {
    match m {
        Mirror::None => {
            w.u8(0);
        }
        Mirror::Shape => {
            w.u8(1);
        }
        Mirror::Exact(volatile) => {
            w.u8(2).list(volatile, |w, f| {
                w.str(f);
            });
        }
    }
}

fn read_mirror(r: &mut Reader) -> Result<Mirror, WireError> {
    Ok(match r.u8()? {
        0 => Mirror::None,
        1 => Mirror::Shape,
        2 => Mirror::Exact(r.list(|r| r.str())?),
        v => {
            return Err(WireError::Invalid {
                what: r.what(),
                value: v as u64,
            })
        }
    })
}

fn write_reach(w: &mut Writer, reach: Reach) {
    w.u8(match reach {
        Reach::Any => 0,
        Reach::Draw => 1,
    });
}

fn read_reach(r: &mut Reader) -> Result<Reach, WireError> {
    Ok(match r.u8()? {
        0 => Reach::Any,
        1 => Reach::Draw,
        v => {
            return Err(WireError::Invalid {
                what: r.what(),
                value: v as u64,
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zend_like() -> RouteTable {
        RouteTable::new(vec![
            Route::new(&[], "/v1/status", Dispatch::Aggregate(Merge::Envelope)),
            Route::new(
                &["POST"],
                "/v1/chat/completions",
                Dispatch::Sticky(vec![
                    KeySource::Header("x-tokera-conversation".into()),
                    KeySource::Body("conv_id".into()),
                    KeySource::Derived(Derived::Passthrough),
                ]),
            ),
            Route::new(&[], "/v1/conversations/:id", Dispatch::sticky_path("id")),
            Route::new(&["GET"], "/v1/world/:wid", Dispatch::Any),
            Route::new(&["PUT", "DELETE"], "/v1/world/:wid", Dispatch::Best),
            Route::new(&[], "/v1/local/*rest", Dispatch::Best),
        ])
        .unwrap()
    }

    #[test]
    fn a_request_finds_its_route_and_parameters() {
        let t = zend_like();
        let m = t.lookup("GET", "/v1/conversations/c-42").unwrap();
        assert_eq!(m.route.dispatch, Dispatch::sticky_path("id"));
        assert_eq!(m.params, vec![("id".to_string(), "c-42".to_string())]);
        assert_eq!(t.dispatch("GET", "/v1/world/w1"), Dispatch::Any);
        assert_eq!(t.dispatch("PUT", "/v1/world/w1"), Dispatch::Best);
        assert_eq!(t.dispatch("GET", "/v1/local/tick/now"), Dispatch::Best);
        // A path no route declares is a UI file.
        assert_eq!(t.dispatch("GET", "/index.html"), Dispatch::Ui);
        assert_eq!(t.dispatch("GET", "/v1/chat/completions"), Dispatch::Ui);
    }

    #[test]
    fn the_table_round_trips_and_encodes_to_exact_bytes() {
        let t = RouteTable::new(vec![
            Route::new(&["GET"], "/v1/x/:id", Dispatch::sticky_path("id")),
            Route {
                methods: vec![],
                path: "/v1/y".into(),
                dispatch: Dispatch::Aggregate(Merge::FirstFound),
                mirror: Mirror::Exact(vec!["t".into()]),
                reach: Reach::Draw,
            },
        ])
        .unwrap();
        let mut want = vec![2, 0, 0, 0];
        want.extend([1, 0, 0, 0, 3, 0, 0, 0]);
        want.extend(b"GET");
        want.extend([9, 0, 0, 0]);
        want.extend(b"/v1/x/:id");
        want.extend([1, 1, 0, 0, 0, 0, 2, 0, 0, 0]);
        want.extend(b"id");
        want.extend([0, 0]);
        want.extend([0, 0, 0, 0, 5, 0, 0, 0]);
        want.extend(b"/v1/y");
        want.extend([3, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0]);
        want.extend(b"t");
        want.push(1);
        assert_eq!(t.encode(), want);
        assert_eq!(RouteTable::decode(&want).unwrap(), t);
        assert_eq!(
            RouteTable::decode(&zend_like().encode()).unwrap(),
            zend_like()
        );
    }

    #[test]
    fn a_malformed_table_is_refused() {
        let mut bad = zend_like().encode();
        bad.push(0);
        assert!(RouteTable::decode(&bad).is_err());
        // Two routes whose patterns conflict cannot be matched.
        assert!(RouteTable::new(vec![
            Route::new(&[], "/v1/npc/:id", Dispatch::Any),
            Route::new(&[], "/v1/npc/:nid", Dispatch::Any),
        ])
        .is_err());
    }
}
