//! The flows one forwarded port carries (`docs/battlecities_app.md` §5):
//! each client's queue to the task that carries its datagrams, and how many
//! there are, in all and from each address.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};

use tokio::sync::mpsc::Sender;

/// The most flows one port carries at once.
pub const MAX_FLOWS: usize = 8_192;

/// The most flows from one client address.
pub const PER_ADDRESS: usize = 64;

/// A flow's place in the table: its number, and where its datagrams are
/// queued.
struct Entry {
    id: u64,
    queue: Sender<Vec<u8>>,
}

#[derive(Default)]
pub struct Table {
    flows: HashMap<SocketAddr, Entry>,
    per_address: HashMap<IpAddr, usize>,
    next: u64,
    /// Datagrams of new flows dropped at a limit.
    pub refused: u64,
}

impl Table {
    /// Where `client`'s datagrams are queued, when it has a flow.
    pub fn queue(&self, client: SocketAddr) -> Option<Sender<Vec<u8>>> {
        self.flows.get(&client).map(|e| e.queue.clone())
    }

    /// Whether a new flow from `client` fits both limits.
    pub fn admits(&self, client: SocketAddr) -> bool {
        let from_address = self.per_address.get(&client.ip()).copied().unwrap_or(0);
        self.flows.len() < MAX_FLOWS && from_address < PER_ADDRESS
    }

    /// Add `client`'s flow, queued on `queue`: its number.
    pub fn add(&mut self, client: SocketAddr, queue: Sender<Vec<u8>>) -> u64 {
        self.next += 1;
        let id = self.next;
        if self.flows.insert(client, Entry { id, queue }).is_none() {
            *self.per_address.entry(client.ip()).or_insert(0) += 1;
        }
        id
    }

    /// Remove `client`'s flow numbered `id`. A flow that ended after the
    /// client's next one began removes nothing.
    pub fn remove(&mut self, client: SocketAddr, id: u64) {
        if self.flows.get(&client).is_some_and(|e| e.id == id) {
            self.flows.remove(&client);
            let ip = client.ip();
            match self.per_address.get_mut(&ip) {
                Some(n) if *n > 1 => *n -= 1,
                _ => {
                    self.per_address.remove(&ip);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.flows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.flows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use tokio::sync::mpsc::channel;

    use super::*;

    fn client(a: u8, b: u8, port: u16) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::new(203, 0, a, b), port))
    }

    fn queue() -> Sender<Vec<u8>> {
        channel(1).0
    }

    #[test]
    fn a_flow_is_found_by_its_client_and_removed_by_its_number() {
        let mut t = Table::default();
        let c = client(0, 9, 40_000);
        assert!(t.queue(c).is_none());
        let first = t.add(c, queue());
        assert!(t.queue(c).is_some());
        assert_eq!(t.len(), 1);
        // The client's next flow takes the place; the first one ending late
        // does not remove it.
        let second = t.add(c, queue());
        assert_ne!(first, second);
        t.remove(c, first);
        assert!(t.queue(c).is_some());
        t.remove(c, second);
        assert!(t.queue(c).is_none());
        assert!(t.is_empty());
    }

    /// **One address opens at most 64 flows**, and gets its room back as they
    /// end.
    #[test]
    fn one_address_is_held_to_its_share() {
        let mut t = Table::default();
        let mut ids = Vec::new();
        for port in 0..PER_ADDRESS as u16 {
            assert!(t.admits(client(0, 9, 1_000 + port)));
            ids.push(t.add(client(0, 9, 1_000 + port), queue()));
        }
        assert!(!t.admits(client(0, 9, 9_999)));
        assert!(t.admits(client(0, 10, 9_999)));
        t.remove(client(0, 9, 1_000), ids[0]);
        assert!(t.admits(client(0, 9, 9_999)));
    }

    /// **A port carries at most 8,192 flows.**
    #[test]
    fn a_port_is_held_to_its_flows() {
        let mut t = Table::default();
        for n in 0..MAX_FLOWS {
            let c = client((n >> 8) as u8, n as u8, 5_000);
            assert!(t.admits(c), "{n}");
            t.add(c, queue());
        }
        assert_eq!(t.len(), MAX_FLOWS);
        assert!(!t.admits(client(200, 1, 5_000)));
    }
}
