//! One forwarded port (`docs/battlecities_app.md` §5): the socket clients
//! send to, and a task for each client that carries its datagrams to one
//! member and the member's back.
//!
//! A flow's task opens a socket of its own towards the member, tells the
//! member which client stands behind that socket's address, and only then
//! sends the client's datagrams on. It says so again every
//! [`ANNOUNCE_EVERY`], ends when the flow carries nothing for [`IDLE`] or its
//! member has left, and takes its place out of the table as it goes.
//! Datagrams of a flow that does not fit the table's limits, or that arrive
//! faster than its task sends them on, are dropped: the game's own protocol
//! sends again what it must.

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use tokio::net::UdpSocket;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::time::{interval_at, sleep_until, Instant, MissedTickBehavior};

use super::pick::pick;
use super::table::Table;

/// How long a flow may carry nothing before it is closed.
pub const IDLE: Duration = Duration::from_secs(60);

/// How often a flow's member is told again whose it is.
pub const ANNOUNCE_EVERY: Duration = Duration::from_secs(30);

/// The largest datagram carried.
pub const DATAGRAM: usize = 2_048;

/// How many of a client's datagrams wait for its flow's task.
const QUEUED: usize = 256;

/// What a member is told, or why it could not be.
pub type Told<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>>;

/// The members a port's flows go to.
pub trait Members: Send + Sync + 'static {
    /// The members that serve now, in node order, each with the address its
    /// machine is reached at.
    fn serving(&self) -> Vec<(u64, IpAddr)>;

    /// Whether `node` is still a member.
    fn has(&self, node: u64) -> bool;

    /// Tell `node` that datagrams from `via` are `client`'s.
    fn announce(&self, node: u64, via: SocketAddr, client: IpAddr) -> Told<'_>;
}

/// One forwarded port: where clients send, the port its members listen on,
/// and its flows.
pub struct Port {
    pub listen: Arc<UdpSocket>,
    pub member_port: u16,
    pub members: Arc<dyn Members>,
    pub table: Arc<Mutex<Table>>,
}

fn held(table: &Mutex<Table>) -> MutexGuard<'_, Table> {
    table.lock().unwrap_or_else(|e| e.into_inner())
}

impl Port {
    pub fn new(listen: UdpSocket, member_port: u16, members: Arc<dyn Members>) -> Arc<Port> {
        Arc::new(Port {
            listen: Arc::new(listen),
            member_port,
            members,
            table: Arc::default(),
        })
    }

    /// Carry the port's datagrams until its socket is gone.
    pub async fn run(self: Arc<Self>) {
        let mut buf = vec![0u8; DATAGRAM];
        loop {
            // An error here is one datagram's: on Windows a send to a client
            // that has gone comes back as a reset on the next receive.
            let Ok((n, client)) = self.listen.recv_from(&mut buf).await else {
                continue;
            };
            let queue = held(&self.table).queue(client);
            let queue = match queue.or_else(|| self.open(client)) {
                Some(queue) => queue,
                None => continue,
            };
            // Full, or the flow has just ended: dropped.
            let _ = queue.try_send(buf[..n].to_vec());
        }
    }

    /// Start `client`'s flow: `None` when no member serves, or the table has
    /// no room for it.
    fn open(self: &Arc<Self>, client: SocketAddr) -> Option<Sender<Vec<u8>>> {
        let (node, at) = pick(&self.members.serving(), client)?;
        let (queue, datagrams) = channel(QUEUED);
        let id = {
            let mut table = held(&self.table);
            if !table.admits(client) {
                table.refused += 1;
                return None;
            }
            table.add(client, queue.clone())
        };
        let flow = Flow {
            port: Arc::clone(self),
            client,
            node,
            member: SocketAddr::new(at, self.member_port),
            id,
        };
        tokio::spawn(flow.carry(datagrams));
        Some(queue)
    }
}

/// One client's flow to one member.
struct Flow {
    port: Arc<Port>,
    client: SocketAddr,
    node: u64,
    member: SocketAddr,
    id: u64,
}

impl Flow {
    async fn carry(self, mut datagrams: Receiver<Vec<u8>>) {
        if let Err(why) = self.run(&mut datagrams).await {
            tracing::warn!(
                client = %self.client,
                member = %self.member,
                "udp: a flow ended: {why}"
            );
        }
        held(&self.port.table).remove(self.client, self.id);
    }

    async fn run(&self, datagrams: &mut Receiver<Vec<u8>>) -> Result<(), String> {
        let any: SocketAddr = match self.member {
            SocketAddr::V4(_) => (Ipv4Addr::UNSPECIFIED, 0).into(),
            SocketAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
        };
        let upstream = UdpSocket::bind(any)
            .await
            .map_err(|e| format!("no socket for it: {e}"))?;
        upstream
            .connect(self.member)
            .await
            .map_err(|e| format!("the member is not reached: {e}"))?;
        let via = upstream
            .local_addr()
            .map_err(|e| format!("its socket has no address: {e}"))?;
        // Before the first datagram, so the member never counts the client
        // as the gateway.
        let members = &self.port.members;
        members.announce(self.node, via, self.client.ip()).await?;
        let mut again = interval_at(Instant::now() + ANNOUNCE_EVERY, ANNOUNCE_EVERY);
        again.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut last = Instant::now();
        let mut buf = vec![0u8; DATAGRAM];
        loop {
            tokio::select! {
                datagram = datagrams.recv() => {
                    let Some(datagram) = datagram else {
                        return Ok(());
                    };
                    last = Instant::now();
                    let _ = upstream.send(&datagram).await;
                }
                answered = upstream.recv(&mut buf) => {
                    if let Ok(n) = answered {
                        last = Instant::now();
                        let _ = self.port.listen.send_to(&buf[..n], self.client).await;
                    }
                }
                _ = again.tick() => {
                    if !members.has(self.node) {
                        return Err("its member left".to_string());
                    }
                    members.announce(self.node, via, self.client.ip()).await?;
                }
                _ = sleep_until(last + IDLE) => return Ok(()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::time::timeout;

    use super::*;

    /// One member, an echo on loopback, that records what it is told.
    struct OneMember {
        at: SocketAddr,
        here: bool,
        told: Mutex<Vec<(u64, SocketAddr, IpAddr)>>,
    }

    impl Members for OneMember {
        fn serving(&self) -> Vec<(u64, IpAddr)> {
            match self.here {
                true => vec![(7, self.at.ip())],
                false => Vec::new(),
            }
        }

        fn has(&self, node: u64) -> bool {
            self.here && node == 7
        }

        fn announce(&self, node: u64, via: SocketAddr, client: IpAddr) -> Told<'_> {
            self.told.lock().unwrap().push((node, via, client));
            Box::pin(async { Ok(()) })
        }
    }

    async fn loopback() -> UdpSocket {
        UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap()
    }

    /// A member that answers each datagram with `re:` and what it got, and
    /// the forwarded port in front of it.
    async fn forwarded(here: bool) -> (SocketAddr, Arc<OneMember>, Arc<Port>) {
        let member = loopback().await;
        let at = member.local_addr().unwrap();
        tokio::spawn(async move {
            let mut buf = [0u8; DATAGRAM];
            while let Ok((n, from)) = member.recv_from(&mut buf).await {
                let mut answer = b"re:".to_vec();
                answer.extend_from_slice(&buf[..n]);
                let _ = member.send_to(&answer, from).await;
            }
        });
        let members = Arc::new(OneMember {
            at,
            here,
            told: Mutex::default(),
        });
        let listen = loopback().await;
        let front = listen.local_addr().unwrap();
        let port = Port::new(listen, at.port(), members.clone());
        tokio::spawn(Arc::clone(&port).run());
        (front, members, port)
    }

    async fn heard(socket: &UdpSocket) -> Option<Vec<u8>> {
        let mut buf = [0u8; DATAGRAM];
        let got = timeout(Duration::from_secs(5), socket.recv_from(&mut buf)).await;
        got.ok()?.ok().map(|(n, _)| buf[..n].to_vec())
    }

    /// **A client's datagrams reach the member and the member's come back**,
    /// from the address the client sent to, and the member was told who
    /// stands behind the flow's socket before the first one arrived.
    #[tokio::test]
    async fn a_clients_datagrams_are_carried_both_ways_and_announced_first() {
        let (front, members, port) = forwarded(true).await;
        let client = loopback().await;
        client.send_to(b"hello", front).await.unwrap();
        assert_eq!(heard(&client).await.as_deref(), Some(&b"re:hello"[..]));
        client.send_to(b"again", front).await.unwrap();
        assert_eq!(heard(&client).await.as_deref(), Some(&b"re:again"[..]));

        let told = members.told.lock().unwrap().clone();
        assert_eq!(told.len(), 1, "one flow, told once so far");
        let (node, via, of) = told[0];
        assert_eq!(node, 7);
        assert_eq!(of, client.local_addr().unwrap().ip());
        assert_ne!(via, client.local_addr().unwrap());
        assert_ne!(via.port(), 0);
        assert_eq!(held(&port.table).len(), 1);

        // A second client is a second flow, on a socket of its own.
        let other = loopback().await;
        other.send_to(b"hi", front).await.unwrap();
        assert_eq!(heard(&other).await.as_deref(), Some(&b"re:hi"[..]));
        let told = members.told.lock().unwrap().clone();
        assert_eq!(told.len(), 2);
        assert_ne!(told[0].1, told[1].1);
        assert_eq!(held(&port.table).len(), 2);
    }

    /// With no member serving, a datagram goes nowhere and no flow is kept.
    #[tokio::test]
    async fn with_no_member_serving_a_datagram_is_dropped() {
        let (front, members, port) = forwarded(false).await;
        let client = loopback().await;
        client.send_to(b"hello", front).await.unwrap();
        let mut buf = [0u8; 16];
        let got = timeout(Duration::from_millis(300), client.recv_from(&mut buf)).await;
        assert!(got.is_err(), "nothing answers");
        assert!(members.told.lock().unwrap().is_empty());
        assert!(held(&port.table).is_empty());
    }
}
