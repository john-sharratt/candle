//! The bulbs' wire (§9.3): one UDP socket, never blocking. A send the socket
//! cannot take now is dropped and said so, for the gate to send again; an
//! answer is read when there is one. Nothing waits on a bulb.

use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};

use effector_bridge::proto::wiz::PORT;

/// What the lights thread sends through, and reads answers from.
pub trait Wire {
    /// Send `bytes` to the bulb at `to`; whether it went.
    fn send(&mut self, to: Ipv4Addr, bytes: &[u8]) -> bool;
    /// An answer, if one has come.
    fn recv(&mut self) -> Option<(Ipv4Addr, Vec<u8>)>;
}

/// The real wire.
pub struct Udp {
    socket: UdpSocket,
    buf: Vec<u8>,
}

impl Udp {
    pub fn open() -> io::Result<Udp> {
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
        socket.set_nonblocking(true)?;
        Ok(Udp {
            socket,
            buf: vec![0u8; 2048],
        })
    }
}

impl Wire for Udp {
    fn send(&mut self, to: Ipv4Addr, bytes: &[u8]) -> bool {
        self.socket
            .send_to(bytes, SocketAddrV4::new(to, PORT))
            .is_ok()
    }

    fn recv(&mut self) -> Option<(Ipv4Addr, Vec<u8>)> {
        match self.socket.recv_from(&mut self.buf) {
            Ok((n, SocketAddr::V4(from))) => Some((*from.ip(), self.buf[..n].to_vec())),
            _ => None,
        }
    }
}

/// A wire that keeps what was sent and hands over what it is given, for the
/// thread's tests.
#[cfg(test)]
#[derive(Default)]
pub struct Recorded {
    pub sent: Vec<(Ipv4Addr, String)>,
    pub answers: std::collections::VecDeque<(Ipv4Addr, Vec<u8>)>,
}

#[cfg(test)]
impl Wire for Recorded {
    fn send(&mut self, to: Ipv4Addr, bytes: &[u8]) -> bool {
        self.sent
            .push((to, String::from_utf8_lossy(bytes).into_owned()));
        true
    }

    fn recv(&mut self) -> Option<(Ipv4Addr, Vec<u8>)> {
        self.answers.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The real wire never blocks**: nothing has come, so nothing is read.
    #[test]
    fn the_wire_never_blocks() {
        let mut w = Udp::open().unwrap();
        assert!(w.recv().is_none());
        assert!(w.send(Ipv4Addr::LOCALHOST, b"{}"));
    }
}
