//! The Android TV Remote v2 protocol — how a Google TV is driven over the LAN,
//! the way its own remote does, with no cloud.
//!
//! Two TLS services share one generated RSA client certificate: pairing on port
//! 6467 and the remote on 6466. Pairing shows a six-hex code on the TV; the
//! secret sent back is `SHA-256` of both certificates' RSA modulus and exponent
//! and the code, which is what proves the two sides saw the same screen. Once
//! paired the certificate is the lasting credential. Messages are protobuf,
//! each framed by a varint of its length (`polo.proto` for pairing,
//! `remotemessage.proto` for the remote); both are hand-encoded here.

use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use rsa::pkcs8::{EncodePrivateKey, LineEnding};
use rsa::traits::PublicKeyParts;
use rsa::RsaPrivateKey;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_native_tls::native_tls::{Identity, TlsConnector as NativeConnector};
use tokio_native_tls::{TlsConnector, TlsStream};
use x509_parser::prelude::{FromDer, X509Certificate};
use x509_parser::public_key::PublicKey;

pub const PAIR_PORT: u16 = 6467;
pub const REMOTE_PORT: u16 = 6466;
const ANSWER: Duration = Duration::from_secs(8);

/// Remote key codes (`RemoteKeyCode`), the ones the world drives.
pub mod key {
    pub const HOME: i32 = 3;
    pub const BACK: i32 = 4;
    pub const UP: i32 = 19;
    pub const DOWN: i32 = 20;
    pub const LEFT: i32 = 21;
    pub const RIGHT: i32 = 22;
    pub const CENTER: i32 = 23;
    pub const VOLUME_UP: i32 = 24;
    pub const VOLUME_DOWN: i32 = 25;
    pub const POWER: i32 = 26;
    pub const MENU: i32 = 82;
    pub const PLAY_PAUSE: i32 = 85;
    pub const STOP: i32 = 86;
    pub const NEXT: i32 = 87;
    pub const PREVIOUS: i32 = 88;
    pub const REWIND: i32 = 89;
    pub const FAST_FORWARD: i32 = 90;
    pub const MUTE: i32 = 91;
    pub const INFO: i32 = 165;
}

/// A minimal protobuf writer and reader — enough for these two schemas.
mod pb {
    /// Append a base-128 varint.
    pub fn varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                out.push(b | 0x80);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn tag(out: &mut Vec<u8>, field: u32, wire: u8) {
        varint(out, ((field as u64) << 3) | wire as u64);
    }

    /// A varint field.
    pub fn put_varint(out: &mut Vec<u8>, field: u32, v: u64) {
        tag(out, field, 0);
        varint(out, v);
    }

    /// A length-delimited field — a string, bytes, or a nested message.
    pub fn put_bytes(out: &mut Vec<u8>, field: u32, b: &[u8]) {
        tag(out, field, 2);
        varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }

    /// One field read off the wire.
    pub enum Val {
        Var(u64),
        Bytes(Vec<u8>),
    }

    /// Parse a message into its `(field, value)` pairs, skipping wire types the
    /// schemas here do not use (fixed32/fixed64).
    pub fn parse(buf: &[u8]) -> Vec<(u32, Val)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < buf.len() {
            let Some(tag) = read_varint(buf, &mut i) else {
                break;
            };
            let field = (tag >> 3) as u32;
            match tag & 7 {
                0 => {
                    let Some(v) = read_varint(buf, &mut i) else {
                        break;
                    };
                    out.push((field, Val::Var(v)));
                }
                2 => {
                    let Some(len) = read_varint(buf, &mut i) else {
                        break;
                    };
                    let end = i + len as usize;
                    if end > buf.len() {
                        break;
                    }
                    out.push((field, Val::Bytes(buf[i..end].to_vec())));
                    i = end;
                }
                5 => i += 4,
                1 => i += 8,
                _ => break,
            }
        }
        out
    }

    pub fn read_varint(buf: &[u8], i: &mut usize) -> Option<u64> {
        let mut shift = 0;
        let mut v = 0u64;
        loop {
            let b = *buf.get(*i)?;
            *i += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some(v);
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }

    /// The first length-delimited field `field` in `fields`.
    pub fn bytes(fields: &[(u32, Val)], field: u32) -> Option<&[u8]> {
        fields.iter().find_map(|(f, v)| match v {
            Val::Bytes(b) if *f == field => Some(b.as_slice()),
            _ => None,
        })
    }

    /// The first varint field `field` in `fields`.
    pub fn var(fields: &[(u32, Val)], field: u32) -> Option<u64> {
        fields.iter().find_map(|(f, v)| match v {
            Val::Var(n) if *f == field => Some(*n),
            _ => None,
        })
    }

    /// Whether a field is present (any wire type).
    pub fn has(fields: &[(u32, Val)], field: u32) -> bool {
        fields.iter().any(|(f, _)| *f == field)
    }
}

/// A generated RSA client identity: the PEM pair native-tls presents, and the
/// public modulus and exponent the pairing secret hashes.
pub struct Identity2 {
    pub cert_pem: String,
    pub key_pem: String,
    pub modulus: Vec<u8>,
    pub exponent: Vec<u8>,
}

/// Generate a fresh RSA-2048 self-signed identity for a client named `name`.
pub fn generate_identity(name: &str) -> Result<Identity2> {
    let mut rng = rand::thread_rng();
    let key = RsaPrivateKey::new(&mut rng, 2048)?;
    let public = key.to_public_key();
    let modulus = strip_zeros(&public.n().to_bytes_be()).to_vec();
    let exponent = strip_zeros(&public.e().to_bytes_be()).to_vec();
    let key_pem = key.to_pkcs8_pem(LineEnding::LF)?.to_string();

    let key_pair = rcgen::KeyPair::from_pkcs8_pem_and_sign_algo(&key_pem, &rcgen::PKCS_RSA_SHA256)?;
    let mut params = rcgen::CertificateParams::new(vec![name.to_string()])?;
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, name);
    let cert = params.self_signed(&key_pair)?;
    Ok(Identity2 {
        cert_pem: cert.pem(),
        key_pem,
        modulus,
        exponent,
    })
}

fn strip_zeros(b: &[u8]) -> &[u8] {
    let mut i = 0;
    while i + 1 < b.len() && b[i] == 0 {
        i += 1;
    }
    &b[i..]
}

/// Decode an even-length hex string to bytes.
fn unhex(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        bail!("odd-length hex");
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| anyhow!("bad hex")))
        .collect()
}

/// The pairing secret: `SHA-256(client_n || client_e || server_n || server_e ||
/// code_bytes)`, with the first digest byte checked against the code's checksum.
pub fn pairing_secret(client: &Identity2, server_der: &[u8], code: &str) -> Result<Vec<u8>> {
    if code.len() != 6 || unhex(code).is_err() {
        bail!("the code is six hex digits");
    }
    let (_, server) = X509Certificate::from_der(server_der)?;
    let (server_n, server_e) = match server.public_key().parsed()? {
        PublicKey::RSA(k) => (
            strip_zeros(k.modulus).to_vec(),
            strip_zeros(k.exponent).to_vec(),
        ),
        _ => bail!("the TV's certificate is not RSA"),
    };
    let mut h = Sha256::new();
    h.update(&client.modulus);
    h.update(&client.exponent);
    h.update(&server_n);
    h.update(&server_e);
    h.update(unhex(&code[2..])?);
    let digest = h.finalize();
    let checksum = u8::from_str_radix(&code[..2], 16).map_err(|_| anyhow!("bad code"))?;
    if digest[0] != checksum {
        bail!("that code did not match — check the digits on the TV");
    }
    Ok(digest.to_vec())
}

type Stream = TlsStream<TcpStream>;

/// Open a mutual-TLS stream presenting `identity`, accepting the TV's cert.
async fn connect(address: &str, port: u16, identity: &Identity2) -> Result<Stream> {
    let id = Identity::from_pkcs8(identity.cert_pem.as_bytes(), identity.key_pem.as_bytes())?;
    let native = NativeConnector::builder()
        .identity(id)
        .danger_accept_invalid_certs(true)
        .danger_accept_invalid_hostnames(true)
        .build()?;
    let connector = TlsConnector::from(native);
    let tcp = timeout(ANSWER, TcpStream::connect((address, port)))
        .await
        .map_err(|_| anyhow!("the TV did not answer in time"))??;
    let stream = timeout(ANSWER, connector.connect(address, tcp))
        .await
        .map_err(|_| anyhow!("the TLS handshake did not finish in time"))??;
    Ok(stream)
}

/// The TV's certificate from an open stream, in DER.
fn peer_der(stream: &Stream) -> Result<Vec<u8>> {
    let cert = stream
        .get_ref()
        .peer_certificate()?
        .ok_or_else(|| anyhow!("the TV presented no certificate"))?;
    Ok(cert.to_der()?)
}

/// Write a length-delimited protobuf message.
async fn send(stream: &mut Stream, msg: &[u8]) -> Result<()> {
    let mut framed = Vec::new();
    pb::varint(&mut framed, msg.len() as u64);
    framed.extend_from_slice(msg);
    stream.write_all(&framed).await?;
    stream.flush().await?;
    Ok(())
}

/// Read a length-delimited protobuf message's body.
async fn recv(stream: &mut Stream) -> Result<Vec<u8>> {
    let read = async {
        let mut len: u64 = 0;
        let mut shift = 0;
        loop {
            let mut one = [0u8; 1];
            stream.read_exact(&mut one).await?;
            len |= ((one[0] & 0x7f) as u64) << shift;
            if one[0] & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        let mut body = vec![0u8; len as usize];
        stream.read_exact(&mut body).await?;
        Ok::<Vec<u8>, anyhow::Error>(body)
    };
    timeout(ANSWER, read)
        .await
        .map_err(|_| anyhow!("the TV went quiet"))?
}

// --- pairing (polo.proto) ---

fn outer(status: u64, field: u32, sub: &[u8]) -> Vec<u8> {
    let mut m = Vec::new();
    pb::put_varint(&mut m, 1, 2); // protocol_version = 2
    pb::put_varint(&mut m, 2, status); // status = STATUS_OK (200)
    pb::put_bytes(&mut m, field, sub);
    m
}

/// A pairing session held open between showing the code and sending the secret.
pub struct Pairing {
    stream: Stream,
    identity: Identity2,
    server_der: Vec<u8>,
}

impl Pairing {
    /// Connect, request pairing, and carry the handshake up to the point the TV
    /// shows its code. The caller then reads the code and calls [`Self::finish`].
    pub async fn start(address: &str, identity: Identity2, client_name: &str) -> Result<Pairing> {
        let mut stream = connect(address, PAIR_PORT, &identity).await?;
        let server_der = peer_der(&stream)?;

        // pairing_request (10): service_name="atvremote", client_name.
        let mut req = Vec::new();
        pb::put_bytes(&mut req, 1, b"atvremote");
        pb::put_bytes(&mut req, 2, client_name.as_bytes());
        send(&mut stream, &outer(200, 10, &req)).await?;
        check_ok(&recv(&mut stream).await?, 11)?;

        // options (20): preferred_role=INPUT(1), input_encodings=[HEX(3), len 6].
        let mut enc = Vec::new();
        pb::put_varint(&mut enc, 1, 3);
        pb::put_varint(&mut enc, 2, 6);
        let mut opts = Vec::new();
        pb::put_bytes(&mut opts, 1, &enc);
        pb::put_varint(&mut opts, 3, 1);
        send(&mut stream, &outer(200, 20, &opts)).await?;
        check_ok(&recv(&mut stream).await?, 20)?;

        // configuration (30): encoding={HEX,6}, client_role=INPUT(1).
        let mut cfg = Vec::new();
        pb::put_bytes(&mut cfg, 1, &enc);
        pb::put_varint(&mut cfg, 2, 1);
        send(&mut stream, &outer(200, 30, &cfg)).await?;
        check_ok(&recv(&mut stream).await?, 31)?;
        // The TV now shows its six-hex code.
        Ok(Pairing {
            stream,
            identity,
            server_der,
        })
    }

    /// Finish with the code shown on the TV, returning the paired identity to
    /// keep — its certificate is the credential for every later connection.
    pub async fn finish(mut self, code: &str) -> Result<Identity2> {
        let secret = pairing_secret(&self.identity, &self.server_der, code)?;
        let mut s = Vec::new();
        pb::put_bytes(&mut s, 1, &secret);
        send(&mut self.stream, &outer(200, 40, &s)).await?;
        check_ok(&recv(&mut self.stream).await?, 41)?;
        Ok(self.identity)
    }
}

/// Fail unless an `OuterMessage` has `status == STATUS_OK` and the expected
/// acknowledgement field.
fn check_ok(msg: &[u8], expect: u32) -> Result<()> {
    let fields = pb::parse(msg);
    match pb::var(&fields, 2) {
        Some(200) => {}
        Some(401) => bail!("the TV rejected the pairing configuration"),
        Some(402) => bail!("the TV rejected the pairing secret"),
        other => bail!("the TV answered with status {:?}", other),
    }
    if !pb::has(&fields, expect) {
        bail!("the TV skipped a pairing step");
    }
    Ok(())
}

// --- remote (remotemessage.proto) ---

/// The features the client asks for: PING|KEY|POWER|VOLUME|APP_LINK.
const FEATURES: u64 = 1 | 2 | 32 | 64 | 512;

/// An open remote-control session, past its handshake and ready for commands.
pub struct Remote {
    stream: Stream,
}

impl Remote {
    /// Connect with the paired `identity` and run the handshake
    /// (configure → set-active → start), answering the opening ping.
    pub async fn open(address: &str, identity: &Identity2) -> Result<Remote> {
        let mut stream = connect(address, REMOTE_PORT, identity).await?;
        loop {
            let fields = pb::parse(&recv(&mut stream).await?);
            if let Some(cfg) = pb::bytes(&fields, 1) {
                // remote_configure: reply with our active feature set.
                let active = pb::var(&pb::parse(cfg), 1).unwrap_or(FEATURES) & FEATURES;
                let mut info = Vec::new();
                pb::put_varint(&mut info, 3, 1); // unknown1
                pb::put_bytes(&mut info, 4, b"1"); // unknown2
                pb::put_bytes(&mut info, 5, b"atvremote"); // package_name
                pb::put_bytes(&mut info, 6, b"1.0.0"); // app_version
                let mut rc = Vec::new();
                pb::put_varint(&mut rc, 1, active);
                pb::put_bytes(&mut rc, 2, &info);
                let mut m = Vec::new();
                pb::put_bytes(&mut m, 1, &rc);
                send(&mut stream, &m).await?;
            } else if pb::has(&fields, 2) {
                // remote_set_active: echo it.
                let mut sa = Vec::new();
                pb::put_varint(&mut sa, 1, FEATURES);
                let mut m = Vec::new();
                pb::put_bytes(&mut m, 2, &sa);
                send(&mut stream, &m).await?;
            } else if pb::has(&fields, 40) {
                // remote_start: ready.
                return Ok(Remote { stream });
            } else if let Some(ping) = pb::bytes(&fields, 8) {
                answer_ping(&mut stream, ping).await?;
            }
        }
    }

    /// Press a key (short press).
    pub async fn key(&mut self, code: i32) -> Result<()> {
        let mut ki = Vec::new();
        pb::put_varint(&mut ki, 1, code as u64);
        pb::put_varint(&mut ki, 2, 3); // direction = SHORT
        let mut m = Vec::new();
        pb::put_bytes(&mut m, 10, &ki);
        send(&mut self.stream, &m).await
    }

    /// Launch an app link — a market id, or a full URL the installed app
    /// handles (a YouTube watch URL opens YouTube on the video).
    pub async fn app_link(&mut self, link: &str) -> Result<()> {
        let mut al = Vec::new();
        pb::put_bytes(&mut al, 1, link.as_bytes());
        let mut m = Vec::new();
        pb::put_bytes(&mut m, 90, &al);
        send(&mut self.stream, &m).await
    }
}

/// Reply to a `remote_ping_request` with its `val1`.
async fn answer_ping(stream: &mut Stream, ping: &[u8]) -> Result<()> {
    let val1 = pb::var(&pb::parse(ping), 1).unwrap_or(0);
    let mut pr = Vec::new();
    pb::put_varint(&mut pr, 1, val1);
    let mut m = Vec::new();
    pb::put_bytes(&mut m, 9, &pr);
    send(stream, &m).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_and_hex() {
        assert_eq!(strip_zeros(&[0, 0, 1, 2]), &[1, 2]);
        assert_eq!(strip_zeros(&[0]), &[0]);
        assert_eq!(unhex("010001").unwrap(), vec![1, 0, 1]);
        assert!(unhex("abc").is_err());
    }

    #[test]
    fn protobuf_round_trips_nested() {
        let mut inner = Vec::new();
        pb::put_varint(&mut inner, 1, 3);
        pb::put_varint(&mut inner, 2, 6);
        let mut outer = Vec::new();
        pb::put_bytes(&mut outer, 1, &inner);
        pb::put_varint(&mut outer, 3, 1);
        let fields = pb::parse(&outer);
        assert_eq!(pb::var(&fields, 3), Some(1));
        let got_inner = pb::bytes(&fields, 1).unwrap();
        let inner_fields = pb::parse(got_inner);
        assert_eq!(pb::var(&inner_fields, 1), Some(3));
        assert_eq!(pb::var(&inner_fields, 2), Some(6));
    }

    #[test]
    fn an_outer_pairing_message_carries_version_and_status() {
        let m = outer(200, 10, b"x");
        let fields = pb::parse(&m);
        assert_eq!(pb::var(&fields, 1), Some(2)); // protocol_version
        assert_eq!(pb::var(&fields, 2), Some(200)); // status OK
        assert_eq!(pb::bytes(&fields, 10), Some(&b"x"[..]));
    }
}
