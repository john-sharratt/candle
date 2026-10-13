//! This gateway's peers (`docs/replicated_substrate.md` §13.4): its node id,
//! kept in the ACME store beside `account.json`; who it is to them — its build,
//! and the digests of the session key and the OAuth client; and its dial loop
//! to each gateway `cluster.peers` names. A peer's own dial arrives at
//! `/v1/cluster/gateway` ([`super::endpoint`]).

use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::peer::{digest, PeerIdentity, Peering, GATEWAY_APP};
use cluster::version::this_build;
use cluster_dial::{tls_connector, Dial, Target};

use crate::acme::store::write_atomic;
use crate::config::{Config, Peer};

/// The node id's file in the ACME store.
pub const NODE_FILE: &str = "node";

/// This gateway's node id when it has no peers, kept beside the group key.
/// A peered gateway uses the ACME store's id instead, the one its `PEER_HELLO`
/// carries, so the tie-break voter is the same id on every side (§5.9).
pub fn local_node(group_key: &Path) -> Result<u64> {
    read_or_mint(&group_key.with_file_name("gateway.node"))
}

fn read_or_mint(path: &Path) -> Result<u64> {
    match fs::read_to_string(path) {
        Ok(text) => u64::from_str_radix(text.trim(), 16)
            .ok()
            .filter(|n| *n != 0)
            .ok_or_else(|| anyhow!("{} holds no node id: {text:?}", path.display())),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let node = loop {
                let n: u64 = rand::random();
                if n != 0 {
                    break n;
                }
            };
            if let Some(dir) = path.parent() {
                fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            write_atomic(path, format!("{node:016x}\n").as_bytes())?;
            Ok(node)
        }
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// This gateway's node id: read from `<store>/node`, or minted there once.
pub fn node_id(store: &Path) -> Result<u64> {
    let path = store.join(NODE_FILE);
    match fs::read_to_string(&path) {
        Ok(text) => u64::from_str_radix(text.trim(), 16)
            .ok()
            .filter(|n| *n != 0)
            .ok_or_else(|| anyhow!("{} holds no node id: {text:?}", path.display())),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let node = loop {
                let n: u64 = rand::random();
                if n != 0 {
                    break n;
                }
            };
            write_atomic(&path, format!("{node:016x}\n").as_bytes())?;
            Ok(node)
        }
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Who this gateway is to its peers, from `cfg`.
pub fn identity(cfg: &Config, node: u64) -> Result<PeerIdentity> {
    let (session_key, oauth_client) = match &cfg.auth {
        Some(auth) => {
            let key = fs::read(&auth.session_secret_file)
                .with_context(|| format!("reading {}", auth.session_secret_file.display()))?;
            (digest(&key), digest(auth.google.client_id.as_bytes()))
        }
        None => (digest(b""), digest(b"")),
    };
    Ok(PeerIdentity {
        node,
        build: this_build(),
        session_key,
        oauth_client,
    })
}

/// The peering `cfg` asks for, logging to `hub`: `None` without
/// `cluster.peers`.
pub fn peering(cfg: &Config, key: &GroupKey, hub: &Arc<Hub>) -> Result<Option<Arc<Peering>>> {
    let Some(cluster) = &cfg.cluster else {
        return Ok(None);
    };
    if cluster.peers.is_empty() {
        return Ok(None);
    }
    let tls = cfg
        .server
        .tls
        .as_ref()
        .ok_or_else(|| anyhow!("peers need the TLS entrance's ACME store"))?;
    let node = node_id(&tls.acme.store)?;
    Ok(Some(Peering::new(
        key.clone(),
        identity(cfg, node)?,
        hub.clone(),
    )))
}

/// Dial every peer in `peers`, each until its loop stops.
pub fn dial_all(peering: &Arc<Peering>, peers: &[Peer]) -> Result<()> {
    let tls = tls_connector()?;
    for p in peers {
        let target = Arc::new(Target::parse(&p.url, p.name.as_deref())?);
        let secure = target.tls_name.is_some();
        let connector = Dial {
            target: target.clone(),
            app: GATEWAY_APP,
            tls: tls.clone(),
        };
        let peering = peering.clone();
        let url = p.url.clone();
        tokio::spawn(async move {
            let stopped = peering.dial(&url, connector, secure).await;
            tracing::info!(peer = %url, ?stopped, "peer dial loop stopped");
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node id is minted once, kept in the store, and read back the same.
    #[test]
    fn the_node_id_is_minted_once_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let first = node_id(dir.path()).unwrap();
        assert_ne!(first, 0);
        assert_eq!(node_id(dir.path()).unwrap(), first);
        assert_eq!(
            fs::read_to_string(dir.path().join(NODE_FILE)).unwrap(),
            format!("{first:016x}\n")
        );
        fs::write(dir.path().join(NODE_FILE), "zz").unwrap();
        assert!(node_id(dir.path()).is_err());
    }
}
