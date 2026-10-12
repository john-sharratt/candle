//! Peer gateways sharing ACME (`docs/replicated_substrate.md` §13.4, §14.1
//! gateway peering), in one process over in-memory links, against a local
//! stand-in for the CA: it takes an order's answers through the gateways' own
//! publishing, checks every gateway its request could reach answers each one,
//! and issues a self-signed certificate valid to a date the test chooses.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use cluster::dial::Connect;
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::peer::{digest, PeerIdentity, Peering};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use cluster::version::this_build;
use http_body_util::BodyExt;
use instant_acme::AccountCredentials;
use rcgen::{date_time_ymd, CertificateParams, KeyPair};
use tempfile::TempDir;
use tokio::time::{sleep, Instant};
use tower::Service;
use web::acme::ca::Ca;
use web::acme::certs::Certs;
use web::acme::challenges::{Challenges, PREFIX};
use web::acme::issue::{IssueError, Issued};
use web::acme::ledger::{Attempt, Ledger, Outcome};
use web::acme::order::order_site;
use web::acme::peers::AcmePeers;
use web::acme::publish::Publish;
use web::acme::store::Store;
use web::tls::Resolver;

const SITE: &str = "tokera";

fn names() -> Vec<String> {
    vec!["tokera.com".into(), "www.tokera.com".into()]
}

fn key() -> GroupKey {
    GroupKey::from_bytes([9; 32])
}

struct Gw {
    _dir: TempDir,
    peering: Arc<Peering>,
    acme: Arc<AcmePeers>,
    certs: Arc<Certs>,
}

fn gateway(node: u64) -> Gw {
    let dir = TempDir::new().unwrap();
    let hub = Hub::new(key(), format!("g{node}"));
    let peering = Peering::new(
        key(),
        PeerIdentity {
            node,
            build: this_build(),
            session_key: digest(b"session key"),
            oauth_client: digest(b"client.apps.googleusercontent.com"),
        },
        hub,
    );
    let certs = Arc::new(Certs {
        store: Store::new(dir.path()),
        resolver: Resolver::new(),
        challenges: Challenges::default(),
    });
    let acme = AcmePeers::new(peering.clone(), certs.clone());
    Gw {
        _dir: dir,
        peering,
        acme,
        certs,
    }
}

/// A dial to `to`, taken as if on its TLS entrance when `secure`.
struct ToPeer {
    to: Arc<Peering>,
    secure: bool,
    open: Arc<AtomicBool>,
}

impl Connect for ToPeer {
    type Tx = MemoryTx;
    type Rx = MemoryRx;
    async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
        if !self.open.load(Ordering::SeqCst) {
            return Err("unreachable".into());
        }
        let (d, a) = pair(1 << 20);
        let (to, secure) = (self.to.clone(), self.secure);
        tokio::spawn(async move { to.accept(a.tx, a.rx, secure).await });
        Ok((d.tx, d.rx))
    }
}

fn link(from: &Gw, to: &Gw, secure: bool) {
    let peering = from.peering.clone();
    let connector = ToPeer {
        to: to.peering.clone(),
        secure,
        open: Arc::new(AtomicBool::new(true)),
    };
    tokio::spawn(async move { peering.dial("peer", connector, secure).await });
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(10)).await;
    }
}

fn linked(g: &Gw) -> usize {
    g.peering.linked().len()
}

/// `site`'s certificate on `g`: its `not_after`, if it holds one.
fn held(g: &Gw) -> Option<u64> {
    g.certs
        .held()
        .unwrap()
        .into_iter()
        .find(|s| s.site == SITE)
        .map(|s| s.not_after)
}

/// What `challenges` answers for `token`, asked as the CA asks: over HTTP.
async fn ask(challenges: &Challenges, token: &str) -> Option<String> {
    let res = challenges
        .router()
        .call(
            Request::get(format!("{PREFIX}{token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    if res.status() != StatusCode::OK {
        return None;
    }
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    Some(String::from_utf8(bytes.to_vec()).unwrap())
}

/// The stand-in CA: validates on every gateway in `reach` — wherever its
/// request might land — and issues a certificate valid to `year`-01-01.
struct FakeCa {
    reach: Vec<Challenges>,
    year: i32,
    /// For each token, how many gateways answered it.
    answered: Mutex<Vec<usize>>,
}

impl Ca for FakeCa {
    async fn order<P: Publish>(&self, names: &[String], publish: &P) -> Result<Issued, IssueError> {
        let tokens: Vec<(String, String)> = names
            .iter()
            .map(|n| (format!("tok-{n}"), format!("tok-{n}.thumbprint")))
            .collect();
        for (t, a) in &tokens {
            publish.publish(t, a).await;
        }
        let mut failed = None;
        for (t, a) in &tokens {
            let mut answered = 0;
            for c in &self.reach {
                if ask(c, t).await.as_deref() == Some(a.as_str()) {
                    answered += 1;
                } else {
                    failed = Some(format!("{t} was not answered on every gateway"));
                }
            }
            self.answered.lock().unwrap().push(answered);
        }
        for (t, _) in &tokens {
            publish.withdraw(t);
        }
        if let Some(why) = failed {
            return Err(IssueError::Failed(why));
        }
        let mut params = CertificateParams::new(names.to_vec()).unwrap();
        params.not_after = date_time_ymd(self.year, 1, 1);
        let key = KeyPair::generate().unwrap();
        let cert = params.self_signed(&key).unwrap();
        Ok(Issued {
            chain_pem: cert.pem(),
            key_pem: key.serialize_pem(),
        })
    }
}

fn ca(reach: &[&Gw], year: i32) -> FakeCa {
    FakeCa {
        reach: reach.iter().map(|g| g.certs.challenges.clone()).collect(),
        year,
        answered: Mutex::new(Vec::new()),
    }
}

/// **One gateway orders; every gateway answers its challenges, and serves
/// what it was issued.** Three gateways in a mesh name one orderer for the
/// site, and only it may order. Its challenges reach both peers, acknowledged,
/// before the CA validates — so the CA's request is answered wherever it
/// lands. The certificate is installed on all three and every challenge is
/// withdrawn everywhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_orderers_challenges_are_answered_everywhere_and_its_certificate_served_everywhere() {
    let gs = [gateway(1), gateway(2), gateway(3)];
    link(&gs[0], &gs[1], true);
    link(&gs[0], &gs[2], true);
    link(&gs[1], &gs[2], true);
    until("a full mesh", || gs.iter().all(|g| linked(g) == 2)).await;

    let orderer = gs[0].peering.orderer(SITE);
    assert!(gs.iter().all(|g| g.peering.orderer(SITE) == orderer));
    let o = gs.iter().position(|g| g.peering.node() == orderer).unwrap();
    for (i, g) in gs.iter().enumerate() {
        assert_eq!(g.acme.orders(SITE).is_ok(), i == o, "gateway {}", i + 1);
    }

    let ca = ca(&gs.iter().collect::<Vec<_>>(), 2031);
    let ordered = order_site(&ca, &gs[o].certs, Some(&gs[o].acme), SITE, &names()).await;
    assert!(ordered.outcome.is_ok(), "{:?}", ordered.outcome.err());
    ordered.installed.unwrap();
    assert_eq!(
        *ca.answered.lock().unwrap(),
        [3, 3],
        "each token on every gateway"
    );

    let not_after = held(&gs[o]).unwrap();
    until("the certificate on every gateway", || {
        gs.iter().all(|g| held(g) == Some(not_after))
    })
    .await;
    for g in &gs {
        assert_eq!(g.certs.resolver.names(), names());
    }
    until("every challenge withdrawn", || {
        gs.iter().all(|g| g.certs.challenges.is_empty())
    })
    .await;
}

/// The first site name gateway `g` is the orderer for, between two.
fn site_ordered_by(g: &Gw) -> String {
    (0..)
        .map(|i| format!("site{i}"))
        .find(|s| g.peering.orderer(s) == g.peering.node())
        .unwrap()
}

/// **A peer's `ACME_ORDERING` holds the site** until its `ACME_DONE`, even
/// for the gateway that would otherwise order it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_announcing_an_order_holds_the_site_until_it_is_done() {
    let (a, b) = (gateway(1), gateway(2));
    link(&a, &b, true);
    until("linked", || linked(&a) == 1 && linked(&b) == 1).await;
    let site = site_ordered_by(&b);
    assert!(b.acme.orders(&site).is_ok());
    a.acme.ordering(&site);
    until("held by a", || {
        b.acme
            .orders(&site)
            .is_err_and(|why| why == "0000000000000001 is ordering it now")
    })
    .await;
    a.acme.done(&site);
    until("released", || b.acme.orders(&site).is_ok()).await;
}

/// **A partition costs at most a duplicate, and the later `not_after` wins.**
/// Unlinked, each gateway is its own orderer and orders alone; linked again,
/// the inventory exchange leaves both serving the later certificate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn after_a_partition_the_later_certificate_wins_on_both() {
    let (a, b) = (gateway(1), gateway(2));
    for (g, year) in [(&a, 2030), (&b, 2032)] {
        assert!(g.acme.orders(SITE).is_ok(), "alone, each orders");
        let ordered = order_site(&ca(&[g], year), &g.certs, Some(&g.acme), SITE, &names()).await;
        assert!(ordered.outcome.is_ok());
    }
    let (early, late) = (held(&a).unwrap(), held(&b).unwrap());
    assert!(late > early);
    link(&a, &b, true);
    until("both hold the later one", || {
        held(&a) == Some(late) && held(&b) == Some(late)
    })
    .await;
}

/// **A certificate key and the account never cross a link that is not over
/// verified TLS.** The order and the ledger still do.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_key_crosses_an_unverified_link() {
    let (a, b) = (gateway(1), gateway(2));
    link(&a, &b, false);
    until("linked", || linked(&a) == 1 && linked(&b) == 1).await;
    let (o, other) = if a.peering.orderer(SITE) == 1 {
        (&a, &b)
    } else {
        (&b, &a)
    };
    let ordered = order_site(
        &ca(&[o, other], 2031),
        &o.certs,
        Some(&o.acme),
        SITE,
        &names(),
    )
    .await;
    assert!(ordered.outcome.is_ok(), "the challenges still cross");
    o.certs
        .store
        .save_account(&serde_json::from_str::<AccountCredentials>(CREDENTIALS).unwrap())
        .unwrap();
    o.acme.account_created().unwrap();
    sleep(Duration::from_millis(300)).await;
    assert_eq!(held(other), None, "no certificate");
    assert!(
        other.certs.store.load_account().unwrap().is_none(),
        "no account"
    );
}

/// The test credentials `instant-acme` itself uses.
const CREDENTIALS: &str = r#"{"id":"id","key_pkcs8":"MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgJVWC_QzOTCS5vtsJp2IG-UDc8cdDfeoKtxSZxaznM-mhRANCAAQenCPoGgPFTdPJ7VLLKt56RxPlYT1wNXnHc54PEyBg3LxKaH0-sJkX0mL8LyPEdsfL_Oz4TxHkWLJGrXVtNhfH","directory":"https://acme-staging-v02.api.letsencrypt.org/directory"}"#;

/// **One account, shared once; ledgers merged.** The gateway that holds the
/// account gives it to a peer that has none on connect; each gateway's ledger
/// reaches the other, and a change made later is merged on arrival.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_account_is_shared_and_the_ledgers_merge() {
    let (a, b) = (gateway(1), gateway(2));
    a.certs
        .store
        .save_account(&serde_json::from_str::<AccountCredentials>(CREDENTIALS).unwrap())
        .unwrap();
    let mut la = Ledger::default();
    la.record(Attempt {
        site: SITE.into(),
        names: names(),
        at: 1_800_000_000,
        outcome: Outcome::Issued,
    });
    la.save_merged(&a.certs.store.ledger_path()).unwrap();

    link(&a, &b, true);
    until("b holds the account and a's ledger", || {
        b.certs.store.load_account().unwrap().is_some()
            && Ledger::load(&b.certs.store.ledger_path())
                .unwrap()
                .attempts
                .len()
                == 1
    })
    .await;

    let mut lb = Ledger::load(&b.certs.store.ledger_path()).unwrap();
    lb.record(Attempt {
        site: "zend".into(),
        names: vec!["code.tokera.com".into()],
        at: 1_800_000_100,
        outcome: Outcome::Failed,
    });
    lb.save_merged(&b.certs.store.ledger_path()).unwrap();
    b.acme.ledger_changed().unwrap();
    until("a merged b's attempt", || {
        Ledger::load(&a.certs.store.ledger_path())
            .unwrap()
            .attempts
            .len()
            == 2
    })
    .await;
    assert_eq!(
        Ledger::load(&a.certs.store.ledger_path()).unwrap(),
        Ledger::load(&b.certs.store.ledger_path()).unwrap()
    );
}
