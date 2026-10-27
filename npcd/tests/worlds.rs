//! npcd worlds across in-process members (`docs/replicated_substrate.md` §9;
//! §14.1's world-state, NPC-lease and disconnected-mode tables).
//!
//! Each member is an npcd [`Runtime`] on its own substrate, replicated through
//! a [`ConversationStore`], joined to one in-process gateway — the web
//! gateway's own router over the cluster [`Hub`], so a write a member sends up
//! to a world's authority travels the path it does in production. A member's
//! connector can be closed to cut it off and opened to let it back.
//!
//! What stands in for something real is the model: no member runs one, so
//! characters take no turns of their own, and the acts here are put to the
//! world directly through [`Runtime::perform`].

use std::net::IpAddr;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use candle_conversation::persistence::{SharedSubstrate, SubstratePersistence};
use candle_conversation::projection::Conversation;
use candle_conversation::substrate::Substrate;
use cluster::consensus::rendezvous::{owner, Weighted};
use cluster::dial::Connect;
use cluster::frame::{BuildId, Capacity, MemberState, ModelId};
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::member::{AppMember, Identity, Link};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use host::cluster::in_flight::InFlight;
use host::cluster::serve::serve;
use host::cluster::store::ConversationStore;
use host::guard::dispatch_table;
use npc_map::world::{Happening, Where};
use npcd::authority::npc_lease::{hex, leases_of, lent_to};
use npcd::authority::ops::{OpReply, WorldOp};
use npcd::authority::routes;
use npcd::authority::Role;
use npcd::engine::act::Act;
use npcd::engine::runtime::Runtime;
use npcd::mind::Mind;
use npcd::world::Hosted;
use npcd::world_state::sim_records;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;
use web::auth::Roles;
use web::cluster::{streams, Gateway};
use web::{Builder, Config};

const MAPS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
const WITHIN: Duration = Duration::from_secs(30);

fn key() -> GroupKey {
    GroupKey::from_bytes([7; 32])
}

fn at(node: &str) -> Where {
    Where::new("vault-casting", node)
}

fn act(tool: &'static str, args: serde_json::Value) -> Act {
    Act {
        tool,
        args: args.as_object().expect("an object").clone(),
    }
}

/// The members' log lines in the test's own output, shown with a failure.
fn logs() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("npcd=debug,cluster=info,host=info")
        .try_init();
}

async fn until(what: &str, within: Duration, ok: impl FnMut() -> bool) {
    until_showing(what, within, ok, String::new).await
}

/// [`until`], saying what `show` sees when it times out.
async fn until_showing(
    what: &str,
    within: Duration,
    mut ok: impl FnMut() -> bool,
    show: impl Fn() -> String,
) {
    let deadline = Instant::now() + within;
    while !ok() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what}: {}",
            show()
        );
        sleep(Duration::from_millis(20)).await;
    }
}

/// The web gateway, in process: its router over a cluster hub, serving the
/// streams members open as it serves a browser.
struct Gw {
    gateway: Arc<Gateway>,
    _router: Router,
}

fn gateway() -> Gw {
    logs();
    let yaml = r#"
cluster: {group_key_file: "unused.key"}
sites:
  - name: npcd
    default: true
    api:
      - {prefix: /, upstream: "http://127.0.0.1:9", cluster: npcd}
"#;
    let cfg = Config::from_yaml(yaml, Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
    let gateway = Gateway::with_hub(Hub::new(key(), "gw"));
    let router = Builder::new(cfg).cluster(gateway.clone()).router();
    Gw {
        gateway,
        _router: router,
    }
}

/// A member's way to the gateway, over a memory pipe; refused while `open`
/// is false — a member cut off.
struct ToGateway {
    gateway: Arc<Gateway>,
    open: Arc<AtomicBool>,
}

impl Connect for ToGateway {
    type Tx = MemoryTx;
    type Rx = MemoryRx;
    async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
        if !self.open.load(Ordering::SeqCst) {
            return Err("unreachable".into());
        }
        let (m, g) = pair(8 << 20);
        let gateway = self.gateway.clone();
        let ip = IpAddr::from([10, 0, 0, 7]);
        tokio::spawn(async move {
            let hub = gateway.hub.clone();
            hub.serve("npcd", ip, g.tx, g.rx, move |s| {
                streams::serve(&gateway, ip, s)
            })
            .await;
        });
        Ok((m.tx, m.rx))
    }
}

/// A member's substrate, before it is a member: what names its node.
struct Store {
    dir: TempDir,
    shared: SharedSubstrate,
    conv: Conversation,
}

fn store() -> Store {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let sp =
        SubstratePersistence::open_in_with_substrate(&dir.path().join("substrate"), &mut substrate)
            .unwrap();
    let shared = SharedSubstrate::new(substrate, sp);
    let conv = Conversation::from_shared(shared.clone());
    Store { dir, shared, conv }
}

impl Store {
    fn node(&self) -> u64 {
        self.conv.node()
    }
}

struct Member {
    rt: Arc<Runtime>,
    app: Arc<AppMember>,
    shared: SharedSubstrate,
    open: Arc<AtomicBool>,
    world: Arc<Hosted>,
    _dir: TempDir,
}

impl Member {
    fn node(&self) -> u64 {
        self.app.node()
    }

    fn role(&self, world: &str) -> Role {
        self.rt.standing.role(world)
    }

    fn awake(&self, npc: u64) -> bool {
        self.rt.scheduler.pace_of(npc).is_some()
    }

    fn joined(&self) -> bool {
        self.app.link() == Link::Joined && self.app.replicator().is_some_and(|r| r.caught_up())
    }

    /// Cut it off: its sockets close and it cannot dial back.
    fn cut_off(&self) {
        self.open.store(false, Ordering::SeqCst);
        self.app.close_all("the network went");
    }

    fn reconnect(&self) {
        self.open.store(true, Ordering::SeqCst);
    }
}

/// Start `store` as a member of `gw`, hosting the vault as `world`, with a
/// body for each of `cast` standing in the green room.
async fn member(gw: &Gw, store: Store, world: &str, cast: &[u64]) -> Member {
    member_as(gw, store, world, cast, true).await
}

/// [`member`], taking leases or not (`Identity::leaseholder`).
async fn member_as(gw: &Gw, store: Store, world: &str, cast: &[u64], leaseholder: bool) -> Member {
    let Store { dir, shared, conv } = store;
    let node = conv.node();
    let replication = ConversationStore::new(conv, 0, || {})
        .replication()
        .unwrap();
    let rt = Runtime::new(Mind::new(None), dir.path()).unwrap();
    rt.set_substrate(shared.clone());
    rt.standing.set_clustered();
    let hosted = rt.host(world, Path::new(MAPS)).expect("the vault loads");
    for &npc in cast {
        let body = Runtime::body_id(npc);
        hosted.with(|w| {
            w.enter(&body, format!("Maker-{npc:02}"), at("green-room"))
                .unwrap()
        });
        rt.embody(npc, world, &body, 0).expect("bound");
    }
    let api = routes::api(Roles::default());
    let table = dispatch_table(api.declared()).unwrap();
    let router = api.into_routes(rt.clone()).router;
    let app = AppMember::new(
        Identity {
            app: "npcd".into(),
            node,
            host: format!("m{node:x}"),
            build: BuildId {
                count: 1,
                hash: "h".into(),
                local: String::new(),
            },
            model: Some(ModelId {
                arch: "Qwen4Exp".into(),
                weights: [1; 32],
                tokenizer: [1; 32],
                rope: "yarn".into(),
                describe: "model".into(),
            }),
            repos: vec![],
            leaseholder,
        },
        key(),
        vec!["https://gw".into()],
        table,
        Box::new(|| Capacity {
            state: MemberState::Ready,
            power: 100,
            idle: true,
            slots_free: 4,
            decodes: 0,
            prefill_backlog: 0,
            vram_free_mib: 1000,
            cores: 8,
            ..Capacity::default()
        }),
        Some(replication),
        Instant::now(),
    );
    let open = Arc::new(AtomicBool::new(true));
    let connector = ToGateway {
        gateway: gw.gateway.clone(),
        open: open.clone(),
    };
    let replicator = app.replicator().cloned();
    // One count for the member, shared by every socket it serves, as the
    // member's cluster holds it.
    let in_flight = InFlight::default();
    let run = app.clone();
    tokio::spawn(async move {
        run.run(0, connector, move |conn| {
            serve(
                router.clone(),
                conn,
                0,
                replicator.clone(),
                in_flight.clone(),
            )
        })
        .await;
    });
    let m = Member {
        rt: rt.clone(),
        app: app.clone(),
        shared,
        open,
        world: hosted,
        _dir: dir,
    };
    until("joined and caught up", WITHIN, || m.joined()).await;
    rt.join_cluster(app);
    m
}

/// The group's members as rendezvous weighs them.
fn weighed(m: &Member) -> Vec<Weighted> {
    m.app
        .group()
        .iter()
        .filter(|i| i.state == MemberState::Ready)
        .map(|i| Weighted {
            node: i.node,
            weight: i.weight as f64,
        })
        .collect()
}

/// The first world name rendezvous over `nodes` gives to `to`.
fn world_for(nodes: &[u64], to: u64) -> String {
    let members: Vec<Weighted> = nodes
        .iter()
        .map(|&node| Weighted {
            node,
            weight: 100.0,
        })
        .collect();
    (0..)
        .map(|i| format!("w{i}"))
        .find(|w| owner(w.as_bytes(), &members) == Some(to))
        .unwrap()
}

/// The first `n` characters rendezvous over `nodes` lends to each of them.
fn cast_for(nodes: &[u64], n: usize) -> Vec<u64> {
    let members: Vec<Weighted> = nodes
        .iter()
        .map(|&node| Weighted {
            node,
            weight: 100.0,
        })
        .collect();
    let mut cast = Vec::new();
    for &node in nodes {
        cast.extend(
            (1u64..)
                .filter(|npc| lent_to(*npc, &members) == Some(node))
                .take(n),
        );
    }
    cast.sort();
    cast
}

/// The most of `members` ever acting as `world`'s authority at once, watched
/// every 5 ms from now on: what a split brain would show.
fn most_authorities(world: &str, members: &[&Member]) -> Arc<AtomicUsize> {
    let most = Arc::new(AtomicUsize::new(0));
    let watched: Vec<Arc<Runtime>> = members.iter().map(|m| m.rt.clone()).collect();
    let world = world.to_string();
    let seen = most.clone();
    tokio::spawn(async move {
        loop {
            let now = watched
                .iter()
                .filter(|rt| rt.standing.role(&world).is_authority())
                .count();
            seen.fetch_max(now, Ordering::SeqCst);
            sleep(Duration::from_millis(5)).await;
        }
    });
    most
}

fn authority_of<'a>(world: &str, members: &'a [&'a Member]) -> Vec<&'a Member> {
    members
        .iter()
        .copied()
        .filter(|m| m.role(world).is_authority())
        .collect()
}

/// **A world has one authority, its state is the same on every member, and
/// its cast runs where rendezvous lends it.** Acts a character makes on a
/// member that only follows the world are made by the authority, and what the
/// authority hears and tells reaches that member.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_world_has_one_authority_and_its_cast_runs_where_it_is_lent() {
    let gw = gateway();
    let (sa, sb) = (store(), store());
    let nodes = [sa.node(), sb.node()];
    // Named for the member that comes up first, which keeps it: what moves a
    // world is the next test's.
    let world = &world_for(&nodes, sa.node());
    let cast = cast_for(&nodes, 3);
    let a = member(&gw, sa, world, &cast).await;
    let b = member(&gw, sb, world, &cast).await;
    let both = [&a, &b];

    until("one authority, every character lent", WITHIN, || {
        let auth = authority_of(world, &both);
        auth.len() == 1 && leases_of(&auth[0].rt.standing.written(world)).len() == cast.len()
    })
    .await;
    let auth = authority_of(world, &both)[0];
    let replica = if auth.node() == a.node() { &b } else { &a };
    assert_eq!(replica.role(world), Role::Replica);

    // Every character is lent where rendezvous names it, and runs only there.
    let members = weighed(auth);
    until("each character awake only where it is lent", WITHIN, || {
        cast.iter().all(|&npc| {
            let at = lent_to(npc, &members).unwrap();
            both.iter().all(|m| m.awake(npc) == (m.node() == at))
        })
    })
    .await;
    let leases = leases_of(&auth.rt.standing.written(world));
    for &npc in &cast {
        assert_eq!(leases[&npc].holder, hex(lent_to(npc, &members).unwrap()));
    }
    // The gateway's page shows it, as the members report it in `COORD`.
    until("the status page shows the world", WITHIN, || {
        gw.gateway.hub.status(0).worlds.iter().any(|w| {
            w.world == *world
                && w.authority.as_deref() == Some(hex(auth.node()).as_str())
                && w.lent.len() == cast.len()
                && w.away.is_empty()
        })
    })
    .await;

    // The world's state is the same on both: the replica holds what the
    // authority wrote, and its world is that state.
    until("the replica follows the records", WITHIN, || {
        replica.rt.standing.written(world) == auth.rt.standing.written(world)
    })
    .await;
    assert_eq!(replica.world.sim(sim_records), auth.world.sim(sim_records));

    // A character running on the replica speaks: the authority's world hears
    // it, and the replica hears it back as the world's pulse.
    let speaker = *cast
        .iter()
        .find(|&&npc| replica.awake(npc))
        .expect("the replica runs a character");
    // Its turn opens from the world, through the authority.
    let opening = replica
        .rt
        .world_write(world, WorldOp::Begin { npc: speaker })
        .await
        .unwrap();
    assert!(matches!(opening, OpReply::Began { .. }), "{opening:?}");
    let said = replica
        .rt
        .perform(
            speaker,
            &act("shout", json!({"intent": "the lift is stuck"})),
            None,
        )
        .await;
    assert!(said.recorded.landed, "{:?}", said.recorded);
    let heard = |w: &Arc<Hosted>| {
        w.read(|w| {
            w.log()
                .iter()
                .any(|e| matches!(&e.what, Happening::Said { words, .. } if words.contains("lift")))
        })
    };
    assert!(heard(&auth.world), "the authority made the act");
    until("the replica hears it as a pulse", WITHIN, || {
        heard(&replica.world)
    })
    .await;

    // A character on the replica walks: the authority walks it, and the
    // replica sees it standing where it stopped once that is written.
    let moved = replica
        .rt
        .perform(
            speaker,
            &act("move_to", json!({"destination": "band one"})),
            None,
        )
        .await;
    assert!(moved.recorded.landed, "{:?}", moved.recorded);
    let body = Runtime::body_id(speaker);
    until("the replica sees where it stopped", WITHIN, || {
        replica.world.place_of(&body) == "vault-casting/band-one"
    })
    .await;

    // A message to that character goes onto the authority's threads, and
    // reaches the character where it runs.
    let before = replica.rt.scheduler.inbox_depth(speaker).unwrap();
    let sent = auth
        .rt
        .make(
            world,
            WorldOp::Message {
                npc: speaker,
                from: "An Operator".into(),
                text: "come up".into(),
            },
            Some(auth.node()),
        )
        .await
        .unwrap();
    assert!(matches!(sent, OpReply::Sent { sent: Some(_) }), "{sent:?}");
    until("the message reaches the character's member", WITHIN, || {
        replica.rt.scheduler.inbox_depth(speaker).unwrap() > before
    })
    .await;
}

/// **A member that takes no lease holds no world and runs no character**,
/// though the world is named for it and half the cast would be lent to it
/// were it weighed like the other: the member that takes leases holds the
/// world and every character, and the other follows the records.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_member_that_takes_no_lease_runs_nothing() {
    let gw = gateway();
    let (sa, sb) = (store(), store());
    let nodes = [sa.node(), sb.node()];
    let world = &world_for(&nodes, sb.node());
    let cast = cast_for(&nodes, 3);
    let a = member(&gw, sa, world, &cast).await;
    let b = member_as(&gw, sb, world, &cast, false).await;
    let both = [&a, &b];

    until("a holds the world and every character", WITHIN, || {
        let leases = leases_of(&a.rt.standing.written(world));
        a.role(world).is_authority()
            && leases.len() == cast.len()
            && leases.values().all(|l| l.holder == hex(a.node()))
    })
    .await;
    until("b follows the records", WITHIN, || {
        b.rt.standing.written(world) == a.rt.standing.written(world)
    })
    .await;
    until("every character awake on a alone", WITHIN, || {
        cast.iter().all(|&npc| a.awake(npc) && !b.awake(npc))
    })
    .await;
    assert_eq!(authority_of(world, &both).len(), 1);
    assert_eq!(b.role(world), Role::Replica);
    assert!(
        weighed(&a)
            .iter()
            .any(|m| m.node == b.node() && m.weight == 0.0),
        "b is listed at weight 0"
    );
}

/// **A world moves to the member rendezvous names, after the one handing it
/// over has written what it had and the next has applied it**; and the
/// characters rendezvous moves go over at their turn boundary, handed back.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_world_and_its_characters_move_to_the_member_rendezvous_names() {
    let gw = gateway();
    let (sa, sb) = (store(), store());
    let nodes = [sa.node(), sb.node()];
    let world = world_for(&nodes, sb.node());
    let cast = cast_for(&nodes, 2);

    // A alone takes the world and every character.
    let a = member(&gw, sa, &world, &cast).await;
    until("a runs the world and the cast", WITHIN, || {
        a.role(&world).is_authority() && cast.iter().all(|&npc| a.awake(npc))
    })
    .await;
    a.world.with_sim(|s| s.table_open = false);
    until("a writes it", WITHIN, || {
        a.rt.standing
            .written(&world)
            .get("sim/meta")
            .is_some_and(|m| m["table_open"] == json!(false))
    })
    .await;

    // B joins, and rendezvous names it for the world.
    let b = member(&gw, sb, &world, &cast).await;
    let most = most_authorities(&world, &[&a, &b]);
    until("b takes the world over", WITHIN, || {
        b.role(&world).is_authority() && a.role(&world) == Role::Replica
    })
    .await;
    assert!(most.load(Ordering::SeqCst) <= 1, "never two authorities");
    assert!(
        !b.world.sim(|s| s.table_open),
        "b runs the world as a left it"
    );

    // Each character runs where rendezvous now lends it — the ones it moved,
    // handed back by a.
    let both = [&a, &b];
    let members = weighed(&b);
    until_showing(
        "each character runs where it is lent now",
        WITHIN,
        || {
            let leases = leases_of(&b.rt.standing.written(&world));
            cast.iter().all(|&npc| {
                let at = lent_to(npc, &members).unwrap();
                leases
                    .get(&npc)
                    .is_some_and(|l| l.holder == hex(at) && l.next.is_none())
                    && both.iter().all(|m| m.awake(npc) == (m.node() == at))
            })
        },
        || {
            format!(
                "a={:x} {:?} {:?} b={:x} {:?} {:?} leases(b)={:?} leases(a)={:?} awake(a)={:?} \
                 awake(b)={:?} handing(a)={:?}",
                a.node(),
                a.role(&world),
                a.rt.moments(),
                b.node(),
                b.role(&world),
                b.rt.moments(),
                leases_of(&b.rt.standing.written(&world)),
                leases_of(&a.rt.standing.written(&world)),
                cast.iter().filter(|&&n| a.awake(n)).collect::<Vec<_>>(),
                cast.iter().filter(|&&n| b.awake(n)).collect::<Vec<_>>(),
                cast.iter()
                    .filter(|&&n| a.rt.standing.handing(n))
                    .collect::<Vec<_>>(),
            )
        },
    )
    .await;
    let moved: Vec<u64> = cast
        .iter()
        .copied()
        .filter(|&npc| lent_to(npc, &members) == Some(b.node()))
        .collect();
    assert!(!moved.is_empty());
    let leases = leases_of(&b.rt.standing.written(&world));
    for npc in moved {
        let after = leases[&npc].after.expect("moved by a hand-back");
        assert_eq!(after.origin, a.node(), "b waited for what a wrote");
    }
}

/// **A member cut off runs its characters in a world of its own and writes
/// nothing of it; its characters are not lent elsewhere while it is away;
/// back, it scraps that world for the authoritative one.**
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_cut_off_member_runs_its_world_locally_and_scraps_it_on_return() {
    let gw = gateway();
    let (sa, sb) = (store(), store());
    let nodes = [sa.node(), sb.node()];
    let world = "w-cut";
    let cast = cast_for(&nodes, 2);
    let a = member(&gw, sa, world, &cast).await;
    let b = member(&gw, sb, world, &cast).await;
    let both = [&a, &b];
    until("one authority, the cast spread and running", WITHIN, || {
        let auth = authority_of(world, &both);
        auth.len() == 1
            && cast
                .iter()
                .all(|&npc| both.iter().filter(|m| m.awake(npc)).count() == 1)
    })
    .await;
    let auth = authority_of(world, &both)[0];
    let away = if auth.node() == a.node() { &b } else { &a };
    let theirs: Vec<u64> = cast
        .iter()
        .copied()
        .filter(|&npc| away.awake(npc))
        .collect();
    let walker = theirs[0];
    let body = Runtime::body_id(walker);
    until("the replica follows", WITHIN, || {
        away.rt.standing.written(world) == auth.rt.standing.written(world)
    })
    .await;

    away.cut_off();
    until("the cut-off member runs the world itself", WITHIN, || {
        away.role(world) == Role::Disconnected
    })
    .await;
    let version = away.shared.custom_objects_version();

    // Its character walks, in its own world.
    let walked = away
        .rt
        .perform(
            walker,
            &act("move_to", json!({"destination": "band one"})),
            None,
        )
        .await;
    assert!(walked.recorded.landed, "{:?}", walked.recorded);
    until("it walks there, locally", WITHIN, || {
        away.world.place_of(&body) == "vault-casting/band-one"
    })
    .await;
    assert_eq!(
        away.shared.custom_objects_version(),
        version,
        "nothing of the world is written cut off"
    );
    // Its characters stay its own while it is away.
    sleep(Duration::from_secs(3)).await;
    let leases = leases_of(&auth.rt.standing.written(world));
    for npc in &theirs {
        assert_eq!(leases[npc].holder, hex(away.node()), "not lent again");
    }
    assert_eq!(auth.world.place_of(&body), "vault-casting/green-room");

    // Back: the local world is scrapped for the authority's.
    away.reconnect();
    until("it follows the world again", WITHIN, || {
        away.joined() && away.role(world) == Role::Replica
    })
    .await;
    until("its world is the authoritative one", WITHIN, || {
        away.world.place_of(&body) == "vault-casting/green-room"
    })
    .await;
}

/// **A world keeps one authority across the loss of the member running it.**
/// Three members; the authority is cut off. It stops writing at once — the
/// world is its own, locally, while it has characters there — and once its
/// term has run out at the others, the member rendezvous names among them
/// takes the world from its records. Back, the old authority scraps what it
/// lived cut off, and — rendezvous naming it again — is handed the world back.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_world_keeps_one_authority_across_the_loss_of_its_authority() {
    let gw = gateway();
    let (sa, sb, sc) = (store(), store(), store());
    let nodes = [sa.node(), sb.node(), sc.node()];
    let world = world_for(&nodes, sa.node());
    let cast = cast_for(&nodes, 1);
    let a = member(&gw, sa, &world, &cast).await;
    let b = member(&gw, sb, &world, &cast).await;
    let c = member(&gw, sc, &world, &cast).await;
    let all = [&a, &b, &c];
    let most = most_authorities(&world, &all);
    until("a is the authority", WITHIN, || {
        a.role(&world).is_authority() && authority_of(&world, &all).len() == 1
    })
    .await;
    a.world.with_sim(|s| s.table_open = false);
    until("b and c hold what a wrote", WITHIN, || {
        [&b, &c].iter().all(|m| {
            m.rt.standing.written(&world) == a.rt.standing.written(&world)
                && m.rt
                    .standing
                    .written(&world)
                    .get("sim/meta")
                    .is_some_and(|v| v["table_open"] == json!(false))
        })
    })
    .await;

    // The group has grown since a took the world alone: a renews under the
    // larger quorum, and b and c's acceptors take its lease.
    let lease = format!("world/{world}");
    let holder = format!("{:016x}", a.node());
    for m in [&b, &c] {
        let r = m.app.replicator().unwrap().clone();
        let deadline = Instant::now() + WITHIN;
        loop {
            let coord = r.coord().await.unwrap();
            let knows = coord.paxos.keys.iter().any(|k| {
                k.key == lease && k.accepted.as_deref().is_some_and(|b| b.ends_with(&holder))
            });
            if knows {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the group never learnt a's lease"
            );
            sleep(Duration::from_millis(50)).await;
        }
    }

    a.cut_off();
    until("a stops running the world as its authority", WITHIN, || {
        !a.role(&world).is_authority()
    })
    .await;
    // Its term runs out at b and c, and the one rendezvous names takes it.
    let rest = [&b, &c];
    until("b or c takes the world", Duration::from_secs(60), || {
        authority_of(&world, &rest).len() == 1
    })
    .await;
    let next = authority_of(&world, &rest)[0];
    assert!(!next.world.sim(|s| s.table_open), "taken from its records");
    assert!(!a.role(&world).is_authority());

    a.reconnect();
    let log = || {
        format!(
            "{:#?}",
            gw.gateway
                .hub
                .log
                .read(None, None, 1000)
                .iter()
                .rev()
                .map(|e| format!("{} {}", e.subject, e.text))
                .collect::<Vec<_>>()
        )
    };
    until_showing(
        "a is handed the world back",
        Duration::from_secs(60),
        || a.joined() && a.role(&world).is_authority() && authority_of(&world, &all).len() == 1,
        log,
    )
    .await;
    assert!(
        !a.world.sim(|s| s.table_open),
        "the world as its records hold it"
    );
    assert!(
        most.load(Ordering::SeqCst) <= 1,
        "never two authorities: {}",
        log()
    );
}
