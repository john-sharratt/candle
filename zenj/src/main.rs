//! `zenj` — Zen as DJ (`docs/zenj.md`). Started at John's logon on
//! BlackWorld, in the console session, from the repository's root:
//!
//! | Command | |
//! |---|---|
//! | `zenj watch` | run `zenj` and start it again whenever it stops: what the logon task runs |
//! | `zenj` | the night: reader, hands, conductor, brain, room, screens, tools |
//! | `zenj login <account>` | sign in once: `spotify-john`, `spotify-yen`, `tidal` |
//! | `zenj library sync` | taste → TIDAL → crates, ahead of the night |
//! | `zenj library analyse` | rekordbox's analysis into the library, once it has analysed the crates |
//! | `zenj library tag` | what Zen knows of each track: singalong, classic, its year |
//! | `zenj midi monitor` | each FLX4 message as the map names it, to check the map on the wire |
//! | `zenj learn` | send each virtual control on Enter, for rekordbox's MIDI LEARN |

use std::fs;
use std::io::{self, BufRead, Write as _};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use arc_swap::ArcSwap;
use clap::{Parser, Subcommand};
use cluster::upgrade::switch::PROOF;
use cluster::version::info::{answer_if_asked, Opens, ZENJ};
use mcp_server::SealedStore;
use rtrb::RingBuffer;
use tokio::net::TcpListener;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use web::log_file::RotatingFileLog;
use winit::event_loop::EventLoop;
use zenj::brain::budget::Budget;
use zenj::brain::curve::Curve;
use zenj::brain::night::Night;
use zenj::brain::run::{Ask, Brain};
use zenj::bus::Bus;
use zenj::conductor::{Command, Conductor, State as ConductorState};
use zenj::config::Config;
use zenj::control::Control;
use zenj::deck::offsets;
use zenj::deck::reader::{self, EXE};
use zenj::deck::Snapshot;
use zenj::events::{Events, Kind};
use zenj::hands::Hands;
use zenj::http::router;
use zenj::library::crates::{crates_of, BUILD, LIVE, NEXT, PEAK};
use zenj::library::sync::{analyse, build, file_crates, Person};
use zenj::library::{tags, Library};
use zenj::lights::wire::Udp;
use zenj::lights::{Fixtures, Lights, View};
use zenj::live::Live;
use zenj::midi::{ports, Map, Out, Reading};
use zenj::music::{Profile, TrackId};
use zenj::oauth::{login, Account};
use zenj::presence;
use zenj::rekordbox::control::Rekordbox;
use zenj::rekordbox::masterdb::{contents, decrypt_to};
use zenj::room::capture;
use zenj::room::{Analyser, Room, BLOCK};
use zenj::screens::{Screens, Shared};
use zenj::spotify::{Spotify, Watch};
use zenj::tidal::Tidal;
use zenj::tools::lights::LightsHandle;
use zenj::tools::Dj;
use zenj::zen::Zen;

/// The offsets for each rekordbox release, shipped in the binary: the
/// release is held from the freeze (§4.2), and the binary with it.
const OFFSETS: &str = include_str!("../offsets/rekordbox.txt");
const LOG_NAME: &str = "zenj.log";
/// Tracks taken from each person's taste.
const PER_PERSON: usize = 1500;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "zenj/zenj.yaml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run `zenj` with the same config, and start it again whenever it stops.
    Watch,
    /// Sign in once as an account: spotify-<person> or tidal.
    Login { account: String },
    #[command(subcommand)]
    Library(LibraryCmd),
    #[command(subcommand)]
    Midi(MidiCmd),
    /// Send each virtual control on Enter, for rekordbox's MIDI LEARN.
    Learn,
}

#[derive(Subcommand)]
enum LibraryCmd {
    Sync,
    Analyse,
    /// Ask Zen what it knows of each track not yet tagged.
    Tag,
}

#[derive(Subcommand)]
enum MidiCmd {
    Monitor,
}

fn logging(state: &Path) {
    let filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let to_file = RotatingFileLog::new(state, LOG_NAME).map(|w| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(w)
            .with_filter(filter())
    });
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(io::stderr)
                .with_filter(filter()),
        )
        .with(to_file)
        .init();
}

fn main() -> anyhow::Result<()> {
    answer_if_asked(ZENJ, Opens::default);
    let args = Args::parse();
    let config = Config::load(&args.config).map_err(|e| anyhow!(e))?;
    std::fs::create_dir_all(&config.state).with_context(|| config.state.display().to_string())?;
    // The watchdog logs nothing of its own: the zenj it runs does.
    if !matches!(args.command, Some(Cmd::Watch)) {
        logging(&config.state);
    }
    match args.command {
        Some(Cmd::Watch) => watch(&args.config),
        None => night(config),
        Some(Cmd::Login { account }) => runtime()?.block_on(sign_in(&config, &account)),
        Some(Cmd::Library(LibraryCmd::Sync)) => runtime()?.block_on(sync(&config)),
        Some(Cmd::Library(LibraryCmd::Analyse)) => analyse_library(&config),
        Some(Cmd::Library(LibraryCmd::Tag)) => runtime()?.block_on(tag(&config)),
        Some(Cmd::Midi(MidiCmd::Monitor)) => monitor(&config),
        Some(Cmd::Learn) => learn(&config),
    }
}

fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

/// Start `zenj` again whenever it stops, a few seconds after.
fn watch(config: &Path) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    loop {
        let started = Instant::now();
        let status = Process::new(&exe).arg("--config").arg(config).status();
        eprintln!(
            "zenj watch: zenj stopped ({status:?}) after {:?}",
            started.elapsed()
        );
        // A crash at start is not retried in a tight loop.
        let wait = if started.elapsed() < Duration::from_secs(30) {
            10
        } else {
            2
        };
        thread::sleep(Duration::from_secs(wait));
    }
}

fn account(config: &Config, name: &str) -> anyhow::Result<Account> {
    let provider = if name == Config::TIDAL_ACCOUNT {
        config.tidal.provider.clone()
    } else {
        config.spotify.provider.clone()
    };
    Account::open(provider, config.state.clone(), name).map_err(|e| anyhow!(e))
}

async fn sign_in(config: &Config, name: &str) -> anyhow::Result<()> {
    let provider = if name == Config::TIDAL_ACCOUNT {
        config.tidal.provider.clone()
    } else if config
        .names()
        .iter()
        .any(|p| Config::spotify_account(p) == name)
    {
        config.spotify.provider.clone()
    } else {
        return Err(anyhow!(
            "`{name}` is no account: tidal, or one of {}",
            config
                .names()
                .iter()
                .map(|p| Config::spotify_account(p))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    };
    login(&provider, &config.state, name)
        .await
        .map_err(|e| anyhow!(e))?;
    println!("Signed in as {name}.");
    Ok(())
}

async fn sync(config: &Config) -> anyhow::Result<()> {
    let tidal = Tidal::new(
        account(config, Config::TIDAL_ACCOUNT)?,
        &config.tidal.country,
    );
    let people: Vec<Person> = config
        .people
        .iter()
        .map(|p| Person {
            name: p.name.clone(),
            export: p.export.clone().filter(|d| d.is_dir()),
            spotify: account(config, &Config::spotify_account(&p.name))
                .ok()
                .map(Spotify::new),
        })
        .collect();
    let mut lib = build(&people, &tidal, PER_PERSON)
        .await
        .map_err(|e| anyhow!(e))?;
    // What was analysed before is kept: a sync does not undo an analysis.
    if let Ok(old) = Library::load(&config.library_path()) {
        lib.playlists = old.playlists.clone();
        for t in old.tracks.into_iter().filter(|t| t.anlz.is_some()) {
            if let Some(new) = lib.get_mut(&t.id) {
                let liked = new.liked.clone();
                *new = t;
                new.liked = liked;
            }
        }
    }
    file_crates(&mut lib, &tidal, &config.names())
        .await
        .map_err(|e| anyhow!(e))?;
    for name in [NEXT, LIVE] {
        if !lib.playlists.contains_key(name) {
            let id = tidal
                .create_playlist(name, "Kept by ZenJ for New Year's Eve")
                .await
                .map_err(|e| anyhow!(e))?;
            lib.playlists.insert(name.to_string(), id);
        }
    }
    // Automix's playlist, should ZenJ step down: the build and peak crates.
    let live = lib.playlists[LIVE].clone();
    let have: Vec<String> = tidal
        .items(&live)
        .await
        .map_err(|e| anyhow!(e))?
        .into_iter()
        .map(|i| i.track)
        .collect();
    let want: Vec<String> = lib
        .tracks
        .iter()
        .filter(|t| matches!(t.crate_name.as_deref(), Some(c) if c == BUILD || c == PEAK))
        .filter_map(|t| t.tidal_id.clone())
        .filter(|t| !have.contains(t))
        .collect();
    tidal
        .add(&live, &want, None)
        .await
        .map_err(|e| anyhow!(e))?;
    lib.save(&config.library_path()).map_err(|e| anyhow!(e))?;
    println!(
        "{} tracks, {} unmatched:",
        lib.tracks.len(),
        lib.unmatched.len()
    );
    for u in &lib.unmatched {
        println!("  {} — {}: {}", u.artist, u.title, u.why);
    }
    Ok(())
}

fn analyse_library(config: &Config) -> anyhow::Result<()> {
    let mut lib = Library::load(&config.library_path()).map_err(|e| anyhow!(e))?;
    let plain = config.state.join("master.plain.db");
    decrypt_to(&config.rekordbox.dir.join("master.db"), &plain).map_err(|e| anyhow!(e))?;
    let c = contents(&plain).map_err(|e| anyhow!(e))?;
    let r = analyse(
        &mut lib,
        &c,
        &config.rekordbox.dir.join("share"),
        &config.names(),
    );
    lib.save(&config.library_path()).map_err(|e| anyhow!(e))?;
    println!(
        "{} analysed; {} waiting for rekordbox; {} failed",
        r.analysed,
        r.waiting.len(),
        r.failed.len()
    );
    for w in r.waiting.iter().take(50) {
        println!("  waiting: {w}");
    }
    for f in &r.failed {
        println!("  failed: {f}");
    }
    Ok(())
}

async fn tag(config: &Config) -> anyhow::Result<()> {
    let mut lib = Library::load(&config.library_path()).map_err(|e| anyhow!(e))?;
    let zen = Zen::new(config.zen.clone(), Path::new(".")).map_err(|e| anyhow!(e))?;
    let untagged: Vec<TrackId> = lib
        .tracks
        .iter()
        .filter(|t| t.tags.is_empty())
        .map(|t| t.id.clone())
        .collect();
    let names = config.names();
    for batch in untagged.chunks(tags::BATCH) {
        let tracks: Vec<&Profile> = batch.iter().filter_map(|id| lib.get(id)).collect();
        let answer = zen
            .ask(tags::SYSTEM, &tags::prompt(&tracks))
            .await
            .map_err(|e| anyhow!(e))?;
        let known = tags::read(&answer, batch.len());
        for (i, (found, year)) in known {
            if let Some(t) = lib.get_mut(&batch[i]) {
                t.tags = found;
                t.year = t.year.or(year);
                t.crate_name = crates_of(t, &names).first().map(|c| c.to_string());
            }
        }
        lib.save(&config.library_path()).map_err(|e| anyhow!(e))?;
        println!("tagged {} of {}", batch.len(), untagged.len());
    }
    Ok(())
}

fn monitor(config: &Config) -> anyhow::Result<()> {
    let map = Map::load(&config.midi.flx4).map_err(|e| anyhow!(e))?;
    println!(
        "MIDI inputs: {:?}",
        ports::input_names().map_err(|e| anyhow!(e))?
    );
    let (tx, mut rx) = RingBuffer::new(1024);
    let _conn = ports::observe(map, tx).map_err(|e| anyhow!(e))?;
    println!(
        "Listening to {}; Ctrl+C to stop.",
        config.midi.flx4.display()
    );
    loop {
        while let Ok(t) = rx.pop() {
            let reading = match t.reading {
                Reading::At(v) => format!("{v:.4}"),
                Reading::Pressed => "pressed".into(),
                Reading::Released => "released".into(),
            };
            println!("{:<20} {reading}", t.control.to_string());
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn learn(config: &Config) -> anyhow::Result<()> {
    let map = Map::load(&config.midi.hands).map_err(|e| anyhow!(e))?;
    println!(
        "MIDI outputs: {:?}",
        ports::output_names().map_err(|e| anyhow!(e))?
    );
    let mut out = Out::open(&map.port).map_err(|e| anyhow!(e))?;
    println!(
        "In rekordbox: MIDI → the {} port → LEARN. For each control, choose its function, \
         then press Enter here to send it.",
        map.port
    );
    let stdin = io::stdin();
    let controls: Vec<Control> = map.controls().map(|(c, _)| *c).collect();
    for c in controls {
        print!("{c:<20} Enter to send… ");
        io::stdout().flush()?;
        let mut line = String::new();
        stdin.lock().read_line(&mut line)?;
        let msgs = match c {
            Control::Knob(k) => [0.0, 0.5, 1.0, 0.5]
                .iter()
                .flat_map(|v| map.set(k, *v).unwrap_or_default())
                .collect(),
            Control::Button(_) => map.press(c).unwrap_or_default(),
        };
        out.send(&msgs).map_err(|e| anyhow!(e))?;
        println!("sent");
    }
    Ok(())
}

/// The night.
fn night(config: Config) -> anyhow::Result<()> {
    let config = Arc::new(config);
    let stop = Arc::new(AtomicBool::new(false));
    let library = Library::load(&config.library_path()).unwrap_or_else(|e| {
        tracing::warn!(
            "no library yet ({e}): run `zenj library sync`, then `zenj library analyse`"
        );
        Library::default()
    });
    tracing::info!(
        tracks = library.tracks.len(),
        playable = library.playable().len(),
        "library read"
    );
    let events = Arc::new(Events::recorded_at(&config.record_path()).map_err(|e| anyhow!(e))?);

    // The deck reader, which also says whether rekordbox runs.
    let snapshot = Arc::new(ArcSwap::from_pointee(Snapshot::blind("not read yet")));
    let rekordbox_running = Arc::new(AtomicBool::new(false));
    let all = offsets::parse(OFFSETS).map_err(|e| anyhow!(e))?;
    let off = all.get(&config.rekordbox.version).cloned().ok_or_else(|| {
        anyhow!(
            "no offsets for rekordbox {} among {:?}",
            config.rekordbox.version,
            all.keys()
        )
    })?;
    {
        let (snap, running, stop, tracks, decks) = (
            snapshot.clone(),
            rekordbox_running.clone(),
            stop.clone(),
            Arc::new(library.clone()),
            config.rekordbox.decks,
        );
        thread::Builder::new()
            .name("zenj-deck-reader".into())
            .spawn(move || reader::run(off, decks, tracks, snap, running, stop))?;
    }

    // The hands: the FLX4 observed, the virtual controller opened.
    let (touch_tx, touch_rx) = RingBuffer::new(1024);
    let flx4 = Map::load(&config.midi.flx4).map_err(|e| anyhow!(e))?;
    let _observed = match ports::observe(flx4, touch_tx) {
        Ok(c) => Some(c),
        Err(e) => {
            events.push(Kind::Fault, format!("the FLX4 is not observed: {e}"));
            None
        }
    };
    let hands_map = Map::load(&config.midi.hands).map_err(|e| anyhow!(e))?;
    let out = match Out::open(&hands_map.port) {
        Ok(o) => Some(o),
        Err(e) => {
            events.push(
                Kind::Fault,
                format!("the virtual controller is not open: {e}"),
            );
            None
        }
    };

    // The lights' queues, when there are bulbs to drive: the thread itself
    // starts once the live view it reads exists.
    let fixtures = Fixtures::load(&config.lights.fixtures).map_err(|e| anyhow!(e))?;
    let (cue_tx, cue_rx) = mpsc::channel();
    let (light_ask_tx, light_ask_rx) = mpsc::channel();
    let lights_view = Arc::new(ArcSwap::from_pointee(View {
        bulbs: fixtures.len(),
        ..View::default()
    }));
    let has_lights = !fixtures.is_empty();
    if !has_lights {
        tracing::info!(
            map = %config.lights.fixtures.display(),
            "lights: no bulbs mapped"
        );
    }

    // The conductor.
    let conductor_state = Arc::new(ArcSwap::from_pointee(ConductorState::default()));
    let (cmd_tx, cmd_rx) = mpsc::channel();
    {
        let c = Conductor {
            map: hands_map,
            out,
            touches: touch_rx,
            commands: cmd_rx,
            snapshot: snapshot.clone(),
            state: conductor_state.clone(),
            events: events.clone(),
            hands: Hands::new(config.midi.combos.clone()),
            // At ~125 BPM a beat is 480 ms: the lead in beats is near enough
            // constant across the night's tempos.
            lead_beats: config.midi.lead_ms / 480.0,
            lights: has_lights.then(|| cue_tx.clone()),
        };
        let stop = stop.clone();
        thread::Builder::new()
            .name("zenj-conductor".into())
            .spawn(move || c.run(stop))?;
    }

    // The room: the FLX4's output for the room's levels and every screen.
    let landing = Arc::new(AtomicU32::new(0.5f32.to_bits()));
    let (room_tx, mut room_rx) = RingBuffer::new(1 << 16);
    let mut sinks = vec![room_tx];
    let mut screen_pcm = Vec::new();
    for _ in &config.screens.outputs {
        let (tx, rx) = RingBuffer::new(1 << 15);
        sinks.push(tx);
        screen_pcm.push(rx);
    }
    let capture = capture::open(&config.audio_device, sinks);
    let rate = match &capture {
        Ok(c) => {
            tracing::info!(device = %c.device, rate = c.rate, "room: capturing");
            c.rate
        }
        Err(e) => {
            events.push(Kind::Fault, format!("the room is not heard: {e}"));
            48_000
        }
    };
    {
        let (landing, stop) = (landing.clone(), stop.clone());
        thread::Builder::new()
            .name("zenj-room".into())
            .spawn(move || {
                let mut a = Analyser::new(rate);
                let mut room = Room::new();
                let mut block = Vec::with_capacity(BLOCK);
                while !stop.load(Ordering::Relaxed) {
                    while let Ok(x) = room_rx.pop() {
                        block.push(x);
                        if block.len() == BLOCK {
                            let now = Instant::now();
                            room.push(now, a.levels(&block));
                            landing.store(room.landing(now).to_bits(), Ordering::Relaxed);
                            block.clear();
                        }
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            })?;
    }

    // The brain, the tools and the cluster, on their runtime.
    let live = Arc::new(ArcSwap::from_pointee(Live::default()));

    // The lights: their own thread, never on the audio's path.
    if has_lights {
        let wire = Udp::open().context("the lights' socket")?;
        let l = Lights {
            fixtures,
            wire,
            snapshot: snapshot.clone(),
            live: live.clone(),
            cues: cue_rx,
            asks: light_ask_rx,
            view: lights_view.clone(),
            events: events.clone(),
            rekordbox: rekordbox_running.clone(),
            lead_ms: config.lights.lead_ms,
        };
        let stop = stop.clone();
        thread::Builder::new()
            .name("zenj-lights".into())
            .spawn(move || l.run(stop))?;
    }
    let lights = has_lights.then(|| LightsHandle {
        asks: light_ask_tx,
        view: lights_view,
    });
    let midnight = config.midnight().map_err(|e| anyhow!(e))?;
    {
        let (config, live, events, snapshot) = (
            config.clone(),
            live.clone(),
            events.clone(),
            snapshot.clone(),
        );
        thread::Builder::new()
            .name("zenj-runtime".into())
            .spawn(move || {
                let rt = match runtime() {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!("zenj: no runtime: {e}");
                        return;
                    }
                };
                if let Err(e) = rt.block_on(serve(
                    config,
                    library,
                    live,
                    events,
                    snapshot,
                    conductor_state,
                    cmd_tx,
                    landing,
                    rekordbox_running,
                    lights,
                )) {
                    tracing::error!("zenj: {e:#}");
                }
            })?;
    }

    // The screens own the main thread, as windows on Windows want.
    if config.screens.outputs.is_empty() {
        loop {
            thread::park();
        }
    }
    let event_loop = EventLoop::new()?;
    let mut screens = Screens::new(
        config.screens.clone(),
        Shared {
            live,
            snapshot,
            midnight,
            sample_rate: rate,
        },
        screen_pcm,
    );
    let _keep_capture = capture;
    event_loop.run_app(&mut screens)?;
    stop.store(true, Ordering::Relaxed);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn serve(
    config: Arc<Config>,
    library: Library,
    live: Arc<ArcSwap<Live>>,
    events: Arc<Events>,
    snapshot: Arc<ArcSwap<Snapshot>>,
    conductor: Arc<ArcSwap<ConductorState>>,
    commands: mpsc::Sender<Command>,
    landing: Arc<AtomicU32>,
    rekordbox_running: Arc<AtomicBool>,
    lights: Option<LightsHandle>,
) -> anyhow::Result<()> {
    let curve = Curve::of(config.night, &config.curve).map_err(|e| anyhow!(e))?;
    let budget = Budget::new(
        config
            .people
            .iter()
            .map(|p| (p.name.clone(), p.share))
            .collect(),
        config.request_share,
    );
    let night = Night::new(
        library,
        curve,
        budget,
        config.midnight().map_err(|e| anyhow!(e))?,
    );
    let zen = match Zen::new(config.zen.clone(), Path::new(".")) {
        Ok(z) => Some(Arc::new(z)),
        Err(e) => {
            events.push(
                Kind::Fault,
                format!("Zen is not reachable, the shortlist chooses: {e}"),
            );
            None
        }
    };
    let tidal = account(&config, Config::TIDAL_ACCOUNT)
        .map(|a| Arc::new(Tidal::new(a, &config.tidal.country)))
        .map_err(|e| events.push(Kind::Fault, format!("TIDAL is not signed in: {e}")))
        .ok();
    let (ask_tx, ask_rx) = tokio::sync::mpsc::channel::<Ask>(256);
    let brain = Brain {
        night,
        snapshot,
        conductor,
        commands,
        events: events.clone(),
        live: live.clone(),
        zen,
        tidal,
        landing,
        rekordbox: rekordbox_running.clone(),
    };
    tokio::spawn(brain.run(ask_rx));

    // rekordbox itself, opened and closed for the house: by Zen's tool, and
    // by the house's mode switch over the gateway's bus.
    let rekordbox = Arc::new(Rekordbox::new(
        config.rekordbox.program.clone(),
        EXE,
        live.clone(),
        events.clone(),
        rekordbox_running.clone(),
    ));
    match fs::read_to_string(&config.zen.key_file) {
        Ok(key) => {
            let bus = Bus::at(&config.zen.gateway, &config.zen.host, key.trim());
            tokio::spawn(bus.serve(rekordbox.clone()));
        }
        Err(e) => {
            events.push(
                Kind::Fault,
                format!(
                    "the house's asks of rekordbox are not answered: {}: {e}",
                    config.zen.key_file.display()
                ),
            );
        }
    }

    // Requests from the phones.
    let requests_owner = config
        .people
        .first()
        .map(|p| Config::spotify_account(&p.name));
    match requests_owner.and_then(|n| account(&config, &n).ok()) {
        Some(a) if !config.spotify.requests_playlist.is_empty() => {
            let spotify = Spotify::new(a);
            let mut watch = Watch::new(config.spotify_people());
            let (list, every, asks, events) = (
                config.spotify.requests_playlist.clone(),
                Duration::from_secs(config.spotify.poll_s),
                ask_tx.clone(),
                events.clone(),
            );
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(every);
                loop {
                    tick.tick().await;
                    let read = async {
                        let snap = spotify.snapshot(&list).await?;
                        if !watch.moved(&snap) {
                            return Ok(Vec::new());
                        }
                        let items = spotify.items(&list).await?;
                        Ok::<_, String>(watch.read(&snap, &items))
                    }
                    .await;
                    match read {
                        Ok(asked) => {
                            for a in asked {
                                let (reply, _) = tokio::sync::oneshot::channel();
                                let query = a
                                    .isrc
                                    .clone()
                                    .unwrap_or_else(|| format!("{} {}", a.artist, a.title));
                                let _ = asks
                                    .send(Ask::Request {
                                        query,
                                        by: a.by,
                                        note: String::new(),
                                        urgency: 0.5,
                                        now: false,
                                        reply,
                                    })
                                    .await;
                            }
                        }
                        Err(e) => {
                            events.push(Kind::Fault, format!("requests could not be read: {e}"));
                        }
                    }
                }
            });
        }
        _ => {
            events.push(
                Kind::Fault,
                "requests from the phones are not read: no Spotify sign-in or no requests playlist",
            );
        }
    }

    let dj = Arc::new(Dj {
        live,
        events,
        asks: ask_tx,
        rekordbox,
        lights,
        people: config
            .people
            .iter()
            .filter_map(|p| Some((p.email.clone()?.to_ascii_lowercase(), p.name.clone())))
            .collect(),
        store: Arc::new(SealedStore),
    });
    let app = router(dj, config.trusted_gateways.clone().into());
    let listener = TcpListener::bind(&config.bind)
        .await
        .with_context(|| config.bind.clone())?;
    presence::join(&config, app.clone())?;
    tracing::info!(bind = %config.bind, "serving");
    eprintln!("{PROOF}");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}
