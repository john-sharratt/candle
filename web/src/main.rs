//! `web` binary — the standalone front door.
//!
//! Two ways to run it, and the flag between them is the whole difference:
//!
//! ```text
//! web                    # gateway: content from disk, APIs forwarded to the daemons
//! web --authoritative    # no daemons at all: APIs answered from the built-in mocks
//! ```
//!
//! The second is what makes the console iterable on a laptop with nothing else
//! running, over real sockets rather than a stubbed `fetch`.

use std::path::PathBuf;

use clap::Parser;
use web::{Builder, Config};

#[derive(Parser, Debug)]
#[command(name = "web", about = "Static host + API gateway, by domain")]
struct Cli {
    /// Site table (YAML).
    #[arg(short, long, default_value = "web.yaml")]
    config: PathBuf,

    /// Override the bind address from the config.
    #[arg(long)]
    bind: Option<std::net::SocketAddr>,

    /// Answer every API route from the built-in mocks instead of forwarding to
    /// a daemon. Nothing outside this process is contacted.
    #[arg(long)]
    authoritative: bool,

    /// Resolve and print the site table, then exit. The fast way to check a
    /// config edit without restarting anything that is serving.
    #[arg(long)]
    check: bool,

    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let level = match cli.verbose {
        0 => "web=info",
        1 => "web=debug",
        _ => "web=trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into()),
        )
        .with_target(false)
        .init();

    let mut cfg = Config::load(&cli.config)?;
    if let Some(bind) = cli.bind {
        cfg.server.bind = bind;
    }

    if cli.check {
        // Resolve sign-in too, and report rather than return: a check that
        // prints "config OK" while the server would refuse to start on an
        // unreadable key file is worse than no check at all. Everything else
        // in the table is still worth printing, so this reports the failure
        // and exits non-zero at the end.
        let auth = cfg
            .auth
            .clone()
            .map(|a| web::auth::Auth::new(a).map(|_| ()));
        print_table(&cfg, cli.authoritative, auth.as_ref().map(|r| r.is_ok()));
        if let Some(Err(e)) = auth {
            anyhow::bail!("sign-in is configured but unusable: {e}");
        }
        return Ok(());
    }

    let mut b = Builder::new(cfg).with_auth()?;
    if cli.authoritative {
        tracing::info!("authoritative: APIs answered from the built-in mocks, nothing forwarded");
        let sites: Vec<String> = b.sites().map(str::to_owned).collect();
        b = b.authoritative();
        for name in sites {
            if let Some(router) = web::mock::for_site(&name) {
                b = b.local_api(&name, router);
            }
        }
    }
    b.serve().await
}

/// `secrets_ok` is `None` when no `auth:` block is configured, else whether
/// its key files actually loaded.
fn print_table(cfg: &Config, authoritative: bool, secrets_ok: Option<bool>) {
    println!(
        "config OK — {} site(s), bind {}{}\nsign-in: {}",
        cfg.sites.len(),
        cfg.server.bind,
        if authoritative { ", authoritative" } else { "" },
        match (&cfg.auth, secrets_ok) {
            (Some(a), ok) => format!(
                "google, cookie on {} ({}, session {}h){}",
                a.cookie_domain,
                if a.google.redirect_uri.starts_with("https://") {
                    "secure cookies"
                } else {
                    "INSECURE cookies — redirect_uri is not https"
                },
                a.session_ttl_hours,
                match ok {
                    Some(true) => " — secrets load",
                    Some(false) => " — SECRETS UNUSABLE, see below",
                    None => "",
                }
            ),
            // Naming the file it looked for and did not find, because "not
            // configured" on its own leaves you guessing whether the path is
            // wrong, the file is missing, or the key is not read from a file
            // at all — three different things to go and check.
            (None, _) => match &cfg.auth_file {
                Some(p) => format!("not configured — no {}", p.display()),
                None => "not configured — no `auth_file:` in this table".to_string(),
            },
        }
    );
    for s in &cfg.sites {
        println!(
            "  {:<10} {:<44} {}{}",
            s.name,
            if s.hosts.is_empty() {
                "*".into()
            } else {
                s.hosts.join(", ")
            },
            s.roots_abs
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(" + "),
            if s.default { "  [default]" } else { "" }
        );
        if let Some(dir) = web::site::papers_dir(s) {
            println!("       {:<10}   papers from {}", "", dir.display());
        }
        // A site this crate is the backend for is already local; a mock only
        // stands in for a daemon.
        let built_in = web::site::built_in(&s.name);
        let mocked = web::mock::for_site(&s.name).is_some();
        for r in &s.api {
            let exact = if r.exact { " (exact)" } else { "" };
            let target = match (&r.upstream, authoritative) {
                _ if built_in => "this process".to_string(),
                (_, true) if mocked => "mock API (in process)".to_string(),
                // Named plainly rather than glossed: --authoritative does not
                // conjure a mock for a site that has none, and a summary that
                // implied otherwise would be the reason someone files a bug.
                (_, true) => format!("NOTHING — no mock for site `{}`", s.name),
                (web::Upstream::Local, _) => "local API".to_string(),
                (web::Upstream::Url(u), _) => u.clone(),
            };
            println!("       {:<10} → {target}{exact}", r.prefix);
        }
    }
}
