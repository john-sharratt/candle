//! The site table, in YAML: everything the gateway serves, either as the owner of
//! its files or as a proxy to the machines that run them.
//!
//! One config shape covers both roles, so a site moves from one to the other by
//! editing one line. Sign-in, TLS, the role table and the cluster's keys live in
//! files the table names, resolved against the config's directory. A named file
//! that is present and broken stops loading, while a missing sign-in or TLS file
//! just turns that feature off. Secret paths resolve against the config file, not
//! the working directory, so the same config works under systemd and from a shell.
//!
//! - `auth_block`: OpenID Connect sign-in across the estate's domains.
//! - `cluster_block`: the group, status, ops and service keys, pushes and ports.
//! - `server_block`: listeners, TLS, cache lifetimes and upstream backoff.
//! - `site_routes`: sites, hostnames, redirects and routes.
//! - `tests_files`: tests of the sign-in, TLS and table files.
//! - `tests_parse`: tests of parsing and of the refusals.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use cluster::gateway::upgrades::{CANARY_DEFAULT, GRACE_DEFAULT};
use cluster::gateway::APPS;
use cluster::machines::MachineTable;
use serde::Deserialize;

use crate::auth::Roles;

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: Server,
    #[serde(default)]
    pub sites: Vec<Site>,
    /// Sign-in, for the whole estate rather than per site — one account across
    /// every hostname is the point. Absent means no sign-in anywhere.
    ///
    /// Normally supplied by [`auth_file`](Self::auth_file) rather than written
    /// here, so this table stays identical on every machine.
    #[serde(default)]
    pub auth: Option<Auth>,
    /// A file to read [`auth`](Self::auth) from, resolved against this config.
    ///
    /// This is what lets the site table be pulled onto a deployment without
    /// touching its credentials. The block used to live inline, which meant the
    /// one machine that had a real `client_id` carried a locally-modified
    /// `web.yaml` forever — and every attempt to update the sites there failed
    /// with *your local changes would be overwritten*, on exactly the file that
    /// had nothing deployment-specific about it except those four lines.
    ///
    /// **A missing file means sign-in is off**, which is the same thing the
    /// commented-out block used to mean: a public site must not stop serving
    /// because a key it does not need is absent. A file that *is* present and
    /// does not parse is fatal, because at that point the deployment has said
    /// it wants sign-in and a gateway that quietly has none is worse.
    #[serde(default)]
    pub auth_file: Option<PathBuf>,
    /// The estate's role table, `host/roles.yaml` — the one file every daemon
    /// compiles in (`host::access::ROLES`), so the gateway and the daemons it
    /// fronts can never disagree about who is an admin. Resolved against this
    /// config's directory and read into [`roles`](Self::roles); named but
    /// missing or unparseable is fatal, since a gateway that quietly had no
    /// admins would refuse its own status page to them.
    #[serde(default)]
    pub roles_file: Option<PathBuf>,
    /// Who is an admin, from [`roles_file`](Self::roles_file). Estate-wide,
    /// because a role that meant something different on each hostname would be
    /// a role nobody could reason about. Empty with no file: no admins.
    #[serde(skip)]
    pub roles: Roles,
    /// The cluster this gateway runs (`docs/replicated_substrate.md` §3): the
    /// group key members prove they hold. Absent, no member can join and every
    /// route goes to its configured upstream, as it always has.
    #[serde(default)]
    pub cluster: Option<Cluster>,
    /// Directory the config was loaded from; relative roots resolve against it.
    #[serde(skip)]
    pub base: PathBuf,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_yaml(&text, path.parent().unwrap_or(Path::new(".")))
            .with_context(|| format!("in {}", path.display()))
    }

    /// Parse from a string — the path an embedded daemon takes, since its
    /// config is `include_str!`d rather than read from disk.
    pub fn from_yaml(text: &str, base: &Path) -> Result<Self> {
        let mut cfg: Config = serde_yaml::from_str(text).context("parsing YAML")?;
        cfg.base = base.to_path_buf();
        cfg.finish()?;
        Ok(cfg)
    }

    /// Read `roles_file` into [`roles`](Self::roles).
    fn load_roles_file(&mut self, base: &Path) -> Result<()> {
        let Some(rel) = self.roles_file.clone() else {
            return Ok(());
        };
        let path = if rel.is_absolute() {
            rel
        } else {
            base.join(rel)
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading the role table {}", path.display()))?;
        self.roles = serde_yaml::from_str(&text)
            .with_context(|| format!("parsing the role table {}", path.display()))?;
        self.roles_file = Some(path);
        Ok(())
    }

    /// Fold `auth_file` into [`auth`](Self::auth).
    ///
    /// Absent file, sign-in off. Present and broken, hard failure — the two
    /// halves of the rule that keeps a public site serving without credentials
    /// while refusing to run a half-configured one.
    fn load_auth_file(&mut self, base: &Path) -> Result<()> {
        let Some(rel) = self.auth_file.clone() else {
            return Ok(());
        };
        if self.auth.is_some() {
            // Silently preferring one would make the other look ignored, and
            // which one won would depend on knowing this function exists.
            bail!("both `auth:` and `auth_file:` are set; use one");
        }

        let path = if rel.is_absolute() {
            rel
        } else {
            base.join(rel)
        };
        // Kept resolved, so `--check` can name the exact path it looked at
        // rather than the relative one it was given.
        self.auth_file = Some(path.clone());

        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            // Distinguished on purpose: a file that is not there is a
            // deployment without sign-in, and any other error — unreadable,
            // a directory, a bad mount — is a deployment that meant to have it.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!("sign-in: no {} — running without it", path.display());
                return Ok(());
            }
            Err(e) => bail!("reading {}: {e}", path.display()),
        };

        self.auth = Some(
            serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        );
        Ok(())
    }

    /// Fold `tls_file` into [`Server::tls`] and resolve the ACME store against
    /// the config. Absent file, no TLS entrance; present and broken, a hard
    /// failure — the rule [`load_auth_file`](Self::load_auth_file) follows.
    fn resolve_tls(&mut self, base: &Path) -> Result<()> {
        if let Some(rel) = self.server.tls_file.clone() {
            if self.server.tls.is_some() {
                bail!("both `tls:` and `tls_file:` are set; use one");
            }
            let path = if rel.is_absolute() {
                rel
            } else {
                base.join(rel)
            };
            self.server.tls_file = Some(path.clone());
            match std::fs::read_to_string(&path) {
                Ok(text) => {
                    self.server.tls = Some(
                        serde_yaml::from_str(&text)
                            .with_context(|| format!("parsing {}", path.display()))?,
                    );
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tracing::info!(
                        "tls: no {} — running without the TLS entrance",
                        path.display()
                    );
                }
                Err(e) => bail!("reading {}: {e}", path.display()),
            }
        }
        if let Some(tls) = &mut self.server.tls {
            if tls.acme.store.is_relative() {
                tls.acme.store = base.join(&tls.acme.store);
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.sites.is_empty() {
            bail!("no sites declared — nothing to serve");
        }
        match self.sites.iter().filter(|s| s.default).count() {
            0 => bail!("no site has `default: true` — an unmatched Host would have nowhere to go"),
            1 => {}
            n => bail!("{n} sites have `default: true`; exactly one may"),
        }

        let base = self.base.clone();

        self.load_auth_file(&base)?;
        self.load_roles_file(&base)?;
        self.resolve_tls(&base)?;

        // Secret paths resolve against the config file, exactly like `roots`
        // and `papers`. Read as given they would follow the working directory
        // instead, so the same config would work under systemd and fail from a
        // shell — and a relative path is what makes the table portable between
        // a laptop and the DMZ box in the first place.
        if let Some(auth) = &mut self.auth {
            for p in [
                &mut auth.session_secret_file,
                &mut auth.google.client_secret_file,
            ] {
                if p.is_relative() {
                    *p = base.join(&*p);
                }
            }
        }
        if let Some(cluster) = &mut self.cluster {
            if cluster.group_key_file.is_relative() {
                cluster.group_key_file = base.join(&cluster.group_key_file);
            }
            if let Some(p) = &mut cluster.status_key_file {
                if p.is_relative() {
                    *p = base.join(&*p);
                }
            }
            if let Some(ops) = &mut cluster.ops_key {
                if ops.file.is_relative() {
                    ops.file = base.join(&ops.file);
                }
            }
            for key in &mut cluster.service_keys {
                if key.key_file.is_relative() {
                    key.key_file = base.join(&key.key_file);
                }
            }
            if let Some(pushes) = &mut cluster.deploy_on_push {
                if let Some(p) = &mut pushes.github_hook_key_file {
                    if p.is_relative() {
                        *p = base.join(&*p);
                    }
                }
            }
            for forward in &cluster.udp_forward {
                if !APPS.contains(&forward.app.as_str()) {
                    bail!(
                        "`cluster.udp_forward` names `{}`, which is no app — one of {}",
                        forward.app,
                        APPS.join(", ")
                    );
                }
                let unnamed = |p: &UdpPort| p.listen == 0 || p.member == 0;
                if forward.ports.is_empty() || forward.ports.iter().any(unnamed) {
                    bail!(
                        "`cluster.udp_forward` for `{}` names no port, or port 0",
                        forward.app
                    );
                }
            }
            let mut names: Vec<&str> = cluster
                .service_keys
                .iter()
                .map(|k| k.name.as_str())
                .collect();
            names.sort_unstable();
            if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
                bail!("`cluster.service_keys` names `{}` twice", w[0]);
            }
            if cluster.machines_file.is_relative() {
                cluster.machines_file = base.join(&cluster.machines_file);
            }
            let text = std::fs::read_to_string(&cluster.machines_file).with_context(|| {
                format!(
                    "reading the machine table {}",
                    cluster.machines_file.display()
                )
            })?;
            cluster.machines = MachineTable::from_yaml(&text).with_context(|| {
                format!(
                    "parsing the machine table {}",
                    cluster.machines_file.display()
                )
            })?;
            // Peers share certificates through the ACME store, which only a
            // gateway with a TLS entrance has; its node id lives there too.
            if !cluster.peers.is_empty() && self.server.tls.is_none() {
                bail!(
                    "`cluster.peers` is set and there is no TLS entrance: peer gateways share \
                     the TLS entrance's certificates, so set `server.tls` or `tls_file`"
                );
            }
        }

        for site in &mut self.sites {
            for root in &site.roots {
                let abs = if root.is_absolute() {
                    root.clone()
                } else {
                    base.join(root)
                };
                let real = abs.canonicalize().with_context(|| {
                    format!(
                        "site `{}`: root {} does not exist",
                        site.name,
                        abs.display()
                    )
                })?;
                site.roots_abs.push(real);
            }
            if let Some(dir) = &site.papers {
                let abs = if dir.is_absolute() {
                    dir.clone()
                } else {
                    base.join(dir)
                };
                site.papers_abs = Some(abs.canonicalize().with_context(|| {
                    format!(
                        "site `{}`: papers directory {} does not exist",
                        site.name,
                        abs.display()
                    )
                })?);
            }
            // A scope is a word a token endpoint lists and a header carries,
            // space-separated: letters, digits, `_` and `-`.
            let word = |s: &String| {
                !s.is_empty()
                    && s.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            };
            if let Some(bad) = site.api_scopes.iter().find(|s| !word(s)) {
                bail!(
                    "site `{}`: `api_scopes` names `{bad}`, which is not a word",
                    site.name
                );
            }
            for r in &site.api {
                if !r.prefix.starts_with('/') {
                    bail!(
                        "site `{}`: prefix `{}` must start with /",
                        site.name,
                        r.prefix
                    );
                }
                if let Upstream::Url(u) = &r.upstream {
                    if !u.starts_with("http://") && !u.starts_with("https://") {
                        bail!(
                            "site `{}`: upstream `{u}` must be `local` or an http(s) URL",
                            site.name
                        );
                    }
                }
                // Said, not ignored: on anything but an https upstream there is
                // no certificate for the flag to describe.
                if r.self_signed && !r.upstream.url().is_some_and(|u| u.starts_with("https://")) {
                    bail!(
                        "site `{}`: `self_signed` on `{}` needs an https:// upstream",
                        site.name,
                        r.prefix
                    );
                }
                if let Some(app) = &r.cluster {
                    if !APPS.contains(&app.as_str()) {
                        bail!(
                            "site `{}`: `cluster: {app}` on `{}` names no app — one of {}",
                            site.name,
                            r.prefix,
                            APPS.join(", ")
                        );
                    }
                    // The configured upstream is what the route falls back to
                    // with no member ready; a route with none has nowhere to go.
                    if r.upstream.url().is_none() {
                        bail!(
                            "site `{}`: `cluster: {app}` on `{}` needs an upstream URL to serve \
                             it while no member is ready",
                            site.name,
                            r.prefix
                        );
                    }
                    if self.cluster.is_none() {
                        bail!(
                            "site `{}`: `cluster: {app}` on `{}`, and no `cluster:` block names \
                             the group key members prove they hold",
                            site.name,
                            r.prefix
                        );
                    }
                }
            }
            // A site with no roots is a pure gateway — legal, and how a proxy
            // that owns no content is expressed. A redirected one owns nothing
            // either, and answers everything: the 301 *is* its content.
            if site.roots.is_empty() && site.api.is_empty() && site.redirect.is_none() {
                bail!(
                    "site `{}` has neither roots, api routes nor a redirect — it can answer nothing",
                    site.name
                );
            }
            // The reverse, which is the mistake worth naming: a redirect wins
            // over both, so anything alongside it is dead configuration that
            // reads as though it were serving.
            if site.redirect.is_some() && !(site.roots.is_empty() && site.api.is_empty()) {
                bail!(
                    "site `{}` has a redirect as well as roots or api routes — the redirect \
                     answers first, so those would never be reached",
                    site.name
                );
            }
            if let Some(to) = &site.redirect {
                if !to.starts_with("http://") && !to.starts_with("https://") {
                    bail!(
                        "site `{}`: redirect `{to}` must be an http(s) URL — it becomes a \
                         `Location` header, which needs an absolute one",
                        site.name
                    );
                }
            }
        }
        Ok(())
    }

    /// The site whose routes `app`'s members serve: where a request a member
    /// forwards up for `app` is routed (§5.8).
    pub fn site_for_app(&self, app: &str) -> Option<&Site> {
        self.sites
            .iter()
            .find(|s| s.api.iter().any(|r| r.cluster.as_deref() == Some(app)))
    }

    pub fn site_for(&self, host: Option<&str>) -> &Site {
        if let Some(raw) = host {
            let h = host_name(raw);
            if let Some(s) = self.sites.iter().find(|s| s.matches_host(&h)) {
                return s;
            }
        }
        self.sites
            .iter()
            .find(|s| s.default)
            .expect("validated: exactly one default")
    }
}

mod auth_block;
mod cluster_block;
mod server_block;
mod site_routes;
#[cfg(test)]
mod tests_files;
#[cfg(test)]
mod tests_parse;

pub use auth_block::*;
pub use cluster_block::*;
pub use server_block::*;
pub use site_routes::*;
