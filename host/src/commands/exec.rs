//! What each maintenance command does (`docs/replicated_substrate.md` §12.4),
//! run as its job. A command that cannot run says why and changes nothing. The
//! upgrade commands go to the process's upgrades (§3.8–§3.11).

use std::sync::Arc;

use candle_conversation::projection::Conversation;
use cluster::replicate::Replicator;
use cluster::upgrade::decisions::Settled;
use cluster::upgrade::runtime::Upgrader;
use serde_json::{json, Map, Value};
use tokio::task::spawn_blocking;

use super::jobs::Job;
use super::parse::Command;
use super::Commands;
use crate::cluster::disk::volume_of;
use crate::stopping::Ending;
use crate::tenant::CheckProgress;

/// How long `offline` waits before closing the sockets — long enough for the
/// job a command sent over one of them to report its end.
const OFFLINE_CLOSE_AFTER: std::time::Duration = std::time::Duration::from_secs(2);

/// One app's substrate, and its replication when it is a member.
struct Target {
    app: &'static str,
    conversation: Conversation,
    replicator: Option<Arc<Replicator>>,
}

impl Commands {
    /// The apps `app` names — every one when `None` — with their replication.
    fn replicators(
        &self,
        app: Option<&str>,
    ) -> Result<Vec<(&'static str, Arc<Replicator>)>, String> {
        let Some(cluster) = &self.cluster else {
            return Err("this daemon runs without --cluster: it has no cluster to act on".into());
        };
        let found: Vec<(&'static str, Arc<Replicator>)> = cluster
            .links()
            .filter(|l| app.is_none_or(|a| a == l.app()))
            .filter_map(|l| Some((l.app(), l.member()?.replicator()?.clone())))
            .collect();
        if found.is_empty() {
            return Err(match app {
                Some(a) => format!("no app {a:?} is joined here"),
                None => "no app is joined yet".into(),
            });
        }
        Ok(found)
    }

    /// The substrates `app` names, each open.
    fn targets(&self, app: Option<&str>) -> Result<Vec<Target>, String> {
        let mut targets = Vec::new();
        for (name, reports) in &self.tenants {
            if app.is_some_and(|a| a != *name) {
                continue;
            }
            let Some(conversation) = reports.conversation() else {
                return Err(format!("{name}'s substrate is not open yet"));
            };
            let replicator = self
                .cluster
                .as_ref()
                .and_then(|c| c.link(name))
                .and_then(|l| l.member())
                .and_then(|m| m.replicator().cloned());
            targets.push(Target {
                app: name,
                conversation,
                replicator,
            });
        }
        if targets.is_empty() {
            return Err(format!("no app {:?} runs here", app.unwrap_or_default()));
        }
        Ok(targets)
    }

    /// The process's upgrades, with `--cluster`.
    fn upgrader(&self) -> Result<Arc<Upgrader>, String> {
        self.cluster
            .as_ref()
            .map(|c| c.upgrader().clone())
            .ok_or_else(|| "this daemon runs without --cluster: it upgrades with no group".into())
    }

    /// Run `command` on `app` as `job`, for `by`.
    pub async fn exec(
        &self,
        job: &Arc<Job>,
        command: &Command,
        app: Option<&str>,
        by: &str,
    ) -> Result<Value, String> {
        match command {
            Command::CertifyAnyway { build } | Command::RejectBuild { build } => {
                let up = self.upgrader()?;
                let (result, reason) = match command {
                    Command::CertifyAnyway { .. } => (Settled::Pass, "certify-anyway"),
                    _ => (Settled::Fail, "reject-build"),
                };
                let mut out = Map::new();
                for (name, _) in self.replicators(app)? {
                    let n = up.settle(name, build, result, by, reason).await?;
                    job.step(name, format!("certify/{name}/{build}/{n}: {reason}"));
                    out.insert(name.into(), json!({ "n": n }));
                }
                Ok(Value::Object(out))
            }
            Command::PauseRollout | Command::ResumeRollout => {
                let up = self.upgrader()?;
                let paused = *command == Command::PauseRollout;
                let mut out = Map::new();
                for (name, _) in self.replicators(app)? {
                    let n = up.pause(name, paused, by).await?;
                    job.step(
                        name,
                        if paused {
                            "the rollout holds between members"
                        } else {
                            "the rollout goes on"
                        },
                    );
                    out.insert(name.into(), json!({ "n": n, "paused": paused }));
                }
                Ok(Value::Object(out))
            }
            Command::Rollout { plan } => {
                let up = self.upgrader()?;
                let app = app.ok_or("rollout needs the `app` it rolls out")?;
                let n = up.rollout(app, plan).await?;
                job.step(
                    app,
                    format!(
                        "rollout/{app}/{n}: {} onto {} members, {} at a time",
                        plan.build,
                        plan.targets.len(),
                        plan.parallel
                    ),
                );
                Ok(json!({ "n": n }))
            }
            Command::Rollback { build } => {
                let name = self.upgrader()?.rollback(build.as_deref())?;
                job.step(
                    "rollback",
                    format!("{name}, from builds/, once upgrade/<app> is this member's"),
                );
                Ok(json!({ "build": name }))
            }
            Command::UpgradeNow => {
                let name = self.upgrader()?.upgrade_now()?;
                job.step(
                    "upgrade",
                    format!("{name}, once upgrade/<app> is this member's"),
                );
                Ok(json!({ "build": name }))
            }
            Command::PruneBuilds { builds, keep } => {
                let gone = self.upgrader()?.prune_builds(builds, *keep)?;
                job.step("prune", format!("deleted {} kept builds", gone.len()));
                Ok(json!({ "deleted": gone }))
            }
            Command::Drain => {
                let mut out = Map::new();
                for (name, r) in self.replicators(app)? {
                    let handed = r.drain().await;
                    job.step(
                        name,
                        format!(
                            "handed back {} claims and {} leases; routed nothing new",
                            handed.claims, handed.leases
                        ),
                    );
                    out.insert(
                        name.into(),
                        json!({"claims": handed.claims, "leases": handed.leases}),
                    );
                }
                Ok(Value::Object(out))
            }
            Command::Undrain => {
                for (name, r) in self.replicators(app)? {
                    r.undrain().map_err(|e| format!("{name}: {e}"))?;
                    job.step(name, "catching up, then taking work again");
                }
                Ok(json!({}))
            }
            Command::Offline | Command::Online => {
                let offline = *command == Command::Offline;
                ::cluster::offline::set(offline)
                    .map_err(|e| format!("the switch is set but not kept across a restart: {e}"))?;
                candle_conversation::set_hub_offline(offline);
                if offline {
                    // Dropped rather than left to their next failure: each
                    // dials again, is refused, and the member writes detached.
                    // A moment late, so a command sent over one of them — the
                    // gateway page's — sees its job end before its socket does.
                    if let Some(cluster) = &self.cluster {
                        let members: Vec<_> = cluster
                            .links()
                            .filter_map(|l| l.member().cloned())
                            .collect();
                        for m in &members {
                            job.step(m.app(), "closing every socket; dialling nothing");
                        }
                        tokio::spawn(async move {
                            tokio::time::sleep(OFFLINE_CLOSE_AFTER).await;
                            // An `online` inside the wait leaves the sockets up.
                            if !::cluster::offline::is_offline() {
                                return;
                            }
                            for m in members {
                                m.close_all("offline: this machine is told the network is down");
                            }
                        });
                    }
                    tracing::warn!(
                        "OFFLINE: no gateway is dialled, nothing is fetched from the internet \
                         and only loopback callers are served, until `online`"
                    );
                } else {
                    job.step(
                        "process",
                        "online: the gateways are dialled again within seconds",
                    );
                    tracing::info!("online again: the gateways are dialled");
                }
                Ok(json!({ "offline": offline }))
            }
            Command::Restart | Command::Shutdown => {
                let ending = if *command == Command::Restart {
                    Ending::Relaunch
                } else {
                    Ending::Exit
                };
                if !self.stopping.ask(ending) {
                    return Err("the process is already stopping".into());
                }
                job.step(
                    "stopping",
                    if ending == Ending::Relaunch {
                        "draining, sending what it wrote, then starting again with the same \
                         arguments"
                    } else {
                        "draining, sending what it wrote, then exiting"
                    },
                );
                Ok(json!({"ending": command.name()}))
            }
            Command::Compact => {
                let mut out = Map::new();
                for t in self.targets(app)? {
                    let (segments, live) = self.compact(job, &t).await?;
                    out.insert(
                        t.app.into(),
                        json!({"segments": segments, "live_bytes": live}),
                    );
                }
                Ok(Value::Object(out))
            }
            Command::Maintain => {
                let mut out = Map::new();
                for t in self.targets(app)? {
                    let faulted = t.replicator.as_ref().is_some_and(|r| r.faulted());
                    job.step(
                        t.app,
                        if faulted {
                            "in fault mode: dropping a dead segment, nothing that needs scratch \
                             space"
                        } else {
                            "one maintenance pass, the planner's pick"
                        },
                    );
                    let conv = t.conversation.clone();
                    let (ran, (segments, last, _)) = spawn_blocking(move || {
                        let ran = if faulted {
                            conv.drop_dead_segment()
                        } else {
                            conv.force_compact_persistence()
                        };
                        (ran, conv.maintenance_status())
                    })
                    .await
                    .map_err(|e| format!("{}: the pass panicked: {e}", t.app))?;
                    let ran = ran.map_err(|e| format!("{}: {e}", t.app))?;
                    out.insert(
                        t.app.into(),
                        json!({"ran": ran, "segments": segments, "last_op": last.map(|(op, _)| op)}),
                    );
                }
                Ok(Value::Object(out))
            }
            Command::ExpireTombstones => {
                let mut out = Map::new();
                for t in self.targets(app)? {
                    let before = t.conversation.tombstone_counts();
                    if before.swept == 0 {
                        job.step(
                            t.app,
                            format!(
                                "none of its {} tombstones has had a complete compaction run past \
                                 it: none can expire yet",
                                before.held
                            ),
                        );
                        out.insert(t.app.into(), json!({"expired": 0, "held": before.held}));
                        continue;
                    }
                    job.step(
                        t.app,
                        format!(
                            "{} tombstones are past a complete compaction: compacting to expire \
                             the old ones",
                            before.swept
                        ),
                    );
                    self.compact(job, &t).await?;
                    let after = t.conversation.tombstone_counts();
                    out.insert(
                        t.app.into(),
                        json!({
                            "expired": after.deleted.saturating_sub(before.deleted),
                            "held": after.held,
                        }),
                    );
                }
                Ok(Value::Object(out))
            }
            Command::Audit => {
                let mut out = Map::new();
                for t in self.targets(app)? {
                    job.step(t.app, "walking every segment, changing nothing");
                    let conv = t.conversation.clone();
                    let audit = spawn_blocking(move || conv.liveness_audit())
                        .await
                        .map_err(|e| format!("{}: the audit panicked: {e}", t.app))?
                        .map_err(|e| format!("{}: {e}", t.app))?;
                    let by_type: Map<String, Value> = audit
                        .by_type()
                        .into_iter()
                        .map(|(rt, c)| {
                            (
                                format!("{rt:?}"),
                                json!({
                                    "records": c.records,
                                    "bytes": c.bytes,
                                    "counted": c.counted,
                                    "carried": c.carried,
                                    "carried_uncounted": c.carried_uncounted,
                                    "counted_uncarried": c.counted_uncarried,
                                }),
                            )
                        })
                        .collect();
                    out.insert(t.app.into(), Value::Object(by_type));
                }
                Ok(Value::Object(out))
            }
            Command::SelfCheck => {
                let mut out = Map::new();
                for (name, reports) in &self.tenants {
                    if app.is_some_and(|a| a != *name) {
                        continue;
                    }
                    let progress: CheckProgress = {
                        let job = job.clone();
                        let name = *name;
                        Arc::new(move |done, total| {
                            job.step(name, format!("{done} of {total} conversations asked"))
                        })
                    };
                    let report = reports
                        .self_check(progress)
                        .await
                        .map_err(|e| format!("{name}: {e}"))?;
                    out.insert((*name).into(), report);
                }
                Ok(Value::Object(out))
            }
            Command::CatchUp { from } => {
                let mut out = Map::new();
                for (name, r) in self.replicators(app)? {
                    let started = r.catch_up_now(*from);
                    job.step(name, format!("{started} catch-ups started"));
                    out.insert(name.into(), json!({"started": started}));
                }
                Ok(Value::Object(out))
            }
            Command::Resync => {
                let mut out = Map::new();
                for (name, r) in self.replicators(app)? {
                    let started = r.resync();
                    job.step(
                        name,
                        "catching up, then offering what it holds to every other member",
                    );
                    out.insert(name.into(), json!({"started": started}));
                }
                Ok(Value::Object(out))
            }
            Command::Rejoin => {
                let Some(cluster) = &self.cluster else {
                    return Err("this daemon runs without --cluster: it has no sockets".into());
                };
                let mut apps = Vec::new();
                for link in cluster.links() {
                    if app.is_some_and(|a| a != link.app()) {
                        continue;
                    }
                    if let Some(m) = link.member() {
                        m.close_all("rejoin: closing every socket to dial it again");
                        job.step(link.app(), "closed every socket; dialling them again");
                        apps.push(link.app());
                    }
                }
                Ok(json!({"apps": apps}))
            }
            Command::CheckDisk => {
                let mut out = Map::new();
                for (name, r) in self.replicators(app)? {
                    let disk = r.check_disk().await;
                    let fault = r.fault();
                    job.step(
                        name,
                        match &fault {
                            Some(f) => format!("in fault mode: {}", f.reason),
                            None => "not in fault mode".into(),
                        },
                    );
                    out.insert(
                        name.into(),
                        json!({
                            "free": disk.map(|d| d.free),
                            "total": disk.map(|d| d.total),
                            "fault": fault.map(|f| f.reason),
                        }),
                    );
                }
                Ok(Value::Object(out))
            }
            Command::ReleaseLeases => {
                let mut out = Map::new();
                for (name, r) in self.replicators(app)? {
                    let store = r.store().clone();
                    let through = spawn_blocking(move || store.own_through())
                        .await
                        .map_err(|e| format!("{name}: {e}"))?;
                    let leases = r.release_leases(through);
                    job.step(name, format!("handed back {leases} leases"));
                    out.insert(name.into(), json!({"leases": leases}));
                }
                Ok(Value::Object(out))
            }
            Command::SetQuorum { max_members } => {
                let mut out = Map::new();
                for (name, r) in self.replicators(app)? {
                    r.set_quorum(*max_members).await;
                    let quorum = *max_members as usize / 2 + 1;
                    job.step(
                        name,
                        format!("max_members set to {max_members}: the quorum is {quorum}"),
                    );
                    out.insert(
                        name.into(),
                        json!({"max_members": max_members, "quorum": quorum}),
                    );
                }
                Ok(Value::Object(out))
            }
        }
    }

    /// A full compaction of `t`: refused in fault mode and when the volume
    /// has less free than the live log the scratch copy needs, and run drained
    /// when the app is a member. Returns the segments after and the live bytes
    /// it copied.
    async fn compact(&self, job: &Arc<Job>, t: &Target) -> Result<(usize, u64), String> {
        if t.replicator.as_ref().is_some_and(|r| r.faulted()) {
            return Err(format!(
                "{} is in fault mode: a compaction needs scratch space it does not have",
                t.app
            ));
        }
        let conv = t.conversation.clone();
        let (live, volume) =
            spawn_blocking(move || (conv.live_bytes(), volume_of(&conv.substrate_dir())))
                .await
                .map_err(|e| format!("{}: {e}", t.app))?;
        let (free, _) =
            volume.ok_or_else(|| format!("{}: its volume could not be measured", t.app))?;
        if free < live {
            return Err(format!(
                "{}: {} MiB free, below the {} MiB of live records the compaction copies",
                t.app,
                free >> 20,
                live >> 20
            ));
        }
        let drained_here = match &t.replicator {
            Some(r) if !r.drained() => {
                job.step(t.app, "draining for the compaction");
                r.drain().await;
                true
            }
            _ => false,
        };
        job.step(
            t.app,
            format!("compacting {} MiB of live records", live >> 20),
        );
        let conv = t.conversation.clone();
        let progress = job.clone();
        let app = t.app;
        let compacted = spawn_blocking(move || {
            let report =
                |phase: usize, of: usize| progress.step(app, format!("phase {phase} of {of}"));
            conv.compact_substrate(Some(&report))
                .map(|()| conv.maintenance_status().0)
        })
        .await
        .map_err(|e| format!("{}: the compaction panicked: {e}", t.app));
        if drained_here {
            if let Some(r) = &t.replicator {
                r.undrain().map_err(|e| format!("{}: {e}", t.app))?;
                job.step(t.app, "undrained: catching up, then taking work again");
            }
        }
        let segments = compacted?.map_err(|e| format!("{}: {e}", t.app))?;
        Ok((segments, live))
    }
}
