//! A maintenance command as it arrives (`docs/replicated_substrate.md` §12.4):
//! its name from the path, its arguments from the JSON body, and a destructive
//! one's `confirm` checked against this machine's name.

use cluster::upgrade::decisions::Rollout;
use cluster::upgrade::runtime::commands::build_of;
use serde::Deserialize;
use serde_json::{json, Value};

/// What a command acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// The process: every app, or every tenant's substrate.
    Process,
    /// One app's cluster membership.
    Cluster,
    /// The tenants' substrates.
    Substrate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Drain,
    Undrain,
    Restart,
    Shutdown,
    Compact,
    Maintain,
    ExpireTombstones,
    Audit,
    SelfCheck,
    /// From the member named, or the one each gateway picks.
    CatchUp {
        from: Option<u64>,
    },
    Resync,
    Rejoin,
    CheckDisk,
    ReleaseLeases,
    SetQuorum {
        max_members: u32,
    },
    /// Accept a build whose certification failed or passed with gaps (§3.8).
    CertifyAnyway {
        build: String,
    },
    /// Keep a build rejected, and drain the member running it (§3.8).
    RejectBuild {
        build: String,
    },
    PauseRollout,
    ResumeRollout,
    /// Decide a forced rollout's confirmed plan (§3.11).
    Rollout {
        plan: Rollout,
    },
    /// Put this member on a build it keeps: the previous by default (§3.9).
    Rollback {
        build: Option<String>,
    },
    /// Delete kept builds: those named, or all but the newest `keep`.
    PruneBuilds {
        builds: Vec<String>,
        keep: Option<usize>,
    },
    /// Install a verified build now (§12.4).
    UpgradeNow,
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Command::Drain => "drain",
            Command::Undrain => "undrain",
            Command::Restart => "restart",
            Command::Shutdown => "shutdown",
            Command::Compact => "compact",
            Command::Maintain => "maintain",
            Command::ExpireTombstones => "expire-tombstones",
            Command::Audit => "audit",
            Command::SelfCheck => "self-check",
            Command::CatchUp { .. } => "catch-up",
            Command::Resync => "resync",
            Command::Rejoin => "rejoin",
            Command::CheckDisk => "check-disk",
            Command::ReleaseLeases => "release-leases",
            Command::SetQuorum { .. } => "set-quorum",
            Command::CertifyAnyway { .. } => "certify-anyway",
            Command::RejectBuild { .. } => "reject-build",
            Command::PauseRollout => "pause-rollout",
            Command::ResumeRollout => "resume-rollout",
            Command::Rollout { .. } => "rollout",
            Command::Rollback { .. } => "rollback",
            Command::PruneBuilds { .. } => "prune-builds",
            Command::UpgradeNow => "upgrade-now",
        }
    }

    /// What it was given, for its job and the coordination log: the app it
    /// acts on, and its own arguments.
    pub fn args(&self, app: Option<&str>) -> Value {
        let mut args = json!({"app": app});
        match self {
            Command::CatchUp { from: Some(n) } => args["from"] = json!(format!("{n:016x}")),
            Command::SetQuorum { max_members } => args["max_members"] = json!(max_members),
            Command::CertifyAnyway { build } | Command::RejectBuild { build } => {
                args["build"] = json!(build)
            }
            Command::Rollout { plan } => {
                args["build"] = json!(plan.build);
                args["from"] = json!(plan.from);
                args["targets"] = json!(plan.targets);
                args["parallel"] = json!(plan.parallel);
            }
            Command::Rollback { build } => args["build"] = json!(build),
            Command::PruneBuilds { builds, keep } => {
                args["builds"] = json!(builds);
                args["keep"] = json!(keep);
            }
            _ => {}
        }
        args
    }

    pub fn reach(&self) -> Reach {
        match self {
            Command::Restart | Command::Shutdown => Reach::Process,
            Command::Compact
            | Command::Maintain
            | Command::ExpireTombstones
            | Command::Audit
            | Command::SelfCheck => Reach::Substrate,
            _ => Reach::Cluster,
        }
    }
}

/// A command's body: every field any command takes.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Body {
    /// The app it acts on — `zend`, `npcd` — or every one when absent.
    pub app: Option<String>,
    /// `catch-up`: the member to catch up from, its node in hex.
    pub from: Option<String>,
    /// `set-quorum`: the new `max_members`.
    pub max_members: Option<u32>,
    /// A build, `count-hash`: `certify-anyway`, `reject-build`, `rollback`.
    pub build: Option<String>,
    /// `prune-builds`: the kept builds to delete.
    pub builds: Option<Vec<String>>,
    /// `prune-builds`: keep only the newest this many.
    pub keep: Option<usize>,
    /// `rollout`: the plan the gateway worked out.
    pub plan: Option<Rollout>,
    /// A destructive command's target, typed out: this machine's name, or for
    /// `rollout` the build's `count-hash`.
    pub confirm: Option<String>,
}

/// `name` with `body`, on the machine called `host`; or why not.
pub fn parse(name: &str, body: &Body, host: &str) -> Result<Command, String> {
    let command = match name {
        "drain" => Command::Drain,
        "undrain" => Command::Undrain,
        "restart" => Command::Restart,
        "shutdown" => Command::Shutdown,
        "compact" => Command::Compact,
        "maintain" => Command::Maintain,
        "expire-tombstones" => Command::ExpireTombstones,
        "audit" => Command::Audit,
        "self-check" => Command::SelfCheck,
        "catch-up" => Command::CatchUp {
            from: match &body.from {
                Some(hex) => Some(
                    u64::from_str_radix(hex, 16)
                        .map_err(|_| format!("`from` is a node in hex, not {hex:?}"))?,
                ),
                None => None,
            },
        },
        "resync" => Command::Resync,
        "rejoin" => Command::Rejoin,
        "check-disk" => Command::CheckDisk,
        "release-leases" => Command::ReleaseLeases,
        "set-quorum" => match body.max_members {
            Some(0) => return Err("`max_members` is at least 1".into()),
            Some(n) => Command::SetQuorum { max_members: n },
            None => return Err("set-quorum needs `max_members`".into()),
        },
        "certify-anyway" => Command::CertifyAnyway {
            build: needs_build(body, name)?,
        },
        "reject-build" => Command::RejectBuild {
            build: needs_build(body, name)?,
        },
        "pause-rollout" => Command::PauseRollout,
        "resume-rollout" => Command::ResumeRollout,
        "rollout" => {
            let plan = body
                .plan
                .clone()
                .ok_or("rollout needs the gateway's `plan`")?;
            if body.app.is_none() {
                return Err("rollout needs the `app` it rolls out".into());
            }
            // A rollout restarts every target: the build is typed to confirm.
            if body.confirm.as_deref() != Some(plan.build.as_str()) {
                return Err(format!(
                    "rollout is destructive: send the build, {:?}, as `confirm`",
                    plan.build
                ));
            }
            Command::Rollout { plan }
        }
        "rollback" => Command::Rollback {
            build: body.build.clone(),
        },
        "prune-builds" => {
            let builds = body.builds.clone().unwrap_or_default();
            if builds.is_empty() && body.keep.is_none() {
                return Err(
                    "prune-builds needs the `builds` to delete or how many to `keep`".into(),
                );
            }
            Command::PruneBuilds {
                builds,
                keep: body.keep,
            }
        }
        "upgrade-now" => Command::UpgradeNow,
        other => return Err(format!("there is no command {other:?}")),
    };
    // Typed out, the target is named twice: once by where the request went,
    // once by the hand that sent it.
    if matches!(
        command,
        Command::Shutdown | Command::SetQuorum { .. } | Command::PruneBuilds { .. }
    ) && body.confirm.as_deref() != Some(host)
    {
        return Err(format!(
            "{} is destructive: send this machine's name, {host:?}, as `confirm`",
            command.name()
        ));
    }
    Ok(command)
}

/// The build a command names, `count-hash`.
fn needs_build(body: &Body, name: &str) -> Result<String, String> {
    let build = body
        .build
        .clone()
        .ok_or_else(|| format!("{name} needs the `build`, `count-hash`"))?;
    build_of(&build)?;
    Ok(build)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(json: &str) -> Body {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn every_command_parses_by_its_name() {
        let none = Body::default();
        for name in [
            "drain",
            "undrain",
            "restart",
            "compact",
            "maintain",
            "expire-tombstones",
            "audit",
            "self-check",
            "catch-up",
            "resync",
            "rejoin",
            "check-disk",
            "release-leases",
            "pause-rollout",
            "resume-rollout",
            "rollback",
            "upgrade-now",
        ] {
            assert_eq!(parse(name, &none, "m1").unwrap().name(), name);
        }
        assert_eq!(
            parse("catch-up", &body(r#"{"from":"00000000000000ab"}"#), "m1"),
            Ok(Command::CatchUp { from: Some(0xab) })
        );
        assert_eq!(
            parse("catch-up", &body(r#"{"from":"zz"}"#), "m1"),
            Err("`from` is a node in hex, not \"zz\"".into())
        );
        assert_eq!(
            parse("frobnicate", &none, "m1"),
            Err("there is no command \"frobnicate\"".into())
        );
    }

    /// `shutdown` and `set-quorum` need the machine's name typed as `confirm`.
    #[test]
    fn a_destructive_command_needs_its_target_named() {
        assert_eq!(
            parse("shutdown", &Body::default(), "m1"),
            Err("shutdown is destructive: send this machine's name, \"m1\", as `confirm`".into())
        );
        assert_eq!(
            parse("shutdown", &body(r#"{"confirm":"m2"}"#), "m1").unwrap_err(),
            "shutdown is destructive: send this machine's name, \"m1\", as `confirm`"
        );
        assert_eq!(
            parse("shutdown", &body(r#"{"confirm":"m1"}"#), "m1"),
            Ok(Command::Shutdown)
        );
        assert_eq!(
            parse("set-quorum", &body(r#"{"max_members":2}"#), "m1").unwrap_err(),
            "set-quorum is destructive: send this machine's name, \"m1\", as `confirm`"
        );
        assert_eq!(
            parse(
                "set-quorum",
                &body(r#"{"max_members":2,"confirm":"m1"}"#),
                "m1"
            ),
            Ok(Command::SetQuorum { max_members: 2 })
        );
        assert_eq!(
            parse(
                "set-quorum",
                &body(r#"{"max_members":0,"confirm":"m1"}"#),
                "m1"
            ),
            Err("`max_members` is at least 1".into())
        );
    }

    /// The upgrade commands: a verdict names its build; `prune-builds` needs
    /// what to delete and the host typed; `rollout` needs the gateway's plan,
    /// its app, and the build typed.
    #[test]
    fn the_upgrade_commands_name_what_they_act_on() {
        assert_eq!(
            parse("certify-anyway", &body(r#"{"build":"12-h12"}"#), "m1"),
            Ok(Command::CertifyAnyway {
                build: "12-h12".into()
            })
        );
        assert_eq!(
            parse("reject-build", &Body::default(), "m1").unwrap_err(),
            "reject-build needs the `build`, `count-hash`"
        );
        assert!(parse("reject-build", &body(r#"{"build":"twelve"}"#), "m1").is_err());
        assert_eq!(
            parse("prune-builds", &body(r#"{"confirm":"m1"}"#), "m1").unwrap_err(),
            "prune-builds needs the `builds` to delete or how many to `keep`"
        );
        assert_eq!(
            parse("prune-builds", &body(r#"{"keep":2}"#), "m1").unwrap_err(),
            "prune-builds is destructive: send this machine's name, \"m1\", as `confirm`"
        );
        assert_eq!(
            parse("prune-builds", &body(r#"{"keep":2,"confirm":"m1"}"#), "m1"),
            Ok(Command::PruneBuilds {
                builds: vec![],
                keep: Some(2)
            })
        );
        let plan = r#"{"build":"10-h10","from":"01","targets":["02"],"parallel":1,"artifacts":[],"rejects":["12-h12"],"by":"ops","at_ms":5}"#;
        let unconfirmed = format!(r#"{{"app":"zend","plan":{plan}}}"#);
        assert_eq!(
            parse("rollout", &body(&unconfirmed), "m1").unwrap_err(),
            "rollout is destructive: send the build, \"10-h10\", as `confirm`"
        );
        let confirmed = format!(r#"{{"app":"zend","plan":{plan},"confirm":"10-h10"}}"#);
        let Command::Rollout { plan } = parse("rollout", &body(&confirmed), "m1").unwrap() else {
            panic!("a rollout");
        };
        assert_eq!((plan.build.as_str(), plan.parallel), ("10-h10", 1));
        let appless = format!(
            r#"{{"plan":{plan_text},"confirm":"10-h10"}}"#,
            plan_text = r#"{"build":"10-h10","from":"01","targets":[],"parallel":1,"artifacts":[],"rejects":[],"by":"o","at_ms":0}"#
        );
        assert_eq!(
            parse("rollout", &body(&appless), "m1").unwrap_err(),
            "rollout needs the `app` it rolls out"
        );
    }

    #[test]
    fn an_unknown_field_is_refused() {
        assert!(serde_json::from_str::<Body>(r#"{"froom":"x"}"#).is_err());
    }

    /// A command's arguments are what it parsed to, not the body's spelling.
    #[test]
    fn the_arguments_are_the_commands_own() {
        assert_eq!(Command::Audit.args(None), json!({"app": null}));
        assert_eq!(
            Command::SetQuorum { max_members: 2 }.args(Some("zend")),
            json!({"app": "zend", "max_members": 2})
        );
        assert_eq!(
            Command::CatchUp { from: Some(0xab) }.args(None),
            json!({"app": null, "from": "00000000000000ab"})
        );
    }

    #[test]
    fn each_command_reaches_what_it_acts_on() {
        assert_eq!(Command::Shutdown.reach(), Reach::Process);
        assert_eq!(Command::Audit.reach(), Reach::Substrate);
        assert_eq!(Command::Drain.reach(), Reach::Cluster);
    }
}
