//! What a maintenance command may be run by (`docs/claude_zend_gateway.md`
//! §8.2): a **Look** changes nothing; a **Care** changes one member's state,
//! recoverably; a **Decide** changes what the cluster runs or who is in it, and
//! is the user's to approve. `offline` and `online` are done on the member's own
//! machine — after `offline` it serves loopback callers alone, so `online`
//! could never reach it through a gateway — and are never run remotely.
//!
//! [`Command::class`] matches every command, so a new one does not build until
//! it is classed; [`class_of`] answers the same for a name, before parsing, for
//! a caller that holds only the name.

use super::parse::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Look,
    Care,
    Decide,
    /// Done on the member's machine, never sent to it.
    OnTheMachine,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Look => "look",
            Class::Care => "care",
            Class::Decide => "decide",
            Class::OnTheMachine => "on the machine",
        }
    }
}

/// Every command by name, with its class.
pub const COMMANDS: [(&str, Class); 25] = [
    ("audit", Class::Look),
    ("self-check", Class::Look),
    ("check-disk", Class::Look),
    ("drain", Class::Care),
    ("undrain", Class::Care),
    ("compact", Class::Care),
    ("maintain", Class::Care),
    ("expire-tombstones", Class::Care),
    ("catch-up", Class::Care),
    ("resync", Class::Care),
    ("rejoin", Class::Care),
    ("release-leases", Class::Care),
    ("pause-rollout", Class::Care),
    ("resume-rollout", Class::Care),
    ("restart", Class::Decide),
    ("shutdown", Class::Decide),
    ("set-quorum", Class::Decide),
    ("certify-anyway", Class::Decide),
    ("reject-build", Class::Decide),
    ("rollout", Class::Decide),
    ("rollback", Class::Decide),
    ("prune-builds", Class::Decide),
    ("upgrade-now", Class::Decide),
    ("offline", Class::OnTheMachine),
    ("online", Class::OnTheMachine),
];

/// The class of the command named `name`; `None` for no command.
pub fn class_of(name: &str) -> Option<Class> {
    COMMANDS.iter().find(|(n, _)| *n == name).map(|(_, c)| *c)
}

impl Command {
    pub fn class(&self) -> Class {
        match self {
            Command::Audit | Command::SelfCheck | Command::CheckDisk => Class::Look,
            Command::Drain
            | Command::Undrain
            | Command::Compact
            | Command::Maintain
            | Command::ExpireTombstones
            | Command::CatchUp { .. }
            | Command::Resync
            | Command::Rejoin
            | Command::ReleaseLeases
            | Command::PauseRollout
            | Command::ResumeRollout => Class::Care,
            Command::Restart
            | Command::Shutdown
            | Command::SetQuorum { .. }
            | Command::CertifyAnyway { .. }
            | Command::RejectBuild { .. }
            | Command::Rollout { .. }
            | Command::Rollback { .. }
            | Command::PruneBuilds { .. }
            | Command::UpgradeNow => Class::Decide,
            Command::Offline | Command::Online => Class::OnTheMachine,
        }
    }
}

#[cfg(test)]
mod tests {
    use cluster::upgrade::decisions::Rollout;

    use super::super::parse::{parse, Body};
    use super::*;

    /// One of every command.
    fn every_command() -> Vec<Command> {
        let plan: Rollout = serde_json::from_str(
            r#"{"build":"10-h10","from":"01","targets":[],"parallel":1,"artifacts":[],"rejects":[],"by":"o","at_ms":0}"#,
        )
        .unwrap();
        vec![
            Command::Drain,
            Command::Undrain,
            Command::Restart,
            Command::Shutdown,
            Command::Compact,
            Command::Maintain,
            Command::ExpireTombstones,
            Command::Audit,
            Command::SelfCheck,
            Command::CatchUp { from: None },
            Command::Resync,
            Command::Rejoin,
            Command::CheckDisk,
            Command::ReleaseLeases,
            Command::SetQuorum { max_members: 1 },
            Command::CertifyAnyway {
                build: "1-a".into(),
            },
            Command::RejectBuild {
                build: "1-a".into(),
            },
            Command::PauseRollout,
            Command::ResumeRollout,
            Command::Rollout { plan },
            Command::Rollback { build: None },
            Command::PruneBuilds {
                builds: vec![],
                keep: Some(1),
            },
            Command::UpgradeNow,
            Command::Offline,
            Command::Online,
        ]
    }

    /// **Every command is classed, by name and by value alike**, and the name
    /// list is exactly the commands that parse.
    #[test]
    fn every_command_is_classed_the_same_by_name_and_by_value() {
        let all = every_command();
        assert_eq!(all.len(), COMMANDS.len());
        for command in &all {
            assert_eq!(
                class_of(command.name()),
                Some(command.class()),
                "{}",
                command.name()
            );
        }
        for (name, _) in COMMANDS {
            let refused = parse(name, &Body::default(), "m1")
                .err()
                .unwrap_or_default();
            assert!(!refused.starts_with("there is no command"), "{name}");
        }
        assert_eq!(class_of("frobnicate"), None);
    }

    /// The classes §8.2 gives, spot-checked from each.
    #[test]
    fn the_classes_are_the_designs() {
        assert_eq!(class_of("audit"), Some(Class::Look));
        assert_eq!(class_of("drain"), Some(Class::Care));
        assert_eq!(class_of("rollout"), Some(Class::Decide));
        assert_eq!(class_of("shutdown"), Some(Class::Decide));
        assert_eq!(class_of("online"), Some(Class::OnTheMachine));
    }
}
