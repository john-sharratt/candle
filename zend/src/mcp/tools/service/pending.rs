//! One deploy of a service on a machine for one commit
//! (`docs/battlecities_app.md` §4.3).
//!
//! A deploy asked for while another of the same service is under way on the
//! same machine is answered from what the machine's agent and keep already
//! hold, so any member answers the same:
//!
//! | What is under way | The commit asked for | The answer |
//! |---|---|---|
//! | a build of it, of that commit | the same | [`Ask::Watch`] it |
//! | a build of it, still fetching origin | any | [`Ask::Watch`] it: it will build what origin holds |
//! | a build of it, of another commit | a newer one | [`Ask::Replace`] it: cancel, and build again |
//! | a trial of it, of that commit | the same | [`Ask::Watch`] it |
//! | a trial of it, of another commit | a newer one | [`Ask::Start`]: the trial is left to end, and the new build follows it |
//! | nothing | any | [`Ask::Start`] |
//!
//! A build is known by its command ([`built_service`]) and is of the commit
//! its agent recorded once it had fetched origin (`JobLine::base`). An agent
//! that records none names no commit, and a deploy beside such a build starts
//! as it always did.

use agent::wire::{JobKind, JobLine, Phase};
use serde_json::Value;

use super::{built_service, trials_of};

/// The branch a deploy builds.
const BRANCH: &str = "main";

/// What a request for a deploy is answered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ask {
    /// Nothing of it is under way: build it.
    Start,
    /// The deploy whose build job is this is already of the commit: watch it.
    Watch(String),
    /// The build job named is of an older commit: cancel it, then build.
    Replace(String),
}

/// The newest build of `service` still under way among `jobs`, an agent's
/// jobs newest first.
pub fn building<'a>(jobs: &'a [JobLine], service: &str) -> Option<&'a JobLine> {
    jobs.iter().find(|job| {
        job.kind == JobKind::Run
            && !job.status.is_finished()
            && job.branch.as_deref() == Some(BRANCH)
            && built_service(&job.command) == Some(service)
    })
}

/// The build job of the deploy keep is trying as `service`, from keep's
/// `state`: the id after the node in the handle the trial records.
pub fn trying(state: &Value, service: &str) -> Option<String> {
    let handle = trials_of(state, service)?
        .get("running")?
        .get("deploy")?
        .as_str()?;
    handle.split_once(':').map(|(_, id)| id.to_string())
}

/// What a request to deploy `service` at commit `tip` is answered with, given
/// the agent's `jobs` (newest first) and the build job of the trial keep is
/// running for it (`trying`).
pub fn decide(jobs: &[JobLine], service: &str, trying: Option<&str>, tip: &str) -> Ask {
    if let Some(job) = building(jobs, service) {
        return match job.base.as_deref() {
            Some(base) if base == tip => Ask::Watch(job.id.clone()),
            Some(_) => Ask::Replace(job.id.clone()),
            None if job.phase == Phase::Fetching => Ask::Watch(job.id.clone()),
            None => Ask::Start,
        };
    }
    let tried = trying.and_then(|id| jobs.iter().find(|job| job.id == id));
    match tried {
        Some(job) if job.base.as_deref() == Some(tip) => Ask::Watch(job.id.clone()),
        _ => Ask::Start,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use zend_vfs::JobStatus;

    use super::*;

    const OLD: &str = "1111111111111111111111111111111111111111";
    const NEW: &str = "2222222222222222222222222222222222222222";

    fn job(id: &str, command: &str, base: Option<&str>, phase: Phase, running: bool) -> JobLine {
        JobLine {
            id: id.into(),
            repo: "battle-cities".into(),
            kind: JobKind::Run,
            branch: Some("main".into()),
            base: base.map(str::to_string),
            command: command.split(' ').map(str::to_string).collect(),
            status: match running {
                true => JobStatus::Running,
                false => JobStatus::Exited { code: Some(0) },
            },
            lines: 0,
            bytes: 0,
            started_ms: 0,
            slot_ms: None,
            ended_ms: None,
            changed: 0,
            phase,
        }
    }

    const GAME: &str = "cargo build --release -p battle-cities-server";

    /// **Asking again for the commit being built is answered with that
    /// deploy**, whoever asked first; a build that has not fetched yet will
    /// build what origin holds, so it is the same deploy too.
    #[test]
    fn a_build_of_the_same_commit_is_watched() {
        let jobs = [job("Bd", GAME, Some(NEW), Phase::Running, true)];
        assert_eq!(
            decide(&jobs, "battlecities", None, NEW),
            Ask::Watch("Bd".into())
        );
        let fetching = [job("Ft", GAME, None, Phase::Fetching, true)];
        assert_eq!(
            decide(&fetching, "battlecities", None, NEW),
            Ask::Watch("Ft".into())
        );
    }

    /// **A newer commit replaces a build of an older one.**
    #[test]
    fn a_build_of_an_older_commit_is_replaced() {
        let jobs = [job("Bd", GAME, Some(OLD), Phase::Running, true)];
        assert_eq!(
            decide(&jobs, "battlecities", None, NEW),
            Ask::Replace("Bd".into())
        );
        let waiting = [job(
            "Wt",
            GAME,
            Some(OLD),
            Phase::Waiting {
                behind: vec!["Xx".into()],
            },
            true,
        )];
        assert_eq!(
            decide(&waiting, "battlecities", None, NEW),
            Ask::Replace("Wt".into())
        );
    }

    /// **A deploy that has reached its trial is left to finish**: the same
    /// commit watches it, a newer one starts and follows it.
    #[test]
    fn a_trial_is_watched_for_its_commit_and_never_replaced() {
        let jobs = [job("Tr", GAME, Some(OLD), Phase::Ended, false)];
        assert_eq!(
            decide(&jobs, "battlecities", Some("Tr"), OLD),
            Ask::Watch("Tr".into())
        );
        assert_eq!(decide(&jobs, "battlecities", Some("Tr"), NEW), Ask::Start);
        // A build that ended with no trial running is no deploy under way.
        assert_eq!(decide(&jobs, "battlecities", None, OLD), Ask::Start);
        // A trial whose build job the agent no longer holds names no commit.
        assert_eq!(decide(&[], "battlecities", Some("Tr"), OLD), Ask::Start);
    }

    /// Another service's build, a build on another branch, a test run and a
    /// build whose agent recorded no commit are not this deploy.
    #[test]
    fn only_a_build_of_the_service_on_main_is_its_deploy() {
        let web = job(
            "Wb",
            "cargo build --release -p web",
            Some(NEW),
            Phase::Running,
            true,
        );
        let mut branch = job("Br", GAME, Some(NEW), Phase::Running, true);
        branch.branch = Some("zen/x".into());
        let test = job(
            "Ts",
            "cargo test -p battle-cities-server",
            Some(NEW),
            Phase::Running,
            true,
        );
        let unnamed = job("Un", GAME, None, Phase::Running, true);
        let jobs = [web.clone(), branch, test, unnamed];
        assert_eq!(building(&jobs, "web").map(|j| j.id.as_str()), Some("Wb"));
        assert_eq!(
            building(&jobs, "battlecities").map(|j| j.id.as_str()),
            Some("Un")
        );
        assert_eq!(decide(&jobs, "battlecities", None, NEW), Ask::Start);
        assert_eq!(decide(&[web], "battlecities", None, NEW), Ask::Start);
    }

    #[test]
    fn the_trial_keep_runs_names_its_build_job() {
        let state = json!({"services": [
            {"name": "battlecities", "trials": {"last": null,
                "running": {"artifact": "x", "deploy": "00000000000031f6:0lt"}}},
            {"name": "web", "trials": {"running": null, "last": {"deploy": "00000000000031f6:Old"}}},
        ]});
        assert_eq!(trying(&state, "battlecities"), Some("0lt".to_string()));
        assert_eq!(trying(&state, "web"), None);
        assert_eq!(trying(&state, "mcp"), None);
    }
}
