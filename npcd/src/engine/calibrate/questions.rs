//! The ways an act is asked for, beyond its examples' situations.
//!
//! Routing happens on the question: what a turn opens on — a message from a
//! person, an entry kept from a conversation, the mind's own watch — is all
//! there is to rank the acts by before anything is decoded. An example gives
//! one situation; these are the many plain phrasings the same act answers,
//! each written the way it actually arrives.
//!
//! Keyed by the act's name. An act with no entry is asked for by its examples'
//! situations alone.

/// The questions written for `act`.
pub fn of(act: &str) -> &'static [&'static str] {
    ASKED
        .iter()
        .find(|(name, _)| *name == act)
        .map_or(&[], |(_, questions)| questions)
}

/// Every act that has questions written for it, with them.
pub const ASKED: &[(&str, &[&str])] = &[
    (
        "terminal_list",
        &[
            "what can you reach from where you are",
            "which terminals do you have",
            "what is there for you to look through right now",
            "is the cluster one of the things you can get to",
            "can you see the house lights from there",
            "what are you able to connect to",
            "list the things you can work",
            "you want to check something outside yourself and do not know what answers",
            "which of your terminals are answering",
            "is the television reachable",
            "what machines can you get onto",
            "show me your terminals",
        ],
    ),
    (
        "terminal_connect",
        &[
            "can you check the cluster for me",
            "are all the machines up and serving right now",
            "is every member of the cluster up, and what build is each one on",
            "what is the status of the home and the machines",
            "connect to the cluster",
            "have a look at the lights",
            "is the television on",
            "go and see whether whiteworld is healthy",
            "John asked you to check the cluster and you said you would",
            "you promised to look at the machines and have not connected to anything yet",
            "you are connected to nothing and somebody wants to know how the cluster stands",
            "get onto blackworld and see what it is doing",
            "what is on the house network tonight",
            "turn the front room lights off",
            "put something on the television",
            "you want to work a thing and it is not the terminal you are connected to",
        ],
    ),
    (
        "survey_network",
        &[
            "what devices are on the house network right now",
            "scan the house network",
            "survey the network and tell me what answers",
            "are there any lights, speakers or a television on the network",
            "what is connected to the wifi at the moment",
            "has anything new joined the network",
            "you are connected to the house network and said you would see what is on it",
            "which devices in the house can you work",
            "find every device in the house",
            "is my phone on the network",
        ],
    ),
    (
        "terminal_disconnect",
        &[
            "you have read what you came for and have nothing more to do here",
            "let go of the cluster",
            "disconnect from the television",
            "you are done with the lights",
            "stop holding that machine",
            "you finished the check and reported it",
            "close the connection you have open",
            "there is nothing more to do at this terminal",
        ],
    ),
    (
        "cluster_status",
        &[
            "are all the machines up and serving right now",
            "is every member up, and what build is each one on",
            "how does the cluster stand",
            "is anything in fault",
            "did the new build reach whiteworld",
            "which members are ready",
            "is blackworld back in the cluster",
            "what build is the cluster on",
            "has every member taken the rollout",
            "is the cluster healthy",
            "who is in the cluster right now",
            "is any member behind the others",
            "you are connected to the cluster and were asked whether everything is up",
            "check the cluster for me",
            "what is the floor build and is everyone on it",
            "did a member drop out overnight",
        ],
    ),
    (
        "cluster_services",
        &[
            "what is each machine's keep running",
            "did the deploy take everywhere",
            "is motherd running on every machine",
            "which build of the gateway is installed on whiteworld",
            "how did the last trial end",
            "is a trial running anywhere right now",
            "what services are held down",
            "is the web gateway up on each machine",
            "which machines are still on the old binary",
            "did keep roll the last deploy back",
        ],
    ),
    (
        "cluster_logs",
        &[
            "why did that member restart",
            "search the logs for a panic",
            "what do the logs say about the out of memory",
            "find every error in the last hour",
            "has anything logged a warning about the substrate",
            "look for NaN in the cluster's logs",
            "when did whiteworld last say it was serving",
            "what went wrong with the deploy, according to the logs",
            "grep the logs for the job that failed",
            "is anything complaining about the disk",
        ],
    ),
    (
        "cluster_member",
        &[
            "tell me everything about whiteworld's member",
            "is that member really joined",
            "what has blackworld not sent yet",
            "which apps does that node host",
            "where is that member in its upgrade",
            "the summary says ready and you want the member's own word",
            "read one member's status in full",
            "is that node a candidate or certified",
        ],
    ),
    (
        "cluster_memory",
        &[
            "is that machine's card full",
            "how much memory is whiteworld using",
            "what is holding the VRAM on blackworld",
            "is a member running out of RAM",
            "why has that member slowed down",
            "how full is the KV on that node",
            "how much room is left on the card",
            "is the machine paging",
        ],
    ),
    (
        "cluster_audit",
        &[
            "who rolled that build out",
            "what was done to the cluster today",
            "did somebody restart a member on purpose",
            "who sent the last command",
            "was a member refused when it tried to join",
            "what changed in the cluster since yesterday",
            "who certified the candidate",
            "has anybody run a command I do not know about",
        ],
    ),
    (
        "cluster_log",
        &[
            "what is the cluster saying right now",
            "show me the newest lines of the log",
            "read the tail of the cluster's log",
            "what did the daemons just write",
            "what is the site's log saying",
            "give me the last lines, I have nothing to search for",
            "what happened in the log just now",
            "is the log quiet",
        ],
    ),
    (
        "cluster_jobs",
        &[
            "what is running on the cluster",
            "is a build going right now",
            "which jobs are in flight",
            "what is blackworld busy with",
            "is anybody running tests",
            "are there deploys under way",
            "what has been running the longest",
            "is the machine free to start something",
        ],
    ),
    (
        "cluster_job",
        &[
            "how did that test run end",
            "is the build finished",
            "did the job pass",
            "where is the deploy's trial up to",
            "what were the first failures",
            "has the task answered yet",
            "you started a job and want to know how it went",
            "is that job still running",
        ],
    ),
    (
        "cluster_job_read",
        &[
            "show me the build's output",
            "read the page the failure is on",
            "what did the test print when it failed",
            "read the next page of that job's log",
            "what does the deploy's trial say",
            "show the task's entries",
            "read the job's log from the start",
            "what is on page two",
        ],
    ),
    (
        "cluster_job_stop",
        &[
            "stop that test run",
            "cancel the build you started",
            "kill that job, it is the wrong one",
            "that run is holding the machine, end it",
            "abort the job",
            "you started the wrong tests",
            "stop the job before it finishes",
            "free the machine up",
        ],
    ),
    (
        "cluster_job_start",
        &[
            "do the web crate's tests still pass on main",
            "run the npcd tests",
            "does candle build at the moment",
            "run clippy over the workspace",
            "check that main compiles",
            "kick off a release build",
            "run the latency gate",
            "is the formatting clean on main",
            "start a test run for zend",
            "find out whether the last commit broke the build",
        ],
    ),
    (
        "machine_check",
        &[
            "is anything listening on that machine that was not yesterday",
            "is the machine healthy",
            "what is using the disk",
            "what does the machine's system log say about the disk",
            "has anything changed on this machine since the last look",
            "which processes are running on it",
            "is the card hot",
            "run the health check",
            "who is logged on to that machine",
            "you are connected to a machine and were asked how it is doing",
        ],
    ),
];

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{of, ASKED};
    use crate::engine::tools;

    #[test]
    fn every_act_asked_for_is_in_the_catalog() {
        for (name, _) in ASKED {
            assert!(tools::by_name(name).is_some(), "`{name}` is no act");
        }
    }

    #[test]
    fn an_act_is_listed_once_and_asks_nothing_twice() {
        let mut names = HashSet::new();
        for (name, questions) in ASKED {
            assert!(names.insert(*name), "`{name}` is listed twice");
            let distinct: HashSet<&&str> = questions.iter().collect();
            assert_eq!(
                distinct.len(),
                questions.len(),
                "`{name}` repeats a question"
            );
            assert!(questions.len() >= 8, "`{name}` asks too few ways");
        }
    }

    #[test]
    fn an_act_with_no_entry_asks_nothing_of_its_own() {
        assert!(of("reflect").is_empty());
        assert_eq!(of("terminal_list").len(), 12);
    }
}
