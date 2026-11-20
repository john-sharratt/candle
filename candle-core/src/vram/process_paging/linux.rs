//! `/proc/<pid>/stat` and `/proc/<pid>/status`.
//!
//! `majflt` is the twelfth field of `stat`, counted after the command name,
//! which is parenthesised and may itself hold spaces and parentheses — so the
//! fields are split after the last `)`.

use super::ProcessPaging;

pub(super) fn own() -> Option<ProcessPaging> {
    read(std::process::id())
}

pub(super) fn all() -> Vec<ProcessPaging> {
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    dir.filter_map(|e| e.ok()?.file_name().to_str()?.parse::<u32>().ok())
        .filter_map(read)
        .collect()
}

fn read(pid: u32) -> Option<ProcessPaging> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let (name, hard_faults) = parse_stat(&stat)?;
    let kb = |key: &str| status_kb(&status, key).unwrap_or(0) * 1024;
    Some(ProcessPaging {
        pid,
        name,
        hard_faults,
        working_set: kb("VmRSS:"),
        private_working_set: kb("RssAnon:"),
        // Resident anonymous pages plus the swapped ones: what it has committed
        // of its own.
        private_bytes: kb("RssAnon:") + kb("VmSwap:"),
    })
}

/// The command name and `majflt` of a `stat` line.
pub(super) fn parse_stat(stat: &str) -> Option<(String, u64)> {
    let open = stat.find('(')?;
    let close = stat.rfind(')')?;
    let name = stat.get(open + 1..close)?.to_string();
    // After the name: state, ppid, pgrp, session, tty_nr, tpgid, flags,
    // minflt, cminflt, majflt.
    let majflt = stat
        .get(close + 1..)?
        .split_whitespace()
        .nth(9)?
        .parse()
        .ok()?;
    Some((name, majflt))
}

fn status_kb(status: &str, key: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with(key))?;
    line.split_whitespace().nth(1)?.parse().ok()
}
