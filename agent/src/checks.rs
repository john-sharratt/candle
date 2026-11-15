//! A machine's checks (`docs/zen_everywhere.md` §5): named, read-only looks
//! at the machine this agent runs on, each declared by a file in the agent's
//! `checks/` folder and run on a member's word.
//!
//! **Named checks, never a shell.** Whoever asks reads logs, and logs carry
//! text from outside; a command line would let what is read decide what is
//! run. A check is a program with a fixed argument list. An argument a caller
//! gives is validated against what the check declares — one of its listed
//! values, or a short run of plain characters — and takes the place of one
//! whole entry of that list: it is never spliced into a string a shell or an
//! interpreter parses.
//!
//! **The list is the machine's own, and it is open.** Each `*.yaml` in the
//! folder is one check, read afresh on every call, so adding a check is adding
//! a file. The first files ([`DEFAULTS`]) are written when the folder does not
//! exist yet and never touched after: who can write to the machine edits or
//! adds to them.
//!
//! **A check says what changed.** The agent keeps each check's last answer and
//! leads the report with the lines that came or went since. A check whose
//! output has a part that moves every time — the busiest processes, a load
//! figure — prints a line of `---` before it: what follows is reported and not
//! compared.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::process::Command;

/// The folder under the agent's own that holds its checks.
pub const FOLDER: &str = "checks";

/// The folder under [`FOLDER`] that keeps each check's last answer.
const LAST: &str = ".last";

/// The line that parts what is compared from what only moves.
pub const MOVES: &str = "---";

/// The most of a report that is handed back.
pub const REPORT_CHARS: usize = 6_000;

/// The most lines of change a report leads with.
const CHANGES_SHOWN: usize = 40;

/// How long a check runs when it does not say.
const DEFAULT_TIMEOUT_S: u64 = 30;

/// The longest any check may run.
const MAX_TIMEOUT_S: u64 = 120;

/// The longest free argument a check may declare.
const MAX_FREE: usize = 120;

/// The first files: each check's declaration and the script it runs, written
/// into a machine's `checks/` folder when it has none.
pub const DEFAULTS: &[(&str, &str)] = &[
    ("health.yaml", include_str!("../checks/health.yaml")),
    ("health.ps1", include_str!("../checks/health.ps1")),
    ("gpu.yaml", include_str!("../checks/gpu.yaml")),
    ("gpu.ps1", include_str!("../checks/gpu.ps1")),
    ("services.yaml", include_str!("../checks/services.yaml")),
    ("services.ps1", include_str!("../checks/services.ps1")),
    ("errors.yaml", include_str!("../checks/errors.yaml")),
    ("errors.ps1", include_str!("../checks/errors.ps1")),
    ("logs.yaml", include_str!("../checks/logs.yaml")),
    ("logs.ps1", include_str!("../checks/logs.ps1")),
    ("logins.yaml", include_str!("../checks/logins.yaml")),
    ("logins.ps1", include_str!("../checks/logins.ps1")),
    ("listeners.yaml", include_str!("../checks/listeners.yaml")),
    ("listeners.ps1", include_str!("../checks/listeners.ps1")),
    ("processes.yaml", include_str!("../checks/processes.yaml")),
    ("processes.ps1", include_str!("../checks/processes.ps1")),
    ("security.yaml", include_str!("../checks/security.yaml")),
    ("security.ps1", include_str!("../checks/security.ps1")),
    ("storage.yaml", include_str!("../checks/storage.yaml")),
    ("storage.ps1", include_str!("../checks/storage.ps1")),
    ("storage.paths", include_str!("../checks/storage.paths")),
];

/// One check, as its file declares it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    /// What it is called: lowercase letters, digits and hyphens.
    pub name: String,
    /// What it reports, as whoever chooses it reads it.
    pub says: String,
    /// The arguments it takes.
    #[serde(default)]
    pub args: Vec<Arg>,
    /// The program it runs.
    pub program: String,
    /// The program's arguments. An entry that is `{name}` whole is replaced by
    /// the validated argument of that name; one beginning `{dir}/` names a
    /// file in the checks folder.
    #[serde(default)]
    pub argv: Vec<String>,
    /// How long it may run, in seconds.
    #[serde(default)]
    pub timeout_s: Option<u64>,
}

/// One argument of a check.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arg {
    pub name: String,
    /// What it is, as whoever gives it reads it.
    #[serde(default)]
    pub says: String,
    /// The values it may take. Empty for a free argument ([`Self::free`]).
    #[serde(default)]
    pub values: Vec<String>,
    /// For an argument with no listed values: the most characters it may run
    /// to, each a letter, a digit, a space or one of `_ . : -`.
    #[serde(default)]
    pub free: Option<usize>,
    /// What it is when the caller gives none. An argument with no default
    /// must be given.
    #[serde(default)]
    pub default: Option<String>,
}

/// A check as a caller is shown it: its name, what it reports, and what each
/// argument may be.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckLine {
    pub name: String,
    pub says: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<ArgLine>,
}

/// One argument of a [`CheckLine`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgLine {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub says: String,
    /// The values it may take; empty for free text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
    /// Whether it must be given.
    #[serde(default)]
    pub required: bool,
}

impl Check {
    fn line(&self) -> CheckLine {
        CheckLine {
            name: self.name.clone(),
            says: self.says.clone(),
            args: self
                .args
                .iter()
                .map(|a| ArgLine {
                    name: a.name.clone(),
                    says: a.says.clone(),
                    values: a.values.clone(),
                    required: a.default.is_none(),
                })
                .collect(),
        }
    }

    /// Why this declaration cannot be run as written, if it cannot.
    fn fault(&self) -> Option<String> {
        if !is_name(&self.name) {
            return Some(format!(
                "its name `{}` is not lowercase letters, digits and hyphens",
                self.name
            ));
        }
        if self.program.trim().is_empty() {
            return Some("it names no program".into());
        }
        for a in &self.args {
            if !is_name(&a.name) {
                return Some(format!("its argument `{}` is not a plain name", a.name));
            }
            match (a.values.is_empty(), a.free) {
                (true, None) => {
                    return Some(format!(
                        "its argument `{}` lists no values and is not free",
                        a.name
                    ))
                }
                (true, Some(n)) if n == 0 || n > MAX_FREE => {
                    return Some(format!(
                        "its argument `{}` may be free for 1 to {MAX_FREE} characters",
                        a.name
                    ))
                }
                _ => {}
            }
            if let Some(d) = &a.default {
                if let Err(why) = a.take(d) {
                    return Some(format!("its argument `{}`'s default: {why}", a.name));
                }
            }
        }
        for entry in &self.argv {
            if let Some(name) = placeholder(entry) {
                if name != "dir" && !self.args.iter().any(|a| a.name == name) {
                    return Some(format!(
                        "its argv names `{{{name}}}`, which it does not take"
                    ));
                }
            }
        }
        None
    }

    /// The arguments it runs with, from what the caller `given`: each
    /// validated, a missing one its default.
    fn taken(&self, given: &BTreeMap<String, String>) -> Result<BTreeMap<String, String>, String> {
        if let Some(extra) = given
            .keys()
            .find(|k| !self.args.iter().any(|a| &a.name == *k))
        {
            return Err(format!("`{}` takes no `{extra}`.", self.name));
        }
        let mut out = BTreeMap::new();
        for a in &self.args {
            let value = match given
                .get(&a.name)
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
            {
                Some(v) => a.take(v)?,
                None => a.default.clone().ok_or_else(|| a.wanted())?,
            };
            out.insert(a.name.clone(), value);
        }
        Ok(out)
    }

    /// The program's arguments with `taken` in their places.
    fn argv_with(&self, dir: &Path, taken: &BTreeMap<String, String>) -> Vec<String> {
        self.argv
            .iter()
            .map(|entry| {
                if let Some(file) = entry.strip_prefix("{dir}/") {
                    return dir.join(file).to_string_lossy().into_owned();
                }
                match placeholder(entry).and_then(|name| taken.get(name)) {
                    Some(value) => value.clone(),
                    None => entry.clone(),
                }
            })
            .collect()
    }
}

impl Arg {
    /// `value` as this argument takes it, or why it does not.
    fn take(&self, value: &str) -> Result<String, String> {
        if !self.values.is_empty() {
            return self
                .values
                .iter()
                .find(|v| v.eq_ignore_ascii_case(value))
                .cloned()
                .ok_or_else(|| self.wanted());
        }
        let max = self.free.unwrap_or(0);
        let plain = value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '.' | ':' | '-'));
        match plain && value.chars().count() <= max {
            true => Ok(value.to_string()),
            false => Err(self.wanted()),
        }
    }

    /// What this argument has to be, said to whoever got it wrong.
    fn wanted(&self) -> String {
        match self.values.is_empty() {
            false => format!("`{}` is one of: {}.", self.name, self.values.join(", ")),
            true => format!(
                "`{}` is up to {} characters: letters, digits, spaces and _ . : -",
                self.name,
                self.free.unwrap_or(0)
            ),
        }
    }
}

fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 40
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// The name an argv entry stands for when the whole of it is `{name}`.
fn placeholder(entry: &str) -> Option<&str> {
    let name = entry.strip_prefix('{')?.strip_suffix('}')?;
    (!name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        .then_some(name)
}

/// The checks folder of an agent whose own folder is `agent_dir`.
pub fn folder(agent_dir: &Path) -> PathBuf {
    agent_dir.join(FOLDER)
}

/// Write the first files into `dir` when it does not exist yet. A folder that
/// exists is its machine's, and is left as it is.
pub fn ensure_defaults(dir: &Path) -> std::io::Result<()> {
    if dir.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    for (name, text) in DEFAULTS {
        std::fs::write(dir.join(name), text)?;
    }
    Ok(())
}

/// Every check `dir` declares, by name, and each file that could not be read
/// as one with why.
pub fn load(dir: &Path) -> (Vec<Check>, Vec<String>) {
    let mut checks: Vec<Check> = Vec::new();
    let mut faults = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (checks, faults);
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yaml" || x == "yml"))
        .collect();
    files.sort();
    for path in files {
        let file = path.file_name().unwrap_or_default().to_string_lossy();
        let parsed = std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_yaml::from_str::<Check>(&text).map_err(|e| e.to_string()));
        match parsed {
            Err(why) => faults.push(format!("{file}: {why}")),
            Ok(check) => match check.fault() {
                Some(why) => faults.push(format!("{file}: {why}")),
                None if checks.iter().any(|c| c.name == check.name) => {
                    faults.push(format!("{file}: `{}` is declared twice", check.name))
                }
                None => checks.push(check),
            },
        }
    }
    (checks, faults)
}

/// The checks of the agent whose own folder is `agent_dir`, as a caller is
/// shown them.
pub fn listed(agent_dir: &Path) -> Vec<CheckLine> {
    let dir = folder(agent_dir);
    if let Err(e) = ensure_defaults(&dir) {
        tracing::warn!(
            "the first checks could not be written to {}: {e}",
            dir.display()
        );
    }
    let (checks, faults) = load(&dir);
    for fault in faults {
        tracing::warn!("a check in {} is not offered: {fault}", dir.display());
    }
    checks.iter().map(Check::line).collect()
}

/// Run check `name` of the agent whose own folder is `agent_dir` with the
/// arguments `given`: its report, leading with what changed since it last ran
/// — or why it was not run.
pub async fn run(
    agent_dir: &Path,
    name: &str,
    given: &BTreeMap<String, String>,
) -> Result<String, String> {
    let dir = folder(agent_dir);
    ensure_defaults(&dir).map_err(|e| format!("This machine's checks could not be read: {e}"))?;
    let (checks, _) = load(&dir);
    let Some(check) = checks.iter().find(|c| c.name.eq_ignore_ascii_case(name)) else {
        let names: Vec<&str> = checks.iter().map(|c| c.name.as_str()).collect();
        return Err(format!(
            "This machine has no check called `{name}`. Its checks: {}.",
            names.join(", ")
        ));
    };
    let taken = check.taken(given)?;
    let wait = Duration::from_secs(
        check
            .timeout_s
            .unwrap_or(DEFAULT_TIMEOUT_S)
            .clamp(1, MAX_TIMEOUT_S),
    );
    let mut command = Command::new(&check.program);
    command
        .args(check.argv_with(&dir, &taken))
        .current_dir(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = command.spawn().map_err(|e| {
        format!(
            "`{}` could not be started ({}): {e}",
            check.name, check.program
        )
    })?;
    let output = match tokio::time::timeout(wait, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => return Err(format!("`{}` did not finish: {e}", check.name)),
        Err(_) => {
            return Err(format!(
                "`{}` was stopped: it ran past its {} seconds.",
                check.name,
                wait.as_secs()
            ))
        }
    };
    let said = String::from_utf8_lossy(&output.stdout).replace('\r', "");
    if !output.status.success() {
        let why = String::from_utf8_lossy(&output.stderr).replace('\r', "");
        return Err(format!(
            "`{}` failed ({}): {}",
            check.name,
            output.status,
            cut(why.trim(), REPORT_CHARS)
        ));
    }
    let key = last_key(&check.name, &taken);
    let before = read_last(&dir, &key);
    let (steady, moves) = parts(&said);
    let changes = changed(before.as_deref(), steady);
    if let Err(e) = write_last(&dir, &key, steady) {
        tracing::warn!(
            "`{}`'s answer could not be kept for the next check: {e}",
            check.name
        );
    }
    Ok(report(&changes, steady, moves))
}

/// A check's report as it is handed back: what changed, then what moves every
/// time — the figures asked for now — then what is compared from run to run,
/// which is the part cut when the whole runs long: a long inventory's news is
/// in the changes above it.
fn report(changes: &str, steady: &str, moves: &str) -> String {
    let mut out = changes.trim_end().to_string();
    for part in [moves, steady] {
        let part = part.trim();
        if !part.is_empty() {
            out.push_str("\n\n");
            out.push_str(part);
        }
    }
    cut(&out, REPORT_CHARS)
}

/// A report as what is compared from run to run and what only moves: the
/// lines before the first [`MOVES`] line, and those after it.
fn parts(said: &str) -> (&str, &str) {
    let mut at = 0;
    for line in said.split_inclusive('\n') {
        if line.trim() == MOVES {
            return (&said[..at], &said[at + line.len()..]);
        }
        at += line.len();
    }
    (said, "")
}

/// What changed between the last answer's compared part, `before`, and this
/// one's, `now`, as the lines the report leads with.
pub fn changed(before: Option<&str>, now: &str) -> String {
    let Some(before) = before else {
        return "This is the first time this check has run here: there is nothing to compare it \
                with."
            .into();
    };
    let lines = |text: &str| -> BTreeSet<String> {
        text.lines()
            .map(|l| l.trim_end().to_string())
            .filter(|l| !l.trim().is_empty())
            .collect()
    };
    let (was, is) = (lines(before), lines(now));
    let came: Vec<&String> = is.difference(&was).collect();
    let went: Vec<&String> = was.difference(&is).collect();
    if came.is_empty() && went.is_empty() {
        return "Nothing changed since the last check.".into();
    }
    let mut out = vec![format!(
        "Changed since the last check — {} new, {} gone:",
        came.len(),
        went.len()
    )];
    let shown = came
        .iter()
        .map(|l| format!("+ {l}"))
        .chain(went.iter().map(|l| format!("- {l}")));
    out.extend(shown.take(CHANGES_SHOWN));
    let total = came.len() + went.len();
    if total > CHANGES_SHOWN {
        out.push(format!("… and {} more.", total - CHANGES_SHOWN));
    }
    out.join("\n")
}

/// The name a check's last answer is kept under: its own, and a digest of the
/// arguments it ran with, so `logs` of one log is not compared with another's.
fn last_key(name: &str, taken: &BTreeMap<String, String>) -> String {
    if taken.is_empty() {
        return name.to_string();
    }
    let mut h = Sha256::new();
    for (k, v) in taken {
        h.update(k.as_bytes());
        h.update([0]);
        h.update(v.as_bytes());
        h.update([0]);
    }
    let digest: String = h
        .finalize()
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("{name}-{digest}")
}

fn read_last(dir: &Path, key: &str) -> Option<String> {
    std::fs::read_to_string(dir.join(LAST).join(format!("{key}.txt"))).ok()
}

fn write_last(dir: &Path, key: &str, steady: &str) -> std::io::Result<()> {
    let last = dir.join(LAST);
    std::fs::create_dir_all(&last)?;
    std::fs::write(last.join(format!("{key}.txt")), steady)
}

/// `text`, cut at `max` characters on a character.
fn cut(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((at, _)) => format!("{}\n… (cut: the report runs on)", &text[..at]),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(yaml: &str) -> Check {
        serde_yaml::from_str(yaml).unwrap()
    }

    fn given(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    const LOGS: &str = "name: logs\nsays: A log searched.\nargs:\n  - name: log\n    values: \
                        [System, Application]\n  - name: grep\n    free: 20\n    default: \"\"\n  \
                        - name: since\n    values: [1h, 24h]\n    default: 24h\nprogram: \
                        powershell\nargv: [\"-File\", \"{dir}/logs.ps1\", \"{log}\", \"{grep}\", \
                        \"{since}\"]\n";

    /// **Every first file is a check that can be offered**, each with the
    /// script its declaration names.
    #[test]
    fn the_first_files_are_checks_that_can_be_offered() {
        let dir = tempfile::tempdir().unwrap();
        let checks = dir.path().join(FOLDER);
        ensure_defaults(&checks).unwrap();
        let (loaded, faults) = load(&checks);
        assert_eq!(faults, Vec::<String>::new());
        let names: Vec<&str> = loaded.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "errors",
                "gpu",
                "health",
                "listeners",
                "logins",
                "logs",
                "processes",
                "security",
                "services",
                "storage"
            ]
        );
        for c in &loaded {
            for entry in &c.argv {
                if let Some(file) = entry.strip_prefix("{dir}/") {
                    assert!(checks.join(file).is_file(), "{} names {file}", c.name);
                }
            }
        }
    }

    /// **Every first check runs on a Windows machine and reports something**,
    /// twice: the second run has the first to compare with. Run by hand on a
    /// machine the checks are for — it takes the machine's own PowerShell and
    /// most of a minute — with `cargo test -p agent --lib checks -- --ignored
    /// --nocapture`.
    #[cfg(windows)]
    #[tokio::test]
    #[ignore = "runs the real checks against this machine"]
    async fn every_first_check_runs_on_this_machine() {
        let dir = tempfile::tempdir().unwrap();
        for line in listed(dir.path()) {
            let args = match line.name.as_str() {
                "logs" => given(&[("log", "System"), ("grep", "error"), ("since", "1h")]),
                _ => BTreeMap::new(),
            };
            let first = run(dir.path(), &line.name, &args)
                .await
                .unwrap_or_else(|why| panic!("{} did not run: {why}", line.name));
            println!("==== {} ====\n{first}\n", line.name);
            assert!(first.starts_with("This is the first time"), "{first}");
            assert!(first.lines().count() > 2, "{} reported nothing", line.name);
            let again = run(dir.path(), &line.name, &args).await.unwrap();
            assert!(
                again.starts_with("Nothing changed") || again.starts_with("Changed since"),
                "{again}"
            );
        }
    }

    /// **A folder that exists is its machine's**: the first files are not
    /// written over what is there, and a file added is a check offered.
    #[test]
    fn a_folder_that_exists_is_left_and_a_file_added_is_a_check() {
        let dir = tempfile::tempdir().unwrap();
        let checks = dir.path().join(FOLDER);
        std::fs::create_dir_all(&checks).unwrap();
        std::fs::write(
            checks.join("uptime.yaml"),
            "name: uptime\nsays: How long it has been up.\nprogram: uptime\n",
        )
        .unwrap();
        ensure_defaults(&checks).unwrap();
        let lines = listed(dir.path());
        assert_eq!(lines.len(), 1);
        assert_eq!(
            (lines[0].name.as_str(), lines[0].says.as_str()),
            ("uptime", "How long it has been up.")
        );
    }

    /// **A declaration that cannot be run safely is not offered**, and says
    /// why: a name that is not plain, an argument with nothing it may be, an
    /// argv naming an argument it does not take, a second of the same name.
    #[test]
    fn a_declaration_that_cannot_be_run_is_not_offered() {
        let dir = tempfile::tempdir().unwrap();
        for (file, text) in [
            ("a.yaml", "name: Bad Name\nsays: x\nprogram: x\n"),
            (
                "b.yaml",
                "name: open\nsays: x\nprogram: x\nargs:\n  - name: what\n",
            ),
            (
                "c.yaml",
                "name: stray\nsays: x\nprogram: x\nargv: [\"{other}\"]\n",
            ),
            ("d.yaml", "name: twice\nsays: x\nprogram: x\n"),
            ("e.yaml", "name: twice\nsays: x\nprogram: x\n"),
            ("f.yaml", "name: shell\nsays: x\nprogram: x\nshell: true\n"),
        ] {
            std::fs::write(dir.path().join(file), text).unwrap();
        }
        let (loaded, faults) = load(dir.path());
        assert_eq!(
            loaded.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["twice"]
        );
        assert_eq!(faults.len(), 5, "{faults:?}");
        assert!(faults[0].contains("lowercase letters"), "{faults:?}");
        assert!(faults[1].contains("lists no values"), "{faults:?}");
        assert!(faults[2].contains("does not take"), "{faults:?}");
        assert!(faults[3].contains("declared twice"), "{faults:?}");
        assert!(faults[4].contains("unknown field"), "{faults:?}");
    }

    /// **An argument is one of its values, or a short run of plain
    /// characters** — and takes the place of one whole entry of the program's
    /// arguments, so nothing given is ever parsed by a shell.
    #[test]
    fn an_argument_is_validated_and_placed_whole() {
        let c = check(LOGS);
        assert_eq!(c.fault(), None);
        let dir = Path::new("/agent/checks");
        let taken = c
            .taken(&given(&[("log", "system"), ("grep", "disk error")]))
            .unwrap();
        assert_eq!(
            taken,
            given(&[("grep", "disk error"), ("log", "System"), ("since", "24h")]),
            "a listed value in its own spelling, a missing one its default"
        );
        let argv = c.argv_with(dir, &taken);
        assert_eq!(argv[0], "-File");
        assert!(argv[1].ends_with("logs.ps1") && argv[1].contains("checks"));
        assert_eq!(&argv[2..], ["System", "disk error", "24h"]);

        for (args, why) in [
            (
                given(&[("log", "Setup")]),
                "`log` is one of: System, Application.",
            ),
            (given(&[]), "`log` is one of: System, Application."),
            (
                given(&[("log", "System"), ("grep", "x; rm -rf /")]),
                "`grep` is up to 20 characters: letters, digits, spaces and _ . : -",
            ),
            (
                given(&[("log", "System"), ("grep", "$(whoami)")]),
                "`grep` is up to 20 characters: letters, digits, spaces and _ . : -",
            ),
            (
                given(&[("log", "System"), ("grep", &"a".repeat(21))]),
                "`grep` is up to 20 characters: letters, digits, spaces and _ . : -",
            ),
            (
                given(&[("log", "System"), ("path", "C:")]),
                "`logs` takes no `path`.",
            ),
        ] {
            assert_eq!(c.taken(&args).unwrap_err(), why);
        }
    }

    /// **A report leads with what changed**: the lines that came and went in
    /// the part that is compared, and nothing of the part that only moves.
    #[test]
    fn a_report_leads_with_what_changed() {
        assert!(changed(None, "a\n").starts_with("This is the first time"));
        assert_eq!(
            changed(Some("a\nb\n"), "b\na\n"),
            "Nothing changed since the last check."
        );
        assert_eq!(
            changed(
                Some("port 80 web\nport 22 ssh\n"),
                "port 80 web\nport 4444 nc\n"
            ),
            "Changed since the last check — 1 new, 1 gone:\n+ port 4444 nc\n- port 22 ssh"
        );
        let (steady, moves) = parts("port 80\n---\ncpu 97%\n");
        assert_eq!((steady, moves), ("port 80\n", "cpu 97%\n"));
        assert_eq!(parts("port 80\n"), ("port 80\n", ""));
        assert_eq!(
            report("Nothing changed since the last check.", steady, moves),
            "Nothing changed since the last check.\n\ncpu 97%\n\nport 80",
            "what moves is read before the inventory, which is what a long report cuts"
        );
        let many: String = (0..60).map(|i| format!("line {i:02}\n")).collect();
        let said = changed(Some(""), &many);
        assert!(said.ends_with("… and 20 more."), "{said}");
    }

    /// **A check runs its program with its arguments and keeps its answer**:
    /// the first run has nothing to compare with, the second says what
    /// changed, and one with other arguments is compared with its own last.
    #[tokio::test]
    async fn a_check_runs_and_says_what_changed_since() {
        let dir = tempfile::tempdir().unwrap();
        let checks = dir.path().join(FOLDER);
        std::fs::create_dir_all(&checks).unwrap();
        // The agent's own binary is a program every machine that runs this
        // test has: `cargo` echoes nothing useful, so the test runs the
        // platform's own echo through a declared argument.
        let (program, argv) = match cfg!(windows) {
            true => ("cmd", "[\"/C\", \"echo\", \"{word}\"]"),
            false => ("echo", "[\"{word}\"]"),
        };
        std::fs::write(
            checks.join("say.yaml"),
            format!(
                "name: say\nsays: Says a word.\nargs:\n  - name: word\n    values: [alpha, \
                 beta]\nprogram: {program}\nargv: {argv}\n"
            ),
        )
        .unwrap();
        let first = run(dir.path(), "say", &given(&[("word", "alpha")]))
            .await
            .unwrap();
        assert!(first.starts_with("This is the first time"), "{first}");
        assert!(first.trim_end().ends_with("alpha"), "{first}");
        let again = run(dir.path(), "say", &given(&[("word", "alpha")]))
            .await
            .unwrap();
        assert!(again.starts_with("Nothing changed"), "{again}");
        let other = run(dir.path(), "say", &given(&[("word", "beta")]))
            .await
            .unwrap();
        assert!(other.starts_with("This is the first time"), "{other}");

        let none = run(dir.path(), "reboot", &BTreeMap::new())
            .await
            .unwrap_err();
        assert_eq!(
            none,
            "This machine has no check called `reboot`. Its checks: say."
        );
        let bad = run(dir.path(), "say", &given(&[("word", "gamma")]))
            .await
            .unwrap_err();
        assert_eq!(bad, "`word` is one of: alpha, beta.");
    }
}
