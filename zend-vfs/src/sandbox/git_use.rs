//! Finding a command that runs git directly.
//!
//! The checkout's branch, index and working tree belong to the sandbox for the
//! length of a run, and the `git_*` tools are how a repository's git state is
//! read and changed. A command that is git — or a shell script that runs git
//! as one of its commands — is refused before it starts, so the caller learns
//! to reach for those tools instead of getting an answer from a checkout the
//! sandbox is about to reset.
//!
//! "Directly" is the line, and the word `git` appearing is not: only git in
//! the position of the program that runs counts. `echo git`,
//! `grep "git status" log`, `cat .gitignore`, `cargo add git2`,
//! `command -v git` and `gitk` run no git; `git log`, `cd x && git log`,
//! `echo "$(git rev-parse HEAD)"` and `timeout 30 git fetch` do.
//!
//! - **The program** is git — `git`, `GIT.EXE`, `/usr/bin/git`, `\git`, or one
//!   of the executables git ships that act on a repository
//!   ([`GIT_PROGRAMS`]). A different program whose name starts `git`
//!   (`git-cliff`, `git-lfs`) is not git.
//! - **A wrapper** runs its arguments as a command — `env`, `timeout`,
//!   `xargs`, `sudo`, `nice`, `exec`, cmd's `start` and `call`, PowerShell's
//!   `Start-Process`, the shell keywords `if`/`then`/`do`/`!`, … Its own
//!   switches are passed over, with the value each switch takes
//!   (`timeout -k 5 30 git`, `xargs -I {} git show {}`), and so is a leading
//!   argument that is not the command (`timeout`'s duration, `start`'s
//!   title). cmd's `if` passes over its condition (`if exist x git status`).
//! - **A shell's script** runs git when one of its commands does. Only the
//!   script is read — the argument after `sh -c`, the rest of the line after
//!   `cmd /C`, what follows `powershell -Command` (or, decoded, its
//!   `-EncodedCommand`) — never a script file's name or the arguments passed
//!   to one. A shell inside a script is read the same way
//!   (`cmd /C "bash -c 'git log'"`), and so is PowerShell's
//!   `Invoke-Expression`.
//!
//! A script is read per dialect — sh and its kin, cmd, PowerShell — honouring
//! each one's quotes and escape character, and cut into the commands it runs
//! ([`split`] names every construct that is text rather than a command). Each
//! command's first word is taken after `NAME=value` assignments,
//! redirections, and cmd's leading `@`, with its quotes and escapes taken off
//! (`"git"`, `g\it`, `g^it`). In PowerShell a quoted first word is a string —
//! an expression — unless the call operator `&` runs it. An address
//! (`https://…`) is never a program.
//!
//! This is how a person reading the script would see it; it is not a shell. A
//! command named through a variable (`$GIT status`) or an alias, and a program
//! that runs git itself — a build script reading the commit id, a script file
//! the command names — are not the caller using git, and run; the sandbox
//! takes back whatever they change (see [`crate::checkout`]).

mod split;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;

use self::split::commands;
use super::command::SandboxCommand;

/// The executables git ships that read or change a repository. Other
/// programs named `git-…` are third-party tools, not git.
const GIT_PROGRAMS: [&str; 5] = [
    "git",
    "git-upload-pack",
    "git-receive-pack",
    "git-upload-archive",
    "git-shell",
];

/// Shells whose script follows a `-c` switch (`-c`, `-lc`, `-ec`, …).
const POSIX_SHELLS: [&str; 6] = ["sh", "bash", "zsh", "dash", "ksh", "fish"];

/// How a shell reads its script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dialect {
    /// sh and its kin: `'…'` and `"…"`, `\` escapes, `$(…)` and backticks,
    /// `#` comments, here-documents.
    Posix,
    /// cmd: `"…"` only, `^` escapes, `rem` and `::` comments.
    Cmd,
    /// PowerShell: `'…'` and `"…"`, backtick escapes, `$(…)`, `#` and
    /// `<# #>` comments.
    PowerShell,
}

impl Dialect {
    fn escape(self) -> char {
        match self {
            Dialect::Posix => '\\',
            Dialect::Cmd => '^',
            Dialect::PowerShell => '`',
        }
    }
}

/// One word of a command: as written, and with its quotes and escapes taken
/// off.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Word {
    raw: String,
    text: String,
}

impl Word {
    /// An argument that reached the program already split, as written.
    fn given(arg: &str) -> Self {
        Self {
            raw: arg.to_string(),
            text: arg.to_string(),
        }
    }
}

/// The part of `command` that runs git directly, as the caller wrote it, or
/// `None` when nothing does. For a shell, that is the script's command that
/// runs git; otherwise the whole command.
pub(crate) fn direct_git(command: &SandboxCommand) -> Option<String> {
    let name = base_name(&command.program);
    if let Some((dialect, script)) = script_of(&name, &command.args) {
        return commands(&script, dialect)
            .into_iter()
            .find(|c| runs_git(&words(c, dialect), dialect))
            .map(|c| c.trim().to_string());
    }
    let argv: Vec<Word> = std::iter::once(&command.program)
        .chain(&command.args)
        .map(|a| Word::given(a))
        .collect();
    runs_git(&argv, Dialect::Posix).then(|| {
        std::iter::once(command.program.as_str())
            .chain(command.args.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ")
    })
}

/// `program`'s file name, lower-cased, unquoted, without a Windows launch
/// extension.
fn base_name(program: &str) -> String {
    let unquoted = program.trim_matches(['"', '\'']);
    let name = unquoted
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(unquoted)
        .to_ascii_lowercase();
    [".exe", ".cmd", ".bat", ".com"]
        .iter()
        .find_map(|ext| name.strip_suffix(ext))
        .map(str::to_string)
        .unwrap_or(name)
}

// ── Shells and their scripts ─────────────────────────────────────────────────

/// The script a shell `name` is given in `args`, with how to read it, or
/// `None` when `name` is not a shell or is given a script file instead.
fn script_of(name: &str, args: &[String]) -> Option<(Dialect, String)> {
    if POSIX_SHELLS.contains(&name) {
        // The first switch cluster holding `c` takes the script next.
        let at = args
            .iter()
            .position(|a| a.starts_with('-') && !a.starts_with("--") && a[1..].contains('c'))?;
        return args.get(at + 1).map(|s| (Dialect::Posix, s.clone()));
    }
    if name == "cmd" {
        let at = args
            .iter()
            .position(|a| a.eq_ignore_ascii_case("/c") || a.eq_ignore_ascii_case("/k"))?;
        // The arguments as written. The quotes cmd strips from a `/C` line are
        // the ones the process launcher wraps around an argument holding a
        // space or a quote, never the script's own.
        return Some((Dialect::Cmd, args[at + 1..].join(" ")));
    }
    if name == "powershell" || name == "pwsh" {
        for (at, arg) in args.iter().enumerate() {
            let arg = arg.to_ascii_lowercase();
            if arg.len() >= 2 && "-command".starts_with(&arg) {
                return Some((Dialect::PowerShell, args[at + 1..].join(" ")));
            }
            if arg == "-ec" || (arg.len() >= 2 && "-encodedcommand".starts_with(&arg)) {
                let script = decode_utf16le(args.get(at + 1)?)?;
                return Some((Dialect::PowerShell, script));
            }
        }
    }
    None
}

/// PowerShell's `-EncodedCommand`: base64 of UTF-16LE text.
fn decode_utf16le(encoded: &str) -> Option<String> {
    let bytes = BASE64.decode(encoded.trim()).ok()?;
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    Some(String::from_utf16_lossy(&units))
}

/// `command` split into words, honouring the dialect's quotes and escape.
fn words(command: &str, dialect: Dialect) -> Vec<Word> {
    let mut out = Vec::new();
    let mut raw = String::new();
    let mut text = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = command.chars();
    while let Some(c) = chars.next() {
        if let Some(q) = quote {
            raw.push(c);
            if c == q {
                quote = None;
            } else if c == '\\' && q == '"' && dialect == Dialect::Posix {
                if let Some(n) = chars.next() {
                    raw.push(n);
                    text.push(n);
                }
            } else {
                text.push(c);
            }
            continue;
        }
        if c.is_whitespace() {
            if in_word {
                out.push(Word {
                    raw: std::mem::take(&mut raw),
                    text: std::mem::take(&mut text),
                });
                in_word = false;
            }
            continue;
        }
        in_word = true;
        raw.push(c);
        if c == '"' || (c == '\'' && dialect != Dialect::Cmd) {
            quote = Some(c);
        } else if c == dialect.escape() {
            if let Some(n) = chars.next() {
                raw.push(n);
                text.push(n);
            }
        } else {
            text.push(c);
        }
    }
    if in_word {
        out.push(Word { raw, text });
    }
    out
}

// ── Reading one command ──────────────────────────────────────────────────────

/// Whether the command `argv` runs git.
fn runs_git(argv: &[Word], dialect: Dialect) -> bool {
    let mut i = 0;
    // Assignments and redirections before the program.
    while let Some(word) = argv.get(i) {
        let powershell_assignment = dialect == Dialect::PowerShell
            && word.text.starts_with('$')
            && argv.get(i + 1).is_some_and(|w| w.text == "=");
        if powershell_assignment {
            // `$x = git log`: what runs is to the right.
            i += 2;
        } else if is_assignment(&word.text) {
            i += 1;
        } else if let Some(takes_target) = redirection(&word.text) {
            i += if takes_target { 2 } else { 1 };
        } else {
            break;
        }
    }
    let Some(first) = argv.get(i) else {
        return false;
    };
    let rest = &argv[i + 1..];
    let program = match dialect {
        Dialect::Cmd => first.text.trim_start_matches('@'),
        _ => first.text.as_str(),
    };
    if program.is_empty() {
        // A bare `@`.
        return runs_git(rest, dialect);
    }
    if program.contains("://") {
        // An address — `start https://…/git.exe` opens it; nothing runs git.
        return false;
    }
    if dialect == Dialect::PowerShell {
        if first.raw.starts_with(['"', '\'']) {
            // A string where a command would start is an expression.
            return false;
        }
        if program == "&" || program == "." {
            // The call operator: the next word names the command, quoted or
            // not.
            let mut called = rest.to_vec();
            if let Some(command) = called.first_mut() {
                command.raw = command.text.clone();
            }
            return runs_git(&called, dialect);
        }
    }
    let name = base_name(program);
    if GIT_PROGRAMS.contains(&name.as_str()) {
        return true;
    }
    let args: Vec<String> = rest.iter().map(|w| w.text.clone()).collect();
    if let Some((inner, script)) = script_of(&name, &args) {
        return commands(&script, inner)
            .iter()
            .any(|c| runs_git(&words(c, inner), inner));
    }
    if name == "invoke-expression" || name == "iex" {
        return rest.first().is_some_and(|script| {
            commands(&script.text, Dialect::PowerShell)
                .iter()
                .any(|c| runs_git(&words(c, Dialect::PowerShell), Dialect::PowerShell))
        });
    }
    if name == "if" && dialect == Dialect::Cmd {
        return cmd_if(rest);
    }
    match wrapper(&name) {
        Some(w) => wrapped(&w, rest, dialect),
        None => false,
    }
}

/// `NAME=value`, a shell's per-command environment setting.
fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty()
            && !name.starts_with(|c: char| c.is_ascii_digit())
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// A redirection word — `>`, `2>`, `>>log`, `2>&1`, `&>`, `<` — and whether
/// its target is the next word.
fn redirection(word: &str) -> Option<bool> {
    let op = word.trim_start_matches(|c: char| c.is_ascii_digit());
    let op = op.strip_prefix('&').unwrap_or(op);
    if !op.starts_with(['>', '<']) {
        return None;
    }
    Some(op.trim_start_matches(['>', '<', '&', '|']).is_empty())
}

/// cmd's `if`: its condition passed over, the command after it read.
fn cmd_if(rest: &[Word]) -> bool {
    let lower = |i: usize| rest.get(i).map(|w| w.text.to_ascii_lowercase());
    let mut i = 0;
    if lower(i).as_deref() == Some("/i") {
        i += 1;
    }
    if lower(i).as_deref() == Some("not") {
        i += 1;
    }
    const COMPARISONS: [&str; 7] = ["==", "equ", "neq", "lss", "leq", "gtr", "geq"];
    match lower(i).as_deref() {
        Some("exist" | "defined" | "errorlevel" | "cmdextversion") => i += 2,
        Some(w) if w.contains("==") => i += 1,
        Some(_) if lower(i + 1).is_some_and(|op| COMPARISONS.contains(&op.as_str())) => i += 3,
        _ => return false,
    }
    rest.get(i..).is_some_and(|r| runs_git(r, Dialect::Cmd))
}

/// A program that runs its arguments as a command.
struct Wrapper {
    /// Switches that take the next word as their value.
    values: &'static [&'static str],
    /// Switches whose value is a whole command line (`env -S`).
    lines: &'static [&'static str],
    /// Switches whose value is the command itself (`-FilePath`).
    commands: &'static [&'static str],
    /// Switches that make it look a command up rather than run it
    /// (`command -v`).
    queries: &'static [&'static str],
    /// Arguments before the command that are not it (`timeout`'s duration).
    leading: usize,
    /// Whether `/x` is a switch (cmd) as well as `-x`.
    slash_switches: bool,
    /// Whether a first quoted argument is a title, not the command (`start`).
    title: bool,
    /// Whether switches match in any case (cmd, PowerShell).
    any_case: bool,
}

const NO: &[&str] = &[];

const PLAIN: Wrapper = Wrapper {
    values: NO,
    lines: NO,
    commands: NO,
    queries: NO,
    leading: 0,
    slash_switches: false,
    title: false,
    any_case: false,
};

/// The wrapper called `name`, if it is one.
fn wrapper(name: &str) -> Option<Wrapper> {
    Some(match name {
        "builtin" | "nohup" | "if" | "then" | "else" | "elif" | "do" | "while" | "until" | "!"
        | "and" | "or" | "not" => PLAIN,
        "exec" => Wrapper {
            values: &["-a"],
            ..PLAIN
        },
        "command" => Wrapper {
            queries: &["-v", "-V"],
            ..PLAIN
        },
        "env" => Wrapper {
            values: &["-u", "-C", "--unset", "--chdir"],
            lines: &["-S", "--split-string"],
            ..PLAIN
        },
        "time" => Wrapper {
            values: &["-f", "-o", "--format", "--output"],
            ..PLAIN
        },
        "sudo" => Wrapper {
            values: &["-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U", "-T"],
            ..PLAIN
        },
        "doas" => Wrapper {
            values: &["-u", "-C"],
            ..PLAIN
        },
        "nice" => Wrapper {
            values: &["-n", "--adjustment"],
            ..PLAIN
        },
        "ionice" => Wrapper {
            values: &["-c", "-n", "--class", "--classdata"],
            ..PLAIN
        },
        "timeout" => Wrapper {
            values: &["-s", "-k", "--signal", "--kill-after"],
            leading: 1,
            ..PLAIN
        },
        "stdbuf" => Wrapper {
            values: &["-i", "-o", "-e", "--input", "--output", "--error"],
            ..PLAIN
        },
        "xargs" => Wrapper {
            values: &[
                "-n",
                "-L",
                "-P",
                "-I",
                "-d",
                "-a",
                "-E",
                "-s",
                "--max-args",
                "--max-lines",
                "--max-procs",
                "--replace",
                "--delimiter",
                "--arg-file",
                "--eof",
                "--max-chars",
            ],
            ..PLAIN
        },
        "watch" => Wrapper {
            values: &["-n", "--interval"],
            ..PLAIN
        },
        "call" => Wrapper {
            any_case: true,
            ..PLAIN
        },
        "start" => Wrapper {
            values: &["/D"],
            slash_switches: true,
            title: true,
            any_case: true,
            ..PLAIN
        },
        "start-process" | "saps" => Wrapper {
            values: &[
                "-ArgumentList",
                "-Args",
                "-WorkingDirectory",
                "-Verb",
                "-WindowStyle",
                "-RedirectStandardOutput",
                "-RedirectStandardError",
                "-RedirectStandardInput",
                "-Credential",
            ],
            commands: &["-FilePath"],
            any_case: true,
            ..PLAIN
        },
        _ => return None,
    })
}

/// Whether the command a wrapper `w` runs, from its arguments `rest`, runs
/// git.
fn wrapped(w: &Wrapper, rest: &[Word], dialect: Dialect) -> bool {
    let named = |list: &[&str], word: &str| {
        list.iter().any(|s| {
            if w.any_case {
                s.eq_ignore_ascii_case(word)
            } else {
                *s == word
            }
        })
    };
    let mut leading = w.leading;
    let mut title = w.title;
    let mut i = 0;
    while let Some(word) = rest.get(i) {
        let t = word.text.as_str();
        if t == "--" {
            return runs_git(&rest[i + 1..], dialect);
        }
        let switch =
            t.len() > 1 && (t.starts_with('-') || (w.slash_switches && t.starts_with('/')));
        if switch {
            if named(w.queries, t) {
                return false;
            }
            if named(w.commands, t) {
                return rest
                    .get(i + 1..)
                    .is_some_and(|command| runs_git(command, dialect));
            }
            if named(w.lines, t) {
                return rest
                    .get(i + 1)
                    .is_some_and(|line| runs_git(&words(&line.text, dialect), dialect));
            }
            i += if named(w.values, t) { 2 } else { 1 };
            continue;
        }
        if title && word.raw.starts_with('"') {
            title = false;
            i += 1;
            continue;
        }
        title = false;
        if leading > 0 {
            leading -= 1;
            i += 1;
            continue;
        }
        return runs_git(&rest[i..], dialect);
    }
    false
}

#[cfg(test)]
mod false_alarms;
#[cfg(test)]
mod tests;
