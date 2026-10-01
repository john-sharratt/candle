//! What counts as running git directly, and — as carefully — what does not.
//!
//! Every table pairs commands that run git with commands that only mention
//! it: an argument, a file name, quoted text, a comment, a here-document, a
//! package name, a question about where git is. A false alarm refuses a
//! legitimate command; a miss lets git run on a checkout about to be reset.

use super::*;

fn found(program: &str, args: &[&str]) -> Option<String> {
    direct_git(&SandboxCommand::new(program).args(args.iter().copied()))
}

fn sh(script: &str) -> Option<String> {
    found("sh", &["-c", script])
}

fn cmd(script: &str) -> Option<String> {
    found("cmd", &["/D", "/C", script])
}

fn pwsh(script: &str) -> Option<String> {
    found("pwsh", &["-NoProfile", "-Command", script])
}

fn assert_runs(what: &str, got: Option<String>) {
    assert!(got.is_some(), "missed git in {what:?}");
}

fn assert_clean(what: &str, got: Option<String>) {
    assert_eq!(got, None, "a false alarm on {what:?}");
}

/// PowerShell's `-EncodedCommand` form of `script`.
fn encoded(script: &str) -> String {
    let bytes: Vec<u8> = script.encode_utf16().flat_map(u16::to_le_bytes).collect();
    BASE64.encode(bytes)
}

// ── The program ──────────────────────────────────────────────────────────────

/// **git as the program is found in every spelling.**
#[test]
fn git_as_the_program_in_every_spelling() {
    for program in [
        "git",
        "Git",
        "GIT",
        "git.exe",
        "GIT.EXE",
        "git.cmd",
        "git.bat",
        "git.com",
        "/usr/bin/git",
        "/usr/local/bin/git",
        "C:\\Program Files\\Git\\cmd\\git.exe",
        "C:/Program Files/Git/bin/git.exe",
        "./git",
        "\"git\"",
        "'git'",
        "git-upload-pack",
        "git-receive-pack",
        "git-upload-archive",
        "GIT-SHELL.EXE",
    ] {
        assert_runs(program, found(program, &["status"]));
        assert_runs(program, found(program, &[]));
    }
}

/// **A program whose name only starts with, contains or resembles `git` is
/// not git.**
#[test]
fn programs_named_like_git_are_not_git() {
    for program in [
        "gitk",
        "git-gui",
        "git-cliff",
        "git-lfs",
        "git-town",
        "git-crypt",
        "gitleaks",
        "gitui",
        "lazygit",
        "tig",
        "digit",
        "legit.exe",
        "git.txt",
        "git.sh",
        "git.py",
        "./gitlike.sh",
        "github-cli",
        "gh",
        "git_status",
        "git2",
        "gît",
    ] {
        assert_clean(program, found(program, &["status"]));
    }
}

/// **A program that is not a shell is never read as one** — `git` among its
/// arguments is its business.
#[test]
fn git_as_an_argument_to_another_program() {
    for (program, args) in [
        ("cargo", vec!["run", "--", "git", "status"]),
        ("cargo", vec!["add", "git2"]),
        ("cargo", vec!["install", "--git", "https://github.com/x/y"]),
        ("npm", vec!["install", "git"]),
        ("python", vec!["-m", "git"]),
        ("python", vec!["-c", "import os; os.system('git status')"]),
        (
            "node",
            vec!["-e", "require('child_process').execSync('git log')"],
        ),
        ("rg", vec!["git status", "src"]),
        ("grep", vec!["-r", "git", "."]),
        ("make", vec!["git"]),
        ("echo", vec!["git", "status"]),
    ] {
        assert_clean(&format!("{program} {args:?}"), found(program, &args));
    }
}

// ── Wrappers ─────────────────────────────────────────────────────────────────

/// **A wrapper given as the program runs git when its command is git** —
/// past its own switches and their values, its leading arguments, and `--`.
#[test]
fn a_wrapper_as_the_program() {
    let runs = [
        ("env", vec!["GIT_TRACE=1", "git", "status"]),
        ("env", vec!["-i", "git", "status"]),
        ("env", vec!["-u", "HOME", "git", "status"]),
        ("env", vec!["--", "git", "status"]),
        ("env", vec!["-S", "git status"]),
        ("env", vec!["sh", "-c", "git status"]),
        ("xargs", vec!["git", "show"]),
        ("xargs", vec!["-n1", "git", "show"]),
        ("xargs", vec!["-n", "1", "git", "show"]),
        ("xargs", vec!["-I", "{}", "git", "show", "{}"]),
        ("xargs", vec!["-0", "-P", "4", "git", "show"]),
        ("nohup", vec!["git", "fetch"]),
        ("timeout", vec!["10", "git", "status"]),
        ("timeout", vec!["-k", "5", "10", "git", "status"]),
        ("timeout", vec!["--signal=KILL", "10", "git"]),
        ("sudo", vec!["git", "pull"]),
        ("sudo", vec!["-u", "bob", "git", "pull"]),
        ("nice", vec!["-n", "5", "git", "gc"]),
        ("nice", vec!["git", "gc"]),
        ("time", vec!["git", "log"]),
        ("stdbuf", vec!["-oL", "git", "log"]),
        ("stdbuf", vec!["-o", "L", "git", "log"]),
        ("watch", vec!["git", "status"]),
        ("watch", vec!["-n", "5", "git", "status"]),
        ("/usr/bin/env", vec!["git", "log"]),
    ];
    for (program, args) in runs {
        assert_runs(&format!("{program} {args:?}"), found(program, &args));
    }
    let clean = [
        ("env", vec![]),
        ("env", vec!["cargo", "run", "--", "git"]),
        ("env", vec!["-u", "git", "cargo", "test"]),
        ("env", vec!["GIT_DIR=x", "cargo", "test"]),
        ("xargs", vec!["grep", "git"]),
        ("xargs", vec!["-n1", "echo", "git"]),
        ("xargs", vec!["-I", "git", "echo", "git"]),
        ("timeout", vec!["10"]),
        ("timeout", vec!["10", "cargo", "test"]),
        ("sudo", vec!["-u", "git", "cargo", "build"]),
        ("nice", vec!["cargo", "build"]),
        ("time", vec!["-o", "git.log", "cargo", "build"]),
        ("watch", vec!["-n", "5", "cat", "git.log"]),
    ];
    for (program, args) in clean {
        assert_clean(&format!("{program} {args:?}"), found(program, &args));
    }
}

// ── Which arguments are a script ─────────────────────────────────────────────

/// **Only the script a shell is given is read** — never a script file's
/// name, the arguments passed to one, or a shell's own options.
#[test]
fn only_the_script_is_read() {
    let runs = [
        ("sh", vec!["-c", "git status"]),
        ("bash", vec!["-c", "git status", "arg0", "arg1"]),
        ("bash", vec!["-o", "pipefail", "-ec", "git gc"]),
        ("bash", vec!["--rcfile", "x", "-c", "git log"]),
        ("bash", vec!["-lc", "git log"]),
        ("zsh", vec!["-c", "git status"]),
        ("dash", vec!["-c", "git status"]),
        ("ksh", vec!["-c", "git status"]),
        ("fish", vec!["-c", "git status"]),
        ("BASH.EXE", vec!["-c", "git status"]),
        ("/bin/sh", vec!["-c", "git status"]),
        (
            "C:\\Program Files\\Git\\bin\\bash.exe",
            vec!["-lc", "git status"],
        ),
        ("cmd", vec!["/C", "git", "status"]),
        ("cmd", vec!["/c", "git"]),
        ("cmd.exe", vec!["/K", "git log"]),
        ("CMD", vec!["/Q", "/C", "echo x & git log"]),
        ("cmd", vec!["/D", "/S", "/C", "git status"]),
        ("powershell", vec!["-Command", "git status"]),
        ("powershell", vec!["-command", "git", "status"]),
        ("powershell.exe", vec!["-c", "git log"]),
        (
            "pwsh",
            vec![
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "git status",
            ],
        ),
        ("pwsh", vec!["-Com", "git log"]),
    ];
    for (program, args) in runs {
        assert_runs(&format!("{program} {args:?}"), found(program, &args));
    }
    let clean = [
        ("sh", vec![]),
        ("sh", vec!["-c"]),
        ("sh", vec!["build.sh", "git"]),
        ("sh", vec!["-x", "build.sh", "git", "status"]),
        ("bash", vec!["--login", "run.sh"]),
        ("bash", vec!["-euo", "pipefail", "-c", "echo git"]),
        ("bash", vec!["git.sh"]),
        ("cmd", vec![]),
        ("cmd", vec!["/C"]),
        ("cmd", vec!["echo", "git"]),
        ("cmd", vec!["/c", "echo", "git"]),
        // One quoted word: cmd strips the launcher's quotes around the
        // argument, and what is left is a quoted name, not git.
        ("cmd", vec!["/D", "/S", "/C", "\"git status\""]),
        ("cmd", vec!["/C", "\"git status & echo done\""]),
        ("powershell", vec!["-File", "git.ps1"]),
        ("powershell", vec!["-File", "run.ps1", "git"]),
        ("pwsh", vec!["-Command", "-"]),
        ("pwsh", vec!["-c", "Write-Host", "git"]),
        ("pwsh", vec!["-NoProfile", "run.ps1"]),
    ];
    for (program, args) in clean {
        assert_clean(&format!("{program} {args:?}"), found(program, &args));
    }
}

// ── sh and its kin ───────────────────────────────────────────────────────────

/// **git as any command of an sh script is found.**
#[test]
fn posix_scripts_that_run_git() {
    for script in [
        "git status",
        "git",
        "  git  ",
        "git\tstatus",
        "echo hi && git commit -m x",
        "echo hi || git fetch",
        "cd sub; git log",
        "cd sub;git log",
        "ls | git hash-object --stdin",
        "ls |& git hash-object --stdin",
        "echo a\ngit push",
        "echo a\r\ngit push",
        "sleep 1 & git fetch",
        "echo $(git rev-parse HEAD)",
        "echo \"at $(git rev-parse HEAD)\"",
        "echo $(echo $(git log))",
        "x=$(git log -1)",
        "echo `git describe`",
        "echo \"`git describe`\"",
        "FOO=1 BAR=2 git status",
        "GIT_DIR=x git status",
        "if git diff --quiet; then echo same; fi",
        "if true; then git pull; fi",
        "if false; then :; else git pull; fi",
        "if false; then :; elif git pull; then :; fi",
        "while ! git fetch; do sleep 1; done",
        "until git push; do sleep 1; done",
        "for f in a b; do git add $f; done",
        "! git diff --quiet",
        "(git status)",
        "( cd sub && git log )",
        "{ git fetch; }",
        "echo <(git log)",
        "diff <(git show a:x) x",
        "exec git status",
        "exec -a name git status",
        "command git status",
        "env -i git status",
        "env -u HOME git status",
        "env FOO=1 git status",
        "env -S 'git status'",
        "time git status",
        "time -p git status",
        "nohup git fetch &",
        "nice -n 10 git gc",
        "sudo -u bob git pull",
        "timeout 30 git fetch",
        "timeout -k 5 30 git fetch",
        "stdbuf -oL git log",
        "echo x | xargs git add",
        "find . -name '*.rs' | xargs -I {} git add {}",
        "watch -n 5 git status",
        ">out.txt git status",
        "> out.txt git status",
        "2>/dev/null git status",
        "git status >/dev/null 2>&1",
        "git status &>log",
        "\\git status",
        "\"git\" status",
        "'git' status",
        "g\"i\"t status",
        "'g'it status",
        "/usr/bin/git status",
        "\"/c/Program Files/Git/bin/git.exe\" status",
        "cat <<EOF\nx\nEOF\ngit status",
        "cat <<'EOF' >x\nx\nEOF\ngit status",
        "echo hi # note\ngit status",
        "echo 'a' ; git log",
        "echo \"a\" && git log",
        "git-upload-pack .",
        "bash -c 'git status'",
        "sh -c \"echo; git log\"",
        "sh -c \"sh -c 'git push'\"",
        "cmd /c git status",
        "pwsh -c 'git log'",
        "a=1; git stash",
        "echo ok && sudo git pull",
        // Beside the constructs that are text, the commands are still read.
        "echo ${x}; git status",
        "echo ${git}; git status",
        "arr=(a b); git log",
        "(( 1 )) && git log",
        "echo $((1 + 2)) && git log",
        "f() { git status; }",
        "echo {a,b}; git log",
        "case $x in a) git log;; esac",
        "case $x in\n  a)\n    git log\n    ;;\nesac",
        "case $x in a|b) echo; git log;; esac",
        "case $x in a) echo;; esac; git log",
        "case $x in a) echo;; esac || git log",
        "case $x in a) echo;; esac | git hash-object --stdin",
        "case $x in a|b) echo;; esac && git log",
        "cat <<EOF\nx\nEOF\ngit log",
    ] {
        assert_runs(script, sh(script));
    }
}

/// **A here-document runs to its terminator alone on a line** — as sh reads
/// it, a line that only starts with the delimiter does not end it, so what
/// follows is still the document's text.
#[test]
fn a_heredoc_ends_only_at_its_terminator_line() {
    assert_clean("EOF || …", sh("cat <<EOF\nx\nEOF || true\ngit status"));
    assert_runs("EOF alone", sh("cat <<EOF\nx\nEOF\ngit status"));
    assert_runs("<<- tabs", sh("cat <<-EOF\n\tx\n\tEOF\ngit status"));
}

/// **An sh script that only mentions git is not refused.**
#[test]
fn posix_scripts_that_only_mention_git() {
    for script in [
        "echo git",
        "echo git status",
        "echo --git",
        "echo a=git",
        "echo \\git",
        "echo $GIT_DIR",
        "echo git; echo \"git\"",
        "cat .gitignore",
        "cat .gitattributes .gitmodules",
        "ls git/ .github",
        "ls -la | grep git",
        "grep -r git src",
        "grep -e git -e status f",
        "rg 'git status' docs",
        "find . -name '*.git'",
        "test -d .git && echo repo",
        "[ -d .git ] && echo repo",
        "gitk",
        "digit",
        "git-cliff -o CHANGELOG.md",
        "git-lfs pull",
        "gitleaks detect",
        "cargo add git2",
        "cargo install --git https://github.com/x/y",
        "pip install git+https://example.com/x.git",
        "npm install github:user/repo",
        "echo 'use git later; git is fine'",
        "echo \"a; git status\"",
        "echo \"a && git status\"",
        "echo 'a | git log'",
        "echo '\"' git",
        "grep \"a) git status\" log",
        "echo \"(git) is here\"",
        "echo \"$(date) git\"",
        "echo $(date) git status",
        "echo `date` git status",
        "echo \"`date` git status\"",
        "echo $(echo $(date) git) done",
        "command -v git",
        "command -V git",
        "command -v git >/dev/null || echo missing",
        "which git",
        "type git",
        "hash git",
        "man git",
        "cd git-repo && ./build.sh",
        "mkdir git && cd git",
        "touch git.txt",
        "./git_status.sh",
        "./scripts/git-hooks.sh",
        "GIT_DIR=x cargo test",
        "export GIT=git",
        "alias g=git",
        "# git status",
        "echo hi # && git status",
        "echo hi # ; git log",
        "cat <<EOF\ngit status\nEOF",
        "cat <<'EOF'\ngit push\nEOF\necho done",
        "cat <<\"EOF\"\ngit push\nEOF",
        "cat <<-EOF\n\tgit push\n\tEOF",
        "cat <<EOF > notes.md\ngit log\nEOF",
        "cat <<EOF\r\ngit log\r\nEOF\r\n",
        "cat <<< 'git status'",
        "printf '%s\\n' \"git status\"",
        "echo hi 2>&1 | grep git",
        "echo x > git",
        "cat < git",
        "env",
        "timeout 30 cargo test",
        "nice -n 10 cargo build",
        "xargs -n1 echo git",
        "xargs grep git",
        "sudo -u git cargo build",
        "env -u git cargo test",
        "time -o git.log cargo build",
        "exec -a git cargo run",
        "watch -n 5 cat git.log",
        "case $x in git) echo yes;; esac",
        "for g in git hg; do echo $g; done",
        "sh build.sh git",
        "bash -c 'echo git'",
    ] {
        assert_clean(script, sh(script));
    }
}

// ── cmd ──────────────────────────────────────────────────────────────────────

/// **git as any command of a cmd line is found.**
#[test]
fn cmd_lines_that_run_git() {
    for script in [
        "git status",
        "git status & echo done",
        "echo x && git log",
        "echo x || git log",
        "dir | git hash-object --stdin",
        "@git status",
        "@ git status",
        "call git.cmd log",
        "call git log",
        "CALL git log",
        "start git fetch",
        "start /B git fetch",
        "start \"\" git fetch",
        "start \"My title\" /B git fetch",
        "start /D sub /B git status",
        "START /WAIT git status",
        "if exist x (git status)",
        "if exist x git status",
        "if not exist x git status",
        "if errorlevel 1 git log",
        "if defined FOO git log",
        "if \"%X%\"==\"1\" git log",
        "if /i \"%a%\" == \"b\" git log",
        "if %X% EQU 1 git log",
        "if exist x (echo a) else (git status)",
        "for %f in (a b) do git show %f",
        "for /f %l in ('git log') do echo %l",
        "for /f \"usebackq\" %l in (`git log`) do echo %l",
        "for /f \"tokens=1\" %l in ('git rev-parse HEAD') do set H=%l",
        "for %f in (git.txt) do git add %f",
        "set X=1 & git status",
        "cd sub & git log",
        "g^it status",
        "\"git\" status",
        "\"C:\\Program Files\\Git\\cmd\\git.exe\" status",
        "git.exe status",
        "GIT.CMD status",
        "(git status)",
        "echo a\r\ngit log",
        "cmd /c git status",
        "bash -c \"git log\"",
        "powershell -Command git status",
        "2>nul git status",
        // Beside the constructs that are text, the commands are still read.
        "echo {x} & git log",
        "echo x; y & git log",
        "echo (x) && git log",
        "rem note\r\ngit log",
    ] {
        assert_runs(script, cmd(script));
    }
}

/// **A cmd line that only mentions git is not refused** — its caret escapes,
/// comments and printed brackets included.
#[test]
fn cmd_lines_that_only_mention_git() {
    for script in [
        "echo git",
        "echo git status",
        "echo git & type .gitattributes",
        "echo \"x & git status\"",
        "echo a ^& git status",
        "echo a ^| git status",
        "echo a ^&^& git status",
        "where git",
        "where /q git || echo missing",
        "rem git status",
        "rem git status & git log",
        "REM git status",
        ":: git status",
        "@rem git log",
        "echo (git)",
        "echo (git status)",
        "@echo (git) & dir",
        "set GIT=git",
        "set \"G=(git)\"",
        "title git (log)",
        "type .gitignore",
        "dir /s *.git",
        "findstr git notes.txt",
        "gitk",
        "git-cliff",
        "echo git>git.txt",
        "copy git.txt out.txt",
        "start notepad git.txt",
        "start \"git\" notepad",
        "call build.cmd git",
        "if exist git\\x echo yes",
        "if \"%1\"==\"git\" echo yes",
        "for %f in (git hg) do echo %f",
        "for %f in (*.git) do echo %f",
        "for /f %l in (git.txt) do echo %l",
        "for /f \"delims=\" %l in (\"git status\") do echo %l",
        "for /d %d in (git*) do dir %d",
        "echo it's git & dir",
        "echo 2>&1 git",
    ] {
        assert_clean(script, cmd(script));
    }
}

// ── PowerShell ───────────────────────────────────────────────────────────────

/// **git as any command of a PowerShell script is found** — including behind
/// the call operator, in a subexpression, an assignment, `Start-Process`,
/// `Invoke-Expression`, and `-EncodedCommand`.
#[test]
fn powershell_scripts_that_run_git() {
    for script in [
        "git status",
        "& git status",
        "& 'C:\\Program Files\\Git\\cmd\\git.exe' status",
        "& \"git.exe\" log",
        "git log; Write-Host done",
        "Write-Host done; git log",
        "if ($true) { git pull }",
        "Write-Host $(git log -1)",
        "Write-Host \"$(git rev-parse HEAD)\"",
        "$x = git log -1",
        "(git log).Count",
        "Get-ChildItem | ForEach-Object { git show $_ }",
        "Invoke-Expression 'git status'",
        "iex \"git log\"",
        "Start-Process git -ArgumentList status",
        "Start-Process -FilePath git -ArgumentList 'status' -Wait",
        "Start-Process -Wait -NoNewWindow git",
        "saps git",
        "g`it status",
        "git.exe status",
        "Get-Date; git fetch",
        "cmd /c git status",
        "echo ok && git log",
        "Write-Host hi # note\ngit log",
        "<# a note #> git log",
        // Beside the constructs that are text, the commands are still read.
        "& 'git' status",
        ". git log",
        "$h = @{a=1}; git log",
        "${x} = 1; git log",
        "switch ($x) { a { git log } }",
        "function f { git status }",
        "('x'); git log",
        "'x' | Out-Null; git log",
        "@(git log)",
        "echo ok && git log",
        "Start-Process https://x; git log",
    ] {
        assert_runs(script, pwsh(script));
    }
    assert_runs(
        "-EncodedCommand",
        found("powershell", &["-EncodedCommand", &encoded("git status")]),
    );
    assert_runs(
        "-ec",
        found("pwsh", &["-ec", &encoded("Write-Host hi; git log")]),
    );
}

/// **A PowerShell script that only mentions git is not refused.**
#[test]
fn powershell_scripts_that_only_mention_git() {
    for script in [
        "Write-Host git",
        "Write-Output 'git status'",
        "Write-Host 'git status; git log'",
        "Write-Host \"a; git log\"",
        "Get-Command git",
        "Get-Content .gitignore",
        "Select-String git *.md",
        "# git status",
        "<# git status #>",
        "<# multi\nline git #> Write-Host hi",
        "@\"\ngit status\n\"@ | Out-File x.txt",
        "@'\ngit log\n'@",
        "Start-Process notepad -ArgumentList git.txt",
        "Start-Process notepad git",
        "Start-Process -FilePath notepad -ArgumentList git",
        "$env:GIT_DIR = 'x'",
        "$git = 'x'",
        "Invoke-Expression 'Write-Host git'",
        "Write-Host `git",
    ] {
        assert_clean(script, pwsh(script));
    }
    assert_clean(
        "-enc",
        found("powershell", &["-enc", &encoded("Write-Host git")]),
    );
    assert_clean(
        "undecodable",
        found("powershell", &["-EncodedCommand", "not base64!"]),
    );
}

// ── What is reported ─────────────────────────────────────────────────────────

/// **The refusal names the part that runs git, as written**: the command
/// line itself, or the one command of a script that does.
#[test]
fn the_part_that_runs_git_is_named() {
    for ((program, args), want) in [
        (("git", vec!["status"]), "git status"),
        (("git", vec![]), "git"),
        (("GIT.EXE", vec!["log", "-1"]), "GIT.EXE log -1"),
        (("env", vec!["FOO=1", "git", "log"]), "env FOO=1 git log"),
        (
            ("sh", vec!["-c", "cargo build && git commit -am wip"]),
            "git commit -am wip",
        ),
        (
            ("sh", vec!["-c", "echo $(git rev-parse HEAD)"]),
            "git rev-parse HEAD",
        ),
        (("sh", vec!["-c", "cmd /c git status"]), "cmd /c git status"),
        (
            ("cmd", vec!["/D", "/C", "git status & echo done"]),
            "git status",
        ),
        (("cmd", vec!["/C", "echo x & @git log"]), "@git log"),
        (
            ("pwsh", vec!["-c", "Write-Host hi; git log -1"]),
            "git log -1",
        ),
        (
            ("powershell", vec!["-ec", &encoded("dir; git fetch")]),
            "git fetch",
        ),
    ] {
        assert_eq!(
            found(program, &args).as_deref(),
            Some(want),
            "{program} {args:?}"
        );
    }
}

// ── Generated: pieces that run git mixed with pieces that only mention it ────

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

/// Scripts of one to five commands, joined by the dialect's operators, each
/// command drawn from ones that run git and ones that only mention it: git is
/// found exactly when a command that runs it is present, and the first such
/// command is the one named. Four seeds of 3,000 scripts each.
fn mixed(
    seed: u64,
    runs: &[&str],
    mentions: &[&str],
    operators: &[&str],
    check: impl Fn(&str) -> Option<String>,
) {
    for stream in 0..4 {
        let mut rng = Lcg(seed + stream);
        for round in 0..3000 {
            let n = 1 + rng.below(5);
            let mut script = String::new();
            let mut first_git: Option<&str> = None;
            for k in 0..n {
                if k > 0 {
                    script.push_str(operators[rng.below(operators.len())]);
                }
                let piece = if rng.below(3) == 0 {
                    let piece = runs[rng.below(runs.len())];
                    first_git.get_or_insert(piece);
                    piece
                } else {
                    mentions[rng.below(mentions.len())]
                };
                script.push_str(piece);
            }
            assert_eq!(
                check(&script).as_deref(),
                first_git,
                "seed {stream}, round {round}: {script:?}"
            );
        }
    }
}

#[test]
fn generated_posix_scripts() {
    mixed(
        0x5eed,
        &[
            "git status",
            "\\git fetch",
            "'git' push",
            "FOO=1 git gc",
            "command git show",
            "timeout 5 git pull",
            "env -u X git log",
            "xargs -n1 git show",
            "! git diff --quiet",
            "exec git log",
        ],
        &[
            "echo git",
            "grep -r git src",
            "cat .gitignore",
            "cargo add git2",
            "echo \"x; git status\"",
            "echo 'a | git log'",
            "command -v git",
            "ls git/",
            "echo $(date) git",
            "gitk",
            "git-cliff",
            "printf '%s' git",
            "sudo -u git make",
            "echo hi 2>&1",
            "echo ${git}",
            "arr=(git hg)",
            "echo {git}",
            "(( git > 0 ))",
            "echo $((git + 1))",
            "case $x in git|hg) echo vcs;; esac",
            // The terminator alone on its line, as a here-document needs.
            "cat <<EOF\ngit status\nEOF\ntrue",
            "gitfn() { echo x; }",
        ],
        &["; ", " && ", " || ", " | ", "\n", " & ", ";"],
        sh,
    );
}

#[test]
fn generated_cmd_lines() {
    mixed(
        0xc0de,
        &[
            "git status",
            "@git log",
            "call git fetch",
            "start /B git gc",
            "g^it show",
            "\"git\" push",
            "if exist x git status",
        ],
        &[
            "echo git",
            "type .gitignore",
            "echo \"x & git status\"",
            "echo a ^& git log",
            "where git",
            "findstr git a.txt",
            "echo (git)",
            "gitk",
            "set G=git",
            "start \"git\" notepad",
            "echo;git",
            "echo x; git log",
            "echo {git}",
            "for %f in (git hg) do echo %f",
            "start https://example.com/git.exe",
        ],
        &[" & ", " && ", " || ", " | ", "\r\n", "&"],
        cmd,
    );
}

#[test]
fn generated_powershell_scripts() {
    mixed(
        0xbeef,
        &[
            "git status",
            "git log -1",
            "iex 'git fetch'",
            "Start-Process git",
            "g`it show",
        ],
        &[
            "Write-Host git",
            "Get-Command git",
            "Write-Host 'a; git log'",
            "Get-Content .gitignore",
            "Start-Process notepad git",
            "$h = @{ git = 1 }",
            "'git' | Out-Null",
            "switch ($x) { git { 1 } }",
            "Write-Host ${git}",
            "function git-x { Write-Host hi }",
        ],
        &["; ", "\n", " | "],
        pwsh,
    );
}

// ── The reader's parts ───────────────────────────────────────────────────────

/// Words honour each dialect's quotes and escape character.
#[test]
fn words_honour_each_dialects_quotes_and_escape() {
    let texts =
        |s: &str, d: Dialect| -> Vec<String> { words(s, d).into_iter().map(|w| w.text).collect() };
    assert_eq!(
        texts("a \"b c\" 'd e' f\\ g", Dialect::Posix),
        ["a", "b c", "d e", "f g"]
    );
    assert_eq!(
        texts("a \"b c\" 'd e' f^ g", Dialect::Cmd),
        ["a", "b c", "'d", "e'", "f g"]
    );
    assert_eq!(
        texts("a \"b c\" 'd e' f` g", Dialect::PowerShell),
        ["a", "b c", "d e", "f g"]
    );
    assert_eq!(texts("\"\" x", Dialect::Cmd), ["", "x"]);
    assert_eq!(words("\"a b\"", Dialect::Posix)[0].raw, "\"a b\"");
    assert!(texts("   ", Dialect::Posix).is_empty());
}

/// Scripts are cut at operators into the commands that start one.
#[test]
fn scripts_are_cut_into_commands() {
    let cut = |s: &str, d: Dialect| -> Vec<String> {
        commands(s, d)
            .into_iter()
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect()
    };
    assert_eq!(
        cut("a; b && c | d\ne || f & g", Dialect::Posix),
        ["a", "b", "c", "d", "e", "f", "g"]
    );
    assert_eq!(
        cut("a 2>&1 | b &>log", Dialect::Posix),
        ["a 2>&1", "b &>log"]
    );
    assert_eq!(cut("x $(y) z; w", Dialect::Posix), ["x", "y", "w"]);
    assert_eq!(cut("a # b; c\nd", Dialect::Posix), ["a", "d"]);
    assert_eq!(cut("a ^& b & c", Dialect::Cmd), ["a ^& b", "c"]);
    assert_eq!(cut("rem a & b\nc", Dialect::Cmd), ["c"]);
    assert_eq!(cut("a <# b; c #> d; e", Dialect::PowerShell), ["a  d", "e"]);
}

#[test]
fn redirections_are_recognised() {
    for (word, want) in [
        (">", Some(true)),
        (">>", Some(true)),
        ("2>", Some(true)),
        ("&>", Some(true)),
        ("<", Some(true)),
        ("2>&1", Some(false)),
        (">out", Some(false)),
        ("&>log", Some(false)),
        ("<in", Some(false)),
        ("a>b", None),
        ("2", None),
        ("-", None),
        ("git", None),
    ] {
        assert_eq!(redirection(word), want, "{word}");
    }
}

#[test]
fn assignments_are_recognised() {
    for (word, want) in [
        ("A=1", true),
        ("_x=", true),
        ("GIT_DIR=.git", true),
        ("1A=x", false),
        ("--a=b", false),
        ("=x", false),
        ("a.b=c", false),
        ("git", false),
    ] {
        assert_eq!(is_assignment(word), want, "{word}");
    }
}
