//! Commands that mention git without running it — none may be refused.
//!
//! A refusal here would stop a legitimate command and tell the caller to use
//! the git tools for something that has nothing to do with git. Every entry
//! puts the word `git` somewhere a careless reader would take for a command:
//! as an argument, a file or folder, a variable, a package, a URL, a user,
//! quoted text, a comment, a here-document, a pattern, a label, an
//! expression — in each dialect's own syntax.

use super::*;

fn assert_clean(what: &str, got: Option<String>) {
    assert_eq!(got, None, "a false alarm on {what:?}");
}

fn check_all(program: &str, lead: &[&str], scripts: &[&str]) {
    for script in scripts {
        let mut args: Vec<&str> = lead.to_vec();
        args.push(script);
        assert_clean(script, direct_git(&SandboxCommand::new(program).args(args)));
    }
}

/// **sh: git as an argument, a path, a package, a URL, a user.**
#[test]
fn posix_git_as_a_word_of_another_command() {
    check_all(
        "sh",
        &["-c"],
        &[
            "printf git",
            "echo git | tr a-z A-Z",
            "echo --help | grep -i git",
            "history | grep git",
            "ps aux | grep git",
            "pgrep git",
            "pkill -f git-daemon",
            "kill $(pgrep git)",
            "mv foo git",
            "cp -r .git/hooks backup",
            "rm -rf .git",
            "tar czf repo.tgz --exclude=.git .",
            "rsync -a --exclude .git src/ dst/",
            "zip -r x.zip . -x '.git/*'",
            "du -sh .git",
            "stat .git",
            "wc -l .gitignore",
            "diff -r a/.git b/.git",
            "ln -s /usr/bin/git ./mygit",
            "chmod +x git-hook.sh",
            "vim .gitignore",
            "code .gitattributes",
            "nano git.txt",
            "sed -i 's/git/hg/' README.md",
            "awk '/git/ {print}' log",
            "jq '.git' package.json",
            "perl -ne 'print if /git/' f",
            "docker run --rm alpine/git version",
            "docker build -t git .",
            "kubectl get pods -l app=git",
            "curl -O https://github.com/x/y.git",
            "wget https://git.example.com/x",
            "ssh git@github.com",
            "ssh -T git@github.com",
            "scp file git@host:repo",
            "go get github.com/x/git",
            "go install github.com/git-chglog/git-chglog@latest",
            "brew install git",
            "apt-get install -y git",
            "sudo apt install git",
            "sudo -u git make",
            "sudo -g git id",
            "doas -u git id",
            "choco install git",
            "pip install gitpython",
            "python -c 'import git'",
            "node -e \"require('simple-git')\"",
            "cargo run --bin git-sync",
            "cargo test git",
            "make git-hooks",
            "npm run git:hooks",
            "yarn git",
            "pre-commit run --all-files",
            "echo x >> .git/info/exclude",
            "source git-completion.bash",
            ". ./git-prompt.sh",
            "export PATH=$PATH:/opt/git/bin",
            "export GIT_AUTHOR_NAME=x",
            "GIT_PAGER=cat less file",
            "env GIT_DIR=x cargo test",
            "env -C git make",
            "timeout 30s ./scripts/git-check.sh",
            "nohup ./git-daemon.sh &",
            "xargs -a git.list rm",
            "xargs -d , echo git",
            "xargs -E git echo x",
            "xargs --replace=git echo git",
            "nice -n 5 make git",
            "ionice -c 3 make git",
            "stdbuf -o0 grep git",
            "time -f %e make git",
            "exec 3>git.log",
            "exec >git.log 2>&1",
            "command echo git",
            "command cd git",
            "builtin echo git",
            "true git",
            ": git status",
            "false || echo git",
            "cd ~/git",
            "cd /; cd git",
        ],
    );
}

/// **sh: git as a variable, an expansion, arithmetic, an array, a function
/// name, a brace expansion.**
#[test]
fn posix_git_inside_expansions() {
    check_all(
        "sh",
        &["-c"],
        &[
            "echo $git",
            "echo ${git}",
            "echo ${git:-x}",
            "echo ${GIT:-git}",
            "echo ${#git}",
            "echo ${git} status",
            "echo \"${git}\"",
            "echo $((git + 1))",
            "echo $(( git * 2 )) done",
            "(( git > 0 )) && echo positive",
            "((git++))",
            "arr=(git hg); echo ${arr[0]}",
            "arr+=(git)",
            "declare -a tools=(git hg svn)",
            "declare -A m=([git]=1)",
            "local git=/usr/bin/git",
            "declare -a git",
            "read git",
            "unset git",
            "git=1; echo $git",
            "git_dir=/tmp; echo $git_dir",
            "echo {git}",
            "echo {git,hg}",
            "echo {a,git}/x",
            "mkdir -p {src,git}/x",
            "ls {git}",
            "cp file{,.git}",
            "echo \"$(git_version)\"",
            "echo git-$(date +%s)",
            "echo $(cat .gitignore)",
            "echo $(which git)",
            "echo $(command -v git)",
            "echo $(type -p git)",
            "echo \"$(echo git)\"",
            "echo $(echo git) status",
            "echo \"a\\\"; git status\"",
            "echo a\\; git status",
            "echo a \\& git",
            "echo a\\|git",
            "echo 'it'\"'\"'s git'",
            "echo $'git\\tstatus'",
            "echo git # comment",
            "echo foo#bar; echo git",
            "echo 'a' \"b\" git",
            "git_helper() { echo hi; }",
            "mygit() { echo fake; }",
            "printf '%s' \"$(printf git)\"",
        ],
    );
}

/// **sh: git in conditions, loops, `case` patterns and here-documents.**
#[test]
fn posix_git_in_control_flow_and_documents() {
    check_all(
        "sh",
        &["-c"],
        &[
            "if command -v git >/dev/null; then echo yes; fi",
            "if [ -x \"$(command -v git)\" ]; then echo y; fi",
            "if [ git = git ]; then echo same; fi",
            "test -x /usr/bin/git",
            "[[ -d .git ]] && echo repo",
            "[ \"$1\" = git ] && echo yes",
            "[[ $x == git ]] && echo y",
            "while read git; do echo $git; done < list",
            "for git in a b; do echo $git; done",
            "select git in a b; do break; done",
            "until [ -d .git ]; do sleep 1; done",
            "! grep -q git file",
            "then echo git",
            "do echo git; done",
            "case \"$tool\" in git|hg) echo vcs;; esac",
            "case \"$tool\" in hg|git) echo vcs;; esac",
            "case $x in git) echo a;; hg) echo b;; esac",
            "case $x in\n  git) echo a ;;\n  hg|git) echo b ;;\nesac",
            "case $x in (git) echo a;; esac",
            "case $x in git) echo a;; esac; echo done",
            "case $x in *) echo git;; esac",
            "cat <<EOF | grep git\ngit x\nEOF",
            "cat << EOF\ngit\nEOF",
            "cat <<EOF1\nEOF\ngit\nEOF1",
            "cat <<A <<B\ngit\nA\ngit\nB",
            "cat <<'EOF'\n$(git status)\nEOF",
            "echo x <<< git",
            "echo hi>git",
            "echo hi |tee git.log",
            "cat file |& grep git",
            "ls > git.txt 2>&1",
        ],
    );
}

/// **cmd: git as text — its own separators, brackets, escapes, labels,
/// variables and comments.**
#[test]
fn cmd_git_as_text() {
    check_all(
        "cmd",
        &["/D", "/C"],
        &[
            "echo {git}",
            "echo {git} & dir",
            "echo ^(git^)",
            "echo.git",
            "echo:git",
            "echo;git",
            "echo x; git status",
            "echo x;git log",
            "set PATH=%PATH%;C:\\Program Files\\Git\\cmd",
            "setx GIT_HOME C:\\git",
            "echo %git%",
            "set git=1",
            "set /p git=Name: ",
            "echo %cd%\\git",
            "call :git",
            ":git",
            "goto git",
            "pushd git & dir & popd",
            "mkdir git",
            "rd /s /q .git",
            "attrib +h .git",
            "icacls .git",
            "copy /y git.ini c:\\x",
            "type git.log | find \"error\"",
            "find \"git\" notes.txt",
            "echo git | clip",
            "echo \"git\"",
            "echo 'git status'",
            "echo ^&git",
            "echo x & rem git & git log",
            "dir git",
            "tree .git",
            "robocopy src dst /xd .git",
            "xcopy /e /i .git backup",
            "choco install git -y",
            "winget install --id Git.Git",
            "where /r . git.exe",
            "timeout /t 5 & echo git",
        ],
    );
}

/// **cmd: git in conditions, loops and the programs `start` opens.**
#[test]
fn cmd_git_in_control_flow_and_start() {
    check_all(
        "cmd",
        &["/D", "/C"],
        &[
            "if exist .git echo repo",
            "if not exist .git (echo no repo) else (echo repo)",
            "if \"%1\"==\"git\" (echo git)",
            "if %ERRORLEVEL% NEQ 0 echo git failed",
            "if defined GIT echo %GIT%",
            "if errorlevel 1 echo git",
            "for %%g in (git hg) do echo %%g",
            "for /f \"delims=\" %%i in ('where git') do echo %%i",
            "for /f %%i in ('type .gitignore') do echo %%i",
            "for /f \"tokens=*\" %%a in ('echo git') do echo %%a",
            "for %f in (git*.txt) do del %f",
            "start https://git-scm.com",
            "start https://example.com/downloads/git.exe",
            "start notepad .gitignore",
            "start \"\" \"C:\\Program Files\\Git\\git-bash.exe\"",
            "start git-bash",
            "start /min cmd /c echo git",
            "\"C:\\Program Files\\Git\\git-bash.exe\"",
            "git-bash.exe --cd=.",
            "cmd /c echo git",
            "powershell -c Write-Host git",
            "\"C:\\Program Files\\Git\\bin\\bash.exe\" -c \"echo git\"",
        ],
    );
}

/// **PowerShell: git as a string, a variable, a table key, a switch
/// pattern, a function's name, a path, a process name.**
#[test]
fn powershell_git_as_a_value() {
    check_all(
        "pwsh",
        &["-NoProfile", "-Command"],
        &[
            "$git = Get-Command git",
            "$h = @{ git = 1 }",
            "@{ git = 'x' }.git",
            "${git} = 1",
            "Write-Host ${git}",
            "Write-Host $git",
            "Write-Host $env:GIT_DIR",
            "'git' | Out-File x.txt",
            "\"git status\" | Set-Content cmd.txt",
            "('git' + ' status')",
            "Write-Host ('git' + ' status')",
            "Write-Host \"$('git')\"",
            "@('git', 'hg') | ForEach-Object { $_ }",
            "$args = 'git', 'status'",
            "switch ($x) { 'git' { 'vcs' } }",
            "switch ($x) { git { 'vcs' } default { 'other' } }",
            "function git-status { Write-Host hi }",
            "function Invoke-Git { param($a) Write-Host $a }",
            "Get-ChildItem -Recurse -Filter *.git",
            "Test-Path .git",
            "if (Test-Path .git) { Write-Host repo }",
            "Remove-Item -Recurse .git",
            "Get-Process git -ErrorAction SilentlyContinue",
            "Stop-Process -Name git",
            "Get-Command git -ErrorAction Ignore",
            "where.exe git",
            "[System.IO.File]::Exists('.git')",
            "Write-Host (Get-Content .gitignore)",
            "Set-Alias g git",
            "New-Item -ItemType Directory git",
            "Join-Path $PWD git",
            "Start-Job { Write-Host git }",
            "& { Write-Host git }",
            "Invoke-Command -ScriptBlock { Write-Host git }",
            "Write-Host 'a' # git status",
            "# requires git\nWrite-Host hi",
            "Start-Process https://git-scm.com",
            "Start-Process notepad .gitignore",
            "Write-Host @'\ngit\n'@",
        ],
    );
}

/// **A program given directly whose name or path merely involves git.**
#[test]
fn programs_that_involve_git_in_name_or_path() {
    for program in [
        "git-bash",
        "git-bash.exe",
        "git-credential-manager",
        "git-annex",
        "git.exe.bak",
        "git~",
        "git1",
        "_git",
        "C:\\git\\bin\\python.exe",
        "/opt/git/bin/tool",
        "./.git/hooks/pre-commit",
        "https://github.com/git/git",
        "C:/Program Files/Git/git-bash.exe",
    ] {
        assert_clean(program, direct_git(&SandboxCommand::new(program).arg("x")));
    }
}
