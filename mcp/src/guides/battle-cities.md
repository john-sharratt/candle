---
name: battle-cities
description: How to build and test the Battle Cities game on a build agent — the dedicated server and the client separately, never in one cargo call — with the test command that enforces its time limits, and what to check when a machine has never built it.
tools: [repo_job_start, job_status, job_read, job_grep, job_list, job_cancel]
---
# Building and testing Battle Cities

Battle Cities is its own repository, `battle-cities`, checked out on the build
agents beside `candle`. Jobs run on it exactly as the `test` guide says —
`repo_job_start {repo: "battle-cities", branch, program: "cargo", args}`,
then `job_status`, `job_read`, `job_grep` — so read that guide for starting,
waiting and reading logs. This one says what to run.

## Two programs, never one cargo call

The repository builds two binaries over one shared crate, `sim`:

| Binary | Package | Holds |
|---|---|---|
| the dedicated server | `battle-cities-server` | no window, no Bevy |
| the client | `battle-cities` | Bevy rendering, input, sound; its build also packs the assets the server serves |

Cargo unifies features across everything one call builds, so a call naming
both — or `--workspace` — turns the client's Bevy features on for the server
too. **Build and test them in separate jobs, each naming its own package.**
Never `--workspace`.

## The server

| Want | `args` |
|---|---|
| a compile check | `["check", "-p", "battle-cities-server"]` |
| a build | `["build", "-p", "battle-cities-server", "--release"]` |
| **its tests** | `["test-server"]` |
| some of its tests | `["test-server", "drill_tower"]` — the words after it go to nextest |

`cargo test-server` is an alias in the repository's `.cargo/config.toml`. It
runs every server-side crate's tests under cargo-nextest and then checks the
time limits, and fails when either fails:

- **each test within 0.5 s** — nextest stops one past that and reports it
  `TIMEOUT`, a failure;
- **each module's tests within 8 s together** — the run ends with
  `<binary> \`<module>\`: N tests took X s, over the 8 s budget`.

A pass ends with `N tests, every module within 8 s`. Plain `cargo test` runs
the tests with neither limit, so it is not the test run — use it only to
debug one test. Do not run `make test`: it only calls the alias, and `make` is
not on every machine.

Four tests fail on `main` and are known (the repository's fix list, B5):
`math`'s `test_frustum` and `test_overlap`, `sim`'s `test_get_joins_hit` and
`test_get_joins_miss`. A run whose only failures are those four is the
baseline, not a regression — report it so, naming them.

## The client

| Want | `args` |
|---|---|
| a compile check | `["check", "-p", "battle-cities"]` |
| a build | `["build", "-p", "battle-cities", "--release"]` |

The client builds against a fork of Bevy that compiles only on the nightly
named in `rust-toolchain.toml`, with the committed `Cargo.lock`. Never move
the toolchain, update the lock or touch the Bevy fork to make a client build
pass: report the failure to the user instead. A first client build compiles
Bevy — give it `timeout_s: 3600`.

## A machine that has never built it

An agent sets up its battle-cities checkout itself when it starts: the
repository's `.agent/setup.yaml` runs `cargo --version` there, which has
rustup install the pinned nightly, then `cargo install cargo-nextest --locked`.
Each is a run of its own on that agent — `job_list {kind: "build"}` shows them,
with the checkout's command line, before the first job you start — and your
job waits behind them. If a test job still fails for want of a tool, find the
setup run that failed on that machine and `job_read` it first:

| The log says | It lacks | Tell the user |
|---|---|---|
| `toolchain 'nightly-…' is not installed` | the pinned nightly | `rustup toolchain install <that nightly>` on that machine |
| `cargo-nextest is not installed on this machine` | cargo-nextest | `cargo install cargo-nextest --locked` on that machine |
| `no such command: \`test-server\`` | the alias: the branch predates it | run it on a branch that has it |

A setup run that failed is a machine problem the agent could not fix itself
(no network, rustup's auto-install switched off, a full disk): say which
machine, which setup command and what its log says, and wait for the user.

## Deploying

There is no deploy route for Battle Cities: neither binary is a keep service,
so `service_deploy` does not apply to it. Build it here; how a built server
or client is put live is the user's to say.

## Report

As the `test` guide says — PASS or FAIL per job with its handle and machine —
plus, for the server's tests, the budget line it ended with, and whether its
failures are exactly the four known ones.
