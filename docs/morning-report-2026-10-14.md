# Morning report — 14 October 2026

BlackWorld is up and serving. WhiteWorld never accepted a session, so everything that needs two machines is still open. The console bug from last night is fixed and checked again this morning. A user conversation, an agent switch, a cancelled job, and a small Battle Cities edit all ran on the one node that is actually up.

## What is running

| Process | Pid | Role |
|---|---|---|
| tokerad | 55384 | zend and npcd, `tokerad\tokerad.yaml`, listening on 192.168.0.5:8081 and 192.168.0.5:8444 |
| web | 64924 | gateway, ports 80 and 9443 |
| cf-ddns | 12556 | left running the whole night |

tokerad was ready at 14:06 UTC after a restart. Both apps are joined to `https://192.168.0.5:9443`. zend is node `db422f70bc91e081`, npcd is node `63dabd7b1f381ee2`. The in-process agent is node `76c3a0f4d39bd9db` on BlackWorld, low priority, with checkouts for candle, battle-cities, and battle-mind. Build count 3280, hash `d4c5ecd2`. The console build id is `93f561377cce627e`.

The branch is `cursor/gateway-tiebreak-voter-f2d0`, pull request https://github.com/john-sharratt/candle/pull/1. The last commit on it is `3fb6b0c1` (status reads no longer wait out the writer). Nothing new was committed overnight after that. The working tree's only leftover is `.claude/scheduled_tasks.lock`, which was already dirty and was not touched.

## The console

`GET /v1/status` through the gateway used to wait until the substrate writer finished every seal the cast had in flight. The gateway gives a member 3 seconds. npcd's writer was behind those seals, so `bot.tokera.com` came back HTTP 200 with a member status of 504, and the page painted "Connecting".

A read now ends when its body is written. A request that appends records still waits for those records. After the new binary was up, both consoles returned a ready member in well under a second (code about 1–26 ms, bot about 25 ms).

Checked again after the later restart: direct status 4 ms, `code.tokera.com` 99 ms with member status 200 and state ready, `bot.tokera.com` 143 ms the same way. One sample during a compaction returned the 3 second deadline on both consoles while a direct status still succeeded. The next sample, a few seconds later, was ready again. An open tab reloads once when the build id changes, then should stay.

## WhiteWorld

Nothing on this night reached WhiteWorld.

The worker is connected and idle (`0474bb94-291b-4d49-b648-a7fd93d03b20`, machine WhiteWorld, checkout `C:\Users\johna\prog\candle`). From BlackWorld, RDP port 3389 is open. Ports 22, 445, 5985, and 8081 are not. The desktop session there is disconnected, so a remote-desktop window from this session never logs in. A local task cannot name that worker; two attempts that were aimed at it ran here instead and stopped: [WhiteWorld disk and checkout](bc-c4018435-582a-51d3-a135-1a69fff7d18d) and [Confirm WhiteWorld hostname](bc-d6535a83-519a-5188-97d2-33f62e932866). The account's composer start call refuses private workers ("Private workers are not enabled for your account"). There is no user API key in the environment.

So these items were not done, and they cannot be done until a session is actually on that machine:

- Free 300 GB for a substrate. Disk was never inspected.
- Build and start tokerad there.
- Split the ingest backlog across two members, and watch the substrate replicate.
- Take one side down and confirm the other still serves a conversation (the tie-break voter is in the binary and in the design, and it is deployed on BlackWorld, but with one member there is no second side to lose).
- Move NPC leases onto the other node when a user conversation lands on BlackWorld.
- Canary a new binary and watch it replicate, then put the current binary back on both machines.

## Ingest

Only BlackWorld is ingesting. After the first restart of the night the backlog was replanned at 2 of 333, last item `candle/README.md`, and it had reached 4 of 333 (`candle/candle-kernels/build_utils.rs`) before the process was restarted again. After that second restart it was replanned at 0 of 328. That replan is the startup walk, not a second copy of files that had already finished. There is no second member to share it with, and nothing to replicate to.

## Agents

Signed in through the gateway as you, with a session minted on this machine for the tests. It was not written into the repo and the auth config was not changed.

The one agent was switched off and back on, twice (before and after the restart). Off, the listing says `disabled` and `enabled: false`. On, it says `ready` and `enabled: true`. Both calls returned in a few milliseconds.

A conversation (`overnight-cancel-2`) was told to start `node -e` with a timer that does not exit, in battle-cities, and then stop. The agent showed job `q1NrFI3k37I` as running (`node -e`). `DELETE /v1/agents/76c3a0f4d39bd9db/jobs/q1NrFI3k37I` returned 200. The conversation's reply contains the failure the model was given:

> The call failed: `run_command` returned `{"error":"sandbox_failed","detail":"the job failed on its agent: the job ended Cancelled with no outcome"}`.

That is the existing path: a cancel aborts the job, the run has no outcome, and zend reports `sandbox_failed` into the conversation. The model can read it and try again. The wording is the status name (`Cancelled`) rather than a sentence of its own. It was not rewritten; the path already does what you asked.

## Battle Cities edit

Conversation `overnight-edit-1` called `file_edit` on `battle-cities` `README.md`, inserting `<!-- overnight-check -->` after the title line. The reply was `edited`, with no tool error. No build was run. The git working tree at `D:\prog\battle-cities\README.md` does not contain that line: the edit is in the conversation's store, which is where an uncommitted zend edit lives. Nothing was pushed.

## NPCs

Two characters were live on BlackWorld the whole time a decode was possible: `16110025551906366116` and `16620653753833005829`. They recalled, took missions at the command table (coolant valve, then breaker panel), closed a chapter, and kept going. With one npcd member, rendezvous lends every character here. There is nowhere else to put them.

Moving them when a user conversation arrives is not how the design works today, and it was not changed. `docs/replicated_substrate.md` §9.5 fixes each member's weight at admission. Idle and a drifting power measurement do not move a world or a character. A character moves only when membership changes who rendezvous names, and only at that character's next turn boundary, after the holder hands it back. A holder that drops out is not lent again until it rejoins.

Doing what you described — a conversation on BlackWorld releases NPC leases and WhiteWorld takes them — needs a busy signal every member agrees on, used only for NPC lending, with some hysteresis so they do not flap. The scheduler's idle flag is the wrong signal: the cast's backlog stays high the whole time NPCs are decoding, so "idle" would be false whenever anyone is thinking. Shipping that as a wire change on one machine, with the other machine still on the old binary, would make the two sides disagree about who holds a character. It waits until both machines can run the same build.

## A fault, and the restart

The first cancel attempt (`overnight-cancel-1`) prefilled a very large prompt (about 210,000 tokens) and then the forward failed:

`model error: A weight is negative, too large or not a valid number`

That string is cuBLAS rejecting a matrix multiply (a dimension or leading dimension it will not accept). Both NPCs immediately started failing the same way and retrying every 5 seconds. Prefill waves kept being logged; decode did not recover. Expert hit rate had fallen from the mid-90s to about 40% during that prompt, and the weight zone had just conceded a large slice to KV. The error is not one of the sticky CUDA faults the daemon exits on, so the process stayed up and kept retrying.

tokerad was stopped and started again with the same config. It does not have a console, and the shutdown route wants the host name confirmed from an admin call; the stop was `Stop-Process`, not a drained shutdown. web and cf-ddns were not touched. Load took about three minutes. After ready, both NPCs were decoding again (expert hit back around 97–98%) and taking missions. The later cancel conversation and the README edit did not bring the fault back. Those later prompts were smaller (about 66,000 and 63,000 tokens) because thinking was off.

## What was deliberately not done

- No canary and no second binary. A canary needs another member to keep serving the old build while the new one takes a fraction of traffic, and afterwards both machines have to be put back on the current binary. There is no second machine. Restarting this one onto a throwaway build, after it had just recovered, would have taken the model down again for nothing that could replicate.
- No model sweep, and Battle Cities was not compiled.
- The tie-break voter and the per-file ingest lease are already on this branch and in this process. They were not re-proven against a second member.

## When WhiteWorld is reachable

1. Confirm at least 300 GB free for a fresh substrate. Clear the Hugging Face cache or move it if it is not.
2. Build this branch there and start tokerad against a new substrate, same cluster URL, same cluster name, same group key. Do not copy BlackWorld's substrate.
3. Watch the unread backlog split by content key, and watch records replicate.
4. Stop one member and send a conversation through the gateway. The other member plus the gateway vote should still serve it.
5. Then, and only on a build both sides already share, add the NPC-lending signal and check that a conversation on BlackWorld hands characters to WhiteWorld at their next turn boundary.
6. Canary a newer build, watch it install on the other machine, then put the current binary back on both.
