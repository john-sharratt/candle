# Skills for Claude and zend

The workflows Claude follows to use zend through its MCP endpoint
(`docs/claude_zend_gateway.md` §7, §9). One folder per skill, each a
`SKILL.md` whose frontmatter holds only `name` (matching its folder) and
`description` — anything else is refused on upload to claude.ai.

| Skill | Does |
|---|---|
| `zend-thread` | Keeping a zend thread: attach, note, delegate, wait, answer — what every other skill follows |
| `zend-implement` | Implement a written spec on a branch, with tests, and push it |
| `zend-review` | Review code, findings checked by a verifier part |
| `zend-research` | Research a question on the web, every claim cited |
| `zend-ask` | Answer a question about the code, every claim cited |
| `cluster-status` | Summarise how the cluster stands |
| `cluster-logs` | Find what the logs say about a symptom |
| `zend-test` | Build a branch and run its tests on an agent |
| `zend-deploy` | Build a commit on main, stage it, and trial it through keep on one machine |
| `zend-release` | Test, trial, certification and rollout, with the user's approval at each decision |
| `zend-rollback` | Roll the cluster back to a named build, with the user's approval |
| `cluster-care` | Drain, compact, catch up or rejoin a member, and confirm it came back |
| `zend-watch` | Watch a rollout or the logs, reporting only exceptions and the end |

## Installing

- **Claude Code**: copy each folder to `~/.claude/skills/<name>/`, or enable
  them on claude.ai and let the sign-in sync bring them down.
- **claude.ai**: zip each folder (the folder at the zip's top level) and
  upload it under Customize › Skills, with code execution turned on.

`zend/tests/skills.rs` holds each skill to the upload rules.
