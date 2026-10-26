---
name: substrate
description: How to read what the stored analysis holds, find out whether a bad answer comes from it by switching a layer off and asking again, and delete a corrupt conversation only with approval.
tools: [layer_read, layer_toggle, timeline_read, timeline_delete, repo_task_start, repo_task_wait]
---
# Stored analysis

Analysis of the repositories is kept in **layers** (`repo_map`,
`code_reading`, `dependency_analysis`, `code_review`, `security_review`,
`dialogue`); each piece of a layer is a stored conversation, a **timeline**.
Workers read from them. A wrong answer can come from a wrong timeline.
Admins only.

## Read what a layer holds

Reading changes nothing:

1. `layer_read {name}` lists its timelines, each with its id, label,
   repository, turns and tokens, a page at a time.
2. `timeline_read {id}` for one timeline's turns, first line each;
   `timeline_read {id, turn}` for one turn in full.

## Find whether a layer causes a bad answer

1. **Switch the suspect layer off** — `layer_toggle {name}` — and ask the same
   question again (a fresh `repo_task_start`). This changes what every worker
   reads, so say so to the user first; it is held in memory and a restart
   switches it back on. If the answer is now right, the layer is the source.
2. Read the layer and the suspect timeline, as above.
3. Switch the layer back on: `layer_toggle {name}` again.

## Repair

`timeline_delete {id, confirm}` deletes a conversation whose content is wrong,
on every member; one a layer's analysis built is built again by the next
analysis pass. It cannot be undone — ask the user, with what you found, and
pass `confirm: "approved by the user"` only with approval.
