---
name: files
description: How to read and change the user's repositories directly, without a task — reading, editing as one commit, landing a picture or any file from a link or from bytes you hold, and previewing a Yen Bots app before it lands.
tools: [file_read, file_list, file_search, file_edit_and_commit, file_download_and_commit, file_upload_and_commit, log_read, image_generate, repo_task_start]
---
# Files

For a small, direct change — a word on a page, a colour, a new page, a
picture — the file tools are the right size; a task is not. For volume (edits
across many files, tests, a build) delegate instead (`implement`).

## Read

- `file_read {repo, path, …}` reads one file by parts or lines;
  `file_list` lists a directory; `file_search` finds files by name.
- `file_read {repo, paths: [...]}` reads up to 64 files whole in one call —
  text as `text`, anything else as `base64` — 4 MiB in all.
- They read the repository **as committed** on its main branch: what a
  commit landed a second ago, never a task's unpushed work.

## Change — every call is one commit, or nothing

| What | Call |
|---|---|
| Text edits, a new text file, a delete, a move | `file_edit_and_commit {repo, message, changes}` — `edit` replaces `old` (copied exactly, once) with `new`; `write` gives a whole file |
| A file at a public link: a picture `image_generate` drew, a font, a library | `file_download_and_commit {repo, message, files: [{url, path}]}` — Zen fetches it; its bytes never pass through you |
| Bytes you hold: a photo, a page a preview built | `file_upload_and_commit {repo, message, files: [{path, base64}]}` |

- **Read before you edit**, and copy the text you replace exactly.
- **Edits, not rewrites.** Change an existing file with `edit`; a whole-file
  `write` of a long file is minutes of output and drifts from what you meant.
- **Never retype a file you already have**: a link goes in by
  `file_download_and_commit`, bytes by `file_upload_and_commit`.
- The answer is `live` with the commit, or `refused` with why — nothing
  moved anywhere on a refusal. A repository with a gate (Yen Bots) names the
  rule, file and line it refused on; fix that and call again.
- Links are fetched from the public internet only: private and local
  addresses are refused, and a link behind a sign-in (a claude.ai artifact)
  cannot be fetched. At most 8 MiB a file and 16 MiB a call.
- The commit is the repository's own author's, never yours.

## Previewing a Yen Bots app

A change too large to type as edits, a new page or a picture is made in a
**preview**: an artifact on claude.ai that loads the app from `main` with
`file_read {paths}`, lays your changed files over it and runs it, so Yen sees
it before it lands. Its **Publish** sends the page's own files — pictures
first with `file_upload_and_commit` (or `file_download_and_commit` for a
drawn picture's link), then the text with `file_edit_and_commit` — and reads
the app's log after (`log_read {source: "yenbot/<app>"}`). Publish is Yen's
tap, never yours. Build a preview from the kit in yenbot's
`templates/preview/` — the loader and toolbar are written once, you write
only the changed files — and read `yen-apps` for the rest.

## Workers

A worker has the same move without the commit: `file_download` brings a
link's bytes into its checkout and `file_upload` sends a checkout file to a
link, neither through its context; its commits carry them.
