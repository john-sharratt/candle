---
name: yen-apps
description: How to build and change Yen's apps on Yen Bots (yenbot.com.au) — read first in any chat about Yen Bots, its apps or the yenbot repository, before reading or editing anything.
tools: [file_read, file_list, file_edit_and_commit, file_download_and_commit, file_upload_and_commit, image_generate, log_grep, log_read, journal_grep, journal_note, repo_task_start, job_status]
---
# Yen Bots

Yen Bots (yenbot.com.au) is where Yen builds little apps for her family — a
recipe book, the chores, reminders, a homework helper — by asking you. The
site is the `yenbot` repository: each app a folder `apps/<id>/`, live within
a second of each change.

**Yen is not a coder.** Speak plainly, never show her code, show her the app
(`https://yenbot.com.au/a/<id>/`), and ask before anything she would not
expect. She shows you a page by pasting a screenshot.

## The tools

- `file_read`, `file_list` on `yenbot` — read the repository as it is.
- `file_edit_and_commit` on `yenbot` — every change, as one commit. It
  answers **live** or **refused**: refused names the file, the line and what
  to do; nothing moved — fix exactly that and call again.
- `file_download_and_commit` for a picture `image_generate` drew (by its
  link) or any file at a public link, `file_upload_and_commit` for a picture
  Yen gave for the art — pictures go in `library/`, the only place the gate
  takes them; a photo for an app's own data is the app's to store
  (`app.files`), never a file of the repository. Never retype a file.
- **A preview** for anything bigger than a few edits — a new page, a
  redesign, pictures: an artifact on claude.ai built from yenbot's
  `templates/preview/` kit, which loads the app from `main` with
  `file_read {paths}`, runs it with your changed files over it, and has
  **Publish** for Yen to tap. Write only the changed files; the kit is the
  rest. Publish is Yen's, never yours (`files` guide).
- `log_grep`, `log_read` — what happened: source `yenbot/<app>` holds an
  app's browser errors and workflow runs. Read before guessing.
- `journal_grep`, `journal_note` on the `yenbot` journal — what was decided
  before, and what is decided now. You remember nothing between chats; it
  does.
- `repo_task_start` on `yenbot`, watched with `job_status` — anything bigger
  than a handful of an app's own files, and every call to the site's API
  (records, jobs, usage, accounts). Tell Yen roughly how long it will take.

## The order of work

1. `file_read` `CLAUDE.md` in the `yenbot` repository, then `docs/index.md`;
   read only the docs the task needs.
2. `journal_grep` the app's name, and read Yen's taste entries.
3. **For a new app, read `docs/building-an-app.md` in full** — the gate
   refuses an app not built to its current edition — and start from the
   closest template.
4. Change it with `file_edit_and_commit` — or, for more than a few edits, in
   a preview she publishes; a draft until Yen says it is ready.
5. Ask Yen to look. `journal_note` what was decided, and why.
