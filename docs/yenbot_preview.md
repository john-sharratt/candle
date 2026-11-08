# Yen Bots previews: an app on claude.ai, published from the page

**Status: design, not built.** This document is authoritative for how a Yen
Bots app is previewed on claude.ai and published from that preview. It rests
on the file tools and transfers of `docs/mcp_file_tools.md` (§2, §3, §3.6) and
on the app host of `docs/yenbot_app_host.md`, which it amends in §5.4, §16 and
§17.5.

## 1. What it is for

Yen builds her apps in a claude.ai chat. Claude writes an app's files and
lands them with `file_edit_and_commit`, and the live site reloads within two
seconds (`docs/yenbot_app_host.md` §5.4). That loop is fast for an edit and
slow for anything larger. A new page, a rewrite or a picture reaches the
repository only by Claude typing every byte into a tool call: minutes of
output for one page, the copy free to drift from what Claude meant, and no
way at all for a binary. Yen also cannot see a change before it is live.

A **preview** fixes both. It is an artifact on claude.ai, a page Claude
publishes in the chat, that runs the app as the repository holds it with
Claude's changes laid over it. Yen sees it, points at what to change and
taps **Publish**, and the page sends its own files to the repository. No byte
of a published file passes through the model.

## 2. Where it lives

The preview is the artifact. Its code runs in the viewer's browser on claude.ai
or in the Claude app, and it reaches Zen through the viewer's own Zen connector:
an artifact that declares the `mcp` capability calls the connector's tools as
the viewer, after the viewer allows it once. Nothing of the preview runs in
yenbot or in Zen.

A page that declares connectors cannot be shared by public link. A preview is
private to the people Yen shares it with, which suits it: it holds her app.

## 3. The loader and the overlay

A preview does not carry the app. It carries a **loader**, the **overlay** and
the **toolbar**:

- **The loader** reads, with one `file_read {repo: yenbot, paths}`
  (`docs/mcp_file_tools.md` §2), the framework (`web/`, the vendored modules
  under it), the app's own files (`apps/<id>/`) and the library art the app
  names, all from `main` as it stands. A binary answers as base64 and becomes
  a `blob:` URL.
- **The overlay** is what Claude changed this round: each file it wrote, by
  path, and each it deleted. It is the page's own data, so a change Claude
  makes to the preview republishes the overlay and nothing else.
- The loader lays the overlay over what it read and runs the result: the
  app's `index.html` in the page, its modules from the merged files.

So the preview always runs what is on `main` plus this round's change, Claude
writes only the files it changes, and Publish commits exactly the overlay.

### 3.1 The SDK in a preview

There is no yenbot server behind a preview: no `/api`, no store, no sign-in.
The SDK of `yen.js` (`docs/yenbot_app_host.md` §8.2) has a **preview backend**,
chosen when the page is not served by yenbot, each part of it a capability the
artifact declares:

| SDK | In a preview |
|---|---|
| `app.collection` | the artifact's `db`: the same documents and queries over a store of the preview's own, which Claude seeds from the app's fields (§7.5) with data that looks like Yen's |
| `app.me` | `user`: the viewer, as the app's account |
| `app.ai` | `sample`: Claude, answering from the page |
| `app.files` | `assets`: a photo added in the preview is kept with it |
| `app.library` | the loader's library files |
| `app.notify` | shown on the page, sent to nobody |
| `app.workflows` | listed with what each would do; none runs |

A workflow, a schedule and a notification are seen working only on the live
site. A preview shows how an app looks and behaves to the person using it.

## 4. Publish

The toolbar's **Publish** lands the overlay:

1. **New library art first**, the overlay's pictures under `library/` (the
   only place the gate takes a binary; a photo a person adds in the preview is
   the app's data, kept by `app.files`, never committed), with
   `file_upload_and_commit` (`docs/mcp_file_tools.md` §3.6). A picture an
   image tool drew is landed by its link with `file_download_and_commit`
   instead, so its bytes never reach the page.
2. **Then the text**, every text file of the overlay, with
   `file_edit_and_commit`: an `edit` against what the loader read for a
   changed file, a `write` for a new one, a `delete` for one removed.

Each call is one commit through yenbot's gate. An asset landed in step 1 that
step 2 then refuses is unused and harmless; the toolbar says what was refused,
with the gate's rule, file and line, and Publish again lands the rest.

On **live**, the page republishes itself (`artifact`) with the commit as its new
baseline and the overlay empty, and the toolbar reads the app's log since the
commit (`log_read {source: "yenbot/<id>"}`, `docs/mcp_file_tools.md` §6.2),
showing a new error the moment the live app logs it.

Publish and Claude's edit box are offered only to a viewer who can edit the
artifact (`user`). A viewer it is shared with for viewing sees the app and
nothing that changes it.

## 5. Asking for a change from the page

Yen asks for a change in two ways, neither needing the chat:

- **Pointing.** The `comments` capability opens claude.ai's comment composer on
  the part she points at. The chat's Claude reads the thread, changes the
  overlay and republishes, and answers under her comment.
- **The edit box.** She types what she wants in the toolbar. The page asks
  Claude (`sample`) with tools of its own: read a file of the merged app, and
  propose an edit to the overlay. The proposal shows in the preview at once.
  **The edit box never publishes**: Publish stays her tap.

## 6. The kit

The loader, the toolbar and the preview backend are one kit, written once and
kept in the yenbot repository beside the templates (`templates/preview/`,
`docs/yenbot_app_host.md` §16), passing the same checks as an app. Claude
does not write it. A preview is published with the kit's files copied in at
publish time, from a published kit artifact refreshed whenever the kit
changes; Claude writes only the overlay and the page's declaration.

The declaration names what the kit calls and nothing more:

```text
mcp:     Zen: file_read, file_edit_and_commit, file_upload_and_commit,
              file_download_and_commit, log_read, image_generate
db, user, sample, assets, comments, artifact
```

## 7. How Claude knows

Three sources, the last of them ours:

1. **The artifact platform's own skill** teaches the capabilities and their
   call contract, current for the contract version a page is built on.
2. **The connector's tool descriptions** teach each tool's arguments.
3. **Ours** teaches the recipe:
   - The Claude client guide (`mcp/src/guides/memory.md`, `clients:
     [claude-code, claude-ai, other]`) carries one line, at the front of the
     first call of every conversation (`docs/mcp_user_context.md` §2.4–§2.5):
     building or changing a Yen Bots app starts with `yen-apps`, and is
     previewed with the kit. Grok's guide is a different guide and does not
     carry it.
   - `yen-apps` (`docs/yenbot_app_host.md` §17.5) says when to preview, how
     to publish one from the kit, and that Publish is Yen's.
   - The `yenbot` journal (§17.6) holds what is Yen's own, read when a session
     starts on her app.

## 8. Before it is built: three checks

Each decides part of the design, and is answered by trying it:

1. **A claude.ai chat's artifact calls the connector.** The `mcp` capability is
   confirmed for artifacts Claude Code publishes; Yen works in claude.ai chat.
   Publish a one-button page from a claude.ai chat that calls `file_read`. If
   chat artifacts cannot declare it, Yen's previews are made in a claude.ai/code
   session instead and nothing else here changes.
2. **The loader can run what it read.** An artifact loads scripts only from a
   few CDNs. The loader imports modules from `blob:` URLs it made from
   `file_read`'s answer. If the page's policy refuses that, the framework is
   copied into the kit artifact and each preview carries it, and only the
   app's own files are loaded.
3. **The file tools reach claude.ai.** Answered: they do. A claude.ai
   session on 2026-11-08 had none of them only because its list predated the
   release that added them, and nothing told it the list had changed; a
   reconnect brought them in. The aggregator now tells a client when its list
   changes (`docs/mcp_app.md` §3.1).

A custom artifact type would have carried the kit and its instructions on its
own; claude.ai offers only its five built-in types, so the kit is a template.

## 9. Order of work

1. **A client is told when its tools change** (§8.3, `docs/mcp_app.md` §3.1).
2. **The address guard and the transfers** (`docs/mcp_file_tools.md` §3.6–§3.8,
   §9 step 7): the guard first, then `file_download` and `file_upload` for
   workers, then the two `_and_commit` tools and yenbot's gate rules for binary
   files; `file_read` with `paths`.
3. **The platform checks** (§8.1, §8.2).
4. **The kit**: the loader and Publish against the yenbot repository, then the
   preview backend over `db`, `user`, `sample` and `assets`.
5. **Asking from the page**: comments, then the edit box.
6. **The toolbar's log**, after Publish.
7. **The guides**: the line in the Claude client guide, the preview in
   `yen-apps`, and `docs/yenbot_app_host.md` §5.4 naming the preview as the
   way a change is seen before it lands.

## 10. Tests

- **The loader**, in the kit's own tests: an overlay replaces, adds and deletes
  files of what was read; a binary becomes a URL; a module of the overlay is
  the one imported.
- **Publish**: the overlay's text files become exactly the edits, writes and
  deletes `file_edit_and_commit` is sent, against raw expected JSON; binary
  files go to `file_upload_and_commit` first; a refusal leaves the overlay as
  it was and shows the gate's reason.
- **The preview backend**: each SDK call of §3.1 answered by its capability,
  with the capability absent answered as the SDK's own "not available here".
- **End to end**, once, against the live yenbot repository on a draft app:
  a change made in a preview is on `main` byte for byte after Publish, and the
  live app shows it.
