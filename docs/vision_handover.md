# Image turns — handover to WhiteWorld

State as of 2026-11-25, main at `fac1ad2d8`. The design lives in
`docs/flash_next_vision.md` (the pipeline, the pack, tenancy, the API) and
`docs/vision_models.md` (the plug-in surface, rows per tower, tests). This
note is what is built, what is proven, and what is left, in the order to do
it.

## 1. What is built

- **Two models read images:** Qwen3.8 Flash-Next (the cluster's model) and
  Qwen3.6-35B-A3B. A model plugs in with a `VisionTowerSource` constant
  (`qwen_vision/source.rs`), a `ModelSpec::vision` entry, and its engine's
  wave hook (`qwen35/forward/images.rs` + the shared
  `qwen_vision/wave_images.rs`). No new kernel: rows are placed by the tuned
  `rows_scatter`.
- **The tower ships as a twinned model pack**, so placement is a straight
  copy: 150–190 ms for 700 MiB, drain about 720–760 ms.
- **Storage:** one record per image, `vision/image/<id>`: original,
  thumbnail (480 px JPEG) and `"rows": {<tower>: <sha>}`. Every blob is named
  back by `vision/blob/<sha>/<id>` and retired only when nothing names it.
- **Tenancy:** holders (`vision/held/…`), uses keyed by **timeline**
  (`vision/used/<id>/<tl>`, `vision/timeline/<tl>/<id>`), noted before the
  turn reads the images. Every timeline delete path releases them. Unused
  images are retired 24 h after `kept_ms`, which a re-upload renews. The
  sweep runs hourly in `zend/src/session/image_sweep.rs`, never on a request.
- **API / GUI:** `POST /v1/images`, `GET /v1/images/{id}` and `/thumb`
  (holder or admin only), image parts in `/v1/chat/completions`, pictures in
  `zen_chat_send`. The GUI shows thumbnails, opens the original on a click,
  and falls back to the original when an image has no thumbnail.

## 2. What is proven

| What | Where | Result |
|---|---|---|
| Tower tests, both towers (CPU, card, guest arena) and the 7 `wave_images` GPU tests | BW, `f88078fc8`, `motherd` stopped | 29 passed |
| Vision guest, both towers | same | 1 passed |
| 35B end to end, `zend/tests/vision_qwen36.rs` | same | 3 passed: "row 2, column 4", "row 5, column 1", and the kept-id rerun |
| Review fixes, CPU (`vision::`, zend `images`, the 35B hook) | BW agent, `vision-review` | 27 + 7 + 1 passed; clippy `-D warnings` clean |
| Live Flash-Next, MCP picture turns | BW 5530/5531 | right answers ("LANE 14", "SPICE 3") |
| `cd6fcfa81` deploy | BW keep trial | proven; a candidate, **not certified** |

**`fac1ad2d8` (the review fixes) is merged but not deployed**, and its
changed paths have not run live.

## 3. The review, and what was done with each finding

1. **Images kept before `kept_ms` existed are swept as stale; ones kept
   before thumbnails 500 on `/thumb`.** Not changed in storage, because the
   repo takes no backward compatibility. The pre-tenancy images, this
   session's test pictures, were already swept live. The GUI now falls back
   to the original when a thumbnail fails to load.
2. **A re-upload didn't renew `kept_ms`, so the sweep could retire the image
   just returned.** Fixed: `vision::touch` runs on every `POST /v1/images`,
   and the sweep re-judges each image just before it retires it.
3. **`retire_blob` on shared content-addressed blobs.** Fixed with
   `vision/blob_refs.rs`; a blob held before any image named it is named
   `outside` and never released.
4. **Race between a delete and a turn that has read but not yet noted the
   image.** Fixed locally: `note_use` now runs before `resolve`. Across
   members, the replication window remains. It is documented in
   `flash_next_vision.md`.
5. **`DELETE /v1/substrate/zend/timeline/{tl}` never released images.**
   Fixed: tenancy is keyed by timeline, and both delete paths call
   `release_images`.
6. **A replay took whichever tower's rows were kept and could fail a whole
   wave.** Fixed: rows are kept per tower, `Sequence` carries the engine's
   `VisionTowerSource`, and `stored_spans` reads that tower's rows, refusing
   for that sequence alone.
7. **Tower tests downloaded about 1.8 GB in a plain `cargo test`.** Fixed:
   the three tower tests and the guest test are `#[ignore]`d.
8. **A full sweep on every upload.** Fixed: moved to the hourly task.
9. **The 35B hook read host ids twice per wave.** Fixed: one pass, and a
   wave with no virtual id answers from it. One host pass over each wave's
   prefill ids remains for every hybrid model (a CPU copy, not a device
   transfer).
10. **Prepare (decode + thumbnail) ran before the kept-rows check.** Fixed:
    the id is the file's digest, kept rows are checked first, and the file is
    decoded only when the tower must read it.

Records written by `cd6fcfa81` or earlier store `rows` as a string. Under the
per-tower map they read as "no rows for this tower", so the original is read
again (correct). Their old rows blob is never released (a few test images).
This is not migrated, by the no-backward-compatibility rule.

## 4. What is left, in order

1. **Deploy `fac1ad2d8` to BW** (`service_deploy {host: BlackWorld, service:
   motherd}`), then live-check:
   - a fresh picture through MCP `zen_chat_send` answers right;
   - closing that chat logs `its images retired`, with `timeline=`;
   - `/v1/images/{id}/thumb` answers 200 for a new upload;
   - after an hour, the log shows the sweep (`unused kept images retired`),
     or nothing when nothing is stale.
   Use a **new** picture: kept rows are reused and no drain runs otherwise.
2. **The 35B's 47.5 MiB gap-launch buffer.** From the 35B test run:
   `held_ground had no room for a buffer inside a forward,
   site=candle-nn/src/kv_cache/chunked/gap/launch.rs:305:16,
   bytes=49852672, capacity=33554432`. It went to the driver outside the
   span. Find whether images cause it (image prefill) or whether any long 35B
   prefill does. Then price it into the forward's plan, or size the held
   ground from the model (memory note "Held Ground Elimination" is related).
   Reproduce with the zend test below and read its WARN lines.
3. **`no_ticket_mib=34` in the tower's `run()`.** Something in the tower's
   forward allocates outside the guest arena; it was not the load or the
   repack. Add a check to `each_vision_guest_reads_an_image_from_its_ground`
   that fails on any allocation outside the arena (the guest domain's
   no-ticket counter), and narrow stage by stage.
4. **HTTP-API conversation path:** an image turn through
   `/v1/chat/completions` on a web conversation id (data URL and kept id),
   checked in the history.
5. **GUI in a browser:** upload, thumbnail in the bubble, click opens the
   original, Escape closes it, an old image without a thumbnail draws the
   original.
6. **MCP "another member holds this task's turns"** during a member restart:
   routing, not vision; reproduce by restarting one member mid-send.
7. **Certification** of the deployed build is John's decision. Ask; don't
   decide.

## 5. How to run the tests

The card must be free: `motherd` stopped on that machine, which **needs
John's approval**. On BW, stopping `motherd` also stops BW's agent, so the BW
Claude session runs the commands locally. Use one cargo process at a time,
`--test-threads=1`, and log to a file with the exit code.

```text
cargo test --release -p candle-transformers --features cuda --lib qwen_vision -- --include-ignored --test-threads=1 --nocapture
cargo test --release -p candle-conversation --features hub --lib guest::vision -- --include-ignored --test-threads=1 --nocapture
cargo test --release -p zend --test vision_qwen36 -- --include-ignored --test-threads=1 --nocapture
```

CPU only (safe with `motherd` running):

```text
cargo test --release -p candle-conversation --features hub --lib vision:: -- --test-threads=1
cargo test --release -p zend --lib images -- --test-threads=1
cargo test --release -p candle-transformers --features cuda --lib qwen35::forward::images -- --test-threads=1
```

## 6. Rules that bind this work

- Commit and push when done, staging paths explicitly; no AI attribution.
- Visible rollouts: post the machine, handle, commit and each step to John and
  both peers.
- A `service_control` stop, and certification, need John's word.
- No env flags, no stubs, one concern per file, raw-value tests (AGENTS.md).
