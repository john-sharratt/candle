//! The record of which files are already read, judged by whether each file's
//! ingest chain finished rather than by its bare content key.
//!
//! A file counts as read only once its chain has finished, because a reading
//! that ran out of decode budget still wrote its content key and would otherwise
//! count as read forever. Every gate asks the same question, so a file is never
//! announced as queued and then skipped as present. Only this member's own
//! crashed partials are retired at boot, or a peer's once its grace has passed.
//! Uploads are kept out of branch checks, since they live on no branch and their
//! absence from one is not a deletion.

use super::*;

/// Every committed `code_reading` conversation of a repository's branches
/// whose chain FINISHED: its content key and path — what the pass plans
/// against. Uploads are not among them: their endpoint keeps them, never the
/// pass.
///
/// # A file counts as read only once its chain finished
///
/// The content key is written once the file's conversation stops calling
/// tools, and that is not the same thing as answering. A README read spent its
/// whole decode budget deliberating over how many `file_read` calls to issue,
/// was cut off mid-word, and emitted no tool call — so no coupling, no second
/// turn and no summary, yet the turn came back `Ok` and the key was written.
/// The substrate stored it faithfully (CRC-clean, chunk counts consistent), the
/// file counted as read forever, and its one turn stayed in the corpus as a
/// worked example of an assistant that deliberates and produces nothing, which
/// later conversations then imitated.
///
/// So each keyed timeline is asked whether its chain finished, and its reads
/// found the file ([`reading_break`]). An unfinished one is left out of what the pass plans
/// against, so its key reads as held by nothing and the file is queued again;
/// [`process_one_file`] defers the unfinished generation and retires it only
/// once the rebuild commits — recovery with nothing lost in between.
///
/// **One predicate for every gate.** The plan (this), the resume snapshot
/// ([`finished_content_keys`]) and the fast path ([`chain_finished`]) each ask
/// "is this file already read?" from their own query. A gate left reading the
/// bare key would announce the file as queued and then skip it as present,
/// which reports recovery that never happens — or, for the fast path, hands the
/// unfinished chain to a live conversation as a file it has read.
pub fn committed(engine: &ConversationEngine) -> Vec<Committed> {
    let scan = scan_ingest_chains(engine);
    // One line per unfinished chain per pass. A handful is ordinary recovery; a
    // corpus-wide sweep of them means the decode budget or the opening prompt
    // is wrong for this model, and every pass will keep redoing the same work.
    // A peer's chain whose later turns are still on their way is not among
    // them: it is `arriving`, and left to the peer.
    for (path, why) in &scan.broken {
        if scan.finished.iter().any(|c| c.subject == *path) {
            continue; // a complete chain for the same file covers it
        }
        tracing::info!(
            target: "zend::code_read::ingest",
            file = %path,
            ?why,
            "ingest chain unfinished — the file is queued again and will be rebuilt",
        );
    }
    scan.finished
        .into_iter()
        .filter(|c| !is_upload_path(&c.subject))
        .collect()
}

/// Every keyed `code_reading` conversation a peer committed whose later turns
/// have not arrived yet ([`Whose::Peers`]): what the pass plans against beside
/// [`committed`], so a file a peer has just read is not queued here again and
/// its reading is not superseded by a second one. Not a reading this member
/// may serve — that waits for the turns ([`chain_finished`]).
pub fn arriving(engine: &ConversationEngine) -> Vec<Committed> {
    scan_ingest_chains(engine)
        .arriving
        .into_iter()
        .filter(|c| !is_upload_path(&c.subject))
        .collect()
}

/// What [`scan_ingest_chains`] found: the keyed conversations whose chain
/// finished, the peers' whose later turns are on their way, and the paths
/// whose chain stopped part-way or never read its file.
pub(super) struct IngestChains {
    pub(super) finished: Vec<Committed>,
    pub(super) arriving: Vec<Committed>,
    pub(super) broken: Vec<(String, Broken)>,
}

/// Every keyed `code_reading` conversation, uploads included, sorted by
/// whether it holds its file: its chain finished and its reads found the file
/// ([`reading_break`]).
pub(super) fn scan_ingest_chains(engine: &ConversationEngine) -> IngestChains {
    let eng = engine;
    let keys: HashMap<TimelineId, String> = eng
        .conversations_with_metadata_key(CONTENT_KEY)
        .into_iter()
        .collect();
    let paths = eng.conversations_with_metadata_key(PATH_KEY);
    let quants = quant_tags(eng);
    let conv = eng.conversation();
    let substrate = conv.read();
    let (node, now) = (conv.node(), now_ms());
    let mut out = IngestChains {
        finished: Vec::new(),
        arriving: Vec::new(),
        broken: Vec::new(),
    };
    for (timeline, path) in paths {
        let Some(key) = keys.get(&timeline) else {
            continue;
        };
        let first = substrate.first_turn_stamp(timeline.raw());
        let unit = |subject: String| Committed {
            timeline,
            key: key.clone(),
            subject,
            stamp: first.unwrap_or_default(),
            quant: quants.get(&timeline).copied(),
        };
        match reading_break(&substrate, timeline) {
            None => out.finished.push(unit(path)),
            // Still arriving: a gap is a turn whose records are on their way,
            // and a peer's reading inside its grace may be missing its last.
            Some(_) if substrate.holds_gap(timeline) || whose(first, node, now) == Whose::Peers => {
                out.arriving.push(unit(path))
            }
            Some(why) => out.broken.push((path, why)),
        }
    }
    out
}

/// Content keys whose ingest chain finished, or a peer's is arriving, uploads
/// included — the snapshot [`ingest_jobs`] probes before spending a
/// conversation on a file.
///
/// Replaces a bare sweep of every content key, which counted an unfinished
/// chain's key as present and so skipped the very file the plan had just
/// queued for rebuild.
pub(super) fn finished_content_keys(engine: &ConversationEngine) -> HashSet<String> {
    let scan = scan_ingest_chains(engine);
    scan.finished
        .into_iter()
        .chain(scan.arriving)
        .map(|c| c.key)
        .collect()
}

/// Whether `timeline`'s ingest chain finished — the fast path's gate before it
/// hands a file conversation to a live one as already read (see [`committed`]).
pub(crate) fn chain_finished(engine: &ConversationEngine, timeline: TimelineId) -> bool {
    reading_break(&engine.conversation().read(), timeline).is_none()
}

/// Whether a workspace-relative `path` (with `/` separators) lives in the
/// daemon's `uploads` repository ([`UPLOADS_REPO`]). Matched on the FIRST
/// segment only, and case-insensitively (the win32 FS is case-insensitive, so
/// an existing `Uploads/` dir still resolves to the daemon's uploads dir) — so
/// an `uploads/` folder inside a repository is NOT matched. Uploads are
/// endpoint-managed and on no branch, so the branch pass must never read their
/// absence from the branches as a deletion.
pub(crate) fn is_upload_path(path: &str) -> bool {
    path.split('/')
        .next()
        .unwrap_or("")
        .eq_ignore_ascii_case(UPLOADS_REPO)
}

/// Retire every crashed-partial `code_read` conversation, up front.
///
/// The `code_read` twin of `repo_scan::retire_crashed_partials`, and the same
/// completion protocol: `path` is written at conversation creation and the
/// content key only once the file's ingest succeeds, so `path` without a key
/// means "started, never committed". A conversation ingested before files
/// were keyed by content carries no key either, and goes the same way. An
/// attempt that fails retires its own conversation ([`process_one_file`]), so
/// what is left for here is what a crash left: the debris would otherwise
/// stay live and keep competing in the provenance gather.
///
/// **Uploads are not exempt.** An upload's conversation without a content
/// key is in no conversation's scope (`crate::retrieval_scope`) — it holds
/// nothing anything retrieves.
///
/// Called ONCE per boot from the session's ingest pre-loop, for every layer not
/// named by `--disable-layer`, before any pool or upload runs. Never called
/// from [`ingest_jobs`]: a pass can overlap a live pool, and an in-flight file
/// is indistinguishable from a crashed one by metadata alone.
///
/// **Only this member's partials.** Nothing of this member's is in flight at
/// boot, but a peer's readings are, and they replicate here begun and unkeyed.
/// A partial is retired only when this member began it, or the peer that did
/// is past [`PEER_GRACE`](crate::branch_ingest::reader::PEER_GRACE), and none of
/// its turns is still arriving ([`settles_here`]).
pub(crate) fn retire_crashed_partials(engine: &ConversationEngine) {
    let e = engine;
    let mut retired = 0usize;
    for (tl, path) in e.conversations_with_metadata_key(PATH_KEY) {
        let committed = e.has_conversation_metadata_key(tl, CONTENT_KEY);
        if committed || !settles_here(e, tl) {
            continue;
        }
        match e.tombstone_timeline(tl) {
            Ok(()) => retired += 1,
            Err(err) => tracing::warn!(
                target: "zend::code_read::ingest",
                path = %path,
                "tombstone of crashed-partial conversation failed: {err:#}",
            ),
        }
    }
    if retired > 0 {
        tracing::info!(
            target: "zend::code_read::ingest",
            retired,
            "retired code_read conversations with no content key (crashed partials, \
             or ingested before files were keyed by content) so they leave the \
             provenance gather",
        );
    }
}

/// Tombstone EVERY `code_reading` conversation, committed or not —
/// `--wipe-layer code_reading`'s targeted counterpart to
/// [`retire_crashed_partials`], which only removes the never-committed half.
///
/// **Uploads are exempt.** They live in the endpoint-managed uploads folder,
/// on no branch, so no pass reads them again: tombstoning one would delete
/// uploaded content with nothing to rebuild it from.
///
/// Called from the session's ingest pre-loop, before the background ingest
/// worker's first pass — which then finds nothing committed and reads every
/// file as new, exactly as it would on a truly fresh install. Unlike
/// `--wipe-substrate`, every other layer's content survives untouched.
pub(crate) fn wipe_layer(engine: &ConversationEngine) {
    let e = engine;
    let mut wiped = 0usize;
    for (tl, path) in e.conversations_with_metadata_key(PATH_KEY) {
        if is_upload_path(&path) {
            continue;
        }
        match e.tombstone_timeline(tl) {
            Ok(()) => wiped += 1,
            Err(err) => tracing::warn!(
                target: "zend::code_read::ingest",
                path = %path,
                "--wipe-layer code_reading: tombstone failed: {err:#}",
            ),
        }
    }
    tracing::info!(
        target: "zend::code_read::ingest",
        wiped,
        "--wipe-layer code_reading: every code_reading conversation tombstoned \
         (uploads kept) — the next pass re-ingests the whole layer",
    );
}
