//! One file into the knowledge base (`docs/provenance_system_collection.md`
//! §7): cut, sealed as members, recorded — and the record of what it replaces
//! retired only once its own has committed.
//!
//! The file's earlier version stays placeable until the new one commits: a
//! failed, cancelled or interrupted attempt leaves the earlier record in place.
//! A file the card had no room for goes back through admission whole.

use super::*;
use crate::files::lead::lead_of;

/// Ingest one file: skip it when its key is already recorded with its members
/// sealed here, or it is binary; otherwise cut it, seal its members behind
/// the knowledge base's prefix ([`crate::files::ingest::ingest_text`]), write
/// its record, and retire the records of its path no branch holds any more.
/// A record already written whose members this process has not sealed is
/// sealed again — its members restored from their persisted streams — and
/// written back unchanged (§12 D15).
pub(super) fn process_one_file(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    job: &Job<'_>,
    present_keys: &HashSet<String>,
    failures: &Failures,
) -> anyhow::Result<()> {
    let file = job.file;
    if present_keys.contains(job.key) {
        tracing::debug!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "skip: file already recorded and sealed",
        );
        return Ok(());
    }
    if job.binary.lock().unwrap().contains(job.key) {
        return Ok(());
    }
    // Another member that planned this same unread file holds its lease and is
    // the one that seals it. The record replicates, so this pass leaves the
    // file. A lease just handed back is sealed here only when that member's
    // record has arrived and the key is still absent.
    let hold = match ingest_lease::share(job.key) {
        Share::Theirs => {
            tracing::info!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "left to the member holding its ingest lease",
            );
            job.left.store(true, Ordering::Relaxed);
            return Ok(());
        }
        Share::Mine(hold) => hold,
    };
    if hold.recheck() && sealed_keys(ctx.engine).contains(job.key) {
        return Ok(());
    }
    let Some(tools) = file_tools(&ctx.tool_ctx, file) else {
        failures.record(
            &file.path,
            "its repository is not read through git".to_string(),
        );
        return Ok(());
    };
    // Content guard: an allowlisted extension does NOT guarantee text. A
    // compiled fatbin / object dump committed as `*.txt` clears both the
    // extension gate and the size gate. Sniffed from the blob before anything
    // is sealed, and remembered, so it is sniffed once.
    let (repo, inner) = split(&file.path);
    let bytes = tools
        .files
        .repo(repo)
        .ok()
        .and_then(|store| store.read_bytes(inner).ok().flatten());
    let text = match bytes {
        Some(bytes) if is_binary_sample(&bytes) => {
            tracing::debug!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "skip: binary content behind a text extension",
            );
            job.binary.lock().unwrap().insert(job.key.to_string());
            // Recorded, so no later pass, restart or member plans it again.
            let conversation = ctx.engine.conversation();
            if let Err(e) = binary_mark::record(&conversation, job.key) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "could not record the file as binary; the next pass sniffs it again: {e:#}",
                );
            }
            return Ok(());
        }
        Some(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        None => {
            failures.record(
                &file.path,
                "the file could not be read at its commit".to_string(),
            );
            return Ok(());
        }
    };

    // The records of this path this one replaces once it commits: every other
    // version no branch the caller holds current carries. A version another
    // branch still holds is left alone.
    let conversation = ctx.engine.conversation();
    let superseded: Vec<FileRecord> = records::of_path(&conversation, &file.path)
        .into_iter()
        .filter(|r| r.content_key != job.key && !job.live.contains(&r.content_key))
        .collect();

    let blob = file.blob.to_string();
    let source = FileSource {
        content_key: job.key,
        path: &file.path,
        blob: &blob,
        unit: FileUnit::File,
        lang: extension_of(&file.path),
        branches: &file.branches,
    };
    // Sealed behind its repository's listing, so each section is computed
    // as a part of that repository (`files::lead`).
    let lead = lead_of(&conversation, &file.path, &file.branches);
    let record = match ingest_text(base, &source, &text, &lead) {
        Ok(record) => record,
        Err(e) => {
            sealing_failed(job, failures, &file.path, superseded.len(), e);
            return Ok(());
        }
    };
    if let Err(err) = records::put(&conversation, &record) {
        // Not recorded: no pass or projection finds it, and the earlier
        // version stays placeable, exactly as on the failure path.
        tracing::warn!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "failed to write the file's record: {err:#}",
        );
        failures.record(&file.path, format!("the record was not written: {err:#}"));
        return Ok(());
    }
    tracing::debug!(
        target: "zend::code_read::ingest",
        file = %file.path,
        chunks = record.chunks.len(),
        tokens = record.tokens(),
        "file sealed into the knowledge base",
    );
    ctx.retrieval.mark_stale();
    ingest_rate::note_committed();

    // The replaced versions go now that this one is recorded.
    for old in &superseded {
        if let Err(err) = records::retire(&conversation, old) {
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "retiring the superseded record failed: {err:#}",
            );
        }
    }
    Ok(())
}

/// A failed seal, classified as the pool's cap counts it: a shutdown cancel
/// and a refusal for want of ground are no failure of the file, a stall is
/// the card's; anything else counts against the pass's cap.
fn sealing_failed(
    job: &Job<'_>,
    failures: &Failures,
    path: &str,
    superseded_kept: usize,
    e: anyhow::Error,
) {
    let msg = format!("{e:#}");
    // A graceful shutdown latched the cancel flag: this is the prefill wait
    // unwinding, not a failure. The file is sealed next run.
    if crate::ingest_cancel::cancelled() {
        tracing::debug!(
            target: "zend::code_read::ingest",
            file = %path,
            "shutdown cancelled the seal mid-file; sealed again next run",
        );
        return;
    }
    // **No ground is not a bad file.** The device refused the file's prefill
    // for want of room the rest of the round is holding, and gives back as it
    // finishes: the file goes back through admission after the round.
    if let Some(resubmit) = job.resubmit {
        if is_ground_refusal_message(&msg) {
            tracing::info!(
                target: "zend::code_read::ingest",
                file = %path,
                "file seal found no ground; resubmitted after this round: {msg}",
            );
            resubmit.store(true, Ordering::Relaxed);
            return;
        }
    }
    if is_stall_message(&msg) {
        tracing::warn!(
            target: "zend::code_read::ingest",
            file = %path,
            superseded_kept,
            "file seal stalled (sealed again next pass; the earlier version kept; \
             not counted against the pass's cap): {msg}",
        );
        ingest_holdoff::cool_down(path, &msg);
        return;
    }
    let n = failures.record(path, msg.clone());
    tracing::warn!(
        target: "zend::code_read::ingest",
        file = %path,
        superseded_kept,
        "file seal failed (sealed again next run; the earlier version kept): {msg}",
    );
    ingest_holdoff::cool_down(path, &msg);
    if n > MAX_DECODE_FAILURES {
        tracing::error!(
            target: "zend::code_read::ingest",
            n, cap = MAX_DECODE_FAILURES,
            "code_read ingest stopping early: failure cap reached (last: {msg})",
        );
        failures.set_abort();
    }
}

/// The extension `path` ends in, the record's `lang`; empty for none.
fn extension_of(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => ext,
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_files_lang_is_its_extension() {
        assert_eq!(extension_of("candle/zend/src/main.rs"), "rs");
        assert_eq!(extension_of("candle/docs/a.b/README.md"), "md");
        assert_eq!(extension_of("candle/Makefile"), "");
        assert_eq!(extension_of("candle/.gitignore"), "");
    }
}
