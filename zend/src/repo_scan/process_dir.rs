//! One folder into the knowledge base (`docs/provenance_system_collection.md`
//! §12 D14): its listing — its immediate children at the commit it was found
//! on, repository-relative, a folder ending in `/`, a file with its size — cut
//! and sealed as members as a file's text is, and recorded. The record of
//! what it replaces is retired only once its own has committed, so a failed
//! attempt leaves the earlier listing placeable.

use super::*;

/// Seal `job`'s folder: list it at its commit, seal the listing as the
/// knowledge base's members, write its record, and retire the records of
/// the folder no branch lists that way any more (`live` is every folder key
/// a branch holds).
pub(super) fn process_one_dir(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    job: &UnitJob,
    live: &HashSet<String>,
    failures: &Failures,
) -> anyhow::Result<()> {
    let unit = &job.unit;
    let text = match listing_at(&ctx.tool_ctx, job) {
        Ok(text) => text,
        Err(e) => {
            let n = failures.record(&unit.dir, format!("the folder could not be listed: {e:#}"));
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                "skip: the folder could not be listed at its commit: {e:#}",
            );
            if n > MAX_DECODE_FAILURES {
                failures.set_abort();
            }
            return Ok(());
        }
    };
    let conversation = ctx.engine.conversation();
    let superseded: Vec<FileRecord> = records::of_path(&conversation, &unit.dir)
        .into_iter()
        .filter(|r| r.content_key != unit.key && !live.contains(&r.content_key))
        .collect();
    let blob = unversioned(&unit.key).to_string();
    let source = FileSource {
        content_key: &unit.key,
        path: &unit.dir,
        blob: &blob,
        unit: FileUnit::Folder,
        lang: "",
        branches: &job.branches,
    };
    let record = match ingest_text(base, &source, &text) {
        Ok(record) => record,
        Err(e) => {
            let msg = format!("{e:#}");
            if crate::ingest_cancel::cancelled() {
                return Ok(());
            }
            if is_stall_message(&msg) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "folder seal stalled (sealed again next pass): {msg}",
                );
                ingest_holdoff::cool_down(&unit.dir, &msg);
                return Ok(());
            }
            let n = failures.record(&unit.dir, msg.clone());
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                "folder seal failed (sealed again next run; the earlier listing kept): {msg}",
            );
            ingest_holdoff::cool_down(&unit.dir, &msg);
            if n > MAX_DECODE_FAILURES {
                failures.set_abort();
            }
            return Ok(());
        }
    };
    if let Err(err) = records::put(&conversation, &record) {
        failures.record(&unit.dir, format!("the record was not written: {err:#}"));
        return Ok(());
    }
    ctx.retrieval.mark_stale();
    for old in &superseded {
        if let Err(err) = records::retire(&conversation, old) {
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                "retiring the superseded listing failed: {err:#}",
            );
        }
    }
    Ok(())
}

/// The listing of `job`'s folder at the commit it was found on.
fn listing_at(tools: &ToolContext, job: &UnitJob) -> anyhow::Result<String> {
    let unit = &job.unit;
    let files = tools
        .files
        .fresh_at(&unit.repo, &job.at)
        .ok_or_else(|| anyhow::anyhow!("its repository is not read through git"))?;
    let store = files.repo(&unit.repo)?;
    let base = store
        .base()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("its repository has no branch"))?;
    let tree = store
        .tree_at(&base)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("its tree could not be read"))?;
    Ok(listing_text(
        &tree.children(inner_dir(&unit.repo, &unit.dir)),
    ))
}

/// `dir` (`"{repo}/a/b/"`) inside its repository, as a tree names it:
/// `"a/b"`, and `""` for the repository's root.
fn inner_dir<'a>(repo: &str, dir: &'a str) -> &'a str {
    dir.strip_prefix(repo)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or("")
        .trim_end_matches('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_is_named_inside_its_repository_as_its_tree_names_it() {
        assert_eq!(inner_dir("candle", "candle/zend/src/"), "zend/src");
        assert_eq!(inner_dir("candle", "candle/"), "");
    }
}
