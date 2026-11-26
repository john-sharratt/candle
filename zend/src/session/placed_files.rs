//! The knowledge base's files a projection point placed, as the projection
//! panel shows them: under the layer each belongs to, with the lines of it
//! placed and the text the model read (`docs/provenance_system_collection.md`
//! §3).
//!
//! A placed file is what its layer contributes to the prompt — a reading of
//! `code_reading`'s, a folder listing of `repo_map`'s — so the panel draws it
//! in that layer's place, not inside the system prompt it is spliced into.

use std::collections::{HashMap, HashSet};

use candle_conversation::files::names::{chunk_name, heading_name, GAP_NAME};
use candle_conversation::files::{
    files_owner, FileKey, FileRecord, FileUnit, LineSpan, PlacedFile,
};
use serde::Serialize;

use super::ZendSession;
use crate::files::records;
use crate::ingest::{IngestLayer, IngestMode};

/// One placed file as the panel draws it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlacedFileContent {
    /// The file's key, as [`FileKey::hex`].
    pub key: String,
    pub path: String,
    /// The layer the file belongs to.
    pub layer: String,
    /// A folder's listing rather than a file's text.
    pub folder: bool,
    /// The lines of it placed, in the order they sit.
    pub lines: Vec<LineSpan>,
    /// The tokens its heading and placed chunks hold.
    pub tokens: usize,
    /// Its heading and placed chunks as the model read them, the gap piece
    /// between runs that are not contiguous.
    pub text: String,
}

/// The layer whose units are `unit`s: the one that reads files, or the one
/// that lists folders. Empty when the schema declares neither.
pub fn layer_of(unit: FileUnit, layers: &[IngestLayer]) -> String {
    let mode = match unit {
        FileUnit::File => IngestMode::Files,
        FileUnit::Folder => IngestMode::Folders,
    };
    layers
        .iter()
        .find(|l| l.mode == mode)
        .map(|l| l.name.clone())
        .unwrap_or_default()
}

/// The names of `record`'s members `lines` places, in order — its heading,
/// then each chunk, the gap piece between two runs that are not contiguous.
pub fn placed_member_names(record: &FileRecord, lines: &[LineSpan]) -> Vec<String> {
    let mut names = vec![heading_name(&record.path, &record.blob)];
    for (i, &span) in lines.iter().enumerate() {
        if i > 0 && !span.continues(lines[i - 1]) {
            names.push(GAP_NAME.to_string());
        }
        names.push(chunk_name(&record.path, &record.blob, span));
    }
    names
}

impl ZendSession {
    /// `placed`, each resolved to what the panel draws; a file this process
    /// holds no record or member of is left out.
    pub fn placed_file_contents(&self, placed: &[PlacedFile]) -> Vec<PlacedFileContent> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let conversation = state.engine.conversation();
        let wanted: HashSet<FileKey> = placed.iter().map(|p| p.file).collect();
        let by_key: HashMap<FileKey, FileRecord> = records::all(&conversation)
            .into_iter()
            .filter(|r| wanted.contains(&r.key()))
            .map(|r| (r.key(), r))
            .collect();
        placed
            .iter()
            .filter_map(|p| {
                let record = by_key.get(&p.file)?;
                let read = conversation.read();
                let members: Vec<Vec<u32>> = placed_member_names(record, &p.chunks)
                    .iter()
                    .map(|name| {
                        read.owned_section_named(files_owner(), name)
                            .map(|id| read.section_tokens_of(id).to_vec())
                    })
                    .collect::<Option<_>>()?;
                drop(read);
                let ids: Vec<u32> = members.concat();
                Some(PlacedFileContent {
                    key: p.file.hex(),
                    path: record.path.clone(),
                    layer: layer_of(record.unit, &state.ingest_layers),
                    folder: record.unit == FileUnit::Folder,
                    lines: p.chunks.clone(),
                    tokens: ids.len(),
                    text: state
                        .engine
                        .tokenizer()
                        .decode(&ids, false)
                        .unwrap_or_default(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::{ChunkRecord, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;

    use super::*;

    fn span(first: u32, last: u32) -> LineSpan {
        LineSpan { first, last }
    }

    fn record() -> FileRecord {
        let member = MemberRef {
            address: ContentAddress {
                prefix_hash: ContentHash::ZERO,
                section_hash: ContentHash::ZERO,
            },
            tokens: 1,
        };
        FileRecord {
            content_key: "v1:candle/a.rs@b".into(),
            path: "candle/a.rs".into(),
            blob: "b".into(),
            unit: FileUnit::File,
            lines: 600,
            lang: "rs".into(),
            branches: vec![],
            heading: member,
            chunks: [span(1, 200), span(201, 400), span(401, 600)]
                .map(|lines| ChunkRecord { lines, member })
                .to_vec(),
        }
    }

    /// **A placement reads as its heading and chunks, a gap between runs**:
    /// contiguous chunks join, a skipped run puts the gap piece in.
    #[test]
    fn a_placement_reads_as_heading_chunks_and_gaps() {
        let r = record();
        assert_eq!(
            placed_member_names(&r, &[span(1, 200), span(201, 400)]),
            vec![
                heading_name("candle/a.rs", "b"),
                chunk_name("candle/a.rs", "b", span(1, 200)),
                chunk_name("candle/a.rs", "b", span(201, 400)),
            ]
        );
        assert_eq!(
            placed_member_names(&r, &[span(1, 200), span(401, 600)]),
            vec![
                heading_name("candle/a.rs", "b"),
                chunk_name("candle/a.rs", "b", span(1, 200)),
                GAP_NAME.to_string(),
                chunk_name("candle/a.rs", "b", span(401, 600)),
            ]
        );
    }
}
