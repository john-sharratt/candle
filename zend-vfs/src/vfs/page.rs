//! Paging a file into fixed-size pages of lines, streamed rather than held whole.
//!
//! A page is `PAGE_LINES` lines, the same size as the excerpts the rest of the
//! system already works with, so a live read and a prefilled excerpt are the same
//! kind of object. A request past the end clamps to the last page instead of
//! failing. The file is streamed line by line and split on `\n` only, so a
//! trailing carriage return on a CRLF file stays part of its line. A file that is
//! not valid UTF-8 is discovered while streaming and refused.

use super::*;

/// Lines per [`VfsStore::read_page`] page.
///
/// The same size the rest of the system already treats as one excerpt, so a
/// live read and the prefilled excerpts the model was conditioned on are the
/// same kind of object — a scope, not a module. `zend`'s `repo_scan::anchor`
/// bounds its anchor excerpts at 200 by the identical `start + LIMIT - 1`
/// clamp, and the `code_reading` ingest carves scopes at 150
/// (`MAX_SCOPE_LINES`), so every `file_read` exchange in the corpus already
/// fits inside this cap and none had to be re-cut for it.
///
/// Chosen this small on measurement, not guesswork: one `file_read` of a
/// 2,499-line module at a wider size put 144 KB into a live conversation, and
/// three such reads made the next turn a 53,288-token prefill — three minutes
/// and fifty seconds of wall clock for one turn with the KV pool ratcheted 6
/// GB against a card already at 99%.
pub const PAGE_LINES: u32 = 200;

/// One page of a file's lines, 0-based.
#[derive(Debug)]
pub struct PageResult {
    /// The page actually returned. Clamped into range the same way the file
    /// tools clamp a listing page: a request past the end yields the last page
    /// rather than an empty one.
    pub page: u32,
    /// First line of the page, 1-based.
    pub start_line: u32,
    /// Last line of the page, 1-based and inclusive. `0` for an empty file.
    pub end_line: u32,
    pub total_lines: u32,
    pub total_pages: u32,
    /// The page's lines, joined by `\n` — never the whole file.
    pub body: String,
}

impl VfsStore {
    /// One [`PAGE_LINES`]-line page of a file, resolved through the overlay the
    /// same way [`Self::read`] is. `Ok(None)` means the path does not exist in
    /// either layer.
    ///
    /// A folder's file is streamed line by line rather than materialised
    /// (`file_read` used to read the whole file into a `String` and a
    /// `Vec<&str>` slice of it just to return 300 lines); a branch's blob,
    /// already bounded by [`MAX_LOWER_FILE_BYTES`], is paged from its text. A
    /// file the session changed is replayed from its deltas once and paged
    /// from that text.
    pub fn read_page(&self, path: &str, page: u32) -> Result<Option<PageResult>, VfsError> {
        let norm = Self::normalize(path);
        Self::guard(&norm)?;
        let view = self.view()?;
        if self.holds(&norm) {
            let Some(text) = self.current(&view, &norm)? else {
                return Ok(None);
            };
            return Self::paginate(std::io::Cursor::new(text.as_bytes()), page)
                .map(Some)
                .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")));
        }
        view.read_page(&norm, page)
    }

    /// Stream `reader` line by line and return page `requested_page`
    /// ([`PAGE_LINES`]-line strides, 0-based), without ever holding more than
    /// two pages' worth of lines at once.
    ///
    /// A request past the end clamps to the last page — the same "over-shoot
    /// reads as the tail, not an error" rule the file tools' paging applies to
    /// `file_list` — which is why the last [`PAGE_LINES`] lines are
    /// held in `tail` the whole way through: by the time EOF says the request
    /// was out of range, the candidate page's lines are long gone, and a
    /// second pass would cost exactly the whole-file read this exists to avoid.
    pub(super) fn paginate(
        mut reader: impl std::io::BufRead,
        requested_page: u32,
    ) -> std::io::Result<PageResult> {
        let want_start = u64::from(requested_page) * u64::from(PAGE_LINES);
        let want_end = want_start + u64::from(PAGE_LINES);

        let mut candidate: Vec<String> = Vec::new();
        let mut tail: std::collections::VecDeque<String> =
            std::collections::VecDeque::with_capacity(PAGE_LINES as usize + 1);
        let mut total_lines: u64 = 0;
        let mut buf = String::new();
        loop {
            buf.clear();
            // `read_line` requires valid UTF-8 (an `io::Error` otherwise), which
            // is the same "the whole file must be text" contract a whole-file
            // read enforces up front — just discovered while streaming instead of
            // before returning anything.
            let n = reader.read_line(&mut buf)?;
            if n == 0 {
                break;
            }
            // Split on '\n' only, matching `file_read`'s old `content.split('\n')`
            // exactly: a trailing '\r' on a CRLF file stays part of the line.
            let line = buf.strip_suffix('\n').unwrap_or(&buf).to_string();
            if total_lines >= want_start && total_lines < want_end {
                candidate.push(line.clone());
            }
            tail.push_back(line);
            if tail.len() as u32 > PAGE_LINES {
                tail.pop_front();
            }
            total_lines += 1;
        }

        if total_lines == 0 {
            return Ok(PageResult {
                page: 0,
                start_line: 1,
                end_line: 0,
                total_lines: 0,
                total_pages: 0,
                body: String::new(),
            });
        }

        let total_pages = ((total_lines - 1) / u64::from(PAGE_LINES) + 1) as u32;
        let (page, start_line_0, lines) = if want_start < total_lines {
            (requested_page, want_start, candidate)
        } else {
            // `tail` holds the last (up to) `PAGE_LINES` lines seen, which is
            // wider than the last page whenever the file's length is not an
            // exact multiple of the stride — take only the suffix the last
            // page actually covers, not the whole buffer.
            let last_page = total_pages - 1;
            let last_start = u64::from(last_page) * u64::from(PAGE_LINES);
            let need = (total_lines - last_start) as usize;
            let skip = tail.len().saturating_sub(need);
            (
                last_page,
                last_start,
                tail.into_iter().skip(skip).collect::<Vec<_>>(),
            )
        };

        let start_line = (start_line_0 + 1) as u32;
        let end_line = start_line + (lines.len() as u32).saturating_sub(1);
        Ok(PageResult {
            page,
            start_line,
            end_line,
            total_lines: total_lines as u32,
            total_pages,
            body: lines.join("\n"),
        })
    }
}
