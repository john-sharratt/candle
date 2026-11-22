//! Parses a record's header and payload from a byte buffer, without verifying the
//! payload.
//!
//! The header is one JSON line that must fit in one sector, so the scan for its
//! end is bounded and a malformed record fails early. Payload verification is a
//! separate step, made where the bytes are consumed. Chunk records are excluded
//! from it, because their checksum is a GPU golden and not a CRC over the payload.

use super::*;

/// Decode the header at the start of `probe` without touching the
/// payload. Returns the decoded header and the number of bytes the
/// header occupies on disk **including** the trailing newline.
///
/// `probe` must contain the first [`HEADER_PROBE_SIZE`] bytes of the
/// record (or fewer if EOF arrives first — in which case the parse
/// fails clean).
pub fn decode_header(probe: &[u8]) -> Result<(RecordHeader, usize)> {
    if probe.is_empty() {
        return Err(PersistenceError::Truncated { need: 1, have: 0 });
    }
    if probe[0] != b'{' {
        return Err(PersistenceError::Corrupt(format!(
            "expected JSON header starting with '{{', found 0x{:02x}",
            probe[0]
        )));
    }
    // Cap the scan window: a JSON header is required to fit in one
    // sector. Bounding the search keeps a malformed record from
    // scanning arbitrarily far for a newline.
    let scan_len = probe.len().min(ALIGN);
    let newline_pos = probe[..scan_len]
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| {
            PersistenceError::Corrupt(
                "no newline found within the header probe — header missing or too large"
                    .to_string(),
            )
        })?;
    let header_line = std::str::from_utf8(&probe[..newline_pos])
        .map_err(|e| PersistenceError::Corrupt(format!("record header is not valid UTF-8: {e}")))?;
    let header: RecordHeader = serde_json::from_str(header_line)
        .map_err(|e| PersistenceError::Corrupt(format!("record header JSON parse: {e}")))?;
    Ok((header, newline_pos + 1))
}

/// Decode one full record from the start of `buf` — header + payload
/// slice + padded total. Returns `(header, payload, total_bytes)`
/// where `payload` is a **borrowed** view into `buf` (zero-copy) and
/// `total_bytes` is the record's padded on-disk size so a walker can
/// advance.
///
/// **Does not CRC-verify the payload.** Verification is a separate
/// concern handled by [`verify_record_crc`] at the payload's
/// consumption point (cold-load decode, `read_record_at`); the
/// recovery walk skips payload bytes entirely for the bulk record
/// types and never pays for verification it can't use.
///
/// Cold-load uses the payload slice in place against the pinned
/// scratch buffer; callers that need owned bytes — typically
/// random-access reads like [`super::log_file::read_record_at`] —
/// `.to_vec()` at the boundary where ownership starts.
pub fn decode_record(buf: &[u8]) -> Result<(RecordHeader, &[u8], usize)> {
    let (header, header_bytes) = decode_header(buf)?;
    let total = padded_record_len(header_bytes - 1, header.payload_len);
    if buf.len() < total {
        return Err(PersistenceError::Truncated {
            need: total,
            have: buf.len(),
        });
    }
    let payload_end = header_bytes + header.payload_len as usize;
    let payload = &buf[header_bytes..payload_end];
    Ok((header, payload, total))
}

/// CRC-verify the payload of a metadata record against the header's stored CRC.
///
/// Called at every payload **consumption point** — the pipelined
/// cold-load decode, the batched stream read, and `read_record_at` —
/// so latent bit rot fails loudly exactly where the bytes are about to
/// be used, at the cost of one checksum pass over bytes just read from
/// disk. `RecordType::Unknown` records are skipped — the payload
/// format may not be readable in our world view and the walker has
/// already decided to advance past them.
///
/// `Chunk` records are **skipped here**. Their `header.crc` holds the GPU golden
/// (a Fletcher-32 over the arena bytes taken *before* the device→host copy, see
/// candle-kernels `simple/fletcher32.cu`), not a crc32 over the payload — so this
/// crc32 comparison never applies. Chunk integrity is checked against that golden
/// separately, with the severity each site warrants: a hard error at the write
/// boundary ([`SubstratePersistence::append_record`]) and a non-fatal warning
/// when the bytes are read into RAM/VRAM (cold-load / elevation). Keeping chunks
/// off this path also keeps the restart/recovery read — which only verifies
/// metadata — from ever putting chunk bytes on the golden path, protecting
/// restart time.
///
/// **Only in a build with the `crc-check` feature.** Without it nothing is
/// computed and every payload passes: a read pays for no checksum it does not
/// ask for. Every writer still stamps its CRC into the header, so a build with
/// the feature verifies every record ever written. A tool whose job is the
/// check calls [`check_record_crc`], which verifies in every build.
pub fn verify_record_crc(header: &RecordHeader, payload: &[u8]) -> Result<()> {
    if !cfg!(feature = "crc-check") {
        return Ok(());
    }
    check_record_crc(header, payload)
}

/// [`verify_record_crc`] in every build, whatever its features.
pub fn check_record_crc(header: &RecordHeader, payload: &[u8]) -> Result<()> {
    if header.record_type == RecordType::Unknown || header.record_type == RecordType::Chunk {
        return Ok(());
    }
    let computed = crc32(payload);
    if computed != header.crc {
        return Err(PersistenceError::BadChecksum {
            header: header.crc,
            computed,
        });
    }
    Ok(())
}
