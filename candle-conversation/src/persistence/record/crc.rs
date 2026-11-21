//! The CRC-32 that checksums log records and the superblock.
//!
//! It is the IEEE 802.3 polynomial, so every existing record still verifies. The
//! accumulator can be fed a payload in pieces, which is how the chunked writers
//! use it. The init and finish functions hide the inversion the underlying crate
//! needs when a computation is resumed.

// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).

/// Initial CRC-32 accumulator.
pub fn crc32_init() -> u32 {
    0xFFFF_FFFF
}

/// Fold `data` into a running CRC-32 accumulator.
///
/// The accumulator is the raw register (`!` of a finished CRC), which is what
/// [`crc32_init`] starts and [`crc32_finish`] inverts. `crc32fast` resumes from a
/// *finished* value, so the register is inverted on the way in and out — the
/// values are exactly the IEEE CRC-32 the log has always carried.
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new_with_initial(!crc);
    hasher.update(data);
    !hasher.finalize()
}

/// Finalize a CRC-32 accumulator.
pub fn crc32_finish(crc: u32) -> u32 {
    !crc
}

/// One-shot CRC-32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    crc32_finish(crc32_update(crc32_init(), data))
}
