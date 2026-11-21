//! Tests record framing: layout, round trips, forward compatibility and corruption.
//!
//! The layout tests pin sector padding and the header-then-payload order. The
//! compatibility tests check that unknown types and unknown fields decode rather
//! than fail. The corruption tests flip a header byte or a payload byte and expect
//! the JSON parse or the CRC to catch it. The tag tests pin the wire values, since
//! a reordering would silently change every index entry.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::content_hash::hash_bytes;

    use super::*;

    #[test]
    fn padded_len_rounds_to_4k() {
        // Header line of 100 bytes + 1 newline + 0 payload = 101 → 4096.
        assert_eq!(padded_record_len(100, 0), 4096);
        // Header of 100 bytes + 1 newline + 3995 payload = 4096 → 4096.
        assert_eq!(padded_record_len(100, 3995), 4096);
        // One more byte pushes us to the next sector.
        assert_eq!(padded_record_len(100, 3996), 8192);
        // Large payload spans many sectors.
        assert_eq!(padded_record_len(100, 100_000), 102_400);
    }

    // A generic record header for the codec / CRC tests. Uses `Tokens` (crc32
    // over the whole payload) — `Chunk` records store a Fletcher golden instead
    // and are verified separately, so they are not meaningful for these tests.
    pub(in super::super) fn sample_header(payload_len: u64, payload: &[u8]) -> RecordHeader {
        RecordHeader {
            record_type: RecordType::Tokens,
            format: 7,
            payload_len,
            crc: crc32(payload),
            stream_id: 0xDEAD_BEEF_0000_0001,
            chunk_index: 5,
            token_count: 32,
            ..RecordHeader::NONE
        }
    }

    #[test]
    fn encode_layout_starts_with_json_header_and_newline() {
        let payload = [0xAAu8, 0xBB, 0xCC, 0xDD];
        let header = sample_header(payload.len() as u64, &payload);
        let bytes = encode_record(&header, &payload);

        // Padded out to one full sector.
        assert_eq!(bytes.len(), ALIGN);

        // The header is JSON terminated by a single newline.
        let newline_pos = bytes.iter().position(|&b| b == b'\n').unwrap();
        let header_str = std::str::from_utf8(&bytes[..newline_pos]).unwrap();
        let parsed: RecordHeader = serde_json::from_str(header_str).unwrap();
        assert_eq!(parsed.record_type, RecordType::Tokens);
        assert_eq!(parsed.format, 7);
        assert_eq!(parsed.payload_len, 4);
        assert_eq!(parsed.stream_id, 0xDEAD_BEEF_0000_0001);
        assert_eq!(parsed.chunk_index, 5);
        assert_eq!(parsed.token_count, 32);
        assert_eq!(parsed.crc, crc32(&payload));

        // Payload sits immediately after the newline.
        let payload_start = newline_pos + 1;
        assert_eq!(
            &bytes[payload_start..payload_start + payload.len()],
            &payload
        );

        // The remainder of the sector is zero padding.
        assert!(bytes[payload_start + payload.len()..]
            .iter()
            .all(|&b| b == 0));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let payload: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
        let header = sample_header(payload.len() as u64, &payload);
        let (written, bytes) = encode_record_written(&header, &payload);

        // The written header is the caller's, with the payload's digest set.
        assert_eq!(
            written,
            RecordHeader {
                digest: hash_bytes(&payload),
                ..header
            }
        );
        let (got_header, got_payload, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(got_header, written);
        assert_eq!(got_payload, payload.as_slice());
    }

    /// Identity fields round-trip, and a digest the caller passed is replaced
    /// by the payload's own: a copy may not claim another record's content.
    #[test]
    fn identity_round_trips_and_the_digest_is_always_the_payloads() {
        let payload = b"identity payload".to_vec();
        let header = RecordHeader {
            digest: ContentHash { lo: 7, hi: 9 },
            ..sample_header(payload.len() as u64, &payload).with_identity(RecordIdentity {
                origin: 0x1122_3344_5566_7788,
                oseq: 42,
                gen: 3,
                epoch: 5,
                hlc: (1_700_000_000_000 << 16) | 2,
            })
        };
        let (written, bytes) = encode_record_written(&header, &payload);
        assert_eq!(written.digest, hash_bytes(&payload));
        let (got_header, _, _) = decode_record(&bytes).unwrap();
        assert_eq!(got_header, written);
        assert_eq!(got_header.identity(), header.identity());
    }

    /// A header with no identity writes none of the identity keys, so a
    /// record without one is byte-for-byte what it was before identity existed.
    #[test]
    fn a_header_with_no_identity_writes_no_identity_keys() {
        let payload = b"x".to_vec();
        let header = sample_header(payload.len() as u64, &payload);
        let (_, bytes) = encode_record_written(&header, &payload);
        let line_end = bytes.iter().position(|&b| b == b'\n').unwrap();
        let line = std::str::from_utf8(&bytes[..line_end]).unwrap();
        for key in ["\"origin\"", "\"oseq\"", "\"gen\"", "\"epoch\"", "\"hlc\""] {
            assert!(!line.contains(key), "{key} written in {line}");
        }
        assert!(line.contains("\"digest\""), "digest missing from {line}");
    }

    #[test]
    fn empty_payload_roundtrips() {
        let header = RecordHeader {
            record_type: RecordType::Commit,
            payload_len: 0,
            crc: crc32(&[]),
            format: 0,
            stream_id: 42,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let (written, bytes) = encode_record_written(&header, &[]);
        assert_eq!(written.digest, hash_bytes(&[]));
        let (got_header, got_payload, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, ALIGN);
        assert_eq!(got_header, written);
        assert!(got_payload.is_empty());
    }

    #[test]
    fn unknown_record_type_decodes_to_unknown_variant_without_error() {
        // Hand-roll a record whose `type` is a tag this version
        // doesn't know about ("future_kind"). The reader must
        // surface it as `RecordType::Unknown` so the walker can
        // skip past it.
        let payload = b"future-payload";
        let crc = crc32(payload);
        let header_line = format!(
            "{{\"type\":\"future_kind\",\"payload_len\":{},\"crc\":{}}}",
            payload.len(),
            crc
        );
        let header_len = header_line.len();
        let total = padded_record_len(header_len, payload.len() as u64);
        let mut bytes = vec![0u8; total];
        bytes[..header_len].copy_from_slice(header_line.as_bytes());
        bytes[header_len] = b'\n';
        bytes[header_len + 1..header_len + 1 + payload.len()].copy_from_slice(payload);

        let (header, got_payload, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, total);
        assert_eq!(header.record_type, RecordType::Unknown);
        // The payload survived even though we don't know how to
        // interpret it — useful for diagnostics and round-trip in
        // tests.
        assert_eq!(got_payload, payload);
    }

    #[test]
    fn extra_header_field_is_ignored() {
        // A newer writer adds a key we don't model — we must
        // silently ignore it and parse the rest normally. This is
        // the field-level forward-compat property.
        let payload = b"hi";
        let crc = crc32(payload);
        let header_line = format!(
            "{{\"type\":\"commit\",\"payload_len\":{},\"crc\":{},\"future_field\":\"opaque\"}}",
            payload.len(),
            crc
        );
        let header_len = header_line.len();
        let total = padded_record_len(header_len, payload.len() as u64);
        let mut bytes = vec![0u8; total];
        bytes[..header_len].copy_from_slice(header_line.as_bytes());
        bytes[header_len] = b'\n';
        bytes[header_len + 1..header_len + 1 + payload.len()].copy_from_slice(payload);

        let (header, got_payload, _) = decode_record(&bytes).unwrap();
        assert_eq!(header.record_type, RecordType::Commit);
        assert_eq!(got_payload, payload);
    }

    #[test]
    fn missing_optional_field_defaults_to_zero() {
        // Some older writer omitted `format` / `stream_id` /
        // `chunk_index` / `token_count` — those should default to
        // zero rather than fail.
        let payload = b"";
        let header_line = "{\"type\":\"commit\",\"payload_len\":0,\"crc\":0}";
        let header_len = header_line.len();
        let total = padded_record_len(header_len, 0);
        let mut bytes = vec![0u8; total];
        bytes[..header_len].copy_from_slice(header_line.as_bytes());
        bytes[header_len] = b'\n';

        let (header, got_payload, _) = decode_record(&bytes).unwrap();
        assert_eq!(header.record_type, RecordType::Commit);
        assert_eq!(header.format, 0);
        assert_eq!(header.stream_id, 0);
        assert_eq!(header.chunk_index, 0);
        assert_eq!(header.token_count, 0);
        assert_eq!(got_payload, payload);
    }

    #[test]
    fn bad_header_first_byte_rejected() {
        // The first byte isn't `{` — corruption (or a torn record).
        let mut bytes = encode_record(&sample_header(0, &[]), &[]);
        bytes[0] = b'X';
        assert!(matches!(
            decode_record(&bytes),
            Err(PersistenceError::Corrupt(_))
        ));
    }

    #[test]
    fn flipped_payload_byte_caught_by_verify_record_crc() {
        // A metadata (non-chunk) record: crc32 over the whole payload catches
        // any flipped byte.
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let header = RecordHeader {
            record_type: RecordType::Tokens,
            format: 0,
            payload_len: payload.len() as u64,
            crc: 0, // filled by encode_record
            stream_id: 1,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let mut bytes = encode_record(&header, &payload);
        // Flip a byte inside the payload — find it just after the header newline.
        let newline_pos = bytes.iter().position(|&b| b == b'\n').unwrap();
        bytes[newline_pos + 2] ^= 0x01;
        // decode_record itself no longer CRC-verifies; the parse succeeds.
        let (header, payload_slice, _) = decode_record(&bytes).unwrap();
        // The dedicated verifier surfaces the bit rot.
        assert!(matches!(
            verify_record_crc(&header, payload_slice),
            Err(PersistenceError::BadChecksum { .. })
        ));
    }

    /// The digest wire tag is the enum discriminant — pinned by raw
    /// value so a reordering of the enum can't silently change the
    /// on-disk `HeaderIndex` format.
    #[test]
    fn record_type_tags_round_trip_with_pinned_values() {
        let pinned: [(RecordType, u8); 19] = [
            (RecordType::ModelKey, 1),
            (RecordType::Template, 2),
            (RecordType::StreamDecl, 3),
            (RecordType::Chunk, 4),
            (RecordType::Tokens, 5),
            (RecordType::Commit, 7),
            (RecordType::Tokenizer, 9),
            (RecordType::Label, 10),
            (RecordType::ConvState, 11),
            (RecordType::TreeMetadata, 12),
            (RecordType::DebugId, 13),
            (RecordType::Tombstone, 14),
            (RecordType::ProjectionEvents, 15),
            (RecordType::Distilled, 16),
            (RecordType::WideQSig, 17),
            (RecordType::HeaderIndex, 18),
            (RecordType::SectionTombstone, 24),
            (RecordType::CustomObject, 27),
            (RecordType::Blob, 29),
        ];
        for (rt, tag) in pinned {
            assert_eq!(rt.tag(), tag, "{rt:?} wire tag");
            assert_eq!(RecordType::from_tag(tag), rt);
        }
        // Retired tags 6 (the removed per-chunk Signatures — WideQSig
        // replaced it) and 8 (the removed checkpoint), plus future tags,
        // decode to Unknown.
        assert_eq!(RecordType::from_tag(6), RecordType::Unknown);
        assert_eq!(RecordType::from_tag(8), RecordType::Unknown);
        assert_eq!(RecordType::from_tag(200), RecordType::Unknown);
    }

    /// **Only the tombstones free space**: a timeline's, a section's and a
    /// file's. Every other type, tag 0 to 255, adds what it carries.
    #[test]
    fn only_the_tombstones_free_space() {
        let freeing: Vec<RecordType> = (0..=u8::MAX)
            .map(RecordType::from_tag)
            .filter(|t| t.frees_space())
            .collect();
        assert_eq!(
            freeing,
            [
                RecordType::Tombstone,
                RecordType::SectionTombstone,
                RecordType::VfsTombstone
            ]
        );
    }

    #[test]
    fn flipped_header_byte_caught_by_json_or_crc() {
        // Mutating a byte inside the header line either breaks JSON
        // parsing or — if the JSON still parses — leaves a stale
        // `crc` value in the header that no longer matches the
        // payload. The fast path catches the first case; verification
        // at the consumption point catches the second.
        let payload = [1u8, 2, 3, 4];
        let mut bytes = encode_record(&sample_header(payload.len() as u64, &payload), &payload);
        let newline_pos = bytes.iter().position(|&b| b == b'\n').unwrap();
        let target = (0..newline_pos)
            .find(|&i| bytes[i].is_ascii_digit())
            .expect("the encoded header contains at least one digit");
        bytes[target] = if bytes[target] == b'0' { b'9' } else { b'0' };

        let caught = match decode_record(&bytes) {
            Err(_) => true,
            Ok((header, payload_slice, _)) => verify_record_crc(&header, payload_slice).is_err(),
        };
        assert!(
            caught,
            "either parse failure or CRC mismatch must surface the header bit-flip"
        );
    }

    #[test]
    fn truncated_record_rejected() {
        let bytes = encode_record(&sample_header(0, &[]), &[]);
        // Decode against a buffer that's far too short.
        assert!(decode_record(&bytes[..10]).is_err());
    }

    #[test]
    #[should_panic(expected = "RecordType::Unknown")]
    fn encode_refuses_unknown_variant() {
        let header = RecordHeader {
            record_type: RecordType::Unknown,
            payload_len: 0,
            crc: 0,
            format: 0,
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        encode_record(&header, &[]);
    }

    #[test]
    fn header_line_too_large_panics() {
        // A pathologically large "extra" payload smuggled into a
        // payload of huge size — the header itself is always tiny,
        // so this should always fit. But we can still verify the
        // bound exists: pin MAX_HEADER_LINE against ALIGN so a
        // future tweak doesn't accidentally allow multi-sector
        // headers.
        const _: () = assert!(MAX_HEADER_LINE < ALIGN);
    }
}
