# bytes — bytes_transcode, bytes_pack, bytes_unpack, bytes_xor

Low-level byte encoding and binary format tools for protocol work.

## Files

| File | Tool | Description |
|------|------|-------------|
| `transcode.rs` | `bytes_transcode` | Convert between hex/base64/base64url/utf8 |
| `pack.rs` | `bytes_pack` | Encode typed values into a binary buffer |
| `unpack.rs` | `bytes_unpack` | Decode a binary buffer into typed JSON values |
| `xor.rs` | `bytes_xor` | XOR two byte sequences |
| `mod.rs` | — | `BytesError`; format parser; pack/unpack helpers; codec |

## bytes_transcode

Converts between four encodings:

| Name | Description |
|------|-------------|
| `hex` | Lower-case hex string (`deadbeef`) |
| `base64` | Standard base64 with padding |
| `base64url` | URL-safe base64 without padding |
| `utf8` | Raw UTF-8 text bytes |

## bytes_pack / bytes_unpack format strings

Python `struct`-compatible format with an optional endianness prefix:

| Prefix | Endianness |
|--------|-----------|
| `>` (default) | Big-endian |
| `<` | Little-endian |

Field codes:

| Code | Type | Size |
|------|------|------|
| `B` / `b` | u8 / i8 | 1 |
| `H` / `h` | u16 / i16 | 2 |
| `I` / `L` / `i` / `l` | u32 / i32 | 4 |
| `Q` / `q` | u64 / i64 | 8 |
| `f` | f32 | 4 |
| `d` | f64 | 8 |
| `Ns` | N-byte string (null-padded on pack) | N |

Repeat prefix: `4B` = four u8 fields.

Example: pack a DNS header → `">HHHHHH"` (6 × u16, big-endian).

## bytes_xor

XOR two hex-encoded byte sequences of equal length.  Used for stream-cipher
keystream application, obfuscation analysis, or CRC operations.

## Shared utilities

- `decode_bytes(s, enc)` / `encode_bytes(b, enc)` — codec helpers
- `parse_format(fmt)` → `(big_endian, Vec<FormatField>)` — format string parser
- `pack_field(buf, field, val, big)` — pack one value
- `unpack_field(cursor, field, big)` — unpack one value

## Error codes

| Code | When |
|------|------|
| `invalid_encoding` | Unknown encoding name |
| `decode_failed` | Bad hex or base64 input |
| `encode_failed` | Bytes not representable in target encoding (e.g. non-UTF8 → `utf8`) |
| `invalid_format` | Unknown format character or malformed format string |
| `pack_failed` | JSON value does not match the format field type |
| `unpack_failed` | Buffer too short for the format |
