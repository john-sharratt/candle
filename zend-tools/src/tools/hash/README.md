# hash — hash_compute, hash_scan

Cryptographic digest tools backed by the RustCrypto digest crates.

## Files

| File | Tool | Description |
|------|------|-------------|
| `compute.rs` | `hash_compute` | Compute a digest of a value |
| `scan.rs` | `hash_scan` | Identify which algorithm produced a given digest |
| `mod.rs` | — | `HashError`; `compute_hash`; `decode_data`; `encode_output` |

## Supported algorithms

`sha256`, `sha512`, `sha1`, `md5`, `sha3_256`, `sha3_512`, `blake3`

## Input / output encoding

`hash_compute` accepts `encoding` parameter: `"text"` (UTF-8 bytes), `"hex"`, or `"base64"`.
Output `format` parameter: `"hex"` (default) or `"base64"`.

## hash_scan

Given a hex or base64 digest value and the pre-image data, `hash_scan` tries
all supported algorithms and returns the one that produces a matching digest.
Useful for identifying unknown digests in protocol captures, firmware headers,
or legacy code without documentation.

## Shared helpers (used by hash_state too)

- `compute_hash(data, algo)` — returns `Vec<u8>` or `HashError::UnknownAlgorithm`
- `decode_data(s, encoding)` — decode input given encoding name
- `encode_output(bytes, encoding)` — hex or base64 output

## Error codes

| Code | When |
|------|------|
| `unknown_algorithm` | Algorithm name not in the supported set |
| `invalid_data_encoding` | Bad hex or base64 input |
| `no_match` | No algorithm produced the target digest (`hash_scan`) |
