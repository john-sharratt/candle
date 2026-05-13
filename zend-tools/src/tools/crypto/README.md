# crypto — cryptographic primitive tools (8 tools)

Pure-Rust cryptography via the RustCrypto family of crates.

## Files

| File | Tool | Primitive | Crates |
|------|------|-----------|--------|
| `aead_encrypt.rs` | `aead_encrypt` | AES-128/256-GCM, ChaCha20-Poly1305 | `aes-gcm`, `chacha20poly1305` |
| `aead_decrypt.rs` | `aead_decrypt` | same | same |
| `hmac_compute.rs` | `hmac_compute` | HMAC-SHA256/512 | `hmac`, `sha2` |
| `signature_sign.rs` | `signature_sign` | Ed25519, ECDSA-P256 | `ed25519-dalek`, `p256` |
| `signature_verify.rs` | `signature_verify` | same | same |
| `kdf_derive.rs` | `kdf_derive` | Argon2id, PBKDF2-SHA256, scrypt | `argon2`, `pbkdf2`, `scrypt` |
| `hkdf_extract.rs` | `hkdf_extract` | HKDF-Extract (RFC 5869) | `hkdf`, `sha2` |
| `hkdf_expand_label.rs` | `hkdf_expand_label` | HKDF-Expand-Label (TLS 1.3 §7.1) | `hkdf`, `sha2` |
| `mod.rs` | — | Shared `CryptoError` enum; `decode_data`/`encode_output` helpers | |

## Data encoding

All binary inputs and outputs are passed as strings.  The `encoding` parameter
accepts `"hex"` (default for most tools), `"base64"`, or `"text"` (UTF-8 bytes).

```
decode_data(s, "hex")    → Vec<u8>
decode_data(s, "base64") → Vec<u8>
decode_data(s, "text")   → s.as_bytes()

encode_output(bytes, "base64") → base64 string
encode_output(bytes, _)        → hex string (default)
```

## AEAD algorithms

| Name | Key size | Nonce size | Tag size |
|------|----------|------------|----------|
| `aes-128-gcm` | 16 bytes | 12 bytes | 16 bytes |
| `aes-256-gcm` | 32 bytes | 12 bytes | 16 bytes |
| `chacha20-poly1305` | 32 bytes | 12 bytes | 16 bytes |

## Signature algorithms

| Name | Key type |
|------|----------|
| `ed25519` | Ed25519 private key (PEM or raw hex) |
| `ecdsa-p256` | P-256 private key (PEM) |

`signature_sign` accepts either an inline key or a `credential_id` referencing
a `signing_key` credential.

## KDF parameters

| Algorithm | Key params |
|-----------|-----------|
| `argon2id` | `memory_kib`, `iterations`, `parallelism` |
| `pbkdf2-sha256` | `iterations` |
| `scrypt` | `n`, `r`, `p` |

## Error codes

| Code | When |
|------|------|
| `invalid_algorithm` | Algorithm string not recognised for this tool |
| `invalid_key` | Key material cannot be parsed or has wrong length |
| `invalid_nonce` | Nonce has wrong length or bad encoding |
| `encryption_failed` | AEAD encrypt error |
| `decryption_failed` | Authentication tag mismatch |
| `unknown_algorithm` | Algorithm name not in the supported set |
| `invalid_data_encoding` | Bad hex or base64 input |
| `signing_failed` | Private key cannot produce a signature |
| `credential_not_found` | Named credential absent from store |
| `invalid_credential_type` | Credential is not `signing_key` |
| `derivation_failed` | KDF parameter error |
| `expand_failed` | HKDF expand error |
