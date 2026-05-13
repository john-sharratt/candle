# credentials — credential_save, credential_list, credential_delete

Named authentication material stored in the in-memory [`CredentialStore`] and
referenced by session-open tools.

## Files

| File | Tool | Description |
|------|------|-------------|
| `save.rs` | `credential_save` | Validate and insert a new credential |
| `list.rs` | `credential_list` | Return metadata (never secrets) |
| `delete.rs` | `credential_delete` | Remove by name |
| `mod.rs` | — | `CredError` enum with all error codes |

## Credential types

| Type | Requires | Used by |
|------|----------|---------|
| `ssh_key` | `username` + PEM/OpenSSH private key, opt. `passphrase` | `ssh_session_open`, `remote_fs_session_open` |
| `ssh_password` | `username` | `ssh_session_open` |
| `telnet_password` | `username` | `telnet_session_open` |
| `http_bearer` | — | `http_session_open` |
| `http_basic` | `username` | `http_session_open` |
| `http_header` | `header_name` | `http_session_open` |
| `totp_secret` | — (base32 seed) | `totp_generate` |
| `sql_password` | `username`, opt. `default_database` | `sql_session_open` |
| `remote_fs_password` | `username`, opt. `domain` | `remote_fs_session_open` |
| `tls_client_cert` | — (cert+key PEM bundle) | `tls_session_open` |
| `signing_key` | — (PEM private key) | `signature_sign` |

Aliases `api_key` and `ed25519_key` are accepted for backward compatibility.

## Error codes

| Code | Variant | When |
|------|---------|------|
| `duplicate_name` | `CredError::DuplicateName` | Name already in store |
| `not_found` | `CredError::NotFound` | Name absent (delete) |
| `missing_field` | `CredError::MissingField` | Type-required field absent (`username`, `header_name`) |
| `invalid_credential_type` | `CredError::InvalidType` | `type` not in the accepted list |
| `invalid_key` | `CredError::InvalidKey` | `ssh_key` secret missing PEM/OpenSSH header |

## Confirmation

`credential_save` confirms every call (shows type + name).
`credential_list` and `credential_delete` do not confirm.
