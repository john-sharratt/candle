# hash_state — hash_state_{init,update,finalize}

Streaming hash computation for data too large to pass in a single tool call.

## Files

| File | Tool | Description |
|------|------|-------------|
| `init.rs` | `hash_state_init` | Create a named context with a fixed algorithm |
| `update.rs` | `hash_state_update` | Feed a data chunk |
| `finalize.rs` | `hash_state_finalize` | Produce digest; discard context |
| `mod.rs` | — | `HashStateError`; `decode_data` helper |

## Workflow

```
hash_state_init   { id: "my-hash", algorithm: "sha256" }
  → { id: "my-hash" }

hash_state_update { id: "my-hash", data: "first chunk", encoding: "text" }
  → { id: "my-hash", bytes_fed: 11 }

hash_state_update { id: "my-hash", data: "deadbeef", encoding: "hex" }
  → { id: "my-hash", bytes_fed: 4 }

hash_state_finalize { id: "my-hash", format: "hex" }
  → { id: "my-hash", digest: "...", algorithm: "sha256" }
```

The algorithm is fixed at init; individual updates can use different encodings.

## Supported algorithms

`sha256`, `sha512`, `sha1`, `md5`, `sha3_256`, `sha3_512`, `blake3`

## Storage

Contexts live in [`crate::state::HashStateStore`] for the session lifetime.
`finalize` removes the context; subsequent updates to the same ID return
`not_found`.

## Error codes

| Code | When |
|------|------|
| `unknown_algorithm` | Algorithm name not in supported set |
| `id_already_exists` | `init` called with a duplicate ID |
| `not_found` | ID not in store (`update`, `finalize`) |
| `invalid_data_encoding` | Bad hex or base64 chunk data |
