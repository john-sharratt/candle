# Reprojection scan time on the production gallery — 2026-11-24

What one reprojection's provenance scan takes on BlackWorld's live gallery, read from the
daemon's own log while the battery ran (C1 run 7 and R1 run 4, sixteen answers in flight).
The gallery is the one `needle.md` ranks over: 7,943 files, 36.1M tokens.

Every line the scheduler logged for a reprojection between 21:12:46Z and 22:08:45Z (UTC)
— the whole of the log's reach at the time it was read — is in
`reproject_scans_2026-11-24.json`, as `GET /v1/logs` answered it:

```bash
curl -H "x-tokera-user: service:battery" -H "x-tokera-provider: service" \
  -H "x-tokera-node: 223ebb209828d84c" \
  "http://192.168.1.2:8081/v1/logs?grep=reproject%3A.*scan_ms%3D&level=TRACE&limit=5000"
```

`scan_ms` is the scan's wall time on the scheduler's thread, queueing behind the other
sessions' work on the card included.

| Reprojections | Median | p10 | p90 | p99 | Max |
|---:|---:|---:|---:|---:|---:|
| 1,277 | 83 ms | 61 ms | 182 ms | 549 ms | 794 ms |

The ~9.5 ms of `docs/archived/paged_gallery_arena.md` is the same scan on the RTX 4090
Laptop GPU's 1-bit tensor cores over a 591K-token gallery (788 files), one session, the
machine otherwise idle. A reprojection runs every 64 decoded tokens, off the decode's
critical path: the decode continues on the working set it has while the scan runs.
