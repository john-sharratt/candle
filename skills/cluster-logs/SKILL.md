---
name: cluster-logs
description: Find what the cluster members' logs say about a symptom — a crash, a stall, an error, a slow request — with the lines cited, using the zend MCP's cluster_logs tool to search every member at once. Use when diagnosing any cluster problem before changing anything.
---

# Cluster logs

## Do

1. Turn the symptom into a search: a regular expression for the line's text,
   the least `level` (WARN for problems, ERROR for failures), the `tenant`
   (zend, npcd) when known, and `since` — an RFC 3339 prefix — a little
   before it started.
2. `cluster_logs {grep, level, tenant, since, limit}`. Every member's matching
   lines come back under that member, oldest first, the latest `limit` kept.
3. Narrow: a different pattern, a later `since`, one member's `node` (an
   Admin's). Read on through time with a later `since`.
4. Answer with:
   - **What happened**, in a sentence or two.
   - **The lines**, quoted exactly, each with its member — the evidence.
   - **What it suggests**, marked as your conclusion, and the next check.

## Notes

- Lines come back as written, timestamps included; the timestamps are UTC.
- A pattern that matches nothing across every member is a finding too — say
  so, with the search.
- Logs are a page, never a stream; for a live watch use zend-watch.
