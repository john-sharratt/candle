---
name: logs
description: How to find what the members' logs say about a symptom — a crash, a stall, an error — searching every member at once, with the lines quoted as evidence.
tools: [cluster_logs, cluster_status]
---
# Reading the logs

## Do

1. Turn the symptom into a search:
   - `grep`: a regular expression found anywhere in a line;
   - `level`: WARN for problems, ERROR for failures;
   - `tenant`: `zend` or `npcd` when known;
   - `since`: an RFC 3339 prefix (UTC) a little before it started.
2. `cluster_logs {grep, level, tenant, since, limit}` — every member's
   matching lines, under that member, oldest first.
3. Narrow: another pattern, a later `since`, one member's `node` (Admins).
   Read on through time with a later `since` — it is a page, never a stream.
4. Answer with:
   - **What happened**, in a sentence or two.
   - **The lines**, quoted exactly, each with its member — the evidence.
   - **What it suggests**, marked as your conclusion, and the next check.

## Notes

- A pattern that matches nothing on every member is a finding too: say so,
  with the search.
- To keep watching for something over time, delegate it (`watch`).
