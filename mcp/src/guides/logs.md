---
name: logs
description: How to find what the logs say about a symptom — a crash, a stall, an error, a broken Yen Bots app — searching the cluster's members and the sites at once, with the lines quoted as evidence.
tools: [log_list, log_read, log_grep, cluster_status]
---
# Reading the logs

## Do

1. Know the sources: `log_list` names them — `cluster` (the machines running
   the daemons), and each site that keeps its own, such as `yenbot` and one
   `yenbot/<app>` per app (browser errors, workflow runs, check refusals).
2. Turn the symptom into a search:
   - `grep`: a regular expression found anywhere in a line;
   - `level`: WARN for problems, ERROR for failures;
   - `source`: one source when you know where to look; every source when not;
   - `tenant`: `zend` (the coding service) or `npcd` (the bots), for the cluster;
   - `since`: an RFC 3339 prefix (UTC) a little before it started.
3. `log_grep {grep, source, level, since, limit}` — the matching lines, under
   each member or source, oldest first. `log_read {source}` reads one source's
   newest lines without a pattern.
4. Narrow: another pattern, a later `since`, one member's `node` (Admins).
   Read on through time with a later `since` — it is a page, never a stream.
5. Answer with:
   - **What happened**, in a sentence or two.
   - **The lines**, quoted exactly, each with its member or source — the evidence.
   - **What it suggests**, marked as your conclusion, and the next check.

## Notes

- A pattern that matches nothing anywhere is a finding too: say so, with the
  search.
- To keep watching for something over time, delegate it (`watch`).
