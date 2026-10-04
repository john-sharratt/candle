# Engine against gate — RTX PRO 5000 Blackwell 72 GB, 2026-10-04

The engine probe's clean C5×8 baseline (`kv_fragmentation`'s `*_combined` tests:
eight C5 story turns through the real `ConversationEngine`, clean pool) set
beside the forward gate's C5×8 row from the same load. One `cargo` process per
run, card to itself, both on `Int8Mode::Precision`, which is what `auto`
resolves to on this card and what `ModelBuilder::load_model` now reports.

Prefill is submit → last first token, decode the tokens after that up to the
last one, both summed over the eight sessions. The engine's prefill counts the
turn's own prompt (`TurnStats::turn_prefill_tokens`), not the sequence's KV
depth. Single runs, so a gap under ~5% is noise.

| Model | row | prefill t/s | decode t/s | story |
|---|---|---:|---:|---|
| Qwen3.6-35B-A3B | gate C5×8, standalone | 10,078.3 | 962.6 | 8/8 |
| Qwen3.6-35B-A3B | gate C5×8, combined load | 10,103.5 | 988.7 | 8/8 |
| Qwen3.6-35B-A3B | engine C5×8 | 9,058.2 | 915.3 | 8/8 |
| Qwen3.6-35B-A3B | engine ×16 (phase B) | 9,450.3 | 1,674.5 | 16/16 |
| Qwen3.8-Flash-Next | gate C5×8, combined load | 3,354.7 | 521.2 | 8/8 |
| Qwen3.8-Flash-Next | engine C5×8 | 3,370.1 | 532.3 | 8/8 |
| Qwen3.8-Flash-Next | engine ×8 (phase B) | 3,317.6 | 561.7 | 8/8 |

Engine against the combined-load gate row: Qwen3.6 prefill −10.3%, decode −7.4%;
Flash-Next prefill +0.5%, decode +2.1%.

## Against the earlier figures

| Model | | earlier | now |
|---|---|---:|---:|
| Qwen3.6-35B-A3B | engine C5×8 prefill | 11,265 | 9,058.2 |
| Qwen3.6-35B-A3B | engine C5×8 decode | 911 | 915.3 |
| Qwen3.8-Flash-Next | engine C5×8 decode | 530.3 | 532.3 |

The earlier Qwen3.6 prefill was overcounted: it divided the sequence's KV depth,
projected context included, by the prefill window. Counting the turn's own
prompt puts it at 9,058 — under the gate, not over it. Decode is unchanged
within noise on both models, so the review fixes (sampler count tables stamped
by a parallel kernel on the sampler's stream, snapshot retraction, tool-call
cleanup without deferral, the closing-tail batch split by adapter) cost nothing
measurable here.

## At 64 tokens per session (later the same day)

The clean baseline now generates 64 tokens per session, and the combined test
runs the gate's C5×8 row at that same count beside it. At the ladder's 10 tokens
the decode window was ~4 speculative steps (~80 ms), and one compaction pass
landing in it moved decode by 20%. Built with two prefill changes: a prefill
group within the model's 25% slack of its width cap rides one forward
(`prefill_group_budget`), and the first tokens of turns finishing together are
sampled in one dispatch.

| Model | row | prefill t/s | decode t/s |
|---|---|---:|---:|
| Qwen3.6-35B-A3B | gate C5×8, 64 tokens | 9,992.1 | 1,100.4 |
| Qwen3.6-35B-A3B | engine C5×8, 64 tokens | 9,847.6 | 1,070.4 |
| Qwen3.8-Flash-Next | gate C5×8, 64 tokens | 3,336.8 | 553.7 |
| Qwen3.8-Flash-Next | engine C5×8, 64 tokens | 3,447.0 | 584.1 |

Engine against gate: Qwen3.6 prefill −1.4%, decode −2.7%; Flash-Next prefill
+3.3%, decode +5.5%. Phase B: Qwen3.6 ×16 9,730.0 / 1,674.7, Flash-Next ×8
3,380.1 / 564.8. Story 16/16 and 8/8. Flash-Next passed all three probe gates
(efficiency 100%, weight uptake 93%); Qwen3.6's efficiency gate read 81% (the
known pool-efficiency issue on `main`, failing with and without these changes).

## Engine probe gates

| Probe | story | worst sustained eff% | weight uptake |
|---|---:|---:|---:|
| qwen36-35b-a3b-q4 | 16/16 | 92% | weights fully resident |
| qwen38-flash-next | 8/8 | 90% | 95% |

Both pass all three gates.
