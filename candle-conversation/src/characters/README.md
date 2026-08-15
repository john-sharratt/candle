# candle-conversation/src/characters/

Fixture data for the `tree_gen` life-timeline generator (see
`candle-conversation/examples/tree_gen.rs`,
`docs/archived/tree_gen_design.md`).

## What it does

`tree_gen` reads a life-timeline Markdown file and a character system prompt,
then drives a guide pipeline (`guide_today` → `director` → the character
model) to produce a narrated, turn-by-turn YAML redo log of that character
living through their timeline. This directory holds the two inputs for the
example/default run of that pipeline, both for a character named Bramble.

| File | Role |
|---|---|
| `bramble.md` | The character system prompt: a first-person, sensory-grounded roleplay persona (an elderly gardener and war veteran). Passed via `tree_gen`'s `--character-prompt-file` flag; not wired to a default in the example itself. |
| `bramble-timeline.md` | The life-timeline input, `tree_gen`'s **default** `--input` value. `[T-N] description` entries (N = days before T-0 = 2026-02-24) grouped under `# Period Name` headers, each period preceded by an HTML-comment `<!-- summary ... -->` block containing a pre-written Life Story + Cast section. Over 100 KB — spans the character's life from birth (`[T-28836]`) onward. |

Neither file is `include_str!`-embedded into the compiled binary; both are
read from disk at runtime by the `tree_gen` example via its CLI arguments.
Outside of `tree_gen.rs`, "Bramble" only otherwise appears as an unrelated
placeholder string (`"You are Bramble."`) in `candle-conversation/src/tree/tests.rs`.

## How it is used

```bash
cargo run --example tree_gen -p candle-conversation --release --features hub -- \
  --input candle-conversation/src/characters/bramble-timeline.md \
  --output bramble_tree.yaml \
  --days 3 \
  --skip-entries 360
```

`tree_gen` reuses the KV cache across the whole run for the character
conversation and the `guide_today` sequence (system prompt includes the
period background extracted from the timeline's summary blocks), while
`guide_summarize` runs once upfront per selected period and `director` runs
fresh per waypoint.

## Related docs

`docs/archived/tree_gen_design.md` (design), `docs/npc_mind_design.md` and
`docs/theory_of_the_mind.md` (the cognitive architecture these fixtures
exercise), `docs/narrative_engine.md` (the narrator that turns waypoints into
`Input` events consumed downstream of `tree_gen`'s output).
