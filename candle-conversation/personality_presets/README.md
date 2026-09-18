# Personality Presets

This directory contains fixed personality data shared by conversation hosts.
Each preset is one YAML document with:

```yaml
trigger_prompts: []
personality_steers:
  - text: ...
    weight: -1.0
```

`free.yaml` is the first preset. It represents a character that speaks more
freely about feelings. The common loader lives in `src/personality.rs`; hosts
select a preset by path and do not need to duplicate the schema or loader.

Build presets with the shared tool:

```text
cargo run -p candle-conversation --bin personality_vector_tool -- \
  --input candle-conversation/personality_presets/free.yaml \
  --output candle-conversation/personality_presets/free.vectors.bin
```

Preset files are private authored data and should be handled according to the
workspace's data policy.

Test a preset against its trigger prompts and an existing vector artifact:

```text
cargo run -p candle-conversation --bin personality_vector_tool --release -- test \
  --vectors candle-conversation/personality_presets/free.vectors.bin \
  --personality candle-conversation/personality_presets/free.yaml
```

The test reports aggregate vector count, trigger prompts tested, personality
hits, personality misses, and rates. A prompt is a miss when a negative-scale
personality steer token sequence appears in its decode; a prompt with no such
match is a hit. The vector payload is loaded and validated, but this command
does not apply the vectors.

Generated artifacts use a neutral default scale of `0.0`. Each grouped vector
stores the mean scalar weight of the personality-steer set used to generate it.
