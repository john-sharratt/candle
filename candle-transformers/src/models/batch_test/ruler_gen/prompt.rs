//! Builds the Qwen3 chat prompt for a RULER sample with thinking suppressed, and
//! holds the filler text and word pool the generators draw from.
//!
//! The prompt is the ChatML frame with the thinking switch set off and an empty
//! thinking block, so the answer is the first thing generated. The filler corpus is
//! tiled to any length and is written to avoid the words in the common-word pool and
//! the needle numbers, so filler never answers a question by accident. Token counts
//! come from the caller's tokenizer, which keeps the length accounting exact for the
//! model being measured.

use super::*;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Prefill chunk size — must match KV cache CHUNK_SIZE (32).
pub(super) const PREFILL_STEP: usize = 32;

/// Qwen3 EOS token IDs.
pub const QWEN3_EOS_IDS: &[u32] = &[151645, 151643];

// <|im_end|>, <|endoftext|>

// ChatML prompt fragments (Qwen3).
pub(super) const SYS_START: &str = "<|im_start|>system\n";

pub(super) const SYS_END: &str = "<|im_end|>\n";

pub(super) const USER_START: &str = "<|im_start|>user\n";

pub(super) const USER_END: &str = "<|im_end|>\n";

pub(super) const ASST_PREFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Pool of unusual color/dye names used as "common words" in CWE tasks.
/// These are chosen to be absent from the filler corpus.
pub(super) const CWE_WORD_POOL: &[&str] = &[
    "crimson",
    "azure",
    "violet",
    "amber",
    "scarlet",
    "cobalt",
    "magenta",
    "tawny",
    "ochre",
    "sienna",
    "umber",
    "taupe",
    "mauve",
    "cerise",
    "fuchsia",
    "vermillion",
    "periwinkle",
    "chartreuse",
    "gamboge",
    "cinnabar",
    "carmine",
    "saffron",
    "turquoise",
    "verdigris",
    "puce",
    "russet",
    "bisque",
    "ecru",
    "alabaster",
    "celadon",
];

/// Varied public-domain-style prose (~750 words, ~1050 Qwen3 tokens).
/// Tiled to fill any context length. Deliberately avoids words in CWE_WORD_POOL
/// and needle number sequences.
pub(super) const FILLER_CORPUS: &str =
    "The solar system formed approximately 4.6 billion years ago \
from a dense region of a molecular cloud. The gravitational collapse of this region caused it to \
spin and flatten into a protoplanetary disk, from which the Sun and planets eventually formed. \
The four inner planets—Mercury, Venus, Earth, and Mars—have solid rocky surfaces, while the four \
outer planets—Jupiter, Saturn, Uranus, and Neptune—are composed primarily of gas and ice.\n\n\
The ocean covers more than 70 percent of Earth's surface and contains about 97 percent of all \
the water on the planet. Deep ocean currents, driven by differences in temperature and salinity, \
circulate water around the globe over centuries. The ocean absorbs heat from the Sun and releases \
it slowly, moderating temperatures in coastal regions. Coral reefs support an estimated \
25 percent of all marine species despite covering less than one percent of the ocean floor.\n\n\
Languages are used by humans to communicate thoughts, feelings, and information. \
There are approximately 7,000 languages spoken in the world today, but more than half of the \
world's population speaks one of just 23 languages as their first language. Languages evolve \
continuously, with new words being added and old ones becoming obsolete. Writing systems, \
developed independently in several ancient civilizations, allowed language to be recorded and \
transmitted across generations.\n\n\
Mountain ranges are formed by the movement of tectonic plates, volcanic activity, or erosion. \
The Himalayas, which include Mount Everest, the highest peak on Earth at 8,849 meters, were \
formed by the collision of the Indian and Eurasian tectonic plates beginning about \
50 million years ago. Mountains affect local climate by forcing air masses to rise, cool, \
and deposit precipitation on their windward slopes.\n\n\
The human brain contains approximately 86 billion neurons, each connected to thousands of \
others through synapses. Neural signals travel as electrochemical impulses along axons and \
across synapses. The cerebral cortex, the outer layer of the brain, is responsible for higher \
cognitive functions including reasoning, language, and consciousness. Sleep is essential for \
brain function, allowing neural connections to consolidate memories.\n\n\
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy \
into chemical energy stored as glucose. In the chloroplasts of plant cells, chlorophyll absorbs \
red and blue light while reflecting green light, giving plants their characteristic color. \
The light-dependent reactions produce oxygen as a byproduct, which is released into the \
atmosphere. Over geological time, photosynthesis has transformed Earth's atmosphere.\n\n\
Trade routes connecting civilizations around the ancient world facilitated the exchange of \
goods, ideas, and technologies. The Silk Road, a network of overland routes stretching from \
China to the Mediterranean, was used from around 200 BCE to the 15th century CE. Merchants \
carried silk, spices, glassware, and other commodities across thousands of kilometers. Along \
with physical goods, religions, art styles, and diseases spread through these trading networks.\n\n\
Rivers are a vital component of the freshwater cycle, transporting water from elevated terrain \
to the sea. The Amazon River in South America carries the largest volume of water of any river \
in the world, discharging about 20 percent of all fresh water that flows into the world's \
oceans. Rivers shape the landscape through erosion and deposition, forming features such as \
canyons, alluvial fans, and deltas.\n\n\
The periodic table of elements, first organized by Dmitri Mendeleev in 1869, arranges the \
known chemical elements by increasing atomic number and groups them by similar chemical \
properties. Elements in the same group share the same number of valence electrons, which \
determines how they bond with other elements. Metals make up the majority of elements and \
are generally shiny, malleable, and good conductors of heat and electricity.\n\n\
Architecture has evolved significantly over thousands of years, reflecting advances in \
materials, construction techniques, and cultural values. Ancient Egyptian architects built \
massive stone monuments such as the pyramids, which have endured for millennia. The Romans \
developed concrete and the arch, enabling the construction of large-scale infrastructure \
like aqueducts and amphitheaters. Modern architects use steel, reinforced concrete, and \
glass to create skyscrapers and other complex structures that were unimaginable to their \
predecessors.";

// ── Internal filler helpers ───────────────────────────────────────────────────

/// Tokenize `FILLER_CORPUS` and tile to approximately `target_len` tokens.
/// Returns a string that decodes to approximately that many tokens when
/// re-tokenized as part of a longer prompt (±2%).
pub(super) fn make_filler(tokenizer: &Tokenizer, target_len: usize) -> String {
    let corpus_tokens: Vec<u32> = tokenizer
        .encode(FILLER_CORPUS, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let n = corpus_tokens.len();
    if n == 0 || target_len == 0 {
        return String::new();
    }

    // Tile to slightly under target (leave room for BPE-boundary expansion).
    let safe = target_len.saturating_sub(target_len / 20);
    let mut tiled: Vec<u32> = Vec::with_capacity(safe + n);
    while tiled.len() < safe {
        let need = safe - tiled.len();
        tiled.extend_from_slice(&corpus_tokens[..need.min(n)]);
    }

    tokenizer.decode(&tiled, true).unwrap_or_default()
}

/// Count tokens for a string using the given tokenizer.
pub(super) fn token_count(tokenizer: &Tokenizer, text: &str) -> usize {
    tokenizer.encode(text, false).map(|e| e.len()).unwrap_or(0)
}

// ── Prompt assembly ───────────────────────────────────────────────────────────

/// Build a complete Qwen3 ChatML prompt with thinking suppressed.
pub(super) fn build_prompt(body: &str, question: &str) -> String {
    format!(
        "{SYS_START}You are a helpful assistant.{SYS_END}\
         {USER_START}/no_think\n\
         {body}\n\n\
         Question: {question}\n\
         Answer with the exact value only.{USER_END}\
         {ASST_PREFIX}"
    )
}
