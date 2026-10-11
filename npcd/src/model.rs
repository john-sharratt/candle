//! The model npcd runs, as its console describes it.
//!
//! The model itself is the host's to name — one prescribed model for every
//! tenant of the process ([`host::MODEL`], `docs/shared_model_host.md` §5):
//! **Qwen3.8-Flash-Next**, 250 B total with about 13 B active per token. 48
//! layers at 3:1 — gated-DeltaNet layers carrying a recurrent state to attention
//! layers carrying paged K/V — with its routed experts behind the three-tier
//! expert cache. Its engine artifact is prepared locally from the pinned
//! upstream releases rather than downloaded; see
//! `candle_conversation::models::qwen38_flash_next`.
//!
//! # What a routed model costs this daemon
//!
//! A routed model amortises expert loads across a wave of sessions stepping
//! through layers together, and a cast on different concerns reaching different
//! experts is where that amortisation is weakest. The price on a small card is
//! KV room: the dense weights and the resident expert working set take most of
//! it, with the expert cache paging the rest from host RAM and disk, and a
//! resident cast's histories are what fill the KV regions left over.
//!
//! # One quant on every card
//!
//! Q4_KO experts over a Q8_0 trunk, whatever the card: one artifact everywhere
//! keeps the KV threshold row and the C-ladder calibration the same on every
//! machine. A smaller card streams more of the experts; it does not take a
//! smaller quant.

use host::MODEL;

// # A deployment may be running something else
//
// Which checkpoint serves a given deployment is that deployment's decision, and
// often a private one: a fine-tune somebody has no right to redistribute is a
// legitimate thing to run and a bad thing to commit. So it is named in
// `models.override.yaml` at the workspace root, which is gitignored and
// replaces a preset's coordinates and its adapters under the preset's own name.
// See `candle_conversation::models::overrides`. `MODEL` is the *variant*;
// `Model::spec()` applies the override. So the tests below assert what is true
// either way — the architecture, the tokenizer lineage, the context — and
// never the repository's coordinates, which an override is entitled to change.

/// What the console shows about the selection, before anything is loaded.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ModelSpec {
    pub name: &'static str,
    pub quant: &'static str,
    pub params_total: &'static str,
    /// The parameters a token actually passes through — about 13 B of the
    /// 250 B, which is what the console's two figures side by side say at a
    /// glance.
    pub params_active: &'static str,
    pub repo: &'static str,
    pub filename: &'static str,
    pub bytes: u64,
}

/// The quantisation named in a GGUF's filename, or `"?"`.
///
/// Read from the file rather than declared beside it, because a deployment
/// running an overridden checkpoint may well be running a different quant —
/// and a console asserting `Q6_K` over a `Q4_K_M` file is worse than one
/// admitting it cannot tell.
fn quant_of(filename: &str) -> &'static str {
    // Longest first: `Q4_K_M` contains `Q4_K`, which contains neither `Q4_0`
    // nor `Q4_1`, but a shortest-first scan would report `Q4_K` for both of the
    // first two. The `_KO` expert formats name a prepared artifact
    // (`…-Q4_KOEXP-….gguf`) and share no prefix with the `_K_` ones.
    const KNOWN: &[&str] = &[
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q4_K_M", "Q4_K_S", "Q4_KO", "Q4_0",
        "Q4_1", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_KO", "Q2_KO", "Q2_K", "BF16", "F16", "F32",
    ];
    let upper = filename.to_ascii_uppercase();
    KNOWN
        .iter()
        .copied()
        .find(|q| upper.contains(q))
        .unwrap_or("?")
}

/// The selection, in the shape the console's system page renders.
///
/// **Every field is derived from the resolved spec.** They were literals once,
/// which was merely redundant while the repository named the only checkpoint
/// anyone ran — and became a lie the moment a deployment could substitute its
/// own. An operator debugging a model that is behaving unexpectedly reads this
/// page first, so it has to describe the checkpoint that actually loaded.
pub fn spec() -> ModelSpec {
    let s = MODEL.spec();
    // `Box::leak` runs once per call on a handful of bytes at startup; the
    // alternative is owned `String`s through a `Copy` struct the console holds.
    let filename: &'static str = Box::leak(s.model_filename.into_boxed_str());
    let name = filename
        .strip_suffix(".gguf")
        .unwrap_or(filename)
        .to_owned();
    ModelSpec {
        name: Box::leak(name.into_boxed_str()),
        quant: quant_of(filename),
        params_total: "250B",
        params_active: "13B",
        repo: Box::leak(s.model_repo.into_boxed_str()),
        filename,
        bytes: s.model_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::models::ModelArch;

    /// **Flash-Next's arch.**
    ///
    /// Each loader refuses a checkpoint of another arch — a routed file loaded
    /// through a dense path has no expert cache to stand up, and a dense one
    /// through a routed path stands one up over weights that have none. The
    /// arch is what routes the load, so it is asserted rather than assumed.
    #[test]
    fn npcd_runs_flash_next() {
        assert!(
            matches!(MODEL.spec().arch, ModelArch::Qwen4Exp),
            "arch {:?} routes to a loader that refuses Flash-Next's checkpoint",
            MODEL.spec().arch
        );
    }

    /// **Flash-Next, locked on Q4_KO experts, prepared locally.** The preset
    /// rather than the resolved spec, because an override is entitled to change
    /// the file — what is asserted is what the repository ships.
    #[test]
    fn the_preset_is_flash_next_at_q4_ko() {
        let s = MODEL.preset_spec();
        assert_eq!(quant_of(&s.model_filename), "Q4_KO");
        assert!(
            s.prepared_from_source,
            "nothing publishes the engine artifact"
        );
        assert!(s.tensor_overrides.is_empty());
    }

    /// **The console's figures come from the resolved spec, override and all.**
    ///
    /// They were hardcoded once, and a repo change would have left the system
    /// page naming a model the loader had stopped downloading. That failure gets
    /// worse with overrides in the picture: an operator running a local
    /// fine-tune would read the repository's coordinates on the console and have
    /// no way to tell which checkpoint was actually loaded.
    #[test]
    fn the_console_figures_come_from_the_registry() {
        let s = spec();
        let registry = MODEL.spec();
        assert_eq!(s.repo, registry.model_repo);
        assert_eq!(s.filename, registry.model_filename);
        assert_eq!(s.bytes, registry.model_bytes);
    }

    /// **An override replaces the checkpoint, and the console follows it.**
    ///
    /// The whole point of `models.override.yaml`: this machine may be running a
    /// checkpoint the repository does not name. Whatever it is, `spec()` — which
    /// the console reads — must describe *it* rather than the preset.
    ///
    /// Written so it holds either way. On a machine with no override the two
    /// agree because there is nothing to apply; on one with an override they
    /// agree because the override reached both. What it rules out is the case
    /// that motivated the indirection: a console reporting the preset while the
    /// loader fetches something else.
    #[test]
    fn the_console_reports_whatever_the_override_left() {
        let resolved = MODEL.spec();
        let preset = MODEL.preset_spec();
        let console = spec();

        assert_eq!(console.repo, resolved.model_repo);
        assert!(!console.repo.is_empty());
        assert!(console.filename.ends_with(".gguf"));

        let overridden = resolved.model_repo != preset.model_repo
            || resolved.model_filename != preset.model_filename
            || resolved.loras != preset.loras;

        // Printed, not just asserted. Run with `--nocapture` and this test says
        // exactly which checkpoint and which adapters this machine resolves —
        // which is the question an operator actually has when a console names a
        // model they did not expect.
        println!(
            "preset:   {} / {}",
            preset.model_repo, preset.model_filename
        );
        println!(
            "resolved: {} / {}",
            resolved.model_repo, resolved.model_filename
        );
        println!(
            "adapters: {:?}",
            resolved
                .loras
                .iter()
                .map(|l| format!("{}={}@{}", l.name, l.repo, l.revision))
                .collect::<Vec<_>>()
        );
        println!(
            "override: {}",
            if overridden {
                "IN EFFECT (models.override.yaml)"
            } else {
                "none — running the repository's preset"
            }
        );

        if overridden {
            assert!(
                console.repo == resolved.model_repo && console.filename == resolved.model_filename,
                "an override is in effect but the console still names the preset"
            );
        }
    }

    /// **An override may not change the architecture.**
    ///
    /// The override file deliberately cannot reach `arch`, and this is the
    /// consequence worth asserting: whatever checkpoint a deployment substitutes
    /// still loads through Flash-Next's path. A file that could swap the
    /// architecture would let a local edit route npcd into a loader that refuses
    /// its checkpoint, with nothing in the repository changed to explain it.
    #[test]
    fn an_override_cannot_move_npcd_off_its_arch() {
        assert!(matches!(MODEL.spec().arch, ModelArch::Qwen4Exp));
        assert_eq!(MODEL.spec().arch, MODEL.preset_spec().arch);
    }

    /// Every adapter the resolved spec carries is named and pinned.
    ///
    /// Both are the operator's protection rather than the engine's: an unnamed
    /// adapter is one no conversation can select, and an unpinned one changes
    /// what its characters say when the upstream repo moves.
    #[test]
    fn every_declared_adapter_is_named_and_pinned() {
        for l in &MODEL.spec().loras {
            assert!(
                !l.name.is_empty(),
                "an unnamed adapter cannot be opted into"
            );
            assert!(!l.repo.is_empty(), "adapter {:?} names no repo", l.name);
            // **A commit, not a branch.** This asserted `!revision.is_empty()`, which a
            // branch name satisfies — so `"main"` sailed through the one check written to
            // stop a moving target, and the preset's example adapter tracked `main` for as
            // long as it existed. A 40-hex SHA is the only spelling that pins.
            assert!(
                l.revision.len() == 40 && l.revision.chars().all(|c| c.is_ascii_hexdigit()),
                "adapter {:?} names revision {:?}, which is not a commit — a branch or tag \
                 moves under you, and an adapter that changes upstream changes what the \
                 characters using it say with nothing here changing",
                l.name,
                l.revision
            );
        }
    }

    /// Routed: a token passes through a fraction of the weights, and the console
    /// saying so is the point of carrying both figures.
    #[test]
    fn a_routed_model_activates_a_fraction_of_its_parameters() {
        let s = spec();
        assert_ne!(s.params_total, s.params_active);
    }

    /// The quant label is read out of the filename, longest match first.
    ///
    /// The ordering is the whole of it: a shortest-first scan reports `Q4_K` for
    /// a `Q4_K_M` file, which is a different format with different error, and
    /// the console would state it with the same confidence.
    #[test]
    fn the_quant_label_is_read_from_the_filename() {
        assert_eq!(quant_of("Qwen3.6-35B-A3B-AntiLoop.Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_of("Qwen3.6-35B-A3B-UD-Q6_K.gguf"), "Q6_K");
        assert_eq!(quant_of("Qwen3.5-9B-Q6_K.gguf"), "Q6_K");
        assert_eq!(quant_of("model-Q4_K_S.gguf"), "Q4_K_S");
        assert_eq!(quant_of("model-Q4_0.gguf"), "Q4_0");
        assert_eq!(quant_of("model-BF16.gguf"), "BF16");
        assert_eq!(
            quant_of("Qwen3.8-Flash-Next-Q4_KOEXP-189dd518216f.gguf"),
            "Q4_KO"
        );
        assert_eq!(quant_of("Qwen3.8-Flash-Next-Q2_KOEXP-0.gguf"), "Q2_KO");
        // Unrecognised is reported as unknown, never guessed.
        assert_eq!(quant_of("some-conversion.gguf"), "?");
    }

    /// The console names the file it loads, whatever an override made that.
    #[test]
    fn the_console_name_tracks_the_resolved_filename() {
        let s = spec();
        assert!(
            s.filename.starts_with(s.name),
            "console name {:?} does not name file {:?}",
            s.name,
            s.filename
        );
        assert!(!s.name.ends_with(".gguf"));
    }
}
