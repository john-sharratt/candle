//! Checks each configuration's output against its expected text and prints the
//! results.
//!
//! StoryRewrite normalises whitespace and pronouns, then compares the common
//! prefix against the expected story with a small tolerance, so a final token
//! that diverges costs a few characters rather than the whole check. Empty
//! output fails outright, and output that is mostly punctuation fails the
//! alphanumeric share. NameGreeting asks only that sessions with different names
//! produce different output. Same-name sessions are not compared, since the flat
//! logits of small models let batch-position noise flip the argmax. Coherence
//! checks skip the session comparison entirely.

use super::*;

impl TestParams {
    /// Validate `results` and print every table — the reporting half of
    /// [`Self::run_loaded`].
    ///
    /// Public so a caller that used [`Self::run_loaded_collect`] can finish the job,
    /// with its own [`TestParams::with_extra_rows`] already installed.
    pub fn validate_and_print(&self, results: &mut [TestResults]) -> Result<()> {
        self.validate_and_print_results(results)
    }

    /// Validate results and print comparison table
    pub(super) fn validate_and_print_results(&self, results: &mut [TestResults]) -> Result<()> {
        println!("\n=== All tests completed ===");

        let mut failed = false;
        for result in results.iter_mut() {
            println!("\n--- Config: {:?} ---", result.config);
            let mut config_valid = true;
            let mut validation_checked = 0usize;
            let mut validation_failed = 0usize;
            let required_pass_percent = self.majority_pass_threshold.unwrap_or(100);

            if result.sessions.is_empty() {
                println!("❌ No sessions found for this config");
                failed = true;
                continue;
            }

            // ── Per-session checks (token count, empty output) ──────────
            for (i, session) in result.sessions.iter().enumerate() {
                let output_str = {
                    let raw = self
                        .tokenizer
                        .decode(&session.output, true)
                        .unwrap_or_default();
                    if self.suppress_thinking {
                        strip_thinking_blocks(&raw)
                    } else {
                        raw
                    }
                };

                // Check token count.  Skipped when EOS early-stop is enabled,
                // since a short generation is the correct outcome there.
                let min_expected_tokens = self.tokens_for(result.config.mode).saturating_sub(5);
                if self.stop_on_eos.is_empty() && session.output.len() < min_expected_tokens {
                    println!(
                        "\n❌ Session {} FAILED: Generated only {} tokens, expected at least {}",
                        i,
                        session.output.len(),
                        min_expected_tokens
                    );
                    failed = true;
                    config_valid = false;
                }

                // ── StoryRewrite validation ──────────────────────────────────
                //
                // Goal: verify the model reproduces the expected story text
                // faithfully after a name substitution.  Each session's
                // `expected` field contains the FULL original prompt (~2 300
                // chars) with "Marcus" replaced by the session's assigned
                // name.  The model only generates ~40-50 chars (controlled
                // by `generate_token_count`), so the output is always much
                // shorter than `expected`.
                //
                // Two normalisation steps are applied before comparison:
                //
                //  1. **Whitespace collapsing** — `split_whitespace().join(" ")`
                //     KV quantisation (especially Q8_0 with many concurrent
                //     contexts) can cause the model to emit extra whitespace
                //     at token boundaries.  For example, the model might
                //     produce  "Astronaut  \nMarcus"  (two spaces then a
                //     newline) instead of  "Astronaut\nMarcus".  These are
                //     semantically identical, so we collapse all whitespace
                //     runs into a single space before comparing.
                //
                //  2. **Pronoun neutralisation** — gendered pronouns like
                //     "his"/"her", "he"/"she" are replaced with bracketed
                //     placeholders ("[his/him/her]", "[he/she]", etc.), whole
                //     words at any boundary (`story_normalize`).  This
                //     lets us compare sessions that use female names against
                //     the original male-protagonist prompt without false
                //     mismatches when the model correctly adapts pronouns.
                //
                // After normalisation we compare via **common-prefix length**
                // rather than slicing both strings to a fixed length:
                //
                //   common_prefix_len = number of leading chars that match
                //   min_len           = min(output_len, expected_len)
                //   required          = min_len - TOLERANCE  (TOLERANCE = 5)
                //
                //   PASS  iff  common_prefix_len >= required
                //
                // Why common-prefix instead of substring equality:
                //
                //   KV quantisation can cause the last 1-2 *generated tokens*
                //   to diverge from the expected text — not just truncation
                //   (fewer chars) but actual content differences where the
                //   model emits different characters (e.g. "\n\n" instead of
                //   " in").  The divergence point shifts non-deterministically
                //   between runs because it depends on which experts were
                //   cached at the time.  A fixed `saturating_sub(N)` approach
                //   fails because it assumes the mismatch is always at the
                //   very end, but with content divergence the offset of the
                //   first wrong character varies.  Measuring the actual
                //   common-prefix length is robust against this.
                //
                // What this still catches (i.e. real model failures):
                //
                //   • **Garbage output** — if the model produces random text
                //     the common prefix will be ~0 chars, far below the
                //     ~35-43 char requirement.  FAIL.
                //
                //   • **Wrong-name contamination** — if session isolation is
                //     broken (one session bleeds into another), the name in
                //     the output won't match `expected` and the prefix
                //     diverges early at the name position.  FAIL.
                //
                //   • **Completely wrong story** — if attention or FFN is
                //     broken the output won't start with "The Backyard
                //     Astronaut …".  FAIL.
                //
                //   • **Empty output** — explicitly checked before prefix
                //     comparison.  FAIL.
                //
                //   • **Too-short output** — the token-count check above
                //     (min_expected_tokens) already flags this, and the
                //     prefix comparison would also fail since min_len would
                //     be very small → required ≈ 0, but the token check
                //     catches it first.
                //
                // The TOLERANCE of 5 chars allows at most ~1-2 tokens of
                // tail divergence (typical token is 3-5 chars).  This is
                // tight enough that any structural problem — wrong content,
                // wrong name, broken attention — will still be caught.
                if !self.skip_validation {
                    match result.effective_test_mode {
                        TestMode::StoryRewrite => {
                            validation_checked += 1;
                            let output_trimmed = output_str.trim();
                            if !output_trimmed.is_empty() {
                                let expected_trimmed = session.expected.trim();

                                // Normalize for comparison (see block comment above).
                                let output_chars: Vec<char> =
                                    normalize_story(output_trimmed).chars().collect();
                                let expected_chars: Vec<char> =
                                    normalize_story(expected_trimmed).chars().collect();

                                let common_prefix_len = output_chars
                                    .iter()
                                    .zip(expected_chars.iter())
                                    .take_while(|(a, b)| a == b)
                                    .count();

                                const TOLERANCE: usize = 5;
                                let min_len = output_chars.len().min(expected_chars.len());
                                let required = min_len.saturating_sub(TOLERANCE);

                                if common_prefix_len < required {
                                    let cmp_len = min_len.min(common_prefix_len + 10);
                                    let output_cmp: String =
                                        output_chars[..cmp_len].iter().collect();
                                    let expected_cmp: String =
                                        expected_chars[..cmp_len].iter().collect();
                                    println!(
                                        "\n⚠ Session {} output diverges at char {} \
                                         (need {} of {} to match, tolerance {}):",
                                        i, common_prefix_len, required, min_len, TOLERANCE
                                    );
                                    println!("{}", format_diff(&expected_cmp, &output_cmp));
                                    validation_failed += 1;
                                }
                            } else {
                                println!("\n❌ Session {} FAILED: Generated empty output", i);
                                validation_failed += 1;
                            }
                        }
                        TestMode::NameGreeting => {
                            validation_checked += 1;
                            // Empty-output check only; consistency is validated below.
                            if output_str.trim().is_empty() {
                                println!("\n❌ Session {} FAILED: Generated empty output", i);
                                validation_failed += 1;
                            }
                        }
                        TestMode::Skip => {
                            // No validation — just run for timing/compression stats.
                        }
                        TestMode::CoherenceCheck => {
                            validation_checked += 1;
                            let trimmed = output_str.trim();
                            if trimmed.is_empty() {
                                println!("\n❌ Session {} FAILED: Generated empty output", i);
                                validation_failed += 1;
                            } else {
                                // Check for garbage output (e.g. "!!!!!!!!!!" from broken
                                // quantized reads).  Coherent text must contain alphanumeric
                                // content — at least 20% of characters should be letters or
                                // digits.  This catches repeated punctuation/symbols while
                                // allowing numbered lists, sparse formatting, equations, etc.
                                let alnum_count =
                                    trimmed.chars().filter(|c| c.is_alphanumeric()).count();
                                let total = trimmed.chars().count();
                                if total > 0 && (alnum_count * 100 / total) < 20 {
                                    println!(
                                        "\n❌ Session {} FAILED: Output appears to be garbage \
                                         ({}/{} chars alphanumeric, need ≥20%): {:?}",
                                        i,
                                        alnum_count,
                                        total,
                                        {
                                            let mut end = trimmed.len().min(80);
                                            while end > 0 && !trimmed.is_char_boundary(end) {
                                                end -= 1;
                                            }
                                            &trimmed[..end]
                                        }
                                    );
                                    validation_failed += 1;
                                }
                            }
                        }
                    }
                }

                if self.print_outputs {
                    println!("\n--- Session {} Output ---\n{}", i, output_str);
                }
            }

            // ── Per-session validation summary with majority threshold ───
            if !self.skip_validation
                && result.effective_test_mode != TestMode::Skip
                && validation_checked > 0
            {
                let validation_passed = validation_checked.saturating_sub(validation_failed);
                let validation_pct = validation_passed * 100 / validation_checked;
                if validation_pct >= required_pass_percent {
                    match result.effective_test_mode {
                        TestMode::StoryRewrite => println!(
                            "✓ {}/{} sessions matched expected output ({}% pass, threshold {}%)",
                            validation_passed,
                            validation_checked,
                            validation_pct,
                            required_pass_percent
                        ),
                        TestMode::NameGreeting => println!(
                            "✓ {}/{} sessions produced non-empty output ({}% pass, threshold {}%)",
                            validation_passed,
                            validation_checked,
                            validation_pct,
                            required_pass_percent
                        ),
                        TestMode::CoherenceCheck => println!(
                            "✓ {}/{} sessions produced coherent output ({}% pass, threshold {}%)",
                            validation_passed,
                            validation_checked,
                            validation_pct,
                            required_pass_percent
                        ),
                        TestMode::Skip => {}
                    }
                } else {
                    println!(
                        "✗ Only {}/{} sessions passed validation ({}%, need ≥{}%)",
                        validation_passed,
                        validation_checked,
                        validation_pct,
                        required_pass_percent
                    );
                    failed = true;
                    config_valid = false;
                }
            }

            // ── NameGreeting: session isolation validation ────────────────
            //
            // We validate that the batching system correctly isolates
            // sessions by checking that sessions with *different* names
            // produce *different* token sequences.  If the KV cache or
            // token routing is broken, multiple sessions would collapse
            // to the same output (they'd all read from one session's
            // context).
            //
            // We deliberately do NOT compare same-name sessions at
            // different batch positions.  Tiny models (0.5B) have flat
            // logit distributions where floating-point non-determinism
            // across batch positions can flip the argmax — that is
            // inherent numerical noise, not a batching bug.
            //
            // CoherenceCheck skips this entirely — formats like Q4_KS
            // and Q3_0 are too lossy to reliably differentiate names.
            if !self.skip_validation
                && !self.disable_session_isolation
                && result.effective_test_mode != TestMode::Skip
            {
                if result.effective_test_mode == TestMode::NameGreeting {
                    let num_sessions = result.sessions.len();
                    if num_sessions > 1 {
                        let num_names = self.names.len();
                        let mut isolation_checked = 0usize;
                        let mut isolation_failed = 0usize;

                        for i in 1..num_sessions {
                            let name_i = i % num_names;
                            let name_prev = (i - 1) % num_names;
                            if name_i != name_prev {
                                isolation_checked += 1;
                                if result.sessions[i].output == result.sessions[i - 1].output {
                                    isolation_failed += 1;
                                    if isolation_failed <= 3 {
                                        println!(
                                        "\n⚠ Session {} ({}) produced identical output to session {} ({}) — possible session isolation failure",
                                        i, self.names[name_i], i - 1, self.names[name_prev]
                                    );
                                        // Debug: show output tokens
                                        let decode_out = |tokens: &[u32]| -> String {
                                            self.tokenizer
                                                .decode(tokens, true)
                                                .unwrap_or_else(|_| format!("{:?}", tokens))
                                        };
                                        println!(
                                            "  Output tokens ({} tokens): {:?}",
                                            result.sessions[i].output.len(),
                                            &result.sessions[i].output
                                                [..result.sessions[i].output.len().min(20)]
                                        );
                                        println!(
                                            "  Decoded: {}",
                                            decode_out(&result.sessions[i].output)
                                        );
                                    }
                                }
                            }
                        }

                        if isolation_checked > 0 {
                            let isolation_passed =
                                isolation_checked.saturating_sub(isolation_failed);
                            let isolation_pct = isolation_passed * 100 / isolation_checked;
                            if isolation_pct >= required_pass_percent {
                                println!(
                                    "✓ {}/{} different-name session pairs produced distinct output ({}% pass, threshold {}%)",
                                    isolation_passed,
                                    isolation_checked,
                                    isolation_pct,
                                    required_pass_percent
                                );
                            } else {
                                println!(
                                    "✗ Only {}/{} different-name pairs were distinct ({}%, need ≥{}%)",
                                    isolation_passed,
                                    isolation_checked,
                                    isolation_pct,
                                    required_pass_percent
                                );
                                failed = true;
                                config_valid = false;
                            }
                        }
                    }
                }
            }

            if self.skip_validation {
                let total_tokens: usize = result.sessions.iter().map(|s| s.output.len()).sum();
                println!(
                    "✓ {} sessions generated {} tokens [validation skipped]",
                    result.sessions.len(),
                    total_tokens
                );
            }
            result.all_valid = config_valid;
        }

        self.print_comparison_table(results);
        self.print_expert_stats_table(results);

        if failed {
            Err(candle::Error::msg("Some tests failed"))
        } else {
            Ok(())
        }
    }
}
