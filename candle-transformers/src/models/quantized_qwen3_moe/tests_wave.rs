//! Equivalence tests for the wave engine: a co-batched forward must match the
//! same rows run separately.
//!
//! Each test compares rows run together with the same rows run alone. The cases
//! cover decode and prefill in one launch, split sweeps across layer windows,
//! ragged batches, and a step whose creep group stops mid-stack. A row handed to
//! the wrong place would corrupt decode or the ingest, so every scored row must
//! equal its solo full-sweep logits. The comparison is by cosine, which tolerates
//! the small reordering of a fused kernel. The tests run on the GPU and are
//! ignored by default.

use super::*;

// Tests

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_question_mark, clippy::type_complexity)]

    use super::*;

    use super::super::tests_gates::tests::{batched, instruct_2507_pack, INSTRUCT_2507_CONTEXT};

    /// Continuous-fair-wave equivalence gate (`docs/continuous_fair_waves.md`):
    /// the co-batched `forward_wave` (decode + prefill through the mixed dispatch +
    /// shared MoE) must produce the SAME logits as running decode and prefill as
    /// separate forwards, and the re-entrant `forward_batched_layers` split sweep
    /// must match a full sweep. GPU + a real model; ignored so it never runs in the
    /// normal (fast) suite.
    ///
    /// Run with:
    ///   cargo test -p candle-transformers --release --features cuda --lib \
    ///     quantized_qwen3_moe::tests::wave_equivalence -- --ignored --nocapture
    #[test]
    #[ignore]
    fn wave_equivalence() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ wave_equivalence requires --features cuda");
            Ok(())
        }
        #[cfg(feature = "cuda")]
        {
            use crate::models::batched_inference::{
                BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, WaveResult,
            };
            use crate::models::residual_rows::CreepResidual;
            use crate::models::wave_driver::{StepCreep, StepRows, StepSpec};

            let device = match Device::new_cuda(0) {
                Ok(d) => d,
                Err(_) => {
                    println!("skip: no CUDA device");
                    return Ok(());
                }
            };
            let pack = instruct_2507_pack(&device, None)?;
            let raw = ModelWeights::from_pack(&pack, &device, None)?;
            let model = batched(raw, INSTRUCT_2507_CONTEXT, 4096, &device)?;
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let n = model.num_layers();

            let mk = |t: &[u32]| -> Result<Tensor> { Tensor::new(t, &device)?.unsqueeze(0) };
            // Cosine of two logit rows, robust to tiny per-kernel fp reordering.
            let cos = |a: &Tensor, b: &Tensor| -> Result<f32> {
                let a = a.flatten_all()?.to_dtype(DType::F32)?;
                let b = b.flatten_all()?.to_dtype(DType::F32)?;
                let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
                let na = (&a * &a)?.sum_all()?.to_scalar::<f32>()?.sqrt();
                let nb = (&b * &b)?.sum_all()?.to_scalar::<f32>()?.sqrt();
                Ok(dot / (na * nb + 1e-6))
            };
            // Single-group reference forward through the unified wave entry: one
            // prefill group, full sweep. `forward_wave` reclassifies q=1 rows into
            // the decode group internally, so this is exactly the old `forward_batched`.
            let fb = |session: &mut BatchedInferenceSession,
                      seqs: &[usize],
                      inputs: &[Tensor]|
             -> Result<Vec<Tensor>> {
                Ok(model
                    .forward_wave(session, &[], &[], seqs, inputs, &[], &[], 0, n, None)?
                    .logits_owned()?)
            };
            // Layer-range reference through the wave entry (the retired
            // `forward_batched_layers`): prefill group, layers `[ls, le)`, residual.
            let fbl = |session: &mut BatchedInferenceSession,
                       seqs: &[usize],
                       inputs: &[Tensor],
                       ls: usize,
                       le: usize,
                       residual: &mut CreepResidual|
             -> Result<WaveResult> {
                model.forward_wave(
                    session,
                    &[],
                    &[],
                    seqs,
                    inputs,
                    &[],
                    &[],
                    ls,
                    le,
                    Some(residual),
                )
            };
            // Prefill an identical `ctx` context into `s`, leaving offset at ctx.len().
            let prep =
                |session: &mut BatchedInferenceSession, s: usize, ctx: &[u32]| -> Result<()> {
                    let _ = fb(session, &[s], &[mk(ctx)?])?;
                    session.advance_sequence(s, ctx.len())?;
                    Ok(())
                };

            let ctx_d: Vec<u32> = (100..=115u32).collect();
            let ctx_p: Vec<u32> = (200..=215u32).collect();
            let dec_tok = [42u32];
            let pre_tok = [51u32, 52, 53, 54, 55];

            // ── Test 1: co-batch (forward_wave) == separate forwards ──────────
            let d0 = session.create_sequence()?;
            let d1 = session.create_sequence()?;
            let p0 = session.create_sequence()?;
            let p1 = session.create_sequence()?;
            prep(&mut session, d0, &ctx_d)?;
            prep(&mut session, d1, &ctx_d)?;
            prep(&mut session, p0, &ctx_p)?;
            prep(&mut session, p1, &ctx_p)?;

            let sep_dec = fb(&mut session, &[d0], &[mk(&dec_tok)?])?;
            let sep_pre = fb(&mut session, &[p0], &[mk(&pre_tok)?])?;
            let wave = model
                .forward_wave(
                    &mut session,
                    &[d1],
                    &[mk(&dec_tok)?],
                    &[p1],
                    &[mk(&pre_tok)?],
                    &[],
                    &[],
                    0,
                    n,
                    None,
                )?
                .logits_owned()?;
            assert_eq!(wave.len(), 2, "wave logits = decode + prefill rows");
            let c_dec = cos(&sep_dec[0], &wave[0])?;
            let c_pre = cos(&sep_pre[0], &wave[1])?;
            println!("forward_wave vs separate: decode cos={c_dec:.5} prefill cos={c_pre:.5}");
            assert!(c_dec > 0.999, "decode logits diverged (cos={c_dec})");
            assert!(c_pre > 0.999, "prefill logits diverged (cos={c_pre})");

            // ── Test 2: re-entrant split sweep == full sweep ──────────────────
            let a = session.create_sequence()?;
            let b = session.create_sequence()?;
            prep(&mut session, a, &ctx_p)?;
            prep(&mut session, b, &ctx_p)?;
            let full = fb(&mut session, &[a], &[mk(&pre_tok)?])?;
            let k = n / 2;
            let mut mid = model.claim_residual(&session, pre_tok.len())?;
            fbl(&mut session, &[b], &[mk(&pre_tok)?], 0, k, &mut mid)?;
            let split =
                fbl(&mut session, &[b], &[mk(&pre_tok)?], k, n, &mut mid)?.logits_owned()?;
            let c_split = cos(&full[0], &split[0])?;
            println!("split sweep [0,{k})+[{k},{n}) vs full: cos={c_split:.5}");
            assert!(c_split > 0.999, "re-entrant sweep diverged (cos={c_split})");

            // ── Test 3a: decode-only wave with D>1 (flat single decode group) ──
            // Guards the mixed-dispatch fix: a >1-sequence decode wave packs flat
            // [1,D,h] and must be reshaped to the decode kernel's [D,1,h] and
            // routed by header type, not by dim(1). (D>1 wasn't covered by Test 1.)
            let e0 = session.create_sequence()?;
            let e1 = session.create_sequence()?;
            let e2 = session.create_sequence()?;
            let e3 = session.create_sequence()?;
            for &s in &[e0, e1, e2, e3] {
                prep(&mut session, s, &ctx_d)?;
            }
            let sep_d2 = fb(&mut session, &[e0, e1], &[mk(&dec_tok)?, mk(&dec_tok)?])?;
            let wave_d2 = model
                .forward_wave(
                    &mut session,
                    &[e2, e3],
                    &[mk(&dec_tok)?, mk(&dec_tok)?],
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    n,
                    None,
                )?
                .logits_owned()?;
            assert_eq!(wave_d2.len(), 2, "decode-only wave = 2 decode rows");
            let c_d0 = cos(&sep_d2[0], &wave_d2[0])?;
            let c_d1 = cos(&sep_d2[1], &wave_d2[1])?;
            println!("multi-decode wave vs separate: cos={c_d0:.5},{c_d1:.5}");
            assert!(c_d0 > 0.999 && c_d1 > 0.999, "multi-decode wave diverged");

            // ── Test 3b: co-batch a 1-token PREFILL ───────────────────────────
            // A single-token prefill is operationally a decode; the wave routes it
            // through the DECODE kernel (the canonical single-token path — the paged
            // prefill kernel diverges for q_len==1), so it must match the separate
            // 1-token forward, which `forward_batched` also decodes.
            let g0 = session.create_sequence()?;
            let g1 = session.create_sequence()?;
            let dg = session.create_sequence()?;
            let dg2 = session.create_sequence()?;
            prep(&mut session, g0, &ctx_p)?;
            prep(&mut session, g1, &ctx_p)?;
            prep(&mut session, dg, &ctx_d)?;
            prep(&mut session, dg2, &ctx_d)?;
            let one_tok = [61u32];
            let sep_dec1 = fb(&mut session, &[dg], &[mk(&dec_tok)?])?;
            let sep_pre1 = fb(&mut session, &[g0], &[mk(&one_tok)?])?;
            let wave_1 = model
                .forward_wave(
                    &mut session,
                    &[dg2],
                    &[mk(&dec_tok)?],
                    &[g1],
                    &[mk(&one_tok)?],
                    &[],
                    &[],
                    0,
                    n,
                    None,
                )?
                .logits_owned()?;
            assert_eq!(wave_1.len(), 2, "co-batch = decode + 1-token prefill");
            let c_1d = cos(&sep_dec1[0], &wave_1[0])?;
            let c_1p = cos(&sep_pre1[0], &wave_1[1])?;
            println!("1-token-prefill co-batch vs separate: decode={c_1d:.5} prefill={c_1p:.5}");
            assert!(c_1d > 0.999, "co-batch decode diverged (cos={c_1d})");
            assert!(c_1p > 0.999, "1-token prefill misrouted (cos={c_1p})");

            // ── Test 3c: a RAGGED forward_batched batch mixing a multi-token and a
            // 1-token sequence. `max_input_len>1` would route the whole batch as a
            // prefill, sending the 1-token row through the divergent prefill kernel;
            // the mixed-batch delegation must route it through the decode kernel so
            // each row matches its homogeneous single-sequence reference.
            let r0 = session.create_sequence()?;
            let r1 = session.create_sequence()?;
            let r0s = session.create_sequence()?;
            let r1s = session.create_sequence()?;
            prep(&mut session, r0, &ctx_p)?;
            prep(&mut session, r1, &ctx_d)?;
            prep(&mut session, r0s, &ctx_p)?;
            prep(&mut session, r1s, &ctx_d)?;
            // References: each row run alone (r0s = multi-token prefill kernel, r1s
            // = 1-token → decode kernel via `max_input_len==1`).
            let ref_multi = fb(&mut session, &[r0s], &[mk(&pre_tok)?])?;
            let ref_one = fb(&mut session, &[r1s], &[mk(&one_tok)?])?;
            // Mixed ragged batch [5-token, 1-token] in ONE wave prefill group.
            let mixed = fb(&mut session, &[r0, r1], &[mk(&pre_tok)?, mk(&one_tok)?])?;
            assert_eq!(mixed.len(), 2, "ragged batch = one logit row per sequence");
            let c_m = cos(&ref_multi[0], &mixed[0])?;
            let c_s = cos(&ref_one[0], &mixed[1])?;
            println!("ragged mixed-batch vs separate: multi={c_m:.5} single={c_s:.5}");
            assert!(c_m > 0.999, "ragged multi-token row diverged (cos={c_m})");
            assert!(c_s > 0.999, "ragged 1-token row misrouted (cos={c_s})");

            // ── Test 3d: a MIXED cohort split across TWO layer windows. This is the
            // re-entrant residual path with reclassification: single-token prefills
            // are folded into the decode group internally, but the residual is held
            // in CALLER order — the forward moves it out of its internal layout into
            // the arena, and gathers it back in on resume. A broken layout would
            // corrupt the whole cohort. Split-sweep logits must equal the full-sweep
            // of the same mixed batch (and each row its solo reference).
            let w0 = session.create_sequence()?;
            let w1 = session.create_sequence()?;
            prep(&mut session, w0, &ctx_p)?;
            prep(&mut session, w1, &ctx_d)?;
            let kk = n / 2;
            let mut mixed_res = model.claim_residual(&session, pre_tok.len() + one_tok.len())?;
            fbl(
                &mut session,
                &[w0, w1],
                &[mk(&pre_tok)?, mk(&one_tok)?],
                0,
                kk,
                &mut mixed_res,
            )?;
            let mixed_split = fbl(
                &mut session,
                &[w0, w1],
                &[mk(&pre_tok)?, mk(&one_tok)?],
                kk,
                n,
                &mut mixed_res,
            )?
            .logits_owned()?;
            assert_eq!(mixed_split.len(), 2, "split cohort = one logit row per seq");
            let c_sm = cos(&ref_multi[0], &mixed_split[0])?;
            let c_ss = cos(&ref_one[0], &mixed_split[1])?;
            println!("mixed cohort split-sweep vs separate: multi={c_sm:.5} single={c_ss:.5}");
            assert!(
                c_sm > 0.999,
                "split mixed multi-token row diverged (cos={c_sm})"
            );
            assert!(
                c_ss > 0.999,
                "split mixed 1-token row corrupted on resume (cos={c_ss})"
            );

            // ── Test 3e: a STEP — decode co-batched with a creep group whose
            // window stops mid-stack, then the creep alone to the head
            // (`docs/wave_step.md`). The decode row crosses every layer in the
            // first step; the group (prefill-multi, section-multi, section-1tok —
            // the 1-token member folded into the decode kernel internally) parks
            // its rows in its arena at the window's end and resumes from there.
            // Every row must equal its solo full-sweep — a row handed to the wrong
            // place would corrupt decode or the ingest.
            let kk2 = n / 2;
            // Solo references (each row alone, full sweep) on their own sequences.
            let rda = session.create_sequence()?;
            let rga = session.create_sequence()?;
            let rgb = session.create_sequence()?;
            let rgc = session.create_sequence()?;
            for &s in &[rda, rga, rgb, rgc] {
                prep(&mut session, s, &ctx_d)?;
            }
            let sref_d = fb(&mut session, &[rda], &[mk(&dec_tok)?])?;
            let sref_a = fb(&mut session, &[rga], &[mk(&pre_tok)?])?;
            let sref_b = fb(&mut session, &[rgb], &[mk(&pre_tok)?])?;
            let sref_c = fb(&mut session, &[rgc], &[mk(&one_tok)?])?;
            // Co-batch decode [da] + group [ga, gb, gc], paused at kk2.
            let da = session.create_sequence()?;
            let ga = session.create_sequence()?;
            let gb = session.create_sequence()?;
            let gc = session.create_sequence()?;
            for &s in &[da, ga, gb, gc] {
                prep(&mut session, s, &ctx_d)?;
            }
            let dec_in = [mk(&dec_tok)?];
            let grp_in = [mk(&pre_tok)?, mk(&pre_tok)?, mk(&one_tok)?];
            let mut grp_res: Option<CreepResidual> = None;
            let first = model.forward_step(
                &mut session,
                StepSpec {
                    decode: StepRows {
                        seqs: &[da],
                        inputs: &dec_in,
                    },
                    verify: StepRows::NONE,
                    glue: StepRows::NONE,
                    creep: Some(StepCreep {
                        rows: StepRows {
                            seqs: &[ga, gb, gc],
                            inputs: &grp_in,
                        },
                        cursor: 0,
                        win_end: kk2,
                        residual: &mut grp_res,
                    }),
                },
            )?;
            assert!(first.creep_advanced, "the window ran and committed");
            let dec_fin = first.head?.logits_owned()?;
            // The group alone, from the layer its window stopped at to the head.
            let rest = model.forward_step(
                &mut session,
                StepSpec {
                    decode: StepRows::NONE,
                    verify: StepRows::NONE,
                    glue: StepRows::NONE,
                    creep: Some(StepCreep {
                        rows: StepRows {
                            seqs: &[ga, gb, gc],
                            inputs: &grp_in,
                        },
                        cursor: kk2,
                        win_end: n,
                        residual: &mut grp_res,
                    }),
                },
            )?;
            assert!(rest.creep_advanced);
            assert!(
                grp_res.is_none(),
                "a creep that reached the head gives its arena back"
            );
            let grp_fin = rest.head?.logits_owned()?;
            assert_eq!(dec_fin.len(), 1, "the step scored its decode row");
            assert_eq!(grp_fin.len(), 3, "group resume = one row per member");
            let c_ed = cos(&sref_d[0], &dec_fin[0])?;
            let c_ea = cos(&sref_a[0], &grp_fin[0])?;
            let c_eb = cos(&sref_b[0], &grp_fin[1])?;
            let c_ec = cos(&sref_c[0], &grp_fin[2])?;
            println!("step: decode={c_ed:.5} creep resumed=[{c_ea:.5},{c_eb:.5},{c_ec:.5}]");
            assert!(
                c_ed > 0.999,
                "decode corrupted crossing the step (cos={c_ed})"
            );
            assert!(
                c_ea > 0.999 && c_eb > 0.999 && c_ec > 0.999,
                "group member corrupted through its arena (cos=[{c_ea},{c_eb},{c_ec}])"
            );

            // ── Test 3f: the THREE-group step — decode + a creep group + a GLUE
            // group — with the creep's window ending mid-stack. This is the step's
            // full geometry (`decode_forward_cobatched` folding deferred glue as a
            // full-sweep member): caller order `[decode | creep | glue]`, so the
            // segment after the window takes the decode rows from ahead of the
            // creep and the glue rows from behind it, and the creep's rows leave
            // from between them. The creep carries a 1-token member (folded into
            // decode internally) so the layouts differ from caller order. (The
            // glue group runs as a plain ragged group — no pending scatter
            // descriptors — which is enough for the geometry; the glue kernel's
            // scatter/mask is covered by the gap-fill tests. Glue scores nothing,
            // so its rows are checked by the decode row it rides beside.) Every
            // scored row must equal its solo full-sweep.
            let kk3 = n / 2;
            let hda = session.create_sequence()?;
            let hca = session.create_sequence()?;
            let hcb = session.create_sequence()?;
            let hgl = session.create_sequence()?;
            for &s in &[hda, hca, hcb, hgl] {
                prep(&mut session, s, &ctx_d)?;
            }
            // Solo references (each row alone, full sweep), on sequences of their
            // own: a reference run writes its rows into its sequence's KV.
            let rhd = session.create_sequence()?;
            let rhca = session.create_sequence()?;
            let rhcb = session.create_sequence()?;
            for &s in &[rhd, rhca, rhcb] {
                prep(&mut session, s, &ctx_d)?;
            }
            let href_d = fb(&mut session, &[rhd], &[mk(&dec_tok)?])?;
            let href_ca = fb(&mut session, &[rhca], &[mk(&pre_tok)?])?;
            let href_cb = fb(&mut session, &[rhcb], &[mk(&one_tok)?])?;
            let dec_in3 = [mk(&dec_tok)?];
            let creep_in3 = [mk(&pre_tok)?, mk(&one_tok)?];
            let glue_in3 = [mk(&pre_tok)?];
            let mut creep_res3: Option<CreepResidual> = None;
            let first3 = model.forward_step(
                &mut session,
                StepSpec {
                    decode: StepRows {
                        seqs: &[hda],
                        inputs: &dec_in3,
                    },
                    verify: StepRows::NONE,
                    glue: StepRows {
                        seqs: &[hgl],
                        inputs: &glue_in3,
                    },
                    creep: Some(StepCreep {
                        rows: StepRows {
                            seqs: &[hca, hcb],
                            inputs: &creep_in3,
                        },
                        cursor: 0,
                        win_end: kk3,
                        residual: &mut creep_res3,
                    }),
                },
            )?;
            let dec_fin3 = first3.head?.logits_owned()?;
            let rest3 = model.forward_step(
                &mut session,
                StepSpec {
                    decode: StepRows::NONE,
                    verify: StepRows::NONE,
                    glue: StepRows::NONE,
                    creep: Some(StepCreep {
                        rows: StepRows {
                            seqs: &[hca, hcb],
                            inputs: &creep_in3,
                        },
                        cursor: kk3,
                        win_end: n,
                        residual: &mut creep_res3,
                    }),
                },
            )?;
            let creep_fin3 = rest3.head?.logits_owned()?;
            let c_fd = cos(&href_d[0], &dec_fin3[0])?;
            let c_fca = cos(&href_ca[0], &creep_fin3[0])?;
            let c_fcb = cos(&href_cb[0], &creep_fin3[1])?;
            println!("three-group step: decode={c_fd:.5} creep resumed=[{c_fca:.5},{c_fcb:.5}]");
            assert!(c_fd > 0.999, "decode corrupted beside glue (cos={c_fd})");
            assert!(
                c_fca > 0.999 && c_fcb > 0.999,
                "creep member corrupted leaving from between decode and glue \
                 (cos=[{c_fca},{c_fcb}])"
            );
            // Glue scores nothing, so its rows are checked through the next
            // token: a decode on the glue sequence attends to the K/V the step
            // wrote for it, and must equal a decode on a sequence that took the
            // same rows alone.
            let glue_next = |session: &mut BatchedInferenceSession, glue: usize| -> Result<f32> {
                let r = session.create_sequence()?;
                prep(session, r, &ctx_d)?;
                fb(session, &[r], &[mk(&pre_tok)?])?;
                let want = fb(session, &[r], &[mk(&dec_tok)?])?;
                let got = fb(session, &[glue], &[mk(&dec_tok)?])?;
                cos(&want[0], &got[0])
            };
            let c_fg = glue_next(&mut session, hgl)?;
            println!("three-group step: glue's next token={c_fg:.5}");
            assert!(
                c_fg > 0.999,
                "glue rows corrupted beside the creep (cos={c_fg})"
            );

            // ── Test 3g: a step of THREE segments — the creep resuming mid-stack
            // with decode and glue riding every layer — so the full-sweep rows
            // run alone up to the creep's window, are handed around the creep
            // through it, and run alone again after it. The creep carries a
            // three-token and a one-token member, so every hand-off crosses a
            // layout that differs from caller order.
            let (kk4a, kk4b) = (n / 4, 3 * n / 4);
            let gda = session.create_sequence()?;
            let gca = session.create_sequence()?;
            let gcb = session.create_sequence()?;
            let ggl = session.create_sequence()?;
            let rgd = session.create_sequence()?;
            let rgca = session.create_sequence()?;
            let rgcb = session.create_sequence()?;
            for &s in &[gda, gca, gcb, ggl, rgd, rgca, rgcb] {
                prep(&mut session, s, &ctx_d)?;
            }
            let gref_d = fb(&mut session, &[rgd], &[mk(&dec_tok)?])?;
            let gref_ca = fb(&mut session, &[rgca], &[mk(&pre_tok)?])?;
            let gref_cb = fb(&mut session, &[rgcb], &[mk(&one_tok)?])?;
            let creep_in4 = [mk(&pre_tok)?, mk(&one_tok)?];
            let creep4 = StepRows {
                seqs: &[gca, gcb],
                inputs: &creep_in4,
            };
            let mut creep_res4: Option<CreepResidual> = None;
            // The creep alone to its first stop.
            let lead = model.forward_step(
                &mut session,
                StepSpec {
                    decode: StepRows::NONE,
                    verify: StepRows::NONE,
                    glue: StepRows::NONE,
                    creep: Some(StepCreep {
                        rows: creep4,
                        cursor: 0,
                        win_end: kk4a,
                        residual: &mut creep_res4,
                    }),
                },
            )?;
            assert!(lead.creep_advanced);
            lead.head?;
            // Decode and glue across every layer, the creep across [kk4a, kk4b).
            let dec_in4 = [mk(&dec_tok)?];
            let glue_in4 = [mk(&pre_tok)?];
            let mid = model.forward_step(
                &mut session,
                StepSpec {
                    decode: StepRows {
                        seqs: &[gda],
                        inputs: &dec_in4,
                    },
                    verify: StepRows::NONE,
                    glue: StepRows {
                        seqs: &[ggl],
                        inputs: &glue_in4,
                    },
                    creep: Some(StepCreep {
                        rows: creep4,
                        cursor: kk4a,
                        win_end: kk4b,
                        residual: &mut creep_res4,
                    }),
                },
            )?;
            assert!(mid.creep_advanced);
            let dec_fin4 = mid.head?.logits_owned()?;
            assert_eq!(dec_fin4.len(), 1, "the step scored its decode row alone");
            // The creep alone to the head.
            let tail = model.forward_step(
                &mut session,
                StepSpec {
                    decode: StepRows::NONE,
                    verify: StepRows::NONE,
                    glue: StepRows::NONE,
                    creep: Some(StepCreep {
                        rows: creep4,
                        cursor: kk4b,
                        win_end: n,
                        residual: &mut creep_res4,
                    }),
                },
            )?;
            assert!(
                creep_res4.is_none(),
                "a creep at the head gives its arena back"
            );
            let creep_fin4 = tail.head?.logits_owned()?;
            let c_gd = cos(&gref_d[0], &dec_fin4[0])?;
            let c_gca = cos(&gref_ca[0], &creep_fin4[0])?;
            let c_gcb = cos(&gref_cb[0], &creep_fin4[1])?;
            let c_gg = glue_next(&mut session, ggl)?;
            println!(
                "three-segment step: decode={c_gd:.5} creep=[{c_gca:.5},{c_gcb:.5}] \
                 glue's next token={c_gg:.5}"
            );
            assert!(
                c_gd > 0.999,
                "decode corrupted around a mid-stack creep (cos={c_gd})"
            );
            assert!(
                c_gca > 0.999 && c_gcb > 0.999,
                "creep corrupted resuming beside full-sweep rows (cos=[{c_gca},{c_gcb}])"
            );
            assert!(
                c_gg > 0.999,
                "glue corrupted around a mid-stack creep (cos={c_gg})"
            );

            Ok(())
        }
    }
}
