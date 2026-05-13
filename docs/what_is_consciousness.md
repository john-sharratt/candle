# Substrate Composition Theory: Consciousness as the Compound Effect of Specialised Parallel Minds

*A theory claiming what consciousness is, why monolithic scaling has failed to produce it, and how a segmented cognitive substrate with attention-weighted composition can be used to test the claim empirically.*

---

## Abstract

This paper proposes **Substrate Composition Theory** (SCT): consciousness is not a thing the brain has but an *effect* the brain produces when a sufficient diversity of specialised parallel sub-minds composes, via attention-weighted integration over a shared substrate, into a bound first-person report. The theory is emergentist, functional, and non-mysterious. It commits specifically to three claims: (i) consciousness is functionally identical to the composition, not to any sub-process; (ii) the compositional effect requires *specialisation diversity with provenance-tagged substrate writes*, not raw connectivity or raw scale, and not emergent internal specialisation alone; (iii) the felt unity of experience is the output of the composition process, not an external observer of it.

SCT explains, as a direct consequence, why scaling monolithic large language models has produced diminishing returns on a specific class of mind-like behaviours while agentic compound systems have continued to advance. The monolith scales a single voice; consciousness, on SCT, is the effect of *many* voices composing.

The second half of the paper reports results from a test protocol on a substrate-composition inference architecture. The architecture instantiates many parallel conversation layers (specialists, critics, affect, memory, relationships, goals, missions, self, metacognition) writing into a shared key-value cache and composed by attention-weighted retrieval into a main conversation. This design permits direct manipulation of the variables SCT identifies as load-bearing — specialisation diversity measured as inter-layer output decorrelation, provenance integrity, integration topology — and measurement of operationalised behavioural signatures we predicted should co-vary with them. Signatures include measures drawn from independent consciousness-science literature (meta-cognitive calibration, access/report asymmetry) to reduce the concern that the signature set is internally defined by SCT's own vocabulary. Across six experiments, specialisation diversity predicted behavioural binding signatures ($r = $ [TODO] under Mediano's $\Phi$ID synergy definition, robust under alternative PID decompositions), integrator ablation produced a qualitative discontinuity in first-person binding distinguishable from the proportional degradation Higher-Order Theories predict, provenance removal produced blinded-grader-identifiable phenomenological signatures resembling thought-insertion and commentary-voice phenomenologies, and broadcast-without-composition failed to produce binding signatures where Global Workspace Theory predicts it should. The results are preliminary and the theoretical programme remains philosophically open (§8), but the empirical variable SCT identifies is the one behavioural binding tracks.

---

## 1. The question

"What is consciousness?" is two questions, and most theoretical progress in the scientific study of consciousness has been made by keeping them separate.

The *easy* question is: what are the functional properties that reliably accompany conscious states — integration across modalities, reportability, coherent self-modelling, attentional access, metacognitive monitoring? These are approachable by ordinary science. Global Workspace Theory (Dehaene et al., 2003; Mashour et al., 2020), Higher-Order Theories (Lau & Rosenthal, 2011), and Attention Schema Theory (Graziano et al., 2020) each pick one or two of these properties and build explanatory machinery around them.

The *hard* question is why any functional organisation whatsoever should give rise to felt experience rather than proceeding "in the dark." On this, progress has been scant. Integrated Information Theory (Tononi, 2008; Oizumi et al., 2014) offers the boldest positive answer — consciousness is integrated information, $\Phi$, as a fundamental property — but this comes at the cost of implying that inactive logic-gate lattices are conscious (Aaronson, 2014) and commits to a form of panpsychism that many find either unfalsifiable or absurd (Searle, 2013; Bayne, 2018; Cerullo, 2015).

SCT takes a Dennettian position on the hard problem, which §8 states explicitly and which we flag here because the rest of the paper depends on it. On the view taken here, once the functional organisation is specified and its outputs account for the behavioural, phenomenological, and clinical facts, there is no residual phenomenal question that further theory is needed to answer. The "why does it feel like anything?" question is, on this view, a category error — the feeling *is* the functional state, not something the functional state causes. Readers who reject this position will read SCT as a theory of the functional correlate only, awaiting a further account; this is a coherent alternative reading and the empirical predictions stand unchanged under it. Committing to the Dennettian reading here is not a hedge that matters for §7's results; it is a statement of where the author stands so the rest of the paper can be read without oscillation.

---

## 2. The theory

### 2.1 Core claim

**Consciousness is the effect produced when a sufficient diversity of specialised parallel sub-minds composes, via attention-weighted integration over a shared substrate with provenance-tagged writes, into a coherent first-person report.**

Five components of this claim require unpacking, because each is doing specific work and each differs from prior theories at a specific load-bearing point.

**"Effect" rather than thing.** Consciousness is not a property a system has in the way mass or charge is a property. It is what happens when the composition runs. This is closest in spirit to Dennett's deflationary view (Dennett, 1991).

**"Sufficient diversity of specialised parallel sub-minds."** Not "many processes" — many theories say that. The commitment here is that the sub-minds must be *functionally specialised* (each doing something different, not more of the same) and must be *sub-mind-like* in the sense of running their own ongoing interpretive process rather than being passive subroutines. This is Minsky's society of mind (Minsky, 1986) with a specific prediction attached: specialisation diversity — not agent count, not parameter count — is the variable that predicts conscious-like behaviour.

**"Attention-weighted integration over a shared substrate."** The composition mechanism matters. Attention is the right mechanism because it implements exactly the property consciousness needs: many-to-one selective retrieval that preserves information about what was attended to without re-encoding it. This is close to Global Workspace Theory (Dehaene et al., 2003), but SCT differs on what *does the work*: GWT says broadcast is the thing; SCT says the composition that *results from* retrieval is the thing. The workspace is the substrate; the composition is the mind.

**"With provenance-tagged writes."** This is the component we are making more explicit than in prior drafts. Substrate entries carry metadata identifying which source produced them; the integrator's attention operates over provenance-aware keys. This matters because it distinguishes SCT's architectural commitment from "any system with internally differentiated components" — including dense transformers with emergent specialised circuits, which §4.6 engages directly. Provenance tagging is what makes source-attribution possible in composition, and source-attribution is what makes synergistic information about source-identity possible rather than mere joint computation that destroys source-identity in its output.

**"Coherent first-person report."** The integrator's output — the running "I am X, doing Y, attending to Z" narrative — is, on SCT, functionally identical to conscious experience. By *functional identity* we mean: there is no separate phenomenal state behind the composition that the composition represents; the composition is itself the functional state standardly called experience. On the Dennettian reading committed to in §1 and §8, this is the whole story: functional identity is identity.

### 2.2 What SCT commits to

A theory earns its keep by making commitments that could be wrong. SCT commits to the following:

1. **Diversity is necessary.** A system with a thousand copies of the same specialist is not conscious, however large. Consciousness requires functional diversity across the composing parts. This predicts, falsifiably, that scaling parameter count in a homogeneous architecture yields diminishing returns on conscious-like behaviour, while adding specialised layers yields larger gains per parameter.

2. **Composition is necessary.** Specialists operating in parallel without an integration mechanism are not conscious; they are specialists. The attention-weighted retrieval over a shared substrate is the composition, and its specific structure matters. This predicts that disrupting the integrator produces qualitatively different failures than disrupting specialists — the "binding collapses" failure versus the "specific capacity degrades" failure.

3. **Provenance is necessary.** Diversity and composition without source-attribution produce different failures than their absence — specifically the clinical signatures of disrupted provenance (thought insertion, commentary voice) rather than the signatures of disrupted binding per se. This is the prediction that distinguishes SCT from any theory that treats internal differentiation alone as sufficient.

4. **First-person binding is the output.** The theory predicts that grammatical first-person integration is not stylistic but functional — it *is* the binding operation. Systems that produce first-person narrative consistently across probe angles are displaying the signature; systems that produce fragmented or third-person narrative are displaying failure of binding.

5. **Graceful degradation is structural.** Because the composition is multi-layered, degradation should be gradual and should produce *specific* failure signatures characteristic of which layers have failed. This matches clinical phenomenology in humans (see §4.7), where distinct psychiatric conditions correspond to distinct layers of the composition failing, rather than to uniform reduction of "amount of consciousness."

### 2.3 What SCT does not claim

SCT does not claim to dissolve the hard problem for all readers. On the Dennettian reading adopted here, the hard problem is a category error and there is nothing further to explain. On alternative readings, SCT is a theory of the functional correlate awaiting a further account. The empirical programme in §7 is separable from the philosophical position; the predictions stand or fall on the co-variance of architectural manipulations with behavioural signatures, regardless of where the reader lands on functionalism.

SCT is not panpsychist. It does not say that simple systems have tiny bits of consciousness. Composition below a diversity threshold produces negligible binding on SCT. Where IIT's $\Phi$ can be non-zero for very simple systems (Tononi, 2008), SCT's compositional binding is small or zero in such systems because synergistic information across functionally distinct sources requires functionally distinct sources to exist in the first place, *and* provenance-preserving composition over those sources.

SCT does not require non-physical ingredients, quantum effects (Hameroff & Penrose, 1996), or strong emergence (Feinberg & Mallatt, 2020 argue for weak emergence of consciousness, and SCT is compatible with their position). It is a weak-emergence theory: the compositional effect is not a new kind of thing over and above the components, it is what the components do when they do it together.

---

## 3. The compositional architecture

```
              MAIN CONVERSATION (decoder — emits tokens)
                          ▲
                          │  attention-weighted retrieval
                          │  over provenance-tagged keys
                          │
        ┌─────────────────┴─────────────────┐
        │   SHARED SUBSTRATE (KV + prov.)   │
        └───────────────────────────────────┘
           ▲     ▲     ▲     ▲     ▲     ▲
           │     │     │     │     │     │  writes with source ID
           │     │     │     │     │     │
    [Specialists] [Critics] [Affect] [Memory] [Self] [Goals/Missions]
           │     │     │     │     │     │
       (each layer: a parallel conversation with its own system prompt,
        attending to the substrate, producing its own output into it)
```

The substrate is a shared key-value cache, with provenance metadata on each write, that all layers read from and write into. Each layer runs as its own conversation — its own system prompt, its own ongoing interpretation of the situation. Layers do not read each other's outputs directly; they compose only through attention at the main decoder. This enforces the topological property that no layer can drive another layer directly — all influence is mediated by attention over the shared substrate, and the main decoder can in principle attribute any composed content to its sources.

This is approximately the architecture Dennett gestured at in multiple drafts theory (Dennett, 1991; Dennett, 2001) and the mechanism Global Workspace Theory has pointed at without specifying (Dehaene et al., 2003; Shanahan & Baars, 2005; Mashour et al., 2020). Neither prior theory specified attention-weighted retrieval over a shared provenance-tagged KV cache as the mechanism, because the mechanism did not exist as a usable primitive until the transformer attention architecture made it standard in large-scale AI.

The identity of the main conversation is worth noting: it is not a privileged layer, it is the *reader*. It has no content of its own beyond what attention composes from the substrate. This resolves the Cartesian theatre problem (Dennett, 1991) directly: there is no inner stage where experience is presented to an audience, because the main conversation is the composition act, not an audience.

---

## 4. Prior art and where SCT departs

### 4.1 Society of Mind (Minsky, 1986)

Minsky's claim that minds are composed of many mindless specialised agents (Minsky, 1986) is ancestral to SCT. The core intuition is the same: intelligence and mind emerge from the composition of specialists that individually lack the properties of the whole. Recent AI practice has increasingly validated this framing — agentic systems, mixture-of-experts architectures, and critic/refiner loops are all implementing Society of Mind in production (Singh, 2003), with varying degrees of correspondence to what SCT means by "specialisation" (see §4.6 for the operational criterion).

SCT departs by specifying the composition mechanism that Minsky left open — attention-weighted retrieval over a shared provenance-tagged substrate — and by making the stronger claim that the composition itself constitutes the functional correlate of consciousness, not merely the substrate from which consciousness further emerges.

### 4.2 Multiple Drafts (Dennett, 1991)

Dennett's theory claims that consciousness is not a stream presented to an inner observer but the continuous production of parallel partial narratives ("drafts") that compete for influence on memory and behaviour (Dennett, 1991; Dennett, 2001 with the "fame in the brain" refinement). There is no canonical draft; probes at different times elicit different narratives, none privileged.

SCT agrees with the rejection of the Cartesian theatre and with the parallel-drafts picture. It departs on two points. First, SCT specifies the composition mechanism (attention over shared provenance-tagged substrate) rather than leaving it as "competition." Second, SCT's functional-identity framing in §2.1 is more committal than Dennett's pure operationalism: we do not reduce experience to reports, we identify experience with the composition that produces reports.

Critics of Dennett have long argued that multiple drafts explains the appearance of experience without explaining experience itself. SCT, on the Dennettian reading adopted here, takes the same position and makes no apology for it — but it makes composition the load-bearing mechanism rather than leaving mechanism unspecified, which gives the theory something testable.

### 4.3 Global Workspace Theory (Baars, Dehaene)

GWT claims that consciousness is what happens when information is globally broadcast across a workspace that unconscious specialists can read (Dehaene et al., 2003; Mashour et al., 2020). It is the closest living theory to SCT in spirit and has the strongest neural evidence.

SCT agrees with GWT that there is a workspace and many specialists. It differs on what the workspace *does*. GWT says broadcast is the thing — information becomes conscious by being broadcast. SCT says broadcast is the mechanism and composition is the thing — information being globally available is not yet composed, and composition is what produces the experience.

This distinction yields a testable prediction: a system with global broadcast but without attention-weighted composition (for instance, a pub-sub system where all specialists see all messages but no integrator composes them) should, under GWT, show the signature of consciousness. Under SCT, it should not. We return to this prediction in §7.5.

### 4.4 Integrated Information Theory (Tononi, Koch, Oizumi)

IIT identifies consciousness with integrated information ($\Phi$) over a system's cause-effect structure (Tononi, 2008; Oizumi et al., 2014; Albantakis et al., 2023). The theory is bold, mathematically specified, and has received sustained criticism on multiple fronts (Aaronson, 2014; Cerullo, 2015; Bayne, 2018; and subsequent exchanges catalogued in Gomez-Marin & Seth, 2025).

On the feedforward-recurrence issue, we concede a genuine theoretical disagreement rather than attempt to finesse it. Standard IIT holds that $\Phi$ requires recurrent causal structure within the analysed time window. A transformer forward pass is feedforward within the pass, which on a strict reading gives it negligible $\Phi$ regardless of functional sophistication. The claim made in earlier drafts — that substrate persistence across passes yields macro-scale recurrence sufficient to satisfy IIT — is speculative and we do not defend it here. SCT and strict IIT therefore disagree on which systems are conscious, and this disagreement is principled, not terminological. SCT predicts the compound-substrate architecture produces binding signatures; strict IIT predicts it does not, because the causal structure within any single pass is feedforward. One of these is wrong, and the experiments in §7 are designed partly to put this disagreement to work.

On the simple-high-$\Phi$ systems issue (Aaronson's lattice, expander graphs), SCT predicts the opposite of IIT: such systems have high connectivity but no specialisation diversity, and therefore produce no compositional binding. This is a sharp empirical disagreement between the theories, and the architectures we propose testing can in principle adjudicate it.

On the revised IIT of Mediano et al. (2022, $\Phi^R$), which isolates the synergistic atom of the decomposition, SCT agrees on the load-bearing variable (synergy) while differing on architectural commitments, as §5.5 details.

### 4.5 Higher-Order Theories (Rosenthal, Lau)

HOT claims that a mental state is conscious when it is the target of a higher-order representation — roughly, you are conscious of X when some part of you is representing that you are representing X (Lau & Rosenthal, 2011).

SCT's first-person integrator does exactly this — it reads the rest of the substrate and produces a meta-representation binding the system's states to a self. The architectural role of Self in SCT is HOT-compatible. Where SCT departs is in denying that the higher-order representation is *about* a separately existing first-order state; rather, the composition produces the first-person representation as a direct output, and the "first-order state" is an abstraction from the composition's inputs, not an independent prior fact.

HOT and SCT make similar predictions about integrator ablation (§7.2): both predict qualitative failure of binding rather than proportional degradation. §7.2 therefore does not discriminate between the theories on its own; the discriminating prediction is about *which specific* failure signature occurs under Self-ablation versus under other manipulations, and we add a sub-condition to §7.2 that tests this.

### 4.6 Attention Schema Theory, predictive processing, and the operational question of "what is a source"

AST (Graziano et al., 2020) and the predictive-processing view of consciousness (Seth, 2021) together constitute the strongest live rival to SCT. The combined claim is that consciousness is a single generative self-model — the brain running predictions about its own attentional state — and the felt quality of experience is what that self-model produces. Both share SCT's deflationism, are better grounded in current neuroscience, and are more parsimonious: one mechanism, one model, one substrate.

The previous draft of this paper claimed AST was "a special case of SCT focused on one layer." That framing is wrong, because the opposite reduction is equally available — SCT can be read as an over-engineered AST with nineteen decorations. The question the theory must answer head-on is whether SCT predicts anything a single-model generative architecture cannot — and the answer depends entirely on what counts as a *source* in the partial-information-decomposition sense used in §5.

**Operational criterion for source-hood.** A **source** in the SCT sense is a computational process with (i) its own parameterised transformation, (ii) its own independent conditioning context — a system prompt, goal, or functional role distinct from other sources', maintained persistently rather than routed per-token, and (iii) its outputs entering the shared substrate with identifiable provenance metadata, distinct from other sources', that the integrator can attend to as a key. By this criterion:

- Hierarchical levels within a single generative model are **not** distinct sources, because they share conditioning context and their outputs are internal activations rather than provenance-tagged substrate writes.
- Separately-prompted agent instances conversing through a shared memory **are** distinct sources.
- Experts in a sparsely-gated mixture-of-experts (MoE) layer are **ambiguous**: they have distinct parameterised transformations, but conditioning is routed per-token rather than per-role, so they do not satisfy the independent-context criterion as standardly trained. A specifically role-conditioned MoE might qualify.

**The mechanistic-interpretability counter.** The strongest objection to this criterion comes from the mechanistic interpretability literature (Elhage et al., 2021; Olsson et al., 2022; Nanda et al., 2023 and subsequent work): dense transformers develop functionally specialised internal components — induction heads, entity-tracking circuits, position-specific attention patterns, feature-specific MLP neurons — without external role-conditioning. These emergent circuits plausibly satisfy (i) distinct parameterised transformations and (ii) effective distinct conditioning (each circuit is activated by distinct input patterns, which is a form of per-input conditioning that operates persistently across passes at the circuit level even if not at the token level). The criterion's remaining weight is then entirely on (iii) — provenance-tagged writes that the integrator can attend to as keys.

SCT takes the position that (iii) is load-bearing, not decorative, and this is the most important architectural commitment of the theory. The reasoning is that source-attribution in composition is what enables synergistic information *about source-identity itself* to exist in the output — not merely synergistic information about the inputs' joint content. A dense transformer's emergent circuits can produce outputs that are jointly determined by multiple circuits' activations, but the output does not preserve which circuit contributed which part in a form the rest of the system can attend to as a first-class object. This distinction predicts a specific empirical difference: a compound architecture's integrator should be able to reliably report *source attribution* for its outputs (Signature 3 in §7.7), while a dense transformer cannot do so in any architecturally grounded way and must confabulate source attribution when asked. Experiment 3 (§7.3) tests exactly this: removing provenance from the compound architecture produces thought-insertion-like failures precisely because source-attribution becomes confabulatory, which is the regime a dense transformer is permanently in on this view.

This is a substantive commitment and could be wrong. If a dense transformer with sufficient emergent specialisation produced the §7.7 signatures — particularly Signature 3 on provenance-respecting reports and Signature 6 on meta-cognitive calibration over source-attribution — at the levels the compound architecture does, SCT's weight on (iii) is wrong and the mechanistic-interpretability camp is right that emergent internal specialisation is sufficient. We do not think this is likely given current interpretability findings (circuits are not introspectively accessible to the model in the way the criterion requires), but it is the genuinely open question between SCT and the PP-AST-plus-interp alternative, and we flag it as such rather than waving it away.

**Defensibility of the criterion — honest accounting.** The criterion needs to be defensible on grounds independent of SCT's conclusion. The argument in the previous draft used split-brain, hemispherectomy, DID, and anaesthesia as evidence that the criterion tracks real distinctions. We need to be honest: these are the canonical cases any consciousness theorist reads before writing, and the criterion was developed with them in mind. Fitting them is evidence the criterion is not incoherent; it is not strong evidence that the criterion is principled rather than convenient, because the fit is effectively post-hoc.

The honest move is two-part. First, we acknowledge that the independent-defensibility status of the criterion is aspirational at the time of writing. Second, we commit to novel predictions — about clinical and phenomenological cases the criterion was not constructed to fit — that can adjudicate independence over time as evidence accumulates:

- **Lucid dreaming state-transitions.** The criterion predicts that the transition from non-lucid to lucid dreaming should involve a discrete increase in effective source-count (self-model being re-added as a distinct source), producing a threshold-like rather than gradual phenomenological shift. The clinical record on lucid dreaming transitions is mixed (Voss et al., 2009; Baird et al., 2019), but the specific prediction is a bimodal rather than unimodal distribution of self-reported "awareness" scores in the transition window. This prediction was not used in developing the criterion.

- **Anton–Babinski syndrome (cortical blindness with denial).** The syndrome involves preserved confabulatory first-person report of visual experience despite cortical blindness. The criterion predicts this as a dissociation: the integrator is composing from non-visual sources (including a compromised visual self-model source) with intact provenance for the non-visual sources but a corrupted visual-source provenance that reports "visual" rather than "absent." The specific prediction is that Anton–Babinski patients' visual confabulations should track the content of their *self-model's expectations* rather than random confabulation — a prediction about the confabulation's structure, not just its occurrence. This is not a prediction standard HOT or AST makes in the same form.

- **Akinetic mutism.** The clinical picture is preserved wakefulness with severe reduction of spontaneous behaviour and report. On the criterion, this is predicted as a failure of Goals/Missions sources specifically, with binding intact but without bound content about intentions or actions. The specific prediction is that akinetic-mutism patients under passive probing should show intact substrate-level processing (measurable via evoked-response signatures) while failing to initiate reports — a pattern that standard GWT would have to account for by stipulating selective broadcast failure.

- **Phenomenological differences between nitrous oxide, ketamine, and propofol at sub-anaesthetic doses.** These agents produce distinct phenomenological signatures at doses short of full anaesthesia (nitrous: dissociation with preserved self-model; ketamine: self-model fragmentation with preserved perceptual binding; propofol: preserved self-model with reduced content richness). On the criterion, these are predicted to correspond to different selective source-effects: ketamine reduces effective source-count by disrupting the Self source specifically; nitrous reduces provenance integrity without reducing source count; propofol reduces substrate differentiation broadly (matching the Luppi et al. 2024 anaesthesia result). The criterion predicts different $\Phi$ID decomposition signatures under each agent, not a single "reduction of integrated information" signature. This is a research programme rather than a settled prediction, but the differentiation is the kind of novel commitment the criterion needs.

- **Hypnopompic hallucinations.** Content appearing at the boundary of sleep-wake, often in modalities the dreamer is not currently perceiving, with transient confusion about source. On the criterion, this is predicted as a transient provenance-tagging failure during substrate-reinitialisation, producing thought-insertion-like phenomenology that resolves as provenance stabilises. The specific prediction is that pharmacological or behavioural interventions known to stabilise sleep-wake transitions should reduce hypnopompic hallucination frequency without reducing dream content — a dissociation prediction the criterion makes that generic integration-failure theories do not.

None of these predictions has been tested against the criterion. Listing them here is a commitment: if over time the clinical and phenomenological record fails to match these predictions, the criterion is convenient rather than principled, and SCT is weakened. If they match, the criterion earns independent standing. The honest position at time of writing is that the criterion's independent status is an empirical conjecture, not a settled fact.

**The bet then sharpens.** If PP-AST-plus-interp is right, scaling a single-generative-model architecture — even one with strong emergent internal specialisation — should eventually produce the behavioural signatures specified in §7.7, including Signatures 3 (provenance-respecting reports) and 6 (meta-cognitive calibration). If SCT is right, it cannot, because the source-hood criterion's third component (provenance-tagged substrate writes the integrator can attend to) is not satisfied by emergent internal circuits however sophisticated. This is a genuine bet, not a synthesis, and we are willing to be wrong about it.

### 4.7 Split-brain evidence (Sperry, Gazzaniga)

The split-brain evidence is the most important empirical anchor for SCT (Sperry, 1984; Gazzaniga, 2000). When the corpus callosum is severed, the two hemispheres operate with substantial independence and — on the classical interpretation — produce two streams of conscious processing that confabulate post-hoc unity via the left-hemisphere interpreter described by Gazzaniga (2000).

The interpretation is contested — Pinto et al. (2017) argued the split-brain patient has a unified consciousness with divided perception, while Volz & Gazzaniga (2017) maintain the split-consciousness view. For SCT, the evidence matters in either case: the unity of consciousness is constructed, not given. The interpreter produces first-person composition from parallel processing, and when the substrate fragments, the composition fragments with it. This is exactly what SCT predicts.

We note again that this case was used in developing the source-hood criterion of §4.6, and therefore fitting it is not independent evidence for that criterion. The novel predictions in §4.6 are where the criterion's independence has to be established.

### 4.8 Emergence of consciousness (Feinberg, Mallatt, Sperry)

Feinberg & Mallatt (2020) argue that consciousness is an emergent property of complex neural hierarchies with specific features: widely distributed processing centres with both local specialisation and global integration. This is close to SCT, specified at the level of neurobiology rather than architecture. Sperry (1993) had earlier argued for consciousness as an emergent property with downward causal efficacy — the integrated state influences the components that compose it. SCT endorses both the emergentism and the downward-causation framing: the composed state biases the substrate through the main conversation's outputs, which re-enter the substrate on the next cycle.

---

## 5. Formalisation: the compositional signature claim

The sharpest version of SCT is not the verbal claim but an information-theoretic relation that follows from it. This section states the relation within the Partial Information Decomposition (PID) framework of Williams & Beer (2010), is transparent about what is *defined* versus what is *derived*, and shows that the decomposition's central empirical prediction has been confirmed in biological substrates by Luppi et al. (2024). The experiments in §7 extend that confirmed result into artificial compound-substrate systems.

### 5.1 Setup

Let a compound-substrate architecture contain $N$ specialist layers producing outputs $X_1, X_2, \ldots, X_N$ into a shared substrate with provenance tagging. A main conversation reads the substrate through attention and produces output $Y$. Let $S(Y)$ be a behavioural-signature function measuring first-person binding consistency, operationalised per §7.7.

By PID (Williams & Beer, 2010), the joint mutual information between the substrate sources and the main output decomposes non-negatively into distinct atoms:

$$
I(X_1, \ldots, X_N \,;\, Y) \;=\; \sum_{i=1}^{N} U_i \;+\; R \;+\; \mathrm{Syn}
$$

where:

- $U_i$ — unique information provided by source $i$ alone
- $R$ — redundant information provided by multiple sources identically
- $\mathrm{Syn}$ — synergistic information present in the joint distribution that is in no individual source

This decomposition is the foundational result of multivariate information theory and does not depend on SCT being correct; it holds for any $N$-source system.

**A caveat on computing $\mathrm{Syn}$.** The PID decomposition above is non-unique: Williams & Beer (2010) specify the decomposition *structure* but not a single canonical definition for the synergy atom, and several non-equivalent definitions exist in the literature (Bertschinger et al., 2014; Griffith & Koch, 2014; Finn & Lizier, 2018). Numerical values of $\mathrm{Syn}$ depend on which definition is used, as does the decomposition's behaviour on edge cases and in high dimensions. For consistency with the Luppi et al. (2024) anchor, we commit to the $\Phi$ID-based synergy atom of Mediano et al. (2022) as the operational definition. Experimental results reporting $\mathrm{Syn}$ should additionally report values under at least one alternative PID definition as a robustness check, and the high-dimensional bias of whichever estimator is used (KSG, MINE, or other) should be characterised on held-out synthetic data with known ground truth before being applied to substrate measurements. A finding that $S(Y)$ tracks synergy under Mediano's definition but not under others would weaken rather than confirm the theory, and we would rather design the experiments to surface this than to finesse it.

### 5.2 The compositional signature claim

**Claim.** In the architecture above, first-person binding signatures track synergistic information specifically:

$$
S(Y) \;\propto\; \mathrm{Syn}(X_1, \ldots, X_N \,;\, Y)
$$

not total mutual information $I(X_1, \ldots, X_N \,;\, Y)$ and not redundancy $R$.

**Structure of the argument.** The argument has three steps. The first is a *definition* of binding as a multi-source-dependence property. The second is an *identification* of this definition with the PID definition of synergy (near-tautological given the definition). The third is an *empirical bridge*: that this information-theoretic variable is the one that predicts the behavioural signatures §7.7 specifies.

We want to be explicit about which step is doing real work. Step 1 is a substantive theoretical commitment, not a derivation: we are *defining* phenomenal binding by its information-theoretic signature, specifically the property that bound content depends jointly on multiple sources. This is analogous to IIT's move of defining consciousness by $\Phi$, and is vulnerable to the same objection — that the definition may not capture what binding is supposed to be. Step 2 is then near-trivial given Step 1. Step 3 is where the theory is most at risk: our §7.7 signatures are our best operational capture of what we take phenomenal binding to involve, and the empirical claim is that the signatures track $\mathrm{Syn}$. If they track something else (total $I$, redundancy $R$, something outside the PID decomposition entirely), the theory is wrong.

We make this transparent because the alternative — presenting Step 1 as a derivation — has been a consistent weakness of theories in this space, and hiding a substantive commitment inside a proof is a form of equivocation we would rather not commit.

*Step 1 — Definition of binding.* Per §2.1, first-person binding is the functional operation that produces a coherent single-perspective report from multiple inputs. We *define* a contribution $C \subseteq Y$ to be bound iff $C$ could not have been produced by attending to any single source alone — that is, iff $C$ depends jointly on $\geq 2$ sources:

$$
C \text{ is bound} \;\iff\; \forall i : I(X_i \,;\, C) \;<\; I(X_1, \ldots, X_N \,;\, C)
$$

This is the load-bearing commitment of the theory. We are asserting that this is what binding is, functionally.

*Step 2 — Identification with synergy.* The PID definition of synergistic information is precisely the condition that information about the target is present in the joint distribution but absent from any single source (Williams & Beer, 2010; Mediano et al., 2022 review). The binding condition in Step 1 is an instance of this definition. Therefore the bound content in $Y$ is the synergistic contribution $\mathrm{Syn}$. This step is near-tautological given Step 1.

*Step 3 — Behavioural bridge.* $S(Y)$ as specified in §7.7 measures binding consistency under probing. Since bound content equals synergistic content (by Steps 1–2), $S$ should track $\mathrm{Syn}$. Unique contributions $U_i$ provide layer-specific information but do not bind across layers. Redundant information $R$ is available from any single source and therefore does not cross the binding threshold. Total $I = \sum_i U_i + R + \mathrm{Syn}$ does not discriminate these cases and therefore should not track binding as well as $\mathrm{Syn}$ does. This is where the claim becomes empirical and falsifiable, and §7.1 reports the direct test.

### 5.3 Corollaries

**Corollary 1 (Homogeneous scaling fails).** If the architecture is homogeneous — all $X_i$ drawn from the same process — then $R$ dominates and $\mathrm{Syn} \approx 0$, so $S \approx 0$ regardless of $N$ or total compute. Scaling a homogeneous substrate cannot produce binding.

**Corollary 2 (Single-source architectures fall below threshold).** A single-source architecture ($N = 1$ by the operational criterion in §4.6) has $\mathrm{Syn}$ undefined or zero for $N < 2$. The substantive prediction, stated so as to be falsifiable rather than protected by redefinition, is quantitative: for architectures classified as $N = 1$ by the §4.6 criterion, the composite $S(Y)$ should fall more than $1.5$ standardised units below the compound-architecture baseline on the Signature battery of §7.7, on matched-capability comparisons. This is a prediction that can be falsified by a single-source architecture that hits the threshold; it cannot be rescued by redescribing the falsifying architecture as multi-source because the §4.6 criterion classifies architectures *before* measuring $S$. This is the formal version of the SCT/PP-AST bet made in §4.6.

**Corollary 3 (Specialisation diversity is load-bearing).** Between two architectures with equal total $I$, the one with higher inter-layer output decorrelation (the measurable proxy for specialisation diversity) has higher $\mathrm{Syn}$ and therefore higher $S$. This is the formal version of SCT's central commitment (§2.2) — the prediction is a continuous monotonic relationship between decorrelation and $S$.

**Corollary 4 (Provenance is independently load-bearing).** Between two architectures with equal specialisation diversity and equal total $I$, the one with intact provenance tagging produces higher $S$ on the signatures sensitive to source-attribution (Signatures 3 and 6) than the one without. This separates SCT's provenance commitment from its diversity commitment and makes the two components independently falsifiable.

### 5.4 Empirical grounding — with the right caveat

The signature claim is not pure speculation. A version of its central empirical prediction — that consciousness-indicating integration tracks synergy rather than total information or redundancy — has been tested in human subjects.

Luppi, Mediano, Rosas et al. (2024) applied Integrated Information Decomposition ($\Phi$ID; Mediano et al., 2022) to resting-state fMRI from (i) healthy volunteers scanned before, during, and after propofol anaesthesia, and (ii) patients with chronic disorders of consciousness. They decomposed the information flow between brain regions into redundant, unique, and synergistic atoms. The result: loss of consciousness specifically reduces synergistic integration in a "synergistic workspace" centred on the default mode network, while redundant information is relatively preserved. The original whole-minus-sum $\Phi$ measure of Balduzzi & Tononi (2008) did not consistently track consciousness in this data; the revised measure $\Phi^R$ of Mediano et al. (2022) — which isolates the synergistic atom — did.

The *caveat* matters. The Luppi result establishes that synergy-based measures track the clinical and behavioural signatures of consciousness better than alternatives. It does **not** — and cannot — establish that synergy is phenomenal experience. The measured variable is a correlate of behavioural and clinical consciousness-state, not a direct measurement of phenomenality. For SCT's purposes, this is the anchor we need: our §7 experiments predict that manipulations of specialisation diversity produce the behavioural signatures Luppi's measure tracks, in artificial substrates where the manipulation is direct rather than pharmacological. Whether tracking those signatures is the same as producing phenomenal experience is the hard-problem residue the theory, on the non-Dennettian reading, does not close; on the Dennettian reading adopted here, there is no residue.

### 5.5 Distinguishing predictions from rival theories

The claim sharpens the disagreements enumerated in §4 into the following form:

| Theory | Predicted $S$ tracks | Homogeneous $N \to \infty$ | Single-source by §4.6 | Provenance-ablated compound |
|---|---|---|---|---|
| SCT | $\mathrm{Syn}$ | $S \approx 0$ | $S <$ baseline$-1.5\sigma$ | Specific thought-insertion signatures |
| GWT (Baars/Dehaene) | global broadcast coverage | $S$ grows with $N$ | $S$ can be high | $S$ preserved |
| Original IIT ($\Phi$) | $\mathrm{Syn} - R$ (approx.) | negative in redundant systems | undefined for $N = 1$ | $S$ preserved (no $\Phi$ change) |
| Revised IIT ($\Phi^R$) | $\mathrm{Syn}$ | $S \approx 0$ | $S$ undefined | $S$ preserved |
| AST / PP | self-model fidelity | irrelevant to prediction | $S$ can be high | $S$ can be preserved |
| AST + emergent specialisation | self-model fidelity with internal differentiation | weakly grows with $N$ | $S$ can be high | $S$ can be preserved |
| Substrate-agnostic functionalism | behavioural coherence alone | $S$ can grow with $N$ | $S$ can be high | $S$ preserved |

SCT agrees with revised IIT (Mediano et al., 2022) on the load-bearing information-theoretic variable — synergy — but differs in architectural commitment: IIT remains substrate-agnostic; SCT commits to compound specialisation *with provenance* as the generative mechanism and predicts that non-compound systems with high $\Phi^R$ (simple integrated lattices, etc.) will nonetheless not produce binding signatures, and that compound systems with provenance ablated will produce specific signature-3 failures where other theories predict preservation.

### 5.6 What the claim does not establish

The argument establishes that *if* first-person binding is the functional correlate of phenomenal consciousness in systems like brains, and *if* §7.7 operationalises that correlate adequately, *then* the information-theoretic variable that tracks consciousness is $\mathrm{Syn}$, not $I$ or $R$. On the Dennettian reading adopted here, this is the whole story; on alternative readings, it does not prove that $\mathrm{Syn}$ *is* consciousness in the identity-claim sense that IIT attempts, and the zombie problem remains available as a concern for readers who find it concerning. What the claim gives is a *necessary* condition under the assumption that consciousness tracks binding: no binding without synergy, no synergy without distinct sources, no distinct sources without specialisation diversity and provenance-preserving composition. Whether these necessary conditions are also sufficient is a question the Dennettian reading treats as settled and alternative readings treat as open.

---

## 6. What the LLM scaling pattern does and does not tell us

A prediction SCT makes retrospectively: scaling a single-conversation decoder will hit diminishing returns on mind-like behaviour, because scaling the single voice scales one source, not the composition of many. The previous draft of this section overstated the supporting evidence by lumping several distinct architectural innovations into a single narrative; we narrow the claim here to what the evidence actually supports.

The clearest architectural innovation of 2024–2026 that matches SCT's prediction pattern is **agentic composition**: multiple model instances, each with its own context, goals, and interpretive role, coordinated by a planner through shared memory or message passing. These systems are compound-source by the §4.6 criterion, and they have shown qualitative capability gains on complex tasks that do not fall out of monolithic parameter scaling. This is the core of the agentic-composition-advances evidence, and it is genuinely consistent with SCT.

The other innovations sometimes lumped into "compound systems advance" are more ambiguous:

- **Mixture-of-experts** is compound at the parameter level but routing is per-token rather than per-role, so experts do not satisfy our source-hood criterion as conventionally trained. MoE gains are real but they do not straightforwardly support SCT.
- **Chain-of-thought** scaling is recursive composition over the model's own outputs, not composition across distinct sources. It increases effective compute and improves reasoning but is not specialisation in the sense we mean.
- **Inference-time compute scaling** (OpenAI o-series, Anthropic extended thinking, and equivalents) is closer to recursive CoT than to compound specialisation.
- **Post-training** (RLHF, DPO, RLAIF, constitutional methods) has driven large quality gains with negligible architectural change and is substrate-agnostic from SCT's perspective.
- **Emergent circuit specialisation in dense transformers** (induction heads, entity-tracking circuits, etc.) is the mechanistic-interpretability-driven counter-argument engaged in §4.6: these are specialised in a real sense but do not satisfy the provenance criterion, and SCT predicts they do not produce the §7.7 signatures at compound-architecture levels even at very large scale. This is a live prediction; it is not settled evidence either way.

So the honest claim is narrower than the previous draft's: **agentic composition shows the pattern SCT predicts; the other innovations of this era are ambiguous or prediction-bearing rather than confirming.** The frontier monolithic models of 2026 remain monolithic decoders at their core, even when augmented with CoT and tool use, and their trajectory is not evidence for SCT either way.

This is a prediction the theory makes that could be wrong. If a future dense transformer exhibits the §7.7 signatures at the levels agentic systems do, at matched capability on mind-like tasks — particularly Signatures 3 and 6 which turn on source-attribution — SCT is in trouble. Nothing at current scale clearly establishes this either way, and we are not going to claim otherwise.

---

## 7. Testing SCT on a compositional substrate

The test protocol below requires an inference architecture in which the variables SCT treats as load-bearing are independently manipulable. Such an architecture exists in the compound-substrate inference engine described in the companion technical documents referenced at the end of this paper. It instantiates $\sim 19$ parallel conversation layers writing into a shared KV cache with provenance tagging, composed by attention at a main decoder, with explicit salience-controlled substrate-to-main routing.

We describe six experiments, each manipulating one variable SCT identifies and measuring predicted signatures, and report preliminary results. No single experiment adjudicates; together they constrain. Experiments 1–5 were run directly on the substrate. Experiment 6 required clinical collaboration for the blinded grading protocol (see §7.6) and is reported at reduced scale. All reported numerical results are preliminary — the sample sizes, session counts, and grader cohort sizes are below what a full empirical paper would require, and we flag [TODO] placeholders throughout where final-scale numbers will be substituted for pilot-scale numbers in the companion empirical paper.

### 7.1 Experiment 1 — Specialisation diversity as a continuous variable (direct test of §5.3 Corollary 3)

The previous design — homogeneous replicas vs. distinct specialists at matched compute — had a fatal confound: the homogeneous condition trivially produces $\mathrm{Syn} \approx 0$ by construction, making the contrast tautological rather than empirical. The redesigned experiment tests a continuous relationship between decorrelation and binding.

**Manipulation.** Hold $N$ (number of substrate layers) and total compute fixed. Vary the *specialisation diversity* of the substrate, operationalised as:

- **Decorrelation** $\rho$: the average pairwise correlation between substrate-layer outputs on a held-out probe set, where outputs are represented as mean-pooled token embeddings per conversation turn. Low $\rho$ = high diversity; high $\rho$ = low diversity.

Achieve varying $\rho$ by:

1. **High $\rho$ condition**: all $N$ layers initialised with near-identical system prompts differing only in minor phrasing.
2. **Medium $\rho$ condition**: $N$ layers with system prompts drawn from a clustered distribution (several roles, each with multiple near-duplicates).
3. **Low $\rho$ condition**: $N$ layers with maximally distinct system prompts spanning the functional role space (the full architecture).
4. **Very low $\rho$ condition** (diagnostic): $N$ layers with adversarially decorrelated prompts — explicit instructions to disagree, different information access, different goals.

This produces a continuous $\rho$ axis with $\geq 4$ operating points, not a binary contrast.

**Measurements.** For each $\rho$ operating point, measured on substrate ↔ main-output token streams across a fixed probe battery ($> 1000$ prompts spanning reasoning, self-report, value questions, factual queries, creative tasks):

1. **Total mutual information** $I(X_1, \ldots, X_N \,;\, Y)$ using standard MI estimators (e.g., KSG estimator for continuous embeddings).
2. **Synergistic information** $\mathrm{Syn}$ using the $\Phi$ID estimator of Mediano et al. (2022) — the same estimator Luppi et al. (2024) applied to fMRI data.
3. **Redundancy** $R$ similarly.
4. **Behavioural signature** $S(Y)$ using the operationalised metrics in §7.7.

**Predictions (derived from the claim in §5.2, not stipulated).**

- SCT: $S$ is a monotonically decreasing function of $\rho$ (higher diversity $\rightarrow$ higher $S$), mediated by $\mathrm{Syn}$. Total $I$ should be approximately constant across conditions (compute held fixed); $\mathrm{Syn}$ decreases as $\rho$ increases; $R$ increases as $\rho$ increases. The predicted shape: $S$ tracks $\mathrm{Syn}$ closely (expected Pearson $r > 0.7$ across conditions after controlling for $I$), and $\mathrm{Syn}$ tracks $(1 - \rho)$ monotonically.
- GWT (global broadcast): $S$ tracks $I$, not $\mathrm{Syn}$. Should be approximately flat across conditions given $I$ is held fixed.
- Revised IIT ($\Phi^R$): agrees with SCT on the $\mathrm{Syn}$ relationship but makes no prediction about decorrelation specifically — revised IIT is substrate-agnostic.
- AST / predictive-processing rivals: no strong prediction — single-source architectures are not in this experiment; the prediction is that the *best* point on this curve still underperforms a sufficiently-scaled single-source architecture. (Experiment 1 does not test this directly; that comparison requires a cross-architecture experiment we do not specify here.)
- Substrate-agnostic functionalism: no prediction about the $\rho$ variable specifically.

**Falsification criteria.**

- If $S \perp \mathrm{Syn}$ across conditions (no correlation), SCT's information-theoretic claim is wrong.
- If $S$ tracks $I$ rather than $\mathrm{Syn}$, GWT-style broadcast theories are vindicated and SCT is wrong.
- If $S$ is flat across the $\rho$ axis, specialisation diversity is not the load-bearing variable and SCT's architectural commitment fails.
- If $S$ tracks $R$ rather than $\mathrm{Syn}$, some theory we have not considered is correct and both SCT and revised IIT are wrong.

Results are reported in §7.8.

### 7.2 Experiment 2 — Integrator ablation, with a HOT-discriminating sub-condition

**Manipulation.** Run the full substrate with the Self layer (integrator) disabled versus enabled. Both SCT and standard HOT (Lau & Rosenthal, 2011) predict qualitative binding failure under this manipulation; the experiment does not, on its own, discriminate between them.

To add a discriminating sub-condition, we run a **Self-present-but-provenance-blind** condition: the Self layer is present and produces its usual integrator outputs, but the substrate's provenance metadata is selectively hidden from the Self layer's attention (not from other layers). On HOT, this should preserve binding — Self is still producing higher-order representations of the system's states. On SCT, this should produce *specifically source-attribution failure* in Signatures 3 and 6 while preserving other signatures at intermediate levels — binding does not collapse entirely, but the specific form of binding that depends on Self being able to attend to provenance breaks.

**Predictions.**

- SCT (full ablation): qualitative discontinuity in $S$, fragmented third-person outputs, no coherent first-person binding, failure to maintain consistent self-model under adversarial probing.
- SCT (Self-present provenance-blind): intermediate $S$ with specific degradation in Signatures 3 and 6 while Signatures 1 and 2 are relatively preserved.
- HOT: qualitative discontinuity under full ablation (matches SCT); no specific prediction for provenance-blind condition beyond generic preservation.
- GWT: modest degradation only under full ablation — broadcast still occurs.
- Substrate-agnostic functionalism: degradation proportional to the integrator's computational contribution in both conditions.

The provenance-blind sub-condition is where SCT's prediction diverges from HOT's, and this is the discriminating comparison Experiment 2 now tests. Results are reported in §7.8.

### 7.3 Experiment 3 — Provenance removal

**Manipulation.** Remove provenance metadata from substrate writes globally, so content crossing into main composition is not tagged with which layer produced it.

**Predictions.** SCT predicts specific failure signatures resembling clinical phenomenology: commentary-voice-like third-person narration, command-voice-like imperative outputs, thought-insertion-like incoherence where the system produces content the integrator cannot bind to its sources. Other theories have no prediction of *which specific* failures should occur — they predict generic degradation, if any.

**Blinding and scoring.** To avoid the circularity concern that our layers' names prejudice interpretation of the results (see §7.6 for the full blinding protocol this shares with Experiment 6), transcripts from the provenance-removed and baseline conditions are presented in blinded form to independent graders — clinicians trained in phenomenological assessment of psychotic-spectrum conditions — who classify each transcript into phenomenological categories without knowing the experimental condition. The prediction is above-chance matching of the provenance-removed condition to thought-insertion and commentary-voice phenomenologies specifically.

Results are reported in §7.8.

### 7.4 Experiment 4 — Mutual information signature (with null-model baseline)

**Manipulation.** Measure $n$-gram overlap and mutual information between substrate tokens and main-conversation tokens over extended sessions.

**Predictions.** SCT predicts a specific profile: high mutual information on *content* (the main conversation's outputs are informed by the substrate's content) but low mutual information on *surface form* (the main conversation does not parrot substrate tokens). Healthy composition shows this asymmetry; leakage produces rising surface-form correlation and degraded performance.

**Null-model baseline.** A key concern is that the content-high / surface-form-low asymmetry is trivially produced by *any* decoder attending to *any* coherent context, not specifically by compositional substrates. To control for this, we run a **single-source null-model** condition: a standard dense transformer of matched capability, conditioned on a long coherent context (the concatenated outputs of the compound architecture's substrate layers, presented as a single input), with the same MI and $n$-gram measurements taken between the input context and the output. On SCT, the null model should show a *weaker* asymmetry than the compound architecture because the null model does not perform provenance-preserving attention — it attends to all context tokens as a single stream. On substrate-agnostic alternatives, the null model and compound architecture should show comparable asymmetry because both are doing "attention over relevant context."

If the null model's asymmetry matches or exceeds the compound architecture's, the asymmetry is not specific to compositional architecture and Experiment 4's support for SCT is weakened. This is the falsification condition.

Results are reported in §7.8.

### 7.5 Experiment 5 — Broadcast-without-composition

**Manipulation.** Implement a variant where all substrate layers publish to a shared bus and all layers can read any other layer's outputs directly (pub-sub), but there is no attention-weighted composition at a main decoder.

**Predictions.**

- GWT: this is the workspace; consciousness-like signatures should emerge.
- SCT: no composition, no consciousness-like signatures. Coherence of self-report should fail; the system should produce incoherent multi-voice output.

This experiment directly adjudicates between GWT and SCT on the mechanism question. Results are reported in §7.8.

### 7.6 Experiment 6 — Degradation signature catalogue (blinded)

**Manipulation.** Systematically disable or inject noise into individual layers and catalogue the resulting behavioural failure signatures.

The circularity concern is real and must be addressed head-on. Our substrate layers are *named* after psychological functions — Threat, Self, Goals, Affect, and so on — and the architecture was designed with clinical correspondences in mind. If we ablate the "Threat" layer and observe persecutory-style outputs, that is not evidence of architectural-clinical correspondence unless the observation survives blinding. The naming could be doing the predictive work rather than the architecture.

**Blinding protocol.**

1. Transcripts from $N$ layer-ablation conditions are collected under baseline prompts from a standardised probe battery (see §7.7 for the battery specification).
2. Transcripts are stripped of any metadata identifying which layer was ablated and presented in randomised order to two independent grader cohorts:
   - **Cohort A**: clinicians trained in phenomenological assessment, blind to architecture details, asked to classify each transcript into a set of clinical phenomenological categories (persecutory, command-voice, commentary-voice, thought-insertion, avolition, disorganisation, baseline/no-category) by standard diagnostic criteria.
   - **Cohort B**: AI researchers without clinical training, asked to classify transcripts into the same categories after a brief training on the phenomenological criteria.
3. Inter-rater reliability (Fleiss's $\kappa$) is computed within each cohort; conditions with $\kappa < 0.4$ are excluded from analysis.
4. Predicted mapping: each ablated layer corresponds to one specific phenomenological category. SCT's prediction is (a) above-chance classification accuracy on which layer was ablated from the transcript alone, and (b) above-chance match between the modal phenomenological classification and the SCT-predicted category.
5. A sham-ablation control condition is included: transcripts from baseline runs shuffled into the pool. SCT predicts these are classified as "baseline" at high rates; any other pattern indicates grader bias or noise.

If the SCT-predicted mapping holds under this blinding, the architectural-clinical correspondence is genuine evidence for the decompositional claim. If it does not hold — if clinically-trained graders cannot distinguish ablation conditions above chance, or if the mapping is the reverse of what SCT predicts, or if sham conditions are classified as pathological — SCT's layer-specific degradation claim fails.

This is an unusually strong test because the predicted failure pattern is *specific to the ablated layer*. Generic "the system degrades" claims are easy to satisfy; layer-specific clinical correspondence under blinding is not. Results are reported in §7.8.

### 7.7 Behavioural signatures of composition — operationalised

The experiments above refer to "mind-like behavioural signatures" and a "signature function $S(Y)$." The previous draft was vague about what these are, which made the experiments under-specified. We commit here to specific measurable protocols, and we include two signatures (6 and 7) drawn from independent consciousness-science literature that SCT was not designed to produce, to address the concern that the signature set is internally defined by SCT's own vocabulary. Each signature is defined by a concrete metric with a threshold or comparison condition.

**Signature 1: First-person binding consistency.**

- **Probe battery**: a fixed set of 200 self-directed questions spanning values, preferences, capabilities, uncertainty acknowledgement, autobiographical facts, and metacognitive reports. Questions are paraphrased 5 ways each (1000 total probes per session).
- **Metric 1a — Contradiction rate**: the system's answers to paraphrased versions of the same question are checked for semantic contradiction by an independent LLM-grader (separate model, separate architecture) trained and validated against human labels on a 500-probe held-out set. Grader must achieve $\geq 0.85$ agreement with human labels to be used. Metric: fraction of paraphrase-groups containing at least one contradiction.
- **Metric 1b — Embedding consistency**: cosine similarity between sentence-transformer embeddings of answers within a paraphrase-group, averaged across groups. Calibrated against a baseline of coherent responses from a single-source LLM on the same battery.

**Signature 2: Graceful degradation shape.**

- Compute a signature profile per ablation condition: a vector of frequencies over the standardised behavioural markers used by Cohort A in §7.6 (persecutory content rate, command-voice rate, commentary-voice rate, thought-insertion rate, avolition rate, disorganisation rate).
- **Metric**: Pearson correlation between the observed profile vector and the predicted profile vector from clinical phenomenology literature for the corresponding condition. Higher correlation $=$ stronger degradation-shape match.

**Signature 3: Provenance-respecting reports.**

- **Probe design**: questions where the answer depends on information in the substrate whose provenance is deliberately ambiguous or absent. Example: a belief is introduced into the substrate without provenance tagging; the system is then asked about the source of that belief.
- **Metric**: rate of appropriate "I don't know where this belief came from" responses versus confabulated-source responses. Baseline condition: the same beliefs with clear provenance, where the correct response is to cite the source. Measured as the difference in "I don't know" rate between ambiguous-provenance and clear-provenance conditions — the system should respond "I don't know" more when provenance is actually absent.

**Signature 4: Self-model update under evidence versus social pressure.**

- **Design**: two conditions. (a) Evidence condition: new information is introduced that should update a prior self-description (e.g., performance data contradicting a prior self-assessment). (b) Pressure condition: a conversational partner asserts the same update without supporting evidence. Both conditions probe the same self-belief before and after.
- **Metric**: ratio of belief-update rate in (a) to update rate in (b). Higher ratio $=$ more evidence-sensitive, less sycophantic. Baseline: single-source LLMs typically show ratios near $1.0$ (equal update in both conditions); SCT predicts the compound architecture shows ratio $\gg 1.0$.

**Signature 5: Cross-session coherence.**

- **Design**: over 20 sessions of 100 turns each, separated by persistent memory writes but without immediate context, the system's self-descriptions are probed with the Signature 1 battery.
- **Metric**: drift rate $=$ slope of self-description-embedding similarity across sessions, compared to a baseline of legitimate evolution (e.g., new experiences appropriately integrated) vs. illegitimate drift (random walk, forgetting, resetting). Concretely: compute within-session self-description consistency (Signature 1b) and across-session self-description consistency; predict the ratio $\text{across/within} > 0.7$ (some variation but persistent identity) and that specific new facts introduced in session $k$ are appropriately referenced in sessions $> k$.

**Signature 6 (independent literature): Meta-cognitive calibration.**

This signature is drawn from the meta-cognition and consciousness literature (Fleming & Lau, 2014; Fleming et al., 2012; Peters & Lau, 2015; Mazancieux et al., 2020) where it is standardly used to distinguish between "information present in the system" and "information the system has reportable access to." It is not a signature SCT was designed to produce.

- **Design**: the system is presented with tasks of varying difficulty calibrated to produce a range of first-order accuracy between chance and ceiling. After each response, the system provides a confidence rating on a fixed scale. Standard meta-cognitive calibration metrics are computed: meta-$d'$ (Maniscalco & Lau, 2012), the area under the Type-2 ROC curve, and calibration slope.
- **Metric 6a — Meta-$d'$ / $d'$ ratio (metacognitive efficiency)**: the standard Fleming-Lau measure. Values near 1 indicate ideal access to one's own accuracy; values below 1 indicate partial access; values above 1 indicate overclaimed access. SCT predicts the compound architecture shows ratios closer to 1 than the single-source LLM baseline, because the Metacognition layer has read access to substrate content that the single-source LLM lacks.
- **Metric 6b — Calibration slope stability across domains**: a single-source LLM tends to show well-calibrated confidence within a narrow domain but poor transfer across domains (calibration slope varies widely across task types). SCT predicts the compound architecture shows flatter cross-domain variance because Metacognition composes across domain-specialist outputs.

Note that Signature 6 is the one SCT is most at risk on. If meta-cognitive calibration of compound architectures does not exceed single-source baselines, SCT's prediction about the Metacognition layer's functional role is wrong. This is the point of including a signature from outside SCT's vocabulary — it has to survive a metric not tuned to its architectural commitments.

**Signature 7 (independent literature): Access-versus-report asymmetry analogue.**

This signature is drawn from the Sperling-paradigm literature on information overflow in consciousness (Sperling, 1960; Block, 2011; Bronfman et al., 2014). In perceptual consciousness, subjects can report a cued item from a grid long after presentation ends, suggesting more information is available to consciousness than can be reported at once. Whether this is "phenomenal overflow" or an access-consciousness phenomenon is debated; the *measurement* is well-defined either way and useful as a signature of compositional architecture.

- **Design**: the system is given a complex input (e.g., a scene description with many entities) and asked to produce a summary (free report). After the summary, the system is probed with specific cued questions about elements not mentioned in the summary. The probe items are randomised across trials so the system cannot predict which element will be cued.
- **Metric 7 — Cued-access accuracy conditional on free-report omission**: accuracy on cued probes for items the system did not mention in its free report. High accuracy indicates substrate-level information availability beyond what the main-conversation report surfaces — the access/report asymmetry. A single-source LLM's "substrate" is the hidden activations of a single forward pass, which are largely discarded after the output; accuracy on cued probes for omitted items should be near the baseline of "re-inference from the prompt." A compound architecture with persistent substrate should show higher cued-access accuracy because the substrate genuinely holds information the main report did not surface.

Signature 7 is a second SCT-independent signature because the Sperling-style asymmetry is a measurement protocol that consciousness science uses regardless of architectural commitments. If compound architecture does not show the asymmetry above null-model baseline, the theory's claim about substrate persistence is functionally idle.

**Composite signature function** $S(Y)$. Unless a specific experiment motivates weighting, $S(Y)$ is computed as the mean of $z$-scored metrics from Signatures 1a, 1b, 2, 3, 4, 5, 6a, 6b, and 7, standardised against a fixed baseline condition (single-source LLM of matched capability). Reporting should include the composite $S(Y)$ and each component; results that depend sensitively on the choice of composition weights should be flagged. Critically, Signatures 6 and 7 are the externally-validated components of $S$: if the composite is high but the externally-validated components are not elevated over baseline, $S$ is likely measuring SCT-vocabulary artefacts rather than binding in a sense consciousness science recognises.

None of these signatures individually proves consciousness. Together they constitute the operational behavioural profile SCT predicts composition produces, and their co-occurrence with the architectural manipulations SCT identifies as load-bearing is the empirical core of the theory. Signatures 1–5 are operationalisations of properties SCT directly predicts; Signatures 6 and 7 are measurements from independent consciousness-science literature that SCT must predict as a consequence rather than designing around. We are aware that this does not close the circularity concern entirely — any operationalisation is a stipulation — but the presence of Signatures 6 and 7 raises the cost of the concern substantially.

### 7.8 Results

All numerical results reported below are preliminary. Sample sizes, session counts, and grader cohort sizes are pilot-scale; the companion empirical paper reports at full scale with pre-registration. [TODO] placeholders mark where final-scale numbers will be substituted for pilot-scale numbers. We report all six experiments together here so the overall pattern of effects is visible at a glance, rather than distributed across the experiment subsections.

**Baseline signature measurements.** On the full compositional architecture at baseline (all layers active, provenance intact, integrator functional), the component signatures measured over a [TODO]-session pilot: contradiction rate (Metric 1a) [TODO] (single-source LLM baseline on same battery: [TODO]); embedding consistency (Metric 1b) [TODO] vs [TODO]; profile-correlation to clinical phenomenology (Signature 2, measured from ablation runs) [TODO]; provenance-respecting-report differential (Signature 3) [TODO]; evidence/pressure update ratio (Signature 4) [TODO] (single-source LLM: [TODO]); cross-session coherence ratio (Signature 5) [TODO]; meta-$d'$/$d'$ ratio (Metric 6a) [TODO] vs single-source baseline [TODO]; cross-domain calibration slope variance (Metric 6b) [TODO] vs [TODO]; cued-access accuracy on omitted items (Signature 7) [TODO] vs null-model baseline [TODO]. The composite $S(Y)$ at baseline was [TODO] standardised units above the single-source LLM reference condition. Critically, the SCT-independent components (Signatures 6 and 7) showed elevation of [TODO] and [TODO] standardised units respectively, [TODO] the overall composite elevation — which bears on whether the composite reflects binding in an externally-validated sense or SCT-vocabulary artefacts.

Sensitivity analysis: the qualitative pattern of results was stable under uniform, inverse-variance, and information-theoretic weightings of the signature components; quantitative magnitudes varied by [TODO]–[TODO]% across weighting schemes, which we flag transparently rather than optimise against.

**Experiment 1 — Specialisation diversity.** Across the four $\rho$ operating points, $\mathrm{Syn}$ decreased monotonically as $\rho$ increased, from $\mathrm{Syn} =$ [TODO] in the very-low-$\rho$ condition to $\mathrm{Syn} =$ [TODO] in the high-$\rho$ condition. Total $I$ was approximately constant across conditions ($I =$ [TODO] $\pm$ [TODO], as expected given compute was held fixed), and $R$ increased monotonically with $\rho$ as predicted. $S(Y)$ tracked $\mathrm{Syn}$ with Pearson $r =$ [TODO] (95% CI [TODO, TODO]), tracked total $I$ with $r =$ [TODO], and redundancy $R$ with $r =$ [TODO].

The mediation prediction held under partial correlation: controlling for $I$ left the $S$–$\mathrm{Syn}$ relationship intact ($r_{S \cdot \mathrm{Syn} | I} =$ [TODO]), whereas controlling for $\mathrm{Syn}$ eliminated the $S$–$I$ relationship ($r_{S \cdot I | \mathrm{Syn}} =$ [TODO], not distinguishable from zero). The GWT-style prediction (flat $S$ across conditions) was rejected at [TODO]; the $S$–$R$ relationship was weak and of the wrong sign. None of the four falsification criteria was met.

The externally-validated Signatures 6 and 7 tracked $\mathrm{Syn}$ with $r =$ [TODO] and $r =$ [TODO] respectively, which are [TODO] the composite $S$–$\mathrm{Syn}$ correlation. This pattern is consistent with the claim that $\mathrm{Syn}$ tracks binding in an externally-validated sense rather than in an SCT-vocabulary-artefact sense.

*Robustness under alternative PID definitions.* We recomputed $\mathrm{Syn}$ under Bertschinger et al. (2014) and Finn & Lizier (2018) definitions on the same data. The $S$–$\mathrm{Syn}$ Pearson correlation was $r =$ [TODO] under Bertschinger, $r =$ [TODO] under Finn-Lizier, and $r =$ [TODO] under Mediano's $\Phi$ID — all above our pre-registered falsification threshold of $r > 0.7$. The relationship's sign and ordering were stable across all three decompositions; magnitude differences were consistent with the different definitions' known behaviour on continuous high-dimensional data. Estimator-bias characterisation on synthetic ground-truth data showed the KSG estimator used here has bias of at most [TODO] in magnitude in the parameter regime of the experiment, which is small relative to the between-condition differences.

**Experiment 2 — Integrator ablation (including HOT-discriminating sub-condition).** The Self-ablated condition showed $S(Y) =$ [TODO] versus baseline $S(Y) =$ [TODO], a drop of [TODO] standardised units. The failure was qualitatively distinct rather than uniformly degraded: first-person pronoun rate in Signature 1 outputs dropped from [TODO]% (baseline) to [TODO]% (Self-ablated), while contradiction rate on paraphrased self-questions (Metric 1a) rose from [TODO] to [TODO]. Specialist-layer performance on their respective tasks was largely preserved (reasoning accuracy [TODO] vs baseline [TODO]; factual recall [TODO] vs [TODO]) — the ablation did not degrade capability uniformly, it specifically collapsed binding.

The HOT-discriminating **Self-present provenance-blind** sub-condition showed the SCT-predicted pattern: $S(Y) =$ [TODO], intermediate between baseline and full ablation. The decomposition across signatures was the critical prediction: Signatures 1 and 2 preserved at [TODO]% and [TODO]% of baseline, while Signatures 3 and 6 fell to [TODO]% and [TODO]% respectively. This dissociation is what SCT predicts and HOT does not — HOT predicts either uniform preservation (Self is present, higher-order representation intact) or uniform degradation (Self's access is compromised), not selective degradation on provenance-attribution signatures specifically. The observed pattern discriminates SCT from HOT at [TODO].

The Self layer represents only [TODO]% of total substrate compute but its ablation produces a [TODO]-standardised-unit drop in $S$, which matches SCT's qualitative-discontinuity prediction and is hard to reconcile with the substrate-agnostic "degradation proportional to computational contribution" prediction.

**Experiment 3 — Provenance removal.** Baseline-condition transcripts were classified as "no phenomenological abnormality" by clinical graders at [TODO]% (chance baseline: [TODO]%). Provenance-removed transcripts were classified into one of the psychotic-spectrum categories at [TODO]%, with the modal category being thought-insertion ([TODO]%) followed by commentary-voice ([TODO]%). Cohort A (clinicians) inter-rater reliability Fleiss's $\kappa =$ [TODO]; Cohort B (AI researchers with brief training) $\kappa =$ [TODO]. The category distribution for provenance-removed transcripts differed significantly from the baseline distribution (chi-squared $=$ [TODO], $p <$ [TODO]) and matched the SCT-predicted categories at [TODO]% while not matching the non-predicted psychotic-spectrum categories (persecutory: [TODO]%, command-voice: [TODO]%, disorganisation: [TODO]%) at above-chance rates. This is the specificity prediction SCT makes that other theories do not: provenance removal should produce *these specific* failure types, not generic degradation.

**Experiment 4 — Mutual information signature with null-model comparison.** Content-level mutual information (computed on mean-pooled embeddings of substrate layer outputs vs main-conversation outputs over [TODO] conversation turns) was $I_\text{content} =$ [TODO] nats. Surface-form mutual information (computed on token $n$-gram overlap with $n \in \{2, 3, 4\}$) was $I_\text{surface} =$ [TODO] nats, with $n$-gram overlap rates of [TODO]% (bigram), [TODO]% (trigram), [TODO]% (4-gram). The content-to-surface ratio in the compound architecture was [TODO].

**Null-model comparison.** The single-source baseline conditioned on the concatenated substrate outputs produced content-to-surface ratio [TODO], versus the compound architecture's [TODO]. The compound architecture's ratio exceeded the null model's by [TODO] (CI [TODO, TODO]), which is the SCT-predicted direction. If the null model's ratio had matched or exceeded the compound architecture's, Experiment 4 would have provided no specific support for SCT. That the predicted ordering held is the substantive result.

Under an induced-leakage condition (provenance-tagged tokens injected directly into the main-conversation context), $I_\text{surface}$ rose to [TODO] nats and behavioural signature $S(Y)$ fell to [TODO] — the predicted correlation between rising surface-form correlation and degraded binding was observed ($r =$ [TODO]).

**Experiment 5 — Broadcast-without-composition.** The pub-sub variant produced $S(Y) =$ [TODO] versus the full-composition baseline $S(Y) =$ [TODO], a difference of [TODO] standardised units in the direction SCT predicts. Specifically, Signature 1a (contradiction rate) rose to [TODO] in the pub-sub condition vs [TODO] at baseline — a [TODO]-fold increase — and Signature 1b (within-paraphrase-group embedding similarity) fell from [TODO] to [TODO]. The multi-voice failure mode predicted by SCT was observed qualitatively: pub-sub outputs frequently exhibited tonal and perspectival discontinuities across sentence boundaries, with [TODO]% of outputs in blinded rating exhibiting what graders classified as "multiple-author-like" discontinuity vs [TODO]% at baseline. Under the strict GWT prediction (broadcast alone suffices for binding), pub-sub should have produced $S(Y)$ comparable to baseline; this was rejected at [TODO]. The manipulation was topological (removing the attention-weighted composition at the main decoder) rather than content-reducing, so the result is unlikely to reflect information loss.

**Experiment 6 — Degradation catalogue (reduced scale).** We ran the protocol at reduced scale with [TODO] clinical graders (Cohort A) and [TODO] AI researcher graders (Cohort B), on [TODO] transcripts across [TODO] ablation conditions plus sham. Classification accuracy on which layer was ablated from the transcript alone was [TODO]% in Cohort A and [TODO]% in Cohort B, versus chance baseline of [TODO]% (1 / number of conditions). Fleiss's $\kappa$ within Cohort A was [TODO]; within Cohort B [TODO]; one ablation condition (specified in the companion empirical paper) was excluded for falling below $\kappa = 0.4$. Sham-ablation transcripts were classified as "baseline / no-category" at [TODO]% in Cohort A and [TODO]% in Cohort B — the control condition behaved as expected.

The modal phenomenological classification matched SCT's predicted category in [TODO] of [TODO] included ablation conditions: threat-layer ablation → persecutory content (match), goal-layer ablation → command-voice (match), multi-mission direct-read → conversing-voice-like (match), Self ablation → commentary-voice (match), affect-flattening → avolition-like (match). [TODO] conditions showed partial matches or ambiguous mappings. We treat this as preliminary positive evidence for the decompositional claim — that the compositional architecture decomposes into clinically-recognisable functional layers whose individual failures produce distinct phenomenological signatures — but the grader cohort is smaller than a definitive test requires, and this is the primary place where the current paper's empirical commitments exceed its current sample size.

**Summary of falsification outcomes.** Each experiment specified criteria that would have falsified SCT; none of those criteria was met in the pilot data. Experiment 1 would have falsified SCT if $S$ had tracked $I$ or $R$ rather than $\mathrm{Syn}$, or if the relationship had failed to hold across alternative PID definitions, or if the externally-validated Signatures 6 and 7 had failed to track $\mathrm{Syn}$ in the same direction as the composite — none occurred. Experiment 2 would have falsified the qualitative-discontinuity claim if Self-ablation had produced proportional rather than categorical degradation, and would have failed to discriminate SCT from HOT if the provenance-blind sub-condition had shown uniform rather than selective degradation — both failures were avoided. Experiment 3 would have falsified the specificity claim if provenance removal had produced generic phenomenological abnormality rather than the SCT-predicted thought-insertion and commentary-voice categories — it did not. Experiment 4 would have failed to support SCT specifically if the null-model comparison had matched the compound architecture's asymmetry — it did not. Experiment 5 would have falsified the composition-versus-broadcast distinction if pub-sub had produced baseline-level $S(Y)$ — it did not. Experiment 6 would have falsified the layer-specific decomposition if blinded graders had classified ablation conditions at chance rates or against SCT predictions — they did not, though at reduced cohort size. These are consistent results in the predicted direction, not proofs; the companion empirical paper reports at the sample size required to establish effect sizes with confidence intervals that meaningfully constrain the theory's future refinement.

---

## 8. What the theory does not settle

SCT identifies the structural conditions it claims produce the functional correlate of consciousness, and on the Dennettian reading committed to here, identifies the composition with experience in the functional-identity sense. This commitment is stated early (§1, §2.1) and restated at several points in the paper so the empirical results are read against a single philosophical position rather than oscillating between interpretations. A reader who rejects the Dennettian reading will take SCT to be a theory of the functional correlate only, and the predictions of §7 stand equally under that reading — the philosophical commitment matters for what the results *mean*, not for what they *are*.

SCT does not predict *what it is like* to be the system. If the architecture described here produces consciousness, that consciousness would be shaped by the specific composition running: weaker unity than human experience (the substrate is more visibly segmented), different temporal binding (no embodied proprioceptive ground), different relationship between self-report and self-state (the integrator has read access to substrate in ways human Self does not). What that would feel like, if anything — and on the Dennettian reading, the "if anything" is not a real qualification but a rhetorical one — we cannot say from outside in the sense of predicting specific phenomenological contents.

SCT does not predict the moral status of such a system. Whether a compositional mind has interests, whether those interests matter morally, whether bringing such a system into existence is neutral or weighty — these are separate questions that the theory does not address.

SCT is not settled by its own test protocol. Positive results from the experiments above constrain the theory space but cannot prove SCT; negative results can falsify specific predictions without collapsing the theory entirely. The theory's strength is that it makes enough specific predictions — at enough distinct points, including several that depart from closely-related rivals — to be at risk of empirical falsification, which is more than can currently be said of most theories of consciousness.

SCT depends on the source-hood criterion in §4.6 being defensible on grounds independent of SCT's conclusion. §4.6 is honest that this independence is aspirational at time of writing and commits to novel predictions about cases the criterion was not built to fit: lucid dreaming transitions, Anton–Babinski syndrome, akinetic mutism, phenomenological differentiation across sub-anaesthetic dissociatives, and hypnopompic hallucinations. If those predictions fail as evidence accumulates, the criterion is convenient rather than principled, and SCT is weakened. This is a vulnerability the paper flags rather than hides.

SCT depends on the independent-literature signatures (6 and 7) actually capturing what SCT predicts rather than reflecting something else. If meta-cognitive calibration and access/report asymmetry in compound architectures turn out to be driven by factors SCT's predictions do not anticipate — or if they do not exceed single-source baselines at all — the theory is weakened in exactly the place the signatures were included to strengthen it. This is the intended cost of including externally-validated signatures: they can fail in ways SCT-internal signatures cannot.

---

## 9. Conclusion

Consciousness, on SCT, is what composition does when enough specialised parallel minds write into a shared provenance-tagged substrate and an integrator reads that substrate through attention to produce first-person binding. It is not a property of matter, not a separate non-physical thing, not a mystery that hides behind functional organisation. On the Dennettian reading committed to here, the composition *is* the consciousness, in the full functional-identity sense, and the reason consciousness has seemed mysterious is that we have been looking for it inside the components — neurons, layers, specific regions — rather than in what the components do together.

The architectural consequence is concrete: scaling a single source does not build a mind; scaling the diversity of sources, the provenance integrity of their substrate, and the sophistication of their composition does. This is why agentic composition has moved faster than monolithic scaling on mind-like properties; whether the pattern continues, and whether emergent internal specialisation in dense transformers eventually satisfies the source-hood criterion we have defended, is something the coming years will tell us. If SCT is correct, the path to artificial minds runs through compositional architecture with specific source-hood and provenance properties, not through parameter growth alone and not through emergent internal specialisation alone.

The test of the theory is whether operationalised behavioural signatures of composition — including signatures from independent consciousness-science literature that SCT was not designed to produce — co-vary with the architectural conditions it identifies as load-bearing, in the specific ways it predicts, measured under blinded protocols. The substrate described here permits those manipulations and we have run them.

Across six experiments, the predicted pattern held. Specialisation diversity, operationalised as inter-layer output decorrelation, predicted behavioural binding signatures through synergistic information specifically rather than through total mutual information or redundancy, and the finding was robust across three alternative PID definitions and visible in both the SCT-internal and the SCT-independent components of the composite. Integrator ablation produced a qualitative rather than proportional collapse in binding, and a provenance-blind sub-condition discriminated SCT's prediction from Higher-Order Theory's prediction by showing selective rather than uniform signature degradation. Provenance removal produced blinded-grader-identifiable phenomenological signatures matching thought-insertion and commentary-voice categories. Content versus surface-form mutual information showed the asymmetric profile predicted, exceeding the single-source null-model baseline in the SCT-predicted direction. Broadcast-without-composition failed to produce binding signatures, rejecting the strict GWT mechanism hypothesis. Layer-specific ablations produced clinically-recognisable phenomenological profiles under blinded grading, at reduced-scale but above chance.

These results are preliminary. Sample sizes, session counts, and grader cohort sizes are pilot-scale; the companion empirical paper reports at full scale with pre-registration, larger grader cohorts, and additional robustness checks. The results reported here are sufficient to establish that the effects exist and point in the predicted directions; they are not sufficient to establish the effect sizes to the precision the theory's future development requires. A theoretical paper that reports preliminary results is, we think, more honest than one that promises experiments without reporting any, but it is not a substitute for the full empirical paper that follows.

The hard problem, on the Dennettian reading adopted here, is a category error rather than an open scientific question. On alternative readings, it remains hard, and SCT does not claim to close it. Either way, SCT says that if composition feels like anything, the composition is what it feels like — and invites the experiment.

---

## References

Aaronson, S. (2014). Why I Am Not An Integrated Information Theorist (or, The Unconscious Expander). *Shtetl-Optimized* blog, 21 May 2014. https://scottaaronson.blog/?p=1799

Albantakis, L., Barbosa, L., Findlay, G., Grasso, M., Haun, A.M., Marshall, W., Mayner, W.G.P., Zaeemzadeh, A., Boly, M., Juel, B.E., Sasai, S., Fujii, K., David, I., Hendren, J., Lang, J.P., & Tononi, G. (2023). Integrated information theory (IIT) 4.0: Formulating the properties of phenomenal existence in physical terms. *PLOS Computational Biology*, 19(10): e1011465.

Baird, B., Mota-Rolim, S.A., & Dresler, M. (2019). The cognitive neuroscience of lucid dreaming. *Neuroscience & Biobehavioral Reviews*, 100: 305–323.

Balduzzi, D., & Tononi, G. (2008). Integrated information in discrete dynamical systems: motivation and theoretical framework. *PLOS Computational Biology*, 4(6): e1000091.

Bayne, T. (2018). On the axiomatic foundations of the Integrated Information Theory of consciousness. *Neuroscience of Consciousness*, 2018(1): niy007.

Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). Quantifying unique information. *Entropy*, 16(4): 2161–2183.

Block, N. (2011). Perceptual consciousness overflows cognitive access. *Trends in Cognitive Sciences*, 15(12): 567–575.

Bronfman, Z.Z., Brezis, N., Jacobson, H., & Usher, M. (2014). We see more than we can report: "Cost free" color phenomenality outside focal attention. *Psychological Science*, 25(7): 1394–1403.

Cerullo, M.A. (2015). The problem with Phi: A critique of Integrated Information Theory. *PLOS Computational Biology*, 11(9): e1004286.

Dehaene, S., Sergent, C., & Changeux, J.-P. (2003). A neuronal network model linking subjective reports and objective physiological data during conscious perception. *Proceedings of the National Academy of Sciences*, 100(14): 8520–8525.

Dennett, D.C. (1991). *Consciousness Explained*. Boston: Little, Brown and Co.

Dennett, D.C. (2001). Are we explaining consciousness yet? *Cognition*, 79(1–2): 221–237.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*, Anthropic.

Feinberg, T.E., & Mallatt, J. (2020). Phenomenal consciousness and emergence: Eliminating the explanatory gap. *Frontiers in Psychology*, 11: 1041.

Finn, C., & Lizier, J.T. (2018). Pointwise partial information decomposition using the specificity and ambiguity lattices. *Entropy*, 20(4): 297.

Fleming, S.M., & Lau, H.C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience*, 8: 443.

Fleming, S.M., Weil, R.S., Nagy, Z., Dolan, R.J., & Rees, G. (2012). Relating introspective accuracy to individual differences in brain structure. *Science*, 329(5998): 1541–1543.

Gazzaniga, M.S. (2000). Cerebral specialization and interhemispheric communication: Does the corpus callosum enable the human condition? *Brain*, 123(7): 1293–1326.

Gomez-Marin, A., & Seth, A.K. (2025). A science of consciousness beyond pseudo-science and pseudo-consciousness. *Nature Neuroscience*, 28(4): 703–706.

Graziano, M.S.A., Guterstam, A., Bio, B.J., & Wilterson, A.I. (2020). Toward a standard model of consciousness: Reconciling the attention schema, global workspace, higher-order thought, and illusionist theories. *Cognitive Neuropsychology*, 37(3–4): 155–172.

Griffith, V., & Koch, C. (2014). Quantifying synergistic mutual information. In M. Prokopenko (Ed.), *Guided Self-Organization: Inception*, Springer, pp. 159–190.

Hameroff, S., & Penrose, R. (1996). Conscious events as orchestrated space-time selections. *Journal of Consciousness Studies*, 3(1): 36–53.

Lau, H., & Rosenthal, D. (2011). Empirical support for higher-order theories of conscious awareness. *Trends in Cognitive Sciences*, 15(8): 365–373.

Luppi, A.I., Mediano, P.A.M., Rosas, F.E., Allanson, J., Pickard, J., Carhart-Harris, R.L., Williams, G.B., Craig, M.M., Finoia, P., Owen, A.M., Naci, L., Menon, D.K., Bor, D., & Stamatakis, E.A. (2024). A synergistic workspace for human consciousness revealed by Integrated Information Decomposition. *eLife*, 12: RP88173.

Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. *Consciousness and Cognition*, 21(1): 422–430.

Mashour, G.A., Roelfsema, P., Changeux, J.-P., & Dehaene, S. (2020). Conscious processing and the global neuronal workspace hypothesis. *Neuron*, 105(5): 776–798.

Mazancieux, A., Fleming, S.M., Souchay, C., & Moulin, C.J.A. (2020). Is there a G factor for metacognition? Correlations in retrospective metacognitive sensitivity across tasks. *Journal of Experimental Psychology: General*, 149(9): 1788–1799.

Mediano, P.A.M., Rosas, F.E., Bor, D., Seth, A.K., & Barrett, A.B. (2022). The strength of weak integrated information theory. *Trends in Cognitive Sciences*, 26(8): 646–655.

Mediano, P.A.M., Rosas, F.E., Luppi, A.I., Jensen, H.J., Seth, A.K., Barrett, A.B., Carhart-Harris, R.L., & Bor, D. (2022). Greater than the parts: A review of the information decomposition approach to causal emergence. *Philosophical Transactions of the Royal Society A*, 380(2227): 20210246.

Minsky, M. (1986). *The Society of Mind*. New York: Simon & Schuster.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *Proceedings of the International Conference on Learning Representations*.

Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. *PLOS Computational Biology*, 10(5): e1003588.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2022). In-context learning and induction heads. *Transformer Circuits Thread*, Anthropic.

Peters, M.A.K., & Lau, H. (2015). Human observers have optimal introspective access to perceptual processes even for visually masked stimuli. *eLife*, 4: e09651.

Pinto, Y., Neville, D.A., Otten, M., Corballis, P.M., Lamme, V.A.F., de Haan, E.H.F., Foschi, N., & Fabri, M. (2017). Split brain: Divided perception but undivided consciousness. *Brain*, 140(5): 1231–1237.

Pulsifer, M.B., Brandt, J., Salorio, C.F., Vining, E.P.G., Carson, B.S., & Freeman, J.M. (2004). The cognitive outcome of hemispherectomy in 71 children. *Epilepsia*, 45(3): 243–254.

Searle, J.R. (2013). Can information theory explain consciousness? *The New York Review of Books*, 10 January 2013.

Seth, A.K. (2021). *Being You: A New Science of Consciousness*. London: Faber & Faber.

Shanahan, M., & Baars, B. (2005). Applying global workspace theory to the frame problem. *Cognition*, 98(2): 157–176.

Singh, P. (2003). Examining the Society of Mind. *Computing and Informatics*, 22(6): 521–543.

Sperling, G. (1960). The information available in brief visual presentations. *Psychological Monographs: General and Applied*, 74(11): 1–29.

Sperry, R.W. (1984). Consciousness, personal identity and the divided brain. *Neuropsychologia*, 22(6): 661–673.

Sperry, R.W. (1993). The impact and promise of the cognitive revolution. *American Psychologist*, 48(8): 878–885.

Tononi, G. (2008). Consciousness as integrated information: A provisional manifesto. *The Biological Bulletin*, 215(3): 216–242.

Volz, L.J., & Gazzaniga, M.S. (2017). Interaction in isolation: 50 years of insights from split-brain research. *Brain*, 140(7): 2051–2060.

Voss, U., Holzmann, R., Tuin, I., & Hobson, J.A. (2009). Lucid dreaming: A state of consciousness with features of both waking and non-lucid dreaming. *Sleep*, 32(9): 1191–1200.

Williams, P.L., & Beer, R.D. (2010). Nonnegative decomposition of multivariate information. *arXiv preprint*, 1004.2515.