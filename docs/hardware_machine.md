# Battle Cities Inference Engine — Production Workstation Build

A 2-GPU custom-water-cooled workstation for running the composed-attention inference architecture described in *"Composing a Local Model with a Frontier Model through Shared Attention."* Targets Qwen3-235B-A22B serving at wave-coherent concurrent execution, with disk-paged KV cache backed by provenance-guided prefetch, running the author's Candle fork with full CUDA SM_120 support.

Approximate MSRP: USD ~15,750. Sydney street pricing: AUD ~26,000–30,000 depending on 5090 availability and AUD/USD rate at order time.

---

## Machine summary

| Category | Spec |
|---|---|
| Compute | 2× RTX 5090 32GB GDDR7 (water-cooled, modest sustained overclock) + Threadripper 7970X 32-core 4.0 GHz |
| Memory | 512GB DDR5-5200 ECC RDIMM (quad-channel, ~200 GB/s aggregate) |
| Storage | 16TB PCIe 5.0 NVMe RAID 0 for paged KV cache (~45 GB/s sustained) + 2TB PCIe 4.0 for OS |
| Interconnect | 2× PCIe 5.0 x16 per GPU (uncontested), 4× PCIe 5.0 M.2 CPU-direct |
| Cooling | Full custom water loop: both GPUs and CPU on water, 2× 420mm radiators, Mayhems X1 Eco UV Clear coolant (5-year service life) |
| Power | Single Corsair AX1600i Titanium, ~1,600W peak system draw |
| Chassis | Phanteks Enthoo 719 (fits 2× 420mm radiators) |
| Expected noise under sustained load | ~28–32 dBA |
| Expected GPU sustained temp under load | ~55–62°C junction |

---

## Use case

The primary purpose of this machine is to run the composed-attention inference architecture in production as the author's daily research and development platform, and as the engine powering Battle Cities NPC cognition.

**Research rig.** Runs the full tactical substrate (specialist lattice with observable attention, research augmentation, attention-trace context selection, line-granular interleaved inference with LRU eviction) and the cognitive stack (beliefs, self, goals, relationships, missions) on top. Primary local model is Qwen3-235B-A22B at Q4/Q6 weights. Frontier interleaving handled via Claude API through the composition described in §7 of the architecture paper — approximately 1% of token volume, targeted at mission breakpoints where frontier capability genuinely differentiates.

**Battle Cities backing engine.** The same hardware serves the inference engine backing NPC cognition in Battle Cities. NPCs run on local models (Qwen3-30B-A3B class for most characters, with selective frontier interleaving for high-stakes interactions). Wave-coherent concurrent execution allows many NPCs to share the 235B reservoir when a heavier model is needed for specific narrative moments. The same box supports development and serving — no separate production fleet during early access.

**Coding workstation.** Daily-use machine for NeurIPS paper follow-up work, Battle Cities development, and general engineering. The composed-attention coding engine runs continuously in the background, with the specialist lattice attending to active codebases and cognitive stack maintaining cross-session continuity. Frontier interleaving via Claude Max 20x at approximately USD 200/month covers the ~1% of token volume requiring frontier capability, as estimated in §19 of the architecture paper.

---

## Inference architecture running on this machine

The hardware is sized specifically to the requirements of the inference architecture. Each layer of the architecture places constraints on hardware that drove specific component choices.

### Tactical substrate

The specialist lattice runs approximately 32 concurrent contexts per active user session, each a conversation under its own system prompt, all sharing a KV prefix (the codebase or current operating context). Shared KV amortises encoding cost across specialists — the codebase is encoded once and all specialists attend to it.

Per-session working memory is dominated by specialist KV deltas (roughly 100–200 MB per specialist at 32K context with palette4 compression) plus the shared prefix (4–16 GB depending on codebase size). Working VRAM set per active user: roughly 25–50 GB.

### Wave-coherent 235B serving

Qwen3-235B-A22B weights at Q4/Q6 mix (~135 GB total) live in system RAM as a reservoir. The wave model streams active experts for the current layer front into VRAM, prefetches next-layer experts via the Markov transition matrix (69% hit rate on 30B-A3B baseline, to be validated on 235B), and evicts experts the wave has passed.

Because all specialists and the main decoder step through layers coherently (the wave), expert weights for the current layer are shared across all concurrent streams. Weight bandwidth demand does not scale linearly with concurrency — this is the key architectural insight that makes concurrent serving of a 235B model viable on consumer hardware.

Pipeline split across the 2 GPUs: layers 0–46 on GPU A, 47–93 on GPU B. Activations cross the PCIe boundary once per forward pass, a negligible ~400 KB per token at realistic batch sizes. Each card holds its portion of the wave plus KV for all active sessions, giving approximately 30 GB of working VRAM per card for active serving.

### Disk-paged KV with attentional provenance retrieval

KV cache for archived sessions lives on the 16 TB PCIe 5.0 NVMe RAID 0, not in RAM. Active session KV stays in RAM; historical KV pages in from disk on retrieval triggered by attentional provenance (Binary Directional Provenance: sign(Q_PCA^T @ K) per token, flat multithreaded XOR+popcount scan).

Disk bandwidth demand at steady state: approximately 24 GB/s during active paged retrieval, assuming 85% provenance prediction accuracy. The 4-drive RAID 0 provides ~45 GB/s sustained, giving ~2× headroom for burst demand and misprediction tails.

### Cognitive stack

Beliefs, self, goals, relationships, and missions each run as event-driven conversations under system prompts, writing into the shared KV substrate. CPU-side orchestration and the Binary Directional Provenance scan — which is the heaviest CPU work in the architecture — run on the Threadripper 7970X's 32 cores with AVX-512, bandwidth-bound against the quad-channel DDR5 at ~200 GB/s.

---

## Full parts list

| Component | Part | USD |
|---|---|---|
| GPU 1 | NVIDIA RTX 5090 Founders Edition | 2,000 |
| GPU 2 | NVIDIA RTX 5090 Founders Edition | 2,000 |
| CPU | AMD Threadripper 7970X (32C/64T, 4.0 GHz base / 5.3 GHz boost, 350W) | 2,499 |
| Motherboard | Gigabyte TRX50 AI TOP (sTR5, 4× PCIe 5.0 x16, 4× PCIe 5.0 M.2 CPU-direct) | 1,200 |
| RAM | 4× 128GB DDR5-5200 ECC RDIMM (V-Color TRQJ564G52S428KWK or Kingston KF552R40BBK4-512) | 2,800 |
| KV storage | 4× Crucial T705 4TB PCIe 5.0 NVMe (software RAID 0 via mdadm) | 2,000 |
| OS storage | 1× Samsung 990 Pro 2TB PCIe 4.0 NVMe | 150 |
| NVMe heatsinks | Active M.2 heatsinks with fans | 150 |
| GPU waterblocks | 2× EKWB Quantum Vector2 RTX 5090 | 500 |
| CPU waterblock | EKWB Velocity² sTR5 | 180 |
| Pump + reservoir | EKWB Quantum Kinetic TBE 200 D5 PWM | 280 |
| Radiators | 2× Hardware Labs Black Ice Nemesis 420GTS | 320 |
| Radiator fans | 6× Noctua NF-A14x25 PWM 140mm | 240 |
| Case fans | 3× Noctua NF-A14 PWM 140mm (non-radiator positions) | 90 |
| Fittings | 16× EKWB Torque compression (G1/4 to 16mm hard tube) | 128 |
| Tubing | EKWB ZMT PETG hard tube 16mm OD × 3m (plus 50% spare for bending) | 60 |
| Tube bending kit | EKWB Loop tubing bending tool kit | 40 |
| Coolant | Mayhems X1 Eco UV Clear 2L + silver kill coil | 85 |
| Flow monitor | Aquacomputer high-flow USB flow sensor | 90 |
| Leak insurance | EKWB leak protection | 40 |
| Case | Phanteks Enthoo 719 (fits 2× 420mm radiators) | 250 |
| Power supply | Corsair AX1600i fully modular Titanium | 500 |
| Custom PSU cables | Sleeved modular cables (optional aesthetic) | 150 |
| **Total (USD, MSRP)** | | **~15,752** |

Sydney street pricing runs higher due to 5090 scarcity premium and AUD/USD. Expect AUD 26,000–30,000 all-in. Source from Scorptec, Mwave, Centre Com, Umart, or PCCG. Build-service labor adds AUD 600–900 if outsourcing the custom loop assembly.

---

## Justification

### GPUs — 2× RTX 5090 Founders Edition

64 GB aggregate VRAM across 2 cards is sufficient for wave-coherent execution of Qwen3-235B-A22B because weights live in system RAM, not VRAM. Each card holds only its pipeline stage's active wave plus KV for concurrent sessions — roughly 30 GB per card. The 235B weight set (135 GB at Q4/Q6 mix) does not need to be VRAM-resident.

The 5090's 1,792 GB/s memory bandwidth and ~1,676 TFLOPS FP16 tensor compute are comfortable headroom for the wave at realistic concurrency. Compute is not the binding constraint at 2-card scale; PCIe 5.0 x16 ingress bandwidth from system RAM is. Each card gets uncontested x16 on TRX50, ~60 GB/s effective ingress, sufficient for synchronous expert-miss transfers at the measured Markov hit rate plus async-overlapped prefetch traffic.

Founders Edition over AIB partner cards for three reasons. First, reference PCB means standard waterblock compatibility (EKWB Quantum Vector2 is designed for FE PCB). Second, stock cooler removal for waterblock install is clean and well-documented. Third, FE cards tend to be more widely available at MSRP than partner models — Sydney availability is still erratic but Founders Edition is generally the easier find.

Modest sustained overclock (+150 MHz core, +1500 MHz memory, +5% power limit) via MSI Afterburner — software only, no BIOS flash. Water cooling lets the cards hold these offsets indefinitely without thermal throttling. Expected throughput gain ~10% over stock water-cooled, compounding across thousands of hours of inference.

Single failure domain per card (no P2P required on consumer Blackwell drivers — pipeline split and wave-coherent execution work over standard PCIe without NCCL's P2P path). This matches the architecture naturally: pipeline parallelism composes with wave execution cleanly, while tensor parallelism (which would require P2P) is not relevant to the workload.

### CPU — AMD Threadripper 7970X

32 cores at 4.0 GHz base / 5.3 GHz boost, full AVX-512, 350W TDP. Three considerations drove this choice over the 24-core 7960X.

The Binary Directional Provenance scan is CPU-side and memory-bandwidth-bound. It parallelises trivially across cores up to the DDR5 bandwidth ceiling. On TRX50 quad-channel at ~200 GB/s, the useful core count for scan workloads is approximately 24–32 cores — past that, cores sit idle waiting on RAM. The 7970X at 32 cores saturates the available bandwidth; going to 64 cores on the 7980X would waste silicon on a bandwidth-capped workload while dropping base clock to 3.2 GHz and losing single-thread scan latency.

Specialist lattice orchestration adds concurrent load — 32 specialist conversations per user session spawn Rust async tasks that the CPU must schedule. 24 cores handle single-user orchestration; 32 cores give headroom for multi-user scenarios (Battle Cities serving with multiple concurrent NPCs running the full cognitive stack).

AVX-512 throughput is load-bearing for provenance scan performance. Each 5090 decode step triggers attention-based KV selection; the CPU must complete scans fast enough not to block the GPU. Threadripper 7000 has full AVX-512 support with per-core throughput at ~2 vectors/cycle at 4 GHz. At 32 cores this gives ~4 TB/s theoretical scan throughput, capped by the ~200 GB/s RAM bandwidth ceiling — cores run out of data before running out of compute, which means high core utilisation without diminishing returns.

The PRO variant (7975WX on WRX90) was considered and rejected. PRO would double RAM bandwidth to ~400 GB/s via 8-channel and provide 128 PCIe lanes, but neither benefit is justified for 2-GPU wave-coherent execution. 128 lanes matter for 4+ GPUs; 8-channel matters for scan workloads that exceed TRX50's 200 GB/s ceiling. Neither is true for this build's target workload. The platform upgrade cost (~$2,500) buys headroom this architecture doesn't need.

### Motherboard — Gigabyte TRX50 AI TOP

Four considerations: PCIe 5.0 x16 to both GPUs uncontested, native 4× PCIe 5.0 M.2 CPU-direct slots for the NVMe RAID, upgrade path to 4-GPU if requirements change, and thermal design adequate for sustained full-load operation.

The AI TOP is the only TRX50 board with 4× PCIe 5.0 M.2 slots wired directly to the CPU. Standard TRX50 boards expose 3× PCIe 5.0 M.2 CPU-direct plus M.2 via chipset (PCIe 4.0). The 4th CPU-direct slot matters because the KV-paging workload demands all 4 NVMe drives be on the same bandwidth tier for RAID 0 to perform predictably.

4× PCIe 5.0 x16 slots mean future expansion to 4 GPUs remains possible without motherboard replacement. The non-PRO Threadripper 7970X's 80 PCIe lanes allow x16/x16/x8/x8 with four GPUs populated (x16/x16 for the first two, x8/x8 for the expansion pair) plus 16 lanes for NVMe RAID. That's not full x16 across all four, but it's workable. For now, 2× GPU at x16 each plus 4× NVMe at x4 each uses 48 lanes, well under the 80-lane budget.

Thermal and power design adequate for sustained 350W CPU plus 1,150W combined GPU load. The AI TOP's VRM is rated for overclocked PRO CPUs at 400W+; running stock 7970X is well within envelope.

The ASRock TRX50 WS at ~$800 was considered as a cheaper alternative. It saves $400 but requires a PCIe 5.0 AIC card (~$250) to carry the 4th NVMe drive. Net saving after AIC: ~$150. The AI TOP's native 4-slot topology is cleaner, avoids AIC thermal hotspot, simplifies cable routing, and preserves the x16 slot that would otherwise carry the AIC. Worth the $150 premium for topology simplicity alone.

### RAM — 512GB DDR5-5200 ECC RDIMM (4× 128GB)

512 GB is the weight reservoir capacity. Qwen3-235B-A22B at Q4/Q6 mix occupies ~135 GB. The remaining ~380 GB provides headroom for:

- Alternative models loaded simultaneously for research comparison (DeepSeek V3 671B at Q4 = ~340 GB fits in remaining headroom with tight packing)
- Active session KV caches for concurrent user sessions (~3–5 GB per session × dozens of sessions)
- System RAM, page cache, and normal workstation overhead (~40 GB reasonable allocation)
- Future model growth without requiring DIMM replacement

Four 128GB DIMMs populate all four DDR5 channels at 1DPC, which is the Threadripper 7000 non-PRO specification for full DDR5-5200 operation. 8 DIMMs at 64GB each on the Gigabyte AI TOP's 8-slot option would technically match capacity but run at DDR5-4400 (2DPC downclock), reducing effective RAM bandwidth by ~15–20%. That bandwidth loss directly hurts expert streaming to GPUs and CPU-side provenance scans — exactly the workloads this machine is optimised for. 4× 128GB 1DPC at full speed is the correct configuration.

ECC RDIMM over UDIMM for reliability during 24/7 operation. DDR5 RDIMMs run cooler and more reliably than UDIMMs at 128GB module capacity; ECC catches bit flips during sustained inference where a single corrupted weight could produce subtly wrong outputs across hours of work before anyone notices.

V-Color or Kingston kits specifically — these vendors maintain TRX50 AI TOP QVL validation for 4× 128GB configurations. Non-QVL kits frequently fail to train at DDR5-5200 with 4 DIMMs populated, dropping to 4800 or failing POST entirely.

### KV storage — 4× Crucial T705 4TB PCIe 5.0 NVMe RAID 0

The architectural decision to page KV from disk (rather than keeping all historical KV in RAM) dictates the storage spec. Disk bandwidth becomes part of the hot path, not just cold storage.

Sustained bandwidth target: ~24 GB/s under active provenance-guided retrieval with 85% prediction accuracy. 4× PCIe 5.0 NVMe in RAID 0 delivers ~45 GB/s sustained — nearly 2× headroom over the target, covering burst demand and misprediction tails without becoming the bottleneck.

Crucial T705 4TB over alternatives: each drive delivers ~13 GB/s sustained read under thermal load, matching premium alternatives (Samsung 9100 Pro) at ~15% lower per-drive cost. For RAID 0 where aggregate bandwidth dominates and per-drive peak differences are small, the more affordable drive wins. 16 TB total capacity holds approximately 3,000+ full session KV archives at ~5 GB per session — enough for cross-session provenance retrieval over the machine's expected 3–5 year service life.

Software RAID 0 via Linux mdadm rather than hardware RAID. Hardware NVMe RAID controllers add latency and cost with no bandwidth benefit. Linux kernel 6.x handles 4-drive NVMe RAID 0 cleanly. Stripe size of 128 KB matches typical KV chunk granularity in the architecture — important to set explicitly; the 4 KB default fragments access patterns.

RAID 0 has zero redundancy. Acceptable because session KV is regeneratable (it's derived from session inputs, which are logged separately) and cross-session archival KV is a research convenience, not a single source of truth. Single drive failure means rebuilding the archive from session logs, not catastrophic data loss. A mirror would double storage cost and halve bandwidth — not worth the trade for this workload.

### Water cooling — full custom loop

Three reasons the custom loop is justified despite adding ~$2,000 to the build over air cooling.

First, sustained load thermals. Air-cooled 5090s throttle from 2.41 GHz boost to ~2.3 GHz under sustained serving load due to thermal limits. Water cooling holds the full boost indefinitely, which compounds to meaningful throughput across thousands of hours of inference. The 10% overclock headroom on top of that is gravy.

Second, noise. Under sustained full-load serving, air-cooled 2× 5090 + Threadripper produces ~45–50 dBA at the case — audible across a home office, fatiguing over long work sessions. Custom loop with large radiators and slow-spinning Noctua fans drops sustained noise to ~28–32 dBA, which is genuinely office-quiet. For a machine running 6+ hours daily in the author's home office, this matters.

Third, component life. GPU junction temperatures of 55–62°C under sustained load (vs 78–82°C on air) translate to longer VRM and memory module life. On a $4,000 GPU pair running continuously for 3+ years, lower thermal stress is measurable reliability gain.

Component choices:

**EKWB Quantum Vector2** waterblocks are the current-gen design for 5090 FE PCB. Full-cover design (GPU die, memory modules, VRM), nickel-plated copper baseplate, acrylic top for visibility. Three-year warranty.

**EKWB Velocity² sTR5** CPU block sized for Threadripper's large IHS. Copper base, high fin density, threaded for standard G1/4 fittings.

**EKWB Quantum Kinetic TBE 200 D5 PWM** pump/reservoir combo. D5 pump class (not DDC) because the loop has 3 blocks + 2 radiators, which loads DDC pumps near their head-pressure limit. D5 runs quiet at moderate speed with headroom for future expansion.

**2× Hardware Labs Black Ice Nemesis 420GTS** radiators provide approximately 2,000W dissipation capacity, comfortable headroom over the ~1,600W peak system load with overclock active. The GTS profile is 54mm thick — aggressive but acceptable in the Phanteks 719's radiator bays.

**Noctua NF-A14x25 PWM** fans are the quietest-at-performance 140mm fans available for radiator work, ~24 dBA at 1500 RPM delivering adequate airflow.

**PETG hard tubing** over soft tubing for aesthetic and long-term reliability. Soft tubing plasticises over years of coolant exposure; PETG is chemically stable for the coolant's 5-year service life. Hard tubing adds 2–3 hours to build time and requires a bending kit ($40 one-time).

**Mayhems X1 Eco UV Clear coolant** with silver kill coil — rated 5-year service life in sealed loops. Clear base (no dye to degrade), established corrosion inhibitor package, biocide for biological growth prevention. Clear coolant shows contamination early; colored coolants obscure the diagnostic signal.

**Aquacomputer high-flow USB sensor** provides continuous flow monitoring visible in the OS. Early warning if pump begins failing or if air ingress reduces flow — catches problems before thermal emergency. Small investment ($90) against loop health.

**EKWB leak insurance** ($40) covers component replacement if EKWB blocks fail and cause damage. Marginal cost against $4,000 of GPUs.

### Power supply — Corsair AX1600i

Peak system draw under sustained overclocked load: approximately 1,600W. Components:

- 2× RTX 5090 at ~615W each (overclocked, +7% power limit) = 1,230W
- Threadripper 7970X sustained = 350W
- RAM, NVMe drives, fans, pump = 100W peak

1,600W Titanium-efficiency PSU provides margin without forcing dual-PSU complexity. Corsair AX1600i specifically for three reasons: digital controller with USB monitoring, individual 12V rail current monitoring, and the highest reliability record in the 1500W+ range based on years of community reports. Single-rail design simplifies cable routing and avoids multi-rail balance issues under heavy GPU loads.

Fully modular cables mean only needed connections run, simplifying case build and airflow. The optional custom sleeved cables ($150) are aesthetic, not functional — skip if budget-sensitive.

At 1,600W peak on a Sydney 10A 240V household circuit (2,400W capacity), there's approximately 800W of overhead for anything else sharing the circuit. Adequate for a dedicated home-office circuit without electrical work.

### Chassis — Phanteks Enthoo 719

Three requirements drove case selection. First, dual 420mm radiator support — needed for the cooling spec. Second, physical clearance for 2× triple-slot 5090s with waterblocks attached (each card becomes single-slot with block installed, so 2 adjacent slots comfortable). Third, serviceability for the custom loop — drain ports accessible, pump/reservoir mount clean, cable routing hidden behind the motherboard tray.

The Phanteks Enthoo 719 (formerly Enthoo 719) fits all three plus provides ServerMB support (EEB form factor if the motherboard ever changes), 140mm front intake capacity for positive pressure, and sound-dampened side panel optional. Not a "showcase" case aesthetically — the Lian Li O11 Dynamic EVO XL is more visually striking — but the 719 fits 2× 420mm radiators where the O11 Evo XL is limited to 1× 420mm + 1× 360mm. For thermal headroom with overclock, the 719's additional radiator capacity is worth the aesthetic trade-off.

Three Noctua NF-A14 case fans in non-radiator positions provide supplementary airflow and slight positive pressure to reduce dust ingress.

### OS storage — Samsung 990 Pro 2TB PCIe 4.0 NVMe

Separate OS drive isolates OS I/O churn (logs, package installs, Docker layer operations, swap) from the KV RAID, which needs to serve sustained sequential reads without random I/O contention. A single drive failure on the OS drive does not compromise the KV archive; a failure on the KV RAID does not compromise the boot environment.

PCIe 4.0 rather than 5.0 because OS workloads are latency-bound (small random reads), not bandwidth-bound. The 990 Pro's ~7 GB/s sustained read is more than OS operations ever saturate. PCIe 5.0 OS drives exist but their premium buys no practical benefit.

2TB capacity holds OS (100 GB), home directory with code and configs (~200 GB), Docker images and build caches (~300 GB), model weight staging area (~500 GB), with ~900 GB headroom. Samsung 990 Pro over alternatives for Linux driver maturity and thermal behaviour at sustained load.

---

## Pre-build validation plan

Before committing to the order, validate three assumptions empirically using rental hardware on Runpod, Tensordock, or similar:

1. **Markov prefetch accuracy on 235B.** The 69% hit rate measured on Qwen3-30B-A3B must transfer to 235B's routing patterns. Measure hit rate against real 235B traffic with the author's engine; if it drops below 55%, reconsider RAM bandwidth assumptions (potentially WRX90 upgrade becomes relevant).

2. **Wave coherence under concurrent load.** Run 32 concurrent specialist streams against 235B and verify the wave stays coherent rather than smearing. If wave coherence degrades, the concurrent-session scaling assumption weakens and the build's value proposition narrows.

3. **Provenance-prefetch accuracy on disk-paged KV.** The 85% prediction accuracy assumed for disk KV misses must hold on real session histories. Test by archiving real session KV to NVMe RAID, running retrieval queries, measuring actual miss rate versus predicted miss rate. If accuracy drops below 70%, the 24 GB/s sustained bandwidth target becomes inadequate and the NVMe spec may need upgrading (8-drive RAID becomes relevant).

Rental cost for validation: approximately USD 150–300 for a weekend of testing, against a USD 15,750 build commitment. Obvious due diligence.

---

## Execution sequence

1. Complete and submit NeurIPS paper on current RTX 4090 Mobile hardware. Nothing in this build helps the paper; timing matters.
2. Validate the three assumptions above on rental hardware post-submission.
3. Get Sydney quotes on all components from Scorptec, Mwave, Centre Com, Umart, and PCCG. Expect 2–6 week lead times on 5090 availability specifically.
4. Confirm RAM QVL with Gigabyte for the specific V-Color or Kingston kit before ordering.
5. Decide custom loop build path: self-build (weekend project), or system-integrator build (AUD 600–900 labour, warranty intact).
6. Order complete kit with electrical review confirmed — Sydney 10A 240V household circuit handles 1,600W peak without issue; verify the specific circuit isn't shared with other heavy loads during sustained use.
7. Build time: 6–8 hours for experienced custom-loop builder, a full weekend for first-time custom-loop build. Add 24-hour leak test before powering components on.
8. Burn-in period: 72 hours of stress testing (Prime95 for CPU, FurMark + 3DMark stress for GPUs, mixed loads for thermals) before committing production workload.

Target production running: approximately 2 months after NeurIPS submission, aligned with Battle Cities Steam early access mid-2026.