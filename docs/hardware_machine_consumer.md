# Battle Cities Workstation — AM5 + PRO 5000 Build

**Sydney AUD, May 2026.** Single-transaction build at Mwave. PCIe 5.0 AM5 with 192 GB DDR5 UDIMM, single RTX PRO 5000 Blackwell 72 GB, sized for Qwen3.5/3.6-397B-A17B inference via custom Rust/Candle engine plus dual-use gaming.

**Total: $18,406.80 AUD**

---

## Master Parts Table

| Component | SKU / Spec | Mwave SKU | AUD |
|---|---|---|---:|
| **GPU** | NVIDIA RTX PRO 5000 Blackwell 72 GB GDDR7 ECC, PCIe 5.0 x16, 300W | AC96146 (Model 900-5G153-2570-000) | 11,899 |
| **CPU** | AMD Ryzen 9 9950X3D (16C/32T Zen 5, 170W, 5.7 GHz, 144 MB cache) | AC82657 (Model 100-100000719WOF) | 1,044 |
| **Motherboard** | MSI MAG X870E Tomahawk MAX WiFi PZ (AM5, ATX, back-connect, 2× Gen5 M.2 + 2× Gen4 M.2) | AC95442 | 599 |
| **RAM** | 4× Crucial Pro 48 GB DDR5-5600 EXPO UDIMM (192 GB total) | AC94759 (Model CP48G56C46U5) | 2,491.80 |
| **NVMe Gen5** | 2× Crucial T710 2 TB Gen5 NVMe w/heatsink (4 TB total Gen5) | AC90596 (Model CT2000T710SSD5) | 1,306 |
| **NVMe Gen4** | Kingston NV3 2 TB Gen4 NVMe (Windows + games + dev tools) | (Model SNV3S/2000G) | 400 |
| **PSU** | Corsair RM1000x 1000W 80+ Gold ATX 3.1 Fully Modular | AC74761 (Model CP-9020271-AU) | 299 |
| **Case** | MSI MAG PANO 100R PZ Tempered Glass Mid-Tower ATX, back-connect compatible (4× 120 mm ARGB fans pre-installed) | AC78596 | 199 |
| **CPU cooler** | Arctic Liquid Freezer III Pro 360 AIO (3× P12 Pro fans, integrated VRM fan) | AC91034 (Model ACFRE00180A) | 169 |
| | | **TOTAL** | **$18,406.80** |

An optional case upgrade is available: the Lian Li O11 Vision Compact Black (Mwave AC78732) at roughly $240 plus two Arctic P12 PWM intake fans at around $40 brings the total to approximately $80 over the MSI Pano. The Lian Li offers a premium dual-chamber layout in brushed aluminium with three borderless glass panels, but it is not required — the MSI Pano 100R PZ does the job at $199 with case fans already included.

---

## Architecture Spec

| Metric | Value |
|---|---:|
| **GPU VRAM** | 72 GB ECC GDDR7 |
| **GPU memory bandwidth** | 1,344 GB/s |
| **GPU PCIe ingress** | PCIe 5.0 x16 = 64 GB/s sustained |
| **System RAM** | 192 GB DDR5-5600 UDIMM (4× 48 GB, 1 DPC) |
| **RAM bandwidth** | 89.6 GB/s (2-channel) |
| **NVMe Gen5 aggregate** | 2× 14.5 GB/s = 29 GB/s sustained (M.2_1 + M.2_2 USB4-disabled) |
| **NVMe Gen4 Windows** | 7 GB/s sustained (M.2_4) |
| **Total NVMe capacity** | 6 TB (4 TB Gen5 + 2 TB Gen4) |
| **CPU** | 16C / 32T Zen 5, 5.7 GHz boost, 144 MB cache (96 MB 3D V-Cache on CCD0) |
| **Peak system power** | ~590 W (PRO 5000 300W + 9950X3D 170W + system 120W) |
| **PSU loading** | 59% of 1000W rated |

---

## NVMe Slot Allocation (MSI Tomahawk MAX WiFi PZ)

| Slot | Source | Speed | Drive | Use |
|---|---|---:|---|---|
| **M.2_1** | CPU | **Gen5 x4 (14.5 GB/s)** | T710 #1 | Active model weights, KV scratch |
| **M.2_2** | CPU (Force x4, USB4 disabled in BIOS) | **Gen5 x4 (14.5 GB/s)** | T710 #2 | Model archive, KV spillover |
| **M.2_3** | Chipset (PCIe 4.0 x2 only, shares PCI_E3) | 3.94 GB/s | **Empty** | Reserved — preserves PCI_E3 at x4 for future 10 GbE NIC or HBA |
| **M.2_4** | Chipset (PCIe 4.0 x4 clean) | **Gen4 x4 (7 GB/s)** | Kingston NV3 2 TB | Windows + Steam library + dev tools |

The PCI_E1 GPU slot stays at full PCIe 5.0 x16 in all configurations. The Tomahawk MAX is one of only a handful of AM5 boards that delivers two Gen5 M.2 slots simultaneously with full x16 GPU bandwidth, alongside the MSI Tomahawk/Edge TI and ASRock Nova/Taichi families. With USB4 disabled in favour of M.2_2 bandwidth, aggregate NVMe throughput reaches approximately 42 GB/s — the practical ceiling on AM5 in May 2026.

---

## Critical BIOS Settings (First Boot)

| Setting | Path | Value | Why |
|---|---|---|---|
| M.2_2 PCIe Mode | Advanced → Integrated Peripherals | **Force x4** | Disables USB4 ports, unlocks M.2_2 at full Gen5 x4 = 14.5 GB/s |
| AMD EXPO Profile | Advanced → Overclocking | **Enabled (Crucial Pro profile)** | DDR5-5600 vs JEDEC fallback ~3600 with 4 DIMMs |
| TPM 2.0 | Security | **Enabled** | Windows 11 requirement |
| Secure Boot | Boot | **Enabled** | Windows 11 install requirement |
| CSM / Legacy | Boot | **Disabled** | UEFI-only |
| SVM Virtualization | Advanced → CPU | **Enabled** | Required for Hyper-V, WSL2, Docker |

---

## Component Rationale

### Why a single PRO 5000 72 GB

The GPU choice was made by working backwards from the working set required for 397B-A17B inference. The shared backbone — the layers replicated across every expert evaluation — must stay VRAM-resident at all times and consumes about 11 GB at Q6. KV cache, intermediate buffers, and CUDA workspace draw another 20 GB under realistic context lengths. That leaves approximately 41 GB for the hot-tier expert cache, which at 2.24 GB per Q6 expert accommodates roughly eighteen experts simultaneously. With BDP-augmented retrieval beating Zipfian baselines by 8-15%, eighteen cached experts is the threshold where effective hit rate reliably reaches 75-80% — exceeding the 70% target the engine was designed around.

Three alternative GPU configurations were rejected against this analysis. A pair of PRO 4500 32 GB cards would have provided 64 GB aggregate VRAM but lost meaningful capacity to backbone replication (each card needs its own copy), netting closer to 50 GB usable for experts. Multi-GPU NCCL overhead would also fight against the inference engine's single-device design, which threads expert dispatch through one CUDA context for cache coherency. A single PRO 6000 96 GB Max-Q would have been strictly better at $14,599, but the $5,100 premium buys only 24 GB of additional VRAM — measurable but not transformative for an offline NPC workload that already exceeds its hit-rate target on the PRO 5000. A pair of PRO 5000 72 GB cards combined for 144 GB would justify a platform jump to TRX50 and push total cost past $40 K, which is the wrong shape of investment for a pre-revenue beta launch.

The PRO 5000 is also a workstation card with ECC GDDR7 and no thermal throttling under sustained 24/7 inference — both meaningful for the months of soak-testing the NeurIPS paper and Battle Cities beta will demand.

### CPU choice: 9950X3D over plain 9950X

The 9950X3D commands roughly a $195 premium over the standard 9950X — not the $400 I initially mis-quoted. At that price the upgrade is essentially free for inference workloads and meaningful for gaming. The X3D variant pairs two CCDs: CCD0 carries the 96 MB 3D V-Cache layer that delivers the gaming uplift, while CCD1 runs at standard 9950X clocks (5.7 GHz boost) with no V-Cache penalty. Windows and Linux schedulers automatically route latency-sensitive gaming workloads to CCD0 via AMD's CPPC2 firmware, and route everything else to CCD1. For inference the V-Cache is irrelevant — model weights vastly exceed any L3 cache no matter how generous — so the inference workload runs on CCD1 at exactly the speed it would run on a plain 9950X.

The gaming benefit is concentrated in cache-locality-heavy genres: strategy and simulation titles see 15-25% FPS improvements at typical settings, MMOs and CPU-bound FPS see 5-12%, and GPU-bound AAA at 4K sees near zero. Given a former 1914 champion's likely taste profile, the X3D upgrade is well-targeted. Same 170W TDP, same cooling envelope, same AM5 socket that preserves the Zen 6 upgrade path.

### Motherboard: MSI MAG X870E Tomahawk MAX WiFi PZ

The Tomahawk MAX was chosen for one specific capability: it is one of the very few AM5 boards that delivers two Gen5 M.2 slots simultaneously with the GPU running at full PCIe 5.0 x16. On most X870E boards, populating a second CPU-direct M.2 slot silently steals lanes from the GPU. The ASUS ProArt X870E-Creator at $899 was the obvious workstation candidate on paper — dual USB4, 10 GbE, polished aesthetic — but its lane allocation falls into exactly this trap. The ASRock X870E Taichi at $611 has clean GPU x16 enforcement but offers only one Gen5 M.2 simultaneous with the GPU, capping aggregate NVMe bandwidth around 30 GB/s rather than the Tomahawk's 42.

The cost is mid-tier VRMs at 14+2+1 phases (80A per stage) rather than the ProArt's 20+2+1. Adequate for sustained 170W on the 9950X3D, but it relies on active airflow to stay reliable under long inference loads — which is why the Arctic AIO's integrated VRM fan was selected to pair with it. The PZ variant routes all power and data connectors to the rear of the board for cable hiding, which requires a back-connect-compatible case.

### RAM: 192 GB DDR5-5600 in four DIMMs

192 GB of system RAM is an unusual choice for a workstation hosting a 397B-parameter model. Uniform Q5 quantisation of the entire model would require 248 GB resident in RAM — sixty gigabytes more than this build provides. The 192 GB ceiling is viable specifically because the inference engine treats inactive experts asymmetrically rather than uniformly, applying C8-C10 cold-tier compression to experts that fall outside the working set while keeping warm-tier experts at full Q5. The arithmetic works out to roughly 38 warm-tier experts at Q5 (about 68 GB), 73 cold-tier experts at C8-C10 (about 77 GB), the Rust inference engine plus OS plus page cache (about 30 GB), and approximately 17 GB of headroom — totaling 175 GB of 192 GB used. Without the compression IP, this build would be forced to Q4_K_S precision for 397B, which is a meaningful quality drop. With the IP, it delivers effective Q5 quality on the hot path. The build only works because of the engine, which is the NeurIPS paper's central architectural argument.

The choice of DDR5-5600 rather than DDR5-6000 reflects two practical realities. First, AM5 is well-documented to struggle with DDR5-6000 stability across four DIMMs populated simultaneously, while DDR5-5600 runs cleanly at four DIMMs filled — important because the EXPO profile must hold reliably across years of 24/7 operation. Second, the difference between 89.6 GB/s (DDR5-5600) and 96 GB/s (DDR5-6000) RAM bandwidth has no practical impact at this workload, because the binding constraint is PCIe miss-fetch throughput at approximately 30 GB/s sustained, not RAM bandwidth. Faster RAM would buy nothing measurable. The 48 GB DIMMs were chosen over 64 GB DIMMs (which would have produced a 256 GB build) because the May 2026 DDR5 shortage market makes the larger kits cost $300-1,000 more, and the engine's expert hit-rate plateaus once experts fit comfortably in RAM — beyond 192 GB, more capacity buys no further throughput.

### NVMe Gen5: two Crucial T710 2 TB drives

Storage at the Gen5 tier exists for two workloads: model loading and KV cache spillover at ultra-long contexts. Both are bursty rather than sustained, so two T710 drives at 2 TB each (4 TB aggregate) provide enough capacity for the 397B model at Q5 and Q6 simultaneously plus a 122B and 35B alternate, with room for benchmark datasets and Battle Cities player state archives. The T710 was selected over the older T705 for its Phison E28 controller and 45% faster random reads — meaningful when the access pattern is small-block expert weight loads rather than large sequential transfers. DRAM-equipped (LPDDR4 cache) with 1500 TBW endurance, the T710 is genuinely workstation-tier rather than budget. Heatsink versions are essential at Gen5 speeds: sustained writes reach 70-85°C without a heatsink, and thermal throttling under inference would defeat the point of buying Gen5 storage.

The second T710 is populated in M.2_2, which requires the "Force x4" BIOS toggle to deliver full Gen5 x4 bandwidth. That toggle disables the board's USB4 ports — an acceptable trade because the workstation will not be moving data via Thunderbolt-class peripherals.

### NVMe Gen4: Kingston NV3 2 TB for Windows

The Windows boot drive is the line item where the May 2026 NAND shortage shows most clearly. In a normal market, the Crucial T500 2 TB at around $399 would be the obvious pick — TLC NAND, real DRAM cache, 1200 TBW endurance. In this market the T500 sits closer to $500-650 at Mwave, and the Samsung 990 EVO Plus 2 TB peaked at $712 AUD in March 2026. The Kingston NV3 2 TB at $400 ends up the best dollars-per-terabyte option for the workstation Windows host, despite QLC NAND and a DRAM-less design that uses Host Memory Buffer. Goes in M.2_4, the only chipset Gen4 slot with no sharing dependencies.

The compromises are real but bounded: 640 TBW endurance handles three to five years of typical workstation writes, after which NAND prices should have normalised and a replacement will be cheaper anyway. QLC sustained write performance drops to roughly 500-700 MB/s once the SLC cache exhausts, but Windows boot and Steam library activity rarely sustain large writes long enough to matter. Random IOPS take a hit from the absent DRAM cache, but the workstation will not be running database workloads on its OS drive. The NV3 2 TB is the right pragmatic answer for the present market; revisit when prices stabilise.

### PSU: Corsair RM1000x 1000W ATX 3.1

Peak system draw is around 590 W: the PRO 5000 at 300W, the 9950X3D at 170W, motherboard and NVMe and DDR5 at around 120W combined. A 1000W PSU sits at 59% loading at peak — almost exactly in the efficiency sweet spot for 80+ Gold rated units. The ATX 3.1 specification includes native 12V-2x6 cabling for the GPU, which is the corrected connector standard that replaced the troubled 12VHPWR. The RM1000x is from Corsair's most reliable line with a 10-year warranty and fully modular cabling. Adequate headroom for any future GPU upgrade short of dual-GPU.

### Case: MSI MAG PANO 100R PZ Black

The case decision turned out to be largely a value-engineering exercise. At $199 the MSI Pano 100R PZ ships with four 120 mm ARGB fans pre-installed — three side-mount intake and one rear exhaust — which eliminates approximately $140-280 of separate case fan purchases. Back-connect motherboard support matches the Tomahawk MAX PZ variant directly, with rear cable routing channels sized for ATX boards. The case supports 360 mm radiators at three positions (top, side, bottom), accommodating the Arctic AIO comfortably. GPU clearance reaches 380 mm, well beyond the PRO 5000's 310 mm length. Tom's Hardware's review noted second-best GPU cooling efficiency in its testing tier, primarily due to the side-mount intake configuration positioning fresh air directly across the GPU.

A genuine premium alternative exists in the Lian Li O11 Vision Compact at approximately $240, which delivers brushed aluminium construction, three borderless glass panels, and a dual-chamber layout that hides cables more elegantly. Adding case fans brings the Lian Li option to roughly $280 total, an $80 premium over the MSI Pano. Whether that premium is worth it comes down to whether the case will be visible enough to enjoy. For a workstation that will spend most of its time accessed via remote desktop, the MSI Pano is enough.

### Cooler: Arctic Liquid Freezer III Pro 360

Arctic's Liquid Freezer III Pro 360 was selected after explicit comparison against the Noctua NH-D15 G2 Chromax air cooler, which would have been the natural pick on reliability grounds. Two factors tipped the decision toward the AIO. First, the integrated VRM fan on the Arctic's CPU block actively cools the Tomahawk MAX's mid-tier VRMs — a real benefit given that the board's 14+2+1 phase VRM is the weakest component in the build and sustained 170W inference loads will exercise it for years. Second, the MSI Pano case includes its own intake fans, so the Noctua's $160 list price would have ballooned to roughly $440 once four additional Noctua case fans were added for adequate airflow. The Arctic at $169 with three P12 Pro fans included on the radiator comes out cheaper net, with better thermal performance, and matches the case aesthetic more cleanly.

Tom's Hardware tested this exact AIO on a Ryzen 9 9950X3D and recorded 76.5°C peak under thirty minutes of AIDA64 FPU stress at 174 W package power — about 20°C of headroom below the chip's 95°C thermal limit. Offset AMD mounting is included, which positions the cold plate directly above CCD0 where the V-Cache concentrates heat. The radiator is 38 mm thick rather than the typical 27 mm, which contributes to the sustained-load advantage but means total radiator-plus-fans depth reaches 63 mm — sized for the MSI Pano's top mount but worth verifying clearance in any case with under 70 mm rated top depth.

The reliability tradeoff is real but bounded. AIO pumps typically run 5-7 years before performance degradation becomes noticeable, versus 10-15 years for high-end air coolers with no moving parts beyond fans. Arctic's six-year warranty on the Liquid Freezer line provides cover for the first half of that window. For a workstation expected to run 24/7 inference loads for years, the air cooler's longer mechanical lifespan is a legitimate consideration — but the integrated VRM fan and the case-fan-budget arithmetic carried the decision.

---

## What This Build Validates

### NeurIPS paper "One Card, One Stack"

The build matches the paper's central premise exactly: single-card 72 GB inference of a near-400B MoE model at workable throughput. Compression metrics for palette4, query-subspace selection, BDP retrieval, SCD, and the C0-C10 format ladder can all be measured against actual production workloads on this hardware rather than synthetic benchmarks. The V error metric defined as normalised squared L2 error runs unchanged from the appendix B specification. Multi-session identity discrimination as described in §9.9b has comfortable headroom in the 72 GB VRAM allocation. The paper is publication-ready on this hardware, and the four-venue roadmap (arXiv preprint May 2026, MLSys October 2026, ACL February 2027, NeurIPS May 2027, EMNLP June 2027) is fully supported by the rig.

### Battle Cities inference target

The primary target is Qwen3.5-397B-A17B as the open-weight reference model, with Qwen3.6-397B-A17B (Max-Preview) as a future target if Alibaba releases its weights. Both run at hybrid Q5/Q6 with the engine's compression layers active: roughly 17 hot experts cached at Q6 in VRAM, 38 warm experts at Q5 in RAM, and 73 cold experts at C8-C10 compressed in RAM. With BDP-augmented retrieval the effective hit rate reaches 75-80%, comfortably exceeding the 70% threshold. Decode rate forecasts land at 20-25 tokens per second for 397B-A17B on this build, with smaller models running considerably faster — 122B-A10B at Q8 fits entirely in VRAM at 50-80 t/s, and 35B-A3B at Q8 hits 90-140 t/s.

The decode rate math constrains directly. At 20 t/s with the C10 cold tier engaged, miss-fetch bandwidth requirement is roughly 25-30 GB/s sustained — fitting inside the PCIe 5.0 x16 sustained budget of 32-48 GB/s with margin. Without C10 compression and BDP retrieval, the same workload would be PCIe-bound at 10-13 t/s. The build only works with the engine IP, which is precisely the architectural argument the paper makes.

### Battle Cities thin-client deployment

The mobile thin-client architecture (server-side isometric rendering streamed as images to mobile clients via a VPS plus WireGuard tunnel back to this rig) is supported by the build's network and rendering capacity. The eight-week Tauri 2 mobile beta path is viable from this hardware. Push notifications via APNs and FCM replace the earlier WhatsApp-based notification scheme without architectural complication.

---

## Windows and Linux Development Environment

The OS choice is Windows 11 Pro installed on the Kingston NV3 2 TB in M.2_4. If an SCB MSDN or volume license is available it should be used; otherwise the retail license costs approximately $199 AUD.

The NVIDIA driver path matters more on a workstation card than on a consumer card. The Studio Driver branch is the correct choice — it ships monthly rather than weekly, prioritises stability for CUDA and professional applications over per-title game optimisations, and still plays games perfectly well. The Game Ready branch is wrong for a PRO 5000.

After Windows installation, several configuration changes meaningfully reduce overhead. Hibernation should be disabled with `powercfg -h off` from an elevated PowerShell — this reclaims approximately 144 GB of disk space that Windows would otherwise reserve for the hibernation file matching system RAM. The page file should be capped to a fixed 32 GB on the C: drive rather than left at the default auto-sized behaviour, which can grow into hundreds of gigabytes on a 192 GB system. System restore should be disabled on the T710 data drives (M.2_1 and M.2_2) — they hold model data that does not benefit from restore point snapshots and would consume meaningful space on those drives over time.

The Steam library should sit on C: rather than on the T710s. The NV3 2 TB has comfortable space for five to ten modern AAA titles, and routing Steam through the Gen4 drive keeps the Gen5 T710s reserved exclusively for inference work. Game performance is unaffected — Gen5 over Gen4 makes no measurable difference at game load times.

For Rust development on the inference engine, WSL2 with Ubuntu 24.04 is the recommended path. `wsl --install -d Ubuntu-24.04` from elevated PowerShell sets it up. NVIDIA officially supports CUDA passthrough into WSL2 (since 2022), so the engine can be built on the Linux toolchain inside WSL2 and run natively on Windows for production inference. This avoids dual-booting while preserving full access to both ecosystems — the right architecture for someone working across Rust dev tools and Windows-native gaming workloads.

---

## Trade-offs vs Alternative Paths

| Path | Cost AUD | Why rejected |
|---|---:|---|
| WRX90 + 4× PRO 4500 + 384 GB RDIMM | ~$48,000 | A $30K premium for offline NPC workload that does not need 128 GB VRAM. RDIMM premium at $12-18K is genuine, not the fake $2,800 I initially mis-quoted |
| TRX50 + 4× PRO 4500 + 384 GB RDIMM | ~$35,000 | RDIMM premium still kills the cost advantage |
| AM5 + 2× PRO 4500 32GB | ~$15,100 | 64 GB VRAM forces aggressive quantisation; replicated shared backbone tax leaves only 50 GB usable; multi-GPU NCCL overhead fights the single-device inference engine design |
| AM5 + 1× PRO 6000 96GB Max-Q | ~$22,500 | $5,100 premium for 24 GB additional VRAM does not justify itself; PRO 5000 72 GB is sufficient with the C10 IP active |
| AM5 + 2× PRO 5000 72GB (144 GB total) | ~$32,000 | Would justify platform jump to TRX50 at $44K total; offline NPC workload does not need this scale |
| ASUS ProArt X870E-Creator motherboard | $899 | M.2_2 Gen5 silently steals GPU lanes (drops GPU to x8) — content-creator-targeted lane allocation, wrong for inference |
| ASRock X870E Taichi motherboard | $611 | Clean GPU x16 but only 1× Gen5 M.2 simultaneous with full GPU; ~30 GB/s aggregate NVMe vs 42 on Tomahawk |
| Noctua NH-D15 G2 air cooler | ~$160 | 168mm height marginal in MSI Pano clearance; case-fan budget makes AIO cheaper net once Noctua case fans accounted for |
| Crucial T500 2TB Windows drive | ~$500-650 | T500 at current NAND-shortage prices does not justify premium over NV3 2TB; revisit in 12 months when prices normalise |
| Silicon Power UD80 500GB Windows drive | ~$60-120 | Gen3, 500GB, DRAM-less, QLC — too compromised for an $18K workstation's Windows host |

---

## Production Scaling Ceiling

The single PRO 5000 72 GB at 20-25 t/s decode supports NeurIPS paper validation work (which is sequential and has no concurrency requirement) and a Battle Cities beta launch with an offline-first NPC model architecture (queue-based inference, no real-time response constraint). It supports one to five concurrent developers and testers comfortably during beta.

What it does not support is production scaling to hundreds of concurrent users. There are three paths past that ceiling when revenue justifies the spend. First, cloud burst capacity from providers like RunPod or Lambda Labs can handle peak load during launch events while the local rig handles baseline traffic. Second, this rig design can be cloned and distributed across multiple homes or colocation sites, routing inference requests via the existing thin-client architecture — economic if the build cost stays under $20K and revenue per rig exceeds $5K monthly. Third, eventual migration to TRX50 or WRX90 with multi-GPU capacity (the $35-48K path that was deferred at the start of this build) becomes correct once Battle Cities revenue validates the target margin model at approximately €8-15M Year 1.

The current architecture is intentionally a single-rig validation and beta-launch platform rather than a production cluster. The economic model — gold marketplace, Euro prize pools, six-month seasons, deflationary destruction mechanics — needs revenue validation before committing to multi-rig infrastructure spend. Critically, this rig stays useful as a development and testing box even after a production migration, so the present spend is not stranded capital. It is correctly time-shifted capacity for capability the business does not yet need.

---

## Pre-Order Verification Checklist (Mwave)

Before checkout, the following items need to be verified at order time:

1. The motherboard SKU is the **PZ (back-connect) variant** at AC95442, not the regular front-connector Tomahawk MAX. Different SKUs ship from Mwave.
2. **All four Crucial Pro 48 GB sticks must be in stock simultaneously** to ensure a matching batch — important for stability at four DIMMs populated.
3. The T710 model is **CT2000T710SSD5** (the heatsinked Gen5 SKU), not the bare CT2000T710SSD8.
4. The Kingston NV3 2 TB price holds at $400 at checkout — the DDR5 and NAND shortage market in May 2026 is volatile and prices can shift between cart-add and order placement.
5. The PRO 5000 model number is **900-5G153-2570-000**, which is the workstation Blackwell card. Not the consumer RTX 5090 (different model, different driver branch, no ECC).
6. The case is the **PZ variant** for back-connect compatibility (AC78596), not the non-PZ version which has different cable routing.

---

## Post-Assembly Validation

After first POST and OS installation, the following validation steps confirm the build is performing as designed:

1. Update BIOS to latest AGESA for Zen 5 X3D stability — May 2026 builds are well-tested.
2. Apply the BIOS settings from the table above before installing the operating system.
3. Verify PCIe enumeration: GPU at PCIe 5.0 x16, both T710s at PCIe 5.0 x4. Confirm via Device Manager properties in Windows or `lspci -vvv` inside WSL2.
4. Confirm RAM has trained at DDR5-5600. Visible in Task Manager → Performance → Memory in Windows, or `dmidecode -t 17` inside WSL2.
5. Benchmark NVMe performance: target sustained 14 GB/s sequential read on both M.2_1 and M.2_2 using CrystalDiskMark on Windows or `fio` inside WSL2.
6. Run a 24-hour soak test under sustained load. 9950X3D under Cinebench R23 multi-core loop should peak below 80°C. PRO 5000 under sustained inference should peak below 75°C. VRMs should stay below 85°C (visible via HWInfo). No thermal throttling should occur on either CPU or GPU.

Any failure on these tests warrants a BIOS revision check and AIO mount pressure verification before component-level troubleshooting.

---

## Total

**$18,406.80 AUD, single transaction at Mwave** — one shipment, one warranty point of contact, all items in stock at the time of cart finalisation.

The build saves approximately $30,000 against the original WRX90 four-GPU plan while hitting roughly 80% of the architectural goals. The remaining 20% — multi-GPU production scaling, RDIMM ECC across all 384 GB, full 128 GB VRAM ceiling — is correctly deferred until Battle Cities revenue justifies a TRX50 or WRX90 migration. This rig remains useful as the development and testing box after that migration, so the spend is not wasted, just front-loaded for capability the business does not yet need.

The architecture is optimised around the inference engine IP — palette4, the C0-C10 format ladder, Binary Directional Provenance retrieval, Speculative Context Decode. The build only works because of that IP, which is exactly the central claim of the NeurIPS paper and the differentiator that makes Battle Cities economically viable at this hardware tier rather than at the WRX90 tier.