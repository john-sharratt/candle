# Battle Cities Production Workstation — Final Build

**Sydney AUD, May 2026.** Architecture target: 4× RTX PRO 4500 Blackwell with 16× PCIe 5.0 NVMe array sized to saturate aggregate GPU ingress. AU local sourcing throughout, all components new with full AU warranty.

**Total: $47,713 AUD / ~$30,800 USD**

---

## Master Parts Table (Final Build at Completion)

| Component | SKU / Spec | Vendor | Unit AUD | Qty | Subtotal | Install |
|---|---|---|---:|---:|---:|---:|
| **GPU** | NVIDIA RTX PRO 4500 Blackwell, 32 GB GDDR7 ECC, PCIe 5.0 x16, 200W | Scorptec | 4,999 | 4 | 19,996 | 1, 2, 4, 4 |
| **CPU** | AMD Threadripper PRO 9955WX, 16C/32T Zen 5, 5.4 GHz boost, 128 PCIe 5.0 lanes | Scorptec | 2,599 | 1 | 2,599 | 1 |
| **Motherboard** | ASUS Pro WS WRX90E-SAGE SE (new, 3-yr AU warranty) | Scorptec | 2,479 | 1 | 2,479 | 1 |
| **RAM** | Kingston FURY Renegade Pro 128 GB (4×32 GB) DDR5-6000 ECC RDIMM kit (KF560R32RBEK4-128) | Scorptec | 5,999 | 2 | 11,998 | 1, 3 |
| **NVMe inference array** | Crucial T705 2 TB Gen5 NVMe, no heatsink (CT2000T705SSD3) | Scorptec | 479 | 16 | 7,664 | 4 / 4 / 8 |
| **OS drive** | Kingston A400 480 GB SATA III SSD | Scorptec | 125 | 1 | 125 | 1 |
| **PSU** | FSP MEGA TI 1650W 80+ Titanium, ATX 3.1, PCIe 5.1, fully modular, 10-yr warranty | Scorptec | 599 | 1 | 599 | 1 |
| **CPU cooler** | Noctua NH-U14S TR5-SP6 (350W rated air, sTR5-specific) | Scorptec | 270 | 1 | 270 | 1 |
| **Fans — 120mm** | Noctua NF-A12x25 PWM chromax.black.swap (bottom intake) | Scorptec | 70 | 3 | 210 | 1 |
| **Fans — 140mm dual** | Noctua NF-A14x25 G2 PWM Sx2-PP chromax dual fan set (front intake, 2 fans) | Scorptec | 149 | 1 | 149 | 1 |
| **Fans — 140mm single** | Noctua NF-A14x25 G2 PWM chromax black (rear exhaust, 1 fan) | Scorptec | 75 | 1 | 75 | 1 |
| **Chassis** | Phanteks Enthoo Pro 2 Server Edition Tempered Glass (PH-ES620PTG_BK02), 11 PCI slots, SSI-EEB | PC Case Gear | 289 | 1 | 289 | 1 |
| **Bifurcation cards** | ASUS Hyper M.2 x16 Gen5 (4× M.2 per card) | Scorptec / Mwave | 240 | 3 | 720 | 2×1, 3×2 |
| **PCIe risers** | LinkUp Ultra PCIe 5.0 x16, 300mm | Mwave / direct | 180 | 3 | 540 | 2×1, 3×2 |
| | | | | **Total** | **47,713** | |

---

## Installment Breakdown

| Install | Components added | Stage AUD | Cumulative |
|---:|---|---:|---:|
| **1** | Platform (mobo, CPU, RAM kit, case, PSU, cooling) + GPU #1 + 4 NVMe + OS drive | **19,709** | 19,709 |
| **2** | GPU #2 + bif card #1 + 4 NVMe + riser | **7,335** | 27,044 |
| **3** | RAM kit #2 (→ 256 GB / 8-ch) + bif cards #2 & #3 + 8 NVMe + 2 risers | **10,671** | 37,715 |
| **4** | GPU #3 + GPU #4 (architecture complete) | **9,998** | **47,713** |

---

## Install 1 Detail

### Scorptec order ($19,420 confirmed cart)

| Line | Unit AUD | Qty | Subtotal |
|---|---:|---:|---:|
| ASUS Pro WS WRX90E-SAGE SE | 2,479 | 1 | 2,479 |
| AMD Threadripper PRO 9955WX | 2,599 | 1 | 2,599 |
| NVIDIA RTX PRO 4500 Blackwell (GPU #1) | 4,999 | 1 | 4,999 |
| Kingston FURY Renegade Pro 128GB DDR5-6000 ECC kit | 5,999 | 1 | 5,999 |
| Crucial T705 2TB | 479 | 4 | 1,916 |
| Kingston A400 480GB SATA (OS drive) | 125 | 1 | 125 |
| FSP MEGA TI 1650W (was $699, on special at $599) | 599 | 1 | 599 |
| Noctua NH-U14S TR5-SP6 CPU cooler | 270 | 1 | 270 |
| Noctua NF-A12x25 PWM chromax (bottom intake) | 70 | 3 | 210 |
| Noctua NF-A14x25 G2 PWM dual fan set (front intake) | 149 | 1 | 149 |
| Noctua NF-A14x25 G2 PWM single (rear exhaust) | 75 | 1 | 75 |
| **Scorptec subtotal** | | | **19,420** |

Total Scorptec savings: $104 (PSU + fan discount). Free delivery applies.

### Separate vendor

| Item | Vendor | AUD |
|---|---|---:|
| Phanteks Enthoo Pro 2 Server Tempered Glass | PC Case Gear | 289 |
| **Install 1 total** | | **19,709** |

---

## Architecture Progression

| Install | GPUs | VRAM | RAM | Channels | Domain 2 BW | NVMe drives | NVMe BW | NVMe→GPU saturation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 32 GB | 128 GB | 4 | 220 GB/s | 4 | 58 GB/s | 97% (vs 1 GPU) |
| 2 | 2 | 64 GB | 128 GB | 4 | 220 GB/s | 8 | 116 GB/s | 97% (vs 2 GPUs) |
| 3 | 2 | 64 GB | **256 GB** | **8** | **384 GB/s** | **16** | **232 GB/s** | 193% (substrate ready) |
| 4 | 4 | 128 GB | 256 GB | 8 | 384 GB/s | 16 | 232 GB/s | **97% (vs 4 GPUs, design target)** |

Domain 2 BW at 8-channel DDR5-6000 = 384 GB/s — better than the 358 GB/s projected at DDR5-5600. Kingston FURY Renegade Pro at DDR5-6000 outperforms the alternatives.

---

## Final Architecture at Completion

| Metric | Value |
|---|---:|
| Total GPU VRAM | 128 GB (4× 32 GB ECC GDDR7) |
| Aggregate GPU memory bandwidth | 3,136 GB/s |
| Aggregate PCIe 5.0 GPU ingress (Domain 1) | 240 GB/s |
| Aggregate NVMe sustained read | 232 GB/s |
| **NVMe saturation of GPU ingress** | **97%** |
| System RAM | 256 GB DDR5-6000 ECC RDIMM (2× Kingston FURY Renegade Pro 4×32 GB kits) |
| RAM bandwidth (Domain 2) | 384 GB/s @ 8-channel |
| NVMe storage capacity | 32 TB (inference) + 480 GB (OS) |
| CPU | TR PRO 9955WX (16C / 32T Zen 5) @ 5.4 GHz boost |
| Peak system power | ~1,300W |
| Single 10A 240V circuit utilisation | 54% peak |
| Warm tier ratio (RAM:VRAM) | 2:1 |

---

## Slot Allocation at Completion

| PCIe slot | Device | Mount |
|---|---|---|
| 1 | RTX PRO 4500 #1 (Install 1) | Direct |
| 2 | ASUS Hyper M.2 #1 → 4× T705 (Install 2) | Riser |
| 3 | RTX PRO 4500 #2 (Install 2) | Direct |
| 4 | ASUS Hyper M.2 #2 → 4× T705 (Install 3) | Riser |
| 5 | RTX PRO 4500 #3 (Install 4) | Direct |
| 6 | ASUS Hyper M.2 #3 → 4× T705 (Install 3) | Riser |
| 7 | RTX PRO 4500 #4 (Install 4) | Direct |
| On-board M.2 ×4 | 4× T705 (Install 1) | Direct |
| 2.5" bay | Kingston A400 480GB SATA OS drive (Install 1) | SATA cable + power from PSU |

Phanteks Enthoo Pro 2 Server has 11 PCI slots — 8 for 4 dual-slot GPUs + 3 for bifurcation cards via risers. Exact fit. OS drive on SATA mount frees all M.2 capacity for inference array.

---

## Power & Cable Plan

### Motherboard power inputs (ASUS Pro WS WRX90E-SAGE SE)

| Connector | Type | Purpose | Required |
|---|---|---|---|
| Main ATX | 24-pin | Motherboard primary power | Yes |
| EATX12V_1 / EATX12V_2 | 2× 8-pin EPS | CPU primary +12V (ProCool II) | Yes |
| PCIE_CPU_12V_1 / 2 | 2× 8-pin PCIe form | Supplementary CPU power | Yes (350W TDP) |
| PCIE_8P_PWR_1 / 2 | 2× 8-pin PCIe | Supplementary PCIe slot power | Yes (multi-GPU) |

### FSP MEGA TI 1650W includes in retail box

- 1× 24-pin ATX
- 2× EPS 8-pin (CPU)
- 8× PCIe 8-pin (across 4 daisy-chained cables)
- **2× 12V-2x6 native cables (600W each)**
- Multiple SATA / Molex

### Cable allocation per installment

| Install | New cables to connect | All from FSP box? |
|---:|---|:---:|
| 1 | 24-pin + 2× EPS + 4× PCIe 8-pin (motherboard supplementary) + 1× 12V-2x6 (GPU #1) | ✓ |
| 2 | 1× 12V-2x6 (GPU #2 — second FSP cable from box) | ✓ |
| 3 | None (bif cards draw from PCIe slot) | ✓ |
| 4 | 2× 12VHPWR for GPUs #3 & #4 → use PNY-included 2×8-pin → 12VHPWR adapters | ✓ (PNY GPU box) |

**Zero additional cables to buy for the entire 4-installment build.** Everything ships with the PSU, motherboard, and GPU boxes.

**Optional polish for Install 4**: 2× FSP-branded native 12V-2x6 cables (~$60-100 AUD) for cleaner connections on GPUs #3 & #4 instead of PNY adapters. Source from FSP direct or Mwave when ordering Install 4.

---

## Why This Sequencing

**Install 1 stands up the full platform** with substantial RAM (128 GB 4-channel) and full thermal/power infrastructure. Single-GPU dev work has comfortable headroom from day one. No platform upgrades across installments — only adds.

**Install 3 is the validation gate.** With 2 GPUs already installed, this install brings the substrate to full architectural spec: 256 GB at 8-channel and the complete 16-drive NVMe array. Two GPUs stress-test the substrate against full memory bandwidth and full NVMe ingress. If anything fails — DIMM training, bifurcation enumeration, riser signal integrity, RAID 0 stability — it surfaces here, before GPUs #3 and #4 are committed.

**Install 4 commits the final $9,998** only after the substrate is proven. Risk-managed final spend.

**RAM uses two identical Kingston FURY Renegade Pro 4×32 GB kits.** Same SKU at both installs ensures clean 8-channel training. Each kit is one product purchase, dodging Scorptec's 2-per-household limits.

**OS on SATA freed up all M.2 capacity for inference array.** Kingston A400 480GB at $125 is the simplest, cheapest OS path — SATA cable, 2.5" mount, no special adapters.

---

## Noise & Thermals

**Realistic noise profile** with the Noctua fan setup (3× A12x25 + 3× A14x25 G2):

| Workload | dB | Comparison |
|---|---:|---|
| Idle / dev work | 25-28 | Whisper, library quiet |
| 1-GPU inference | 32-37 | Quiet conversation |
| 2-GPU inference | 40-45 | Office ambient |
| 4-GPU full load | 48-53 | Noticeable but not loud |

GPU blower fans (not case fans) dominate noise under multi-GPU load. Case fans run silent at idle thanks to PWM curves staying below 40% until load.

FSP MEGA TI 1650W has semi-fanless Zero-RPM mode below 40% load — silent PSU during idle and Install 1-2 dev work.

**For genuine silence at the desk**: relocate the workstation to a closet/utility space and access via 10 GbE LAN. ASUS Pro WS WRX90E-SAGE SE has dual 10 GbE built in. Add a 10m Cat6A cable (~$80) and the workstation can be physically isolated from your work area while remaining fully accessible.

---

## Validation at Each Stage

**After Install 1:**
- Platform POST, BIOS up to date for sTR5 + 9955WX
- 4-channel RAM training stable at DDR5-6000 (document actual trained speed)
- Single-GPU GDS path: CUDA 12.2+ / Linux 6.4+ DMA-BUF works with T705
- 4-drive RAID 0 benchmark: target 50 GB/s aggregate read
- Inference engine boot on 1 GPU; run small model (Qwen3-30B baseline)
- OS drive runs cleanly from Kingston A400 — boot time, package install, dev tool perf

**After Install 2:**
- 2-GPU layer pipeline functional
- 8-drive RAID 0: target 100 GB/s
- Bifurcation card #1 enumerates all 4 drives at PCIe 5.0 x4
- Layer pipeline efficiency: target ≥ 70% sustained utilisation per card

**After Install 3 (the critical validation gate):**
- **8-channel RAM training stable at DDR5-6000** with all 8 Kingston FURY Renegade Pro sticks populated (may step down to 5600 or 5200 with full population — document actual trained speed)
- Domain 2 BW benchmark: target ≥ 340 GB/s (90% of 384 GB/s ideal)
- 16-drive RAID 0: target 200+ GB/s aggregate read
- Bifurcation cards #2 and #3 enumerate all drives correctly
- All 3 PCIe risers signal integrity test at sustained PCIe 5.0 x16
- 2-GPU concurrent ingress with provenance scan: confirm Domains 1 and 2 don't contend on IO die
- 24-hour soak at 2-GPU sustained load — confirm thermals and stability

**After Install 4:**
- 4-GPU layer pipeline with 1T MoE: validate Markov staging accuracy
- 24-hour soak at sustained 1.3 kW: thermals, no throttling
- Single 10A circuit holds under transient spikes
- Full production validation

---

## Risks / Open Items

- **8× Kingston FURY Renegade Pro sticks at DDR5-6000 may train down on the ASUS Pro WS WRX90E-SAGE SE.** Kingston FURY Renegade Pro is on ASUS QVL but 8-stick 1DPC training at 6000 MT/s is variable. Expect possible step-down to 5600 (Domain 2 BW = 358 GB/s — still adequate) or 5200 (333 GB/s — still adequate). Functional risk only, not capacity-fatal.
- **T705 stock fluctuates.** Scorptec is $479 (limited to 5 per household per transaction). Lock pricing per installment; if Scorptec runs low, PCCG and Mwave both stock the same SKU.
- **T705 household limit (5/transaction) needs management.** Install 1 uses 4 (OK). Install 2 uses 4 (OK). Install 3 uses 8 (need 2 transactions or split across vendors). Plan ahead.
- **PCIe riser signal integrity at sustained max bandwidth.** Test each riser independently after install. LinkUp Ultra 5.0 cables rated to spec but a stack of 3 risers is unusual — confirm no negotiation drops under 24-hour load.
- **Kingston FURY Renegade Pro SKU continuity.** Confirm Scorptec stock at Install 3 — same SKU (KF560R32RBEK4-128) as Install 1 for clean 8-channel training.
- **A400 endurance** rated 160 TBW. OS-only workload typically consumes 10-20 TBW per 5-year period. Headroom adequate but the A400 is DRAM-less — don't use it as cache for Docker/build artifacts. Keep it as pure boot + OS.
- **Refurb motherboard path abandoned** — paid the $580 premium for new-in-box Scorptec with 3-year AU warranty. Right call given architectural criticality of the board.

---

## Final Numbers

- **4 installments totalling AUD $47,713 / USD ~$30,800**
- Install 1: **$19,709** (41% — full platform + first GPU + 128 GB RAM + 4 NVMe + OS + PSU + case)
- Install 2: **$7,335** (15% — second GPU + first bifurcation card + 4 NVMe)
- Install 3: **$10,671** (22% — RAM to 256 GB / 8-ch + NVMe to full 16 drives — substrate validation gate)
- Install 4: **$9,998** (21% — final 2 GPUs — architecture complete)

**At completion:**
- 128 GB VRAM / 256 GB RAM (8-channel @ DDR5-6000) / 32 TB Gen5 NVMe inference + 480 GB SATA OS
- 97% NVMe-to-GPU ingress saturation
- AU local warranty on every major component
- Single 10A 240V circuit at 54% peak utilisation
- **Payback vs Opus 4.7 retail output: 0.6-1.4 months at production utilisation**

---

## Ready to Ship — Install 1

**Scorptec cart locked at $19,420.** Confirmed line items match plan. Free delivery applies. Single transaction, all major components.

**Remaining Install 1 spend:**
- Phanteks Enthoo Pro 2 Server Tempered Glass — order from PC Case Gear ($289)

**Total Install 1 spend: $19,709 across 2 vendors.** Then build.