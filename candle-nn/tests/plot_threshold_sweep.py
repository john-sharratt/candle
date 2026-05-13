#!/usr/bin/env python3
"""
Plot KV-cache threshold sweep data for research paper.

Reads kv_threshold_sweep.csv and produces:
  1. Per-level K vs V BPE curves showing candidate-list asymmetry
  2. Combined CR contour/heatmap with separate K/V threshold axes
  3. SNR vs threshold for both K and V (C9 candidates)
  4. Total CR curves when K and V thresholds are varied independently

Usage:
    python plot_threshold_sweep.py
"""

import csv
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "kv_threshold_sweep.csv")
OUT_DIR = SCRIPT_DIR


def parse_csv(path):
    """Parse the multi-section CSV into separate data structures."""
    k_sweep = []
    v_sweep = []
    grid_k_thrs = []
    grid_v_thrs = []
    grid_cr = []
    # Per-level sweeps: {level: [(threshold, bpe, snr, p95), ...]}
    perlevel_k = defaultdict(list)
    perlevel_v = defaultdict(list)

    with open(path, 'r') as f:
        section = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('# SECTION:'):
                if 'PER_LEVEL_K_SWEEP' in line:
                    section = 'perlevel_k'
                elif 'PER_LEVEL_V_SWEEP' in line:
                    section = 'perlevel_v'
                elif 'K_SWEEP' in line:
                    section = 'k_sweep'
                elif 'V_SWEEP' in line:
                    section = 'v_sweep'
                elif 'COMBINED_CR_GRID' in line:
                    section = 'grid'
                continue
            if line.startswith('#'):
                continue

            if section == 'k_sweep':
                if line.startswith('k_threshold'):
                    continue
                parts = line.split(',')
                if len(parts) >= 6:
                    k_sweep.append({
                        'threshold': float(parts[0]),
                        'bpe': float(parts[1]),
                        'snr_db': float(parts[2]),
                        'nrmse': float(parts[3]),
                        'cos_p95': float(parts[4]),
                        'cr': float(parts[5]),
                    })

            elif section == 'v_sweep':
                if line.startswith('v_threshold'):
                    continue
                parts = line.split(',')
                if len(parts) >= 6:
                    v_sweep.append({
                        'threshold': float(parts[0]),
                        'bpe': float(parts[1]),
                        'snr_db': float(parts[2]),
                        'nrmse': float(parts[3]),
                        'cos_p95': float(parts[4]),
                        'cr': float(parts[5]),
                    })

            elif section == 'grid':
                if line.startswith('k_thr'):
                    parts = line.split(',')
                    grid_v_thrs = [float(x) for x in parts[1:]]
                    continue
                if line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) > 1:
                    grid_k_thrs.append(float(parts[0]))
                    grid_cr.append([float(x) for x in parts[1:]])

            elif section == 'perlevel_k':
                if line.startswith('level'):
                    continue
                parts = line.split(',')
                if len(parts) >= 5:
                    lvl = int(parts[0])
                    perlevel_k[lvl].append({
                        'threshold': float(parts[1]),
                        'bpe': float(parts[2]),
                        'snr_db': float(parts[3]),
                        'cos_p95': float(parts[4]),
                    })

            elif section == 'perlevel_v':
                if line.startswith('level'):
                    continue
                parts = line.split(',')
                if len(parts) >= 5:
                    lvl = int(parts[0])
                    perlevel_v[lvl].append({
                        'threshold': float(parts[1]),
                        'bpe': float(parts[2]),
                        'snr_db': float(parts[3]),
                        'cos_p95': float(parts[4]),
                    })

    return (k_sweep, v_sweep, grid_k_thrs, grid_v_thrs, np.array(grid_cr),
            dict(perlevel_k), dict(perlevel_v))


def find_step_changes(data, key='bpe', min_delta=0.3):
    """Find threshold values where BPE drops significantly."""
    steps = []
    prev = data[0][key]
    for d in data[1:]:
        if prev - d[key] > min_delta:
            steps.append(d)
            prev = d[key]
    return steps


def main():
    (k_sweep, v_sweep, grid_k_thrs, grid_v_thrs, grid_cr,
     perlevel_k, perlevel_v) = parse_csv(CSV_PATH)

    k_thr = np.array([d['threshold'] for d in k_sweep])
    k_bpe = np.array([d['bpe'] for d in k_sweep])
    k_snr = np.array([d['snr_db'] for d in k_sweep])
    k_cr = np.array([d['cr'] for d in k_sweep])

    v_thr = np.array([d['threshold'] for d in v_sweep])
    v_bpe = np.array([d['bpe'] for d in v_sweep])
    v_snr = np.array([d['snr_db'] for d in v_sweep])
    v_cr = np.array([d['cr'] for d in v_sweep])

    # Color maps for levels
    level_colors = plt.cm.viridis(np.linspace(0.1, 0.95, 10))

    # ── Figure 1: Per-level K vs V asymmetry (4-panel) ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('KV Cache Adaptive Quantization: K/V Asymmetry by Compression Level',
                 fontsize=14, fontweight='bold')

    # Panel (a): Per-level K BPE curves
    ax = axes[0, 0]
    for lvl in range(10):
        if lvl not in perlevel_k:
            continue
        thr = np.array([d['threshold'] for d in perlevel_k[lvl]])
        bpe = np.array([d['bpe'] for d in perlevel_k[lvl]])
        ax.semilogx(thr, bpe, color=level_colors[lvl], linewidth=1.5,
                     label=f'C{lvl}', alpha=0.85)
    ax.set_xlabel('Cosine Distance Threshold (log scale)')
    ax.set_ylabel('Bits per Element (BPE)')
    ax.set_title('(a) K-side BPE by Level (different candidate lists)')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2, 16.5)
    ax.axhline(y=8.5, color='gray', linestyle=':', alpha=0.4)
    ax.text(1e-6, 8.7, 'Q8_0 (8.5 bpe)', fontsize=7, color='gray')
    ax.axhline(y=4.5, color='gray', linestyle='--', alpha=0.4)
    ax.text(1e-6, 4.7, 'Q4_0 (4.5 bpe)', fontsize=7, color='gray')

    # Panel (b): Per-level V BPE curves
    ax = axes[0, 1]
    for lvl in range(10):
        if lvl not in perlevel_v:
            continue
        thr = np.array([d['threshold'] for d in perlevel_v[lvl]])
        bpe = np.array([d['bpe'] for d in perlevel_v[lvl]])
        ax.semilogx(thr, bpe, color=level_colors[lvl], linewidth=1.5,
                     label=f'C{lvl}', alpha=0.85)
    ax.set_xlabel('Cosine Distance Threshold (log scale)')
    ax.set_ylabel('Bits per Element (BPE)')
    ax.set_title('(b) V-side BPE by Level (different candidate lists)')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2, 16.5)
    ax.axhline(y=8.5, color='gray', linestyle=':', alpha=0.4)
    ax.text(1e-6, 8.7, 'Q8_0 (8.5 bpe)', fontsize=7, color='gray')
    ax.axhline(y=4.5, color='gray', linestyle='--', alpha=0.4)
    ax.text(1e-6, 4.7, 'Q4_0 (4.5 bpe)', fontsize=7, color='gray')

    # Panel (c): K vs V BPE at same threshold for select levels — shows asymmetry
    ax = axes[1, 0]
    for lvl in [0, 2, 4, 6, 8, 9]:
        if lvl not in perlevel_k or lvl not in perlevel_v:
            continue
        k_thr_lvl = np.array([d['threshold'] for d in perlevel_k[lvl]])
        k_bpe_lvl = np.array([d['bpe'] for d in perlevel_k[lvl]])
        v_thr_lvl = np.array([d['threshold'] for d in perlevel_v[lvl]])
        v_bpe_lvl = np.array([d['bpe'] for d in perlevel_v[lvl]])
        ax.semilogx(k_thr_lvl, k_bpe_lvl, color=level_colors[lvl],
                     linewidth=2, linestyle='-', alpha=0.8)
        ax.semilogx(v_thr_lvl, v_bpe_lvl, color=level_colors[lvl],
                     linewidth=2, linestyle='--', alpha=0.8)
        # Label at end
        ax.text(k_thr_lvl[-1]*1.05, k_bpe_lvl[-1], f'K{lvl}',
                fontsize=7, color=level_colors[lvl], va='center')
        ax.text(v_thr_lvl[-1]*1.05, v_bpe_lvl[-1], f'V{lvl}',
                fontsize=7, color=level_colors[lvl], va='center')
    # Legend proxy
    ax.plot([], [], 'k-', linewidth=2, label='K (solid)')
    ax.plot([], [], 'k--', linewidth=2, label='V (dashed)')
    ax.set_xlabel('Cosine Distance Threshold (log scale)')
    ax.set_ylabel('Bits per Element (BPE)')
    ax.set_title('(c) K vs V at Same Threshold — Asymmetry by Level')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2, 16.5)

    # Panel (d): Combined CR heatmap with separate K/V axes
    ax = axes[1, 1]
    if len(grid_k_thrs) > 0 and len(grid_v_thrs) > 0 and grid_cr.size > 0:
        K, V = np.meshgrid(grid_v_thrs, grid_k_thrs)
        cs = ax.contourf(
            np.log10(K), np.log10(V), grid_cr,
            levels=np.arange(1.0, 6.5, 0.25),
            cmap='RdYlGn_r'
        )
        cbar = plt.colorbar(cs, ax=ax, label='Combined CR (×)')
        cl = ax.contour(
            np.log10(K), np.log10(V), grid_cr,
            levels=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            colors='black', linewidths=0.5
        )
        ax.clabel(cl, fmt='%.1f×', fontsize=8)
        ax.set_xlabel('log₁₀(V Threshold)')
        ax.set_ylabel('log₁₀(K Threshold)')
        ax.set_title('(d) Total CR = f(K_θ, V_θ) — Off-diagonal = asymmetric gains')

        # Diagonal line (K=V, symmetric approach)
        diag_range = np.linspace(-6, -0.5, 100)
        ax.plot(diag_range, diag_range, 'w--', linewidth=1.5, alpha=0.7, label='K=V (symmetric)')
        # Example asymmetric path: V relaxed 10× more than K
        ax.plot(diag_range, diag_range - 1.0, 'c-', linewidth=1.5, alpha=0.7,
                label='V_θ = 10× K_θ')
        ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'kv_threshold_sweep.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # ── Figure 2: Total CR with asymmetric K/V thresholds ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Total Compression Ratio with Independent K/V Error Margins',
                 fontsize=14, fontweight='bold')

    # Panel (a): Fix K threshold, sweep V — total CR
    ax = axes[0]
    # Pick several fixed K thresholds and show how relaxing V improves total CR
    k_fix_points = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    # Use C9 data (broadest candidate list) for this
    for k_fix in k_fix_points:
        # Find closest K BPE
        idx = np.argmin(np.abs(k_thr - k_fix))
        k_bpe_fixed = k_bpe[idx]
        # Total CR as V threshold varies
        total_cr = 16.0 / ((k_bpe_fixed + v_bpe) / 2.0)
        ax.semilogx(v_thr, total_cr, linewidth=2, alpha=0.8,
                     label=f'K_θ={k_fix:.0e} (K={k_bpe_fixed:.1f} bpe)')
    ax.set_xlabel('V Cosine Distance Threshold (log scale)')
    ax.set_ylabel('Total Compression Ratio (×)')
    ax.set_title('(a) Fix K threshold, sweep V → Total CR')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1, 7)

    # Panel (b): BPE gap — how much cheaper V is vs K at the same threshold
    ax = axes[1]
    # Interpolate V BPE at K threshold points
    v_bpe_interp = np.interp(k_thr, v_thr, v_bpe)
    bpe_gap = k_bpe - v_bpe_interp  # positive = V is cheaper
    ax.semilogx(k_thr, bpe_gap, 'purple', linewidth=2.5, alpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.fill_between(k_thr, 0, bpe_gap, where=(bpe_gap > 0),
                     alpha=0.15, color='green', label='V more compressible')
    ax.fill_between(k_thr, 0, bpe_gap, where=(bpe_gap < 0),
                     alpha=0.15, color='red', label='K more compressible')
    ax.set_xlabel('Cosine Distance Threshold (log scale)')
    ax.set_ylabel('BPE Gap (K_BPE − V_BPE)')
    ax.set_title('(b) K/V BPE Gap at Same Threshold (C9 candidates)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # Annotate the asymmetry explanation
    ax.annotate(
        'K errors amplified through\nsoftmax nonlinearity\n→ K needs more bits\n(AsymKV, COLING 2025)',
        xy=(1e-4, max(bpe_gap)*0.7), fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    plt.tight_layout()
    out_path2 = os.path.join(OUT_DIR, 'kv_threshold_asymmetry.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")
    plt.close()

    # ── Figure 3: Per-level total CR with asymmetric selection ──
    # For each level, show total CR when K_θ is fixed and V_θ is relaxed by multipliers
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Per-Level Total CR: Effect of V Threshold Relaxation\n'
                 '(V_θ = multiplier × K_θ, K_θ swept independently per level)',
                 fontsize=13, fontweight='bold')

    multipliers = [1.0, 2.0, 5.0, 10.0]
    bar_width = 0.18
    x = np.arange(10)

    for mi, mult in enumerate(multipliers):
        crs = []
        for lvl in range(10):
            if lvl not in perlevel_k or lvl not in perlevel_v:
                crs.append(1.0)
                continue
            # Use mid-range threshold for this level
            k_data = perlevel_k[lvl]
            v_data = perlevel_v[lvl]
            k_thrs = np.array([d['threshold'] for d in k_data])
            k_bpes = np.array([d['bpe'] for d in k_data])
            v_thrs = np.array([d['threshold'] for d in v_data])
            v_bpes = np.array([d['bpe'] for d in v_data])
            # Pick threshold where K BPE first drops below midpoint
            k_mid = (k_bpes[0] + k_bpes[-1]) / 2.0
            k_idx = np.argmin(np.abs(k_bpes - k_mid))
            k_thr_chosen = k_thrs[k_idx]
            k_bpe_chosen = k_bpes[k_idx]
            # V threshold = mult × K threshold
            v_thr_chosen = k_thr_chosen * mult
            v_idx = np.argmin(np.abs(v_thrs - v_thr_chosen))
            v_bpe_chosen = v_bpes[v_idx]
            total_cr = 16.0 / ((k_bpe_chosen + v_bpe_chosen) / 2.0)
            crs.append(total_cr)
        bars = ax.bar(x + mi * bar_width, crs, bar_width,
                       label=f'V_θ = {mult:.0f}× K_θ', alpha=0.85)
        for i, cr in enumerate(crs):
            ax.text(x[i] + mi * bar_width, cr + 0.03, f'{cr:.2f}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Compression Level', fontsize=12)
    ax.set_ylabel('Total Compression Ratio (×)', fontsize=12)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([f'C{i}' for i in range(10)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    out_path3 = os.path.join(OUT_DIR, 'kv_threshold_c_selection.png')
    plt.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path3}")
    plt.close()

    # ── Print threshold selection guidance ──
    print("\n" + "=" * 90)
    print("  THRESHOLD SELECTION GUIDANCE FROM SWEEP DATA")
    print("=" * 90)

    print("\n  K-side format transitions (C9 candidates):")
    for target_bpe in [13.0, 12.0, 11.0, 10.0, 9.0, 8.5, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0]:
        for d in k_sweep:
            if d['bpe'] <= target_bpe:
                print(f"    K_BPE ≤ {target_bpe:5.1f}  at  K_θ = {d['threshold']:.7f}  (SNR={d['snr_db']:.1f} dB, cos_p95={d['cos_p95']:.7f})")
                break

    print("\n  V-side format transitions (C9 candidates):")
    for target_bpe in [13.0, 12.0, 11.0, 10.0, 9.0, 8.5, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0]:
        for d in v_sweep:
            if d['bpe'] <= target_bpe:
                print(f"    V_BPE ≤ {target_bpe:5.1f}  at  V_θ = {d['threshold']:.7f}  (SNR={d['snr_db']:.1f} dB, cos_p95={d['cos_p95']:.7f})")
                break

    # Per-level K/V asymmetry summary
    print("\n  Per-level BPE at threshold=0.005 (K vs V):")
    print(f"  {'Level':>6}  {'K_BPE':>7}  {'V_BPE':>7}  {'Gap':>7}  {'Total CR':>9}")
    for lvl in range(10):
        if lvl not in perlevel_k or lvl not in perlevel_v:
            continue
        k_data = perlevel_k[lvl]
        v_data = perlevel_v[lvl]
        k_thrs = np.array([d['threshold'] for d in k_data])
        k_bpes = np.array([d['bpe'] for d in k_data])
        v_thrs = np.array([d['threshold'] for d in v_data])
        v_bpes = np.array([d['bpe'] for d in v_data])
        ki = np.argmin(np.abs(k_thrs - 0.005))
        vi = np.argmin(np.abs(v_thrs - 0.005))
        kb = k_bpes[ki]
        vb = v_bpes[vi]
        cr = 16.0 / ((kb + vb) / 2.0)
        print(f"  C{lvl:>4}  {kb:>7.2f}  {vb:>7.2f}  {kb-vb:>+7.2f}  {cr:>8.2f}×")


if __name__ == '__main__':
    main()
