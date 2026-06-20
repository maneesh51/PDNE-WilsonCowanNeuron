"""
Threshold Sensitivity Analysis — Node Classification
=====================================================
Shows how the E-specific / I-specific / Shared / Peripheral node counts
and percentages change as the W_out weight threshold varies.

The key question for the paper: are the qualitative findings
(~19% E, ~23% I, ~20% Shared) robust to the choice of threshold,
or do they depend critically on θ = 0.05?

What this script produces
--------------------------
One figure with three panels:

  [A] Mean ± std node counts for each class vs threshold
      (absolute numbers — shows how many nodes shift between classes)

  [B] Mean ± std percentage composition vs threshold
      (normalised — shows fractional balance is preserved or not)

  [C] Structural sign correspondence (3/4 WC signs) vs threshold
      (re-runs the population connectivity analysis at each θ and
       counts how many of the 4 WC sign predictions are matched
       in the mean structural matrix)

Usage — paste at end of notebook after the existing analyses:

    from threshold_sensitivity import plot_threshold_sensitivity

    plot_threshold_sensitivity(
        FM_tst_W_outs  = FM_tst_W_outs,
        FM_tst_Graphs  = FM_tst_Graphs,
        FM_tst_OutsNodes = FM_tst_OutsNodes,
        FM_train_ResStates = FM_train_ResStates,
        thresholds     = [0.01, 0.05, 0.10, 0.20],
        save=True, save_dir=DataDir)

Dependencies: numpy, matplotlib, scipy, networkx
The function _flatten_int_nodes and _classify_nodes are re-implemented
here so the script is self-contained (no import from notebook cells).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES (self-contained — mirrors notebook cell 15 logic)
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(nodes):
    if isinstance(nodes, np.ndarray):
        nodes = nodes.tolist()
    if isinstance(nodes, (list, tuple, set)):
        out = []
        for item in nodes:
            out.extend(_flatten(item))
        return out
    return [int(nodes)]


def _classify_nodes_thresh(W_outs, OutsNodes, G, thresh):
    """
    Classify nodes into E-spec / I-spec / Shared / Peripheral
    for a given weight threshold.  Returns counts and percentages.
    """
    nl  = list(G.nodes())
    nti = {n: i for i, n in enumerate(nl)}
    N   = len(nl)
    out_E = set(_flatten(OutsNodes[0]))
    out_I = set(_flatten(OutsNodes[1]))

    contrib = np.zeros((2, N))
    for ch, (w, grp) in enumerate(zip(W_outs,
                                       [OutsNodes[0], OutsNodes[1]])):
        w  = np.asarray(w).ravel()
        gf = _flatten(grp)
        for k, nd in enumerate(gf):
            if k < len(w) and nd in nti:
                contrib[ch, nti[nd]] = abs(w[k])
    for ch in range(2):
        mx = contrib[ch].max()
        if mx > 0:
            contrib[ch] /= mx

    score_E = contrib[0]
    score_I = contrib[1]

    in_E = np.array([n in out_E and score_E[nti[n]] > thresh for n in nl])
    in_I = np.array([n in out_I and score_I[nti[n]] > thresh for n in nl])

    E_idx = np.where( in_E & ~in_I)[0]
    I_idx = np.where(~in_E &  in_I)[0]
    S_idx = np.where( in_E &  in_I)[0]
    P_idx = np.where(~in_E & ~in_I)[0]

    return dict(
        N=N,
        n_E=len(E_idx), n_I=len(I_idx),
        n_S=len(S_idx), n_P=len(P_idx),
        pct_E=100*len(E_idx)/N, pct_I=100*len(I_idx)/N,
        pct_S=100*len(S_idx)/N, pct_P=100*len(P_idx)/N,
        E_idx=E_idx, I_idx=I_idx, S_idx=S_idx,
        node_list=nl, node_to_idx=nti,
        score_E=score_E, score_I=score_I,
    )


def _pop_connectivity(G, cl):
    """
    Mean signed edge weight between E-spec and I-spec populations.
    Returns a 2×2 matrix M where M[i,j] = mean weight FROM pop_i TO pop_j
    (row = source, col = target) — matching the convention used in
    compute_composite_popmat_allreps (cell 22 of the notebook).

    WC effective coupling signs in this convention:
      M[E-src, E-tgt] = EE > 0  (E excites E)
      M[E-src, I-tgt] = EI < 0  (E→I edges in reservoir carry negative weights,
                                  consistent with WC effective inhibitory coupling)
      M[I-src, E-tgt] = IE > 0  (I→E edges carry positive weights)
      M[I-src, I-tgt] = II = 0  (unconstrained in WC)
    """
    A   = nx.to_numpy_array(G)   # A[i,j] = weight of edge i→j (networkx convention)
    nti = cl['node_to_idx']

    pops = [cl['E_idx'], cl['I_idx']]
    M = np.full((2, 2), np.nan)
    for pi, src_idx in enumerate(pops):      # pi = source population index
        for pj, tgt_idx in enumerate(pops):  # pj = target population index
            edges = []
            for si in src_idx:
                for ti in tgt_idx:
                    w = A[si, ti]            # edge from source si to target ti
                    if w != 0:
                        edges.append(w)
            if edges:
                M[pi, pj] = np.mean(edges)  # M[source, target]
    return M


def _sign_match_count(mean_M):
    """
    Count how many of the 3 constrained WC sign predictions are matched.
    M is in [source, target] convention (matching cell 22).

    WC effective coupling signs:
      M[E-src, E-tgt] (0,0): > 0  ✓ if positive
      M[E-src, I-tgt] (0,1): < 0  ✓ if negative
      M[I-src, E-tgt] (1,0): > 0  ✓ if positive
      M[I-src, I-tgt] (1,1): unconstrained (WC w_II=0) — excluded from count
    """
    WC_signs = {(0, 0): +1,   # E→E: positive
                (0, 1): -1,   # E→I: negative (WC effective sign)
                (1, 0): +1}   # I→E: positive (WC effective sign)
    # (1,1) = I→I excluded: WC has w_II=0, no sign prediction
    n_match = 0
    n_total = 0
    for (r, c), wc_sign in WC_signs.items():
        val = mean_M[r, c]
        if not np.isnan(val):
            n_total += 1
            if np.sign(val) == wc_sign:
                n_match += 1
    return n_match, n_total


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _run_threshold(FM_tst_W_outs, FM_tst_Graphs, FM_tst_OutsNodes, thresh):
    """Run node classification and pop-connectivity at one threshold value."""
    n_reps = len(FM_tst_Graphs)
    per_rep = []
    pop_mats = []

    for rp in range(n_reps):
        G  = FM_tst_Graphs[rp]
        cl = _classify_nodes_thresh(
            FM_tst_W_outs[rp], FM_tst_OutsNodes[rp], G, thresh)
        per_rep.append(cl)

        # Population connectivity only meaningful if both pops are non-empty
        if len(cl['E_idx']) > 0 and len(cl['I_idx']) > 0:
            M = _pop_connectivity(G, cl)
            pop_mats.append(M)

    # Aggregate node counts / percentages
    n_E   = np.array([r['n_E']   for r in per_rep], dtype=float)
    n_I   = np.array([r['n_I']   for r in per_rep], dtype=float)
    n_S   = np.array([r['n_S']   for r in per_rep], dtype=float)
    n_P   = np.array([r['n_P']   for r in per_rep], dtype=float)
    pct_E = np.array([r['pct_E'] for r in per_rep], dtype=float)
    pct_I = np.array([r['pct_I'] for r in per_rep], dtype=float)
    pct_S = np.array([r['pct_S'] for r in per_rep], dtype=float)
    pct_P = np.array([r['pct_P'] for r in per_rep], dtype=float)

    # Mean population connectivity matrix and sign match
    if pop_mats:
        stack     = np.stack([m for m in pop_mats
                              if not np.any(np.isnan(m))], axis=0)
        mean_M    = np.nanmean(stack, axis=0) if len(stack) > 0 else np.full((2,2), np.nan)
        med_M     = np.nanmedian(stack, axis=0) if len(stack) > 0 else np.full((2,2), np.nan)
        n_match_mean, n_total = _sign_match_count(mean_M)
        n_match_med,  _       = _sign_match_count(med_M)
    else:
        mean_M = med_M = np.full((2, 2), np.nan)
        n_match_mean = n_match_med = n_total = 0

    return dict(
        thresh=thresh,
        n_E=n_E, n_I=n_I, n_S=n_S, n_P=n_P,
        pct_E=pct_E, pct_I=pct_I, pct_S=pct_S, pct_P=pct_P,
        mean_M=mean_M, med_M=med_M,
        n_match_mean=n_match_mean,
        n_match_med=n_match_med,
        n_total_signs=n_total,
        n_valid_reps=len(pop_mats),
    )


def _print_summary(results):
    """Print a compact console table across all thresholds."""
    print("\n" + "=" * 72)
    print(f"  THRESHOLD SENSITIVITY — NODE CLASSIFICATION")
    print("=" * 72)
    print(f"  {'θ':>6}  {'n_E':>12}  {'n_I':>12}  {'n_S':>12}  {'n_P':>12}  {'signs (mean)':>14}")
    print("-" * 72)
    for r in results:
        th = r['thresh']
        print(
            f"  {th:>6.2f}  "
            f"{np.mean(r['n_E']):5.1f}±{np.std(r['n_E']):4.1f}  "
            f"{np.mean(r['n_I']):5.1f}±{np.std(r['n_I']):4.1f}  "
            f"{np.mean(r['n_S']):5.1f}±{np.std(r['n_S']):4.1f}  "
            f"{np.mean(r['n_P']):5.1f}±{np.std(r['n_P']):4.1f}  "
            f"  {r['n_match_mean']}/{r['n_total_signs']} (med: {r['n_match_med']}/{r['n_total_signs']})"
        )
    print("=" * 72)

    print("\n  Percentage composition (mean ± std):")
    print(f"  {'θ':>6}  {'%E':>12}  {'%I':>12}  {'%Shared':>12}  {'%Periph':>12}")
    print("-" * 60)
    for r in results:
        th = r['thresh']
        print(
            f"  {th:>6.2f}  "
            f"{np.mean(r['pct_E']):5.1f}±{np.std(r['pct_E']):4.1f}  "
            f"{np.mean(r['pct_I']):5.1f}±{np.std(r['pct_I']):4.1f}  "
            f"{np.mean(r['pct_S']):5.1f}±{np.std(r['pct_S']):4.1f}  "
            f"{np.mean(r['pct_P']):5.1f}±{np.std(r['pct_P']):4.1f}"
        )
    print("=" * 60)

    print("\n  Population connectivity mean matrix per threshold (M[source, target]):")
    print("  Convention matches cell 22: EI = mean E→I edge weight, IE = mean I→E edge weight")
    print("  WC effective signs: EE>0, EI<0, IE>0  (II unconstrained)")
    for r in results:
        M = r['mean_M']
        print(f"  θ={r['thresh']:.2f}:  "
              f"EE(E→E)={M[0,0]:+.4f}  EI(E→I)={M[0,1]:+.4f}  "
              f"IE(I→E)={M[1,0]:+.4f}  II(I→I)={M[1,1]:+.4f}  "
              f"→ {r['n_match_mean']}/3 signs correct")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def _panel_node_counts(ax, results):
    """Panel A: mean ± std node counts per class vs threshold."""
    thresholds = [r['thresh'] for r in results]
    x = np.arange(len(thresholds))
    w = 0.18

    colors = {'E': 'royalblue', 'I': 'tomato', 'S': 'mediumaquamarine', 'P': 'lightgray'}

    for k, (key, label, col) in enumerate([
        ('n_E', 'E-specific', colors['E']),
        ('n_I', 'I-specific', colors['I']),
        ('n_S', 'Shared',     colors['S']),
        ('n_P', 'Peripheral', colors['P']),
    ]):
        means = [np.mean(r[key]) for r in results]
        stds  = [np.std(r[key])  for r in results]
        offset = (k - 1.5) * w
        ax.bar(x + offset, means, w, yerr=stds, capsize=4,
               color=col, alpha=0.85, label=label,
               error_kw=dict(lw=1.2, capthick=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels([f'θ = {t}' for t in thresholds], fontsize=10)
    ax.set_ylabel('Node count (mean ± std across reps)', fontsize=11)
    ax.set_title('(A) Node counts vs classification threshold',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.3)
    ax.set_ylim(bottom=0)


def _panel_pct_composition(ax, results):
    """Panel B: stacked % composition per threshold (mean across reps)."""
    thresholds = [r['thresh'] for r in results]
    x = np.arange(len(thresholds))

    pE = [np.mean(r['pct_E']) for r in results]
    pI = [np.mean(r['pct_I']) for r in results]
    pS = [np.mean(r['pct_S']) for r in results]
    pP = [np.mean(r['pct_P']) for r in results]

    eE = [np.std(r['pct_E']) for r in results]
    eI = [np.std(r['pct_I']) for r in results]
    eS = [np.std(r['pct_S']) for r in results]

    bars_E = ax.bar(x, pE, color='royalblue',        alpha=0.85, label='E-specific')
    bars_I = ax.bar(x, pI, bottom=pE,                color='tomato',          alpha=0.85, label='I-specific')
    bars_S = ax.bar(x, pS, bottom=np.add(pE, pI),    color='mediumaquamarine',alpha=0.85, label='Shared')
    bars_P = ax.bar(x, pP, bottom=np.add(np.add(pE, pI), pS),
                    color='lightgray', alpha=0.7, label='Peripheral')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}' for t in thresholds], fontsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.set_ylabel('Mean % of nodes (across reps)', fontsize=19)
    ax.set_xlabel('$\\theta$', fontsize=22)
    ax.set_ylim(0, 100)
    # ax.set_title('(B) Percentage composition vs threshold\n(dashed lines = θ=0.05 reference)',
    #              fontsize=11, fontweight='bold')
    ax.legend(fontsize=14, loc='upper left', framealpha=0.3)


def _panel_sign_match(ax, results):
    """
    Panel C: for each threshold, show the mean population connectivity
    matrix entries (EE, EI, IE) and mark which ones match WC sign.
    Also shows the sign-match count as a summary bar.
    """
    thresholds = [r['thresh'] for r in results]
    x = np.arange(len(thresholds))
    w = 0.22

    # EE, EI, IE entries from mean matrix
    EE = [r['mean_M'][0, 0] for r in results]
    EI = [r['mean_M'][0, 1] for r in results]
    IE = [r['mean_M'][1, 0] for r in results]
    II = [r['mean_M'][1, 1] for r in results]

    ax.bar(x - 1.5*w, EE, w, color='royalblue', alpha=0.75, label='E→E (WC: >0)')
    ax.bar(x - 0.5*w, EI, w, color='steelblue', alpha=0.75, label='E→I (WC: <0)')
    ax.bar(x + 0.5*w, IE, w, color='tomato',    alpha=0.75, label='I→E (WC: >0)')
    ax.bar(x + 1.5*w, II, w, color='salmon',    alpha=0.55, label='I→I (WC: 0, \nunconstrained)')

    ax.axhline(0, color='k', lw=0.8, ls='-')

    # WC sign reference lines (conceptual — very faint)
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.3)

    # Annotate sign-match count above each threshold group
    for i, r in enumerate(results):
        nm = r['n_match_mean']
        nt = r['n_total_signs']
        col = 'green' if nm == nt else 'orange' 
        ax.text(x[i], max(max(EE), max(IE), 0) * 1.0,
                f'{nm}/{nt}',
                ha='center', fontsize=14, fontweight='bold', color=col)

    ax.set_ylim(-0.05, 0.04)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}' for t in thresholds], fontsize=16)
    # ax.set_yticklabels(fontsize=16)
    ax.set_ylabel('Mean population edge weight', fontsize=19)
    ax.set_xlabel('$\\theta$', fontsize=22)
    # ax.set_title('(C) Population connectivity sign match vs threshold\n'
    #              '(n/3 = number of WC signs correctly recovered)',
    #              fontsize=11, fontweight='bold')
    ax.legend(fontsize=13, loc='lower left')


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_sensitivity(
        FM_tst_W_outs,
        FM_tst_Graphs,
        FM_tst_OutsNodes,
        FM_train_ResStates=None,   # kept for API compatibility, not used here
        thresholds=None,
        save=False,
        save_dir=None):
    """
    Sensitivity analysis of node classification to the W_out weight threshold.

    Parameters
    ----------
    FM_tst_W_outs      : list (n_reps) of [W_out_E, W_out_I]
    FM_tst_Graphs      : list (n_reps) of NetworkX DiGraphs
    FM_tst_OutsNodes   : list (n_reps) of [out_E_nodes, out_I_nodes]
    FM_train_ResStates : unused, kept for API consistency
    thresholds         : list of float, default [0.01, 0.05, 0.10, 0.20]
    save               : bool
    save_dir           : str
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.10, 0.20]

    # Guard: 0.05 must be in thresholds for the reference lines in panel B
    if 0.05 not in thresholds:
        thresholds = sorted(set(thresholds) | {0.05})
        print(f"  Note: θ=0.05 added to thresholds for reference lines: {thresholds}")

    print(f"\nRunning threshold sensitivity across θ = {thresholds} ...")
    results = []
    for th in thresholds:
        print(f"\n  θ = {th}")
        r = _run_threshold(FM_tst_W_outs, FM_tst_Graphs, FM_tst_OutsNodes, th)
        results.append(r)

    _print_summary(results)
    
    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    plt.subplots_adjust(wspace=0.25)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    # ax_C = fig.add_subplot(gs[0, 2])

    # _panel_node_counts(ax_A, results)
    _panel_pct_composition(ax_A, results)
    _panel_sign_match(ax_B, results)
    plt.yticks(fontsize=16)
    n_reps = len(FM_tst_Graphs)
    # fig.suptitle(
    #     f'Node classification sensitivity to W_out weight threshold\n'
    #     f'N = {n_reps} repetitions — qualitative pattern should be stable across θ',
    #     fontsize=12, fontweight='bold'  )
    plt.tight_layout()

    if save and save_dir:
        fig.savefig(f"{save_dir}/Threshold_Sensitivity.svg", bbox_inches='tight')
        fig.savefig(f"{save_dir}/Threshold_Sensitivity.png",
                    bbox_inches='tight', dpi=300)
        print("\n  Threshold sensitivity figure saved.")
    plt.show()
    return results