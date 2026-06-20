"""
W_out Contribution Analysis for PDNE-WC Revision
=================================================
Four analyses that combine reservoir states, W_out readout weights,
and node classification to ask: how does the evolved reservoir actually
compute E(t) and I(t) through its classified node populations?

Key structural fact (confirmed from data):
  FM_tst_W_outs[rep] = [W_out_E, W_out_I]
  len(W_out_E) = |O_E|   (e.g. 25 for rep 0)
  len(W_out_I) = |O_I|   (e.g. 29 for rep 0)
  |O_E| ≠ |O_I| in general; nodes can be in both (non-exclusive)

  Ê(t) = sum_k  W_out_E[k] * r_{O_E[k]}(t)
  Î(t) = sum_k  W_out_I[k] * r_{O_I[k]}(t)

  Contribution of node n to channel c at time t:
    contrib_c(n, t) = W_out_c[pos_of_n_in_O_c] * r_n(t)  if n ∈ O_c
                    = 0                                     otherwise

State extraction (confirmed):
  np.array(trs).shape = (N, B, T)  →  R = transpose(1,2,0).reshape(B*T, N)
  Per amplitude b: R_b = array(trs)[:, b, :].T  →  (T, N)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(nodes):
    if isinstance(nodes, np.ndarray): nodes = nodes.tolist()
    if isinstance(nodes, (list, tuple, set)):
        out = []
        for item in nodes: out.extend(_flatten(item))
        return out
    return [int(nodes)]


def _classify_nodes(W_outs, OutsNodes, G, thresh=0.05):
    """Returns classification dict with node_to_idx, per-channel weight maps."""
    nl  = list(G.nodes())
    nti = {n: i for i, n in enumerate(nl)}
    N   = len(nl)
    out_E = set(_flatten(OutsNodes[0]))
    out_I = set(_flatten(OutsNodes[1]))

    # Build full-N weight vectors (signed, not absolute)
    w_E_full = np.zeros(N); w_I_full = np.zeros(N)
    w_E_arr  = np.asarray(W_outs[0]).ravel()
    w_I_arr  = np.asarray(W_outs[1]).ravel()
    outE_ordered = _flatten(OutsNodes[0])
    outI_ordered = _flatten(OutsNodes[1])
    for k, nd in enumerate(outE_ordered):
        if nd in nti and k < len(w_E_arr):
            w_E_full[nti[nd]] = w_E_arr[k]
    for k, nd in enumerate(outI_ordered):
        if nd in nti and k < len(w_I_arr):
            w_I_full[nti[nd]] = w_I_arr[k]

    # Normalized absolute weights for classification
    abs_E = np.abs(w_E_full); abs_I = np.abs(w_I_full)
    abs_E = abs_E / abs_E.max() if abs_E.max() > 0 else abs_E
    abs_I = abs_I / abs_I.max() if abs_I.max() > 0 else abs_I

    in_E = np.array([nd in out_E and abs_E[nti[nd]] > thresh for nd in nl])
    in_I = np.array([nd in out_I and abs_I[nti[nd]] > thresh for nd in nl])

    return dict(
        E_spec_idx = np.where( in_E & ~in_I)[0],
        I_spec_idx = np.where(~in_E &  in_I)[0],
        shared_idx = np.where( in_E &  in_I)[0],
        periph_idx = np.where(~in_E & ~in_I)[0],
        w_E_full   = w_E_full,   # signed, full-N
        w_I_full   = w_I_full,   # signed, full-N
        abs_E_norm = abs_E,
        abs_I_norm = abs_I,
        node_list  = nl,
        node_to_idx= nti,
        N = N,
        out_E = out_E, out_I = out_I,
    )


def _get_states(trs):
    """(N,B,T) → (B*T, N)"""
    R = np.array(trs, dtype=float)
    N, B, T = R.shape
    return R.transpose(1,2,0).reshape(B*T, N), B, T


def _get_states_per_amp(trs, amp_idx):
    """(N,B,T) → (T, N) for one amplitude."""
    return np.array(trs, dtype=float)[:, amp_idx, :].T


def _class_contribution(R, cl, channel='E'):
    """
    Returns dict: class_name → contribution time series (T_total,)

    For each node n in a class, its contribution to channel c is:
        w_c_full[n] * r_n(t)
    summed over all nodes in that class.

    R: (T_total, N)
    """
    w = cl['w_E_full'] if channel == 'E' else cl['w_I_full']
    N = cl['N']
    classes = {
        'E-spec'  : cl['E_spec_idx'],
        'I-spec'  : cl['I_spec_idx'],
        'Shared'  : cl['shared_idx'],
        'Periph'  : cl['periph_idx'],
        'Total'   : np.arange(N),
    }
    contribs = {}
    for name, idx in classes.items():
        if len(idx) == 0:
            contribs[name] = np.zeros(R.shape[0])
        else:
            # sum of w[n] * r_n(t) for n in class
            contribs[name] = R[:, idx] @ w[idx]
    return contribs


def _r2(pred, target):
    """R² between pred and target time series."""
    if np.std(pred) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    return float(np.corrcoef(pred, target)[0,1])**2


def _signed_r(pred, target):
    """Signed correlation (preserves direction)."""
    if np.std(pred) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    return float(np.corrcoef(pred, target)[0,1])


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1+2: CONTRIBUTION DECOMPOSITION + R² BY NODE CLASS
# ─────────────────────────────────────────────────────────────────────────────

def contribution_decomposition_analysis(
        FM_tst_Graphs, FM_tst_W_outs, FM_tst_OutsNodes,
        FM_train_ResStates, Outs_O,
        thresh=0.05, save=False, save_dir=None):
    """
    For each rep, decompose Ê(t) and Î(t) into contributions from
    E-specific, I-specific, Shared, and Peripheral node classes.

    Measures:
      A1. Mean contribution magnitude per class per channel (bar chart)
      A2. R²(class → channel) — how much of E(t)/I(t) variance each
          class explains through its readout-weighted activity
      A3. Signed correlation: sign tells whether class contributes
          constructively or destructively to each channel prediction

    Key test: is R²(E-spec → E) > R²(E-spec → I)?
              is R²(I-spec → I) > R²(I-spec → E)?
    This tests whether the structural classification corresponds to
    functional readout specialization.
    """
    n_reps = len(FM_tst_Graphs)
    class_names = ['E-spec', 'I-spec', 'Shared', 'Periph']

    # Storage: [rep, class, channel]  where channel 0=E, 1=I
    r2_mat    = np.full((n_reps, 4, 2), np.nan)  # R²
    corr_mat  = np.full((n_reps, 4, 2), np.nan)  # signed r
    frac_mat  = np.full((n_reps, 4, 2), np.nan)  # fraction of total prediction

    # Flatten targets
    E_tgt = Outs_O[:,0,:].ravel()   # (B*T,)
    I_tgt = Outs_O[:,1,:].ravel()

    for rp in range(n_reps):
        G  = FM_tst_Graphs[rp]
        cl = _classify_nodes(FM_tst_W_outs[rp], FM_tst_OutsNodes[rp], G, thresh)
        N  = cl['N']

        R, B, T = _get_states(FM_train_ResStates[rp])
        if R.shape[1] != N:
            print(f"  Rep {rp:2d}: SKIPPED shape mismatch")
            continue

        T_use = min(R.shape[0], len(E_tgt))
        R_use = R[:T_use]; E_use = E_tgt[:T_use]; I_use = I_tgt[:T_use]

        # Decompose each channel
        contribs_E = _class_contribution(R_use, cl, 'E')
        contribs_I = _class_contribution(R_use, cl, 'I')

        total_E = contribs_E['Total']
        total_I = contribs_I['Total']

        for ci, cname in enumerate(class_names):
            cE = contribs_E[cname]; cI = contribs_I[cname]

            # R² against actual targets
            r2_mat[rp, ci, 0]   = _r2(cE, E_use)
            r2_mat[rp, ci, 1]   = _r2(cI, I_use)
            corr_mat[rp, ci, 0] = _signed_r(cE, E_use)
            corr_mat[rp, ci, 1] = _signed_r(cI, I_use)

            # Fraction of total prediction variance explained
            # = R² of class contribution vs total prediction
            frac_mat[rp, ci, 0] = _r2(cE, total_E) if np.std(total_E) > 1e-10 else 0.0
            frac_mat[rp, ci, 1] = _r2(cI, total_I) if np.std(total_I) > 1e-10 else 0.0

        n_E = len(cl['E_spec_idx']); n_I = len(cl['I_spec_idx'])
        n_S = len(cl['shared_idx']); n_P = len(cl['periph_idx'])
        print(f"  Rep {rp:2d} | N_E={n_E:3d} N_I={n_I:3d} N_S={n_S:3d} N_P={n_P:3d} | "
              f"R²(E-spec→E)={r2_mat[rp,0,0]:.3f} "
              f"R²(E-spec→I)={r2_mat[rp,0,1]:.3f} | "
              f"R²(I-spec→E)={r2_mat[rp,1,0]:.3f} "
              f"R²(I-spec→I)={r2_mat[rp,1,1]:.3f}")

    # Aggregate
    mean_r2  = np.nanmean(r2_mat,   axis=0)   # (4, 2)
    std_r2   = np.nanstd(r2_mat,    axis=0)
    mean_corr= np.nanmean(corr_mat, axis=0)
    mean_frac= np.nanmean(frac_mat, axis=0)

    print("\n" + "="*65)
    print("  R² DECOMPOSITION — MEAN ACROSS REPS")
    print("="*65)
    print(f"  {'Class':10s}  {'R²→E(t)':>10s}  {'R²→I(t)':>10s}  "
          f"{'r→E(t)':>10s}  {'r→I(t)':>10s}  {'Spec?':>6s}")
    for ci, cname in enumerate(class_names):
        r2E = mean_r2[ci,0]; r2I = mean_r2[ci,1]
        rE  = mean_corr[ci,0]; rI = mean_corr[ci,1]
        spec = '✓' if (ci==0 and r2E>r2I) or (ci==1 and r2I>r2E) else '✗'
        print(f"  {cname:10s}  {r2E:>10.4f}  {r2I:>10.4f}  "
              f"{rE:>+10.4f}  {rI:>+10.4f}  {spec:>6s}")

    # Key statistical tests
    # Test: R²(E-spec→E) > R²(E-spec→I) across reps?
    valid = ~np.isnan(r2_mat[:,0,0])
    if valid.sum() > 1:
        t_Espec, p_Espec = stats.ttest_rel(
            r2_mat[valid,0,0], r2_mat[valid,0,1])
        t_Ispec, p_Ispec = stats.ttest_rel(
            r2_mat[valid,1,1], r2_mat[valid,1,0])
        print(f"\n  Paired t R²(E-spec→E) > R²(E-spec→I): "
              f"t={t_Espec:.3f}  p={p_Espec:.4f}  "
              f"{'✓' if p_Espec<0.05 and mean_r2[0,0]>mean_r2[0,1] else '✗'}")
        print(f"  Paired t R²(I-spec→I) > R²(I-spec→E): "
              f"t={t_Ispec:.3f}  p={p_Ispec:.4f}  "
              f"{'✓' if p_Ispec<0.05 and mean_r2[1,1]>mean_r2[1,0] else '✗'}")
    else:
        t_Espec=p_Espec=t_Ispec=p_Ispec=np.nan

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['royalblue','tomato','mediumaquamarine','lightgray']

    # 1. Mean R² heatmap (class × channel)
    ax = fig.add_subplot(gs[0, 0])
    vmax = min(mean_r2.max(), 1.0)
    im = ax.imshow(mean_r2, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks(range(4))
    ax.set_xticklabels(['→ E(t)', '→ I(t)'], fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_title('Mean R² per Class per Channel\n(diagonal pattern = specialization)',
                 fontsize=11, fontweight='bold')
    for i in range(4):
        for j in range(2):
            ax.text(j, i, f'{mean_r2[i,j]:.3f}\n±{std_r2[i,j]:.3f}',
                    ha='center', va='center', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. Signed correlation heatmap
    ax = fig.add_subplot(gs[0, 1])
    im2 = ax.imshow(mean_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks(range(4))
    ax.set_xticklabels(['→ E(t)', '→ I(t)'], fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_title('Mean Signed Correlation\n(sign = constructive/destructive)',
                 fontsize=11, fontweight='bold')
    for i in range(4):
        for j in range(2):
            ax.text(j, i, f'{mean_corr[i,j]:+.3f}',
                    ha='center', va='center', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # 3. Specialization index = R²(correct) - R²(cross) per rep
    ax = fig.add_subplot(gs[0, 2])
    spec_E = r2_mat[:,0,0] - r2_mat[:,0,1]   # R²(E-spec→E) - R²(E-spec→I)
    spec_I = r2_mat[:,1,1] - r2_mat[:,1,0]   # R²(I-spec→I) - R²(I-spec→E)
    ri = np.arange(n_reps)
    ax.bar(ri-0.2, spec_E, 0.35, color='royalblue', alpha=0.8, label='E-spec index')
    ax.bar(ri+0.2, spec_I, 0.35, color='tomato',    alpha=0.8, label='I-spec index')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axhline(np.nanmean(spec_E), color='royalblue', ls='--', lw=1.5, alpha=0.7,
               label=f"mean E={np.nanmean(spec_E):+.3f}")
    ax.axhline(np.nanmean(spec_I), color='tomato', ls='--', lw=1.5, alpha=0.7,
               label=f"mean I={np.nanmean(spec_I):+.3f}")
    if not np.isnan(p_Espec):
        ax.text(0.98, 0.95,
                f'E-spec p={p_Espec:.3f}{"*" if p_Espec<0.05 else ""}\n'
                f'I-spec p={p_Ispec:.3f}{"*" if p_Ispec<0.05 else ""}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_xlabel('Model Rep', fontsize=12)
    ax.set_ylabel('Specialization Index\nR²(correct)−R²(cross)', fontsize=11)
    ax.set_title('Readout Specialization per Rep\n(positive = correct channel preferred)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xticks(ri[::3])

    # 4. Per-rep R² E-spec row
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ri, r2_mat[:,0,0], 'o-', color='royalblue', lw=1.5, label='R²(E-spec→E)')
    ax.plot(ri, r2_mat[:,0,1], 's--', color='royalblue', lw=1.5, alpha=0.5,
            label='R²(E-spec→I)')
    ax.fill_between(ri, r2_mat[:,0,0], r2_mat[:,0,1],
                    where=r2_mat[:,0,0]>r2_mat[:,0,1],
                    alpha=0.2, color='royalblue', label='E-spec prefers E ✓')
    ax.fill_between(ri, r2_mat[:,0,0], r2_mat[:,0,1],
                    where=r2_mat[:,0,0]<r2_mat[:,0,1],
                    alpha=0.2, color='red', label='E-spec prefers I ✗')
    ax.set_xlabel('Model Rep', fontsize=12); ax.set_ylabel('R²', fontsize=12)
    ax.set_title('E-specific nodes: R² for E(t) vs I(t)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xticks(ri[::3])

    # 5. Per-rep R² I-spec row
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ri, r2_mat[:,1,1], 'o-', color='tomato', lw=1.5, label='R²(I-spec→I)')
    ax.plot(ri, r2_mat[:,1,0], 's--', color='tomato', lw=1.5, alpha=0.5,
            label='R²(I-spec→E)')
    ax.fill_between(ri, r2_mat[:,1,1], r2_mat[:,1,0],
                    where=r2_mat[:,1,1]>r2_mat[:,1,0],
                    alpha=0.2, color='tomato', label='I-spec prefers I ✓')
    ax.fill_between(ri, r2_mat[:,1,1], r2_mat[:,1,0],
                    where=r2_mat[:,1,1]<r2_mat[:,1,0],
                    alpha=0.2, color='blue', label='I-spec prefers E ✗')
    ax.set_xlabel('Model Rep', fontsize=12); ax.set_ylabel('R²', fontsize=12)
    ax.set_title('I-specific nodes: R² for I(t) vs E(t)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xticks(ri[::3])

    # 6. Summary: mean R² 2×2 matrix (E-spec vs I-spec rows, E/I channels cols)
    ax = fig.add_subplot(gs[1, 2])
    summary = mean_r2[:2, :]   # just E-spec and I-spec rows
    vmax2 = max(summary.max(), 0.01)
    im3 = ax.imshow(summary, cmap='YlOrRd', vmin=0, vmax=vmax2, aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['R²→E(t)', 'R²→I(t)'], fontsize=12)
    ax.set_yticklabels(['E-specific\nnodes', 'I-specific\nnodes'], fontsize=11)
    ax.set_title('E/I Specialization Matrix\n(diagonal > off-diagonal = ✓)',
                 fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            diag = (i == j)
            ax.text(j, i,
                    f'{summary[i,j]:.4f}',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if summary[i,j] > 0.5*vmax2 else 'black')
            if diag:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor='green', lw=3))
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('W_out Contribution Decomposition: How Each Node Class Predicts E(t) and I(t)\n'
                 f'(N=30 reps, N₀∈{{15,25,35}})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save and save_dir:
        plt.savefig(f"{save_dir}/Contribution_Decomposition.svg", bbox_inches='tight')
    plt.show()

    return dict(r2_mat=r2_mat, corr_mat=corr_mat, frac_mat=frac_mat,
                mean_r2=mean_r2, std_r2=std_r2, mean_corr=mean_corr,
                p_Espec=p_Espec, p_Ispec=p_Ispec,
                t_Espec=t_Espec, t_Ispec=t_Ispec)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: W_out SIGN CONSISTENCY WITH NODE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def wout_sign_analysis(
        FM_tst_Graphs, FM_tst_W_outs, FM_tst_OutsNodes,
        thresh=0.05, save=False, save_dir=None):
    """
    For each classified node, examine the sign of its W_out weights.

    E-specific nodes: expected positive W_out_E (add to Ê, like excitatory neurons)
    I-specific nodes: expected positive W_out_I (add to Î, predicting inhibitory activity)
                      and near-zero or any-sign W_out_E

    Reports:
      - Fraction of E-spec nodes with positive W_out_E per rep
      - Fraction of I-spec nodes with positive W_out_I per rep
      - Mean signed W_out_E for E-spec, and W_out_I for I-spec
      - Statistical test: is the mean W_out_E of E-spec nodes > 0?
    """
    n_reps = len(FM_tst_Graphs)
    results = []

    for rp in range(n_reps):
        G  = FM_tst_Graphs[rp]
        cl = _classify_nodes(FM_tst_W_outs[rp], FM_tst_OutsNodes[rp], G, thresh)

        E_idx = cl['E_spec_idx']; I_idx = cl['I_spec_idx']

        # Signed W_out weights for classified nodes
        wE_Espec = cl['w_E_full'][E_idx]   # W_out_E weights of E-specific nodes
        wI_Ispec = cl['w_I_full'][I_idx]   # W_out_I weights of I-specific nodes
        wE_Ispec = cl['w_E_full'][I_idx]   # W_out_E weights of I-specific nodes (should be ~0)
        wI_Espec = cl['w_I_full'][E_idx]   # W_out_I weights of E-specific nodes (should be ~0)

        # Fraction with correct sign
        frac_Epos = float(np.mean(wE_Espec > 0)) if len(wE_Espec)>0 else np.nan
        frac_Ipos = float(np.mean(wI_Ispec > 0)) if len(wI_Ispec)>0 else np.nan

        rep_res = dict(
            rep=rp, N_E=len(E_idx), N_I=len(I_idx),
            wE_Espec=wE_Espec, wI_Ispec=wI_Ispec,
            wE_Ispec=wE_Ispec, wI_Espec=wI_Espec,
            mean_wE_Espec = float(np.mean(wE_Espec)) if len(wE_Espec)>0 else np.nan,
            mean_wI_Ispec = float(np.mean(wI_Ispec)) if len(wI_Ispec)>0 else np.nan,
            mean_wE_Ispec = float(np.mean(wE_Ispec)) if len(wE_Ispec)>0 else np.nan,
            frac_Epos=frac_Epos, frac_Ipos=frac_Ipos,
        )
        results.append(rep_res)
        print(f"  Rep {rp:2d} | N_E={len(E_idx):3d} N_I={len(I_idx):3d} | "
              f"W_out_E(E-spec): mean={rep_res['mean_wE_Espec']:+.4f} "
              f"frac+={frac_Epos:.2f} | "
              f"W_out_I(I-spec): mean={rep_res['mean_wI_Ispec']:+.4f} "
              f"frac+={frac_Ipos:.2f}")

    # Pool across reps
    all_wE_E = np.concatenate([r['wE_Espec'] for r in results if len(r['wE_Espec'])>0])
    all_wI_I = np.concatenate([r['wI_Ispec'] for r in results if len(r['wI_Ispec'])>0])
    all_wE_I = np.concatenate([r['wE_Ispec'] for r in results if len(r['wE_Ispec'])>0])

    tE_stat, tE_p = stats.ttest_1samp(all_wE_E, 0)
    tI_stat, tI_p = stats.ttest_1samp(all_wI_I, 0)
    # Is W_out_E of I-spec nodes closer to zero than W_out_E of E-spec nodes?
    tEI_stat, tEI_p = stats.ttest_ind(np.abs(all_wE_E), np.abs(all_wE_I))

    mean_frac_E = np.nanmean([r['frac_Epos'] for r in results])
    mean_frac_I = np.nanmean([r['frac_Ipos'] for r in results])

    print("\n" + "="*60)
    print("  W_out SIGN CONSISTENCY")
    print("="*60)
    print(f"  Mean W_out_E of E-specific nodes: {np.mean(all_wE_E):+.4f}  "
          f"t={tE_stat:.3f}  p={tE_p:.4f}  "
          f"{'> 0 ✓' if np.mean(all_wE_E)>0 else '< 0 ✗'}")
    print(f"  Mean W_out_I of I-specific nodes: {np.mean(all_wI_I):+.4f}  "
          f"t={tI_stat:.3f}  p={tI_p:.4f}  "
          f"{'> 0 ✓' if np.mean(all_wI_I)>0 else '< 0 ✗'}")
    print(f"  Mean W_out_E of I-specific nodes: {np.mean(all_wE_I):+.4f}  "
          f"(should be ~0 or weaker than E-spec)")
    print(f"  |W_out_E| E-spec > I-spec: t={tEI_stat:.3f}  p={tEI_p:.4f}  "
          f"{'✓' if tEI_p<0.05 and np.mean(np.abs(all_wE_E))>np.mean(np.abs(all_wE_I)) else '✗'}")
    print(f"  Frac E-spec with positive W_out_E: {mean_frac_E:.3f}")
    print(f"  Frac I-spec with positive W_out_I: {mean_frac_I:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    bins = np.linspace(-1.5, 1.5, 40)
    ax.hist(all_wE_E, bins=bins, color='royalblue', alpha=0.7,
            label=f'W_out_E of E-nodes\nmean={np.mean(all_wE_E):+.3f}')
    ax.hist(all_wE_I, bins=bins, color='tomato', alpha=0.7,
            label=f'W_out_E of I-nodes\nmean={np.mean(all_wE_I):+.3f}')
    ax.axvline(0, color='k', lw=1.5, ls='--')
    ax.axvline(np.mean(all_wE_E), color='royalblue', lw=2)
    ax.axvline(np.mean(all_wE_I), color='tomato', lw=2)
    ax.set_xlabel('W_out_E weight (signed)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('W_out_E weights:\nE-nodes vs I-nodes', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(all_wI_I, bins=bins, color='tomato', alpha=0.7,
            label=f'W_out_I of I-nodes\nmean={np.mean(all_wI_I):+.3f}')
    ax.axvline(0, color='k', lw=1.5, ls='--')
    ax.axvline(np.mean(all_wI_I), color='tomato', lw=2)
    ax.set_xlabel('W_out_I weight (signed)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('W_out_I weights of I-specific nodes\n'
                 f'(p={tI_p:.4f} vs 0{"*" if tI_p<0.05 else ""})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[2]
    rids = [r['rep'] for r in results]
    fE = [r['frac_Epos'] for r in results]
    fI = [r['frac_Ipos'] for r in results]
    ax.bar(np.array(rids)-0.2, fE, 0.35, color='royalblue', alpha=0.85,
           label=f'E-nodes +W_out_E (mean={mean_frac_E:.2f})')
    ax.bar(np.array(rids)+0.2, fI, 0.35, color='tomato', alpha=0.85,
           label=f'I-nodes +W_out_I (mean={mean_frac_I:.2f})')
    ax.axhline(0.5, color='k', lw=1, ls=':', label='50% (chance)')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Model Rep', fontsize=12)
    ax.set_ylabel('Fraction with positive weight', fontsize=12)
    ax.set_title('Fraction of nodes with\ncorrect-sign readout weights',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xticks(np.array(rids)[::3])

    plt.suptitle('Analysis 3: W_out Sign Consistency with Node Classification\n'
                 'E-spec nodes should have positive W_out_E; I-spec positive W_out_I',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save and save_dir:
        plt.savefig(f"{save_dir}/Wout_Sign_Analysis.svg", bbox_inches='tight')
    plt.show()

    return results, dict(
        mean_wE_Espec=float(np.mean(all_wE_E)),
        mean_wI_Ispec=float(np.mean(all_wI_I)),
        tE_p=tE_p, tI_p=tI_p, tEI_p=tEI_p,
        mean_frac_E=mean_frac_E, mean_frac_I=mean_frac_I,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4: TIMESCALE RECOVERY IN READOUT-WEIGHTED DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def timescale_analysis(
        FM_tst_Graphs, FM_tst_W_outs, FM_tst_OutsNodes,
        FM_train_ResStates, Train_Inputs, Outs_O,
        thresh=0.05, decay_window=30,
        save=False, save_dir=None):
    """
    Tests whether E-node contributions to Ê(t) decay with a timescale
    closer to τ_E=10 than τ_I=5, and vice versa for I-nodes.

    This is the key test the paper could not do with topology alone:
    path-length analysis showed no timescale asymmetry in graph structure,
    but the READOUT-WEIGHTED DYNAMICS may encode the τ_E vs τ_I difference.

    For each trial b and each pulse p:
      1. Find pulse end time pe from Train_Inputs[b]
      2. Extract post-pulse window: r_n(t) for t in [pe+1, pe+decay_window]
      3. Compute class contribution: contrib_class(t) = sum W_out[n]*r_n(t)
      4. Fit exponential decay: f(t) = A * exp(-t/tau) + C
      5. Compare tau(E-node contrib to Ê) vs tau(I-node contrib to Î)

    WC ground truth: τ_E=10, τ_I=5 (in simulation units, subsampled by 10
    → 1 index = 1s of simulation time → τ_E≈10 indices, τ_I≈5 indices).
    """
    n_reps = len(FM_tst_Graphs)
    # Parse pulse timing from Train_Inputs
    # Train_Inputs shape: (B, T)
    B, T_inp = Train_Inputs.shape
    pulse_segs_all = []
    for b in range(B):
        inp = Train_Inputs[b]
        pidx = np.where(inp > 0)[0]
        if len(pidx) == 0:
            pulse_segs_all.append([])
            continue
        gaps = np.where(np.diff(pidx) > 1)[0]
        segs = []; start = int(pidx[0])
        for g in gaps:
            segs.append((start, int(pidx[g]))); start = int(pidx[g+1])
        segs.append((start, int(pidx[-1])))
        pulse_segs_all.append(segs)

    def fit_decay(y, dt=1):
        """Fit A*exp(-t/tau)+C, return tau or nan."""
        t = np.arange(len(y), dtype=float) * dt
        try:
            p0 = [y[0]-y[-1], 5.0, y[-1]]
            bounds = ([0, 0.5, -np.inf], [np.inf, 100, np.inf])
            popt, _ = curve_fit(lambda t,A,tau,C: A*np.exp(-t/tau)+C,
                                t, y, p0=p0, bounds=bounds, maxfev=2000)
            return float(popt[1])
        except Exception:
            return np.nan

    all_tau_E = []   # tau of E-node contrib to Ê
    all_tau_I = []   # tau of I-node contrib to Î

    for rp in range(n_reps):
        G  = FM_tst_Graphs[rp]
        cl = _classify_nodes(FM_tst_W_outs[rp], FM_tst_OutsNodes[rp], G, thresh)
        N  = cl['N']
        rep_tau_E = []; rep_tau_I = []

        for b in range(B):
            R_b = _get_states_per_amp(FM_train_ResStates[rp], b)  # (T, N)
            if R_b.shape[1] != N: continue
            T_b = R_b.shape[0]

            for (ps, pe) in pulse_segs_all[b]:
                # Post-pulse window
                t_start = pe + 1
                t_end   = min(pe + 1 + decay_window, T_b)
                if t_end - t_start < 5: continue

                R_win = R_b[t_start:t_end]   # (decay_window, N)

                # E-node contribution to Ê in decay window
                E_idx = cl['E_spec_idx']
                I_idx = cl['I_spec_idx']
                if len(E_idx) > 0:
                    cE = R_win[:, E_idx] @ cl['w_E_full'][E_idx]
                    tau = fit_decay(cE - cE[-1])   # subtract baseline
                    if not np.isnan(tau) and 0.5 < tau < 50:
                        rep_tau_E.append(tau)
                if len(I_idx) > 0:
                    cI = R_win[:, I_idx] @ cl['w_I_full'][I_idx]
                    tau = fit_decay(cI - cI[-1])
                    if not np.isnan(tau) and 0.5 < tau < 50:
                        rep_tau_I.append(tau)

        m_tE = float(np.mean(rep_tau_E)) if rep_tau_E else np.nan
        m_tI = float(np.mean(rep_tau_I)) if rep_tau_I else np.nan
        all_tau_E.append(m_tE); all_tau_I.append(m_tI)
        print(f"  Rep {rp:2d} | tau(E-contrib→Ê)={m_tE:.2f}  "
              f"tau(I-contrib→Î)={m_tI:.2f}  "
              f"(WC: τ_E=10, τ_I=5)")

    all_tau_E = np.array(all_tau_E); all_tau_I = np.array(all_tau_I)
    valid = ~(np.isnan(all_tau_E) | np.isnan(all_tau_I))

    tE_vs_tI_stat, tE_vs_tI_p = (
        stats.ttest_rel(all_tau_E[valid], all_tau_I[valid])
        if valid.sum() > 1 else (np.nan, np.nan))
    # One-sample t-tests vs WC ground truth
    tauE_vs10_stat, tauE_vs10_p = (
        stats.ttest_1samp(all_tau_E[~np.isnan(all_tau_E)], 10)
        if (~np.isnan(all_tau_E)).sum() > 1 else (np.nan, np.nan))
    tauI_vs5_stat, tauI_vs5_p = (
        stats.ttest_1samp(all_tau_I[~np.isnan(all_tau_I)], 5)
        if (~np.isnan(all_tau_I)).sum() > 1 else (np.nan, np.nan))

    print("\n" + "="*60)
    print("  TIMESCALE RECOVERY")
    print("="*60)
    print(f"  mean tau(E-contrib→Ê) = {np.nanmean(all_tau_E):.3f}  "
          f"(WC τ_E = 10)  p vs 10: {tauE_vs10_p:.4f}")
    print(f"  mean tau(I-contrib→Î) = {np.nanmean(all_tau_I):.3f}  "
          f"(WC τ_I =  5)  p vs 5:  {tauI_vs5_p:.4f}")
    print(f"  tau_E > tau_I: t={tE_vs_tI_stat:.3f}  p={tE_vs_tI_p:.4f}  "
          f"{'✓' if (not np.isnan(tE_vs_tI_p) and tE_vs_tI_p<0.05 and np.nanmean(all_tau_E)>np.nanmean(all_tau_I)) else '✗'}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ri = np.arange(n_reps)

    ax = axes[0]
    ax.scatter(ri[valid], all_tau_E[valid], color='royalblue', s=60, zorder=3,
               label=f'tau(E-contrib) mean={np.nanmean(all_tau_E):.2f}')
    ax.scatter(ri[valid], all_tau_I[valid], color='tomato', s=60, marker='s',
               zorder=3, label=f'tau(I-contrib) mean={np.nanmean(all_tau_I):.2f}')
    ax.axhline(10, color='royalblue', ls='--', lw=1.5, alpha=0.6, label='WC τ_E=10')
    ax.axhline(5,  color='tomato',    ls='--', lw=1.5, alpha=0.6, label='WC τ_I=5')
    ax.set_xlabel('Model Rep', fontsize=12); ax.set_ylabel('Decay timescale τ', fontsize=12)
    ax.set_title('Readout-Weighted Decay Timescales\nvs WC Ground Truth',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xticks(ri[::3])

    ax = axes[1]
    ax.scatter(all_tau_E[valid], all_tau_I[valid],
               c=ri[valid], cmap='viridis', s=60, zorder=3)
    ax.axhline(5,  color='tomato',    ls='--', lw=1.5, alpha=0.6, label='WC τ_I=5')
    ax.axvline(10, color='royalblue', ls='--', lw=1.5, alpha=0.6, label='WC τ_E=10')
    ax.plot([0,50],[0,50],'k--',lw=0.8,alpha=0.4,label='τ_E=τ_I line')
    ax.set_xlabel('tau(E-contrib→Ê)', fontsize=12)
    ax.set_ylabel('tau(I-contrib→Î)', fontsize=12)
    ax.set_title(f'τ_E vs τ_I per Rep\n'
                 f'(paired t p={tE_vs_tI_p:.3f}{"*" if not np.isnan(tE_vs_tI_p) and tE_vs_tI_p<0.05 else ""})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[2]
    bins = np.linspace(0, 30, 25)
    ax.hist(all_tau_E[~np.isnan(all_tau_E)], bins=bins, color='royalblue',
            alpha=0.7, label=f'τ(E-contrib) mean={np.nanmean(all_tau_E):.2f}')
    ax.hist(all_tau_I[~np.isnan(all_tau_I)], bins=bins, color='tomato',
            alpha=0.7, label=f'τ(I-contrib) mean={np.nanmean(all_tau_I):.2f}')
    ax.axvline(10, color='royalblue', ls='--', lw=2, label='WC τ_E=10')
    ax.axvline(5,  color='tomato',    ls='--', lw=2, label='WC τ_I=5')
    ax.set_xlabel('Decay timescale τ (indices)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Recovered Timescales', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle('Analysis 4: Timescale Recovery in Readout-Weighted Node Contributions\n'
                 'Do E-node contributions to Ê(t) decay with τ≈τ_E=10? I-nodes with τ≈τ_I=5?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save and save_dir:
        plt.savefig(f"{save_dir}/Timescale_Recovery.svg", bbox_inches='tight')
    plt.show()

    return dict(tau_E=all_tau_E, tau_I=all_tau_I,
                mean_tau_E=float(np.nanmean(all_tau_E)),
                mean_tau_I=float(np.nanmean(all_tau_I)),
                tE_vs_tI_p=tE_vs_tI_p,
                tauE_vs10_p=tauE_vs10_p, tauI_vs5_p=tauI_vs5_p)


# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
"""
from wout_contribution_analysis import (
    contribution_decomposition_analysis,
    wout_sign_analysis,
    timescale_analysis,
)

# Analysis 1+2: R² decomposition (most important)
decomp = contribution_decomposition_analysis(
    FM_tst_Graphs, FM_tst_W_outs, FM_tst_OutsNodes,
    FM_train_ResStates, Outs_O,
    thresh=0.05, save=True, save_dir=DataDir)

# Analysis 3: W_out sign consistency
sign_res, sign_agg = wout_sign_analysis(
    FM_tst_Graphs, FM_tst_W_outs, FM_tst_OutsNodes,
    thresh=0.05, save=True, save_dir=DataDir)

# Analysis 4: Timescale recovery (needs Train_Inputs)
# Train_Inputs = np.load('WC_Train_Inputs.npy')  # (B=6, T=200)
ts_res = timescale_analysis(
    FM_tst_Graphs, FM_tst_W_outs, FM_tst_OutsNodes,
    FM_train_ResStates, Train_Inputs, Outs_O,
    thresh=0.05, decay_window=30,
    save=True, save_dir=DataDir)
"""
