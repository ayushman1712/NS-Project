"""
Phase 2 — Network Construction
Builds 11 NetworkX graphs: 5 temporal + 6 issue-specific.
Uses pre-computed agreement S-scores from Harvard Dataverse (Voeten et al.)
for temporal networks, and Pearson correlation with per-issue θ sweep for
issue-specific networks (no Dataverse S-scores exist at issue level).
Run from NS-Project root: python scripts/run_p2_networks.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC   = os.path.join(ROOT, 'data', 'processed')
EXT    = os.path.join(ROOT, 'data', 'external')
DV     = os.path.join(ROOT, 'dataverse_files')
NETS   = os.path.join(ROOT, 'results', 'networks')
PLOTS  = os.path.join(ROOT, 'results', 'plots')
TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(NETS, exist_ok=True)

print("="*60)
print("PHASE 2: Network Construction")
print("="*60)

# ── Threshold for temporal networks (tuned on S-score distribution) ──
THETA       = 0.70   # S-scores range [0,1]; 0.70 is the literature standard
THETA_ISSUE = 0.55   # Issue networks use Pearson, sparser

ERAS = ['full', 'cold_war', 'post_cw', 'post_9_11', 'recent']
ERA_YEARS = {
    'full':     (1946, 2015),
    'cold_war': (1946, 1991),
    'post_cw':  (1991, 2001),
    'post_9_11':(2001, 2014),
    'recent':   (2014, 2015),
}
ISSUE_MAP = {
    'me': 'israel_palestine', 'co': 'colonization',
    'hr': 'human_rights',     'di': 'disarmament',
    'nu': 'nuclear_weapons',  'ec': 'economic_development'
}

# ── External attributes ───────────────────────────────────────────
region_df  = pd.read_csv(os.path.join(EXT, 'un_regional_groups.csv'))
income_df  = pd.read_csv(os.path.join(EXT, 'wb_income_groups.csv'))
region_map = dict(zip(region_df['country'], region_df['region']))
income_map = dict(zip(income_df['country'], income_df['income_group']))

# ── Load Dataverse S-scores + ccode→country mapping ──────────────
print("\n[0/5] Loading Dataverse agreement S-scores...")
agree_df = pd.read_csv(os.path.join(DV, 'AgreementScores.csv'),
                        usecols=['ccode1', 'ccode2', 'agree', 'year'])
votes_clean = pd.read_csv(os.path.join(PROC, 'votes_clean.csv'),
                           usecols=['ccode', 'country'])
ccode_map = dict(votes_clean.drop_duplicates('ccode')[['ccode','country']].values)
agree_df['country_a'] = agree_df['ccode1'].map(ccode_map)
agree_df['country_b'] = agree_df['ccode2'].map(ccode_map)
agree_df = agree_df.dropna(subset=['country_a', 'country_b', 'agree'])
print(f"  Loaded {len(agree_df):,} dyad-year S-scores, {agree_df['country_a'].nunique()} countries")
print(f"  S-score range: {agree_df['agree'].min():.3f} – {agree_df['agree'].max():.3f}")
print(f"  S-score mean: {agree_df['agree'].mean():.3f}, median: {agree_df['agree'].median():.3f}")


def build_agreement_matrix_sscore(agree_sub):
    """Build symmetric agreement matrix from Dataverse S-scores.
    Averages S-scores across all years in the subset for each dyad."""
    dyad_mean = agree_sub.groupby(['country_a', 'country_b'])['agree'].mean().reset_index()
    countries = sorted(set(dyad_mean['country_a']) | set(dyad_mean['country_b']))
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)
    for _, row in dyad_mean.iterrows():
        i, j = idx.get(row['country_a']), idx.get(row['country_b'])
        if i is not None and j is not None:
            mat[i, j] = row['agree']
            mat[j, i] = row['agree']
    return pd.DataFrame(mat, index=countries, columns=countries)


def build_agreement_matrix_pearson(df, min_votes=5):
    """Pearson correlation on vote pivot — used only for issue networks
    where Dataverse S-scores are not available at issue level."""
    pivot = df.pivot_table(index='rcid', columns='country', values='v', aggfunc='first')
    pivot = pivot.dropna(axis=1, thresh=min_votes)
    corr  = pivot.corr(method='pearson', min_periods=min_votes)
    return corr


def build_graph(corr_matrix, theta, label=''):
    """Threshold agreement matrix into a weighted NetworkX graph."""
    countries = corr_matrix.index.tolist()
    G = nx.Graph(name=label)
    for c in countries:
        G.add_node(c,
                   region=region_map.get(c, 'Unknown'),
                   income=income_map.get(c, 'Unknown'))
    n = len(countries)
    for i in range(n):
        for j in range(i+1, n):
            w = corr_matrix.iloc[i, j]
            if not np.isnan(w) and w >= theta:
                G.add_edge(countries[i], countries[j], weight=float(w))
    return G


def find_knee_theta(mat, thetas=np.arange(0.30, 0.96, 0.05)):
    """Find θ at the 'knee' where giant component fraction drops fastest."""
    countries = mat.columns.tolist()
    n = len(countries)
    fracs = []
    for theta in thetas:
        G = nx.Graph()
        G.add_nodes_from(countries)
        for i in range(n):
            for j in range(i+1, n):
                w = mat.iloc[i, j]
                if not np.isnan(w) and w >= theta:
                    G.add_edge(countries[i], countries[j])
        if G.number_of_edges() == 0:
            fracs.append(0)
        else:
            gcc = max(nx.connected_components(G), key=len)
            fracs.append(len(gcc) / n)
    # Knee = largest negative second derivative
    fracs = np.array(fracs)
    d2 = np.diff(fracs, n=2)
    knee_idx = np.argmin(d2) + 1  # +1 for diff offset
    return round(thetas[knee_idx], 2), fracs


# ── Step 2.1: Build temporal agreement matrices + graphs using S-scores
print("\n[1/5] Building temporal networks from Dataverse S-scores...")
era_matrices = {}
era_graphs   = {}
stats_rows   = []

for era in ERAS:
    y0, y1 = ERA_YEARS[era]
    print(f"  Processing {era} ({y0}–{y1})...", end=' ', flush=True)
    sub = agree_df[(agree_df['year'] >= y0) & (agree_df['year'] <= y1)]
    mat = build_agreement_matrix_sscore(sub)
    era_matrices[era] = mat
    G = build_graph(mat, THETA, label=era)
    era_graphs[era] = G
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = nx.number_connected_components(G)
    gcc_size = len(max(nx.connected_components(G), key=len)) if m > 0 else 0
    stats_rows.append({'network': era, 'nodes': n, 'edges': m,
                       'density': round(nx.density(G), 5),
                       'avg_degree': round(2*m/n, 2) if n > 0 else 0,
                       'giant_component': gcc_size, 'components': comps,
                       'method': 'Dataverse S-score', 'theta': THETA})
    print(f"N={n}, M={m}, density={nx.density(G):.4f}")

    for key in ['United States of America', 'China', 'Russian Federation',
                'Russia', 'India']:
        if key in G:
            print(f"    {key}: degree = {G.degree(key)}")

# Save agreement matrix for validation
mat_full = era_matrices['full']
mat_full.to_csv(os.path.join(PROC, 'agree_matrix_full.csv'))
print(f"  Saved: agree_matrix_full.csv ({mat_full.shape})")

# ── Step 2.2: Threshold sweep plot ────────────────────────────────
print("\n[2/5] Generating threshold sweep plot (S-scores)...")
thresholds = np.arange(0.30, 0.96, 0.05)
sweep_rows = []
countries_full = mat_full.columns.tolist()
n_full = len(countries_full)
for theta in thresholds:
    G_sw = nx.Graph()
    G_sw.add_nodes_from(countries_full)
    for i in range(n_full):
        for j in range(i+1, n_full):
            w = mat_full.iloc[i, j]
            if not np.isnan(w) and w >= theta:
                G_sw.add_edge(countries_full[i], countries_full[j])
    m_sw = G_sw.number_of_edges()
    if m_sw == 0:
        gf = 0
    else:
        gcc = max(nx.connected_components(G_sw), key=len)
        gf  = len(gcc) / n_full
    sweep_rows.append({'theta': round(theta, 2), 'n_edges': m_sw, 'giant_frac': gf})

sweep_df = pd.DataFrame(sweep_rows)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

axes[0].plot(sweep_df['theta'], sweep_df['giant_frac'], 'o-', color='#58a6ff', lw=2)
axes[0].axvline(THETA, color='#f85149', ls='--', label=f'θ={THETA} (chosen)')
axes[0].set_xlabel('Threshold θ (S-score)', color='white')
axes[0].set_ylabel('Giant component fraction', color='white')
axes[0].set_title('Giant Component vs Threshold', color='white')
axes[0].legend(facecolor='#161b22', labelcolor='white')

axes[1].plot(sweep_df['theta'], sweep_df['n_edges'], 's-', color='#3fb950', lw=2)
axes[1].axvline(THETA, color='#f85149', ls='--', label=f'θ={THETA}')
axes[1].set_xlabel('Threshold θ (S-score)', color='white')
axes[1].set_ylabel('Number of edges', color='white')
axes[1].set_title('Edge Count vs Threshold', color='white')
axes[1].legend(facecolor='#161b22', labelcolor='white')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'p2_threshold_sweep.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved: p2_threshold_sweep.png")

# ── Step 2.3: Issue-specific graphs with per-issue θ sweep ────────
print("\n[3/5] Building issue-specific graphs (Pearson + per-issue θ)...")
full_df      = pd.read_csv(os.path.join(PROC, 'votes_full.csv'))
issue_graphs = {}
issue_thetas = {}

for short, col in ISSUE_MAP.items():
    if short not in full_df.columns:
        print(f"  WARNING: column '{short}' not found, skipping")
        continue
    sub = full_df[full_df[short] == 1].copy()
    n_res = sub['rcid'].nunique()
    n_ctr = sub['country'].nunique()
    print(f"  issue_{short}: {n_res} resolutions, {n_ctr} countries", end=' ')
    if n_res < 5:
        print("(too few, skipping)")
        continue
    mat_iss = build_agreement_matrix_pearson(sub)
    # Per-issue θ sweep to find the knee
    theta_iss, _ = find_knee_theta(mat_iss)
    theta_iss = max(theta_iss, 0.30)  # floor at 0.30
    issue_thetas[short] = theta_iss
    G_iss = build_graph(mat_iss, theta_iss, label=f'issue_{short}')
    issue_graphs[short] = G_iss
    n_i, m_i = G_iss.number_of_nodes(), G_iss.number_of_edges()
    gcc_i = len(max(nx.connected_components(G_iss), key=len)) if m_i > 0 else 0
    comps_i = nx.number_connected_components(G_iss)
    print(f"θ={theta_iss} --> N={n_i}, M={m_i}")
    stats_rows.append({'network': f'issue_{short}', 'nodes': n_i, 'edges': m_i,
                       'density': round(nx.density(G_iss), 5),
                       'avg_degree': round(2*m_i/n_i, 2) if n_i > 0 else 0,
                       'giant_component': gcc_i, 'components': comps_i,
                       'method': 'Pearson', 'theta': theta_iss})

# ── Step 2.4: Export ──────────────────────────────────────────────
print("\n[4/5] Exporting all graphs...")
all_graphs = {**era_graphs, **{f'issue_{k}': v for k, v in issue_graphs.items()}}
for name, G in all_graphs.items():
    nx.write_graphml(G, os.path.join(NETS, f'{name}.graphml'))
    with open(os.path.join(NETS, f'{name}.pkl'), 'wb') as f:
        pickle.dump(G, f)
print(f"  Exported {len(all_graphs)} graphs to results/networks/")

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(TABLES, 'p2_network_stats.csv'), index=False)
print("\nNetwork statistics:")
print(stats_df.to_string(index=False))

# ── Step 2.5: Full network visualisation ──────────────────────────
print("\n[5/5] Generating full network visualisation...")
G_full = era_graphs['full']
REGION_COLORS = {
    'Africa':'#f85149','AsiaPacific':'#58a6ff','EasternEurope':'#3fb950',
    'GRULAC':'#d2a8ff','WEOG':'#ffa657','ArabStates':'#79c0ff','Unknown':'#8b949e'
}
node_colors = [REGION_COLORS.get(G_full.nodes[n].get('region','Unknown'),'#8b949e')
               for n in G_full.nodes]

print("  Computing spring layout...")
pos = nx.spring_layout(G_full, seed=42, k=0.3, iterations=50)

fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
nx.draw_networkx_edges(G_full, pos, ax=ax, alpha=0.25, width=0.5, edge_color='#8b949e')
nx.draw_networkx_nodes(G_full, pos, ax=ax, node_color=node_colors, node_size=30, alpha=0.92)
key_countries = {
    'United States of America':'USA','China':'China','Russian Federation':'Russia',
    'India':'India','Brazil':'Brazil','Germany':'Germany','South Africa':'S.Africa',
    'Nigeria':'Nigeria','Saudi Arabia':'S.Arabia','Japan':'Japan'
}
labels_dict = {n: lbl for n, lbl in key_countries.items() if n in G_full}
nx.draw_networkx_labels(G_full, pos, labels=labels_dict, ax=ax, font_size=7, font_color='white')
legend_handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items() if r != 'Unknown']
ax.legend(handles=legend_handles, loc='lower left', fontsize=8,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
ax.set_title('UNGA Voting Network — Full Era (1946–2015) | Dataverse S-scores, θ=0.70',
             color='white', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'p2_full_network.png'), bbox_inches='tight',
            facecolor='#0d1117', dpi=150)
plt.close()
print("  Saved: p2_full_network.png")
print("\nPhase 2 complete!")
