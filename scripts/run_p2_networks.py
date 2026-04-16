"""
Phase 2 — Network Construction
Builds 11 NetworkX graphs: 5 temporal + 6 issue-specific.
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
NETS   = os.path.join(ROOT, 'results', 'networks')
PLOTS  = os.path.join(ROOT, 'results', 'plots')
TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(NETS, exist_ok=True)

print("="*60)
print("PHASE 2: Network Construction")
print("="*60)

THETA       = 0.45   # Adjusted: max pairwise corr ~0.65; 0.70 gives isolates
THETA_ISSUE = 0.40   # Issue networks already denser

# ── External attributes ───────────────────────────────────────────
region_df  = pd.read_csv(os.path.join(EXT, 'un_regional_groups.csv'))
income_df  = pd.read_csv(os.path.join(EXT, 'wb_income_groups.csv'))
region_map = dict(zip(region_df['country'], region_df['region']))
income_map = dict(zip(income_df['country'], income_df['income_group']))

ERAS = ['full', 'cold_war', 'post_cw', 'post_9_11', 'recent']
ISSUE_MAP = {
    'me': 'israel_palestine', 'co': 'colonization',
    'hr': 'human_rights',     'di': 'disarmament',
    'nu': 'nuclear_weapons',  'ec': 'economic_development'
}
# Short codes → column names in votes_full.csv
ISSUE_CODES = {'me':'me','co':'co','hr':'hr','di':'di','nu':'nu','ec':'ec'}


def build_agreement_matrix(df, min_votes=5):
    """Pearson correlation on vote pivot (rcid x country)."""
    pivot = df.pivot_table(index='rcid', columns='country', values='v', aggfunc='first')
    pivot = pivot.dropna(axis=1, thresh=min_votes)
    corr  = pivot.corr(method='pearson', min_periods=min_votes)
    return corr


def build_graph(corr_matrix, theta, label=''):
    """Threshold correlation matrix into a weighted NetworkX graph."""
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


# ── Step 2.1+2.3: Build temporal agreement matrices + graphs ──────
print("\n[1/4] Building temporal agreement matrices and graphs...")
era_matrices = {}
era_graphs   = {}
stats_rows   = []

for era in ERAS:
    print(f"  Processing {era}...", end=' ', flush=True)
    vote_df = pd.read_csv(os.path.join(PROC, f'votes_{era}.csv'))
    mat = build_agreement_matrix(vote_df)
    era_matrices[era] = mat
    G = build_graph(mat, THETA, label=era)
    era_graphs[era] = G
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = nx.number_connected_components(G)
    gcc_size = len(max(nx.connected_components(G), key=len)) if m > 0 else 0
    stats_rows.append({'network': era, 'nodes': n, 'edges': m,
                       'density': round(nx.density(G), 5),
                       'avg_degree': round(2*m/n, 2) if n > 0 else 0,
                       'giant_component': gcc_size, 'components': comps})
    print(f"N={n}, M={m}, density={nx.density(G):.4f}")

    # Sanity check key countries
    for key in ['United States of America', 'China', 'Russian Federation',
                'Russia', 'India']:
        if key in G:
            print(f"    {key}: degree = {G.degree(key)}")

# ── Step 2.2: Threshold sweep plot ────────────────────────────────
print("\n[2/4] Generating threshold sweep plot...")
thresholds = np.arange(0.30, 0.96, 0.05)
sweep_rows = []
mat_full   = era_matrices['full']
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
axes[0].axvline(0.7, color='#f85149', ls='--', label='theta=0.7 (chosen)')
axes[0].set_xlabel('Threshold theta', color='white')
axes[0].set_ylabel('Giant component fraction', color='white')
axes[0].set_title('Giant Component vs Threshold', color='white')
axes[0].legend(facecolor='#161b22', labelcolor='white')

axes[1].plot(sweep_df['theta'], sweep_df['n_edges'], 's-', color='#3fb950', lw=2)
axes[1].axvline(0.7, color='#f85149', ls='--', label='theta=0.7')
axes[1].set_xlabel('Threshold theta', color='white')
axes[1].set_ylabel('Number of edges', color='white')
axes[1].set_title('Edge Count vs Threshold', color='white')
axes[1].legend(facecolor='#161b22', labelcolor='white')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'p2_threshold_sweep.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved: p2_threshold_sweep.png")

# ── Step 2.4: Issue-specific graphs ──────────────────────────────
print("\n[3/4] Building issue-specific graphs...")
full_df      = pd.read_csv(os.path.join(PROC, 'votes_full.csv'))
issue_graphs = {}

for short, col in ISSUE_CODES.items():
    if col not in full_df.columns:
        print(f"  WARNING: column '{col}' not found, skipping {short}")
        continue
    sub = full_df[full_df[col] == 1].copy()
    n_res = sub['rcid'].nunique()
    n_ctr = sub['country'].nunique()
    print(f"  issue_{short}: {n_res} resolutions, {n_ctr} countries", end=' ')
    if n_res < 5:
        print("(too few, skipping)")
        continue
    theta_iss = 0.25 if n_res < 20 else THETA_ISSUE
    mat_iss   = build_agreement_matrix(sub)
    G_iss     = build_graph(mat_iss, theta_iss, label=f'issue_{short}')
    issue_graphs[short] = G_iss
    n_i, m_i = G_iss.number_of_nodes(), G_iss.number_of_edges()
    gcc_i = len(max(nx.connected_components(G_iss), key=len)) if m_i > 0 else 0
    comps_i = nx.number_connected_components(G_iss)
    print(f"--> N={n_i}, M={m_i}")
    stats_rows.append({'network': f'issue_{short}', 'nodes': n_i, 'edges': m_i,
                       'density': round(nx.density(G_iss), 5),
                       'avg_degree': round(2*m_i/n_i, 2) if n_i > 0 else 0,
                       'giant_component': gcc_i, 'components': comps_i})

# ── Step 2.5: Export ──────────────────────────────────────────────
print("\n[4/4] Exporting all graphs...")
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

# ── Full network visualisation ────────────────────────────────────
print("\nGenerating full network visualisation...")
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
edge_weights = [G_full[u][v]['weight'] for u,v in G_full.edges]
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
ax.set_title('UNGA Voting Network - Full Era (1946-2015) | Colored by UN Regional Group',
             color='white', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'p2_full_network.png'), bbox_inches='tight',
            facecolor='#0d1117', dpi=150)
plt.close()
print("  Saved: p2_full_network.png")
print("\nPhase 2 complete!")
