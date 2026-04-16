"""
Phases 3–6: Topology, Communities, Robustness, Dynamics
Run from NS-Project root: python scripts/run_p3_p6.py
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from itertools import combinations
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
import statsmodels.api as sm
import powerlaw
warnings.filterwarnings('ignore')

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC   = os.path.join(ROOT, 'data', 'processed')
EXT    = os.path.join(ROOT, 'data', 'external')
NETS   = os.path.join(ROOT, 'results', 'networks')
PLOTS  = os.path.join(ROOT, 'results', 'plots')
TABLES = os.path.join(ROOT, 'results', 'tables')

def load_g(name):
    with open(os.path.join(NETS, f'{name}.pkl'), 'rb') as f:
        return pickle.load(f)

ERAS   = ['full','cold_war','post_cw','post_9_11','recent']
ISSUES = ['me','co','hr','di','nu','ec']
ISSUE_NAMES = {'me':'Israel/Palestine','co':'Colonization','hr':'Human Rights',
                'di':'Disarmament','nu':'Nuclear','ec':'Economic Dev'}

region_df  = pd.read_csv(os.path.join(EXT, 'un_regional_groups.csv'))
income_df  = pd.read_csv(os.path.join(EXT, 'wb_income_groups.csv'))
region_map = dict(zip(region_df['country'], region_df['region']))
income_map = dict(zip(income_df['country'], income_df['income_group']))

plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22',
    'axes.edgecolor':'#30363d','text.color':'white',
    'axes.labelcolor':'white','xtick.color':'white','ytick.color':'white',
    'figure.dpi':150,'grid.color':'#30363d','grid.alpha':0.4
})

G_full = load_g('full')
print("="*60); print("PHASE 3: Topology"); print("="*60)

# ── P3.1: Degree distribution & power-law ────────────────────────
degrees = np.array([d for _,d in G_full.degree()])
fit = powerlaw.Fit(degrees[degrees>0], discrete=True, verbose=False)
gamma = fit.power_law.alpha
R, p_lr = fit.distribution_compare('power_law','exponential',normalized_ratio=True)
print(f"Power-law gamma = {gamma:.3f}")
print(f"LR test vs exponential: R={R:.3f}, p={p_lr:.4f}")

fig, axes = plt.subplots(1,2,figsize=(13,5))
fig.suptitle('Degree Distribution -- Full UNGA Network', color='white', fontsize=13)
axes[0].hist(degrees, bins=30, color='#58a6ff', alpha=0.8, edgecolor='#0d1117')
axes[0].set_xlabel('Degree k'); axes[0].set_ylabel('Count')
axes[0].set_title('Degree Histogram', color='white')
fit.plot_ccdf(ax=axes[1], color='#58a6ff', label='Empirical')
fit.power_law.plot_ccdf(ax=axes[1], color='#f85149', ls='--',
                        label=f'Power law (g={gamma:.2f})')
fit.exponential.plot_ccdf(ax=axes[1], color='#3fb950', ls=':', label='Exponential')
axes[1].set_title('Log-Log CCDF', color='white')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p3_degree_distribution.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Saved: p3_degree_distribution.png")

# ── P3.2: Small-world test ────────────────────────────────────────
gcc_nodes = max(nx.connected_components(G_full), key=len)
G_gcc  = G_full.subgraph(gcc_nodes).copy()
N, M   = G_gcc.number_of_nodes(), G_gcc.number_of_edges()
p_er   = M / (N*(N-1)/2)
C_real = nx.average_clustering(G_gcc)

# Approximate d_real by sampling
sample_n = min(30, N)
np.random.seed(42)
sn = list(np.random.choice(list(G_gcc.nodes), sample_n, replace=False))
lengths_real = []
for s in sn:
    spl = nx.single_source_shortest_path_length(G_gcc, s)
    lengths_real.extend(spl.values())
d_real = np.mean(lengths_real)

# ER baseline, 50 runs
n_runs = 50
C_rand_list, d_rand_list = [], []
for _ in range(n_runs):
    Gr  = nx.erdos_renyi_graph(N, p_er)
    gcc_r = max(nx.connected_components(Gr), key=len)
    Gcc_r = Gr.subgraph(gcc_r).copy()
    if Gcc_r.number_of_nodes() < 2: continue
    C_rand_list.append(nx.average_clustering(Gcc_r))
    snn = list(np.random.choice(list(Gcc_r.nodes), min(15,Gcc_r.number_of_nodes()), replace=False))
    ll = []
    for s in snn:
        spl = nx.single_source_shortest_path_length(Gcc_r, s)
        ll.extend(spl.values())
    d_rand_list.append(np.mean(ll) if ll else np.nan)

C_rand = np.mean(C_rand_list)
d_rand = np.nanmean(d_rand_list)
sigma  = (C_real/C_rand)/(d_real/d_rand)
print(f"Small-world: C_real={C_real:.4f}, C_rand={C_rand:.4f}, "
      f"d_real={d_real:.4f}, d_rand={d_rand:.4f}, sigma={sigma:.3f}")

sw_df = pd.DataFrame({'metric':['C','d','sigma'],
                      'real':[C_real,d_real,sigma],
                      'ER_mean':[C_rand,d_rand,1.0]})
sw_df.to_csv(os.path.join(TABLES,'p3_small_world.csv'), index=False)

# ── P3.3: Centrality ─────────────────────────────────────────────
print("Computing centrality...")
deg_c = nx.degree_centrality(G_full)
btw_c = nx.betweenness_centrality(G_full, weight='weight', normalized=True)
try:
    eig_c = nx.eigenvector_centrality(G_full, weight='weight', max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eig_c = {n:0.0 for n in G_full.nodes}

cent_df = pd.DataFrame({
    'country': list(deg_c.keys()),
    'degree_centrality': [deg_c[c] for c in deg_c],
    'betweenness_centrality': [btw_c[c] for c in deg_c],
    'eigenvector_centrality': [eig_c[c] for c in deg_c]
})
cent_df['degree_rank']      = cent_df['degree_centrality'].rank(ascending=False).astype(int)
cent_df['betweenness_rank'] = cent_df['betweenness_centrality'].rank(ascending=False).astype(int)
cent_df['eigenvector_rank'] = cent_df['eigenvector_centrality'].rank(ascending=False).astype(int)
cent_df.sort_values('betweenness_rank').to_csv(os.path.join(TABLES,'p3_centrality.csv'), index=False)

india_c = cent_df[cent_df['country']=='India']
print("India centrality ranks:")
print(india_c[['country','degree_rank','betweenness_rank','eigenvector_rank']].to_string(index=False))
print("Top 10 betweenness:")
print(cent_df.nsmallest(10,'betweenness_rank')[['country','betweenness_rank']].to_string(index=False))

# Centrality bar chart
highlight_names = ['United States of America','China','Russian Federation',
                   'India','Brazil','South Africa','Germany','Nigeria']
hdf = cent_df[cent_df['country'].isin(highlight_names)].copy()
short = {'United States of America':'USA','Russian Federation':'Russia',
         'South Africa':'S.Africa'}
hdf['sname'] = hdf['country'].map(lambda x: short.get(x,x))
hdf = hdf.sort_values('betweenness_centrality', ascending=False)

fig, axes = plt.subplots(1,3,figsize=(15,5))
fig.suptitle('Centrality Comparison -- Major Countries', color='white', fontsize=13)
for ax,(metric,label,color) in zip(axes, [
    ('degree_centrality','Degree','#58a6ff'),
    ('betweenness_centrality','Betweenness','#f85149'),
    ('eigenvector_centrality','Eigenvector','#3fb950')]):
    ax.barh(hdf['sname'], hdf[metric], color=color, alpha=0.85)
    ax.set_title(label+' Centrality', color='white')
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p3_centrality.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()

# ── P3.4: C(k) hierarchical ──────────────────────────────────────
colors_map = {'full':'#58a6ff','cold_war':'#f85149','post_cw':'#3fb950',
              'post_9_11':'#d2a8ff','recent':'#ffa657'}
hier_rows = []
fig, ax = plt.subplots(figsize=(9,6))
ax.set_title('C(k) vs k log-log -- Hierarchical Test', color='white', fontsize=12)
for era in ERAS:
    G = load_g(era)
    c_dict = nx.clustering(G)
    degs   = np.array([G.degree(n) for n in G.nodes])
    clusts = np.array([c_dict[n]   for n in G.nodes])
    ck_df  = pd.DataFrame({'k':degs,'c':clusts})
    ck     = ck_df[ck_df['k']>=2].groupby('k')['c'].mean()
    if len(ck) < 3: continue
    log_k = np.log10(ck.index.astype(float))
    log_c = np.log10(ck.values.clip(min=1e-6))
    valid = np.isfinite(log_k) & np.isfinite(log_c)
    slope = np.nan
    if valid.sum() >= 3:
        slope, interc, _, _, _ = scipy_stats.linregress(log_k[valid], log_c[valid])
        x_fit = np.linspace(log_k[valid].min(), log_k[valid].max(), 50)
        ax.plot(x_fit, interc + slope*x_fit, '--', color=colors_map[era], lw=1)
    ax.scatter(log_k, log_c, s=20, alpha=0.5, color=colors_map[era],
               label=f'{era} (slope={slope:.2f})' if not np.isnan(slope) else era)
    hier_rows.append({'era':era, 'ck_slope': round(slope,3) if not np.isnan(slope) else np.nan})

x_ref = np.linspace(0.5,2.2,50)
ax.plot(x_ref, -x_ref+0.3, 'w--', lw=1, alpha=0.3, label='slope=-1')
ax.set_xlabel('log10(k)'); ax.set_ylabel('log10 C(k)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p3_hierarchical_ck.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()

topo_rows = []
for era in ERAS:
    G = load_g(era)
    degs = [d for _,d in G.degree()]
    topo_rows.append({'era':era,'N':G.number_of_nodes(),'M':G.number_of_edges(),
                      'density':round(nx.density(G),4),'avg_degree':round(np.mean(degs),2),
                      'max_degree':max(degs),'clustering':round(nx.average_clustering(G),4),
                      'n_components':nx.number_connected_components(G)})
pd.DataFrame(topo_rows).to_csv(os.path.join(TABLES,'p3_topology.csv'), index=False)
pd.DataFrame(hier_rows).to_csv(os.path.join(TABLES,'p3_hierarchy_slopes.csv'), index=False)
print("Phase 3 done.")

print("\n"+"="*60); print("PHASE 4: Community Detection"); print("="*60)

def louvain_best(G, n_runs=10):
    best_Q, best_p = -1, None
    for seed in range(n_runs):
        p = community_louvain.best_partition(G, weight='weight', random_state=seed)
        Q = community_louvain.modularity(p, G, weight='weight')
        if Q > best_Q:
            best_Q, best_p = Q, p
    return best_p, best_Q

# P4.1: Full network Louvain
print("Louvain on full network...")
full_part, full_Q = louvain_best(G_full, 10)
n_comm = len(set(full_part.values()))
print(f"  Q={full_Q:.4f}, communities={n_comm}")

comm_members = {}
for c, cid in full_part.items():
    comm_members.setdefault(cid,[]).append(c)

community_labels = {}
for cid, members in comm_members.items():
    regions = [region_map.get(c,'Unknown') for c in members]
    dominant = pd.Series(regions).value_counts().idxmax()
    community_labels[cid] = f'C{cid}:{dominant}'
print("  Community labels:", community_labels)

part_df = pd.DataFrame([{'country':c,'community':cid,'community_label':community_labels[cid]}
                         for c,cid in full_part.items()])
part_df.to_csv(os.path.join(TABLES,'p4_louvain_full.csv'), index=False)
india_row = part_df[part_df['country']=='India']
print("  India:", india_row[['country','community','community_label']].to_string(index=False))

# Visualise communities
COMM_PALETTE = [
    '#58a6ff','#f85149','#3fb950','#d2a8ff','#ffa657',
    '#79c0ff','#ffa8a8','#56d364','#e3b341','#ff7b72'
]
node_colors_c = [COMM_PALETTE[full_part.get(n,0) % len(COMM_PALETTE)] for n in G_full.nodes]
pos = nx.spring_layout(G_full, seed=42, k=0.3, iterations=50)
fig, ax = plt.subplots(figsize=(15,11))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
nx.draw_networkx_edges(G_full, pos, ax=ax, alpha=0.12, width=0.4, edge_color='#8b949e')
nx.draw_networkx_nodes(G_full, pos, ax=ax, node_color=node_colors_c, node_size=35, alpha=0.95)
key_c = {n:n.split()[-1] for n in ['United States of America','China','Russian Federation',
          'India','Brazil','Germany','South Africa','Nigeria','Saudi Arabia'] if n in G_full}
nx.draw_networkx_labels(G_full, pos, labels=key_c, ax=ax, font_size=7, font_color='white')
legend_h = [mpatches.Patch(color=COMM_PALETTE[cid%len(COMM_PALETTE)], label=lbl)
            for cid,lbl in sorted(community_labels.items())]
ax.legend(handles=legend_h, loc='lower left', fontsize=7,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
ax.set_title(f'UNGA Full Network -- Louvain Communities (Q={full_Q:.3f})',
             color='white', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p4_communities_full.png'), bbox_inches='tight',
            facecolor='#0d1117', dpi=150)
plt.close()
print("  Saved: p4_communities_full.png")

# P4.2: Temporal Louvain
print("Louvain on temporal networks...")
era_partitions = {'full': full_part}
era_qs = {'full': full_Q}
for era in ['cold_war','post_cw','post_9_11','recent']:
    G_era = load_g(era)
    if G_era.number_of_edges() < 5: continue
    p, Q = louvain_best(G_era, 10)
    era_partitions[era] = p
    era_qs[era] = Q
    nc = len(set(p.values()))
    print(f"  {era}: Q={Q:.4f}, comms={nc}")
    pdf = pd.DataFrame([{'country':c,'community':cid} for c,cid in p.items()])
    pdf.to_csv(os.path.join(TABLES,f'p4_louvain_{era}.csv'), index=False)

# P4.3: Sankey (save data for plotly in notebooks; here generate a static PNG summary)
print("Generating community size heatmap across eras...")
era_order = ['cold_war','post_cw','post_9_11','recent']
valid_eras = [e for e in era_order if e in era_partitions]

if len(valid_eras) >= 2:
    # Build NMI
    nmi_rows = []
    pairs = [('cold_war','post_cw','CW->PCW'),
             ('post_cw','post_9_11','PCW->9/11'),
             ('post_9_11','recent','9/11->Recent')]
    for e1, e2, lbl in pairs:
        if e1 not in era_partitions or e2 not in era_partitions: continue
        p1, p2 = era_partitions[e1], era_partitions[e2]
        common = sorted(set(p1) & set(p2))
        if len(common) < 10: continue
        v1 = [p1[c] for c in common]; v2 = [p2[c] for c in common]
        nmi = normalized_mutual_info_score(v1, v2)
        nmi_rows.append({'transition':lbl,'NMI':round(nmi,4),
                         'interpretation':('stable' if nmi>0.7 else 'partial rupture' if nmi>0.4 else 'structural rupture')})
        print(f"  NMI {lbl}: {nmi:.4f}")
    nmi_df = pd.DataFrame(nmi_rows)
    nmi_df.to_csv(os.path.join(TABLES,'p4_nmi.csv'), index=False)

    # Sankey-like static alluvial plot
    fig, axes = plt.subplots(1, len(valid_eras), figsize=(14, 8))
    if len(valid_eras) == 1: axes = [axes]
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Community Composition Across Eras', color='white', fontsize=13)

    era_short = {'cold_war':'Cold War','post_cw':'Post-CW','post_9_11':'Post-9/11','recent':'Recent'}
    for idx, era in enumerate(valid_eras):
        p = era_partitions[era]
        comm_counts = pd.Series(p).value_counts().sort_index()
        ax = axes[idx]
        ax.set_facecolor('#161b22')
        bars = ax.bar(range(len(comm_counts)), comm_counts.values,
                      color=[COMM_PALETTE[c % len(COMM_PALETTE)] for c in comm_counts.index])
        ax.set_xticks(range(len(comm_counts)))
        ax.set_xticklabels([f'C{c}' for c in comm_counts.index], fontsize=8)
        ax.set_title(era_short.get(era, era), color='white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        if idx == 0: ax.set_ylabel('Countries', color='white')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,'p4_community_eras.png'), bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("  Saved: p4_community_eras.png")

# P4.5: Issue network Louvain -> India Multi-Alignment Matrix (D1)
print("Issue-network Louvain (India Multi-Alignment Matrix)...")
india_matrix = []
issue_partitions = {}
for issue in ISSUES:
    try:
        G_iss = load_g(f'issue_{issue}')
    except FileNotFoundError:
        continue
    if G_iss.number_of_edges() < 3: continue
    p, Q = louvain_best(G_iss, 10)
    issue_partitions[issue] = p
    india_comm = p.get('India', None)
    if india_comm is None:
        india_comm = next((v for k,v in p.items() if 'india' in k.lower()), None)
    top_covoters = []
    if 'India' in G_iss and india_comm is not None:
        nbrs = [(n, G_iss['India'][n]['weight']) for n in G_iss.neighbors('India')]
        top_covoters = sorted(nbrs, key=lambda x:-x[1])[:5]
    nc = len(set(p.values()))
    india_matrix.append({
        'issue': issue, 'issue_name': ISSUE_NAMES[issue],
        'n_communities': nc, 'modularity_Q': round(Q,4),
        'india_community': india_comm,
        'india_top_covoters': ', '.join([f'{c}({w:.2f})' for c,w in top_covoters])
    })
    print(f"  {issue}: Q={Q:.4f}, n_comms={nc}, India->C{india_comm}")

d1_df = pd.DataFrame(india_matrix)
d1_df.to_csv(os.path.join(TABLES,'p4_india_alignment_D1.csv'), index=False)
print("[D1] India Multi-Alignment Matrix saved.")
print(d1_df[['issue_name','india_community','n_communities','modularity_Q']].to_string(index=False))

# P4.6: Girvan-Newman bridge edges
print("Girvan-Newman bridge analysis (HR, Nuclear)...")
gn_rows = []
for issue in ['hr','nu']:
    try:
        G_iss = load_g(f'issue_{issue}')
    except FileNotFoundError:
        continue
    edge_btw = nx.edge_betweenness_centrality(G_iss, weight='weight', normalized=True)
    top_bridges = sorted(edge_btw.items(), key=lambda x:-x[1])[:5]
    print(f"  Top bridge edges for issue_{issue}:")
    for (u,v), btw in top_bridges:
        w = G_iss[u][v]['weight']
        gn_rows.append({'issue':issue,'country_a':u,'country_b':v,'betweenness':round(btw,5),'weight':round(w,4)})
        print(f"    {u[:25]:25s} -- {v[:25]:25s} | btw={btw:.4f}")
if gn_rows:
    pd.DataFrame(gn_rows).to_csv(os.path.join(TABLES,'p4_bridge_edges.csv'), index=False)

print("Phase 4 done.")

print("\n"+"="*60); print("PHASE 5: Robustness & Motifs"); print("="*60)

def attack_simulation(G, removal_order, stop_frac=0.5):
    H = G.copy(); N_orig = H.number_of_nodes()
    max_rm = int(stop_frac * N_orig); results = []
    for i, node in enumerate(removal_order):
        if i >= max_rm: break
        if node not in H: continue
        H.remove_node(node)
        if H.number_of_nodes() == 0:
            results.append({'removed_frac':(i+1)/N_orig,'giant_frac':0}); break
        if H.number_of_edges() == 0:
            results.append({'removed_frac':(i+1)/N_orig,'giant_frac':1/max(H.number_of_nodes(),1)})
        else:
            gcc = max(nx.connected_components(H), key=len)
            results.append({'removed_frac':(i+1)/N_orig,'giant_frac':len(gcc)/N_orig})
    return pd.DataFrame(results)

def run_robustness(G, n_rand=30, stop_frac=0.5):
    btw = nx.betweenness_centrality(G, weight='weight', normalized=True)
    t_order = sorted(btw, key=btw.get, reverse=True)
    t_df    = attack_simulation(G, t_order, stop_frac)
    nodes   = list(G.nodes)
    x_grid  = np.linspace(0, stop_frac, 80)
    traces  = np.zeros((n_rand, len(x_grid)))
    for i in range(n_rand):
        r_ord = np.random.permutation(nodes).tolist()
        r_df  = attack_simulation(G, r_ord, stop_frac)
        if r_df.empty: continue
        traces[i] = np.interp(x_grid, r_df['removed_frac'], r_df['giant_frac'], left=1.0, right=0.0)
    return t_df, x_grid, traces.mean(axis=0), traces.std(axis=0), t_order

print("Running robustness on full network...")
t_df, x_g, r_mean, r_std, t_order = run_robustness(G_full, n_rand=30)

fig, ax = plt.subplots(figsize=(10,6))
ax.set_title('Robustness: Full UNGA Network', color='white', fontsize=13)
ax.plot(t_df['removed_frac'], t_df['giant_frac'], 'o-', color='#f85149', lw=2, ms=4, label='Targeted (betweenness)')
ax.plot(x_g, r_mean, '-', color='#3fb950', lw=2, label='Random (mean)')
ax.fill_between(x_g, r_mean-r_std, r_mean+r_std, alpha=0.2, color='#3fb950', label='+-1 SD')
ax.set_xlabel('Fraction removed'); ax.set_ylabel('Giant component fraction')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0,0.5); ax.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p5_robustness_full.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()

# P5.3: Temporal robustness comparison plot
temporal_eras = ['cold_war','post_cw','post_9_11','recent']
era_labels_map = {'cold_war':'Cold War','post_cw':'Post-CW',
                  'post_9_11':'Post-9/11','recent':'Recent'}
fig, axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle('Robustness Across Eras', color='white', fontsize=14)
axes_flat = axes.flatten()
era_rob = {}
for idx, era in enumerate(temporal_eras):
    G_era = load_g(era)
    if G_era.number_of_edges() < 5: continue
    print(f"  Robustness for {era}...", flush=True)
    t_e, x_e, rm_e, rs_e, to_e = run_robustness(G_era, n_rand=20)
    era_rob[era] = {'targeted':t_e,'rand_x':x_e,'rand_mean':rm_e,'targeted_order':to_e}
    ax = axes_flat[idx]
    ax.set_facecolor('#161b22')
    ax.plot(t_e['removed_frac'], t_e['giant_frac'], 'o-', color='#f85149', lw=2, ms=3, label='Targeted')
    ax.plot(x_e, rm_e, '-', color='#3fb950', lw=2, label='Random')
    ax.fill_between(x_e, rm_e-rs_e, rm_e+rs_e, alpha=0.2, color='#3fb950')
    ax.set_title(era_labels_map[era], color='white')
    ax.set_xlabel('Fraction removed'); ax.set_ylabel('Giant component')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0,0.5); ax.set_ylim(0,1.05)
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p5_robustness_eras.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()

# P5.4: Superpower Fragility Index
print("Computing Superpower Fragility Index (D3)...")
N_orig = G_full.number_of_nodes()
gcc_orig = max(nx.connected_components(G_full), key=len)
S_orig = len(gcc_orig) / N_orig
frag_rows = []
for i, node in enumerate(G_full.nodes):
    H = G_full.copy(); H.remove_node(node)
    if H.number_of_nodes() == 0: dS = S_orig
    elif H.number_of_edges() == 0: dS = S_orig - 1/max(H.number_of_nodes(),1)
    else:
        gcc_new = max(nx.connected_components(H), key=len)
        dS = S_orig - len(gcc_new)/N_orig
    frag_rows.append({'country':node,'delta_S':dS})
    if (i+1) % 50 == 0: print(f"  {i+1}/{N_orig}", flush=True)

frag_df = pd.DataFrame(frag_rows).sort_values('delta_S', ascending=False)
frag_df['rank'] = range(1, len(frag_df)+1)
frag_df.to_csv(os.path.join(TABLES,'p5_fragility_index_D3.csv'), index=False)
print("Top 15 Fragility Index:")
print(frag_df.head(15).to_string(index=False))
india_f = frag_df[frag_df['country']=='India']
print(f"\nIndia: rank={india_f['rank'].values[0]}, delta_S={india_f['delta_S'].values[0]:.5f}")

top10 = frag_df.head(10)
names_short = {'United States of America':'USA','Russian Federation':'Russia','South Africa':'S.Africa'}
t10_names = [names_short.get(c,c) for c in top10['country']]
colors_fr = plt.cm.YlOrRd(np.linspace(0.9,.3,10))
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title('Superpower Fragility Index -- Top 10 (D3)', color='white', fontsize=13)
bars = ax.barh(t10_names[::-1], top10['delta_S'].values[::-1], color=colors_fr)
ax.set_xlabel('Delta S (drop in giant component)')
for bar, val in zip(bars, top10['delta_S'].values[::-1]):
    ax.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8, color='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p5_fragility_index.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()

# P5.5: Frustrated triangles (sample)
print("Frustrated triangle motif search (sample)...")
sample_nodes = list(list(max(nx.connected_components(G_full),key=len))[:60])
G_samp = G_full.subgraph(sample_nodes).copy()
n_frust = sum(1 for a,b,c in combinations(G_samp.nodes,3)
              if (G_samp.has_edge(a,b)+G_samp.has_edge(b,c)+G_samp.has_edge(a,c))==2)
rand_fr = []
for _ in range(30):
    try:
        H = G_samp.copy()
        nx.double_edge_swap(H, nswap=5*H.number_of_edges(), max_tries=5*H.number_of_edges()*20)
        rf = sum(1 for a,b,c in combinations(H.nodes,3)
                 if (H.has_edge(a,b)+H.has_edge(b,c)+H.has_edge(a,c))==2)
        rand_fr.append(rf)
    except Exception: pass

if rand_fr:
    mean_r = np.mean(rand_fr); std_r = np.std(rand_fr)
    z_score = (n_frust - mean_r) / (std_r + 1e-9)
    print(f"  Frustrated: real={n_frust}, rand={mean_r:.1f}+-{std_r:.1f}, Z={z_score:.3f}")
    pd.DataFrame({'metric':['real','rand_mean','rand_std','z_score'],
                  'value':[n_frust,mean_r,std_r,z_score]}).to_csv(
        os.path.join(TABLES,'p5_frustrated_motifs.csv'), index=False)

print("Phase 5 done.")

print("\n"+"="*60); print("PHASE 6: Dynamics & Synthesis"); print("="*60)

df_full = pd.read_csv(os.path.join(PROC,'votes_full.csv'))

# P6.1+P6.2: Homophily OLS
print("Homophily regression (D4)...")
GDP = {
    'United States of America':20936,'China':14723,'Japan':5065,'Germany':3806,
    'India':2709,'United Kingdom':2708,'United Kingdom of Great Britain and Northern Ireland':2708,
    'France':2716,'Italy':1886,'Canada':1644,'Republic of Korea':1631,'South Korea':1631,
    'Russian Federation':1478,'Russia':1478,'Brazil':1445,'Australia':1330,
    'Spain':1281,'Mexico':1090,'Indonesia':1058,'Netherlands':910,
    'Saudi Arabia':703,'Turkey':720,'Turkiye':720,'Switzerland':703,
    'Argentina':383,'Poland':594,'Belgium':524,'Sweden':537,
    'Norway':363,'Israel':402,'South Africa':335,'Egypt':363,
    'Pakistan':263,'Bangladesh':302,'Nigeria':432,'Kenya':98,'Ethiopia':96,
}

def make_edge_df(G):
    rows = []
    for u,v,data in G.edges(data=True):
        sr = int(region_map.get(u)==region_map.get(v) and region_map.get(u) is not None)
        si = int(income_map.get(u)==income_map.get(v) and income_map.get(u) is not None)
        gu,gv = GDP.get(u,np.nan),GDP.get(v,np.nan)
        lgr = abs(np.log(gu/gv)) if (not np.isnan(gu) and not np.isnan(gv) and gu>0 and gv>0) else np.nan
        rows.append({'country_a':u,'country_b':v,'agreement':data.get('weight',np.nan),
                     'same_region':sr,'same_income':si,'log_gdp_ratio':lgr})
    return pd.DataFrame(rows)

def run_ols(df_e, label=''):
    sub = df_e.dropna(subset=['agreement','same_region','same_income','log_gdp_ratio'])
    if len(sub) < 20: return None
    y = sub['agreement']
    X = sub[['same_region','same_income','log_gdp_ratio']]
    X_std = (X - X.mean()) / (X.std() + 1e-9)
    X_sm  = sm.add_constant(X_std)
    model = sm.OLS(y, X_sm).fit()
    return model

beta_rows = []
for era in ['full','cold_war','post_cw','post_9_11','recent']:
    G_e   = load_g(era)
    e_df  = make_edge_df(G_e)
    model = run_ols(e_df, era)
    if model is None: continue
    params, pvals = model.params, model.pvalues
    beta_rows.append({
        'era':era,'R2':round(model.rsquared,4),
        'b_same_region':   round(params.get('same_region',np.nan),4),
        'b_same_income':   round(params.get('same_income',np.nan),4),
        'b_log_gdp_ratio': round(params.get('log_gdp_ratio',np.nan),4),
        'p_same_region':   round(pvals.get('same_region',np.nan),4),
        'p_same_income':   round(pvals.get('same_income',np.nan),4),
    })
    print(f"  {era}: R2={round(model.rsquared,4)}, b_region={round(params.get('same_region',np.nan),4)}, "
          f"b_income={round(params.get('same_income',np.nan),4)}")

beta_df = pd.DataFrame(beta_rows)
beta_df.to_csv(os.path.join(TABLES,'p6_homophily_D4.csv'), index=False)

# Beta over time plot
temporal_betas = beta_df[beta_df['era'].isin(['cold_war','post_cw','post_9_11','recent'])].copy()
if len(temporal_betas) >= 2:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title('Homophily Beta Coefficients Over Time (D4)', color='white', fontsize=13)
    era_x = range(len(temporal_betas))
    ax.plot(era_x, temporal_betas['b_same_region'], 'o-', color='#58a6ff', lw=2, ms=8, label='beta(same region)')
    ax.plot(era_x, temporal_betas['b_same_income'], 's-', color='#f85149', lw=2, ms=8, label='beta(same income)')
    ax.plot(era_x, temporal_betas['b_log_gdp_ratio'], '^-', color='#3fb950', lw=2, ms=8, label='beta(log GDP ratio)')
    ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
    ax.set_xticks(list(era_x))
    ax.set_xticklabels(['Cold War','Post-CW','Post-9/11','Recent'][:len(temporal_betas)])
    ax.set_ylabel('Standardised beta')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,'p6_homophily_betas_D4.png'), bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("  Saved: p6_homophily_betas_D4.png")

# P6.3: Pivot events
print("Finding pivot events (D5)...")
session_year = df_full[['session','year']].drop_duplicates()
pivot_rows = []
for country in ['United States of America','Russian Federation','China','India']:
    cdf = df_full[df_full['country']==country]
    if cdf.empty: continue
    sess_means = cdf.groupby('session')['v'].agg(['mean','std','count']).reset_index()
    sess_means.columns = ['session','mean_vote','std_vote','n_votes']
    overall_mean = cdf['v'].mean(); overall_std = cdf['v'].std()
    sess_means['deviation'] = (sess_means['mean_vote'] - overall_mean).abs() / (overall_std + 1e-9)
    pivots = sess_means[sess_means['deviation'] > 1.5].nlargest(5, 'deviation')
    pivots = pivots.merge(session_year, on='session', how='left')
    pivots['country'] = country
    for _, r in pivots.iterrows():
        pivot_rows.append({'country':country,'session':int(r['session']),
                           'year':r.get('year',np.nan),'mean_vote':round(r['mean_vote'],3),
                           'deviation':round(r['deviation'],3)})

pivot_df = pd.DataFrame(pivot_rows)
pivot_df.to_csv(os.path.join(TABLES,'p6_pivot_events_D5.csv'), index=False)
print(pivot_df.to_string(index=False))

# P6.4: Cascade radius (simplified)
print("Cascade radius analysis (D5)...")
sessions = sorted(df_full['session'].unique())
gcc_sub_nodes = list(max(nx.connected_components(G_full), key=len))[:80]
G_sub = G_full.subgraph(gcc_sub_nodes).copy()

cascade_rows = []
if not pivot_df.empty:
    for _, prow in pivot_df[pivot_df['country']=='India'].head(3).iterrows():
        country = prow['country']
        sess    = int(prow['session'])
        if country not in G_sub: continue
        idx = list(sessions).index(sess) if sess in sessions else -1
        if idx < 0 or idx+1 >= len(sessions): continue
        next_sess = sessions[idx+1]
        v_s  = df_full[(df_full['country']==country)&(df_full['session']==sess)]['v'].mean()
        v_s1 = df_full[(df_full['country']==country)&(df_full['session']==next_sess)]['v'].mean()
        delta = v_s1 - v_s
        for hop in range(1,4):
            hop_nodes = [n for n in G_sub if n != country and
                         nx.has_path(G_sub,country,n) and
                         nx.shortest_path_length(G_sub,country,n)==hop]
            updated = total = 0
            for nb in hop_nodes:
                vn0 = df_full[(df_full['country']==nb)&(df_full['session']==sess)]['v'].mean()
                vn1 = df_full[(df_full['country']==nb)&(df_full['session']==next_sess)]['v'].mean()
                if np.isnan(vn0) or np.isnan(vn1): continue
                total += 1
                if delta != 0 and (vn1-vn0)*delta > 0: updated += 1
            cascade_rows.append({'country':country,'session':sess,'hop':hop,
                                 'n_countries':total,'updated':updated,
                                 'frac_updated':updated/total if total>0 else np.nan})

cascade_df = pd.DataFrame(cascade_rows)
cascade_df.to_csv(os.path.join(TABLES,'p6_cascade_D5.csv'), index=False)
print(cascade_df.to_string(index=False))

if not cascade_df.empty:
    fig, ax = plt.subplots(figsize=(9,5))
    ax.set_title('Vote Contagion Decay Curve (D5)', color='white', fontsize=13)
    for (country,sess), grp in cascade_df.groupby(['country','session']):
        ax.plot(grp['hop'], grp['frac_updated'], 'o-', ms=7, lw=2, label=f'{country} s{sess}')
    ax.set_xlabel('Network hops from pivot country')
    ax.set_ylabel('Fraction updating vote')
    ax.set_xticks([1,2,3]); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS,'p6_cascade_D5.png'), bbox_inches='tight', facecolor='#0d1117')
    plt.close()

# P6.5: India era ranks
print("India centrality ranks across eras...")
era_rank_rows = []
for era in ERAS:
    G_e = load_g(era)
    if 'India' not in G_e:
        era_rank_rows.append({'era':era,'india_btw_rank':np.nan,'india_deg_rank':np.nan})
        continue
    btw = nx.betweenness_centrality(G_e, weight='weight', normalized=True)
    deg = nx.degree_centrality(G_e)
    sb  = sorted(btw, key=btw.get, reverse=True)
    sd  = sorted(deg, key=deg.get, reverse=True)
    br  = sb.index('India')+1
    dr  = sd.index('India')+1
    era_rank_rows.append({'era':era,'india_btw_rank':br,'india_deg_rank':dr})
    print(f"  {era:15s}: btw_rank={br:>4}, deg_rank={dr:>4}")

era_rank_df = pd.DataFrame(era_rank_rows)
era_rank_df.to_csv(os.path.join(TABLES,'p6_india_era_ranks.csv'), index=False)

fig, ax = plt.subplots(figsize=(9,5))
ax.set_title("India's Centrality Rank Across Eras", color='white', fontsize=13)
era_xs = range(len(era_rank_df))
era_ns = ['Full','CW','Post-CW','Post-9/11','Recent']
valid_btw = era_rank_df['india_btw_rank'].fillna(990)
valid_deg = era_rank_df['india_deg_rank'].fillna(990)
ax.plot(era_xs, valid_btw.values, 'o-', color='#58a6ff', lw=2, ms=8, label='Betweenness rank')
ax.plot(era_xs, valid_deg.values, 's-', color='#f85149', lw=2, ms=8, label='Degree rank')
ax.invert_yaxis()
ax.set_xticks(list(era_xs)); ax.set_xticklabels(era_ns[:len(era_rank_df)])
ax.set_ylabel('Rank (lower = more central)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'p6_india_ranks.png'), bbox_inches='tight', facecolor='#0d1117')
plt.close()

# P6.6: India Verdict table
print("Building India verdict table...")
verdict_rows = []
if 'india_btw_rank' in era_rank_df.columns:
    full_row = era_rank_df[era_rank_df['era']=='full']
    if not full_row.empty:
        verdict_rows.append(['Betweenness rank (full)', full_row['india_btw_rank'].values[0], 'lower=more central'])
        verdict_rows.append(['Degree rank (full)', full_row['india_deg_rank'].values[0], 'lower=more connected'])

if not india_f.empty:
    verdict_rows.append(['Fragility rank (D3)', india_f['rank'].values[0],
                         f"delta_S={india_f['delta_S'].values[0]:.5f}"])

if os.path.exists(os.path.join(TABLES,'p4_india_alignment_D1.csv')):
    d1 = pd.read_csv(os.path.join(TABLES,'p4_india_alignment_D1.csv'))
    for _, r in d1.iterrows():
        verdict_rows.append([f"Community: {r['issue_name']}", f"C{r['india_community']}", f"Q={r['modularity_Q']}"])

if os.path.exists(os.path.join(TABLES,'p4_nmi.csv')):
    nmi_d = pd.read_csv(os.path.join(TABLES,'p4_nmi.csv'))
    for _, r in nmi_d.iterrows():
        verdict_rows.append([f"NMI {r['transition']}", r['NMI'], r['interpretation']])

verdict_df = pd.DataFrame(verdict_rows, columns=['Metric','India Value','Notes'])
verdict_df.to_csv(os.path.join(TABLES,'p6_india_verdict.csv'), index=False)
print("="*60)
print("INDIA VERDICT TABLE"); print("="*60)
print(verdict_df.to_string(index=False))

print("\nAll phases complete! Summary of outputs:")
import glob
for t in sorted(glob.glob(os.path.join(TABLES,'*.csv'))):
    print(f"  TABLE: {os.path.basename(t)}")
for p in sorted(glob.glob(os.path.join(PLOTS,'*.png'))):
    print(f"  PLOT:  {os.path.basename(p)}")
