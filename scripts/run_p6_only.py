"""
Phase 6: Dynamics & Synthesis
Run from NS-Project root: python scripts/run_p6_only.py
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

print("="*60); print("PHASE 6: Dynamics & Synthesis"); print("="*60)
G_full = load_g('full')
df_full = pd.read_csv(os.path.join(PROC,'votes_full.csv'))

# P6.3: Pivot events
print("Finding pivot events (D5)...")
session_year = df_full[['session','year']].drop_duplicates()
pivot_rows = []
for country in ['United States of America','Russian Federation','Russia','China','India']:
    cdf = df_full[df_full['country']==country]
    if cdf.empty: continue
    sess_means = cdf.groupby('session')['v'].agg(['mean','std','count']).reset_index()
    sess_means.columns = ['session','mean_vote','std_vote','n_votes']
    overall_mean = cdf['v'].mean(); overall_std = cdf['v'].std()
    sess_means['deviation'] = (sess_means['mean_vote'] - overall_mean).abs() / (overall_std + 1e-9)
    # Lower threshold to get at least some pivots
    pivots = sess_means[sess_means['deviation'] > 0.8].nlargest(5, 'deviation')
    if pivots.empty: 
        pivots = sess_means.nlargest(3, 'deviation')
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

if os.path.exists(os.path.join(TABLES,'p5_fragility_index_D3.csv')):
    frag_df = pd.read_csv(os.path.join(TABLES,'p5_fragility_index_D3.csv'))
    india_f = frag_df[frag_df['country']=='India']
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
