"""
Phase 1 — Data Preparation
Cleans votes, merges metadata, creates temporal slices, saves external attributes.
Run from NS-Project root: python scripts/run_p1_data_prep.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW    = os.path.join(ROOT, 'kaggle_files')
DV     = os.path.join(ROOT, 'dataverse_files')
PROC   = os.path.join(ROOT, 'data', 'processed')
EXT    = os.path.join(ROOT, 'data', 'external')
PLOTS  = os.path.join(ROOT, 'results', 'plots')
TABLES = os.path.join(ROOT, 'results', 'tables')

for d in [PROC, EXT, PLOTS, TABLES]:
    os.makedirs(d, exist_ok=True)

print("="*60)
print("PHASE 1: Data Preparation")
print("="*60)

# ── Load ──────────────────────────────────────────────────────────
print("\n[1/6] Loading raw files...")
votes_raw = pd.read_csv(os.path.join(RAW, 'votes.csv'))
res_raw   = pd.read_csv(os.path.join(RAW, 'resolutions.csv'))
states_raw = pd.read_csv(os.path.join(RAW, 'states.csv'))
print(f"  votes_raw:  {votes_raw.shape}")
print(f"  res_raw:    {res_raw.shape}")
print(f"  states_raw: {states_raw.shape}")

# ── Session→year mapping ──────────────────────────────────────────
session_year = states_raw[['assembly_session','year']].drop_duplicates()

ISSUES = {
    'co': 'colonization',
    'hr': 'human_rights',
    'me': 'israel_palestine',
    'di': 'disarmament',
    'nu': 'nuclear_weapons',
    'ec': 'economic_development'
}
ISSUE_COLS = list(ISSUES.values())

# ── Clean & recode ────────────────────────────────────────────────
print("\n[2/6] Cleaning and recoding votes...")
df = votes_raw.merge(session_year, on='assembly_session', how='left')
df = df[df['vote'].isin([1, 2, 3])].copy()
vote_recode = {1: 1, 2: 0, 3: -1}
df['v'] = df['vote'].map(vote_recode)

# Merge issue flags
df = df.merge(res_raw[['vote_id'] + [c for c in ISSUE_COLS if c in res_raw.columns]],
              on='vote_id', how='left')

# Rename
rename_map = {
    'vote_id': 'rcid', 'state_code': 'ccode', 'state_name': 'country',
    'assembly_session': 'session', 'significant_vote': 'importantvote',
    'colonization': 'co', 'human_rights': 'hr', 'israel_palestine': 'me',
    'disarmament': 'di', 'nuclear_weapons': 'nu', 'economic_development': 'ec'
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

for col in ['co','hr','me','di','nu','ec']:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

print(f"  After cleaning: {df.shape}")
print(f"  Countries: {df['country'].nunique()}")
print(f"  Resolutions: {df['rcid'].nunique()}")
print(f"  Year range: {df['year'].min()} – {df['year'].max()}")

MASTER = os.path.join(PROC, 'votes_clean.csv')
df.to_csv(MASTER, index=False)
print(f"  Saved master: {MASTER}")

# ── Temporal slices ───────────────────────────────────────────────
print("\n[3/6] Creating temporal slices...")
ERAS = {
    'full':     (1946, 2015),
    'cold_war': (1946, 1991),
    'post_cw':  (1991, 2001),
    'post_9_11':(2001, 2014),
    'recent':   (2014, 2015),
}

summary_rows = []
for name, (y0, y1) in ERAS.items():
    sub = df[(df['year'] >= y0) & (df['year'] <= y1)].copy()
    out = os.path.join(PROC, f'votes_{name}.csv')
    sub.to_csv(out, index=False)
    row = {
        'era': name, 'years': f'{y0}–{y1}',
        'n_rows': len(sub),
        'n_countries': sub['country'].nunique(),
        'n_resolutions': sub['rcid'].nunique()
    }
    summary_rows.append(row)
    print(f"  {name:15s}: {row['n_rows']:>9,} rows | "
          f"{row['n_countries']:>4} countries | "
          f"{row['n_resolutions']:>5} resolutions")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TABLES, 'p1_era_summary.csv'), index=False)

# ── UN Regional Groups ────────────────────────────────────────────
print("\n[4/6] Saving UN regional groups...")
un_regional_groups = {
    # African Group
    'Algeria':'Africa','Angola':'Africa','Benin':'Africa','Botswana':'Africa',
    'Burkina Faso':'Africa','Burundi':'Africa','Cabo Verde':'Africa','Cameroon':'Africa',
    'Central African Republic':'Africa','Chad':'Africa','Comoros':'Africa',
    "Côte d'Ivoire":'Africa','Djibouti':'Africa','Egypt':'Africa',
    'Equatorial Guinea':'Africa','Eritrea':'Africa','Eswatini':'Africa',
    'Ethiopia':'Africa','Gabon':'Africa','Gambia':'Africa','Ghana':'Africa',
    'Guinea':'Africa','Guinea-Bissau':'Africa','Kenya':'Africa','Lesotho':'Africa',
    'Liberia':'Africa','Libya':'Africa','Madagascar':'Africa','Malawi':'Africa',
    'Mali':'Africa','Mauritania':'Africa','Mauritius':'Africa','Morocco':'Africa',
    'Mozambique':'Africa','Namibia':'Africa','Niger':'Africa','Nigeria':'Africa',
    'Rwanda':'Africa','Sao Tome and Principe':'Africa','Senegal':'Africa',
    'Seychelles':'Africa','Sierra Leone':'Africa','Somalia':'Africa',
    'South Africa':'Africa','South Sudan':'Africa','Sudan':'Africa',
    'Togo':'Africa','Tunisia':'Africa','Uganda':'Africa',
    'United Republic of Tanzania':'Africa','Tanzania':'Africa',
    'Zambia':'Africa','Zimbabwe':'Africa',
    # Asia-Pacific
    'Afghanistan':'AsiaPacific','Bangladesh':'AsiaPacific','Bhutan':'AsiaPacific',
    'Cambodia':'AsiaPacific','China':'AsiaPacific','Fiji':'AsiaPacific',
    'India':'AsiaPacific','Indonesia':'AsiaPacific','Japan':'AsiaPacific',
    'Kazakhstan':'AsiaPacific','Kyrgyzstan':'AsiaPacific',
    'Lao People Democratic Republic':'AsiaPacific',
    'Malaysia':'AsiaPacific','Maldives':'AsiaPacific','Marshall Islands':'AsiaPacific',
    'Mongolia':'AsiaPacific','Myanmar':'AsiaPacific','Nauru':'AsiaPacific',
    'Nepal':'AsiaPacific','New Zealand':'AsiaPacific','Pakistan':'AsiaPacific',
    'Palau':'AsiaPacific','Papua New Guinea':'AsiaPacific','Philippines':'AsiaPacific',
    'Republic of Korea':'AsiaPacific','South Korea':'AsiaPacific',
    'Samoa':'AsiaPacific','Singapore':'AsiaPacific','Solomon Islands':'AsiaPacific',
    'Sri Lanka':'AsiaPacific','Tajikistan':'AsiaPacific','Thailand':'AsiaPacific',
    'Timor-Leste':'AsiaPacific','Tonga':'AsiaPacific','Turkmenistan':'AsiaPacific',
    'Tuvalu':'AsiaPacific','Uzbekistan':'AsiaPacific','Vanuatu':'AsiaPacific',
    'Viet Nam':'AsiaPacific','Vietnam':'AsiaPacific','Australia':'AsiaPacific',
    "Iran (Islamic Republic of)":'AsiaPacific','Iran':'AsiaPacific',
    'Democratic People Republic of Korea':'AsiaPacific','North Korea':'AsiaPacific',
    # Eastern European
    'Albania':'EasternEurope','Armenia':'EasternEurope','Azerbaijan':'EasternEurope',
    'Belarus':'EasternEurope','Bosnia and Herzegovina':'EasternEurope',
    'Bulgaria':'EasternEurope','Croatia':'EasternEurope','Czechia':'EasternEurope',
    'Czech Republic':'EasternEurope',
    'Estonia':'EasternEurope','Georgia':'EasternEurope','Hungary':'EasternEurope',
    'Latvia':'EasternEurope','Lithuania':'EasternEurope','Moldova':'EasternEurope',
    'Republic of Moldova':'EasternEurope',
    'Montenegro':'EasternEurope','North Macedonia':'EasternEurope',
    'Poland':'EasternEurope','Romania':'EasternEurope','Russia':'EasternEurope',
    'Russian Federation':'EasternEurope',
    'Serbia':'EasternEurope','Slovakia':'EasternEurope','Slovenia':'EasternEurope',
    'Ukraine':'EasternEurope',
    # GRULAC
    'Antigua and Barbuda':'GRULAC','Argentina':'GRULAC','Bahamas':'GRULAC',
    'Barbados':'GRULAC','Belize':'GRULAC','Bolivia':'GRULAC','Brazil':'GRULAC',
    'Chile':'GRULAC','Colombia':'GRULAC','Costa Rica':'GRULAC','Cuba':'GRULAC',
    'Dominica':'GRULAC','Dominican Republic':'GRULAC','Ecuador':'GRULAC',
    'El Salvador':'GRULAC','Grenada':'GRULAC','Guatemala':'GRULAC','Guyana':'GRULAC',
    'Haiti':'GRULAC','Honduras':'GRULAC','Jamaica':'GRULAC','Mexico':'GRULAC',
    'Nicaragua':'GRULAC','Panama':'GRULAC','Paraguay':'GRULAC','Peru':'GRULAC',
    'Saint Kitts and Nevis':'GRULAC','Saint Lucia':'GRULAC',
    'Saint Vincent and the Grenadines':'GRULAC','Suriname':'GRULAC',
    'Trinidad and Tobago':'GRULAC','Uruguay':'GRULAC','Venezuela':'GRULAC',
    'Bolivarian Republic of Venezuela':'GRULAC',
    # WEOG
    'Andorra':'WEOG','Austria':'WEOG','Belgium':'WEOG','Canada':'WEOG',
    'Cyprus':'WEOG','Denmark':'WEOG','Finland':'WEOG','France':'WEOG',
    'Germany':'WEOG','Greece':'WEOG','Iceland':'WEOG','Ireland':'WEOG',
    'Israel':'WEOG','Italy':'WEOG','Liechtenstein':'WEOG','Luxembourg':'WEOG',
    'Malta':'WEOG','Monaco':'WEOG','Netherlands':'WEOG','Norway':'WEOG',
    'Portugal':'WEOG','San Marino':'WEOG','Spain':'WEOG','Sweden':'WEOG',
    'Switzerland':'WEOG','Turkey':'WEOG','Türkiye':'WEOG',
    'United Kingdom':'WEOG','United Kingdom of Great Britain and Northern Ireland':'WEOG',
    'United States of America':'WEOG','United States':'WEOG',
    # Arab States
    'Bahrain':'ArabStates','Iraq':'ArabStates','Jordan':'ArabStates',
    'Kuwait':'ArabStates','Lebanon':'ArabStates','Oman':'ArabStates',
    'Qatar':'ArabStates','Saudi Arabia':'ArabStates','Syria':'ArabStates',
    'Syrian Arab Republic':'ArabStates',
    'United Arab Emirates':'ArabStates','Yemen':'ArabStates',
}

region_df = pd.DataFrame(list(un_regional_groups.items()), columns=['country','region'])
region_df.to_csv(os.path.join(EXT, 'un_regional_groups.csv'), index=False)
print(f"  Saved {len(region_df)} country-region mappings")

# ── WB Income Groups ──────────────────────────────────────────────
print("\n[5/6] Saving WB income groups...")
income_groups = {
    'High income': [
        'United States of America','United States','Canada','United Kingdom',
        'United Kingdom of Great Britain and Northern Ireland',
        'France','Germany','Japan','Australia','Italy','Spain',
        'Netherlands','Sweden','Norway','Denmark','Finland','Austria',
        'Belgium','Switzerland','New Zealand','Republic of Korea','South Korea',
        'Israel','Singapore','United Arab Emirates','Qatar','Kuwait','Bahrain',
        'Saudi Arabia','Oman','Iceland','Ireland','Luxembourg','Portugal',
        'Greece','Cyprus','Malta','Czechia','Czech Republic','Estonia','Latvia',
        'Lithuania','Slovakia','Slovenia','Hungary','Poland','Croatia',
        'Romania','Bulgaria','Andorra','Monaco','Liechtenstein','San Marino',
        'Bahamas','Barbados','Trinidad and Tobago','Chile','Panama','Uruguay',
    ],
    'Upper middle income': [
        'China','Brazil','Russia','Russian Federation','Mexico','Argentina',
        'Colombia','Peru','South Africa','Turkey','Türkiye','Thailand',
        'Malaysia','Iran','Iran (Islamic Republic of)','Iraq','Jordan',
        'Cuba','Dominican Republic','Ecuador','Guatemala','Venezuela',
        'Bolivarian Republic of Venezuela','Algeria','Libya','Gabon','Botswana',
        'Namibia','Mauritius','Belarus','Serbia','Montenegro','North Macedonia',
        'Albania','Bosnia and Herzegovina','Azerbaijan','Armenia','Georgia',
        'Kazakhstan','Turkmenistan','Jamaica','Paraguay','Suriname','Fiji','Tonga',
        'Samoa','Costa Rica','Belize',
    ],
    'Lower middle income': [
        'India','Pakistan','Bangladesh','Sri Lanka','Philippines','Indonesia',
        'Viet Nam','Vietnam','Myanmar','Cambodia',
        'Lao People Democratic Republic','Nepal','Bhutan','Mongolia',
        'Timor-Leste','Papua New Guinea','Egypt','Morocco','Tunisia','Sudan',
        'Nigeria','Ghana','Kenya','United Republic of Tanzania','Tanzania',
        'Zambia','Zimbabwe','Uganda','Ethiopia','Senegal','Cameroon',
        "Côte d'Ivoire",'Mali','Mauritania','Honduras','Nicaragua',
        'El Salvador','Bolivia','Guyana','Uzbekistan','Kyrgyzstan',
        'Tajikistan','Republic of Moldova','Moldova','Ukraine','Lebanon',
        'Syria','Syrian Arab Republic','Yemen',
    ],
    'Low income': [
        'Afghanistan','Haiti','Somalia','South Sudan','Chad','Niger',
        'Burkina Faso','Guinea','Guinea-Bissau','Liberia','Sierra Leone',
        'Central African Republic','Eritrea','Rwanda','Burundi','Malawi',
        'Mozambique','Togo','Benin','Madagascar','Djibouti','Comoros',
        'Lesotho','Eswatini','Gambia','Sao Tome and Principe',
        'Equatorial Guinea','Democratic People Republic of Korea','North Korea',
    ]
}

rows = []
for income, countries in income_groups.items():
    for c in countries:
        rows.append({'country': c, 'income_group': income})
income_df = pd.DataFrame(rows).drop_duplicates('country')
income_df.to_csv(os.path.join(EXT, 'wb_income_groups.csv'), index=False)
print(f"  Saved {len(income_df)} country income classifications")

# ── EDA Plot ──────────────────────────────────────────────────────
print("\n[6/6] Generating EDA plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')

rcid_per_year = df.groupby('year')['rcid'].nunique()
axes[0].bar(rcid_per_year.index, rcid_per_year.values, color='#58a6ff', alpha=0.8)
axes[0].set_xlabel('Year', color='white')
axes[0].set_ylabel('Resolutions', color='white')
axes[0].set_title('Resolutions per Year', color='white')
axes[0].tick_params(colors='white')
for sp in axes[0].spines.values(): sp.set_edgecolor('#30363d')

yearly_v = df.groupby('year')['v'].mean()
axes[1].plot(yearly_v.index, yearly_v.values, color='#3fb950', lw=2)
axes[1].axhline(0, color='#f85149', lw=0.8, ls='--')
axes[1].set_xlabel('Year', color='white')
axes[1].set_ylabel('Mean vote', color='white')
axes[1].set_title('Mean Vote Value Over Time', color='white')
axes[1].tick_params(colors='white')
for sp in axes[1].spines.values(): sp.set_edgecolor('#30363d')

plt.tight_layout()
out_fig = os.path.join(PLOTS, 'p1_vote_overview.png')
plt.savefig(out_fig, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"  Saved: {out_fig}")

print("\n✅ Phase 1 complete!")
print("   Outputs:")
for f in os.listdir(PROC):
    print(f"   • data/processed/{f}")
for f in os.listdir(EXT):
    print(f"   • data/external/{f}")
