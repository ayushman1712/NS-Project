import pandas as pd, numpy as np, os
ROOT = r'D:\BTP\Cell2FireMksh\Ayushman\NS\NS-Project'
df = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'votes_full.csv'))
for country in ['India', 'United States of America', 'China', 'Russian Federation', 'Russia']:
    sub = df[df['country']==country]
    if len(sub) > 0:
        print(country, len(sub), 'votes, mean=', round(sub['v'].mean(), 3))

india = df[df['country']=='India'][['rcid','v']].rename(columns={'v':'v_india'})
usa   = df[df['country']=='United States of America'][['rcid','v']].rename(columns={'v':'v_usa'})
m1 = india.merge(usa, on='rcid')
print('India-USA common votes:', len(m1), 'corr:', round(m1['v_india'].corr(m1['v_usa']), 4))

china = df[df['country']=='China'][['rcid','v']].rename(columns={'v':'v_china'})
m2 = india.merge(china, on='rcid')
print('India-China common votes:', len(m2), 'corr:', round(m2['v_india'].corr(m2['v_china']), 4))

# Check top correlations for India
pivot = df.pivot_table(index='rcid', columns='country', values='v', aggfunc='first')
if 'India' in pivot.columns:
    corr_india = pivot.corr(min_periods=50)['India'].drop('India').sort_values(ascending=False)
    print("\nTop 10 correlated with India:")
    print(corr_india.head(10).to_string())
    print("\nBottom 10 (anti-correlated):")
    print(corr_india.tail(10).to_string())
    print("\nCorr with USA:", round(corr_india.get('United States of America', float('nan')), 4))
    print("Corr with China:", round(corr_india.get('China', float('nan')), 4))
