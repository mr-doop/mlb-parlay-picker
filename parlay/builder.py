from __future__ import annotations
import numpy as np, pandas as pd
def build_parlay_greedy(df: pd.DataFrame, target_decimal_odds=6.0, min_legs=4, max_legs=7, mode='SAFETY'):
    pool=df.copy(); pool=pool[pool['decimal_odds']>1].dropna(subset=['q'])
    pool['score']= (pool['q']*np.log(pool['decimal_odds'])) if mode.upper()=='VALUE' else (pool['q']+1e-6*np.log(pool['decimal_odds']))
    pool=pool.sort_values(['score','q','decimal_odds'], ascending=[False,False,False])
    chosen=[]; used=set(); total=1.0
    for _,r in pool.iterrows():
        if r['game_id'] in used: continue
        chosen.append(r); used.add(r['game_id']); total*=r['decimal_odds']
        if total>=target_decimal_odds and len(chosen)>=min_legs: break
        if len(chosen)>=max_legs: break
    picks=pd.DataFrame(chosen); picks.attrs['total_decimal_odds']=total
    picks.attrs['est_hit_prob']=picks['q'].prod() if not picks.empty else float('nan')
    return picks
