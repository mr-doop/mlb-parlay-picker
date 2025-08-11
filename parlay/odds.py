from __future__ import annotations
import numpy as np, pandas as pd
def american_to_decimal(a):
    a=float(a); return 1.0 + (a/100.0 if a>0 else 100.0/abs(a))
def implied_prob_from_american(a):
    a=float(a); return 100.0/(a+100.0) if a>0 else abs(a)/(abs(a)+100.0)
def vig_free_binary(p1,p2):
    s=p1+p2; return (p1,p2) if s<=0 else (p1/s,p2/s)
def compute_vig_free_probs(df: pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    df['implied_p']=df['american_odds'].apply(implied_prob_from_american)
    df['decimal_odds']=df['american_odds'].apply(american_to_decimal)
    df['vigfree_p']=df['implied_p']
    def key(r):
        mt=r['market_type']; gid=r['game_id']
        if mt in {'PITCHER_KS','PITCHER_OUTS','PITCHER_WALKS'}: return ('PROP',gid,mt,r.get('player_id'),r.get('alt_line'))
        if mt=='PITCHER_WIN': return ('WIN',gid,mt,r.get('player_id'),None)
        if mt=='MONEYLINE': return ('ML',gid,mt,None,None)
        if mt in {'RUN_LINE','ALT_RUN_LINE'}:
            spread=r.get('alt_line'); spread_abs=abs(float(spread)) if spread is not None else None
            return ('RL',gid,mt,None,spread_abs)
        return ('OTHER',gid,mt,None,r.get('alt_line'))
    df['_k']=df.apply(key,axis=1)
    rows=[]
    for k,g in df.groupby('_k', dropna=False):
        g=g.copy(); sides=g['side'].astype(str).str.upper().tolist()
        if 'HOME' in sides and 'AWAY' in sides:
            ph=g.loc[g['side'].str.upper()=='HOME','implied_p'].iloc[0]
            pa=g.loc[g['side'].str.upper()=='AWAY','implied_p'].iloc[0]
            vfh,vfa=vig_free_binary(ph,pa)
            g.loc[g['side'].str.upper()=='HOME','vigfree_p']=vfh
            g.loc[g['side'].str.upper()=='AWAY','vigfree_p']=vfa
        elif 'OVER' in sides and 'UNDER' in sides:
            pov=g.loc[g['side'].str.upper()=='OVER','implied_p'].iloc[0]
            pun=g.loc[g['side'].str.upper()=='UNDER','implied_p'].iloc[0]
            vfo,vfu=vig_free_binary(pov,pun)
            g.loc[g['side'].str.upper()=='OVER','vigfree_p']=vfo
            g.loc[g['side'].str.upper()=='UNDER','vigfree_p']=vfu
        elif 'YES' in sides and 'NO' in sides:
            pys=g.loc[g['side'].str.upper()=='YES','implied_p'].iloc[0]
            pno=g.loc[g['side'].str.upper()=='NO','implied_p'].iloc[0]
            vfy,vfn=vig_free_binary(pys,pno)
            g.loc[g['side'].str.upper()=='YES','vigfree_p']=vfy
            g.loc[g['side'].str.upper()=='NO','vigfree_p']=vfn
        elif k[0]=='RL' and g['team'].nunique()>=2:
            teams=g['team'].unique()[:2]
            p1=g.loc[g['team']==teams[0],'implied_p'].iloc[0]
            p2=g.loc[g['team']==teams[1],'implied_p'].iloc[0]
            vf1,vf2=vig_free_binary(p1,p2)
            g.loc[g['team']==teams[0],'vigfree_p']=vf1
            g.loc[g['team']==teams[1],'vigfree_p']=vf2
        rows.append(g)
    return pd.concat(rows, ignore_index=True).drop(columns=['_k'])
