from __future__ import annotations
import math, numpy as np, pandas as pd
def _clip(x,lo,hi): return max(lo,min(hi,x))
def _nb_tail(mu,alpha,thr):
    var=mu+alpha*mu*mu
    if mu<=0 or var<=0: return 0.0
    z=(math.ceil(thr)-0.5-mu)/math.sqrt(var)
    return 0.5*math.erfc(z/(2**0.5))
def project_expected_ip(last5_pitch_ct_mean=85, days_rest=5, leash_bias=0.0, favorite_flag=0, bullpen_freshness=6.0):
    base=(last5_pitch_ct_mean or 85)/15.0
    adj=0.15*leash_bias + 0.15*favorite_flag - 0.10*_clip((bullpen_freshness/7.5),0,1) + 0.10*_clip((days_rest-4)/2.0,-0.1,0.2)
    return _clip(base+adj,3.0,7.5)
def project_ks_mu(expected_ip, pitcher_k_rate=0.24, opp_k_rate=0.22, park_k_factor=1.0, ump_k_bias=1.0):
    bf=expected_ip*4.2; eff= pitcher_k_rate*(0.5+0.5*opp_k_rate/0.22)*park_k_factor*ump_k_bias
    return _clip(bf*eff,0.5,14.0)
def project_bb_mu(expected_ip, pitcher_bb_rate=0.08, opp_bb_rate=0.08, ump_bb_bias=1.0):
    bf=expected_ip*4.2; eff=pitcher_bb_rate*(0.5+0.5*opp_bb_rate/0.08)*ump_bb_bias
    return _clip(bf*eff,0.0,8.0)
def prob_over_count(mu,alpha,line): return float(_nb_tail(mu,alpha,line))
def project_outs_probs(expected_ip, leash_bias=0.0):
    mean_outs=expected_ip*3.0; alpha=_clip(0.10+0.25*(1.0-leash_bias),0.05,0.5)
    pmf={}
    for outs in range(0,28):
        var=mean_outs+alpha*mean_outs**2
        zlo=((outs-0.5)-mean_outs)/math.sqrt(var); zhi=((outs+0.5)-mean_outs)/math.sqrt(var)
        cdf_lo=0.5*math.erfc(-zlo/(2**0.5)); cdf_hi=0.5*math.erfc(-zhi/(2**0.5))
        pmf[outs]=max(0.0,cdf_hi-cdf_lo)
    def tail_at(xh): t=int(xh-0.5); return sum(pmf.get(o,0.0) for o in range(t+1,28))
    return {x: tail_at(x) for x in [14.5,15.5,16.5,17.5,18.5,19.5]}
def project_win_prob(vf_ml, p_ip_ge5, bullpen_hold=0.90): return _clip(vf_ml*p_ip_ge5*bullpen_hold,0.0,0.9)
def apply_projections(legs_df: pd.DataFrame, features_df: pd.DataFrame)->pd.DataFrame:
    df=legs_df.copy(); feats=features_df.copy()
    df=df.merge(feats, on='player_id', how='left', suffixes=('','_feat'))
    df['expected_ip']=df.apply(lambda r: project_expected_ip(r.get('last5_pitch_ct_mean',85), r.get('days_rest',5),
                                                            r.get('leash_bias',0.0), int(r.get('favorite_flag',0) or 0),
                                                            r.get('bullpen_freshness',6.0)), axis=1)
    # Ks
    mask_ks = (df['market_type']=='PITCHER_KS') & df['alt_line'].notna()
    df.loc[mask_ks,'mu_ks']=df[mask_ks].apply(lambda r: project_ks_mu(r['expected_ip'], r.get('pitcher_k_rate',0.24),
                                                                      r.get('opp_k_rate',0.22), r.get('park_k_factor',1.0),
                                                                      r.get('ump_k_bias',1.0)), axis=1)
    df.loc[mask_ks,'q_proj']=df[mask_ks].apply(lambda r: prob_over_count(r['mu_ks'],0.20, float(r['alt_line']) if str(r['side']).upper()=='OVER' else float(r['alt_line'])-1.0), axis=1)
    df.loc[mask_ks & df['side'].str.upper().eq('UNDER'),'q_proj'] = 1.0 - df.loc[mask_ks & df['side'].str.upper().eq('UNDER'),'q_proj']

    # Walks
    mask_bb = (df['market_type']=='PITCHER_WALKS') & df['alt_line'].notna()
    df.loc[mask_bb,'mu_bb']=df[mask_bb].apply(lambda r: project_bb_mu(r['expected_ip'], r.get('pitcher_bb_rate',0.08),
                                                                      r.get('opp_bb_rate',0.08), r.get('ump_bb_bias',1.0)), axis=1)
    df.loc[mask_bb,'q_proj']=df[mask_bb].apply(lambda r: prob_over_count(r['mu_bb'],0.25, float(r['alt_line']) if str(r['side']).upper()=='OVER' else float(r['alt_line'])-1.0), axis=1)
    df.loc[mask_bb & df['side'].str.upper().eq('UNDER'),'q_proj'] = 1.0 - df.loc[mask_bb & df['side'].str.upper().eq('UNDER'),'q_proj']

    # Outs
    mask_outs = (df['market_type']=='PITCHER_OUTS') & df['alt_line'].notna()
    tails = df[mask_outs].apply(lambda r: project_outs_probs(r['expected_ip'], r.get('leash_bias',0.0)), axis=1)
    df.loc[mask_outs,'q_proj']=[ (t.get(r['alt_line'], None) if str(r['side']).upper()=='OVER' else (1.0 - t.get(r['alt_line'], None)) ) for t,(_,r) in zip(tails, df[mask_outs].iterrows()) ]

    # Pitcher Win
    mask_win = df['market_type'].eq('PITCHER_WIN')
    vf_team = df.get('team_ml_vigfree', pd.Series(index=df.index)).fillna(df.get('vigfree_p', 0.5))
    p_ip_ge5 = df.apply(lambda r: project_outs_probs(r.get('expected_ip',5.5), r.get('leash_bias',0.0)).get(14.5, 0.6), axis=1)
    df.loc[mask_win,'q_proj'] = [ project_win_prob(vf_team.iloc[i], p_ip_ge5.iloc[i]) for i in df.index ]
    df.loc[mask_win & df['side'].str.upper().eq('NO'),'q_proj'] = 1.0 - df.loc[mask_win & df['side'].str.upper().eq('NO'),'q_proj']

    return df
