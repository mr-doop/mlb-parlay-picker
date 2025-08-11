from __future__ import annotations
def describe_row(r):
    mt=r.get('market_type',''); team=r.get('team',''); side=r.get('side','')
    player=r.get('player_name',''); alt=r.get('alt_line',None); game=r.get('game_id','')
    if mt in {'PITCHER_KS','PITCHER_OUTS','PITCHER_WALKS'}:
        return f"{player} {mt.replace('PITCHER_','').title()} {side} {alt} ({game})"
    if mt=='PITCHER_WIN': return f"{player} To Win: {side} ({game})"
    if mt in {'RUN_LINE','ALT_RUN_LINE'}: return f"{team} {alt:+} RL ({game})"
    if mt=='MONEYLINE': return f"{side} ML ({game})"
    return f"{mt} {side} {alt} ({game})"
