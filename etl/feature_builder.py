def build_features(date: str, dk_df: pd.DataFrame) -> pd.DataFrame:
    # join on player_id/game_id/team as appropriate
    feats = []
    feats.append(weather.build(date, dk_df))
    feats.append(bullpen.build(date, dk_df))
    feats.append(opp_rates.build(date, dk_df))
    feats.append(park_factors.build(date, dk_df))
    # ... add more packs
    features = reduce_join(feats)  # left-join in a stable order
    # fill NaNs with safe defaults
    features = fill_feature_defaults(features)
    # write to features_{date}.csv
    return features