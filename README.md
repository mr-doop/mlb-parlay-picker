# MLB Parlay Picker — Streamlit Cloud Mode
See app for usage. Cloud Mode auto-fetches DK board and adds download buttons.

Feature
Type
Default
Used by
Why it moves outcomes
temp_f, humidity, wind_speed_mph, wind_out_to_cf, roof_closed
numeric/bool
70, 50, 0, 0, 0
Ks, Outs, ML/RL
Hot/windy‑out = HR↑ Ks↓; roof closed neutralizes
air_density_index
numeric
0
Ks, ML/RL
Composite of temp/humidity/pressure
bullpen_ip_1d, bullpen_ip_3d, bullpen_freshness_index
numeric
0, 0, 6.0
Outs, ML
Tired pens → longer leashes
opp_k_rate_30/14, opp_bb_rate_30/14 (+handedness)
numeric
0.22/0.08
Ks/BB
Rolling opponent discipline
park_run_factor, park_k_factor
numeric
1.00
Ks, ML/RL
Contextual park effects
ump_k_bias, ump_bb_bias
numeric
0
Ks/BB
Plate ump tendencies
last3_pc_mean, last3_ip_mean, days_rest_exact, leash_bias
numeric
90, 5.5, 5, 0
Outs, Ks
Recent usage & rest
usage_slider/..., opp_whiff_vs_slider/...
numeric
0
Ks
Arsenal vs opponent weaknesses
travel_miles_24h, day_after_night, west_to_east
numeric/bool
0
ML/RL, Outs
Fatigue effects
odds_drift_1h, odds_drift_3h, is_off_consensus
numeric/bool
0
ranking
Market intel

