import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
deliveries_df = pd.read_csv(r"c:\Users\pradeep m\termal\deliveries.csv")
matches_df = pd.read_csv(r"c:\Users\pradeep m\termal\matches.csv")

print("Matches DataFrame:")
print(matches_df.head())

print("\nDeliveries DataFrame:")
print(deliveries_df.head())

# Team name normalization
team_name_mapping = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Rising Pune Supergiants': 'Rising Pune Supergiants',
    'Pune Warriors': 'Pune Warriors India',
    'Kings XI Punjab': 'Punjab Kings',
    'Gujarat Lions': 'Gujarat Titans'
}
matches_df.replace(team_name_mapping, inplace=True)
deliveries_df.replace(team_name_mapping, inplace=True)

matches_df['team1'] = matches_df['team1'].replace(team_name_mapping)
matches_df['team2'] = matches_df['team2'].replace(team_name_mapping)
matches_df['winner'] = matches_df['winner'].replace(team_name_mapping)
deliveries_df['batting_team'] = deliveries_df['batting_team'].replace(team_name_mapping)
deliveries_df['bowling_team'] = deliveries_df['bowling_team'].replace(team_name_mapping)

# Process seasons
seasons = matches_df['season'].unique()

# Compile team statistics
team_stats = []
for season in seasons:
    season_matches = matches_df[matches_df['season'] == season]
    teams = pd.concat([season_matches['team1'], season_matches['team2']]).unique()

    for team in teams:
        played = season_matches[(season_matches['team1'] == team) | (season_matches['team2'] == team)]
        won = season_matches[season_matches['winner'] == team]

        matches_played = played.shape[0]
        matches_won = won.shape[0]
        win_rate = matches_won / matches_played if matches_played > 0 else 0

        runs_scored = deliveries_df[(deliveries_df['batting_team'] == team) &
                                    (deliveries_df['match_id'].isin(played['id']))]['total_runs'].sum()
        runs_conceded = deliveries_df[(deliveries_df['bowling_team'] == team) &
                                      (deliveries_df['match_id'].isin(played['id']))]['total_runs'].sum()
        net_run_diff = runs_scored - runs_conceded

        team_stats.append({
            'season': season,
            'team': team,
            'matches_played': matches_played,
            'matches_won': matches_won,
            'win_rate': win_rate,
            'runs_scored': runs_scored,
            'runs_conceded': runs_conceded,
            'net_run_diff': net_run_diff
        })

team_stats_df = pd.DataFrame(team_stats)
print(team_stats_df.head())

# Determine finalists
finalist_teams = []
for season in seasons:
    season_matches = matches_df[matches_df['season'] == season]
    final_match = season_matches[season_matches['date'] == season_matches['date'].max()]
    finalists = list(final_match[['team1', 'team2']].values[0])
    for team in finalists:
        finalist_teams.append((season, team))

team_stats_df['is_finalist'] = team_stats_df.apply(
    lambda row: 1 if (row['season'], row['team']) in finalist_teams else 0, axis=1
)

# Features and labels
features = ['matches_played', 'matches_won', 'win_rate', 'runs_scored', 'runs_conceded', 'net_run_diff']
X = team_stats_df[features]
y = team_stats_df['is_finalist']

# Split data
train_data = team_stats_df[team_stats_df['season'] != '2024']
test_data = team_stats_df[team_stats_df['season'] == '2024']

X_train = train_data[features]
y_train = train_data['is_finalist']
X_test = test_data[features]
y_test = test_data['is_finalist']

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction for 2024
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prediction for 2025 using 2024 performance
latest_stats = test_data[features]
team_names_2024 = test_data['team'].values
finalist_preds = model.predict_proba(latest_stats)[:, 1]

predictions = pd.DataFrame({'team': team_names_2024, 'finalist_probability': finalist_preds})
predicted_finalists = predictions.sort_values(by='finalist_probability', ascending=False).head(2)
print("\nPredicted Finalists for 2025:")
print(predicted_finalists)
