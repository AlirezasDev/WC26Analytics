"""
Configuration and constants for WC26 Analytics Engine.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DATA_DIR / "artifacts"
SRC_DIR = DATA_DIR / "src"

RESULT_LABELS = {0: "loss", 1: "draw", 2: "win"}
INT_TO_LABEL = RESULT_LABELS
LABEL_TO_INT = {label: value for value, label in RESULT_LABELS.items()}

ROLE_MAP = {
    "Goalkeeper": "goalkeeper",
    "Centre-Back": "defender",
    "Left-Back": "defender",
    "Right-Back": "defender",
    "Defender": "defender",
    "Left Wing-Back": "defender",
    "Right Wing-Back": "defender",
    "Defensive Midfield": "midfielder",
    "Central Midfield": "midfielder",
    "Midfield": "midfielder",
    "Attacking Midfield": "midfielder",
    "Left Midfield": "midfielder",
    "Right Midfield": "midfielder",
    "Left Winger": "forward",
    "Right Winger": "forward",
    "Centre-Forward": "forward",
    "Forward": "forward",
    "Second Striker": "forward",
}

TEAM_NAME_ALIASES = {
    "usa": "united states",
    "czechia": "czech republic",
    "ivory coast": "ivory coast",
    "cote divoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cote d ivoire": "ivory coast",
    "dr congo": "congo",
    "democratic republic of the congo": "congo",
    "curaao": "curacao",
    "curacao": "curacao",
    "south arica": "south africa",
    "korea, south": "south korea",
    "republic of ireland": "ireland",
    "bosnia": "bosnia-herzegovina",
    "trkiye": "turkiye",
    "türkiye": "turkiye",
    "turkey": "turkiye",
}

# Known national team IDs in games.csv
KNOWN_NATIONAL_TEAM_IDS = {
    "algeria": 3614.0,
    "argentina": 3437.0,
    "australia": 3433.0,
    "austria": 3383.0,
    "belgium": 3382.0,
    "bosnia-herzegovina": 3446.0,
    "brazil": 3439.0,
    "canada": 3510.0,
    "cape verde": 4311.0,
    "colombia": 3816.0,
    "congo": 3854.0,
    "croatia": 3556.0,
    "curacao": 32364.0,
    "czech republic": 3445.0,
    "ecuador": 5750.0,
    "egypt": 3672.0,
    "england": 3299.0,
    "france": 3377.0,
    "germany": 3262.0,
    "ghana": 3441.0,
    "haiti": 14161.0,
    "iran": 3582.0,
    "iraq": 3560.0,
    "ivory coast": 3591.0,
    "cote divoire": 3591.0,
    "cote d'ivoire": 3591.0,
    "japan": 3435.0,
    "jordan": 15737.0,
    "mexico": 6303.0,
    "morocco": 3575.0,
    "netherlands": 3379.0,
    "new zealand": 9171.0,
    "norway": 3440.0,
    "panama": 3577.0,
    "paraguay": 3581.0,
    "portugal": 3300.0,
    "qatar": 14162.0,
    "saudi arabia": 3807.0,
    "scotland": 3380.0,
    "senegal": 3499.0,
    "south africa": 3806.0,
    "south korea": 3589.0,
    "spain": 3375.0,
    "sweden": 3557.0,
    "switzerland": 3384.0,
    "tunisia": 3670.0,
    "turkiye": 3381.0,
    "united states": 3505.0,
    "uruguay": 3449.0,
    "uzbekistan": 3563.0,
}

REQUIRED_CSV_FILES = [
    "appearances.csv",
    "club_games.csv",
    "clubs.csv",
    "competitions.csv",
    "countries.csv",
    "game_events.csv",
    "game_lineups.csv",
    "games.csv",
    "national_teams.csv",
    "player_valuations.csv",
    "players.csv",
    "transfers.csv",
]
