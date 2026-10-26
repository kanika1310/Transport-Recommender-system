# Transport Recommender System

A small machine-learning project that recommends a transport mode (bus, metro, auto-rickshaw, cab, bike)
for an urban trip, given trip and context features. Built for the ML Lab course and described in the
accompanying research paper.

> **Status:** replace `data/trips.csv` with the real dataset before reporting any numbers.
> `data/trips_synthetic.csv` is simulated and only exists to check that the code runs.

## What it does
1. Loads a trip dataset (one row per trip, with the mode the traveller actually chose).
2. Compares five approaches with stratified 5-fold cross-validation:
   majority class, a hand-written rule-based recommender, Logistic Regression, Random Forest, Gradient Boosting.
3. Reports Accuracy, Macro-F1 and Top-2 accuracy (is the true mode among the two recommended ones?).
4. Runs a feature-group ablation with Random Forest.
5. Saves tables and figures to `results/`.

## Project structure
```
transport-recommender-system/
├── data/
│   ├── trips.csv                # YOUR real dataset (not included)
│   └── trips_synthetic.csv      # simulated data for dry runs only
├── src/
│   ├── generate_synthetic.py    # creates the simulated dataset
│   └── run_experiments.py       # comparison + ablation + plots
├── results/                     # generated tables and figures
├── docs/                        # paper PDF / report go here
├── requirements.txt
└── LICENSE
```

## Setup
```bash
git clone https://github.com/<your-username>/transport-recommender-system.git
cd transport-recommender-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/run_experiments.py --data data/trips.csv
# dry run without real data:
python src/generate_synthetic.py
python src/run_experiments.py --data data/trips_synthetic.csv --out results_dryrun
```

## Dataset format
| column | type | description |
|---|---|---|
| distance_km | float | trip distance in km |
| hour | int 0-23 | departure hour |
| is_weekend | 0/1 | weekend trip |
| weather | clear / rain / hot | weather during the trip |
| traffic | low / medium / high | traffic level |
| budget | low / medium / high | traveller's budget for this trip |
| group_size | int | number of people travelling together |
| has_luggage | 0/1 | carrying luggage |
| mode | bus / metro / auto / cab / bike | **target:** mode actually used |

## Outputs
`results/model_comparison.csv`, `results/ablation.csv`, `results/model_comparison.png`,
`results/confusion_matrix_rf.png`, `results/feature_importance.png`, `results/run_info.json`.

## Limitations
Small survey-style data, self-reported choices, no real-time traffic or fare APIs. See the paper for details.

## Authors
[Your Name], [Teammates] — [College / Department], ML Lab.

## License
MIT, see `LICENSE`.
