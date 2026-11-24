# cfDARKO: college football → NFL draft projections

A research pipeline that projects how college football players will turn out in the NFL, inspired by Kostya Medvedovsky's DARKO model for the NBA. It turns college play-by-play and box-score data into time-weighted skill estimates for each player, then values draft prospects in rookie-contract terms.

The main focus is honest evaluation: avoiding look-ahead bias, pre-registering experiments, and adopting a change only when it passes its pre-stated test.

## What it does

1. **Ingests** college game and play-by-play data (CollegeFootballData API), NFL outcomes and contracts (nflverse), combine measurements, and stadium orientation for wind and weather context.
2. **Adjusts for context.** Contextual models (including per-play LightGBM models) estimate what an average player would have produced in the same situation. A player's residual is his production above that baseline.
3. **Estimates skill over time** with exponentially weighted averages and Kalman-filter priors, so recent seasons count more and small samples are shrunk toward position norms.
4. **Labels NFL outcomes** as value delivered on the rookie contract, priced from second-contract pay.
5. **Builds draft models and a board.** It combines per-position and cross-position models, calibrates the scores, models each player's chance of still being available at a given pick, and generates player comps.
6. **Serves a Streamlit app** (`frontend/app.py`) for exploring prospects.

## Research process

- **Pre-registered experiments.** Each test's hypothesis, primary metric and decision rule were committed before any results were seen. Bug fixes found mid-test are logged as such, before results. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full log of pre-registrations and outcomes.
- **Leakage audit.** [`docs/leakage_audit.md`](docs/leakage_audit.md) documents where historical validation could see the future, and how that affects (and doesn't affect) current prospect rankings.
- **Negative results are kept.** Several plausible ideas failed their primary tests and were not adopted. For example, per-player adjustments on top of the consensus board carried signal, but not enough to beat the board out of sample.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "CFBD_API_KEY=your_key" > .env        # free key: https://collegefootballdata.com/key
.venv/bin/python run_pipeline.py --test    # small test run
.venv/bin/python run_pipeline.py --no-tune # full run, reusing the tuned parameters committed in data/processed/
.venv/bin/streamlit run frontend/app.py
.venv/bin/pytest
```

Tuned hyperparameters, feature selections and calibration constants are committed as JSON under `data/processed/`, so a `--no-tune` run rebuilds the models without the multi-hour retune.

## Data not included

This public copy leaves out inputs that come from paid or third-party sources:

- **PFF grades.** `src/scrapers/pff.py` reads season CSV exports from a PFF subscription, placed in `data/raw/pff/{year}/` (college) and `data/raw/pff_nfl/` (NFL). Without them, PFF-based features are unavailable and the rookie-value label falls back to neutral quality, so results will differ from the author's runs. The export download script is not included.
- **Historical consensus big boards** (Arif Hasan, 2014–2024) expected at `hasan.csv`. Recent years are fetched from the public sheet at run time.
- **Combine supplements** from Pro-Football-Reference, used only to fill years that nflverse hasn't published yet.

## Layout

- `run_pipeline.py`: runs the numbered steps in `pipeline/` in order (`--from`, `--step`, `--no-tune`, `--retune`)
- `src/models/`: Kalman filter, decay, contextual, draft-slot, availability and board models
- `src/scrapers/`: data loaders (CFBD, nflverse, PFF exports)
- `experiments/`: experiment scripts and result files
- `tests/`: pytest suite
- `config.py`: all modeling constants in one place
