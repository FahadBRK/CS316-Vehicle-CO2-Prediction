# Vehicle CO2 Emissions Prediction Using Machine Learning

**CS316 - Introduction to AI and Data Science, Phase 2**  
Prince Sultan University  
Fahad Albarrak (223110478), Meshal Alsuhaibani (223110309), Nawaf Alkhazy (223111223)  
Supervisor: Dr. Mohammed Al Zamil

This project implements the uploaded Phase 1 proposal using the official 2025
Canadian fuel-consumption dataset. It retains the proposed numerical linear
regression and evaluates a small, interpretable extension with separate fuel-type
intercepts and fuel-consumption slopes. The extension is still ordinary least
squares, with no deep learning or application framework.

## Run the analysis

Use Python **3.12**. Unzip this folder, open a terminal in it, and run:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the recorded versions and execute:

```bash
python -m pip install -r requirements.txt
python -m src.analysis
python -m unittest discover -s tests -v
```

The dataset is included, so the analysis does not require an internet connection after the dependencies are installed. Running the analysis updates the results/ folder and saves the trained models in models/. The supplied results were generated using the fixed random seed and package versions listed in requirements.txt.
An optional notebook is in `notebooks/Vehicle_CO2_Analysis.ipynb`. Install JupyterLab
separately (`python -m pip install jupyterlab`) and run its cells in order.
The notebook calls the same analysis code and includes outputs from the executed run.

## Example prediction

```bash
python predict.py --engine 2.0 --cylinders 4 --combined 8.0 --fuel X
```

Fuel codes: X = regular gasoline, Z = premium gasoline, D = diesel, E = E85.
Engine size is in litres, combined fuel consumption is in L/100 km, and the output
is rated CO2 in g/km. The CLI checks positive inputs and warns about extrapolation.
Only load trusted `.joblib` files.

## Data and validation

- Raw snapshot: 701 rows and 15 columns, retrieved 14 November 2026.
- No missing values, exact full-row duplicates, or invalid rows were found.
- Make plus the exact stripped model name defines 600 catalog groups.
- Fixed seed 42 holds out 120 groups (137 rows). Training contains 480 groups
  (564 rows), with zero exact make/model group overlap.
- Five grouped folds within training compare all five predeclared candidates.
  Minimum mean validation MAE selects the model before test predictions are made.
- All imputers and fuel encodings learn from training data only. Target-derived
  CO2 rating, smog rating, redundant mpg and city/highway columns are excluded.
- Cluster bootstrap intervals resample held-out catalog groups 2,000 times.
  They describe uncertainty in test MAE conditional on this trained model and split.

## Executed results

| Model | Grouped CV MAE (g/km) | Test MAE (g/km) | Test MSE ((g/km)^2) | Test R2 |
|---|---:|---:|---:|---:|
| Training-mean baseline | 52.018 | 55.471 | 5278.973 | -0.000012 |
| Engine and cylinders | 31.173 | 31.431 | 1778.011 | 0.663186 |
| Combined fuel consumption | 3.146 | 2.937 | 39.885 | 0.992445 |
| Phase 1 numerical linear regression | 3.331 | 3.090 | 40.314 | 0.992363 |
| Fuel-interaction linear regression | 1.421 | 1.449 | 6.217 | 0.998822 |

The selected model's grouped CV MAE is 1.421 +/- 0.102 (sample standard deviation
across five folds). Its test RMSE is 2.493 g/km. The 95% cluster-bootstrap interval
for its test MAE is [1.157, 1.844] g/km. MAE falls 53.1% relative to the proposed
Phase 1 numerical model on this held-out set. R2 is not classification accuracy.

Fuel consumption and published CO2 ratings have a strong physical relationship.
These results therefore assess reconstruction of catalog ratings from related
ratings. They do not validate prediction of independently measured real-world
driving emissions, future model years, Saudi vehicles, or lifecycle emissions.
The test set has only three diesel rows and no E85 rows. Distinct model strings
can still share platforms or engines. No claim of fleet-wide representativeness
or causal effects is made.

## Files

| Path | Purpose |
|---|---|
| `src/analysis.py` | Audit, grouped splitting, cross-validation, test metrics, bootstrap and figures |
| `src/features.py` | Training-fitted, vectorized fuel-interaction transformer |
| `predict.py` | Validated example prediction CLI |
| `data/raw/` | Unmodified official CSV snapshot |
| `data/SOURCE.md` | Source URL, licence attribution and file hash |
| `results/` | Metrics, data audit, coefficients, per-row predictions, split IDs and figures |
| `models/` | Selected and Phase 1 models fitted to training data only |
| `notebooks/` | Executed analysis notebook with reproducible code |
| `tests/` | Source, leakage, metric and model-artifact consistency checks |

## Memory efficiency

`load_and_audit()` selects nine relevant source columns, uses categorical storage
for repeated text, downcasts numeric columns, and measures `memory_usage(deep=True)`.
On the same 701 rows and selected columns, memory decreases from 194,969 bytes to
72,774 bytes (62.7%). This is DataFrame storage, not a claim about peak process RAM.
Feature construction uses NumPy broadcasting, group aggregates avoid duplicating
rows for bootstrap sampling, folds are reused across models, and figures are
closed immediately.

## References

1. Natural Resources Canada, 2025 Fuel Consumption Ratings. Source and licence: `data/SOURCE.md`.
2. NRCan, Understanding the tables, official dataset guide linked in `data/SOURCE.md`.
3. NRCan, Learn the facts: Fuel consumption and CO2, https://natural-resources.canada.ca/sites/www.nrcan.gc.ca/files/oee/pdf/transportation/fuel-efficient-technologies/autosmart_factsheet_6_e.pdf
4. scikit-learn, LinearRegression, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
5. scikit-learn, GroupKFold, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
6. scikit-learn, Regression metrics, https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
7. United Nations, Goal 13, https://sdgs.un.org/goals/goal13
