# Road-accident fatality classification (France, 2023)

A coursework project: can we predict whether a person involved in a road accident was killed, from the
characteristics of the accident, the location, the vehicle and the person? It uses the official French road-accident
open data (BAAC 2023) and compares logistic regression, random forest and a small neural network, with SMOTE for the
heavy class imbalance.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6B7280)

> ⚠️ **Read before using the numbers.** The saved notebook keeps the `grav` (injury severity) column as a feature,
> and the target `mortality` is defined as `grav == 2` (killed). That is **target leakage**: the Random Forest gets
> 100% test accuracy on the original split. In addition, SMOTE is applied before the train/test split. The metrics
> below are kept for transparency. They are **not** valid estimates of real predictive performance. See
> [Fixing the evaluation](#fixing-the-evaluation).

## Data

Four BAAC 2023 tables from [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/)
(Licence Ouverte / Etalab), included as CSVs:

| File | Rows | Content |
|---|---|---|
| `caract-2023.csv` | 54,822 | accident characteristics (date, light, weather, location type) |
| `lieux-2023.csv` | 70,860 | road / location |
| `vehicules-2023.csv` | 93,585 | vehicles |
| `usagers-2023.csv` | 125,789 | people involved (incl. `grav` severity) |

The tables are inner-joined on `Num_Acc`, which gives 309,341 rows and 57 columns. 6,037 rows (about 2%) are
fatalities. Because the join uses only the accident id, every person is paired with every vehicle of the same
accident. This duplicates people across rows, so the row count is larger than the 125,789 people.

## What the notebook does

1. Merges and cleans the four tables. Creates `mortality = 1 if grav == 2 else 0`.
2. Runs EDA: fatalities by sex, month, age, vehicle category and hour of day.
3. Trains logistic regression, random forest (`n_estimators=50, max_depth=5, class_weight="balanced"`) and an MLP (50-25).
4. Rebalances with SMOTE, runs 5-fold cross-validation (macro F1) and tries a decision threshold of 0.6.
5. Starts a `GridSearchCV` for the random forest. That run was interrupted, so there are no results.

### Results as recorded (affected by the leakage described above)

| Model | Setting | Test accuracy | Notes |
|---|---|---|---|
| Logistic regression | original split | 0.98 | predicts "not fatal" for every row (fatal-class recall 0.00) |
| Random forest (default) | original split | 1.00 | leakage through `grav` |
| Random forest (shallow, balanced) | trained on SMOTE data, tested on the original split | 0.94 | fatal class: recall 0.87, precision 0.24 |
| MLP | SMOTE split | 0.76 | CV macro F1 0.74 |

## Fixing the evaluation

1. Drop `grav`, and any other column recorded after the outcome, from the features.
2. Join `usagers` to `vehicules` on (`Num_Acc`, `id_vehicule`), not on `Num_Acc` alone.
3. Split train/test **first** (grouped by `Num_Acc`, so one accident is never in both), then apply SMOTE to the
   training fold only, ideally inside an `imblearn` `Pipeline`.
4. Report PR-AUC and fatal-class recall/precision rather than accuracy, since 98% is the "always not fatal" baseline.

## Files

- `machine_learning_project.ipynb`: the full analysis
- `Machine_Learning_Accident_Analysis_Report.pdf`: written report
- `machine learning project ppt.pptx`: slides
- `*-2023.csv`: raw BAAC data (about 22 MB)

## What I learned

- High accuracy on imbalanced data says little. Look at minority-class recall and precision.
- Leakage hides in "obvious" columns. The label must not be derivable from the features.
- Resampling belongs inside the cross-validation loop, never before the split.
