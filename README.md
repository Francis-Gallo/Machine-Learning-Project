# Road-accident fatality classification (France, 2023)

A coursework project: can we predict whether a person involved in a road accident was killed, from the
characteristics of the accident, the location, the vehicle and the person? It uses the official French road-accident
open data (BAAC 2023) and compares logistic regression, random forest and a small neural network, with SMOTE for the
heavy class imbalance.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6B7280)

> The notebook used to keep `grav` in the feature matrix and fit SMOTE on the whole table before the split. The scores
> in the table below are that saved run. They are a record of the leak, and they are not a measure of predictive
> performance. This revision drops `grav` and fits SMOTE only inside an `imblearn` pipeline on the training fold.
> Re-running it on the committed CSVs stops before any model is fit, so there is no replacement score. See
> [Re-run](#re-run).

## Data

Four BAAC 2023 tables from [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/)
(Licence Ouverte / Etalab), included as CSVs:

| File | Rows | Content |
|---|---|---|
| `caract-2023.csv` | 54,822 | accident characteristics (date, light, weather, location type) |
| `lieux-2023.csv` | 70,860 | road / location |
| `vehicules-2023.csv` | 93,585 | vehicles |
| `usagers-2023.csv` | 125,789 | people involved (incl. `grav` severity) |

The saved notebook inner-joined the tables on `Num_Acc` and got 309,341 rows and 57 columns. 6,037 of those rows
(about 2%) were fatalities (`grav == 2`). Because that join used only the accident id, every person was paired with
every vehicle of the same accident, so the row count was larger than the 125,789 people.

## What the notebook does

1. Loads the four CSVs from the repository root and checks that `Num_Acc` still identifies accidents.
2. Merges and cleans the tables. Creates `mortality = 1 if grav == 2 else 0`.
3. Runs EDA: fatalities by sex, month, age, vehicle category and hour of day.
4. Drops `grav` (and any other column whose name contains `grav`) before the split. `mortality` is the target.
5. Splits train/test first (`test_size=0.2`, `random_state=42`). Fits logistic regression and a default random forest on that split with no resampling.
6. Fits logistic regression (`class_weight="balanced"`), a shallow random forest (`n_estimators=50`, `max_depth=5`, `class_weight="balanced"`) and an MLP (`hidden_layer_sizes=(50, 25)`) inside an `imblearn` pipeline: `StandardScaler`, then `SMOTE(random_state=42)`, then the classifier. The scaler and SMOTE run only on the training fold. The test fold stays at the original class mix. Scores are accuracy, fatal-class precision/recall/F1, and PR-AUC. A 0.6 threshold is applied to the shallow forest's test probabilities. 5-fold macro F1 uses the same pipeline.
7. Runs the previous `GridSearchCV` (180 fits) on that pipeline, training split only. The earlier search was interrupted and this one has no saved score.

### Results recorded before this fix

These numbers are copied from the previous notebook outputs. `grav` was still a feature, and SMOTE was fit on the full table before `train_test_split`. Do not use them as performance estimates.

| Model | Setting in that run | Test accuracy | Fatal class |
|---|---|---|---|
| Logistic regression | original split, `grav` still a feature | 0.9802 | precision 0.00, recall 0.00 (every test row predicted non-fatal) |
| Random forest (default) | original split, `grav` still a feature | 1.0000 | precision 1.00, recall 1.00 |
| Random forest (shallow, balanced) | SMOTE fit on all rows, then tested on the original split | 0.9435 | precision 0.24, recall 0.87 |
| MLP (50-25) | SMOTE fit on all rows, then tested on a resampled test split | 0.7557 | CV macro F1 mean 0.7420 |

## Re-run

The modeling cells above were executed against the CSVs in this repository. They do not produce a new metric.

Every `Num_Acc` value in the four committed files is the same collapsed id. Pandas reads it as the float `202300000000.0` (`nunique` is 1 in each file; the text in the file is `2.023E+11`). An inner join on that column is a cartesian product, not the 309,341-row table from the saved run. The notebook raises before the merge, and no estimator is fit.

`lat` / `long` in `caract-2023.csv` are damaged in the same export: the decimal comma was treated as a separator, so `lat` is only the integer part and `long` is empty on every row. `id_vehicule` and `id_usager` are still unique, which is why the vehicle and user files can be checked row by row, but they do not restore the accident key for `caract-2023.csv` and `lieux-2023.csv`.

No held-out accuracy, fatal-class precision/recall, or PR-AUC is reported for the corrected models, because none was computed.

## What is still open

The corrected code does not change two other limits of the saved analysis:

- `usagers` is still joined to `vehicules` on `Num_Acc` alone, once a real accident id exists, so people remain duplicated across vehicles of the same accident.
- The split is still a row split, not a split grouped by `Num_Acc`.

## Files

- `machine_learning_project.ipynb`: the full analysis
- `Machine_Learning_Accident_Analysis_Report.pdf`: written report
- `machine learning project ppt.pptx`: slides
- `*-2023.csv`: raw BAAC data (about 22 MB)

## What I learned

- High accuracy on imbalanced data says little. Look at minority-class recall, precision, and PR-AUC.
- Leakage hides in "obvious" columns. The label must not be derivable from the features.
- Resampling belongs inside the training fold, including inside cross-validation, and the test fold stays untouched.
