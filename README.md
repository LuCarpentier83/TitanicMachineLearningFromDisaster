# Titanic – Machine Learning from Disaster

Kaggle's getting-started competition: predict which Titanic passengers survived
(`Survived` = 1) from their information (class, sex, age, family, ticket, fare…).
The Kaggle score is **accuracy** (share of correct predictions) on the 418 test passengers:
each passenger is worth 1/418 ≈ **0.24 point**.

**Best score: 0.80143** (335 / 418), using simple rules (sex + family survival +
solo third-class women), without any machine learning model.

---

## Setup

The Python environment is managed by [uv](https://docs.astral.sh/uv/) (exact versions in `uv.lock`).

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1   # Windows
```
```bash
./install.sh                                             # Linux / macOS
```

The script installs uv if needed, creates `.venv` with the dependencies, registers the Jupyter kernel
"Python (titanic)" and enables the git hooks (`.githooks/commit-msg` rejects any commit containing
an AI attribution). It is safe to re-run.

Submitting to Kaggle requires a Kaggle API token (`%USERPROFILE%\.kaggle\access_token` or `kaggle.json`).

## Layout

| Path | Purpose |
|---|---|
| `src/titanic/features.py` | All shared code: cleaning (pipeline pieces), grouped CV, group survival, rules |
| `notebooks/02_random_forest.ipynb` | Random Forest |
| `notebooks/03_gradient_boosting.ipynb` | Gradient Boosting (GBC), CV grouped by ticket |
| `notebooks/04_survie_groupe.ipynb` | Group survival (`GroupSurv` feature and woman/boy rules) |
| `scripts/submit.py` | Kaggle submission: checks the format, refuses an already-submitted file, waits for the score, logs everything in `submissions/history.csv` |
| `submissions/` | Submitted files and score history |
| `data/train.csv`, `data/test.csv` | Kaggle data |
| `data/ntrain.csv`, `ntest.csv`, `submit.csv` | Old pre-cleaned CSVs (from before the switch to pipelines), no longer used |

```powershell
uv run jupyter lab                                                    # notebooks
uv run python scripts/submit.py submissions/x.csv -m "description"    # submit
uv run python scripts/submit.py --list                                # latest Kaggle submissions
```

### Reproducing the best submission (0.80143)

```python
from titanic.features import load_data, group_rules, solo_female_rule, save_submission

X, y, test = load_data()
pred = (test.Sex == "female").astype(int).values           # 1. gender model: women live, men die
pred = group_rules(X, y, test, pred, max_ticket_gap=1)      # 2. corrections from family survival
pred = solo_female_rule(test, pred)                         # 3. solo third-class women
save_submission(test, pred, "genre_regles_nom_femmes_seules")
```

---

## Investigations, in order

### Glossary

- **CV (cross-validation)**: split the training set into 5 parts, train on 4 and score on the 5th, five times,
  and repeat 5 times ("5 × 5"). Estimates the score on unseen data without spending a submission.
- **Stratified CV**: each part keeps the same share of survivors as the training set.
- **Ticket-grouped CV**: a given ticket (a family, a travel group) is never in both training and validation.
- **Overfitting**: the model learns details specific to the training set that don't carry over; it shines in CV and drops on Kaggle.
- **Data leakage**: during training the model sees information it wouldn't have in reality (e.g. its own answer).
- **Leave-one-out**: for a training passenger, compute a group statistic *without them*, so they never see their own answer.
- **RF / GBC**: Random Forest and Gradient Boosting, two families of models built from decision trees.

### 1. Tree models (RF, GBC) — ceiling around 0.775

Cleaning in `features.py`: cabin deck (unknown = deck "U"), missing port = S,
titles extracted from the name (Mr, Mrs, Miss, Master, "Rare" containing only men),
missing age = median of the title, then RF and GBC.

| Version | CV | Kaggle |
|---|---|---|
| RF + KNN (first model) | | 0.77511 |
| GBC titles, no PassengerId | | 0.74641 |
| RF titles, no PassengerId | | 0.75358 |
| GBC without cabin count | | 0.77511 |
| GBC deck U, port S, age by title, grouped titles | | 0.77511 |
| GBC + FamilySize, TicketGroup, **FarePerPerson** | 0.8323 | **0.73684** |
| GBC + FamilySize, TicketGroup, raw Fare | 0.8310 | 0.76794 |

**Finding**: the GBC loses ~6 points between CV (0.83) and Kaggle (0.77); it overfits.
The **FarePerPerson** trap (fare divided by the number of passengers on the ticket) shows it: best CV, worst Kaggle score.

### 2. Ticket-grouped CV — would not have caught the trap

Hypothesis: stratified CV is too optimistic because members of the same family end up in both training
*and* validation. So we set up a CV where a ticket is never split (`group_cv`).

- Lower, more honest scores: GBC 0.8193, RF 0.8063.
- But it **still ranked the FarePerPerson version first**: it would not have avoided the trap.

### 3. Group survival — the real lever

Idea: on the Titanic, women and children of the same family often lived or died together.

**`GroupSurv` feature** (survival rate of the *other* members of the same ticket, leave-one-out on the training set):
only **+0.2 pt** in the GBC. The signal is strong but concerns few passengers, and the trees drown it.
Note: ticket-grouped CV is unsuitable for this feature (0% of validation passengers have a known group,
versus 36% of the Kaggle test set); stratified CV is needed (34%).

**Rules applied after the model** (`group_rules`), looking only at the women and boys (title Master) of the group:
- a woman whose known women/boys in the group all died → **dies**;
- a boy whose known women/boys in the group all survived → **survives**.

| Version | Strat. CV | Kaggle |
|---|---|---|
| Gender only (women live, men die) | 0.7867 | 0.76555 |
| **Gender + rules** | 0.8368 | **0.79665** |
| GBC + rules | 0.8364 | 0.77272 |

- Gender + rules: 15 predictions changed versus gender only, **14 correct**.
- GBC + rules: differs from gender + rules on 40 passengers; the GBC is right 15 times and **wrong 25 times**.
  When the GBC contradicts the sex rule, it is wrong more often than right.
  CV (0.8364 vs 0.8368) did not see the difference.

### 4. Groups by surname

Hypothesis: some families have different tickets; also grouping by surname (+ class + port) would find more groups.

- **Surname alone** (ticket or surname): 3 predictions changed on the test set. On the training set: 6 corrections gained, **6 lost**.
  The losses all come from a **false group**: two unrelated Andersson families (tickets 347082 and 3101281).
- **Surname + nearby tickets** (`max_ticket_gap`): same surname, class, port, ticket prefix, and ticket numbers at most *k* apart.
  In the training set, every real family found is **1 apart** (Vander Planke 345763/4, Jussila 4136/7, Bourke 364848/9):
  6 corrections, 6 correct, and the false Andersson group disappears. Only k = 1 is validated by the training set; beyond it
  we link cases that cannot be checked (including another Andersson 9 apart).

| Version | Strat. CV | Kaggle |
|---|---|---|
| Gender + rules, ticket | 0.8368 | 0.79665 |
| **Gender + rules, ticket + surname (gap ≤ 1)** | 0.8431 | **0.79904** |

On the test set, a single prediction changes (Ida Ilmakangas), and it was correct. Tickets already capture almost every family.

### 5. State of the art (Chris Deotte's notebooks)

Read for comparison:
[Titanic using Name only](https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818) and
[Titanic WCG+XGBoost](https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688).

- His **WCG** (woman-child-groups) model applies the same two rules, with groups by surname
  (excluding adult men) completed by ticket.
- **Caution**: his scores (0.818 → 0.847) were computed on **209 passengers** (half of the test set);
  Kaggle now scores all 418. He himself estimated WCG at ~0.809 on the full test set.
  Our 0.799–0.801 is therefore on par with WCG.
- Predicting which **adult men** survive: patterns exist in the training set (1st class, age 20–36), but zero gain on the test set.
- Predicting which **solo third-class women** die: his best lever after WCG.

### 6. Solo third-class women

Rule (`solo_female_rule`, thresholds from one of Deotte's CART trees): a third-class woman with no family aboard,
alone on her ticket, with a **fare per person > $8 or age > 25 → dies**.

| Version | Strat. CV | Kaggle |
|---|---|---|
| Ticket + surname groups | 0.8431 | 0.79904 |
| **+ solo women, fixed thresholds ($8 / 25 years)** | 0.8498 | **0.80143** |
| + solo women, tree re-learned in each fold (honest version) | 0.8444 | not submitted |

- Training set: 20 women flagged, 13 actually died (65%).
- Test set: 17 changed, **9 correct and 8 wrong** (53%), a net gain of a single passenger.
- The honest version (the tree must find its thresholds without seeing the validation part) gains only +0.13 pt:
  the $8 / 25-year thresholds are partly tailored to the training set. A **fragile** rule, kept because it gives the best score.

---

## What we learned

1. **Sex does most of the work** (0.766). Family adds ~3 points. The rest comes down to a handful of passengers.
2. **Tree models overfit** on this small dataset: their deviations from the sex rule are mostly noise.
   A simple rule + a few targeted corrections beats all our RF/GBC models.
3. **CV is not enough to sort good ideas from bad**: it missed the FarePerPerson trap and the weakness of GBC + rules,
   and it overestimates rules whose thresholds were chosen on the whole training set. For a rule, prefer an
   "honest" version where the thresholds are re-learned in each fold.
4. **Each submission hinges on a few passengers**: a difference of 1 or 2 correct answers (0.24 to 0.5 pt) is down to chance.
5. **The realistic ceiling** without cheating is around 0.81–0.82 on the full test set.

## Next steps

- Align our groups with Deotte's WCG (surname alone + linking by ticket): he flags 14 women, we flag 8.
- Update notebook 04 to use `max_ticket_gap=1` and `solo_female_rule`.
- Cleanup: same imputer for RF and GBC, drop the StandardScaler (useless for trees), add an Age_missing column,
  handle Fare = 0, remove dead code from `features.py` (ticket_transformer, isNaN, rare_titles, outdated docstrings)
  and the old CSVs in `data/`.
