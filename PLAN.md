# PLAN — US PERM Visa Outcome Classification (v2: preprocessing + model placeholders)

**Module:** IT2011 AI/ML group assignment, Y2S1 2026 · **Group size:** 6 · **Group ID:** 2026-Y02-S1-MLB-B9G1-08 · **Plan owner:** IT25100285 (Stage 6: Feature Selection)
**Template:** the `Default-creditcard-client` repo structure. Its stage order is kept; all dataset-specific logic is replaced.
**Scope of this plan:** a cleaned, leakage-free, model-ready dataset that teammates can start training on, plus **6 model placeholders**. Model training, tuning and comparison come in a later plan.

---

## What changed from v1 (and why)

| # | v1 | v2 | Reason |
|---|---|---|---|
| 1 | Plan ran to tuning, test evaluation and saved models | **Stops at preprocessing + model placeholders** | Your request: hand the cleaned data over first |
| 2 | Extra folders: `src/`, `data/interim`, `data/sample`, `models/`, `results/metrics` | **Removed.** Only the guideline's folders are used, plus `data/external/` for lookup tables | Progress Review I guideline fixes the repo layout (see §1) |
| 3 | Visa-specific stage split | **Teammate repo's order:** S1 Missing/Invalid · S2 Encoding · S3 Outliers · S4 Feature Creation · S5 Scaling · **S6 Feature Selection (you)** | Your choice |
| 4 | 70/15/15 train/val/test | **80/20 stratified train/test + stratified 5-fold CV on train** | The spec, Step 4, requires k-fold cross-validation, so CV replaces a separate validation set |
| 5 | Optional temporal (2016) test | **Dropped.** Drift is shown in one EDA plot and listed as a report limitation | Not required by any guideline; it would add a second evaluation regime for all 6 members |
| 6 | SMOTE "if used" | **No resampling in the handover data.** Class weights by default; SMOTE is an *optional model variety*, used only inside CV training folds | Final Evaluation asks for "varieties with different pre-processing", so resampling belongs to each member's experiments, not to the shared data |
| 7 | One final dataset | **Two handover datasets:** A = cleaned and unscaled with all features; B = scaled and feature-selected | Your choice, and it gives members preprocessing varieties to compare |
| 8 | Stage 0 loader in `src/load_data.py` | Moved into a **"Stage 0: Load & integrate"** section at the top of the S1 notebook and of `group_pipeline.ipynb` | No `src/` in the guideline layout |
| 9 | `employer_filing_count` in S4; `wage_range_width` | Frequency encoding of employer moved to **S2**; `wage_range_width` **dropped** (`wage_offer_to` is 80% missing) | Stage logic and missing-data rule |

**Model choices** (you asked me to object where a model is a poor fit) are in §5.

---

## 1. Guideline cross-check (Progress Review I, page 2 layout)

| Guideline item | This plan | Status |
|---|---|---|
| `README.md`: overview, dataset, member roles, how to run | Yes, plus a feature dictionary and handover guide | ✅ |
| `data/raw/`: dataset as provided | `data/raw/us_perm_visas.csv` (**moved from the folder root**) | ✅ (gitignored on GitHub, but goes in the Courseweb zip; see risks) |
| `data/external/`: external reference data | `us_states.csv` (name ↔ code ↔ census region), `soc_major_groups.csv`, `naics_sectors.csv` | ✅ |
| `notebooks/IT_Number_Technique.ipynb`, one per member | 6 notebooks named with each member's IT number | ✅ |
| `group_pipeline.ipynb`: integrated pipeline | Yes, plus 6 model workspaces | ✅ |
| `results/eda_visualizations/` | `m0..m6_*.png` | ✅ |
| `results/logs/` (optional) | `pipeline_run.log` with rows and columns per stage | ✅ |
| `results/outputs/`: final processed dataset / features | Two handover `.csv.gz` files + feature list; stage intermediates as parquet | ✅ |
| Not in guideline but kept | `requirements.txt`, `.gitignore`, `.gitattributes`, `scripts/strip_notebook_runtime.js` (strips notebook outputs) | Dev tooling only; can be left out of the zip |

**Other guideline requirements this plan must support**
- **Spec, Step 2:** clean the data (missing values, outliers, encoding), produce EDA visualisations, and write up insights. ✅
- **Spec, Step 4:** k-fold CV, and metrics such as Accuracy, F1 and the confusion matrix. ✅ The split and CV protocol are fixed in the handover.
- **Final Evaluation:** each member trains several *varieties* of their model (different preprocessing and hyperparameters), and the group compares the 6 best models. ✅ Datasets A and B make this possible.
- **Spec, Step 5.7:** the report needs an **AI Tool Usage Declaration**. Record that Claude (Claude Code) helped plan the pipeline and profile the data.

### Final repo layout (`~/Desktop/us_prem_visa/` → zip root `2026-Y02-S1-MLB-B9G1-08/`)
```
us_prem_visa/
├── README.md · PLAN.md · requirements.txt · .gitignore · .gitattributes
├── data/
│   ├── raw/us_perm_visas.csv                      # 298.6 MB, gitignored
│   └── external/us_states.csv, soc_major_groups.csv, naics_sectors.csv
├── notebooks/
│   ├── IT25101547_MissingData.ipynb               # S1
│   ├── IT25103364_Encoding.ipynb                  # S2
│   ├── IT25101145_OutlierRemoval.ipynb            # S3
│   ├── IT25102357_FeatureEngineering.ipynb  # S4
│   ├── IT25103041_Scaling.ipynb                   # S5
│   └── IT25100285_FeatureSelection.ipynb          # S6 (you)
├── group_pipeline.ipynb                           # Stage 0–6 functions, run_pipeline(), 6 model workspaces
├── results/
│   ├── eda_visualizations/m0_dataset_overview.png … m6_feature_selection.png
│   ├── logs/pipeline_run.log
│   └── outputs/
│       ├── stage1_missing_handled.parquet … stage5_scaled.parquet   # gitignored, regenerated by run_pipeline()
│       ├── cleaned_unscaled_allfeatures.csv.gz    # HANDOVER A (committed)
│       ├── final_processed.csv.gz                 # HANDOVER B (committed)
│       ├── selected_features.txt · feature_dictionary.csv · scaler_params.csv
└── scripts/strip_notebook_runtime.js
```

---

## 2. Data facts (from chunked reads; the full file was never loaded at once)

- **Size:** 374,362 rows × 154 columns, 298.6 MB. Nearly every column loads as text; about 110k rows store numbers with commas (`"83,366.00"`).
- **Target `case_status`:** Certified 48.6% · Certified-Expired 39.7% · **Denied 6.85%** · Withdrawn 4.86%.
  - **Binary target `denied`** (1 = Denied; 0 = Certified or Certified-Expired), with Withdrawn dropped: 356,168 rows, **7.20% positive**.
  - Predicting "approved" for everyone scores **92.8% accuracy**, so accuracy alone is misleading.
- **Mixed form versions:** the same field appears under different column names. Combining them leaves almost nothing missing:
  - `country_of_citizenship` + `country_of_citzenship` → 0.02% missing
  - `wage_offer_from_9089` + `wage_offered_from_9089` → 0.03%
  - the three NAICS code columns → 30.8%
- **Missing blocks:** whole groups of columns are missing for exactly 36.1%, 59.8% or 76.3% of rows (one group per form version). About 30 columns are more than 90% missing.
- **Drift:** denial rate is ~10–13% in 2011–13 and ~5–6% in 2014–16. Volume grows from 11k rows (2011) to 124k (2016).
- **Dirty values:**
  - wage units mixed across 10 spellings (Year/yr/Hour/hr/Week/wk/Bi-Weekly/bi/Month/mth)
  - `employer_state` has 113 distinct values (codes and full names mixed)
  - `employer_yr_estab` minimum is 0
  - `employer_num_employees` maximum is 263.5M
  - 1,337 duplicate case IDs
- **High-cardinality columns:**

  | Column | Distinct values |
  |---|---|
  | `employer_name` | 71.7k |
  | `foreign_worker_info_major` | 21.9k |
  | `agent_firm_name` | 10.1k |
  | `job_info_work_city` | 9.4k |
  | `pw_job_title_9089` | 8.3k |
  | `pw_soc_code` | 1.4k |
  | country of citizenship | 202 |
  | `class_of_admission` | 57 |

- **Environment:** Python 3.13 with pandas 3.0.5 and numpy 2.5.3. **Not yet installed:** scikit-learn, pyarrow, imbalanced-learn. Node is available (needed for the notebook filter).

---

## 3. Repo audit and dataset comparison (summary from v1, still valid)

**Teammate repo:**
- 6 stage functions in `group_pipeline.ipynb`, each mirrored by one member notebook.
- Each stage reads the previous stage's CSV and writes its own.
- No split, models, evaluation or saved models; the 6 model cells are empty.

| Reuse as-is | Replace | Fix (don't copy) |
|---|---|---|
| Stage-handoff pattern, `plot_dir` argument, project-root finder, notebook markdown template (Technique → Justification → Why this stage order → EDA interpretation), `run_pipeline()` harness, notebook-output filter | All column names, category codes, target name, NT$ labels, the "top 15" rule, credit-specific features | Scaling and feature selection are fit on **all rows before any split** (leakage). Code is duplicated between notebooks and the pipeline. Notebook filter setup is undocumented. README links to files that don't exist. The S6 notebook says "PCA" but does none |

| | Teammates (UCI credit) | Yours (PERM visas) |
|---|---|---|
| Size | 30k × 25 | 374k × 154 (356k after framing) |
| Types | All numeric codes | Text, dates, numbers stored as text; 4–71k-level categoricals |
| Missing | None | Heavy and structured by form version |
| Positive class | 22.1% | **7.2%** |
| Drift | None | 2011–2016 |

### Leakage rules (enforced with code asserts)

| Column | Treatment |
|---|---|
| `decision_date` | **EDA-only.** Kept through S1–S5 for the drift and leakage plots; **dropped in S6** before either handover file is written (the "keep for preprocessing, drop later" item you agreed to) |
| Processing time (`decision_date − case_received_date`) | Median days: Denied 399, Certified 125. Plotted once in m0 to justify excluding it; **never created as a feature**. `case_received_date` is dropped after m0 |
| `case_no` / `case_number` / `orig_case_no` | Used only for de-duplication, then dropped |
| `application_type` and form-version "is-missing" flags | Act as a proxy for the year → dropped / never created |
| `pw_amount_9089` missingness | 8.3% missing for Denied vs ~0% for the other classes. Imputed **without** a flag; noted in the README so members can test removing it |
| `employer_name` | **Frequency** encoding fit on train only. No target encoding |
| `case_status` | Kept through S1 for the 4-class EDA, then replaced by `denied` |

---

## 4. Stage-by-stage design (each stage fits on train rows and transforms all rows)

**Shared conventions**
- Every notebook starts with an identical **Config cell**: paths, `TARGET="denied"`, `SEED=42`, `DEV_SAMPLE=None|0.10`, `EDA_ONLY_COLS`.
- Each stage reads the previous parquet file and writes its own.
- `split ∈ {train, test}` is a column, so a single file still flows between stages.
- `DEV_SAMPLE=0.10` gives a stratified ~35k-row run (similar in size to the teammates' dataset) for fast iteration.

### S1 — Missing & Invalid Data Handling · `IT25101547_MissingData.ipynb` (heaviest stage)

**Stage 0: load and integrate**
- Chunked `read_csv(usecols=~32 columns, dtype=str, chunksize=100_000)`.
- Combine duplicate-name columns: citizenship, wage from/to/unit, NAICS (3 columns), case ID.
- Strip commas, then `to_numeric` / `to_datetime`. Convert to `category` / `float32`.

**Target and rows**
- Build `denied`, drop Withdrawn.
- Remove duplicate case IDs, keeping the row with the latest decision.

**Split**
- Stratified 80/20 split on `denied` with `SEED=42`, stored in the `split` column.

**Invalid values**
- Convert wages to annual amounts (hour ×2080, week ×52, bi-weekly ×26, month ×12).
- Annual wage below $10k or above $1M → NaN.
- `employer_yr_estab` before 1800 or after 2016 → NaN.
- `employer_num_employees` ≤ 0 → NaN.
- Normalise states to 2-letter codes using `data/external/us_states.csv`.

**Missing values**
- Drop columns still more than 60% missing after combining.
- Numeric columns → train median. Categorical columns → `"Unknown"`.
- No missing-indicator flags (they would encode the form version).

**Plot:** `m1_missingness_before_after.png` (missing % per column: raw → combined → final).

**Done when:**
- 0 NaNs in kept columns.
- Denied rate in train and test is within ±0.2 pp of each other.
- Leakage columns are absent (asserted).
- Row counts are logged.

### S2 — Categorical Encoding · `IT25103364_Encoding.ipynb`

**Ordinal encoding**
- `pw_level_9089`: I–IV → 1–4.
- Worker and job education: High School → Doctorate as 1–5; Other/Unknown → 0.

**Binary encoding**
- `job_info_experience` Y/N.

**Grouping, then one-hot** (category lists fit on train; unseen values → Other)
- SOC code → 2-digit major group (~23).
- NAICS → 2-digit sector (~20).
- Citizenship → top 20 + Other.
- `class_of_admission` → top 10 + Other.
- State → census region.
- `pw_source_name`.

**Frequency encoding**
- `employer_name` → `employer_filing_count` (counts from train).

**Passthrough**
- Raw employer and work state are kept for S4 and dropped there.

**Plot:** `m2_denial_rate_by_category.png`.

**Done when:**
- All model features are numeric.
- An assert confirms category lists and counts came from train rows only.

### S3 — Outlier Treatment · `IT25101145_OutlierRemoval.ipynb`
- `log1p` of `employer_num_employees` and `employer_filing_count`.
- Winsorise at **train** P1/P99: annual wage, annual prevailing wage, log employer size, employer establishment year.
- **No rows dropped.** Denied cases tend to be the unusual ones, so deleting outliers would remove minority signal.
- Caps are saved in the log.

**Plot:** `m3_outlier_boxplots.png` (raw wage clearly shows the mixed-units problem; treated wage does not).

### S4 — Feature Engineering (Creation) · `IT25102357_FeatureEngineering.ipynb`

New features:
- `wage_to_pw_ratio`: offered wage ÷ prevailing wage, clipped to [0, 5].
- `wage_below_pw`: flag, ratio < 1.
- `employer_age`: 2016 − year established.
- `has_agent`: an attorney or agent firm is present.
- `same_state`: employer state equals work state; passthrough columns are dropped after this.
- `edu_meets_requirement`: worker education ≥ required education, where both are known.
- `is_h1b`: worker currently holds an H-1B.

**Plot:** `m4_engineered_feature_signal.png` (denial rate by `wage_to_pw_ratio` bins and by `has_agent`).

**Done when:** at least 6 new features exist and none contain inf or NaN.

### S5 — Scaling · `IT25103041_Scaling.ipynb`
- Z-score the continuous and ordinal columns using **train** mean and standard deviation (with a guard against σ≈0).
- One-hot and binary columns stay unscaled.
- Save `scaler_params.csv`.

**Plot:** `m5_scaling_comparison.png`.

**Done when:** train mean is ≈0 and std ≈1 (asserted). Test values not being exactly 0/1 is expected and should be explained in the viva.

### S6 — Feature Selection · `IT25100285_FeatureSelection.ipynb` (**you**)

**Steps**
1. Drop `EDA_ONLY_COLS` (`decision_date`, `case_status`) and assert that no leakage column remains.
2. **Correlation filter:** compute pairwise Pearson |r| on train and drop one column from each pair above 0.85 (e.g. annual wage vs prevailing wage, keeping the ratio and the prevailing wage). Plot a correlation heatmap.
3. **Supervised ranking:** `mutual_info_classif` on train, with the binary/one-hot columns marked as discrete and a fixed `random_state`. Choose k (~20–25) at the point where cumulative MI flattens.

**Outputs**
- **Handover A:** `cleaned_unscaled_allfeatures.csv.gz`. This is the S4 output minus the EDA-only columns: all candidate features, unscaled, plus `denied` and `split`.
- **Handover B:** `final_processed.csv.gz`. This is the S5 output restricted to the selected features: scaled, plus `denied` and `split`.
- `selected_features.txt` and `feature_dictionary.csv` (column, meaning, type, scaled?, in B?).

**Plot:** `m6_feature_selection.png` (MI ranking bars + correlation heatmap).

**Viva angle (why MI and not Pearson like the teammate repo):**
- MI captures non-linear and categorical relationships with the binary target.
- MI is fit on train only, which fixes the teammate repo's leakage.

### Group — `group_pipeline.ipynb`
- **m0 dataset overview** (group EDA): 4-class → binary target bars, denial rate by year (drift), processing-time box plot by class (the leakage evidence).
- Stage functions 0–6 and `run_pipeline(dev_sample=None)`, which writes `results/logs/pipeline_run.log`.
- **Integration check:** `assert_frame_equal(run_pipeline() output, the notebook chain's handover B)`.
- **6 model workspaces** (placeholders only; see §5).

---

## 5. Model placeholders and handover guide (for teammates)

Each workspace has a markdown heading (`### <IT number> — <Model>`) and a skeleton covering:
- loading A or B, with `X = df.drop(columns=["denied","split"])` on `split=="train"`
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `GridSearchCV` / `RandomizedSearchCV` with `scoring="average_precision"` (F1 also reported)
- a table of at least 3 varieties (A vs B, class weights vs SMOTE-in-CV, hyperparameter sets)
- a single final evaluation on `split=="test"`

**Metrics** (the spec lists Accuracy, F1 and confusion matrix):
- F1, precision and recall for **Denied**
- macro-F1
- PR-AUC
- ROC-AUC
- confusion matrix
- accuracy, always shown next to the **92.8% "always approve" baseline**

| Workspace | Suggested model | Dataset | Verdict for *this* task |
|---|---|---|---|
| IT25100285 (you) | **Logistic Regression** (`class_weight="balanced"`) | B | ✅ Good baseline and interpretable (coefficients tie to the S6 features) |
| IT25101547 | Decision Tree | A | ✅ Fine; no scaling needed |
| IT25103364 | Random Forest | A | ✅ Strong; use `n_jobs=-1` and run the grid search on a stratified subsample |
| IT25101145 | **XGBoost / LightGBM** (`scale_pos_weight≈12.9`) | A | ✅ **Recommended addition.** Usually the best model on tabular data like this; likely group winner |
| IT25102357 | **SVM** | B | ⚠️ **With conditions.** Kernel (RBF) SVC scales roughly O(n²) and is infeasible on ~285k training rows × 5 folds × a grid. Use `LinearSVC` on the full data, *or* RBF on a stratified subsample of ≤30k. No `predict_proba`, so use `decision_function` for the AUC metrics |
| IT25103041 | **MLP** (`MLPClassifier`, `early_stopping=True`) | B (must be scaled) | ✅ Replaces KNN (dropped: slow on 285k rows, distances poorly defined over one-hot columns, minority outvoted). MLP handles the size and mixed features, and is the guideline's "Deep Learning (MLP)" example. No `class_weight` → use SMOTE-in-CV or threshold tuning as its imbalance variety |

**Imbalance policy for all members:**
- Use class weights (or `scale_pos_weight`) by default.
- SMOTE is allowed only through `imblearn.pipeline.Pipeline` inside CV, so it only touches training folds.
- Never resample before the split, and never tune the threshold on the test set.

---

## 6. Sprint backlog (preprocessing scope)

Effort: S ≈ ≤2 h · M ≈ half a day to 1 day · L ≈ 1–2 days

| # | Task | Done when | Effort | Depends on |
|---|---|---|---|---|
| 1 | venv + `requirements.txt`: pandas, numpy, pyarrow, scikit-learn, matplotlib, seaborn, ipykernel, plus teammates' imbalanced-learn and xgboost/lightgbm, pinned to versions that also work on Colab (pandas 2.2.x) | `python -c "import pandas, pyarrow, sklearn"` works in VS Code's selected kernel | S | — |
| 2 | Scaffold: `git init`, guideline folders (+ `.gitkeep`), **move the CSV into `data/raw/`**, `.gitignore` (raw CSV, `*.parquet`, venv), copy `.gitattributes` + filter script, run `git config filter.notebook-runtime.clean "node scripts/strip_notebook_runtime.js"` | `git status` doesn't list the CSV; a committed notebook has no outputs | S | 1 |
| 3 | `data/external/` lookup tables (states → code → region, SOC major groups, NAICS sectors) | All 113 raw state values map to a code or `Unknown`; tables are under 10 KB | S | 2 |
| 4 | S1 notebook (Stage 0 + target + dedupe + split + invalid values + missing values) + m1 | S1 "done when" (§4) passes on DEV_SAMPLE and on the full data; log shows ~356k rows | L | 2, 3 |
| 5 | m0 overview cell in `group_pipeline.ipynb` | m0 PNG saved with the drift and processing-time panels | S | 4 |
| 6 | S2 Encoding + m2 | S2 "done when" | M | 4 |
| 7 | S3 Outliers + m3 | Rows in = rows out; caps logged | S | 6 |
| 8 | S4 Feature creation + m4 | ≥6 new features, no inf/NaN | M | 7 |
| 9 | S5 Scaling + m5 | Train mean/std asserts pass | S | 8 |
| 10 | **S6 Feature Selection (you)** + m6 + both handover files + feature dictionary | A and B written; B has ~20–25 features + `denied` + `split`; leakage assert passes | M | 9 |
| 11 | `run_pipeline()` + integration check + run log | A clean kernel runs end-to-end on the full data in under ~5 min; `assert_frame_equal` passes | M | 4–10 |
| 12 | 6 model workspace placeholders + handover-guide markdown (§5) | Each skeleton loads A or B and splits X/y without errors | S | 10 |
| 13 | README: overview, dataset, member/stage table, mermaid flow, how to run (VS Code + Colab), leakage table, feature dictionary link | A teammate can follow the README from a fresh clone | M | 11 |
| 14 | Handover: commit the `.csv.gz` files, push, teammate smoke test | One teammate loads B in Colab and one in VS Code, and gets the same shape and denied rate | S | 12, 13 |

### Risks and mitigations
- **S1 is the heaviest notebook:** do it first, on DEV_SAMPLE, and keep Stage 0 as a separate function.
- **Form-version proxy:** "Unknown"/median imputation for the 36%-missing block (employer size, year established, education) still partly encodes the year. Document it as a limitation; members can compare A with and without those columns as a variety.
- **pandas 3 (you) vs pandas 2.2 (Colab):** avoid pandas-3-only behaviour and chained assignment; the step-14 smoke test catches mismatches.
- **Duplicated Config cells across 7 notebooks:** the integration `assert_frame_equal` catches drift.
- **Repo size:** each `.csv.gz` is ~10–20 MB. Commit only at handover milestones, not after every run.
- **Courseweb zip:** the guideline wants the raw data "as provided" in `data/raw/`. 298 MB zips to roughly 40–60 MB; if the upload limit is lower, include a download link in the README instead.

---

## Assumptions
1. The CSV is **moved** to `data/raw/us_perm_visas.csv`, gitignored, and shared via the Kaggle link or Courseweb zip.
2. Target `denied` = 1 for Denied; Certified-Expired counts as approved; Withdrawn is dropped.
3. Only information available when the application is filed is used as a feature.
4. 80/20 stratified split + 5-fold stratified CV on train (`SEED=42`); no temporal split.
5. No resampling in the shared data; SMOTE is a per-member variety inside CV only.
6. Intermediate stage outputs are parquet (gitignored); the two handover datasets are `.csv.gz` (committed), which pandas reads directly in VS Code and Colab.
7. Stage notebooks for the other 5 members were written by IT25100285 and are owned / presented by each member (IT numbers in the filenames).
8. EDA is one plot per stage, chosen to justify a cleaning decision, plus the m0 group overview.

## Open items (non-blocking; needed before submission)
- The other 5 members' IT numbers, and which member gets which model workspace.
- Courseweb upload size limit (decides whether the raw CSV goes in the zip).

---

## Status — 2026-10-02 (preprocessing phase built)

**Done (backlog tasks 1–13):**
- Environment set up (`.venv`, `requirements.txt`).
- Repo scaffolded; CSV moved to `data/raw/`.
- `data/external/` lookup tables written.
- Stage notebooks S1–S6 written, executed and passing their asserts.
- `group_pipeline.ipynb` runs end to end in ~40 s (peak RAM 1.3 GB); integration check passed.
- 6 model workspaces added (KNN replaced by MLP).
- README written.

**Final numbers:**

| | |
|---|---|
| Rows | 354,849 (1,337 duplicate IDs removed, Withdrawn dropped) |
| Denied | 24,524 (6.91%) → "always approve" accuracy = **93.1%** (replaces the 92.8% estimate above) |
| Split | 283,879 train / 70,970 test |
| Handover A | 107 features, unscaled |
| Handover B | 25 features, scaled |
| Sanity check (untuned LR on B) | PR-AUC 0.32 vs 0.069 baseline, ROC-AUC 0.77 |

**Deviations from the plan (and why):**
- The git output-stripping filter is applied to `group_pipeline.ipynb` only, as in the teammate repo, so the stage notebooks keep their outputs (plots, checks) for GitHub viewing and the viva.
- `pwsrc_Unknown` (prevailing wage not recorded) is kept as a feature. It is filing-time information, not post-decision leakage. It is documented as a shortcut, and an ablation is recommended.
- The S6 notebook is named `IT25100285_FeatureSelection.ipynb`, and S3 outputs `stage3_outliers_treated.parquet`.

**Remaining:**
- Task 14: commit, push, teammate smoke test.
- Check the Courseweb upload limit for the raw CSV.
