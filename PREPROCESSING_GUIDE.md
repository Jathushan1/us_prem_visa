# Preprocessing Guide: Build the Dataset Yourself, Block by Block

**For:** IT25100285 (Stage 6 owner, Logistic Regression) · **Group:** 2026-Y02-S1-MLB-B9G1-08
**Goal:** understand *what* was done to turn the raw US PERM visa file into `final_processed.csv.gz`, *how* each step works, and *why* it was needed. You do it with your own hands, from scratch, and finish by proving your result is identical to the official dataset.

> Your stage (Stage 6, Feature Selection) gets the most detail: see Part 2 → Stage 6, then the viva questions in Part 4.

---

## Contents
- [Part 0: How to use this guide](#part-0-how-to-use-this-guide)
- [Part 1: The big picture](#part-1-the-big-picture)
- [Part 2: Hands-on, block by block](#part-2-hands-on-block-by-block)
  - [Block 1: Config](#block-1-config-shared-settings)
  - [Stage 0: Load & integrate](#stage-0-load--integrate-the-298-mb-file)
  - [Stage 1: Missing & invalid data](#stage-1-missing--invalid-data-it25101547)
  - [Stage 2: Encoding](#stage-2-categorical-encoding-it25103364)
  - [Stage 3: Outliers](#stage-3-outlier-treatment-it25101145)
  - [Stage 4: Feature creation](#stage-4-feature-engineering-it25102357)
  - [Stage 5: Scaling](#stage-5-scaling-it25103041)
  - [Stage 6: Feature selection (YOU)](#stage-6-feature-selection-it25100285--your-stage)
  - [Final exam: does your dataset match?](#final-exam-does-your-dataset-match-the-official-one)
- [Part 3: How the repo notebooks fit together](#part-3-how-the-repo-notebooks-fit-together)
- [Part 4: Viva preparation](#part-4-viva-preparation)
- [Part 5: Glossary](#part-5-glossary)
- [Part 6: What comes next (your Logistic Regression)](#part-6-what-comes-next)

---

## Part 0: How to use this guide

### Setup (once)
1. Open the repo folder `us_prem_visa` in VS Code.
2. Make sure the raw file exists: `data/raw/us_perm_visas.csv` (298.6 MB).
3. Activate the environment in the VS Code terminal:
   ```bash
   source .venv/bin/activate
   ```
4. Create a folder `practice/` in the repo root and a new notebook `practice/IT25100285_practice.ipynb`. Select the `.venv` kernel (top-right in VS Code).
5. Keep practice work out of GitHub by adding this line to `.gitignore`:
   ```
   practice/
   ```

### Rules for learning
- **Type the code; don't paste it.** Typing forces you to read every line.
- **One block = one notebook cell.** Run it, then compare with the **✅ Expected** output.
- After each block, answer the **🧠 Check yourself** questions out loud. That is viva practice.
- Your practice code **never writes into `results/outputs/`**, so you can't break the official files.

---

## Part 1: The big picture

### The problem
The US Department of Labor (DOL) decides **PERM labor certification** applications: an employer wants to hire a foreign worker permanently and must prove it is not undercutting US workers. We predict whether an application will be **denied**.

| | |
|---|---|
| Raw data | 374,362 applications × 154 columns, 2011–2016 |
| Target | `case_status` → **`denied`**: Denied = 1, Certified / Certified-Expired = 0, Withdrawn removed |
| Final data | 354,849 rows, **6.91% denied** (strong class imbalance) |
| Task | Binary classification |

### The three golden rules (repeat these in the viva)
1. **Fit on train only.** Every number we *learn from the data* (medians, category lists, percentiles, mean/std, feature rankings) is computed on the **training rows**, then applied to the test rows. Otherwise the test set "leaks" into training and the scores look better than they really are.
2. **No leakage columns.** A feature must be known **when the application is filed**. Anything known only after the decision (decision date, processing time) is forbidden.
3. **Don't delete the minority class.** Only 6.9% of rows are denials. We never remove rows to "clean" outliers, because unusual applications are exactly where denials live.

### The pipeline
```
raw CSV (374,362 × 154)
  │ Stage 0  load in chunks, keep 32 columns, merge duplicate form fields, parse numbers ── (374,362 × 26)
  │ Stage 1  target, de-duplicate, TRAIN/TEST SPLIT, fix invalid values, fill missing ──── (354,849 × 21)
  │ Stage 2  encode categories into numbers ───────────────────────────────────────────── (354,849 × 106)
  │ Stage 3  cap outliers (log + winsorise) ────────────────────────────────────────────── (354,849 × 106)
  │ Stage 4  create 8 new features ─────────────────────────────────────────────────────── (354,849 × 111)
  │ Stage 5  standardise 12 numeric columns ─────────────────────────────────────────────── (354,849 × 111)
  │ Stage 6  select 25 features (YOU)
  ├──► Handover B  final_processed.csv.gz              25 features, scaled   (DEFAULT for all models)
  └──► Handover A  cleaned_unscaled_allfeatures.csv.gz  107 features, unscaled (optional variety)
```
Each table has 2 extra non-feature columns at the end, `denied` and `split`. Until Stage 6 there are also 2 EDA-only columns at the start (`case_status`, `decision_date`).

---

## Part 2: Hands-on, block by block

> Every block below = one cell in `practice/IT25100285_practice.ipynb`. Run them **in order**. The full run takes about 1 minute.

### Block 1: Config (shared settings)

**What:** imports, file paths, and settings every stage uses.
**Why:** one place for constants means all 6 notebooks behave identically (same seed, same target, same lookups).

```python
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

# Find the project root: walk up until we find data/raw
ROOT = os.path.abspath(os.getcwd())
while not os.path.isdir(os.path.join(ROOT, 'data', 'raw')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

RAW_CSV = os.path.join(ROOT, 'data', 'raw', 'us_perm_visas.csv')
EXT_DIR = os.path.join(ROOT, 'data', 'external')
OUT_DIR = os.path.join(ROOT, 'results', 'outputs')     # official outputs: we only READ from here

TARGET = 'denied'
SEED = 42                 # same random seed everywhere -> reproducible results
TEST_SIZE = 0.20          # 80% train / 20% test
EDA_ONLY_COLS = ['case_status', 'decision_date']                                         # for charts only, removed in S6
LEAKAGE_COLS = ['case_id', 'case_received_date', 'application_type', 'processing_days']   # must never reach a model
NON_FEATURE_COLS = EDA_ONLY_COLS + [TARGET, 'split']
REFERENCE_YEAR = 2016
S4_PASSTHROUGH = ['employer_state', 'job_info_work_state', 'agent_firm_name']
SCALE_COLS = ['wage_offer_annual', 'pw_annual', 'wage_gap_annual', 'wage_to_pw_ratio',
              'employer_num_employees_log', 'employer_filing_count_log', 'employer_yr_estab', 'employer_age',
              'pw_level', 'edu_worker', 'edu_required', 'edu_gap']

# Lookup tables from data/external (see README section on external data)
_states = pd.read_csv(os.path.join(EXT_DIR, 'us_states.csv'))
STATE_LOOKUP = {**dict(zip(_states['state_name'], _states['state_code'])),
                **dict(zip(_states['state_code'], _states['state_code']))}
STATE_REGION = dict(zip(_states['state_code'], _states['census_region']))
VALID_SOC = set(pd.read_csv(os.path.join(EXT_DIR, 'soc_major_groups.csv'), dtype=str)['soc_major_group'])
_naics = pd.read_csv(os.path.join(EXT_DIR, 'naics_sectors.csv'), dtype=str)
VALID_NAICS = set(_naics['naics_2digit'])
NAICS_SECTOR_KEY = dict(zip(_naics['naics_2digit'], _naics['sector_key']))
print('Project root:', ROOT)
print(STATE_LOOKUP['CALIFORNIA'], STATE_LOOKUP['CA'], STATE_REGION['CA'])
```
✅ **Expected:** your project path, then `CA CA West`.

🧠 **Check yourself**
- Why does `STATE_LOOKUP` map *both* `CALIFORNIA` and `CA` to `CA`? *(The raw file mixes full names and codes.)*
- Why is `SEED` fixed? *(The same random split and the same results every run, for every member.)*

---

### Stage 0: Load & integrate the 298 MB file

#### Block 2: Decide which columns to keep
**What:** keep 32 of 154 columns; list the duplicate-named fields to merge.
**Why:**
- 73 of the 154 columns are more than 60% missing.
- Many columns are free text (addresses, job titles, phone numbers) or recruitment-ad details.
- We keep the core facts: employer, job, wage, worker, plus the target and the dates/IDs needed for cleaning and EDA.

```python
RAW_KEEP = [
    'case_status', 'decision_date', 'case_received_date', 'case_no', 'case_number', 'application_type',
    'employer_name', 'employer_state', 'employer_num_employees', 'employer_yr_estab', 'job_info_work_state',
    'pw_soc_code', 'pw_level_9089', 'pw_amount_9089', 'pw_unit_of_pay_9089', 'pw_source_name_9089',
    'wage_offer_from_9089', 'wage_offered_from_9089', 'wage_offer_unit_of_pay_9089', 'wage_offered_unit_of_pay_9089',
    'naics_2007_us_code', 'naics_us_code', 'naics_code',
    'country_of_citizenship', 'country_of_citzenship', 'class_of_admission',
    'foreign_worker_info_education', 'job_info_education', 'job_info_experience',
    'agent_firm_name', 'refile', 'us_economic_sector',
]
# Same field, different names in different versions of the DOL form
COALESCE_MAP = {
    'case_id': ['case_no', 'case_number'],
    'country_of_citizenship': ['country_of_citizenship', 'country_of_citzenship'],   # note the typo column!
    'wage_offer_from': ['wage_offer_from_9089', 'wage_offered_from_9089'],
    'wage_offer_unit': ['wage_offer_unit_of_pay_9089', 'wage_offered_unit_of_pay_9089'],
    'naics_code': ['naics_2007_us_code', 'naics_us_code', 'naics_code'],
}
NUMERIC_TEXT_COLS = ['wage_offer_from', 'pw_amount_9089', 'employer_num_employees', 'employer_yr_estab']
print(len(RAW_KEEP), 'columns kept')
```
✅ **Expected:** `32 columns kept`

#### Block 3: Read the file in chunks
**What:** read 100,000 rows at a time, count missing values for **all** 154 columns, keep only our 32.
**Why:**
- Loading all 154 columns at once wastes memory.
- Reading as `dtype=str` stops pandas guessing types wrongly; for example, numbers stored as `"83,366.00"` would break.

```python
n_rows, notna, parts = 0, None, []
for chunk in pd.read_csv(RAW_CSV, dtype=str, chunksize=100_000):
    n_rows += len(chunk)
    counts = chunk.notna().sum()                        # non-missing count per column in this chunk
    notna = counts if notna is None else notna + counts
    parts.append(chunk[RAW_KEEP])
df = pd.concat(parts, ignore_index=True)
raw_missing = (1 - notna / n_rows).mul(100).rename('missing_pct_raw').to_frame()

print(n_rows, 'rows,', len(notna), 'raw columns ->', df.shape)
print((raw_missing['missing_pct_raw'] > 60).sum(), 'raw columns are more than 60% missing')
```
✅ **Expected:** `374362 rows, 154 raw columns -> (374362, 32)` and `73 raw columns are more than 60% missing`

#### Block 4: Merge duplicate fields, parse numbers and dates
**What:**
- `combine_first` takes the value from the first column and fills its gaps from the next one.
- Commas are removed from numbers.
- Dates are parsed.
- Text is upper-cased so `"ca "` and `"CA"` match.

**Why:** this is the dataset's biggest hidden problem. Citizenship is 5.5% missing in one column and 94.5% in the misspelt one. They are the same field from different form versions, and merged they are only **0.02%** missing.

```python
for new_col, old_cols in COALESCE_MAP.items():
    merged = df[old_cols[0]]
    for col in old_cols[1:]:
        merged = merged.combine_first(df[col])
    df = df.drop(columns=old_cols)
    df[new_col] = merged
    print(f'{new_col:<24} missing after merge: {merged.isna().mean():.2%}')

for col in NUMERIC_TEXT_COLS:
    df[col] = pd.to_numeric(df[col].str.replace(',', '', regex=False), errors='coerce')
for col in ['decision_date', 'case_received_date']:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
for col in ['employer_state', 'job_info_work_state', 'country_of_citizenship', 'employer_name', 'agent_firm_name']:
    df[col] = df[col].str.strip().str.upper()

df_0 = df.reset_index(drop=True)
print(df_0.shape)
```
✅ **Expected:** `case_id 0.00%`, `country_of_citizenship 0.02%`, `wage_offer_from 0.03%`, `wage_offer_unit 12.08%`, `naics_code 30.79%`, then `(374362, 26)`.

🧠 **Check yourself**
- Why are there 26 columns now, not 32? *(5 groups of 2–3 columns were merged into 5 columns.)*
- Explore it yourself: `df_0['pw_unit_of_pay_9089'].value_counts()` shows `Year`, `yr`, `Hour`, `hr`, … That's why Stage 1 needs to annualise wages.

---

### Stage 1: Missing & invalid data (IT25101547)

#### Block 5: Target, duplicates, leakage
**What:**
- Keep only the latest decision per case ID.
- Drop *Withdrawn*.
- Create `denied`.
- Drop the leakage columns.

**Why:**
- *Withdrawn* is the applicant's choice, not a DOL decision.
- `case_received_date` lets you compute **processing time**, which is only known *after* the decision. Denied cases took a median of **398 days** vs **125** for certified ones, so a model would "cheat" with it.
- `application_type` mostly identifies the old paper form, which is a proxy for the year.

```python
d = df_0.copy()
n_raw = len(d)
d = d.sort_values('decision_date', kind='stable').drop_duplicates('case_id', keep='last').sort_index()
print('duplicate case IDs removed:', n_raw - len(d))

d = d[d['case_status'] != 'Withdrawn'].copy()
d[TARGET] = (d['case_status'] == 'Denied').astype('int8')
d = d.drop(columns=['case_id', 'case_received_date', 'application_type']).reset_index(drop=True)
print(d.shape, f'denied rate = {d[TARGET].mean():.2%}')
```
✅ **Expected:** `duplicate case IDs removed: 1337` then `(354849, 24) denied rate = 6.91%`

#### Block 6: The train/test split (the most important line in the project)
**What:** a stratified 80/20 split, stored as a `split` column.
**Why:**
- *Stratified* keeps exactly 6.91% denied in both parts.
- It happens **now**, before anything is learned from the data, so later stages can fit on train rows only.
- Storing it as a column means every member uses the **same rows**.

```python
_, test_idx = train_test_split(d.index, test_size=TEST_SIZE, stratify=d[TARGET], random_state=SEED)
d['split'] = 'train'
d.loc[test_idx, 'split'] = 'test'
train = d['split'].eq('train')          # True/False mask we reuse below
print(d.groupby('split')[TARGET].agg(rows='size', denied='sum', rate='mean'))
```
✅ **Expected:** test 70,970 rows / 4,905 denied; train 283,879 / 19,619; both rates 0.0691.

🧠 **Check yourself:** what would go wrong if we computed medians on *all* rows and *then* split? *(The test rows would influence the training data: leakage.)*

#### Block 7: Fix invalid wages (unit conversion)
**What:** convert every wage to **USD per year**.
**Why:**
- The same job can be listed as `"45.00 hr"` or `"93,600 Year"`.
- Unknown units are inferred: ≤ 200 looks like an hourly rate.
- Results outside USD 10k–1M are impossible, so they become `NaN` and are filled later.

```python
UNIT_TO_ANNUAL = {'year': 1, 'yr': 1, 'hour': 2080, 'hr': 2080, 'week': 52, 'wk': 52,
                  'bi-weekly': 26, 'bi': 26, 'month': 12, 'mth': 12}      # 2080 = 40 h x 52 weeks
WAGE_MIN, WAGE_MAX = 10_000, 1_000_000

def annualise_wage(amount, unit):
    multiplier = unit.str.strip().str.lower().map(UNIT_TO_ANNUAL)
    inferred = pd.Series(np.where(amount <= 200, 2080, 1), index=amount.index)
    annual = amount * multiplier.fillna(inferred)
    return annual.where(annual.between(WAGE_MIN, WAGE_MAX))

d['wage_offer_annual'] = annualise_wage(d['wage_offer_from'], d['wage_offer_unit'].fillna(d['pw_unit_of_pay_9089']))
d['pw_annual'] = annualise_wage(d['pw_amount_9089'], d['pw_unit_of_pay_9089'])
d = d.drop(columns=['wage_offer_from', 'wage_offer_unit', 'pw_amount_9089', 'pw_unit_of_pay_9089'])
print(d[['wage_offer_annual', 'pw_annual']].describe().round(0))
```
✅ **Expected:** both columns have a max of at most 1,000,000 and a median around 86k–92k.

#### Block 8: Fix invalid states, codes, years and sizes
**What:**
- States are mapped to 2-letter codes.
- SOC and NAICS codes are cut to 2 digits and validated against `data/external`.
- Impossible founding years and employer sizes become `NaN`.

**Why:**
- About 3,400 SOC codes were corrupted into dates (`2021-11-01`).
- Founding year `0` and employer size 263 million are impossible.

```python
for col in ['employer_state', 'job_info_work_state']:
    d[col] = d[col].map(STATE_LOOKUP)
soc = d['pw_soc_code'].str.extract(r'^\s*(\d{2})-\d{4}')[0]
d['soc_major_group'] = soc.where(soc.isin(VALID_SOC))
naics = d['naics_code'].str.extract(r'^\s*(\d{2})')[0]
d['naics_2digit'] = naics.where(naics.isin(VALID_NAICS))
d = d.drop(columns=['pw_soc_code', 'naics_code'])
d['employer_yr_estab'] = d['employer_yr_estab'].where(d['employer_yr_estab'].between(1800, REFERENCE_YEAR))
d['employer_num_employees'] = d['employer_num_employees'].where(d['employer_num_employees'].between(1, 2_500_000))
print(d['soc_major_group'].value_counts().head(3))
```
✅ **Expected:** `15` (computer jobs) is by far the most common group.

#### Block 9: Missing values (drop, then fill)
**What:**
- Drop columns more than 60% missing (measured on train).
- Fill numeric columns with the **train median** and text columns with `"Unknown"`.

**Why:**
- More than 60% missing means we'd invent most of the column.
- The median is robust to skewed wages.
- **No "was missing" flag columns:** missingness mostly reveals the *form version*, which is a proxy for the *year*, and the denial rate changes by year.

```python
features = [c for c in d.columns if c not in NON_FEATURE_COLS]
miss_train = d.loc[train, features].isna().mean()
dropped = miss_train[miss_train > 0.60].index.tolist()
d = d.drop(columns=dropped)
features = [c for c in features if c not in dropped]
numeric = [c for c in features if pd.api.types.is_numeric_dtype(d[c])]
categorical = [c for c in features if c not in numeric]
medians = d.loc[train, numeric].median()               # learned from TRAIN only
d[numeric] = d[numeric].fillna(medians).astype('float32')
d[categorical] = d[categorical].fillna('Unknown')

df_1 = d[EDA_ONLY_COLS + sorted(features) + [TARGET, 'split']]
print('dropped:', dropped)
print('medians:', medians.round(0).to_dict())
print(df_1.shape, '| NaNs left:', int(df_1[sorted(features)].isna().sum().sum()))
```
✅ **Expected:** `dropped: ['refile', 'us_economic_sector']`, medians `employer_num_employees 1269, employer_yr_estab 1996, wage_offer_annual 91562, pw_annual 86278`, then `(354849, 21) | NaNs left: 0`.

🧠 **Check yourself:** a viva examiner asks "why median, not mean?" *(Wages and employer sizes are right-skewed; a few huge values pull the mean up, but the median ignores them.)*

---

### Stage 2: Categorical encoding (IT25103364)

#### Block 10: Ordinal and binary encoding
**What:** turn ordered categories into numbers that keep the order.
**Why:**
- Wage levels I → IV and education High School → Doctorate have a real order, so numbers are honest.
- `Unknown` / `Other` become 0.

```python
EDU_ORDER = {'High School': 1, "Associate's": 2, "Bachelor's": 3, "Master's": 4, 'Doctorate': 5}
PW_LEVEL_ORDER = {'Level I': 1, 'Level II': 2, 'Level III': 3, 'Level IV': 4}

d = df_1.copy()
d['pw_level'] = d['pw_level_9089'].map(PW_LEVEL_ORDER).fillna(0).astype('int8')
d['edu_worker'] = d['foreign_worker_info_education'].map(EDU_ORDER).fillna(0).astype('int8')
d['edu_required'] = d['job_info_education'].map(EDU_ORDER).fillna(0).astype('int8')
d['experience_required'] = d['job_info_experience'].eq('Y').astype('int8')
print(d.groupby('edu_worker')[TARGET].mean().round(3))
```
✅ **Expected:** denial rates of about 0.21 for High School (1) and about 0.03 for Doctorate (5). The order clearly matters.

#### Block 11: One-hot encoding with categories learned from train
**What:**
- Group big code lists first: SOC → 23 major groups, NAICS → 20 sectors, state → 5 regions, top-20 countries, top-10 visa classes.
- Then make one 0/1 column per category.

**Why:**
- No false order between categories.
- Grouping avoids thousands of near-empty columns.
- Category lists come from **train rows**; anything else goes to `Other` / `Unknown`.

```python
TOP_K = {'country_of_citizenship': 20, 'class_of_admission': 10}

def _clean_name(value):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(value)).strip('_')

def one_hot(series, categories, prefix, fallback='Other'):
    cats = list(categories) + ([fallback] if fallback not in categories else [])
    values = pd.Categorical(series.where(series.isin(cats), fallback), categories=cats)
    dummies = pd.get_dummies(values, dtype='int8')
    dummies.columns = [f'{prefix}_{_clean_name(c)}' for c in cats]
    dummies.index = series.index
    return dummies

train = d['split'].eq('train')
d['employer_region'] = d['employer_state'].map(STATE_REGION).fillna('Unknown')
d['naics_sector'] = d['naics_2digit'].map(NAICS_SECTOR_KEY).fillna('Unknown')
specs = {'soc_major_group': 'soc', 'naics_sector': 'naics', 'employer_region': 'region',
         'pw_source_name_9089': 'pwsrc', 'country_of_citizenship': 'country', 'class_of_admission': 'coa'}
fitted = {}
for col, prefix in specs.items():
    counts = d.loc[train, col].value_counts()                       # TRAIN only
    fitted[col] = sorted(counts.head(TOP_K[col]).index) if col in TOP_K else sorted(counts.index)
encoded = [one_hot(d[col], fitted[col], prefix, fallback='Other' if col in TOP_K else 'Unknown')
           for col, prefix in specs.items()]
print({prefix: e.shape[1] for (col, prefix), e in zip(specs.items(), encoded)})
```
✅ **Expected:** `{'soc': 23, 'naics': 21, 'region': 7, 'pwsrc': 7, 'country': 21, 'coa': 11}`

#### Block 12: Frequency encoding, then assemble
**What:** replace `employer_name` (71,709 distinct values) with *how many other train applications that employer filed*.
**Why:**
- One-hot would create 71k columns.
- The count captures "big repeat filer vs one-off employer", which is very predictive.
- We subtract the row itself on train, so a train row and a test row mean the same thing ("other filings").

```python
counts = d.loc[train, 'employer_name'].value_counts()
filing = d['employer_name'].map(counts).fillna(0)
d['employer_filing_count'] = (filing - train.astype(int)).astype('int32')

drop = list(specs) + ['pw_level_9089', 'foreign_worker_info_education', 'job_info_education',
                      'job_info_experience', 'naics_2digit', 'employer_name']
d = pd.concat([d.drop(columns=drop)] + encoded, axis=1)
features = [c for c in d.columns if c not in NON_FEATURE_COLS]
df_2 = d[EDA_ONLY_COLS + features + [TARGET, 'split']]
print(df_2.shape)
```
✅ **Expected:** `(354849, 106)`

🧠 **Check yourself:** why not *target encoding* (replace employer with its denial rate)? *(It uses the label itself; without careful out-of-fold handling it leaks the answer into the feature.)*

---

### Stage 3: Outlier treatment (IT25101145)

#### Block 13: log1p + winsorise at train P1/P99
**What:**
- Log-transform the counts (employer size, filing count).
- Clip 5 columns to the train 1st–99th percentiles.

**Why:**
- Employer size ranges from 1 to 2.3 million; on a log scale it becomes readable.
- Clipping keeps extreme values from dominating Logistic Regression, SVM and MLP.
- **No row is deleted.**

```python
d = df_2.copy()
train = d['split'].eq('train')
d['employer_num_employees_log'] = np.log1p(d['employer_num_employees']).astype('float32')
d['employer_filing_count_log'] = np.log1p(d['employer_filing_count']).astype('float32')
d = d.drop(columns=['employer_num_employees', 'employer_filing_count'])

for col in ['wage_offer_annual', 'pw_annual', 'employer_num_employees_log', 'employer_filing_count_log', 'employer_yr_estab']:
    lo, hi = d.loc[train, col].quantile([0.01, 0.99])          # TRAIN percentiles
    print(f'{col:<28} cap to [{lo:,.2f}, {hi:,.2f}]  ({((d[col] < lo) | (d[col] > hi)).mean():.2%} of rows capped)')
    d[col] = d[col].clip(lo, hi).astype('float32')
df_3 = d
print(df_3.shape)
```
✅ **Expected:** offered wage capped to [16,869, 225,000] (1.95% of rows), and shape `(354849, 106)`, so **no rows lost**.

🧠 **Check yourself:** "Why not remove outliers with the IQR rule?" *(It would delete thousands of rows, many of them denials, the very cases we want to learn.)*

---

### Stage 4: Feature engineering (IT25102357)

#### Block 14: Create 8 domain features
**What / why:** PERM law requires the offered wage to be **≥ the prevailing wage**. We turn that rule, and other domain knowledge, into features.

```python
d = df_3.copy()
d['wage_to_pw_ratio'] = (d['wage_offer_annual'] / d['pw_annual'].clip(lower=1)).clip(0, 5).astype('float32')
d['wage_below_pw'] = (d['wage_to_pw_ratio'] < 1).astype('int8')
d['wage_gap_annual'] = (d['wage_offer_annual'] - d['pw_annual']).astype('float32')
d['employer_age'] = (REFERENCE_YEAR - d['employer_yr_estab']).clip(lower=0).astype('float32')
d['has_agent'] = d['agent_firm_name'].ne('Unknown').astype('int8')
d['same_state'] = (d['employer_state'].eq(d['job_info_work_state']) & d['employer_state'].ne('Unknown')).astype('int8')
known = (d['edu_worker'] > 0) & (d['edu_required'] > 0)
d['edu_meets_requirement'] = (known & (d['edu_worker'] >= d['edu_required'])).astype('int8')
d['edu_gap'] = np.where(known, d['edu_worker'] - d['edu_required'], 0).astype('int8')
df_4 = d.drop(columns=S4_PASSTHROUGH)

t = df_4[df_4['split'].eq('train')]
print(df_4.shape)
print('denial rate when wage_below_pw = 1:', round(t.loc[t.wage_below_pw == 1, TARGET].mean(), 3),
      '| = 0:', round(t.loc[t.wage_below_pw == 0, TARGET].mean(), 3))
```
✅ **Expected:** `(354849, 111)` and `0.65` vs `0.065`. **This is the strongest finding in the project.**

---

### Stage 5: Scaling (IT25103041)

#### Block 15: z-score with train mean/std
**What:** `z = (x − mean) / std` for 12 numeric/ordinal columns. One-hot and 0/1 flags are left unchanged.
**Why:**
- Wages (~100,000) and ratios (~1) live on different scales.
- Logistic Regression (yours!), SVM and MLP need comparable scales.
- The mean and std come from **train** only.

```python
d = df_4.copy()
train = d['split'].eq('train')
mu = d.loc[train, SCALE_COLS].mean()
sigma = d.loc[train, SCALE_COLS].std().replace(0, 1.0)
d[SCALE_COLS] = ((d[SCALE_COLS] - mu) / sigma).astype('float32')
df_5 = d
print('train mean (max |.|):', round(float(df_5.loc[train, SCALE_COLS].mean().abs().max()), 4))
print('train std  (max |1-.|):', round(float((df_5.loc[train, SCALE_COLS].std() - 1).abs().max()), 4))
print('test mean of wage_offer_annual:', round(float(df_5.loc[~train, 'wage_offer_annual'].mean()), 4))
```
✅ **Expected:** about 0 and about 0 for train. The test mean is *close to* but **not exactly** 0, which is correct: the test rows were scaled with the train statistics.

---

### Stage 6: Feature selection (IT25100285): YOUR STAGE

> Read this part slowly. Everything here is likely to be asked in your viva.

#### 6.1 The problem your stage solves
After Stage 5 there are **107 candidate features** for 283,879 training rows:
- **Too many weak features.** Most one-hot columns (e.g. `country_ITALY`) carry almost no information about denial. They add noise, slow models down and make Logistic Regression harder to interpret.
- **Redundant pairs.** Some features say the same thing twice. That causes **multicollinearity**: in Logistic Regression the coefficients of correlated features become unstable and hard to explain.

So your stage answers two questions:
1. **Which features carry information about `denied`?** Answered with **mutual information**.
2. **Which features are duplicates of each other?** Answered with the **correlation filter**.

#### 6.2 Concept: mutual information (MI)
**In one sentence:** MI measures *how much knowing a feature reduces your uncertainty about the target*.
- MI = 0 means the feature tells you nothing about denial (independent).
- A larger MI means the feature tells you more.
- It is measured in *nats* (natural-log units), and it is never negative.

**The formula, for two discrete variables:**

$$ MI(X;Y) = \sum_{x}\sum_{y} P(x,y)\,\log\frac{P(x,y)}{P(x)\,P(y)} $$

If X and Y are independent, then P(x,y) = P(x)·P(y), every log term is log(1) = 0, and so MI = 0.

**A worked example on your own data: `wage_below_pw` vs `denied` (train rows)**

| | denied = 0 | denied = 1 | total |
|---|---|---|---|
| wage_below_pw = 0 | 263,502 | 18,212 | 281,714 |
| wage_below_pw = 1 | 758 | **1,407** | 2,165 |

- Overall, 6.9% are denied. When `wage_below_pw = 1`, **65%** are denied.
- Only 0.76% of applications have this flag, so on average it reduces uncertainty only a little: MI = **0.0087 nats**. This is computed by hand on all train rows; Block 17 prints ≈0.0090 because it uses the 60k-row sample.
- That is why MI ranks it 10th, not 1st. MI rewards features that help for *many* rows (like the wage itself), not just rare but dramatic ones.

**For continuous features** (wages, ratios), sklearn's `mutual_info_classif` uses a **k-nearest-neighbour estimator** instead of counting. That's why we tell it which columns are discrete (`discrete_features=`), and why it needs a `random_state` (it adds tiny noise to break ties).

#### 6.3 Why MI and not Pearson correlation (your strongest viva argument)
Pearson correlation only measures **straight-line** relationships.

| Feature | Pearson r with `denied` | What really happens |
|---|---|---|
| `wage_to_pw_ratio` | **−0.066** (looks almost useless!) | Denial jumps from ~7% to ~65% when the ratio drops below 1: a **threshold**, not a line |

MI catches non-linear patterns like this threshold; Pearson misses them. The teammate repo we used as a template ranked features with Pearson r. For this dataset that would have under-rated the most important mechanism.

MI also works naturally for **binary / one-hot** features, which make up most of our 107 columns.

#### 6.4 Concept: the correlation (redundancy) filter
Two features with |r| > 0.85 carry nearly the same information. Keeping both:
- adds nothing new for the model,
- makes Logistic Regression coefficients unstable (multicollinearity),
- double-counts one idea.

**The greedy algorithm (what the code does):**
1. Sort all features by MI, highest first.
2. Walk down the list. For each feature, check whether it has |r| > 0.85 with any feature **already kept**:
   - no → **keep** it
   - yes → **drop** it and record which kept feature it duplicates

Because we walk in MI order, in every redundant pair the **more informative** feature survives.

**The 5 pairs it removed (train rows):**

| Dropped | Kept instead | r | Why they are duplicates |
|---|---|---|---|
| `pw_annual` | `wage_offer_annual` | +0.859 | Employers usually offer ≈ the prevailing wage |
| `wage_gap_annual` | `wage_to_pw_ratio` | +0.910 | Both measure offer vs prevailing wage |
| `edu_worker` | `edu_required` | +0.952 | Workers usually have exactly the required education |
| `employer_yr_estab` | `employer_age` | **−1.000** | `age = 2016 − year`: the same information |
| `pwsrc_Other` | `pwsrc_OES` | −0.885 | Two dummies of one column; when one is 1 the other is 0 |

#### 6.5 Choosing k (how many features to keep)
- **Rule:** keep features in MI order until they cover **90% of the total MI** of the remaining features, with at least 15 and at most 30.
- **Result:** 24 features cover 89.5% and 25 cover **90.2%** → **k = 25**.
- **Why a rule and not a guess:** it's reproducible and explainable. The last ~75 features together add only 10% of the information.

#### 6.6 Why it's leakage-safe
- MI and correlations are computed on **train rows only**. (MI uses a stratified 60,000-row train subsample to save time; the ranking is stable at that size.)
- `decision_date` and `case_status` are dropped here, and an `assert` stops the code if any leakage column survives.

#### Block 16: Drop EDA-only columns and check for leakage
```python
CORR_THRESHOLD, MI_SAMPLE, MI_CUMULATIVE, K_BOUNDS = 0.85, 60_000, 0.90, (15, 30)

a = df_4.drop(columns=EDA_ONLY_COLS)      # unscaled   -> handover A
s = df_5.drop(columns=EDA_ONLY_COLS)      # scaled     -> handover B
leaked = [c for c in LEAKAGE_COLS + EDA_ONLY_COLS if c in a.columns or c in s.columns]
assert not leaked, f'Leakage columns present: {leaked}'
features = [c for c in s.columns if c not in (TARGET, 'split')]
print(len(features), 'candidate features, no leakage columns')
```
✅ **Expected:** `107 candidate features, no leakage columns`

#### Block 17: Rank by mutual information
```python
train = s['split'].eq('train')
X, y = s.loc[train, features], s.loc[train, TARGET]
X_mi, _, y_mi, _ = train_test_split(X, y, train_size=MI_SAMPLE, stratify=y, random_state=SEED)
discrete = np.array([c not in SCALE_COLS for c in features])      # one-hot / binary = discrete
mi = pd.Series(mutual_info_classif(X_mi, y_mi, discrete_features=discrete, random_state=SEED), index=features)
mi = mi.sort_values(ascending=False)
print(mi.head(10).round(4))
```
✅ **Expected:** top 5 are `wage_offer_annual ≈ 0.0407`, `pw_annual ≈ 0.0402`, `employer_filing_count_log ≈ 0.0254`, `employer_num_employees_log ≈ 0.0236`, `wage_to_pw_ratio ≈ 0.0235`. (Takes about 20 seconds.)

#### Block 18: The greedy correlation filter
```python
corr = X.corr().abs()
kept, dropped_for = [], {}
for col in mi.index:                                   # highest MI first
    partner = next((k for k in kept if corr.loc[col, k] > CORR_THRESHOLD), None)
    if partner is None:
        kept.append(col)
    else:
        dropped_for[col] = partner
print('dropped as redundant:', dropped_for)
```
✅ **Expected:** the 5 pairs from the table in 6.4.

#### Block 19: Choose k and build the two handover datasets
```python
candidates = mi[kept][mi[kept] > 0]
cumulative = candidates.cumsum() / candidates.sum()
k = int(np.clip((cumulative < MI_CUMULATIVE).sum() + 1, *K_BOUNDS))
selected = candidates.index[:k].tolist()

handover_a = a[features + [TARGET, 'split']]          # all 107 features, unscaled
handover_b = s[selected + [TARGET, 'split']]          # 25 features, scaled  <- the official dataset
print('k =', k, '| cumulative MI at k:', round(float(cumulative.iloc[k - 1]), 3))
print(selected)
```
✅ **Expected:** `k = 25 | cumulative MI at k: 0.902` and the list starting `['wage_offer_annual', 'employer_filing_count_log', 'employer_num_employees_log', 'wage_to_pw_ratio', 'edu_required', 'pwsrc_Unknown', ...]`

🧠 **Check yourself:** read your 25 features out loud and give a one-line reason why each could affect denial. That's a great viva warm-up.

---

### Final exam: does your dataset match the official one?

Save your result *inside `practice/`* in the same format, read both files back, and compare them cell by cell.

```python
mine_path = os.path.join(ROOT, 'practice', 'my_final_processed.csv.gz')
handover_b.to_csv(mine_path, index=False, float_format='%.6g')

mine = pd.read_csv(mine_path)
official = pd.read_csv(os.path.join(OUT_DIR, 'final_processed.csv.gz'))
pd.testing.assert_frame_equal(mine, official)
print(f'🎉 IDENTICAL: your {mine.shape} dataset matches results/outputs/final_processed.csv.gz exactly')
```
✅ **Expected:** `🎉 IDENTICAL: your (354849, 27) dataset matches ...`

If you get an error, the message shows the first column or value that differs. Go back to the block that creates it and compare your typing line by line.

---

## Part 3: How the repo notebooks fit together

What you just did in one practice notebook is split across the repo like this:

| You typed | Lives in | Saved as (gitignored) |
|---|---|---|
| Blocks 2–4 (Stage 0) | `stage0_load_and_integrate()` in `1_IT25101547_MissingData.ipynb` and `group_pipeline.ipynb` | — |
| Blocks 5–9 | `stage1_missing_and_invalid_data()` in `1_IT25101547_MissingData.ipynb` | `stage1_missing_handled.parquet` |
| Blocks 10–12 | `stage2_categorical_encoding()` in `2_IT25103364_Encoding.ipynb` | `stage2_encoded.parquet` |
| Block 13 | `stage3_outlier_treatment()` in `3_IT25101145_OutlierRemoval.ipynb` | `stage3_outliers_treated.parquet` |
| Block 14 | `stage4_feature_engineering()` in `4_IT25102357_FeatureEngineering.ipynb` | `stage4_features_created.parquet` |
| Block 15 | `stage5_feature_scaling()` in `5_IT25103041_Scaling.ipynb` | `stage5_scaled.parquet` |
| Blocks 16–19 | `stage6_feature_selection()` in **`6_IT25100285_FeatureSelection.ipynb`** | **`final_processed.csv.gz`** (B) + `cleaned_unscaled_allfeatures.csv.gz` (A) |

- **Member notebooks** run one stage each: read the previous file, run the stage, save the next file. Run them in order S1 → S6.
- **`group_pipeline.ipynb`** contains the same six functions and runs them all with `run_pipeline()` in about 40 seconds. It then checks its result is **identical** to the member-notebook chain; that's the "integration check" for the group mark.
- The repo versions also draw the EDA plots (m0–m6) and save the fitted parameters (`encoding_maps.json`, `outlier_caps.csv`, `scaler_params.csv`, `feature_selection_report.csv`). Your practice version skipped those to focus on the logic.

---

## Part 4: Viva preparation

### Questions on YOUR stage (Stage 6)
| Question | Answer |
|---|---|
| What does your stage do? | It selects the most informative, non-redundant features: MI ranking, then a correlation filter, then the top k. 107 candidates → 25 features. |
| What is mutual information? | How much knowing a feature reduces uncertainty about the target. 0 = independent. It captures non-linear relationships. |
| Why MI instead of Pearson correlation? | The key relationship is a threshold: the denial rate jumps at wage ratio = 1. Pearson r is only −0.066 for `wage_to_pw_ratio`, but MI ranks it 4th. MI also handles binary/one-hot features. |
| Why not chi-square? | Chi-square works only for categorical features; ~12 of ours are continuous. MI handles both. |
| Why remove correlated features if MI already ranks them? | MI scores each feature *alone*, so two near-duplicates both score high. Keeping both adds no information and causes multicollinearity in Logistic Regression. |
| Why 0.85 as the threshold? | A common rule of thumb for "strong" correlation (0.8–0.9). At 0.85 it removed the 5 true duplicates and no meaningful feature. |
| How did you choose 25 features? | The smallest set covering 90% of the total MI of the non-redundant features (bounded 15–30): 24 features → 89.5%, 25 → 90.2%. |
| Why compute MI on a 60k sample? | Speed: the kNN estimator for continuous features is slow on 284k rows. A stratified 60k sample keeps the 6.9% denial rate and gives a stable ranking. |
| Isn't selecting features with the target leakage? | Not if it uses train rows only. We never look at the test rows. |
| Why not PCA? | PCA creates synthetic components that destroy interpretability, and it's unsupervised (ignores the target). We need to explain *which* real factors drive denial. |
| What are handover A and B? | B = 25 selected scaled features, the default for all six models. A = all 107 unscaled features, an optional "different preprocessing" variety. |
| Why is `pwsrc_Unknown` selected? Is it leakage? | It flags a missing prevailing-wage source: 9.3% of denied vs ~0.4% of approved. It's on the form at filing time (an incomplete application), so it isn't leakage, but it is a shortcut. We document it and suggest an ablation without it. |
| Weaknesses of your method? | MI is univariate (it scores features one at a time, so it can miss interactions). The 0.85 and 90% thresholds are judgement calls. A wrapper or embedded method (e.g. L1 Logistic Regression) could be compared as future work. |

### Questions on the whole pipeline (any member can be asked)
| Question | Short answer |
|---|---|
| Why 80/20 and stratified? | 80% gives enough training data; stratified keeps 6.91% denied in both parts. |
| Where does k-fold CV come in? | Members run 5-fold `StratifiedKFold` on the **train rows** for tuning; the test set is used once at the end. |
| Why drop Withdrawn? | It's the applicant's choice, not a decision outcome. |
| Why is accuracy misleading here? | Predicting "always approve" gives 93.1% accuracy but catches zero denials. Use F1/recall for Denied and PR-AUC. |
| What is leakage? Give an example. | Information not available at prediction time. Processing time: denied cases took a median of 398 days vs 125, but it's only known after the decision. |
| Why winsorise instead of deleting outliers? | Deleting would remove denials, the minority class we care about. |
| Limitations of the data? | Concept drift (denial rate 10–14% in 2011–13 vs 5–6% after); missingness tied to the form version; country of citizenship raises fairness concerns (for the ethics section). |

---

## Part 5: Glossary

| Term | Meaning |
|---|---|
| **Target / label** | What we predict: `denied` (1/0) |
| **Feature** | An input column the model uses |
| **Stratified split** | A split that keeps the class percentages equal in train and test |
| **Leakage** | Information in training that wouldn't be available at prediction time, which inflates scores |
| **Coalesce** | Merge duplicate columns: take the first non-missing value |
| **Imputation** | Filling missing values (median / `"Unknown"`) |
| **Ordinal encoding** | Ordered categories → ordered numbers (Level I–IV → 1–4) |
| **One-hot encoding** | One 0/1 column per category |
| **Frequency encoding** | A category → how often it occurs |
| **Winsorising** | Clipping values to percentile limits instead of deleting rows |
| **log1p** | log(1 + x): compresses huge ranges and works with 0 |
| **z-score / standardisation** | (x − mean) / std → mean 0, std 1 |
| **Mutual information** | How much a feature reduces uncertainty about the target (≥ 0) |
| **Multicollinearity** | Features strongly correlated with each other, which makes linear-model coefficients unstable |
| **Class imbalance** | One class is much rarer (6.9% denied) |
| **PR-AUC** | Area under the precision–recall curve; the best single metric for a rare positive class |

---

## Part 6: What comes next

Your model is **Logistic Regression** on handover B. The plan, which we'll build together next:
1. **V1:** plain LR → shows the "accuracy trap" (≈93% accuracy, almost no denials caught).
2. **V2:** `class_weight='balanced'` → recall for Denied goes up.
3. **V3:** `GridSearchCV` over `C` and L1/L2, with 5-fold `StratifiedKFold` and `scoring='average_precision'`.
4. **V4:** SMOTE inside an `imblearn` Pipeline (a different-preprocessing variety).
5. **V5:** handover A + `StandardScaler` in a Pipeline (all 107 features vs your 25: this tests your own Stage 6).
6. Compare the varieties (CV mean ± std), tune the decision threshold, evaluate the test set **once**, then interpret the coefficients.
