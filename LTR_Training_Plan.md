# LTR Model Training Plan: HDB Flat Recommendation for Singles (35+)

---

## Overview

Learning to Rank (LTR) is a supervised ML approach where the model learns to order a list of items (HDB flat listings) by relevance to a query (a user's preference profile). Rather than predicting an absolute price or score, the model learns *relative ordering* — which flats are more suitable than others for a given user.

---

## Step 1: Choose Your LTR Paradigm


| Approach | How it works | Example Algorithm | Recommended? |
|---|---|---|---|
| **Pointwise** | Predict a relevance score per flat independently | Ridge Regression, RankSVM | ❌ Ignores order |
| **Pairwise** | Learn which flat is better between two candidates | RankNet, LambdaRank | ✓ Good baseline |
| **Listwise** | Optimise the ranking of the entire result list | LambdaMART, ListNet | ✅ Best for this use case |

**Recommended: LambdaMART** (via LightGBM's `lambdarank` objective or XGBoost's `rank:ndcg`). It is fast, interpretable, handles tabular data well, and directly optimises NDCG — the standard LTR metric.

---

## Step 2: Data Sources

You need three types of data, all publicly available in Singapore.

### A. Flat / Listing Data (Item Features)
| Source | Data Available |
|---|---|
| [HDB Resale Flat Prices (data.gov.sg)](https://data.gov.sg/dataset/resale-flat-prices) | Block, street, town, flat type, storey range, floor area, lease commence date, resale price |
| [HDB Property Information](https://data.gov.sg/dataset/hdb-property-information) | Total units, year completed, block details |
| OneMap API | Coordinates of each block (for distance calculations) |
| LTA DataMall | MRT/LRT station coordinates, bus stop locations |
| MOH / Google Maps API | Polyclinic, hospital coordinates |
| HDB Map Services | Hawker centres, community clubs, parks |

### B. User Preference Data (Query Features)
These come directly from the user at query time (input form / onboarding survey):
- Budget range (min / max in SGD)
- Flat type preference (2-room Flexi, 3-room, etc.)
- BTO vs Resale preference
- Home MRT station or postal code of workplace
- Postal code of parents' home
- Lifestyle preferences (noise sensitivity, greenery, nightlife proximity)
- Importance weights per preference (slider: 1–5)

### C. Relevance Labels (Ground Truth)
This is the hardest part. You need to know which flats are "good" for which user profiles. Three strategies:

| Strategy | Description | Pros | Cons |
|---|---|---|---|
| **Expert annotation** | Manually label 200–500 (user profile, flat) pairs with scores 0–4 | High quality | Expensive, slow |
| **Implicit feedback proxy** | Use transaction data — flats that were actually bought by singles in a given profile | Realistic | Needs demographic enrichment |
| **Synthetic / rule-based labels** | Generate labels via scoring function (e.g. distance < 500m to MRT = +2, budget fit = +3) | Scalable | Noisier, less realistic |

**Recommended for a student project: Synthetic labels as a starting point, validated with a small expert-annotated set.**

---

## Step 3: Feature Engineering (Model Inputs)

Features are grouped into three categories: flat features, user features, and interaction features (flat × user).

### A. Flat-Level Features (Static)
These describe each listing and are computed once.

| Feature | Type | How to Compute |
|---|---|---|
| Resale price (SGD) | Numerical | From HDB resale dataset |
| Price per square foot | Numerical | Price / floor area |
| Floor area (sqm) | Numerical | From dataset |
| Storey band (low / mid / high) | Ordinal | Encode storey range |
| Flat type | Categorical | One-hot encode |
| Remaining lease (years) | Numerical | 99 - (current year - lease commence year) |
| Town / estate | Categorical | Label encode |
| Distance to nearest MRT (m) | Numerical | Haversine(block coords, MRT coords) |
| Distance to nearest hawker (m) | Numerical | Haversine(block coords, hawker coords) |
| Distance to nearest polyclinic (m) | Numerical | Haversine(block coords, clinic coords) |
| Distance to nearest park (m) | Numerical | Haversine(block coords, park coords) |
| Number of MRT stations within 1km | Numerical | Count from LTA data |
| Is mature estate | Binary | Based on HDB estate classification |
| Price trend (12-month delta %) | Numerical | From historical resale data |

### B. User-Level Features (Query)
These describe the user's preferences and are provided at query time.

| Feature | Type | Source |
|---|---|---|
| Budget max (SGD) | Numerical | User input |
| Preferred flat type | Categorical | User input |
| BTO or resale preference | Binary | User input |
| Workplace coordinates | Numerical | User input (geocoded) |
| Parents' home coordinates | Numerical | User input (geocoded) |
| Importance weight: transport | Numerical (1–5) | User slider |
| Importance weight: budget | Numerical (1–5) | User slider |
| Importance weight: parents proximity | Numerical (1–5) | User slider |
| Importance weight: lifestyle/F&B | Numerical (1–5) | User slider |
| Importance weight: investment value | Numerical (1–5) | User slider |

### C. Interaction Features (Flat × User) — Most Important
These are computed per (user, flat) pair and capture relevance directly.

| Feature | Formula |
|---|---|
| Budget feasibility score | `1 - max(0, (flat_price - user_budget_max) / user_budget_max)` |
| Distance to workplace (m) | `Haversine(flat_coords, workplace_coords)` |
| Distance to parents' home (m) | `Haversine(flat_coords, parents_coords)` |
| Distance to nearest MRT (weighted) | `mrt_distance × user_weight_transport` |
| Is flat type matched | `1 if flat_type == preferred_type else 0` |
| Is BTO/resale matched | `1 if flat_tenure == preferred_tenure else 0` |
| Weighted composite score | Weighted sum of all distance/budget features by user importance weights |

---

## Step 4: Label Generation (Model Output Target)

Labels represent the **relevance of a flat to a user profile**. Use a 5-point graded relevance scale:

| Score | Meaning |
|---|---|
| 4 | Perfect match — meets all preferences, within budget, close to work and parents |
| 3 | Good match — meets most preferences, minor trade-offs |
| 2 | Acceptable — meets some preferences, clear gaps |
| 1 | Poor match — budget or distance constraints barely met |
| 0 | Irrelevant — over budget, wrong flat type, too far on all dimensions |

### Synthetic Label Formula (for bootstrapping)
```
relevance = 0
+ 2 × budget_feasibility_score          # most important
+ 1 × (1 if flat_type_match else 0)
+ 1 × (1 if tenure_match else 0)
+ weighted_transport_score              # user weight × (1 - normalised MRT distance)
+ weighted_parents_score               # user weight × (1 - normalised parent distance)
+ weighted_workplace_score             # user weight × (1 - normalised workplace distance)

Normalise final score → [0, 4] and round to integer
```

---

## Step 5: Training Pipeline

```
Raw Data (HDB resale, LTA, MOH, OneMap)
        ↓
Feature Engineering (flat features + interaction features)
        ↓
Label Generation (synthetic scoring function)
        ↓
Group Construction (group = all flats for one user query)
        ↓
Train / Val / Test Split (80/10/10 by user profile, not by flat)
        ↓
LambdaMART Training (LightGBM lambdarank)
        ↓
Hyperparameter Tuning (num_leaves, learning_rate, min_data_in_leaf)
        ↓
Evaluation (NDCG@5, NDCG@10, MRR)
        ↓
Output: Ranked list of (block, street, flat type, price, score)
```

### Key Note on Groups
LTR models require a **group** parameter — the set of candidates being ranked for a single query. In your case, a group = all available HDB listings being ranked for one user's preference profile. LightGBM uses `lgb.Dataset(..., group=[n1, n2, ...])` where each value is the number of items in that query group.

---

## Step 6: Model Training Code Skeleton

```python
import lightgbm as lgb
import pandas as pd
import numpy as np

# X: interaction features (flat × user)
# y: relevance labels (0–4)
# groups: number of listings per user query

train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
val_data   = lgb.Dataset(X_val,   label=y_val,   group=val_groups)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "label_gain": [0, 1, 3, 7, 15]  # importance weights per relevance level
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

# Predict relevance scores and rank
scores = model.predict(X_test)
ranked_listings = listings_df.assign(score=scores).sort_values("score", ascending=False)
```

---

## Step 7: Evaluation Metrics

| Metric | What it measures | Target |
|---|---|---|
| **NDCG@5** | Quality of top 5 recommendations | Primary metric — aim for > 0.75 |
| **NDCG@10** | Quality of top 10 recommendations | Secondary metric |
| **MRR** | Mean Reciprocal Rank — how high the first relevant result appears | > 0.6 |
| **Precision@5** | Fraction of top 5 that are relevant (score ≥ 3) | > 0.6 |

**Why NDCG?** NDCG (Normalised Discounted Cumulative Gain) penalises relevant results appearing low in the list — exactly what matters in a recommendation system where users rarely scroll past the top 5.

---

## Step 8: Output Format

Each recommendation returned to the user should include:

```
Rank | Block | Street Name         | Town       | Flat Type | Est. Price  | Match Score | Key Highlights
-----|-------|---------------------|------------|-----------|-------------|-------------|-------------------------
  1  | 123A  | Toa Payoh Lorong 1  | Toa Payoh  | 3-room    | $350,000    | 94%         | 4 min to MRT, 1.2km to parents
  2  | 456B  | Bishan St 13        | Bishan     | 3-room    | $370,000    | 91%         | 6 min to MRT, within budget
  3  | 789C  | Clementi Ave 4      | Clementi   | 3-room    | $340,000    | 87%         | Near hawker, 20 min commute
```

---

## Summary of Inputs and Outputs

### Inputs
- **User preferences** (query time): budget, flat type, tenure, workplace, parents' location, importance weights
- **Flat features** (pre-computed): price, size, MRT distance, lease, estate type, hawker/clinic proximity
- **Interaction features** (computed per query): distance to workplace/parents, budget fit, type match

### Output
- A **ranked list of specific HDB flat listings** (block + street + flat type) ordered by predicted relevance score for that user, with top-5 highlighted and key match reasons displayed.

---

## Recommended Tools & Libraries

| Purpose | Tool |
|---|---|
| LTR model | LightGBM (`lambdarank`) or XGBoost (`rank:ndcg`) |
| Geocoding / distances | OneMap Singapore API + `geopy` (Haversine) |
| Data processing | `pandas`, `numpy` |
| Evaluation | `sklearn.metrics`, `pyltr` |
| Data sources | data.gov.sg, LTA DataMall, OneMap API |
