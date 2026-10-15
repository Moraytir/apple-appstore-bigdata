# What makes an app "hit" the App Store? 🌿

Analysis of **1.23 million Apple App Store apps** with PySpark, Spark SQL, Spark MLlib and scikit-learn on Databricks, plus an interactive web explorer with an app simulator.

> 🇹🇭 **สรุปสั้น ๆ:** วิเคราะห์แอปบน App Store 1.23 ล้านแอป ว่าแอปแบบไหนมีโอกาสได้รับความนิยม (วัดจากจำนวนรีวิว) พบว่าตลาดกระจุกตัวสุดขีด (แอป 1% บนสุดถือรีวิว 95.8%) โมเดล Spark MLlib ทำนายแอปที่ติดตลาดได้ ROC-AUC 0.884 โดยไม่ได้อาศัยแค่อายุแอป และ k-means แบ่งแอปเป็น 4 กลุ่ม ตั้งแต่ "ดาวเด่น" ถึง "ซอมบี้จากโรงงานแอป" มีเว็บให้ลองจำลองแอปของตัวเองได้

**🔗 Live demo:** `<add your GitHub Pages link here>`

Course project for 040563105 Statistical Analysis for Big Data (KMITNB).

## Key findings

| Question | Finding |
|---|---|
| How unequal is the market? | **55.6%** of apps have no reviews at all. The top **1%** of apps hold **95.8%** of all reviews (Gini = **0.996**). |
| Which categories have the most hits? | Share of apps with more than 1,000 reviews: Weather 7.75%, Graphics & Design 4.89%, Games 4.66%. Lowest: Business 0.54%, Developer Tools 0.46%, Stickers 0.06%. |
| Which categories have the most abandoned apps? | Share not updated for over 2 years: Stickers 65.6%, Games 58.7%. Lowest: Shopping 20.5%, Finance 21.5%. |
| Does app size matter? | Hit rate rises with size: under 10 MB **0.32%**, 100-500 MB **5.43%**, 500+ MB **8.97%**. |
| Free or paid? | Raw hit rate is higher for free apps (2.04% vs 1.12%), but in the models "free" adds almost no predictive power once age, size and genre are accounted for. |
| Indie or app factory? | Indie developers (1 app) average **4.32** stars versus **3.84** for developers with 100+ apps (apps with at least one review). This is the opposite of the hypothesis that big developers rate higher (Cliff's delta -0.271). |
| Do updates matter? | Among apps released 2016-2018, those updated within the last 3-12 months are hits 6.76% of the time versus 0.70% for apps untouched for over 2 years. This is a correlation: popular apps may simply earn enough to keep updating. |
| Can we predict hits? (Spark MLlib) | On all 1.23M apps (1.96% hits): GBT **ROC-AUC 0.884 / PR-AUC 0.199** (about 10× the 0.0196 baseline), Random Forest 0.874 / 0.180, Logistic Regression 0.852 / 0.125. Top predictors: genre, size, app age, minimum iOS version. |
| Is the model just learning app age? | No. App age alone gives ROC-AUC **0.667**; dropping app age still gives **0.828**, and dropping both age and iOS version gives 0.809. |
| What kinds of apps are there? (k-means, k = 4) | Among apps with reviews: **Stars** (31% hits, 4.39 stars, largest apps, 21.5% abandoned), **Newcomers** (largest group, 2.9 years old, 4.61 stars, few reviews yet), **Fading** (1.86 stars, 60% abandoned), **App-factory zombies** (89% abandoned, median developer has 18 apps). |

## Web explorer

A single-page site (`docs/index.html`) that runs entirely in the browser on a pre-aggregated table (`results/cube for web.csv`, embedded in the page).

- **App simulator:** pick genre, free/paid, size, team type and cohort, then see how similar apps performed against the market, and what changes if you vary one factor at a time.
- **Explorer:** preset questions and free filtering with comparison charts.
- Groups with fewer than 30 apps are hidden or flagged.

The design follows the GreenLeaf design system.

## Repository structure

```
├── README.md
├── notebook/
│   └── apple_appstore_analysis.ipynb    full analysis with outputs (Databricks export)
├── apple_appstore_analysis_current.py   same notebook as Databricks source
├── results/
│   ├── cube for web.csv                 aggregated table that powers the web page
│   └── *.csv                            summary tables for each analysis section
├── docs/
│   └── index.html                       web explorer (GitHub Pages)
└── image/                               chart screenshots
```

## Data

[Apple App Store Apps on Kaggle](https://www.kaggle.com/datasets/gauthamp10/apple-appstore-apps) (21 columns, 1,230,376 apps). The file is dated 14 October 2021 and very few apps were updated after mid-October 2021, so the snapshot was most likely taken around then (see Limitations). The raw file (about 400 MB) is **not included**; download it from Kaggle and check the dataset's license before reuse.

## How to reproduce

1. Download the CSV from Kaggle.
2. In Databricks (Free Edition works), upload it via **Add data** and create the table `workspace.default.apple_app_data`.
3. Import the notebook and run the cells top to bottom on Serverless compute (the MLlib section takes a few minutes; lower `FRACTION` in section 10.2 if needed).
4. For the web table, run the cube cell (section 11) and download the **full** results (not the 10,000-row preview).

## Method

- **Cleaning:** de-duplicate on `App_Id`; derive size in MB, app age, days since update, website flag, minimum iOS major version, and developer app count. Reference date is the latest `Updated` value, not today. Years 2008-2021 only (2022 is incomplete).
- **Definitions:** *hit* = more than 1,000 reviews; *abandoned* = no update for over 730 days.
- **Inequality:** Lorenz curve and Gini coefficient of review counts.
- **Correlation:** Spearman correlation matrix with `pyspark.ml.stat.Correlation`.
- **Model (Spark MLlib, full data):** `StringIndexer` → `OneHotEncoder` → `Imputer` → `VectorAssembler` pipeline; Logistic Regression, Random Forest and GBT with class weights; ROC-AUC and PR-AUC (accuracy is misleading at 2% positives); an ablation that removes app age and iOS version to check what the model really learns.
- **Model (scikit-learn, first pass):** HistGradientBoostingClassifier on a 25% sample (AUC 0.895) with permutation importance, kept for comparison.
- **Clustering:** k-means on standardised, log-transformed features; k chosen with the elbow method and silhouette score; PCA for a 2D view.
- **Hypothesis tests:** Mann-Whitney U with Cliff's delta, because with hundreds of thousands of rows p-values are tiny even for negligible effects.

## Limitations

- A single snapshot: no history per app, so nothing here shows cause and effect.
- No ranking, featuring or download data, so this says nothing about Apple's algorithm.
- Review count is a proxy for popularity, not for users or downloads.
- Newer apps have had less time to collect reviews, so cohorts are not directly comparable.
- `Current_Version_Score` equals `Average_User_Rating` and `Current_Version_Reviews` equals `Reviews` on every row, so rating trends and "still popular vs faded" could not be measured.
- The reference date (`max(Updated)` = 2022-04-09) is probably about six months later than the real snapshot (mid-October 2021). App age and days since update are therefore inflated by about 176 days, so the 730-day abandoned threshold corresponds to roughly 1.5 years without updates. The value was kept so all sections stay comparable.
- Apps with no reviews store a rating of 0, which inflates the correlation between review count and rating (0.88).

## Tech stack

Databricks, PySpark, Spark SQL, Spark MLlib, pandas, scikit-learn, SciPy, Matplotlib, Chart.js.

Built by [@Moraytir](https://github.com/Moraytir).
