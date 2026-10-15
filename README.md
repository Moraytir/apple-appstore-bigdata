# What makes an app "hit" the App Store? 🌿

Analysis of **1.23 million Apple App Store apps** with PySpark, SQL and scikit-learn on Databricks, plus an interactive web explorer with an app simulator.

> 🇹🇭 **สรุปสั้น ๆ:** วิเคราะห์แอปบน App Store 1.23 ล้านแอป ว่าแอปแบบไหนมีโอกาสได้รับความนิยม (วัดจากจำนวนรีวิว) พบว่าตลาดกระจุกตัวสุดขีด (แอป 1% บนสุดถือรีวิว 95.8%) และ "ขนาดแอป" ทำนายความสำเร็จได้ดีกว่า "ฟรีหรือเสียเงิน" มีเว็บให้ลองจำลองแอปของตัวเองได้

**🔗 Live demo:** https://moraytir.github.io/apple-appstore-bigdata

## Key findings

| Question | Finding |
|---|---|
| How unequal is the market? | **55.6%** of apps have no reviews at all. The top **1%** of apps hold **95.8%** of all reviews (Gini = **0.996**). |
| Which categories have the most hits? | Share of apps with more than 1,000 reviews: Weather 7.75%, Graphics & Design 4.89%, Games 4.66%. Lowest: Business 0.54%, Developer Tools 0.46%, Stickers 0.06%. |
| Which categories have the most abandoned apps? | Share not updated for over 2 years: Stickers 65.6%, Games 58.7%. Lowest: Shopping 20.5%, Finance 21.5%. |
| Does app size matter? | Hit rate rises with size: under 10 MB **0.32%**, 100-500 MB **5.43%**, 500+ MB **8.97%**. |
| Free or paid? | Raw hit rate is higher for free apps (2.04% vs 1.12%), but in the model "free" adds almost no predictive power once age, size and genre are accounted for. |
| Indie or app factory? | Indie developers (1 app) average **4.32** stars versus **3.84** for developers with 100+ apps (apps with at least one review). This is the opposite of the hypothesis that big developers rate higher (Cliff's delta -0.271). |
| Do updates matter? | Among apps released 2016-2018, those updated within the last 3-12 months are hits 6.76% of the time versus 0.70% for apps untouched for over 2 years. This is a correlation: popular apps may simply earn enough to keep updating. |
| Can we predict hits? | Gradient boosting reaches **AUC 0.895** (about 2% of apps are hits). Top predictors: app age, size, minimum iOS version, genre. |

## Web explorer

A single-page site (`docs/index.html`) that runs entirely in the browser on a pre-aggregated table (`results/cube for web.csv`).

- **App simulator:** pick genre, free/paid, size, team type and cohort, then see how similar apps performed against the market, and what changes if you vary one factor at a time.
- **Explorer:** preset questions and free filtering with comparison charts.
- Groups with fewer than 30 apps are hidden or flagged.

The design follows the GreenLeaf design system.

## Repository structure

```
├── README.md
├── .github/workflows/static.yml         deploys the site to GitHub Pages
├── notebook/
│   └── apple_appstore_analysis.ipynb    full analysis with outputs (Databricks export)
├── results/
│   ├── cube for web.csv                 aggregated table that powers the web page
│   └── *.csv                            summary tables for each analysis section
├── docs/
│   └── index.html                       web explorer (GitHub Pages)
└── image/                               chart screenshots
```

## Charts

![Review inequality across the App Store](image/Review%20inequality%20ascross%20the%20App%20Store.png)

![Permutation importance](image/Permutation%20importance.png)

## Data

[Apple App Store Apps on Kaggle](https://www.kaggle.com/datasets/gauthamp10/apple-appstore-apps) (21 columns, snapshot taken April 2022). The raw file (about 400 MB) is **not included**; download it from Kaggle and check the dataset's license before reuse.

## How to reproduce

1. Download the CSV from Kaggle.
2. In Databricks (Free Edition works), upload it via **Add data** and create the table `workspace.default.apple_app_data`.
3. Import the notebook and run the cells top to bottom on Serverless compute.
4. For the web table, run the cube cell and download the **full** results (not the 10,000-row preview).

## Method

- **Cleaning:** de-duplicate on `App_Id`; derive size in MB, app age, days since update, website flag, minimum iOS major version, and developer app count. Reference date is the latest `Updated` value, not today. Years 2008-2021 only (2022 is incomplete).
- **Definitions:** *hit* = more than 1,000 reviews; *abandoned* = no update for over 730 days.
- **Inequality:** Lorenz curve and Gini coefficient of review counts.
- **Model:** HistGradientBoostingClassifier on a 25% sample, stratified 80/20 split, evaluated with AUC (accuracy is misleading at 2% positives); permutation importance for feature ranking. A baseline check (section 9.1) compares the full model with an age-only model and with models that drop `age_days` or `ios_ver`, and reports PR-AUC, to show how much of the AUC comes from app age alone.
- **Hypothesis tests:** Mann-Whitney U with Cliff's delta, because with hundreds of thousands of rows p-values are tiny even for negligible effects.

## Limitations

- A single snapshot: no history per app, so nothing here shows cause and effect.
- No ranking, featuring or download data, so this says nothing about Apple's algorithm.
- Review count is a proxy for popularity, not for users or downloads.
- Newer apps have had less time to collect reviews, so cohorts are not directly comparable.
- `Current_Version_Score` was identical to `Average_User_Rating` in this data, so a rating-trend feature could not be built.

## Tech stack

Databricks, PySpark, SQL, pandas, scikit-learn, SciPy, Matplotlib, Chart.js.

Built by [@Moraytir](https://github.com/Moraytir).
